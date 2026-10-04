//! Saving and loading a fitted GLM: a versioned JSON artifact.
//!
//! The artifact holds what governance needs to audit and reuse the fit
//! (`docs/design/models.md`, "Model artifacts"): the spec, the estimates
//! and their covariance, the fit statistics, the fitted values, and
//! provenance with the crate version and a hash of the training data.
//! Numbers round-trip bit for bit; a non-finite value is written as the
//! string `"NaN"`, `"inf"` or `"-inf"`.
//!
//! The artifact does not hold the training data, so a loaded fit predicts
//! and simulates but cannot give a sandwich covariance. Factor levels and
//! other coding live with the design (`act_models::Coding`), not here.

use act_core::{Error, Result};
use act_models::{Family, Link};
use serde_json::{Map, Number, Value, json};

use crate::{Dispersion, Glm, GlmFit};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.glm_fit";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

impl GlmFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use act_glm::{Glm, GlmFit};
    /// use act_models::{Design, Family, Link, Model};
    ///
    /// let d = Design::new(
    ///     vec!["(Intercept)".into(), "x".into()],
    ///     vec![vec![1.0; 4], vec![0.0, 1.0, 2.0, 3.0]],
    /// )
    /// .unwrap();
    /// let fit = Glm::new(Family::Poisson, Link::Log).fit(&d, &[1.0, 2.0, 2.0, 5.0]).unwrap();
    /// let back = GlmFit::from_json(&fit.to_json()).unwrap();
    /// assert_eq!(back, fit);
    /// ```
    pub fn to_json(&self) -> String {
        let spec = &self.spec;
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": {
                "model": "glm",
                "versions": [["act-glm", env!("CARGO_PKG_VERSION")]],
                "input_hash": self.input_hash,
            },
            "spec": {
                "family": family_json(spec.family),
                "link": link_json(spec.link),
                "dispersion": match spec.dispersion {
                    Dispersion::Fixed(v) => json!({"kind": "fixed", "value": num(v)}),
                    Dispersion::Pearson => json!({"kind": "pearson"}),
                    Dispersion::Deviance => json!({"kind": "deviance"}),
                },
                "tolerance": num(spec.tolerance),
                "max_iterations": spec.max_iterations,
            },
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "unscaled_covariance": nums(&self.unscaled_covariance),
            "dispersion": num(self.dispersion),
            "deviance": num(self.deviance),
            "null_deviance": num(self.null_deviance),
            "log_likelihood": num(self.log_likelihood),
            "n_obs": self.n_obs,
            "iterations": self.iterations,
            "fitted": nums(&self.fitted),
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc: Value = serde_json::from_str(text)
            .map_err(|e| Error::Data(format!("model artifact is not JSON: {e}")))?;
        let top = object(&doc, "artifact")?;
        if top.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Err(Error::Data(format!("not a {FORMAT} artifact")));
        }
        let version = uint(top, "format_version")?;
        if version == 0 || version > FORMAT_VERSION {
            return Err(Error::Data(format!(
                "artifact format version {version}; this build reads 1 to {FORMAT_VERSION}"
            )));
        }
        let provenance = object(field(top, "provenance")?, "provenance")?;
        let input_hash = field(provenance, "input_hash")?
            .as_str()
            .ok_or_else(|| bad("provenance.input_hash"))?
            .to_string();

        let spec = object(field(top, "spec")?, "spec")?;
        let family = family_from(object(field(spec, "family")?, "family")?)?;
        family.validate()?;
        let link = link_from(object(field(spec, "link")?, "link")?)?;
        let disp = object(field(spec, "dispersion")?, "dispersion")?;
        let dispersion_spec = match field(disp, "kind")?.as_str() {
            Some("fixed") => Dispersion::Fixed(float(field(disp, "value")?, "dispersion.value")?),
            Some("pearson") => Dispersion::Pearson,
            Some("deviance") => Dispersion::Deviance,
            _ => return Err(bad("spec.dispersion.kind")),
        };
        let glm = Glm {
            family,
            link,
            dispersion: dispersion_spec,
            tolerance: float(field(spec, "tolerance")?, "tolerance")?,
            max_iterations: uint(spec, "max_iterations")? as usize,
        };

        let names: Vec<String> = field(top, "names")?
            .as_array()
            .ok_or_else(|| bad("names"))?
            .iter()
            .map(|v| v.as_str().map(String::from).ok_or_else(|| bad("names")))
            .collect::<Result<_>>()?;
        let coefficients = floats(top, "coefficients")?;
        let unscaled_covariance = floats(top, "unscaled_covariance")?;
        let fitted = floats(top, "fitted")?;
        let n_obs = uint(top, "n_obs")? as usize;
        let p = names.len();
        if coefficients.len() != p || unscaled_covariance.len() != p * p || fitted.len() != n_obs {
            return Err(Error::Data(
                "model artifact sizes disagree: names, coefficients, covariance, fitted and n_obs"
                    .into(),
            ));
        }
        Ok(GlmFit {
            spec: glm,
            names,
            coefficients,
            unscaled_covariance,
            dispersion: float(field(top, "dispersion")?, "dispersion")?,
            deviance: float(field(top, "deviance")?, "deviance")?,
            null_deviance: float(field(top, "null_deviance")?, "null_deviance")?,
            log_likelihood: float(field(top, "log_likelihood")?, "log_likelihood")?,
            n_obs,
            iterations: uint(top, "iterations")? as usize,
            fitted,
            input_hash,
        })
    }
}

fn family_json(f: Family) -> Value {
    match f {
        Family::NegativeBinomial { theta } => json!({"name": f.name(), "theta": num(theta)}),
        Family::Tweedie { power } => json!({"name": f.name(), "power": num(power)}),
        _ => json!({"name": f.name()}),
    }
}

fn family_from(o: &Map<String, Value>) -> Result<Family> {
    Ok(match field(o, "name")?.as_str() {
        Some("gaussian") => Family::Gaussian,
        Some("poisson") => Family::Poisson,
        Some("gamma") => Family::Gamma,
        Some("inverse_gaussian") => Family::InverseGaussian,
        Some("binomial") => Family::Binomial,
        Some("negative_binomial") => Family::NegativeBinomial {
            theta: float(field(o, "theta")?, "family.theta")?,
        },
        Some("tweedie") => Family::Tweedie {
            power: float(field(o, "power")?, "family.power")?,
        },
        _ => return Err(bad("spec.family.name")),
    })
}

fn link_json(l: Link) -> Value {
    let name = match l {
        Link::Identity => "identity",
        Link::Log => "log",
        Link::Logit => "logit",
        Link::Probit => "probit",
        Link::Cloglog => "cloglog",
        Link::Inverse => "inverse",
        Link::InverseSquared => "inverse_squared",
        Link::Power(p) => return json!({"name": "power", "power": num(p)}),
    };
    json!({"name": name})
}

fn link_from(o: &Map<String, Value>) -> Result<Link> {
    Ok(match field(o, "name")?.as_str() {
        Some("identity") => Link::Identity,
        Some("log") => Link::Log,
        Some("logit") => Link::Logit,
        Some("probit") => Link::Probit,
        Some("cloglog") => Link::Cloglog,
        Some("inverse") => Link::Inverse,
        Some("inverse_squared") => Link::InverseSquared,
        Some("power") => Link::Power(float(field(o, "power")?, "link.power")?),
        _ => return Err(bad("spec.link.name")),
    })
}

fn num(x: f64) -> Value {
    match Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None if x.is_nan() => Value::String("NaN".into()),
        None if x > 0.0 => Value::String("inf".into()),
        None => Value::String("-inf".into()),
    }
}

fn nums(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| num(x)).collect())
}

fn float(v: &Value, what: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| bad(what)),
        Value::String(s) => match s.as_str() {
            "NaN" => Ok(f64::NAN),
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            _ => Err(bad(what)),
        },
        _ => Err(bad(what)),
    }
}

fn floats(o: &Map<String, Value>, key: &str) -> Result<Vec<f64>> {
    field(o, key)?
        .as_array()
        .ok_or_else(|| bad(key))?
        .iter()
        .map(|v| float(v, key))
        .collect()
}

fn uint(o: &Map<String, Value>, key: &str) -> Result<u64> {
    field(o, key)?.as_u64().ok_or_else(|| bad(key))
}

fn field<'a>(o: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    o.get(key)
        .ok_or_else(|| Error::Data(format!("model artifact has no {key:?}")))
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| bad(what))
}

fn bad(what: &str) -> Error {
    Error::Data(format!("model artifact field {what:?} is invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_models::{Design, Model};

    fn design() -> Design {
        Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 6], vec![0.1, 0.7, 1.3, 2.2, 2.9, 3.4]],
        )
        .unwrap()
        .with_weights(vec![1.0, 2.0, 0.5, 1.0, 1.5, 1.0])
        .unwrap()
    }

    #[test]
    fn every_family_and_link_round_trips_bit_for_bit() {
        let d = design();
        let cases = [
            (
                Glm::new(Family::Gaussian, Link::Identity),
                vec![1.1, 2.3, 2.9, 4.4, 5.2, 6.0],
            ),
            (
                Glm::over_dispersed_poisson(),
                vec![0.0, 2.0, 1.0, 4.0, 3.0, 7.0],
            ),
            (
                Glm::new(Family::Gamma, Link::Inverse),
                vec![1.1, 2.3, 2.9, 4.4, 5.2, 6.0],
            ),
            (
                Glm::new(Family::Tweedie { power: 1.37 }, Link::Power(-0.37))
                    .dispersion(Dispersion::Deviance),
                vec![0.0, 2.3, 0.0, 4.4, 5.2, 6.0],
            ),
            (
                Glm::new(Family::NegativeBinomial { theta: 1.7 }, Link::Log),
                vec![0.0, 2.0, 1.0, 4.0, 3.0, 7.0],
            ),
            (
                Glm::new(Family::Binomial, Link::Probit),
                vec![0.0, 0.5, 0.0, 1.0, 0.5, 1.0],
            ),
        ];
        for (glm, y) in cases {
            let fit = glm.fit(&d, &y).unwrap();
            let back = GlmFit::from_json(&fit.to_json()).unwrap();
            assert_eq!(back, fit, "{:?}", glm.family);
            assert!(back.input_hash().starts_with("blake3:"));
        }
    }

    #[test]
    fn non_finite_values_round_trip() {
        let mut fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(&design(), &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
            .unwrap();
        fit.log_likelihood = f64::NAN;
        fit.deviance = f64::INFINITY;
        let back = GlmFit::from_json(&fit.to_json()).unwrap();
        assert!(back.log_likelihood.is_nan());
        assert_eq!(back.deviance, f64::INFINITY);
    }

    #[test]
    fn the_hash_identifies_the_training_data() {
        let glm = Glm::new(Family::Gaussian, Link::Identity);
        let a = glm.fit(&design(), &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]).unwrap();
        let b = glm.fit(&design(), &[1.0, 2.0, 3.0, 4.0, 5.0, 7.5]).unwrap();
        assert_ne!(a.input_hash(), b.input_hash());
        assert_eq!(
            a.input_hash(),
            glm.fit(&design(), &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
                .unwrap()
                .input_hash()
        );
    }

    #[test]
    fn rejects_other_formats_versions_and_broken_fields() {
        let fit = Glm::new(Family::Poisson, Link::Log)
            .fit(&design(), &[0.0, 2.0, 1.0, 4.0, 3.0, 7.0])
            .unwrap();
        let mut doc: Value = serde_json::from_str(&fit.to_json()).unwrap();
        assert!(GlmFit::from_json("{").is_err());
        let edit = |f: &dyn Fn(&mut Value)| {
            let mut d = doc.clone();
            f(&mut d);
            GlmFit::from_json(&d.to_string())
        };
        assert!(edit(&|d| d["format"] = json!("other")).is_err());
        let err = edit(&|d| d["format_version"] = json!(2)).unwrap_err();
        assert!(err.to_string().contains("version 2"));
        assert!(edit(&|d| d["coefficients"] = json!([1.0])).is_err());
        assert!(edit(&|d| d["spec"]["link"]["name"] = json!("sqrt")).is_err());
        assert!(edit(&|d| d["spec"]["family"] = json!({"name": "tweedie", "power": 3.0})).is_err());
        doc.as_object_mut().unwrap().remove("fitted");
        assert!(GlmFit::from_json(&doc.to_string()).is_err());
    }
}
