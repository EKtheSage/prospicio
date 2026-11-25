//! Saving and loading GLMM fits: a versioned JSON artifact in the envelope
//! the other GLM artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the specification, the fixed effects and their
//! covariance, the variance components with their standard errors and
//! correlations, the optimizer's parameters and their covariance, the
//! dispersion, the predicted random effects of every grouping factor with
//! their variances, and the fit statistics; numbers round-trip bit for
//! bit. A loaded fit predicts on new data with the same grouping factors.
//! The fit keeps no hash of its training data, so the provenance's
//! `input_hash` is empty.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, array, bad, family_from, family_json, field, float, floats, input_hash, num, nums,
    object, parse, strings, uint,
};
use serde_json::{Map, Value, json};

use super::{Glmm, GlmmFit, Integration, RandomEffects};

const FORMAT: &str = "risk_rs.glmm_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

fn matrix_json(rows: &[Vec<f64>]) -> Value {
    rows.iter().map(|r| nums(r)).collect()
}

fn matrix_from(o: &Map<String, Value>, key: &str) -> Result<Vec<Vec<f64>>> {
    array(o, key)?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| bad(key))?
                .iter()
                .map(|v| float(v, key))
                .collect()
        })
        .collect()
}

fn spec_json(spec: &Glmm) -> Value {
    json!({
        "family": family_json(spec.family),
        "integration": match spec.integration {
            Integration::Laplace => json!({"kind": "laplace"}),
            Integration::Quadrature { nodes } => json!({"kind": "quadrature", "nodes": nodes}),
        },
        "tolerance": num(spec.tolerance),
        "max_iterations": spec.max_iterations,
    })
}

fn spec_from(o: &Map<String, Value>) -> Result<Glmm> {
    let family = family_from(object(field(o, "family")?, "family")?)?;
    let integration = object(field(o, "integration")?, "integration")?;
    let integration = match field(integration, "kind")?.as_str() {
        Some("laplace") => Integration::Laplace,
        Some("quadrature") => Integration::Quadrature {
            nodes: uint(integration, "nodes")? as usize,
        },
        _ => return Err(bad("spec.integration.kind")),
    };
    let spec = Glmm {
        family,
        integration,
        tolerance: float(field(o, "tolerance")?, "tolerance")?,
        max_iterations: uint(o, "max_iterations")? as usize,
    };
    spec.power()?;
    Ok(spec)
}

fn effects_json(e: &RandomEffects) -> Value {
    json!({
        "name": e.name,
        "terms": e.terms,
        "levels": e.levels,
        "values": matrix_json(&e.values),
        "conditional_variance": matrix_json(&e.conditional_variance),
        "prediction_variance": matrix_json(&e.prediction_variance),
    })
}

fn effects_from(v: &Value) -> Result<RandomEffects> {
    let o = object(v, "random_effects")?;
    let e = RandomEffects {
        name: field(o, "name")?
            .as_str()
            .ok_or_else(|| bad("random_effects.name"))?
            .to_string(),
        terms: strings(o, "terms")?,
        levels: strings(o, "levels")?,
        values: matrix_from(o, "values")?,
        conditional_variance: matrix_from(o, "conditional_variance")?,
        prediction_variance: matrix_from(o, "prediction_variance")?,
    };
    let shaped =
        |m: &[Vec<f64>]| m.len() == e.terms.len() && m.iter().all(|r| r.len() == e.levels.len());
    if !(shaped(&e.values) && shaped(&e.conditional_variance) && shaped(&e.prediction_variance)) {
        return Err(Error::Data(format!(
            "model artifact's random effects of {} disagree with its terms and levels",
            e.name
        )));
    }
    Ok(e)
}

impl GlmmFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::glmm::{Glmm, GlmmFit, Grouping};
    /// use prospicio_models::{Design, Family};
    ///
    /// let x: Vec<f64> = (0..24).map(|i| f64::from(i % 5) / 4.0).collect();
    /// let g: Vec<String> = (0..24).map(|i| format!("g{}", i % 6)).collect();
    /// let y: Vec<f64> = (0..24).map(|i| f64::from(1 + (i * 7) % 5 + i % 6)).collect();
    /// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 24], x]).unwrap();
    /// let fit = Glmm::new(Family::Poisson).fit(&d, &[Grouping::new("g", g)], &y).unwrap();
    /// let back = GlmmFit::from_json(&fit.to_json()).unwrap();
    /// assert_eq!(back.to_json(), fit.to_json());
    /// assert_eq!(back.coefficients(), fit.coefficients());
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("glmm", "prospicio-glm", env!("CARGO_PKG_VERSION"), ""),
            "spec": spec_json(&self.spec),
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "covariance": nums(&self.covariance),
            "group_names": self.group_names,
            "term_names": self.term_names,
            "sd": nums(&self.sd),
            "sd_standard_errors": nums(&self.sd_standard_errors),
            "correlations": matrix_json(&self.correlations),
            "parameters": nums(&self.parameters),
            "parameter_covariance": nums(&self.parameter_covariance),
            "dispersion": num(self.dispersion),
            "dispersion_standard_error": num(self.dispersion_standard_error),
            "log_likelihood": num(self.log_likelihood),
            "n_obs": self.n_obs,
            "fitted": nums(&self.fitted),
            "random_effects": self.random_effects.iter().map(effects_json).collect::<Vec<_>>(),
            "iterations": self.iterations,
            "converged": self.converged,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        input_hash(top)?;
        let spec = spec_from(object(field(top, "spec")?, "spec")?)?;
        let names = strings(top, "names")?;
        let coefficients = floats(top, "coefficients")?;
        let covariance = floats(top, "covariance")?;
        let group_names = strings(top, "group_names")?;
        let term_names = strings(top, "term_names")?;
        let sd = floats(top, "sd")?;
        let sd_standard_errors = floats(top, "sd_standard_errors")?;
        let parameters = floats(top, "parameters")?;
        let parameter_covariance = floats(top, "parameter_covariance")?;
        let fitted = floats(top, "fitted")?;
        let n_obs = uint(top, "n_obs")? as usize;
        let (p, k, m) = (names.len(), sd.len(), parameters.len());
        if coefficients.len() != p
            || covariance.len() != p * p
            || group_names.len() != k
            || term_names.len() != k
            || sd_standard_errors.len() != k
            || parameter_covariance.len() != m * m
            || fitted.len() != n_obs
        {
            return Err(Error::Data(
                "model artifact sizes disagree: names, coefficients, covariances, variance \
                 components and fitted"
                    .into(),
            ));
        }
        let random_effects = array(top, "random_effects")?
            .iter()
            .map(effects_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(GlmmFit {
            spec,
            names,
            coefficients,
            covariance,
            group_names,
            term_names,
            sd,
            sd_standard_errors,
            correlations: matrix_from(top, "correlations")?,
            parameters,
            parameter_covariance,
            dispersion: float(field(top, "dispersion")?, "dispersion")?,
            dispersion_standard_error: float(
                field(top, "dispersion_standard_error")?,
                "dispersion_standard_error",
            )?,
            log_likelihood: float(field(top, "log_likelihood")?, "log_likelihood")?,
            n_obs,
            fitted,
            random_effects,
            iterations: uint(top, "iterations")? as usize,
            converged: field(top, "converged")?
                .as_bool()
                .ok_or_else(|| bad("converged"))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::Grouping;
    use prospicio_models::{Design, Family};

    fn data() -> (Design, Vec<Grouping>, Vec<f64>) {
        let n = 48;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.5).collect();
        let g: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
        let h: Vec<String> = (0..n).map(|i| format!("h{}", i % 3)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (1.0 + ((i * 13) % 7) as f64 + (i % 8) as f64 * 0.5 + x[i]).round())
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x],
        )
        .unwrap();
        (d, vec![Grouping::new("g", g), Grouping::new("h", h)], y)
    }

    #[test]
    fn laplace_and_quadrature_fits_round_trip_and_predict_alike() {
        let (d, groups, y) = data();
        for (spec, g) in [
            (Glmm::new(Family::Gamma), &groups[..]),
            (Glmm::new(Family::Poisson).quadrature(5), &groups[..1]),
        ] {
            let fit = spec.fit(&d, g, &y).unwrap();
            let back = GlmmFit::from_json(&fit.to_json()).unwrap();
            // A variance component at zero leaves NaN standard errors,
            // which `==` never matches: compare the documents instead.
            assert_eq!(back.to_json(), fit.to_json());
            assert_eq!(back.predict(&d, g).unwrap(), fit.predict(&d, g).unwrap());
        }
    }

    #[test]
    fn refuses_other_documents() {
        let (d, groups, y) = data();
        let j = Glmm::new(Family::Poisson)
            .fit(&d, &groups[..1], &y)
            .unwrap()
            .to_json();
        assert!(GlmmFit::from_json(&j.replace("risk_rs.glmm_fit", "risk_rs.glm_fit")).is_err());
        assert!(
            GlmmFit::from_json(&j.replace("\"format_version\":1", "\"format_version\":2")).is_err()
        );
        assert!(
            GlmmFit::from_json(&j.replace("\"name\":\"poisson\"", "\"name\":\"binomial\""))
                .is_err()
        );
        assert!(GlmmFit::from_json(&j.replace("\"levels\":[\"g0\",", "\"levels\":[")).is_err());
    }
}
