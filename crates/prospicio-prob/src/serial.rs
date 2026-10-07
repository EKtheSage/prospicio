//! Saving and loading distributions: [`Dist::to_json`] and
//! [`Dist::from_json`], a versioned JSON document per distribution.
//!
//! The document holds the family and the parameters the family's
//! constructor takes, so a loaded distribution is rebuilt by the same
//! validated constructor and equals the saved one. Numbers round-trip bit
//! for bit; a non-finite value is written as `"NaN"`, `"inf"` or `"-inf"`.
//! A mixture is saved with its components, so it must have been built from
//! native severities ([`Mixture::from_dists`], which the bindings use). A
//! [`Custom`](crate::Custom) cannot be saved: it is a function in the
//! caller's language.
//!
//! ```json
//! {"format": "risk_rs.distribution", "format_version": 1,
//!  "family": "lognormal", "meanlog": 7.0, "sdlog": 0.5}
//! ```

use std::sync::Arc;

use prospicio_core::{Error, Result};
use serde_json::{Map, Number, Value, json};

use crate::dist::{Dist, SeverityDist};
use crate::evt::Gpd;
use crate::sampled::Empirical;
use crate::{
    Gamma, Grid, LogAffinePareto, Loglogistic, Lognormal, Mixture, Pareto, PiecewisePareto,
    Sampled, Truncation, Tweedie, Weibull,
};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.distribution";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

impl Dist {
    /// The distribution as a JSON document; [`from_json`](Self::from_json)
    /// reads it back to an equal distribution.
    ///
    /// Fails for a [`Custom`](crate::Custom) (a function in the caller's
    /// language) and for a mixture built from trait objects rather than
    /// with [`Mixture::from_dists`].
    ///
    /// ```
    /// use prospicio_prob::{Dist, Distribution, Lognormal};
    ///
    /// let d = Dist::from(Lognormal::new(7.0, 0.5).unwrap());
    /// let back = Dist::from_json(&d.to_json().unwrap()).unwrap();
    /// assert_eq!(back.family(), "lognormal");
    /// assert_eq!(back.mean(), d.mean());
    /// ```
    pub fn to_json(&self) -> Result<String> {
        let mut doc = Map::new();
        doc.insert("format".into(), json!(FORMAT));
        doc.insert("format_version".into(), json!(FORMAT_VERSION));
        for (k, v) in body(self)? {
            doc.insert(k, v);
        }
        Ok(serde_json::to_string(&Value::Object(doc)).expect("a JSON value serializes"))
    }

    /// Reads a document written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, an unknown
    /// family, or parameters the family's constructor refuses.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc: Value = serde_json::from_str(text)
            .map_err(|e| Error::Data(format!("not a JSON document: {e}")))?;
        let top = object(&doc, "document")?;
        match top.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            other => {
                return Err(Error::Data(format!(
                    "not a distribution document: format {other:?}, expected {FORMAT:?}"
                )));
            }
        }
        let version = top
            .get("format_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Data("format_version is missing".into()))?;
        if version > FORMAT_VERSION {
            return Err(Error::Data(format!(
                "format_version {version} is newer than this build reads ({FORMAT_VERSION})"
            )));
        }
        from_body(top)
    }
}

/// The family and parameters, without the format fields.
fn body(d: &Dist) -> Result<Map<String, Value>> {
    let mut m = Map::new();
    m.insert("family".into(), json!(d.family()));
    let mut put = |k: &str, v: Value| {
        m.insert(k.into(), v);
    };
    match d {
        Dist::Lognormal(x) => {
            put("meanlog", num(x.meanlog()));
            put("sdlog", num(x.sdlog()));
        }
        Dist::Pareto(x) => {
            put("t", num(x.t()));
            put("alpha", num(x.alpha()));
            put("truncation", x.truncation().map_or(Value::Null, num));
        }
        Dist::PiecewisePareto(x) => {
            put("t", nums(x.thresholds()));
            put("alpha", nums(x.alphas()));
            match x.truncation() {
                Some((t, kind)) => {
                    put("truncation", num(t));
                    put("truncation_type", json!(truncation_name(kind)));
                }
                None => put("truncation", Value::Null),
            }
        }
        Dist::LogAffinePareto(x) => {
            put("t", num(x.t()));
            put("alpha0", num(x.alpha0()));
            put("gamma", num(x.gamma()));
        }
        Dist::GeneralizedPareto(x) => {
            put("xi", num(x.xi()));
            put("beta", num(x.beta()));
            put("location", num(x.location()));
        }
        Dist::Gamma(x) => {
            put("shape", num(x.shape()));
            put("scale", num(x.scale()));
        }
        Dist::Tweedie(x) => {
            put("mean", num(x.mean_param()));
            put("dispersion", num(x.dispersion()));
            put("power", num(x.power()));
        }
        Dist::Weibull(x) => {
            put("shape", num(x.shape()));
            put("scale", num(x.scale()));
        }
        Dist::Loglogistic(x) => {
            put("shape", num(x.shape()));
            put("scale", num(x.scale()));
        }
        Dist::Mixture(x) => {
            let dists = x.dists().ok_or_else(|| {
                Error::Data(
                    "this mixture was built from trait objects; build it with \
                     Mixture::from_dists to save it"
                        .into(),
                )
            })?;
            put("weights", nums(x.weights()));
            let parts = dists
                .iter()
                .map(|c| body(c.dist()).map(Value::Object))
                .collect::<Result<Vec<_>>>()?;
            put("components", Value::Array(parts));
        }
        Dist::Grid(x) => {
            put("step", num(x.step()));
            put("probs", nums(x.probs()));
        }
        Dist::Sampled(x) => {
            put("draws", nums(x.draws()));
        }
        Dist::Custom(x) => {
            return Err(Error::Data(format!(
                "custom distribution {:?} is a function in the caller's language and \
                 cannot be saved",
                x.name()
            )));
        }
    }
    Ok(m)
}

fn from_body(m: &Map<String, Value>) -> Result<Dist> {
    let family = m
        .get("family")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Data("family is missing".into()))?;
    let f = |k: &str| float(m, k);
    Ok(match family {
        "lognormal" => Lognormal::new(f("meanlog")?, f("sdlog")?)?.into(),
        "pareto" => {
            let p = Pareto::new(f("t")?, f("alpha")?)?;
            match optional(m, "truncation")? {
                Some(t) => p.truncated(t)?,
                None => p,
            }
            .into()
        }
        "piecewise_pareto" => {
            let p = PiecewisePareto::new(floats(m, "t")?, floats(m, "alpha")?)?;
            match optional(m, "truncation")? {
                Some(t) => {
                    let kind = match m.get("truncation_type").and_then(Value::as_str) {
                        Some("lp") => Truncation::LastPiece,
                        Some("wd") => Truncation::WholeDistribution,
                        other => {
                            return Err(Error::Data(format!(
                                "truncation_type must be \"lp\" or \"wd\", got {other:?}"
                            )));
                        }
                    };
                    p.truncated(t, kind)?
                }
                None => p,
            }
            .into()
        }
        "log_affine_pareto" => LogAffinePareto::new(f("t")?, f("alpha0")?, f("gamma")?)?.into(),
        "generalized_pareto" => Gpd::new(f("xi")?, f("beta")?)?
            .shifted(f("location")?)?
            .into(),
        "gamma" => Gamma::new(f("shape")?, f("scale")?)?.into(),
        "tweedie" => Tweedie::new(f("mean")?, f("dispersion")?, f("power")?)?.into(),
        "weibull" => Weibull::new(f("shape")?, f("scale")?)?.into(),
        "loglogistic" => Loglogistic::new(f("shape")?, f("scale")?)?.into(),
        "mixture" => {
            let weights = floats(m, "weights")?;
            let comps = m
                .get("components")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Data("components is missing".into()))?;
            if comps.len() != weights.len() {
                return Err(Error::Data("one weight per mixture component".into()));
            }
            let parts = weights
                .into_iter()
                .zip(comps)
                .map(|(w, c)| {
                    let d = from_body(object(c, "component")?)?;
                    let s = SeverityDist::try_from(d).map_err(|_| {
                        Error::Data("a mixture component cannot be sampled draws".into())
                    })?;
                    Ok((w, s))
                })
                .collect::<Result<Vec<_>>>()?;
            Dist::Mixture(Arc::new(Mixture::from_dists(parts)?))
        }
        "grid" => Grid::new(f("step")?, floats(m, "probs")?)?.into(),
        "sampled" => Sampled::new(floats(m, "draws")?)?.into(),
        other => {
            return Err(Error::Data(format!("unknown family {other:?}")));
        }
    })
}

fn truncation_name(kind: Truncation) -> &'static str {
    match kind {
        Truncation::LastPiece => "lp",
        Truncation::WholeDistribution => "wd",
    }
}

fn num(x: f64) -> Value {
    match Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None if x.is_nan() => json!("NaN"),
        None if x > 0.0 => json!("inf"),
        None => json!("-inf"),
    }
}

fn nums(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| num(x)).collect())
}

fn to_f64(v: &Value, name: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
    .ok_or_else(|| Error::Data(format!("{name} must be a number")))
}

fn float(m: &Map<String, Value>, name: &str) -> Result<f64> {
    to_f64(
        m.get(name)
            .ok_or_else(|| Error::Data(format!("{name} is missing")))?,
        name,
    )
}

fn optional(m: &Map<String, Value>, name: &str) -> Result<Option<f64>> {
    match m.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => to_f64(v, name).map(Some),
    }
}

fn floats(m: &Map<String, Value>, name: &str) -> Result<Vec<f64>> {
    m.get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Data(format!("{name} must be a list of numbers")))?
        .iter()
        .map(|v| to_f64(v, name))
        .collect()
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error::Data(format!("the {what} must be a JSON object")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Severity};

    fn every() -> Vec<Dist> {
        let ln = Lognormal::new(7.0, 0.5).unwrap();
        vec![
            ln.into(),
            Pareto::new(1e5, 1.5).unwrap().into(),
            Pareto::new(1e5, 1.5)
                .unwrap()
                .truncated(1e7)
                .unwrap()
                .into(),
            PiecewisePareto::new(vec![1.0, 10.0, 100.0], vec![1.2, 1.8, 2.5])
                .unwrap()
                .into(),
            PiecewisePareto::new(vec![1.0, 10.0], vec![1.2, 1.8])
                .unwrap()
                .truncated(1000.0, Truncation::WholeDistribution)
                .unwrap()
                .into(),
            LogAffinePareto::new(100.0, 1.5, 0.3).unwrap().into(),
            Gpd::new(0.25, 3.0).unwrap().shifted(10.0).unwrap().into(),
            Gamma::new(2.0, 500.0).unwrap().into(),
            Tweedie::new(1000.0, 2.0, 1.5).unwrap().into(),
            Weibull::new(1.5, 1000.0).unwrap().into(),
            Loglogistic::new(4.0, 900.0).unwrap().into(),
            Dist::Mixture(Arc::new(
                Mixture::from_dists(vec![
                    (0.7, SeverityDist::try_from(Dist::from(ln)).unwrap()),
                    (
                        0.3,
                        SeverityDist::try_from(Dist::from(Pareto::new(1e5, 2.0).unwrap())).unwrap(),
                    ),
                ])
                .unwrap(),
            )),
            Grid::new(0.5, vec![0.1, 0.4, 0.3, 0.2]).unwrap().into(),
            Sampled::new(vec![3.0, 1.0, 2.0, 0.1 + 0.2]).unwrap().into(),
        ]
    }

    #[test]
    fn every_family_round_trips_exactly() {
        for d in every() {
            let text = d.to_json().unwrap();
            let back = Dist::from_json(&text).unwrap();
            assert_eq!(back.family(), d.family(), "{text}");
            assert_eq!(back.to_json().unwrap(), text, "{text}");
            assert_eq!(back.mean().to_bits(), d.mean().to_bits(), "{text}");
            for p in [0.1, 0.5, 0.9] {
                let (a, b) = (back.quantile(p).unwrap(), d.quantile(p).unwrap());
                assert_eq!(a.to_bits(), b.to_bits(), "{text} at {p}");
            }
            if let (Some(a), Some(b)) = (back.as_severity(), d.as_severity()) {
                assert_eq!(a.lev(1500.0).to_bits(), b.lev(1500.0).to_bits(), "{text}");
            }
        }
    }

    #[test]
    fn refuses_what_it_cannot_save_or_read() {
        let custom = crate::Custom::new(
            "c",
            Arc::new(|x: f64| Ok((1.0 - (-x).exp()).max(0.0))),
            None,
            true,
        )
        .unwrap();
        assert!(Dist::from(custom).to_json().is_err());
        let boxed = Mixture::new(vec![(
            1.0,
            Box::new(Gamma::new(2.0, 1.0).unwrap()) as Box<dyn Severity + Send + Sync>,
        )])
        .unwrap();
        assert!(Dist::from(boxed).to_json().is_err());
        assert!(Dist::from_json("not json").is_err());
        assert!(Dist::from_json(r#"{"format": "risk_rs.glm_fit", "format_version": 1}"#).is_err());
        let newer = r#"{"format": "risk_rs.distribution", "format_version": 2, "family": "gamma", "shape": 2, "scale": 1}"#;
        assert!(Dist::from_json(newer).is_err());
        let bad = r#"{"format": "risk_rs.distribution", "format_version": 1, "family": "gamma", "shape": -2, "scale": 1}"#;
        assert!(Dist::from_json(bad).is_err());
        let unknown =
            r#"{"format": "risk_rs.distribution", "format_version": 1, "family": "cauchy"}"#;
        assert!(Dist::from_json(unknown).is_err());
    }
}
