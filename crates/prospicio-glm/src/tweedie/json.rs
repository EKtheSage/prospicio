//! Saving and loading Tweedie fits: a versioned JSON artifact in the
//! envelope the other GLM artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the estimated power and dispersion, the maximized
//! log-likelihood, the power's interval with which ends are open, whether
//! the power hit a bound, the number of profile evaluations, and the GLM
//! at that power as a nested `risk_rs.glm_fit` artifact, which keeps its
//! own hash of the training data. Numbers round-trip bit for bit.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{self, bad, field, float, floats, num, nums, object, parse, uint};
use serde_json::{Map, Value, json};

use super::TweedieFit;
use crate::GlmFit;

const FORMAT: &str = "risk_rs.tweedie_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

fn flag(o: &Map<String, Value>, key: &str) -> Result<bool> {
    field(o, key)?.as_bool().ok_or_else(|| bad(key))
}

impl TweedieFit {
    /// The fit as a versioned JSON artifact (format `risk_rs.tweedie_fit`),
    /// which [`TweedieFit::from_json`] reads back; the GLM at the
    /// estimated power is nested as a `risk_rs.glm_fit` artifact.
    ///
    /// ```
    /// use prospicio_glm::tweedie::{TweedieFit, TweedieGlm};
    /// use prospicio_models::{Design, Link, Model};
    ///
    /// let n = 60;
    /// let x: Vec<f64> = (0..n).map(|i| f64::from(i % 3)).collect();
    /// let y: Vec<f64> = (0..n)
    ///     .map(|i| if i % 4 == 0 { 0.0 } else { 1.0 + f64::from((i * 7) % 5) + x[i as usize] })
    ///     .collect();
    /// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; n as usize], x])
    ///     .unwrap();
    /// let fit = TweedieGlm::new(Link::Log).fit(&d, &y).unwrap();
    /// let back = TweedieFit::from_json(&fit.to_json()).unwrap();
    /// assert_eq!(back.to_json(), fit.to_json());
    /// assert_eq!(back.power(), fit.power());
    /// ```
    pub fn to_json(&self) -> String {
        let glm: Value = serde_json::from_str(&self.glm.to_json()).expect("a GLM artifact is JSON");
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("tweedie", "prospicio-glm", env!("CARGO_PKG_VERSION"), ""),
            "power": num(self.power),
            "dispersion": num(self.phi),
            "log_likelihood": num(self.log_likelihood),
            "interval": nums(&[self.interval.0, self.interval.1]),
            "interval_open": [self.interval_open.0, self.interval_open.1],
            "at_boundary": self.at_boundary,
            "evaluations": self.evaluations,
            "glm": glm,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        let interval = match floats(top, "interval")?.as_slice() {
            [lo, hi] => (*lo, *hi),
            _ => return Err(bad("interval")),
        };
        let open = field(top, "interval_open")?
            .as_array()
            .ok_or_else(|| bad("interval_open"))?;
        let interval_open = match open.as_slice() {
            [Value::Bool(lo), Value::Bool(hi)] => (*lo, *hi),
            _ => return Err(bad("interval_open")),
        };
        let power = float(field(top, "power")?, "power")?;
        if !power.is_finite() {
            return Err(bad("power"));
        }
        let glm = GlmFit::from_json(&field(top, "glm")?.to_string())?;
        if glm.spec().family != (prospicio_models::Family::Tweedie { power }) {
            return Err(Error::Data(
                "model artifact fields disagree: power and the nested GLM's family".into(),
            ));
        }
        Ok(TweedieFit {
            power,
            phi: float(field(top, "dispersion")?, "dispersion")?,
            log_likelihood: float(field(top, "log_likelihood")?, "log_likelihood")?,
            interval,
            interval_open,
            at_boundary: flag(top, "at_boundary")?,
            evaluations: usize::try_from(uint(top, "evaluations")?)
                .map_err(|_| bad("evaluations"))?,
            glm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tweedie::TweedieGlm;
    use prospicio_models::{Design, Link, Model};

    fn fit() -> TweedieFit {
        let n = 80;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    0.5 + f64::from((i * 7) % 5) * (1.0 + x[i as usize])
                }
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n as usize], x],
        )
        .unwrap();
        TweedieGlm::new(Link::Log).fit(&d, &y).unwrap()
    }

    #[test]
    fn round_trips_bit_for_bit() {
        let fit = fit();
        let back = TweedieFit::from_json(&fit.to_json()).unwrap();
        assert_eq!(back, fit);
        assert_eq!(back.to_json(), fit.to_json());
    }

    #[test]
    fn bad_artifacts_are_refused() {
        let j = fit().to_json();
        assert!(
            TweedieFit::from_json(&j.replace("risk_rs.tweedie_fit", "risk_rs.glm_fit")).is_err()
        );
        assert!(
            TweedieFit::from_json(&j.replacen("\"format_version\":1", "\"format_version\":2", 1))
                .is_err()
        );
        assert!(
            TweedieFit::from_json(&j.replace("\"at_boundary\":false", "\"at_boundary\":1"))
                .is_err()
        );
        assert!(TweedieFit::from_json("[]").is_err());
    }
}
