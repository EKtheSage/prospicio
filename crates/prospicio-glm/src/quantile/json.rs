//! Saving and loading fitted quantile and expectile regressions: versioned
//! JSON artifacts in the envelope the other GLM artifacts share
//! (`docs/design/models.md`, "Model artifacts"; #978).
//!
//! The artifact holds the level, the coefficient names, the estimates and
//! their covariance, and the fit statistics; numbers round-trip bit for
//! bit. The fits keep no hash of their training data, so the provenance's
//! `input_hash` is empty.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, field, float, floats, input_hash, num, nums, object, parse, strings, uint,
};
use serde_json::{Map, Value, json};

use super::{ExpectileFit, QuantileFit};

const QUANTILE_FORMAT: &str = "risk_rs.quantile_fit";
const EXPECTILE_FORMAT: &str = "risk_rs.expectile_fit";
/// The newest version of each format this build reads.
const FORMAT_VERSION: u64 = 1;

fn provenance(model: &str) -> Value {
    artifact::provenance(model, "prospicio-glm", env!("CARGO_PKG_VERSION"), "")
}

/// The names, coefficients and covariance, checked to agree in size.
fn estimates(top: &Map<String, Value>) -> Result<(Vec<String>, Vec<f64>, Vec<f64>)> {
    let names = strings(top, "names")?;
    let coefficients = floats(top, "coefficients")?;
    let covariance = floats(top, "covariance")?;
    let p = names.len();
    if coefficients.len() != p || covariance.len() != p * p {
        return Err(Error::Data(
            "model artifact sizes disagree: names, coefficients and covariance".into(),
        ));
    }
    Ok((names, coefficients, covariance))
}

impl QuantileFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::quantile::{QuantileFit, QuantileRegression};
    /// use prospicio_models::{Design, Model};
    ///
    /// let d = Design::new(
    ///     vec!["(Intercept)".into(), "x".into()],
    ///     vec![vec![1.0; 6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
    /// )
    /// .unwrap();
    /// let fit = QuantileRegression::new(0.5).fit(&d, &[1.0, 2.5, 2.0, 4.0, 5.5, 5.0]).unwrap();
    /// assert_eq!(QuantileFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": QUANTILE_FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": provenance("quantile_regression"),
            "tau": num(self.tau),
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "covariance": nums(&self.covariance),
            "bandwidth": num(self.bandwidth),
            "objective": num(self.objective),
            "iterations": self.iterations,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, QUANTILE_FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        input_hash(top)?;
        let (names, coefficients, covariance) = estimates(top)?;
        Ok(QuantileFit {
            tau: float(field(top, "tau")?, "tau")?,
            names,
            coefficients,
            covariance,
            bandwidth: float(field(top, "bandwidth")?, "bandwidth")?,
            objective: float(field(top, "objective")?, "objective")?,
            iterations: uint(top, "iterations")? as usize,
        })
    }
}

impl ExpectileFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::quantile::{ExpectileFit, ExpectileRegression};
    /// use prospicio_models::{Design, Model};
    ///
    /// let d = Design::new(
    ///     vec!["(Intercept)".into(), "x".into()],
    ///     vec![vec![1.0; 6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
    /// )
    /// .unwrap();
    /// let fit = ExpectileRegression::new(0.8).fit(&d, &[1.0, 2.5, 2.0, 4.0, 5.5, 5.0]).unwrap();
    /// assert_eq!(ExpectileFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": EXPECTILE_FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": provenance("expectile_regression"),
            "tau": num(self.tau),
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "covariance": nums(&self.covariance),
            "objective": num(self.objective),
            "iterations": self.iterations,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, EXPECTILE_FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        input_hash(top)?;
        let (names, coefficients, covariance) = estimates(top)?;
        Ok(ExpectileFit {
            tau: float(field(top, "tau")?, "tau")?,
            names,
            coefficients,
            covariance,
            objective: float(field(top, "objective")?, "objective")?,
            iterations: uint(top, "iterations")? as usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantile::{ExpectileRegression, QuantileRegression};
    use prospicio_models::{Design, Model};

    fn data() -> (Design, Vec<f64>) {
        let x: Vec<f64> = (0..40).map(|i| f64::from(i) / 4.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| 1.0 + 0.5 * x + ((i * 37) % 11) as f64 / 3.0)
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 40], x],
        )
        .unwrap();
        (d, y)
    }

    #[test]
    fn quantile_fits_round_trip_and_predict_alike() {
        let (d, y) = data();
        let fit = QuantileRegression::new(0.9).fit(&d, &y).unwrap();
        let back = QuantileFit::from_json(&fit.to_json()).unwrap();
        assert_eq!(back, fit);
        assert_eq!(
            prospicio_models::Fitted::predict(&back, &d).unwrap(),
            prospicio_models::Fitted::predict(&fit, &d).unwrap()
        );
    }

    #[test]
    fn expectile_fits_round_trip() {
        let (d, y) = data();
        let fit = ExpectileRegression::new(0.2).fit(&d, &y).unwrap();
        assert_eq!(ExpectileFit::from_json(&fit.to_json()).unwrap(), fit);
    }

    #[test]
    fn refuses_other_documents() {
        let (d, y) = data();
        let q = QuantileRegression::new(0.5).fit(&d, &y).unwrap().to_json();
        assert!(ExpectileFit::from_json(&q).is_err());
        assert!(
            QuantileFit::from_json(&q.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(
            QuantileFit::from_json(&q.replace("\"names\":[\"(Intercept)\",", "\"names\":["))
                .is_err()
        );
    }
}
