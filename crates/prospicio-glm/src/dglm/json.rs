//! Saving and loading double GLM fits: a versioned JSON artifact in the
//! envelope the other GLM artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the family, link and method, both linear predictors'
//! names, estimates and covariances, the fitted means and dispersions, and
//! the fit statistics; numbers round-trip bit for bit. The fit keeps no
//! hash of its training data, so the provenance's `input_hash` is empty.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, bad, family_from, family_json, field, float, floats, input_hash, link_from, link_json,
    num, nums, object, parse, strings, uint,
};
use serde_json::json;

use super::{DispersionMethod, DoubleGlmFit};

const FORMAT: &str = "risk_rs.double_glm_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

impl DoubleGlmFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::dglm::{DoubleGlm, DoubleGlmFit};
    /// use prospicio_models::{Design, Family, Link};
    ///
    /// let urban = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    /// let d = Design::new(vec!["(Intercept)".into(), "urban".into()], vec![vec![1.0; 8], urban])
    ///     .unwrap();
    /// let y = [9.0, 11.0, 10.0, 10.5, 4.0, 18.0, 6.0, 15.0];
    /// let fit = DoubleGlm::new(Family::Gamma, Link::Log).fit(&d, &d, &y).unwrap();
    /// assert_eq!(DoubleGlmFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("double_glm", "prospicio-glm", env!("CARGO_PKG_VERSION"), ""),
            "family": family_json(self.family),
            "link": link_json(self.link),
            "method": match self.method {
                DispersionMethod::Ml => "ml",
                DispersionMethod::Reml => "reml",
            },
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "covariance": nums(&self.covariance),
            "dispersion_names": self.dispersion_names,
            "dispersion_coefficients": nums(&self.dispersion_coefficients),
            "dispersion_covariance": nums(&self.dispersion_covariance),
            "fitted": nums(&self.fitted),
            "dispersion_fitted": nums(&self.dispersion_fitted),
            "minus_two_log_likelihood": num(self.minus_two_log_likelihood),
            "reml_criterion": num(self.reml_criterion),
            "iterations": self.iterations,
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
        let family = family_from(object(field(top, "family")?, "family")?)?;
        family.validate()?;
        let link = link_from(object(field(top, "link")?, "link")?)?;
        let method = match field(top, "method")?.as_str() {
            Some("ml") => DispersionMethod::Ml,
            Some("reml") => DispersionMethod::Reml,
            _ => return Err(bad("method")),
        };
        let names = strings(top, "names")?;
        let coefficients = floats(top, "coefficients")?;
        let covariance = floats(top, "covariance")?;
        let dispersion_names = strings(top, "dispersion_names")?;
        let dispersion_coefficients = floats(top, "dispersion_coefficients")?;
        let dispersion_covariance = floats(top, "dispersion_covariance")?;
        let fitted = floats(top, "fitted")?;
        let dispersion_fitted = floats(top, "dispersion_fitted")?;
        let (p, q) = (names.len(), dispersion_names.len());
        if coefficients.len() != p
            || covariance.len() != p * p
            || dispersion_coefficients.len() != q
            || dispersion_covariance.len() != q * q
            || dispersion_fitted.len() != fitted.len()
        {
            return Err(Error::Data(
                "model artifact sizes disagree: names, coefficients, covariances and fitted".into(),
            ));
        }
        Ok(DoubleGlmFit {
            family,
            link,
            method,
            names,
            coefficients,
            covariance,
            dispersion_names,
            dispersion_coefficients,
            dispersion_covariance,
            fitted,
            dispersion_fitted,
            minus_two_log_likelihood: float(
                field(top, "minus_two_log_likelihood")?,
                "minus_two_log_likelihood",
            )?,
            reml_criterion: float(field(top, "reml_criterion")?, "reml_criterion")?,
            iterations: uint(top, "iterations")? as usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dglm::DoubleGlm;
    use prospicio_models::{Design, Family, Link};

    fn data() -> (Design, Vec<f64>) {
        let urban: Vec<f64> = (0..24).map(|i| f64::from(i % 2)).collect();
        let x: Vec<f64> = (0..24).map(|i| f64::from(i) / 8.0).collect();
        let y: Vec<f64> = (0..24)
            .map(|i| {
                5.0 + f64::from(i % 2) * 3.0 + f64::from((i * 7) % 5) * (1.0 + f64::from(i % 2))
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "urban".into(), "x".into()],
            vec![vec![1.0; 24], urban, x],
        )
        .unwrap();
        (d, y)
    }

    #[test]
    fn ml_and_reml_fits_round_trip() {
        let (d, y) = data();
        for spec in [
            DoubleGlm::new(Family::Gamma, Link::Log),
            DoubleGlm::new(Family::Tweedie { power: 1.5 }, Link::Log).reml(),
        ] {
            let fit = spec.fit(&d, &d, &y).unwrap();
            let back = DoubleGlmFit::from_json(&fit.to_json()).unwrap();
            assert_eq!(back, fit);
            assert_eq!(back.predict(&d).unwrap(), fit.predict(&d).unwrap());
        }
    }

    #[test]
    fn refuses_other_documents() {
        let (d, y) = data();
        let j = DoubleGlm::new(Family::Gamma, Link::Log)
            .fit(&d, &d, &y)
            .unwrap()
            .to_json();
        assert!(
            DoubleGlmFit::from_json(&j.replace("risk_rs.double_glm_fit", "risk_rs.glm_fit"))
                .is_err()
        );
        assert!(
            DoubleGlmFit::from_json(&j.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(
            DoubleGlmFit::from_json(&j.replace("\"method\":\"ml\"", "\"method\":\"em\"")).is_err()
        );
        assert!(
            DoubleGlmFit::from_json(&j.replace("\"names\":[\"(Intercept)\",", "\"names\":["))
                .is_err()
        );
    }
}
