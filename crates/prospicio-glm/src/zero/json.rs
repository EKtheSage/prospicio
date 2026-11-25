//! Saving and loading zero-inflated and hurdle fits: a versioned JSON
//! artifact in the envelope the other GLM artifacts share
//! (`docs/design/models.md`, "Model artifacts"; #978).
//!
//! The artifact holds the kind, the count distribution and the zero part's
//! link, both parts' names, the parameters `(β, γ, ln θ)` and their
//! covariance, the pointwise log-likelihood, the fitted means and the fit
//! statistics; numbers round-trip bit for bit. The fit keeps no hash of its
//! training data, so the provenance's `input_hash` is empty.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, bad, field, float, floats, input_hash, link_from, link_json, num, nums, object, parse,
    strings, uint,
};
use serde_json::json;

use super::{CountDistribution, TwoPart, TwoPartFit};

const FORMAT: &str = "risk_rs.two_part_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

impl TwoPartFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::zero::{Hurdle, TwoPartFit};
    /// use prospicio_models::{Design, Model};
    ///
    /// let x: Vec<f64> = (0..60).map(|i| f64::from(i % 4) / 3.0).collect();
    /// let y: Vec<f64> = (0..60)
    ///     .map(|i| if i % 3 == 0 { 0.0 } else { f64::from(1 + (i * 7) % 5) })
    ///     .collect();
    /// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 60], x]).unwrap();
    /// let fit = Hurdle::poisson().fit(&d, &y).unwrap();
    /// assert_eq!(TwoPartFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance(self.kind.name(), "prospicio-glm", env!("CARGO_PKG_VERSION"), ""),
            "kind": self.kind.name(),
            "count": self.count.name(),
            "zero_link": link_json(self.zero_link),
            "count_names": self.count_names,
            "zero_names": self.zero_names,
            "parameters": nums(&self.par),
            "covariance": nums(&self.covariance),
            "pointwise": nums(&self.pointwise),
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
        let doc = parse(text, FORMAT, FORMAT_VERSION)?;
        let top = object(&doc, "artifact")?;
        input_hash(top)?;
        let kind = match field(top, "kind")?.as_str() {
            Some("zero_inflated") => TwoPart::ZeroInflated,
            Some("hurdle") => TwoPart::Hurdle,
            _ => return Err(bad("kind")),
        };
        let count = field(top, "count")?
            .as_str()
            .and_then(CountDistribution::from_name)
            .ok_or_else(|| bad("count"))?;
        let zero_link = link_from(object(field(top, "zero_link")?, "zero_link")?)?;
        let count_names = strings(top, "count_names")?;
        let zero_names = strings(top, "zero_names")?;
        let par = floats(top, "parameters")?;
        let covariance = floats(top, "covariance")?;
        let pointwise = floats(top, "pointwise")?;
        let fitted = floats(top, "fitted")?;
        let k = count_names.len()
            + zero_names.len()
            + usize::from(count == CountDistribution::NegativeBinomial);
        if par.len() != k || covariance.len() != k * k || pointwise.len() != fitted.len() {
            return Err(Error::Data(
                "model artifact sizes disagree: names, parameters, covariance, pointwise and \
                 fitted"
                    .into(),
            ));
        }
        Ok(TwoPartFit {
            kind,
            count,
            zero_link,
            count_names,
            zero_names,
            par,
            covariance,
            pointwise,
            log_likelihood: float(field(top, "log_likelihood")?, "log_likelihood")?,
            n_obs: uint(top, "n_obs")? as usize,
            iterations: uint(top, "iterations")? as usize,
            fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero::{Hurdle, ZeroInflated};
    use prospicio_models::{Design, Fitted, Model};

    fn data() -> (Design, Vec<f64>) {
        let x: Vec<f64> = (0..120).map(|i| f64::from(i % 4) / 3.0).collect();
        let y: Vec<f64> = (0..120)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    f64::from(((i * 13) % 11) * (i % 5 + 1) / 2)
                }
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 120], x],
        )
        .unwrap();
        (d, y)
    }

    #[test]
    fn every_kind_and_count_round_trips() {
        let (d, y) = data();
        let fits = [
            ZeroInflated::poisson().fit(&d, &y).unwrap(),
            ZeroInflated::negative_binomial().fit(&d, &y).unwrap(),
            Hurdle::poisson().fit(&d, &y).unwrap(),
            Hurdle::negative_binomial()
                .zero_link(prospicio_models::Link::Probit)
                .fit(&d, &y)
                .unwrap(),
        ];
        for fit in fits {
            let back = TwoPartFit::from_json(&fit.to_json()).unwrap();
            assert_eq!(back, fit);
            assert_eq!(back.predict(&d).unwrap(), fit.predict(&d).unwrap());
        }
    }

    #[test]
    fn refuses_other_documents() {
        let (d, y) = data();
        let j = ZeroInflated::poisson().fit(&d, &y).unwrap().to_json();
        assert!(
            TwoPartFit::from_json(&j.replace("risk_rs.two_part_fit", "risk_rs.glm_fit")).is_err()
        );
        assert!(
            TwoPartFit::from_json(&j.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        // A Poisson artifact relabelled negative binomial lacks ln θ.
        assert!(
            TwoPartFit::from_json(
                &j.replace("\"count\":\"poisson\"", "\"count\":\"negative_binomial\"")
            )
            .is_err()
        );
    }
}
