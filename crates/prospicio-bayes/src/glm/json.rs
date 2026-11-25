//! Saving and loading sampled Bayesian GLMs: a versioned JSON artifact in
//! the envelope the GLM artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the specification (family, link, priors and sampler
//! settings, seed included), the coefficient names, every kept draw of the
//! coefficients and the dispersion, chain-major, and the number of
//! divergences; numbers round-trip bit for bit, so a loaded fit gives the
//! same summaries, predictions and pointwise log-likelihood. The fit keeps
//! no hash of its training data, so the provenance's `input_hash` is
//! empty.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, bad, family_from, family_json, field, float, floats, input_hash, link_from, link_json,
    num, nums, object, parse, strings, uint,
};
use serde_json::{Map, Value, json};

use super::{BayesGlm, BayesGlmFit, DispersionPrior, Sampler};

const FORMAT: &str = "risk_rs.bayes_glm_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

/// A sampler's settings as JSON, shared with the GLMM's artifact.
pub(crate) fn sampler_json(s: &Sampler) -> Value {
    json!({
        "chains": s.chains,
        "tune": s.tune,
        "draws": s.draws,
        "seed": s.seed,
        "target_accept": num(s.target_accept),
        "max_depth": s.max_depth,
    })
}

/// The sampler [`sampler_json`] wrote.
pub(crate) fn sampler_from(o: &Map<String, Value>) -> Result<Sampler> {
    let s = Sampler {
        chains: uint(o, "chains")? as usize,
        tune: uint(o, "tune")? as usize,
        draws: uint(o, "draws")? as usize,
        seed: uint(o, "seed")?,
        target_accept: float(field(o, "target_accept")?, "target_accept")?,
        max_depth: uint(o, "max_depth")?,
    };
    s.validate()?;
    Ok(s)
}

impl BayesGlmFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_bayes::glm::{BayesGlm, BayesGlmFit, Sampler};
    /// use prospicio_models::{Design, Family, Link, Model};
    ///
    /// let x: Vec<f64> = (0..20).map(|i| (i % 4) as f64 - 1.5).collect();
    /// let y: Vec<f64> = (0..20).map(|i| [1.0, 2.0, 3.0, 5.0][i % 4]).collect();
    /// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 20], x]).unwrap();
    /// let sampler = Sampler { chains: 1, tune: 50, draws: 20, ..Sampler::default() };
    /// let fit = BayesGlm::new(Family::Poisson, Link::Log).sampler(sampler).fit(&d, &y).unwrap();
    /// assert_eq!(BayesGlmFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let spec = &self.spec;
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("bayes_glm", "prospicio-bayes", env!("CARGO_PKG_VERSION"), ""),
            "spec": {
                "family": family_json(spec.family),
                "link": link_json(spec.link),
                "prior_sd": num(spec.prior_sd),
                "intercept_sd": num(spec.intercept_sd),
                "dispersion": match spec.dispersion {
                    DispersionPrior::Fixed(v) => json!({"kind": "fixed", "value": num(v)}),
                    DispersionPrior::HalfNormal(s) => json!({"kind": "half_normal", "scale": num(s)}),
                },
                "sampler": sampler_json(&spec.sampler),
            },
            "names": self.names,
            "coefficients": nums(&self.coefficients),
            "dispersion": nums(&self.dispersion),
            "divergences": self.divergences,
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
        let o = object(field(top, "spec")?, "spec")?;
        let d = object(field(o, "dispersion")?, "spec.dispersion")?;
        let dispersion = match field(d, "kind")?.as_str() {
            Some("fixed") => DispersionPrior::Fixed(float(field(d, "value")?, "dispersion.value")?),
            Some("half_normal") => {
                DispersionPrior::HalfNormal(float(field(d, "scale")?, "dispersion.scale")?)
            }
            _ => return Err(bad("spec.dispersion.kind")),
        };
        let spec = BayesGlm {
            family: family_from(object(field(o, "family")?, "family")?)?,
            link: link_from(object(field(o, "link")?, "link")?)?,
            prior_sd: float(field(o, "prior_sd")?, "prior_sd")?,
            intercept_sd: float(field(o, "intercept_sd")?, "intercept_sd")?,
            dispersion,
            sampler: sampler_from(object(field(o, "sampler")?, "sampler")?)?,
        };
        spec.check()?;
        let names = strings(top, "names")?;
        let coefficients = floats(top, "coefficients")?;
        let dispersion = floats(top, "dispersion")?;
        let n = spec.sampler.chains * spec.sampler.draws;
        if dispersion.len() != n || coefficients.len() != n * names.len() {
            return Err(Error::Data(
                "model artifact sizes disagree: the sampler's draws, names, coefficients and \
                 dispersion"
                    .into(),
            ));
        }
        Ok(BayesGlmFit {
            spec,
            names,
            coefficients,
            dispersion,
            divergences: uint(top, "divergences")? as usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_models::{Design, Family, Fitted, Link, Model};

    fn data() -> (Design, Vec<f64>) {
        let x: Vec<f64> = (0..30).map(|i| (i % 5) as f64 / 2.0 - 1.0).collect();
        let y: Vec<f64> = (0..30)
            .map(|i| 1.0 + (i % 5) as f64 * 0.7 + ((i * 7) % 3) as f64)
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 30], x],
        )
        .unwrap();
        (d, y)
    }

    fn quick() -> Sampler {
        Sampler {
            chains: 2,
            tune: 60,
            draws: 25,
            seed: 7,
            ..Sampler::default()
        }
    }

    #[test]
    fn fixed_and_sampled_dispersions_round_trip() {
        let (d, y) = data();
        for spec in [
            BayesGlm::new(Family::Poisson, Link::Log),
            BayesGlm::new(Family::Gamma, Link::Log),
        ] {
            let fit = spec.sampler(quick()).fit(&d, &y).unwrap();
            let back = BayesGlmFit::from_json(&fit.to_json()).unwrap();
            assert_eq!(back, fit);
            assert_eq!(back.predict(&d).unwrap(), fit.predict(&d).unwrap());
        }
    }

    #[test]
    fn refuses_other_documents() {
        let (d, y) = data();
        let j = BayesGlm::new(Family::Poisson, Link::Log)
            .sampler(quick())
            .fit(&d, &y)
            .unwrap()
            .to_json();
        assert!(
            BayesGlmFit::from_json(&j.replace("risk_rs.bayes_glm_fit", "risk_rs.glm_fit")).is_err()
        );
        assert!(
            BayesGlmFit::from_json(&j.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        // Fewer draws than the sampler says.
        assert!(BayesGlmFit::from_json(&j.replace("\"draws\":25", "\"draws\":26")).is_err());
        // A Poisson dispersion cannot be sampled.
        assert!(
            BayesGlmFit::from_json(&j.replace(
                "{\"kind\":\"fixed\",\"value\":1.0}",
                "{\"kind\":\"half_normal\",\"scale\":1.0}"
            ))
            .is_err()
        );
    }
}
