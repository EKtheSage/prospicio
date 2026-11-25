//! Saving and loading sampled Bayesian GLMMs: a versioned JSON artifact in
//! the envelope the model artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the specification (family, priors and sampler
//! settings), the fixed effects' names, the posterior draws of the
//! parameters and of the random effects, where each factor's effects sit
//! in those draws, and the Laplace fit the chains started from as a nested
//! `risk_rs.glmm_fit` artifact; numbers round-trip bit for bit, so a loaded
//! fit summarizes and predicts as the original. The fit keeps no hash of
//! its training data, so the provenance's `input_hash` is empty.

use prospicio_core::{Error, Result};
use prospicio_glm::glmm::GlmmFit;
use prospicio_models::artifact::{
    self, array, family_from, family_json, field, float, input_hash, num, object, parse, strings,
    uint,
};
use serde_json::{Value, json};

use super::{BayesGlmm, BayesGlmmFit, EffectsLayout, PosteriorDraws};
use crate::glm::json::{sampler_from, sampler_json};

const FORMAT: &str = "risk_rs.bayes_glmm_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

impl BayesGlmmFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    pub fn to_json(&self) -> String {
        let s = &self.spec;
        let layout: Vec<Value> = self
            .layout
            .iter()
            .map(|l| {
                json!({
                    "name": l.name,
                    "terms": l.terms,
                    "levels": l.levels,
                    "start": l.start,
                    "d": l.d,
                })
            })
            .collect();
        let laplace: Value =
            serde_json::from_str(&self.laplace.to_json()).expect("a GLMM artifact is JSON");
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("bayes_glmm", "prospicio-bayes", env!("CARGO_PKG_VERSION"), ""),
            "spec": {
                "family": family_json(s.family),
                "prior_sd": num(s.prior_sd),
                "intercept_sd": num(s.intercept_sd),
                "sd_scale": num(s.sd_scale),
                "lkj": num(s.lkj),
                "dispersion_scale": num(s.dispersion_scale),
                "sampler": sampler_json(&s.sampler),
            },
            "names": self.names,
            "parameters": self.parameters.json_value(),
            "effects": self.effects.json_value(),
            "layout": layout,
            "laplace": laplace,
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
        let spec = BayesGlmm {
            family: family_from(object(field(o, "family")?, "family")?)?,
            prior_sd: float(field(o, "prior_sd")?, "prior_sd")?,
            intercept_sd: float(field(o, "intercept_sd")?, "intercept_sd")?,
            sd_scale: float(field(o, "sd_scale")?, "sd_scale")?,
            lkj: float(field(o, "lkj")?, "lkj")?,
            dispersion_scale: float(field(o, "dispersion_scale")?, "dispersion_scale")?,
            sampler: sampler_from(object(field(o, "sampler")?, "sampler")?)?,
        };
        spec.check()?;
        let parameters =
            PosteriorDraws::from_json_value(object(field(top, "parameters")?, "parameters")?)?;
        let effects = PosteriorDraws::from_json_value(object(field(top, "effects")?, "effects")?)?;
        let layout = array(top, "layout")?
            .iter()
            .map(|v| {
                let l = object(v, "layout")?;
                Ok(EffectsLayout {
                    name: field(l, "name")?
                        .as_str()
                        .ok_or_else(|| artifact::bad("layout.name"))?
                        .to_string(),
                    terms: strings(l, "terms")?,
                    levels: strings(l, "levels")?,
                    start: uint(l, "start")? as usize,
                    d: uint(l, "d")? as usize,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if layout
            .iter()
            .any(|l| l.d != l.terms.len() || l.start + l.d * l.levels.len() > effects.dim())
        {
            return Err(Error::Data(
                "model artifact's effect layout disagrees with its effect draws".into(),
            ));
        }
        let laplace = GlmmFit::from_json(&field(top, "laplace")?.to_string())?;
        Ok(BayesGlmmFit {
            spec,
            names: strings(top, "names")?,
            parameters,
            effects,
            layout,
            laplace,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nuts::Sampler;
    use prospicio_glm::glmm::Grouping;
    use prospicio_models::{Design, Family};

    #[test]
    fn fits_round_trip_and_predict_alike() {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.5).collect();
        let g: Vec<String> = (0..n).map(|i| format!("g{}", i % 5)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (1.0 + ((i * 13) % 7) as f64 + (i % 5) as f64 * 0.5).round())
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x],
        )
        .unwrap();
        let groups = [Grouping::new("g", g)];
        let fit = BayesGlmm::new(Family::Poisson)
            .sampler(Sampler {
                chains: 1,
                tune: 60,
                draws: 20,
                seed: 3,
                ..Sampler::default()
            })
            .fit(&d, &groups, &y)
            .unwrap();
        let text = fit.to_json();
        let back = BayesGlmmFit::from_json(&text).unwrap();
        assert_eq!(back.to_json(), text);
        assert_eq!(
            back.predict(&d, &groups).unwrap(),
            fit.predict(&d, &groups).unwrap()
        );
        assert!(BayesGlmmFit::from_json(&text.replace("risk_rs.bayes_glmm_fit", "x")).is_err());
        assert!(
            BayesGlmmFit::from_json(&text.replacen(
                "\"format_version\":1",
                "\"format_version\":2",
                1
            ))
            .is_err()
        );
    }
}
