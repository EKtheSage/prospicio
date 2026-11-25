//! Saving and loading MARS fits: a versioned JSON artifact in the envelope
//! the other GLM artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the predictors' names, the selected and forward
//! terms, the forward pass step by step, why it stopped, the coefficients
//! and the fit statistics, and with [`Mars::glm`](super::Mars::glm) the
//! GLM fitted on the selected terms as a nested `risk_rs.glm_fit` artifact.
//! Numbers round-trip bit for bit. The MARS fit keeps no hash of its
//! training data, so the provenance's `input_hash` is empty; the nested
//! GLM keeps its own.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{
    self, array, bad, field, float, floats, input_hash, num, nums, object, parse, strings,
};
use serde_json::{Map, Value, json};

use super::{Direction, MarsFit, MarsStep, MarsTerm, Termination};
use crate::GlmFit;

const FORMAT: &str = "risk_rs.mars_fit";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

fn direction_json(d: Direction) -> &'static str {
    match d {
        Direction::Up => "up",
        Direction::Down => "down",
        Direction::Linear => "linear",
    }
}

fn termination_json(t: Termination) -> &'static str {
    match t {
        Termination::NoImprovement => "no_improvement",
        Termination::Threshold => "threshold",
        Termination::Gcv => "gcv",
        Termination::Rsq => "rsq",
        Termination::MaxTerms => "max_terms",
    }
}

fn terms_json(terms: &[MarsTerm]) -> Value {
    terms
        .iter()
        .map(|t| {
            json!({
                "variable": t.variable,
                "knot": num(t.knot),
                "direction": direction_json(t.direction),
            })
        })
        .collect()
}

fn terms_from(o: &Map<String, Value>, key: &str, p: usize) -> Result<Vec<MarsTerm>> {
    array(o, key)?
        .iter()
        .map(|v| {
            let t = object(v, key)?;
            let variable = field(t, "variable")?
                .as_u64()
                .ok_or_else(|| bad("variable"))? as usize;
            if variable >= p {
                return Err(Error::Data(format!(
                    "model artifact term names predictor {variable} of {p}"
                )));
            }
            let direction = match field(t, "direction")?.as_str() {
                Some("up") => Direction::Up,
                Some("down") => Direction::Down,
                Some("linear") => Direction::Linear,
                _ => return Err(bad("direction")),
            };
            Ok(MarsTerm {
                variable,
                knot: float(field(t, "knot")?, "knot")?,
                direction,
            })
        })
        .collect()
}

impl MarsFit {
    /// The fit as a JSON artifact; [`from_json`](Self::from_json) reads it
    /// back exactly.
    ///
    /// ```
    /// use prospicio_glm::mars::{Mars, MarsFit};
    ///
    /// let x: Vec<f64> = (0..60).map(f64::from).collect();
    /// let y: Vec<f64> = x.iter().map(|v| 1.0 + 0.5 * v + 3.0 * (v - 31.0).max(0.0)).collect();
    /// let fit = Mars::new().fit(&["x"], &[x], &y, None).unwrap();
    /// assert_eq!(MarsFit::from_json(&fit.to_json()).unwrap(), fit);
    /// ```
    pub fn to_json(&self) -> String {
        let steps: Vec<Value> = self
            .steps
            .iter()
            .map(|s| {
                json!({
                    "terms": terms_json(&s.terms),
                    "rss": num(s.rss),
                    "rsq": num(s.rsq),
                    "gcv": num(s.gcv),
                })
            })
            .collect();
        let glm = match &self.glm {
            Some(g) => serde_json::from_str(&g.to_json()).expect("a GLM artifact is JSON"),
            None => Value::Null,
        };
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "provenance": artifact::provenance("mars", "prospicio-glm", env!("CARGO_PKG_VERSION"), ""),
            "names": self.names,
            "terms": terms_json(&self.terms),
            "forward": terms_json(&self.forward),
            "steps": steps,
            "termination": termination_json(self.termination),
            "coefficients": nums(&self.coefficients),
            "rss": num(self.rss),
            "gcv": num(self.gcv),
            "rsq": num(self.rsq),
            "grsq": num(self.grsq),
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
        input_hash(top)?;
        let names = strings(top, "names")?;
        let p = names.len();
        let terms = terms_from(top, "terms", p)?;
        let coefficients = floats(top, "coefficients")?;
        if coefficients.len() != terms.len() + 1 {
            return Err(Error::Data(
                "model artifact sizes disagree: terms and coefficients".into(),
            ));
        }
        let steps = array(top, "steps")?
            .iter()
            .map(|v| {
                let s = object(v, "steps")?;
                Ok(MarsStep {
                    terms: terms_from(s, "terms", p)?,
                    rss: float(field(s, "rss")?, "rss")?,
                    rsq: float(field(s, "rsq")?, "rsq")?,
                    gcv: float(field(s, "gcv")?, "gcv")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let termination = match field(top, "termination")?.as_str() {
            Some("no_improvement") => Termination::NoImprovement,
            Some("threshold") => Termination::Threshold,
            Some("gcv") => Termination::Gcv,
            Some("rsq") => Termination::Rsq,
            Some("max_terms") => Termination::MaxTerms,
            _ => return Err(bad("termination")),
        };
        let glm = match field(top, "glm")? {
            Value::Null => None,
            v => Some(GlmFit::from_json(&v.to_string())?),
        };
        Ok(MarsFit {
            names,
            terms,
            forward: terms_from(top, "forward", p)?,
            steps,
            termination,
            coefficients,
            rss: float(field(top, "rss")?, "rss")?,
            gcv: float(field(top, "gcv")?, "gcv")?,
            rsq: float(field(top, "rsq")?, "rsq")?,
            grsq: float(field(top, "grsq")?, "grsq")?,
            glm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mars::Mars;
    use prospicio_models::{Family, Link};

    fn data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x1: Vec<f64> = (0..80).map(|i| f64::from(i) / 2.0).collect();
        let x2: Vec<f64> = (0..80).map(|i| f64::from((i * 17) % 23)).collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .enumerate()
            .map(|(i, (a, b))| {
                2.0 + 0.3 * a
                    + 1.5 * (a - 20.0).max(0.0)
                    + 0.2 * (8.0 - b).max(0.0)
                    + ((i * 7) % 5) as f64 / 10.0
            })
            .collect();
        (vec![x1, x2], y)
    }

    #[test]
    fn least_squares_fits_round_trip() {
        let (x, y) = data();
        let fit = Mars::new()
            .threshold(0.0)
            .prune(true)
            .fit(&["a", "b"], &x, &y, None)
            .unwrap();
        assert!(!fit.steps().is_empty());
        assert_eq!(MarsFit::from_json(&fit.to_json()).unwrap(), fit);
    }

    #[test]
    fn glm_fits_round_trip_with_the_nested_glm() {
        let (x, y) = data();
        let fit = Mars::new()
            .glm(Family::Gamma, Link::Log)
            .fit(&["a", "b"], &x, &y, None)
            .unwrap();
        assert!(fit.glm_fit().is_some());
        let back = MarsFit::from_json(&fit.to_json()).unwrap();
        assert_eq!(back, fit);
    }

    #[test]
    fn refuses_other_documents() {
        let (x, y) = data();
        let j = Mars::new()
            .fit(&["a", "b"], &x, &y, None)
            .unwrap()
            .to_json();
        assert!(MarsFit::from_json(&j.replace("risk_rs.mars_fit", "risk_rs.glm_fit")).is_err());
        assert!(
            MarsFit::from_json(&j.replace("\"format_version\":1", "\"format_version\":2")).is_err()
        );
        assert!(MarsFit::from_json(&j.replace("\"names\":[\"a\",\"b\"]", "\"names\":[]")).is_err());
    }
}
