//! Saving and loading posterior draws: a versioned JSON artifact in the
//! envelope the model artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! The artifact holds the parameter names, the number of chains, every
//! draw chain-major, the divergences of each chain and the seed the draws
//! came from; numbers round-trip bit for bit.

use prospicio_core::{Error, Result};
use prospicio_models::artifact::{field, floats, object, parse, strings, uint};
use serde_json::{Map, Value, json};

use super::PosteriorDraws;
use prospicio_models::artifact::{self, nums};

const FORMAT: &str = "risk_rs.posterior_draws";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

impl PosteriorDraws {
    /// The draws as a JSON value, for the artifacts that hold draws.
    pub(crate) fn json_value(&self) -> Value {
        json!({
            "names": self.names,
            "chains": self.chains,
            "values": nums(&self.values),
            "divergences": self.divergences,
            "seed": self.seed,
        })
    }

    /// The draws [`json_value`](Self::json_value) wrote.
    pub(crate) fn from_json_value(o: &Map<String, Value>) -> Result<Self> {
        let names = strings(o, "names")?;
        let chains = uint(o, "chains")? as usize;
        let values = floats(o, "values")?;
        let divergences = field(o, "divergences")?
            .as_array()
            .ok_or_else(|| artifact::bad("divergences"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|d| d as usize)
                    .ok_or_else(|| artifact::bad("divergences"))
            })
            .collect::<Result<Vec<_>>>()?;
        let dim = names.len();
        if dim == 0
            || chains == 0
            || divergences.len() != chains
            || values.is_empty()
            || !values.len().is_multiple_of(chains * dim)
        {
            return Err(Error::Data(
                "draws artifact sizes disagree: names, chains, values and divergences".into(),
            ));
        }
        let per_chain = values.len() / chains;
        let mut draws = Self::from_chains(
            names,
            values.chunks(per_chain).map(<[f64]>::to_vec).collect(),
            divergences,
        )?;
        draws.seed = uint(o, "seed")?;
        Ok(draws)
    }

    /// The draws as a JSON artifact; [`from_json`](Self::from_json) reads
    /// them back exactly.
    ///
    /// ```
    /// use prospicio_bayes::nuts::PosteriorDraws;
    ///
    /// let d = PosteriorDraws::from_chains(
    ///     vec!["mu".into(), "sigma".into()],
    ///     vec![vec![0.1, 1.0, 0.2, 1.1], vec![0.0, 0.9, f64::NAN, 1.2]],
    ///     vec![0, 1],
    /// )
    /// .unwrap();
    /// let back = PosteriorDraws::from_json(&d.to_json()).unwrap();
    /// assert_eq!(back.to_json(), d.to_json());
    /// ```
    pub fn to_json(&self) -> String {
        let mut doc = self.json_value();
        let o = doc.as_object_mut().expect("draws are an object");
        o.insert("format".into(), json!(FORMAT));
        o.insert("format_version".into(), json!(FORMAT_VERSION));
        o.insert(
            "provenance".into(),
            artifact::provenance(
                "posterior_draws",
                "prospicio-bayes",
                env!("CARGO_PKG_VERSION"),
                "",
            ),
        );
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads an artifact written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or fields
    /// that are missing or inconsistent.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, FORMAT, FORMAT_VERSION)?;
        Self::from_json_value(object(&doc, "artifact")?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_round_trip_with_their_seed_and_divergences() {
        let mut d = PosteriorDraws::from_chains(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                (0..12).map(|i| f64::from(i) * 0.1).collect(),
                (0..12).map(|i| f64::from(i).sqrt()).collect(),
            ],
            vec![2, 0],
        )
        .unwrap();
        d.seed = 41;
        let back = PosteriorDraws::from_json(&d.to_json()).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn refuses_other_documents() {
        let d =
            PosteriorDraws::from_chains(vec!["a".into()], vec![vec![1.0, 2.0]], vec![]).unwrap();
        let j = d.to_json();
        assert!(PosteriorDraws::from_json(&j.replace("risk_rs.posterior_draws", "x")).is_err());
        assert!(
            PosteriorDraws::from_json(&j.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(PosteriorDraws::from_json(&j.replace("\"chains\":1", "\"chains\":3")).is_err());
    }
}
