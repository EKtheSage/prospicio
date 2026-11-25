//! [`PredictiveDistribution`] as a JSON document: [`to_json`] and
//! [`from_json`], for saving a distribution without the `arrow` feature
//! and for Python's `pickle` (#978).
//!
//! The document holds the dimension names, each component's key and the
//! provenance in the same JSON forms as the Arrow IPC file's metadata
//! (`crate::ipc`, the `arrow` feature), and the draws simulation-major.
//! Numbers round-trip bit for bit.
//!
//! ```json
//! {"format": "risk_rs.predictive_distribution", "format_version": 1,
//!  "dims": ["lob"], "components": [[{"text": "motor"}]],
//!  "provenance": {"model": "…", …}, "n_sims": 2, "draws": [1.0, 2.0]}
//! ```
//!
//! [`to_json`]: PredictiveDistribution::to_json
//! [`from_json`]: PredictiveDistribution::from_json

use prospicio_core::{Error, Grain, Month, Period, Result};
use serde_json::{Value, json};

use crate::predictive::{ComponentKey, KeyValue, PredictiveDistribution};
use crate::provenance::Provenance;

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.predictive_distribution";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

impl PredictiveDistribution {
    /// This distribution as a JSON document;
    /// [`from_json`](Self::from_json) reads it back to an equal
    /// distribution, draws, keys and provenance bit for bit.
    ///
    /// ```
    /// use prospicio_prob::PredictiveDistribution;
    /// use prospicio_prob::predictive::KeyValue;
    /// use prospicio_prob::provenance::Provenance;
    ///
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["lob".into()],
    ///     vec![vec![KeyValue::Text("motor".into())], vec![KeyValue::Text("home".into())]],
    ///     vec![1.0, 2.0, 0.1, 0.3],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let back = PredictiveDistribution::from_json(&pd.to_json()).unwrap();
    /// assert_eq!(back.draw_matrix(), pd.draw_matrix());
    /// assert_eq!(back.components(), pd.components());
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "dims": self.dims(),
            "components": self
                .components()
                .iter()
                .map(|key| Value::Array(key.iter().map(key_value_to_json).collect()))
                .collect::<Vec<_>>(),
            "provenance": provenance_to_json(self.provenance()),
            "n_sims": self.n_sims(),
            "draws": self.draw_matrix(),
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads a document written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or data
    /// that [`from_draws`](Self::from_draws) refuses.
    pub fn from_json(text: &str) -> Result<Self> {
        let bad = |msg: String| Error::Data(format!("predictive distribution JSON: {msg}"));
        let doc: Value =
            serde_json::from_str(text).map_err(|e| bad(format!("not a JSON document: {e}")))?;
        if doc.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Err(bad(format!("format is not {FORMAT:?}")));
        }
        match doc.get("format_version").and_then(Value::as_u64) {
            Some(v) if v <= FORMAT_VERSION => {}
            v => return Err(bad(format!("format version {v:?} is not supported"))),
        }
        let dims: Vec<String> = doc
            .get("dims")
            .and_then(Value::as_array)
            .and_then(|a| a.iter().map(|d| d.as_str().map(String::from)).collect())
            .ok_or_else(|| bad("dims is not an array of strings".into()))?;
        let components: Vec<ComponentKey> = doc
            .get("components")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("components is not an array".into()))?
            .iter()
            .map(component_key_from_json)
            .collect::<std::result::Result<_, _>>()
            .map_err(bad)?;
        let provenance = provenance_from_json(
            doc.get("provenance")
                .ok_or_else(|| bad("no provenance".into()))?,
        )
        .map_err(bad)?;
        let draws: Vec<f64> = doc
            .get("draws")
            .and_then(Value::as_array)
            .and_then(|a| a.iter().map(Value::as_f64).collect())
            .ok_or_else(|| bad("draws is not an array of numbers".into()))?;
        let n_sims = doc.get("n_sims").and_then(Value::as_u64);
        if n_sims.map(|n| n as usize * components.len()) != Some(draws.len()) {
            return Err(bad(
                "n_sims times the components is not the number of draws".into(),
            ));
        }
        PredictiveDistribution::from_draws(dims, components, draws, provenance)
    }
}

/// Spelled out rather than taken from `Grain`'s `Display`, so the file
/// format cannot change with it.
fn grain_code(grain: Grain) -> &'static str {
    match grain {
        Grain::Month => "M",
        Grain::Quarter => "Q",
        Grain::Semester => "S",
        Grain::Year => "Y",
    }
}

pub(crate) fn key_value_to_json(v: &KeyValue) -> Value {
    match v {
        KeyValue::Int(i) => json!({ "int": i }),
        KeyValue::Text(s) => json!({ "text": s }),
        KeyValue::Period(p) => json!({ "period": {
            "start": p.start().to_string(),
            "grain": grain_code(p.grain()),
        }}),
    }
}

pub(crate) fn component_key_from_json(v: &Value) -> std::result::Result<ComponentKey, String> {
    v.as_array()
        .ok_or_else(|| String::from("a component key is not an array"))?
        .iter()
        .map(key_value_from_json)
        .collect()
}

fn key_value_from_json(v: &Value) -> std::result::Result<KeyValue, String> {
    let bad = || format!("{v} is not a key value");
    let obj = v.as_object().filter(|o| o.len() == 1).ok_or_else(bad)?;
    let (tag, inner) = obj.iter().next().ok_or_else(bad)?;
    match tag.as_str() {
        "int" => inner.as_i64().map(KeyValue::Int).ok_or_else(bad),
        "text" => inner
            .as_str()
            .map(|s| KeyValue::Text(s.into()))
            .ok_or_else(bad),
        "period" => {
            let start = inner.get("start").and_then(Value::as_str).ok_or_else(bad)?;
            let grain = match inner.get("grain").and_then(Value::as_str) {
                Some("M") => Grain::Month,
                Some("Q") => Grain::Quarter,
                Some("S") => Grain::Semester,
                Some("Y") => Grain::Year,
                _ => return Err(bad()),
            };
            let (year, month) = start.rsplit_once('-').ok_or_else(bad)?;
            let month = Month::new(
                year.parse().map_err(|_| bad())?,
                month.parse().map_err(|_| bad())?,
            )
            .map_err(|_| bad())?;
            let period = Period::containing(month, grain);
            if period.start() != month {
                return Err(format!(
                    "period start {start} is not the first month of a {grain} period"
                ));
            }
            Ok(KeyValue::Period(period))
        }
        _ => Err(bad()),
    }
}

fn pairs_to_json(pairs: &[(String, String)]) -> Value {
    pairs.iter().map(|(k, v)| json!([k, v])).collect()
}

fn pairs_from_json(
    v: Option<&Value>,
    what: &str,
) -> std::result::Result<Vec<(String, String)>, String> {
    let bad = || format!("provenance {what} is not an array of [name, value]");
    v.and_then(Value::as_array)
        .ok_or_else(bad)?
        .iter()
        .map(|pair| match pair.as_array().map(Vec::as_slice) {
            Some([Value::String(k), Value::String(v)]) => Ok((k.clone(), v.clone())),
            _ => Err(bad()),
        })
        .collect()
}

pub(crate) fn provenance_to_json(p: &Provenance) -> Value {
    json!({
        "model": p.model,
        "parameters": pairs_to_json(&p.parameters),
        "seed": p.seed.map(|s| s.to_string()),
        "stream_scheme": p.stream_scheme,
        "samplers": p.samplers.as_deref().map(pairs_to_json),
        "versions": pairs_to_json(&p.versions),
        "input_hash": p.input_hash,
    })
}

pub(crate) fn provenance_from_json(v: &Value) -> std::result::Result<Provenance, String> {
    let optional_str = |name: &str| -> std::result::Result<Option<String>, String> {
        match v.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format!("provenance {name} is not a string")),
        }
    };
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| String::from("provenance has no model"))?
        .to_string();
    let seed = optional_str("seed")?
        .map(|s| {
            s.parse()
                .map_err(|_| format!("provenance seed {s:?} is not a u64"))
        })
        .transpose()?;
    // Absent from files written before samplers were recorded
    // (2026-10-08). Families must be strictly increasing, one id each, as
    // written, so that records equal as sets compare equal in
    // `Provenance::replays_same_draws`.
    let samplers = match v.get("samplers") {
        None | Some(Value::Null) => None,
        Some(s) => {
            let pairs = pairs_from_json(Some(s), "samplers")?;
            if !pairs.windows(2).all(|w| w[0].0 < w[1].0) {
                return Err(String::from(
                    "provenance samplers are not sorted by family, one id per family",
                ));
            }
            Some(pairs)
        }
    };
    Ok(Provenance {
        model,
        parameters: pairs_from_json(v.get("parameters"), "parameters")?,
        seed,
        stream_scheme: optional_str("stream_scheme")?,
        samplers,
        versions: pairs_from_json(v.get("versions"), "versions")?,
        input_hash: optional_str("input_hash")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{InputHasher, SIM_INDEX_SCHEME};

    fn sample() -> PredictiveDistribution {
        let q = |y, m| Period::containing(Month::new(y, m).unwrap(), Grain::Quarter);
        PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into(), "layer".into()],
            vec![
                vec!["Auto".into(), q(2020, 1).into(), 1.into()],
                vec!["Auto".into(), q(2020, 4).into(), 1.into()],
                vec!["Home".into(), Period::year(2019).into(), 2.into()],
            ],
            vec![
                1.0,
                0.1 + 0.2,
                -0.0,
                f64::MIN_POSITIVE,
                5e-324,
                6.25,
                7.0,
                f64::MAX,
                9.0e12,
                std::f64::consts::PI,
                11.0,
                -12.5,
            ],
            Provenance::new("test")
                .param("note", "a \"quoted\" value")
                .seed(u64::MAX, SIM_INDEX_SCHEME)
                .input_hash(InputHasher::new().str("tri").finish()),
        )
        .unwrap()
    }

    #[test]
    fn round_trips_bit_for_bit() {
        let pd = sample();
        let back = PredictiveDistribution::from_json(&pd.to_json()).unwrap();
        assert_eq!(back.dims(), pd.dims());
        assert_eq!(back.components(), pd.components());
        assert_eq!(back.n_sims(), pd.n_sims());
        let bits = |p: &PredictiveDistribution| -> Vec<u64> {
            p.draw_matrix().iter().map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&back), bits(&pd));
        assert_eq!(back.provenance(), pd.provenance());
        assert_eq!(back.to_json(), pd.to_json());
    }

    #[test]
    fn refuses_other_documents() {
        let text = sample().to_json();
        for bad in [
            "not json".to_string(),
            text.replace("risk_rs.predictive_distribution", "risk_rs.distribution"),
            text.replace("\"format_version\":1", "\"format_version\":2"),
            text.replace("\"n_sims\":4", "\"n_sims\":5"),
        ] {
            assert!(PredictiveDistribution::from_json(&bad).is_err(), "{bad}");
        }
    }
}
