//! [`EventSet`] as a JSON document: [`to_json`] and [`from_json`], for
//! saving simulated years exactly as they are and for Python's `pickle`
//! (#978).
//!
//! The document holds the year offsets and the losses, and when present
//! each loss's sum insured, time and labels, each year's aggregate part,
//! the seed, the samplers and the input hash. Numbers round-trip bit for
//! bit; a non-finite value is written as `"NaN"`, `"inf"` or `"-inf"`.
//!
//! ```json
//! {"format": "risk_rs.event_set", "format_version": 1,
//!  "offsets": [0, 2, 3], "losses": [1.0, 2.0, 3.0], "seed": 7,
//!  "sums_insured": null, "times": null, "labels": null,
//!  "aggregate": null, "samplers": null, "input_hash": null}
//! ```
//!
//! [`to_json`]: EventSet::to_json
//! [`from_json`]: EventSet::from_json

use prospicio_core::{Error, Result};
use serde_json::{Value, json};

use super::{AggregatePart, EventSet, Labels};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.event_set";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

type Parse<T> = std::result::Result<T, String>;

impl EventSet {
    /// These years as a JSON document; [`from_json`](Self::from_json)
    /// reads them back to an equal event set, every number bit for bit.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![1.0, 2.0], vec![], vec![3.0]], 7).unwrap();
    /// assert_eq!(EventSet::from_json(&events.to_json()).unwrap(), events);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "offsets": self.offsets,
            "losses": numbers(&self.losses),
            "sums_insured": self.sums_insured.as_deref().map(numbers),
            "times": self.times.as_deref().map(numbers),
            "labels": self.labels.as_ref().map(|l| json!({
                "dims": l.dims,
                "combinations": l.combinations,
                "of_loss": l.of_loss,
            })),
            "aggregate": self.aggregate.as_ref().map(|a| json!({
                "of_column": a.of_column,
                "amounts": numbers(&a.amounts),
            })),
            "seed": self.seed,
            "samplers": self.samplers,
            "input_hash": self.input_hash,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads a document written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or parts
    /// that do not fit together.
    pub fn from_json(text: &str) -> Result<Self> {
        read(text).map_err(|msg| Error::Data(format!("event set JSON: {msg}")))
    }
}

fn read(text: &str) -> Parse<EventSet> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("not a JSON document: {e}"))?;
    if doc.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(format!("format is not {FORMAT:?}"));
    }
    match doc.get("format_version").and_then(Value::as_u64) {
        Some(v) if v <= FORMAT_VERSION => {}
        v => return Err(format!("format version {v:?} is not supported")),
    }
    let field = |name: &str| doc.get(name).ok_or_else(|| format!("missing {name:?}"));
    let optional = |name: &str| field(name).map(|v| (!v.is_null()).then_some(v));

    let offsets = counts(field("offsets")?, "\"offsets\"")?;
    let losses = floats(field("losses")?, "\"losses\"")?;
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&losses.len())
        || offsets.windows(2).any(|w| w[0] > w[1])
    {
        return Err("\"offsets\" do not index the losses year by year".into());
    }
    let n_years = offsets.len() - 1;
    let per_loss = |name: &str| -> Parse<Option<Vec<f64>>> {
        optional(name)?
            .map(|v| {
                let xs = floats(v, name)?;
                if xs.len() == losses.len() {
                    Ok(xs)
                } else {
                    Err(format!("{name:?} does not have one value per loss"))
                }
            })
            .transpose()
    };
    let sums_insured = per_loss("sums_insured")?;
    let times = per_loss("times")?;
    let labels = optional("labels")?
        .map(|v| -> Parse<Labels> {
            let dims = strings(field_of(v, "dims")?, "label dims")?;
            let combinations = field_of(v, "combinations")?
                .as_array()
                .ok_or("label combinations are not an array")?
                .iter()
                .map(|c| strings(c, "a label combination"))
                .collect::<Parse<Vec<_>>>()?;
            if combinations.iter().any(|c| c.len() != dims.len()) {
                return Err("a label combination does not have one value per dimension".into());
            }
            let of_loss = counts(field_of(v, "of_loss")?, "label of_loss")?;
            if of_loss.len() != losses.len() || of_loss.iter().any(|&i| i >= combinations.len()) {
                return Err("label of_loss does not give each loss a combination".into());
            }
            Ok(Labels {
                dims,
                combinations,
                of_loss,
            })
        })
        .transpose()?;
    let n_combinations = labels.as_ref().map_or(1, |l| l.combinations.len());
    let aggregate = optional("aggregate")?
        .map(|v| -> Parse<AggregatePart> {
            let of_column = counts(field_of(v, "of_column")?, "aggregate of_column")?;
            let amounts = floats(field_of(v, "amounts")?, "aggregate amounts")?;
            if of_column.iter().any(|&i| i >= n_combinations)
                || amounts.len() != n_years * of_column.len()
            {
                return Err("the aggregate part does not fit the years and labels".into());
            }
            Ok(AggregatePart { of_column, amounts })
        })
        .transpose()?;
    let samplers = optional("samplers")?
        .map(|v| -> Parse<Vec<(String, String)>> {
            v.as_array()
                .ok_or("\"samplers\" is not an array")?
                .iter()
                .map(|pair| {
                    let p = strings(pair, "a sampler")?;
                    match <[String; 2]>::try_from(p) {
                        Ok([a, b]) => Ok((a, b)),
                        Err(_) => Err("a sampler is not a pair of strings".to_string()),
                    }
                })
                .collect::<Parse<Vec<_>>>()
        })
        .transpose()?;
    let input_hash = optional("input_hash")?
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or("\"input_hash\" is not a string")
        })
        .transpose()?;
    Ok(EventSet {
        offsets,
        losses,
        sums_insured,
        times,
        labels,
        aggregate,
        seed: field("seed")?
            .as_u64()
            .ok_or("\"seed\" is not an unsigned integer")?,
        samplers,
        input_hash,
    })
}

fn field_of<'a>(v: &'a Value, name: &str) -> Parse<&'a Value> {
    v.get(name).ok_or_else(|| format!("missing {name:?}"))
}

fn numbers(xs: &[f64]) -> Vec<Value> {
    xs.iter()
        .map(|&x| {
            if x.is_finite() {
                json!(x)
            } else if x.is_nan() {
                json!("NaN")
            } else if x > 0.0 {
                json!("inf")
            } else {
                json!("-inf")
            }
        })
        .collect()
}

fn floats(v: &Value, what: &str) -> Parse<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|x| match x {
            Value::Number(n) => n.as_f64().ok_or_else(|| format!("{n} is not a float")),
            Value::String(s) if s == "NaN" => Ok(f64::NAN),
            Value::String(s) if s == "inf" => Ok(f64::INFINITY),
            Value::String(s) if s == "-inf" => Ok(f64::NEG_INFINITY),
            _ => Err(format!("{what} holds {x}, not a number")),
        })
        .collect()
}

fn counts(v: &Value, what: &str) -> Parse<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("{what} holds {x}, not a count"))
        })
        .collect()
}

fn strings(v: &Value, what: &str) -> Parse<Vec<String>> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|s| {
            s.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{what} holds a non-string"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(xs: &[f64]) -> Vec<u64> {
        xs.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn round_trips_every_part() {
        let mut e =
            EventSet::from_years(vec![vec![1.5, -0.0], vec![], vec![1e-310, f64::MAX]], 9).unwrap();
        e.sums_insured = Some(vec![2.0, 1.0, 1.0, f64::MAX]);
        e.times = Some(vec![0.1, 0.9, 0.0, 1.0]);
        e.labels = Some(Labels {
            dims: vec!["peril".into()],
            combinations: vec![vec!["flood".into()], vec!["wind".into()]],
            of_loss: vec![1, 0, 0, 1],
        });
        e.aggregate = Some(AggregatePart {
            of_column: vec![0, 1],
            amounts: vec![1.0, 2.0, 0.0, 0.5, 3.0, 4.0],
        });
        e.samplers = Some(vec![("count".into(), "inverse".into())]);
        e.input_hash = Some("abc".into());
        let back = EventSet::from_json(&e.to_json()).unwrap();
        assert_eq!(back, e);
        assert_eq!(bits(&back.losses), bits(&e.losses));
    }

    #[test]
    fn refuses_other_documents() {
        let e = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 1).unwrap();
        let text = e.to_json();
        assert!(EventSet::from_json("{}").is_err());
        assert!(EventSet::from_json(&text.replace("risk_rs.event_set", "x")).is_err());
        assert!(
            EventSet::from_json(&text.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(EventSet::from_json(&text.replace("[0,2,3]", "[0,2,4]")).is_err());
        assert!(EventSet::from_json(&text.replace("\"times\":null", "\"times\":[0.5]")).is_err());
    }
}
