//! [`Triangle`] as a JSON document: [`to_json`] and [`from_json`], for
//! saving a triangle exactly as it is and for Python's `pickle` (#978).
//!
//! The document holds every part of the triangle: the shape, the key
//! names and index labels, the measure names, the origins and ages with
//! their grains, the valuation, whether it is cumulative, and the values
//! and observation mask in storage order (index, column, origin, age).
//! Numbers round-trip bit for bit; a non-finite value is written as the
//! string `"NaN"`, `"inf"` or `"-inf"`, which JSON numbers cannot hold.
//!
//! ```json
//! {"format": "risk_rs.triangle", "format_version": 1,
//!  "shape": [1, 1, 2, 2], "keys": [], "index": [[]], "columns": ["paid"],
//!  "origins": ["2020-01", "2021-01"], "origin_grain": "Y",
//!  "development": [12, 24], "development_grain": "Y",
//!  "valuation": "2021-12", "cumulative": true,
//!  "values": [100.0, 150.0, 110.0, 0.0], "mask": [true, true, true, false]}
//! ```
//!
//! [`to_json`]: Triangle::to_json
//! [`from_json`]: Triangle::from_json

use prospicio_core::{Grain, Month};
use serde_json::{Value, json};

use super::{Label, Triangle};
use crate::error::{Error, Result};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.triangle";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

impl Triangle {
    /// This triangle as a JSON document; [`from_json`](Self::from_json)
    /// reads it back to an equal triangle, every stored value bit for bit.
    ///
    /// ```
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let origin = [2020, 2020, 2021].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 110.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// assert_eq!(Triangle::from_json(&tri.to_json()).unwrap(), tri);
    /// ```
    pub fn to_json(&self) -> String {
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "shape": self.shape,
            "keys": self.keys,
            "index": self.index.iter().map(Label::parts).collect::<Vec<_>>(),
            "columns": self.columns,
            "origins": self.origins.iter().map(|&m| month_to_json(m)).collect::<Vec<_>>(),
            "origin_grain": grain_code(self.origin_grain),
            "development": self.development,
            "development_grain": grain_code(self.development_grain),
            "valuation": month_to_json(self.valuation),
            "cumulative": self.cumulative,
            "values": self.values.iter().map(|&v| number_to_json(v)).collect::<Vec<_>>(),
            "mask": self.mask,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads a document written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, or parts
    /// that do not fit together (lengths that disagree with the shape, an
    /// index label with the wrong number of parts).
    pub fn from_json(text: &str) -> Result<Self> {
        read(text).map_err(|msg| {
            Error::Core(prospicio_core::Error::Data(format!("triangle JSON: {msg}")))
        })
    }
}

fn read(text: &str) -> std::result::Result<Triangle, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("not a JSON document: {e}"))?;
    if doc.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(format!("format is not {FORMAT:?}"));
    }
    match doc.get("format_version").and_then(Value::as_u64) {
        Some(v) if v <= FORMAT_VERSION => {}
        v => return Err(format!("format version {v:?} is not supported")),
    }
    let field = |name: &str| doc.get(name).ok_or_else(|| format!("missing {name:?}"));
    let array = |name: &str| {
        field(name)?
            .as_array()
            .ok_or_else(|| format!("{name:?} is not an array"))
    };
    let strings = |v: &Value, what: &str| -> std::result::Result<Vec<String>, String> {
        v.as_array()
            .ok_or_else(|| format!("{what} is not an array"))?
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{what} holds a non-string"))
            })
            .collect()
    };

    let shape_values = array("shape")?;
    if shape_values.len() != 4 {
        return Err("\"shape\" does not have four axes".into());
    }
    let mut shape = [0usize; 4];
    for (s, v) in shape.iter_mut().zip(shape_values) {
        *s = v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("\"shape\" holds a non-count")?;
    }
    let keys = strings(field("keys")?, "\"keys\"")?;
    let index = array("index")?
        .iter()
        .map(|l| strings(l, "an index label").map(Label))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if index.iter().any(|l| l.parts().len() != keys.len()) {
        return Err("an index label does not have one part per key".into());
    }
    let columns = strings(field("columns")?, "\"columns\"")?;
    let origins = array("origins")?
        .iter()
        .map(month_from_json)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let development = array("development")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| "\"development\" holds a non-age".to_string())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if [index.len(), columns.len(), origins.len(), development.len()] != shape {
        return Err("the axes' lengths do not match \"shape\"".into());
    }
    let size = shape.iter().product::<usize>();
    let values = array("values")?
        .iter()
        .map(number_from_json)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mask = array("mask")?
        .iter()
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| "\"mask\" holds a non-boolean".to_string())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != size || mask.len() != size {
        return Err("\"values\" or \"mask\" does not have one entry per cell".into());
    }
    Ok(Triangle {
        values,
        mask,
        shape,
        keys,
        index,
        columns,
        origins,
        origin_grain: grain_from_json(field("origin_grain")?)?,
        development,
        development_grain: grain_from_json(field("development_grain")?)?,
        valuation: month_from_json(field("valuation")?)?,
        cumulative: field("cumulative")?
            .as_bool()
            .ok_or("\"cumulative\" is not a boolean")?,
    })
}

fn grain_code(grain: Grain) -> &'static str {
    match grain {
        Grain::Month => "M",
        Grain::Quarter => "Q",
        Grain::Semester => "S",
        Grain::Year => "Y",
    }
}

fn grain_from_json(v: &Value) -> std::result::Result<Grain, String> {
    match v.as_str() {
        Some("M") => Ok(Grain::Month),
        Some("Q") => Ok(Grain::Quarter),
        Some("S") => Ok(Grain::Semester),
        Some("Y") => Ok(Grain::Year),
        _ => Err(format!("{v} is not a grain (\"M\", \"Q\", \"S\" or \"Y\")")),
    }
}

fn month_to_json(m: Month) -> Value {
    Value::String(format!("{:04}-{:02}", m.year(), m.month()))
}

fn month_from_json(v: &Value) -> std::result::Result<Month, String> {
    let bad = || format!("{v} is not a month (\"YYYY-MM\")");
    let s = v.as_str().ok_or_else(bad)?;
    let (y, m) = s.rsplit_once('-').ok_or_else(bad)?;
    let year = y.parse().map_err(|_| bad())?;
    let month = m.parse().map_err(|_| bad())?;
    Month::new(year, month).map_err(|_| bad())
}

fn number_to_json(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else if x.is_nan() {
        json!("NaN")
    } else if x > 0.0 {
        json!("inf")
    } else {
        json!("-inf")
    }
}

fn number_from_json(v: &Value) -> std::result::Result<f64, String> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| format!("{n} is not a float")),
        Value::String(s) if s == "NaN" => Ok(f64::NAN),
        Value::String(s) if s == "inf" => Ok(f64::INFINITY),
        Value::String(s) if s == "-inf" => Ok(f64::NEG_INFINITY),
        _ => Err(format!("{v} is not a number")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DevelopmentColumn, Long};

    fn sample() -> Triangle {
        let origin = [2020, 2020, 2021, 2020].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[("lob", &["Auto", "Auto", "Auto", "Home"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12]),
            values: &[
                ("paid", &[100.0, 150.0, 110.0, -0.0]),
                ("incurred", &[1e-310, f64::MAX, 0.1, 50.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn round_trips_bit_for_bit() {
        for tri in [sample(), sample().to_incremental(), sample().link_ratios()] {
            let back = Triangle::from_json(&tri.to_json()).unwrap();
            assert_eq!(back.shape, tri.shape);
            assert_eq!(back.mask, tri.mask);
            assert_eq!(
                back.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                tri.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                (
                    &back.keys,
                    &back.index,
                    &back.columns,
                    &back.origins,
                    &back.development
                ),
                (
                    &tri.keys,
                    &tri.index,
                    &tri.columns,
                    &tri.origins,
                    &tri.development
                )
            );
            assert_eq!(
                (
                    back.origin_grain,
                    back.development_grain,
                    back.valuation,
                    back.cumulative
                ),
                (
                    tri.origin_grain,
                    tri.development_grain,
                    tri.valuation,
                    tri.cumulative
                )
            );
        }
    }

    #[test]
    fn non_finite_values_survive() {
        let mut tri = sample();
        tri.values[0] = f64::NAN;
        tri.values[1] = f64::INFINITY;
        tri.values[2] = f64::NEG_INFINITY;
        let back = Triangle::from_json(&tri.to_json()).unwrap();
        assert!(back.values[0].is_nan());
        assert_eq!(back.values[1], f64::INFINITY);
        assert_eq!(back.values[2], f64::NEG_INFINITY);
    }

    #[test]
    fn refuses_other_documents() {
        let text = sample().to_json();
        assert!(Triangle::from_json("[]").is_err());
        assert!(Triangle::from_json(&text.replace("risk_rs.triangle", "other")).is_err());
        assert!(
            Triangle::from_json(&text.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(Triangle::from_json(&text.replace("\"shape\":[2,", "\"shape\":[3,")).is_err());
        assert!(Triangle::from_json(&text.replace("\"Auto\"]", "\"Auto\",\"x\"]")).is_err());
    }
}
