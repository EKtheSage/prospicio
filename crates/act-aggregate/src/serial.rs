//! Saving and loading reinsurance programmes: [`Tower::to_json`] and
//! [`Tower::from_json`], a versioned JSON document per tower.
//!
//! The document lists the stages in inuring order, each a list of layers
//! with every term. Loading rebuilds each layer through the same validated
//! builders ([`Layer::xol`], [`Layer::surplus`], [`Layer::share`], ...), so
//! a document with an impossible term is refused, and a loaded tower equals
//! the saved one. Numbers round-trip bit for bit; a non-finite value (an
//! unlimited layer) is written as `"inf"`, as in distribution documents.
//!
//! ```json
//! {"format": "risk_rs.tower", "format_version": 1,
//!  "stages": [[{"name": "5x5", "basis": "loss", "limit": 5000000.0,
//!               "attachment": 5000000.0, "share": 1.0,
//!               "aggregate_deductible": 0.0, "aggregate_limit": 10000000.0,
//!               "premium": 0.0, "reinstatement_rates": [],
//!               "pro_rata_time": false}]]}
//! ```

use act_core::{Error, Result};
use serde_json::{Map, Number, Value, json};

use crate::reinsurance::{Basis, Layer, Tower};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.tower";
/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

impl Tower {
    /// The programme as a JSON document; [`from_json`](Self::from_json)
    /// reads it back to an equal tower.
    ///
    /// ```
    /// use act_aggregate::{Layer, Tower};
    ///
    /// let tower = Tower::inuring(vec![
    ///     vec![Layer::surplus("surplus", 1e6, 4.0).unwrap()],
    ///     vec![Layer::xol("xl", 2e6, 1e6).unwrap()
    ///         .paid_reinstatements(1e5, vec![1.0]).unwrap()],
    /// ])
    /// .unwrap();
    /// let back = Tower::from_json(&tower.to_json()).unwrap();
    /// assert_eq!(back, tower);
    /// ```
    pub fn to_json(&self) -> String {
        let mut stages: Vec<Value> = Vec::new();
        for (layer, &stage) in self.layers.iter().zip(&self.stages) {
            if stage == stages.len() {
                stages.push(Value::Array(Vec::new()));
            }
            if let Some(Value::Array(s)) = stages.last_mut() {
                s.push(layer_json(layer));
            }
        }
        let doc = json!({
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "stages": stages,
        });
        serde_json::to_string(&doc).expect("a JSON value serializes")
    }

    /// Reads a document written by [`to_json`](Self::to_json). Fails on
    /// malformed JSON, another format, a newer format version, an unknown
    /// basis, or a term the layer builders refuse.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc: Value = serde_json::from_str(text)
            .map_err(|e| Error::Data(format!("not a JSON document: {e}")))?;
        let top = object(&doc, "document")?;
        match top.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            other => {
                return Err(Error::Data(format!(
                    "not a tower document: format {other:?}, expected {FORMAT:?}"
                )));
            }
        }
        let version = top
            .get("format_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Data("format_version is missing".into()))?;
        if version > FORMAT_VERSION {
            return Err(Error::Data(format!(
                "format_version {version} is newer than this build reads ({FORMAT_VERSION})"
            )));
        }
        let stages = array(top, "stages")?
            .iter()
            .map(|stage| {
                stage
                    .as_array()
                    .ok_or_else(|| Error::Data("each stage must be a list of layers".into()))?
                    .iter()
                    .map(|l| layer_from(object(l, "layer")?))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Tower::inuring(stages)
    }
}

fn layer_json(l: &Layer) -> Value {
    let mut m = Map::new();
    m.insert("name".into(), json!(l.name));
    match l.basis {
        Basis::Loss => {
            m.insert("basis".into(), json!("loss"));
        }
        Basis::Surplus { retention, lines } => {
            m.insert("basis".into(), json!("surplus"));
            m.insert("retention".into(), num(retention));
            m.insert("lines".into(), num(lines));
        }
    }
    m.insert("limit".into(), num(l.limit));
    m.insert("attachment".into(), num(l.attachment));
    m.insert("share".into(), num(l.share));
    m.insert("aggregate_deductible".into(), num(l.aggregate_deductible));
    m.insert("aggregate_limit".into(), num(l.aggregate_limit));
    m.insert("premium".into(), num(l.premium));
    m.insert(
        "reinstatement_rates".into(),
        Value::Array(l.reinstatement_rates.iter().map(|&r| num(r)).collect()),
    );
    m.insert("pro_rata_time".into(), json!(l.pro_rata_time));
    Value::Object(m)
}

fn layer_from(m: &Map<String, Value>) -> Result<Layer> {
    let name = m
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Data("a layer's name must be a string".into()))?;
    let mut layer = Layer::xol(name, float(m, "limit")?, float(m, "attachment")?)?;
    match m.get("basis").and_then(Value::as_str) {
        Some("loss") => {}
        Some("surplus") => {
            // The surplus builder checks the retention and lines.
            let s = Layer::surplus(name, float(m, "retention")?, float(m, "lines")?)?;
            layer.basis = s.basis;
        }
        other => {
            return Err(Error::Data(format!(
                "layer {name:?}: basis {other:?} is not \"loss\" or \"surplus\""
            )));
        }
    }
    layer = layer
        .share(float(m, "share")?)?
        .aggregate_deductible(float(m, "aggregate_deductible")?)?;
    let premium = float(m, "premium")?;
    let rates = array(m, "reinstatement_rates")?
        .iter()
        .map(|v| to_f64(v, "reinstatement_rates"))
        .collect::<Result<Vec<_>>>()?;
    if rates.is_empty() {
        if !(premium.is_finite() && premium >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: premium,
                reason: "must be finite and non-negative",
            });
        }
        layer.premium = premium;
    } else {
        layer = layer.paid_reinstatements(premium, rates)?;
    }
    // After paid reinstatements, which set it from the limit.
    layer = layer.aggregate_limit(float(m, "aggregate_limit")?)?;
    match m.get("pro_rata_time") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => layer = layer.pro_rata_as_to_time()?,
        Some(_) => return Err(Error::Data("pro_rata_time must be true or false".into())),
    }
    Ok(layer)
}

fn num(x: f64) -> Value {
    match Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None if x.is_nan() => json!("NaN"),
        None if x > 0.0 => json!("inf"),
        None => json!("-inf"),
    }
}

fn to_f64(v: &Value, name: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
    .ok_or_else(|| Error::Data(format!("{name} must be a number")))
}

fn float(m: &Map<String, Value>, name: &str) -> Result<f64> {
    to_f64(
        m.get(name)
            .ok_or_else(|| Error::Data(format!("{name} is missing")))?,
        name,
    )
}

fn array<'a>(m: &'a Map<String, Value>, name: &str) -> Result<&'a Vec<Value>> {
    m.get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Data(format!("{name} must be a list")))
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error::Data(format!("the {what} must be a JSON object")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate_events;
    use act_prob::{Lognormal, Poisson};

    fn programme() -> Tower {
        Tower::inuring(vec![
            vec![
                Layer::quota_share("QS 30%", 0.3).unwrap(),
                Layer::surplus("surplus", 1e6, 4.0)
                    .unwrap()
                    .aggregate_limit(2e7)
                    .unwrap(),
            ],
            vec![
                Layer::xol("2x1", 2e6, 1e6)
                    .unwrap()
                    .share(0.6)
                    .unwrap()
                    .aggregate_deductible(5e5)
                    .unwrap()
                    .paid_reinstatements(3e5, vec![1.0, 0.5])
                    .unwrap()
                    .pro_rata_as_to_time()
                    .unwrap(),
                Layer::xol("free", 5e6, 3e6)
                    .unwrap()
                    .reinstatements(2)
                    .unwrap(),
            ],
            vec![Layer::stop_loss("SL", f64::INFINITY, 2e7).unwrap()],
        ])
        .unwrap()
    }

    #[test]
    fn round_trips_every_term() {
        let tower = programme();
        let text = tower.to_json();
        assert!(text.contains("\"inf\""));
        let back = Tower::from_json(&text).unwrap();
        assert_eq!(back, tower);
        assert_eq!(back.to_json(), text);
        // The loaded programme cedes the same, event by event.
        let events = simulate_events(
            &Poisson::new(3.0).unwrap(),
            &Lognormal::from_mean_cv(1e6, 2.0).unwrap(),
            500,
            3,
        )
        .unwrap();
        let si: Vec<f64> = (0..events.n_sims())
            .flat_map(|i| events.events(i).iter().map(|&x| 2.0 * x + 1e5))
            .collect();
        let events = events.with_sums_insured(si).unwrap().with_uniform_times();
        let a = tower.apply(&events).unwrap();
        let b = back.apply(&events).unwrap();
        for i in 0..a.n_sims() {
            assert_eq!(a.row(i).unwrap(), b.row(i).unwrap());
        }
    }

    #[test]
    fn refuses_bad_documents() {
        let good = Tower::new(vec![Layer::xol("L", 1.0, 1.0).unwrap()])
            .unwrap()
            .to_json();
        assert!(Tower::from_json("not json").is_err());
        assert!(Tower::from_json(&good.replace("risk_rs.tower", "risk_rs.distribution")).is_err());
        assert!(
            Tower::from_json(&good.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(Tower::from_json(&good.replace("\"loss\"", "\"cat\"")).is_err());
        // A term the builders refuse.
        assert!(Tower::from_json(&good.replace("\"share\":1.0", "\"share\":1.5")).is_err());
        assert!(Tower::from_json(&good.replace("\"stages\":[[", "\"stages\":[[],[")).is_err());
    }
}
