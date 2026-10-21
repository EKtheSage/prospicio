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
//!
//! A layer with a loss corridor also has `"loss_corridor": {"lower": …,
//! "upper": …, "retained": …}`; a document without one (all those written
//! before corridors) loads with none. Contract terms are written the same
//! way, only when present: `"ceding_commission": rate`, `"sliding_scale":
//! [[commission, loss_ratio], …]`, `"profit_commission": {"share": …,
//! "allowance": …}` and `"swing": {"basic": …, "lcm": …, "minimum": …,
//! "maximum": …}` (at 100% of the layer). A carry-forward over a term is
//! `"profit_commission_carry": {"years": n}` and `"sliding_scale_carry":
//! {"deficit": …, "credit": …, "years": n}`, with `years` null without
//! limit; documents without them load with none (format version 1).
//! A deal's deductions (#485, #477) are `"brokerage": rate` or
//! `"brokerage_scale": [[rate, loss_ratio], …]`, `"deductions": [{"label":
//! …, "basis": "premium" | "loss" | "fixed", "value": …, "inside_nmbf":
//! …}, …]`, and the flags `"scales_before_annual_terms": true` and
//! `"brokerage_within_commission_range": true` (format version 3).

use prospicio_core::{Error, Result};
use serde_json::{Map, Number, Value, json};

use crate::deal::Deduction;
use crate::pnl::ExpenseBasis;
use crate::reinsurance::{Basis, Commission, Layer, LimitAllocation, LossSensitivePremium, Tower};

/// Value of the `format` field.
const FORMAT: &str = "risk_rs.tower";
/// The newest format version this build reads. A document is stamped at
/// the oldest version that reads it: 3 when a layer has brokerage,
/// deductions or a scale read before annual terms, 2 when a layer carries
/// deficits or credits over a term (which older readers would drop), else
/// 1.
const FORMAT_VERSION: u64 = 3;

impl Tower {
    /// The programme as a JSON document; [`from_json`](Self::from_json)
    /// reads it back to an equal tower.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
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
        let carries = self
            .layers
            .iter()
            .any(|l| l.profit_commission_carry.is_some() || l.sliding_scale_carry.is_some());
        let deal = self.layers.iter().any(|l| {
            l.brokerage.is_some()
                || !l.deductions.is_empty()
                || l.scales_before_annual_terms
                || l.brokerage_within_commission_range
        });
        let doc = json!({
            "format": FORMAT,
            "format_version": if deal { 3 } else if carries { 2 } else { 1 },
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
    // Written only when not the default, so older documents read the same.
    if l.allocation == LimitAllocation::ProRata {
        m.insert("allocation".into(), json!("pro_rata"));
    }
    if let Some(c) = &l.corridor {
        m.insert(
            "loss_corridor".into(),
            json!({"lower": num(c.lower), "upper": num(c.upper), "retained": num(c.retained)}),
        );
    }
    match &l.commission {
        Some(Commission::Flat(c)) => {
            m.insert("ceding_commission".into(), num(*c));
        }
        Some(Commission::SlidingScale(anchors)) => {
            let pairs = anchors
                .iter()
                .map(|&(c, lr)| Value::Array(vec![num(c), num(lr)]))
                .collect();
            m.insert("sliding_scale".into(), Value::Array(pairs));
        }
        None => {}
    }
    if let Some(pc) = &l.profit_commission {
        m.insert(
            "profit_commission".into(),
            json!({"share": num(pc.share), "allowance": num(pc.allowance)}),
        );
    }
    if let Some(cf) = &l.profit_commission_carry {
        m.insert("profit_commission_carry".into(), json!({"years": cf.years}));
    }
    if let Some(cf) = &l.sliding_scale_carry {
        m.insert(
            "sliding_scale_carry".into(),
            json!({"deficit": cf.deficit, "credit": cf.credit, "years": cf.years}),
        );
    }
    if let Some(s) = &l.swing {
        m.insert(
            "swing".into(),
            json!({"basic": num(s.basic), "lcm": num(s.lcm),
                   "minimum": num(s.minimum), "maximum": num(s.maximum)}),
        );
    }
    match &l.brokerage {
        Some(Commission::Flat(b)) => {
            m.insert("brokerage".into(), num(*b));
        }
        Some(Commission::SlidingScale(anchors)) => {
            let pairs = anchors
                .iter()
                .map(|&(b, lr)| Value::Array(vec![num(b), num(lr)]))
                .collect();
            m.insert("brokerage_scale".into(), Value::Array(pairs));
        }
        None => {}
    }
    if !l.deductions.is_empty() {
        let list = l
            .deductions
            .iter()
            .map(|d| {
                json!({"label": d.label, "basis": d.basis.name(), "value": num(d.value),
                       "inside_nmbf": d.inside_nmbf})
            })
            .collect();
        m.insert("deductions".into(), Value::Array(list));
    }
    if l.scales_before_annual_terms {
        m.insert("scales_before_annual_terms".into(), json!(true));
    }
    if l.brokerage_within_commission_range {
        m.insert("brokerage_within_commission_range".into(), json!(true));
    }
    Value::Object(m)
}

/// A carry's `years`: a positive integer, or null without limit.
fn carry_years(cf: &Map<String, Value>) -> Result<Option<u32>> {
    match cf.get("years") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| Error::Data("a carry's years must be a whole number or null".into())),
    }
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
    // Absent in documents written before it, and for event order.
    match m.get("allocation") {
        None => {}
        Some(Value::String(a)) if a == "event_order" => {}
        Some(Value::String(a)) if a == "pro_rata" => {
            layer = layer.allocation(LimitAllocation::ProRata)?;
        }
        Some(_) => {
            return Err(Error::Data(
                "allocation must be \"event_order\" or \"pro_rata\"".into(),
            ));
        }
    }
    // Absent in documents written before corridors, and for a layer
    // without one.
    match m.get("loss_corridor") {
        None | Some(Value::Null) => {}
        Some(Value::Object(c)) => {
            layer = layer.loss_corridor(
                float(c, "lower")?,
                float(c, "upper")?,
                float(c, "retained")?,
            )?;
        }
        Some(_) => return Err(Error::Data("loss_corridor must be an object".into())),
    }
    // Contract terms, after the premium they read; absent in documents
    // written before them.
    if m.contains_key("ceding_commission") {
        layer = layer.ceding_commission(float(m, "ceding_commission")?)?;
    }
    match m.get("sliding_scale") {
        None | Some(Value::Null) => {}
        Some(Value::Array(pairs)) => {
            let anchors = pairs
                .iter()
                .map(|p| match p.as_array().map(Vec::as_slice) {
                    Some([c, lr]) => {
                        Ok((to_f64(c, "sliding_scale")?, to_f64(lr, "sliding_scale")?))
                    }
                    _ => Err(Error::Data(
                        "sliding_scale must list [commission, loss_ratio] pairs".into(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            layer = layer.sliding_scale(anchors)?;
        }
        Some(_) => return Err(Error::Data("sliding_scale must be a list".into())),
    }
    match m.get("profit_commission") {
        None | Some(Value::Null) => {}
        Some(Value::Object(pc)) => {
            layer = layer.profit_commission(float(pc, "share")?, float(pc, "allowance")?)?;
        }
        Some(_) => return Err(Error::Data("profit_commission must be an object".into())),
    }
    // Carry-forward over a term, after the terms it carries.
    match m.get("profit_commission_carry") {
        None | Some(Value::Null) => {}
        Some(Value::Object(cf)) => {
            layer = layer.profit_commission_carry_forward(carry_years(cf)?)?;
        }
        Some(_) => {
            return Err(Error::Data(
                "profit_commission_carry must be an object".into(),
            ));
        }
    }
    match m.get("sliding_scale_carry") {
        None | Some(Value::Null) => {}
        Some(Value::Object(cf)) => {
            let flag = |k: &str| {
                cf.get(k)
                    .and_then(Value::as_bool)
                    .ok_or_else(|| Error::Data(format!("sliding_scale_carry needs a boolean {k}")))
            };
            layer = layer.sliding_scale_carry_forward(
                flag("deficit")?,
                flag("credit")?,
                carry_years(cf)?,
            )?;
        }
        Some(_) => return Err(Error::Data("sliding_scale_carry must be an object".into())),
    }
    match m.get("swing") {
        None | Some(Value::Null) => {}
        Some(Value::Object(s)) => {
            let terms = LossSensitivePremium::new(
                float(s, "basic")?,
                float(s, "lcm")?,
                Some(float(s, "minimum")?),
                Some(float(s, "maximum")?),
            )?;
            layer = layer.swing_rated(terms)?;
        }
        Some(_) => return Err(Error::Data("swing must be an object".into())),
    }
    // A deal's deductions, after the premium and commission they read.
    if m.contains_key("brokerage") {
        layer = layer.brokerage(float(m, "brokerage")?)?;
    }
    match m.get("brokerage_scale") {
        None | Some(Value::Null) => {}
        Some(Value::Array(pairs)) => {
            let anchors = pairs
                .iter()
                .map(|p| match p.as_array().map(Vec::as_slice) {
                    Some([b, lr]) => Ok((
                        to_f64(b, "brokerage_scale")?,
                        to_f64(lr, "brokerage_scale")?,
                    )),
                    _ => Err(Error::Data(
                        "brokerage_scale must list [rate, loss_ratio] pairs".into(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            layer = layer.brokerage_scale(anchors)?;
        }
        Some(_) => return Err(Error::Data("brokerage_scale must be a list".into())),
    }
    match m.get("deductions") {
        None | Some(Value::Null) => {}
        Some(Value::Array(list)) => {
            for d in list {
                let d = object(d, "deduction")?;
                let label = d
                    .get("label")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Data("a deduction's label must be a string".into()))?;
                let basis = match d.get("basis").and_then(Value::as_str) {
                    Some("premium") => ExpenseBasis::Premium,
                    Some("loss") => ExpenseBasis::Loss,
                    Some("fixed") => ExpenseBasis::Fixed,
                    other => {
                        return Err(Error::Data(format!(
                            "a deduction's basis must be premium, loss or fixed, not {other:?}"
                        )));
                    }
                };
                let mut deduction = Deduction::new(label, basis, float(d, "value")?)?;
                match d.get("inside_nmbf") {
                    None | Some(Value::Bool(true)) => {}
                    Some(Value::Bool(false)) => deduction = deduction.outside_nmbf(),
                    Some(_) => {
                        return Err(Error::Data("inside_nmbf must be a boolean".into()));
                    }
                }
                layer = layer.deduction(deduction)?;
            }
        }
        Some(_) => return Err(Error::Data("deductions must be a list".into())),
    }
    for (k, set) in [
        (
            "scales_before_annual_terms",
            Layer::scales_before_annual_terms as fn(Layer) -> Layer,
        ),
        (
            "brokerage_within_commission_range",
            Layer::brokerage_within_commission_range,
        ),
    ] {
        match m.get(k) {
            None | Some(Value::Null | Value::Bool(false)) => {}
            Some(Value::Bool(true)) => layer = set(layer),
            Some(_) => return Err(Error::Data(format!("{k} must be a boolean"))),
        }
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
    use crate::{LossSensitivePremium, simulate_events};
    use prospicio_prob::{Lognormal, Poisson};

    fn programme() -> Tower {
        Tower::inuring(vec![
            vec![
                Layer::quota_share("QS 30%", 0.3)
                    .unwrap()
                    .loss_corridor(1e6, 3e6, 0.5)
                    .unwrap()
                    .premium_rate(1.0, 2e7)
                    .unwrap()
                    .sliding_scale(vec![(0.35, 0.55), (0.3, 0.6), (0.2, 0.75)])
                    .unwrap()
                    .profit_commission(0.15, 0.35)
                    .unwrap()
                    .sliding_scale_carry_forward(true, true, Some(3))
                    .unwrap()
                    .profit_commission_carry_forward(None)
                    .unwrap(),
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
                    .unwrap()
                    .ceding_commission(0.1)
                    .unwrap(),
                Layer::xol("free", 5e6, 3e6)
                    .unwrap()
                    .reinstatements(2)
                    .unwrap()
                    .swing_rated(
                        LossSensitivePremium::new(1e5, 1.25, Some(2e5), Some(2e6)).unwrap(),
                    )
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
        for key in [
            "sliding_scale",
            "profit_commission",
            "ceding_commission",
            "swing",
            "sliding_scale_carry",
            "profit_commission_carry",
        ] {
            assert!(text.contains(key), "{key}");
        }
        // A carry needs version 2; a tower without one stays version 1.
        assert!(text.contains("\"format_version\":2"));
        let plain = Tower::new(vec![Layer::xol("L", 1.0, 1.0).unwrap()]).unwrap();
        assert!(plain.to_json().contains("\"format_version\":1"));
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
    fn a_deals_deductions_round_trip() {
        let layer = Layer::quota_share("QS", 1.0)
            .unwrap()
            .aggregate_limit(120.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
            .unwrap()
            .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
            .unwrap()
            .deduction(Deduction::premium("tax", 0.03).unwrap())
            .unwrap()
            .deduction(Deduction::loss("tpa", 0.02).unwrap())
            .unwrap()
            .deduction(Deduction::fixed("fronting", 1.5).unwrap().outside_nmbf())
            .unwrap()
            .scales_before_annual_terms()
            .brokerage_within_commission_range();
        let flat = Layer::xol("XL", 50.0, 50.0)
            .unwrap()
            .premium_rate(1.0, 20.0)
            .unwrap()
            .brokerage(0.1)
            .unwrap();
        let tower = Tower::new(vec![layer, flat]).unwrap();
        let text = tower.to_json();
        assert!(text.contains("\"format_version\":3"));
        for key in [
            "brokerage_scale",
            "\"brokerage\":",
            "deductions",
            "scales_before_annual_terms",
            "brokerage_within_commission_range",
            "\"inside_nmbf\":false",
        ] {
            assert!(text.contains(key), "{key}");
        }
        let back = Tower::from_json(&text).unwrap();
        assert_eq!(back, tower);
        assert_eq!(back.to_json(), text);
        let bad = text.replace("\"basis\":\"loss\"", "\"basis\":\"lots\"");
        assert!(Tower::from_json(&bad).is_err());
        let bad = text.replace(
            "\"scales_before_annual_terms\":true",
            "\"scales_before_annual_terms\":1",
        );
        assert!(Tower::from_json(&bad).is_err());
        let bad = text.replace("\"label\":\"tpa\"", "\"label\":\"tax\"");
        assert!(Tower::from_json(&bad).is_err());
    }

    #[test]
    fn the_limit_allocation_round_trips() {
        let layer = Layer::xol("L", 10.0, 5.0)
            .unwrap()
            .reinstatements(1)
            .unwrap()
            .allocation(LimitAllocation::ProRata)
            .unwrap();
        let tower = Tower::new(vec![layer]).unwrap();
        let json = tower.to_json();
        assert!(json.contains("\"allocation\":\"pro_rata\""));
        let back = Tower::from_json(&json).unwrap();
        assert_eq!(back.layers[0].allocation, LimitAllocation::ProRata);
        assert_eq!(back, tower);
        // Event order is the default and is not written; it reads back
        // from either spelling.
        let plain = Tower::new(vec![Layer::xol("L", 10.0, 5.0).unwrap()])
            .unwrap()
            .to_json();
        assert!(!plain.contains("allocation"));
        let spelled = plain.replace(
            "\"pro_rata_time\":false",
            "\"pro_rata_time\":false,\"allocation\":\"event_order\"",
        );
        assert_eq!(
            Tower::from_json(&spelled).unwrap().layers[0].allocation,
            LimitAllocation::EventOrder
        );
        let bad = plain.replace(
            "\"pro_rata_time\":false",
            "\"pro_rata_time\":false,\"allocation\":3",
        );
        assert!(Tower::from_json(&bad).is_err());
    }

    #[test]
    fn refuses_bad_documents() {
        let good = Tower::new(vec![Layer::xol("L", 1.0, 1.0).unwrap()])
            .unwrap()
            .to_json();
        assert!(Tower::from_json("not json").is_err());
        assert!(Tower::from_json(&good.replace("risk_rs.tower", "risk_rs.distribution")).is_err());
        assert!(
            Tower::from_json(&good.replace("\"format_version\":1", "\"format_version\":4"))
                .is_err()
        );
        assert!(Tower::from_json(&good.replace("\"loss\"", "\"cat\"")).is_err());
        // A term the builders refuse.
        assert!(Tower::from_json(&good.replace("\"share\":1.0", "\"share\":1.5")).is_err());
        // A document without a corridor (as all were before corridors)
        // loads with none; a corridor upside down is refused.
        assert!(!good.contains("loss_corridor"));
        assert_eq!(Tower::from_json(&good).unwrap().layers[0].corridor, None);
        let with = |c: &str| {
            good.replace(
                "\"pro_rata_time\":false",
                &format!("\"pro_rata_time\":false,\"loss_corridor\":{c}"),
            )
        };
        let ok = Tower::from_json(&with(r#"{"lower":1.0,"upper":2.0,"retained":1.0}"#)).unwrap();
        assert_eq!(ok.layers[0].corridor.unwrap().upper, 2.0);
        assert!(Tower::from_json(&with(r#"{"lower":2.0,"upper":1.0,"retained":1.0}"#)).is_err());
        assert!(Tower::from_json(&with("3")).is_err());
        // Contract terms: a sliding scale needs the premium; terms that do
        // not combine are refused; a malformed scale is refused.
        assert!(Tower::from_json(&with(r#"null,"sliding_scale":[[0.3,0.6]]"#)).is_err());
        let priced = good.replace("\"premium\":0.0", "\"premium\":1.0");
        let terms = |t: &str| {
            priced.replace(
                "\"pro_rata_time\":false",
                &format!("\"pro_rata_time\":false,{t}"),
            )
        };
        let slid = Tower::from_json(&terms(r#""sliding_scale":[[0.3,0.6],[0.2,0.7]]"#)).unwrap();
        assert!(slid.layers[0].commission.is_some());
        let swing = r#""swing":{"basic":0.0,"lcm":1.0,"minimum":0.0,"maximum":"inf"}"#;
        assert!(
            Tower::from_json(&terms(swing)).unwrap().layers[0]
                .swing
                .is_some()
        );
        assert!(
            Tower::from_json(&terms(&format!(r#""sliding_scale":[[0.3,0.6]],{swing}"#))).is_err()
        );
        assert!(Tower::from_json(&terms(r#""sliding_scale":[[0.3]]"#)).is_err());
        assert!(Tower::from_json(&terms(r#""profit_commission":{"share":0.2}"#)).is_err());
        assert!(Tower::from_json(&good.replace("\"stages\":[[", "\"stages\":[[],[")).is_err());
    }
}
