//! Saving and loading deals and books: [`DealRun::to_json`] and
//! [`DealBook::to_json`], versioned JSON documents of every stored part
//! (#978), which the bindings' pickling uses.
//!
//! A run holds its name, premium, loss, deductions, inception month, and
//! its monthly patterns and investor flows when it has them; a book holds
//! its runs in order. Numbers round-trip bit for bit. Reading rebuilds
//! through the same validated builders ([`DealRun::new`],
//! [`MonthlyPatterns::new`],
//! [`DealRun::deduction`], [`DealRun::flows`], [`DealBook::new`]), so a
//! document that would not build is refused.

use prospicio_core::{Error, Result};
use serde_json::{Map, Value, json};

use super::{DealBook, DealRun};
use crate::cashflow::MonthlyPatterns;

const RUN_FORMAT: &str = "risk_rs.deal_run";
const BOOK_FORMAT: &str = "risk_rs.deal_book";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

fn bad(what: &str) -> Error {
    Error::Data(format!("deal document field {what:?} is invalid"))
}

fn field<'a>(o: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    o.get(key)
        .ok_or_else(|| Error::Data(format!("deal document has no {key:?}")))
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| bad(what))
}

fn string(o: &Map<String, Value>, key: &str) -> Result<String> {
    field(o, key)?
        .as_str()
        .map(String::from)
        .ok_or_else(|| bad(key))
}

fn uint(o: &Map<String, Value>, key: &str) -> Result<usize> {
    field(o, key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| bad(key))
}

fn floats(o: &Map<String, Value>, key: &str) -> Result<Vec<f64>> {
    field(o, key)?
        .as_array()
        .ok_or_else(|| bad(key))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| bad(key)))
        .collect()
}

/// Parses `text` and checks its format and version.
fn parse(text: &str, format: &str) -> Result<Value> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| Error::Data(format!("deal document is not JSON: {e}")))?;
    let top = object(&doc, "document")?;
    if top.get("format").and_then(Value::as_str) != Some(format) {
        return Err(Error::Data(format!("not a {format} document")));
    }
    let version = top
        .get("format_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| bad("format_version"))?;
    if version == 0 || version > FORMAT_VERSION {
        return Err(Error::Data(format!(
            "deal document format version {version}; this build reads 1 to {FORMAT_VERSION}"
        )));
    }
    Ok(doc)
}

fn run_value(run: &DealRun) -> Value {
    json!({
        "name": run.name,
        "premium": run.premium,
        "loss": run.loss,
        "deductions": run.lines.iter().map(|l| json!({
            "name": l.name,
            "outside_nmbf": l.outside_nmbf,
            "values": l.values,
        })).collect::<Vec<_>>(),
        "inception": run.inception,
        "patterns": run.patterns.as_ref().map(|p| json!({
            "written": p.written,
            "earned": p.earned,
            "paid": p.paid,
        })),
        "flows": run.flows.as_ref().map(|f| json!({
            "months": f.months,
            "flows": f.flows,
            "funded": f.funded,
            "opening": f.opening,
        })),
    })
}

fn run_from(v: &Value) -> Result<DealRun> {
    let o = object(v, "deal")?;
    let mut run = DealRun::new(
        &string(o, "name")?,
        floats(o, "premium")?,
        floats(o, "loss")?,
    )?;
    for d in field(o, "deductions")?
        .as_array()
        .ok_or_else(|| bad("deductions"))?
    {
        let d = object(d, "deductions")?;
        let outside = field(d, "outside_nmbf")?
            .as_bool()
            .ok_or_else(|| bad("outside_nmbf"))?;
        run = run.deduction(&string(d, "name")?, floats(d, "values")?, outside)?;
    }
    run = run.inception(uint(o, "inception")?);
    match field(o, "flows")? {
        Value::Null => {}
        f => {
            let f = object(f, "flows")?;
            run = run.flows(
                uint(f, "months")?,
                floats(f, "flows")?,
                floats(f, "funded")?,
                floats(f, "opening")?,
            )?;
        }
    }
    match field(o, "patterns")? {
        Value::Null => {}
        p => {
            let p = object(p, "patterns")?;
            run = run.patterns(MonthlyPatterns::new(
                floats(p, "written")?,
                floats(p, "earned")?,
                floats(p, "paid")?,
            )?);
        }
    }
    Ok(run)
}

fn document(format: &str, key: &str, body: Value) -> String {
    let mut doc = Map::new();
    doc.insert("format".into(), format.into());
    doc.insert("format_version".into(), FORMAT_VERSION.into());
    doc.insert(key.into(), body);
    Value::Object(doc).to_string()
}

impl DealRun {
    /// The run as a versioned JSON document (format `risk_rs.deal_run`),
    /// which [`DealRun::from_json`] reads back to an equal run.
    ///
    /// ```
    /// use prospicio_pricing::book::DealRun;
    ///
    /// let run = DealRun::new("cat", vec![100.0, 100.0], vec![40.0, 150.0]).unwrap()
    ///     .deduction("brokerage", vec![2.5, 2.5], false).unwrap()
    ///     .inception(6);
    /// assert_eq!(DealRun::from_json(&run.to_json()).unwrap(), run);
    /// ```
    pub fn to_json(&self) -> String {
        document(RUN_FORMAT, "deal", run_value(self))
    }

    /// Reads a run that [`DealRun::to_json`] wrote. Fails on text that is
    /// not such a document, one of a newer format version, or a run that
    /// [`DealRun::new`] and its builders refuse.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, RUN_FORMAT)?;
        run_from(field(object(&doc, "document")?, "deal")?)
    }
}

impl DealBook {
    /// The book as a versioned JSON document (format `risk_rs.deal_book`),
    /// every run in order, which [`DealBook::from_json`] reads back to an
    /// equal book.
    ///
    /// ```
    /// use prospicio_pricing::book::{DealBook, DealRun};
    ///
    /// let a = DealRun::new("a", vec![100.0, 100.0], vec![40.0, 80.0]).unwrap();
    /// let b = DealRun::new("b", vec![300.0, 300.0], vec![150.0, 330.0]).unwrap().inception(3);
    /// let book = DealBook::new(vec![a, b]).unwrap();
    /// assert_eq!(DealBook::from_json(&book.to_json()).unwrap(), book);
    /// ```
    pub fn to_json(&self) -> String {
        document(
            BOOK_FORMAT,
            "deals",
            Value::Array(self.deals.iter().map(run_value).collect()),
        )
    }

    /// Reads a book that [`DealBook::to_json`] wrote. Fails on text that is
    /// not such a document, one of a newer format version, or deals that
    /// [`DealBook::new`] refuses.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text, BOOK_FORMAT)?;
        let deals = field(object(&doc, "document")?, "deals")?
            .as_array()
            .ok_or_else(|| bad("deals"))?
            .iter()
            .map(run_from)
            .collect::<Result<_>>()?;
        DealBook::new(deals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_run() -> DealRun {
        DealRun::new("cat", vec![100.0, 100.1], vec![40.0, 0.1 + 0.2])
            .unwrap()
            .deduction("brokerage", vec![2.5, 2.5], false)
            .unwrap()
            .deduction("fee", vec![1.0, 1.25], true)
            .unwrap()
            .inception(6)
            .flows(
                2,
                vec![-10.0, 3.5, -12.0, 1.0 / 3.0],
                vec![10.0, 12.0],
                vec![5.0, 5.0],
            )
            .unwrap()
            .patterns(MonthlyPatterns {
                written: vec![1.0, 1.0],
                earned: vec![0.5, 1.0],
                paid: vec![0.1, 0.4],
            })
    }

    #[test]
    fn runs_and_books_round_trip_with_every_part() {
        let run = full_run();
        let back = DealRun::from_json(&run.to_json()).unwrap();
        assert_eq!(back, run);
        let other = DealRun::new("b", vec![1.0, 2.0], vec![0.0, 3.0]).unwrap();
        let book = DealBook::new(vec![run, other]).unwrap();
        assert_eq!(DealBook::from_json(&book.to_json()).unwrap(), book);
    }

    #[test]
    fn bad_documents_are_refused() {
        let text = full_run().to_json();
        assert!(DealBook::from_json(&text).is_err(), "wrong format");
        assert!(
            DealRun::from_json(&text.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        assert!(DealRun::from_json(&text.replace("\"loss\":[40.0,", "\"loss\":[")).is_err());
        assert!(
            DealRun::from_json(&text.replace("\"name\":\"fee\"", "\"name\":\"margin\"")).is_err()
        );
        assert!(DealRun::from_json("{}").is_err());
        let a = DealRun::new("a", vec![1.0], vec![0.0]).unwrap();
        let mut doc: Value =
            serde_json::from_str(&DealBook::new(vec![a]).unwrap().to_json()).unwrap();
        let deals = doc["deals"].as_array_mut().unwrap();
        deals.push(deals[0].clone());
        assert!(
            DealBook::from_json(&doc.to_string()).is_err(),
            "two deals named a"
        );
    }
}
