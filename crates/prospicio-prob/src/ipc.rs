//! [`PredictiveDistribution`] to and from Arrow IPC files (the `arrow`
//! feature).
//!
//! # Format, version 1
//!
//! One `Float64` column per component, in component order, with no nulls:
//! column `j` is component `j`'s draws and row `i` is simulation `i`. Any
//! Arrow reader (pyarrow, R `arrow`, Polars) therefore sees a table whose
//! columns are the marginals and whose rows are joint draws.
//!
//! Schema metadata:
//!
//! | Key | Value |
//! |---|---|
//! | `risk_rs.format` | `"predictive_distribution"` |
//! | `risk_rs.format_version` | `"1"` |
//! | `risk_rs.dims` | JSON array of dimension names |
//! | `risk_rs.provenance` | JSON object, see below |
//!
//! Each field has `risk_rs.key`: a JSON array with one tagged value per
//! dimension, `{"int": 2019}`, `{"text": "Auto"}` or
//! `{"period": {"start": "2019-01", "grain": "Y"}}` (grain `M`, `Q`, `S` or
//! `Y`). The field name is the key's values joined by `/` (`"Auto/2019"`,
//! or `"total"` for an empty key); it is for people, and readers use
//! `risk_rs.key`.
//!
//! The provenance object has `model`, `parameters` and `versions` (arrays
//! of `[name, value]` pairs), and `seed`, `stream_scheme` and `input_hash`
//! (strings or `null`). The seed is a decimal string because JSON numbers
//! lose precision above 2^53.
//!
//! A change to any of this is a new `format_version`. Readers reject
//! versions they do not know.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use prospicio_core::{Grain, Month, Period};
use serde_json::{Value, json};

use crate::predictive::{ComponentKey, KeyValue, PredictiveDistribution};
use crate::provenance::Provenance;

/// Schema metadata key naming the format.
pub const FORMAT_KEY: &str = "risk_rs.format";
/// Value of [`FORMAT_KEY`] for a `PredictiveDistribution`.
pub const FORMAT: &str = "predictive_distribution";
/// Schema metadata key holding the format version.
pub const FORMAT_VERSION_KEY: &str = "risk_rs.format_version";
/// The format version this module writes and reads.
pub const FORMAT_VERSION: &str = "1";
/// Schema metadata key holding the dimension names.
pub const DIMS_KEY: &str = "risk_rs.dims";
/// Schema metadata key holding the provenance.
pub const PROVENANCE_KEY: &str = "risk_rs.provenance";
/// Field metadata key holding a component's key.
pub const COMPONENT_KEY: &str = "risk_rs.key";

/// Why a distribution could not be written or read.
#[derive(Debug)]
pub enum IpcError {
    /// Arrow could not write or read the file.
    Arrow(ArrowError),
    /// The data is not a version-1 `PredictiveDistribution`.
    Format(String),
    /// The data is well formed but fails `PredictiveDistribution`'s checks
    /// (for example, a repeated key or a non-finite draw).
    Invalid(prospicio_core::Error),
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arrow(e) => write!(f, "arrow: {e}"),
            Self::Format(msg) => write!(f, "not a predictive distribution: {msg}"),
            Self::Invalid(e) => write!(f, "invalid predictive distribution: {e}"),
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arrow(e) => Some(e),
            Self::Format(_) => None,
            Self::Invalid(e) => Some(e),
        }
    }
}

impl From<ArrowError> for IpcError {
    fn from(e: ArrowError) -> Self {
        Self::Arrow(e)
    }
}

impl From<prospicio_core::Error> for IpcError {
    fn from(e: prospicio_core::Error) -> Self {
        Self::Invalid(e)
    }
}

fn format_error(msg: impl Into<String>) -> IpcError {
    IpcError::Format(msg.into())
}

impl PredictiveDistribution {
    /// This distribution as one Arrow record batch, in the format described
    /// in the [module docs](crate::ipc).
    pub fn to_record_batch(&self) -> Result<RecordBatch, IpcError> {
        let n = self.n_components();
        let mut fields = Vec::with_capacity(n);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(n);
        for (j, key) in self.components().iter().enumerate() {
            let key_json = Value::Array(key.iter().map(key_value_to_json).collect());
            let field = Field::new(field_name(key), DataType::Float64, false).with_metadata(
                HashMap::from([(COMPONENT_KEY.to_string(), key_json.to_string())]),
            );
            let column: Float64Array = self
                .draw_matrix()
                .iter()
                .skip(j)
                .step_by(n)
                .copied()
                .collect();
            fields.push(field);
            columns.push(Arc::new(column));
        }
        let metadata = HashMap::from([
            (FORMAT_KEY.to_string(), FORMAT.to_string()),
            (FORMAT_VERSION_KEY.to_string(), FORMAT_VERSION.to_string()),
            (DIMS_KEY.to_string(), json!(self.dims()).to_string()),
            (
                PROVENANCE_KEY.to_string(),
                provenance_to_json(self.provenance()).to_string(),
            ),
        ]);
        let schema = Schema::new_with_metadata(fields, metadata);
        Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
    }

    /// The distribution in `batch`, as written by
    /// [`to_record_batch`](Self::to_record_batch).
    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self, IpcError> {
        from_batches(batch.schema_ref(), std::slice::from_ref(batch))
    }

    /// Writes this distribution to `w` as an Arrow IPC file (Feather v2),
    /// readable by `pyarrow.feather.read_table` or R's
    /// `arrow::read_feather`.
    pub fn write_ipc<W: Write>(&self, w: W) -> Result<(), IpcError> {
        let batch = self.to_record_batch()?;
        let mut writer = FileWriter::try_new(w, batch.schema_ref())?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    /// Reads a distribution written by [`write_ipc`](Self::write_ipc). The
    /// file may hold several record batches; their rows are the simulations
    /// in order.
    pub fn read_ipc<R: Read + Seek>(r: R) -> Result<Self, IpcError> {
        let reader = FileReader::try_new(r, None)?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        from_batches(&schema, &batches)
    }
}

fn from_batches(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<PredictiveDistribution, IpcError> {
    let meta = schema.metadata();
    let get = |k: &str| {
        meta.get(k)
            .ok_or_else(|| format_error(format!("schema metadata has no {k}")))
    };
    if get(FORMAT_KEY)? != FORMAT {
        return Err(format_error(format!("{FORMAT_KEY} is not {FORMAT:?}")));
    }
    let version = get(FORMAT_VERSION_KEY)?;
    if version != FORMAT_VERSION {
        return Err(format_error(format!(
            "format version {version:?} is not supported (expected {FORMAT_VERSION:?})"
        )));
    }
    let dims: Vec<String> = parse_json(get(DIMS_KEY)?, DIMS_KEY)?
        .as_array()
        .and_then(|a| a.iter().map(|d| d.as_str().map(String::from)).collect())
        .ok_or_else(|| format_error(format!("{DIMS_KEY} is not an array of strings")))?;
    let provenance = provenance_from_json(&parse_json(get(PROVENANCE_KEY)?, PROVENANCE_KEY)?)?;

    let mut components = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if field.data_type() != &DataType::Float64 {
            return Err(format_error(format!(
                "column {:?} is {}, not Float64",
                field.name(),
                field.data_type()
            )));
        }
        let raw = field.metadata().get(COMPONENT_KEY).ok_or_else(|| {
            format_error(format!("column {:?} has no {COMPONENT_KEY}", field.name()))
        })?;
        components.push(component_key_from_json(&parse_json(raw, COMPONENT_KEY)?)?);
    }

    let n_components = components.len();
    let n_sims: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let mut draws = vec![0.0; n_sims * n_components];
    let mut first_row = 0;
    for batch in batches {
        for (j, column) in batch.columns().iter().enumerate() {
            let column = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| format_error("a column is not Float64"))?;
            if column.null_count() > 0 {
                return Err(format_error(format!(
                    "column {:?} has nulls",
                    schema.field(j).name()
                )));
            }
            for (i, &x) in column.values().iter().enumerate() {
                draws[(first_row + i) * n_components + j] = x;
            }
        }
        first_row += batch.num_rows();
    }
    Ok(PredictiveDistribution::from_draws(
        dims, components, draws, provenance,
    )?)
}

fn parse_json(raw: &str, what: &str) -> Result<Value, IpcError> {
    serde_json::from_str(raw).map_err(|e| format_error(format!("{what} is not JSON: {e}")))
}

fn field_name(key: &ComponentKey) -> String {
    if key.is_empty() {
        return "total".into();
    }
    key.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("/")
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

fn key_value_to_json(v: &KeyValue) -> Value {
    match v {
        KeyValue::Int(i) => json!({ "int": i }),
        KeyValue::Text(s) => json!({ "text": s }),
        KeyValue::Period(p) => json!({ "period": {
            "start": p.start().to_string(),
            "grain": grain_code(p.grain()),
        }}),
    }
}

fn component_key_from_json(v: &Value) -> Result<ComponentKey, IpcError> {
    v.as_array()
        .ok_or_else(|| format_error(format!("{COMPONENT_KEY} is not an array")))?
        .iter()
        .map(key_value_from_json)
        .collect()
}

fn key_value_from_json(v: &Value) -> Result<KeyValue, IpcError> {
    let bad = || format_error(format!("{v} is not a key value"));
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
                return Err(format_error(format!(
                    "period start {start} is not the first month of a {grain} period"
                )));
            }
            Ok(KeyValue::Period(period))
        }
        _ => Err(bad()),
    }
}

fn pairs_to_json(pairs: &[(String, String)]) -> Value {
    pairs.iter().map(|(k, v)| json!([k, v])).collect()
}

fn pairs_from_json(v: Option<&Value>, what: &str) -> Result<Vec<(String, String)>, IpcError> {
    let bad = || {
        format_error(format!(
            "provenance {what} is not an array of [name, value]"
        ))
    };
    v.and_then(Value::as_array)
        .ok_or_else(bad)?
        .iter()
        .map(|pair| match pair.as_array().map(Vec::as_slice) {
            Some([Value::String(k), Value::String(v)]) => Ok((k.clone(), v.clone())),
            _ => Err(bad()),
        })
        .collect()
}

fn provenance_to_json(p: &Provenance) -> Value {
    json!({
        "model": p.model,
        "parameters": pairs_to_json(&p.parameters),
        "seed": p.seed.map(|s| s.to_string()),
        "stream_scheme": p.stream_scheme,
        "versions": pairs_to_json(&p.versions),
        "input_hash": p.input_hash,
    })
}

fn provenance_from_json(v: &Value) -> Result<Provenance, IpcError> {
    let optional_str = |name: &str| -> Result<Option<String>, IpcError> {
        match v.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format_error(format!("provenance {name} is not a string"))),
        }
    };
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| format_error("provenance has no model"))?
        .to_string();
    let seed = optional_str("seed")?
        .map(|s| {
            s.parse()
                .map_err(|_| format_error(format!("provenance seed {s:?} is not a u64")))
        })
        .transpose()?;
    Ok(Provenance {
        model,
        parameters: pairs_from_json(v.get("parameters"), "parameters")?,
        seed,
        stream_scheme: optional_str("stream_scheme")?,
        versions: pairs_from_json(v.get("versions"), "versions")?,
        input_hash: optional_str("input_hash")?,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

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
                1.0, 2.0, 3.0, -0.5, 5.5, 6.25, 7.0, 8.0, 9.0e12, 10.0, 11.0, 12.0,
            ],
            Provenance::new("test")
                .param("n_sims", 4)
                .param("note", "a \"quoted\" value")
                .seed(u64::MAX, SIM_INDEX_SCHEME)
                .input_hash(InputHasher::new().str("tri").finish()),
        )
        .unwrap()
    }

    fn assert_same(a: &PredictiveDistribution, b: &PredictiveDistribution) {
        assert_eq!(a.dims(), b.dims());
        assert_eq!(a.components(), b.components());
        assert_eq!(a.n_sims(), b.n_sims());
        assert_eq!(a.draw_matrix(), b.draw_matrix());
        assert_eq!(a.provenance(), b.provenance());
    }

    #[test]
    fn ipc_round_trip() {
        let pd = sample();
        let mut buf = Vec::new();
        pd.write_ipc(&mut buf).unwrap();
        let back = PredictiveDistribution::read_ipc(Cursor::new(buf)).unwrap();
        assert_same(&pd, &back);
    }

    #[test]
    fn record_batch_layout() {
        let batch = sample().to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 3);
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["Auto/2020Q1/1", "Auto/2020Q2/1", "Home/2019/2"]);
        let col1 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(col1.values(), &[2.0, 5.5, 8.0, 11.0]);
        let schema = batch.schema();
        assert_eq!(
            schema.field(1).metadata()[COMPONENT_KEY],
            r#"[{"text":"Auto"},{"period":{"grain":"Q","start":"2020-04"}},{"int":1}]"#
        );
        let prov: Value = serde_json::from_str(&batch.schema().metadata()[PROVENANCE_KEY]).unwrap();
        assert_eq!(prov["seed"], json!(u64::MAX.to_string()));
        assert_eq!(prov["parameters"][0], json!(["n_sims", "4"]));
    }

    #[test]
    fn several_batches_are_concatenated() {
        let pd = sample();
        let batch = pd.to_record_batch().unwrap();
        let mut buf = Vec::new();
        let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        writer.write(&batch.slice(0, 1)).unwrap();
        writer.write(&batch.slice(1, 3)).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let back = PredictiveDistribution::read_ipc(Cursor::new(buf)).unwrap();
        assert_same(&pd, &back);
    }

    #[test]
    fn empty_key_is_named_total() {
        let total = sample().aggregate(&[]).unwrap();
        let batch = total.to_record_batch().unwrap();
        assert_eq!(batch.schema().field(0).name(), "total");
        assert_same(
            &total,
            &PredictiveDistribution::from_record_batch(&batch).unwrap(),
        );
    }

    fn with_schema_meta(batch: &RecordBatch, k: &str, v: Option<&str>) -> RecordBatch {
        let mut meta = batch.schema().metadata().clone();
        match v {
            Some(v) => meta.insert(k.into(), v.into()),
            None => meta.remove(k),
        };
        let schema = Schema::new_with_metadata(batch.schema().fields().clone(), meta);
        RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap()
    }

    fn with_key(batch: &RecordBatch, j: usize, key: &str) -> RecordBatch {
        let schema = batch.schema();
        let fields: Vec<Field> = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let f = f.as_ref().clone();
                if i == j {
                    f.with_metadata(HashMap::from([(COMPONENT_KEY.into(), key.into())]))
                } else {
                    f
                }
            })
            .collect();
        let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
        RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap()
    }

    fn format_err(batch: &RecordBatch) -> String {
        match PredictiveDistribution::from_record_batch(batch) {
            Err(IpcError::Format(msg)) => msg,
            other => panic!("expected a format error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_other_formats_and_versions() {
        let batch = sample().to_record_batch().unwrap();
        assert!(format_err(&with_schema_meta(&batch, FORMAT_KEY, None)).contains(FORMAT_KEY));
        assert!(
            format_err(&with_schema_meta(&batch, FORMAT_KEY, Some("triangle"))).contains(FORMAT)
        );
        assert!(
            format_err(&with_schema_meta(&batch, FORMAT_VERSION_KEY, Some("2"))).contains("\"2\"")
        );
        assert!(format_err(&with_schema_meta(&batch, DIMS_KEY, Some("[1]"))).contains(DIMS_KEY));
        assert!(
            format_err(&with_schema_meta(&batch, PROVENANCE_KEY, Some("{}"))).contains("model")
        );
        assert!(
            format_err(&with_schema_meta(
                &batch,
                PROVENANCE_KEY,
                Some(r#"{"model":"m","parameters":[],"versions":[],"seed":"-1"}"#)
            ))
            .contains("seed")
        );
    }

    #[test]
    fn rejects_bad_keys() {
        let batch = sample().to_record_batch().unwrap();
        for bad in [
            "{}",
            "[{}]",
            r#"[{"int":"1"},{"int":1},{"int":1}]"#,
            r#"[{"int":1,"text":"a"},{"int":1},{"int":1}]"#,
            r#"[{"date":"x"},{"int":1},{"int":1}]"#,
            r#"[{"text":"a"},{"period":{"start":"2020-13","grain":"Q"}},{"int":1}]"#,
            r#"[{"text":"a"},{"period":{"start":"2020-04","grain":"W"}},{"int":1}]"#,
        ] {
            format_err(&with_key(&batch, 0, bad));
        }
        let msg = format_err(&with_key(
            &batch,
            0,
            r#"[{"text":"a"},{"period":{"start":"2020-05","grain":"Q"}},{"int":1}]"#,
        ));
        assert!(msg.contains("first month"), "{msg}");
    }

    #[test]
    fn rejects_non_float_and_null_columns() {
        let batch = sample().to_record_batch().unwrap();
        let schema = batch.schema();

        let nulls: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(1.0),
            Some(1.0),
        ]));
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields[0] = fields[0].clone().with_nullable(true);
        let mut cols = batch.columns().to_vec();
        cols[0] = nulls;
        let b = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            cols,
        )
        .unwrap();
        assert!(format_err(&b).contains("nulls"));

        let ints: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3, 4]));
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields[0] = fields[0].clone().with_data_type(DataType::Int64);
        let mut cols = batch.columns().to_vec();
        cols[0] = ints;
        let b = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            cols,
        )
        .unwrap();
        assert!(format_err(&b).contains("not Float64"));
    }

    #[test]
    fn invalid_contents_are_reported_as_invalid() {
        let batch = sample().to_record_batch().unwrap();
        // Two columns with the same key.
        let key = batch.schema().field(0).metadata()[COMPONENT_KEY].clone();
        let dup = with_key(&batch, 1, &key);
        assert!(matches!(
            PredictiveDistribution::from_record_batch(&dup),
            Err(IpcError::Invalid(_))
        ));
    }

    #[test]
    fn not_an_arrow_file() {
        assert!(matches!(
            PredictiveDistribution::read_ipc(Cursor::new(b"not arrow".to_vec())),
            Err(IpcError::Arrow(_))
        ));
    }
}
