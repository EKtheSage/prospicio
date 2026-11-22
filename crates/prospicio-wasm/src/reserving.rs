//! Reserving: triangles, the chain ladder, Mack, Bornhuetter-Ferguson and
//! the ODP bootstrap.

use crate::convert::{
    Fields, Obj, Pairs, err, numbers, options as read_options, read, strings, table,
};
use crate::predictive::Predictive;
use js_sys::Array;
use prospicio_reserving as core;
use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, ReserveFit, SegmentFits};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// A loss triangle: segments by measure columns by origin periods by
/// development ages.
#[wasm_bindgen]
pub struct Triangle {
    inner: core::Triangle,
}

/// An origin as JavaScript gives it: a year (`2020`) or a month as text
/// (`"2020-04"`, or `"2020"` for January).
#[derive(Deserialize)]
#[serde(untagged)]
enum Origin {
    Year(i32),
    Text(String),
}

impl Origin {
    fn month(&self) -> Result<Month, JsError> {
        let bad = |s: &str| JsError::new(&format!("origin {s:?} is not a year or \"YYYY-MM\""));
        match self {
            Self::Year(y) => Ok(Month::january(*y)),
            Self::Text(s) => {
                let (y, m) = s.split_once('-').unwrap_or((s.as_str(), "1"));
                let year: i32 = y.trim().parse().map_err(|_| bad(s))?;
                let month: u8 = m.trim().parse().map_err(|_| bad(s))?;
                Month::new(year, month).map_err(err)
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LongInput {
    origin: Vec<Origin>,
    /// Development age of each row, in months.
    development: Vec<u32>,
    /// One measure, stored as the column `"value"`.
    #[serde(default)]
    value: Option<Vec<Option<f64>>>,
    /// Several measures by name.
    #[serde(default)]
    values: Option<Pairs<Vec<Option<f64>>>>,
    /// Key columns by name: each distinct combination is a segment.
    #[serde(default)]
    keys: Option<Pairs<Vec<String>>>,
    #[serde(default)]
    origin_grain: Option<String>,
    #[serde(default)]
    development_grain: Option<String>,
    #[serde(default)]
    cumulative: Option<bool>,
}

impl Fields for LongInput {
    const FIELDS: &'static [&'static str] = &[
        "origin",
        "development",
        "value",
        "values",
        "keys",
        "originGrain",
        "developmentGrain",
        "cumulative",
    ];
}

fn grain(name: &Option<String>, arg: &str) -> Result<Grain, JsError> {
    match name {
        None => Ok(Grain::Year),
        Some(n) => Grain::from_name(n.trim().to_ascii_lowercase().as_str()).ok_or_else(|| {
            let names: Vec<&str> = Grain::NAMES.iter().map(|(n, _)| *n).collect();
            JsError::new(&format!(
                "{arg} must be one of {}, got {n:?}",
                names.join(", ")
            ))
        }),
    }
}

#[wasm_bindgen]
impl Triangle {
    /// A triangle from a long table, one row per observed cell:
    ///
    /// ```js
    /// const tri = Triangle.fromLong({
    ///   origin: [2020, 2020, 2021],        // a year, or "YYYY-MM"
    ///   development: [12, 24, 12],         // age in months
    ///   value: [100, 150, 110],            // or values: { paid: [...], premium: [...] }
    ///   // keys: { lob: ["Auto", "Auto", "Auto"] },
    ///   // originGrain: "year", developmentGrain: "year", cumulative: true,
    /// });
    /// ```
    ///
    /// `value` is stored as the column `"value"`. A `null` or `NaN` marks a
    /// measure that a row does not observe. The grains default to `"year"`
    /// and the values to cumulative.
    #[wasm_bindgen(js_name = fromLong)]
    pub fn from_long(table: JsValue) -> Result<Triangle, JsError> {
        let input: LongInput = read(table, "Triangle.fromLong")?;
        let origin: Vec<Month> = input
            .origin
            .iter()
            .map(Origin::month)
            .collect::<Result<_, _>>()?;
        let columns: Vec<(String, Vec<f64>)> = match (&input.value, &input.values) {
            (Some(v), None) => vec![("value".to_owned(), numbers(v))],
            (None, Some(Pairs(vs))) => vs.iter().map(|(n, v)| (n.clone(), numbers(v))).collect(),
            (Some(_), Some(_)) => {
                return Err(JsError::new(
                    "Triangle.fromLong: give value or values, not both",
                ));
            }
            (None, None) => {
                return Err(JsError::new(
                    "Triangle.fromLong: give value (one measure) or values (several by name)",
                ));
            }
        };
        let values: Vec<(&str, &[f64])> = columns
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let key_columns = input.keys.map(|k| k.0).unwrap_or_default();
        let key_refs: Vec<(&str, Vec<&str>)> = key_columns
            .iter()
            .map(|(n, v)| (n.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        let keys: Vec<(&str, &[&str])> = key_refs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let inner = core::Triangle::from_long(&Long {
            keys: &keys,
            origin: &origin,
            development: DevelopmentColumn::Age(&input.development),
            values: &values,
            origin_grain: grain(&input.origin_grain, "originGrain")?,
            development_grain: grain(&input.development_grain, "developmentGrain")?,
            cumulative: input.cumulative.unwrap_or(true),
        })
        .map_err(err)?;
        Ok(Self { inner })
    }

    /// The numbers of segments, columns, origins and ages.
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&n| n as u32).collect()
    }

    /// The names of the measure columns.
    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> Array {
        strings(self.inner.columns())
    }

    /// The names of the key columns.
    #[wasm_bindgen(getter, js_name = keyNames)]
    pub fn key_names(&self) -> Array {
        strings(self.inner.key_names())
    }

    /// The origin periods, as `"2020"`, `"2020Q1"` or `"2020-04"`.
    #[wasm_bindgen(getter)]
    pub fn origins(&self) -> Array {
        strings(self.inner.origins().iter().map(ToString::to_string))
    }

    /// The development ages, in months.
    #[wasm_bindgen(getter)]
    pub fn development(&self) -> Vec<u32> {
        self.inner.development().to_vec()
    }

    /// Whether the values are cumulative.
    #[wasm_bindgen(getter, js_name = isCumulative)]
    pub fn is_cumulative(&self) -> bool {
        self.inner.is_cumulative()
    }

    /// The valuation month, as `"YYYY-MM"`.
    #[wasm_bindgen(getter)]
    pub fn valuation(&self) -> String {
        self.inner.valuation().to_string()
    }

    /// The triangle with incremental values.
    #[wasm_bindgen(js_name = toIncremental)]
    pub fn to_incremental(&self) -> Triangle {
        Self {
            inner: self.inner.to_incremental(),
        }
    }

    /// The triangle with cumulative values.
    #[wasm_bindgen(js_name = toCumulative)]
    pub fn to_cumulative(&self) -> Triangle {
        Self {
            inner: self.inner.to_cumulative(),
        }
    }

    /// The triangle as a long table, one row per observed cell: the key
    /// columns, `origin` (the period's first month, `"YYYY-MM"`),
    /// `development` (age in months) and each measure as a `Float64Array`
    /// with `NaN` where a row does not observe it.
    #[wasm_bindgen(js_name = toLong)]
    pub fn to_long(&self) -> JsValue {
        let long = self.inner.to_long();
        let mut o = Obj::new();
        for (name, values) in &long.keys {
            o = o.strings(name, values);
        }
        let ages: Vec<f64> = long.development.iter().map(|&a| f64::from(a)).collect();
        o = o
            .strings("origin", long.origin.iter().map(ToString::to_string))
            .numbers("development", &ages);
        for (name, values) in &long.values {
            o = o.numbers(name, values);
        }
        o.done()
    }
}

/// The three long tables of a fit over segments: `byOrigin` (one row per
/// segment and origin), `totals` (one row per segment) and `development`
/// (one row per segment and age).
fn tables<T: ReserveFit>(fits: &SegmentFits<T>) -> JsValue {
    Obj::new()
        .set("byOrigin", table(&fits.to_long()))
        .set("totals", table(&fits.totals()))
        .set("development", table(&fits.development_table()))
        .done()
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ChainLadderOptions {
    /// A constant tail factor past the oldest age.
    #[serde(default)]
    tail: Option<f64>,
}

impl Fields for ChainLadderOptions {
    const FIELDS: &'static [&'static str] = &["tail"];
}

fn tail(factor: Option<f64>) -> core::Tail {
    match factor {
        None => core::Tail::default(),
        Some(factor) => core::Tail::Constant(core::TailConstant {
            factor,
            ..Default::default()
        }),
    }
}

/// The chain ladder on `column` of every segment of `triangle`.
///
/// `options`: `{ tail }`, a constant tail factor. Returns `{ byOrigin,
/// totals, development }`, each a table of columns: `byOrigin` has
/// `origin`, `latest`, `ultimate` and `reserve`; `development` has `age`,
/// `ldf` and `cdf`.
#[wasm_bindgen(js_name = chainLadder)]
pub fn chain_ladder(
    triangle: &Triangle,
    column: &str,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: ChainLadderOptions = read_options(options, "chainLadder options")?;
    let method = core::ChainLadder {
        tail: tail(o.tail),
        ..Default::default()
    };
    let fits = method.fit_segments(&triangle.inner, column).map_err(err)?;
    Ok(tables(&fits))
}

/// Mack's chain ladder on `column` of every segment: the chain ladder's
/// tables with each origin's and the total's standard error.
///
/// `options`: `{ tail }`, a constant tail factor.
#[wasm_bindgen]
pub fn mack(triangle: &Triangle, column: &str, options: JsValue) -> Result<JsValue, JsError> {
    let o: ChainLadderOptions = read_options(options, "mack options")?;
    let method = core::Mack {
        tail: tail(o.tail),
        ..Default::default()
    };
    let fits = method.fit_segments(&triangle.inner, column).map_err(err)?;
    Ok(tables(&fits))
}

/// One a-priori loss ratio for every origin, or one per origin, oldest
/// first.
#[derive(Deserialize)]
#[serde(untagged)]
enum AprioriInput {
    All(f64),
    InOrder(Vec<f64>),
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BornhuetterFergusonOptions {
    /// The exposure column (premium), which the a-priori ratio multiplies.
    exposure: String,
    apriori: AprioriInput,
    #[serde(default)]
    tail: Option<f64>,
}

impl Fields for BornhuetterFergusonOptions {
    const FIELDS: &'static [&'static str] = &["exposure", "apriori", "tail"];
}

/// The Bornhuetter-Ferguson method on `column` of every segment.
///
/// `options`: `{ exposure, apriori, tail }`: the name of the exposure
/// column, the a-priori loss ratio (one number, or one per origin oldest
/// first) and an optional constant tail factor. Returns the same three
/// tables as `chainLadder`, with `ulr` (the ultimate over the exposure) in
/// `byOrigin` and `totals`.
#[wasm_bindgen(js_name = bornhuetterFerguson)]
pub fn bornhuetter_ferguson(
    triangle: &Triangle,
    column: &str,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: BornhuetterFergusonOptions = read(options, "bornhuetterFerguson options")?;
    let apriori = match o.apriori {
        AprioriInput::All(a) => core::Apriori::All(a),
        AprioriInput::InOrder(a) => core::Apriori::InOrder(a),
    };
    let method = core::BornhuetterFerguson {
        apriori,
        chain_ladder: core::ChainLadder {
            tail: tail(o.tail),
            ..Default::default()
        },
    };
    let fits = method
        .fit_segments(&triangle.inner, column, &o.exposure)
        .map_err(err)?;
    Ok(tables(&fits))
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BootstrapOptions {
    #[serde(default)]
    n_sims: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

impl Fields for BootstrapOptions {
    const FIELDS: &'static [&'static str] = &["nSims", "seed"];
}

/// The ODP bootstrap of the chain ladder on `column` of every segment.
///
/// `options`: `{ nSims, seed }`, by default 10,000 simulations and seed 0.
/// Returns `{ reserves, byOrigin, totals }`: the simulated reserves as a
/// [`Predictive`] with the key columns and `origin` as dimensions (its
/// `summary()` has each origin's simulated mean and standard deviation),
/// and the chain ladder the bootstrap resamples, by origin and in total
/// (`latest`, `ultimate`, `reserve`, and the ODP `scale` in `totals`). The same seed gives the same draws as in Rust,
/// Python and R.
#[wasm_bindgen(js_name = odpBootstrap)]
pub fn odp_bootstrap(
    triangle: &Triangle,
    column: &str,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: BootstrapOptions = read_options(options, "odpBootstrap options")?;
    let default = core::OdpBootstrap::default();
    let method = core::OdpBootstrap {
        n_sims: o.n_sims.unwrap_or(default.n_sims),
        seed: o.seed.unwrap_or(default.seed),
        ..default
    };
    let fits = method.fit_segments(&triangle.inner, column).map_err(err)?;
    let (by_origin, totals) = (
        table(&fits.segments.to_long()),
        table(&fits.segments.totals()),
    );
    Ok(Obj::new()
        .set("reserves", Predictive::from(fits.reserves))
        .set("byOrigin", by_origin)
        .set("totals", totals)
        .done())
}
