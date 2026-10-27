//! `prospicio.reserving` (Reserving lane): the loss triangle, the chain
//! ladder, Mack's model, the expected-loss methods, Clark's growth curves
//! and the ODP bootstrap over `prospicio_reserving` (`docs/design/triangle.md`,
//! `docs/design/reserving-v02.md`).
//!
//! Long tables come in as array-likes (lists, numpy arrays, pandas or
//! Polars columns) and go out as dicts of lists; numpy and pandas are used
//! when the caller passes them but are not required.

use prospicio_core::{Grain, Lag, Month, Period};
use prospicio_reserving::{
    Apriori, Average, Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder,
    ChainLadderFit, ClaimsDevelopmentResult, ClarkCapeCod, ClarkFit, ClarkLdf, CurveShape,
    Development, DevelopmentColumn, DiagonalAge, ExpectedLoss, ExpectedLossFit, FitTable,
    GrowthCurve, Label, LatestDiagonals, Long, LongDiagonal, Mack, MackBootstrap,
    MackBootstrapFits, MackBootstrapSegment, MackFit, MackProcess, OdpBootstrap, OdpBootstrapFits,
    OdpBootstrapSegment, OneYearFits, OneYearMethod, OneYearPattern, PastPatternEnd, PatternKind,
    PatternWeight, ProcessDistribution, ReserveFit, SegmentDependence, SegmentFits, SegmentMap,
    SelectedPattern, SigmaInterpolation, Tail, TailBondy, TailConstant, TailCurve, Triangle,
    UnderwritingBasis, view,
};
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDelta, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple,
};

use prospicio_bayes::nuts::{ParamSummary, Sampler as BayesSampler};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesBfNormalised, BayesBfNormalisedFit, BayesOdpFits, BayesOdpSegment,
    BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment, Compartmental,
    CompartmentalCandidate, CompartmentalFit, CompartmentalFits, CompartmentalLikelihood,
    CompartmentalOneYearFits, CompartmentalPriors, CompartmentalSegment, CompartmentalStart,
    Fitted, HalfPrior, LayerSharePrior, LightRefit, MeyersFits, MeyersSegment, OneYearRefit,
    Outstanding, PaidResponse, PositivePrior, RateOrder, RealPrior, Scale as CompartmentalScale,
    ShareDependence,
};
use prospicio_reserving::{
    Backtest, BayesOdp, BayesOdpFit, Diagnostics, GlmCandidate, Meyers, MeyersFit, MeyersModel,
    NamedModel, TriangleFrame, TriangleModel, diagonal_backtest as run_backtest,
};

use prospicio_glm::Glm;
use prospicio_models::{Column, Terms};

use crate::distributions::PyPredictiveDistribution;
use crate::models::{PyElpd, PyGlm, PyTerms};
use crate::to_py;

fn err(e: prospicio_reserving::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn grain(name: &str) -> PyResult<Grain> {
    match name.to_ascii_uppercase().as_str() {
        "M" => Ok(Grain::Month),
        "Q" => Ok(Grain::Quarter),
        "S" => Ok(Grain::Semester),
        "Y" => Ok(Grain::Year),
        _ => Err(PyValueError::new_err(format!(
            "grain must be \"M\", \"Q\", \"S\" or \"Y\", got {name:?}"
        ))),
    }
}

fn average(name: &str) -> PyResult<Average> {
    match name {
        "volume" => Ok(Average::Volume),
        "simple" => Ok(Average::Simple),
        "regression" => Ok(Average::Regression),
        "geometric" => Ok(Average::Geometric),
        _ => Err(PyValueError::new_err(format!(
            "average must be \"volume\", \"simple\", \"regression\" or \"geometric\", got {name:?}"
        ))),
    }
}

fn average_name(a: Average) -> &'static str {
    match a {
        Average::Volume => "volume",
        Average::Simple => "simple",
        Average::Regression => "regression",
        Average::Geometric => "geometric",
    }
}

fn sigma_interpolation(name: &str) -> PyResult<SigmaInterpolation> {
    match name {
        "log-linear" => Ok(SigmaInterpolation::LogLinear),
        "mack" => Ok(SigmaInterpolation::Mack),
        _ => Err(PyValueError::new_err(format!(
            "sigma_interpolation must be \"log-linear\" or \"mack\", got {name:?}"
        ))),
    }
}

fn process_distribution(name: &str) -> PyResult<ProcessDistribution> {
    match name {
        "gamma" => Ok(ProcessDistribution::Gamma),
        "none" => Ok(ProcessDistribution::None),
        _ => Err(PyValueError::new_err(format!(
            "process must be \"gamma\" or \"none\", got {name:?}"
        ))),
    }
}

fn process_name(p: ProcessDistribution) -> &'static str {
    match p {
        ProcessDistribution::Gamma => "gamma",
        ProcessDistribution::None => "none",
    }
}

/// The dependence between segments named `name`, with the Spearman matrix
/// that `"rank_correlation"` needs and the others refuse.
fn segment_dependence(name: &str, spearman: Option<Vec<Vec<f64>>>) -> PyResult<SegmentDependence> {
    match (name, spearman) {
        ("independent", None) => Ok(SegmentDependence::Independent),
        ("synchronized", None) => Ok(SegmentDependence::Synchronized),
        ("rank_correlation", Some(rows)) => {
            if rows.iter().any(|r| r.len() != rows.len()) {
                return Err(PyValueError::new_err(
                    "spearman must be a square matrix, one row and column per segment",
                ));
            }
            Ok(SegmentDependence::RankCorrelation {
                spearman: rows.into_iter().flatten().collect(),
            })
        }
        ("rank_correlation", None) => Err(PyValueError::new_err(
            "dependence=\"rank_correlation\" needs a spearman matrix",
        )),
        ("independent" | "synchronized", Some(_)) => Err(PyValueError::new_err(
            "spearman is taken only with dependence=\"rank_correlation\"",
        )),
        _ => Err(PyValueError::new_err(format!(
            "dependence must be \"independent\", \"synchronized\" or \"rank_correlation\", \
             got {name:?}"
        ))),
    }
}

fn dependence_name(d: &SegmentDependence) -> &'static str {
    match d {
        SegmentDependence::Independent => "independent",
        SegmentDependence::Synchronized => "synchronized",
        SegmentDependence::RankCorrelation { .. } => "rank_correlation",
    }
}

/// The Spearman matrix of a rank correlation, as rows; `None` otherwise.
fn dependence_spearman(d: &SegmentDependence) -> Option<Vec<Vec<f64>>> {
    match d {
        SegmentDependence::RankCorrelation { spearman } => {
            let n = (spearman.len() as f64).sqrt().round() as usize;
            Some(spearman.chunks(n.max(1)).map(<[f64]>::to_vec).collect())
        }
        _ => None,
    }
}

/// `, dependence=...` for a bootstrap's repr; empty when independent.
fn dependence_repr(d: &SegmentDependence) -> String {
    match dependence_spearman(d) {
        None if *d == SegmentDependence::Independent => String::new(),
        None => format!(", dependence={:?}", dependence_name(d)),
        Some(m) => format!(", dependence={:?}, spearman={m:?}", dependence_name(d)),
    }
}

fn sigma_interpolation_name(s: SigmaInterpolation) -> &'static str {
    match s {
        SigmaInterpolation::LogLinear => "log-linear",
        SigmaInterpolation::Mack => "mack",
    }
}

fn development(
    average_: &str,
    sigma_interpolation_: &str,
    pattern: Option<SelectedPattern>,
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> PyResult<Development> {
    Ok(Development {
        average: average(average_)?,
        sigma_interpolation: sigma_interpolation(sigma_interpolation_)?,
        pattern,
        n_periods,
        drop_high,
        drop_low,
    })
}

fn pattern_kind(name: &str) -> PyResult<PatternKind> {
    match name {
        "ldf" => Ok(PatternKind::AgeToAge),
        "cdf" => Ok(PatternKind::AgeToUltimate),
        "percent_developed" => Ok(PatternKind::PercentDeveloped),
        _ => Err(PyValueError::new_err(format!(
            "ldf_kind must be \"ldf\" (age-to-age factors), \"cdf\" (factors to ultimate) \
             or \"percent_developed\", got {name:?}"
        ))),
    }
}

fn pattern_kind_name(kind: PatternKind) -> &'static str {
    match kind {
        PatternKind::AgeToAge => "ldf",
        PatternKind::AgeToUltimate => "cdf",
        PatternKind::PercentDeveloped => "percent_developed",
    }
}

pub(crate) fn py_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr().map_or_else(|_| "?".into(), |r| r.to_string())
}

/// `(age, value)` pairs of a dict (or any mapping, a pandas Series too)
/// from age in months to a number.
pub(crate) fn age_values(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<(Lag, f64)>> {
    if obj.is_instance_of::<PyString>() || !obj.hasattr("items")? {
        return Err(PyTypeError::new_err(format!(
            "{what} must be a dict from age in months to a number, got {}",
            py_repr(obj)
        )));
    }
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        let age: i64 = if key.is_instance_of::<PyBool>() {
            None
        } else {
            key.extract().ok()
        }
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: an age must be a whole number of months, got {}",
                py_repr(&key)
            ))
        })?;
        let age = Lag::try_from(age)
            .map_err(|_| PyValueError::new_err(format!("{what}: {age} is not an age in months")))?;
        let value: f64 = if value.is_instance_of::<PyBool>() {
            None
        } else {
            value.extract().ok()
        }
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: the value at age {age} must be a number, got {}",
                py_repr(&value)
            ))
        })?;
        out.push((age, value));
    }
    Ok(out)
}

/// The selected pattern of the ``ldf``, ``ldf_kind`` and ``ldf_weight``
/// arguments: ``None`` without ``ldf``. The kind is required: age-to-age
/// factors and factors to ultimate cannot be told apart.
fn selected_pattern(
    ldf: Option<&Bound<'_, PyAny>>,
    ldf_kind: Option<&str>,
    ldf_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<SelectedPattern>> {
    let ldf_weight = ldf_weight.filter(|o| !o.is_none());
    let Some(ldf) = ldf.filter(|o| !o.is_none()) else {
        if ldf_kind.is_some() {
            return Err(PyValueError::new_err("ldf_kind is given without ldf"));
        }
        if ldf_weight.is_some() {
            return Err(PyValueError::new_err("ldf_weight is given without ldf"));
        }
        return Ok(None);
    };
    let Some(kind) = ldf_kind else {
        return Err(PyValueError::new_err(
            "ldf needs ldf_kind: \"ldf\" (age-to-age factors), \"cdf\" (factors to \
             ultimate) or \"percent_developed\"; age-to-age factors and factors to \
             ultimate cannot be told apart",
        ));
    };
    let pattern = SelectedPattern::new(pattern_kind(kind)?, age_values(ldf, "ldf")?);
    let weight = match ldf_weight {
        None => PatternWeight::default(),
        Some(w) if w.hasattr("items")? => PatternWeight::ByAge(age_values(w, "ldf_weight")?),
        Some(w) if !w.is_instance_of::<PyBool>() => {
            PatternWeight::All(w.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "ldf_weight must be a number or a dict from age in months to a number, got {}",
                    py_repr(w)
                ))
            })?)
        }
        Some(w) => {
            return Err(PyTypeError::new_err(format!(
                "ldf_weight must be a number or a dict from age in months to a number, got {}",
                py_repr(w)
            )));
        }
    };
    Ok(Some(pattern.with_weight(weight)))
}

/// A dict from age in months to value.
fn age_dict<'py>(py: Python<'py>, pairs: &[(Lag, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for &(age, v) in pairs {
        d.set_item(age, v)?;
    }
    Ok(d)
}

/// A method's ``ldf`` as Python sees it: a dict from age to value, or
/// ``None`` without a pattern.
fn pattern_values<'py>(
    py: Python<'py>,
    pattern: Option<&SelectedPattern>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    pattern.map(|p| age_dict(py, &p.values)).transpose()
}

/// A method's ``ldf_kind``, ``None`` without a pattern.
fn pattern_kind_of(pattern: Option<&SelectedPattern>) -> Option<&'static str> {
    pattern.map(|p| pattern_kind_name(p.kind))
}

/// A method's ``ldf_weight``: a float, a dict from age to weight, or
/// ``None`` without a pattern.
fn pattern_weight<'py>(
    py: Python<'py>,
    pattern: Option<&SelectedPattern>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    pattern
        .map(|p| match &p.weight {
            PatternWeight::All(w) => Ok(PyFloat::new(py, *w).into_any()),
            PatternWeight::ByAge(by_age) => Ok(age_dict(py, by_age)?.into_any()),
        })
        .transpose()
}

/// `, n_periods=..., drop_high=..., drop_low=...` for the exclusions that are
/// set, then the selected pattern; empty at the defaults.
fn development_arg_repr(d: &Development) -> String {
    let mut out = exclusions_arg_repr(d.n_periods, d.drop_high, d.drop_low);
    out.push_str(&pattern_arg_repr(d.pattern.as_ref()));
    out
}

/// The exclusions that are set, as `, n_periods=..., drop_high=...,
/// drop_low=...`; empty at the defaults.
fn exclusions_arg_repr(n_periods: Option<usize>, drop_high: usize, drop_low: usize) -> String {
    let mut out = String::new();
    if let Some(n) = n_periods {
        out.push_str(&format!(", n_periods={n}"));
    }
    if drop_high != 0 {
        out.push_str(&format!(", drop_high={drop_high}"));
    }
    if drop_low != 0 {
        out.push_str(&format!(", drop_low={drop_low}"));
    }
    out
}

/// `, ldf={...}, ldf_kind=..., ldf_weight=...` for a method's repr; empty
/// without a pattern, and no weight when it is the default.
fn pattern_arg_repr(pattern: Option<&SelectedPattern>) -> String {
    let Some(p) = pattern else {
        return String::new();
    };
    let map = |pairs: &[(Lag, f64)]| {
        let items: Vec<String> = pairs.iter().map(|(a, v)| format!("{a}: {v:?}")).collect();
        format!("{{{}}}", items.join(", "))
    };
    let weight = match &p.weight {
        PatternWeight::All(w) if *w == 1.0 => String::new(),
        PatternWeight::All(w) => format!(", ldf_weight={w:?}"),
        PatternWeight::ByAge(by_age) => format!(", ldf_weight={}", map(by_age)),
    };
    format!(
        ", ldf={}, ldf_kind={:?}{weight}",
        map(&p.values),
        pattern_kind_name(p.kind)
    )
}

/// A column as a plain Python sequence: numpy arrays and pandas or Polars
/// series become lists.
fn plain<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.hasattr("tolist")? {
        obj.call_method0("tolist")
    } else {
        Ok(obj.clone())
    }
}

fn month_of(item: &Bound<'_, PyAny>, what: &str) -> PyResult<Month> {
    // Python ints and integer scalars such as `numpy.int64` (via `__index__`).
    if !item.is_instance_of::<PyBool>()
        && (item.is_instance_of::<PyInt>() || item.hasattr("__index__")?)
    {
        return Ok(Month::january(item.extract()?));
    }
    if item.hasattr("year")? && item.hasattr("month")? {
        let year: i32 = item.getattr("year")?.extract()?;
        let month: u8 = item.getattr("month")?.extract()?;
        return Month::new(year, month).map_err(to_py);
    }
    Err(PyTypeError::new_err(format!(
        "{what} must hold dates or integer years, got {}",
        item.repr()?
    )))
}

/// Months of a column of dates (numpy `datetime64`, `datetime.date`,
/// pandas `Timestamp`) or integer years.
fn months(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Month>> {
    let py = obj.py();
    if obj.hasattr("dtype")? {
        // A numpy array or a pandas/Polars series: convert datetimes in bulk.
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (obj,))?;
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind == "M" {
            let ordinals: Vec<i64> = arr
                .call_method1("astype", ("datetime64[M]",))?
                .call_method1("astype", ("int64",))?
                .call_method0("tolist")?
                .extract()?;
            return ordinals
                .into_iter()
                .map(|o| {
                    if o == i64::MIN {
                        Err(PyValueError::new_err(format!("{what} has a missing date")))
                    } else {
                        Ok(Month::january(1970).add_months(o))
                    }
                })
                .collect();
        }
        return arr
            .call_method0("tolist")?
            .try_iter()?
            .map(|item| month_of(&item?, what))
            .collect();
    }
    obj.try_iter()?.map(|item| month_of(&item?, what)).collect()
}

fn ages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Lag>> {
    let raw: Vec<i64> = plain(obj)?
        .extract()
        .map_err(|_| PyTypeError::new_err("development must hold integer ages in months"))?;
    raw.into_iter()
        .map(|a| {
            Lag::try_from(a).map_err(|_| {
                PyValueError::new_err(format!("development age {a} is not a positive month count"))
            })
        })
        .collect()
}

fn floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    plain(obj)?
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("column {name:?} must hold numbers")))
}

/// Values of the key column `name` as strings (`str()` of each value).
/// `None`, float NaN and pandas' `NA` and `NaT` are missing values, which a
/// key may not have.
fn key_values(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    let missing_error = || PyValueError::new_err(format!("key column {name:?} has missing values"));
    // A pandas series knows its own missing values, whatever its dtype.
    if obj.hasattr("isna")? && obj.call_method0("isna")?.call_method0("any")?.is_truthy()? {
        return Err(missing_error());
    }
    plain(obj)?
        .try_iter()?
        .map(|item| {
            let item = item?;
            let missing = item.is_none()
                || (item.is_instance_of::<PyFloat>() && item.extract::<f64>()?.is_nan())
                || matches!(
                    item.get_type().name()?.extract::<String>()?.as_str(),
                    "NAType" | "NaTType"
                );
            if missing {
                return Err(missing_error());
            }
            Ok(item.str()?.to_string())
        })
        .collect()
}

/// A label as Python sees it: ``"Total"`` without keys, a str with one key,
/// a tuple with several.
fn label_to_py<'py>(py: Python<'py>, n_keys: usize, label: &Label) -> PyResult<Bound<'py, PyAny>> {
    match (n_keys, label.parts()) {
        (0, _) => Ok(PyString::new(py, &label.to_string()).into_any()),
        (1, [one]) => Ok(PyString::new(py, one).into_any()),
        (_, parts) => Ok(PyTuple::new(py, parts)?.into_any()),
    }
}

/// Key values to select: one value (a string, number, date, ...) or any
/// other iterable of values (a list, tuple, set, NumPy array or pandas
/// Series); each compared as ``str()`` of it, as key columns are stored.
fn selection_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
        return Ok(vec![obj.str()?.to_string()]);
    }
    match obj.try_iter() {
        Ok(items) => items.map(|item| Ok(item?.str()?.to_string())).collect(),
        Err(_) => Ok(vec![obj.str()?.to_string()]),
    }
}

fn names(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = obj.extract::<String>() {
        Ok(vec![name])
    } else {
        obj.extract()
    }
}

/// Measure columns from a dict of name to column, or one unnamed column
/// (named ``"values"``).
fn value_columns(values: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Vec<f64>)>> {
    if let Ok(dict) = values.cast::<PyDict>() {
        dict.iter()
            .map(|(k, v)| {
                let name: String = k
                    .extract()
                    .map_err(|_| PyTypeError::new_err("value column names must be strings"))?;
                let column = floats(&v, &name)?;
                Ok((name, column))
            })
            .collect()
    } else {
        Ok(vec![("values".to_string(), floats(values, "values")?)])
    }
}

struct LongArgs {
    origin: Vec<Month>,
    development: Vec<Lag>,
    valuations: Vec<Month>,
    development_is_valuation: bool,
    values: Vec<(String, Vec<f64>)>,
    keys: Vec<(String, Vec<String>)>,
    origin_grain: Grain,
    development_grain: Grain,
    cumulative: bool,
}

impl LongArgs {
    #[allow(clippy::too_many_arguments)]
    fn new(
        origin: &Bound<'_, PyAny>,
        development: &Bound<'_, PyAny>,
        development_is_valuation: bool,
        values: Vec<(String, Vec<f64>)>,
        keys: Vec<(String, Vec<String>)>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
    ) -> PyResult<Self> {
        let (development, valuations) = if development_is_valuation {
            (Vec::new(), months(development, "development")?)
        } else {
            (ages(development)?, Vec::new())
        };
        Ok(Self {
            origin: months(origin, "origin")?,
            development,
            valuations,
            development_is_valuation,
            values,
            keys,
            origin_grain: grain(origin_grain)?,
            development_grain: grain(development_grain)?,
            cumulative,
        })
    }

    fn build(&self, py: Python<'_>) -> PyResult<PyTriangle> {
        let values: Vec<(&str, &[f64])> = self
            .values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let key_values: Vec<Vec<&str>> = self
            .keys
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let keys: Vec<(&str, &[&str])> = self
            .keys
            .iter()
            .zip(&key_values)
            .map(|((n, _), v)| (n.as_str(), v.as_slice()))
            .collect();
        let long = Long {
            keys: &keys,
            origin: &self.origin,
            development: if self.development_is_valuation {
                DevelopmentColumn::Valuation(&self.valuations)
            } else {
                DevelopmentColumn::Age(&self.development)
            },
            values: &values,
            origin_grain: self.origin_grain,
            development_grain: self.development_grain,
            cumulative: self.cumulative,
        };
        let inner = py.detach(|| Triangle::from_long(&long)).map_err(err)?;
        Ok(PyTriangle { inner })
    }
}

/// A loss triangle with four axes: index (segment), column (measure), origin
/// and development age, in chainladder-python's order.
///
/// Segments are named by key columns such as ``"lob"`` and ``"state"``:
/// ``keys`` gives their names and ``index`` one label per segment. A
/// triangle without keys has one segment, ``"Total"``.
///
/// Build one from a long table with ``from_long`` or ``from_frame``. Ages
/// are whole months from the start of the origin period, so age 12 on a
/// 2021 accident year is valued at December 2021. Cells that were not
/// observed are ``nan`` in ``values``; an observed zero stays zero.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Triangle
/// >>> tri = Triangle.from_long(
/// ...     origin=[2020, 2020, 2021],
/// ...     development=[12, 24, 12],
/// ...     values={"paid": [100.0, 150.0, 110.0]},
/// ... )
/// >>> tri.shape
/// (1, 1, 2, 2)
/// >>> tri.origins, tri.development, tri.valuation
/// (['2020', '2021'], [12, 24], datetime.date(2021, 12, 31))
/// >>> tri.values[0][0]
/// [[100.0, 150.0], [110.0, nan]]
#[pyclass(name = "Triangle", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTriangle {
    pub(crate) inner: Triangle,
}

fn wrap(inner: Triangle) -> PyTriangle {
    PyTriangle { inner }
}

/// The last day of `month` as a ``datetime.date``.
fn month_end<'py>(py: Python<'py>, month: Month) -> PyResult<Bound<'py, PyDate>> {
    // The first of the next month, minus one day.
    let next = month.add_months(1);
    let first_of_next = PyDate::new(py, next.year(), next.month(), 1)?;
    let one_day = PyDelta::new(py, 1, 0, 0, false)?;
    first_of_next
        .call_method1("__sub__", (one_day,))?
        .cast_into::<PyDate>()
        .map_err(Into::into)
}

/// pandas when it is installed, so tables come out as DataFrames; the
/// tables are dicts of lists otherwise.
fn pandas(py: Python<'_>) -> PyResult<Option<Bound<'_, PyModule>>> {
    match py.import("pandas") {
        Ok(module) => Ok(Some(module)),
        Err(e) if e.is_instance_of::<PyImportError>(py) => Ok(None),
        Err(e) => Err(e),
    }
}

#[pymethods]
impl PyTriangle {
    /// Builds a triangle from the columns of a long table, one row per
    /// (keys, origin, development).
    ///
    /// Origins span every period from the earliest to the latest row and
    /// ages every development period from the youngest to the oldest. Rows
    /// with the same (keys, origin, age) are summed; ``nan`` values are
    /// missing. Incremental input treats a missing row as a period without
    /// movement, as chainladder-python does.
    ///
    /// Parameters
    /// ----------
    /// origin : array-like
    ///     Any date in each row's origin period (numpy ``datetime64``,
    ///     ``datetime.date``, pandas ``Timestamp``), or integer years.
    /// development : array-like
    ///     Development age of each row in months (12, 24, ...), or its
    ///     valuation date when ``development_is_valuation`` is true.
    /// values : dict of str to array-like, or array-like
    ///     Measure columns by name. A single array-like is one column named
    ///     ``"values"``.
    /// keys : dict of str to array-like, optional
    ///     Key columns by name, such as ``{"lob": [...], "state": [...]}``,
    ///     in key order. Values are stored as strings (``str()`` of each);
    ///     ``None`` and ``nan`` are not allowed. Each distinct combination is
    ///     a segment. By default every row is in one segment, ``"Total"``.
    /// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
    ///     Length of an origin period.
    /// development_grain : {"Y", "S", "Q", "M"}, default "Y"
    ///     Spacing of development ages; must divide the origin grain.
    /// cumulative : bool, default True
    ///     Whether the values are cumulative (otherwise incremental).
    /// development_is_valuation : bool, default False
    ///     Whether ``development`` holds valuation dates instead of ages.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If columns differ in length, an age is not on the development
    ///     grid, a value is infinite, the grains are incompatible, a key name
    ///     is repeated or also a value column, or a key has missing values.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     origin=[2020, 2020, 2021, 2020],
    /// ...     development=[12, 24, 12, 12],
    /// ...     values={"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"], "state": ["CA", "CA", "CA", "NY"]},
    /// ... )
    /// >>> tri.keys, tri.index
    /// (['lob', 'state'], [('Auto', 'CA'), ('Home', 'NY')])
    ///
    /// Valuation dates instead of ages:
    ///
    /// >>> import datetime
    /// >>> from prospicio.reserving import Triangle
    /// >>> d = datetime.date
    /// >>> tri = Triangle.from_long(
    /// ...     origin=[d(2021, 2, 1), d(2021, 2, 1), d(2021, 5, 1)],
    /// ...     development=[d(2021, 3, 31), d(2021, 6, 30), d(2021, 6, 30)],
    /// ...     values=[10.0, 25.0, 7.0],
    /// ...     origin_grain="Q",
    /// ...     development_grain="Q",
    /// ...     development_is_valuation=True,
    /// ... )
    /// >>> tri.origins, tri.development
    /// (['2021Q1', '2021Q2'], [3, 6])
    #[staticmethod]
    #[pyo3(signature = (origin, development, values, keys = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false))]
    #[allow(clippy::too_many_arguments)]
    fn from_long(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        development: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
    ) -> PyResult<Self> {
        let keys = match keys {
            None => Vec::new(),
            Some(keys) => {
                let dict = keys.cast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("keys must be a dict of key name to column")
                })?;
                dict.iter()
                    .map(|(k, v)| {
                        let name: String = k
                            .extract()
                            .map_err(|_| PyTypeError::new_err("key names must be strings"))?;
                        let column = key_values(&v, &name)?;
                        Ok((name, column))
                    })
                    .collect::<PyResult<_>>()?
            }
        };
        LongArgs::new(
            origin,
            development,
            development_is_valuation,
            value_columns(values)?,
            keys,
            origin_grain,
            development_grain,
            cumulative,
        )?
        .build(py)
    }

    /// Builds a triangle from a data frame in long format.
    ///
    /// Columns are looked up with ``data[name]``, so a pandas or Polars
    /// DataFrame works, as does a dict of columns.
    ///
    /// Parameters
    /// ----------
    /// data : DataFrame or dict
    /// origin : str
    ///     Name of the origin column (dates or integer years).
    /// development : str
    ///     Name of the development column (ages in months, or valuation
    ///     dates when ``development_is_valuation`` is true).
    /// columns : str or list of str
    ///     Names of the measure columns.
    /// keys : str or list of str, optional
    ///     Names of the key columns, such as ``["lob", "state"]``. By
    ///     default every row is in one segment, ``"Total"``.
    /// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// development_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// cumulative : bool, default True
    /// development_is_valuation : bool, default False
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As for ``from_long``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> df = {
    /// ...     "lob": ["Auto", "Auto", "Auto", "Home"],
    /// ...     "year": [2020, 2020, 2021, 2020],
    /// ...     "age": [12, 24, 12, 12],
    /// ...     "paid": [100.0, 150.0, 110.0, 50.0],
    /// ... }
    /// >>> tri = Triangle.from_frame(df, "year", "age", "paid", keys="lob")
    /// >>> tri.keys, tri.index, tri.shape
    /// (['lob'], ['Auto', 'Home'], (2, 1, 2, 2))
    #[staticmethod]
    #[pyo3(signature = (data, origin, development, columns, keys = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false))]
    #[allow(clippy::too_many_arguments)]
    fn from_frame(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        origin: &str,
        development: &str,
        columns: &Bound<'_, PyAny>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
    ) -> PyResult<Self> {
        let values = names(columns)?
            .into_iter()
            .map(|name| {
                let column = floats(&data.get_item(&name)?, &name)?;
                Ok((name, column))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let keys = match keys {
            None => Vec::new(),
            Some(keys) => names(keys)?
                .into_iter()
                .map(|name| {
                    let column = key_values(&data.get_item(&name)?, &name)?;
                    Ok((name, column))
                })
                .collect::<PyResult<_>>()?,
        };
        LongArgs::new(
            &data.get_item(origin)?,
            &data.get_item(development)?,
            development_is_valuation,
            values,
            keys,
            origin_grain,
            development_grain,
            cumulative,
        )?
        .build(py)
    }

    /// Axis lengths: ``(index, column, origin, development)``.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let [i, c, o, d] = self.inner.shape();
        (i, c, o, d)
    }

    /// Names of the key columns, in key order; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Segment labels: a str each with one key, a tuple of key values with
    /// several, and ``["Total"]`` without keys.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let n_keys = self.inner.key_names().len();
        self.inner
            .index()
            .iter()
            .map(|l| label_to_py(py, n_keys, l))
            .collect()
    }

    /// Measure column names.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    /// Origin periods, oldest first: ``"2021"``, ``"2021H1"``,
    /// ``"2021Q3"`` or ``"2021-07"`` by grain.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins().iter().map(|p| p.to_string()).collect()
    }

    /// Origin grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn origin_grain(&self) -> String {
        self.inner.origin_grain().to_string()
    }

    /// Development ages in months, youngest first.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.development().to_vec()
    }

    /// Development grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn development_grain(&self) -> String {
        self.inner.development_grain().to_string()
    }

    /// Valuation date of the latest diagonal: the last day of its month,
    /// as a ``datetime.date``.
    #[getter]
    fn valuation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDate>> {
        month_end(py, self.inner.valuation())
    }

    /// Whether the values are cumulative (otherwise incremental).
    #[getter]
    fn is_cumulative(&self) -> bool {
        self.inner.is_cumulative()
    }

    /// Values as nested lists indexed ``[index][column][origin][development]``,
    /// ``nan`` where unobserved. ``numpy.asarray`` gives the 4-D array.
    #[getter]
    fn values(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let [ni, nc, no, nd] = self.inner.shape();
        (0..ni)
            .map(|i| {
                (0..nc)
                    .map(|c| {
                        (0..no)
                            .map(|o| {
                                (0..nd)
                                    .map(|d| self.inner.get(i, c, o, d).unwrap_or(f64::NAN))
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// The triangle as a long table: a dict of equal-length lists with one
    /// entry per key column (by name), ``"origin"`` (start of the origin
    /// period, a ``datetime.date``), ``"development"`` (age in months) and
    /// one per measure column, with a row per (segment, origin, age) that
    /// has an observed measure. It feeds back into ``from_frame`` (with
    /// ``keys=tri.keys``) or ``pandas.DataFrame``.
    ///
    /// Returns
    /// -------
    /// dict of str to list
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key or measure column is named ``origin`` or
    ///     ``development``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020], [12, 12], {"paid": [1.0, 2.0]}, keys={"lob": ["Auto", "Home"]}
    /// ... )
    /// >>> long = tri.to_long()
    /// >>> list(long), long["lob"]
    /// (['lob', 'origin', 'development', 'paid'], ['Auto', 'Home'])
    fn to_long<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(c) = self
            .inner
            .key_names()
            .iter()
            .chain(self.inner.columns())
            .find(|c| ["origin", "development"].contains(&c.as_str()))
        {
            return Err(PyValueError::new_err(format!(
                "column {c:?} clashes with the {c:?} column of the long table"
            )));
        }
        let long = self.inner.to_long();
        let date = py.import("datetime")?.getattr("date")?;
        let out = PyDict::new(py);
        let origin = long
            .origin
            .iter()
            .map(|m| date.call1((m.year(), m.month(), 1)))
            .collect::<PyResult<Vec<_>>>()?;
        for (name, values) in long.keys {
            out.set_item(name, values)?;
        }
        out.set_item("origin", origin)?;
        out.set_item("development", long.development)?;
        for (name, values) in long.values {
            out.set_item(name, values)?;
        }
        Ok(out)
    }

    /// The long table of ``to_long`` as a pandas DataFrame. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let long = self.to_long(py)?;
        py.import("pandas")?.call_method1("DataFrame", (long,))
    }

    /// Incremental values: each observed value minus the previous observed
    /// value in its row.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn to_incremental(&self) -> Self {
        wrap(self.inner.to_incremental())
    }

    /// Cumulative values: running sums of the observed increments.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn to_cumulative(&self) -> Self {
        wrap(self.inner.to_cumulative())
    }

    /// The latest observed value of each origin, as nested lists indexed
    /// ``[index][column][origin]``, ``nan`` for an origin with no value.
    ///
    /// Returns
    /// -------
    /// list of list of list of float
    fn latest_diagonal(&self) -> Vec<Vec<Vec<f64>>> {
        let diagonal = self.inner.latest_diagonal();
        let [ni, nc, no] = diagonal.shape();
        (0..ni)
            .map(|i| {
                (0..nc)
                    .map(|c| {
                        (0..no)
                            .map(|o| diagonal.get(i, c, o).map_or(f64::NAN, |(_, v)| v))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Age-to-age link ratios of the cumulative values. Development
    /// position ``d`` holds the ratio from age ``d`` to age ``d + 1``,
    /// observed where both ages are observed and the earlier value is not
    /// zero.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn link_ratios(&self) -> Self {
        wrap(self.inner.link_ratios())
    }

    /// The segments whose key values match, and the measure columns named.
    ///
    /// Each keyword names a key and gives one value or a list of values to
    /// keep; segments must match every keyword, and keep their order.
    /// Values are compared as strings, as keys are stored (``str()`` of a
    /// value). A key named ``columns`` cannot be selected this way.
    ///
    /// Parameters
    /// ----------
    /// columns : str or list of str, optional
    ///     Measure columns to keep, in this order. By default every column.
    /// **keys : value or list of values
    ///     For example ``lob="Auto"`` or ``state=["CA", "NY"]``; any
    ///     iterable that is not a string (a tuple, set, NumPy array or
    ///     pandas Series) is a list of values.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key, value or column is unknown or given twice, a list of
    ///     values is empty, or no segment matches.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020],
    /// ...     [12, 12, 12],
    /// ...     {"paid": [1.0, 2.0, 3.0], "incurred": [2.0, 3.0, 4.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
    /// ... )
    /// >>> tri.select(state="NY").index
    /// [('Auto', 'NY'), ('Home', 'NY')]
    /// >>> tri.select(lob="Auto", state=["CA", "NY"], columns="paid").shape
    /// (2, 1, 1, 1)
    #[pyo3(signature = (columns = None, **keys))]
    fn select(
        &self,
        columns: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let conditions: Vec<(String, Vec<String>)> = match keys {
            None => Vec::new(),
            Some(keys) => keys
                .iter()
                .map(|(k, v)| Ok((k.extract::<String>()?, selection_values(&v)?)))
                .collect::<PyResult<_>>()?,
        };
        let values: Vec<Vec<&str>> = conditions
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let conditions: Vec<(&str, &[&str])> = conditions
            .iter()
            .zip(&values)
            .map(|((k, _), v)| (k.as_str(), v.as_slice()))
            .collect();
        let mut out = self.inner.select(&conditions).map_err(err)?;
        if let Some(columns) = columns {
            let columns = names(columns)?;
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            out = out.select_columns(&columns).map_err(err)?;
        }
        Ok(wrap(out))
    }

    /// Sums the segments that share the values of ``keys``, dropping the
    /// other keys.
    ///
    /// Cumulative values are summed cell by cell, and a cell is observed if
    /// any segment in the group observes it. An incremental triangle is
    /// summed as cumulative values and returned incremental.
    ///
    /// Parameters
    /// ----------
    /// keys : str or list of str
    ///     The keys to keep, in the order the result has them. ``[]`` sums
    ///     every segment into one, labelled ``"Total"``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key is unknown or named twice.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020],
    /// ...     [12, 12, 12],
    /// ...     {"paid": [1.0, 2.0, 3.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
    /// ... )
    /// >>> by_lob = tri.group_by("lob")
    /// >>> by_lob.index, by_lob.to_long()["paid"]
    /// (['Auto', 'Home'], [3.0, 3.0])
    /// >>> tri.group_by([]).to_long()["paid"]
    /// [6.0]
    fn group_by(&self, keys: &Bound<'_, PyAny>) -> PyResult<Self> {
        let keys = names(keys)?;
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.inner.group_by(&keys).map(wrap).map_err(err)
    }

    /// The triangle at a coarser origin and/or development grain.
    ///
    /// Parameters
    /// ----------
    /// origin_grain : {"Y", "S", "Q", "M"}
    /// development_grain : {"Y", "S", "Q", "M"}, optional
    ///     By default ``origin_grain``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a grain is finer than the current one, or the development
    ///     grain does not divide the origin grain.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> q = Triangle.from_long(
    /// ...     [2020, 2020], [3, 6], [1.0, 2.0], origin_grain="Q", development_grain="Q"
    /// ... )
    /// >>> y = q.grain("Y")
    /// >>> y.origins, y.development_grain
    /// (['2020'], 'Y')
    #[pyo3(signature = (origin_grain, development_grain = None))]
    fn grain(&self, origin_grain: &str, development_grain: Option<&str>) -> PyResult<Self> {
        let origin = grain(origin_grain)?;
        let dev = match development_grain {
            Some(g) => grain(g)?,
            None => origin,
        };
        self.inner.grain(origin, dev).map(wrap).map_err(err)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyTriangle>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    /// One segment and measure as an origin × development table.
    ///
    /// Parameters
    /// ----------
    /// column : str, optional
    ///     The measure; may be left out when the triangle has one column.
    /// **keys : value
    ///     One value per key, such as ``lob="Auto"``, compared as ``str()``
    ///     of it. Keys not named may take any value, but the choice must
    ///     leave one segment; a triangle with one segment needs none. A key
    ///     named ``column`` cannot be chosen this way (``select`` it first).
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     With pandas installed, a DataFrame with the origin labels as its
    ///     index (named ``origin``), the ages in months as its columns
    ///     (named ``development``) and ``nan`` where a cell is not observed.
    ///     Without pandas, a dict of lists as the other tables of this
    ///     module: ``"origin"``, then one list per age keyed by the age.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key, value or column is unknown, the keys match several
    ///     segments, or the column is left out and there are several.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2021, 2020],
    /// ...     [12, 24, 12, 12],
    /// ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
    /// ... )
    /// >>> v = tri.view(lob="Auto")
    /// >>> list(v.index), list(v.columns), float(v.loc["2021", 12])
    /// (['2020', '2021'], [12, 24], 110.0)
    #[pyo3(signature = (column = None, **keys))]
    fn view<'py>(
        &self,
        py: Python<'py>,
        column: Option<&str>,
        keys: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let view = self.inner.view(&keys, column).map_err(err)?;
        let origins: Vec<String> = view.origins.iter().map(ToString::to_string).collect();
        let (no, nd) = (view.origins.len(), view.development.len());
        let cell = |o: usize, d: usize| view.get(o, d).unwrap_or(f64::NAN);
        match pandas(py)? {
            Some(pd) => {
                let rows: Vec<Vec<f64>> = (0..no)
                    .map(|o| (0..nd).map(|d| cell(o, d)).collect())
                    .collect();
                let index_args = PyDict::new(py);
                index_args.set_item("name", "origin")?;
                let index = pd.call_method("Index", (origins,), Some(&index_args))?;
                let columns_args = PyDict::new(py);
                columns_args.set_item("name", "development")?;
                let columns =
                    pd.call_method("Index", (view.development.clone(),), Some(&columns_args))?;
                let args = PyDict::new(py);
                args.set_item("index", index)?;
                args.set_item("columns", columns)?;
                pd.call_method("DataFrame", (rows,), Some(&args))
            }
            None => {
                let out = PyDict::new(py);
                out.set_item("origin", origins)?;
                for (d, age) in view.development.iter().enumerate() {
                    let values: Vec<f64> = (0..no).map(|o| cell(o, d)).collect();
                    out.set_item(age, values)?;
                }
                Ok(out.into_any())
            }
        }
    }

    /// One row per segment and measure: the key values, ``column``,
    /// ``n_origins`` (origins with an observed value), ``first_origin`` and
    /// ``last_origin`` of those, ``valuation`` (the last day of the latest
    /// valuation with an observed value), ``latest`` (the sum over origins of
    /// the latest cumulative value, so for an incremental triangle the sum
    /// of every increment) and ``cumulative``. An origin or valuation is
    /// missing (``None``, which pandas may show as ``NaN``) when the segment
    /// has no observed value of the measure.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     A DataFrame with pandas installed, a dict of lists otherwise.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key has the name of one of the summary's columns.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2021, 2020],
    /// ...     [12, 24, 12, 12],
    /// ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
    /// ... )
    /// >>> s = tri.summary()
    /// >>> s["lob"].tolist(), s["n_origins"].tolist(), s["latest"].tolist()
    /// (['Auto', 'Home'], [2, 1], [260.0, 50.0])
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        const COLUMNS: [&str; 7] = [
            "column",
            "n_origins",
            "first_origin",
            "last_origin",
            "valuation",
            "latest",
            "cumulative",
        ];
        if let Some(k) = self
            .inner
            .key_names()
            .iter()
            .find(|k| COLUMNS.contains(&k.as_str()))
        {
            return Err(PyValueError::new_err(format!(
                "key {k:?} clashes with the {k:?} column of the summary"
            )));
        }
        let summary = self.inner.summary();
        let rows = &summary.rows;
        let out = PyDict::new(py);
        for (k, name) in summary.key_names.iter().enumerate() {
            let values: Vec<&str> = rows.iter().map(|r| r.label.parts()[k].as_str()).collect();
            out.set_item(name, values)?;
        }
        let label = |p: Option<prospicio_core::Period>| p.map(|p| p.to_string());
        out.set_item(
            "column",
            rows.iter().map(|r| r.column.as_str()).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "n_origins",
            rows.iter().map(|r| r.n_origins).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "first_origin",
            rows.iter()
                .map(|r| label(r.first_origin))
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "last_origin",
            rows.iter()
                .map(|r| label(r.last_origin))
                .collect::<Vec<_>>(),
        )?;
        let valuation = rows
            .iter()
            .map(|r| r.valuation.map(|m| month_end(py, m)).transpose())
            .collect::<PyResult<Vec<_>>>()?;
        out.set_item("valuation", valuation)?;
        out.set_item("latest", rows.iter().map(|r| r.latest).collect::<Vec<_>>())?;
        out.set_item("cumulative", vec![summary.cumulative; rows.len()])?;
        match pandas(py)? {
            Some(pd) => pd.call_method1("DataFrame", (out,)),
            None => Ok(out.into_any()),
        }
    }

    /// The printout as text: the origin × development grid for a triangle
    /// with one segment and one measure (as ``view``), otherwise the
    /// ``summary`` table. Numbers are rounded for reading; ``view`` and
    /// ``summary`` give exact values.
    ///
    /// Parameters
    /// ----------
    /// max_rows : int, default 20
    ///     Rows shown before the middle ones are left out; 0 for no limit.
    /// max_cols : int, default 12
    ///     Development ages shown before the middle ones are left out; 0
    ///     for no limit.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [1000.0, 1500.0, 1100.0]})
    /// >>> print(tri.to_string())
    /// Triangle: paid (cumulative, valuation 2021-12)
    ///          12     24
    /// 2020  1,000  1,500
    /// 2021  1,100
    #[pyo3(signature = (max_rows = view::MAX_ROWS, max_cols = view::MAX_COLS))]
    fn to_string(&self, max_rows: usize, max_cols: usize) -> String {
        self.inner.to_text(max_rows, max_cols)
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn _repr_html_(&self) -> String {
        self.inner.to_html(view::MAX_ROWS, view::MAX_COLS)
    }
}

/// A given tail factor, as chainladder-python's ``TailConstant``.
///
/// The factor applies from the attachment age to ultimate. Past the
/// attachment it is spread over the following periods as
/// ``1 + x * decay**k``, the last factor making up the difference; this
/// shapes the factors past the attachment, not the factor to ultimate. An
/// attachment before the oldest age replaces the estimated factors from
/// there.
///
/// Parameters
/// ----------
/// factor : float, default 1.0
///     Factor from the attachment age to ultimate; finite and positive.
/// decay : float, default 0.5
///     Share of each period's development kept in the next, from 0 to 1.
/// attachment_age : int, optional
///     Age in months the factor attaches at (the first age at or after
///     it); the oldest age by default. An age at or before the youngest
///     replaces every estimated factor (chainladder-python ignores such an
///     attachment).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailConstant, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder(tail=TailConstant(1.05)).fit(tri, "paid")
/// >>> fit.tail, round(fit.ultimate[1], 6)
/// (1.05, 315.0)
#[pyclass(name = "TailConstant", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailConstant {
    inner: TailConstant,
}

#[pymethods]
impl PyTailConstant {
    #[new]
    #[pyo3(signature = (factor = 1.0, decay = 0.5, attachment_age = None))]
    fn new(factor: f64, decay: f64, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailConstant {
                factor,
                decay,
                attachment_age,
            },
        }
    }

    /// Factor from the attachment age to ultimate.
    #[getter]
    fn factor(&self) -> f64 {
        self.inner.factor
    }

    /// Share of each period's development kept in the next.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// Age in months the factor attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Constant(self.inner))
    }
}

/// A curve fitted to the estimated factors and extrapolated, as
/// chainladder-python's ``TailCurve``.
///
/// Factors above 1.00001 in the fit period are regressed by least squares:
/// ``ln(f - 1)`` on the 1-based development index ``k`` (exponential) or on
/// ``ln(k)`` (inverse power). The fitted curve replaces the factors from the
/// attachment age on and runs ``extrap_periods`` periods past the oldest
/// age.
///
/// Parameters
/// ----------
/// curve : {"exponential", "inverse_power"}, default "exponential"
/// fit_period : tuple of (int or None, int or None), default (None, None)
///     Ages in months whose factors enter the fit: from the last age at or
///     before the first (inclusive) to the last age at or before the second
///     (exclusive), as chainladder-python reads them; ``None`` is
///     open-ended.
/// extrap_periods : int, default 100
///     Number of periods past the oldest age the curve is extrapolated.
/// attachment_age : int, optional
///     Age in months the curve attaches at (the first age at or after it);
///     the oldest age by default.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailCurve, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = ChainLadder(tail=TailCurve()).fit(tri, "values")
/// >>> 1.0 < fit.tail < 1.05
/// True
#[pyclass(name = "TailCurve", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailCurve {
    inner: TailCurve,
}

#[pymethods]
impl PyTailCurve {
    #[new]
    #[pyo3(signature = (curve = "exponential", fit_period = (None, None), extrap_periods = 100, attachment_age = None))]
    fn new(
        curve: &str,
        fit_period: (Option<Lag>, Option<Lag>),
        extrap_periods: usize,
        attachment_age: Option<Lag>,
    ) -> PyResult<Self> {
        let curve = match curve {
            "exponential" => CurveShape::Exponential,
            "inverse_power" => CurveShape::InversePower,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "curve must be \"exponential\" or \"inverse_power\", got {curve:?}"
                )));
            }
        };
        Ok(Self {
            inner: TailCurve {
                curve,
                fit_period,
                extrap_periods,
                attachment_age,
            },
        })
    }

    /// The curve fitted to ``f - 1``.
    #[getter]
    fn curve(&self) -> &'static str {
        curve_name(self.inner.curve)
    }

    /// Ages whose factors enter the fit, from (inclusive) and to
    /// (exclusive).
    #[getter]
    fn fit_period(&self) -> (Option<Lag>, Option<Lag>) {
        self.inner.fit_period
    }

    /// Number of periods past the oldest age the curve is extrapolated.
    #[getter]
    fn extrap_periods(&self) -> usize {
        self.inner.extrap_periods
    }

    /// Age in months the curve attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Curve(self.inner))
    }
}

/// The Bondy tail, as chainladder-python's ``TailBondy``.
///
/// Each log factor from ``earliest_age`` on is taken as ``b`` times the one
/// before it, ``b`` fitted by least squares. The fitted factors are
/// ``f0 ** (b ** j)`` from the factor ``f0`` at ``earliest_age``, and those
/// past the next one multiply to the last fitted factor raised to
/// ``b / (1 - b)``. With the default ``earliest_age`` (the age of the last
/// factor) ``b`` is 1/2 and the tail repeats the last factor.
///
/// Parameters
/// ----------
/// earliest_age : int, optional
///     First age in months whose factor enters the fit (the last age at or
///     before it, as chainladder-python reads it); the age of the last
///     factor by default.
/// attachment_age : int, optional
///     The factor from this age (the last age at or before it) to the next
///     is kept and the fitted ones replace those after it; the age of the
///     last factor by default. Not before ``earliest_age``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailBondy, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 110.0, 170.0, 120.0],
/// ... )
/// >>> round(ChainLadder(tail=TailBondy()).fit(tri, "values").tail, 12)
/// 1.1
#[pyclass(name = "TailBondy", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailBondy {
    inner: TailBondy,
}

#[pymethods]
impl PyTailBondy {
    #[new]
    #[pyo3(signature = (earliest_age = None, attachment_age = None))]
    fn new(earliest_age: Option<Lag>, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailBondy {
                earliest_age,
                attachment_age,
            },
        }
    }

    /// First age whose factor enters the fit; ``None`` is the age of the
    /// last factor.
    #[getter]
    fn earliest_age(&self) -> Option<Lag> {
        self.inner.earliest_age
    }

    /// Age after which the fitted factors replace the estimated ones;
    /// ``None`` is the age of the last factor.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Bondy(self.inner))
    }
}

/// R ChainLadder's ``tail = TRUE`` rule (its ``tailfactor`` function).
///
/// When the third- and second-last factors multiply to more than 1.0001,
/// ``ln(f - 1)`` is regressed on the development index over the factors
/// above 1 and the next 100 extrapolated factors are multiplied; otherwise
/// the tail is 1. A tail above 2 is reset to 1, as R does.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, TailLogLinear, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack(tail=TailLogLinear()).fit(tri, "values")
/// >>> fit.tail > 1.0 and fit.standard_error[0] > 0.0
/// True
#[pyclass(name = "TailLogLinear", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailLogLinear;

#[pymethods]
impl PyTailLogLinear {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::LogLinear)
    }
}

fn curve_name(c: CurveShape) -> &'static str {
    match c {
        CurveShape::Exponential => "exponential",
        CurveShape::InversePower => "inverse_power",
    }
}

/// A constant tail with the default decay and attachment, shown as its
/// factor.
fn plain_factor(tail: &Tail) -> Option<f64> {
    match tail {
        Tail::Constant(t) if t.decay == 0.5 && t.attachment_age.is_none() => Some(t.factor),
        _ => None,
    }
}

fn tail_repr(tail: &Tail) -> String {
    let age = |a: Option<Lag>| a.map_or("None".to_string(), |a| a.to_string());
    match tail {
        Tail::Constant(t) => format!(
            "TailConstant(factor={:?}, decay={:?}, attachment_age={})",
            t.factor,
            t.decay,
            age(t.attachment_age)
        ),
        Tail::Curve(t) => format!(
            "TailCurve(curve={:?}, fit_period=({}, {}), extrap_periods={}, attachment_age={})",
            curve_name(t.curve),
            age(t.fit_period.0),
            age(t.fit_period.1),
            t.extrap_periods,
            age(t.attachment_age)
        ),
        Tail::Bondy(t) => format!(
            "TailBondy(earliest_age={}, attachment_age={})",
            age(t.earliest_age),
            age(t.attachment_age)
        ),
        Tail::LogLinear => "TailLogLinear()".to_string(),
    }
}

/// The ``tail`` argument of ``ChainLadder``, ``Mack`` and the
/// expected-loss methods: ``None`` (no tail), a number (a constant factor)
/// or a tail estimator.
pub(crate) fn tail_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Tail> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Tail::default());
    };
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailConstant>>() {
        return Ok(Tail::Constant(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailCurve>>() {
        return Ok(Tail::Curve(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailBondy>>() {
        return Ok(Tail::Bondy(t.inner));
    }
    if obj.extract::<PyRef<'_, PyTailLogLinear>>().is_ok() {
        return Ok(Tail::LogLinear);
    }
    if !obj.is_instance_of::<PyBool>() {
        if let Ok(factor) = obj.extract::<f64>() {
            return Ok(factor.into());
        }
    }
    Err(PyTypeError::new_err(format!(
        "tail must be a number, TailConstant, TailCurve, TailBondy or TailLogLinear, got {}",
        obj.repr()?
    )))
}

/// The ``tail`` of a method as Python sees it: a plain
/// constant as its factor, any other tail as its estimator.
fn tail_object<'py>(py: Python<'py>, tail: &Tail) -> PyResult<Bound<'py, PyAny>> {
    if let Some(factor) = plain_factor(tail) {
        return Ok(PyFloat::new(py, factor).into_any());
    }
    Ok(match *tail {
        Tail::Constant(inner) => Bound::new(py, PyTailConstant { inner })?.into_any(),
        Tail::Curve(inner) => Bound::new(py, PyTailCurve { inner })?.into_any(),
        Tail::Bondy(inner) => Bound::new(py, PyTailBondy { inner })?.into_any(),
        Tail::LogLinear => Bound::new(py, PyTailLogLinear)?.into_any(),
    })
}

/// ``repr`` of a ``tail`` argument: a plain constant as its factor.
fn tail_arg_repr(tail: &Tail) -> String {
    plain_factor(tail).map_or_else(|| tail_repr(tail), |f| format!("{f:?}"))
}

// Latest diagonals (`reserving-v02.md`, decisions 15 to 19).

/// The name of a grain, as ``origin_grain`` gives it.
fn grain_name(g: Grain) -> &'static str {
    match g {
        Grain::Month => "M",
        Grain::Quarter => "Q",
        Grain::Semester => "S",
        Grain::Year => "Y",
    }
}

/// How a method converts its selected pattern at fit time: ``ldf_origin``,
/// the origin grain it was measured on, and ``ldf_underwriting``, the
/// underwriting basis it is converted to.
#[derive(Debug, Clone, Default)]
struct LdfBasis {
    origin: Option<Grain>,
    underwriting: Option<UnderwritingBasis>,
}

/// The ``ldf_origin`` and ``ldf_underwriting`` arguments: each ``None``,
/// or, with ``ldf``, the origin grain the selected pattern was measured on
/// and the underwriting basis it is converted to.
fn ldf_basis_arg(
    ldf_origin: Option<&str>,
    ldf_underwriting: Option<&PyUnderwritingBasis>,
    ldf: Option<&Bound<'_, PyAny>>,
) -> PyResult<LdfBasis> {
    let no_ldf = ldf.is_none_or(|l| l.is_none());
    if no_ldf && ldf_underwriting.is_some() {
        return Err(PyValueError::new_err(
            "ldf_underwriting is given without ldf",
        ));
    }
    let underwriting = ldf_underwriting.map(|b| b.inner.clone());
    let Some(name) = ldf_origin else {
        return Ok(LdfBasis {
            origin: None,
            underwriting,
        });
    };
    if no_ldf {
        return Err(PyValueError::new_err("ldf_origin is given without ldf"));
    }
    let origin = match name {
        "M" => Grain::Month,
        "Q" => Grain::Quarter,
        "S" => Grain::Semester,
        "Y" => Grain::Year,
        _ => {
            return Err(PyValueError::new_err(format!(
                "ldf_origin must be \"M\", \"Q\", \"S\" or \"Y\", got {name:?}"
            )));
        }
    };
    Ok(LdfBasis {
        origin: Some(origin),
        underwriting,
    })
}

/// A selected pattern converted as `basis` says, for origins of `origin`:
/// to an underwriting basis from accident periods of ``ldf_origin`` (the
/// target's grain without it, `SelectedPattern::to_underwriting_basis`),
/// its writing no longer than a period of `origin`
/// (`UnderwritingBasis::check_within`),
/// or averaged from origin periods of ``ldf_origin`` over the periods of
/// `origin` (`SelectedPattern::averaged_over_origins`); unchanged with
/// neither.
fn averaged(
    pattern: &Option<SelectedPattern>,
    basis: &LdfBasis,
    origin: Grain,
) -> PyResult<Option<SelectedPattern>> {
    match (pattern, basis) {
        (
            Some(p),
            LdfBasis {
                underwriting: Some(u),
                origin: from,
            },
        ) => {
            u.check_within(origin).map_err(err)?;
            Ok(Some(
                p.to_underwriting_basis(from.unwrap_or(origin), u)
                    .map_err(err)?,
            ))
        }
        (
            Some(p),
            LdfBasis {
                origin: Some(g), ..
            },
        ) => Ok(Some(p.averaged_over_origins(*g, origin).map_err(err)?)),
        _ => Ok(pattern.clone()),
    }
}

/// A chain ladder with its pattern converted as `basis` says.
fn chain_ladder_on(cl: &ChainLadder, basis: &LdfBasis, origin: Grain) -> PyResult<ChainLadder> {
    let mut cl = cl.clone();
    cl.development.pattern = averaged(&cl.development.pattern, basis, origin)?;
    Ok(cl)
}

/// A method's ``repr`` with ``ldf_origin`` and ``ldf_underwriting`` before
/// its closing parenthesis, when they are set.
fn with_ldf_basis(repr: String, basis: &LdfBasis) -> String {
    let mut extra = String::new();
    if let Some(g) = basis.origin {
        extra.push_str(&format!(", ldf_origin={:?}", grain_name(g)));
    }
    if let Some(u) = &basis.underwriting {
        extra.push_str(&format!(", ldf_underwriting={}", underwriting_repr(u)));
    }
    match repr.strip_suffix(')') {
        Some(head) if !extra.is_empty() => format!("{head}{extra})"),
        _ => repr,
    }
}

fn underwriting_repr(u: &UnderwritingBasis) -> String {
    format!(
        "UnderwritingBasis(writing={:?}, policy_months={})",
        u.writing, u.policy_months
    )
}

/// How an underwriting period's policies are written and earned, to convert
/// an accident-period development pattern to the underwriting period's.
///
/// Parameters
/// ----------
/// writing : sequence of float, optional
///     The share of the period's policies written in each month from its
///     start, each written at an even rate through its month: finite, not
///     negative, with a positive sum, and scaled to sum to 1. Not given,
///     even writing over 12 months, an underwriting year's. The writing,
///     its trailing zeros dropped, must be no longer than the triangle's
///     origin period, within which its policies are written: on
///     underwriting quarters give three months (``[1, 1, 1]`` for even
///     writing); longer writing is an error at fit time.
/// policy_months : int, default 12
///     Each policy's term in months; it earns evenly over the term.
///
/// The underwriting period's exposure in its ``t``-th accident month is the
/// writing convolved with a policy's earning by month (``exposure``), and
/// the period's percent developed at age ``a`` from its start is
/// ``P(a) = sum_t e(t) p(a - t)``, ``p`` an accident month's (Rust's
/// ``SelectedPattern::to_underwriting_basis``). Pass it as
/// ``ldf_underwriting`` to a method that takes ``ldf``. The default, even
/// writing over a year and 12-month policies, is the parallelogram: the
/// exposure is triangular over 24 months.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import UnderwritingBasis
/// >>> e = UnderwritingBasis().exposure()
/// >>> len(e), round(e[0] * 288, 12), round(e[12] * 288, 12)
/// (24, 1.0, 23.0)
#[pyclass(name = "UnderwritingBasis", module = "prospicio.reserving", frozen)]
pub(crate) struct PyUnderwritingBasis {
    inner: UnderwritingBasis,
}

#[pymethods]
impl PyUnderwritingBasis {
    #[new]
    #[pyo3(signature = (writing = None, policy_months = 12))]
    fn new(writing: Option<Vec<f64>>, policy_months: u32) -> PyResult<Self> {
        let inner = match writing {
            Some(w) => UnderwritingBasis::new(w, policy_months),
            None => UnderwritingBasis::even(12, policy_months),
        };
        inner.exposure().map_err(err)?;
        Ok(Self { inner })
    }

    /// The writing shares by month, as given.
    #[getter]
    fn writing(&self) -> Vec<f64> {
        self.inner.writing.clone()
    }

    /// The policy term in months.
    #[getter]
    fn policy_months(&self) -> u32 {
        self.inner.policy_months
    }

    /// The share of the period's exposure in each accident month from its
    /// start, summing to 1.
    fn exposure(&self) -> PyResult<Vec<f64>> {
        self.inner.exposure().map_err(err)
    }

    fn __repr__(&self) -> String {
        underwriting_repr(&self.inner)
    }
}

fn past_pattern_end(name: &str) -> PyResult<PastPatternEnd> {
    match name {
        "reject" => Ok(PastPatternEnd::Reject),
        "no_development" => Ok(PastPatternEnd::NoDevelopment),
        "tail" => Ok(PastPatternEnd::Tail),
        _ => Err(PyValueError::new_err(format!(
            "past_pattern_end must be \"reject\", \"no_development\" or \"tail\", got {name:?}"
        ))),
    }
}

fn past_pattern_end_name(rule: PastPatternEnd) -> &'static str {
    match rule {
        PastPatternEnd::Reject => "reject",
        PastPatternEnd::NoDevelopment => "no_development",
        PastPatternEnd::Tail => "tail",
    }
}

/// Key columns of a dict from key name to column, in its order.
fn key_columns(keys: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(String, Vec<String>)>> {
    let Some(keys) = keys.filter(|k| !k.is_none()) else {
        return Ok(Vec::new());
    };
    let dict = keys
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("keys must be a dict of key name to column"))?;
    dict.iter()
        .map(|(k, v)| {
            let name: String = k
                .extract()
                .map_err(|_| PyTypeError::new_err("key names must be strings"))?;
            let column = key_values(&v, &name)?;
            Ok((name, column))
        })
        .collect()
}

/// One valuation month: a date (``datetime.date``, a pandas ``Timestamp``
/// or a numpy ``datetime64``), not an integer year, which would read as
/// January.
fn valuation_month(obj: &Bound<'_, PyAny>) -> PyResult<Month> {
    if obj.is_instance_of::<PyBool>()
        || obj.is_instance_of::<PyInt>()
        || (obj.hasattr("__index__")? && !obj.hasattr("dtype")?)
    {
        return Err(PyTypeError::new_err(format!(
            "give the valuation month as a date, got {}: an integer year would read as January",
            py_repr(obj)
        )));
    }
    if obj.hasattr("dtype")? {
        let flat = obj
            .py()
            .import("numpy")?
            .call_method1("asarray", (obj,))?
            .call_method1("reshape", (-1,))?;
        let kind: String = flat.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind != "M" {
            return Err(PyTypeError::new_err(format!(
                "give the valuation month as a date, got {}",
                py_repr(obj)
            )));
        }
        let all = months(&flat, "valuation")?;
        return match all.as_slice() {
            [one] => Ok(*one),
            _ => Err(PyValueError::new_err("valuation must be a single date")),
        };
    }
    month_of(obj, "valuation")
}

/// What `LatestDiagonal`'s constructors read: the long table's columns.
struct DiagonalArgs {
    keys: Vec<(String, Vec<String>)>,
    origin: Vec<Month>,
    age: DiagonalArgAge,
    latest: Vec<f64>,
    exposure: Option<Vec<f64>>,
    origin_grain: Grain,
    past_pattern_end: PastPatternEnd,
}

enum DiagonalArgAge {
    Ages(Vec<Lag>),
    Valuation(Month),
    Valuations(Vec<Month>),
}

impl DiagonalArgs {
    fn build(&self, py: Python<'_>) -> PyResult<PyLatestDiagonal> {
        let key_values: Vec<Vec<&str>> = self
            .keys
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let keys: Vec<(&str, &[&str])> = self
            .keys
            .iter()
            .zip(&key_values)
            .map(|((n, _), v)| (n.as_str(), v.as_slice()))
            .collect();
        let long = LongDiagonal {
            keys: &keys,
            origin: &self.origin,
            age: match &self.age {
                DiagonalArgAge::Ages(a) => DiagonalAge::Ages(a),
                DiagonalArgAge::Valuation(v) => DiagonalAge::Valuation(*v),
                DiagonalArgAge::Valuations(v) => DiagonalAge::Valuations(v),
            },
            latest: &self.latest,
            exposure: self.exposure.as_deref(),
            origin_grain: self.origin_grain,
        };
        let rule = self.past_pattern_end;
        let inner = py
            .detach(|| LatestDiagonals::from_long(&long))
            .map_err(err)?
            .with_past_pattern_end(rule);
        Ok(PyLatestDiagonal { inner })
    }
}

/// A latest diagonal on its own, one value per origin at its own age, as a
/// bordereau holds it, in one or several segments; projected with a
/// selected pattern from elsewhere by the ``fit_diagonal`` of
/// ``ChainLadder``, ``ExpectedLoss``, ``BornhuetterFerguson``,
/// ``Benktander`` and ``CapeCod`` (``docs/design/reserving-v02.md``,
/// decisions 15 to 19).
///
/// Each origin is given once per segment, at a positive age, with its
/// latest cumulative value (0 for an origin with nothing reported yet).
/// The ages may be off the pattern's grid (18 months against an annual
/// pattern): the fit reads them on the pattern's own ages, ``log(cdf)``
/// linear between them. An origin older than the pattern's end is an error
/// unless ``past_pattern_end`` says how to develop it.
///
/// Segments are named by key columns, as a ``Triangle``'s are, and sorted
/// by label; within a segment the origins keep the order given. Rows run
/// segment by segment in that order, as ``to_frame()`` lists them.
///
/// Parameters
/// ----------
/// origin : array-like
///     Any date in each row's origin period, or integer years.
/// age : array-like
///     Each row's age in months: from the start of the origin period to
///     the end of the valuation month (an accident year 2024 at December
///     2024 is 12 months).
/// latest : array-like
///     Each row's latest cumulative value.
/// exposure : array-like, optional
///     Each row's exposure (premium), for the expected-loss methods.
/// keys : dict of str to array-like, optional
///     Key columns by name, as ``Triangle.from_long`` takes them.
/// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
/// past_pattern_end : {"reject", "no_development", "tail"}, default "reject"
///     How an origin older than the pattern's end (its last age; for
///     age-to-age factors, the end of their span) is developed:
///     ``"reject"`` is an error that names it; ``"no_development"`` a
///     factor of 1 there, with no ``tail`` (a cumulative pattern's level
///     at its end is then rescaled to 100% for every origin, so with a
///     level other than 100% the young origins move too); ``"tail"`` the
///     method's ``tail`` continued from the pattern's end at one step per
///     origin period, a constant (``attachment_age`` unset) or one
///     estimated on the pattern's own factors.
///
/// Examples
/// --------
/// >>> import datetime
/// >>> from prospicio.reserving import BornhuetterFerguson, ChainLadder, LatestDiagonal
/// >>> d = LatestDiagonal.at_valuation(
/// ...     [2022, 2023, 2024], datetime.date(2024, 12, 31), [900.0, 600.0, 150.0],
/// ...     exposure=[1000.0, 1000.0, 1000.0],
/// ... )
/// >>> d.ages
/// [36, 24, 12]
/// >>> cl = ChainLadder(ldf={12: 2.0, 24: 1.25, 36: 1.0}, ldf_kind="cdf")
/// >>> cl.fit_diagonal(d).ultimate
/// [900.0, 750.0, 300.0]
/// >>> bf = BornhuetterFerguson(apriori=0.7, ldf={12: 2.0, 24: 1.25, 36: 1.0}, ldf_kind="cdf")
/// >>> round(bf.fit_diagonal(d).ultimate[2], 9)
/// 500.0
#[pyclass(name = "LatestDiagonal", module = "prospicio.reserving", frozen)]
pub(crate) struct PyLatestDiagonal {
    inner: LatestDiagonals,
}

#[pymethods]
impl PyLatestDiagonal {
    #[new]
    #[pyo3(signature = (origin, age, latest, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        age: &Bound<'_, PyAny>,
        latest: &Bound<'_, PyAny>,
        exposure: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        DiagonalArgs {
            keys: key_columns(keys)?,
            origin: months(origin, "origin")?,
            age: DiagonalArgAge::Ages(ages(age)?),
            latest: floats(latest, "latest")?,
            exposure: exposure
                .filter(|e| !e.is_none())
                .map(|e| floats(e, "exposure"))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// A diagonal valued at one month: each origin's age is the months from
    /// its start to the end of the valuation month (an accident year 2024
    /// at June 2024 is six months).
    ///
    /// Parameters
    /// ----------
    /// origin : array-like
    ///     Any date in each row's origin period, or integer years.
    /// valuation : date
    ///     Any date in the valuation month (``datetime.date``, a pandas
    ///     ``Timestamp`` or a numpy ``datetime64``); an integer is an error,
    ///     since a year would read as January.
    /// latest, exposure, keys, origin_grain, past_pattern_end
    ///     As for the constructor.
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``valuation`` is not a date.
    /// ValueError
    ///     If the valuation is before an origin starts (naming the row), an
    ///     origin is given twice in a segment, or a column has another
    ///     length.
    #[staticmethod]
    #[pyo3(signature = (origin, valuation, latest, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn at_valuation(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        valuation: &Bound<'_, PyAny>,
        latest: &Bound<'_, PyAny>,
        exposure: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        DiagonalArgs {
            keys: key_columns(keys)?,
            origin: months(origin, "origin")?,
            age: DiagonalArgAge::Valuation(valuation_month(valuation)?),
            latest: floats(latest, "latest")?,
            exposure: exposure
                .filter(|e| !e.is_none())
                .map(|e| floats(e, "exposure"))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// A diagonal from a data frame, one row per segment and origin.
    ///
    /// Columns are looked up with ``data[name]``, so a pandas or Polars
    /// DataFrame works, as does a dict of columns.
    ///
    /// Parameters
    /// ----------
    /// data : DataFrame or dict
    /// origin : str
    ///     Name of the origin column (dates or integer years).
    /// latest : str
    ///     Name of the latest-value column.
    /// age : str, optional
    ///     Name of the age column, in months.
    /// valuation : date or str, optional
    ///     The valuation month of every row (a date), or the name of a
    ///     column of each row's valuation date. Give exactly one of ``age``
    ///     and ``valuation``.
    /// exposure : str, optional
    ///     Name of the exposure column.
    /// keys : str or list of str, optional
    ///     Names of the key columns.
    /// origin_grain, past_pattern_end
    ///     As for the constructor.
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Examples
    /// --------
    /// >>> import datetime
    /// >>> from prospicio.reserving import LatestDiagonal
    /// >>> df = {
    /// ...     "lob": ["b", "a", "a"],
    /// ...     "year": [2024, 2023, 2024],
    /// ...     "paid": [50.0, 400.0, 150.0],
    /// ... }
    /// >>> d = LatestDiagonal.from_frame(
    /// ...     df, "year", "paid", valuation=datetime.date(2024, 12, 31), keys="lob"
    /// ... )
    /// >>> d.index, d.ages
    /// (['a', 'b'], [24, 12, 12])
    #[staticmethod]
    #[pyo3(signature = (data, origin, latest, age = None, valuation = None, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn from_frame(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        origin: &str,
        latest: &str,
        age: Option<&str>,
        valuation: Option<&Bound<'_, PyAny>>,
        exposure: Option<&str>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        let valuation = valuation.filter(|v| !v.is_none());
        let age = match (age, valuation) {
            (Some(name), None) => DiagonalArgAge::Ages(ages(&data.get_item(name)?)?),
            (None, Some(v)) if v.is_instance_of::<PyString>() => {
                let name: String = v.extract()?;
                DiagonalArgAge::Valuations(months(&data.get_item(name)?, "valuation")?)
            }
            (None, Some(v)) => DiagonalArgAge::Valuation(valuation_month(v)?),
            _ => {
                return Err(PyValueError::new_err(
                    "give exactly one of age (a column of ages) and valuation (a date, or a \
                     column of valuation dates)",
                ));
            }
        };
        let keys = match keys.filter(|k| !k.is_none()) {
            None => Vec::new(),
            Some(keys) => names(keys)?
                .into_iter()
                .map(|name| {
                    let column = key_values(&data.get_item(&name)?, &name)?;
                    Ok((name, column))
                })
                .collect::<PyResult<_>>()?,
        };
        DiagonalArgs {
            keys,
            origin: months(&data.get_item(origin)?, "origin")?,
            age,
            latest: floats(&data.get_item(latest)?, latest)?,
            exposure: exposure
                .map(|e| floats(&data.get_item(e)?, e))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// The latest diagonal of a triangle column, in every segment: each
    /// origin at the age of its last observed cell, with its cumulative
    /// value there, as chainladder-python's ``latest_diagonal``. With
    /// ``exposure``, each origin also takes that column's last observed
    /// value. This is how a fit is applied to another book:
    /// ``source.to_chain_ladder().fit_diagonal(LatestDiagonal.from_triangle(book, "paid"))``
    /// is chainladder-python's ``fit(source).predict(book)``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     With any number of segments.
    /// column : str
    /// exposure : str, optional
    /// past_pattern_end : {"reject", "no_development", "tail"}, default "reject"
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is unknown, an origin has no observed cell, or an
    ///     exposure is not finite and positive; with keys, the message names
    ///     the segment.
    #[staticmethod]
    #[pyo3(signature = (triangle, column, exposure = None, past_pattern_end = "reject"))]
    fn from_triangle(
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: Option<&str>,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        let rule = self::past_pattern_end(past_pattern_end)?;
        let inner = LatestDiagonals::from_triangle(&triangle.inner, column, exposure)
            .map_err(err)?
            .with_past_pattern_end(rule);
        Ok(Self { inner })
    }

    /// The diagonal with an exposure per row, in the row order of
    /// ``to_frame()``: the segments in order and, within one, its origins
    /// in the order given.
    fn with_exposure(&self, exposure: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exposure = floats(exposure, "exposure")?;
        let inner = self.inner.clone().with_exposure(&exposure).map_err(err)?;
        Ok(Self { inner })
    }

    /// The diagonal with another rule for origins older than the pattern's
    /// end: ``"reject"``, ``"no_development"`` or ``"tail"``.
    fn with_past_pattern_end(&self, rule: &str) -> PyResult<Self> {
        let rule = past_pattern_end(rule)?;
        Ok(Self {
            inner: self.inner.clone().with_past_pattern_end(rule),
        })
    }

    /// The one segment whose keys have the given values, as
    /// ``ChainLadderFit.segment``.
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    /// Names of the key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let n_keys = self.inner.key_names().len();
        self.inner
            .labels()
            .iter()
            .map(|l| label_to_py(py, n_keys, l))
            .collect()
    }

    /// Origin grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn origin_grain(&self) -> &'static str {
        grain_name(self.inner.origin_grain())
    }

    /// Origin period of each row.
    #[getter]
    fn origins(&self) -> Vec<String> {
        let grain = self.inner.origin_grain();
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.origins().iter())
            .map(|&m| Period::containing(m, grain).to_string())
            .collect()
    }

    /// Age of each row in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.ages().iter().copied())
            .collect()
    }

    /// Latest value of each row.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.latest().iter().copied())
            .collect()
    }

    /// Exposure of each row; ``None`` when not given.
    #[getter]
    fn exposure(&self) -> Option<Vec<f64>> {
        self.inner
            .diagonals()
            .iter()
            .map(|d| d.exposure().map(<[f64]>::to_vec))
            .collect::<Option<Vec<_>>>()
            .map(|e| e.concat())
    }

    /// The rule for origins older than the pattern's end.
    #[getter]
    fn past_pattern_end(&self) -> &'static str {
        past_pattern_end_name(self.inner.diagonals()[0].past_pattern_end())
    }

    /// The rows as a pandas DataFrame: the key columns, then ``origin``,
    /// ``age``, ``latest`` and, when given, ``exposure``.
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = PyDict::new(py);
        for (k, name) in self.inner.key_names().iter().enumerate() {
            let values: Vec<String> = self
                .inner
                .iter()
                .flat_map(|(l, d)| std::iter::repeat_n(l.parts()[k].clone(), d.origins().len()))
                .collect();
            out.set_item(name, values)?;
        }
        out.set_item("origin", self.origins())?;
        out.set_item("age", self.ages())?;
        out.set_item("latest", self.latest())?;
        if let Some(e) = self.exposure() {
            out.set_item("exposure", e)?;
        }
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    fn __repr__(&self) -> String {
        let segments = if self.inner.len() > 1 {
            format!("segments={}, ", self.inner.len())
        } else {
            String::new()
        };
        format!(
            "LatestDiagonal({segments}origins={}, origin_grain={:?}, exposure={}, past_pattern_end={:?})",
            self.ages().len(),
            self.origin_grain(),
            if self.exposure().is_some() {
                "True"
            } else {
                "False"
            },
            self.past_pattern_end()
        )
    }
}

/// A segment label as ``index`` writes it: ``"Total"`` without keys, a str
/// with one key, a tuple with several.
fn label_from_py(obj: &Bound<'_, PyAny>, n_keys: usize) -> PyResult<Label> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let parts: Vec<String> = tuple
            .iter()
            .map(|p| Ok(p.str()?.to_string()))
            .collect::<PyResult<_>>()?;
        return Ok(Label::new(parts));
    }
    let s = obj.str()?.to_string();
    Ok(if n_keys == 0 && s == "Total" {
        Label::default()
    } else {
        Label::from(s)
    })
}

/// The development of each segment of `diagonal` from the ``development``
/// and ``segment_map`` arguments of ``fit_diagonal``: ``None`` without
/// ``development``.
fn diagonal_development(
    diagonal: &LatestDiagonals,
    development: Option<&Bound<'_, PyAny>>,
    segment_map: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<SegmentFits<ChainLadder>>> {
    let segment_map = segment_map.filter(|m| !m.is_none());
    let Some(development) = development.filter(|d| !d.is_none()) else {
        if segment_map.is_some() {
            return Err(PyValueError::new_err(
                "segment_map is given without development",
            ));
        }
        return Ok(None);
    };
    let source = development
        .extract::<PyRef<'_, PyChainLadderFit>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "development must be a ChainLadderFit; for another method's fit, pass its \
                 .chain_ladder, got {}",
                py_repr(development)
            ))
        })?;
    let map = match segment_map {
        None => SegmentMap::ByLabel,
        Some(m) => {
            let dict = m.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err(
                    "segment_map must be a dict from a segment of the diagonal to a segment of \
                     the development",
                )
            })?;
            let (n_target, n_source) = (diagonal.key_names().len(), source.inner.key_names.len());
            SegmentMap::Pairs(
                dict.iter()
                    .map(|(t, s)| Ok((label_from_py(&t, n_target)?, label_from_py(&s, n_source)?)))
                    .collect::<PyResult<_>>()?,
            )
        }
    };
    diagonal
        .developments(&source.inner, &map)
        .map(Some)
        .map_err(err)
}

/// The chain-ladder method: each origin's latest value projected to
/// ultimate with age-to-age factors estimated from the triangle and a tail.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged into one factor per age: volume
///     weighted, their mean, or least squares through the origin (Mack's
///     ``alpha`` of 1, 0 and 2), or their geometric mean. A geometric
///     average estimates the factors only: ``Mack`` and the bootstraps
///     reject it, since it has no variance model.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a variance parameter with a single link ratio is filled in.
/// n_periods : int, optional
///     Average only the link ratios that start in the latest ``n_periods``
///     calendar diagonals of the triangle; an age with fewer uses all of
///     them. All of them by default.
/// drop_high : int, default 0
///     Leave out this many of the highest link ratios at each age, among
///     those ``n_periods`` keeps. Skipped at an age where it would leave
///     none.
/// drop_low : int, default 0
///     As ``drop_high``, for the lowest link ratios.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age: a number is a constant factor from
///     the oldest age to ultimate. No tail (a factor of 1) by default.
/// ldf : dict, optional
///     A selected development pattern, from age in months to value, which
///     replaces (or, with ``ldf_weight``, blends with) the factors
///     estimated from the triangle at the ages it gives; the other ages
///     stay estimated, so a benchmark for the immature ages only is a
///     partial pattern. As ``ldf_kind`` says, the values are age-to-age
///     factors, or factors to ultimate or percents developed, which select
///     the factors between the first age and the last (``cdf[a] / cdf[next
///     age]``); one factor to ultimate (or percent developed) is the tail
///     from its age, so it is given at the oldest age, with no ``tail`` and
///     ``ldf_weight`` 1, and on a triangle observed at one age it is the
///     whole projection. When every age given is a development
///     age of the triangle, an age-to-age factor links its age to the
///     triangle's next age (any age but the oldest: development past it is
///     the tail's), and a development age the cumulative values skip is
///     interpolated: between two given ages ``b < c``, ``log(cdf)`` is
///     linear in the age, so the factor from ``a`` to the next age ``a'``
///     is ``(cdf[b] / cdf[c]) ** ((a' - a) / (c - b))`` (percents
///     developed are converted to factors to ultimate first). Otherwise the
///     pattern is read on its own ages: an age-to-age factor runs to the
///     next age given (the last one by the spacing before it), a link of
///     the triangle within the pattern's span is selected with ``log(cdf)``
///     linear between the pattern's ages, and what the pattern gives past
///     the triangle's oldest age folds into a constant tail there, at full
///     weight. So a monthly pattern applies to an annual triangle. Adding
///     one off-grid age to a sparse age-to-age pattern changes how its
///     other factors are read: each then runs to the pattern's next age,
///     not the triangle's. On RAA, ``{12: 1.5, 18: 1.1, 36: 1.2}`` selects
///     1.5484 from 12 to 24, where ``{12: 1.5, 36: 1.2}`` selects 1.5, and
///     nothing warns of the change. Their level at their last age selects
///     nothing. With no ``tail`` and a last age that is the oldest, the
///     level is the tail: a constant tail factor equal to the factor to
///     ultimate there (1 over the percent developed), with no parameter
///     error, as chainladder-python's ``DevelopmentConstant`` keeps it
///     (``ChainLadderFit.pattern_tail``). Otherwise the level is rescaled
///     to the factors and tail in use (``ChainLadderFit.pattern_rescale``).
///     A selected factor at or after a tail's attachment age is an error.
/// ldf_kind : {"ldf", "cdf", "percent_developed"}, optional
///     How ``ldf`` is expressed; required with it, since age-to-age factors
///     and factors to ultimate cannot be told apart. Factors to ultimate
///     and percents developed need only be finite and positive, in any
///     order.
/// ldf_weight : float or dict, optional
///     The credibility ``w`` on each selected factor, from 0 to 1: the
///     factor used is ``w * selected + (1 - w) * estimated``. A number
///     applies to every selected factor, a dict from age (the age that
///     starts the factor's link) to weight to those listed, the others
///     fully selected. Not given, every selected factor is used as is
///     (``w = 1``).
/// ldf_origin : {"M", "Q", "S", "Y"}, optional
///     The origin grain ``ldf`` was measured on, when it is not the
///     triangle's: an accident-month pattern is ``"M"``. At fit time the
///     pattern is averaged over the sub-periods of each origin period, with
///     the exposure even through the period, into percents developed at
///     every month (Rust's ``SelectedPattern::averaged_over_origins``). The
///     ultimates are exact for origins up to the pattern's last age; past
///     it the development of the period's later sub-periods is deferred to
///     the period's last age, which overstates the ultimate slightly. An
///     estimated tail is then fitted on monthly factors, so a constant tail
///     taken from the monthly pattern's own tail is the safer choice. The
///     triangle's origin grain must be a whole multiple of it; a weight by
///     age (``ldf_weight`` a dict, applied before the averaging; the
///     averaged pattern cannot be weighted by age here) and a single
///     age-to-age factor are errors; the triangle's own grain leaves the
///     pattern as given. Requires ``ldf``. With ``ldf_underwriting`` it is
///     the grain of the accident periods ``ldf`` is converted from instead.
/// ldf_underwriting : UnderwritingBasis, optional
///     Converts ``ldf``, an accident-period pattern, to the pattern of the
///     triangle's underwriting periods at fit time: an underwriting year's
///     share developed at age ``a`` from its start is ``sum_t e(t) p(a -
///     t)``, with ``e`` its exposure by accident month (the writing
///     convolved with even earning over the policy term,
///     ``UnderwritingBasis.exposure``) and ``p`` an accident month's (Rust's
///     ``SelectedPattern::to_underwriting_basis``). ``ldf`` is on accident
///     periods of ``ldf_origin``, or of the triangle's origin grain without
///     it: an accident-year pattern on underwriting-year origins needs no
///     ``ldf_origin``. An accident-month pattern (``ldf_origin="M"``)
///     converts under any writing and term; a pattern of longer accident
///     periods needs a policy term that is a whole multiple of them or
///     writing even within each of them, and is read between its ages by
///     the rule above, so an annual accident-year pattern is approximate in
///     the first two years (31% too much developed at 12 months on a
///     simulated parallelogram); give it at every month, or an
///     accident-month pattern, where it matters. The result is longer than
///     ``ldf`` by the writing months plus the policy term less the accident
///     period (12 months for an accident-year pattern under the default,
///     23 for an accident-month one), and a constant tail from ``ldf``'s
///     own tail is the safer choice there. The writing must be no longer
///     than the triangle's origin period (the default's 12 months suit
///     underwriting years only; on underwriting quarters give
///     ``UnderwritingBasis([1, 1, 1])``), or the fit is an error. Requires
///     ``ldf``.
///
/// A selected factor carries no parameter error, and a blended one
/// ``(1 - w)`` times the estimate's, in ``Mack`` and the bootstraps; the
/// process error stays the triangle's. A supplied pattern therefore lowers
/// the reported uncertainty, and a wrong benchmark is not reflected in it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder().fit(tri, "paid")
/// >>> fit.ldf, fit.ultimate, fit.total_reserve
/// ([1.5], [150.0, 300.0], 100.0)
///
/// A benchmark for the immature ages only: 50% developed at 12 months and
/// 80% at 24, so the factor from 12 to 24 is 1.6; the factor from 24 to 36
/// stays the triangle's.
///
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0]},
/// ... )
/// >>> bench = ChainLadder(ldf={12: 0.5, 24: 0.8}, ldf_kind="percent_developed").fit(tri, "paid")
/// >>> [round(f, 4) for f in bench.ldf], [round(f, 4) for f in bench.estimated_ldf]
/// ([1.6, 1.1], [1.5238, 1.1])
/// >>> round(bench.ultimate[2], 2), round(bench.pattern_rescale, 4)
/// (211.2, 1.1364)
///
/// A pattern to the oldest age with no tail keeps its level as the tail:
/// 95% developed at 36 months is a tail of 1 / 0.95.
///
/// >>> full = ChainLadder(ldf={12: 0.5, 24: 0.8, 36: 0.95}, ldf_kind="percent_developed")
/// >>> fit = full.fit(tri, "paid")
/// >>> round(fit.pattern_tail, 4), round(fit.ultimate[2], 2)
/// (1.0526, 240.0)
#[pyclass(name = "ChainLadder", module = "prospicio.reserving", frozen)]
pub(crate) struct PyChainLadder {
    inner: ChainLadder,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyChainLadder {
    #[new]
    #[pyo3(signature = (
        average = "volume",
        sigma_interpolation = "log-linear",
        tail = None,
        ldf = None,
        ldf_kind = None,
        ldf_weight = None,
        n_periods = None,
        drop_high = 0,
        drop_low = 0,
        ldf_origin = None,
        ldf_underwriting = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        let pattern = selected_pattern(ldf, ldf_kind, ldf_weight)?;
        Ok(Self {
            ldf_basis,
            inner: ChainLadder {
                development: development(
                    average,
                    sigma_interpolation,
                    pattern,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                tail: tail_arg(tail)?,
            },
        })
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.development.pattern.as_ref())
    }

    /// Fits one measure column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative or incremental, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the column is unknown, a factor cannot be estimated or the tail
    ///     cannot be fitted (a constant that is not positive, a curve with
    ///     fewer than two factors above 1 to fit); with keys, the message
    ///     names the segment.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyChainLadderFit> {
        let tri = &triangle.inner;
        let cl = self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| cl.fit_segments(tri, column)).map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    /// Projects a latest diagonal, in every segment, with the fully
    /// selected pattern ``ldf`` (``ldf_weight`` 1, no ``n_periods`` or
    /// drops) or with the development of a fit (``development``):
    /// ``docs/design/reserving-v02.md``, decisions 15 to 19. Each origin's
    /// latest value is multiplied by the pattern's factor to ultimate at
    /// its own age, read ``log(cdf)``-linearly between the pattern's ages,
    /// and the tail. The fit's development axis is the grid the diagonal is
    /// placed on, which spans the pattern's ages as well as the
    /// diagonal's; its origins run oldest first over every period from the
    /// earliest to the latest. An origin older than the pattern's end is
    /// developed as the diagonal's ``past_pattern_end`` says.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    /// development : ChainLadderFit, optional
    ///     A fit whose factors and tail develop the diagonal, each segment by
    ///     the source segment of the same label (a single-segment source
    ///     develops every segment): chainladder-python's
    ///     ``fit(source).predict(target)``. For another method's fit, pass
    ///     its ``.chain_ladder``. The method itself then takes no ``ldf``,
    ///     no ``tail`` other than 1 and no ``n_periods``, ``drop_high`` or
    ///     ``drop_low``. The source's factors are read at each origin's own
    ///     age, as fitted, with no ``ldf_origin``: a source fitted on
    ///     another origin grain (accident quarters for a diagonal of
    ///     accident years, say) goes through ``to_chain_ladder()`` instead,
    ///     with ``ldf_origin`` added to the arguments it gives, one segment
    ///     at a time.
    /// segment_map : dict, optional
    ///     With ``development``, the source segment of each segment of the
    ///     diagonal, as ``{target: source}`` with labels written as
    ///     ``index`` writes them (``"Total"``, a str, or a tuple); every
    ///     segment of the diagonal listed once.
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there is no pattern, it is not fully selected, an origin is
    ///     younger than its first age, or older than its end with
    ///     ``past_pattern_end="reject"``; if the map leaves a segment
    ///     without a source; with keys, the message names the segment.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import ChainLadder, LatestDiagonal, Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    /// >>> own = ChainLadder().fit(tri, "paid")
    /// >>> d = LatestDiagonal.from_triangle(tri, "paid")
    /// >>> ChainLadder().fit_diagonal(d, development=own).ultimate
    /// [150.0, 300.0]
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyChainLadderFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyChainLadder {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<ChainLadder> {
        chain_ladder_on(&self.inner, &self.ldf_basis, origin)
    }

    fn base_repr(&self) -> String {
        format!(
            "ChainLadder(average={:?}, sigma_interpolation={:?}, tail={}{})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail),
            development_arg_repr(&self.inner.development)
        )
    }
}

/// The one fit of a single-segment result, or an error naming what to use
/// instead: the long table `instead` (if any) or `segment(...)`.
fn single<'a, T>(fits: &'a SegmentFits<T>, field: &str, instead: &str) -> PyResult<&'a T> {
    match fits.fits.as_slice() {
        [one] => Ok(one),
        _ => {
            let table = if instead.is_empty() {
                String::new()
            } else {
                format!("{instead} or ")
            };
            Err(PyValueError::new_err(format!(
                "{field} needs a single-segment fit, and this one has {} segments; \
                 use {table}segment(...)",
                fits.len()
            )))
        }
    }
}

/// Age from which a fit's selected factors are the tail's.
fn attachment_age(fit: &ChainLadderFit) -> PyResult<Lag> {
    fit.development
        .development
        .get(fit.tail.attachment)
        .copied()
        .ok_or_else(|| PyValueError::new_err("the fit has no development ages"))
}

/// Per-origin values of every segment, in the order of the long rows.
fn by_origin<T>(fits: &SegmentFits<T>, f: impl Fn(&T) -> Vec<f64>) -> Vec<f64> {
    fits.fits.iter().flat_map(f).collect()
}

/// Origin labels of every segment, in the order of the long rows.
fn origin_labels<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.fits
        .iter()
        .flat_map(|f| f.chain_ladder().origins.iter().map(ToString::to_string))
        .collect()
}

/// Labels of the segments, as ``Triangle.index``.
fn segment_labels<'py, T>(
    py: Python<'py>,
    fits: &SegmentFits<T>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let n_keys = fits.key_names.len();
    fits.labels
        .iter()
        .map(|l| label_to_py(py, n_keys, l))
        .collect()
}

/// Key values choosing one segment, compared as ``str()`` of each value.
fn segment_keys(keys: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, String)>> {
    match keys {
        None => Ok(Vec::new()),
        Some(keys) => keys
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.str()?.to_string())))
            .collect(),
    }
}

fn pick<T: Clone>(
    fits: &SegmentFits<T>,
    keys: Option<&Bound<'_, PyDict>>,
) -> PyResult<SegmentFits<T>> {
    let keys = segment_keys(keys)?;
    let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    fits.segment(&keys).map_err(err)
}

/// A long result table as a pandas DataFrame: the key columns, then
/// ``origin`` (str) or ``development`` (age), then the values.
fn table_frame<'py>(py: Python<'py>, table: FitTable) -> PyResult<Bound<'py, PyAny>> {
    let mut names: Vec<String> = Vec::new();
    if table.origin.is_some() {
        names.push("origin".into());
    }
    if table.age.is_some() {
        names.push("development".into());
    }
    names.extend(table.values.iter().map(|(n, _)| n.clone()));
    if let Some((k, _)) = table.keys.iter().find(|(k, _)| names.contains(k)) {
        return Err(PyValueError::new_err(format!(
            "key {k:?} clashes with the {k:?} column of the result"
        )));
    }
    let out = PyDict::new(py);
    for (name, values) in table.keys {
        out.set_item(name, values)?;
    }
    if let Some(origin) = table.origin {
        let labels: Vec<String> = origin.iter().map(ToString::to_string).collect();
        out.set_item("origin", labels)?;
    }
    if let Some(age) = table.age {
        out.set_item("development", age)?;
    }
    for (name, values) in table.values {
        out.set_item(name, values)?;
    }
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// The ``repr`` prefix naming the number of segments when there are several.
fn segments_prefix<T>(fits: &SegmentFits<T>) -> String {
    if fits.len() > 1 {
        format!("segments={}, ", fits.len())
    } else {
        String::new()
    }
}

/// A fitted chain-ladder projection of every segment of a triangle column.
///
/// Per-origin lists (``origins``, ``latest``, ``ultimate``, ``reserve``)
/// run over the origins of each segment in turn, like the rows of
/// ``to_frame()``, so a single-segment fit has one value per origin.
/// Per-age lists (``ldf``, ``cdf``, ``sigma``, ``std_err``) and the tail
/// need a single-segment fit; for several segments use
/// ``development_frame()`` (per age), ``totals_frame()`` (``tail``,
/// ``tail_sigma``, ``tail_std_err``) or ``segment(...)``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021] * 2,
/// ...     [12, 24, 12] * 2,
/// ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0]},
/// ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
/// ... )
/// >>> fit = ChainLadder().fit(tri, "paid")
/// >>> fit.index, fit.reserve
/// (['Auto', 'Home'], [0.0, 100.0, 0.0, 30.0])
/// >>> fit.segment(lob="Home").ldf
/// [2.0]
#[pyclass(name = "ChainLadderFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyChainLadderFit {
    inner: SegmentFits<ChainLadderFit>,
}

#[pymethods]
impl PyChainLadderFit {
    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0].development.development.clone()
    }

    /// The fit frozen as a ``ChainLadder``: its factors in use (``ldf``) as
    /// fully selected age-to-age factors at its ages, and its tail factor
    /// as a constant tail (``docs/design/reserving-v02.md``, decision 17).
    /// Applied to the fit's own latest diagonal it projects as the fit did,
    /// the factors to the bit; to another book's
    /// (``LatestDiagonal.from_triangle``) it is chainladder-python's
    /// ``fit(source).predict(book)``. Its ``ldf``, ``ldf_kind`` and ``tail``
    /// unpack into any method's arguments, as
    /// ``BornhuetterFerguson(apriori=0.6, ldf=cl.ldf, ldf_kind=cl.ldf_kind,
    /// tail=cl.tail)``. It does not carry sigma, the standard errors or the
    /// tail's form. Needs a single-segment fit: for several segments pass
    /// the fit as ``development`` to ``fit_diagonal``, or use
    /// ``segment(...)``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import ChainLadder, LatestDiagonal, Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    /// >>> cl = ChainLadder(tail=1.1).fit(tri, "paid").to_chain_ladder()
    /// >>> cl.ldf, cl.ldf_kind, cl.tail
    /// ({12: 1.5}, 'ldf', 1.1)
    fn to_chain_ladder(&self) -> PyResult<PyChainLadder> {
        let f = single(&self.inner, "to_chain_ladder", "")?;
        Ok(PyChainLadder {
            inner: f.to_chain_ladder(),
            ldf_basis: LdfBasis::default(),
        })
    }

    /// Selected age-to-age factors, which the projection uses: the
    /// estimated ones (or a selected pattern's, ``blended_ldf``), replaced
    /// by the tail's from its attachment age. Factor ``k`` links age ``k``
    /// to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.ldf().to_vec())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .cdf
            .clone())
    }

    /// Variance parameter of each factor, with unestimable ones
    /// interpolated (``nan`` where that is impossible).
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "sigma", "development_frame()")?;
        Ok(f.development.sigma.clone())
    }

    /// Standard error of each factor used before the tail: ``(1 -
    /// ldf_weight)`` times ``estimated_std_err``, so 0 where a selected
    /// pattern replaces the estimate.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.blended_std_err())
    }

    /// Number of link ratios behind each estimated factor, after
    /// ``n_periods`` and the drops; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn n_links(&self) -> PyResult<Vec<usize>> {
        let f = single(&self.inner, "n_links", "development_frame()")?;
        Ok(f.development.n_links.clone())
    }

    /// Whether ``n_periods`` or the drops left out a link ratio at some age.
    #[getter]
    fn excluded(&self) -> PyResult<bool> {
        let f = single(&self.inner, "excluded", "development_frame()")?;
        Ok(f.development.excluded)
    }

    /// Age-to-age factors as estimated from the triangle, before a selected
    /// pattern or the tail replaced any.
    #[getter]
    fn estimated_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "estimated_ldf", "")?;
        Ok(f.development.ldf.clone())
    }

    /// Standard error of each estimated factor.
    #[getter]
    fn estimated_std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "estimated_std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    /// The selected pattern's age-to-age factor of each link, after
    /// conversion from ``ldf_kind``; ``None`` where it selects none.
    #[getter]
    fn selected_ldf(&self) -> PyResult<Vec<Option<f64>>> {
        let f = single(&self.inner, "selected_ldf", "development_frame()")?;
        Ok(f.development.selected.clone())
    }

    /// The weight on each link's selected factor; 0 where none is
    /// selected.
    #[getter]
    fn ldf_weight(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf_weight", "development_frame()")?;
        Ok(f.development.weight.clone())
    }

    /// The factors before the tail: ``w * selected + (1 - w) * estimated``
    /// where a factor is selected, the estimate elsewhere.
    #[getter]
    fn blended_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "blended_ldf", "development_frame()")?;
        Ok(f.development.blended_ldf())
    }

    /// For a pattern given as factors to ultimate or percents developed,
    /// the factor that multiplies the given percents developed (and divides
    /// the given factors to ultimate) to reconcile the pattern's level at
    /// its last age with the factors and tail in use: ``cdf_given /
    /// cdf_used`` there, so that a pattern given to the oldest age with a
    /// ``tail``, rescaled, times the tail is 100%. 1 for a pattern to the
    /// oldest age with no tail, whose level is the tail (``pattern_tail``).
    /// At earlier ages the rescaled values are the projection's only where
    /// every link is fully selected; with ``ldf_weight`` below 1, read
    /// ``cdf`` for the factors in use. ``None`` for an age-to-age pattern
    /// or none.
    #[getter]
    fn pattern_rescale(&self) -> PyResult<Option<f64>> {
        Ok(single(&self.inner, "pattern_rescale", "totals_frame()")?.pattern_rescale)
    }

    /// The tail a pattern of factors to ultimate or percents developed
    /// implies: with no ``tail`` and a pattern to the oldest age, its
    /// factor to ultimate there (1 over its percent developed), used as a
    /// constant tail with no parameter error (``tail_std_err`` 0, its sigma
    /// extrapolated as a constant tail's). ``None`` with a ``tail``, for a
    /// pattern that stops before the oldest age, an age-to-age pattern, or
    /// none.
    #[getter]
    fn pattern_tail(&self) -> PyResult<Option<f64>> {
        Ok(single(&self.inner, "pattern_tail", "totals_frame()")?.pattern_tail)
    }

    /// Age from which ``ldf`` holds the tail's factors rather than the
    /// estimated ones; the oldest age when the tail replaced none.
    #[getter]
    fn tail_attachment_age(&self) -> PyResult<Lag> {
        attachment_age(single(&self.inner, "tail_attachment_age", "")?)
    }

    /// Tail factor from the oldest age to ultimate.
    #[getter]
    fn tail(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail", "totals_frame()")?.tail.factor)
    }

    /// Factors past the oldest age, which multiply to ``tail``: one per
    /// development period of the following year and one to ultimate, as
    /// chainladder-python's ``ldf_`` (a single factor for
    /// ``TailLogLinear``).
    #[getter]
    fn tail_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "tail_ldf", "")?;
        Ok(f.tail.ldf[f.development.ldf.len()..].to_vec())
    }

    /// The tail's variance parameter, extrapolated log-linearly; 0 without
    /// a tail (a factor of 1), ``nan`` if it cannot be extrapolated. A tail
    /// below 1 is read where a tail of 1.001 would be, as chainladder-python
    /// does.
    #[getter]
    fn tail_sigma(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail_sigma", "totals_frame()")?
            .tail
            .sigma)
    }

    /// Standard error of the tail factor, extrapolated log-linearly.
    #[getter]
    fn tail_std_err(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail_std_err", "totals_frame()")?
            .tail
            .std_err)
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.latest.clone())
    }

    /// Projected ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ChainLadderFit::reserves)
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate`` and ``reserve``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, and its ``tail``,
    /// ``tail_sigma`` and ``tail_std_err``; with a selected pattern, its
    /// ``pattern_rescale`` and ``pattern_tail`` too (``nan`` where there is
    /// none). Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age: the key columns, ``development``,
    /// ``ldf`` (the selected factor to the next age), ``cdf`` (to ultimate,
    /// with the tail), ``sigma`` and ``std_err``; the oldest age has ``nan``
    /// for ``ldf``, ``sigma`` and ``std_err``, and the tail factor as its
    /// ``cdf``. With a selected pattern, also ``estimated_ldf``,
    /// ``selected_ldf`` (``nan`` where none), ``ldf_weight``,
    /// ``blended_ldf`` and ``estimated_std_err``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values (compared as ``str()``
    /// of each value). Keys not named may take any value, so a fit with one
    /// segment needs none.
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key or value is unknown, or the choice matches several
    ///     segments.
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLadderFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// Mack's distribution-free chain ladder: the chain-ladder projection plus
/// the standard error of each origin's reserve and of the total, split into
/// process and parameter risk (Mack 1993, 1999).
///
/// A tail other than 1 is one more development step, from the oldest age
/// to ultimate, with its own sigma and standard error, as R ChainLadder's
/// ``MackChainLadder(tail = ...)``; unless given, both are extrapolated
/// log-linearly. Every origin, the oldest included, carries the tail's risk.
/// A tail below 1 follows chainladder-python: it scales the ultimates and
/// carries the risk read where a tail of 1.001 would be. R's
/// ``MackChainLadder`` ignores a tail below 1 altogether.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``. Mack's standard errors follow the link ratios
///     that are kept (R's ``MackChainLadder(weights = ...)``).
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// tail_sigma : float, optional
///     The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
///     Unused when the tail factor is 1.
/// tail_std_err : float, optional
///     The tail factor's standard error (R's ``tail.se``); extrapolated if
///     not given. Unused when the tail factor is 1.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``. A selected
///     factor is taken as known: it has no parameter error, and a blended
///     one ``(1 - w)`` times the estimate's standard error, so the
///     parameter risk is ``(1 - w)`` times Mack's on those links (none for
///     a fully selected pattern). The process risk keeps the triangle's
///     sigmas, around the factors used. A supplied pattern therefore lowers
///     the reported standard errors, and a wrong benchmark is not reflected
///     in them. ``claims_development_result()`` takes the pattern too. With
///     no ``tail``, a cumulative pattern to the oldest age implies a constant
///     tail (``ChainLadderFit.pattern_tail``) with no parameter error
///     (``tail_std_err`` 0 unless given) and its sigma extrapolated, as a
///     constant tail without ``tail_sigma``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack().fit(tri, "values")
/// >>> fit.total_standard_error > 0 and fit.standard_error[0] == 0
/// True
#[pyclass(name = "Mack", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMack {
    inner: Mack,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyMack {
    #[new]
    #[pyo3(signature = (
        average = "volume",
        sigma_interpolation = "log-linear",
        tail = None,
        tail_sigma = None,
        tail_std_err = None,
        ldf = None,
        ldf_kind = None,
        ldf_weight = None,
        n_periods = None,
        drop_high = 0,
        drop_low = 0,
        ldf_origin = None,
        ldf_underwriting = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        tail_sigma: Option<f64>,
        tail_std_err: Option<f64>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        let pattern = selected_pattern(ldf, ldf_kind, ldf_weight)?;
        Ok(Self {
            ldf_basis,
            inner: Mack {
                development: development(
                    average,
                    sigma_interpolation,
                    pattern,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                tail: tail_arg(tail)?,
                tail_sigma,
                tail_std_err,
            },
        })
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The given tail sigma, or ``None`` to extrapolate it.
    #[getter]
    fn tail_sigma(&self) -> Option<f64> {
        self.inner.tail_sigma
    }

    /// The given standard error of the tail factor, or ``None`` to
    /// extrapolate it.
    #[getter]
    fn tail_std_err(&self) -> Option<f64> {
        self.inner.tail_std_err
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.development.pattern.as_ref())
    }

    /// Fits one measure column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// MackFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, and if the triangle has fewer than three
    ///     ages, a variance parameter can be neither estimated nor
    ///     interpolated, or the tail's sigma or standard error can neither be
    ///     extrapolated nor is given.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyMackFit> {
        let tri = &triangle.inner;
        let mack = self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| mack.fit_segments(tri, column)).map_err(err)?;
        Ok(PyMackFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyMack {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<Mack> {
        let mut m = self.inner.clone();
        m.development.pattern = averaged(&m.development.pattern, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        let given =
            |name: &str, v: Option<f64>| v.map_or(String::new(), |v| format!(", {name}={v:?}"));
        format!(
            "Mack(average={:?}, sigma_interpolation={:?}, tail={}{}{}{})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail),
            given("tail_sigma", self.inner.tail_sigma),
            given("tail_std_err", self.inner.tail_std_err),
            development_arg_repr(&self.inner.development),
        )
    }
}

/// A fitted Mack model of every segment: the chain-ladder fields, plus
/// standard errors of each origin's reserve and of each segment's total.
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``. Per-age lists, the tail and the totals'
/// standard errors need a single-segment fit; for several segments use
/// ``development_frame()``, ``totals_frame()`` (the totals' standard errors
/// and the tail) or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
#[pyclass(name = "MackFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackFit {
    inner: SegmentFits<MackFit>,
}

impl PyMackFit {
    /// The chain ladder of a single-segment fit, for a `field` found in the
    /// long table `instead` (if any) of a fit with several segments.
    fn one(&self, field: &str, instead: &str) -> PyResult<&ChainLadderFit> {
        Ok(&single(&self.inner, field, instead)?.chain_ladder)
    }
}

#[pymethods]
impl PyMackFit {
    /// The underlying chain-ladder projection.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|m| m.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Selected age-to-age factors, as ``ChainLadderFit.ldf``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("ldf", "development_frame()")?.ldf().to_vec())
    }

    /// Age-to-ultimate factors, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("cdf", "development_frame()")?.cdf.clone())
    }

    /// Tail factor from the oldest age to ultimate.
    #[getter]
    fn tail(&self) -> PyResult<f64> {
        Ok(self.one("tail", "totals_frame()")?.tail.factor)
    }

    /// Factors as estimated, as ``ChainLadderFit.estimated_ldf``.
    #[getter]
    fn estimated_ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("estimated_ldf", "")?.development.ldf.clone())
    }

    /// Standard error of each estimated factor, as
    /// ``ChainLadderFit.estimated_std_err``.
    #[getter]
    fn estimated_std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("estimated_std_err", "development_frame()")?
            .development
            .std_err
            .clone())
    }

    /// The selected pattern's factor of each link, as
    /// ``ChainLadderFit.selected_ldf``.
    #[getter]
    fn selected_ldf(&self) -> PyResult<Vec<Option<f64>>> {
        Ok(self
            .one("selected_ldf", "development_frame()")?
            .development
            .selected
            .clone())
    }

    /// The weight on each link's selected factor, as
    /// ``ChainLadderFit.ldf_weight``.
    #[getter]
    fn ldf_weight(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("ldf_weight", "development_frame()")?
            .development
            .weight
            .clone())
    }

    /// The factors before the tail, as ``ChainLadderFit.blended_ldf``.
    #[getter]
    fn blended_ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("blended_ldf", "development_frame()")?
            .development
            .blended_ldf())
    }

    /// The rescale of a cumulative pattern, as
    /// ``ChainLadderFit.pattern_rescale``.
    #[getter]
    fn pattern_rescale(&self) -> PyResult<Option<f64>> {
        Ok(self
            .one("pattern_rescale", "totals_frame()")?
            .pattern_rescale)
    }

    /// The tail a cumulative pattern implies, as
    /// ``ChainLadderFit.pattern_tail``.
    #[getter]
    fn pattern_tail(&self) -> PyResult<Option<f64>> {
        Ok(self.one("pattern_tail", "totals_frame()")?.pattern_tail)
    }

    /// Age from which ``ldf`` holds the tail's factors, as
    /// ``ChainLadderFit.tail_attachment_age``.
    #[getter]
    fn tail_attachment_age(&self) -> PyResult<Lag> {
        attachment_age(self.one("tail_attachment_age", "")?)
    }

    /// Factors past the oldest age, as ``ChainLadderFit.tail_ldf``.
    #[getter]
    fn tail_ldf(&self) -> PyResult<Vec<f64>> {
        let cl = self.one("tail_ldf", "")?;
        Ok(cl.tail.ldf[cl.development.ldf.len()..].to_vec())
    }

    /// The tail's sigma used in the process risk: given, or extrapolated
    /// log-linearly; 0 without a tail (a factor of 1).
    #[getter]
    fn tail_sigma(&self) -> PyResult<f64> {
        Ok(self.one("tail_sigma", "totals_frame()")?.tail.sigma)
    }

    /// The tail factor's standard error used in the parameter risk: given,
    /// or extrapolated log-linearly; 0 without a tail (a factor of 1).
    #[getter]
    fn tail_std_err(&self) -> PyResult<f64> {
        Ok(self.one("tail_std_err", "totals_frame()")?.tail.std_err)
    }

    /// Variance parameter of each factor.
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("sigma", "development_frame()")?
            .development
            .sigma
            .clone())
    }

    /// Standard error of each factor used before the tail, as
    /// ``ChainLadderFit.std_err``: the parameter risk's.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("std_err", "development_frame()")?
            .development
            .blended_std_err())
    }

    /// Number of link ratios behind each estimated factor, as
    /// ``ChainLadderFit.n_links``.
    #[getter]
    fn n_links(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .one("n_links", "development_frame()")?
            .development
            .n_links
            .clone())
    }

    /// Whether ``n_periods`` or the drops left out a link ratio at some age.
    #[getter]
    fn excluded(&self) -> PyResult<bool> {
        Ok(self
            .one("excluded", "development_frame()")?
            .development
            .excluded)
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Projected ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.ultimate.clone())
    }

    /// Reserve per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.reserves())
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// Process standard error per origin.
    #[getter]
    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    /// Parameter (estimation) standard error per origin.
    #[getter]
    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    /// Mack standard error per origin: ``sqrt(process**2 + parameter**2)``.
    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    /// Process standard error of the total reserve.
    #[getter]
    fn total_process_risk(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_process_risk", "totals_frame()")?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, including the
    /// correlation between origins that share estimated factors.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_parameter_risk", "totals_frame()")?.total_parameter_risk)
    }

    /// Mack standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_standard_error", "totals_frame()")?.total_standard_error)
    }

    /// Coefficient of variation of the total reserve.
    #[getter]
    fn total_cv(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_cv", "totals_frame()")?.total_cv())
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``process_risk``,
    /// ``parameter_risk`` and ``standard_error``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, and the ``process_risk``,
    /// ``parameter_risk`` and ``standard_error`` of its total reserve.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// Merz and Wüthrich's (2008) one-year view: the standard error of the
    /// claims development result of each origin and in total, in the next
    /// calendar year and in every later one, as R ChainLadder's
    /// ``CDR(MackChainLadder(x), dev = "all")``.
    ///
    /// The method form of the ``cdr`` property, kept as its alias.
    ///
    /// With a selected pattern (``Mack(ldf=...)``), each factor used varies
    /// with ``1 - w`` times the estimate's parameter error, so a fully
    /// selected link adds process error only; with weights strictly
    /// between 0 and 1 the calendar years' CDRs are correlated, and their
    /// standard errors need not add up to Mack's
    /// (``docs/design/reserving-v02.md``, decision 12).
    ///
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``), the
    ///     factors are not volume-weighted, the fit has a tail (a factor
    ///     other than 1, or one that replaces estimated factors), the fit
    ///     excludes link ratios (``n_periods``, ``drop_high`` or
    ///     ``drop_low``; use ``MackBootstrap.one_year``, which follows
    ///     them), or the latest values do not lie on one calendar diagonal
    ///     with one new origin per period.
    fn claims_development_result(&self) -> PyResult<PyClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", "")?;
        let inner = fit.claims_development_result().map_err(err)?;
        Ok(PyClaimsDevelopmentResult { inner })
    }

    /// Merz and Wüthrich's one-year view, as a property: the same
    /// ``ClaimsDevelopmentResult`` that ``claims_development_result()``
    /// returns (that method stays as an alias), named as ``OneYearFit.cdr``
    /// names the claims development result. This one holds the analytic
    /// standard errors; ``OneYearFit.cdr`` is the simulated distribution.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``claims_development_result()``.
    ///
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    #[getter]
    fn cdr(&self) -> PyResult<PyClaimsDevelopmentResult> {
        self.claims_development_result()
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// MackFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        let se = match self.inner.fits.as_slice() {
            [one] => format!(", total_standard_error={:?}", one.total_standard_error),
            _ => String::new(),
        };
        format!(
            "MackFit({}origins={}, total_reserve={:?}{se})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_reserve(),
        )
    }
}

/// Merz and Wüthrich's (2008) one-year view of a Mack fit: standard errors
/// of the claims development result (CDR), the change in the chain-ladder
/// ultimate over a calendar year, per origin and in total. The total
/// includes the covariance between origins. Year ``k`` of the run-off is
/// R ChainLadder's ``CDR(k)S.E.``; summed in square over the years, the
/// run-off gives back Mack's standard error.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> mack = Mack().fit(tri, "values")
/// >>> cdr = mack.claims_development_result()
/// >>> len(cdr.by_calendar_year), cdr.one_year_standard_error[0]
/// (3, 0.0)
/// >>> abs(cdr.total_run_off_standard_error - mack.total_standard_error) < 1e-9
/// True
#[pyclass(
    name = "ClaimsDevelopmentResult",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyClaimsDevelopmentResult {
    inner: ClaimsDevelopmentResult,
}

#[pymethods]
impl PyClaimsDevelopmentResult {
    /// Origin periods, oldest first.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Standard error of each origin's CDR in the next calendar year, R's
    /// ``CDR(1)S.E.``.
    #[getter]
    fn one_year_standard_error(&self) -> Vec<f64> {
        self.inner.one_year_standard_error.clone()
    }

    /// Standard error of the total CDR in the next calendar year.
    #[getter]
    fn total_one_year_standard_error(&self) -> f64 {
        self.inner.total_one_year_standard_error
    }

    /// Standard error of each origin's CDR in each future calendar year:
    /// ``by_calendar_year[k - 1][i]`` is year ``k`` (R's ``CDR(k)S.E.``) of
    /// origin ``i``, zero once the origin is fully developed. One year per
    /// age-to-age factor.
    #[getter]
    fn by_calendar_year(&self) -> Vec<Vec<f64>> {
        self.inner.by_calendar_year.clone()
    }

    /// Standard error of the total CDR in each future calendar year.
    #[getter]
    fn total_by_calendar_year(&self) -> Vec<f64> {
        self.inner.total_by_calendar_year.clone()
    }

    /// Standard error of each origin's full run-off, the square root of
    /// the sum of its yearly mean squared errors; equals Mack's.
    #[getter]
    fn run_off_standard_error(&self) -> Vec<f64> {
        self.inner.run_off_standard_error()
    }

    /// Standard error of the total full run-off; equals Mack's.
    #[getter]
    fn total_run_off_standard_error(&self) -> f64 {
        self.inner.total_run_off_standard_error()
    }

    /// One row per origin: ``origin``, then ``cdr_1``, ``cdr_2``, ... the
    /// standard error of the CDR in each future calendar year (R's
    /// ``CDR(k)S.E.``), and ``run_off``, that of the full run-off. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut values: Vec<(String, Vec<f64>)> = self
            .inner
            .by_calendar_year
            .iter()
            .enumerate()
            .map(|(t, year)| (format!("cdr_{}", t + 1), year.clone()))
            .collect();
        values.push(("run_off".into(), self.inner.run_off_standard_error()));
        let table = FitTable {
            keys: Vec::new(),
            origin: Some(self.inner.origins.clone()),
            age: None,
            values,
        };
        table_frame(py, table)
    }

    fn __repr__(&self) -> String {
        format!(
            "ClaimsDevelopmentResult(origins={}, total_one_year_standard_error={:?}, \
             total_run_off_standard_error={:?})",
            self.inner.origins.len(),
            self.inner.total_one_year_standard_error,
            self.inner.total_run_off_standard_error()
        )
    }
}

/// The chain ladder an expected-loss method estimates its development
/// pattern with.
#[allow(clippy::too_many_arguments)]
fn pattern(
    average: &str,
    sigma_interpolation: &str,
    tail: Option<&Bound<'_, PyAny>>,
    ldf: Option<&Bound<'_, PyAny>>,
    ldf_kind: Option<&str>,
    ldf_weight: Option<&Bound<'_, PyAny>>,
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> PyResult<ChainLadder> {
    let pattern = selected_pattern(ldf, ldf_kind, ldf_weight)?;
    Ok(ChainLadder {
        development: development(
            average,
            sigma_interpolation,
            pattern,
            n_periods,
            drop_high,
            drop_low,
        )?,
        tail: tail_arg(tail)?,
    })
}

/// The ``repr`` arguments describing a method's development pattern.
fn pattern_repr(cl: &ChainLadder) -> String {
    format!(
        "average={:?}, sigma_interpolation={:?}, tail={}{}",
        average_name(cl.development.average),
        sigma_interpolation_name(cl.development.sigma_interpolation),
        tail_arg_repr(&cl.tail),
        development_arg_repr(&cl.development)
    )
}

/// An ``apriori`` argument: a number for every origin, a sequence of one
/// per origin in origin order (oldest first, over the triangle's or the
/// diagonals' origins), or a dict from origin (a label as fits report
/// origins, ``"2021"`` or ``"2021Q3"``, or an integer year) to its loss
/// ratio. A pandas Series is read by its index, as a dict; one with a
/// default ``RangeIndex`` is an error, since its order cannot be checked.
/// ``None`` is 1.
fn apriori_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Apriori> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Apriori::default());
    };
    let type_error = || {
        PyTypeError::new_err(format!(
            "apriori must be a number, a sequence of numbers in origin order or a dict from              origin to number, got {}",
            py_repr(obj)
        ))
    };
    if obj.is_instance_of::<PyBool>() {
        return Err(type_error());
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return apriori_by_origin(dict.iter().map(Ok));
    }
    if is_pandas_series(obj)? {
        // A default index, 0 to n - 1 by 1; `range(1988, 1998)` is years.
        let index = obj.getattr("index")?;
        if index.get_type().name()?.extract::<String>()? == "RangeIndex"
            && index.getattr("start")?.extract::<i64>()? == 0
            && index.getattr("step")?.extract::<i64>()? == 1
        {
            return Err(PyValueError::new_err(
                "apriori is a pandas Series with a default RangeIndex, so the origin of each                  value cannot be checked: index it by origin label or integer year (as                  df.set_index(\"origin\")[\"elr\"]), or pass .tolist() for values in origin                  order",
            ));
        }
        // `items()`, not `to_dict()`, so an origin given twice stays an error.
        return apriori_by_origin(
            obj.call_method0("items")?
                .try_iter()?
                .map(|item| item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()),
        );
    }
    if let Ok(value) = obj.extract::<f64>() {
        return Ok(Apriori::All(value));
    }
    plain(obj)?
        .extract::<Vec<f64>>()
        .map(Apriori::InOrder)
        .map_err(|_| type_error())
}

/// Whether `obj` is a pandas Series, found without importing pandas: its
/// type comes from a `pandas` module and is named `Series`.
fn is_pandas_series(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let ty = obj.get_type();
    let module = ty
        .getattr("__module__")?
        .extract::<String>()
        .unwrap_or_default();
    Ok((module == "pandas" || module.starts_with("pandas."))
        && ty.name()?.extract::<String>()? == "Series")
}

/// An ``apriori`` keyed by origin from `(key, value)` pairs: a key is an
/// origin label or an integer year.
fn apriori_by_origin<'py>(
    items: impl Iterator<Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>>,
) -> PyResult<Apriori> {
    let pairs = items
        .map(|item| {
            let (k, v) = item?;
            let label = if k.is_instance_of::<PyString>() {
                k.extract::<String>()?
            } else if !k.is_instance_of::<PyBool>()
                && (k.is_instance_of::<PyInt>() || k.hasattr("__index__")?)
            {
                k.extract::<i32>()?.to_string()
            } else {
                return Err(PyTypeError::new_err(format!(
                    "apriori's keys must be origin labels such as \"2021\" or \"2021Q3\", or                      integer years, got {}",
                    py_repr(&k)
                )));
            };
            let value = v.extract::<f64>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "apriori's value for {label} must be a number, got {}",
                    py_repr(&v)
                ))
            })?;
            Ok((label, value))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Apriori::from_labels(pairs).map_err(err)
}

/// A method's ``apriori`` as Python sees it: a float, a list in origin
/// order or a dict by origin label.
fn apriori_object<'py>(py: Python<'py>, apriori: &Apriori) -> PyResult<Bound<'py, PyAny>> {
    Ok(match apriori {
        Apriori::All(a) => PyFloat::new(py, *a).into_any(),
        Apriori::InOrder(values) => PyList::new(py, values)?.into_any(),
        Apriori::ByOrigin(pairs) => {
            let d = PyDict::new(py);
            for (origin, v) in pairs {
                d.set_item(origin.to_string(), v)?;
            }
            d.into_any()
        }
    })
}

/// ``apriori`` in a method's repr.
fn apriori_repr(apriori: &Apriori) -> String {
    match apriori {
        Apriori::All(a) => format!("{a:?}"),
        Apriori::InOrder(values) => format!("{values:?}"),
        Apriori::ByOrigin(pairs) => {
            let items: Vec<String> = pairs.iter().map(|(o, v)| format!("'{o}': {v:?}")).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// The expected loss ratio method: each origin's ultimate is ``apriori``
/// times its exposure, whatever has been observed. The chain ladder is
/// still fitted for the development pattern the fit reports.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio: the ultimate per unit of exposure, positive:
///     one for every origin, one per origin in origin order (oldest first,
///     over the triangle's origins or, for a latest diagonal, its origins),
///     or a dict from origin (its label as fits report it, ``"2021"`` or
///     ``"2021Q3"``, or an integer year) to its loss ratio. Every origin
///     fitted needs one; a value for an origin the triangle does not have
///     is an error. Over segments each segment takes its own origins'
///     values. A pandas Series is read by its index, as a dict, so
///     ``df.set_index("origin")["elr"]`` works in any row order; one with
///     a default ``RangeIndex`` is an error, since its order cannot be
///     checked. A Polars Series or numpy array is a sequence. On a latest
///     diagonal a sequence needs the rows oldest first, each origin once,
///     and is otherwise an error, since the exposure is read by row.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ExpectedLoss, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = ExpectedLoss(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.ultimate, fit.reserve
/// ([125.0, 200.0], [-25.0, 0.0])
#[pyclass(name = "ExpectedLoss", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExpectedLoss {
    inner: ExpectedLoss,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyExpectedLoss {
    #[new]
    #[pyo3(signature = (apriori = None, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: ExpectedLoss {
                apriori: apriori_arg(apriori)?,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio: a float, a list in origin order or a dict by origin
    /// label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``apriori`` is not positive, or if an
    ///     origin has no observed, finite, positive exposure (the message
    ///     names it and, with keys, its segment).
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``ExpectedLoss`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyExpectedLoss {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<ExpectedLoss> {
        let mut m = self.inner.clone();
        m.chain_ladder = chain_ladder_on(&m.chain_ladder, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        format!(
            "ExpectedLoss(apriori={}, {})",
            apriori_repr(&self.inner.apriori),
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Bornhuetter–Ferguson method: each origin's latest value plus the
/// expected loss ``apriori * exposure`` times the share still to develop,
/// ``1 - 1 / cdf``, as chainladder-python's ``BornhuetterFerguson``.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio: the expected ultimate per unit of exposure,
///     positive: one for every origin, one per origin in origin order
///     (oldest first, over the triangle's origins or, for a latest
///     diagonal, its origins), or a dict from origin (its label as fits
///     report it, ``"2021"`` or ``"2021Q3"``, or an integer year) to its
///     loss ratio. Every origin fitted needs one; a value for an origin the
///     triangle does not have is an error. Over segments each segment takes
///     its own origins' values. A pandas Series is read by its index, as
///     a dict, so ``df.set_index("origin")["elr"]`` works in any row
///     order; one with a default ``RangeIndex`` is an error, since its
///     order cannot be checked. A Polars Series or numpy array is a
///     sequence. On a latest diagonal a sequence needs the rows oldest
///     first, each origin once, and is otherwise an error, since the
///     exposure is read by row.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 266.67]
#[pyclass(name = "BornhuetterFerguson", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBornhuetterFerguson {
    inner: BornhuetterFerguson,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyBornhuetterFerguson {
    #[new]
    #[pyo3(signature = (apriori = None, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: BornhuetterFerguson {
                apriori: apriori_arg(apriori)?,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio: a float, a list in origin order or a dict by origin
    /// label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``BornhuetterFerguson`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyBornhuetterFerguson {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<BornhuetterFerguson> {
        let mut m = self.inner.clone();
        m.chain_ladder = chain_ladder_on(&m.chain_ladder, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        format!(
            "BornhuetterFerguson(apriori={}, {})",
            apriori_repr(&self.inner.apriori),
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Benktander (iterated Bornhuetter–Ferguson) method: starting from
/// ``U(0) = apriori * exposure``, ``U(k) = latest + (1 - 1 / cdf) * U(k-1)``
/// for ``n_iters`` steps, as chainladder-python's ``Benktander``.
/// ``n_iters=0`` is the expected loss method, 1 is Bornhuetter–Ferguson,
/// and many iterations approach the chain ladder. The steps are summed in
/// closed form, so a large ``n_iters`` is cheap; where an origin's ``cdf``
/// is below 1/2 they diverge instead.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio of the starting ultimate, positive: one for
///     every origin, one per origin in origin order (oldest first, over the
///     triangle's origins or, for a latest diagonal, its origins), or a
///     dict from origin (its label as fits report it, ``"2021"`` or
///     ``"2021Q3"``, or an integer year) to its loss ratio. Every origin
///     fitted needs one; a value for an origin the triangle does not have
///     is an error. Over segments each segment takes its own origins'
///     values. A pandas Series is read by its index, as a dict, so
///     ``df.set_index("origin")["elr"]`` works in any row order; one with
///     a default ``RangeIndex`` is an error, since its order cannot be
///     checked. A Polars Series or numpy array is a sequence. On a latest
///     diagonal a sequence needs the rows oldest first, each origin once,
///     and is otherwise an error, since the exposure is read by row.
/// n_iters : int, default 1
///     Number of Bornhuetter–Ferguson steps.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Benktander, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = Benktander(apriori=0.5, n_iters=2).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 288.89]
#[pyclass(name = "Benktander", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBenktander {
    inner: Benktander,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyBenktander {
    #[new]
    #[pyo3(signature = (apriori = None, n_iters = 1, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        n_iters: usize,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: Benktander {
                apriori: apriori_arg(apriori)?,
                n_iters,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio of the starting ultimate: a float, a list in origin
    /// order or a dict by origin label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// Number of Bornhuetter–Ferguson steps.
    #[getter]
    fn n_iters(&self) -> usize {
        self.inner.n_iters
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``Benktander`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori`` and ``n_iters``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyBenktander {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<Benktander> {
        let mut m = self.inner.clone();
        m.chain_ladder = chain_ladder_on(&m.chain_ladder, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        format!(
            "Benktander(apriori={}, n_iters={}, {})",
            apriori_repr(&self.inner.apriori),
            self.inner.n_iters,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Cape Cod (Stanard–Bühlmann) method: Bornhuetter–Ferguson with each
/// origin's apriori estimated from the triangle, as chainladder-python's
/// ``CapeCod``.
///
/// Origin ``j``'s used-up exposure is ``exposure[j] / cdf[j]`` and its
/// latest value is trended to the triangle's valuation by
/// ``(1 + trend) ** (months / 12)``, the months running from the end of
/// the origin period. Origin ``i``'s trended apriori is the sum of the
/// trended latest values weighted by ``decay ** abs(i - j)`` over the same
/// weighted sum of used-up exposures; dividing by its own trend factor
/// gives the apriori of its Bornhuetter–Ferguson ultimate.
///
/// Parameters
/// ----------
/// trend : float, default 0.0
///     Annual trend of the loss ratio; above -1.
/// decay : float, default 1.0
///     Weight of an origin ``n`` periods away, ``decay ** n``; from 0 to 1.
///     With 1 every origin shares one loss ratio.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod().fit(tri, "paid", "premium")
/// >>> [round(a, 4) for a in fit.apriori], [round(u, 2) for u in fit.ultimate]
/// ([0.6774, 0.6774], [150.0, 290.32])
#[pyclass(name = "CapeCod", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCapeCod {
    inner: CapeCod,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyCapeCod {
    #[new]
    #[pyo3(signature = (trend = 0.0, decay = 1.0, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: CapeCod {
                trend,
                decay,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Annual trend of the loss ratio.
    #[getter]
    fn trend(&self) -> f64 {
        self.inner.trend
    }

    /// Weight of an origin one period away.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure and apriori. Trend runs to the triangle's valuation.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``trend`` or ``decay`` is out of
    ///     range, or if an origin has no observed, finite, positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyCapeCodFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyCapeCodFit { inner })
    }

    /// ``CapeCod`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one: its apriori
    /// is estimated on the diagonal and trended to one valuation, the
    /// latest month any origin of any segment is valued at.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``trend`` and ``decay``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCapeCodFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyCapeCodFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyCapeCod {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<CapeCod> {
        let mut m = self.inner.clone();
        m.chain_ladder = chain_ladder_on(&m.chain_ladder, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        format!(
            "CapeCod(trend={:?}, decay={:?}, {})",
            self.inner.trend,
            self.inner.decay,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// A fitted expected-loss method (``ExpectedLoss``,
/// ``BornhuetterFerguson`` or ``Benktander``) of every segment of a
/// triangle column.
///
/// Per-origin lists (``origins``, ``latest``, ``exposure``, ``apriori``,
/// ``ultimate``, ``reserve``) run over the origins of each segment in turn,
/// like the rows of ``to_frame()``. ``ultimate`` and ``reserve`` are this
/// method's; ``chain_ladder`` holds the chain ladder's. Per-age lists need
/// a single-segment fit; for several segments use ``development_frame()``
/// or ``segment(...)``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021] * 2,
/// ...     [12, 24, 12] * 2,
/// ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0],
/// ...      "premium": [250.0, 250.0, 400.0, 500.0, 500.0, 800.0]},
/// ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.exposure, fit.segment(lob="Home").ultimate
/// ([250.0, 400.0, 500.0, 800.0], [20.0, 230.0])
#[pyclass(name = "ExpectedLossFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExpectedLossFit {
    inner: SegmentFits<ExpectedLossFit>,
}

#[pymethods]
impl PyExpectedLossFit {
    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .chain_ladder
            .cdf
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone())
    }

    /// Expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.apriori.clone())
    }

    /// This method's ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ExpectedLossFit::reserves)
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure`` and ``apriori``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpectedLossFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// A fitted Cape Cod of every segment of a triangle column: the fields of
/// ``ExpectedLossFit``, with ``apriori`` the detrended loss ratio applied
/// to each origin (chainladder-python's ``detrended_apriori_``), plus
/// ``trended_apriori`` before detrending (its ``apriori_``).
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod(trend=0.1).fit(tri, "paid", "premium")
/// >>> round(fit.trended_apriori[0] / fit.apriori[0], 10)
/// 1.1
#[pyclass(name = "CapeCodFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCapeCodFit {
    inner: SegmentFits<CapeCodFit>,
}

#[pymethods]
impl PyCapeCodFit {
    /// The expected-loss fit: ultimates, exposures and the detrended
    /// apriori.
    #[getter]
    fn expected_loss(&self) -> PyExpectedLossFit {
        PyExpectedLossFit {
            inner: self.inner.map(|f| f.expected_loss.clone()),
        }
    }

    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.expected_loss.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .expected_loss
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "cdf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.cdf.clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.exposure.clone())
    }

    /// Detrended expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.apriori.clone())
    }

    /// Expected loss ratio per origin at the valuation date's price level,
    /// before detrending. For a treaty year that incepts later, trend it
    /// one more year (or the years from the valuation date to the treaty
    /// year's average date) beyond this value.
    #[getter]
    fn trended_apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.trended_apriori.clone())
    }

    /// Cape Cod ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, CapeCodFit::reserves)
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure``, ``apriori`` and
    /// ``trended_apriori``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CapeCodFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// The growth curve named `name`.
fn growth_curve(name: &str) -> PyResult<GrowthCurve> {
    match name {
        "loglogistic" => Ok(GrowthCurve::LogLogistic),
        "weibull" => Ok(GrowthCurve::Weibull),
        other => Err(PyValueError::new_err(format!(
            "unknown growth curve {other:?}; expected \"loglogistic\" or \"weibull\""
        ))),
    }
}

fn growth_curve_name(curve: GrowthCurve) -> &'static str {
    match curve {
        GrowthCurve::LogLogistic => "loglogistic",
        GrowthCurve::Weibull => "weibull",
    }
}

/// Clark's LDF method (Clark 2003), as R ChainLadder's ``ClarkLDF``: each
/// origin's expected ultimate and a growth curve are fitted to the
/// incremental losses by over-dispersed Poisson maximum likelihood, with
/// ages measured from the average date of loss (the middle of the origin
/// period, R's ``adol = TRUE``).
///
/// The ultimate is the latest value developed by the fitted curve to
/// ``max_age``. Process risk is the scale times the fitted reserve, and
/// parameter risk the delta method on the parameters' covariance, the
/// scale times the inverse Fisher information.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve ``G``: ``x**omega / (x**omega + theta**omega)`` or
///     ``1 - exp(-(x / theta)**omega)``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf(curve="weibull", max_age=120).fit(tri, "values")
/// >>> fit.omega > 0 and fit.total_standard_error > fit.total_process_risk
/// True
/// >>> round(fit.ultimate[2] * fit.growth(36) / fit.growth(120), 6)
/// 410.0
#[pyclass(name = "ClarkLdf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkLdf {
    inner: ClarkLdf,
}

#[pymethods]
impl PyClarkLdf {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkLdf {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// Fits one loss column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if the triangle has fewer than four ages,
    ///     ``max_age`` is before its last age, an origin's latest value is
    ///     not positive, or the likelihood search does not converge.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| m.fit_segments(tri, column)).map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkLdf(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

fn max_age_repr(max_age: Option<f64>) -> String {
    max_age.map_or_else(|| "None".to_string(), |m| format!("{m:?}"))
}

/// Clark's Cape Cod method (Clark 2003), as R ChainLadder's
/// ``ClarkCapeCod``: one expected loss ratio times each origin's exposure
/// and a growth curve are fitted to the incremental losses by
/// over-dispersed Poisson maximum likelihood, with ages measured from the
/// average date of loss.
///
/// The reserve is the fitted ``elr * exposure * (G(max_age) - G(age))``;
/// process and parameter risk are as in ``ClarkLdf``.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve, as in ``ClarkLdf``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkCapeCod, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     {"paid": [v for row in rows for v in row],
/// ...      "premium": [800.0 for row in rows for _ in row]},
/// ... )
/// >>> fit = ClarkCapeCod().fit(tri, "paid", "premium")
/// >>> 0 < fit.elr < 1 and fit.expected_ultimate == [fit.elr * 800.0] * 5
/// True
#[pyclass(name = "ClarkCapeCod", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkCapeCod {
    inner: ClarkCapeCod,
}

#[pymethods]
impl PyClarkCapeCod {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkCapeCod {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed value is its
    ///     exposure.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ClarkLdf.fit``, and if an origin has no observed, finite,
    ///     positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkCapeCod(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

/// A fitted Clark LDF or Cape Cod model of every segment of a triangle
/// column.
///
/// Per-origin lists (``origins``, ``latest``, ``expected_ultimate``,
/// ``ultimate``, ``reserve`` and the standard errors) run over the origins
/// of each segment in turn, like the rows of ``to_frame()``. The fitted
/// parameters and the standard errors of the total need a single-segment
/// fit; for several segments use ``totals_frame()`` or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf().fit(tri, "values")
/// >>> len(fit.covariance), fit.elr, fit.growth(float("inf"))
/// (7, None, 1.0)
#[pyclass(name = "ClarkFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkFit {
    inner: SegmentFits<ClarkFit>,
}

impl PyClarkFit {
    /// The fit of a single-segment result, for `field`.
    fn one(&self, field: &str) -> PyResult<&ClarkFit> {
        single(&self.inner, field, "totals_frame()")
    }
}

#[pymethods]
impl PyClarkFit {
    /// The volume-weighted chain ladder of the same column, with the chain
    /// ladder's ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.fits[0].curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.fits[0].max_age
    }

    /// Fitted shape of the growth curve.
    #[getter]
    fn omega(&self) -> PyResult<f64> {
        Ok(self.one("omega")?.omega)
    }

    /// Fitted scale of the growth curve, in months.
    #[getter]
    fn theta(&self) -> PyResult<f64> {
        Ok(self.one("theta")?.theta)
    }

    /// Expected loss ratio (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn elr(&self) -> PyResult<Option<f64>> {
        if self.inner.fits[0].elr.is_none() {
            return Ok(None);
        }
        Ok(self.one("elr")?.elr)
    }

    /// Length of the origin period in months; ages are shifted by half of
    /// it to the average date of loss.
    #[getter]
    fn origin_width(&self) -> f64 {
        self.inner.fits[0].origin_width
    }

    /// Number of observed incremental values fitted; ``scale`` divides by
    /// this less the number of parameters.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        Ok(single(&self.inner, "n_observations", "segment(...)")?.n_observations)
    }

    /// Over-dispersion ``sigma**2``: squared Pearson residuals over the
    /// observed incremental values less the number of parameters.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale")?.scale)
    }

    /// Covariance of the parameters: the expected ultimates (LDF) or the
    /// expected loss ratio (Cape Cod), then ``omega`` and ``theta``. NaN if
    /// the Fisher information is singular.
    #[getter]
    fn covariance(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(single(&self.inner, "covariance", "segment(...)")?
            .covariance
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn exposure(&self) -> Option<Vec<f64>> {
        self.inner.fits[0].exposure.as_ref()?;
        Some(by_origin(&self.inner, |f| {
            f.exposure.clone().unwrap_or_default()
        }))
    }

    /// Expected ultimate per origin, developed to infinity: fitted (LDF)
    /// or ``elr * exposure`` (Cape Cod).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_ultimate.clone())
    }

    /// Ultimate per origin: the latest value plus the reserve.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ClarkFit::reserves)
    }

    /// Process standard error per origin.
    #[getter]
    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    /// Parameter standard error per origin.
    #[getter]
    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    /// Standard error per origin: ``sqrt(process**2 + parameter**2)``.
    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// Process standard error of the total reserve.
    #[getter]
    fn total_process_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_process_risk")?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, with the covariance
    /// between origins.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_parameter_risk")?.total_parameter_risk)
    }

    /// Standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(self.one("total_standard_error")?.total_standard_error)
    }

    /// Share of the expected ultimate developed by a development age.
    ///
    /// Parameters
    /// ----------
    /// age : float
    ///     Development age in months, before the shift to the average date
    ///     of loss; ``inf`` gives 1.
    ///
    /// Returns
    /// -------
    /// float
    fn growth(&self, age: f64) -> PyResult<f64> {
        Ok(single(&self.inner, "growth", "segment(...)")?.growth(age))
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, (Cape Cod) ``exposure``,
    /// ``expected_ultimate``, ``process_risk``, ``parameter_risk`` and
    /// ``standard_error``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, the ``process_risk``,
    /// ``parameter_risk`` and ``standard_error`` of its total reserve, and
    /// its ``omega``, ``theta``, ``scale`` and (Cape Cod) ``elr``. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        let method = if self.inner.fits[0].elr.is_some() {
            "cape_cod"
        } else {
            "ldf"
        };
        let se = match self.inner.fits.as_slice() {
            [one] => format!(", total_standard_error={:?}", one.total_standard_error),
            _ => String::new(),
        };
        format!(
            "ClarkFit({}method={method:?}, curve={:?}, origins={}, total_reserve={:?}{se})",
            segments_prefix(&self.inner),
            self.curve(),
            self.origins().len(),
            self.inner.total_reserve(),
        )
    }
}

/// Over-dispersed Poisson bootstrap of the chain ladder (England and
/// Verrall 2002), as R ChainLadder's ``BootChainLadder``: adjusted Pearson
/// residuals of the volume-weighted chain ladder are resampled into pseudo
/// triangles, each is re-projected, and process error is added to every
/// future incremental value. Simulation ``i`` uses random stream ``i`` of
/// ``seed`` for every segment in turn, so results do not depend on the
/// number of threads.
///
/// Parameters
/// ----------
/// n_sims : int, default 10000
///     Number of simulations; positive.
/// seed : int, default 0
///     Seed of the simulation streams, from 0 to ``2**64 - 1``. R accepts
///     seeds below ``2**53``; a seed in both ranges gives the same draws.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future incremental value: Gamma with
///     the expected value as mean and variance ``scale * |mean|`` (R's
///     ``process.distr = "gamma"``), or none for parameter error only.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age in the lifetime view (``fit``), as
///     ``ChainLadder``'s; no tail by default, as R's ``BootChainLadder``.
///     An estimated tail (``TailCurve``, ``TailBondy``, ``TailLogLinear``)
///     is refitted on each simulation's pseudo factors, which gives its
///     parameter error; a constant one is fixed unless ``tail_std_err`` is
///     given. The step from the oldest age to ultimate is one more future
///     increment, with mean the pseudo value at the oldest age times the
///     tail factor less 1 and the same process error. ``one_year`` values
///     that step one development period after the oldest age and takes it
///     for the origins whose step falls in the coming year, closing them;
///     ``method``'s tail must then be this one. R's rule
///     (``TailLogLinear``) refitted on the ODP's pseudo factors is often
///     exactly 1 (5.7% of simulations on RAA) and on average high (RAA's
///     oldest origin's mean reserve 78% above the plug-in); a constant tail
///     with ``tail_std_err`` has neither.
/// tail_std_err : float, optional
///     Standard error of a constant tail factor: each simulation draws the
///     factor from the lognormal with the factor as mean and this standard
///     deviation. Not given, the factor is fixed, unlike
///     ``MackBootstrap``, which extrapolates Mack's standard error. Unused
///     when the factor is 1; an error with an estimated tail.
/// dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit`` and ``one_year``: each resamples its own residuals
///     independently; every segment resamples the residuals of the same
///     origins and ages in each simulation (the synchronous bootstrap of
///     Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so
///     the lines' parameter error takes the correlation of their residuals
///     and process error stays independent, which needs the same origins,
///     ages and observed cells in every segment; or each is bootstrapped
///     independently and the segments' simulations are reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, so that row ``i`` is
///     no longer simulation ``i`` but any subset of rows is a fair sample.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, one row and column per segment in index
///     order, symmetric with a unit diagonal. It is converted to the normal
///     scores' correlation ``2 sin(pi rho / 6)``, which must be positive
///     definite.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``. The residuals
///     and the scale come from the triangle's own volume-weighted factors;
///     each simulation then uses ``w * selected + (1 - w) * pseudo`` at
///     the selected ages, so a fully selected factor is the same in every
///     simulation (no parameter error) and a blended one moves by ``1 - w``
///     times the pseudo factor's deviation. The origins' level still moves
///     (each simulation projects from its pseudo latest values), so even a
///     fully selected pattern keeps part of the spread: with every factor
///     selected and no process error, about 73% of the unblended standard
///     deviation on RAA. This is the ODP's parameter error in the origins'
///     levels, which a development pattern does not fix, and it is kept on
///     purpose; ``MackBootstrap`` projects from the observed latest values
///     and keeps only the process error. A supplied pattern lowers the
///     reported uncertainty. With no ``tail``, a cumulative pattern to the
///     oldest age implies a constant tail (``ChainLadderFit.pattern_tail``),
///     the same in every simulation unless ``tail_std_err`` is given. In
///     ``one_year``, a bootstrap without a pattern takes ``method``'s
///     (``bootstrap_ldf``); one with a pattern needs ``method`` to select the
///     same factors, and the tail its pattern implies to be ``method``'s.
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``, for the volume-weighted chain ladder the
///     bootstrap fits (it has no average setting). The fitted values are
///     back-cast with the factors under the exclusions; an incremental cell
///     has a residual only if the link ratio it closes is kept (the first
///     age's cells always do), and the pseudo factors average the kept
///     links only. The drops are fixed on the observed triangle: the
///     simulated triangles are not ranked again. The scale is the sum of
///     the kept squared residuals over ``N - p`` (chainladder-python's
///     rule), which is biased low under exclusions: on RAA with
///     ``n_periods=3`` it is about 70% of the plain scale (694 against
///     984), and the standard deviations are correspondingly narrow. In
///     ``one_year``, a bootstrap that sets none of them takes ``method``'s;
///     one that sets its own must set the same as ``method``, else a
///     ``ValueError`` names both. The mean of the claims development
///     result is then not zero: the window moves and the drops are ranked
///     again on the closing triangle, as the next year's reserving does.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square; if ``n_periods`` is zero.
/// TypeError
///     If ``tail`` is not a number or a tail estimator.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import OdpBootstrap, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = OdpBootstrap(n_sims=2000, seed=42).fit(tri, "values")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.reserves.mean() > 0
/// True
#[pyclass(name = "OdpBootstrap", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpBootstrap {
    inner: OdpBootstrap,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyOdpBootstrap {
    #[new]
    #[pyo3(signature = (
        n_sims = 10_000,
        seed = 0,
        process = "gamma",
        tail = None,
        tail_std_err = None,
        dependence = "independent",
        spearman = None,
        ldf = None,
        ldf_kind = None,
        ldf_weight = None,
        n_periods = None,
        drop_high = 0,
        drop_low = 0,
        ldf_origin = None,
        ldf_underwriting = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_sims: usize,
        seed: u64,
        process: &str,
        tail: Option<&Bound<'_, PyAny>>,
        tail_std_err: Option<f64>,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            ldf_basis,
            inner: OdpBootstrap {
                n_sims,
                seed,
                process: process_distribution(process)?,
                tail: tail_arg(tail)?,
                tail_std_err,
                dependence: segment_dependence(dependence, spearman)?,
                pattern: selected_pattern(ldf, ldf_kind, ldf_weight)?,
                one_year_pattern: OneYearPattern::Method,
                n_periods,
                drop_high,
                drop_low,
            },
        })
    }

    /// Number of simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the simulation streams.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// The lifetime view's tail: a constant factor as a number, otherwise
    /// its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The standard error a constant tail factor is drawn with, or
    /// ``None`` to keep it fixed.
    #[getter]
    fn tail_std_err(&self) -> Option<f64> {
        self.inner.tail_std_err
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.pattern.as_ref())
    }

    /// How the segments depend on each other: ``"independent"``,
    /// ``"synchronized"`` or ``"rank_correlation"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Link ratios that start before the latest ``n_periods`` calendar
    /// diagonals are left out; ``None`` keeps all.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.n_periods
    }

    /// How many of the highest link ratios at each age are left out.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.drop_high
    }

    /// How many of the lowest link ratios at each age are left out.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.drop_low
    }

    /// Bootstraps one measure column in every segment of a cumulative
    /// triangle, each with its own residuals and scale, into one joint
    /// distribution of the reserves, the segments depending on each other
    /// as ``dependence`` says. Every origin must be observed from the first
    /// age up to its latest.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// OdpBootstrapFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, and if an origin has a gap before its
    ///     latest age, a segment has too few observed cells for the
    ///     degrees of freedom to be positive, synchronized segments differ
    ///     in their origins, ages or observed cells, or ``spearman`` does
    ///     not fit the segments; with an estimated ``tail``, if it cannot be
    ///     refitted on some simulation's pseudo factors (the message counts
    ///     them and gives one).
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyOdpBootstrapFit> {
        let tri = &triangle.inner;
        let boot = &self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| boot.fit_segments(tri, column)).map_err(err)?;
        Ok(PyOdpBootstrapFit { inner })
    }

    /// The one-year view of any reserving method: the claims development
    /// result over the coming year, by re-reserving on the bootstrap
    /// ("actuary in the box"). The coming year is every cell valued in the
    /// twelve months after the segment's valuation: one per origin for an
    /// annual development grain, four for a quarterly one (fewer for an
    /// origin that reaches the last age). Each simulation resamples the
    /// residuals for the volume-weighted factors, projects the increments
    /// of those cells in turn from the origin's resampled latest value with
    /// the bootstrap's process error, as ``fit`` projects, adds them to the
    /// observed latest value, appends the cells to the triangle, refits
    /// ``method`` and records ``CDR = opening ultimate - closing
    /// ultimate``; a negative CDR is an adverse development. An origin whose
    /// remaining cells all fall in the year thus has its lifetime bootstrap
    /// reserve as its one-year view; an origin at the last age gets no new
    /// cell; an origin short of the latest diagonal develops from its own
    /// latest cell, and only the year's cells are appended. Unlike
    /// ``MackFit.claims_development_result()`` (Merz and Wüthrich), any
    /// averaging, tail and development grain are allowed.
    ///
    /// Without a bootstrap ``tail``, development past the oldest age moves
    /// only through ``method``'s refitted tail. With one, which must be
    /// ``method``'s tail, the tail is drawn in each simulation as ``fit``
    /// draws it, and its step to ultimate is valued one development period
    /// after the oldest age: an origin whose step falls in the coming year
    /// (with an annual grain, one already at the oldest age) takes it and
    /// is closed at that simulated ultimate. The step is the whole tail: a
    /// curve or a decayed constant tail is taken as the product of its
    /// factors in that one step, so an origin that takes it has its
    /// lifetime tail risk in the coming year, an upper bound on its
    /// one-year risk.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``. Its latest value per origin is kept for
    ///     the end of the year.
    /// bootstrap_ldf : bool, default True
    ///     With a bootstrap that has no ``ldf`` of its own, whether it takes
    ///     ``method``'s selected pattern, so that each simulation's pseudo
    ///     factors are blended with it as the method's factors are (as R's
    ///     ``odp_one_year(bootstrap_ldf = TRUE)`` and the Rust default,
    ///     ``OneYearPattern::Method``). ``False`` simulates from the
    ///     triangle's own factors and lets the pattern enter only when the
    ///     method re-reserves; it is an error with a bootstrap that has its
    ///     own ``ldf``. A bootstrap's own ``ldf`` must select the same factors
    ///     and weights as ``method``'s, in any order or kind; and when it
    ///     implies a tail (a cumulative pattern to the oldest age, no
    ///     ``tail``), ``method`` must use the same tail. The ``ldf`` taken
    ///     from ``method`` is read as ``method`` reads it: with no ``tail``
    ///     on either, a cumulative pattern to the oldest age implies the
    ///     same tail for both; when ``method`` has a ``tail``, to which it
    ///     rescales the pattern's level, the pattern implies no tail for
    ///     the bootstrap either.
    ///
    /// Returns
    /// -------
    /// OneYearFit
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``method`` is not one of the classes above.
    /// ValueError
    ///     As ``fit`` and the method's own ``fit``; if ``exposure`` is
    ///     missing for an expected-loss method or given for
    ///     ``ChainLadder``; if the bootstrap has a ``tail`` that is not the
    ///     one ``method`` uses (its own, or the one its ``ldf`` implies), or
    ///     an ``ldf`` that selects other factors or weights
    ///     than ``method``'s; or if the refit (or the bootstrap's tail refit)
    ///     fails in any simulation (the message counts them and gives one).
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BornhuetterFerguson, OdpBootstrap, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [250.0] * 4 + [260.0] * 3 + [270.0] * 2 + [280.0],
    /// ...     },
    /// ... )
    /// >>> boot = OdpBootstrap(n_sims=2000, seed=42)
    /// >>> fit = boot.one_year(tri, "paid", BornhuetterFerguson(apriori=0.7), exposure="premium")
    /// >>> fit.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> fit.opening_reserve[0]
    /// 0.0
    /// >>> fit.cdr.variance() < boot.fit(tri, "paid").reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, method, exposure = None, bootstrap_ldf = true))]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        method: &Bound<'_, PyAny>,
        exposure: Option<String>,
        bootstrap_ldf: bool,
    ) -> PyResult<PyOneYearFit> {
        let origin = triangle.inner.origin_grain();
        let method = one_year_method(method, exposure, origin)?;
        let mut boot = self.resolved(origin)?;
        boot.one_year_pattern = one_year_pattern(boot.pattern.is_some(), bootstrap_ldf)?;
        let tri = &triangle.inner;
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Odp(inner),
        })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyOdpBootstrap {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<OdpBootstrap> {
        let mut m = self.inner.clone();
        m.pattern = averaged(&m.pattern, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        let std_err = self
            .inner
            .tail_std_err
            .map_or(String::new(), |v| format!(", tail_std_err={v:?}"));
        format!(
            "OdpBootstrap(n_sims={}, seed={}, process={:?}, tail={}{std_err}{}{}{})",
            self.inner.n_sims,
            self.inner.seed,
            self.process(),
            tail_arg_repr(&self.inner.tail),
            dependence_repr(&self.inner.dependence),
            exclusions_arg_repr(
                self.inner.n_periods,
                self.inner.drop_high,
                self.inner.drop_low,
            ),
            pattern_arg_repr(self.inner.pattern.as_ref()),
        )
    }
}

/// A fitted ODP bootstrap of every segment.
///
/// ``reserves`` is one joint distribution with the triangle's keys and
/// ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists run over the origins of
/// each segment in turn, like the rows of ``to_frame()`` and the
/// components of ``reserves``. ``fitted``, ``residuals`` and ``scale``
/// need a single-segment fit; for several segments use ``segment(...)`` or
/// ``totals_frame()``. ``fitted`` and ``residuals`` are nested lists
/// indexed ``[origin][development]``, like one segment of
/// ``Triangle.values``, with ``nan`` where the triangle is not observed.
#[pyclass(name = "OdpBootstrapFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpBootstrapFit {
    inner: OdpBootstrapFits,
}

impl PyOdpBootstrapFit {
    /// A row-major origin x development vector as nested lists.
    fn grid(&self, flat: &[f64]) -> Vec<Vec<f64>> {
        let n_dev = self.development().len();
        flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect()
    }

    fn one(&self, field: &str, instead: &str) -> PyResult<&OdpBootstrapSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyOdpBootstrapFit {
    /// The deterministic volume-weighted chain ladder the bootstrap is
    /// centred on, with the bootstrap's tail.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Fitted incremental values, ``[origin][development]``.
    #[getter]
    fn fitted(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.one("fitted", "")?.fitted))
    }

    /// Adjusted Pearson residuals ``(x - m) / sqrt(|m|) * sqrt(n / (n - p))``,
    /// ``[origin][development]``; ``nan`` where not observed or where the
    /// fitted value is zero.
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.one("residuals", "")?.residuals))
    }

    /// The scale parameter ``phi``: the sum of squared unadjusted residuals
    /// over the degrees of freedom ``n - p``.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    /// Joint distribution of the reserve (the sum of future incremental
    /// values) by segment and origin: the triangle's keys and ``"origin"``
    /// are its dimensions, one component per segment and origin, one row
    /// per simulation. Its ``mean`` and ``quantile`` describe the total
    /// reserve. Columns of ``draw_matrix()`` follow ``origins``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the bootstrapped reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals, the
    /// ``scale``, and the ``mean`` and ``std_dev`` of the segment's
    /// bootstrapped total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The bootstrap of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// OdpBootstrapFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let scale = match self.inner.segments.fits.as_slice() {
            [one] => format!(", scale={:?}", one.scale),
            _ => String::new(),
        };
        format!(
            "OdpBootstrapFit({}origins={}, n_sims={}{scale})",
            segments_prefix(&self.inner.segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// Which factors a bootstrap simulates the one-year view with: with
/// `bootstrap_ldf`, the method's pattern when it has none of its own (the
/// Rust default, as R's); without, the triangle's own factors. `own` is
/// whether the bootstrap has a pattern of its own.
fn one_year_pattern(own: bool, bootstrap_ldf: bool) -> PyResult<OneYearPattern> {
    match (own, bootstrap_ldf) {
        (_, true) => Ok(OneYearPattern::Method),
        (false, false) => Ok(OneYearPattern::Estimated),
        (true, false) => Err(PyValueError::new_err(
            "bootstrap_ldf=False takes a bootstrap without ldf: the bootstrap has its own",
        )),
    }
}

/// The method of ``OdpBootstrap.one_year``: a chain ladder without
/// exposure, or an expected-loss method with its exposure column.
fn one_year_method(
    method: &Bound<'_, PyAny>,
    exposure: Option<String>,
    origin: Grain,
) -> PyResult<OneYearMethod> {
    if let Ok(m) = method.extract::<PyRef<'_, PyChainLadder>>() {
        if exposure.is_some() {
            return Err(PyValueError::new_err(
                "ChainLadder takes no exposure column",
            ));
        }
        return Ok(OneYearMethod::ChainLadder(m.resolved(origin)?));
    }
    let needs_exposure = |name: &str| {
        exposure.clone().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name} needs an exposure column: pass exposure=..."
            ))
        })
    };
    if let Ok(m) = method.extract::<PyRef<'_, PyExpectedLoss>>() {
        return Ok(OneYearMethod::ExpectedLoss(
            m.resolved(origin)?,
            needs_exposure("ExpectedLoss")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBornhuetterFerguson>>() {
        return Ok(OneYearMethod::BornhuetterFerguson(
            m.resolved(origin)?,
            needs_exposure("BornhuetterFerguson")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBenktander>>() {
        return Ok(OneYearMethod::Benktander(
            m.resolved(origin)?,
            needs_exposure("Benktander")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyCapeCod>>() {
        return Ok(OneYearMethod::CapeCod(
            m.resolved(origin)?,
            needs_exposure("CapeCod")?,
        ));
    }
    Err(PyTypeError::new_err(format!(
        "method must be ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or \
         CapeCod, got {}",
        method.repr()?
    )))
}

/// Mack's bootstrap for the lifetime and one-year views (England, Verrall
/// and Wüthrich 2019, Appendix 1): the scaled bias-adjusted residuals of
/// the link ratios are resampled into pseudo factors, and each future
/// cumulative value, to the last age (``fit``) or over the coming year
/// (``one_year``), is drawn from the one before ``C`` (the observed latest
/// value for the first) with mean ``f* C`` and Mack's variance
/// ``sigma**2 * abs(C)**(2 - alpha)``. The lifetime view's standard
/// deviation approximates Mack's analytic standard error. Beside
/// ``OdpBootstrap`` (variance ``scale`` times the mean increment), it gives
/// the one-year view under Mack's process: with the volume-weighted chain
/// ladder and no tail, its standard deviation is
/// ``MackFit.claims_development_result()``'s (Merz and Wüthrich) within
/// Monte Carlo error. The residuals are centred by default
/// (``centre_residuals``), so the mean CDR is Merz and Wüthrich's zero and
/// the lifetime mean reserve the chain ladder's, and EVW's Table 4 expected
/// reserves agree. Uncentred, as EVW's Appendix 1 is written, the pool's
/// non-zero mean biases the pseudo factors: the mean CDR is about -0.2
/// (RAA), -0.04 (GenIns) and +0.18 (ABC) times its standard deviation, the
/// lifetime mean reserve about +17%, +0.7% and -0.8% off the chain
/// ladder's, and the one-year standard deviation up to 1.3% wide on RAA.
/// Simulation ``i`` uses random stream ``i`` of ``seed`` for every segment
/// in turn.
///
/// Parameters
/// ----------
/// n_sims : int, default 10000
///     Number of simulations; positive.
/// seed : int, default 0
///     Seed of the simulation streams, from 0 to ``2**64 - 1``.
/// process : {"gamma", "lognormal", "residuals", "normal", "none"}, default "gamma"
///     Process error on each next cumulative value: Gamma or lognormal
///     (negated for a negative mean) or normal, with Mack's mean and
///     variance; the mean plus a resampled residual times the standard
///     deviation, which carries the residuals' mean and variance; or none
///     for parameter error only.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How Mack's model averages the link ratios (its ``alpha``).
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a sigma behind a single link ratio is filled in.
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``. The bootstrap keeps the link ratios that are
///     kept at the observed triangle (the drops are not ranked again in
///     each simulation): only they have residuals and enter the pseudo
///     factors, so Mack's standard error under the same settings is its
///     standard deviation's reference. In ``one_year``, a bootstrap that
///     sets none of them takes ``method``'s; one that sets its own must set
///     the same as ``method``, else a ``ValueError`` names both. The mean
///     of the claims development result is then not zero: the window moves
///     and the drops are ranked again on the closing triangle, as the next
///     year's reserving does.
/// centre_residuals : bool, default True
///     Subtract the residuals' mean before resampling them, so that the
///     pseudo factors are unbiased, the mean CDR is about zero and the
///     lifetime mean reserve is the chain ladder's (without a tail or with
///     a constant one). ``False`` resamples them uncentred, as EVW's
///     Appendix 1 is written.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age in the lifetime view (``fit``), as
///     ``Mack``'s: one more step, to ultimate, with the process error of
///     the tail's sigma (Mack 1999). An estimated tail (``TailCurve``,
///     ``TailBondy``, ``TailLogLinear``) is refitted on each simulation's
///     pseudo factors, which gives its parameter error; a constant one is
///     drawn from the lognormal with the factor as mean and the tail's
///     standard error. No tail by default. ``one_year`` takes the step for
///     the origins whose step falls in the coming year, as
///     ``OdpBootstrap``'s does; ``method``'s tail must then be this one.
/// tail_sigma : float, optional
///     The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
///     Unused when the tail factor is 1.
/// tail_std_err : float, optional
///     The constant tail factor's standard error (R's ``tail.se``);
///     extrapolated if not given, unlike ``OdpBootstrap``, which then keeps
///     the factor fixed. Unused when the tail factor is 1; an error with an
///     estimated tail.
/// dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit`` and ``one_year``: each resamples its own residuals
///     independently; every segment resamples the link-ratio residuals of
///     the same origins and ages in each simulation (the synchronous bootstrap of
///     Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so
///     the lines' parameter error takes the correlation of their residuals
///     and process error stays independent, which needs the same origins,
///     ages and observed cells in every segment; or each is bootstrapped
///     independently and the segments' simulations are reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, so that row ``i`` is
///     no longer simulation ``i`` but any subset of rows is a fair sample.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, one row and column per segment in index
///     order, symmetric with a unit diagonal. It is converted to the normal
///     scores' correlation ``2 sin(pi rho / 6)``, which must be positive
///     definite.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``. Mack's sigmas
///     and residuals come from the triangle's own estimate; each simulation
///     then uses ``w * selected + (1 - w) * pseudo`` at the selected ages,
///     so a fully selected factor has no parameter error and a blended one
///     ``1 - w`` times the pseudo factor's deviation, as ``Mack`` charges
///     it; the process error is unchanged. A supplied pattern lowers the
///     reported uncertainty. Each origin is projected from its observed
///     latest value, so with every factor selected only the process error
///     is left (unlike ``OdpBootstrap``, which keeps the origins' level
///     error). With no ``tail``, a cumulative pattern to the oldest age
///     implies a constant tail (``ChainLadderFit.pattern_tail``), the same
///     in every simulation unless ``tail_std_err`` is given. In
///     ``one_year``, a bootstrap without a pattern takes ``method``'s
///     (``bootstrap_ldf``); one with a pattern needs ``method`` to select
///     the same factors.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, a setting is unknown, or ``spearman`` is
///     missing for ``"rank_correlation"``, given for another dependence or
///     not square.
/// TypeError
///     If ``tail`` is not a number or a tail estimator.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Mack, MackBootstrap, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = MackBootstrap(n_sims=2000, seed=42).one_year(tri, "values", ChainLadder())
/// >>> fit.model
/// 'mack'
/// >>> fit.cdr.variance() ** 0.5 < Mack().fit(tri, "values").total_standard_error
/// True
#[pyclass(name = "MackBootstrap", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackBootstrap {
    inner: MackBootstrap,
    ldf_basis: LdfBasis,
}

fn mack_process(name: &str) -> PyResult<MackProcess> {
    match name {
        "gamma" => Ok(MackProcess::Gamma),
        "lognormal" => Ok(MackProcess::Lognormal),
        "residuals" => Ok(MackProcess::Residuals),
        "normal" => Ok(MackProcess::Normal),
        "none" => Ok(MackProcess::None),
        _ => Err(PyValueError::new_err(format!(
            "process must be \"gamma\", \"lognormal\", \"residuals\", \"normal\" or \"none\", \
             got {name:?}"
        ))),
    }
}

fn mack_process_name(p: MackProcess) -> &'static str {
    match p {
        MackProcess::Gamma => "gamma",
        MackProcess::Lognormal => "lognormal",
        MackProcess::Residuals => "residuals",
        MackProcess::Normal => "normal",
        MackProcess::None => "none",
    }
}

#[pymethods]
impl PyMackBootstrap {
    #[new]
    #[pyo3(signature = (
        n_sims = 10_000,
        seed = 0,
        process = "gamma",
        average = "volume",
        sigma_interpolation = "log-linear",
        centre_residuals = true,
        tail = None,
        tail_sigma = None,
        tail_std_err = None,
        dependence = "independent",
        spearman = None,
        ldf = None,
        ldf_kind = None,
        ldf_weight = None,
        n_periods = None,
        drop_high = 0,
        drop_low = 0,
        ldf_origin = None,
        ldf_underwriting = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_sims: usize,
        seed: u64,
        process: &str,
        average: &str,
        sigma_interpolation: &str,
        centre_residuals: bool,
        tail: Option<&Bound<'_, PyAny>>,
        tail_sigma: Option<f64>,
        tail_std_err: Option<f64>,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            ldf_basis,
            inner: MackBootstrap {
                n_sims,
                seed,
                process: mack_process(process)?,
                development: development(
                    average,
                    sigma_interpolation,
                    selected_pattern(ldf, ldf_kind, ldf_weight)?,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                centre_residuals,
                tail: tail_arg(tail)?,
                tail_sigma,
                tail_std_err,
                dependence: segment_dependence(dependence, spearman)?,
                one_year_pattern: OneYearPattern::Method,
            },
        })
    }

    /// Number of simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the simulation streams.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Process error: ``"gamma"``, ``"lognormal"``, ``"residuals"``,
    /// ``"normal"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        mack_process_name(self.inner.process)
    }

    /// How Mack's model averages the link ratios.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.development.average)
    }

    /// How a sigma behind a single link ratio is filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.development.drop_low
    }

    /// Whether the residuals are centred before they are resampled.
    #[getter]
    fn centre_residuals(&self) -> bool {
        self.inner.centre_residuals
    }

    /// The lifetime view's tail: a constant factor as a number, otherwise
    /// its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The given tail sigma, or ``None`` to extrapolate it.
    #[getter]
    fn tail_sigma(&self) -> Option<f64> {
        self.inner.tail_sigma
    }

    /// The given standard error of a constant tail factor, or ``None`` to
    /// extrapolate it.
    #[getter]
    fn tail_std_err(&self) -> Option<f64> {
        self.inner.tail_std_err
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.development.pattern.as_ref())
    }

    /// How the segments depend on each other: ``"independent"``,
    /// ``"synchronized"`` or ``"rank_correlation"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// The lifetime view: bootstraps one measure column in every segment of
    /// a cumulative triangle, each with its own Mack model and residuals,
    /// into one joint distribution of the reserves (EVW's Appendix 1), the
    /// segments depending on each other as ``dependence`` says. Each
    /// simulation resamples the residuals into pseudo factors and draws
    /// every cumulative value from the latest observed one to the last age,
    /// each from the one before, with Mack's mean and variance; an origin's
    /// reserve is its last drawn value less its latest. With a ``tail``
    /// each origin takes one more step, to ultimate, with the tail's sigma,
    /// the tail factor refitted (an estimated tail) or drawn (a constant
    /// one) per simulation. The standard deviation approximates Mack's
    /// analytic standard error (``Mack.fit``, with the same tail; for an
    /// estimated tail the parameter error is the refit's, not Mack's
    /// extrapolated one); the mean is the chain ladder's reserve with
    /// ``centre_residuals`` (the default), without a tail or with a
    /// constant one, about 17% above it on RAA without centring.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain, every origin observed from the first age to its latest
    ///     with no negative value.
    /// column : str
    ///
    /// Returns
    /// -------
    /// MackBootstrapFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``Mack.fit``, if an origin has a gap before its latest age, if
    ///     a cumulative value is negative, if synchronized segments differ in
    ///     their origins, ages or observed cells, if ``spearman`` does not
    ///     fit the segments, if ``tail_std_err`` is given with an estimated
    ///     tail, or if the tail cannot be refitted on some simulation's
    ///     pseudo factors (the message counts them and gives one).
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import MackBootstrap, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ... )
    /// >>> fit = MackBootstrap(n_sims=2000, seed=42).fit(tri, "values")
    /// >>> fit.reserves.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> abs(fit.reserves.mean() / fit.chain_ladder.total_reserve - 1) < 0.05
    /// True
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyMackBootstrapFit> {
        let tri = &triangle.inner;
        let boot = &self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| boot.fit_segments(tri, column)).map_err(err)?;
        Ok(PyMackBootstrapFit { inner })
    }

    /// The one-year view of any reserving method under Mack's process, as
    /// ``OdpBootstrap.one_year``: each simulation draws the cells of the
    /// coming twelve months from Mack's bootstrap, each from the one before
    /// with the same pseudo factors, appends them to the triangle, refits
    /// ``method`` and records ``CDR = opening ultimate - closing
    /// ultimate``. Without a bootstrap ``tail``, development past the
    /// oldest age moves only through ``method``'s refitted tail; with one,
    /// which must be ``method``'s, the origins whose tail step falls in
    /// the coming year take Mack's step to ultimate and are closed, as in
    /// ``OdpBootstrap.one_year``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain, every origin observed from the first age to its latest
    ///     with no negative value.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``.
    /// bootstrap_ldf : bool, default True
    ///     As ``OdpBootstrap.one_year``: without an ``ldf`` of its own, the
    ///     bootstrap takes ``method``'s selected pattern and reads it as
    ///     ``method`` does (no implied tail when ``method`` has a ``tail``);
    ///     ``False`` simulates from the triangle's own factors.
    ///
    /// Returns
    /// -------
    /// OneYearFit
    ///     With ``model == "mack"``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``method`` is not one of the classes above.
    /// ValueError
    ///     As ``OdpBootstrap.one_year`` and ``Mack.fit``, if a cumulative
    ///     value is negative, and if the bootstrap has a ``tail`` that is
    ///     not ``method``'s.
    #[pyo3(signature = (triangle, column, method, exposure = None, bootstrap_ldf = true))]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        method: &Bound<'_, PyAny>,
        exposure: Option<String>,
        bootstrap_ldf: bool,
    ) -> PyResult<PyOneYearFit> {
        let origin = triangle.inner.origin_grain();
        let method = one_year_method(method, exposure, origin)?;
        let mut boot = self.resolved(origin)?;
        boot.one_year_pattern =
            one_year_pattern(boot.development.pattern.is_some(), bootstrap_ldf)?;
        let tri = &triangle.inner;
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Mack(inner),
        })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyMackBootstrap {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<MackBootstrap> {
        let mut m = self.inner.clone();
        m.development.pattern = averaged(&m.development.pattern, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        let given =
            |name: &str, v: Option<f64>| v.map_or(String::new(), |v| format!(", {name}={v:?}"));
        format!(
            "MackBootstrap(n_sims={}, seed={}, process={:?}, average={:?}, \
             sigma_interpolation={:?}, centre_residuals={}, tail={}{}{}{}{})",
            self.inner.n_sims,
            self.inner.seed,
            self.process(),
            self.average(),
            self.sigma_interpolation(),
            if self.inner.centre_residuals {
                "True"
            } else {
                "False"
            },
            tail_arg_repr(&self.inner.tail),
            given("tail_sigma", self.inner.tail_sigma),
            given("tail_std_err", self.inner.tail_std_err),
            dependence_repr(&self.inner.dependence),
            development_arg_repr(&self.inner.development),
        )
    }
}

/// A fitted bootstrap of Mack's model, the lifetime view, of every
/// segment (``MackBootstrap.fit``).
///
/// ``reserves`` is one joint distribution with the triangle's keys and
/// ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists run over the origins of
/// each segment in turn, like the rows of ``to_frame()`` and the
/// components of ``reserves``. ``residuals`` needs a single-segment fit;
/// for several segments use ``segment(...)``.
#[pyclass(name = "MackBootstrapFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackBootstrapFit {
    inner: MackBootstrapFits,
}

#[pymethods]
impl PyMackBootstrapFit {
    /// The chain ladder of Mack's model (its averaging), with the
    /// bootstrap's tail: the reserves the bootstrap's mean equals with
    /// ``centre_residuals=True``, the default, without a tail or with a
    /// constant one (an estimated tail's refit has its own mean).
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.mack.chain_ladder.clone()),
        }
    }

    /// Mack's model on the observed triangle, with the bootstrap's tail:
    /// the factors and sigmas the simulation uses, and the analytic
    /// standard errors the simulated standard deviations approximate (with
    /// an estimated tail, up to the refit's own parameter error).
    #[getter]
    fn mack(&self) -> PyMackFit {
        PyMackFit {
            inner: self.inner.segments.map(|s| s.mack.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .mack
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Mack's scaled bias-adjusted residuals of the link ratios,
    /// ``[origin][development]``, ``[o][k]`` the link from age ``k`` to
    /// ``k + 1``; ``nan`` where there is none, from a zero, or behind a
    /// factor with a single link ratio or a zero sigma. Never centred.
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        let flat = &single(&self.inner.segments, "residuals", "")?.residuals;
        let n_dev = self.development().len();
        Ok(flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect())
    }

    /// Joint distribution of the reserve (each origin's last simulated
    /// cumulative value less its latest) by segment and origin: the
    /// triangle's keys and ``"origin"`` are its dimensions, one component
    /// per segment and origin, one row per simulation. Its ``mean`` and
    /// ``quantile`` describe the total reserve. Columns of
    /// ``draw_matrix()`` follow ``origins``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the bootstrapped reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals, and
    /// the ``mean`` and ``std_dev`` of the segment's bootstrapped total
    /// reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The bootstrap of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// MackBootstrapFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MackBootstrapFit({}origins={}, n_sims={})",
            segments_prefix(&self.inner.segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// A one-year view from either bootstrap.
enum OneYear {
    Odp(OneYearFits),
    Mack(OneYearFits<MackBootstrapSegment>),
}

/// `$body` on whichever one-year view `$fit` holds, bound to `$f`.
macro_rules! each_one_year {
    ($fit:expr, $f:ident => $body:expr) => {
        match $fit {
            OneYear::Odp($f) => $body,
            OneYear::Mack($f) => $body,
        }
    };
}

/// The simulated one-year view of every segment, from
/// ``OdpBootstrap.one_year`` or ``MackBootstrap.one_year`` (``model``
/// says which).
///
/// ``cdr`` is one joint distribution of the claims development result
/// with the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments, and
/// ``cdr.quantile(0.005)`` is minus the one-year value at risk at 99.5%.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``cdr``. ``fitted``,
/// ``residuals`` and ``scale`` need a single-segment fit; for several
/// segments use ``segment(...)`` or ``totals_frame()``. ``fitted`` and
/// ``scale`` are the ODP bootstrap's (``OdpBootstrapFit``'s), ``mack`` is
/// Mack's bootstrap's model (a ``MackFit`` of every segment, as
/// ``chain_ladder``), and ``residuals`` are either's.
#[pyclass(name = "OneYearFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOneYearFit {
    inner: OneYear,
}

impl PyOneYearFit {
    /// A row-major origin x development vector as nested lists.
    fn grid(&self, flat: &[f64]) -> Vec<Vec<f64>> {
        let n_dev = self.development().len();
        flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect()
    }

    /// The ODP bootstrap of a single-segment ODP fit, for `field`.
    fn odp(&self, field: &str, instead: &str) -> PyResult<&OdpBootstrapSegment> {
        match &self.inner {
            OneYear::Odp(f) => Ok(&single(&f.segments, field, instead)?.bootstrap),
            OneYear::Mack(_) => Err(PyValueError::new_err(format!(
                "{field} is the ODP bootstrap's; this one-year view is Mack's"
            ))),
        }
    }
}

#[pymethods]
impl PyOneYearFit {
    /// The bootstrap's model: ``"odp"`` or ``"mack"``.
    #[getter]
    fn model(&self) -> &'static str {
        match self.inner {
            OneYear::Odp(_) => "odp",
            OneYear::Mack(_) => "mack",
        }
    }

    /// The bootstrap's chain ladder: the factors the simulated next cells
    /// develop with (volume-weighted for the ODP, Mack's averaging for
    /// Mack's).
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: each_one_year!(&self.inner, f => f
                .segments
                .map(|s| s.bootstrap.chain_ladder().clone())),
        }
    }

    /// Mack's model behind ``MackBootstrap.one_year``, with its lifetime
    /// standard errors and ``claims_development_result()``.
    #[getter]
    fn mack(&self) -> PyResult<PyMackFit> {
        match &self.inner {
            OneYear::Mack(f) => Ok(PyMackFit {
                inner: f.segments.map(|s| s.bootstrap.mack.clone()),
            }),
            OneYear::Odp(_) => Err(PyValueError::new_err(
                "mack is Mack's bootstrap's model; this one-year view is the ODP's",
            )),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        each_one_year!(&self.inner, f => f.segments.key_names.clone())
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        each_one_year!(&self.inner, f => segment_labels(py, &f.segments))
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        each_one_year!(&self.inner, f => origin_labels(&f.segments))
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| {
            s.bootstrap.chain_ladder().latest.clone()
        }))
    }

    /// The method's ultimate per origin on the observed triangle.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_ultimate.clone()))
    }

    /// The opening ultimate less the latest value, per origin.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_reserve.clone()))
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        each_one_year!(&self.inner, f => f.segments.fits[0]
            .bootstrap
            .chain_ladder()
            .development
            .development
            .clone())
    }

    /// The ODP bootstrap's fitted incremental values,
    /// ``[origin][development]``.
    #[getter]
    fn fitted(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.odp("fitted", "")?.fitted))
    }

    /// The residuals the bootstrap resamples, ``[origin][development]``:
    /// the ODP's adjusted Pearson residuals, as
    /// ``OdpBootstrapFit.residuals``, or Mack's scaled bias-adjusted
    /// residuals of the link ratios, ``[o][k]`` the link from age ``k`` to
    /// ``k + 1`` (``nan`` where there is none, from a zero, or behind a
    /// factor with a single link ratio).
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        let flat = match &self.inner {
            OneYear::Odp(f) => &single(&f.segments, "residuals", "")?.bootstrap.residuals,
            OneYear::Mack(f) => &single(&f.segments, "residuals", "")?.bootstrap.residuals,
        };
        Ok(self.grid(flat))
    }

    /// The ODP bootstrap's scale parameter ``phi``.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.odp("scale", "totals_frame()")?.scale)
    }

    /// Joint distribution of the claims development result (opening less
    /// closing ultimate) by segment and origin: the triangle's keys and
    /// ``"origin"`` are its dimensions, one component per segment and
    /// origin, one row per simulation. Its ``mean``, ``variance`` and
    /// ``quantile`` describe the total. Columns of ``draw_matrix()`` follow
    /// ``origins``.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, the method's ``opening_ultimate`` and
    /// ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// simulated claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_one_year!(&self.inner, f => f.to_long()))
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the ODP bootstrap's ``scale`` (not for Mack's), and the
    /// ``cdr_mean`` and ``cdr_std_dev`` of the segment's total claims
    /// development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_one_year!(&self.inner, f => f.totals()))
    }

    /// The bootstrap's chain ladders' development factors, one row per
    /// segment and age, as ``ChainLadderFit.development_frame``. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_one_year!(&self.inner, f => f.segments.development_table()),
        )
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// OneYearFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let inner = match &self.inner {
            OneYear::Odp(f) => OneYear::Odp(f.segment(&keys).map_err(err)?),
            OneYear::Mack(f) => OneYear::Mack(f.segment(&keys).map_err(err)?),
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let (prefix, n_sims) = each_one_year!(&self.inner, f => (
            segments_prefix(&f.segments),
            f.cdr.n_sims(),
        ));
        let detail = match &self.inner {
            OneYear::Odp(f) => match f.segments.fits.as_slice() {
                [one] => format!(", scale={:?}", one.bootstrap.scale),
                _ => String::new(),
            },
            OneYear::Mack(_) => ", model=\"mack\"".to_string(),
        };
        format!(
            "OneYearFit({prefix}origins={}, n_sims={n_sims}{detail})",
            self.origins().len(),
        )
    }
}

/// The Bayesian chain ladder of England and Verrall (2002, 2006): the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``exp(c + a[w] + b[d])`` (``a[1] = b[1] = 0``),
/// normal priors on the log scale, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) plugged in. The posterior is sampled
/// with NUTS, and each draw projects the future cells up to the oldest
/// observed age (no tail) with the process error of ``process``, as
/// ``OdpBootstrap`` does. Every origin's and every development column's
/// observed increments must sum to more than zero; negative cells are
/// allowed.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``BayesOdpFit.diagnostics``.
///
/// Parameters
/// ----------
/// intercept_sd : float, default 10.0
///     Standard deviation of the normal prior on the intercept ``c``,
///     centred on the log of the mean observed increment.
/// origin_sd : float, default 10.0
///     Standard deviation of the normal prior on each origin effect
///     ``a[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each development effect
///     ``b[d]``, ``d >= 2``, centred on 0.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; above
///     NUTS's usual 0.8, which on triangles such as RAA usually leaves a
///     few divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it):
///     each segment's own posterior and simulations, with seeds derived
///     from ``seed`` and the segment's position so that the segments share
///     no random numbers; or the segments' simulations reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, as
///     ``OdpBootstrap``'s. ``"synchronized"`` is accepted here, as the
///     bootstraps take it, but ``fit_segments`` raises: it resamples the
///     residuals of the same cells in every segment, and a posterior has no
///     residuals to share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesOdp, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = BayesOdp(n_sims=2000, seed=1).fit(tri, "values")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.reserves.mean() > 0
/// True
/// >>> fit.parameter_names[:2]
/// ['c', 'a[2]']
#[pyclass(name = "BayesOdp", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdp {
    inner: BayesOdp,
}

#[pymethods]
impl PyBayesOdp {
    #[new]
    #[pyo3(signature = (
        intercept_sd = 10.0,
        origin_sd = 10.0,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesOdp {
                intercept_sd,
                origin_sd,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// Standard deviation of the intercept's prior.
    #[getter]
    fn intercept_sd(&self) -> f64 {
        self.inner.intercept_sd
    }

    /// Standard deviation of each origin effect's prior.
    #[getter]
    fn origin_sd(&self) -> f64 {
        self.inner.origin_sd
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How segments depend on each other in ``fit_segments``:
    /// ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of one measure column of a single-segment
    /// cumulative triangle and simulates its reserves. Every origin must be
    /// observed from the first age up to its latest, there must be more
    /// observed cells than parameters, and every origin's and every
    /// development column's increments must sum to more than zero. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     a prior standard deviation is not positive and finite or a
    ///     sampler setting is invalid; or if an origin's or a development
    ///     column's increments do not sum to more than zero.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column)).map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.diagnostics.warnings)?;
        Ok(PyBayesOdpFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently), into one joint
    /// distribution of the reserves, the segments depending on each other
    /// as ``dependence`` says. Segment ``k`` (in index order) takes seeds
    /// derived from ``seed`` and ``k`` for its chains and its simulations,
    /// so independent segments share no random numbers, and a segment's
    /// draws differ from ``fit`` of that segment alone. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment,
    /// each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = BayesOdp(tune=300, draws=300, n_sims=1000).fit_segments(tri, "paid")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").scale > 0
    /// True
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit_segments(tri, column)).map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        Ok(PyBayesOdpFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 16 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the year's increments from
    /// it with ``process`` error (the cells valued in the next twelve
    /// months, as in ``OdpBootstrap.one_year``), reweights every posterior
    /// draw by the year's quasi likelihood with ``scale`` and the priors
    /// unchanged, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is
    ///
    /// ``CDR[w] = E[R[w] | D0] - (X[w] + E[R'[w] | D0, X])``
    ///
    /// with ``R`` the expected run-off at the opening, ``X`` the year's
    /// payments and ``R'`` the expected run-off after the year, each
    /// expectation the (weighted) mean over the posterior draws; its mean
    /// is about zero. A simulation whose ``k`` is above ``k_threshold`` is
    /// flagged. With ``refit="moment_match_then_refit"`` (the default) its
    /// posterior draws are first moved by affine maps to the moments the
    /// weights estimate (moment matching, as R's
    /// ``loo::loo_moment_match``), and only a simulation whose ``k`` stays
    /// above ``k_threshold`` is refitted with NUTS on the triangle extended
    /// by its year, with this model's sampler. 14% to 26% of the
    /// simulations are flagged on RAA and GenIns; matching rescues nearly
    /// all of them, within 0.04 to 0.06 sd of their refits, at about 1 ms
    /// each against about 26 ms for a refit, so a default run of 10,000
    /// simulations takes a few seconds in a release build.
    /// ``"refit_flagged"`` refits every flagged simulation (one to two
    /// minutes there), and ``"reweight_only"`` takes about a second and
    /// leaves the flagged simulations' reweighting as it is. Every cell up to the end of the
    /// year is observed and conditioned on, also for an origin that lags
    /// the valuation. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted (the threshold of Vehtari et al. 2024 for a few
    ///     thousand draws).
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation: moment-matched, and
    ///     refitted only if its ``k`` stays above ``k_threshold``
    ///     (``moment_matched`` lists the rescued); refitted with NUTS
    ///     (``"refit_flagged"``); or reweighted anyway (fast, biased where
    ///     ``k`` is high); ``"refit_all"`` refits every simulation, to
    ///     check the reweighting.
    /// max_refits : int, optional
    ///     The most simulations refitted with NUTS: the first this many due
    ///     for a refit, in simulation order; the others are reweighted as
    ///     they are (as unreliable as their ``k`` says), listed in
    ///     ``unrefitted`` and warned about. ``None`` (the default) refits
    ///     every one, ``0`` none.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ... )
    /// >>> view = BayesOdp(n_sims=1000, seed=1).one_year(tri, "values", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> len(view.pareto_k), view.refitted
    /// (1000, [])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesOdp",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Odp(inner),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle: each segment's posterior and simulations as in
    /// ``fit_segments`` (segment ``k``'s own seeds), reweighted and
    /// refitted as in ``one_year``, into one joint claims development
    /// result, the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results, as the bootstraps'
    /// one-year view does. Raises one ``UserWarning`` listing the failed
    /// diagnostics of every segment's fit, and another if any refit failed
    /// its diagnostics, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    /// k_threshold : float, default 0.7
    ///     As in ``one_year``.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     As in ``one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``, for each segment: ``k`` segments
    ///     refit at most ``k`` times this many.
    /// n_sims : int, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> model = BayesOdp(tune=300, draws=300, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(tri, "paid", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> len(views.segment(lob="Home").pareto_k)
    /// 500
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        warn_refits(py, "BayesOdp", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Odp(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesOdp(intercept_sd={:?}, origin_sd={:?}, development_sd={:?}, process={:?}, \
             chains={}, tune={}, draws={}, n_sims={}, seed={}{})",
            self.inner.intercept_sd,
            self.inner.origin_sd,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            dependence_repr(&self.inner.dependence),
        )
    }
}

/// A fitted ``BayesOdp``: the posterior of ``c``, ``a[2..n]`` and
/// ``b[2..m]``, its MCMC diagnostics, and the posterior predictive
/// distribution of the reserves, laid out as ``OdpBootstrapFit.reserves``.
/// The frames' ``latest``, ``ultimate`` and ``reserve`` are the
/// volume-weighted chain ladder's (``chain_ladder``); ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesOdpFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFit {
    inner: BayesOdpFit,
}

#[pymethods]
impl PyBayesOdpFit {
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale (the
    /// ODP bootstrap's).
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``c``, ``a[2]`` to ``a[n]``, ``b[2]`` to ``b[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is the normalised over-dispersed Poisson density at the cell's
    /// mean and ``scale``, so compare the result only with models of the
    /// same incremental values.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, and the ``mean`` and ``std_dev`` of
    /// the posterior predictive reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the ``scale``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesOdpFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// One ``UserWarning`` from `model` listing failed diagnostics, if any.
fn warn_diagnostics(py: Python<'_>, model: &str, warnings: &[String]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let message = format!(
        "{model}: the MCMC diagnostics failed; check them before relying on the fit: {}",
        warnings.join("; ")
    );
    let message =
        std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &message, 1)
}

/// MCMC diagnostics as a dict, as ``BayesOdpFit.diagnostics``.
fn diagnostics_dict<'py>(py: Python<'py>, d: &Diagnostics) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("max_rhat", d.max_rhat)?;
    out.set_item("min_ess_bulk", d.min_ess_bulk)?;
    out.set_item("min_ess_tail", d.min_ess_tail)?;
    out.set_item("divergences", d.divergences)?;
    out.set_item("warnings", d.warnings.clone())?;
    Ok(out)
}

/// Every segment's failed diagnostics, each message prefixed with its
/// segment's label when the triangle has keys.
fn segment_warnings<T>(segments: &SegmentFits<T>, diag: fn(&T) -> &Diagnostics) -> Vec<String> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .flat_map(|(label, fit)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            diag(fit)
                .warnings
                .iter()
                .map(move |w| format!("{prefix}{w}"))
        })
        .collect()
}

/// One row per segment: the key columns, then ``max_rhat``,
/// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and ``warnings``
/// (the segment's messages joined by ``"; "``).
fn segment_diagnostics_frame<'py, T>(
    py: Python<'py>,
    segments: &SegmentFits<T>,
    diag: fn(&T) -> &Diagnostics,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    for (k, name) in segments.key_names.iter().enumerate() {
        let values: Vec<String> = segments
            .labels
            .iter()
            .map(|l| l.parts().get(k).cloned().unwrap_or_default())
            .collect();
        out.set_item(name, values)?;
    }
    let col = |g: fn(&Diagnostics) -> f64| {
        segments
            .fits
            .iter()
            .map(|f| g(diag(f)))
            .collect::<Vec<f64>>()
    };
    out.set_item("max_rhat", col(|d| d.max_rhat))?;
    out.set_item("min_ess_bulk", col(|d| d.min_ess_bulk))?;
    out.set_item("min_ess_tail", col(|d| d.min_ess_tail))?;
    out.set_item(
        "divergences",
        segments
            .fits
            .iter()
            .map(|f| diag(f).divergences)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "warnings",
        segments
            .fits
            .iter()
            .map(|f| diag(f).warnings.join("; "))
            .collect::<Vec<_>>(),
    )?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// A posterior summary as a pandas DataFrame, as ``BayesOdpFit.summary``.
fn summary_frame<'py>(py: Python<'py>, s: &[ParamSummary]) -> PyResult<Bound<'py, PyAny>> {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    let out = PyDict::new(py);
    out.set_item(
        "parameter",
        s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item("mean", col(|p| p.mean))?;
    out.set_item("sd", col(|p| p.sd))?;
    out.set_item("q05", col(|p| p.q05))?;
    out.set_item("q50", col(|p| p.q50))?;
    out.set_item("q95", col(|p| p.q95))?;
    out.set_item("rhat", col(|p| p.rhat))?;
    out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
    out.set_item("ess_tail", col(|p| p.ess_tail))?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// ``BayesOdp.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distribution of the reserves, laid out as
/// ``OdpBootstrapFit``.
///
/// ``reserves`` has the triangle's keys and ``"origin"`` as dimensions, so
/// ``reserves.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``reserves``. The
/// posterior of one segment (``scale``, ``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()`` and
/// ``loo()``) needs a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales) or
/// ``diagnostics_frame()``.
#[pyclass(name = "BayesOdpFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFits {
    inner: BayesOdpFits,
}

impl PyBayesOdpFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&BayesOdpSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyBayesOdpFits {
    /// The volume-weighted chain ladder of every segment, whose latest
    /// values and Pearson scales the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// The plugged-in scale ``phi`` of a single-segment fit.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the triangle's keys and ``"origin"`` are its dimensions,
    /// one component per segment and origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``BayesOdpFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``BayesOdpFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// ``BayesOdpFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments, or an observed increment is
    ///     negative.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = self.one("loo()", "")?;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the ``scale``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let scale = match segments.fits.as_slice() {
            [one] => format!(", scale={:?}", one.scale),
            _ => String::new(),
        };
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "BayesOdpFits({}origins={}, n_sims={}{scale}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name (`None` for the model's one-year default), the refit
/// budget and the number of simulations (`None` for the model's
/// one-year default).
fn one_year_settings(
    k_threshold: f64,
    refit: Option<&str>,
    max_refits: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<BayesOneYear> {
    let refit = match refit {
        None => None,
        Some("refit_flagged") => Some(OneYearRefit::RefitFlagged),
        Some("reweight_only") => Some(OneYearRefit::ReweightOnly),
        Some("refit_all") => Some(OneYearRefit::RefitAll),
        Some("moment_match_then_refit") => Some(OneYearRefit::MomentMatchThenRefit),
        Some(refit) => {
            return Err(PyValueError::new_err(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got {refit:?}"
            )));
        }
    };
    Ok(BayesOneYear {
        k_threshold,
        refit,
        max_refits,
        n_sims,
        light_refit: None,
        no_light_refit: false,
    })
}

/// A compartmental one-year view's ``light_refit``: ``None`` for the
/// model's default (model 2's light refit), ``False`` to turn it off,
/// ``True`` for [`LightRefit::default`] (4 chains of 150 warm-up and 250 kept
/// draws), or ``(chains, tune, draws)``. Returns the sampler and whether the
/// default is turned off.
fn light_refit_arg(light_refit: Option<&Bound<'_, PyAny>>) -> PyResult<(Option<LightRefit>, bool)> {
    let Some(arg) = light_refit else {
        return Ok((None, false));
    };
    if arg.is_none() {
        return Ok((None, false));
    }
    if let Ok(flag) = arg.cast::<pyo3::types::PyBool>() {
        return Ok(if flag.is_true() {
            (Some(LightRefit::default()), false)
        } else {
            (None, true)
        });
    }
    let message = "light_refit must be None, a bool or a tuple (chains, tune, draws) of                    positive integers";
    let (chains, tune, draws): (usize, usize, usize) =
        arg.extract().map_err(|_| PyValueError::new_err(message))?;
    Ok((
        Some(LightRefit {
            chains,
            tune,
            draws,
        }),
        false,
    ))
}

/// One ``UserWarning`` from `model` if any NUTS refit of a one-year view
/// failed its diagnostics, and another if any simulation due for a refit
/// was left unrefitted by ``max_refits``: `refits` holds each segment's
/// label prefix, its number of refits, how many of them warned and how
/// many simulations were left unrefitted.
fn warn_refits(
    py: Python<'_>,
    model: &str,
    refits: &[(String, usize, usize, usize)],
) -> PyResult<()> {
    let warn = |message: String| -> PyResult<()> {
        let message =
            std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, category.as_any(), &message, 1)
    };
    let failed: Vec<String> = refits
        .iter()
        .filter(|(_, _, warned, _)| *warned > 0)
        .map(|(prefix, n, warned, _)| format!("{prefix}{warned} of {n} refits"))
        .collect();
    if !failed.is_empty() {
        warn(format!(
            "{model}: the MCMC diagnostics of NUTS refits in the one-year view failed (R-hat, \
             ESS or a divergence): {}",
            failed.join("; ")
        ))?;
    }
    let left: Vec<String> = refits
        .iter()
        .filter(|(_, _, _, left)| *left > 0)
        .map(|(prefix, _, _, left)| format!("{prefix}{left}"))
        .collect();
    if !left.is_empty() {
        warn(format!(
            "{model}: simulations due for a refit beyond max_refits were reweighted instead, \
             their reweighting as unreliable as their Pareto k (listed in unrefitted): {}",
            left.join("; ")
        ))?;
    }
    Ok(())
}

/// The one-year view of a Bayesian model on a single-segment triangle,
/// from ``BayesOdp.one_year``, ``BayesBf.one_year``,
/// ``BayesBfNormalised.one_year`` or ``Meyers.one_year``: the lifetime fit,
/// the opening reserve, the claims development result over the coming
/// year and how reliable each simulation's importance reweighting was.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``), the opening reserve less the year's
/// payment and the posterior mean reserve after the year, so a negative
/// value is an adverse development and ``cdr.quantile(0.005)`` is minus
/// the one-year value at risk at 99.5%. ``pareto_k`` has one Pareto ``k``
/// per simulation; ``flagged_share`` is the share above ``k_threshold``,
/// whether or not they were refitted, ``refitted`` lists the simulations
/// refitted with NUTS and ``moment_matched`` those moment matching
/// rescued (``refit="moment_match_then_refit"``, the default). ``fit`` is
/// the model's own lifetime fit, a ``BayesOdpFit``, a ``BayesBfFit``, a
/// ``BayesBfNormalisedFit`` or a ``MeyersFit``.
#[pyclass(name = "BayesOneYearFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFit {
    inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
enum BayesOneYearOf {
    Odp(BayesOneYearFit<BayesOdpFit>),
    Bf(BayesOneYearFit<BayesBfFit>),
    BfNormalised(BayesOneYearFit<BayesBfNormalisedFit>),
    Meyers(Box<BayesOneYearFit<MeyersFit>>),
}

/// Evaluates `$body` with `$f` bound to the view, whatever its model.
macro_rules! each_bayes_one_year {
    ($view:expr, $f:ident => $body:expr) => {
        match $view {
            BayesOneYearOf::Odp($f) => $body,
            BayesOneYearOf::Bf($f) => $body,
            BayesOneYearOf::BfNormalised($f) => $body,
            BayesOneYearOf::Meyers($f) => $body,
        }
    };
}

impl PyBayesOneYearFit {
    /// The lifetime fit's chain ladder.
    fn chain_ladder_fit(&self) -> &ChainLadderFit {
        each_bayes_one_year!(&self.inner, f => f.fit.chain_ladder())
    }
}

#[pymethods]
impl PyBayesOneYearFit {
    /// The model fitted to the observed triangle, with its posterior and
    /// lifetime reserves, as the model's ``fit`` gives it: a
    /// ``BayesOdpFit`` from ``BayesOdp.one_year``, a ``BayesBfFit`` from
    /// ``BayesBf.one_year``, a ``BayesBfNormalisedFit`` from
    /// ``BayesBfNormalised.one_year``, a ``MeyersFit`` from
    /// ``Meyers.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOdpFit, BayesBfFit, BayesBfNormalisedFit or MeyersFit
    #[getter]
    fn fit(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.inner {
            BayesOneYearOf::Odp(f) => Py::new(
                py,
                PyBayesOdpFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Bf(f) => Py::new(
                py,
                PyBayesBfFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::BfNormalised(f) => Py::new(
                py,
                PyBayesBfNormalisedFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Meyers(f) => Py::new(
                py,
                PyMeyersFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
        })
    }

    /// The model's volume-weighted chain ladder.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.chain_ladder_fit().clone()],
            },
        }
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.chain_ladder_fit()
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.chain_ladder_fit().development.development.clone()
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.chain_ladder_fit().latest.clone()
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by origin:
    /// dimension ``"origin"``, one component per origin, one row per
    /// simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// Each simulation's Pareto ``k``, in the order of the simulations
    /// (``inf`` when too few weights differ to fit the tail).
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.refitted.clone())
    }

    /// The simulations moment matching rescued (their ``k`` brought to at
    /// most ``k_threshold``), in order (0-based); empty unless ``refit``
    /// was ``"moment_match_then_refit"``, when each flagged simulation is
    /// in this, in ``refitted`` or in ``unrefitted``.
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.moment_matched.clone())
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead (as unreliable as their ``k`` says), in order (0-based);
    /// empty without ``max_refits``.
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.unrefitted.clone())
    }

    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    #[getter]
    fn refit_warnings(&self) -> usize {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    /// One row per origin: ``origin``, ``latest``, ``opening_ultimate``,
    /// ``opening_reserve``, the model's own columns (``BayesBf``'s
    /// ``exposure`` and ``prior_ultimate``), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    /// One row: the totals of ``to_frame()``'s columns, the ``scale``, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_year!(&self.inner, f => f.fit.development_table()),
        )
    }

    fn __repr__(&self) -> String {
        each_bayes_one_year!(&self.inner, f => format!(
            "BayesOneYearFit(origins={}, n_sims={}, flagged_share={:?}, refitted={})",
            f.opening_reserve.len(),
            f.cdr.n_sims(),
            f.flagged_share(),
            f.refitted.len(),
        ))
    }
}

/// The one-year view of a Bayesian model in every segment of a triangle,
/// from ``BayesOdp.one_year_segments`` or ``Meyers.one_year_segments``,
/// laid out as ``OneYearFit``.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``cdr``. ``scale`` (the
/// ODP's only), ``premium`` (Meyers' only), ``diagnostics``, ``pareto_k``,
/// ``refitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales or premiums and the
/// flagged shares) or ``diagnostics_frame()``.
#[pyclass(name = "BayesOneYearFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFits {
    inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
enum BayesOneYearFitsOf {
    Odp(BayesOneYearFits<BayesOdpSegment>),
    Meyers(BayesOneYearFits<MeyersSegment>),
}

/// Evaluates `$body` with `$f` bound to the views over segments, whatever
/// the model.
macro_rules! each_bayes_one_years {
    ($views:expr, $f:ident => $body:expr) => {
        match $views {
            BayesOneYearFitsOf::Odp($f) => $body,
            BayesOneYearFitsOf::Meyers($f) => $body,
        }
    };
}

/// A segment's label prefix, its number of refits and how many of them
/// warned, for ``warn_refits``.
fn segment_refits<S>(
    segments: &SegmentFits<BayesOneYearSegment<S>>,
) -> Vec<(String, usize, usize, usize)> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .map(|(label, s)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            (
                prefix,
                s.refitted.len(),
                s.refit_warnings,
                s.unrefitted.len(),
            )
        })
        .collect()
}

#[pymethods]
impl PyBayesOneYearFits {
    /// The volume-weighted chain ladder of every segment.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => f.segments.key_names.clone())
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        each_bayes_one_years!(&self.inner, f => segment_labels(py, &f.segments))
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => origin_labels(&f.segments))
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0]
            .fit
            .chain_ladder
            .development
            .development
            .clone())
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .fit
            .chain_ladder
            .latest
            .clone()))
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale ``phi`` of a single-segment ``BayesOdp`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without a scale
    ///     (``Meyers``).
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(PyValueError::new_err(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none",
            )),
        }
    }

    /// The premium of each origin of a single-segment ``Meyers`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without premium
    ///     (``BayesOdp``).
    #[getter]
    fn premium(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(PyValueError::new_err(
                "premium is Meyers' input; BayesOdp has none",
            )),
        }
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        each_bayes_one_years!(&self.inner, f => diagnostics_dict(
            py,
            &single(&f.segments, "diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        ))
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS.
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .clone()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued.
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .clone()))
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead.
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .clone()))
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings))
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` of the segment's posterior, as
    /// ``BayesOdpFits.diagnostics_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``opening_ultimate``, ``opening_reserve``, the model's
    /// own columns (Meyers' ``premium``), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the model's own totals (the ODP's ``scale``, Meyers'
    /// ``premium``), the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_years!(&self.inner, f => f.segments.development_table()),
        )
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(err)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(err)?)
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        each_bayes_one_years!(&self.inner, f => {
            let segments = &f.segments;
            let detail = match segments.fits.as_slice() {
                [one] => format!(
                    ", flagged_share={:?}, refitted={}",
                    one.flagged_share(),
                    one.refitted.len()
                ),
                _ => String::new(),
            };
            format!(
                "BayesOneYearFits({}origins={}, n_sims={}{detail})",
                segments_prefix(segments),
                origin_labels(segments).len(),
                f.cdr.n_sims(),
            )
        })
    }
}

/// The Bayesian Bornhuetter–Ferguson of England, Verrall and Wüthrich
/// (2012, Model 3.1):the over-dispersed Poisson model of a cumulative
/// triangle's incremental values ``X`` with mean ``mu[w] * gamma[d]``, an
/// informative gamma prior on each origin's row parameter ``mu[w]`` with
/// mean ``apriori * exposure[w]`` and coefficient of variation ``cv``, a
/// vague normal prior on each ``log gamma[d]`` (the pattern is not
/// normalised), and the
/// chain ladder's Pearson scale ``phi`` (``OdpBootstrapFit.scale``) plugged
/// in. The posterior is sampled with NUTS, and each draw projects the
/// future cells up to the oldest observed age (no tail) with the process
/// error of ``process``, as ``BayesOdp`` does.
///
/// The prior's strength moves the reserve between two limits. With
/// ``cv=float("inf")`` (a flat prior on ``log mu[w]``) the model is the
/// Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
/// Monte Carlo error. As ``cv`` falls to 0, ``mu[w]`` is fixed at the prior
/// ultimate ``m[w]`` and each origin's mean reserve tends to ``m[w]`` times
/// the sum of the future ``S[d] / M[d]`` (``S[d]`` the observed increments
/// of column ``d``, ``M[d]`` the prior ultimates of the origins observed
/// there): EVW's strong-prior limit, Mack's (2008) Bornhuetter–Ferguson
/// with the raw pattern. That equals the classical
/// ``BornhuetterFerguson`` reserve ``(1 - 1 / cdf) * m[w]`` only when the
/// prior ultimates are the chain ladder's.
///
/// ``mu[w]`` is EVW's row parameter, not the origin's expected ultimate:
/// that is ``mu[w]`` times the sum of the ``gamma[d]``, which under a tight
/// prior is ``m[w]`` times the sum of ``S[d] / M[d]`` over all columns. On
/// EVW's triangle that sum is 0.885, so the implied ultimate loss ratio is
/// about 11.5% below ``apriori``.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than ``-phi / cv**2``; negative cells
/// are allowed. One ``seed`` replays a fit exactly, whatever the number of
/// threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
/// below 100 per chain, any divergent transition) do not fail the fit:
/// ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``BayesBfFit.diagnostics``.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of each origin's ``mu[w]`` per unit of exposure;
///     positive and finite. It is the expected loss ratio only when the
///     pattern ``gamma`` sums to 1 (see above).
/// cv : float, default 0.1
///     Coefficient of variation of the gamma prior on each ``mu[w]`` (shape
///     ``1 / cv**2``, EVW's ``a``); positive, ``float("inf")`` for a flat
///     prior.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each ``log gamma[d]``,
///     centred on 0. With a flat ``cv``, a much vaguer prior (1,000, say)
///     leaves divergent transitions.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesBf, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
/// ...     },
/// ... )
/// >>> fit = BayesBf(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.prior_ultimate[3] == 0.85 * 260.0
/// True
/// >>> fit.parameter_names[:2]
/// ['alpha[1]', 'alpha[2]']
#[pyclass(name = "BayesBf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBf {
    inner: BayesBf,
}

#[pymethods]
impl PyBayesBf {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        cv = 0.1,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesBf {
                apriori,
                cv,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
            },
        })
    }

    /// Prior mean of each origin's ``mu[w]`` per unit of exposure.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of each origin's prior; ``inf`` is flat.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// Samples the posterior of one loss column of a single-segment
    /// cumulative triangle, with each origin's exposure from another
    /// column, and simulates its reserves. Every origin must be observed
    /// from the first age up to its latest and there must be more observed
    /// cells than chain-ladder parameters. Raises one ``UserWarning``
    /// listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure, as for ``BornhuetterFerguson``.
    ///
    /// Returns
    /// -------
    /// BayesBfFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     an origin has no observed, finite, positive exposure; if
    ///     ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
    ///     setting is; if a development column's increments do not sum to
    ///     more than zero; or if an origin's sum to ``-phi / cv**2`` or
    ///     less.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyBayesBfFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, column, exposure))
            .map_err(err)?;
        let warnings = &inner.diagnostics.warnings;
        if !warnings.is_empty() {
            let message = format!(
                "BayesBf: the MCMC diagnostics failed; check them before relying on the fit: {}",
                warnings.join("; ")
            );
            let message = std::ffi::CString::new(message)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyBayesBfFit { inner })
    }

    /// The one-year view of one loss column of a single-segment cumulative
    /// triangle, with each origin's exposure from another column: the
    /// claims development result over the coming year, by Pareto smoothed
    /// importance sampling of ``fit``'s posterior (decisions 16 and 17 of
    /// ``docs/design/reserving-bayes.md``), exactly as
    /// ``BayesOdp.one_year`` with this model's means
    /// ``mu[w] * gamma[d]``. Each simulation draws the year's increments
    /// from a posterior draw with ``process`` error, reweights every
    /// posterior draw by the year's quasi likelihood (``scale``, the
    /// priors and the prior ultimates unchanged), and takes each origin's
    /// claims development result as its posterior mean reserve at the
    /// opening less its payment in the year and its weighted posterior mean
    /// reserve after it. A simulation whose Pareto ``k`` is above
    /// ``k_threshold`` is handled as ``refit`` says: by default
    /// moment-matched, and refitted with NUTS on the extended triangle
    /// only if matching cannot bring its ``k`` to the threshold. A flat prior (``cv=inf``) gives about
    /// ``BayesOdp.one_year``'s view; a tighter prior a narrower one, in
    /// which the development pattern learns from the year and the origins'
    /// levels hardly do. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``BayesBfFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesBf, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    /// ...     },
    /// ... )
    /// >>> model = BayesBf(apriori=0.85, cv=0.1, n_sims=1000, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> type(view.fit).__name__, len(view.pareto_k)
    /// ('BayesBfFit', 1000)
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesBf", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesBf",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Bf(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesBf(apriori={:?}, cv={:?}, development_sd={:?}, process={:?}, chains={}, \
             tune={}, draws={}, n_sims={}, seed={})",
            self.inner.apriori,
            self.inner.cv,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
        )
    }
}

/// A fitted ``BayesBf``: the posterior of ``alpha[1..n]`` (``log mu[w]``)
/// and ``beta[1..m]`` (``log gamma[d]``), its MCMC diagnostics, and the
/// posterior predictive distribution of the reserves, laid out as
/// ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
/// ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
/// with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesBfFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfFit {
    inner: BayesBfFit,
}

#[pymethods]
impl PyBayesBfFit {
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The prior mean of each origin's row parameter ``mu[w]``,
    /// ``apriori * exposure`` (EVW's prior ultimate). The model's expected
    /// ultimate is ``mu[w]`` times the sum of the ``gamma[d]``, not
    /// ``mu[w]``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[1]`` to
    /// ``beta[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
    /// pointwise term is the normalised over-dispersed Poisson density at
    /// the cell's mean and ``scale``, so the result compares with
    /// ``BayesOdp``'s on the same column.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``exposure`` and
    /// ``prior_ultimate``, and the ``mean`` and ``std_dev`` of the
    /// posterior predictive reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``exposure`` and
    /// ``prior_ultimate``, the ``scale``, and the ``mean`` and ``std_dev``
    /// of the posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesBfFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// The Bayesian Bornhuetter–Ferguson whose prior sits on the ultimate: the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``U[w] * p[d]``, the origin's expected ultimate
/// ``U[w]`` times the share ``p[d]`` of it that falls at age ``d``, the
/// shares summing to 1 (``p[d] = exp(beta[d]) / sum(exp(beta))`` with
/// ``beta[1] = 0``). The informative gamma prior is on ``U[w]`` itself,
/// with mean ``apriori * exposure[w]`` and coefficient of variation
/// ``cv``, so ``apriori`` is the expected ultimate loss ratio and ``cv``
/// says how firmly it is held, as in the classical
/// ``BornhuetterFerguson``. The shares have a vague normal prior on their
/// log ratios to the first age's, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) is plugged in. The posterior is
/// sampled with NUTS, and each draw projects the future cells up to the
/// oldest observed age (no tail) with the process error of ``process``, as
/// ``BayesOdp`` does.
///
/// ``BayesBf`` (England, Verrall and Wüthrich's Model 3.1) instead
/// leaves the pattern unnormalised and puts its prior on ``mu[w]``, which
/// is not the ultimate: there the level of ``apriori`` is cancelled by the
/// pattern and barely moves the reserve. Use this model when the
/// a-priori loss ratio is a judgment about the ultimate; use ``BayesBf``
/// to match England, Verrall and Wüthrich's published results.
///
/// The prior's strength moves the reserve between two limits. With
/// ``cv=float("inf")`` (a flat prior on ``log U[w]``) the model is the
/// Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
/// Monte Carlo error. As ``cv`` falls to 0, ``U[w]`` is fixed at the prior
/// ultimate ``m[w]`` and the shares are ``S[d] / (M[d] + lambda)`` (``S[d]``
/// the observed increments of column ``d``, ``M[d]`` the prior ultimates of
/// the origins observed there, ``lambda`` such that they sum to 1). When
/// the prior ultimates are the chain ladder's, ``lambda`` is 0 and the
/// reserve is exactly the classical ``BornhuetterFerguson`` reserve
/// ``(1 - 1 / cdf) * m[w]``; otherwise the shares move with the ultimates,
/// where the classical method keeps the chain ladder's pattern.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than ``-phi / cv**2``; negative cells
/// are allowed. One ``seed`` replays a fit exactly, whatever the number of
/// threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
/// below 100 per chain, any divergent transition) do not fail the fit:
/// ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``BayesBfNormalisedFit.diagnostics``.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of each origin's expected ultimate ``U[w]`` per unit of
///     exposure, the expected ultimate loss ratio; positive and finite.
/// cv : float, default 0.1
///     Coefficient of variation of the gamma prior on each ``U[w]`` (shape
///     ``1 / cv**2``); positive, ``float("inf")`` for a flat prior.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each ``beta[d]`` (the log
///     ratio of the share at age ``d`` to the first age's), centred on 0.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesBfNormalised, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
/// ...     },
/// ... )
/// >>> fit = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.prior_ultimate[3] == 0.85 * 260.0
/// True
/// >>> fit.parameter_names[:2]
/// ['alpha[1]', 'alpha[2]']
#[pyclass(name = "BayesBfNormalised", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalised {
    pub(crate) inner: BayesBfNormalised,
}

#[pymethods]
impl PyBayesBfNormalised {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        cv = 0.1,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        common_cv = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        common_cv: Option<f64>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesBfNormalised {
                apriori,
                cv,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                common_cv,
            },
        })
    }

    /// The coefficient of variation of the multiplier common to all
    /// origins' prior ultimates, or ``None`` if there is none.
    #[getter]
    fn common_cv(&self) -> Option<f64> {
        self.inner.common_cv
    }

    /// Prior mean of each origin's expected ultimate per unit of exposure.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of each origin's prior; ``inf`` is flat.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// Samples the posterior of one loss column of a single-segment
    /// cumulative triangle, with each origin's exposure from another
    /// column, and simulates its reserves. Every origin must be observed
    /// from the first age up to its latest and there must be more observed
    /// cells than chain-ladder parameters. Raises one ``UserWarning``
    /// listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure, as for ``BornhuetterFerguson``.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     an origin has no observed, finite, positive exposure; if
    ///     ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
    ///     setting is; if a development column's increments do not sum to
    ///     more than zero; or if an origin's sum to ``-phi / cv**2`` or
    ///     less.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, column, exposure))
            .map_err(err)?;
        let warnings = &inner.diagnostics.warnings;
        if !warnings.is_empty() {
            let message = format!(
                "BayesBfNormalised: the MCMC diagnostics failed; check them before relying on the fit: {}",
                warnings.join("; ")
            );
            let message = std::ffi::CString::new(message)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of one loss column of a single-segment cumulative
    /// triangle, with each origin's exposure from another column: the
    /// claims development result over the coming year, by Pareto smoothed
    /// importance sampling of ``fit``'s posterior (decisions 16 and 17 of
    /// ``docs/design/reserving-bayes.md``), exactly as
    /// ``BayesOdp.one_year`` with this model's means
    /// ``exp(alpha[w]) * p[d]``. Each simulation draws the year's increments
    /// from a posterior draw with ``process`` error, reweights every
    /// posterior draw by the year's quasi likelihood (``scale``, the
    /// priors and the prior ultimates unchanged), and takes each origin's
    /// claims development result as its posterior mean reserve at the
    /// opening less its payment in the year and its weighted posterior mean
    /// reserve after it. A simulation whose Pareto ``k`` is above
    /// ``k_threshold`` is handled as ``refit`` says: by default
    /// moment-matched, and refitted with NUTS on the extended triangle
    /// only if matching cannot bring its ``k`` to the threshold. A flat
    /// prior (``cv=inf``) gives about ``BayesOdp.one_year``'s view; a
    /// tighter prior holds each origin's ultimate more firmly and gives a
    /// narrower one. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``BayesBfNormalisedFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesBfNormalised, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    /// ...     },
    /// ... )
    /// >>> model = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=1000, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> type(view.fit).__name__, len(view.pareto_k)
    /// ('BayesBfNormalisedFit', 1000)
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesBfNormalised", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesBfNormalised",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesBfNormalised(apriori={:?}, cv={:?}, development_sd={:?}, process={:?}, chains={}, \
             tune={}, draws={}, n_sims={}, seed={})",
            self.inner.apriori,
            self.inner.cv,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
        )
    }
}

/// A fitted ``BayesBfNormalised``: the posterior of ``alpha[1..n]`` (the log
/// expected ultimates) and ``beta[2..m]`` (the log ratios of the shares to
/// the first age's), its MCMC diagnostics, and the
/// posterior predictive distribution of the reserves, laid out as
/// ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
/// ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
/// with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesBfNormalisedFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalisedFit {
    inner: BayesBfNormalisedFit,
}

#[pymethods]
impl PyBayesBfNormalisedFit {
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The prior mean of each origin's expected ultimate,
    /// ``apriori * exposure``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The posterior mean of each age's share ``p[d]`` of the ultimate, in
    /// development order; the shares sum to 1.
    #[getter]
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern()
    }

    /// The posterior mean of each origin's expected ultimate
    /// ``exp(alpha[w])``, the quantity that ``apriori`` and ``cv`` put the
    /// prior on (the reserve's process error and the cells already paid are
    /// not in it).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate()
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[2]`` to
    /// ``beta[m]`` (the first age is the corner, ``beta[1] = 0``).
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
    /// pointwise term is the normalised over-dispersed Poisson density at
    /// the cell's mean and ``scale``, so the result compares with
    /// ``BayesOdp``'s on the same column.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``exposure`` and
    /// ``prior_ultimate``, and the ``mean`` and ``std_dev`` of the
    /// posterior predictive reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``exposure`` and
    /// ``prior_ultimate``, the ``scale``, and the ``mean`` and ``std_dev``
    /// of the posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesBfNormalisedFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// The prior of a layer's expected ultimate for ``BayesBfNormalised``, from
/// the ground-up expected loss ratio and the layer share, each with an
/// uncertainty (``docs/design/reserving-excess.md``, PR 4).
///
/// The layer's a-priori ultimate for origin ``i`` is the ground-up exposure
/// times ``apriori`` times the share. The normalised BF puts a gamma prior
/// on the layer's expected ultimate, centred there, whose coefficient of
/// variation is that of a product of two independent variables,
/// ``cv = sqrt((1 + apriori_cv**2) * (1 + share_cv**2) - 1)``: the moments
/// are exact. One share error is common to all origins (the default,
/// ``share_dependence="common"``): the model carries one more parameter, a
/// gamma multiplier ``s`` with mean 1 and cv ``share_cv``, sampled with the
/// others, and each origin's prior ultimate is a gamma with mean
/// ``exposure * apriori * share * s`` and cv ``apriori_cv``, so every origin
/// keeps the marginal cv above, the origins' priors are correlated and the
/// data update the share for every origin at once (``log_share`` in the
/// fit's parameters). ``share_dependence="per_origin"`` is the first
/// version: an independent share error at each origin, which understates
/// the total's interval. The layer's development pattern is estimated from
/// the layer's own triangle under the model's vague prior on the pattern;
/// an excess pattern from ``ExcessPattern`` does not enter (open decision
/// 22 of the design note).
///
/// Parameters
/// ----------
/// apriori : float
///     The ground-up expected ultimate loss ratio.
/// shares : float or list of float
///     The layer share (``layer_share`` of a severity, or an exposure
///     curve's ``layer_share``), one for all origins or one per origin,
///     each in ``(0, 1]``.
/// apriori_cv : float, default 0.1
///     The loss ratio's coefficient of variation (``inf`` is flat).
/// share_cv : float, default 0.0
///     The share's coefficient of variation, the same at every origin.
/// share_dependence : {"common", "per_origin"}, default "common"
///     Whether one share error is common to all origins or each origin has
///     its own. A known share (``share_cv`` 0) or a flat prior has nothing
///     to share and is the per-origin model; a common share error needs a
///     positive ``apriori_cv``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import LayerSharePrior
/// >>> prior = LayerSharePrior(0.65, 0.2, apriori_cv=0.1, share_cv=0.3)
/// >>> round(prior.layer_apriori(0), 12), round(prior.cv ** 2, 12)
/// (0.13, 0.1009)
/// >>> prior.share_dependence
/// 'common'
#[pyclass(name = "LayerSharePrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyLayerSharePrior {
    inner: LayerSharePrior,
}

fn share_dependence_arg(name: &str) -> PyResult<ShareDependence> {
    match name {
        "common" => Ok(ShareDependence::Common),
        "per_origin" => Ok(ShareDependence::PerOrigin),
        _ => Err(PyValueError::new_err(format!(
            "unknown share_dependence {name:?}; expected \"common\" or \"per_origin\""
        ))),
    }
}

fn shares_arg(shares: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(one) = shares.extract::<f64>() {
        return Ok(vec![one]);
    }
    shares.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "shares must be a number or a list of numbers, got {}",
            py_repr(shares)
        ))
    })
}

#[pymethods]
impl PyLayerSharePrior {
    #[new]
    #[pyo3(signature = (apriori, shares, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn new(
        apriori: f64,
        shares: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::new(apriori, apriori_cv, &shares_arg(shares)?, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior from one share and its standard error: the share's
    /// coefficient of variation is ``share_se / share``.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// share : float
    /// share_se : float
    /// apriori_cv : float, default 0.1
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, share, share_se, apriori_cv = 0.1, share_dependence = "common"))]
    fn from_share_se(
        apriori: f64,
        share: f64,
        share_se: f64,
        apriori_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::from_share_se(apriori, apriori_cv, share, share_se)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior with the share of a severity in a layer (``layer_share``,
    /// ultimate basis) and its coefficient of variation.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// severity : severity distribution
    /// layer : ExcessLayer
    /// apriori_cv : float, default 0.1
    /// share_cv : float, default 0.0
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, severity, layer, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn from_severity(
        apriori: f64,
        severity: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        let sev = crate::distributions::extract_severity(severity)?;
        let layer = crate::excess::layer_of(layer, "layer")?;
        Ok(Self {
            inner: LayerSharePrior::from_severity(apriori, apriori_cv, &sev, layer, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// The ground-up expected ultimate loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// The loss ratio's coefficient of variation.
    #[getter]
    fn apriori_cv(&self) -> f64 {
        self.inner.apriori_cv
    }

    /// The layer shares, one or one per origin.
    #[getter]
    fn shares(&self) -> Vec<f64> {
        self.inner.shares.clone()
    }

    /// The share's coefficient of variation.
    #[getter]
    fn share_cv(&self) -> f64 {
        self.inner.share_cv
    }

    /// ``"common"`` or ``"per_origin"``: whether one share error is common
    /// to all origins.
    #[getter]
    fn share_dependence(&self) -> &'static str {
        match self.inner.dependence {
            ShareDependence::Common => "common",
            ShareDependence::PerOrigin => "per_origin",
        }
    }

    /// The marginal coefficient of variation of the prior on each origin's
    /// expected ultimate: ``sqrt((1 + apriori_cv**2) * (1 + share_cv**2) -
    /// 1)``, ``inf`` if either is. It is the gamma's cv under
    /// ``"per_origin"``; under ``"common"`` the model's own cv is
    /// ``apriori_cv`` and the multiplier's ``share_cv``.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv()
    }

    /// Whether the model carries a multiplier common to all origins: the
    /// share error is ``"common"`` and both cvs are positive and finite.
    #[getter]
    fn has_common_factor(&self) -> bool {
        self.inner.has_common_factor()
    }

    /// The mean and standard deviation of the prior of the layer's total
    /// expected ultimate, in closed form: ``c_e**2 * (1 + c_s**2) * sum(m**2)
    /// + c_s**2 * sum(m)**2`` for a common share error and ``((1 + c_e**2) *
    /// (1 + c_s**2) - 1) * sum(m**2)`` for independent ones, with ``m`` each
    /// origin's prior ultimate ``exposure * apriori * share``.
    ///
    /// Parameters
    /// ----------
    /// exposure : list of float
    ///     The ground-up exposure of each origin.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     ``(mean, sd)``; ``sd`` is ``inf`` for a flat prior.
    fn prior_total(&self, exposure: Vec<f64>) -> PyResult<(f64, f64)> {
        self.inner.prior_total(&exposure).map_err(err)
    }

    /// The layer's a-priori loss ratio at origin ``i`` (0-based): the
    /// ground-up loss ratio times that origin's share.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///
    /// Returns
    /// -------
    /// float
    fn layer_apriori(&self, i: usize) -> PyResult<f64> {
        if self.inner.shares.len() > 1 && i >= self.inner.shares.len() {
            return Err(PyValueError::new_err(format!(
                "origin {i} is out of range for {} shares",
                self.inner.shares.len()
            )));
        }
        Ok(self.inner.layer_apriori(i))
    }

    /// ``model`` (default settings if omitted) with this prior's ground-up
    /// ``apriori`` and its ``cv``, for the triangle of ``triangle``.
    ///
    /// Parameters
    /// ----------
    /// model : BayesBfNormalised, optional
    ///     Supplies the sampler, process error, simulations and seed; its
    ///     own ``apriori``, ``cv`` and ``common_cv`` are replaced.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A common share error with ``apriori_cv`` 0.
    #[pyo3(signature = (model = None))]
    fn model(
        &self,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalised> {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        Ok(PyBayesBfNormalised {
            inner: self.inner.model(&base).map_err(err)?,
        })
    }

    /// ``triangle`` with a ``"layer_exposure"`` column: the ground-up
    /// exposure column times each origin's share.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// exposure : str
    ///     The ground-up exposure column.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn triangle(&self, triangle: PyRef<'_, PyTriangle>, exposure: &str) -> PyResult<PyTriangle> {
        Ok(PyTriangle {
            inner: self
                .inner
                .triangle(&triangle.inner, exposure)
                .map_err(err)?,
        })
    }

    /// Fits the layer losses ``column`` of a single-segment cumulative
    /// triangle that also holds the ground-up ``exposure`` column, with
    /// this prior: ``BayesBfNormalised.fit`` on the layer exposure. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The layer losses.
    /// exposure : str
    ///     The ground-up exposure column.
    /// model : BayesBfNormalised, optional
    ///     The sampler and other settings; its ``apriori`` and ``cv`` are
    ///     replaced by the prior's.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``BayesBfNormalised.fit``; if the prior's ``cv`` is 0.
    #[pyo3(signature = (triangle, column, exposure, model = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.fit(&base, tri, column, exposure))
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.diagnostics.warnings)?;
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of the same fit (``BayesBfNormalised.one_year``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    /// k_threshold : float, default 0.7
    /// refit : str, default "moment_match_then_refit"
    /// max_refits : int, optional
    /// n_sims : int, optional
    ///     As in ``BayesBfNormalised.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    #[pyo3(signature = (triangle, column, exposure, model = None, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.one_year(&base, tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "LayerSharePrior",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerSharePrior(apriori={:?}, shares={:?}, apriori_cv={:?}, share_cv={:?}, share_dependence={:?})",
            self.inner.apriori,
            self.inner.shares,
            self.inner.apriori_cv,
            self.inner.share_cv,
            self.share_dependence(),
        )
    }
}

fn meyers_model(name: &str) -> PyResult<MeyersModel> {
    match name.to_ascii_lowercase().as_str() {
        "crc" => Ok(MeyersModel::Crc),
        "csr" => Ok(MeyersModel::Csr),
        "cay" => Ok(MeyersModel::Cay),
        _ => Err(PyValueError::new_err(format!(
            "unknown Meyers model {name:?}; expected \"crc\", \"csr\" or \"cay\""
        ))),
    }
}

/// Meyers' (2019, CAS Monograph 8) log-normal models of a cumulative
/// triangle ``C`` with premium ``P``, for origins ``w = 1..n`` and ages
/// ``d = 1..m``: ``log C[w, d]`` is normal with mean
/// ``log P[w] + logelr + alpha[w] + beta[d] g[w]`` (``alpha[1] = 0``,
/// ``beta[m] = 0``) and sd ``sigma[d]``, decreasing with age. ``g[w]`` is 1
/// for the correlated chain ladder without its correlation between origins
/// (``"crc"``) and ``(1 - gamma)^(w - 1)`` for the changing settlement rate
/// (``"csr"``). The correlated accident year model (``"cay"``) keeps
/// ``g[w] = 1`` and adds ``rho`` times the previous origin's residual at
/// the same age, ``log C[w - 1, d]`` less its mean, to the mean of every
/// origin after the first, so neighbouring origins are correlated; Meyers
/// fits it to incurred data. The priors are those of Meyers' Stan code:
/// ``logelr`` normal with mean ``logelr_mean`` and sd ``logelr_sd``,
/// ``alpha`` and ``beta`` normal around 0, ``gamma`` normal around 0 with
/// sd ``gamma_sd``, ``(1 - rho) / 2`` beta with both shapes ``rho_shape``,
/// and each step of ``sigma[d]^2`` uniform on ``(1 - exp(-1e-5), 1)``.
///
/// The ultimate is the value at the oldest age (no tail): each posterior
/// draw gives each open origin a log-normal ultimate; an origin observed at
/// the oldest age keeps its value; under ``"cay"`` the ultimates are drawn
/// origin by origin, each from the previous origin's residual at the oldest
/// age, observed for a closed origin and simulated otherwise.
/// The reserve is the ultimate less the latest value of the fitted column
/// (an outstanding reserve on paid data, IBNR on incurred data) and can be
/// negative. Every observed cell must be positive, and under ``"cay"``
/// every observed cell after the first origin needs the previous origin's
/// cell at the same age.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``MeyersFit.diagnostics``. With the defaults, all 120 fits of a seed
/// sweep on Meyers' group 353 data (paid and incurred, CRC and CSR, 30
/// seeds) passed every threshold, and CAY on incurred passed at 199 of 200
/// seeds (one divergent transition in 8,000 draws at the other).
///
/// Parameters
/// ----------
/// model : {"crc", "csr", "cay"}, default "crc"
///     The correlated chain ladder without its correlation between origins,
///     the changing settlement rate model, or the correlated accident year
///     model.
/// logelr_mean : float, default -0.4
///     Mean of the normal prior on ``logelr``, the log expected loss ratio.
/// logelr_sd : float, default 3.162
///     Standard deviation of the normal prior on ``logelr`` (3.162 is
///     Meyers' rounding of ``sqrt(10)``).
/// origin_sd : float, default 3.162
///     Standard deviation of the normal prior on each origin effect
///     ``alpha[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 3.162
///     Standard deviation of the normal prior on each development effect
///     ``beta[d]``, ``d < m``, centred on 0.
/// gamma_sd : float, default 0.05
///     Standard deviation of the normal prior on ``gamma``, centred on 0
///     (``"csr"`` only).
/// rho_shape : float, default 2.0
///     Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"``
///     only); Meyers' 2 makes the prior density of ``rho`` proportional to
///     ``1 - rho^2``.
/// chains, tune, draws : int, default 4, 1000, 2000
///     NUTS chains, warm-up draws per chain and kept draws per chain; with
///     1,000 kept draws the late ``sigma[d]`` sometimes miss the R-hat
///     threshold.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; at NUTS's
///     usual 0.8 the paid fits of group 353 leave divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it),
///     as ``BayesOdp``'s: each segment's own posterior and simulations, with
///     seeds derived from ``seed`` and the segment's position; or the
///     segments' simulations reordered as whole rows (Iman–Conover on the
///     segments' total reserves, each segment's ultimates moving with its
///     reserves) to the Spearman matrix ``spearman`` and then put in a
///     random order. ``"synchronized"`` is accepted here, as the bootstraps
///     take it, but ``fit_segments`` raises: a posterior has no residuals to
///     share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' total reserves, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``model`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Meyers, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
/// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
/// >>> rows = [(1988 + w, 12 * (d + 1), float(v), float(premium[w]))
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = Meyers("csr", tune=300, draws=300, n_sims=2000, seed=1).fit(
/// ...         tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('1988',), ('1989',), ('1990',), ('1991',), ('1992',), ('1993',)]
/// >>> fit.parameter_names[0], fit.parameter_names[-1]
/// ('logelr', 'sigma[6]')
/// >>> fit.premium[:2]
/// [5812.0, 4908.0]
#[pyclass(name = "Meyers", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyers {
    inner: Meyers,
}

#[pymethods]
impl PyMeyers {
    #[new]
    #[pyo3(signature = (
        model = "crc",
        logelr_mean = -0.4,
        logelr_sd = 3.162,
        origin_sd = 3.162,
        development_sd = 3.162,
        gamma_sd = 0.05,
        rho_shape = 2.0,
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: Meyers {
                model: meyers_model(model)?,
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model.name()
    }

    /// Mean of the prior on ``logelr``.
    #[getter]
    fn logelr_mean(&self) -> f64 {
        self.inner.logelr_mean
    }

    /// Standard deviation of the prior on ``logelr``.
    #[getter]
    fn logelr_sd(&self) -> f64 {
        self.inner.logelr_sd
    }

    /// Standard deviation of each origin effect's prior.
    #[getter]
    fn origin_sd(&self) -> f64 {
        self.inner.origin_sd
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Standard deviation of the prior on ``gamma`` (``"csr"`` only).
    #[getter]
    fn gamma_sd(&self) -> f64 {
        self.inner.gamma_sd
    }

    /// Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"`` only).
    #[getter]
    fn rho_shape(&self) -> f64 {
        self.inner.rho_shape
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How segments depend on each other in ``fit_segments``:
    /// ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of one measure column of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value, as ``CapeCod`` reads exposure), and simulates the
    /// ultimates and reserves. Every origin must be observed from the first
    /// age up to its latest, every observed cell must be positive, and
    /// there must be more observed cells than parameters. Raises one
    /// ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is missing, the triangle has several segments or an
    ///     origin has a gap; if a premium is missing or not positive; if a
    ///     prior standard deviation or ``rho_shape`` is not positive and
    ///     finite, ``logelr_mean`` is not finite or a sampler setting is
    ///     invalid; if an observed cell is not positive; if there are no more
    ///     observed cells than parameters; or, under ``"cay"``, if a cell
    ///     lacks the previous origin's cell at the same age.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column, premium)).map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.diagnostics.warnings)?;
        Ok(PyMeyersFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into one joint
    /// distribution of the ultimates and one of the reserves, the segments
    /// depending on each other as ``dependence`` says. Segment ``k`` (in
    /// index order) takes seeds derived from ``seed`` and ``k`` for its
    /// chains and its simulations, so independent segments share no random
    /// numbers, and a segment's draws differ from ``fit`` of that segment
    /// alone. Raises one ``UserWarning`` listing the failed diagnostics of
    /// every segment, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = Meyers(tune=200, draws=200, n_sims=1000).fit_segments(
    /// ...         tri, "paid", "premium")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, column, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        Ok(PyMeyersFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle with premium: the claims development result over
    /// the coming year, by Pareto smoothed importance sampling of the
    /// posterior (decision 19 of ``docs/design/reserving-bayes.md``). The
    /// posterior and the lifetime ultimates and reserves are ``fit``'s for
    /// the same settings. Each simulation picks a posterior draw, draws the
    /// cumulative values of the year's cells from it (the cells valued in
    /// the next twelve months, as in ``OdpBootstrap.one_year``), reweights
    /// every posterior draw by the likelihood of those values, and smooths
    /// the weights (one Pareto ``k`` per simulation). Origin ``w``'s claims
    /// development result is
    ///
    /// ``CDR[w] = E[U[w] | D0] - E[U[w] | D0, C]``
    ///
    /// with ``U`` the ultimate (the value at the oldest age) and each
    /// expectation the (weighted) posterior mean of the ultimate's expected
    /// value given a draw; an origin the year takes to the oldest age has
    /// its value there instead of the second term. The mean is about zero.
    /// A simulation whose ``k`` is above ``k_threshold`` is flagged and
    /// handled as ``refit`` says: by default moment-matched, and refitted
    /// with NUTS only if its ``k`` stays above the threshold. On Meyers'
    /// group 353 nearly every simulation is flagged and matching rescues
    /// about nine in ten, so 10,000 simulations take 4 to 7 minutes on 16
    /// threads in a release build, and ``"refit_flagged"`` about 6.7 times
    /// as long. The view therefore runs at most 2,000 simulations by
    /// default (about a fifth of that time), whatever the model's
    /// ``n_sims`` is above that; ``n_sims`` here sets the count. Raises one ``UserWarning``
    /// listing the failed diagnostics of the fit, and another if any refit
    /// failed its diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged and costs a moment match or a refit. The
    ///     lifetime view in ``fit`` keeps ``n_sims``; a smaller count
    ///     gives the first simulations of a larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``MeyersFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
    /// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = Meyers(tune=300, draws=300, n_sims=1000, seed=1).one_year(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> type(view.fit).__name__, view.cdr.components()[:2]
    /// ('MeyersFit', [('1988',), ('1989',)])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, premium, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Meyers",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Meyers(Box::new(inner)),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle with premium: each segment's posterior and
    /// simulations as in ``fit_segments`` (segment ``k``'s own seeds), its
    /// view as in ``one_year``, into one joint claims development result,
    /// the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment's
    /// fit, and another if any refit failed its diagnostics, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    /// premium : str
    /// k_threshold : float, default 0.7
    ///     As in ``one_year``.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     As in ``one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``, for each segment: ``k`` segments
    ///     refit at most ``k`` times this many.
    /// n_sims : int, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = Meyers(tune=200, draws=200, n_sims=500).one_year_segments(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        warn_refits(py, "Meyers", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Meyers(inner),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Meyers(model={:?}, logelr_mean={:?}, logelr_sd={:?}, origin_sd={:?}, \
             development_sd={:?}, gamma_sd={:?}, rho_shape={:?}, chains={}, tune={}, \
             draws={}, n_sims={}, seed={}{})",
            m.model.name(),
            m.logelr_mean,
            m.logelr_sd,
            m.origin_sd,
            m.development_sd,
            m.gamma_sd,
            m.rho_shape,
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
            dependence_repr(&m.dependence),
        )
    }
}

/// A fitted ``Meyers``: the posterior of ``logelr``, ``alpha[2..n]``,
/// ``beta[1..m-1]``, ``gamma`` (``"csr"`` only) or ``rho`` (``"cay"``
/// only), and ``sigma[1..m]``, its
/// MCMC diagnostics, and the posterior predictive distributions of the
/// ultimates and reserves by origin. The frames' ``latest``, ``ultimate``
/// and ``reserve`` are the volume-weighted chain ladder's
/// (``chain_ladder``); ``mean`` and ``std_dev`` are the posterior
/// predictive reserve's, and ``ultimate_mean`` and ``ultimate_std_dev``
/// the posterior predictive ultimate's.
#[pyclass(name = "MeyersFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFit {
    inner: MeyersFit,
}

#[pymethods]
impl PyMeyersFit {
    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model().name()
    }

    /// The volume-weighted chain ladder of the fitted column, whose latest
    /// values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The premium of each origin.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension ``"origin"``, one component
    /// per origin, one row per simulation. An origin observed at the
    /// oldest age keeps its value.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates`` and as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``logelr``, ``alpha[2]`` to ``alpha[n]``,
    /// ``beta[1]`` to ``beta[m-1]``, ``gamma`` (``"csr"`` only) or ``rho``
    /// (``"cay"`` only), and ``sigma[1]`` to ``sigma[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is Meyers', the normal density of ``log C[w, d]``, so the
    /// result is on the log scale and comparable only between ``"crc"``,
    /// ``"csr"`` and ``"cay"`` fits of the same column (Meyers 2019,
    /// Tables 6.4, 7.3 and 8.3). Under ``"cay"`` each cell's density is
    /// conditional on the previous origin's cell, as in Meyers' code.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``premium``, the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, and the
    /// ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
    /// predictive ultimate. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``premium``, the
    /// ``mean`` and ``std_dev`` of the posterior predictive total reserve,
    /// and the ``ultimate_mean`` and ``ultimate_std_dev`` of the total
    /// ultimate. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "MeyersFit(model={:?}, origins={}, draws={}, n_sims={}, warnings={})",
            self.inner.model().name(),
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Meyers.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distributions of the ultimates and the reserves,
/// laid out as ``BayesOdpFits``.
///
/// ``ultimates`` and ``reserves`` have the triangle's keys and ``"origin"``
/// as dimensions, so ``reserves.aggregate(["lob"])`` keeps the dependence
/// between segments. Per-origin lists (``origins``, ``premium``) run over
/// the origins of each segment in turn, like the rows of ``to_frame()``
/// and the components of ``reserves``. The posterior of one segment
/// (``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()`` and ``loo()``) needs a single-segment fit: for several
/// segments use ``segment(...)`` or ``diagnostics_frame()``.
#[pyclass(name = "MeyersFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFits {
    inner: MeyersFits,
}

impl PyMeyersFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&MeyersSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyMeyersFits {
    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.segments.fits[0].model().name()
    }

    /// The volume-weighted chain ladder of every segment's fitted column,
    /// whose latest values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.premium.iter().copied())
            .collect()
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by segment and origin: the triangle's keys and
    /// ``"origin"`` are its dimensions, one component per segment and
    /// origin, one row per simulation.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``MeyersFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``MeyersFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``MeyersFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``MeyersFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells on Meyers' log
    /// scale, as ``MeyersFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = self.one("loo()", "")?;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, the
    /// ``premium``, the ``mean`` and ``std_dev`` of the posterior
    /// predictive reserve, and the ``ultimate_mean`` and
    /// ``ultimate_std_dev`` of the posterior predictive ultimate. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the total ``premium``, the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve, and the ``ultimate_mean`` and
    /// ``ultimate_std_dev`` of its total ultimate. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint ultimates and
    /// reserves (same dimensions).
    ///
    /// Returns
    /// -------
    /// MeyersFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "MeyersFits(model={:?}, {}origins={}, n_sims={}, warnings={warnings})",
            self.model(),
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

// ---------------------------------------------------------------------------
// Gesmann and Morris' compartmental model (docs/design/reserving-compartmental.md)
// ---------------------------------------------------------------------------

/// `(median, sd)` of a log-normal prior on a positive parameter.
fn positive_prior(name: &str, value: (f64, f64)) -> PyResult<PositivePrior> {
    let (median, sd) = value;
    if !(median.is_finite() && median > 0.0 && sd.is_finite() && sd > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be (median, sd) with both finite and positive, got ({median}, {sd})"
        )));
    }
    Ok(PositivePrior::LogNormal { median, sd })
}

fn positive_prior_tuple(prior: PositivePrior) -> (f64, f64) {
    let PositivePrior::LogNormal { median, sd } = prior;
    (median, sd)
}

/// A half prior from ``("student_t", nu, scale)`` or ``("normal", scale)``.
fn half_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<HalfPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, scale) or (\"normal\", scale)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 3) => Ok(HalfPrior::StudentT {
            nu: num(1)?,
            scale: num(2)?,
        }),
        ("normal", 2) => Ok(HalfPrior::Normal { scale: num(1)? }),
        _ => Err(bad()),
    }
}

fn half_prior_tuple<'py>(py: Python<'py>, prior: HalfPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        HalfPrior::StudentT { nu, scale } => ("student_t", nu, scale).into_pyobject(py),
        HalfPrior::Normal { scale } => ("normal", scale).into_pyobject(py),
    }
}

/// A real prior from ``("student_t", nu, location, scale)`` or
/// ``("normal", mean, sd)``.
fn real_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<RealPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, location, scale) or (\"normal\", mean, sd)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 4) => Ok(RealPrior::StudentT {
            nu: num(1)?,
            location: num(2)?,
            scale: num(3)?,
        }),
        ("normal", 3) => Ok(RealPrior::Normal {
            mean: num(1)?,
            sd: num(2)?,
        }),
        _ => Err(bad()),
    }
}

fn real_prior_tuple<'py>(py: Python<'py>, prior: RealPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        RealPrior::StudentT {
            nu,
            location,
            scale,
        } => ("student_t", nu, location, scale).into_pyobject(py),
        RealPrior::Normal { mean, sd } => ("normal", mean, sd).into_pyobject(py),
    }
}

fn log_sigma_priors(value: &[Bound<'_, PyTuple>]) -> PyResult<[RealPrior; 2]> {
    if value.len() != 2 {
        return Err(PyValueError::new_err(
            "log_sigma must hold two priors: log sigma_os, then log sigma_pd",
        ));
    }
    Ok([
        real_prior("log_sigma[0]", &value[0])?,
        real_prior("log_sigma[1]", &value[1])?,
    ])
}

/// An optional half prior: ``None`` (no such effect) or a half prior's
/// tuple.
fn optional_half_prior(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => {
            let t = v.cast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} must be None, (\"student_t\", nu, scale) or (\"normal\", scale)"
                ))
            })?;
            Ok(Some(half_prior(name, t)?))
        }
    }
}

fn optional_half_prior_tuple<'py>(
    py: Python<'py>,
    prior: Option<HalfPrior>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    prior.map(|p| half_prior_tuple(py, p)).transpose()
}

/// The development-year effect sds on ``k_er``, ``k_p``, ``rlr`` and
/// ``rrf``: ``None`` (none) or four items, each ``None`` or a half prior.
fn development_effect_sds(value: Option<&Bound<'_, PyAny>>) -> PyResult<[Option<HalfPrior>; 4]> {
    let Some(v) = value.filter(|v| !v.is_none()) else {
        return Ok([None; 4]);
    };
    let bad = || {
        PyValueError::new_err(
            "development_effect_sd must be None or four priors, on k_er, k_p, rlr and rrf in \
             that order, each None or a half prior",
        )
    };
    let items: Vec<Bound<'_, PyAny>> = v.try_iter().map_err(|_| bad())?.collect::<PyResult<_>>()?;
    if items.len() != 4 {
        return Err(bad());
    }
    let mut out = [None; 4];
    for (i, (slot, item)) in out.iter_mut().zip(&items).enumerate() {
        *slot = optional_half_prior(&format!("development_effect_sd[{i}]"), Some(item))?;
    }
    Ok(out)
}

/// ``replace``'s value for an optional effect: ``None`` keeps it,
/// ``False`` removes it, a half prior's tuple sets it.
fn replace_effect(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
    current: Option<HalfPrior>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(current),
        Some(v) if v.is_none() => Ok(current),
        Some(v) if v.is_instance_of::<PyBool>() => {
            if v.extract::<bool>()? {
                Err(PyValueError::new_err(format!(
                    "{name} must be False (no effect) or a half prior, not True"
                )))
            } else {
                Ok(None)
            }
        }
        Some(v) => optional_half_prior(name, Some(v)),
    }
}

/// The priors of ``Compartmental``, a required argument: the rates and
/// ratios are specific to a line and a book, so the library does not choose
/// them silently. ``CompartmentalPriors.gesmann_morris()`` is the paper's
/// model 1 and ``CompartmentalPriors.gesmann_morris_2()`` its model 2;
/// check any set with ``Compartmental.prior_predictive`` before fitting.
/// ``CompartmentalPriors.diffuse()`` is the starting point for a fit to paid alone
/// (``fitted="paid_only"``): it leaves the rates to the data. A fit to incurred alone
/// (``fitted="incurred_only"``) refuses a ``k_p`` prior with a log-scale sd above 0.5,
/// so give it a prior you believe, such as the paper's. Use a narrower prior than the
/// preset also when you hold information about the rates that is not in the data.
///
/// Each positive parameter ``θ`` (``k_er``, ``k_p``, ``rlr``, ``rrf``) is
/// ``median * exp(sd * o)`` with ``o ~ N(0, 1)``: a log-normal prior with
/// that median and log-scale sd, given as ``(median, sd)``. The effects
/// live on that ``o`` scale.
///
/// The hierarchy follows from the priors. ``RLR`` and ``RRF`` always vary
/// by origin, with correlated effects. ``ker_effect_sd`` and
/// ``kp_effect_sd`` add origin effects on the rates, uncorrelated with the
/// others, and ``development_effect_sd`` adds development-year effects on
/// any of the four parameters; ``None`` (model 1) means no such effect.
/// A development age with no observed cell takes a new effect from the
/// population in each simulation, as an origin with no cell does.
///
/// Parameters
/// ----------
/// k_er, k_p : tuple of (float, float)
///     ``(median, sd)`` of the reporting rate and the payment rate, per year.
/// rlr, rrf : tuple of (float, float)
///     ``(median, sd)`` of the population reported loss ratio and reserve
///     robustness factor (paid over reported).
/// rlr_effect_sd, rrf_effect_sd : tuple
///     Priors on the sds of the origin effects on the ``o`` scale:
///     ``("student_t", nu, scale)`` for a half Student-t or
///     ``("normal", scale)`` for a half normal.
/// lkj_eta : float
///     LKJ shape of the correlation of the two effects (1: uniform).
/// log_sigma : sequence of two tuples
///     Priors on ``log sigma_os`` and ``log sigma_pd``, in that order, each
///     ``("student_t", nu, location, scale)`` or ``("normal", mean, sd)``.
/// ker_effect_sd, kp_effect_sd : tuple, optional
///     Priors on the sds of origin effects on ``k_er`` and ``k_p``, in the
///     form of ``rlr_effect_sd``; ``None``: no such effect.
/// development_effect_sd : sequence of four, optional
///     Priors on the sds of development-year effects on ``k_er``, ``k_p``,
///     ``rlr`` and ``rrf``, in that order, each ``None`` (no effect) or a
///     half prior; ``None``: no development effect.
///
/// Raises
/// ------
/// ValueError
///     If a prior has the wrong form or a log-normal median or sd is not
///     finite and positive. The other settings are checked by the fit.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CompartmentalPriors
/// >>> p = CompartmentalPriors.gesmann_morris()
/// >>> p.k_er, p.rlr_effect_sd, p.log_sigma[0]
/// ((3.0, 0.1), ('student_t', 10.0, 0.2), ('student_t', 1.0, 0.0, 1000.0))
/// >>> p.replace(k_er=(2.0, 0.3)).k_er
/// (2.0, 0.3)
/// >>> q = CompartmentalPriors.gesmann_morris_2()
/// >>> q.kp_effect_sd, q.development_effect_sd[2]
/// (('student_t', 10.0, 0.3), ('student_t', 10.0, 0.7))
/// >>> q.replace(kp_effect_sd=False).kp_effect_sd is None
/// True
#[pyclass(name = "CompartmentalPriors", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalPriors {
    inner: CompartmentalPriors,
}

#[pymethods]
impl PyCompartmentalPriors {
    #[new]
    #[pyo3(signature = (
        *,
        k_er,
        k_p,
        rlr,
        rrf,
        rlr_effect_sd,
        rrf_effect_sd,
        lkj_eta,
        log_sigma,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_er: (f64, f64),
        k_p: (f64, f64),
        rlr: (f64, f64),
        rrf: (f64, f64),
        rlr_effect_sd: Bound<'_, PyTuple>,
        rrf_effect_sd: Bound<'_, PyTuple>,
        lkj_eta: f64,
        log_sigma: Vec<Bound<'_, PyTuple>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CompartmentalPriors {
                k_er: positive_prior("k_er", k_er)?,
                k_p: positive_prior("k_p", k_p)?,
                rlr: positive_prior("rlr", rlr)?,
                rrf: positive_prior("rrf", rrf)?,
                rlr_effect_sd: half_prior("rlr_effect_sd", &rlr_effect_sd)?,
                rrf_effect_sd: half_prior("rrf_effect_sd", &rrf_effect_sd)?,
                lkj_eta,
                log_sigma: log_sigma_priors(&log_sigma)?,
                ker_effect_sd: optional_half_prior("ker_effect_sd", ker_effect_sd.as_ref())?,
                kp_effect_sd: optional_half_prior("kp_effect_sd", kp_effect_sd.as_ref())?,
                development_effect_sd: development_effect_sds(development_effect_sd.as_ref())?,
            },
        })
    }

    /// Gesmann and Morris' (2020) case study, model 1: ``k_er`` (3, 0.1),
    /// ``k_p`` (1, 0.1), ``rlr`` (0.7, 0.2), ``rrf`` (0.8, 0.1), half
    /// Student-t(10) effect sds with scales 0.2 and 0.1, LKJ(1), and
    /// Student-t(1, 0, 1000) on each ``log sigma`` (effectively flat), with
    /// no effect on the rates and no development effect. On company 337 the
    /// data pull the rates 5 to 9 prior sds from these medians: the
    /// posterior is a compromise with the preset.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris(),
        }
    }

    /// A diffuse set for model 1: the paper's medians (``k_er`` 3, ``k_p`` 1,
    /// ``rlr`` 0.7, ``rrf`` 0.8) with a log-scale sd of 1 on each of the four,
    /// 5 to 10 times the paper's, and its effect sds and ``log sigma`` priors.
    /// Use it as the starting point for a fit to paid alone
    /// (``fitted="paid_only"``), where the paid cells pin the rates. It is
    /// refused for a fit to incurred alone (``fitted="incurred_only"``): no
    /// paid cell is fitted there, so the payment rate ``k_p`` and with it the
    /// paid run-off and the reserve follow the prior on ``k_p`` at every width
    /// (on company 337 -51,000 with an sd of about 150,000 at a log-scale sd of
    /// 1, 203,000 at 0.3), and a ``k_p`` prior with a log-scale sd above 0.5 is
    /// an error. For incurred alone replace ``k_p`` with a prior you believe
    /// (``replace``), for example one from the book's paid history or from a
    /// paid-and-incurred fit of a similar book; the paper's narrow prior put
    /// the reserve on company 337 26% above the fit to paid and outstanding.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn diffuse() -> Self {
        Self {
            inner: CompartmentalPriors::diffuse(),
        }
    }

    /// Gesmann and Morris' (2020) case study, model 2 (their code, sec.
    /// 7.2.3, where it and the text disagree): model 1's rate and ratio
    /// priors; origin effects on all four parameters (``RLR`` and ``RRF``
    /// correlated, LKJ(1)) and development-year effects on all four, every
    /// effect sd half Student-t(10) on the ``o`` scale with one scale per
    /// parameter, shared by its two groups: 0.3 for ``k_er`` and ``k_p``,
    /// 0.7 for ``rlr`` and 0.5 for ``rrf``; and ``("normal", log 0.2,
    /// 0.2)`` on each ``log sigma`` (the text gives a median of 0.1). Use
    /// it with ``Compartmental.model_2``.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris_2() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris_2(),
        }
    }

    /// A copy with the given priors replaced, each in the constructor's
    /// form. For ``ker_effect_sd`` and ``kp_effect_sd``, ``None`` keeps the
    /// current prior and ``False`` removes the effect; a
    /// ``development_effect_sd`` replaces all four, its ``None`` items
    /// removing those effects.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[pyo3(signature = (
        *,
        k_er = None,
        k_p = None,
        rlr = None,
        rrf = None,
        rlr_effect_sd = None,
        rrf_effect_sd = None,
        lkj_eta = None,
        log_sigma = None,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        k_er: Option<(f64, f64)>,
        k_p: Option<(f64, f64)>,
        rlr: Option<(f64, f64)>,
        rrf: Option<(f64, f64)>,
        rlr_effect_sd: Option<Bound<'_, PyTuple>>,
        rrf_effect_sd: Option<Bound<'_, PyTuple>>,
        lkj_eta: Option<f64>,
        log_sigma: Option<Vec<Bound<'_, PyTuple>>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut p = self.inner;
        if let Some(v) = k_er {
            p.k_er = positive_prior("k_er", v)?;
        }
        if let Some(v) = k_p {
            p.k_p = positive_prior("k_p", v)?;
        }
        if let Some(v) = rlr {
            p.rlr = positive_prior("rlr", v)?;
        }
        if let Some(v) = rrf {
            p.rrf = positive_prior("rrf", v)?;
        }
        if let Some(v) = rlr_effect_sd {
            p.rlr_effect_sd = half_prior("rlr_effect_sd", &v)?;
        }
        if let Some(v) = rrf_effect_sd {
            p.rrf_effect_sd = half_prior("rrf_effect_sd", &v)?;
        }
        if let Some(v) = lkj_eta {
            p.lkj_eta = v;
        }
        if let Some(v) = log_sigma {
            p.log_sigma = log_sigma_priors(&v)?;
        }
        p.ker_effect_sd = replace_effect("ker_effect_sd", ker_effect_sd.as_ref(), p.ker_effect_sd)?;
        p.kp_effect_sd = replace_effect("kp_effect_sd", kp_effect_sd.as_ref(), p.kp_effect_sd)?;
        if let Some(v) = development_effect_sd.filter(|v| !v.is_none()) {
            p.development_effect_sd = development_effect_sds(Some(&v))?;
        }
        Ok(Self { inner: p })
    }

    /// ``(median, sd)`` of the reporting rate.
    #[getter]
    fn k_er(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_er)
    }

    /// ``(median, sd)`` of the payment rate.
    #[getter]
    fn k_p(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_p)
    }

    /// ``(median, sd)`` of the population reported loss ratio.
    #[getter]
    fn rlr(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rlr)
    }

    /// ``(median, sd)`` of the population reserve robustness factor.
    #[getter]
    fn rrf(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rrf)
    }

    /// Prior on the sd of the origin effects on ``RLR``.
    #[getter]
    fn rlr_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rlr_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``RRF``.
    #[getter]
    fn rrf_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rrf_effect_sd)
    }

    /// LKJ shape of the correlation of the two effects.
    #[getter]
    fn lkj_eta(&self) -> f64 {
        self.inner.lkj_eta
    }

    /// Priors on ``log sigma_os`` and ``log sigma_pd``.
    #[getter]
    fn log_sigma<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .log_sigma
            .iter()
            .map(|&p| real_prior_tuple(py, p))
            .collect()
    }

    /// Prior on the sd of the origin effects on ``k_er``; ``None``: no such
    /// effect.
    #[getter]
    fn ker_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.ker_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``k_p``; ``None``: no such
    /// effect.
    #[getter]
    fn kp_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.kp_effect_sd)
    }

    /// Priors on the sds of the development-year effects on ``k_er``,
    /// ``k_p``, ``rlr`` and ``rrf``, each ``None`` when there is no such
    /// effect.
    #[getter]
    fn development_effect_sd<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Option<Bound<'py, PyTuple>>>> {
        self.inner
            .development_effect_sd
            .iter()
            .map(|&p| optional_half_prior_tuple(py, p))
            .collect()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        let p = &self.inner;
        let half = |h: HalfPrior| match h {
            HalfPrior::StudentT { nu, scale } => format!("(\"student_t\", {nu:?}, {scale:?})"),
            HalfPrior::Normal { scale } => format!("(\"normal\", {scale:?})"),
        };
        let maybe = |h: Option<HalfPrior>| h.map_or_else(|| "None".to_string(), half);
        let real = |r: RealPrior| match r {
            RealPrior::StudentT {
                nu,
                location,
                scale,
            } => format!("(\"student_t\", {nu:?}, {location:?}, {scale:?})"),
            RealPrior::Normal { mean, sd } => format!("(\"normal\", {mean:?}, {sd:?})"),
        };
        let pos = |v: PositivePrior| format!("{:?}", positive_prior_tuple(v));
        let dev = if p.development_effect_sd.iter().all(Option::is_none) {
            "None".to_string()
        } else {
            let items: Vec<String> = p.development_effect_sd.iter().map(|&h| maybe(h)).collect();
            format!("[{}]", items.join(", "))
        };
        format!(
            "CompartmentalPriors(k_er={}, k_p={}, rlr={}, rrf={}, rlr_effect_sd={}, \
             rrf_effect_sd={}, lkj_eta={:?}, log_sigma=[{}, {}], ker_effect_sd={}, \
             kp_effect_sd={}, development_effect_sd={})",
            pos(p.k_er),
            pos(p.k_p),
            pos(p.rlr),
            pos(p.rrf),
            half(p.rlr_effect_sd),
            half(p.rrf_effect_sd),
            p.lkj_eta,
            real(p.log_sigma[0]),
            real(p.log_sigma[1]),
            maybe(p.ker_effect_sd),
            maybe(p.kp_effect_sd),
            dev,
        )
    }
}

/// A setting given as a string, with its accepted names in order.
fn named<T: Copy>(setting: &str, name: &str, values: &[(&str, T)]) -> PyResult<T> {
    let lower = name.to_ascii_lowercase();
    values
        .iter()
        .find(|(n, _)| *n == lower)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let expected: Vec<String> = values.iter().map(|(n, _)| format!("{n:?}")).collect();
            PyValueError::new_err(format!(
                "unknown {setting} {name:?}; expected one of {}",
                expected.join(", ")
            ))
        })
}

/// The name of a setting's value.
fn name_of<T: PartialEq>(value: T, values: &[(&'static str, T)]) -> &'static str {
    values
        .iter()
        .find(|(_, v)| *v == value)
        .map_or("?", |(n, _)| n)
}

const LIKELIHOODS: [(&str, CompartmentalLikelihood); 2] = [
    ("normal", CompartmentalLikelihood::Normal),
    ("lognormal", CompartmentalLikelihood::LogNormal),
];
const SCALES: [(&str, CompartmentalScale); 2] = [
    ("amount", CompartmentalScale::Amount),
    ("loss_ratio", CompartmentalScale::LossRatio),
];
const PAID_RESPONSES: [(&str, PaidResponse); 2] = [
    ("cumulative", PaidResponse::Cumulative),
    ("incremental", PaidResponse::Incremental),
];
const FITTED: [(&str, Fitted); 4] = [
    ("paid_and_outstanding", Fitted::PaidAndOutstanding),
    ("paid_and_incurred", Fitted::PaidAndIncurred),
    ("paid_only", Fitted::PaidOnly),
    ("incurred_only", Fitted::IncurredOnly),
];
const RATE_ORDERS: [(&str, RateOrder); 2] =
    [("free", RateOrder::Free), ("ordered", RateOrder::Ordered)];
const STARTS: [(&str, CompartmentalStart); 2] = [
    ("data", CompartmentalStart::Data),
    ("prior_centre", CompartmentalStart::PriorCentre),
];

/// Gesmann and Morris' (2020) hierarchical compartmental reserving model
/// with one outstanding compartment, their models 1 and 2
/// (``docs/design/reserving-compartmental.md``). Claims flow from exposure
/// into outstanding as they are reported, at rate ``k_er``, and from
/// outstanding into paid as they settle, at rate ``k_p``. For origin ``i``
/// with premium ``P_i`` and development time ``t`` in years, with
/// ``g(t) = (exp(-k_p t) - exp(-k_er t)) / (k_er - k_p)``::
///
///     OS_i(t) = P_i RLR_i k_er g(t)
///     PD_i(t) = P_i RLR_i RRF_i [(1 - exp(-k_p t)) - k_p g(t)]
///
/// so ``RLR_i`` is the reported loss ratio, ``RRF_i`` the reserve
/// robustness factor (paid over reported) and ``RLR_i RRF_i`` the ultimate
/// loss ratio. ``RLR`` and ``RRF`` vary by origin with correlated effects
/// (LKJ prior on the correlation) on the priors' ``o`` scale; the priors
/// can add origin effects on the rates and development-year effects on any
/// of the four (``CompartmentalPriors``), each cell then taking its own
/// parameters. NUTS samples the posterior on a pilot run's dense scale,
/// with the gradient from dual numbers.
///
/// Model 1 (the defaults) has normal outstanding and cumulative paid
/// amounts about these curves, each type with its own sigma. Model 2
/// (``Compartmental.model_2``, with
/// ``CompartmentalPriors.gesmann_morris_2()``) has log-normal outstanding
/// and incremental paid loss ratios to premium; every combination of
/// ``likelihood``, ``scale`` and ``paid_response`` is built. Under
/// ``likelihood="lognormal"`` every response must be positive, and under
/// ``paid_response="incremental"`` each paid cell needs the paid at the
/// age before; ``fit`` raises ``ValueError`` naming the cell otherwise.
/// ``fitted="paid_and_incurred"`` fits paid and incurred, with incurred in place of
/// outstanding as the second response: the mean of an incurred cell is ``RLR_i (OS(t) +
/// RRF_i PD(t))`` times premium (or as a loss ratio), with the cumulative ``PD`` even
/// under ``paid_response="incremental"``, and ``log_sigma_os`` is the incurred sigma.
/// Give the incurred column as ``incurred=``, or an outstanding column as
/// ``outstanding=`` and the incurred is outstanding plus paid where both are observed.
/// The fit's ``outstanding`` then holds the future incurred cells.
///
/// ``fitted="paid_only"`` and ``fitted="incurred_only"`` fit one response alone and need
/// ``rate_order="ordered"`` (``ValueError`` under ``"free"``): from one response the means
/// do not change under the rate swap below, and from paid alone only ``RLR x RRF`` is
/// identified, so the priors carry the split of ``RLR`` and ``RRF``. The model then has one
/// sigma, ``sigma_pd`` or ``sigma_os``. Under ``"paid_only"`` the second column is not read
/// (``outstanding=`` and ``incurred=`` may both be left out) and the fit has no
/// ``outstanding`` (``None``). Under ``"incurred_only"`` the incurred is given as for
/// ``"paid_and_incurred"`` and the paid column supplies each origin's latest paid only: the
/// reserve is the expected paid at the last age less the latest paid, and the future paid
/// cells are expected values, without noise, since no paid cell is fitted.
///
/// ``CompartmentalPriors.diffuse()`` is the starting point for ``"paid_only"``. Under
/// ``"incurred_only"`` no paid cell is fitted, so the incurred does not identify the payment
/// rate ``k_p`` and its prior sets the paid run-off and the reserve at every width (on company
/// 337 the paper's narrow prior put the reserve 26% above the fit to paid and outstanding):
/// the fit refuses a ``k_p`` prior with a log-scale sd above 0.5, so the diffuse preset is an
/// error there, and you give a ``k_p`` prior you believe, for example from the book's paid
/// history or from a paid-and-incurred fit of a similar book.
///
/// An origin with a premium and no observed cell (the papers' training
/// triangle has one) draws its effects from the population in each
/// simulation, and so does a development age with no observed cell. The
/// reserve is the paid at the square's last age less the latest paid; the
/// square runs for as many origin periods as the premium column has
/// origins, in development periods (40 quarterly ages for ten annual
/// origins). The development grain may be finer than the origin grain
/// (the triangle's own grains), and each cell's development time is its
/// age in years on every grain, so the rates are per year and an
/// incremental paid cell is the increment over one development period.
/// The coming year of ``one_year`` is the next twelve months of calendar
/// time, up to four quarterly cells per origin. One ``seed`` replays a fit
/// exactly: the chains and the simulations each take a seed derived from
/// it.
///
/// The means are exactly invariant under swapping the two rates (with
/// ``RLR`` and ``RRF`` rescaled), so only the priors tell the two modes
/// apart. Under ``rate_order="free"`` (the paper's priors) the fit warns
/// when any posterior draw has population rates with ``k_er <= k_p``;
/// ``rate_order="ordered"`` samples ``k_er = k_p + exp(delta)``, the same
/// priors restricted to ``k_er > k_p``. With effects on the rates,
/// ``CompartmentalFit.cell_rate_swaps`` reports the share of draws and
/// cells with ``k_er <= k_p``.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) and rate swaps do not fail the
/// fit: ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``CompartmentalFit.diagnostics``. With the defaults, 194 of 200 seeds on
/// company 337 passed every threshold for model 1, and with
/// ``model_2``'s sampler 195 of 200 for model 2; each of the others just
/// missed one.
///
/// Parameters
/// ----------
/// priors : CompartmentalPriors
///     The priors; required.
/// likelihood : str
///     ``"normal"`` (model 1) or ``"lognormal"`` (model 2).
/// scale : str
///     The response scale: ``"amount"`` (model 1) or ``"loss_ratio"``,
///     amounts over the origin's premium (model 2).
/// paid_response : str
///     ``"cumulative"`` (model 1) or ``"incremental"`` (model 2).
/// fitted : str
///     ``"paid_and_outstanding"``, ``"paid_and_incurred"`` (incurred replaces
///     outstanding as the second response, as above), ``"paid_only"`` or
///     ``"incurred_only"`` (one response alone, with ``rate_order="ordered"``).
/// rate_order : str
///     ``"free"`` or ``"ordered"``, as above.
/// start : str
///     Where the pilot run starts: ``"data"``, a least-squares fit of the
///     population curve (the prior centre if that fit fails, and under one
///     response if its log posterior is below the prior centre's), or
///     ``"prior_centre"``.
/// chains, tune, draws : int
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int
///     Number of predictive simulations; positive.
/// seed : int
///     Seed of the chains and of the simulations.
/// target_accept : float
///     Target mean acceptance rate (the paper's ``adapt_delta``, 0.99).
/// max_depth : int
///     Largest NUTS tree depth.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
/// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
/// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
/// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
/// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
/// >>> rows = [(1988 + w, 12 * (d + 1), paid[w][d], incurred[w][d], premium[w])
/// ...         for w in range(5) for d in range(len(paid[w]))]
/// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": pd, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2,
/// ...                       tune=150, draws=150, target_accept=0.9, n_sims=1000)
/// >>> # Check the priors first: the mean paid of 1988 at age 60 over its premium.
/// >>> paths = model.prior_predictive(tri, "premium", n_sims=1000)
/// >>> ratio = paths["paid"].marginal(("1988", 60))
/// >>> 0.2 < ratio.quantile(0.05) / 104.4 and ratio.quantile(0.95) / 104.4 < 1.2
/// True
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = model.fit(tri, "paid", "premium", incurred="incurred")
/// >>> fit.reserves.dims, fit.parameter_names[0]
/// (['origin'], 'k_er')
#[pyclass(name = "Compartmental", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmental {
    inner: Compartmental,
}

#[pymethods]
impl PyCompartmental {
    #[new]
    #[pyo3(signature = (
        priors,
        *,
        likelihood = "normal",
        scale = "amount",
        paid_response = "cumulative",
        fitted = "paid_and_outstanding",
        rate_order = "free",
        start = "data",
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.99,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        priors: PyRef<'_, PyCompartmentalPriors>,
        likelihood: &str,
        scale: &str,
        paid_response: &str,
        fitted: &str,
        rate_order: &str,
        start: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: Compartmental {
                priors: priors.inner,
                likelihood: named("likelihood", likelihood, &LIKELIHOODS)?,
                scale: named("scale", scale, &SCALES)?,
                paid_response: named("paid_response", paid_response, &PAID_RESPONSES)?,
                fitted: named("fitted", fitted, &FITTED)?,
                rate_order: named("rate_order", rate_order, &RATE_ORDERS)?,
                start: named("start", start, &STARTS)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// Gesmann and Morris' model 2 with ``priors`` (the paper's are
    /// ``CompartmentalPriors.gesmann_morris_2()``): the log-normal
    /// likelihood on loss ratios with incremental paid, and a longer, finer
    /// sampler than the constructor's, four chains of 2,000 warm-up and
    /// 8,000 kept draws with target acceptance 0.999. With 87 parameters,
    /// effect sds whose posteriors pile up near 0 and a slowly mixing
    /// ``k_p``, model 1's sampler left divergent transitions or an R-hat
    /// above 1.01 on most seeds; this one passes every threshold on 195 of
    /// 200 seeds on company 337. A fit takes about a minute. The other
    /// arguments are the constructor's.
    ///
    /// Returns
    /// -------
    /// Compartmental
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors
    /// >>> m = Compartmental.model_2(CompartmentalPriors.gesmann_morris_2())
    /// >>> m.likelihood, m.scale, m.paid_response, m.tune, m.draws, m.target_accept
    /// ('lognormal', 'loss_ratio', 'incremental', 2000, 8000, 0.999)
    #[staticmethod]
    #[pyo3(signature = (
        priors,
        *,
        rate_order = "free",
        start = "data",
        chains = 4,
        tune = 2000,
        draws = 8000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.999,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn model_2(
        priors: PyRef<'_, PyCompartmentalPriors>,
        rate_order: &str,
        start: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        Self::new(
            priors,
            "lognormal",
            "loss_ratio",
            "incremental",
            "paid_and_outstanding",
            rate_order,
            start,
            chains,
            tune,
            draws,
            n_sims,
            seed,
            target_accept,
            max_depth,
            dependence,
            spearman,
        )
    }

    /// The priors.
    #[getter]
    fn priors(&self) -> PyCompartmentalPriors {
        PyCompartmentalPriors {
            inner: self.inner.priors,
        }
    }

    /// The likelihood: ``"normal"`` or ``"lognormal"``.
    #[getter]
    fn likelihood(&self) -> &'static str {
        name_of(self.inner.likelihood, &LIKELIHOODS)
    }

    /// The response scale: ``"amount"`` or ``"loss_ratio"``.
    #[getter]
    fn scale(&self) -> &'static str {
        name_of(self.inner.scale, &SCALES)
    }

    /// The paid response: ``"cumulative"`` or ``"incremental"``.
    #[getter]
    fn paid_response(&self) -> &'static str {
        name_of(self.inner.paid_response, &PAID_RESPONSES)
    }

    /// The responses fitted.
    #[getter]
    fn fitted(&self) -> &'static str {
        name_of(self.inner.fitted, &FITTED)
    }

    /// How the rates are kept apart: ``"free"`` or ``"ordered"``.
    #[getter]
    fn rate_order(&self) -> &'static str {
        name_of(self.inner.rate_order, &RATE_ORDERS)
    }

    /// Where the pilot run starts: ``"data"`` or ``"prior_centre"``.
    #[getter]
    fn start(&self) -> &'static str {
        name_of(self.inner.start, &STARTS)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How ``fit_segments`` joins the segments: ``"independent"`` or
    /// ``"rank_correlation"`` (``"synchronized"`` is refused by
    /// ``fit_segments``). ``fit`` ignores it.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of paid and outstanding of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value), and simulates the future cells of the square and
    /// the reserves. Outstanding comes from ``outstanding``, a column of
    /// outstanding amounts, or from ``incurred``, a column of incurred
    /// amounts less paid cell by cell (the papers' outstanding on company
    /// 337 is incurred with bulk and IBNR less paid); give exactly one, or
    /// neither under ``fitted="paid_only"``, which does not read it.
    /// Raises one ``UserWarning`` listing the failed diagnostics and any
    /// rate swap.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin; the fit's origins are this column's.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``fitted`` is ``"paid_only"`` or ``"incurred_only"`` and
    ///     ``rate_order`` is not ``"ordered"``; if not exactly one of
    ///     ``outstanding`` and ``incurred`` is given (at most one under
    ///     ``"paid_only"``); if a column is
    ///     missing, the triangle has several segments, or a premium is
    ///     missing or not positive; if a prior or sampler setting is
    ///     invalid; or if there are no more observed cells than parameters.
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, paid, os, premium))
            .map_err(err)?;
        warn_diagnostics(py, "Compartmental", &inner.diagnostics.warnings)?;
        Ok(PyCompartmentalFit { inner })
    }

    /// Samples the posterior of paid and outstanding in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into joint distributions
    /// of the reserves and of the future paid and outstanding cells, the
    /// segments depending on each other as ``dependence`` says. Under
    /// ``"rank_correlation"`` each segment's reserves, future paid cells and
    /// future outstanding cells are reordered together, ranked on the
    /// reserves, so every output of a segment stays consistent. Segment
    /// ``k`` (in index order) takes seeds derived from ``seed`` and ``k``
    /// for its chains and its simulations, so independent segments share no
    /// random numbers, and a segment's draws differ from ``fit`` of that
    /// segment alone. Raises one ``UserWarning`` listing the failed
    /// diagnostics and any rate swap of every segment, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = model.fit_segments(tri, "paid", "premium", outstanding="case")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, paid, os, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "Compartmental", &warnings)?;
        Ok(PyCompartmentalFits { inner })
    }

    /// The one-year view: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 20 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the paid and outstanding of
    /// the cells valued in the next twelve months from it (every origin's
    /// next diagonal), reweights every posterior draw by the likelihood of
    /// those cells, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is the
    /// expected paid to the square's last age given the observed cells,
    /// less the year's paid and the same expectation given the year too;
    /// its mean is about zero. An origin with no observed cell (1997 on
    /// the papers' triangle) and, with development effects (model 2), an
    /// age with none take their effects as part of each posterior draw, so
    /// the year's cells update them. A simulation whose ``k`` is above
    /// ``k_threshold`` is flagged and handled as ``refit`` says. On company
    /// 337 about four in five of model 1's simulations are flagged and all
    /// of model 2's; for model 1 moment matching rescues about nineteen in
    /// twenty of them, for model 2 none, and a refit takes as long as a fit
    /// (for model 2 about 31 s of wall time with 8 running side by side).
    /// So by default (``refit=None``, the user's decisions of 2026-10-16 and
    /// 2026-10-21) model 1 moment-matches its flagged simulations and refits
    /// with NUTS only those whose ``k`` stays above the threshold, and a
    /// model with development effects (model 2) refits them with the light
    /// refit (``light_refit``): a warm-started NUTS run of 4 chains, 150
    /// warm-up and 250 kept draws, within about 0.03 sd of the total claims
    /// development result of a full refit, at about 20 times the run time
    /// of reweighting only (about 30 minutes for the default 2,000
    /// simulations on company 337). ``light_refit=False`` turns it off:
    /// model 2 then reweights its flagged simulations only, which is
    /// approximate (about 0.4 sd off a refit per simulation) and warns so
    /// (``approximate``). An explicit ``refit`` is used as it is, with the
    /// fit's own sampler unless ``light_refit`` is given, with
    /// ``max_refits`` as a budget. The view runs at most 2,000 simulations
    /// by default (``n_sims``).
    ///
    /// Raises one ``UserWarning`` listing the failed diagnostics of the
    /// fit, another if any refit failed its diagnostics, another if
    /// ``max_refits`` left simulations unrefitted, and another if the
    /// view is approximate (model 2 with ``light_refit=False``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    ///     ``None`` (the default) takes the model's own:
    ///     ``"moment_match_then_refit"`` without development effects (model
    ///     1), ``"refit_flagged"`` with the light refit with them (model 2;
    ///     ``light_refit=False`` makes it ``"reweight_only"``, with a
    ///     warning).
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged. The lifetime view in ``fit`` keeps
    ///     ``n_sims``; a smaller count gives the first simulations of a
    ///     larger one.
    /// light_refit : bool or tuple, optional
    ///     The sampler of the refits: a shorter, warm-started NUTS run
    ///     instead of the fit's own. ``None`` (the default) is the model's
    ///     own: the light refit for model 2 when ``refit`` is ``None``, the
    ///     fit's own sampler otherwise. ``True`` takes 4 chains of 150
    ///     warm-up and 250 kept draws, a tuple ``(chains, tune, draws)`` its
    ///     own. ``False`` turns model 2's default off: with ``refit=None``
    ///     the view reweights only and is approximate, with a warning. Each
    ///     refit scales its sampler by the opening posterior and starts
    ///     there, with no pilot run. On company 337 it comes within about
    ///     0.03 sd of the total claims development result of a full refit
    ///     of model 2, at about a thirtieth of the cost. A refit warns when
    ///     its closing reserve has R-hat above 1.05, fewer than 100
    ///     effective draws or a divergence.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
    /// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
    /// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
    /// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
    /// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d], premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": pd, "incurred": inc, "premium": prem})
    /// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2, tune=150,
    /// ...                       draws=150, target_accept=0.9, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = model.one_year(tri, "paid", "premium", incurred="incurred",
    /// ...                           refit="reweight_only")
    /// >>> view.cdr.components()[:2]
    /// [('1988',), ('1989',)]
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, paid, os, premium, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "Compartmental", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Compartmental",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFit { inner })
    }

    /// The one-year view of every segment of a triangle: each segment's
    /// view as ``one_year`` gives it, on the segment's own seeds as in
    /// ``fit_segments`` (a segment's simulations differ from
    /// ``one_year`` of the segment alone), and one joint claims development
    /// result across segments, which depend on each other as ``dependence``
    /// says: ``"independent"`` as drawn, ``"rank_correlation"`` by
    /// Iman-Conover on the segments' total claims development results.
    /// Everything else is as ``one_year``: the refit default of the model
    /// (moment matching for model 1, the light refit for model 2, or
    /// reweighting only with a warning with ``light_refit=False``), at most 2,000 simulations by default, and ``max_refits``
    /// as a budget of each segment, so ``k`` segments refit at most ``k``
    /// times this many. Raises one ``UserWarning`` listing the failed
    /// diagnostics of each segment's fit (with the segment's label), another
    /// if any refit failed its diagnostics, another if ``max_refits`` left
    /// simulations unrefitted, and another if the view is approximate.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit_segments``.
    /// k_threshold : float, default 0.7
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    /// max_refits : int, optional
    /// n_sims : int, optional
    /// light_refit : bool or tuple, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(
    /// ...         tri, "paid", "premium", outstanding="case", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, paid, os, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.fit.diagnostics);
        warn_diagnostics(py, "Compartmental", &warnings)?;
        warn_refits(py, "Compartmental", &segment_refits(&inner.segments))?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFits { inner })
    }

    /// Paths of the mean outstanding and cumulative paid of each origin of
    /// the premium column at each age of the square, drawn from the priors
    /// alone (Gesmann and Morris, sec. 3.4): a check of a prior set before
    /// fitting. The paths are means, without the likelihood's noise, which
    /// under the preset's effectively flat prior on ``log sigma`` would
    /// swamp them. The seed is derived from ``seed``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// premium : str
    ///     The premium of each origin.
    /// n_sims : int
    ///     Number of paths; positive.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"outstanding": PredictiveDistribution, "paid":
    ///     PredictiveDistribution}``, each with dimensions ``origin`` and
    ///     ``age`` (months).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a prior setting is invalid, ``n_sims`` is 0, or the premium
    ///     column is missing or not positive.
    #[pyo3(signature = (triangle, premium, n_sims = 1000))]
    fn prior_predictive<'py>(
        &self,
        py: Python<'py>,
        triangle: PyRef<'_, PyTriangle>,
        premium: &str,
        n_sims: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let paths = py
            .detach(|| model.prior_predictive(tri, premium, n_sims))
            .map_err(err)?;
        let out = PyDict::new(py);
        out.set_item(
            "outstanding",
            PyPredictiveDistribution {
                inner: paths.outstanding,
            },
        )?;
        out.set_item("paid", PyPredictiveDistribution { inner: paths.paid })?;
        Ok(out)
    }

    /// This model as a candidate of ``diagonal_backtest``, with the
    /// columns it reads: the cumulative ``paid`` column (the backtest's
    /// ``column``), the outstanding (from ``outstanding`` or from
    /// ``incurred``, as ``fit``) and each origin's ``premium``. The
    /// backtest fits the model to the cells valued before each held-out
    /// diagonal and forecasts the paid increments of the diagonal; the
    /// candidate's ``outstanding_backtest`` scores the held-out outstanding
    /// of the same diagonals on the **same fits**. The settings are
    /// ``fit``'s, except that the backtest's ``n_sims`` and ``seed``
    /// replace ``n_sims`` and ``seed``, and the sampler runs once per
    /// split (the whole ``chains`` x ``draws`` posterior of ``fit``, so
    /// size them for the number of diagonals).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     The cumulative triangle with one segment. It must be the one
    ///     given to ``diagonal_backtest``, with ``paid`` as its ``column``.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalCandidate
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If not exactly one of ``outstanding`` and ``incurred`` is given
    ///     (at most one under ``fitted="paid_only"``).
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn candidate(
        &self,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalCandidate> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        Ok(PyCompartmentalCandidate {
            inner: CompartmentalCandidate::new(
                self.inner.clone(),
                &triangle.inner,
                paid,
                os,
                premium,
            ),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Compartmental(likelihood={:?}, scale={:?}, paid_response={:?}, fitted={:?}, \
             rate_order={:?}, start={:?}, chains={}, tune={}, draws={}, n_sims={}, seed={})",
            name_of(m.likelihood, &LIKELIHOODS),
            name_of(m.scale, &SCALES),
            name_of(m.paid_response, &PAID_RESPONSES),
            name_of(m.fitted, &FITTED),
            name_of(m.rate_order, &RATE_ORDERS),
            name_of(m.start, &STARTS),
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
        )
    }
}

/// A ``Compartmental`` for ``diagonal_backtest``, from
/// ``Compartmental.candidate``: the model's settings with the triangle and
/// the paid, outstanding (or incurred) and premium columns it reads.
///
/// Put it in the ``models`` of ``diagonal_backtest`` (a list, or a dict
/// that names it) to score the paid increments of the held-out diagonals
/// beside the other models, and call ``outstanding_backtest`` for the
/// second table, the held-out outstanding of the same diagonals. Both
/// tables use the same fits (one run of the sampler per split, kept by the
/// candidate), whichever is asked for first.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
/// ...         [983, 2211, 2830], [1657, 2685], [932]]
/// >>> incurred = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
/// ...             [1400, 1100, 700], [1200, 900], [1100]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214]
/// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d] + v, premium[w])
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": values, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris())
/// >>> cand = model.candidate(tri, "paid", "premium", incurred="incurred")
/// >>> cand.name, cand.paid, cand.premium, cand.outstanding_from
/// ('compartmental_gm1', 'paid', 'premium', 'incurred')
#[pyclass(
    name = "CompartmentalCandidate",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalCandidate {
    inner: CompartmentalCandidate,
}

#[pymethods]
impl PyCompartmentalCandidate {
    /// The model's name in a backtest: ``"compartmental_gm1"`` for model 1,
    /// ``"compartmental_gm2"`` for model 2, ``"compartmental"`` otherwise.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// The cumulative paid column, the backtest's ``column``.
    #[getter]
    fn paid(&self) -> String {
        self.inner.paid().to_string()
    }

    /// The premium column.
    #[getter]
    fn premium(&self) -> String {
        self.inner.premium().to_string()
    }

    /// The column the outstanding comes from.
    #[getter]
    fn outstanding(&self) -> String {
        match self.inner.outstanding() {
            Outstanding::Column(c) | Outstanding::FromIncurred(c) => c.to_string(),
        }
    }

    /// ``"column"`` when ``outstanding`` is a column of outstanding
    /// amounts, ``"incurred"`` when it is a column of incurred amounts less
    /// paid.
    #[getter]
    fn outstanding_from(&self) -> &'static str {
        match self.inner.outstanding() {
            Outstanding::Column(_) => "column",
            Outstanding::FromIncurred(_) => "incurred",
        }
    }

    /// Scores the held-out **outstanding** of the latest ``n_diagonals``
    /// calendar diagonals, a second table beside ``diagonal_backtest``'s
    /// paid one: for each, the model is fitted to the cells valued before
    /// it (the outstanding of the held-out diagonal is always withheld) and
    /// forecasts the outstanding of the diagonal's cells, scored by the
    /// backtest's metrics. The splits, ``n_sims``, ``seed`` and
    /// ``interval`` are ``diagonal_backtest``'s, so a backtest with the
    /// same arguments and this candidate (or a copy of it in its
    /// ``models``) has fitted the model already and the fits are reused.
    /// The one model of the table is named after the candidate with
    /// ``_outstanding``. A held-out cell without an outstanding in the
    /// triangle, or without a fitted origin or development period, is
    /// counted in ``excluded``.
    ///
    /// Parameters
    /// ----------
    /// n_diagonals : int
    ///     The number of latest calendar diagonals to hold out.
    /// n_sims : int, default 1000
    ///     Simulations per forecast.
    /// seed : int, default 0
    ///     Seed of every fit and simulation.
    /// interval : float, default 0.9
    ///     The central interval whose coverage is scored.
    ///
    /// Returns
    /// -------
    /// Backtest
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a diagonal cannot be held out, the model cannot be fitted or
    ///     a setting is invalid.
    #[pyo3(signature = (n_diagonals, *, n_sims = 1000, seed = 0, interval = 0.9))]
    fn outstanding_backtest(
        &self,
        py: Python<'_>,
        n_diagonals: usize,
        n_sims: usize,
        seed: u64,
        interval: f64,
    ) -> PyResult<PyBacktest> {
        let candidate = &self.inner;
        let (inner, diagonals) = py
            .detach(|| {
                let cells = TriangleFrame::new(candidate.triangle(), candidate.paid(), None)?;
                let bt =
                    candidate.outstanding_backtest(&cells, n_diagonals, n_sims, seed, interval)?;
                let diagonals = diagonal_labels(&cells, &bt);
                Ok((bt, diagonals))
            })
            .map_err(err)?;
        Ok(PyBacktest { inner, diagonals })
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalCandidate(name={}, paid={}, premium={})",
            quoted(self.inner.name()),
            quoted(self.inner.paid()),
            quoted(self.inner.premium())
        )
    }
}

/// Where a compartmental fit's outstanding comes from: exactly one of a
/// column of outstanding amounts and a column of incurred amounts, or
/// under ``fitted="paid_only"``, which does not read it, at most one.
fn compartmental_outstanding<'a>(
    fitted: Fitted,
    outstanding: Option<&'a str>,
    incurred: Option<&'a str>,
) -> PyResult<Outstanding<'a>> {
    match (outstanding, incurred) {
        (Some(c), None) => Ok(Outstanding::Column(c)),
        (None, Some(c)) => Ok(Outstanding::FromIncurred(c)),
        // Not read under paid alone.
        (None, None) if fitted == Fitted::PaidOnly => Ok(Outstanding::Column("")),
        _ => Err(PyValueError::new_err(
            "give exactly one of outstanding (a column of outstanding amounts) and \
             incurred (a column of incurred amounts); under fitted=\"paid_only\" neither \
             is needed",
        )),
    }
}

/// The one-year view of a ``Compartmental`` model, from
/// ``Compartmental.one_year``: the lifetime fit, the opening reserve, the
/// claims development result over the coming year and how reliable each
/// simulation's importance reweighting was, by the premium column's
/// origins.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``): the opening reserve less the year's
/// paid and the posterior mean reserve after the year; a negative value
/// is an adverse development. ``pareto_k`` holds one Pareto ``k`` per
/// simulation, ``flagged_share`` the share above ``k_threshold``;
/// ``moment_matched``, ``refitted`` and ``unrefitted`` list what happened
/// to the flagged simulations; ``approximate`` says why the view is
/// approximate when it reweighted them only by its model's default.
#[pyclass(
    name = "CompartmentalOneYearFit",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFit {
    inner: BayesOneYearFit<CompartmentalFit>,
}

#[pymethods]
impl PyCompartmentalOneYearFit {
    /// The model fitted to the observed triangle, as ``Compartmental.fit``
    /// gives it.
    #[getter]
    fn fit(&self) -> PyCompartmentalFit {
        PyCompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// expected paid to the square's last age less the latest paid.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// Joint distribution of the claims development result by origin.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Each simulation's Pareto ``k``, in order.
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        self.inner.refitted.clone()
    }

    /// The simulations moment matching rescued, in order (0-based).
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        self.inner.moment_matched.clone()
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead, in order (0-based).
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        self.inner.unrefitted.clone()
    }

    /// Why the view is approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// How many of the refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> usize {
        self.inner.refit_warnings
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``,
    /// ``opening_ultimate``, ``opening_reserve``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the totals of ``to_frame``'s columns, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalOneYearFit(origins={}, n_sims={}, flagged_share={:.3}, \
             moment_matched={}, refitted={}, unrefitted={})",
            self.inner.fit.origins.len(),
            self.inner.pareto_k.len(),
            self.inner.flagged_share(),
            self.inner.moment_matched.len(),
            self.inner.refitted.len(),
            self.inner.unrefitted.len(),
        )
    }
}

/// A fitted ``Compartmental``: the posterior, its MCMC diagnostics, and the
/// posterior predictive distributions of the reserves and of the future
/// paid and outstanding cells, all from the same simulations.
#[pyclass(name = "CompartmentalFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFit {
    inner: CompartmentalFit,
}

#[pymethods]
impl PyCompartmentalFit {
    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Each origin's premium.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    /// The origins with no observed cell, whose effects each simulation
    /// draws from the population.
    #[getter]
    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Joint posterior predictive distribution of each origin's reserve, the
    /// paid at the square's last age less the latest paid: dimension
    /// ``"origin"``, laid out as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Posterior predictive cumulative paid of each future cell of the
    /// square: dimensions ``"origin"`` and ``"age"`` (months).
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred, the second response)
    /// of each future cell of the square, laid out as ``paid``; ``None``
    /// under ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// Each origin's posterior mean ultimate loss ratio ``RLR_i RRF_i``,
    /// without process error past the square.
    #[getter]
    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    /// The number of posterior draws whose population rates have
    /// ``k_er <= k_p`` (0 under ``rate_order="ordered"``).
    #[getter]
    fn rate_swaps(&self) -> usize {
        self.inner.rate_swaps
    }

    /// With effects on the rates (each cell its own ``k_er`` and ``k_p``),
    /// the share of posterior draws and observed cells with
    /// ``k_er <= k_p``: a diagnostic, not a warning, since each cell's
    /// means are invariant under the swap on their own. ``None`` when the
    /// rates are shared.
    #[getter]
    fn cell_rate_swaps(&self) -> Option<f64> {
        self.inner.cell_rate_swaps
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold and one for rate swaps; empty when all is well).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(py, &self.inner.diagnostics)
    }

    /// Parameter names: on the natural scale ``k_er``, ``k_p``, ``RLR``,
    /// ``RRF`` (the population values, at a zero effect), ``sigma_os``,
    /// ``sigma_pd``, ``RLR[o]`` and ``RRF[o]`` for each origin ``o`` with
    /// an observed cell (at a zero development effect), with origin effects
    /// on the rates ``k_er[o]`` and ``k_p[o]`` too, and with development
    /// effects ``dev_ker[a]``, ``dev_kp[a]``, ``dev_RLR[a]`` and
    /// ``dev_RRF[a]``, each development age ``a`` (months) with an observed
    /// cell's effect on the ``o`` scale; then on the scale the paper prints
    /// ``b_ker``, ``b_kp``, ``b_RLR``, ``b_RRF``, ``tau_RLR``,
    /// ``tau_RRF`` (and ``tau_ker``, ``tau_kp``), ``rho`` (and
    /// ``tau_dev_ker``, ``tau_dev_kp``, ``tau_dev_RLR``, ``tau_dev_RRF``),
    /// ``log_sigma_os`` and ``log_sigma_pd``. Each name in parentheses is
    /// there only when the priors have that effect.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter, as ``MeyersFit.summary``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of the observed cells, outstanding cells first and then
    /// paid cells, with each cell's relative efficiency from the chains.
    /// The pointwise term is the full log density of each response, normal
    /// or log-normal (with its ``-log y`` term), as brms' ``log_lik``, on
    /// the response's scale: amounts or loss ratios, cumulative or
    /// incremental paid. Compare it only with fits of the same responses.
    /// ``loo_by_type`` splits the elpd.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// The elpd of ``loo`` split by type: a dict with ``n_outstanding``
    /// (the number of outstanding cells, which come first in the pointwise
    /// values), ``elpd_outstanding`` and ``elpd_paid``.
    ///
    /// Returns
    /// -------
    /// dict
    fn loo_by_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("n_outstanding", l.n_outstanding)?;
        out.set_item("elpd_outstanding", l.elpd_outstanding)?;
        out.set_item("elpd_paid", l.elpd_paid)?;
        Ok(out)
    }

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``, the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve, and the
    /// posterior mean ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the total ``premium`` and ``latest_paid``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per future paid cell: ``origin``, ``development`` (age in
    /// months), and the ``mean``, ``std_dev``, ``q05``, ``q50`` and ``q95``
    /// of the posterior predictive cumulative paid. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalFit(origins={}, draws={}, n_sims={}, rate_swaps={}, warnings={})",
            self.inner.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.rate_swaps,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Compartmental.fit_segments``: each segment's own posterior and the
/// joint posterior predictive distributions of the reserves and of the
/// future paid and outstanding cells.
///
/// ``reserves``, ``paid`` and ``outstanding`` have the triangle's keys as
/// their first dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists (``origins``,
/// ``premium``, ``latest_paid``) run over the origins of each segment in
/// turn, like the rows of ``to_frame()`` and the components of
/// ``reserves``. The posterior of one segment (``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()``, ``loo()``)
/// needs a single-segment fit: for several segments use ``segment(...)`` or
/// ``diagnostics_frame()``.
#[pyclass(name = "CompartmentalFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFits {
    inner: CompartmentalFits,
}

impl PyCompartmentalFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&CompartmentalSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalFits {
    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.premium.iter().copied())
            .collect()
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.latest_paid.iter().copied())
            .collect()
    }

    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the triangle's keys and ``"origin"`` are its dimensions,
    /// one component per segment and origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Posterior predictive cumulative paid of each future cell of every
    /// segment: dimensions the keys, ``"origin"`` and ``"age"`` (months),
    /// from the same simulations as ``reserves``.
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred) of each future cell of
    /// every segment, laid out as ``paid``; ``None`` under
    /// ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// The number of posterior draws with ``k_er <= k_p`` in each segment,
    /// in index order.
    #[getter]
    fn rate_swaps(&self) -> Vec<usize> {
        self.inner
            .segments
            .fits
            .iter()
            .map(|s| s.fit.rate_swaps)
            .collect()
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``CompartmentalFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, as
    /// ``CompartmentalFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = &self.one("loo()", "")?.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, the ``mean`` and ``std_dev`` of the
    /// posterior predictive reserve, and the posterior mean
    /// ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the total ``premium`` and
    /// ``latest_paid``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and future paid cell, as
    /// ``CompartmentalFit.paid_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint
    /// distributions (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let warnings: usize = segments
            .fits
            .iter()
            .map(|s| s.fit.diagnostics.warnings.len())
            .sum();
        format!(
            "CompartmentalFits({}origins={}, n_sims={}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// The one-year view of a ``Compartmental`` in every segment of a
/// triangle, from ``Compartmental.one_year_segments``: each segment's
/// posterior, opening reserve and Pareto ``k``, and the joint claims
/// development result across segments.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists (``origins``, ``premium``, ``latest_paid``,
/// ``opening_ultimate``, ``opening_reserve``) run over the origins of each
/// segment in turn, like the rows of ``to_frame()`` and the components of
/// ``cdr``. ``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()``, ``loo()``, ``pareto_k``, ``moment_matched``, ``refitted``,
/// ``unrefitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (which has the flagged shares) or
/// ``diagnostics_frame()``. ``approximate`` says why the views are
/// approximate when they reweighted the flagged simulations only by their
/// model's default (``light_refit=False``).
#[pyclass(
    name = "CompartmentalOneYearFits",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFits {
    inner: CompartmentalOneYearFits,
}

impl PyCompartmentalOneYearFits {
    fn one(
        &self,
        field: &str,
        instead: &str,
    ) -> PyResult<&BayesOneYearSegment<CompartmentalSegment>> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalOneYearFits {
    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.premium.clone())
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.latest_paid.clone())
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Why the views are approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, one list per draw.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.fit.posterior.chains())
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("pareto_k", "")?.pareto_k.clone())
    }

    /// The simulations of a single-segment fit refitted with NUTS
    /// (0-based).
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("refitted", "")?.refitted.clone())
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued (0-based).
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("moment_matched", "")?.moment_matched.clone())
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead (0-based).
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("unrefitted", "")?.unrefitted.clone())
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        Ok(self.one("refit_warnings", "")?.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.segments.fits[0].k_threshold
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        Ok(self.one("flagged_share", "totals_frame()")?.flagged_share())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = &self.one("loo()", "")?.fit.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, ``opening_ultimate``,
    /// ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let detail = match segments.fits.as_slice() {
            [one] => format!(
                ", flagged_share={:?}, refitted={}",
                one.flagged_share(),
                one.refitted.len()
            ),
            _ => String::new(),
        };
        format!(
            "CompartmentalOneYearFits({}origins={}, n_sims={}{detail})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.cdr.n_sims(),
        )
    }
}

/// A generalized linear model of the cells of a triangle, for
/// ``diagonal_backtest``: the terms and the GLM, on the columns of the
/// backtest's cell table (``origin``, ``development`` and ``calendar``
/// factors; ``origin_index``, ``development_index``, ``calendar_index``,
/// ``age`` and ``exposure`` numbers; ``exposure`` is ``premium`` or 1).
/// The response is the incremental value of the column.
///
/// Parameters
/// ----------
/// terms : prospicio.models.Terms
///     The model's terms.
/// glm : prospicio.models.Glm
///     The family, link and dispersion.
/// name : str, default "glm"
///     The model's name in a ``diagonal_backtest`` given a list.
///
/// Examples
/// --------
/// >>> from prospicio.models import Glm, Terms
/// >>> from prospicio.reserving import GlmCandidate
/// >>> GlmCandidate.odp()
/// GlmCandidate(name='odp')
/// >>> GlmCandidate(Terms().intercept().factor("origin"), Glm("poisson", "log"), name="origin only").name
/// 'origin only'
#[pyclass(name = "GlmCandidate", module = "prospicio.reserving", frozen)]
pub(crate) struct PyGlmCandidate {
    inner: GlmCandidate,
}

#[pymethods]
impl PyGlmCandidate {
    #[new]
    #[pyo3(signature = (terms, glm, name = "glm"))]
    fn new(terms: PyRef<'_, PyTerms>, glm: PyRef<'_, PyGlm>, name: &str) -> Self {
        Self {
            inner: GlmCandidate {
                name: name.to_string(),
                terms: terms.inner.clone(),
                glm: glm.inner,
            },
        }
    }

    /// The over-dispersed Poisson GLM of the chain ladder: an intercept, an
    /// origin factor and a development factor, with the over-dispersed
    /// Poisson family. Its name is ``"odp"``.
    ///
    /// Returns
    /// -------
    /// GlmCandidate
    #[staticmethod]
    fn odp() -> Self {
        Self {
            inner: GlmCandidate {
                name: "odp".into(),
                terms: Terms::new()
                    .intercept()
                    .factor("origin")
                    .factor("development"),
                glm: Glm::over_dispersed_poisson(),
            },
        }
    }

    /// The name in the backtest.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The model's terms.
    #[getter]
    fn terms(&self) -> PyTerms {
        PyTerms {
            inner: self.inner.terms.clone(),
        }
    }

    /// The model's family, link and dispersion.
    #[getter]
    fn glm(&self) -> PyGlm {
        PyGlm {
            inner: self.inner.glm,
        }
    }

    fn __repr__(&self) -> String {
        format!("GlmCandidate(name={})", quoted(&self.inner.name))
    }
}

/// `text` as a Python string literal in single quotes, for a `repr`.
fn quoted(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// `items` as a Python list of strings, for a `repr`.
fn quoted_list(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|i| quoted(i))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// A model of a ``diagonal_backtest``, read from a Python object.
enum Candidate {
    Odp(BayesOdp),
    Meyers(Meyers),
    Glm(GlmCandidate),
    Compartmental(Box<CompartmentalCandidate>),
}

impl Candidate {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(m) = obj.extract::<PyRef<'_, PyBayesOdp>>() {
            Ok(Self::Odp(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyMeyers>>() {
            Ok(Self::Meyers(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyGlmCandidate>>() {
            Ok(Self::Glm(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyCompartmentalCandidate>>() {
            // A clone shares the fits the object keeps, so its
            // outstanding_backtest reuses what the backtest sampled.
            Ok(Self::Compartmental(Box::new(m.inner.clone())))
        } else {
            let kind = obj
                .get_type()
                .name()
                .map_or_else(|_| "?".into(), |n| n.to_string());
            Err(PyTypeError::new_err(format!(
                "a backtest model must be a BayesOdp, a Meyers, a GlmCandidate or a \
                 CompartmentalCandidate, got {kind}"
            )))
        }
    }

    fn default_name(&self) -> String {
        match self {
            Self::Odp(m) => m.name().to_string(),
            Self::Meyers(m) => m.name().to_string(),
            Self::Glm(m) => m.name.clone(),
            Self::Compartmental(m) => m.name().to_string(),
        }
    }

    fn into_named(self, name: String) -> NamedModel {
        match self {
            Self::Odp(m) => NamedModel::new(name, m),
            Self::Meyers(m) => NamedModel::new(name, m),
            Self::Glm(m) => NamedModel::new(name, m),
            Self::Compartmental(m) => NamedModel::new(name, *m),
        }
    }
}

/// The models of a backtest as named models: a dict's keys are the names,
/// a list's are the models' own names (``"bayes_odp"``, ``"meyers_crc"``,
/// ``"meyers_csr"``, ``"meyers_cay"``, a ``GlmCandidate``'s ``name``).
fn named_models(
    models: &Bound<'_, PyAny>,
) -> PyResult<(Vec<NamedModel>, bool, Vec<CompartmentalCandidate>)> {
    let mut named = Vec::new();
    let mut has_meyers = false;
    let mut compartmental = Vec::new();
    if let Ok(dict) = models.cast::<PyDict>() {
        for (key, value) in dict.iter() {
            let name: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("the keys of models must be model names"))?;
            let c = Candidate::from_py(&value)?;
            has_meyers |= matches!(c, Candidate::Meyers(_));
            if let Candidate::Compartmental(m) = &c {
                compartmental.push((**m).clone());
            }
            named.push(c.into_named(name));
        }
    } else {
        let iter = models.try_iter().map_err(|_| {
            PyTypeError::new_err("models must be a dict of name to model or a list of models")
        })?;
        for item in iter {
            let c = Candidate::from_py(&item?)?;
            has_meyers |= matches!(c, Candidate::Meyers(_));
            if let Candidate::Compartmental(m) = &c {
                compartmental.push((**m).clone());
            }
            let name = c.default_name();
            if named.iter().any(|m: &NamedModel| m.name == name) {
                return Err(PyValueError::new_err(format!(
                    "two models are named {}: pass a dict {{name: model}} to name each one",
                    quoted(&name)
                )));
            }
            named.push(c.into_named(name));
        }
    }
    if named.is_empty() {
        return Err(PyValueError::new_err("models must hold at least one model"));
    }
    Ok((named, has_meyers, compartmental))
}

/// The calendar period of each split of `bt`, as the frame labels it.
fn diagonal_labels(cells: &TriangleFrame, bt: &Backtest) -> Vec<String> {
    let labels = match cells.frame().column("calendar") {
        Some(Column::Categorical(c)) => Some(c),
        _ => None,
    };
    bt.scored_rows()
        .iter()
        .zip(bt.diagonals())
        .map(|(rows, &index)| match (labels, rows.first()) {
            (Some(c), Some(&r)) => c[r].clone(),
            _ => index.to_string(),
        })
        .collect()
}

/// A table as a pandas DataFrame, or as a dict of lists without pandas.
fn backtest_table<'py>(
    py: Python<'py>,
    columns: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    match py.import("pandas") {
        Ok(pandas) => pandas.call_method1("DataFrame", (&columns,)),
        Err(_) => Ok(columns.into_any()),
    }
}

/// Backtests models on the latest calendar diagonals of one measure column
/// of a single-segment triangle: for each of the last ``n_diagonals``
/// calendar periods, every model is fitted to the cells valued before it
/// and forecasts the cells valued in it, and the forecasts are scored
/// against what came in.
///
/// The table of cells has one row per origin and development cell, with
/// the incremental values of ``column`` as the response (a cumulative
/// triangle is differenced). The cells of each held-out diagonal are the
/// test set and the earlier ones the training set, so a model sees only
/// what was known. A held-out cell whose origin or development period has
/// no training cell is left out of the scoring of every model (see
/// ``excluded``).
///
/// A fit of a sampler (``BayesOdp``, ``Meyers``) whose diagnostics fail a
/// threshold is **still scored**: a study over many triangles wants every
/// split and a list of the doubtful ones, not an abort at the first. The
/// failed thresholds are in ``Backtest.warnings`` and
/// ``Backtest.diagnostics()``; a model that cannot be fitted at all raises
/// ``ValueError``.
///
/// Meyers' own statistic, the percentile of the outcome in the predictive
/// distribution of the ultimate, comes from a fit on the whole upper
/// triangle, not from a backtest: it is the ``cdf`` of the fit's
/// predictive total at the known outcome,
/// ``fit.ultimates.total().cdf(outcome)`` (``Sampled.cdf``). The
/// backtest's ``total_percentile`` is the same quantity for the held-out
/// diagonal's total, from the fit on the cells before it.
///
/// Parameters
/// ----------
/// triangle : Triangle
///     Cumulative, with one segment.
/// column : str
///     The measure column to forecast.
/// models : dict or list
///     A dict from name to model, or a list of models named by their
///     kind (``"bayes_odp"``, ``"meyers_crc"``, ``"meyers_csr"``,
///     ``"meyers_cay"``, ``"compartmental_gm1"``, ``"compartmental_gm2"``,
///     a ``GlmCandidate``'s ``name``); two models with one name are an
///     error. A model is a ``BayesOdp`` or a ``Meyers`` (the settings of
///     its own ``fit``, except that ``n_sims`` and ``seed`` are the
///     backtest's), a ``GlmCandidate`` or a ``CompartmentalCandidate``
///     (``Compartmental.candidate``, on this ``triangle`` and ``column``;
///     its ``outstanding_backtest`` scores the held-out outstanding on the
///     same fits).
/// n_diagonals : int
///     The number of latest calendar diagonals to hold out, oldest first.
/// premium : array-like, optional
///     The exposure of each origin, one value per origin. ``Meyers``
///     needs it; a ``GlmCandidate`` sees it as the ``exposure`` column.
/// n_sims : int, default 1000
///     Simulations per forecast.
/// seed : int, default 0
///     Seed of every fit and simulation; the same seed replays the scores.
/// interval : float, default 0.9
///     The central interval whose coverage is scored.
///
/// Returns
/// -------
/// Backtest
///
/// Raises
/// ------
/// ValueError
///     If ``models`` is empty or names two models alike; if ``Meyers`` is
///     given without ``premium``; if a ``CompartmentalCandidate`` was built
///     on another triangle or paid column than ``triangle`` and ``column``;
///     if the triangle has several segments or
///     lacks the column; if a diagonal cannot be held out; or if a model
///     cannot be fitted.
/// TypeError
///     If a model is not a ``BayesOdp``, a ``Meyers``, a
///     ``GlmCandidate`` or a ``CompartmentalCandidate``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import GlmCandidate, Triangle, diagonal_backtest
/// >>> paid = [[100.0, 150.0, 165.0, 170.0], [110.0, 170.0, 180.0],
/// ...         [120.0, 175.0], [130.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> bt = diagonal_backtest(tri, "paid", [GlmCandidate.odp()], 1, n_sims=200)
/// >>> bt.models, bt.n_splits, bt.metrics[:2]
/// (['odp'], 1, ['crps', 'coverage'])
#[pyfunction]
#[pyo3(signature = (
    triangle,
    column,
    models,
    n_diagonals,
    *,
    premium = None,
    n_sims = 1000,
    seed = 0,
    interval = 0.9,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn diagonal_backtest(
    py: Python<'_>,
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
    models: &Bound<'_, PyAny>,
    n_diagonals: usize,
    premium: Option<&Bound<'_, PyAny>>,
    n_sims: usize,
    seed: u64,
    interval: f64,
) -> PyResult<PyBacktest> {
    let (named, has_meyers, compartmental) = named_models(models)?;
    let premium = premium.map(|p| floats(p, "premium")).transpose()?;
    if has_meyers && premium.is_none() {
        return Err(PyValueError::new_err(
            "Meyers needs premium, one value per origin",
        ));
    }
    let tri = &triangle.inner;
    // A candidate reads its own triangle: it must be the backtest's.
    for c in &compartmental {
        if c.paid() != column || c.triangle() != tri {
            return Err(PyValueError::new_err(format!(
                "{} was built on another triangle or paid column than the backtest's: build \
                 it with Compartmental.candidate on this triangle and column {}",
                c.name(),
                quoted(column)
            )));
        }
    }
    let (inner, diagonals) = py
        .detach(|| {
            let cells = TriangleFrame::new(tri, column, premium.as_deref())?;
            let refs: Vec<&dyn TriangleModel> =
                named.iter().map(|m| m as &dyn TriangleModel).collect();
            let bt = run_backtest(&cells, &refs, n_diagonals, n_sims, seed, interval)?;
            let diagonals = diagonal_labels(&cells, &bt);
            Ok((bt, diagonals))
        })
        .map_err(err)?;
    Ok(PyBacktest { inner, diagonals })
}

/// The scores of a ``diagonal_backtest``: every model on every metric and
/// held-out diagonal, the diagnostics of each fit and the pointwise
/// held-out log densities.
///
/// The metrics (``metrics``), each scored per diagonal over the held-out
/// cells that every model fits:
///
/// - ``crps``: the mean over the cells of the continuous ranked
///   probability score of the forecast distribution at the cell's value.
/// - ``coverage``: the share of cells inside the central ``interval`` of
///   their forecast draws.
/// - ``actual_vs_expected``: the diagonal total over the forecast mean of
///   the diagonal total.
/// - ``total_crps``: the CRPS of the diagonal total from the draws summed
///   per simulation.
/// - ``mean_log_density``: the mean over the cells of the model's held-out
///   log predictive density (NaN for a model that gives none).
/// - ``total_percentile``: the share of the simulated diagonal totals at
///   or below the actual total, the probability integral transform of the
///   diagonal: uniform over many triangles for a calibrated model.
///
/// The tables (``scores()``, ``mean()``, ``diagnostics()``) are pandas
/// DataFrames when pandas is installed and dicts of lists otherwise.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import GlmCandidate, Triangle, diagonal_backtest
/// >>> paid = [[100.0, 150.0, 165.0, 170.0], [110.0, 170.0, 180.0],
/// ...         [120.0, 175.0], [130.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> bt = diagonal_backtest(tri, "paid", [GlmCandidate.odp()], 1, n_sims=200)
/// >>> bt
/// Backtest(models=['odp'], diagonals=['2023'], metrics=6, warnings=0)
/// >>> bt.warnings
/// []
#[pyclass(name = "Backtest", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBacktest {
    inner: Backtest,
    diagonals: Vec<String>,
}

#[pymethods]
impl PyBacktest {
    /// Model names, in the order given.
    #[getter]
    fn models(&self) -> Vec<String> {
        self.inner.models().to_vec()
    }

    /// Metric names, in the order of ``scores()``: ``crps``,
    /// ``coverage``, ``actual_vs_expected``, ``total_crps``,
    /// ``mean_log_density`` and ``total_percentile``.
    #[getter]
    fn metrics(&self) -> Vec<String> {
        self.inner
            .metrics()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Number of held-out diagonals.
    #[getter]
    fn n_splits(&self) -> usize {
        self.inner.n_splits()
    }

    /// The calendar period held out in each split (labelled at the
    /// triangle's development grain, as the ``calendar`` column of the
    /// cells), oldest first.
    #[getter]
    fn diagonals(&self) -> Vec<String> {
        self.diagonals.clone()
    }

    /// Every score as a long table with one row per model, diagonal and
    /// metric: columns ``model``, ``diagonal``, ``metric`` and ``value``.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let (mut model, mut diagonal, mut metric, mut value) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (m, name) in bt.models().iter().enumerate() {
            for (k, label) in self.diagonals.iter().enumerate() {
                for (j, metric_name) in bt.metrics().iter().enumerate() {
                    model.push(name.clone());
                    diagonal.push(label.clone());
                    metric.push((*metric_name).to_string());
                    value.push(bt.split_scores(m, j)[k]);
                }
            }
        }
        let out = PyDict::new(py);
        out.set_item("model", model)?;
        out.set_item("diagonal", diagonal)?;
        out.set_item("metric", metric)?;
        out.set_item("value", value)?;
        backtest_table(py, out)
    }

    /// The mean of each metric over the diagonals, one row per model:
    /// the column ``model`` and one column per metric. A metric that is NaN
    /// on a diagonal (``mean_log_density`` of a model with no density) is
    /// NaN in the mean.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let out = PyDict::new(py);
        out.set_item("model", bt.models().to_vec())?;
        for (j, metric) in bt.metrics().iter().enumerate() {
            let means: Vec<f64> = (0..bt.models().len()).map(|m| bt.mean(m, j)).collect();
            out.set_item(*metric, means)?;
        }
        backtest_table(py, out)
    }

    /// Per diagonal, the number of held-out cells left out of the scoring
    /// of every model because their origin or development period has no
    /// training cell that every model fits on.
    #[getter]
    fn excluded(&self) -> Vec<usize> {
        self.inner.excluded().to_vec()
    }

    /// Per diagonal, the rows of the cell table that were scored (origin
    /// by development, row-major, counted from 0; R's ``scored_rows``
    /// count from 1), in the order of
    /// ``log_densities``.
    #[getter]
    fn scored_rows(&self) -> Vec<Vec<usize>> {
        self.inner.scored_rows().to_vec()
    }

    /// Each model's held-out log predictive density of every scored row,
    /// diagonals in order, by model name; ``None`` for a model that gives
    /// none. Every model's list is for the same rows, so the lists are
    /// the input of the stacking functions.
    #[getter]
    fn log_densities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, lpd) in self.inner.models().iter().zip(self.inner.log_densities()) {
            out.set_item(name, lpd.clone())?;
        }
        Ok(out)
    }

    /// The sampler diagnostics of each fit, one row per model and
    /// diagonal: ``model``, ``diagonal``, ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``n_warnings``; a row for
    /// every model and diagonal, models in order. A model with no sampler
    /// (the GLM) has NaN in the three float columns and ``None`` in the
    /// two count columns (NaN in a DataFrame). A fit with warnings is
    /// scored like any other.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let (mut model, mut diagonal) = (Vec::new(), Vec::new());
        let (mut rhat, mut bulk, mut tail) = (Vec::new(), Vec::new(), Vec::new());
        let (mut divergences, mut n_warnings) = (Vec::new(), Vec::new());
        for (m, name) in bt.models().iter().enumerate() {
            for (k, d) in bt.diagnostics(m).iter().enumerate() {
                model.push(name.clone());
                diagonal.push(self.diagonals[k].clone());
                rhat.push(d.as_ref().map_or(f64::NAN, |d| d.max_rhat));
                bulk.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_bulk));
                tail.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_tail));
                divergences.push(d.as_ref().map(|d| d.divergences));
                n_warnings.push(d.as_ref().map(|d| d.warnings.len()));
            }
        }
        let out = PyDict::new(py);
        out.set_item("model", model)?;
        out.set_item("diagonal", diagonal)?;
        out.set_item("max_rhat", rhat)?;
        out.set_item("min_ess_bulk", bulk)?;
        out.set_item("min_ess_tail", tail)?;
        out.set_item("divergences", divergences)?;
        out.set_item("n_warnings", n_warnings)?;
        backtest_table(py, out)
    }

    /// Every failed diagnostic threshold as ``(model, diagonal,
    /// message)``, in model then diagonal order; empty when every fit
    /// passed.
    #[getter]
    fn warnings(&self) -> Vec<(String, String, String)> {
        self.inner
            .warnings()
            .into_iter()
            .map(|(m, k, message)| {
                (
                    self.inner.models()[m].clone(),
                    self.diagonals[k].clone(),
                    message,
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Backtest(models={}, diagonals={}, metrics={}, warnings={})",
            quoted_list(self.inner.models()),
            quoted_list(&self.diagonals),
            self.inner.metrics().len(),
            self.inner.warnings().len(),
        )
    }
}
