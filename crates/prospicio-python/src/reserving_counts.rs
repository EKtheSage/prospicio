//! `prospicio.reserving` methods on claim counts (issues #554 and #555,
//! `docs/design/reserving-v02.md`, decisions 35 and 36): frequency-severity,
//! the disposal rate method, case outstanding development and the
//! Berquist-Sherman adjustments, over
//! `prospicio_reserving::{frequency_severity, berquist_sherman}`.
//!
//! The Berquist-Sherman functions return a `Triangle`, which any method
//! then projects. Counts and amounts are measure columns of one
//! single-segment triangle.

use prospicio_core::{Lag, Period};
use prospicio_reserving::{
    BerquistSherman, CaseAdequacy, CaseOutstanding, CaseOutstandingFit, CaseTrend, CountProjection,
    Disposal, DisposalFit, FrequencySeverity, FrequencySeverityFit, Interpolation, OpenCounts,
    Selection, SettlementRate, SeverityProjection, TrendFit, Trended, Triangle, UltimateCounts,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyTuple};

use crate::reserving::{PyChainLadder, PyTriangle, age_values, err, py_repr};

type Rows = Vec<Vec<Option<f64>>>;

/// A `Selection` from Python: `None` or `"all"` for the simple average of
/// every origin, an integer for the latest n origins, `"latest_diagonal"`,
/// a dict `{age: value}` of given values, or a dict with `n_periods`,
/// `drop_high` and `drop_low`.
fn selection_arg(obj: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Selection> {
    let Some(obj) = obj else {
        return Ok(Selection::default());
    };
    if obj.is_none() {
        return Ok(Selection::default());
    }
    let bad = || {
        PyTypeError::new_err(format!(
            "{what} must be None, \"all\", \"latest_diagonal\", a number of origins, a dict of \
             values by age or a dict of n_periods, drop_high and drop_low, got {}",
            py_repr(obj)
        ))
    };
    if obj.is_instance_of::<PyBool>() {
        return Err(bad());
    }
    if let Ok(name) = obj.extract::<String>() {
        return match name.as_str() {
            "all" => Ok(Selection::default()),
            "latest_diagonal" => Ok(Selection::LatestDiagonal),
            _ => Err(bad()),
        };
    }
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(Selection::latest(n));
    }
    let Ok(dict) = obj.cast::<PyDict>() else {
        return Err(bad());
    };
    if dict.keys().iter().any(|k| k.is_instance_of::<PyString>()) {
        let mut n_periods = None;
        let (mut drop_high, mut drop_low) = (0, 0);
        for (k, v) in dict.iter() {
            match k.extract::<String>().map_err(|_| bad())?.as_str() {
                "n_periods" => n_periods = v.extract::<Option<usize>>()?,
                "drop_high" => drop_high = v.extract()?,
                "drop_low" => drop_low = v.extract()?,
                _ => return Err(bad()),
            }
        }
        return Ok(Selection::Average {
            n_periods,
            drop_high,
            drop_low,
        });
    }
    Ok(Selection::Given(age_values(obj, what)?))
}

/// The ultimate counts Python names, for `tri`: a sequence of numbers, one
/// per origin; a column for the volume chain ladder on it; a pair
/// `(column, ChainLadder)`; or a list of columns or pairs for their mean.
fn ultimate_counts_arg(obj: &Bound<'_, PyAny>, tri: &Triangle) -> PyResult<UltimateCounts> {
    if let Ok(column) = obj.extract::<String>() {
        return Ok(UltimateCounts::development(column));
    }
    if let Ok(pair) = obj.cast::<PyTuple>()
        && pair.len() == 2
        && let Ok(column) = pair.get_item(0)?.extract::<String>()
    {
        let method = pair.get_item(1)?;
        let method = method.extract::<PyRef<'_, PyChainLadder>>().map_err(|_| {
            PyTypeError::new_err(format!(
                "ultimate_counts: a pair is (column, ChainLadder), got {}",
                py_repr(obj)
            ))
        })?;
        return Ok(UltimateCounts::Development {
            column,
            method: method.resolved(tri.origin_grain())?,
        });
    }
    if let Ok(values) = obj.extract::<Vec<f64>>() {
        return Ok(UltimateCounts::Given(values));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let parts = list
            .iter()
            .map(|x| ultimate_counts_arg(&x, tri))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(UltimateCounts::Mean(parts));
    }
    Err(PyTypeError::new_err(format!(
        "ultimate_counts must be numbers (one per origin), a column, a pair (column, \
         ChainLadder) or a list of columns or pairs, got {}",
        py_repr(obj)
    )))
}

/// The origin of `tri` that prints as `label`.
fn origin_labelled(label: &str, tri: &Triangle, what: &str) -> PyResult<Period> {
    tri.origins()
        .iter()
        .find(|p| p.to_string() == label)
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!("{what}: {label} is not an origin of the triangle"))
        })
}

fn trend_dict(fit: Option<TrendFit>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match fit {
        None => Ok(py.None()),
        Some(f) => {
            let d = PyDict::new(py);
            d.set_item("rate", f.rate)?;
            d.set_item("r_squared", f.r_squared)?;
            d.set_item("n", f.n)?;
            Ok(d.into_any().unbind())
        }
    }
}

fn labels(origins: &[Period]) -> Vec<String> {
    origins.iter().map(ToString::to_string).collect()
}

/// A value selected at the level of a base origin and carried to the other
/// origins by an annual trend: at origin ``o``,
/// ``selected * (1 + trend) ** (years(o) - years(base)) / factors[o]``.
/// Friedland's approach #2 selects a frequency and a severity this way from
/// the older, mature years.
///
/// Parameters
/// ----------
/// selected : float
///     The value at the base origin's level.
/// base : int or str
///     The origin whose level ``selected`` is at, as ``Triangle.origins``
///     prints it.
/// trend : float
///     The annual trend.
/// factors : sequence of float, optional
///     One factor per origin, oldest first, that brings the origin's value
///     to the base level for anything besides the trend (Friedland's tort
///     reform factors).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Trended
/// >>> Trended(20.0, 2008, 0.05)
/// Trended(selected=20.0, base="2008", trend=0.05)
#[pyclass(name = "Trended", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTrended {
    selected: f64,
    base: String,
    trend: f64,
    factors: Vec<f64>,
}

impl PyTrended {
    fn resolved(&self, tri: &Triangle) -> PyResult<Trended> {
        Ok(Trended {
            selected: self.selected,
            base: origin_labelled(&self.base, tri, "Trended's base")?,
            trend: self.trend,
            factors: self.factors.clone(),
        })
    }
}

#[pymethods]
impl PyTrended {
    #[new]
    #[pyo3(signature = (selected, base, trend, factors = None))]
    fn new(
        selected: f64,
        base: &Bound<'_, PyAny>,
        trend: f64,
        factors: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            selected,
            base: base.str()?.to_string(),
            trend,
            factors: factors.unwrap_or_default(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Trended(selected={:?}, base={:?}, trend={:?})",
            self.selected, self.base, self.trend
        )
    }
}

/// Frequency-severity: ultimate claims as ultimate counts times an ultimate
/// severity, each projected on its own (Friedland, chapter 11, approaches
/// #1 and #2).
///
/// Parameters
/// ----------
/// amounts : str
///     The claims column (reported or paid) whose ultimate is projected.
/// counts : str
///     The count column the severity triangle divides the amounts by.
/// ultimate_counts : optional
///     Numbers, one per origin; a column for the chain ladder on it; a pair
///     ``(column, ChainLadder)``; a list of columns or pairs for their mean;
///     or a ``Trended`` frequency with ``exposure`` (approach #2). By
///     default the chain ladder on ``counts``.
/// severity : ChainLadder or Trended, optional
///     The chain ladder on the triangle of severities (approach #1, the
///     default) or a severity selected at a base origin and trended.
/// exposure : str, optional
///     The exposure column of a ``Trended`` frequency.
/// exposure_factors : sequence of float, optional
///     One factor per origin that brings the exposure to the base level
///     (on-level factors, wage inflation).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import FrequencySeverity, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12],
/// ...     {"reported": [100.0, 150.0, 120.0], "counts": [10.0, 12.0, 11.0]})
/// >>> fit = FrequencySeverity("reported", "counts").fit(tri)
/// >>> round(fit.ultimate[1], 6)
/// 180.0
#[pyclass(name = "FrequencySeverity", module = "prospicio.reserving", frozen)]
pub(crate) struct PyFrequencySeverity {
    amounts: String,
    counts: String,
    ultimate_counts: Option<Py<PyAny>>,
    severity: Option<Py<PyAny>>,
    exposure: Option<String>,
    exposure_factors: Vec<f64>,
}

#[pymethods]
impl PyFrequencySeverity {
    #[new]
    #[pyo3(signature = (amounts, counts, ultimate_counts = None, severity = None, exposure = None, exposure_factors = None))]
    fn new(
        amounts: String,
        counts: String,
        ultimate_counts: Option<Py<PyAny>>,
        severity: Option<Py<PyAny>>,
        exposure: Option<String>,
        exposure_factors: Option<Vec<f64>>,
    ) -> Self {
        Self {
            amounts,
            counts,
            ultimate_counts,
            severity,
            exposure,
            exposure_factors: exposure_factors.unwrap_or_default(),
        }
    }

    /// Fit to a single-segment triangle holding the columns.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///
    /// Returns
    /// -------
    /// FrequencySeverityFit
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
    ) -> PyResult<PyFrequencySeverityFit> {
        let tri = &triangle.inner;
        let ultimate_counts = match self.ultimate_counts.as_ref().map(|o| o.bind(py)) {
            None => CountProjection::Counts(UltimateCounts::development(self.counts.clone())),
            Some(obj) => match obj.extract::<PyRef<'_, PyTrended>>() {
                Ok(frequency) => CountProjection::Frequency {
                    exposure: self.exposure.clone().ok_or_else(|| {
                        PyValueError::new_err("a Trended frequency needs exposure, a column")
                    })?,
                    exposure_factors: self.exposure_factors.clone(),
                    frequency: frequency.resolved(tri)?,
                },
                Err(_) => CountProjection::Counts(ultimate_counts_arg(obj, tri)?),
            },
        };
        let severity = match self.severity.as_ref().map(|o| o.bind(py)) {
            None => SeverityProjection::Development(Default::default()),
            Some(obj) => {
                if let Ok(t) = obj.extract::<PyRef<'_, PyTrended>>() {
                    SeverityProjection::Trended(t.resolved(tri)?)
                } else if let Ok(c) = obj.extract::<PyRef<'_, PyChainLadder>>() {
                    SeverityProjection::Development(c.resolved(tri.origin_grain())?)
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "severity must be a ChainLadder or a Trended, got {}",
                        py_repr(obj)
                    )));
                }
            }
        };
        let inner = FrequencySeverity {
            amounts: self.amounts.clone(),
            counts: self.counts.clone(),
            ultimate_counts,
            severity,
        }
        .fit(tri)
        .map_err(err)?;
        Ok(PyFrequencySeverityFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencySeverity(amounts={:?}, counts={:?})",
            self.amounts, self.counts
        )
    }
}

/// A fitted ``FrequencySeverity``.
#[pyclass(name = "FrequencySeverityFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyFrequencySeverityFit {
    inner: FrequencySeverityFit,
}

#[pymethods]
impl PyFrequencySeverityFit {
    /// Origin of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        labels(&self.inner.origins)
    }
    /// The latest value of the amounts column.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }
    /// The latest amounts over the latest counts.
    #[getter]
    fn latest_severity(&self) -> Vec<f64> {
        self.inner.latest_severity.clone()
    }
    /// Ultimate claim counts.
    #[getter]
    fn ultimate_counts(&self) -> Vec<f64> {
        self.inner.ultimate_counts.clone()
    }
    /// Ultimate severities.
    #[getter]
    fn ultimate_severity(&self) -> Vec<f64> {
        self.inner.ultimate_severity.clone()
    }
    /// Ultimate claims, counts times severity.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }
    /// Ultimate less latest.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        self.inner.reserves()
    }
    /// The sum of the ultimates.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }
    /// The sum of the reserves.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }
    fn __repr__(&self) -> String {
        format!(
            "FrequencySeverityFit(total_ultimate={:?}, total_reserve={:?})",
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// The disposal rate method (Friedland, chapter 11): incremental closed
/// counts from selected disposal rates, each priced at a selected
/// incremental severity brought to the origin's cost level.
///
/// Parameters
/// ----------
/// closed_counts : str
///     The cumulative closed count column.
/// paid : str
///     The cumulative paid claims column.
/// ultimate_counts
///     Numbers, one per origin; a column for the chain ladder on it; a pair
///     ``(column, ChainLadder)``; or a list of these for their mean.
/// disposal_rates : optional
///     How the rate at each age is selected: ``None`` or ``"all"`` for the
///     simple average of every origin, an integer for the latest n,
///     ``"latest_diagonal"``, a dict of values by age, or a dict of
///     ``n_periods``, ``drop_high`` and ``drop_low``.
/// severities : optional
///     How the incremental severity at each age is selected, as
///     ``disposal_rates``, on severities at the base origin's level.
/// tail_severity_from : int, optional
///     The age from which the severities are pooled into one value, total
///     trended paid over total counts, used to ultimate. By default the
///     last selection carries on.
/// trend : float, default 0.0
///     The annual severity trend.
/// base : int or str, optional
///     The origin of the common cost level; by default the latest origin.
/// factors : sequence of float, optional
///     One factor per origin that brings its severities to the base level
///     besides the trend (tort reform factors).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Disposal, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12],
/// ...     {"closed": [5.0, 10.0, 6.0], "paid": [50.0, 100.0, 60.0]})
/// >>> fit = Disposal("closed", "paid", [10.0, 12.0]).fit(tri)
/// >>> round(fit.ultimate[1], 6)
/// 120.0
#[pyclass(name = "Disposal", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDisposal {
    closed_counts: String,
    paid: String,
    ultimate_counts: Py<PyAny>,
    disposal_rates: Selection,
    severities: Selection,
    tail_severity_from: Option<Lag>,
    trend: f64,
    base: Option<String>,
    factors: Vec<f64>,
}

#[pymethods]
impl PyDisposal {
    #[new]
    #[pyo3(signature = (closed_counts, paid, ultimate_counts, disposal_rates = None, severities = None, tail_severity_from = None, trend = 0.0, base = None, factors = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        closed_counts: String,
        paid: String,
        ultimate_counts: Py<PyAny>,
        disposal_rates: Option<&Bound<'_, PyAny>>,
        severities: Option<&Bound<'_, PyAny>>,
        tail_severity_from: Option<Lag>,
        trend: f64,
        base: Option<&Bound<'_, PyAny>>,
        factors: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            closed_counts,
            paid,
            ultimate_counts,
            disposal_rates: selection_arg(disposal_rates, "disposal_rates")?,
            severities: selection_arg(severities, "severities")?,
            tail_severity_from,
            trend,
            base: base
                .map(|b| Ok::<_, PyErr>(b.str()?.to_string()))
                .transpose()?,
            factors: factors.unwrap_or_default(),
        })
    }

    /// Fit to a single-segment triangle holding the columns.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///
    /// Returns
    /// -------
    /// DisposalFit
    fn fit(&self, py: Python<'_>, triangle: PyRef<'_, PyTriangle>) -> PyResult<PyDisposalFit> {
        let tri = &triangle.inner;
        let base = self
            .base
            .as_ref()
            .map(|b| origin_labelled(b, tri, "base"))
            .transpose()?;
        let inner = Disposal {
            closed_counts: self.closed_counts.clone(),
            paid: self.paid.clone(),
            ultimate_counts: ultimate_counts_arg(self.ultimate_counts.bind(py), tri)?,
            disposal_rates: self.disposal_rates.clone(),
            severities: self.severities.clone(),
            tail_severity_from: self.tail_severity_from,
            trend: self.trend,
            base,
            factors: self.factors.clone(),
        }
        .fit(tri)
        .map_err(err)?;
        Ok(PyDisposalFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Disposal(closed_counts={:?}, paid={:?}, trend={:?})",
            self.closed_counts, self.paid, self.trend
        )
    }
}

/// A fitted ``Disposal``. Cell tables are lists of rows by origin, ``None``
/// where a cell does not apply.
#[pyclass(name = "DisposalFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDisposalFit {
    inner: DisposalFit,
}

#[pymethods]
impl PyDisposalFit {
    /// Origin of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        labels(&self.inner.origins)
    }
    /// Development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }
    /// Ultimate claim counts.
    #[getter]
    fn ultimate_counts(&self) -> Vec<f64> {
        self.inner.ultimate_counts.clone()
    }
    /// Observed closed counts over the ultimate counts.
    #[getter]
    fn disposal_rates(&self) -> Rows {
        self.inner.disposal_rates.clone()
    }
    /// The disposal rate selected at each age.
    #[getter]
    fn selected_rates(&self) -> Vec<f64> {
        self.inner.selected_rates.clone()
    }
    /// Incremental paid over incremental closed counts.
    #[getter]
    fn severities(&self) -> Rows {
        self.inner.severities.clone()
    }
    /// The severities at the base origin's level.
    #[getter]
    fn trended_severities(&self) -> Rows {
        self.inner.trended_severities.clone()
    }
    /// The selected severity at each age and, last, to ultimate.
    #[getter]
    fn selected_severities(&self) -> Vec<f64> {
        self.inner.selected_severities.clone()
    }
    /// Projected incremental closed counts, the step to ultimate last.
    #[getter]
    fn incremental_counts(&self) -> Rows {
        self.inner.incremental_counts.clone()
    }
    /// The severities at each origin's level.
    #[getter]
    fn projected_severities(&self) -> Rows {
        self.inner.projected_severities.clone()
    }
    /// Projected incremental paid, the step to ultimate last.
    #[getter]
    fn incremental_paid(&self) -> Rows {
        self.inner.incremental_paid.clone()
    }
    /// The latest paid claims.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }
    /// Ultimate claims.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }
    /// Ultimate less latest paid.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        self.inner.reserves()
    }
    /// The sum of the ultimates.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }
    /// The sum of the reserves.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }
    fn __repr__(&self) -> String {
        format!(
            "DisposalFit(total_ultimate={:?}, total_reserve={:?})",
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// Case outstanding development (Friedland, chapter 12): each period's paid
/// claims and closing case outstanding as ratios to the case outstanding a
/// period before, run out from the latest diagonal, with paid to ultimate
/// on the last case outstanding.
///
/// Parameters
/// ----------
/// paid : str
///     The cumulative paid claims column.
/// incurred : str
///     The cumulative reported claims column; case outstanding is reported
///     less paid.
/// paid_to_ultimate : float
///     Remaining paid over the case outstanding at the oldest age.
/// paid_on_case, case_on_case : optional
///     How the ratios at each age are selected, as ``Disposal``'s
///     ``disposal_rates``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CaseOutstanding, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [50.0, 90.0, 60.0], "reported": [100.0, 110.0, 120.0]})
/// >>> fit = CaseOutstanding("paid", "reported", 1.0).fit(tri)
/// >>> round(fit.ultimate[0], 6)
/// 110.0
#[pyclass(name = "CaseOutstanding", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCaseOutstanding {
    inner: CaseOutstanding,
}

#[pymethods]
impl PyCaseOutstanding {
    #[new]
    #[pyo3(signature = (paid, incurred, paid_to_ultimate, paid_on_case = None, case_on_case = None))]
    fn new(
        paid: String,
        incurred: String,
        paid_to_ultimate: f64,
        paid_on_case: Option<&Bound<'_, PyAny>>,
        case_on_case: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CaseOutstanding {
                paid,
                incurred,
                paid_on_case: selection_arg(paid_on_case, "paid_on_case")?,
                case_on_case: selection_arg(case_on_case, "case_on_case")?,
                paid_to_ultimate,
            },
        })
    }

    /// Fit to a single-segment triangle holding the columns.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///
    /// Returns
    /// -------
    /// CaseOutstandingFit
    fn fit(&self, triangle: PyRef<'_, PyTriangle>) -> PyResult<PyCaseOutstandingFit> {
        Ok(PyCaseOutstandingFit {
            inner: self.inner.fit(&triangle.inner).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CaseOutstanding(paid={:?}, incurred={:?}, paid_to_ultimate={:?})",
            self.inner.paid, self.inner.incurred, self.inner.paid_to_ultimate
        )
    }
}

/// A fitted ``CaseOutstanding``. Cell tables are lists of rows by origin,
/// ``None`` where a cell does not apply.
#[pyclass(name = "CaseOutstandingFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCaseOutstandingFit {
    inner: CaseOutstandingFit,
}

#[pymethods]
impl PyCaseOutstandingFit {
    /// Origin of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        labels(&self.inner.origins)
    }
    /// Development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }
    /// Incremental paid over the previous case outstanding.
    #[getter]
    fn paid_on_case(&self) -> Rows {
        self.inner.paid_on_case.clone()
    }
    /// Case outstanding over the previous case outstanding.
    #[getter]
    fn case_on_case(&self) -> Rows {
        self.inner.case_on_case.clone()
    }
    /// The paid ratio selected at each age.
    #[getter]
    fn selected_paid_on_case(&self) -> Vec<f64> {
        self.inner.selected_paid_on_case.clone()
    }
    /// The case ratio selected at each age.
    #[getter]
    fn selected_case_on_case(&self) -> Vec<f64> {
        self.inner.selected_case_on_case.clone()
    }
    /// Observed and projected case outstanding.
    #[getter]
    fn case_outstanding(&self) -> Rows {
        self.inner.case_outstanding.clone()
    }
    /// Projected incremental paid, the step to ultimate last.
    #[getter]
    fn incremental_paid(&self) -> Rows {
        self.inner.incremental_paid.clone()
    }
    /// The latest paid claims.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }
    /// The latest reported claims.
    #[getter]
    fn latest_incurred(&self) -> Vec<f64> {
        self.inner.latest_incurred.clone()
    }
    /// Ultimate claims.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }
    /// Ultimate less latest reported.
    #[getter]
    fn ibnr(&self) -> Vec<f64> {
        self.inner.ibnr()
    }
    /// Ultimate less latest paid.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        self.inner.reserves()
    }
    /// The sum of the ultimates.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }
    /// The sum of the reserves.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }
    fn __repr__(&self) -> String {
        format!(
            "CaseOutstandingFit(total_ultimate={:?}, total_reserve={:?})",
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

fn case_trend_arg(trend: &Bound<'_, PyAny>) -> PyResult<CaseTrend> {
    if let Ok(t) = trend.extract::<f64>() {
        return Ok(CaseTrend::Given(t));
    }
    if trend
        .extract::<String>()
        .is_ok_and(|name| name == "average_case")
    {
        return Ok(CaseTrend::AverageCase);
    }
    if let Ok((amount, counts)) = trend.extract::<(String, String)>() {
        return Ok(CaseTrend::Fitted { amount, counts });
    }
    Err(PyTypeError::new_err(format!(
        "trend must be a number, \"average_case\" or a pair (amount column, count column), got {}",
        py_repr(trend)
    )))
}

fn open_counts_arg(
    open_counts: Option<String>,
    reported: Option<String>,
    closed: Option<String>,
) -> PyResult<OpenCounts> {
    match (open_counts, reported, closed) {
        (Some(c), None, None) => Ok(OpenCounts::Column(c)),
        (None, Some(reported), Some(closed)) => {
            Ok(OpenCounts::ReportedLessClosed { reported, closed })
        }
        _ => Err(PyValueError::new_err(
            "give open_counts, or reported and closed, not both",
        )),
    }
}

fn interpolation_arg(name: &str) -> PyResult<Interpolation> {
    match name {
        "exponential" => Ok(Interpolation::Exponential),
        "linear" => Ok(Interpolation::Linear),
        _ => Err(PyValueError::new_err(format!(
            "interpolation must be \"exponential\" or \"linear\", got {name:?}"
        ))),
    }
}

/// The triangle restated at the case reserve adequacy of the latest
/// diagonal (Berquist and Sherman; Friedland, chapter 13): the average case
/// outstanding of each age on the latest diagonal, detrended to the earlier
/// origins, times the open counts, plus paid. The result is a cumulative
/// triangle with ``incurred`` replaced, for any method to project.
///
/// Parameters
/// ----------
/// triangle : Triangle
/// incurred, paid : str
///     The reported and paid claims columns.
/// trend : float, "average_case" or (str, str)
///     The annual severity trend, or a pair of an amount and a count column
///     to fit it from (paid claims and closed counts).
/// open_counts : str, optional
///     A column of open counts. Otherwise give ``reported`` and ``closed``.
/// reported, closed : str, optional
///     Count columns whose difference is the open counts.
/// max_age : int, optional
///     The oldest age adjusted, where older averages are too thin to use.
///
/// Returns
/// -------
/// Triangle
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Triangle, case_adequacy_adjust
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {
/// ...     "incurred": [100.0, 120.0, 150.0], "paid": [50.0, 100.0, 50.0],
/// ...     "open": [5.0, 2.0, 5.0]})
/// >>> out = case_adequacy_adjust(tri, "incurred", "paid", 0.25, open_counts="open")
/// >>> out.columns
/// ['incurred', 'paid', 'open']
#[pyfunction]
#[pyo3(signature = (triangle, incurred, paid, trend, open_counts = None, reported = None, closed = None, max_age = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn case_adequacy_adjust(
    triangle: PyRef<'_, PyTriangle>,
    incurred: String,
    paid: String,
    trend: &Bound<'_, PyAny>,
    open_counts: Option<String>,
    reported: Option<String>,
    closed: Option<String>,
    max_age: Option<Lag>,
) -> PyResult<PyTriangle> {
    let inner = CaseAdequacy {
        incurred,
        paid,
        open_counts: open_counts_arg(open_counts, reported, closed)?,
        trend: case_trend_arg(trend)?,
        max_age,
    }
    .adjust(&triangle.inner)
    .map_err(err)?;
    Ok(PyTriangle { inner })
}

/// The triangle with the columns ``average_case`` and
/// ``adjusted_average_case`` added, the exhibits behind
/// ``case_adequacy_adjust``, which has the same arguments.
///
/// Returns
/// -------
/// Triangle
#[pyfunction]
#[pyo3(signature = (triangle, incurred, paid, trend, open_counts = None, reported = None, closed = None, max_age = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn average_case(
    triangle: PyRef<'_, PyTriangle>,
    incurred: String,
    paid: String,
    trend: &Bound<'_, PyAny>,
    open_counts: Option<String>,
    reported: Option<String>,
    closed: Option<String>,
    max_age: Option<Lag>,
) -> PyResult<PyTriangle> {
    let inner = CaseAdequacy {
        incurred,
        paid,
        open_counts: open_counts_arg(open_counts, reported, closed)?,
        trend: case_trend_arg(trend)?,
        max_age,
    }
    .average_case(&triangle.inner)
    .map_err(err)?;
    Ok(PyTriangle { inner })
}

fn settlement(
    tri: &Triangle,
    paid: String,
    closed_counts: String,
    ultimate_counts: &Bound<'_, PyAny>,
    disposal_rates: Option<&Bound<'_, PyAny>>,
    interpolation: &str,
) -> PyResult<SettlementRate> {
    let rates = match disposal_rates {
        Some(r) if !r.is_none() => selection_arg(Some(r), "disposal_rates")?,
        _ => Selection::LatestDiagonal,
    };
    Ok(SettlementRate {
        paid,
        closed_counts,
        ultimate_counts: ultimate_counts_arg(ultimate_counts, tri)?,
        disposal_rates: rates,
        interpolation: interpolation_arg(interpolation)?,
    })
}

/// The triangle restated at the settlement rate of the latest diagonal
/// (Berquist and Sherman; Friedland, chapter 13): closed counts are the
/// selected disposal rates times each origin's ultimate counts, and paid
/// claims are read off a curve through each origin's observed (closed
/// count, paid) pairs at those counts.
///
/// Parameters
/// ----------
/// triangle : Triangle
/// paid, closed_counts : str
///     The paid claims and closed count columns, both replaced.
/// ultimate_counts
///     Numbers, one per origin; a column for the chain ladder on it; a pair
///     ``(column, ChainLadder)``; or a list of these for their mean.
/// disposal_rates : optional, default "latest_diagonal"
///     As ``Disposal``'s ``disposal_rates``.
/// interpolation : {"exponential", "linear"}, default "exponential"
///     The curve between two successive points.
///
/// Returns
/// -------
/// Triangle
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Triangle, settlement_rate_adjust
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {
/// ...     "paid": [50.0, 100.0, 60.0], "closed": [5.0, 10.0, 6.0]})
/// >>> settlement_rate_adjust(tri, "paid", "closed", [10.0, 12.0]).columns
/// ['paid', 'closed']
#[pyfunction]
#[pyo3(signature = (triangle, paid, closed_counts, ultimate_counts, disposal_rates = None, interpolation = "exponential"))]
pub(crate) fn settlement_rate_adjust(
    triangle: PyRef<'_, PyTriangle>,
    paid: String,
    closed_counts: String,
    ultimate_counts: &Bound<'_, PyAny>,
    disposal_rates: Option<&Bound<'_, PyAny>>,
    interpolation: &str,
) -> PyResult<PyTriangle> {
    let inner = settlement(
        &triangle.inner,
        paid,
        closed_counts,
        ultimate_counts,
        disposal_rates,
        interpolation,
    )?
    .adjust(&triangle.inner)
    .map_err(err)?;
    Ok(PyTriangle { inner })
}

/// Both Berquist-Sherman adjustments: paid claims and closed counts as
/// ``settlement_rate_adjust`` restates them, and incurred claims as the
/// adjusted average case outstanding times the adjusted open counts plus
/// the adjusted paid (Friedland, chapter 13, XYZ Insurer's third
/// projection).
///
/// Parameters
/// ----------
/// triangle : Triangle
/// incurred, paid, reported, closed_counts : str
///     Reported and paid claims, reported and closed counts.
/// trend : float, "average_case" or (str, str)
///     As ``case_adequacy_adjust``'s ``trend``.
/// ultimate_counts
///     As ``settlement_rate_adjust``'s.
/// max_age : int, optional
///     As ``case_adequacy_adjust``'s.
/// disposal_rates, interpolation
///     As ``settlement_rate_adjust``'s.
///
/// Returns
/// -------
/// Triangle
#[pyfunction]
#[pyo3(signature = (triangle, incurred, paid, reported, closed_counts, trend, ultimate_counts, max_age = None, disposal_rates = None, interpolation = "exponential"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn berquist_sherman_adjust(
    triangle: PyRef<'_, PyTriangle>,
    incurred: String,
    paid: String,
    reported: String,
    closed_counts: String,
    trend: &Bound<'_, PyAny>,
    ultimate_counts: &Bound<'_, PyAny>,
    max_age: Option<Lag>,
    disposal_rates: Option<&Bound<'_, PyAny>>,
    interpolation: &str,
) -> PyResult<PyTriangle> {
    let inner = BerquistSherman {
        case_adequacy: CaseAdequacy {
            incurred,
            paid: paid.clone(),
            open_counts: OpenCounts::ReportedLessClosed {
                reported,
                closed: closed_counts.clone(),
            },
            trend: case_trend_arg(trend)?,
            max_age,
        },
        settlement_rate: settlement(
            &triangle.inner,
            paid,
            closed_counts,
            ultimate_counts,
            disposal_rates,
            interpolation,
        )?,
    }
    .adjust(&triangle.inner)
    .map_err(err)?;
    Ok(PyTriangle { inner })
}

/// The exponential trend through points by least squares on the logarithm.
///
/// Parameters
/// ----------
/// x, y : sequence of float
///     Times in years and positive values.
///
/// Returns
/// -------
/// dict or None
///     ``rate``, ``r_squared`` and ``n``; ``None`` with fewer than two
///     usable points.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import exponential_trend
/// >>> round(exponential_trend([0.0, 1.0, 2.0], [100.0, 110.0, 121.0])["rate"], 12)
/// 0.1
#[pyfunction]
pub(crate) fn exponential_trend(py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<Py<PyAny>> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y need the same length"));
    }
    let points: Vec<(f64, f64)> = x.into_iter().zip(y).collect();
    trend_dict(prospicio_reserving::exponential_trend(&points), py)
}

/// The exponential trend of a column at each age, down the origins.
///
/// Returns
/// -------
/// dict
///     Each age with its ``rate``, ``r_squared`` and ``n``, or ``None``.
#[pyfunction]
pub(crate) fn trend_by_age<'py>(
    py: Python<'py>,
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (age, fit) in prospicio_reserving::trend_by_age(&triangle.inner, column).map_err(err)? {
        out.set_item(age, trend_dict(fit, py)?)?;
    }
    Ok(out)
}

/// One exponential trend of a column over every age at once: a common
/// slope with an intercept for each age.
///
/// Returns
/// -------
/// dict or None
///     ``rate``, ``r_squared`` and ``n``.
#[pyfunction]
pub(crate) fn pooled_trend(
    py: Python<'_>,
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
) -> PyResult<Py<PyAny>> {
    trend_dict(
        prospicio_reserving::pooled_trend(&triangle.inner, column).map_err(err)?,
        py,
    )
}

/// The triangle with a column ``name`` that is ``numerator`` over
/// ``denominator`` cell by cell, as average paid or a frequency.
///
/// Returns
/// -------
/// Triangle
#[pyfunction]
pub(crate) fn ratio_column(
    triangle: PyRef<'_, PyTriangle>,
    numerator: &str,
    denominator: &str,
    name: &str,
) -> PyResult<PyTriangle> {
    let inner = prospicio_reserving::ratio_column(&triangle.inner, numerator, denominator, name)
        .map_err(err)?;
    Ok(PyTriangle { inner })
}

/// Friedland's chapter 12 factor from case outstanding to unpaid claims:
/// ``1 + (R - 1) P / (P - R)`` from the reported and paid factors to
/// ultimate at an age.
///
/// Parameters
/// ----------
/// reported_cdf, paid_cdf : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.reserving import case_development_factor
/// >>> round(case_development_factor(1.2, 2.0), 6)
/// 1.5
#[pyfunction]
pub(crate) fn case_development_factor(reported_cdf: f64, paid_cdf: f64) -> PyResult<f64> {
    prospicio_reserving::case_development_factor(reported_cdf, paid_cdf).map_err(err)
}
