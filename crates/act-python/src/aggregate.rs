//! `prospicio.aggregate`: wrappers over `act_aggregate` (compound
//! distributions and simulated events).

use act_aggregate::{CompoundMethod, CompoundReport, EventSet};
use act_prob::{Counting, Grid};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::distributions::{
    PyGrid, PyNegativeBinomial, PyPoisson, PyPredictiveDistribution, extract_severity,
};
use crate::to_py;

/// A claim count accepted by the aggregation functions.
pub(crate) enum AnyCount {
    Poisson(act_prob::Poisson),
    NegativeBinomial(act_prob::NegativeBinomial),
    Binomial(act_prob::Binomial),
}

impl AnyCount {
    pub(crate) fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(n) = obj.extract::<PyRef<'_, PyPoisson>>() {
            return Ok(Self::Poisson(n.inner));
        }
        if let Ok(n) = obj.extract::<PyRef<'_, PyNegativeBinomial>>() {
            return Ok(Self::NegativeBinomial(n.inner));
        }
        if let Ok(n) = obj.extract::<PyRef<'_, crate::pareto::PyBinomial>>() {
            return Ok(Self::Binomial(n.inner));
        }
        Err(PyTypeError::new_err(
            "expected a Poisson, a NegativeBinomial or a Binomial",
        ))
    }

    pub(crate) fn as_counting(&self) -> &(dyn Counting + Sync) {
        match self {
            Self::Poisson(n) => n,
            Self::NegativeBinomial(n) => n,
            Self::Binomial(n) => n,
        }
    }
}

impl Counting for AnyCount {
    fn pmf(&self, k: u64) -> f64 {
        self.as_counting().pmf(k)
    }
    fn mean(&self) -> f64 {
        self.as_counting().mean()
    }
    fn variance(&self) -> f64 {
        self.as_counting().variance()
    }
    fn panjer_ab(&self) -> (f64, f64) {
        self.as_counting().panjer_ab()
    }
    fn pgf(&self, z: f64) -> f64 {
        self.as_counting().pgf(z)
    }
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        self.as_counting().pgf_complex(z)
    }
}

/// What a compound calculation produced and the error it introduced.
///
/// Returned with the aggregate grid by ``panjer`` and ``fft``. Read
/// ``aliasing_error`` first for FFT results: when it is not negligible the
/// grid is unreliable, including ``tail_mass``.
#[pyclass(name = "CompoundReport", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyCompoundReport {
    pub(crate) inner: CompoundReport,
}

#[pymethods]
impl PyCompoundReport {
    /// Method: ``"panjer"`` or ``"fft"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            CompoundMethod::Panjer => "panjer",
            CompoundMethod::Fft => "fft",
        }
    }

    /// Number of points in the aggregate grid.
    #[getter]
    fn points(&self) -> usize {
        self.inner.points
    }

    /// Aggregate probability above the last point, lumped onto it.
    #[getter]
    fn tail_mass(&self) -> f64 {
        self.inner.tail_mass
    }

    /// Largest change in any probability when the FFT buffer doubles; 0 for
    /// Panjer.
    #[getter]
    fn aliasing_error(&self) -> f64 {
        self.inner.aliasing_error
    }

    /// ``E[N] * E[X]`` on the severity grid.
    #[getter]
    fn expected_mean(&self) -> f64 {
        self.inner.expected_mean
    }

    /// Mean of the aggregate grid.
    #[getter]
    fn grid_mean(&self) -> f64 {
        self.inner.grid_mean
    }

    /// ``grid_mean - expected_mean``.
    ///
    /// Returns
    /// -------
    /// float
    fn mean_error(&self) -> f64 {
        self.inner.mean_error()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompoundReport(method={:?}, points={}, tail_mass={:e}, aliasing_error={:e})",
            self.method(),
            self.inner.points,
            self.inner.tail_mass,
            self.inner.aliasing_error
        )
    }
}

fn compound(
    result: act_core::Result<(Grid, CompoundReport)>,
) -> PyResult<(PyGrid, PyCompoundReport)> {
    let (inner, report) = result.map_err(to_py)?;
    Ok((PyGrid { inner }, PyCompoundReport { inner: report }))
}

/// Aggregate loss ``S = X_1 + ... + X_N`` by Panjer's recursion.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Grid
///     Severity on a grid; the result uses its step.
/// points : int
///     Points in the aggregate grid.
///
/// Returns
/// -------
/// tuple of (Grid, CompoundReport)
///
/// Raises
/// ------
/// ValueError
///     If ``points`` is 0 or ``P(S = 0)`` underflows (use ``fft``).
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import panjer
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> sev = Grid(1.0, [0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
/// >>> agg, report = panjer(Poisson(3.0), sev, 100)
/// >>> round(agg.mean(), 6)
/// 6.15
#[pyfunction]
pub(crate) fn panjer(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyGrid>,
    points: usize,
) -> PyResult<(PyGrid, PyCompoundReport)> {
    let n = AnyCount::extract(frequency)?;
    let sev = severity.inner.clone();
    compound(py.detach(|| act_aggregate::panjer(n.as_counting(), &sev, points)))
}

/// Aggregate loss ``S = X_1 + ... + X_N`` by fast Fourier transform.
///
/// Works for large claim counts that make ``panjer`` underflow. Check
/// ``report.aliasing_error`` before using the result.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Grid
/// points : int
///
/// Returns
/// -------
/// tuple of (Grid, CompoundReport)
///
/// Raises
/// ------
/// ValueError
///     If ``points`` is 0.
#[pyfunction]
pub(crate) fn fft(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyGrid>,
    points: usize,
) -> PyResult<(PyGrid, PyCompoundReport)> {
    let n = AnyCount::extract(frequency)?;
    let sev = severity.inner.clone();
    compound(py.detach(|| act_aggregate::fft(n.as_counting(), &sev, points)))
}

/// Simulated years of individual losses, for applying per-loss terms such
/// as reinsurance layers.
///
/// Created by ``simulate_events``. Year ``i`` was drawn from stream ``i`` of
/// the generator keyed by ``seed``, so results do not depend on the number of
/// threads.
#[pyclass(name = "EventSet", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyEventSet {
    pub(crate) inner: EventSet,
}

impl From<EventSet> for PyEventSet {
    fn from(inner: EventSet) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEventSet {
    /// Years of losses from elsewhere (your own simulation, or a
    /// catastrophe model's event loss table by year), optionally with the
    /// sum insured of the risk each loss hit, which a surplus treaty needs.
    ///
    /// Parameters
    /// ----------
    /// years : list of list of float
    ///     Each year's losses, in order.
    /// sums_insured : list of list of float, optional
    ///     The same shape: each loss's sum insured, at least the loss.
    /// seed : int, default 0
    ///     Recorded in results' provenance.
    /// times : list of list of float, optional
    ///     The same shape: each loss's time, as the fraction of the year
    ///     elapsed (in ``[0, 1]``, non-decreasing within a year), which
    ///     reinstatements pro rata as to time need.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[5.0, 2.0], [], [9.0]], [[10.0, 2.0], [], [50.0]])
    /// >>> e.counts(), e.sums_insured(2)
    /// ([2, 0, 1], [50.0])
    #[staticmethod]
    #[pyo3(signature = (years, sums_insured = None, seed = 0, times = None))]
    fn from_years(
        years: Vec<Vec<f64>>,
        sums_insured: Option<Vec<Vec<f64>>>,
        seed: u64,
        times: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        let shape: Vec<usize> = years.iter().map(Vec::len).collect();
        let mut inner = EventSet::from_years(years, seed).map_err(to_py)?;
        if let Some(si) = sums_insured {
            if si.iter().map(Vec::len).ne(shape.iter().copied()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sums_insured must have the shape of years",
                ));
            }
            inner = inner
                .with_sums_insured(si.into_iter().flatten().collect())
                .map_err(to_py)?;
        }
        if let Some(t) = times {
            if t.iter().map(Vec::len).ne(shape.iter().copied()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "times must have the shape of years",
                ));
            }
            inner = inner
                .with_times(t.into_iter().flatten().collect())
                .map_err(to_py)?;
        }
        Ok(Self { inner })
    }

    /// The same events at times spread uniformly over the year.
    ///
    /// Year ``i``'s losses take sorted uniform draws, in their order, from
    /// a stream of the generator keyed by the set's seed apart from the
    /// losses' own, so the losses are unchanged and any year replays alone.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[5.0, 2.0, 7.0]], seed=3).with_uniform_times()
    /// >>> t = e.times(0)
    /// >>> t == sorted(t) and e.has_times
    /// True
    fn with_uniform_times(&self) -> Self {
        Self {
            inner: self.inner.clone().with_uniform_times(),
        }
    }

    /// Whether the losses carry times.
    #[getter]
    fn has_times(&self) -> bool {
        self.inner.has_times()
    }

    /// Year ``sim``'s times, one per loss, or ``None``.
    ///
    /// Parameters
    /// ----------
    /// sim : int
    ///
    /// Returns
    /// -------
    /// list of float or None
    fn times(&self, sim: usize) -> PyResult<Option<Vec<f64>>> {
        if sim >= self.inner.n_sims() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.times(sim).map(<[f64]>::to_vec))
    }

    /// Whether the losses carry sums insured.
    #[getter]
    fn has_sums_insured(&self) -> bool {
        self.inner.has_sums_insured()
    }

    /// Year ``sim``'s sums insured, one per loss, or ``None``.
    ///
    /// Parameters
    /// ----------
    /// sim : int
    ///
    /// Returns
    /// -------
    /// list of float or None
    fn sums_insured(&self, sim: usize) -> PyResult<Option<Vec<f64>>> {
        if sim >= self.inner.n_sims() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.sums_insured(sim).map(<[f64]>::to_vec))
    }

    /// Number of simulated years.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The seed the years were drawn from.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// Year ``sim``'s individual losses, in the order they were drawn.
    ///
    /// Parameters
    /// ----------
    /// sim : int
    ///
    /// Returns
    /// -------
    /// list of float
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``sim`` is not a simulated year.
    fn events(&self, sim: usize) -> PyResult<Vec<f64>> {
        if sim >= self.inner.n_sims() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.events(sim).to_vec())
    }

    /// Number of losses in each year.
    ///
    /// Returns
    /// -------
    /// list of int
    fn counts(&self) -> Vec<usize> {
        self.inner.counts()
    }

    /// Each year's total loss.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn totals(&self) -> PyResult<PyPredictiveDistribution> {
        let inner = self.inner.totals().map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "EventSet(n_sims={}, seed={})",
            self.inner.n_sims(),
            self.inner.seed()
        )
    }
}

/// Simulates ``n_sims`` years of claims: a count from ``frequency``, then that
/// many independent losses from ``severity``.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
/// n_sims : int
/// seed : int
///
/// Returns
/// -------
/// EventSet
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is 0.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import simulate_events
/// >>> from prospicio.distributions import Lognormal, Poisson
/// >>> events = simulate_events(Poisson(5.0), Lognormal.from_mean_cv(1000.0, 1.0), 20_000, 42)
/// >>> abs(events.totals().mean() - 5000.0) < 75.0
/// True
#[pyfunction]
pub(crate) fn simulate_events(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    n_sims: usize,
    seed: u64,
) -> PyResult<PyEventSet> {
    let n = AnyCount::extract(frequency)?;
    let sev = extract_severity(severity)?;
    let inner = py
        .detach(|| act_aggregate::simulate_events(n.as_counting(), &sev, n_sims, seed))
        .map_err(to_py)?;
    Ok(PyEventSet { inner })
}
