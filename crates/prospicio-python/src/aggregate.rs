//! `prospicio.aggregate`: wrappers over `prospicio_aggregate` (compound
//! distributions and simulated events).

use prospicio_aggregate::{CompoundMethod, CompoundReport, EventSet};
use prospicio_prob::{Counting, Grid};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::distributions::{
    PyGrid, PyNegativeBinomial, PyPoisson, PyPredictiveDistribution, extract_severity,
};
use crate::to_py;

/// A claim count accepted by the aggregation functions.
pub(crate) enum AnyCount {
    Poisson(prospicio_prob::Poisson),
    NegativeBinomial(prospicio_prob::NegativeBinomial),
    Binomial(prospicio_prob::Binomial),
    Other(prospicio_prob::count_families::CountDist),
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
        if let Ok(n) = obj.extract::<PyRef<'_, crate::counts::PyCount>>() {
            return Ok(Self::Other(n.inner.clone()));
        }
        Err(PyTypeError::new_err(
            "expected a Poisson, a NegativeBinomial, a Binomial or a Count",
        ))
    }

    pub(crate) fn as_counting(&self) -> &(dyn Counting + Sync) {
        match self {
            Self::Poisson(n) => n,
            Self::NegativeBinomial(n) => n,
            Self::Binomial(n) => n,
            Self::Other(n) => n,
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
    fn panjer_ab(&self) -> Option<(f64, f64)> {
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
    result: prospicio_core::Result<(Grid, CompoundReport)>,
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
    compound(py.detach(|| prospicio_aggregate::panjer(n.as_counting(), &sev, points)))
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
    compound(py.detach(|| prospicio_aggregate::fft(n.as_counting(), &sev, points)))
}

/// A recommended FFT grid, from ``recommend_grid`` or ``fft_auto``.
#[pyclass(name = "GridSize", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyGridSize {
    pub(crate) inner: prospicio_aggregate::GridSize,
}

#[pymethods]
impl PyGridSize {
    /// Bucket size, a ``round_bucket`` rung.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step
    }

    /// Number of points, a power of two.
    #[getter]
    fn points(&self) -> usize {
        self.inner.points
    }

    /// The extent the grid was sized to cover.
    #[getter]
    fn extent(&self) -> f64 {
        self.inner.extent
    }

    /// What sized it: ``"moments"`` or ``"single_big_jump"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            prospicio_aggregate::SizingMethod::Moments => "moments",
            prospicio_aggregate::SizingMethod::SingleBigJump => "single_big_jump",
        }
    }

    /// The moment extent, or ``None`` with an infinite variance.
    #[getter]
    fn moment_extent(&self) -> Option<f64> {
        self.inner.moment_extent
    }

    /// The single-big-jump extent at ``p_star``, or ``None``.
    #[getter]
    fn jump_extent(&self) -> Option<f64> {
        self.inner.jump_extent
    }

    /// ``min(1, E[N] S_X(top))``: roughly the probability beyond the grid
    /// from one claim alone. Raise ``log2`` when it is not small.
    #[getter]
    fn tail_estimate(&self) -> f64 {
        self.inner.tail_estimate
    }

    fn __repr__(&self) -> String {
        format!(
            "GridSize(step={:?}, points={}, method={:?}, tail_estimate={:e})",
            self.inner.step,
            self.inner.points,
            self.method(),
            self.inner.tail_estimate
        )
    }
}

fn sizing(log2: u32, p: f64, p_star: f64) -> prospicio_aggregate::Sizing {
    prospicio_aggregate::Sizing {
        log2,
        p,
        p_star,
        ..prospicio_aggregate::Sizing::default()
    }
}

/// Rounds a bucket size up to a "nice" value, as ``aggregate``'s
/// ``round_bucket``: ``{1, 2, 4, 5, 8} * 10**k`` at 1 and above, a power of
/// two below.
///
/// Parameters
/// ----------
/// bs : float
///     Positive and finite.
///
/// Returns
/// -------
/// float
///
/// Raises
/// ------
/// ValueError
///     If ``bs`` is not positive and finite.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import round_bucket
/// >>> round_bucket(3.4), round_bucket(0.3)
/// (4.0, 0.5)
#[pyfunction]
pub(crate) fn round_bucket(bs: f64) -> PyResult<f64> {
    prospicio_aggregate::round_bucket(bs).map_err(to_py)
}

/// A grid for the compound distribution of ``frequency`` claims of
/// ``severity``, as ``aggregate`` sizes one when no bucket is given: the
/// larger of a lognormal or gamma fitted to the aggregate's mean and
/// variance at ``p``, and one big claim on a typical bulk at ``p``; then
/// one big claim at ``p_star`` when it fits at the same bucket.
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity
///     Any distribution but ``Sampled``.
/// log2 : int, default 16
///     At most ``2**log2`` points.
/// p : float, default 1 - 1e-5
/// p_star : float, default 1 - 1e-12
///
/// Returns
/// -------
/// GridSize
///
/// Raises
/// ------
/// ValueError
///     If the aggregate mean is not finite and positive, or ``log2`` is not
///     in ``1..=30``.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import recommend_grid
/// >>> from prospicio.distributions import Gamma, Poisson
/// >>> g = recommend_grid(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
/// >>> g.step, g.points, g.method
/// (0.0625, 65536, 'moments')
#[pyfunction]
#[pyo3(signature = (frequency, severity, log2 = 16, p = 1.0 - 1e-5, p_star = 1.0 - 1e-12))]
pub(crate) fn recommend_grid(
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    log2: u32,
    p: f64,
    p_star: f64,
) -> PyResult<PyGridSize> {
    let n = AnyCount::extract(frequency)?;
    let sev = crate::distributions::extract_severity(severity)?;
    let inner =
        prospicio_aggregate::recommend_grid(n.as_counting(), &sev, &sizing(log2, p, p_star))
            .map_err(to_py)?;
    Ok(PyGridSize { inner })
}

/// The compound distribution by FFT on the grid ``recommend_grid``
/// chooses, the severity discretized by rounding.
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity
/// log2 : int, default 16
/// p : float, default 1 - 1e-5
/// p_star : float, default 1 - 1e-12
///
/// Returns
/// -------
/// tuple of (Grid, CompoundReport, GridSize)
///
/// Raises
/// ------
/// ValueError
///     As ``recommend_grid``.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import fft_auto
/// >>> from prospicio.distributions import Lognormal, Poisson
/// >>> agg, report, size = fft_auto(Poisson(5.0), Lognormal.from_mean_cv(100.0, 1.0))
/// >>> round(agg.mean()), report.aliasing_error < 1e-12
/// (500, True)
#[pyfunction]
#[pyo3(signature = (frequency, severity, log2 = 16, p = 1.0 - 1e-5, p_star = 1.0 - 1e-12))]
pub(crate) fn fft_auto(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    log2: u32,
    p: f64,
    p_star: f64,
) -> PyResult<(PyGrid, PyCompoundReport, PyGridSize)> {
    let n = AnyCount::extract(frequency)?;
    let sev = crate::distributions::extract_severity(severity)?;
    let s = sizing(log2, p, p_star);
    let (grid, report, size) = py
        .detach(|| prospicio_aggregate::fft_auto(n.as_counting(), &sev, &s))
        .map_err(to_py)?;
    Ok((
        PyGrid { inner: grid },
        PyCompoundReport { inner: report },
        PyGridSize { inner: size },
    ))
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
    ///     Recorded in results' provenance; the samplers are not, since
    ///     the losses were not drawn here.
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

    /// The same events at times drawn from a seasonal density.
    ///
    /// The year is cut into ``len(weights)`` equal periods (12 for months,
    /// 52 for weeks) starting at the contract's inception, and a loss
    /// falls in period ``k`` with probability ``weights[k] / sum(weights)``,
    /// uniformly within it. A zero weight means no losses in that period.
    /// The draws are those of ``with_uniform_times``, mapped through the
    /// season's quantile, so equal weights give the uniform times.
    ///
    /// Parameters
    /// ----------
    /// weights : list of float
    ///     Each period's relative weight: non-negative, not all zero.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[5.0, 2.0, 7.0]], seed=3)
    /// >>> t = e.with_seasonal_times([0.0, 1.0]).times(0)
    /// >>> all(x >= 0.5 for x in t) and t == sorted(t)
    /// True
    fn with_seasonal_times(&self, weights: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_seasonal_times(&weights)
                .map_err(to_py)?,
        })
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
        .detach(|| prospicio_aggregate::simulate_events(n.as_counting(), &sev, n_sims, seed))
        .map_err(to_py)?;
    Ok(PyEventSet { inner })
}
