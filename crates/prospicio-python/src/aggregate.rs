//! `prospicio.aggregate`: wrappers over `prospicio_aggregate` (compound
//! distributions and simulated events).

use prospicio_aggregate::{CompoundMethod, CompoundReport, EventLossTable, EventSet};
use prospicio_prob::{Counting, Grid, Severity, SeverityDist};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    fn third_moment(&self) -> f64 {
        self.as_counting().third_moment()
    }
    fn third_cumulant(&self) -> f64 {
        self.as_counting().third_cumulant()
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
/// Assumes the claim count independent of the severities and the severities
/// independent and identically distributed. For dependence use
/// ``fft_bivariate`` or ``Clash`` (two lines hit by the same events), one
/// ``Count.mixed_poisson`` on a ``Mixture`` (components sharing a mixing
/// variable) or ``PredictiveDistribution.reorder_groups`` on simulated totals.
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
/// Assumes the claim count independent of the severities and the severities
/// independent and identically distributed. For dependence use
/// ``fft_bivariate`` or ``Clash`` (two lines hit by the same events), one
/// ``Count.mixed_poisson`` on a ``Mixture`` (components sharing a mixing
/// variable) or ``PredictiveDistribution.reorder_groups`` on simulated totals.
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
/// Assumes the claim count independent of the severities and the severities
/// independent and identically distributed. For dependence use
/// ``fft_bivariate`` or ``Clash`` (two lines hit by the same events), one
/// ``Count.mixed_poisson`` on a ``Mixture`` (components sharing a mixing
/// variable) or ``PredictiveDistribution.reorder_groups`` on simulated totals.
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

fn extract_units(units: &Bound<'_, PyAny>) -> PyResult<Vec<(AnyCount, SeverityDist)>> {
    let mut out = Vec::new();
    for unit in units.try_iter()? {
        let (n, x): (Bound<'_, PyAny>, Bound<'_, PyAny>) = unit?.extract()?;
        out.push((
            AnyCount::extract(&n)?,
            crate::distributions::extract_severity(&x)?,
        ));
    }
    Ok(out)
}

fn unit_refs(units: &[(AnyCount, SeverityDist)]) -> Vec<(&dyn Counting, &dyn Severity)> {
    units
        .iter()
        .map(|(n, x)| (n.as_counting() as &dyn Counting, x as &dyn Severity))
        .collect()
}

/// A grid for the sum of independent compound units, as ``aggregate``
/// sizes a ``Portfolio``: ``recommend_grid``'s steps on the total, the
/// single big jump taken on whichever unit needs the most room.
///
/// Parameters
/// ----------
/// units : list of (frequency, severity) pairs
/// log2 : int, default 16
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
///     As ``recommend_grid``, or with no units.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import recommend_portfolio_grid
/// >>> from prospicio.distributions import Gamma, Poisson
/// >>> g = recommend_portfolio_grid([(Poisson(10.0), Gamma(2.0, 50.0)),
/// ...                               (Poisson(2.0), Gamma(0.5, 1000.0))])
/// >>> g.points <= 2**16
/// True
#[pyfunction]
#[pyo3(signature = (units, log2 = 16, p = 1.0 - 1e-5, p_star = 1.0 - 1e-12))]
pub(crate) fn recommend_portfolio_grid(
    units: &Bound<'_, PyAny>,
    log2: u32,
    p: f64,
    p_star: f64,
) -> PyResult<PyGridSize> {
    let units = extract_units(units)?;
    let inner =
        prospicio_aggregate::recommend_portfolio_grid(&unit_refs(&units), &sizing(log2, p, p_star))
            .map_err(to_py)?;
    Ok(PyGridSize { inner })
}

/// The cells an audit built and the one it chose, from ``audit_grid`` or
/// ``audit_portfolio_grid``.
#[pyclass(name = "GridAudit", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyGridAudit {
    inner: prospicio_aggregate::GridAudit,
}

fn cell_dict<'py>(
    py: Python<'py>,
    c: &prospicio_aggregate::GridCell,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("d_step", c.d_step)?;
    d.set_item("d_log2", c.d_log2)?;
    d.set_item("step", c.step)?;
    d.set_item("log2", c.log2)?;
    d.set_item("score", c.score)?;
    let t = &c.terms;
    for (k, v) in [
        ("severity_mean", t.severity_mean),
        ("severity_cv", t.severity_cv),
        ("severity_skew", t.severity_skew),
        ("aggregate_mean", t.aggregate_mean),
        ("aggregate_cv", t.aggregate_cv),
        ("aggregate_skew", t.aggregate_skew),
    ] {
        d.set_item(k, v)?;
    }
    d.set_item("tail_mass", c.tail_mass)?;
    d.set_item("aliasing_error", c.aliasing_error)?;
    d.set_item("defective", c.defective)?;
    d.set_item("selected", c.selected)?;
    d.set_item("note", &c.note)?;
    Ok(d)
}

#[pymethods]
impl PyGridAudit {
    /// Every cell built, the center first, as dicts: ``d_step`` and
    /// ``d_log2`` (the moves from the center, in doublings), ``step``,
    /// ``log2``, ``score``, the six terms (``severity_mean`` to
    /// ``aggregate_skew``, ``None`` when skipped), ``tail_mass`` (the mass
    /// beyond the grid), ``aliasing_error``, ``defective`` (mass lost, so
    /// never chosen while a sound cell exists), ``selected`` and ``note``.
    #[getter]
    fn cells<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        self.inner.cells.iter().map(|c| cell_dict(py, c)).collect()
    }

    /// The index of the chosen cell in ``cells``.
    #[getter]
    fn selected(&self) -> usize {
        self.inner.selected
    }

    /// The chosen cell.
    #[getter]
    fn best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        cell_dict(py, self.inner.best())
    }

    /// The chosen step.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.best().step
    }

    /// The chosen ``log2``.
    #[getter]
    fn log2(&self) -> u32 {
        self.inner.best().log2
    }

    /// False when the center met the target and nothing else was built.
    #[getter]
    fn probed(&self) -> bool {
        self.inner.probed
    }

    /// Whether the chosen cell has twice the center's points.
    #[getter]
    fn grew(&self) -> bool {
        self.inner.grew
    }

    /// Why that cell was chosen.
    #[getter]
    fn reason(&self) -> &str {
        &self.inner.reason
    }

    fn __len__(&self) -> usize {
        self.inner.cells.len()
    }

    fn __repr__(&self) -> String {
        let b = self.inner.best();
        format!(
            "GridAudit(step={:?}, log2={}, score={:e}, cells={}, grew={})",
            b.step,
            b.log2,
            b.score,
            self.inner.cells.len(),
            if self.inner.grew { "True" } else { "False" }
        )
    }
}

fn audit_settings(
    eps: f64,
    power: f64,
    good_enough: f64,
    max_doublings: u32,
    log2_cap: u32,
    log2_floor: u32,
) -> prospicio_aggregate::AuditSettings {
    prospicio_aggregate::AuditSettings {
        eps,
        power,
        good_enough,
        max_doublings,
        log2_cap,
        log2_floor,
    }
}

/// Audits an FFT grid as ``aggregate`` 1.2.0's ``sharpen`` does: builds
/// the compound distribution at the center and, unless it meets the
/// target, at doubled and halved steps on the same and half the points,
/// scores each cell by six moment errors against the exact moments, and
/// chooses the best sound cell, preferring fewer points. Twice the points
/// is tried only when nothing smaller meets the target.
///
/// A score of 1 is ``aggregate``'s validation boundary: a relative error
/// of ``eps`` in a mean, ``10 eps`` in a cv, ``100 eps`` in a skewness.
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity
///     Any distribution but ``Sampled``.
/// step : float, optional
///     The center's step; by default ``recommend_grid``'s at ``log2``.
/// log2 : int, optional
///     The center's ``2**log2`` points; by default 16, or
///     ``recommend_grid``'s when ``step`` is not given either.
/// eps : float, default 1e-4
/// power : float, default 2
///     The power mean of the terms.
/// good_enough : float, default 0.5
///     The target score.
/// max_doublings : int, default 4
///     Steps doubled or halved at most this many times each way.
/// log2_cap : int, default 20
///     ``log2`` never grows past it.
/// log2_floor : int, default 8
///     The half-size row is tried only above it.
///
/// Returns
/// -------
/// GridAudit
///
/// Raises
/// ------
/// ValueError
///     If ``step`` is not positive and finite or ``log2`` is not in
///     ``1..=30``.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import audit_grid
/// >>> from prospicio.distributions import Gamma, Poisson
/// >>> a = audit_grid(Poisson(50.0), Gamma.from_mean_cv(10.0, 0.5), step=0.5, log2=10)
/// >>> a.step, a.log2, a.grew
/// (1.0, 11, True)
#[pyfunction]
#[pyo3(signature = (frequency, severity, step = None, log2 = None, *, eps = 1e-4, power = 2.0,
                    good_enough = 0.5, max_doublings = 4, log2_cap = 20, log2_floor = 8))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audit_grid(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    step: Option<f64>,
    log2: Option<u32>,
    eps: f64,
    power: f64,
    good_enough: f64,
    max_doublings: u32,
    log2_cap: u32,
    log2_floor: u32,
) -> PyResult<PyGridAudit> {
    let n = AnyCount::extract(frequency)?;
    let sev = crate::distributions::extract_severity(severity)?;
    let s = audit_settings(eps, power, good_enough, max_doublings, log2_cap, log2_floor);
    let inner = py
        .detach(|| {
            let (step, log2) = match step {
                Some(h) => (h, log2.unwrap_or(16)),
                None => {
                    let g = prospicio_aggregate::recommend_grid(
                        n.as_counting(),
                        &sev,
                        &sizing(log2.unwrap_or(16), 1.0 - 1e-5, 1.0 - 1e-12),
                    )?;
                    (g.step, g.points.trailing_zeros())
                }
            };
            prospicio_aggregate::audit_grid(n.as_counting(), &sev, step, log2, &s)
        })
        .map_err(to_py)?;
    Ok(PyGridAudit { inner })
}

/// ``audit_grid`` for the sum of independent units on one grid. The
/// severity terms score the claim-weighted mixture of the units'
/// severities, the aggregate terms the total.
///
/// Parameters
/// ----------
/// units : list of (frequency, severity) pairs
/// step, log2, eps, power, good_enough, max_doublings, log2_cap, log2_floor
///     As ``audit_grid``; the default center is
///     ``recommend_portfolio_grid``'s.
///
/// Returns
/// -------
/// GridAudit
///
/// Raises
/// ------
/// ValueError
///     As ``audit_grid``, or with no units.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import audit_portfolio_grid
/// >>> from prospicio.distributions import Gamma, Poisson
/// >>> a = audit_portfolio_grid([(Poisson(10.0), Gamma(2.0, 50.0)),
/// ...                           (Poisson(2.0), Gamma(0.5, 1000.0))], log2=12)
/// >>> a.best["score"] < 0.5
/// True
#[pyfunction]
#[pyo3(signature = (units, step = None, log2 = None, *, eps = 1e-4, power = 2.0,
                    good_enough = 0.5, max_doublings = 4, log2_cap = 20, log2_floor = 8))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audit_portfolio_grid(
    py: Python<'_>,
    units: &Bound<'_, PyAny>,
    step: Option<f64>,
    log2: Option<u32>,
    eps: f64,
    power: f64,
    good_enough: f64,
    max_doublings: u32,
    log2_cap: u32,
    log2_floor: u32,
) -> PyResult<PyGridAudit> {
    let units = extract_units(units)?;
    let s = audit_settings(eps, power, good_enough, max_doublings, log2_cap, log2_floor);
    let inner = py
        .detach(|| {
            let refs = unit_refs(&units);
            let (step, log2) = match step {
                Some(h) => (h, log2.unwrap_or(16)),
                None => {
                    let g = prospicio_aggregate::recommend_portfolio_grid(
                        &refs,
                        &sizing(log2.unwrap_or(16), 1.0 - 1e-5, 1.0 - 1e-12),
                    )?;
                    (g.step, g.points.trailing_zeros())
                }
            };
            prospicio_aggregate::audit_portfolio_grid(&refs, step, log2, &s)
        })
        .map_err(to_py)?;
    Ok(PyGridAudit { inner })
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

    /// Years of losses in long form, as a year loss table lists them: one
    /// row per loss with its year, an index from 0 to ``n_years - 1``.
    ///
    /// Within a year the losses keep their order; years with no row are
    /// empty.
    ///
    /// Parameters
    /// ----------
    /// years : list of int
    /// losses : list of float
    /// n_years : int
    ///     The number of simulated years, including those with no loss.
    /// seed : int, default 0
    /// labels : dict of str to list of str, optional
    ///     Each row's labels, a list per label dimension (``{"peril":
    ///     [...], "class": [...]}``), as ``with_labels`` takes them.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_long([2, 0, 2], [5.0, 1.0, 7.0], 4)
    /// >>> e.counts(), e.events(2)
    /// ([1, 0, 2, 0], [5.0, 7.0])
    /// >>> e = EventSet.from_long([2, 0, 2], [5.0, 1.0, 7.0], 4, labels={"peril": ["wind", "quake", "wind"]})
    /// >>> e.labels(0)
    /// {'peril': ['quake']}
    #[staticmethod]
    #[pyo3(signature = (years, losses, n_years, seed = 0, labels = None))]
    fn from_long(
        years: Vec<usize>,
        losses: Vec<f64>,
        n_years: usize,
        seed: u64,
        labels: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let inner = match labels {
            None => EventSet::from_long(&years, &losses, n_years, seed),
            Some(labels) => EventSet::from_long_labeled(
                &years,
                &losses,
                labels_from_dict(labels)?,
                n_years,
                seed,
            ),
        }
        .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Years of losses from a year loss table's columns, checked and
    /// expanded to a run's simulations.
    ///
    /// Rows are ordered by year, then slot (the given order breaking ties),
    /// and each row is one loss: rows sharing a slot, an event hitting
    /// several classes, stay separate losses, so an occurrence layer sees
    /// each of them. ``prospicio.aggregate.read_ylt`` reads the columns
    /// from a CSV or Parquet file.
    ///
    /// Parameters
    /// ----------
    /// years : list of int
    ///     Each row's year, from 1.
    /// losses : list of float
    ///     Finite and non-negative.
    /// slots : list of int, optional
    ///     Each row's event slot, which orders a year's rows.
    /// labels : dict of str to list of str, optional
    ///     Each row's labels, a list per dimension (``{"peril": [...],
    ///     "class": [...]}``).
    /// n_years : int, optional
    ///     The table's years, at least its largest year; by default the
    ///     largest year.
    /// n_sims : int, optional
    ///     The simulations to expand to, by default ``n_years``.
    /// expand : {"tile", "resample"}, default "tile"
    ///     ``"tile"``: simulation ``i`` (from 0) takes year ``i % n_years``,
    ///     so ``n_sims`` must be a multiple of ``n_years``. ``"resample"``:
    ///     each simulation takes a year drawn uniformly with replacement
    ///     from its own stream of ``seed``.
    /// seed : int, default 0
    /// unique_by : list of str, optional
    ///     With slots, no two rows may share a year, a slot and their values
    ///     in these label dimensions (``["peril"]``: one slot may hit several
    ///     classes, not repeat a peril); by default every label dimension.
    /// expected : dict of str to list of str, optional
    ///     The values allowed in each named dimension (``{"peril": ["WS",
    ///     "EQ"]}``).
    /// source_hash : str, optional
    ///     A hash of the source, recorded as results' input hash.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_ylt([2, 1, 2], [5.0, 1.0, 7.0], slots=[9, 1, 3], n_sims=4)
    /// >>> e.events(1), e.events(3)
    /// ([7.0, 5.0], [7.0, 5.0])
    #[staticmethod]
    #[pyo3(signature = (
        years, losses, *, slots = None, labels = None, n_years = None, n_sims = None,
        expand = "tile", seed = 0, unique_by = None, expected = None, source_hash = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_ylt(
        years: Vec<usize>,
        losses: Vec<f64>,
        slots: Option<Vec<u64>>,
        labels: Option<&Bound<'_, PyDict>>,
        n_years: Option<usize>,
        n_sims: Option<usize>,
        expand: &str,
        seed: u64,
        unique_by: Option<Vec<String>>,
        expected: Option<&Bound<'_, PyDict>>,
        source_hash: Option<String>,
    ) -> PyResult<Self> {
        use prospicio_aggregate::ylt::{Expansion, YearLossTable};
        let expansion = match expand {
            "tile" => Expansion::Tile,
            "resample" => Expansion::Resample { seed },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "expand must be \"tile\" or \"resample\", not {other:?}"
                )));
            }
        };
        let mut ylt = YearLossTable::new(&years, &losses).map_err(to_py)?;
        let has_slots = slots.is_some();
        if let Some(slots) = slots {
            ylt = ylt.with_slots(slots).map_err(to_py)?;
        }
        let labels = labels
            .map(labels_from_dict)
            .transpose()?
            .unwrap_or_default();
        let dims: Vec<String> = labels.iter().map(|(d, _)| d.clone()).collect();
        ylt = ylt.with_labels(labels).map_err(to_py)?;
        if let Some(n) = n_years {
            ylt = ylt.with_n_years(n).map_err(to_py)?;
        }
        if let Some(h) = source_hash {
            ylt = ylt.with_source_hash(h);
        }
        if has_slots {
            let by = unique_by.unwrap_or(dims);
            ylt.check_unique(&by.iter().map(String::as_str).collect::<Vec<_>>())
                .map_err(to_py)?;
        }
        if let Some(expected) = expected {
            for (dim, values) in labels_from_dict(expected)? {
                ylt.check_values(&dim, &values.iter().map(String::as_str).collect::<Vec<_>>())
                    .map_err(to_py)?;
            }
        }
        Ok(Self {
            inner: ylt.events(n_sims, expansion).map_err(to_py)?,
        })
    }

    /// The same events with labels, a peril or a class say.
    ///
    /// ``Tower.apply`` then gives gross, each layer's ceded loss and net by
    /// label, with the label dimensions after ``kind`` and ``layer``;
    /// ``merge`` keeps the labels of either set. Replaces any labels the
    /// set had.
    ///
    /// Parameters
    /// ----------
    /// labels : dict of str to list of str
    ///     One list per label dimension, a value per loss, years in order as
    ///     ``events`` lists them. The names may not be ``"kind"`` or
    ///     ``"layer"``.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[5.0, 2.0], [9.0]]).with_labels({"peril": ["wind", "flood", "wind"]})
    /// >>> e.label_dims, e.label_combinations
    /// (['peril'], [['flood'], ['wind']])
    fn with_labels(&self, labels: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_labels(labels_from_dict(labels)?)
                .map_err(to_py)?,
        })
    }

    /// The same events with an aggregate part: amounts in each year that
    /// are no event and have no time, attritional losses say, one column
    /// per component of ``part``.
    ///
    /// ``part`` needs as many simulations as the set has years; simulation
    /// ``i`` goes with year ``i``. Its dimensions are label dimensions, so
    /// the part can be given by peril or class: the set's labels get the
    /// part's dimensions not among them, with ``""`` for every loss. A part
    /// with no dimensions is one unlabelled amount a year. Replaces any
    /// aggregate part the set had; label the events first.
    ///
    /// ``Tower.apply`` counts the part in gross and net, and only layers
    /// that see it (``Layer.sees_aggregate_part``: quota shares and stop
    /// losses) cede from it. In each year it comes before the events, so a
    /// binding annual deductible is used up on it first, unless the layer
    /// shares annual terms pro rata. ``totals`` includes it; ``maxima`` and
    /// ``counts`` do not.
    ///
    /// Parameters
    /// ----------
    /// part : PredictiveDistribution
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``part`` has another number of simulations, a negative or
    ///     non-finite amount, or a dimension ``"kind"`` or ``"layer"``, or
    ///     if labels are given after the part.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> attritional = PredictiveDistribution([], [()], [[10.0], [40.0], [30.0]])
    /// >>> e = EventSet.from_years([[], [50.0], [5.0, 60.0]]).with_aggregate_part(attritional)
    /// >>> r = Tower([Layer("30x25", 30.0, 25.0)]).apply(e)
    /// >>> list(r.marginal(("ceded", "30x25")).draws)
    /// [0.0, 25.0, 30.0]
    fn with_aggregate_part(&self, part: &PyPredictiveDistribution) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_aggregate_part(&part.inner)
                .map_err(to_py)?,
        })
    }

    /// Whether the years carry an aggregate part.
    #[getter]
    fn has_aggregate_part(&self) -> bool {
        self.inner.has_aggregate_part()
    }

    /// Year ``sim``'s aggregate part, one amount per column in the order of
    /// ``aggregate_part_labels``, or ``None``.
    ///
    /// Parameters
    /// ----------
    /// sim : int
    ///
    /// Returns
    /// -------
    /// list of float or None
    fn aggregate_part(&self, sim: usize) -> PyResult<Option<Vec<f64>>> {
        if sim >= self.inner.n_sims() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.aggregate_part(sim).map(<[f64]>::to_vec))
    }

    /// Each column of the aggregate part's labels, one value per label
    /// dimension, or ``None`` without a part.
    #[getter]
    fn aggregate_part_labels(&self) -> Option<Vec<Vec<String>>> {
        self.inner
            .aggregate_part_labels()
            .map(|cols| cols.into_iter().map(<[String]>::to_vec).collect())
    }

    /// Whether the losses carry labels.
    #[getter]
    fn has_labels(&self) -> bool {
        self.inner.has_labels()
    }

    /// The label dimensions, empty without labels.
    #[getter]
    fn label_dims(&self) -> Vec<String> {
        self.inner.label_dims().to_vec()
    }

    /// The distinct combinations of label values, one value per dimension,
    /// sorted: the labels of ``Tower.apply``'s components.
    #[getter]
    fn label_combinations(&self) -> Vec<Vec<String>> {
        self.inner.label_combinations().to_vec()
    }

    /// Year ``sim``'s labels, a list per dimension with a value per loss,
    /// or ``None``.
    ///
    /// Parameters
    /// ----------
    /// sim : int
    ///
    /// Returns
    /// -------
    /// dict of str to list of str or None
    fn labels<'py>(&self, py: Python<'py>, sim: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        if sim >= self.inner.n_sims() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        let Some(year) = self.inner.labels(sim) else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        for (d, dim) in self.inner.label_dims().iter().enumerate() {
            let values: Vec<&str> = year.iter().map(|v| v[d].as_str()).collect();
            out.set_item(dim, values)?;
        }
        Ok(Some(out))
    }

    /// Both sets' losses in each year: a catastrophe event set with a
    /// non-catastrophe one, say.
    ///
    /// The sets need the same number of years. With times on both, each
    /// year's losses are merged in time order (this set's first on a tie);
    /// with times on neither, this set's losses come first. Sums insured are
    /// kept when both sets carry them. Labels are kept when either carries
    /// them: this set's dimensions, then the other's new ones, with ``""``
    /// for a loss without a value in a dimension.
    ///
    /// Parameters
    /// ----------
    /// other : EventSet
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> cat = EventSet.from_years([[100.0], []], times=[[0.5], []])
    /// >>> other = EventSet.from_years([[3.0, 4.0], [5.0]], times=[[0.2, 0.7], [0.1]])
    /// >>> cat.merge(other).events(0)
    /// [3.0, 100.0, 4.0]
    fn merge(&self, other: PyRef<'_, PyEventSet>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.merge(&other.inner).map_err(to_py)?,
        })
    }

    /// The years in another order: year ``i`` of the result is year
    /// ``index[i]`` of this set, with its losses, sums insured, times and
    /// labels. Pass the index ``PredictiveDistribution.reorder_groups``
    /// returns for a class (``return_index=True``) to keep the class's
    /// events with its reordered totals; any index into the years works,
    /// so a resample with repeats does too.
    ///
    /// Parameters
    /// ----------
    /// index : list of int
    ///     0-based years of this set.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the index is empty or names a year past the last.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[], [50.0], [5.0, 60.0]])
    /// >>> [e.permute([2, 0, 1]).events(i) for i in range(3)]
    /// [[5.0, 60.0], [], [50.0]]
    fn permute(&self, index: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.permute(&index).map_err(to_py)?,
        })
    }

    /// The losses whose label ``label`` takes one of ``values``, every
    /// year kept (a year with none of them is empty), with their sums
    /// insured, times and labels, and the aggregate part's columns with
    /// one of the values: what a layer scoped to those values
    /// (``Layer.with_scope``) sees.
    ///
    /// Parameters
    /// ----------
    /// label : str
    /// values : list of str
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Without labels, without the dimension, or with no values.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[20.0, 15.0], [8.0]]).with_labels(
    /// ...     {"peril": ["quake", "wind", "quake"]})
    /// >>> w = e.filter("peril", ["wind"])
    /// >>> w.n_sims, w.events(0), w.events(1)
    /// (2, [15.0], [])
    fn filter(&self, label: &str, values: Vec<String>) -> PyResult<Self> {
        let values: Vec<&str> = values.iter().map(String::as_str).collect();
        Ok(Self {
            inner: self.inner.filter(label, &values).map_err(to_py)?,
        })
    }

    /// Each year's losses grouped into occurrences by an hours clause.
    ///
    /// Losses within ``window_hours`` of the first loss of a window are one
    /// occurrence, whose loss is their sum and whose time is the window's
    /// start; the next loss after the window opens the next one. A year is
    /// 8,760 hours. Windows open at each first loss in turn, not where they
    /// would recover most. Needs times; sums insured and labels are not
    /// carried (group each peril's set before merging them).
    ///
    /// Parameters
    /// ----------
    /// window_hours : float
    ///     Such as 72 for a windstorm, 168 for a flood.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> h = 1 / 8760
    /// >>> e = EventSet.from_years([[10.0, 20.0, 5.0]], times=[[0.5, 0.5 + 30 * h, 0.5 + 80 * h]])
    /// >>> e.grouped(72.0).events(0)
    /// [30.0, 5.0]
    fn grouped(&self, window_hours: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.grouped(window_hours).map_err(to_py)?,
        })
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

    /// Splits the simulations into a term of ``n_years`` consecutive
    /// blocks, for ``Tower.apply_term``: year ``t`` (from 0) holds
    /// simulations ``t * n`` to ``(t + 1) * n - 1``, with ``n = n_sims //
    /// n_years``, so path ``j``'s year ``t`` is simulation ``t * n + j``.
    /// Sums insured and times go with their losses.
    ///
    /// Parameters
    /// ----------
    /// n_years : int
    ///     Must divide ``n_sims``.
    ///
    /// Returns
    /// -------
    /// list of EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[1.0], [2.0], [3.0], [4.0]])
    /// >>> [y.events(1) for y in e.term_years(2)]
    /// [[2.0], [4.0]]
    fn term_years(&self, n_years: usize) -> PyResult<Vec<Self>> {
        Ok(self
            .inner
            .term_years(n_years)
            .map_err(to_py)?
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// A term of ``n_years`` underwriting years whose yearly totals (each
    /// year's losses and aggregate part) have a rank correlation, set by
    /// Iman–Conover. Each year's simulations move as whole years, keeping
    /// their events, times, sums insured, labels and aggregate part, so the
    /// years go to ``Tower.apply_term`` as they come.
    ///
    /// Parameters
    /// ----------
    /// n_years : int
    ///     At least 2.
    /// correlation : float or list of list of float
    ///     One correlation for every pair of years, or the full matrix.
    /// kind : {"spearman", "pearson"}
    ///     ``"spearman"`` (the default) targets Spearman's rho of the
    ///     totals; ``"pearson"`` the correlation of their normal scores, as
    ///     ``PredictiveDistribution.reorder_groups``.
    /// draws : {"same", "blocks"}
    ///     ``"same"`` (the default): every year reorders all the
    ///     simulations, so each has exactly the one-year distribution and
    ///     the term's mean the one-year mean; ``"blocks"``: year ``t`` is
    ///     block ``t`` of ``term_years``.
    /// seed : int
    ///     The same seed gives the same years.
    ///
    /// Returns
    /// -------
    /// list of EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> e = EventSet.from_years([[float(i)] for i in range(10)])
    /// >>> years = e.correlated_years(2, 0.9, seed=1)
    /// >>> sorted(y.events(i)[0] for y in years[:1] for i in range(10)) == [float(i) for i in range(10)]
    /// True
    #[pyo3(signature = (n_years, correlation, *, kind = "spearman", draws = "same", seed = 0))]
    fn correlated_years(
        &self,
        py: Python<'_>,
        n_years: usize,
        correlation: &Bound<'_, PyAny>,
        kind: &str,
        draws: &str,
        seed: u64,
    ) -> PyResult<Vec<Self>> {
        let correlation: Vec<f64> = match correlation.extract::<f64>() {
            Ok(rho) => vec![rho],
            Err(_) => correlation
                .extract::<Vec<Vec<f64>>>()?
                .into_iter()
                .flatten()
                .collect(),
        };
        let (kind, draws) = (correlation_kind(kind)?, year_draws(draws)?);
        let inner = &self.inner;
        Ok(py
            .detach(|| inner.correlated_years(n_years, &correlation, kind, draws, seed))
            .map_err(to_py)?
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// The correlation between every pair of ``n_years`` years for which
    /// ``correlated_years`` gives a term whose mean yearly total has
    /// ``ratio`` times the standard deviation of one year's total, found by
    /// bisection on these simulations. With the same premium each year,
    /// that is the ratio of the term's loss ratio's spread to one year's.
    /// Independent years give ``1 / sqrt(n_years)``, identical ones 1.
    ///
    /// Parameters
    /// ----------
    /// n_years : int
    /// ratio : float
    ///     In ``[0, 1]``.
    /// kind, draws, seed
    ///     As ``correlated_years``.
    ///
    /// Returns
    /// -------
    /// tuple of (float, float, list of EventSet)
    ///     The correlation, the ratio it achieves and the years under it.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> from prospicio.distributions import Lognormal, Poisson
    /// >>> from prospicio.aggregate import simulate_events
    /// >>> e = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(100.0, 1.0), 4000, seed=7)
    /// >>> rho, achieved, years = e.calibrate_year_correlation(3, 0.75, seed=11)
    /// >>> abs(achieved - 0.75) < 1e-3, 0 < rho < 1, len(years)
    /// (True, True, 3)
    #[pyo3(signature = (n_years, ratio, *, kind = "spearman", draws = "same", seed = 0))]
    fn calibrate_year_correlation(
        &self,
        py: Python<'_>,
        n_years: usize,
        ratio: f64,
        kind: &str,
        draws: &str,
        seed: u64,
    ) -> PyResult<(f64, f64, Vec<Self>)> {
        let (kind, draws) = (correlation_kind(kind)?, year_draws(draws)?);
        let inner = &self.inner;
        let fit = py
            .detach(|| inner.calibrate_year_correlation(n_years, ratio, kind, draws, seed))
            .map_err(to_py)?;
        Ok((
            fit.correlation,
            fit.ratio,
            fit.years.into_iter().map(|inner| Self { inner }).collect(),
        ))
    }

    /// The seed the years were drawn from.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// A hash of the years' source (a year loss table's file), which
    /// results record as their provenance's input hash, or ``None``.
    #[getter]
    fn input_hash(&self) -> Option<String> {
        self.inner.input_hash().map(str::to_string)
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

    /// Each year's largest loss (0 in a year with none): its survival
    /// function is the simulated occurrence exceedance (OEP) curve, and its
    /// quantiles the occurrence PMLs.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn maxima(&self) -> PyResult<PyPredictiveDistribution> {
        let inner = self.inner.maxima().map_err(to_py)?;
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

/// The chance that the year's largest occurrence exceeds ``x``,
/// ``1 - G_N(F(x))`` with ``G_N`` the count's generating function: the
/// occurrence exceedance (OEP) curve.
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity
/// x : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.aggregate import occurrence_exceedance
/// >>> from prospicio.distributions import Lognormal, Poisson
/// >>> sev = Lognormal.from_mean_cv(100.0, 2.0)
/// >>> oep = occurrence_exceedance(Poisson(2.0), sev, 500.0)
/// >>> abs(oep - (1 - math.exp(-2 * (1 - sev.cdf(500.0))))) < 1e-12
/// True
#[pyfunction]
pub(crate) fn occurrence_exceedance(
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    x: f64,
) -> PyResult<f64> {
    let n = AnyCount::extract(frequency)?;
    let sev = extract_severity(severity)?;
    Ok(prospicio_aggregate::occurrence_exceedance(
        n.as_counting(),
        &sev,
        x,
    ))
}

/// One point of the occurrence PML curve, from ``occurrence_pml``.
#[pyclass(name = "OccurrencePml", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyOccurrencePml {
    inner: prospicio_aggregate::OccurrencePml,
}

#[pymethods]
impl PyOccurrencePml {
    /// The annual probability asked for.
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability
    }

    /// The loss whose occurrence exceedance is the probability.
    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }

    /// The severity's exceedance ``P(X > loss)``.
    #[getter]
    fn severity_survival(&self) -> f64 {
        self.inner.severity_survival
    }

    /// ``1 / (E[N] P(X > loss))``, the average time between occurrences
    /// above the loss; it can be shorter than a year.
    #[getter]
    fn occurrence_return_period(&self) -> f64 {
        self.inner.occurrence_return_period
    }

    /// ``1 / probability``, the average time between years with such an
    /// occurrence.
    #[getter]
    fn annual_return_period(&self) -> f64 {
        self.inner.annual_return_period
    }

    fn __repr__(&self) -> String {
        format!(
            "OccurrencePml(probability={:?}, loss={:?})",
            self.inner.probability, self.inner.loss
        )
    }
}

/// The occurrence PML at an annual probability ``p``: the loss whose
/// occurrence exceedance is ``p``, for any claim count (the Python package
/// ``aggregate``'s ``oep`` covers Poisson counts).
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity
/// p : float
///     In ``(0, 1 - P(N = 0))``.
///
/// Returns
/// -------
/// OccurrencePml
///
/// Raises
/// ------
/// ValueError
///     If ``p`` is outside ``(0, 1 - P(N = 0))``.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import occurrence_pml
/// >>> from prospicio.distributions import Lognormal, Poisson
/// >>> pml = occurrence_pml(Poisson(2.0), Lognormal.from_mean_cv(100.0, 2.0), 0.01)
/// >>> round(pml.loss, 4), pml.annual_return_period
/// (1171.4979, 100.0)
#[pyfunction]
pub(crate) fn occurrence_pml(
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
    p: f64,
) -> PyResult<PyOccurrencePml> {
    let n = AnyCount::extract(frequency)?;
    let sev = extract_severity(severity)?;
    let inner = prospicio_aggregate::occurrence_pml(n.as_counting(), &sev, p).map_err(to_py)?;
    Ok(PyOccurrencePml { inner })
}

/// Compound moments to third order, from ``compound_moments``.
#[pyclass(name = "CompoundMoments", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyCompoundMoments {
    pub(crate) inner: prospicio_aggregate::CompoundMoments,
}

fn approximation_kind(kind: &str) -> PyResult<prospicio_aggregate::ApproximationKind> {
    use prospicio_aggregate::ApproximationKind as K;
    Ok(match kind {
        "normal" => K::Normal,
        "gamma" => K::Gamma,
        "lognormal" => K::Lognormal,
        "shifted_gamma" => K::ShiftedGamma,
        "shifted_lognormal" => K::ShiftedLognormal,
        "normal_power" => K::NormalPower,
        "wilson_hilferty" => K::WilsonHilferty,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "kind must be \"normal\", \"gamma\", \"lognormal\", \"shifted_gamma\", \"shifted_lognormal\", \"normal_power\" or \"wilson_hilferty\", got {other:?}"
            )));
        }
    })
}

#[pymethods]
impl PyCompoundMoments {
    /// ``E[S]``.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// ``Var[S]``.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance
    }

    /// The third cumulant, ``E[(S - E[S])^3]``.
    #[getter]
    fn third_cumulant(&self) -> f64 {
        self.inner.third_cumulant
    }

    /// The coefficient of variation.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv()
    }

    /// The skewness.
    #[getter]
    fn skew(&self) -> f64 {
        self.inner.skew()
    }

    /// The moment-matched approximation: ``"normal"``, ``"gamma"``,
    /// ``"lognormal"`` (mean and cv), ``"shifted_gamma"``,
    /// ``"shifted_lognormal"``, ``"normal_power"`` or ``"wilson_hilferty"``
    /// (mean, cv and skewness; the last two defined by their quantiles,
    /// Bahnemann's section 4.4).
    ///
    /// Parameters
    /// ----------
    /// kind : str
    ///
    /// Returns
    /// -------
    /// Approximation
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For an unknown kind, a non-positive mean (gamma, lognormal) or a
    ///     non-positive skewness (the three-moment fits).
    fn approximate(&self, kind: &str) -> PyResult<PyApproximation> {
        let inner = self
            .inner
            .approximate(approximation_kind(kind)?)
            .map_err(to_py)?;
        Ok(PyApproximation { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CompoundMoments(mean={:?}, cv={:?}, skew={:?})",
            self.inner.mean,
            self.inner.cv(),
            self.inner.skew()
        )
    }
}

/// A distribution fitted to compound moments, from
/// ``CompoundMoments.approximate``.
#[pyclass(name = "Approximation", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyApproximation {
    inner: prospicio_aggregate::Approximation,
}

#[pymethods]
impl PyApproximation {
    /// The family: ``"normal"``, ``"gamma"``, ``"lognormal"``,
    /// ``"shifted_gamma"``, ``"shifted_lognormal"``, ``"normal_power"`` or
    /// ``"wilson_hilferty"``.
    #[getter]
    fn kind(&self) -> &'static str {
        use prospicio_aggregate::Approximation as A;
        match self.inner {
            A::Normal { .. } => "normal",
            A::Gamma { .. } => "gamma",
            A::Lognormal { .. } => "lognormal",
            A::ShiftedGamma { .. } => "shifted_gamma",
            A::ShiftedLognormal { .. } => "shifted_lognormal",
            A::NormalPower { .. } => "normal_power",
            A::WilsonHilferty { .. } => "wilson_hilferty",
        }
    }

    /// The parameters by name: ``mean`` and ``sd``; ``shape`` and
    /// ``scale``; ``meanlog`` and ``sdlog``; with ``shift`` for the
    /// shifted families; ``mean``, ``sd`` and ``skew`` for the normal power
    /// and Wilson-Hilferty.
    #[getter]
    fn parameters(&self) -> Vec<(&'static str, f64)> {
        use prospicio_aggregate::Approximation as A;
        match self.inner {
            A::Normal { mean, sd } => vec![("mean", mean), ("sd", sd)],
            A::Gamma { shape, scale } => vec![("shape", shape), ("scale", scale)],
            A::Lognormal { meanlog, sdlog } => vec![("meanlog", meanlog), ("sdlog", sdlog)],
            A::ShiftedGamma {
                shape,
                scale,
                shift,
            } => vec![("shape", shape), ("scale", scale), ("shift", shift)],
            A::ShiftedLognormal {
                meanlog,
                sdlog,
                shift,
            } => vec![("meanlog", meanlog), ("sdlog", sdlog), ("shift", shift)],
            A::NormalPower { mean, sd, skew } | A::WilsonHilferty { mean, sd, skew } => {
                vec![("mean", mean), ("sd", sd), ("skew", skew)]
            }
        }
    }

    /// The mean.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// ``P(S <= x)``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, x: f64) -> PyResult<f64> {
        self.inner.cdf(x).map_err(to_py)
    }

    /// The ``p`` quantile.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Approximation(kind={:?}, parameters={:?})",
            self.kind(),
            self.parameters()
        )
    }
}

/// Compound moments to third order: the mean, variance and third cumulant
/// of ``S = X_1 + ... + X_N``, from the count's cumulants and the
/// severity's first three moments, and the approximations fitted to them
/// (the Python package ``aggregate``'s ``approximate``).
///
/// Parameters
/// ----------
/// frequency : a claim count
/// severity : a severity with a closed-form third moment
///
/// Returns
/// -------
/// CompoundMoments
///
/// Raises
/// ------
/// ValueError
///     If the severity has no closed-form third moment, or an infinite one.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import compound_moments
/// >>> from prospicio.distributions import Gamma, Poisson
/// >>> m = compound_moments(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
/// >>> round(m.mean, 6), round(m.cv, 6), round(m.skew, 6)
/// (500.0, 0.386005, 0.512946)
/// >>> fit = m.approximate("shifted_gamma")
/// >>> [(k, round(v, 4)) for k, v in fit.parameters]
/// [('shape', 15.2025), ('scale', 49.5), ('shift', -252.5253)]
#[pyfunction]
pub(crate) fn compound_moments(
    frequency: &Bound<'_, PyAny>,
    severity: &Bound<'_, PyAny>,
) -> PyResult<PyCompoundMoments> {
    let n = AnyCount::extract(frequency)?;
    let sev = extract_severity(severity)?;
    let inner = prospicio_aggregate::compound_moments(n.as_counting(), &sev).map_err(to_py)?;
    Ok(PyCompoundMoments { inner })
}

/// The probability of eventual ruin at each initial surplus on the grid, by
/// the Pollaczeck–Khinchine formula, from ``eventual_ruin``.
#[pyclass(name = "Ruin", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyRuin {
    inner: prospicio_aggregate::Ruin,
}

#[pymethods]
impl PyRuin {
    /// The margin ``rho``; the loss ratio is ``1 / (1 + rho)``.
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho()
    }

    /// The initial surplus at each grid point.
    #[getter]
    fn u(&self) -> Vec<f64> {
        let h = self.inner.step();
        (0..self.inner.psi().len()).map(|j| j as f64 * h).collect()
    }

    /// ``psi(u)`` at each grid point; the last is the mass beyond the grid.
    #[getter]
    fn psi(&self) -> Vec<f64> {
        self.inner.psi().to_vec()
    }

    /// ``psi(u)`` at any initial surplus (constant between grid points).
    ///
    /// Parameters
    /// ----------
    /// u : float
    ///
    /// Returns
    /// -------
    /// float
    fn psi_at(&self, u: f64) -> f64 {
        self.inner.psi_at(u)
    }

    /// The initial surplus for ruin probability ``p``: the smallest grid
    /// point with ``psi(u) < p``, or linear between grid points.
    ///
    /// Parameters
    /// ----------
    /// p : float
    /// interpolate : bool, default False
    ///
    /// Returns
    /// -------
    /// float or None
    ///     None when the grid does not reach it.
    #[pyo3(signature = (p, interpolate=false))]
    fn capital(&self, p: f64, interpolate: bool) -> Option<f64> {
        if interpolate {
            self.inner.capital_interpolated(p)
        } else {
            self.inner.capital(p)
        }
    }

    /// The Lundberg exponent ``R`` with ``psi(u) <= exp(-R u)``, or None
    /// when the grid's severity has no root below ``700 / max(x)``.
    #[getter]
    fn lundberg_exponent(&self) -> Option<f64> {
        self.inner.lundberg_exponent()
    }

    /// The severity's mean on the grid, after any cap, excess or stop loss.
    #[getter]
    fn severity_mean(&self) -> f64 {
        self.inner.severity_mean()
    }

    /// The equilibrium (integrated tail) severity.
    #[getter]
    fn equilibrium(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.equilibrium().clone(),
        }
    }

    /// The compound FFT's report; read ``aliasing_error``.
    #[getter]
    fn report(&self) -> PyCompoundReport {
        PyCompoundReport {
            inner: self.inner.report().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Ruin(rho={}, psi(0)={:.6}, points={})",
            self.inner.rho(),
            self.inner.psi()[0],
            self.inner.psi().len()
        )
    }
}

/// The probability of eventual ruin of the classical surplus process by
/// the Pollaczeck–Khinchine formula, as ``aggregate``'s
/// ``pollaczeck_khinchine`` and ``eventual_ruin``: Poisson claims with this
/// severity and a premium at ``1 + rho`` times the expected loss. The
/// Poisson rate cancels. The maximal aggregate loss is a compound geometric
/// of the equilibrium severity, by FFT.
///
/// Parameters
/// ----------
/// severity : Grid
/// rho : float, optional
///     The margin to loss, positive. Or give ``loss_ratio``.
/// loss_ratio : float, optional
///     ``1 / (1 + rho)``, in (0, 1).
/// points : int, optional
///     Grid points for ``psi``; the severity's by default.
/// cap : float, optional
///     Use ``X | X <= cap``.
/// excess : float, optional
///     Use ``X | X > excess`` (not shifted).
/// stop_loss : float, optional
///     Use ``X`` with the mass above ``stop_loss`` moved onto it.
///
/// Returns
/// -------
/// Ruin
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import eventual_ruin
/// >>> from prospicio.distributions import Gamma, Grid
/// >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(100.0, 1.0), 1.0, 1 << 14)
/// >>> ruin = eventual_ruin(sev, rho=0.2)
/// >>> # Exponential claims: exp(-u / 600) / 1.2 = 0.157; the step costs 1%.
/// >>> round(ruin.psi_at(1000.0), 3)
/// 0.156
#[pyfunction]
#[pyo3(signature = (severity, rho=None, *, loss_ratio=None, points=None, cap=None, excess=None,
                    stop_loss=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eventual_ruin(
    severity: PyRef<'_, PyGrid>,
    rho: Option<f64>,
    loss_ratio: Option<f64>,
    points: Option<usize>,
    cap: Option<f64>,
    excess: Option<f64>,
    stop_loss: Option<f64>,
) -> PyResult<PyRuin> {
    use prospicio_aggregate::SeverityPart;
    use pyo3::exceptions::PyValueError;
    let rho = match (rho, loss_ratio) {
        (Some(r), None) => r,
        (None, Some(lr)) if lr > 0.0 && lr < 1.0 => 1.0 / lr - 1.0,
        (None, Some(lr)) => {
            return Err(PyValueError::new_err(format!(
                "the loss ratio must be in (0, 1), not {lr}"
            )));
        }
        _ => {
            return Err(PyValueError::new_err(
                "give exactly one of rho and loss_ratio",
            ));
        }
    };
    let part = match (cap, excess, stop_loss) {
        (None, None, None) => SeverityPart::All,
        (Some(c), None, None) => SeverityPart::AtMost(c),
        (None, Some(c), None) => SeverityPart::Above(c),
        (None, None, Some(s)) => SeverityPart::StopLoss(s),
        _ => {
            return Err(PyValueError::new_err(
                "give at most one of cap, excess and stop_loss",
            ));
        }
    };
    let points = points.unwrap_or(severity.inner.len());
    let inner = prospicio_aggregate::pollaczeck_khinchine(&severity.inner, rho, points, part)
        .map_err(to_py)?;
    Ok(PyRuin { inner })
}

/// The probability of eventual ruin at each initial surplus on the grid
/// when the waits between claims follow any law, by Wiener–Hopf, from
/// ``wiener_hopf``.
#[pyclass(name = "RenewalRuin", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyRenewalRuin {
    inner: prospicio_aggregate::RenewalRuin,
}

#[pymethods]
impl PyRenewalRuin {
    /// The margin ``rho``.
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho()
    }

    /// The premium rate ``c = (1 + rho) E[X] / E[W]``.
    #[getter]
    fn premium_rate(&self) -> f64 {
        self.inner.premium_rate()
    }

    /// The initial surplus at each grid point.
    #[getter]
    fn u(&self) -> Vec<f64> {
        let h = self.inner.step();
        (0..self.inner.psi().len()).map(|j| j as f64 * h).collect()
    }

    /// ``psi(u)`` at each grid point; the last shows the mass beyond the
    /// grid.
    #[getter]
    fn psi(&self) -> Vec<f64> {
        self.inner.psi().to_vec()
    }

    /// The probabilities of the claim walk's all-time maximum at each grid
    /// point.
    #[getter]
    fn maximum(&self) -> Vec<f64> {
        self.inner.maximum().to_vec()
    }

    /// ``psi(u)`` at any initial surplus (constant between grid points).
    ///
    /// Parameters
    /// ----------
    /// u : float
    ///
    /// Returns
    /// -------
    /// float
    fn psi_at(&self, u: f64) -> f64 {
        self.inner.psi_at(u)
    }

    /// The initial surplus for ruin probability ``p``: the smallest grid
    /// point with ``psi(u) < p``, or linear between grid points.
    ///
    /// Parameters
    /// ----------
    /// p : float
    /// interpolate : bool, default False
    ///
    /// Returns
    /// -------
    /// float or None
    ///     None when the grid does not reach it.
    #[pyo3(signature = (p, interpolate=false))]
    fn capital(&self, p: f64, interpolate: bool) -> Option<f64> {
        if interpolate {
            self.inner.capital_interpolated(p)
        } else {
            self.inner.capital(p)
        }
    }

    /// The severity's mean on the grid.
    #[getter]
    fn severity_mean(&self) -> f64 {
        self.inner.severity_mean()
    }

    /// The grid's mean step of the claim walk ``X - cW``, negative.
    #[getter]
    fn mean_step(&self) -> f64 {
        self.inner.mean_step()
    }

    fn __repr__(&self) -> String {
        format!(
            "RenewalRuin(rho={}, psi(0)={:.6}, points={})",
            self.inner.rho(),
            self.inner.psi()[0],
            self.inner.psi().len()
        )
    }
}

/// The probability of eventual ruin when the waits between claims follow
/// any law (a renewal, or Sparre Andersen, process), as ``aggregate``'s
/// ``wiener_hopf``: claims of this severity, waits drawn from ``wait``,
/// and the premium at the rate ``(1 + rho) E[X] / E[W]``. Ruin comes at a
/// claim, so ``psi(u)`` is the chance that the walk with steps ``X - cW``
/// ever rises above ``u``, found by the cepstral Wiener–Hopf
/// factorization. Exponential waits give ``eventual_ruin``.
///
/// Parameters
/// ----------
/// severity : Grid
/// wait : distribution
///     The law of the wait between claims, with a finite mean.
/// rho : float, optional
///     The margin to loss, positive. Or give ``loss_ratio``.
/// loss_ratio : float, optional
///     ``1 / (1 + rho)``, in (0, 1).
/// points : int, optional
///     Grid points for ``psi``, at least the severity's (the default).
///     Raise it when ``psi`` at the last point is not small.
///
/// Returns
/// -------
/// RenewalRuin
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import eventual_ruin, wiener_hopf
/// >>> from prospicio.distributions import Gamma, Grid
/// >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(10.0, 0.3), 1 / 32, 1 << 14)
/// >>> # Gamma waits (shape 2) are more regular than Poisson arrivals.
/// >>> wh = wiener_hopf(sev, Gamma(2.0, 1.0), rho=0.2)
/// >>> pk = eventual_ruin(sev, rho=0.2)
/// >>> wh.psi_at(50.0) < pk.psi_at(50.0)
/// True
#[pyfunction]
#[pyo3(signature = (severity, wait, rho=None, *, loss_ratio=None, points=None))]
pub(crate) fn wiener_hopf(
    severity: PyRef<'_, PyGrid>,
    wait: &Bound<'_, PyAny>,
    rho: Option<f64>,
    loss_ratio: Option<f64>,
    points: Option<usize>,
) -> PyResult<PyRenewalRuin> {
    use pyo3::exceptions::PyValueError;
    let rho = match (rho, loss_ratio) {
        (Some(r), None) => r,
        (None, Some(lr)) if lr > 0.0 && lr < 1.0 => 1.0 / lr - 1.0,
        (None, Some(lr)) => {
            return Err(PyValueError::new_err(format!(
                "the loss ratio must be in (0, 1), not {lr}"
            )));
        }
        _ => {
            return Err(PyValueError::new_err(
                "give exactly one of rho and loss_ratio",
            ));
        }
    };
    let wait = crate::distributions::extract_dist(wait)?;
    let points = points.unwrap_or(severity.inner.len());
    let inner =
        prospicio_aggregate::wiener_hopf(&severity.inner, &wait, rho, points).map_err(to_py)?;
    Ok(PyRenewalRuin { inner })
}

/// A severity reshaped to layer loss picks, from ``layer_loss_picks``.
#[pyclass(name = "Picks", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyPicks {
    inner: prospicio_aggregate::Picks,
}

#[pymethods]
impl PyPicks {
    /// The adjusted severity, on the same grid.
    #[getter]
    fn severity(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.severity.clone(),
        }
    }

    /// Whether the bottom layer took the straight-line shape.
    #[getter]
    fn straight_bottom(&self) -> bool {
        self.inner.straight_bottom
    }

    /// Each layer, bottom up, as columns: ``bottom``, ``top``, ``before``
    /// and ``target`` (the expected loss per claim before and after),
    /// ``weight`` (on the survival function) and ``above`` (the adjusted
    /// chance of a loss above the layer).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let l = &self.inner.layers;
        let col = |f: fn(&prospicio_aggregate::PickedLayer) -> f64| -> Vec<f64> {
            l.iter().map(f).collect()
        };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("bottom", col(|x| x.bottom))?;
        d.set_item("top", col(|x| x.top))?;
        d.set_item("before", col(|x| x.before))?;
        d.set_item("target", col(|x| x.target))?;
        d.set_item("weight", col(|x| x.weight))?;
        d.set_item("above", col(|x| x.above))?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Picks(layers={}, straight_bottom={})",
            self.inner.layers.len(),
            if self.inner.straight_bottom {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Reshapes a severity so that its expected loss to each layer hits a
/// pick, as ``aggregate``'s ``Aggregate.picks``: in each layer the survival
/// function is rescaled about its value at the layer's top, keeping its
/// shape there; above the top attachment the severity is kept.
///
/// Parameters
/// ----------
/// severity : Grid
/// attachments : list of float
///     Increasing grid points; the layers are 0 to the first, then between
///     consecutive attachments.
/// picks : list of float
///     The loss to each layer for ``claims`` expected ground-up claims.
/// claims : float, default 1.0
///
/// Returns
/// -------
/// Picks
///
/// Raises
/// ------
/// ValueError
///     If an attachment is not a grid point or the picks are infeasible (a
///     pick below what the layers above force through its layer).
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import layer_loss_picks
/// >>> from prospicio.distributions import Gamma, Grid
/// >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(100.0, 1.5), 1.0, 1 << 14)
/// >>> picked = layer_loss_picks(sev, [100, 500, 1000], [480, 420, 60], claims=10)
/// >>> [round(10 * t, 6) for t in picked.to_dict()["target"]]
/// [480.0, 420.0, 60.0]
#[pyfunction]
#[pyo3(signature = (severity, attachments, picks, claims=1.0))]
pub(crate) fn layer_loss_picks(
    severity: PyRef<'_, PyGrid>,
    attachments: Vec<f64>,
    picks: Vec<f64>,
    claims: f64,
) -> PyResult<PyPicks> {
    let inner =
        prospicio_aggregate::layer_loss_picks(&severity.inner, &attachments, &picks, claims)
            .map_err(to_py)?;
    Ok(PyPicks { inner })
}

/// A catastrophe model's event loss table with secondary uncertainty.
///
/// Each row is an event with an annual rate, a mean loss, independent and
/// correlated standard deviations, and an exposure, the most it can lose.
/// Events occur as independent Poisson processes; an event's loss, given
/// that it occurs, is a beta distribution on ``[0, exposure]`` with the
/// row's mean and standard deviation ``sd_independent + sd_correlated``
/// (summed, as catastrophe models report them for one portfolio), or the
/// mean exactly when that is 0. A row whose standard deviation no beta on
/// its exposure can have is an error naming the row.
///
/// Parameters
/// ----------
/// rate, mean, sd_independent, sd_correlated, exposure : list of float
///     One value per event, as the table's columns.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import EventLossTable
/// >>> elt = EventLossTable([0.01, 0.05], [800.0, 200.0], [200.0, 0.0], [100.0, 0.0], [2000.0, 500.0])
/// >>> round(elt.rate, 12), round(elt.expected_annual_loss, 12)
/// (0.06, 18.0)
/// >>> events, ids = elt.simulate(1000, seed=1)
/// >>> events.n_sims
/// 1000
#[pyclass(name = "EventLossTable", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyEventLossTable {
    inner: EventLossTable,
}

#[pymethods]
impl PyEventLossTable {
    #[new]
    fn new(
        rate: Vec<f64>,
        mean: Vec<f64>,
        sd_independent: Vec<f64>,
        sd_correlated: Vec<f64>,
        exposure: Vec<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: EventLossTable::new(&rate, &mean, &sd_independent, &sd_correlated, &exposure)
                .map_err(to_py)?,
        })
    }

    /// The mean number of events a year, the rates' sum.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// The expected annual loss, the sum of rate times mean.
    #[getter]
    fn expected_annual_loss(&self) -> f64 {
        self.inner.expected_annual_loss()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The annual frequency of events losing more than ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn exceedance_frequency(&self, x: f64) -> f64 {
        self.inner.exceedance_frequency(x)
    }

    /// The occurrence exceedance probability at ``x``,
    /// ``1 - exp(-exceedance_frequency(x))``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn oep(&self, x: f64) -> f64 {
        self.inner.oep(x)
    }

    /// The occurrence PML at annual probability ``p``: the loss whose
    /// occurrence exceedance probability is ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn occurrence_pml(&self, p: f64) -> PyResult<f64> {
        self.inner.occurrence_pml(p).map_err(to_py)
    }

    /// Simulated years of events, with each loss's event (row index).
    ///
    /// Year ``i`` draws from its own stream of the seed: a Poisson count,
    /// then for each event the row by its rate and its loss, so any year
    /// replays alone.
    ///
    /// Parameters
    /// ----------
    /// n_years : int
    /// seed : int, default 0
    ///
    /// Returns
    /// -------
    /// tuple of (EventSet, list of list of int)
    #[pyo3(signature = (n_years, seed = 0))]
    fn simulate(&self, n_years: usize, seed: u64) -> PyResult<(PyEventSet, Vec<Vec<usize>>)> {
        let (inner, ids) = self.inner.simulate(n_years, seed).map_err(to_py)?;
        Ok((PyEventSet { inner }, ids))
    }

    fn __repr__(&self) -> String {
        format!(
            "EventLossTable(events={}, rate={})",
            self.inner.len(),
            self.inner.rate()
        )
    }
}

/// A joint distribution of two losses on a grid, ``P(A = i * step_a, B = j
/// * step_b)``: rows are line A and columns line B. The mass past the last
/// row or column is carried on it.
///
/// Parameters
/// ----------
/// steps : tuple of (float, float)
/// probs : list of list of float
///     One row per point of A, one column per point of B, summing to 1.
#[pyclass(name = "JointGrid", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyJointGrid {
    inner: prospicio_aggregate::JointGrid,
}

#[pymethods]
impl PyJointGrid {
    #[new]
    fn new(steps: (f64, f64), probs: Vec<Vec<f64>>) -> PyResult<Self> {
        let cols = probs.first().map_or(0, Vec::len);
        if probs.iter().any(|r| r.len() != cols) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "every row of probs must have the same length",
            ));
        }
        let shape = [probs.len(), cols];
        let flat = probs.into_iter().flatten().collect();
        let inner =
            prospicio_aggregate::JointGrid::new([steps.0, steps.1], shape, flat).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The joint law of two independent losses.
    ///
    /// Parameters
    /// ----------
    /// a, b : Grid
    ///
    /// Returns
    /// -------
    /// JointGrid
    #[staticmethod]
    fn independent(a: PyRef<'_, PyGrid>, b: PyRef<'_, PyGrid>) -> Self {
        Self {
            inner: prospicio_aggregate::JointGrid::independent(&a.inner, &b.inner),
        }
    }

    /// The joint per-event severity of two losses coupled by a copula.
    ///
    /// Each line is hit with its ``trigger`` probability and then loses
    /// from its grid (``a`` for line A, the rows; ``b`` for B), so mass at 0
    /// is an event that misses the line; the two per-event marginals are
    /// joined by the copula's rectangle probabilities over their cdfs, as
    /// ``aggregate``'s copula ``bivariate``. The marginals come back
    /// exactly. Give the result to ``fft_bivariate``.
    ///
    /// Parameters
    /// ----------
    /// a, b : Grid
    ///     The loss to each line given that the event hits it.
    /// copula : GaussianCopula or ArchimedeanCopula
    ///     Two-dimensional, with a distribution function.
    /// trigger_a, trigger_b : float, default 1.0
    ///     The probability that an event hits each line.
    ///
    /// Returns
    /// -------
    /// JointGrid
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import JointGrid
    /// >>> from prospicio.distributions import Grid
    /// >>> from prospicio.risk import ArchimedeanCopula
    /// >>> a = Grid(1.0, [0.25, 0.5, 0.25])
    /// >>> b = Grid(1.0, [0.5, 0.5])
    /// >>> s = JointGrid.from_copula(a, b, ArchimedeanCopula("gumbel", 2.0, 2), trigger_a=0.5)
    /// >>> s.correlation() > 0
    /// True
    #[staticmethod]
    #[pyo3(signature = (a, b, copula, *, trigger_a = 1.0, trigger_b = 1.0))]
    fn from_copula(
        a: PyRef<'_, PyGrid>,
        b: PyRef<'_, PyGrid>,
        copula: &Bound<'_, PyAny>,
        trigger_a: f64,
        trigger_b: f64,
    ) -> PyResult<Self> {
        use prospicio_aggregate::bivariate::{JointGrid, triggered};
        let copula = crate::risk::AnyCopula::extract(copula)?;
        let a = triggered(&a.inner, trigger_a).map_err(to_py)?;
        let b = triggered(&b.inner, trigger_b).map_err(to_py)?;
        Ok(Self {
            inner: JointGrid::from_copula(&a, &b, copula.as_copula()).map_err(to_py)?,
        })
    }

    /// The step on each axis.
    #[getter]
    fn steps(&self) -> (f64, f64) {
        let [a, b] = self.inner.steps();
        (a, b)
    }

    /// The number of rows (line A) and columns (line B).
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let [a, b] = self.inner.shape();
        (a, b)
    }

    /// The probabilities, one list per row.
    #[getter]
    fn probs(&self) -> Vec<Vec<f64>> {
        let cols = self.inner.shape()[1];
        self.inner
            .probs()
            .chunks(cols)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// ``P(A = i * step_a, B = j * step_b)``.
    ///
    /// Parameters
    /// ----------
    /// i, j : int
    ///
    /// Returns
    /// -------
    /// float
    fn prob(&self, i: usize, j: usize) -> PyResult<f64> {
        let [rows, cols] = self.inner.shape();
        if i >= rows || j >= cols {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "cell ({i}, {j}) is outside a {rows} x {cols} grid"
            )));
        }
        Ok(self.inner.prob(i, j))
    }

    /// The marginal distribution of line A (``axis=0``) or B (``axis=1``).
    ///
    /// Parameters
    /// ----------
    /// axis : int
    ///
    /// Returns
    /// -------
    /// Grid
    fn marginal(&self, axis: usize) -> PyResult<PyGrid> {
        Ok(PyGrid {
            inner: self.inner.marginal(axis).map_err(to_py)?,
        })
    }

    /// The distribution of the total ``A + B``: exact when the steps are
    /// equal, otherwise on the larger step with each cell's mass split to
    /// keep the mean.
    ///
    /// Returns
    /// -------
    /// Grid
    fn total(&self) -> PyResult<PyGrid> {
        Ok(PyGrid {
            inner: self.inner.total().map_err(to_py)?,
        })
    }

    /// The raw mixed moment ``E[A^p B^q]``.
    ///
    /// Parameters
    /// ----------
    /// p, q : int
    ///
    /// Returns
    /// -------
    /// float
    fn moment(&self, p: u32, q: u32) -> f64 {
        self.inner.moment(p, q)
    }

    /// The mean of line A (``axis=0``) or B (``axis=1``).
    fn mean(&self, axis: usize) -> PyResult<f64> {
        check_axis(axis)?;
        Ok(self.inner.mean(axis))
    }

    /// The variance of line A (``axis=0``) or B (``axis=1``).
    fn variance(&self, axis: usize) -> PyResult<f64> {
        check_axis(axis)?;
        Ok(self.inner.variance(axis))
    }

    /// ``Cov(A, B)``.
    fn covariance(&self) -> f64 {
        self.inner.covariance()
    }

    /// Pearson's correlation of ``A`` and ``B``; NaN if either is constant.
    fn correlation(&self) -> f64 {
        self.inner.correlation()
    }

    fn __repr__(&self) -> String {
        let [a, b] = self.inner.steps();
        let [r, c] = self.inner.shape();
        format!("JointGrid(steps=({a}, {b}), shape=({r}, {c}))")
    }
}

fn check_axis(axis: usize) -> PyResult<()> {
    if axis > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "axis must be 0 or 1, not {axis}"
        )));
    }
    Ok(())
}

/// What ``fft_bivariate`` produced and the error it introduced.
#[pyclass(name = "BivariateReport", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyBivariateReport {
    pub(crate) inner: prospicio_aggregate::BivariateReport,
}

#[pymethods]
impl PyBivariateReport {
    /// Points on each axis.
    #[getter]
    fn points(&self) -> (usize, usize) {
        (self.inner.points[0], self.inner.points[1])
    }

    /// The mass past the last row and past the last column, carried on them.
    #[getter]
    fn tail_mass(&self) -> (f64, f64) {
        (self.inner.tail_mass[0], self.inner.tail_mass[1])
    }

    /// On each axis, the mass past the end of the FFT buffer, which wraps
    /// around; their sum bounds the probability misplaced. Request more
    /// points when it is not negligible.
    #[getter]
    fn aliasing_mass(&self) -> (f64, f64) {
        (self.inner.aliasing_mass[0], self.inner.aliasing_mass[1])
    }

    /// ``E[N] E[X]`` and ``E[N] E[Y]``: the means before truncation.
    #[getter]
    fn expected_means(&self) -> (f64, f64) {
        (self.inner.expected_means[0], self.inner.expected_means[1])
    }

    /// The means of the result.
    #[getter]
    fn grid_means(&self) -> (f64, f64) {
        (self.inner.grid_means[0], self.inner.grid_means[1])
    }

    fn __repr__(&self) -> String {
        format!(
            "BivariateReport(points={:?}, tail_mass={:?}, aliasing_mass={:?})",
            self.inner.points, self.inner.tail_mass, self.inner.aliasing_mass
        )
    }
}

/// The joint annual loss to two lines hit by the same events, by a 2-D FFT:
/// ``(A, B) = sum over N events of (X_k, Y_k)``, with ``severity`` the joint
/// law of one event's pair of losses (mass at 0 on an axis is an event that
/// misses that line).
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial, Binomial or Count
///     The shared count of events.
/// severity : JointGrid
/// points : tuple of (int, int)
///     Points on each axis of the result.
///
/// Returns
/// -------
/// tuple of (JointGrid, BivariateReport)
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import JointGrid, fft_bivariate
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> a = Grid(1.0, [0.0, 0.6, 0.4])
/// >>> b = Grid(1.0, [0.5, 0.5])
/// >>> joint, report = fft_bivariate(Poisson(2.0), JointGrid.independent(a, b), (64, 64))
/// >>> round(joint.covariance(), 9)  # Var(N) E[X] E[Y]
/// 1.4
#[pyfunction]
pub(crate) fn fft_bivariate(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyJointGrid>,
    points: (usize, usize),
) -> PyResult<(PyJointGrid, PyBivariateReport)> {
    let n = AnyCount::extract(frequency)?;
    let sev = severity.inner.clone();
    let (inner, report) = py
        .detach(|| prospicio_aggregate::fft_bivariate(n.as_counting(), &sev, [points.0, points.1]))
        .map_err(to_py)?;
    Ok((PyJointGrid { inner }, PyBivariateReport { inner: report }))
}

/// Two lines hit by the same events, each line independently within an
/// event: ``aggregate``'s ``clash``.
///
/// Given the expected events a year hitting line A only, line B only and
/// both (the clash count), independence within an event fixes those hitting
/// neither, ``only_a * only_b / both``. Each event hits A with probability
/// ``trigger_a`` and B with ``trigger_b``; the dependence between the lines
/// comes only from the shared count.
///
/// Parameters
/// ----------
/// only_a, only_b : float
///     Expected events a year hitting one line only, non-negative.
/// both : float
///     Expected events a year hitting both lines, positive.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Clash
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> clash = Clash(2.0, 3.0, 1.0)
/// >>> clash.events, clash.trigger_a
/// (12.0, 0.25)
/// >>> a = Grid(1.0, [0.0, 0.5, 0.5])
/// >>> b = Grid(1.0, [0.0, 1.0])
/// >>> joint, report = clash.aggregate(Poisson(clash.events), a, b, (128, 128))
/// >>> round(joint.covariance(), 9)  # Poisson: both * E[X] * E[Y]
/// 1.5
#[pyclass(name = "Clash", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyClash {
    inner: prospicio_aggregate::Clash,
}

#[pymethods]
impl PyClash {
    #[new]
    fn new(only_a: f64, only_b: f64, both: f64) -> PyResult<Self> {
        let inner = prospicio_aggregate::Clash::new(only_a, only_b, both).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Expected events a year hitting line A only.
    #[getter]
    fn only_a(&self) -> f64 {
        self.inner.only_a()
    }

    /// Expected events a year hitting line B only.
    #[getter]
    fn only_b(&self) -> f64 {
        self.inner.only_b()
    }

    /// Expected events a year hitting both lines: the clash count.
    #[getter]
    fn both(&self) -> f64 {
        self.inner.both()
    }

    /// Expected events a year hitting neither line.
    #[getter]
    fn neither(&self) -> f64 {
        self.inner.neither()
    }

    /// Expected events a year: the mean of the shared count.
    #[getter]
    fn events(&self) -> f64 {
        self.inner.events()
    }

    /// The probability that an event hits line A.
    #[getter]
    fn trigger_a(&self) -> f64 {
        self.inner.trigger_a()
    }

    /// The probability that an event hits line B.
    #[getter]
    fn trigger_b(&self) -> f64 {
        self.inner.trigger_b()
    }

    /// The joint per-event severity: ``a`` with probability ``trigger_a``
    /// on line A and ``b`` with ``trigger_b`` on line B, independently.
    ///
    /// Parameters
    /// ----------
    /// a, b : Grid
    ///     The loss to each line given that the event hits it.
    ///
    /// Returns
    /// -------
    /// JointGrid
    fn severity(&self, a: PyRef<'_, PyGrid>, b: PyRef<'_, PyGrid>) -> PyResult<PyJointGrid> {
        Ok(PyJointGrid {
            inner: self.inner.severity(&a.inner, &b.inner).map_err(to_py)?,
        })
    }

    /// The joint annual loss to the two lines by ``fft_bivariate``.
    ///
    /// Parameters
    /// ----------
    /// frequency : Poisson, NegativeBinomial, Binomial or Count
    ///     The shared count; its mean must be ``events``.
    /// a, b : Grid
    ///     The loss to each line given that the event hits it.
    /// points : tuple of (int, int)
    ///
    /// Returns
    /// -------
    /// tuple of (JointGrid, BivariateReport)
    fn aggregate(
        &self,
        py: Python<'_>,
        frequency: &Bound<'_, PyAny>,
        a: PyRef<'_, PyGrid>,
        b: PyRef<'_, PyGrid>,
        points: (usize, usize),
    ) -> PyResult<(PyJointGrid, PyBivariateReport)> {
        let n = AnyCount::extract(frequency)?;
        let (a, b, clash) = (a.inner.clone(), b.inner.clone(), self.inner);
        let (inner, report) = py
            .detach(|| clash.aggregate(n.as_counting(), &a, &b, [points.0, points.1]))
            .map_err(to_py)?;
        Ok((PyJointGrid { inner }, PyBivariateReport { inner: report }))
    }

    fn __repr__(&self) -> String {
        format!(
            "Clash(only_a={}, only_b={}, both={})",
            self.inner.only_a(),
            self.inner.only_b(),
            self.inner.both()
        )
    }
}

/// A hash of a year loss table's file bytes, ``"blake3:<hex>"``, which
/// ``EventSet.from_ylt`` records as its ``source_hash``.
///
/// Parameters
/// ----------
/// data : bytes
///
/// Returns
/// -------
/// str
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import ylt_hash
/// >>> ylt_hash(b"year,value").startswith("blake3:")
/// True
#[pyfunction]
pub fn ylt_hash(data: &[u8]) -> String {
    prospicio_aggregate::ylt::hash_bytes(data)
}

/// Labels from a dict of dimension to values, in the dict's order.
fn labels_from_dict(labels: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Vec<String>)>> {
    labels
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<Vec<String>>()?)))
        .collect()
}

fn correlation_kind(kind: &str) -> PyResult<prospicio_prob::portfolio::CorrelationKind> {
    use prospicio_prob::portfolio::CorrelationKind;
    match kind {
        "pearson" => Ok(CorrelationKind::Pearson),
        "spearman" => Ok(CorrelationKind::Spearman),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "kind must be \"pearson\" or \"spearman\", not {kind:?}"
        ))),
    }
}

fn year_draws(draws: &str) -> PyResult<prospicio_aggregate::term::YearDraws> {
    use prospicio_aggregate::term::YearDraws;
    match draws {
        "same" => Ok(YearDraws::Same),
        "blocks" => Ok(YearDraws::Blocks),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "draws must be \"same\" or \"blocks\", not {draws:?}"
        ))),
    }
}

/// The correlation between every pair of ``n_years`` years whose totals
/// have the same spread for which the mean of a term's yearly totals has
/// ``ratio`` times one year's standard deviation:
/// ``(n_years * ratio**2 - 1) / (n_years - 1)``. Pearson's correlation of
/// the totals: a start; ``EventSet.calibrate_year_correlation`` hits the
/// ratio on the simulations themselves.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import year_correlation_for_ratio
/// >>> year_correlation_for_ratio(3, 1.0)
/// 1.0
#[pyfunction]
pub(crate) fn year_correlation_for_ratio(n_years: usize, ratio: f64) -> PyResult<f64> {
    prospicio_aggregate::term::year_correlation_for_ratio(n_years, ratio).map_err(to_py)
}

/// A cover basis from its name: ``"losses_occurring"`` (or ``"lod"``) or
/// ``"risks_attaching"`` (or ``"rad"``).
pub(crate) fn cover_basis(name: &str) -> PyResult<prospicio_aggregate::timeline::CoverBasis> {
    use prospicio_aggregate::timeline::CoverBasis;
    match name {
        "losses_occurring" | "lod" => Ok(CoverBasis::LossesOccurring),
        "risks_attaching" | "rad" => Ok(CoverBasis::RisksAttaching),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown cover basis {other:?}: expected \"losses_occurring\" or \"risks_attaching\""
        ))),
    }
}

/// The name of a cover basis.
pub(crate) fn cover_basis_name(basis: prospicio_aggregate::timeline::CoverBasis) -> &'static str {
    use prospicio_aggregate::timeline::CoverBasis;
    match basis {
        CoverBasis::LossesOccurring => "losses_occurring",
        CoverBasis::RisksAttaching => "risks_attaching",
    }
}

/// Each underwriting year's share of a full book's exposure in each
/// accident month from the start of a term.
///
/// One row a year, one share a month; a month's shares add to at most 1,
/// the rest belonging to underwriting years outside the term.
///
/// Parameters
/// ----------
/// shares : list of list of float
///     One row per underwriting year, all of the same length.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Exposure
/// >>> e = Exposure.uniform(2, 12)
/// >>> e.n_years, e.months
/// (2, 36)
/// >>> round(e.shares[0][12] + e.shares[1][12], 12)
/// 1.0
#[pyclass(name = "Exposure", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyExposure {
    pub(crate) inner: prospicio_aggregate::timeline::Exposure,
}

#[pymethods]
impl PyExposure {
    #[new]
    fn new(shares: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = prospicio_aggregate::timeline::Exposure::new(shares).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// ``n_years`` underwriting years, each writing its policies evenly
    /// over its twelve months, each policy in force for ``policy_months``
    /// from the middle of the month it is written in.
    ///
    /// Year ``u``'s share of month ``m`` is its policies' time in force in
    /// ``m`` over ``policy_months``: the shares of a month in mid-term add
    /// to 1, and the exposure runs ``12 * n_years + policy_months`` months.
    ///
    /// Parameters
    /// ----------
    /// n_years : int
    /// policy_months : int
    ///
    /// Returns
    /// -------
    /// Exposure
    #[staticmethod]
    fn uniform(n_years: usize, policy_months: usize) -> PyResult<Self> {
        let inner = prospicio_aggregate::timeline::Exposure::uniform(n_years, policy_months)
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The number of underwriting years.
    #[getter]
    fn n_years(&self) -> usize {
        self.inner.n_years()
    }

    /// The number of accident months.
    #[getter]
    fn months(&self) -> usize {
        self.inner.months()
    }

    /// The shares, one row a year.
    #[getter]
    fn shares(&self) -> Vec<Vec<f64>> {
        self.inner.shares().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "Exposure({} underwriting years over {} months)",
            self.inner.n_years(),
            self.inner.months()
        )
    }
}

/// Events over a whole term of accident months, shared by its
/// underwriting years.
///
/// Simulation ``j`` is one path; each event is stored once, with its time
/// as a fraction of the term, and is in accident month ``floor(x * months)``
/// (the last month for ``x = 1``). A cover basis splits the timeline into
/// contract years: ``Tower.apply_timeline``, or ``years`` for the years
/// themselves.
///
/// An aggregate part of the events (``EventSet.with_aggregate_part``) has
/// no time: it is spread over its accident period's months by
/// ``aggregate_pattern`` and each month's amount falls to the contract
/// years as an event in that month would.
///
/// Parameters
/// ----------
/// events : EventSet
///     With times (``with_uniform_times``, say) as fractions of the term.
///     An aggregate part is the whole term's.
/// months : int
///     The number of accident months.
/// aggregate_pattern : list of float, optional
///     Twelve non-negative weights, one for each month of an accident
///     year, that spread the aggregate part over the months (they need
///     not add to 1). Uniform by default.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import EventSet, Exposure, Timeline
/// >>> events = EventSet.from_years([[100.0]], times=[[13.5 / 24]])
/// >>> timeline = Timeline(events, 24)
/// >>> shares = [[0.0] * 24, [0.0] * 24]
/// >>> shares[0][13], shares[1][13] = 0.25, 0.75
/// >>> rad = timeline.years("risks_attaching", Exposure(shares))
/// >>> rad[0].events(0), rad[1].events(0)
/// ([25.0], [75.0])
#[pyclass(name = "Timeline", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyTimeline {
    pub(crate) inner: prospicio_aggregate::timeline::Timeline,
}

impl PyTimeline {
    fn patterned(
        inner: prospicio_aggregate::timeline::Timeline,
        pattern: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = match pattern {
            Some(p) => inner.with_aggregate_pattern(&p).map_err(to_py)?,
            None => inner,
        };
        Ok(Self { inner })
    }
}

#[pymethods]
impl PyTimeline {
    #[new]
    #[pyo3(signature = (events, months, aggregate_pattern = None))]
    fn new(
        events: PyRef<'_, PyEventSet>,
        months: usize,
        aggregate_pattern: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = prospicio_aggregate::timeline::Timeline::new(events.inner.clone(), months)
            .map_err(to_py)?;
        Self::patterned(inner, aggregate_pattern)
    }

    /// A timeline stitched from one event set per accident year.
    ///
    /// Each year needs times within its year and the same number of
    /// simulations; year ``t``'s event at time ``tau`` is at
    /// ``(t + tau) / T`` of the term. The years can come from
    /// ``EventSet.term_years`` or ``EventSet.correlated_years``. Each
    /// year's aggregate part stays its accident year's, spread over its
    /// twelve months by ``aggregate_pattern``.
    ///
    /// Parameters
    /// ----------
    /// years : list of EventSet
    /// aggregate_pattern : list of float, optional
    ///     Twelve non-negative monthly weights; uniform by default.
    ///
    /// Returns
    /// -------
    /// Timeline
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet, Exposure, Timeline
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> part = PredictiveDistribution([], [()], [[120.0]])
    /// >>> year = EventSet.from_years([[]], times=[[]]).with_aggregate_part(part)
    /// >>> halves = [[1.0] * 6 + [0.0] * 6, [0.0] * 6 + [1.0] * 6]
    /// >>> even = Timeline.from_years([year])
    /// >>> [y.aggregate_part(0) for y in even.years("risks_attaching", Exposure(halves))]
    /// [[60.0], [60.0]]
    /// >>> december = Timeline.from_years([year], aggregate_pattern=[0.0] * 11 + [1.0])
    /// >>> [y.aggregate_part(0) for y in december.years("risks_attaching", Exposure(halves))]
    /// [[0.0], [120.0]]
    #[staticmethod]
    #[pyo3(signature = (years, aggregate_pattern = None))]
    fn from_years(
        years: Vec<PyRef<'_, PyEventSet>>,
        aggregate_pattern: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let years: Vec<EventSet> = years.iter().map(|e| e.inner.clone()).collect();
        let inner = prospicio_aggregate::timeline::Timeline::from_years(&years).map_err(to_py)?;
        Self::patterned(inner, aggregate_pattern)
    }

    /// The weight of each month of an accident year that spreads the
    /// aggregate part, as set (uniform by default).
    #[getter]
    fn aggregate_pattern(&self) -> Vec<f64> {
        self.inner.aggregate_pattern().to_vec()
    }

    /// Whether the timeline carries an aggregate part.
    #[getter]
    fn has_aggregate_part(&self) -> bool {
        self.inner.has_aggregate_part()
    }

    /// The events, with times as fractions of the term, without the
    /// aggregate part.
    #[getter]
    fn events(&self) -> PyEventSet {
        PyEventSet {
            inner: self.inner.events().clone(),
        }
    }

    /// The number of accident months.
    #[getter]
    fn months(&self) -> usize {
        self.inner.months()
    }

    /// The number of simulated paths.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The contract years under ``basis``, for ``Tower.apply_term``.
    ///
    /// Under ``"losses_occurring"``, contract year ``k`` takes every event
    /// in accident months ``12k`` to ``12k + 11`` whole; ``exposure`` gives
    /// only the number of years. Under ``"risks_attaching"``, underwriting
    /// year ``u`` takes its share of each event in each month (sums insured
    /// scaled alike), at its time over the months the year has a share.
    ///
    /// Parameters
    /// ----------
    /// basis : {"losses_occurring", "risks_attaching"}
    /// exposure : Exposure
    ///
    /// Returns
    /// -------
    /// list of EventSet
    fn years(
        &self,
        py: Python<'_>,
        basis: &str,
        exposure: PyRef<'_, PyExposure>,
    ) -> PyResult<Vec<PyEventSet>> {
        let basis = cover_basis(basis)?;
        let (timeline, exposure) = (&self.inner, &exposure.inner);
        Ok(py
            .detach(|| timeline.years(basis, exposure))
            .map_err(to_py)?
            .into_iter()
            .map(|inner| PyEventSet { inner })
            .collect())
    }

    /// ``years`` with each year's sources: for each of the year's losses,
    /// in order, its event's index among all the timeline's losses, so the
    /// same event can be found in every year it falls to.
    ///
    /// Returns
    /// -------
    /// tuple of (list of EventSet, list of list of int)
    fn years_with_sources(
        &self,
        py: Python<'_>,
        basis: &str,
        exposure: PyRef<'_, PyExposure>,
    ) -> PyResult<(Vec<PyEventSet>, Vec<Vec<usize>>)> {
        let basis = cover_basis(basis)?;
        let (timeline, exposure) = (&self.inner, &exposure.inner);
        let (years, sources) = py
            .detach(|| timeline.years_with_sources(basis, exposure))
            .map_err(to_py)?;
        Ok((
            years
                .into_iter()
                .map(|inner| PyEventSet { inner })
                .collect(),
            sources,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Timeline({} paths over {} months)",
            self.inner.n_sims(),
            self.inner.months()
        )
    }
}
