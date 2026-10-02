//! `actuarialrs.aggregate`: wrappers over `act_aggregate` (compound
//! distributions, simulated events, reinsurance).

use act_aggregate::{CompoundMethod, CompoundReport, EventSet, Layer, Tower};
use act_prob::{Counting, Grid};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::distributions::{
    AnySeverity, PyGrid, PyNegativeBinomial, PyPoisson, PyPredictiveDistribution,
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

    fn as_counting(&self) -> &(dyn Counting + Sync) {
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
#[pyclass(name = "CompoundReport", module = "actuarialrs.aggregate", frozen)]
pub(crate) struct PyCompoundReport {
    inner: CompoundReport,
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
/// >>> from actuarialrs.aggregate import panjer
/// >>> from actuarialrs.distributions import Grid, Poisson
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
#[pyclass(name = "EventSet", module = "actuarialrs.aggregate", frozen)]
pub(crate) struct PyEventSet {
    inner: EventSet,
}

impl From<EventSet> for PyEventSet {
    fn from(inner: EventSet) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEventSet {
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
/// >>> from actuarialrs.aggregate import simulate_events
/// >>> from actuarialrs.distributions import Lognormal, Poisson
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
    let sev = AnySeverity::extract(severity)?;
    let inner = py
        .detach(|| act_aggregate::simulate_events(n.as_counting(), &sev, n_sims, seed))
        .map_err(to_py)?;
    Ok(PyEventSet { inner })
}

/// A per-occurrence excess-of-loss layer: ``limit`` xs ``attachment`` on each
/// loss, then annual terms.
///
/// For one year, ``ceded = share * min(max(sum of per-loss recoveries -
/// aggregate_deductible, 0), aggregate_limit)``.
///
/// Parameters
/// ----------
/// name : str
/// limit : float
///     Per-occurrence limit; may be ``inf``.
/// attachment : float
/// share : float, default 1.0
///     Placed share, in ``(0, 1]``.
/// aggregate_deductible : float, default 0.0
/// aggregate_limit : float, default inf
/// reinstatements : int, optional
///     Free reinstatements: sets ``aggregate_limit`` to
///     ``limit * (reinstatements + 1)``; cannot be combined with
///     ``aggregate_limit``.
/// premium : float, default 0.0
///     Upfront premium for the placed share; used only by
///     ``reinstatement_rates``.
/// reinstatement_rates : list of float, optional
///     Paid reinstatements, one rate per reinstatement as a fraction of
///     ``premium`` (1.0 is 100%), pro rata as to amount. Sets
///     ``aggregate_limit`` to ``limit * (len(reinstatement_rates) + 1)``;
///     cannot be combined with ``aggregate_limit`` or ``reinstatements``.
///
/// Raises
/// ------
/// ValueError
///     If a term is out of range, or more than one of ``aggregate_limit``,
///     ``reinstatements`` and ``reinstatement_rates`` is given.
///
/// Examples
/// --------
/// >>> from actuarialrs.aggregate import Layer
/// >>> layer = Layer("5x5", 5e6, 5e6, reinstatements=1)
/// >>> layer.ceded([7e6])
/// 2000000.0
/// >>> layer.ceded([12e6, 20e6, 30e6])
/// 10000000.0
/// >>> paid = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
/// >>> paid.reinstatement_premium([22.0, 12.0])
/// 2.2
#[pyclass(name = "Layer", module = "actuarialrs.aggregate", frozen)]
pub(crate) struct PyLayer {
    inner: Layer,
}

#[pymethods]
impl PyLayer {
    #[new]
    #[pyo3(signature = (name, limit, attachment, share = 1.0, aggregate_deductible = 0.0, aggregate_limit = None, reinstatements = None, premium = 0.0, reinstatement_rates = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        limit: f64,
        attachment: f64,
        share: f64,
        aggregate_deductible: f64,
        aggregate_limit: Option<f64>,
        reinstatements: Option<u32>,
        premium: f64,
        reinstatement_rates: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut layer = Layer::xol(name, limit, attachment)
            .and_then(|l| l.share(share))
            .and_then(|l| l.aggregate_deductible(aggregate_deductible))
            .map_err(to_py)?;
        layer = match (aggregate_limit, reinstatements, reinstatement_rates) {
            (Some(aal), None, None) => layer.aggregate_limit(aal).map_err(to_py)?,
            (None, Some(n), None) => layer.reinstatements(n).map_err(to_py)?,
            (None, None, Some(rates)) => {
                layer.paid_reinstatements(premium, rates).map_err(to_py)?
            }
            (None, None, None) => layer,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "give at most one of aggregate_limit, reinstatements and reinstatement_rates",
                ));
            }
        };
        Ok(Self { inner: layer })
    }

    /// A quota share ceding ``cession`` of every loss.
    ///
    /// Unlimited cover from the first unit with ``share = cession``.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// cession : float
    ///     In ``(0, 1]``.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.aggregate import Layer
    /// >>> Layer.quota_share("QS", 0.4).ceded([10.0, 5.0])
    /// 6.0
    #[staticmethod]
    fn quota_share(name: String, cession: f64) -> PyResult<Self> {
        let inner = Layer::quota_share(name, cession).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// An aggregate stop-loss: ``limit`` xs ``retention`` on the year's total.
    ///
    /// Covers the total of the losses it sees: gross, or net of earlier
    /// stages in an inuring ``Tower``.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// limit : float
    ///     Annual limit; may be ``inf``.
    /// retention : float
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.aggregate import Layer
    /// >>> Layer.stop_loss("SL", 50.0, 100.0).ceded([60.0, 70.0])
    /// 30.0
    #[staticmethod]
    fn stop_loss(name: String, limit: f64, retention: f64) -> PyResult<Self> {
        let inner = Layer::stop_loss(name, limit, retention).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Layer name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Per-occurrence limit.
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit
    }

    /// Per-occurrence attachment.
    #[getter]
    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    /// Placed share.
    #[getter]
    fn share(&self) -> f64 {
        self.inner.share
    }

    /// Annual aggregate deductible.
    #[getter]
    fn aggregate_deductible(&self) -> f64 {
        self.inner.aggregate_deductible
    }

    /// Annual aggregate limit.
    #[getter]
    fn aggregate_limit(&self) -> f64 {
        self.inner.aggregate_limit
    }

    /// Upfront premium for the placed share.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// Rate of each paid reinstatement; empty when reinstatements are free.
    #[getter]
    fn reinstatement_rates(&self) -> Vec<f64> {
        self.inner.reinstatement_rates.clone()
    }

    /// Ceded loss for one year's losses.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn ceded(&self, losses: Vec<f64>) -> f64 {
        self.inner.ceded(&losses)
    }

    /// Ceded loss per event for one year, taking losses as chronological.
    ///
    /// The annual deductible absorbs the first recoveries and the annual
    /// limit stops the last ones; the entries sum to ``ceded(losses)``.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.aggregate import Layer
    /// >>> layer = Layer("L", 10.0, 5.0, aggregate_deductible=4.0, aggregate_limit=15.0)
    /// >>> layer.ceded_by_event([8.0, 20.0, 12.0])
    /// [0.0, 9.0, 6.0]
    fn ceded_by_event(&self, losses: Vec<f64>) -> Vec<f64> {
        self.inner.ceded_by_event(&losses)
    }

    /// Reinstatement premium for one year's losses.
    ///
    /// With layer loss ``L`` at 100% after annual terms, ``premium *
    /// sum(rate_k * min(max(L - k * limit, 0), limit) / limit)``; zero when
    /// reinstatements are free.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn reinstatement_premium(&self, losses: Vec<f64>) -> f64 {
        self.inner.reinstatement_premium(&losses)
    }

    fn __repr__(&self) -> String {
        format!(
            "Layer({:?}, limit={:?}, attachment={:?}, share={:?})",
            self.inner.name, self.inner.limit, self.inner.attachment, self.inner.share
        )
    }
}

/// A reinsurance programme: layers in inuring stages.
///
/// ``Tower(layers)`` is one stage: every layer sees the gross losses.
/// ``Tower.inuring(stages)`` applies stages in order, each seeing the losses
/// net of all earlier stages, event by event.
///
/// Parameters
/// ----------
/// layers : list of Layer
///     At least one; names must be unique.
///
/// Raises
/// ------
/// ValueError
///     If there are no layers or two share a name.
///
/// Examples
/// --------
/// >>> from actuarialrs.aggregate import Layer, Tower, simulate_events
/// >>> from actuarialrs.distributions import Lognormal, Poisson
/// >>> events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 1_000, 7)
/// >>> tower = Tower([Layer("5x5", 5e6, 5e6), Layer("15x10", 15e6, 10e6)])
/// >>> result = tower.apply(events)
/// >>> [k[0] for k in result.aggregate(["kind"]).components()]
/// ['gross', 'ceded', 'net']
#[pyclass(name = "Tower", module = "actuarialrs.aggregate", frozen)]
pub(crate) struct PyTower {
    inner: Tower,
}

#[pymethods]
impl PyTower {
    #[new]
    fn new(layers: Vec<PyRef<'_, PyLayer>>) -> PyResult<Self> {
        let inner = Tower::new(layers.iter().map(|l| l.inner.clone()).collect()).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// A tower whose stages inure in order.
    ///
    /// Each stage's layers see the losses net of all earlier stages, event
    /// by event, with annual terms used up in event order.
    ///
    /// Parameters
    /// ----------
    /// stages : list of list of Layer
    ///     No stage may be empty; names must be unique across stages.
    ///
    /// Returns
    /// -------
    /// Tower
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.aggregate import Layer, Tower
    /// >>> tower = Tower.inuring([[Layer.quota_share("QS", 0.5)], [Layer("5x5", 5.0, 5.0)]])
    /// >>> tower.ceded([30.0])
    /// [15.0, 5.0]
    #[staticmethod]
    fn inuring(stages: Vec<Vec<PyRef<'_, PyLayer>>>) -> PyResult<Self> {
        let stages = stages
            .iter()
            .map(|stage| stage.iter().map(|l| l.inner.clone()).collect())
            .collect();
        let inner = Tower::inuring(stages).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Stage of each layer, in order, starting at 0.
    #[getter]
    fn stages(&self) -> Vec<usize> {
        self.inner.stages.clone()
    }

    /// Ceded loss of each layer, in order, for one year's losses.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn ceded(&self, losses: Vec<f64>) -> Vec<f64> {
        self.inner.ceded(&losses)
    }

    /// Layer names, in order.
    #[getter]
    fn layer_names(&self) -> Vec<String> {
        self.inner.layers.iter().map(|l| l.name.clone()).collect()
    }

    /// Applies the tower to every simulated year.
    ///
    /// The result has dimensions ``["kind", "layer"]``: ``("gross",
    /// "ground_up")``, ``("ceded", name)`` per layer, ``("net",
    /// "retained")``, then ``("reinstatement_premium", name)`` per layer with
    /// paid reinstatements. ``aggregate(["kind"])`` gives gross, total ceded
    /// and net; net is a loss, before premiums.
    ///
    /// Parameters
    /// ----------
    /// events : EventSet
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn apply(
        &self,
        py: Python<'_>,
        events: PyRef<'_, PyEventSet>,
    ) -> PyResult<PyPredictiveDistribution> {
        let (tower, events) = (&self.inner, &events.inner);
        let inner = py.detach(|| tower.apply(events)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __repr__(&self) -> String {
        format!("Tower(layers={:?})", self.layer_names())
    }
}
