//! `actuarialrs.reinsurance`: wrappers over `act_aggregate::reinsurance`
//! (layers and towers, applied to simulated losses or on the grid).

use act_aggregate::{Layer, Tower, TowerGrids};
use pyo3::prelude::*;

use crate::aggregate::{AnyCount, PyCompoundReport, PyEventSet};
use crate::distributions::{PyGrid, PyPredictiveDistribution};
use crate::to_py;

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
/// pro_rata_time : bool, default False
///     Paid reinstatements also pro rata as to time: the limit a loss at
///     time ``t`` (the fraction of the year elapsed) uses up is charged at
///     ``1 - t``. Needs ``reinstatement_rates``, and events with times
///     (``EventSet.with_uniform_times``, or ``times=`` in
///     ``EventSet.from_years``).
///
/// Raises
/// ------
/// ValueError
///     If a term is out of range, or more than one of ``aggregate_limit``,
///     ``reinstatements`` and ``reinstatement_rates`` is given.
///
/// Examples
/// --------
/// >>> from actuarialrs.reinsurance import Layer
/// >>> layer = Layer("5x5", 5e6, 5e6, reinstatements=1)
/// >>> layer.ceded([7e6])
/// 2000000.0
/// >>> layer.ceded([12e6, 20e6, 30e6])
/// 10000000.0
/// >>> paid = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
/// >>> paid.reinstatement_premium([22.0, 12.0])
/// 2.2
/// >>> timed = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5],
/// ...               pro_rata_time=True)
/// >>> round(timed.reinstatement_premium([22.0, 12.0], times=[0.25, 0.5]), 12)
/// 1.6
#[pyclass(name = "Layer", module = "actuarialrs.reinsurance", frozen)]
pub(crate) struct PyLayer {
    inner: Layer,
}

#[pymethods]
impl PyLayer {
    #[new]
    #[pyo3(signature = (name, limit, attachment, share = 1.0, aggregate_deductible = 0.0, aggregate_limit = None, reinstatements = None, premium = 0.0, reinstatement_rates = None, pro_rata_time = false))]
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
        pro_rata_time: bool,
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
        if pro_rata_time {
            layer = layer.pro_rata_as_to_time().map_err(to_py)?;
        }
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
    /// >>> from actuarialrs.reinsurance import Layer
    /// >>> Layer.quota_share("QS", 0.4).ceded([10.0, 5.0])
    /// 6.0
    #[staticmethod]
    fn quota_share(name: String, cession: f64) -> PyResult<Self> {
        let inner = Layer::quota_share(name, cession).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// A surplus treaty: each risk cedes the part of its sum insured above
    /// the retention line ``retention``, up to ``lines`` lines, and the
    /// same share of every loss on it.
    ///
    /// With a retention of 1m and 9 lines (a capacity of 9m), a 5m risk
    /// cedes 80% and a 20m risk 45%. The events must carry sums insured
    /// (``EventSet.from_years(..., sums_insured=...)``); it can inure to a
    /// per-risk excess of loss in a later stage of a ``Tower``.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// retention : float
    ///     The retention line; positive.
    /// lines : float
    ///     Number of lines of capacity; positive.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reinsurance import Layer
    /// >>> s = Layer.surplus("surplus", 1e6, 9.0)
    /// >>> round(s.ceded_with_sums_insured([2e6, 2e6], [5e6, 20e6]))
    /// 2500000
    #[staticmethod]
    fn surplus(name: String, retention: f64, lines: f64) -> PyResult<Self> {
        let inner = Layer::surplus(name, retention, lines).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Ceded loss for one year's losses on risks with the given sums
    /// insured, one per loss.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    /// sums_insured : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn ceded_with_sums_insured(&self, losses: Vec<f64>, sums_insured: Vec<f64>) -> PyResult<f64> {
        if losses.len() != sums_insured.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give one sum insured per loss",
            ));
        }
        Ok(self.inner.ceded_with_sums_insured(&losses, &sums_insured))
    }

    /// Whether the layer is a surplus treaty, which needs sums insured.
    #[getter]
    fn needs_sums_insured(&self) -> bool {
        self.inner.needs_sums_insured()
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
    /// >>> from actuarialrs.reinsurance import Layer
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

    /// Whether paid reinstatements are pro rata as to time.
    #[getter]
    fn pro_rata_time(&self) -> bool {
        self.inner.pro_rata_time
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
    /// >>> from actuarialrs.reinsurance import Layer
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
    /// reinstatements are free. Pro rata as to time, the limit each loss
    /// uses up is charged at ``1 - t``, its time's share of the year left.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///     In time order.
    /// times : list of float, optional
    ///     Each loss's time, as the fraction of the year elapsed; needed
    ///     (and only used) when the layer is pro rata as to time.
    ///
    /// Returns
    /// -------
    /// float
    ///     NaN for a layer pro rata as to time without ``times``.
    #[pyo3(signature = (losses, times = None))]
    fn reinstatement_premium(&self, losses: Vec<f64>, times: Option<Vec<f64>>) -> PyResult<f64> {
        match times {
            None => Ok(self.inner.reinstatement_premium(&losses)),
            Some(t) if t.len() == losses.len() => {
                Ok(self.inner.reinstatement_premium_dated(&losses, &t))
            }
            Some(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "give one time per loss",
            )),
        }
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
/// >>> from actuarialrs.aggregate import simulate_events
/// >>> from actuarialrs.reinsurance import Layer, Tower
/// >>> from actuarialrs.distributions import Lognormal, Poisson
/// >>> events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 1_000, 7)
/// >>> tower = Tower([Layer("5x5", 5e6, 5e6), Layer("15x10", 15e6, 10e6)])
/// >>> result = tower.apply(events)
/// >>> [k[0] for k in result.aggregate(["kind"]).components()]
/// ['gross', 'ceded', 'net']
#[pyclass(name = "Tower", module = "actuarialrs.reinsurance", frozen)]
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
    /// >>> from actuarialrs.reinsurance import Layer, Tower
    /// >>> tower = Tower.inuring([[Layer.quota_share("QS", 0.5)], [Layer("5x5", 5.0, 5.0)]])
    /// >>> tower.ceded([30.0])
    /// [15.0, 5.0]
    /// Reads a document written by ``Tower.to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Tower
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On malformed JSON, another format, a newer format version, or a
    ///     term a layer refuses.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let inner = Tower::from_json(text).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The programme as a versioned JSON document: every stage and layer
    /// with all its terms, numbers bit for bit. ``Tower.from_json`` reads it
    /// back to an equal tower, rebuilding each layer through the same
    /// checks; towers also pickle this way.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reinsurance import Layer, Tower
    /// >>> tower = Tower.inuring([[Layer.surplus("S", 1e6, 4.0)], [Layer("xl", 2e6, 1e6)]])
    /// >>> back = Tower.from_json(tower.to_json())
    /// >>> back.to_json() == tower.to_json()
    /// True
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
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
    /// >>> from actuarialrs.reinsurance import Layer, Tower
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

    /// Applies the tower to any predictive distribution, each simulation's
    /// total taken as one aggregate loss: an adverse development cover on a
    /// reserve bootstrap, a stop-loss or quota share on modelled premium
    /// risk. An occurrence layer sees the total as one occurrence, so it
    /// acts as an aggregate excess of loss. Components as ``apply``.
    ///
    /// Parameters
    /// ----------
    /// losses : PredictiveDistribution
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn apply_aggregate(
        &self,
        py: Python<'_>,
        losses: PyRef<'_, PyPredictiveDistribution>,
    ) -> PyResult<PyPredictiveDistribution> {
        let (tower, pd) = (&self.inner, &losses.inner);
        let inner = py.detach(|| tower.apply_aggregate(pd)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// Gross, ceded and net annual distributions on the grid, by FFT.
    ///
    /// Each layer's per-occurrence recoveries form a severity grid, which
    /// is compounded with the same claim count; annual terms and the share
    /// then apply to the total. With boundaries on multiples of the step
    /// the grids are exact for the discretized problem, with no sampling
    /// error. Grids are marginal (use ``apply`` on simulated events for
    /// joint results). ``net`` is given when no layer has annual terms, or
    /// when the last stage is a single aggregate cover such as a
    /// stop-loss; otherwise it is ``None``.
    ///
    /// Parameters
    /// ----------
    /// frequency : Poisson, NegativeBinomial or Binomial
    /// severity : Grid
    /// points : int
    ///     Points in every aggregate grid.
    ///
    /// Returns
    /// -------
    /// TowerGrids
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a layer with annual terms inures to a later stage, or
    ///     ``points`` is 0.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reinsurance import Layer, Tower
    /// >>> from actuarialrs.distributions import Grid, Poisson
    /// >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
    /// >>> r = Tower([Layer("2x2", 2.0, 2.0)]).on_grid(Poisson(3.0), sev, 200)
    /// >>> round(r.ceded[0].mean(), 12), r.on_points
    /// (1.2, True)
    fn on_grid(
        &self,
        py: Python<'_>,
        frequency: &Bound<'_, PyAny>,
        severity: PyRef<'_, PyGrid>,
        points: usize,
    ) -> PyResult<PyTowerGrids> {
        let n = AnyCount::extract(frequency)?;
        let (tower, sev) = (&self.inner, severity.inner.clone());
        let inner = py
            .detach(|| tower.on_grid(n.as_counting(), &sev, points))
            .map_err(to_py)?;
        Ok(PyTowerGrids { inner })
    }

    fn __repr__(&self) -> String {
        format!("Tower(layers={:?})", self.layer_names())
    }
}

/// A tower's annual distributions on the grid, from ``Tower.on_grid``.
///
/// Every grid is a marginal distribution. Ceded grids are at the placed
/// share and after annual terms; a layer with share ``c`` has step
/// ``c * h``.
#[pyclass(name = "TowerGrids", module = "actuarialrs.reinsurance", frozen)]
pub(crate) struct PyTowerGrids {
    inner: TowerGrids,
}

#[pymethods]
impl PyTowerGrids {
    /// Annual gross loss.
    #[getter]
    fn gross(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.gross.clone(),
        }
    }

    /// The compound calculation behind ``gross``.
    #[getter]
    fn gross_report(&self) -> PyCompoundReport {
        PyCompoundReport {
            inner: self.inner.gross_report.clone(),
        }
    }

    /// Annual ceded loss of each layer, in tower order.
    #[getter]
    fn ceded(&self) -> Vec<PyGrid> {
        self.inner
            .ceded
            .iter()
            .map(|g| PyGrid { inner: g.clone() })
            .collect()
    }

    /// The compound calculation behind each layer: its annual recovery at
    /// 100%, before annual terms.
    #[getter]
    fn ceded_reports(&self) -> Vec<PyCompoundReport> {
        self.inner
            .ceded_reports
            .iter()
            .map(|r| PyCompoundReport { inner: r.clone() })
            .collect()
    }

    /// Annual net loss, or ``None`` when it is not a single compound total.
    #[getter]
    fn net(&self) -> Option<PyGrid> {
        self.inner.net.clone().map(|inner| PyGrid { inner })
    }

    /// Expected reinstatement premium of each layer; 0 without paid
    /// reinstatements.
    #[getter]
    fn expected_reinstatement_premium(&self) -> Vec<f64> {
        self.inner.expected_reinstatement_premium.clone()
    }

    /// Whether every boundary and net loss fell on a grid point. When
    /// ``False``, means are still exact but shapes are smeared by up to a
    /// step.
    #[getter]
    fn on_points(&self) -> bool {
        self.inner.on_points
    }

    fn __repr__(&self) -> String {
        format!(
            "TowerGrids(layers={}, net={}, on_points={})",
            self.inner.ceded.len(),
            self.inner.net.is_some(),
            self.inner.on_points
        )
    }
}
