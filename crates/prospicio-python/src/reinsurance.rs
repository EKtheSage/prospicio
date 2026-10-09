//! `prospicio.reinsurance`: wrappers over `prospicio_aggregate::reinsurance`
//! (layers and towers, applied to simulated losses or on the grid).

use prospicio_aggregate::{Commission, Layer, LossSensitivePremium, Tower, TowerGrids};
use pyo3::prelude::*;

use crate::aggregate::{AnyCount, PyCompoundReport, PyEventSet};
use crate::distributions::{PyGrid, PyPredictiveDistribution};
use crate::to_py;

/// A per-occurrence excess-of-loss layer: ``limit`` xs ``attachment`` on each
/// loss, then annual terms.
///
/// For one year, ``ceded = share * min(max(sum of per-loss recoveries -
/// aggregate_deductible, 0), aggregate_limit)``, less any loss corridor
/// (``with_loss_corridor``) before the annual limit.
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
///     Upfront premium for the placed share: the base of paid
///     reinstatements, ceding and profit commissions and a sliding scale's
///     loss ratio. ``with_rate_on_line``, ``with_premium_rate`` and
///     ``with_deposit_premium`` set it from a quote.
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
/// >>> from prospicio.reinsurance import Layer
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
#[pyclass(name = "Layer", module = "prospicio.reinsurance", frozen)]
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
            (None, None, Some(rates)) => {
                layer.paid_reinstatements(premium, rates).map_err(to_py)?
            }
            (aal, n, None) if aal.is_none() || n.is_none() => {
                let layer = match (aal, n) {
                    (Some(aal), _) => layer.aggregate_limit(aal).map_err(to_py)?,
                    (_, Some(n)) => layer.reinstatements(n).map_err(to_py)?,
                    _ => layer,
                };
                with_premium(layer, premium)?
            }
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
    /// >>> from prospicio.reinsurance import Layer
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
    /// >>> from prospicio.reinsurance import Layer
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
    /// >>> from prospicio.reinsurance import Layer
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

    /// The same layer with a loss corridor.
    ///
    /// Of the annual layer loss at 100% after the annual deductible, the
    /// cedant keeps ``retained`` of the part between ``lower`` and
    /// ``upper``; the annual limit then caps what is left, so the reinsurer
    /// still pays up to the full annual limit. Reinstatement premiums
    /// follow the loss after the corridor. A corridor quoted as loss
    /// ratios ``lr`` on the reinsurer's premium ``P`` for a placed share
    /// ``s`` is ``lr * P / s`` (for a quota share, ``P / s`` is the
    /// subject premium).
    ///
    /// Parameters
    /// ----------
    /// lower : float
    ///     Non-negative.
    /// upper : float
    ///     Finite, above ``lower``.
    /// retained : float, default 1.0
    ///     Share of the band the cedant keeps, in ``(0, 1]``.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = Layer.quota_share("QS", 0.3).with_loss_corridor(70.0, 90.0)
    /// >>> round(qs.ceded([50.0, 30.0]), 12), round(qs.ceded([120.0]), 12)
    /// (21.0, 30.0)
    #[pyo3(signature = (lower, upper, retained = 1.0))]
    fn with_loss_corridor(&self, lower: f64, upper: f64, retained: f64) -> PyResult<Self> {
        let inner = self
            .inner
            .clone()
            .loss_corridor(lower, upper, retained)
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The loss corridor as ``(lower, upper, retained)``, or ``None``.
    #[getter]
    fn loss_corridor(&self) -> Option<(f64, f64, f64)> {
        self.inner.corridor.map(|c| (c.lower, c.upper, c.retained))
    }

    /// The same layer with its premium set from a deposit quoted for 100%
    /// of the layer: ``premium = share * amount``.
    ///
    /// Parameters
    /// ----------
    /// amount : float
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> Layer("5x5", 5e6, 5e6, share=0.4).with_deposit_premium(1e6).premium
    /// 400000.0
    fn with_deposit_premium(&self, amount: f64) -> PyResult<Self> {
        self.map(|l| l.deposit_premium(amount))
    }

    /// The same layer with its premium set from a rate on line:
    /// ``premium = share * rol * limit``. Needs a finite limit.
    ///
    /// Parameters
    /// ----------
    /// rol : float
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> Layer("10x10", 10e6, 10e6).with_rate_on_line(0.125).premium
    /// 1250000.0
    fn with_rate_on_line(&self, rol: f64) -> PyResult<Self> {
        self.map(|l| l.rate_on_line(rol))
    }

    /// The same layer with its premium set as a rate on the subject
    /// premium: ``premium = share * rate * subject_premium``. A quota
    /// share's ceded premium is ``with_premium_rate(1.0, subject_premium)``.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    /// subject_premium : float
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> Layer.quota_share("QS", 0.3).with_premium_rate(1.0, 1000.0).premium
    /// 300.0
    fn with_premium_rate(&self, rate: f64, subject_premium: f64) -> PyResult<Self> {
        self.map(|l| l.premium_rate(rate, subject_premium))
    }

    /// The same layer with a flat ceding commission: ``rate`` of the
    /// year's premium (the swing-rated premium, if any) is paid back to
    /// the cedant. Reinstatement premiums carry none.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// Layer
    fn with_ceding_commission(&self, rate: f64) -> PyResult<Self> {
        self.map(|l| l.ceding_commission(rate))
    }

    /// The same layer with a sliding-scale ceding commission.
    ///
    /// The commission rate at the year's ceded loss ratio (ceded loss over
    /// premium) is interpolated linearly between ``(commission,
    /// loss_ratio)`` anchors and flat beyond the first and last, as
    /// ``aggregate``'s ``slide``. Set the premium first; it cannot be
    /// combined with swing rating or paid reinstatements, and replaces a
    /// flat commission.
    ///
    /// Parameters
    /// ----------
    /// anchors : list of (float, float)
    ///     ``(commission, loss_ratio)`` pairs; the commission must not rise
    ///     with the loss ratio.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_sliding_scale([(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)]))
    /// >>> round(qs.ceding_commission_for(65.0), 12)
    /// 35.0
    fn with_sliding_scale(&self, anchors: Vec<(f64, f64)>) -> PyResult<Self> {
        self.map(|l| l.sliding_scale(anchors))
    }

    /// The same layer with a profit commission: ``share`` of ``max(premium
    /// * (1 - allowance) - ceded, 0)``, as ``aggregate``'s ``pc <share>
    /// after <allowance>``.
    ///
    /// The allowance is the reinsurer's expenses and margin as a fraction
    /// of premium, including any ceding commission the contract deducts
    /// before profit. Set the premium first; it cannot be combined with
    /// swing rating or paid reinstatements.
    ///
    /// Parameters
    /// ----------
    /// share : float
    ///     In ``[0, 1]``.
    /// allowance : float, default 0.0
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_profit_commission(0.25, 0.1))
    /// >>> round(qs.profit_commission_for(60.0), 12)
    /// 7.5
    #[pyo3(signature = (share, allowance = 0.0))]
    fn with_profit_commission(&self, share: f64, allowance: f64) -> PyResult<Self> {
        self.map(|l| l.profit_commission(share, allowance))
    }

    /// The same layer, swing rated: the year's premium is ``clip(share *
    /// basic + lcm * ceded, share * minimum, share * maximum)``.
    ///
    /// The terms are quoted for 100% of the layer, as in ``aggregate``; the
    /// ceded loss is already at the placed share. The swing premium
    /// replaces the fixed premium in the tower's results. It cannot be
    /// combined with paid reinstatements, a sliding scale or a profit
    /// commission.
    ///
    /// Parameters
    /// ----------
    /// basic : float
    /// lcm : float
    ///     Loss conversion factor, such as ``100 / 80``.
    /// minimum : float, optional
    ///     Defaults to ``basic``.
    /// maximum : float, optional
    ///     Defaults to no cap.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> layer = Layer("L", 1000.0, 0.0).with_swing_rating(0.0, 1.25, 100.0, 300.0)
    /// >>> [layer.premium_for(x) for x in (40.0, 200.0, 400.0)]
    /// [100.0, 250.0, 300.0]
    #[pyo3(signature = (basic, lcm, minimum = None, maximum = None))]
    fn with_swing_rating(
        &self,
        basic: f64,
        lcm: f64,
        minimum: Option<f64>,
        maximum: Option<f64>,
    ) -> PyResult<Self> {
        let terms = LossSensitivePremium::new(basic, lcm, minimum, maximum).map_err(to_py)?;
        self.map(|l| l.swing_rated(terms))
    }

    /// The flat ceding commission rate, or ``None`` (also when the
    /// commission slides).
    #[getter]
    fn ceding_commission(&self) -> Option<f64> {
        match self.inner.commission {
            Some(Commission::Flat(c)) => Some(c),
            _ => None,
        }
    }

    /// The sliding scale's ``(commission, loss_ratio)`` anchors in
    /// increasing loss ratio, or ``None``.
    #[getter]
    fn sliding_scale(&self) -> Option<Vec<(f64, f64)>> {
        match &self.inner.commission {
            Some(Commission::SlidingScale(a)) => Some(a.clone()),
            _ => None,
        }
    }

    /// The profit commission as ``(share, allowance)``, or ``None``.
    #[getter]
    fn profit_commission(&self) -> Option<(f64, f64)> {
        self.inner
            .profit_commission
            .map(|pc| (pc.share, pc.allowance))
    }

    /// The swing rating at 100% as ``(basic, lcm, minimum, maximum)``, or
    /// ``None``.
    #[getter]
    fn swing_rating(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner
            .swing
            .map(|s| (s.basic, s.lcm, s.minimum, s.maximum))
    }

    /// The year's premium given its ceded loss: the swing-rated premium,
    /// or else the fixed ``premium``.
    ///
    /// Parameters
    /// ----------
    /// ceded : float
    ///     The layer's ceded loss for the year, at the placed share.
    ///
    /// Returns
    /// -------
    /// float
    fn premium_for(&self, ceded: f64) -> f64 {
        self.inner.premium_for(ceded)
    }

    /// The year's ceding commission (flat or sliding) given its ceded loss.
    ///
    /// Parameters
    /// ----------
    /// ceded : float
    ///
    /// Returns
    /// -------
    /// float
    fn ceding_commission_for(&self, ceded: f64) -> f64 {
        self.inner.ceding_commission_for(ceded)
    }

    /// The year's profit commission given its ceded loss.
    ///
    /// Parameters
    /// ----------
    /// ceded : float
    ///
    /// Returns
    /// -------
    /// float
    fn profit_commission_for(&self, ceded: f64) -> f64 {
        self.inner.profit_commission_for(ceded)
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
    /// >>> from prospicio.reinsurance import Layer
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

impl PyLayer {
    /// A copy of the layer with one builder applied.
    fn map(&self, f: impl FnOnce(Layer) -> prospicio_core::Result<Layer>) -> PyResult<Self> {
        let inner = f(self.inner.clone()).map_err(to_py)?;
        Ok(Self { inner })
    }
}

/// Sets the premium of a layer without paid reinstatements, which set it
/// themselves.
fn with_premium(mut layer: Layer, premium: f64) -> PyResult<Layer> {
    if !(premium.is_finite() && premium >= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "premium must be finite and non-negative",
        ));
    }
    layer.premium = premium;
    Ok(layer)
}

/// A premium that depends on the year's loss ``x``: ``clip(basic + lcm *
/// x, minimum, maximum)``.
///
/// The premium of a retrospectively rated policy, with ``x`` the account's
/// loss (net of any reinsurance that inures to it), and of a swing-rated
/// layer (``Layer.with_swing_rating``). For a retro plan, ``basic`` and
/// ``lcm`` include the tax multiplier.
///
/// Parameters
/// ----------
/// basic : float
/// lcm : float
///     Loss conversion factor.
/// minimum : float, optional
///     Defaults to ``basic``.
/// maximum : float, optional
///     Defaults to no cap.
///
/// Examples
/// --------
/// >>> from prospicio.reinsurance import LossSensitivePremium
/// >>> retro = LossSensitivePremium(1000.0, 1.1, maximum=2500.0)
/// >>> [round(p, 9) for p in retro.premium([0.0, 500.0, 1500.0])]
/// [1000.0, 1550.0, 2500.0]
#[pyclass(
    name = "LossSensitivePremium",
    module = "prospicio.reinsurance",
    frozen
)]
pub(crate) struct PyLossSensitivePremium {
    inner: LossSensitivePremium,
}

#[pymethods]
impl PyLossSensitivePremium {
    #[new]
    #[pyo3(signature = (basic, lcm, minimum = None, maximum = None))]
    fn new(basic: f64, lcm: f64, minimum: Option<f64>, maximum: Option<f64>) -> PyResult<Self> {
        let inner = LossSensitivePremium::new(basic, lcm, minimum, maximum).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The premium for each loss.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn premium(&self, losses: Vec<f64>) -> Vec<f64> {
        losses.iter().map(|&x| self.inner.premium(x)).collect()
    }

    /// The basic premium.
    #[getter]
    fn basic(&self) -> f64 {
        self.inner.basic
    }

    /// The loss conversion factor.
    #[getter]
    fn lcm(&self) -> f64 {
        self.inner.lcm
    }

    /// The minimum premium.
    #[getter]
    fn minimum(&self) -> f64 {
        self.inner.minimum
    }

    /// The maximum premium; ``inf`` when uncapped.
    #[getter]
    fn maximum(&self) -> f64 {
        self.inner.maximum
    }

    fn __repr__(&self) -> String {
        format!(
            "LossSensitivePremium(basic={:?}, lcm={:?}, minimum={:?}, maximum={:?})",
            self.inner.basic, self.inner.lcm, self.inner.minimum, self.inner.maximum
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
/// >>> from prospicio.aggregate import simulate_events
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> from prospicio.distributions import Lognormal, Poisson
/// >>> events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 1_000, 7)
/// >>> tower = Tower([Layer("5x5", 5e6, 5e6), Layer("15x10", 15e6, 10e6)])
/// >>> result = tower.apply(events)
/// >>> [k[0] for k in result.aggregate(["kind"]).components()]
/// ['gross', 'ceded', 'net']
#[pyclass(name = "Tower", module = "prospicio.reinsurance", frozen)]
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
    /// >>> from prospicio.reinsurance import Layer, Tower
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
    /// >>> from prospicio.reinsurance import Layer, Tower
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
    /// >>> from prospicio.reinsurance import Layer, Tower
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
    /// paid reinstatements, ``("swing_premium", name)`` per swing-rated
    /// layer, ``("ceding_commission", name)`` per layer with a flat or
    /// sliding commission and ``("profit_commission", name)`` per layer with
    /// a profit commission. ``aggregate(["kind"])`` gives gross, total ceded
    /// and net; net is a loss, before premiums and commissions.
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
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> from prospicio.distributions import Grid, Poisson
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
#[pyclass(name = "TowerGrids", module = "prospicio.reinsurance", frozen)]
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

    /// Expected premium of each layer: the swing-rated premium's mean, or
    /// else the fixed premium (0 when none is set).
    #[getter]
    fn expected_premium(&self) -> Vec<f64> {
        self.inner.expected_premium.clone()
    }

    /// Expected ceding commission (flat or sliding) of each layer.
    #[getter]
    fn expected_ceding_commission(&self) -> Vec<f64> {
        self.inner.expected_ceding_commission.clone()
    }

    /// Expected profit commission of each layer.
    #[getter]
    fn expected_profit_commission(&self) -> Vec<f64> {
        self.inner.expected_profit_commission.clone()
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
