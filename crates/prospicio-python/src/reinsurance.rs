//! `prospicio.reinsurance`: wrappers over `prospicio_aggregate::reinsurance`
//! (layers and towers, applied to simulated losses or on the grid).

use prospicio_aggregate::{
    AnnualGrids, Commission, DealScale, Deduction, EventSet, Layer, LayerStatistics,
    LimitAllocation, LossSensitivePremium, Tower, TowerGrids, TowerPrice, ViewPrice,
};
use prospicio_prob::PredictiveDistribution;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::aggregate::{
    AnyCount, PyCompoundMoments, PyCompoundReport, PyEventSet, PyExposure, PyTimeline,
};
use crate::distributions::{PyGrid, PyPredictiveDistribution};
use crate::risk::PyDistortion;
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
/// allocation : {"event_order", "pro_rata"}, default "event_order"
///     How the annual terms (aggregate deductible and limit, reinstatements)
///     split the year's recovery among its events. ``"event_order"`` uses
///     the aggregate limit up in the order the events occur, so a later
///     event gets what is left. ``"pro_rata"`` scales every event's
///     recovery by the same factor, the annual recovery after the terms
///     over the one before them. The annual total is the same; the split by
///     event, and so what an inuring stage sees, differs. Cannot be
///     combined with ``pro_rata_time``.
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
/// >>> Layer("10x5", 10.0, 5.0, reinstatements=1).ceded_by_event([15.0, 20.0, 10.0])
/// [10.0, 10.0, 0.0]
/// >>> Layer("10x5", 10.0, 5.0, reinstatements=1, allocation="pro_rata").ceded_by_event([15.0, 20.0, 10.0])
/// [8.0, 8.0, 4.0]
#[pyclass(name = "Layer", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyLayer {
    pub(crate) inner: Layer,
}

#[pymethods]
impl PyLayer {
    #[new]
    #[pyo3(signature = (name, limit, attachment, share = 1.0, aggregate_deductible = 0.0, aggregate_limit = None, reinstatements = None, premium = 0.0, reinstatement_rates = None, pro_rata_time = false, allocation = "event_order"))]
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
        allocation: &str,
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
        let allocation = match allocation {
            "event_order" => LimitAllocation::EventOrder,
            "pro_rata" => LimitAllocation::ProRata,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "allocation must be \"event_order\" or \"pro_rata\", not {other:?}"
                )));
            }
        };
        if allocation != LimitAllocation::EventOrder {
            layer = layer.allocation(allocation).map_err(to_py)?;
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

    /// Whether the layer cedes from an ``EventSet``'s aggregate part: it is
    /// on the loss with attachment 0 and no limit, as a quota share or a
    /// stop loss is. A layer with a per-occurrence attachment or limit sees
    /// the events alone.
    #[getter]
    fn sees_aggregate_part(&self) -> bool {
        self.inner.sees_aggregate_part()
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

    /// How the annual terms split the year's recovery among its events:
    /// ``"event_order"`` or ``"pro_rata"``.
    #[getter]
    fn allocation(&self) -> &'static str {
        match self.inner.allocation {
            LimitAllocation::EventOrder => "event_order",
            LimitAllocation::ProRata => "pro_rata",
        }
    }

    /// The losses the layer covers, ``(label, values)``, or None for
    /// every loss.
    #[getter]
    fn scope(&self) -> Option<(String, Vec<String>)> {
        self.inner
            .scope
            .as_ref()
            .map(|s| (s.label.clone(), s.values.clone()))
    }

    /// The same layer covering only the losses whose label ``label``
    /// takes one of ``values``: a windstorm treaty on a multi-peril event
    /// set is scoped to ``("peril", ["windstorm"])``.
    ///
    /// The layer's occurrence and annual terms see only those losses, so
    /// another peril's events neither use its annual limit nor count
    /// toward its deductible; the losses it does not cover pass to the
    /// next inuring stage whole. That differs from reading one label's
    /// share of an unscoped layer, whose annual terms every peril uses
    /// up. A tower with a scoped layer needs labelled events
    /// (``Tower.apply``, ``apply_term``, ``apply_timeline``) and refuses
    /// the grids and annual losses.
    ///
    /// Parameters
    /// ----------
    /// label : str
    ///     A label dimension of the events.
    /// values : list of str
    ///     At least one; at least one must occur.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> e = EventSet.from_years([[20.0, 15.0]]).with_labels({"peril": ["quake", "wind"]})
    /// >>> xl = Layer("10x5", 10.0, 5.0, aggregate_limit=15.0)
    /// >>> wind = Tower([xl.with_scope("peril", ["wind"])]).apply(e)
    /// >>> wind.marginal(("ceded", "10x5", "wind")).mean()
    /// 10.0
    /// >>> Tower([xl]).apply(e).marginal(("ceded", "10x5", "wind")).mean()
    /// 5.0
    fn with_scope(&self, label: &str, values: Vec<String>) -> PyResult<Self> {
        let values: Vec<&str> = values.iter().map(String::as_str).collect();
        self.map(|l| l.scope(label, &values))
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

    /// The same layer, carrying the profit commission's deficits forward
    /// between the years of a term (``Tower.apply_term``).
    ///
    /// A year whose ceded loss exceeds ``premium * (1 - allowance)`` pays no
    /// commission and carries the excess, which later years' profits cover
    /// before any commission is paid. Deficits are used up oldest first.
    /// Set the profit commission first. A single year (``Tower.apply``)
    /// starts with nothing carried.
    ///
    /// Parameters
    /// ----------
    /// years : int, optional
    ///     How many later years a deficit may be carried into; ``None``
    ///     (the default) carries it until it is used up.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_profit_commission(0.25, 0.1)
    /// ...       .with_profit_commission_carry_forward())
    /// >>> qs.profit_commissions_over([120.0, 40.0])
    /// [0.0, 5.0]
    #[pyo3(signature = (years = None))]
    fn with_profit_commission_carry_forward(&self, years: Option<u32>) -> PyResult<Self> {
        self.map(|l| l.profit_commission_carry_forward(years))
    }

    /// The same layer, carrying the sliding scale's deficits (ceded loss
    /// above the last anchor's loss ratio), credits (below the first
    /// anchor's) or both forward between the years of a term
    /// (``Tower.apply_term``).
    ///
    /// Each year adds what was carried to its own ceded loss before the
    /// commission is read off the scale, and carries what is then still
    /// beyond the scale's ends. Amounts are used up oldest first. Set the
    /// sliding scale first.
    ///
    /// Parameters
    /// ----------
    /// deficit : bool, default True
    /// credit : bool, default False
    /// years : int, optional
    ///     How many later years an amount may be carried into; ``None``
    ///     (the default) without limit.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_sliding_scale([(0.45, 0.60), (0.25, 0.70)])
    /// ...       .with_sliding_scale_carry_forward())
    /// >>> [round(c, 12) for c in qs.ceding_commissions_over([80.0, 55.0])]
    /// [25.0, 35.0]
    #[pyo3(signature = (deficit = true, credit = false, years = None))]
    fn with_sliding_scale_carry_forward(
        &self,
        deficit: bool,
        credit: bool,
        years: Option<u32>,
    ) -> PyResult<Self> {
        self.map(|l| l.sliding_scale_carry_forward(deficit, credit, years))
    }

    /// The profit commission's carry-forward as ``{"years": n}`` (``n`` is
    /// ``None`` without limit), or ``None``.
    #[getter]
    fn profit_commission_carry<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .profit_commission_carry
            .map(|cf| {
                let d = PyDict::new(py);
                d.set_item("years", cf.years)?;
                Ok(d)
            })
            .transpose()
    }

    /// The sliding scale's carry-forward as ``{"deficit", "credit",
    /// "years"}``, or ``None``.
    #[getter]
    fn sliding_scale_carry<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .sliding_scale_carry
            .map(|cf| {
                let d = PyDict::new(py);
                d.set_item("deficit", cf.deficit)?;
                d.set_item("credit", cf.credit)?;
                d.set_item("years", cf.years)?;
                Ok(d)
            })
            .transpose()
    }

    /// The profit commissions over consecutive years of a term with these
    /// ceded losses, carrying deficits forward if the layer does.
    ///
    /// Parameters
    /// ----------
    /// ceded : list of float
    ///     The layer's ceded loss in each year, at the placed share.
    ///
    /// Returns
    /// -------
    /// list of float
    fn profit_commissions_over(&self, ceded: Vec<f64>) -> Vec<f64> {
        self.inner.profit_commissions_over(&ceded)
    }

    /// The ceding commissions over consecutive years of a term with these
    /// ceded losses, carrying the sliding scale's deficits and credits
    /// forward if the layer does.
    ///
    /// Parameters
    /// ----------
    /// ceded : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn ceding_commissions_over(&self, ceded: Vec<f64>) -> Vec<f64> {
        self.inner.ceding_commissions_over(&ceded)
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

    /// The same layer with a flat brokerage: ``rate`` of the year's
    /// premium. It replaces any brokerage set before.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0).with_brokerage(0.025)
    /// >>> round(qs.brokerage_for(70.0), 12)
    /// 2.5
    fn with_brokerage(&self, rate: f64) -> PyResult<Self> {
        self.map(|l| l.brokerage(rate))
    }

    /// The same layer with brokerage on a scale of the loss ratio.
    ///
    /// The rate is read from ``(rate, loss_ratio)`` anchors as a sliding
    /// scale is, linear between anchors and flat beyond the first and
    /// last, and may rise or fall with the loss ratio. It reads the ceded
    /// loss ratio, or the one before annual terms with
    /// ``with_scales_before_annual_terms``, held inside a sliding-scale
    /// commission's range with ``with_brokerage_within_commission_range``.
    /// Set the premium first; it cannot be combined with swing rating or
    /// paid reinstatements.
    ///
    /// Parameters
    /// ----------
    /// anchors : list of (float, float)
    ///     ``(rate, loss_ratio)`` pairs.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_brokerage_scale([(0.03, 0.5), (0.01, 0.8)]))
    /// >>> round(qs.brokerage_for(65.0), 12)
    /// 2.0
    fn with_brokerage_scale(&self, anchors: Vec<(f64, f64)>) -> PyResult<Self> {
        self.map(|l| l.brokerage_scale(anchors))
    }

    /// The same layer with another deduction from its premium: a TPA fee,
    /// tax, reinsurance cost, a fronting fee or a manager's fee.
    ///
    /// Each deduction is its own ``(label, layer)`` component in a tower's
    /// results and its own ratio in ``Tower.deal_ratios``.
    ///
    /// Parameters
    /// ----------
    /// label : str
    ///     Its name, distinct from the layer's other deductions and from
    ///     the kinds a tower's result and the deal ratios use.
    /// value : float
    ///     The rate, or the amount for ``basis="fixed"``; non-negative.
    /// basis : {"premium", "loss", "fixed"}
    ///     A rate of the year's premium, of its ceded loss, or an amount.
    /// inside_nmbf : bool
    ///     Whether the nominal margin before fees counts it; ``False`` for
    ///     a fee kept outside it, such as a manager's fee.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer
    /// >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    /// ...       .with_deduction("tax", 0.03)
    /// ...       .with_deduction("tpa", 0.02, basis="loss"))
    /// >>> [round(qs.deduction_for(k, 70.0), 12) for k in ["tax", "tpa"]]
    /// [3.0, 1.4]
    #[pyo3(signature = (label, value, basis = "premium", inside_nmbf = true))]
    fn with_deduction(
        &self,
        label: String,
        value: f64,
        basis: &str,
        inside_nmbf: bool,
    ) -> PyResult<Self> {
        let basis = crate::pnl::basis_of(basis)?;
        let mut d = Deduction::new(label, basis, value).map_err(to_py)?;
        if !inside_nmbf {
            d = d.outside_nmbf();
        }
        self.map(|l| l.deduction(d))
    }

    /// The same layer with its sliding scales (commission and brokerage)
    /// read on the loss ratio before the annual terms.
    ///
    /// That is the placed share of the year's recoveries before the
    /// annual deductible, the corridor and the annual limit, over the
    /// premium: a loss-ratio cap stated as ``aggregate_limit`` then caps
    /// the ceded loss but not what the scales read. A tower's result
    /// then has each year's ``("loss_before_terms", <layer>)``. The grid
    /// methods and the P&L ledger refuse such a layer with annual terms
    /// and a sliding scale.
    ///
    /// Returns
    /// -------
    /// Layer
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> cover = (Layer("cover", 1000.0, 0.0, aggregate_limit=120.0)
    /// ...          .with_premium_rate(1.0, 100.0)
    /// ...          .with_sliding_scale([(0.3, 0.5), (0.1, 1.5)])
    /// ...          .with_scales_before_annual_terms())
    /// >>> res = Tower([cover]).apply_annual([140.0])
    /// >>> [res.marginal([k, "cover"]).draws[0] for k in ["ceded", "loss_before_terms"]]
    /// [120.0, 140.0]
    /// >>> round(res.marginal(["ceding_commission", "cover"]).draws[0], 12)
    /// 12.0
    fn with_scales_before_annual_terms(&self) -> Self {
        Self {
            inner: self.inner.clone().scales_before_annual_terms(),
        }
    }

    /// The same layer with its brokerage scale reading the loss ratio held
    /// inside the sliding-scale commission's range, from its first anchor
    /// to its last.
    ///
    /// Returns
    /// -------
    /// Layer
    fn with_brokerage_within_commission_range(&self) -> Self {
        Self {
            inner: self.inner.clone().brokerage_within_commission_range(),
        }
    }

    /// The flat brokerage rate, or ``None`` (also when it is on a scale).
    #[getter]
    fn brokerage(&self) -> Option<f64> {
        match self.inner.brokerage {
            Some(Commission::Flat(b)) => Some(b),
            _ => None,
        }
    }

    /// The brokerage scale's ``(rate, loss_ratio)`` anchors in increasing
    /// loss ratio, or ``None``.
    #[getter]
    fn brokerage_scale(&self) -> Option<Vec<(f64, f64)>> {
        match &self.inner.brokerage {
            Some(Commission::SlidingScale(a)) => Some(a.clone()),
            _ => None,
        }
    }

    /// The deductions as ``(label, basis, value, inside_nmbf)``.
    #[getter]
    fn deductions(&self) -> Vec<(String, &'static str, f64, bool)> {
        self.inner
            .deductions
            .iter()
            .map(|d| (d.label.clone(), d.basis.name(), d.value, d.inside_nmbf))
            .collect()
    }

    /// Whether the sliding scales read the loss ratio before annual terms.
    #[getter]
    fn scales_before_annual_terms(&self) -> bool {
        self.inner.scales_before_annual_terms
    }

    /// Whether the brokerage scale is held inside the commission's range.
    #[getter]
    fn brokerage_within_commission_range(&self) -> bool {
        self.inner.brokerage_within_commission_range
    }

    /// The year's brokerage given its ceded loss.
    ///
    /// Parameters
    /// ----------
    /// ceded : float
    ///
    /// Returns
    /// -------
    /// float
    fn brokerage_for(&self, ceded: f64) -> f64 {
        self.inner.brokerage_for(ceded)
    }

    /// The year's amount of the deduction labelled ``label`` given its
    /// ceded loss.
    ///
    /// Parameters
    /// ----------
    /// label : str
    /// ceded : float
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the layer has no deduction labelled ``label``.
    fn deduction_for(&self, label: &str, ceded: f64) -> PyResult<f64> {
        let d = self
            .inner
            .deductions
            .iter()
            .find(|d| d.label == label)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "layer {:?} has no deduction labelled {label:?}",
                    self.inner.name
                ))
            })?;
        Ok(self.inner.deduction_for(d, ceded))
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
    pub(crate) inner: LossSensitivePremium,
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
    pub(crate) inner: Tower,
}

impl PyTower {
    /// The deals' names, in the order they were added.
    fn deal_names(&self) -> Vec<String> {
        self.inner.deals.iter().map(|d| d.name.clone()).collect()
    }
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

    /// The same tower with a deal's scales read on the loss ratio of
    /// several layers together.
    ///
    /// The deal's commission and brokerage read the sum of its sections'
    /// keys (each section's ceded loss, or its loss before annual terms
    /// with ``with_scales_before_annual_terms``) over the sum of their
    /// premiums, are paid on that summed premium, and add to whatever each
    /// section pays on its own. Results gain ``("ceding_commission",
    /// name)``, ``("brokerage", name)`` and, when a section reads its
    /// scales before annual terms, ``("loss_before_terms", name)``;
    /// ``deal_ratios`` takes the deal's name. ``on_annual_grid`` gives each
    /// deal's expected commission and brokerage, ``on_grid`` too when no
    /// section has annual terms, and the cedant's P&L ledger books the
    /// deal's commission.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Not a layer's or another deal's name.
    /// sections : list of str
    ///     At least two layers, each with a positive fixed premium (no
    ///     swing rating or paid reinstatements) and in no other deal.
    /// sliding_scale : list of (float, float), optional
    ///     ``(commission, loss_ratio)`` anchors; the commission must not
    ///     rise with the loss ratio.
    /// brokerage : float, optional
    ///     A flat brokerage rate on the deal's premium.
    /// brokerage_scale : list of (float, float), optional
    ///     ``(rate, loss_ratio)`` anchors instead of a flat rate.
    /// carry_deficit, carry_credit : bool, default False
    ///     Carry the deal commission's deficits or credits between the
    ///     years of a term, on the deal's key.
    /// carry_years : int, optional
    ///     How many later years a carried amount lasts; ``None`` without
    ///     limit.
    /// brokerage_within_commission_range : bool, default False
    ///     Hold the brokerage key inside the deal commission's range.
    ///
    /// Returns
    /// -------
    /// Tower
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For a deal with no scale, a taken name, an unknown, repeated or
    ///     unpriced section, or a section of another deal.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> a = Layer.stop_loss("A", 50.0, 0.0).with_premium_rate(1.0, 100.0)
    /// >>> b = Layer.stop_loss("B", float("inf"), 50.0).with_premium_rate(1.0, 100.0)
    /// >>> tower = Tower([a, b]).with_deal_scale(
    /// ...     "deal", ["A", "B"], sliding_scale=[(0.35, 0.5), (0.15, 0.9)])
    /// >>> result = tower.apply_annual([160.0])
    /// >>> round(result.marginal(["ceding_commission", "deal"]).draws[0], 12)
    /// 40.0
    #[pyo3(signature = (
        name,
        sections,
        sliding_scale = None,
        brokerage = None,
        brokerage_scale = None,
        carry_deficit = false,
        carry_credit = false,
        carry_years = None,
        brokerage_within_commission_range = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_deal_scale(
        &self,
        name: &str,
        sections: Vec<String>,
        sliding_scale: Option<Vec<(f64, f64)>>,
        brokerage: Option<f64>,
        brokerage_scale: Option<Vec<(f64, f64)>>,
        carry_deficit: bool,
        carry_credit: bool,
        carry_years: Option<u32>,
        brokerage_within_commission_range: bool,
    ) -> PyResult<Self> {
        if brokerage.is_some() && brokerage_scale.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give brokerage or brokerage_scale, not both",
            ));
        }
        let mut deal = DealScale::new(name, sections).map_err(to_py)?;
        if let Some(anchors) = sliding_scale {
            deal = deal.sliding_scale(anchors).map_err(to_py)?;
        }
        if carry_deficit || carry_credit || carry_years.is_some() {
            deal = deal
                .carry_forward(carry_deficit, carry_credit, carry_years)
                .map_err(to_py)?;
        }
        if let Some(rate) = brokerage {
            deal = deal.brokerage(rate).map_err(to_py)?;
        }
        if let Some(anchors) = brokerage_scale {
            deal = deal.brokerage_scale(anchors).map_err(to_py)?;
        }
        if brokerage_within_commission_range {
            deal = deal.brokerage_within_commission_range();
        }
        let inner = self.inner.clone().deal_scale(deal).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The tower with its cover basis, which ``apply_timeline`` needs:
    /// ``"losses_occurring"`` (contract years by accident period) or
    /// ``"risks_attaching"`` (by underwriting period). A tower is one
    /// programme, so the basis is one for all its layers.
    ///
    /// Parameters
    /// ----------
    /// basis : {"losses_occurring", "risks_attaching"}
    ///
    /// Returns
    /// -------
    /// Tower
    fn with_cover_basis(&self, basis: &str) -> PyResult<Self> {
        let basis = crate::aggregate::cover_basis(basis)?;
        Ok(Self {
            inner: self.inner.clone().cover_basis(basis),
        })
    }

    /// The cover basis's name, or ``None`` until one is set.
    #[getter(cover_basis)]
    fn get_cover_basis(&self) -> Option<&'static str> {
        self.inner
            .cover_basis
            .map(crate::aggregate::cover_basis_name)
    }

    /// The tower with its contracts interlocking or not; off by default.
    ///
    /// Under risks attaching an event that several underwriting years
    /// write falls to each by its share. With interlocking each contract's
    /// limit is cut, for that event, by the year's share of the event's
    /// loss across the programme's underwriting years in the term; the
    /// retention is not cut, and each year's annual terms apply to its
    /// recoveries. Under losses occurring it changes nothing.
    ///
    /// Parameters
    /// ----------
    /// on : bool, default True
    ///
    /// Returns
    /// -------
    /// Tower
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet, Exposure, Timeline
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> events = EventSet.from_years([[100.0]], times=[[13.5 / 24]])
    /// >>> shares = [[0.0] * 24, [0.0] * 24]
    /// >>> shares[0][13], shares[1][13] = 0.25, 0.75
    /// >>> tower = (Tower([Layer("XL", 50.0, 20.0)]).with_cover_basis("risks_attaching")
    /// ...          .with_interlocking())
    /// >>> rad = tower.apply_timeline(Timeline(events, 24), Exposure(shares))
    /// >>> [rad.select("year", [y]).select("kind", ["ceded"]).mean() for y in (1, 2)]
    /// [5.0, 37.5]
    #[pyo3(signature = (on = true))]
    fn with_interlocking(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().interlocking(on),
        }
    }

    /// Whether the tower's contracts interlock.
    #[getter(interlocking)]
    fn get_interlocking(&self) -> bool {
        self.inner.interlocking
    }

    /// Each deal's name and sections, in the order they were added.
    #[getter]
    fn deals(&self) -> Vec<(String, Vec<String>)> {
        self.inner
            .deals
            .iter()
            .map(|d| (d.name.clone(), d.sections.clone()))
            .collect()
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
    /// and net; net is a loss, before premiums and commissions. Take one kind
    /// with ``select("kind", ["ceded"])`` before joining it to a book:
    /// summing over ``kind`` adds gross, ceded and net together.
    ///
    /// With labels on the events (``EventSet.with_labels``) the label
    /// dimensions follow ``kind`` and ``layer``: gross, each layer's ceded
    /// loss and net come once per combination in
    /// ``events.label_combinations``, each the sum over that combination's
    /// events (a layer's split by event under its ``allocation``), and the
    /// other components get ``""`` labels. ``aggregate(["kind", "layer"])``
    /// gives back the unlabelled result.
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

    /// Each event's ceded loss to the named layers, summed over them, as an
    /// event set: the input to an occurrence retro on a reinsurer's own
    /// cessions.
    ///
    /// Each layer's split by event is the one ``apply``'s labels use
    /// (``Layer.ceded_by_event``: after the layer's annual terms, at its
    /// share, behind any earlier inuring stage), so in every year the
    /// set's losses add to the layers' ceded total. The events keep their
    /// times and labels; events that cede nothing are left out, sums
    /// insured are dropped, and a named layer that sees the aggregate part
    /// passes its ceded part on as the set's aggregate part. Join several
    /// deals' sets with ``EventSet.merge``, and group one event's cessions
    /// into one occurrence with ``EventSet.grouped``.
    ///
    /// Parameters
    /// ----------
    /// events : EventSet
    /// layers : str or list of str
    ///     Layer names of this tower.
    ///
    /// Returns
    /// -------
    /// EventSet
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> e = EventSet.from_years([[20.0, 8.0, 15.0], [4.0]])
    /// >>> t = Tower([Layer("10x5", 10.0, 5.0, aggregate_limit=15.0)])
    /// >>> c = t.ceded_events(e, "10x5")
    /// >>> c.events(0), c.events(1)
    /// ([10.0, 3.0, 2.0], [])
    fn ceded_events(
        &self,
        py: Python<'_>,
        events: PyRef<'_, PyEventSet>,
        layers: &Bound<'_, PyAny>,
    ) -> PyResult<PyEventSet> {
        let layers: Vec<String> = match layers.extract::<String>() {
            Ok(l) => vec![l],
            Err(_) => layers.extract()?,
        };
        let names: Vec<&str> = layers.iter().map(String::as_str).collect();
        let (tower, events) = (&self.inner, &events.inner);
        let inner = py
            .detach(|| tower.ceded_events(events, &names))
            .map_err(to_py)?;
        Ok(PyEventSet { inner })
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

    /// Applies the tower to plain annual losses, one per simulation, each
    /// taken as one aggregate loss as in ``apply_aggregate``, with no
    /// ``EventSet`` or ``PredictiveDistribution`` to build. A contract
    /// stated on the loss ratio is a layer on money amounts: a loss-ratio
    /// cap ``c`` on subject premium ``P`` is ``Layer.stop_loss(name, c * P,
    /// 0.0).with_premium_rate(1.0, P)``, so the ceded loss ratio is the
    /// capped one, and a sliding scale keyed on it is
    /// ``with_sliding_scale``. To key the scales on the loss ratio before
    /// the cap, state the cap as ``aggregate_limit`` and use
    /// ``with_scales_before_annual_terms``. A deal of several sections is
    /// one layer per section, each scale reading its own section's loss
    /// ratio, and the amounts add; ``Layer.ceding_commission_for``
    /// evaluates a scale at any ceded loss. Components as ``apply``.
    ///
    /// Parameters
    /// ----------
    /// losses : Sequence[float]
    ///     Each simulation's annual loss, finite and non-negative.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For no losses, a negative or non-finite loss, a surplus treaty or
    ///     reinstatements pro rata as to time.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> cover = (Layer.stop_loss("cover", 120.0, 0.0)
    /// ...          .with_premium_rate(1.0, 100.0)
    /// ...          .with_sliding_scale([(0.3, 0.5), (0.1, 0.9)]))
    /// >>> res = Tower([cover]).apply_annual([40.0, 70.0, 150.0])
    /// >>> list(res.marginal(["ceded", "cover"]).draws)
    /// [40.0, 70.0, 120.0]
    /// >>> [round(x, 12) for x in res.marginal(["ceding_commission", "cover"]).draws]
    /// [30.0, 20.0, 10.0]
    fn apply_annual(&self, py: Python<'_>, losses: Vec<f64>) -> PyResult<PyPredictiveDistribution> {
        let tower = &self.inner;
        let inner = py.detach(|| tower.apply_annual(&losses)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// The deal ratios of one layer or one deal (``with_deal_scale``) in
    /// each simulation of a result of this tower (``apply``,
    /// ``apply_aggregate``, ``apply_annual`` or ``apply_term``).
    ///
    /// Every amount is over the year's premium for the placed share (the
    /// fixed or the swing-rated premium; reinstatement premiums are not in
    /// it). With ``d`` the deductions: ``total_deductions = CC + PC + BK +
    /// sum(d)``, ``combined_ratio = loss_ratio + total_deductions``,
    /// ``margin = 1 - combined_ratio`` and ``nmbf = margin`` plus the
    /// deductions kept outside it. For a deal, every amount is summed over
    /// its sections with the deal's own commission and brokerage added,
    /// over the sections' premium together; a section's own ratios leave
    /// the deal's scales out.
    ///
    /// Parameters
    /// ----------
    /// result : PredictiveDistribution
    ///     A result of this tower.
    /// layer : str
    ///     The layer's or the deal's name.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///     Dimension ``["ratio"]`` (``["ratio", "year"]`` for a term):
    ///     ``loss_ratio``, ``loss_ratio_before_terms`` for a layer whose
    ///     scales read it, ``ceding_commission``, ``profit_commission`` and
    ///     ``brokerage`` where the layer has them, each deduction by its
    ///     label, ``total_deductions``, ``combined_ratio``, ``nmbf`` and
    ///     ``margin``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For an unknown layer or deal, a result that is not this tower's,
    ///     or a premium that is not positive.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> cover = (Layer.stop_loss("cover", 120.0, 0.0)
    /// ...          .with_premium_rate(1.0, 100.0)
    /// ...          .with_sliding_scale([(0.3, 0.5), (0.1, 0.9)])
    /// ...          .with_brokerage(0.025)
    /// ...          .with_deduction("tax", 0.03)
    /// ...          .with_deduction("manager_fee", 0.05, inside_nmbf=False))
    /// >>> tower = Tower([cover])
    /// >>> r = tower.deal_ratios(tower.apply_annual([40.0, 70.0, 150.0]), "cover")
    /// >>> [round(r.marginal([k]).draws[1], 12) for k in ["combined_ratio", "margin", "nmbf"]]
    /// [1.005, -0.005, 0.045]
    fn deal_ratios(
        &self,
        py: Python<'_>,
        result: PyRef<'_, PyPredictiveDistribution>,
        layer: &str,
    ) -> PyResult<PyPredictiveDistribution> {
        let (tower, pd) = (&self.inner, &result.inner);
        let inner = py.detach(|| tower.deal_ratios(pd, layer)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// The reinsurer's P&L ledger of one layer or one deal
    /// (:meth:`with_deal_scale`) over the annual gross losses ``source``:
    /// one group that sells the cover, named ``layer``, with the premium
    /// received against the ceded loss, the commissions paid, the brokerage
    /// and each deduction, every leg at each atom of the source.
    ///
    /// The tower is applied to each atom as one annual loss
    /// (:meth:`apply_annual`), so the amounts are those :meth:`deal_ratios`
    /// divides by the premium. A deal books its sections' amounts with its
    /// own. :meth:`prospicio.aggregate.Book.ledger` books the same tower as
    /// covers bought and leaves brokerage and deductions out.
    ///
    /// The rows are ``Premium``, ``Loss``, ``Ceding commission``, ``Profit
    /// commission``, ``Brokerage`` and each deduction by its label, as the
    /// layer has them. NMBF is not a row: :meth:`deal_ratios` gives it.
    ///
    /// Parameters
    /// ----------
    /// source : Grid or tuple of (list of float, list of float)
    ///     The annual gross loss.
    /// layer : str
    ///     A layer or a deal.
    ///
    /// Returns
    /// -------
    /// prospicio.aggregate.Ledger
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A joint source, a layer that cannot apply to an annual loss, a
    ///     booked layer with paid reinstatements, or an unknown layer or
    ///     deal.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> cover = (Layer.stop_loss("cover", 120.0, 0.0)
    /// ...          .with_premium_rate(1.0, 100.0)
    /// ...          .with_ceding_commission(0.2)
    /// ...          .with_brokerage(0.025)
    /// ...          .with_deduction("tax", 0.03))
    /// >>> ledger = Tower([cover]).deal_ledger(([40.0, 150.0], [0.5, 0.5]), "cover")
    /// >>> ledger.result.values
    /// [-45.5, 34.5]
    fn deal_ledger(
        &self,
        source: &Bound<'_, PyAny>,
        layer: &str,
    ) -> PyResult<crate::pnl::PyLedger> {
        let source = crate::pnl::source_of(source)?;
        Ok(crate::pnl::PyLedger {
            inner: self.inner.deal_ledger(&source, layer).map_err(to_py)?,
        })
    }

    /// Applies the tower over a contract term: ``years[t]`` holds year
    /// ``t + 1``'s simulated losses, and simulation ``j`` of every year
    /// belongs to the same path.
    ///
    /// The result has dimensions ``["kind", "layer", "year"]``: each of
    /// ``apply``'s components once per year, keyed by the year ``1..T``.
    /// Each year's ceded and net loss and premiums are what ``apply`` gives
    /// for that year alone; a profit commission or sliding scale that
    /// carries deficits or credits forward reads what the path's earlier
    /// years left. ``EventSet.term_years`` splits one simulation into the
    /// years of a term.
    ///
    /// Parameters
    /// ----------
    /// years : list of EventSet
    ///     One per year, each with the same number of simulations.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> qs = (Layer.quota_share("QS", 1.0).with_premium_rate(1.0, 100.0)
    /// ...       .with_profit_commission(0.25, 0.1)
    /// ...       .with_profit_commission_carry_forward())
    /// >>> years = [EventSet.from_years([[120.0], [40.0]]), EventSet.from_years([[40.0], [40.0]])]
    /// >>> term = Tower([qs]).apply_term(years)
    /// >>> term.select("year", [2]).select("kind", ["profit_commission"]).mean()
    /// 8.75
    fn apply_term(
        &self,
        py: Python<'_>,
        years: Vec<PyRef<'_, PyEventSet>>,
    ) -> PyResult<PyPredictiveDistribution> {
        let years: Vec<EventSet> = years.iter().map(|e| e.inner.clone()).collect();
        let tower = &self.inner;
        let inner = py.detach(|| tower.apply_term(&years)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// ``apply_term`` on the contract years a timeline splits into under
    /// the tower's cover basis (``with_cover_basis``; an error without one).
    ///
    /// Parameters
    /// ----------
    /// timeline : Timeline
    /// exposure : Exposure
    ///     Each underwriting year's share of each accident month; under
    ///     losses occurring only its number of years is used.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.aggregate import EventSet, Exposure, Timeline
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> events = EventSet.from_years([[100.0]], times=[[13.5 / 24]])
    /// >>> timeline = Timeline(events, 24)
    /// >>> shares = [[0.0] * 24, [0.0] * 24]
    /// >>> shares[0][13], shares[1][13] = 0.25, 0.75
    /// >>> tower = Tower([Layer("XL", 50.0, 20.0)]).with_cover_basis("risks_attaching")
    /// >>> rad = tower.apply_timeline(timeline, Exposure(shares))
    /// >>> [rad.select("year", [y]).select("kind", ["ceded"]).mean() for y in (1, 2)]
    /// [5.0, 50.0]
    fn apply_timeline(
        &self,
        py: Python<'_>,
        timeline: PyRef<'_, PyTimeline>,
        exposure: PyRef<'_, PyExposure>,
    ) -> PyResult<PyPredictiveDistribution> {
        let (tower, timeline, exposure) = (&self.inner, &timeline.inner, &exposure.inner);
        let inner = py
            .detach(|| tower.apply_timeline(timeline, exposure))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// ``apply_term`` on aggregate losses: year ``t`` of path ``j`` is
    /// simulation ``j``'s total of ``years[t]``, as in ``apply_aggregate``.
    ///
    /// Parameters
    /// ----------
    /// years : list of PredictiveDistribution
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn apply_aggregate_term(
        &self,
        py: Python<'_>,
        years: Vec<PyRef<'_, PyPredictiveDistribution>>,
    ) -> PyResult<PyPredictiveDistribution> {
        let years: Vec<PredictiveDistribution> = years.iter().map(|e| e.inner.clone()).collect();
        let tower = &self.inner;
        let inner = py
            .detach(|| tower.apply_aggregate_term(&years))
            .map_err(to_py)?;
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
        Ok(PyTowerGrids {
            inner,
            deal_names: self.deal_names(),
        })
    }

    /// The tower applied to an annual loss on a grid, each year's total
    /// treated as one loss (as ``apply_aggregate`` does for draws): a quota
    /// share or stop-loss on a loss-ratio or annual-loss distribution, with
    /// its expected sliding-scale and profit commission. Inuring stages and
    /// annual terms are all allowed, and the net always exists.
    ///
    /// Parameters
    /// ----------
    /// annual : Grid
    ///     The annual loss.
    ///
    /// Returns
    /// -------
    /// AnnualGrids
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For a surplus treaty or reinstatements pro rata as to time, which
    ///     need risks or event times.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> from prospicio.distributions import Grid
    /// >>> probs = [0.0] * 81
    /// >>> for k in (50, 60, 70, 80):
    /// ...     probs[k] = 0.25
    /// >>> qs = Layer.quota_share("QS", 0.4).with_deposit_premium(100.0).with_ceding_commission(0.3)
    /// >>> r = Tower([qs]).on_annual_grid(Grid(1.0, probs))
    /// >>> round(r.ceded[0].mean(), 12), round(r.expected_ceding_commission[0], 12)
    /// (26.0, 12.0)
    fn on_annual_grid(&self, annual: PyRef<'_, PyGrid>) -> PyResult<PyAnnualGrids> {
        Ok(PyAnnualGrids {
            inner: self.inner.on_annual_grid(&annual.inner).map_err(to_py)?,
            deal_names: self.deal_names(),
        })
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
    pub(crate) inner: TowerGrids,
    deal_names: Vec<String>,
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

    /// Annual ceded loss of each stage, its layers together: the total
    /// cession ``aggregate`` reports as ``Ceded``. ``None`` for a stage of
    /// several layers one of which has annual terms.
    #[getter]
    fn stage_ceded(&self) -> Vec<Option<PyGrid>> {
        self.inner
            .stage_ceded
            .iter()
            .map(|g| g.clone().map(|inner| PyGrid { inner }))
            .collect()
    }

    /// Annual net loss, or ``None`` when it is not a single compound total.
    #[getter]
    fn net(&self) -> Option<PyGrid> {
        self.inner.net.clone().map(|inner| PyGrid { inner })
    }

    /// Annual ceded loss of the whole tower, every layer together, or
    /// ``None`` when it is not a single compound total: several layers
    /// with annual terms (reinstatements, aggregate limits or deductibles),
    /// or an aggregate cover after per-occurrence stages.
    #[getter]
    fn total_ceded(&self) -> Option<PyGrid> {
        self.inner.total_ceded.clone().map(|inner| PyGrid { inner })
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

    /// Expected ceding commission of each deal (``Tower.deal_scale``), by
    /// its name, read on its sections' loss ratio together; 0 for a deal
    /// without one.
    #[getter]
    fn expected_deal_ceding_commission(&self) -> HashMap<String, f64> {
        self.deal_names
            .iter()
            .cloned()
            .zip(self.inner.expected_deal_ceding_commission.iter().copied())
            .collect()
    }

    /// Expected brokerage of each deal, by its name; 0 without one.
    #[getter]
    fn expected_deal_brokerage(&self) -> HashMap<String, f64> {
        self.deal_names
            .iter()
            .cloned()
            .zip(self.inner.expected_deal_brokerage.iter().copied())
            .collect()
    }

    /// Each view priced on its own by a distortion: the stand-alone price
    /// of the gross, of every layer's cession and of the net, as
    /// ``aggregate``'s ``reins_price_df``.
    ///
    /// Parameters
    /// ----------
    /// distortion : Distortion
    ///
    /// Returns
    /// -------
    /// TowerPrice
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> from prospicio.distributions import Grid, Poisson
    /// >>> from prospicio.risk import Distortion
    /// >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
    /// >>> r = Tower([Layer("2x2", 2.0, 2.0)]).on_grid(Poisson(3.0), sev, 200)
    /// >>> p = r.price(Distortion.proportional_hazard(0.7))
    /// >>> round(p.ceded[0].loss, 12), p.ceded[0].premium > p.ceded[0].loss
    /// (1.2, True)
    fn price(&self, distortion: PyRef<'_, PyDistortion>) -> PyTowerPrice {
        PyTowerPrice {
            inner: self.inner.price(&distortion.inner),
        }
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

/// One view's price under a distortion, from ``TowerGrids.price``: the
/// expected loss ``loss``, the premium, the margin ``premium - loss`` and
/// the loss ratio ``loss / premium``.
#[pyclass(name = "ViewPrice", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyViewPrice {
    inner: ViewPrice,
}

#[pymethods]
impl PyViewPrice {
    /// The expected loss.
    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }

    /// The distorted expectation.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// ``premium - loss``.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin
    }

    /// ``loss / premium``.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio
    }

    fn __repr__(&self) -> String {
        format!(
            "ViewPrice(loss={}, premium={})",
            self.inner.loss, self.inner.premium
        )
    }
}

/// A tower's views priced by a distortion, from ``TowerGrids.price``.
#[pyclass(name = "TowerPrice", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyTowerPrice {
    inner: TowerPrice,
}

#[pymethods]
impl PyTowerPrice {
    /// The gross.
    #[getter]
    fn gross(&self) -> PyViewPrice {
        PyViewPrice {
            inner: self.inner.gross,
        }
    }

    /// Each layer's cession, in tower order.
    #[getter]
    fn ceded(&self) -> Vec<PyViewPrice> {
        self.inner
            .ceded
            .iter()
            .map(|&inner| PyViewPrice { inner })
            .collect()
    }

    /// Each stage's total cession, ``None`` where ``stage_ceded`` is.
    #[getter]
    fn stages(&self) -> Vec<Option<PyViewPrice>> {
        self.inner
            .stages
            .iter()
            .map(|v| v.map(|inner| PyViewPrice { inner }))
            .collect()
    }

    /// The net, or ``None`` when the tower has no net grid.
    #[getter]
    fn net(&self) -> Option<PyViewPrice> {
        self.inner.net.map(|inner| PyViewPrice { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "TowerPrice(gross={}, layers={}, net={})",
            self.inner.gross.premium,
            self.inner.ceded.len(),
            self.inner.net.is_some()
        )
    }
}

/// A per-occurrence layer's statistics, from
/// ``occurrence_layer_statistics``.
#[pyclass(name = "LayerStatistics", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyLayerStatistics {
    inner: LayerStatistics,
}

#[pymethods]
impl PyLayerStatistics {
    /// The placed share.
    #[getter]
    fn share(&self) -> f64 {
        self.inner.share
    }

    /// The per-occurrence limit.
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit
    }

    /// The attachment.
    #[getter]
    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    /// ``P(X > attachment)``, per claim.
    #[getter]
    fn pr_attach(&self) -> f64 {
        self.inner.pr_attach
    }

    /// ``P(X >= attachment + limit)``, per claim.
    #[getter]
    fn pr_detach(&self) -> f64 {
        self.inner.pr_detach
    }

    /// The chance of a loss to the layer in the year.
    #[getter]
    fn pr_loss(&self) -> f64 {
        self.inner.pr_loss
    }

    /// The annual ceded mean over ``share * limit``.
    #[getter]
    fn loss_on_line(&self) -> f64 {
        self.inner.loss_on_line
    }

    /// The number of claims that reach the layer.
    #[getter]
    fn count(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.count,
        }
    }

    /// The ceded loss of a claim that reaches the layer, at the share.
    #[getter]
    fn severity(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.severity,
        }
    }

    /// The annual ceded loss, at the share.
    #[getter]
    fn aggregate(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.aggregate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerStatistics({} x {} xs {}, pr_loss={}, loss_on_line={})",
            self.inner.share,
            self.inner.limit,
            self.inner.attachment,
            self.inner.pr_loss,
            self.inner.loss_on_line
        )
    }
}

/// The statistics of ``share`` of ``limit`` xs ``attachment`` per
/// occurrence on a severity grid, as ``aggregate``'s ``reins_stats_df``.
///
/// The chances that a claim reaches the layer (``P(X > a)``, exclusive)
/// and exhausts it (``P(X >= a + l)``, inclusive), and that the year has a
/// loss to it; the excess count (the count thinned to the claims that
/// reach the layer), the layer severity given a loss to it and the annual
/// ceded loss, each by its first three cumulants; and the loss on line.
/// The share scales the moments exactly.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Grid
/// share : float
///     In ``(0, 1]``.
/// limit : float
///     May be ``inf``; the loss on line is then 0.
/// attachment : float
///
/// Returns
/// -------
/// LayerStatistics
///
/// Raises
/// ------
/// ValueError
///     For a share outside ``(0, 1]``, a non-positive limit or a negative
///     attachment.
///
/// Examples
/// --------
/// >>> from prospicio.reinsurance import occurrence_layer_statistics
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
/// >>> s = occurrence_layer_statistics(Poisson(3.0), sev, 1.0, 2.0, 2.0)
/// >>> round(s.pr_attach, 12), round(s.count.mean, 12), round(s.aggregate.mean, 12)
/// (0.3, 0.9, 1.2)
#[pyfunction]
pub(crate) fn occurrence_layer_statistics(
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyGrid>,
    share: f64,
    limit: f64,
    attachment: f64,
) -> PyResult<PyLayerStatistics> {
    let n = AnyCount::extract(frequency)?;
    let inner = prospicio_aggregate::occurrence_layer_statistics(
        n.as_counting(),
        &severity.inner,
        share,
        limit,
        attachment,
    )
    .map_err(to_py)?;
    Ok(PyLayerStatistics { inner })
}

/// The total cession of several per-occurrence layers, from
/// ``occurrence_cession_statistics``.
#[pyclass(name = "CessionStatistics", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyCessionStatistics {
    inner: prospicio_aggregate::CessionStatistics,
}

#[pymethods]
impl PyCessionStatistics {
    /// ``P(X > a)`` at the lowest attachment, per claim.
    #[getter]
    fn pr_attach(&self) -> f64 {
        self.inner.pr_attach
    }

    /// The chance of a loss to the cession in the year.
    #[getter]
    fn pr_loss(&self) -> f64 {
        self.inner.pr_loss
    }

    /// The annual ceded mean over the placed limits; 0 when one is
    /// unlimited.
    #[getter]
    fn loss_on_line(&self) -> f64 {
        self.inner.loss_on_line
    }

    /// The ceded loss of a claim, every claim counted.
    #[getter]
    fn severity(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.severity,
        }
    }

    /// The annual ceded loss.
    #[getter]
    fn aggregate(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.aggregate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CessionStatistics(pr_loss={}, loss_on_line={})",
            self.inner.pr_loss, self.inner.loss_on_line
        )
    }
}

/// The statistics of the total cession of several per-occurrence layers
/// side by side, as ``aggregate``'s ``Ceded`` column of
/// ``reins_stats_df``: a claim cedes the sum of ``share * min(max(x - a,
/// 0), limit)`` over the layers. Moments are exact for the grid.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Grid
/// layers : list of (share, limit, attachment)
///
/// Returns
/// -------
/// CessionStatistics
///
/// Examples
/// --------
/// >>> from prospicio.reinsurance import occurrence_cession_statistics
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
/// >>> c = occurrence_cession_statistics(Poisson(3.0), sev, [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0)])
/// >>> round(c.aggregate.mean, 12)
/// 3.0
#[pyfunction]
pub(crate) fn occurrence_cession_statistics(
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyGrid>,
    layers: Vec<(f64, f64, f64)>,
) -> PyResult<PyCessionStatistics> {
    let n = AnyCount::extract(frequency)?;
    let inner = prospicio_aggregate::occurrence_cession_statistics(
        n.as_counting(),
        &severity.inner,
        &layers,
    )
    .map_err(to_py)?;
    Ok(PyCessionStatistics { inner })
}

/// An aggregate cover's statistics, from ``aggregate_layer_statistics``.
#[pyclass(
    name = "AggregateLayerStatistics",
    module = "prospicio.reinsurance",
    frozen
)]
pub(crate) struct PyAggregateLayerStatistics {
    inner: prospicio_aggregate::AggregateLayerStatistics,
}

#[pymethods]
impl PyAggregateLayerStatistics {
    /// The placed share.
    #[getter]
    fn share(&self) -> f64 {
        self.inner.share
    }

    /// The aggregate limit.
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit
    }

    /// The aggregate retention.
    #[getter]
    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    /// ``P(S > attachment)``: the chance of a loss to the cover.
    #[getter]
    fn pr_attach(&self) -> f64 {
        self.inner.pr_attach
    }

    /// ``P(S >= attachment + limit)``; 0 for an unlimited cover.
    #[getter]
    fn pr_detach(&self) -> f64 {
        self.inner.pr_detach
    }

    /// The ceded mean over ``share * limit``; 0 for an unlimited cover.
    #[getter]
    fn loss_on_line(&self) -> f64 {
        self.inner.loss_on_line
    }

    /// The annual ceded loss, at the share.
    #[getter]
    fn aggregate(&self) -> PyCompoundMoments {
        PyCompoundMoments {
            inner: self.inner.aggregate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AggregateLayerStatistics({} x {} xs {}, pr_attach={}, loss_on_line={})",
            self.inner.share,
            self.inner.limit,
            self.inner.attachment,
            self.inner.pr_attach,
            self.inner.loss_on_line
        )
    }
}

/// The statistics of ``share`` of ``limit`` xs ``attachment`` on an annual
/// total, an aggregate cover such as a stop loss, as ``aggregate``'s
/// ``reins_stats_df`` for its ``agg`` layers: the chances of reaching and
/// exhausting the cover, the loss on line and the ceded loss's first three
/// cumulants, exact for the grid.
///
/// Parameters
/// ----------
/// annual : Grid
///     The annual total the cover sees: a compound grid, or a tower's net.
/// share : float
/// limit : float
///     May be ``inf``.
/// attachment : float
///
/// Returns
/// -------
/// AggregateLayerStatistics
///
/// Examples
/// --------
/// >>> from prospicio.reinsurance import aggregate_layer_statistics
/// >>> from prospicio.distributions import Grid
/// >>> s = aggregate_layer_statistics(Grid(1.0, [0.2, 0.3, 0.3, 0.2]), 1.0, 1.0, 1.0)
/// >>> round(s.pr_attach, 12), round(s.aggregate.mean, 12)
/// (0.5, 0.5)
#[pyfunction]
pub(crate) fn aggregate_layer_statistics(
    annual: PyRef<'_, PyGrid>,
    share: f64,
    limit: f64,
    attachment: f64,
) -> PyResult<PyAggregateLayerStatistics> {
    let inner =
        prospicio_aggregate::aggregate_layer_statistics(&annual.inner, share, limit, attachment)
            .map_err(to_py)?;
    Ok(PyAggregateLayerStatistics { inner })
}

/// A tower applied to an annual loss distribution, from
/// ``Tower.on_annual_grid``. Ceded grids are at the placed share and after
/// annual terms; a layer with share ``c`` has step ``c * h``.
#[pyclass(name = "AnnualGrids", module = "prospicio.reinsurance", frozen)]
pub(crate) struct PyAnnualGrids {
    inner: AnnualGrids,
    deal_names: Vec<String>,
}

#[pymethods]
impl PyAnnualGrids {
    /// The annual loss given.
    #[getter]
    fn gross(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.gross.clone(),
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

    /// Annual net loss.
    #[getter]
    fn net(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.net.clone(),
        }
    }

    /// Expected reinstatement premium of each layer.
    #[getter]
    fn expected_reinstatement_premium(&self) -> Vec<f64> {
        self.inner.expected_reinstatement_premium.clone()
    }

    /// Expected premium of each layer: the swing-rated premium's mean, or
    /// else the fixed premium.
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

    /// Expected ceding commission of each deal (``Tower.deal_scale``), by
    /// its name, read on its sections' loss ratio together; 0 for a deal
    /// without one.
    #[getter]
    fn expected_deal_ceding_commission(&self) -> HashMap<String, f64> {
        self.deal_names
            .iter()
            .cloned()
            .zip(self.inner.expected_deal_ceding_commission.iter().copied())
            .collect()
    }

    /// Expected brokerage of each deal, by its name; 0 without one.
    #[getter]
    fn expected_deal_brokerage(&self) -> HashMap<String, f64> {
        self.deal_names
            .iter()
            .cloned()
            .zip(self.inner.expected_deal_brokerage.iter().copied())
            .collect()
    }

    /// Whether every ceded and net loss fell on a grid point. The
    /// expectations are exact either way.
    #[getter]
    fn on_points(&self) -> bool {
        self.inner.on_points
    }

    fn __repr__(&self) -> String {
        format!(
            "AnnualGrids(layers={}, on_points={})",
            self.inner.ceded.len(),
            self.inner.on_points
        )
    }
}
