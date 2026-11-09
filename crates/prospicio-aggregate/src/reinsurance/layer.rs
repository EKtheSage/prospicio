//! The layer: its terms and how it cedes one year of losses.

use super::*;

/// A per-occurrence excess-of-loss layer: `limit` xs `attachment` on each
/// loss, then annual terms.
///
/// For one year with losses `x_1, …, x_n`:
///
/// ```text
/// recovery  = Σ min(max(x_e - attachment, 0), limit)
/// after AAD = max(recovery - aggregate_deductible, 0)
/// corridor  = after AAD - retained × min(max(after AAD - lower, 0), upper - lower)
/// ceded     = share × min(corridor, aggregate_limit)
/// ```
///
/// where the loss corridor (`lower`, `upper`, `retained`) is optional
/// ([`Layer::loss_corridor`]).
///
/// Terms are plain data, so a tower can be stored and replayed.
///
/// ```
/// use prospicio_aggregate::Layer;
///
/// // 5m xs 5m with one reinstatement: at most 10m a year.
/// let layer = Layer::xol("5x5", 5e6, 5e6).unwrap().reinstatements(1).unwrap();
/// assert_eq!(layer.ceded(&[7e6]), 2e6);
/// assert_eq!(layer.ceded(&[12e6, 20e6, 30e6]), 10e6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    /// What each event's recovery is figured on: the loss, or (a surplus
    /// treaty) a share of it set by the risk's sum insured.
    pub basis: Basis,
    pub attachment: f64,
    /// Per-occurrence limit; may be infinite.
    pub limit: f64,
    /// Share of the layer placed, in `(0, 1]`.
    pub share: f64,
    /// Annual aggregate deductible (AAD), retained before the layer pays.
    pub aggregate_deductible: f64,
    /// Annual aggregate limit (AAL); infinite when unlimited.
    pub aggregate_limit: f64,
    /// Upfront premium for the placed share: the base of paid
    /// reinstatements, ceding and profit commissions, and the loss ratio
    /// of a sliding scale. Set it with [`Layer::deposit_premium`],
    /// [`Layer::rate_on_line`], [`Layer::premium_rate`] or
    /// [`Layer::paid_reinstatements`].
    pub premium: f64,
    /// Premium rate of each paid reinstatement, as a fraction of
    /// `premium` (1.0 is 100%), pro rata as to amount. Empty when
    /// reinstatements are free.
    pub reinstatement_rates: Vec<f64>,
    /// Whether reinstatement premiums are also pro rata as to time: each
    /// event's share is scaled by the part of the year left after it.
    pub pro_rata_time: bool,
    /// How a binding annual deductible or limit is shared among the
    /// year's events; see [`Layer::allocation`].
    pub allocation: LimitAllocation,
    /// A band of the annual layer loss the cedant keeps; see
    /// [`Layer::loss_corridor`].
    pub corridor: Option<Corridor>,
    /// Ceding commission paid back to the cedant on the year's premium:
    /// a flat rate or a sliding scale; see [`Layer::ceding_commission`].
    pub commission: Option<Commission>,
    /// Profit commission; see [`Layer::profit_commission`].
    pub profit_commission: Option<ProfitCommission>,
    /// A no-claims bonus; see [`Layer::no_claims_bonus`].
    pub no_claims_bonus: Option<NoClaimsBonus>,
    /// Deficits of the profit commission carried into later years of a
    /// term; see [`Layer::profit_commission_carry_forward`].
    pub profit_commission_carry: Option<CarryForward>,
    /// Deficits and credits of the sliding scale carried into later years
    /// of a term; see [`Layer::sliding_scale_carry_forward`].
    pub sliding_scale_carry: Option<CarryForward>,
    /// Swing rating at 100% of the layer: the year's premium as a function
    /// of its ceded loss, in place of `premium`; see [`Layer::swing_rated`].
    pub swing: Option<LossSensitivePremium>,
    /// Brokerage on the year's premium: a flat rate or a scale on the
    /// loss ratio; see [`Layer::brokerage`].
    pub brokerage: Option<Commission>,
    /// Other deductions from the premium (a TPA fee, tax, reinsurance
    /// cost, a fronting fee, a manager's fee); see [`Layer::deduction`].
    pub deductions: Vec<Deduction>,
    /// Whether the sliding scales (commission and brokerage) read the loss
    /// ratio before the annual terms; see
    /// [`Layer::scales_before_annual_terms`].
    pub scales_before_annual_terms: bool,
    /// Whether a brokerage scale reads its loss ratio held inside the
    /// sliding-scale commission's range; see
    /// [`Layer::brokerage_within_commission_range`].
    pub brokerage_within_commission_range: bool,
    /// The events the layer sees, by a label's values; see
    /// [`Layer::scope`]. `None` sees every event.
    pub scope: Option<Scope>,
    /// A franchise deductible: an occurrence at or above the attachment
    /// is paid from the ground up, up to the limit; see
    /// [`Layer::franchise`].
    pub franchise: bool,
    /// A disappearing deductible: the retention falls from the attachment
    /// to 0 as the loss rises from the attachment to this amount; see
    /// [`Layer::disappearing`].
    pub disappearing: Option<f64>,
    /// Only the `n`-th and later qualifying events of a year are covered;
    /// see [`Layer::nth_event`].
    pub nth_event: Option<NthEvent>,
    /// The layer drops to a lower attachment once another layer's annual
    /// limit is used up; see [`Layer::top_and_drop`].
    pub drop_down: Option<DropDown>,
}

impl Layer {
    /// `limit` xs `attachment`, fully placed, with no annual terms.
    pub fn xol(name: impl Into<String>, limit: f64, attachment: f64) -> Result<Self> {
        if !attachment.is_finite() || attachment < 0.0 {
            return Err(invalid(
                "attachment",
                attachment,
                "must be finite and non-negative",
            ));
        }
        if limit.is_nan() || limit <= 0.0 {
            return Err(invalid("limit", limit, "must be positive"));
        }
        Ok(Self {
            name: name.into(),
            basis: Basis::Loss,
            attachment,
            limit,
            share: 1.0,
            aggregate_deductible: 0.0,
            aggregate_limit: f64::INFINITY,
            premium: 0.0,
            reinstatement_rates: Vec::new(),
            pro_rata_time: false,
            allocation: LimitAllocation::EventOrder,
            corridor: None,
            commission: None,
            profit_commission: None,
            no_claims_bonus: None,
            profit_commission_carry: None,
            sliding_scale_carry: None,
            swing: None,
            brokerage: None,
            deductions: Vec::new(),
            scales_before_annual_terms: false,
            brokerage_within_commission_range: false,
            scope: None,
            franchise: false,
            disappearing: None,
            nth_event: None,
            drop_down: None,
        })
    }

    /// A quota share ceding `cession` of every loss: unlimited cover from
    /// the first unit, with `share = cession`.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS 40%", 0.4).unwrap();
    /// assert_eq!(qs.ceded(&[10.0, 5.0]), 6.0);
    /// ```
    pub fn quota_share(name: impl Into<String>, cession: f64) -> Result<Self> {
        Self::xol(name, f64::INFINITY, 0.0)?.share(cession)
    }

    /// A surplus treaty: each risk cedes the part of its sum insured above
    /// the retention line `retention`, up to `lines` lines, and the same
    /// share of every loss on it. With a retention of 1m and 9 lines (a
    /// capacity of 9m), a 5m risk cedes 80% and a 20m risk 45%.
    ///
    /// The events must carry sums insured
    /// ([`EventSet::with_sums_insured`], or a risk profile); an event limit
    /// or annual terms can be added as for any layer.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let s = Layer::surplus("surplus", 1e6, 9.0).unwrap();
    /// let ceded = s.ceded_with_sums_insured(&[2e6, 2e6, 2e6], &[1e6, 5e6, 20e6]);
    /// assert!((ceded - (0.0 + 0.8 * 2e6 + 0.45 * 2e6)).abs() < 1e-6);
    /// ```
    pub fn surplus(name: impl Into<String>, retention: f64, lines: f64) -> Result<Self> {
        if !(retention.is_finite() && retention > 0.0) {
            return Err(invalid(
                "retention",
                retention,
                "must be positive and finite",
            ));
        }
        if !(lines.is_finite() && lines > 0.0) {
            return Err(invalid("lines", lines, "must be positive and finite"));
        }
        let mut layer = Self::xol(name, f64::INFINITY, 0.0)?;
        layer.basis = Basis::Surplus { retention, lines };
        Ok(layer)
    }

    /// Whether the layer needs each event's sum insured (a surplus).
    pub fn needs_sums_insured(&self) -> bool {
        matches!(self.basis, Basis::Surplus { .. })
    }

    /// Whether the layer cedes from an [`EventSet`]'s aggregate part
    /// ([`EventSet::with_aggregate_part`]): it has no per-occurrence terms,
    /// that is it is on the loss with attachment 0 and no limit, as a
    /// quota share or a stop loss is. A layer with per-occurrence terms
    /// sees the events alone, since an amount that is no event has no
    /// occurrence to attach to.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// assert!(Layer::quota_share("QS", 0.3).unwrap().sees_aggregate_part());
    /// assert!(Layer::stop_loss("SL", 50.0, 100.0).unwrap().sees_aggregate_part());
    /// assert!(!Layer::xol("5x5", 5.0, 5.0).unwrap().sees_aggregate_part());
    /// ```
    pub fn sees_aggregate_part(&self) -> bool {
        self.basis == Basis::Loss && self.attachment == 0.0 && self.limit == f64::INFINITY
    }

    /// An aggregate stop-loss: `limit` xs `retention` on the year's total
    /// loss. It is an unlimited per-occurrence layer from 0 whose annual
    /// deductible is `retention` and annual limit `limit`, so it covers
    /// the total of whatever losses it sees (gross, or net of earlier
    /// inuring stages).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let sl = Layer::stop_loss("SL", 50.0, 100.0).unwrap();
    /// assert_eq!(sl.ceded(&[60.0, 70.0]), 30.0);
    /// assert_eq!(sl.ceded(&[60.0, 70.0, 90.0]), 50.0);
    /// ```
    pub fn stop_loss(name: impl Into<String>, limit: f64, retention: f64) -> Result<Self> {
        Self::xol(name, f64::INFINITY, 0.0)?
            .aggregate_deductible(retention)?
            .aggregate_limit(limit)
    }

    /// Places `share` of the layer.
    pub fn share(mut self, share: f64) -> Result<Self> {
        if !(share > 0.0 && share <= 1.0) {
            return Err(invalid("share", share, "must be in (0, 1]"));
        }
        self.share = share;
        Ok(self)
    }

    /// Annual aggregate deductible.
    pub fn aggregate_deductible(mut self, aad: f64) -> Result<Self> {
        if !aad.is_finite() || aad < 0.0 {
            return Err(invalid(
                "aggregate_deductible",
                aad,
                "must be finite and non-negative",
            ));
        }
        self.aggregate_deductible = aad;
        Ok(self)
    }

    /// Annual aggregate limit.
    pub fn aggregate_limit(mut self, aal: f64) -> Result<Self> {
        if aal.is_nan() || aal <= 0.0 {
            return Err(invalid("aggregate_limit", aal, "must be positive"));
        }
        self.aggregate_limit = aal;
        Ok(self)
    }

    /// A loss corridor: of the annual layer loss at 100% after the annual
    /// deductible, the cedant keeps `retained` of the part between `lower`
    /// and `upper`; the annual limit then caps what is left. So the
    /// corridor is used up in event order as the deductible is, and the
    /// reinsurer still pays up to the full annual limit. Reinstatement
    /// premiums follow the layer loss after the corridor: limit kept in
    /// the corridor is not reinstated or charged for.
    ///
    /// Corridors are usually quoted as loss ratios on the reinsurer's
    /// premium: a band from `lr_lower` to `lr_upper` on a premium `P` for
    /// a placed share `s` is `lower = lr_lower × P / s`, `upper = lr_upper
    /// × P / s` (for a quota share, `P / s` is the subject premium).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // A 30% quota share; the cedant keeps the ceded loss ratio between
    /// // 70% and 90% of a subject premium of 100.
    /// let qs = Layer::quota_share("QS", 0.3).unwrap()
    ///     .loss_corridor(70.0, 90.0, 1.0).unwrap();
    /// // Losses 80: 10 of them fall in the corridor, so 0.3 × 70 is ceded.
    /// assert!((qs.ceded(&[50.0, 30.0]) - 21.0).abs() < 1e-12);
    /// // Losses 120: the whole band of 20 is kept.
    /// assert!((qs.ceded(&[120.0]) - 30.0).abs() < 1e-12);
    /// ```
    pub fn loss_corridor(mut self, lower: f64, upper: f64, retained: f64) -> Result<Self> {
        if !(lower.is_finite() && lower >= 0.0) {
            return Err(invalid("lower", lower, "must be finite and non-negative"));
        }
        if !(upper.is_finite() && upper > lower) {
            return Err(invalid("upper", upper, "must be finite and above lower"));
        }
        if !(retained > 0.0 && retained <= 1.0) {
            return Err(invalid("retained", retained, "must be in (0, 1]"));
        }
        self.corridor = Some(Corridor {
            lower,
            upper,
            retained,
        });
        Ok(self)
    }

    /// `n` free reinstatements: the annual limit becomes
    /// `limit × (n + 1)`.
    pub fn reinstatements(self, n: u32) -> Result<Self> {
        let aal = self.limit * (f64::from(n) + 1.0);
        self.aggregate_limit(aal)
    }

    /// Paid reinstatements, one per entry of `rates`: the annual limit
    /// becomes `limit × (rates.len() + 1)`, and reinstating the `k`-th
    /// limit costs `rates[k] × premium`, pro rata as to amount (see
    /// [`reinstatement_premium`](Self::reinstatement_premium)). `premium`
    /// is the upfront premium for the placed share.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 10 xs 10, premium 2, the first reinstatement at 100% and the
    /// // second at 50%.
    /// let layer = Layer::xol("10x10", 10.0, 10.0).unwrap()
    ///     .paid_reinstatements(2.0, vec![1.0, 0.5]).unwrap();
    /// assert_eq!(layer.aggregate_limit, 30.0);
    /// // Layer loss 10 + 2 = 12: the first limit is used up and 2 of the
    /// // second.
    /// assert_eq!(
    ///     layer.reinstatement_premium(&[22.0, 12.0]),
    ///     2.0 * (1.0 * 10.0 + 0.5 * 2.0) / 10.0
    /// );
    /// ```
    pub fn paid_reinstatements(mut self, premium: f64, rates: Vec<f64>) -> Result<Self> {
        if !self.limit.is_finite() {
            return Err(invalid(
                "limit",
                self.limit,
                "must be finite for paid reinstatements",
            ));
        }
        if !premium.is_finite() || premium < 0.0 {
            return Err(invalid(
                "premium",
                premium,
                "must be finite and non-negative",
            ));
        }
        if !rates.is_empty()
            && (self.swing.is_some()
                || self.profit_commission.is_some()
                || matches!(self.commission, Some(Commission::SlidingScale(_))))
        {
            return Err(invalid(
                "reinstatement_rates",
                rates.len() as f64,
                "cannot be combined with swing rating, a sliding scale or a profit commission",
            ));
        }
        if let Some(&rate) = rates.iter().find(|r| !r.is_finite() || **r < 0.0) {
            return Err(invalid(
                "reinstatement_rates",
                rate,
                "must be finite and non-negative",
            ));
        }
        let n = u32::try_from(rates.len())
            .map_err(|_| invalid("reinstatement_rates", rates.len() as f64, "too many"))?;
        self = self.reinstatements(n)?;
        self.premium = premium;
        self.reinstatement_rates = rates;
        Ok(self)
    }

    /// Makes the paid reinstatements pro rata as to time as well as to
    /// amount: limit used by a loss at time `t` (the fraction of the year
    /// elapsed) is reinstated for the remaining `1 − t` of the year, so its
    /// premium is scaled by `1 − t`. The events must carry times
    /// ([`EventSet::with_times`], [`EventSet::with_uniform_times`],
    /// [`EventSet::with_seasonal_times`]).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let layer = Layer::xol("10x10", 10.0, 10.0).unwrap()
    ///     .paid_reinstatements(2.0, vec![1.0, 0.5]).unwrap()
    ///     .pro_rata_as_to_time().unwrap();
    /// // The first limit is used up a quarter of the way through the year,
    /// // 2 of the second at half way.
    /// let rp = layer.reinstatement_premium_dated(&[22.0, 12.0], &[0.25, 0.5]);
    /// assert!((rp - 2.0 * (1.0 * 0.75 + 0.5 * 0.2 * 0.5)).abs() < 1e-12);
    /// ```
    pub fn pro_rata_as_to_time(mut self) -> Result<Self> {
        if self.reinstatement_rates.is_empty() {
            return Err(invalid(
                "reinstatement_rates",
                0.0,
                "must be given (paid_reinstatements) before pro rata as to time",
            ));
        }
        if self.allocation == LimitAllocation::ProRata {
            return Err(Error::Data(
                "reinstatements pro rata as to time need the annual limit used in event \
                 order, not pro rata"
                    .into(),
            ));
        }
        self.pro_rata_time = true;
        Ok(self)
    }

    /// Whether the layer needs each event's time (paid reinstatements pro
    /// rata as to time).
    pub fn needs_times(&self) -> bool {
        self.pro_rata_time
    }

    /// Whether the layer reads the order of a year's events: an
    /// nth-event or a top-and-drop cover. A grid or a year's annual total
    /// has no events in order, so those paths refuse it.
    pub fn needs_event_order(&self) -> bool {
        self.nth_event.is_some() || self.drop_down.is_some()
    }

    /// A franchise deductible (#591): an occurrence whose loss is at or
    /// above the attachment is paid from the ground up, `min(x, limit)`,
    /// and one below it pays nothing. On the loss basis only; not with a
    /// top-and-drop cover.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let f = Layer::xol("F", 20.0, 5.0).unwrap().franchise().unwrap();
    /// assert_eq!(f.ceded_by_event(&[4.0, 5.0, 12.0, 30.0]), [0.0, 5.0, 12.0, 20.0]);
    /// ```
    pub fn franchise(mut self) -> Result<Self> {
        if self.basis != Basis::Loss {
            return Err(Error::Data(format!(
                "layer {:?}: a franchise deductible is on the loss basis",
                self.name
            )));
        }
        if self.drop_down.is_some() {
            return Err(Error::Data(format!(
                "layer {:?}: a franchise deductible cannot drop down",
                self.name
            )));
        }
        if self.disappearing.is_some() {
            return Err(Error::Data(format!(
                "layer {:?}: a deductible is franchise or disappearing, not both",
                self.name
            )));
        }
        self.franchise = true;
        Ok(self)
    }

    /// A disappearing deductible (#772), `aggregate`'s `y xs d1 to d2
    /// disappearing`: the retention is the attachment `d1` up to a loss of
    /// `d1`, falls linearly to 0 at a loss of `end` (`d2`), and is 0 above,
    /// so an occurrence of `x` pays `min(x - r(x), limit)`: nothing up to
    /// `d1`, `k (x - d1)` with `k = d2 / (d2 - d1)` up to `d2`, the whole
    /// loss above. The limit is the largest payment, as for the ordinary
    /// and franchise deductibles. On the loss basis only; not with a
    /// franchise or a top-and-drop cover.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 300 xs 50 to 100: the retention falls from 50 to 0 between 50 and 100.
    /// let d = Layer::xol("D", 300.0, 50.0).unwrap().disappearing(100.0).unwrap();
    /// assert_eq!(d.ceded_by_event(&[40.0, 75.0, 100.0, 250.0, 400.0]), [0.0, 50.0, 100.0, 250.0, 300.0]);
    /// ```
    pub fn disappearing(mut self, end: f64) -> Result<Self> {
        if self.basis != Basis::Loss {
            return Err(Error::Data(format!(
                "layer {:?}: a disappearing deductible is on the loss basis",
                self.name
            )));
        }
        if self.franchise || self.drop_down.is_some() {
            return Err(Error::Data(format!(
                "layer {:?}: a disappearing deductible is not with a franchise or a top-and-drop \
                 cover",
                self.name
            )));
        }
        if !(end.is_finite() && end > self.attachment) {
            return Err(invalid(
                "end",
                end,
                "must be finite and above the attachment (equal is a franchise)",
            ));
        }
        self.disappearing = Some(end);
        Ok(self)
    }

    /// The first three raw moments of one occurrence's recovery at 100%,
    /// before annual terms, for a ground-up loss with distribution
    /// `severity`: exact from the severity's layer moments, for an
    /// ordinary, franchise or disappearing deductible on the loss basis.
    ///
    /// A disappearing deductible pays `k A + B` with `A` the layer
    /// `d2 - d1 xs d1`, `k = d2 / (d2 - d1)` and `B` the layer
    /// `limit - d2 xs d2` (`k` times the layer `limit / k xs d1` when the
    /// limit is at most `d2`); `B > 0` only when `A` is full, so the cross
    /// moments are exact. A franchise pays `d 1{X ≥ d} + B` with `B` the
    /// layer `limit - d xs d`.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    /// use prospicio_prob::{Lognormal, Severity};
    ///
    /// let x = Lognormal::from_mean_cv(100.0, 1.0).unwrap();
    /// let plain = Layer::xol("L", 300.0, 50.0).unwrap();
    /// let m = plain.occurrence_moments(&x).unwrap();
    /// assert!((m[0] - x.layer(300.0, 50.0)).abs() < 1e-12);
    /// ```
    pub fn occurrence_moments(&self, severity: &dyn prospicio_prob::Severity) -> Result<[f64; 3]> {
        if self.basis != Basis::Loss {
            return Err(Error::Data(format!(
                "layer {:?}: occurrence moments are on the loss basis",
                self.name
            )));
        }
        let layer = |limit: f64, attachment: f64| {
            [
                severity.layer(limit, attachment),
                severity.layer_second_moment(limit, attachment),
                severity.layer_third_moment(limit, attachment),
            ]
        };
        let (d1, l) = (self.attachment, self.limit);
        if let Some(d2) = self.disappearing {
            let k = d2 / (d2 - d1);
            if l <= d2 {
                let a = layer(l / k, d1);
                return Ok([k * a[0], k * k * a[1], k * k * k * a[2]]);
            }
            let (a, b, w) = (layer(d2 - d1, d1), layer(l - d2, d2), d2 - d1);
            return Ok([
                k * a[0] + b[0],
                k * k * a[1] + 2.0 * k * w * b[0] + b[1],
                k * k * k * a[2] + 3.0 * k * k * w * w * b[0] + 3.0 * k * w * b[1] + b[2],
            ]);
        }
        if self.franchise {
            let s = severity.survival(d1);
            if l <= d1 {
                return Ok([l * s, l * l * s, l * l * l * s]);
            }
            let b = layer(l - d1, d1);
            return Ok([
                d1 * s + b[0],
                d1 * d1 * s + 2.0 * d1 * b[0] + b[1],
                d1 * d1 * d1 * s + 3.0 * d1 * d1 * b[0] + 3.0 * d1 * b[1] + b[2],
            ]);
        }
        Ok(layer(l, d1))
    }

    /// An nth-event cover (#591, [`NthEvent`]): only the `n`-th and later
    /// events of a year whose loss to the layer exceeds `threshold` are
    /// covered; earlier ones, and events at or below the threshold, cede
    /// nothing. Events are counted in the order given, chronological when
    /// the event set has times. `nth_event(1, attachment)` is the plain
    /// layer. Grids and annual totals have no events in order and refuse
    /// it.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // A second-event cover: the first loss above 5 is the cedant's.
    /// let l = Layer::xol("2nd", 10.0, 5.0).unwrap().nth_event(2, 5.0).unwrap();
    /// assert_eq!(l.ceded_by_event(&[12.0, 3.0, 9.0, 20.0]), [0.0, 0.0, 4.0, 10.0]);
    /// ```
    pub fn nth_event(mut self, n: u32, threshold: f64) -> Result<Self> {
        if n == 0 {
            return Err(invalid("n", 0.0, "must be at least 1"));
        }
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(invalid(
                "threshold",
                threshold,
                "must be finite and at least 0",
            ));
        }
        self.nth_event = Some(NthEvent { n, threshold });
        Ok(self)
    }

    /// A top-and-drop cover (#591, [`DropDown`]): the layer sits at its own
    /// attachment until the annual limit of the layer named `underlying`
    /// is used up, and from the next event attaches at `attachment`, with
    /// the same occurrence and annual limits. The underlying layer must be
    /// in the same stage, earlier in the tower, with an annual limit
    /// ([`Tower::inuring`] checks). On the loss basis only; not with a
    /// franchise. A drop-down whose underlying layer never exhausts is the
    /// plain layer.
    pub fn top_and_drop(mut self, underlying: &str, attachment: f64) -> Result<Self> {
        if self.basis != Basis::Loss {
            return Err(Error::Data(format!(
                "layer {:?}: a top-and-drop cover is on the loss basis",
                self.name
            )));
        }
        if self.franchise || self.disappearing.is_some() {
            return Err(Error::Data(format!(
                "layer {:?}: a franchise or disappearing deductible cannot drop down",
                self.name
            )));
        }
        if !(attachment.is_finite() && attachment >= 0.0 && attachment < self.attachment) {
            return Err(invalid(
                "attachment",
                attachment,
                "must be at least 0 and below the layer's own",
            ));
        }
        if underlying == self.name {
            return Err(invalid("underlying", 0.0, "must be another layer"));
        }
        self.drop_down = Some(DropDown {
            underlying: underlying.to_string(),
            attachment,
        });
        Ok(self)
    }

    /// The layer seen only by events whose `label` is one of `values`
    /// (#584): a peril-specific treaty. Every other event, and the
    /// aggregate part's other columns for a layer that sees the part, is
    /// zero to this layer, for its occurrence terms and its annual terms
    /// (deductible, corridor, annual limit, reinstatements) alike, while
    /// the layers in later stages still see it. It needs labelled events
    /// ([`EventSet::with_labels`]) in [`Tower::apply`],
    /// [`Tower::apply_term`] and [`Tower::apply_timeline`]; the grid and
    /// annual paths, which have no events, refuse it. A scope over every
    /// value equals no scope, and the scoped layer on the whole set equals
    /// the unscoped one on [`EventSet::filter`]'s set.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_prob::{Empirical, KeyValue};
    ///
    /// // A windstorm 10 xs 10 with an annual limit of 15 on a year of a
    /// // windstorm of 25 and an earthquake of 30.
    /// let events = EventSet::from_years(vec![vec![25.0, 30.0]], 0).unwrap()
    ///     .with_labels(vec![("peril".into(), vec!["ws".into(), "eq".into()])]).unwrap();
    /// let ws = Layer::xol("WS", 10.0, 10.0).unwrap()
    ///     .aggregate_limit(15.0).unwrap()
    ///     .scope("peril", &["ws"]).unwrap();
    /// let result = Tower::new(vec![ws]).unwrap().apply(&events).unwrap()
    ///     .aggregate(&["kind", "layer"]).unwrap();
    /// let key = vec![KeyValue::from("ceded"), KeyValue::from("WS")];
    /// assert_eq!(Empirical::draws(&result.marginal(&key).unwrap()), [10.0]);
    /// ```
    pub fn scope(mut self, label: &str, values: &[&str]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::Data(format!(
                "a scope on {label:?} needs at least one value"
            )));
        }
        self.scope = Some(Scope {
            label: label.to_string(),
            values: values.iter().map(|v| v.to_string()).collect(),
        });
        Ok(self)
    }

    /// How a binding annual deductible or limit is shared among the
    /// year's events ([`LimitAllocation`]): in event order (the default)
    /// or pro rata. It changes [`ceded_by_event`](Self::ceded_by_event),
    /// and so the losses a later inuring stage sees, not the annual ceded
    /// loss. Pro rata cannot be combined with reinstatements pro rata as
    /// to time, whose premium follows the order in which the limit is
    /// used.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, LimitAllocation};
    ///
    /// // 10 xs 5, one reinstatement: 20 a year. Recoveries 10, 10, 5 = 25.
    /// let layer = Layer::xol("10x5", 10.0, 5.0).unwrap().reinstatements(1).unwrap();
    /// assert_eq!(layer.ceded_by_event(&[15.0, 20.0, 10.0]), [10.0, 10.0, 0.0]);
    /// let pro_rata = layer.allocation(LimitAllocation::ProRata).unwrap();
    /// assert_eq!(pro_rata.ceded_by_event(&[15.0, 20.0, 10.0]), [8.0, 8.0, 4.0]);
    /// ```
    pub fn allocation(mut self, allocation: LimitAllocation) -> Result<Self> {
        if allocation == LimitAllocation::ProRata && self.pro_rata_time {
            return Err(Error::Data(
                "reinstatements pro rata as to time need the annual limit used in event \
                 order, not pro rata"
                    .into(),
            ));
        }
        self.allocation = allocation;
        Ok(self)
    }

    /// Sets the premium from a deposit `amount` quoted for 100% of the
    /// layer: the premium for the placed share is `share × amount`, so
    /// set the share first.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let l = Layer::xol("5x5", 5e6, 5e6).unwrap().share(0.4).unwrap()
    ///     .deposit_premium(1e6).unwrap();
    /// assert_eq!(l.premium, 4e5);
    /// ```
    pub fn deposit_premium(mut self, amount: f64) -> Result<Self> {
        if !(amount.is_finite() && amount >= 0.0) {
            return Err(invalid("amount", amount, "must be finite and non-negative"));
        }
        self.premium = self.share * amount;
        Ok(self)
    }

    /// Sets the premium from a rate on line: `share × rol × limit`, so set
    /// the share first. Needs a finite per-occurrence limit.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 10m xs 10m at a 12.5% rate on line.
    /// let l = Layer::xol("10x10", 10e6, 10e6).unwrap().rate_on_line(0.125).unwrap();
    /// assert_eq!(l.premium, 1.25e6);
    /// ```
    pub fn rate_on_line(mut self, rol: f64) -> Result<Self> {
        if !self.limit.is_finite() {
            return Err(invalid(
                "limit",
                self.limit,
                "must be finite for a rate on line",
            ));
        }
        if !(rol.is_finite() && rol >= 0.0) {
            return Err(invalid("rol", rol, "must be finite and non-negative"));
        }
        self.premium = self.share * rol * self.limit;
        Ok(self)
    }

    /// Sets the premium as a rate on the subject (underlying) premium:
    /// `share × rate × subject_premium`, so set the share first. A quota
    /// share's ceded premium is `premium_rate(1.0, subject_premium)`.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS", 0.3).unwrap().premium_rate(1.0, 1000.0).unwrap();
    /// assert_eq!(qs.premium, 300.0);
    /// // An excess layer at 2.5% of a subject premium of 40m.
    /// let xl = Layer::xol("5x5", 5e6, 5e6).unwrap().premium_rate(0.025, 40e6).unwrap();
    /// assert_eq!(xl.premium, 1e6);
    /// ```
    pub fn premium_rate(mut self, rate: f64, subject_premium: f64) -> Result<Self> {
        if !(rate.is_finite() && rate >= 0.0) {
            return Err(invalid("rate", rate, "must be finite and non-negative"));
        }
        if !(subject_premium.is_finite() && subject_premium >= 0.0) {
            return Err(invalid(
                "subject_premium",
                subject_premium,
                "must be finite and non-negative",
            ));
        }
        self.premium = self.share * rate * subject_premium;
        Ok(self)
    }

    /// A flat ceding commission: the reinsurer pays back `rate` of the
    /// year's premium (the swing-rated premium, if any; reinstatement
    /// premiums carry none).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS", 0.3).unwrap()
    ///     .premium_rate(1.0, 1000.0).unwrap()
    ///     .ceding_commission(0.3).unwrap();
    /// assert!((qs.ceding_commission_for(150.0) - 90.0).abs() < 1e-12);
    /// ```
    pub fn ceding_commission(mut self, rate: f64) -> Result<Self> {
        if !(rate.is_finite() && (0.0..=1.0).contains(&rate)) {
            return Err(invalid("rate", rate, "must be in [0, 1]"));
        }
        self.commission = Some(Commission::Flat(rate));
        self.sliding_scale_carry = None;
        Ok(self)
    }

    /// A sliding-scale ceding commission from `(commission, loss_ratio)`
    /// anchors: the commission at the year's ceded loss ratio (ceded loss
    /// over premium), linear between anchors and flat beyond the first and
    /// last, as `aggregate`'s `slide`. The commission must not rise with
    /// the loss ratio, and the loss ratios must differ. Set the premium
    /// first; a sliding scale cannot be combined with swing rating or paid
    /// reinstatements (as in `aggregate`), and replaces any flat
    /// commission.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 45% at a 60% loss ratio, sliding to 25% at 70% and 19% at 80%.
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .sliding_scale(vec![(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)]).unwrap();
    /// // Ceded loss 65 on a premium of 100: a 65% loss ratio, 35% commission.
    /// assert!((qs.ceding_commission_for(65.0) - 35.0).abs() < 1e-12);
    /// assert!((qs.ceding_commission_for(90.0) - 19.0).abs() < 1e-12);
    /// ```
    pub fn sliding_scale(mut self, anchors: Vec<(f64, f64)>) -> Result<Self> {
        self.check_loss_ratio_terms("sliding_scale")?;
        let anchors = checked_anchors(anchors, true)?;
        self.commission = Some(Commission::SlidingScale(anchors));
        Ok(self)
    }

    /// A profit commission: `share` of `max(premium × (1 − allowance) −
    /// ceded loss, 0)`, as `aggregate`'s `pc <share> after <allowance>`.
    /// The allowance is the reinsurer's expenses and margin as a fraction
    /// of premium, including any ceding commission the contract deducts
    /// before profit. Set the premium first; it cannot be combined with
    /// swing rating or paid reinstatements (as in `aggregate`).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 25% of the profit after a 10% allowance, on a premium of 100.
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .profit_commission(0.25, 0.1).unwrap();
    /// assert!((qs.profit_commission_for(60.0) - 7.5).abs() < 1e-12);
    /// assert_eq!(qs.profit_commission_for(95.0), 0.0);
    /// ```
    pub fn profit_commission(mut self, share: f64, allowance: f64) -> Result<Self> {
        self.check_loss_ratio_terms("profit_commission")?;
        if !(share.is_finite() && (0.0..=1.0).contains(&share)) {
            return Err(invalid("share", share, "must be in [0, 1]"));
        }
        if !(allowance.is_finite() && allowance >= 0.0) {
            return Err(invalid(
                "allowance",
                allowance,
                "must be finite and non-negative",
            ));
        }
        self.profit_commission = Some(ProfitCommission { share, allowance });
        Ok(self)
    }

    /// Carries the profit commission's deficits forward between the years
    /// of a term ([`Tower::apply_term`]): a year whose ceded loss exceeds
    /// `premium × (1 − allowance)` pays no commission and carries the
    /// excess, which later years' profits must cover before any commission
    /// is paid. `years` limits how many later years a deficit may be
    /// carried into (`None` until it is used up). Set the profit commission
    /// first. A single year ([`Tower::apply`]) starts with nothing carried,
    /// so the term has no effect there.
    ///
    /// Over a term with deficits carried to extinction, the commissions
    /// telescope: `Σ PC_t = share × (Σ_t (premium (1 − allowance) − ceded_t)
    /// + D_T)`, with `D_T` the deficit left at the end.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .profit_commission(0.25, 0.1).unwrap()
    ///     .profit_commission_carry_forward(None).unwrap();
    /// // Premium 100, so 90 after the allowance. Year 1 cedes 120 (a deficit
    /// // of 30); year 2 cedes 40 (a profit of 50, less 30 carried).
    /// assert_eq!(qs.profit_commissions_over(&[120.0, 40.0]), [0.0, 5.0]);
    /// ```
    pub fn profit_commission_carry_forward(mut self, years: Option<u32>) -> Result<Self> {
        if self.profit_commission.is_none() {
            return Err(invalid(
                "profit_commission_carry_forward",
                0.0,
                "needs a profit commission first",
            ));
        }
        check_carry_years(years)?;
        self.profit_commission_carry = Some(CarryForward {
            deficit: true,
            credit: false,
            years,
        });
        Ok(self)
    }

    /// Carries the sliding scale's deficits (ceded loss above the last
    /// anchor's loss ratio), its credits (ceded loss below the first
    /// anchor's), or both forward between the years of a term
    /// ([`Tower::apply_term`]); see [`CarryForward`]. `years` limits how
    /// many later years an amount may be carried into (`None` until it is
    /// used up). Set the sliding scale first; a later flat
    /// [`ceding_commission`](Self::ceding_commission) removes it.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 45% at a 60% loss ratio, sliding to 25% at 70%, on a premium of 100.
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .sliding_scale(vec![(0.45, 0.60), (0.25, 0.70)]).unwrap()
    ///     .sliding_scale_carry_forward(true, false, None).unwrap();
    /// // Year 1 at 80%: the minimum 25%, with 10 carried. Year 2 at 55%
    /// // reads 65%: a 35% commission.
    /// let c = qs.ceding_commissions_over(&[80.0, 55.0]);
    /// assert!((c[0] - 25.0).abs() < 1e-12 && (c[1] - 35.0).abs() < 1e-12);
    /// ```
    pub fn sliding_scale_carry_forward(
        mut self,
        deficit: bool,
        credit: bool,
        years: Option<u32>,
    ) -> Result<Self> {
        if !matches!(self.commission, Some(Commission::SlidingScale(_))) {
            return Err(invalid(
                "sliding_scale_carry_forward",
                0.0,
                "needs a sliding scale first",
            ));
        }
        if !(deficit || credit) {
            return Err(invalid(
                "sliding_scale_carry_forward",
                0.0,
                "must carry deficits, credits or both",
            ));
        }
        check_carry_years(years)?;
        self.sliding_scale_carry = Some(CarryForward {
            deficit,
            credit,
            years,
        });
        Ok(self)
    }

    /// The profit commissions over consecutive years of a term with these
    /// ceded losses, carrying deficits forward if the layer does
    /// ([`profit_commission_carry_forward`](Self::profit_commission_carry_forward));
    /// without a carry each is [`profit_commission_for`](Self::profit_commission_for).
    pub fn profit_commissions_over(&self, ceded: &[f64]) -> Vec<f64> {
        let mut carry = Carried::default();
        ceded
            .iter()
            .enumerate()
            .map(|(t, &c)| self.profit_commission_in(c, t, &mut carry))
            .collect()
    }

    /// The ceding commissions over consecutive years of a term with these
    /// ceded losses, carrying the sliding scale's deficits and credits
    /// forward if the layer does
    /// ([`sliding_scale_carry_forward`](Self::sliding_scale_carry_forward));
    /// without a carry each is [`ceding_commission_for`](Self::ceding_commission_for).
    pub fn ceding_commissions_over(&self, ceded: &[f64]) -> Vec<f64> {
        let mut carry = Carried::default();
        ceded
            .iter()
            .enumerate()
            .map(|(t, &c)| self.ceding_commission_in(c, c, t, &mut carry))
            .collect()
    }

    /// A no-claims bonus: `share` of the year's premium returned when the
    /// layer's ceded loss over the last `years` contract years is at most
    /// `threshold` ([`NoClaimsBonus`]). It is figured on its own, beside any
    /// sliding scale or profit commission, on the year's premium as the
    /// layer prices it ([`premium_for`](Self::premium_for), the deposit or
    /// a swing-rated premium), without reinstatement premium. `years > 1`
    /// needs a term ([`Tower::apply_term`]).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let cat = Layer::xol("20x10", 20.0, 10.0).unwrap()
    ///     .premium_rate(0.1, 20.0).unwrap()
    ///     .no_claims_bonus(0.15, 1, 0.0).unwrap();
    /// // A premium of 2: 0.3 back in a year with no loss to the layer.
    /// assert!((cat.no_claims_bonus_for(0.0) - 0.3).abs() < 1e-12);
    /// assert_eq!(cat.no_claims_bonus_for(5.0), 0.0);
    /// ```
    pub fn no_claims_bonus(mut self, share: f64, years: u32, threshold: f64) -> Result<Self> {
        if !(share.is_finite() && share > 0.0 && share <= 1.0) {
            return Err(invalid("share", share, "must be in (0, 1]"));
        }
        if years == 0 {
            return Err(invalid("years", 0.0, "must be at least 1"));
        }
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(invalid(
                "threshold",
                threshold,
                "must be finite and non-negative",
            ));
        }
        if self.premium.is_nan() || self.premium <= 0.0 {
            return Err(invalid(
                "premium",
                self.premium,
                "must be set (positive) before a no-claims bonus",
            ));
        }
        self.no_claims_bonus = Some(NoClaimsBonus {
            share,
            years,
            threshold,
        });
        Ok(self)
    }

    /// The year's no-claims bonus given its ceded loss, the year alone;
    /// zero without one.
    pub fn no_claims_bonus_for(&self, ceded: f64) -> f64 {
        match &self.no_claims_bonus {
            Some(b) if ceded <= b.threshold => b.share * self.premium_for(ceded),
            _ => 0.0,
        }
    }

    /// Year `t`'s no-claims bonus in a term, on the ceded loss of the last
    /// `years` contract years, `history` holding the earlier years'.
    pub(crate) fn no_claims_bonus_in(&self, ceded: f64, t: usize, history: &mut Carried) -> f64 {
        let Some(b) = &self.no_claims_bonus else {
            return 0.0;
        };
        let window = history.window(ceded, t, b.years);
        if window <= b.threshold {
            b.share * self.premium_for(ceded)
        } else {
            0.0
        }
    }

    /// Year `t`'s profit commission in a term, updating what is carried.
    pub(crate) fn profit_commission_in(&self, ceded: f64, t: usize, carry: &mut Carried) -> f64 {
        let (Some(pc), Some(cf)) = (&self.profit_commission, &self.profit_commission_carry) else {
            return self.profit_commission_for(ceded);
        };
        let threshold = self.premium * (1.0 - pc.allowance);
        // Signed loss beyond the threshold, with what earlier years left.
        let total = carry.step(ceded - threshold, t, cf.years, |x| x.max(0.0));
        pc.share * (-total).max(0.0)
    }

    /// Year `t`'s ceding commission in a term, updating what is carried.
    /// `key` is the loss the scale reads: the ceded loss, or with
    /// [`scales_before_annual_terms`](Self::scales_before_annual_terms) the
    /// ceded share of the loss before the annual terms.
    pub(crate) fn ceding_commission_in(
        &self,
        ceded: f64,
        key: f64,
        t: usize,
        carry: &mut Carried,
    ) -> f64 {
        let Some(scale @ Commission::SlidingScale(_)) = &self.commission else {
            return self.ceding_commission_for(ceded);
        };
        scale_in(
            scale,
            self.premium,
            key,
            t,
            carry,
            self.sliding_scale_carry.as_ref(),
        )
    }

    /// Swing rating: the year's premium is `terms.premium(ceded loss)`,
    /// with the terms quoted for 100% of the layer and scaled to the
    /// placed share as `aggregate` does: `clip(share × basic + lcm ×
    /// ceded, share × minimum, share × maximum)`, where the ceded loss is
    /// already at the placed share. It replaces the fixed premium in the
    /// tower's results and in a flat ceding commission. It cannot be
    /// combined with paid reinstatements, a sliding scale or a profit
    /// commission (as in `aggregate`).
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, LossSensitivePremium};
    ///
    /// // Pay back losses at 100/80, between 100 and 300.
    /// let swing = LossSensitivePremium::new(0.0, 1.25, Some(100.0), Some(300.0)).unwrap();
    /// let l = Layer::xol("L", 1000.0, 0.0).unwrap().swing_rated(swing).unwrap();
    /// assert_eq!(l.premium_for(40.0), 100.0);
    /// assert_eq!(l.premium_for(200.0), 250.0);
    /// assert_eq!(l.premium_for(400.0), 300.0);
    /// ```
    pub fn swing_rated(mut self, terms: LossSensitivePremium) -> Result<Self> {
        if !self.reinstatement_rates.is_empty() {
            return Err(invalid(
                "swing",
                terms.basic,
                "cannot be combined with paid reinstatements",
            ));
        }
        if matches!(self.commission, Some(Commission::SlidingScale(_)))
            || matches!(self.brokerage, Some(Commission::SlidingScale(_)))
            || self.profit_commission.is_some()
        {
            return Err(invalid(
                "swing",
                terms.basic,
                "cannot be combined with a sliding scale or a profit commission",
            ));
        }
        self.swing = Some(terms);
        Ok(self)
    }

    /// A sliding scale and a profit commission read the loss ratio on a
    /// fixed premium.
    pub(crate) fn check_loss_ratio_terms(&self, name: &'static str) -> Result<()> {
        if self.swing.is_some() {
            return Err(invalid(name, 0.0, "cannot be combined with swing rating"));
        }
        if !self.reinstatement_rates.is_empty() {
            return Err(invalid(
                name,
                0.0,
                "cannot be combined with paid reinstatements",
            ));
        }
        if self.premium.is_nan() || self.premium <= 0.0 {
            return Err(invalid(
                "premium",
                self.premium,
                "must be set (positive) before a sliding scale or profit commission",
            ));
        }
        Ok(())
    }

    /// The year's premium for the placed share given its ceded loss: the
    /// swing-rated premium, or else the fixed `premium`. Reinstatement
    /// premiums are separate.
    pub fn premium_for(&self, ceded: f64) -> f64 {
        match &self.swing {
            Some(s) => s.placed(self.share).premium(ceded),
            None => self.premium,
        }
    }

    /// The year's ceding commission given its ceded loss: the rate (flat,
    /// or from the sliding scale at the loss ratio `ceded / premium`) times
    /// the year's premium; zero without one.
    /// A layer with
    /// [`scales_before_annual_terms`](Self::scales_before_annual_terms)
    /// reads its scale on another loss, which a tower's result gives.
    pub fn ceding_commission_for(&self, ceded: f64) -> f64 {
        let Some(commission) = &self.commission else {
            return 0.0;
        };
        let premium = self.premium_for(ceded);
        if premium == 0.0 {
            return 0.0;
        }
        commission.rate(ceded / premium) * premium
    }

    /// The year's profit commission given its ceded loss:
    /// `share × max(premium × (1 − allowance) − ceded, 0)`; zero without
    /// one.
    pub fn profit_commission_for(&self, ceded: f64) -> f64 {
        match &self.profit_commission {
            Some(pc) => pc.share * (self.premium * (1.0 - pc.allowance) - ceded).max(0.0),
            None => 0.0,
        }
    }

    /// Ceded loss for one year's losses (NaN for a surplus, which needs
    /// [`ceded_with_sums_insured`](Self::ceded_with_sums_insured)).
    ///
    /// The losses are taken as given: a negative loss lies below any
    /// attachment and cedes nothing, so a file with losses stored as
    /// negative numbers would cede nothing without a word.
    /// [`try_ceded`](Self::try_ceded) refuses such a loss.
    pub fn ceded(&self, losses: &[f64]) -> f64 {
        self.share * self.layer_loss(losses, None, None)
    }

    /// [`ceded`](Self::ceded), refusing a loss that is negative or not
    /// finite, usually a sign error in the data (#593).
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let layer = Layer::xol("10x5", 10.0, 5.0).unwrap();
    /// assert_eq!(layer.try_ceded(&[12.0]), Ok(7.0));
    /// assert!(layer.try_ceded(&[-12.0]).is_err());
    /// ```
    pub fn try_ceded(&self, losses: &[f64]) -> Result<f64> {
        check_losses(losses)?;
        Ok(self.ceded(losses))
    }

    /// [`ceded_by_event`](Self::ceded_by_event), refusing a loss that is
    /// negative or not finite (#593).
    pub fn try_ceded_by_event(&self, losses: &[f64]) -> Result<Vec<f64>> {
        check_losses(losses)?;
        Ok(self.ceded_by_event(losses))
    }

    /// Ceded loss for one year's losses on risks with the given sums
    /// insured, one per loss.
    pub fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> f64 {
        self.share * self.layer_loss(losses, Some(sums_insured), None)
    }

    /// Reinstatement premium for one year's losses: with layer loss `L`
    /// at 100% (after the annual deductible and limit),
    ///
    /// ```text
    /// premium × Σ_k rates[k] × min(max(L - k × limit, 0), limit) / limit
    /// ```
    ///
    /// summing over `k = 0, 1, …` (the `k`-th reinstatement restores the
    /// limit used up by the `k + 1`-th). Zero when reinstatements are free;
    /// NaN when they are pro rata as to time, which needs
    /// [`reinstatement_premium_dated`](Self::reinstatement_premium_dated).
    pub fn reinstatement_premium(&self, losses: &[f64]) -> f64 {
        self.reinstatement_premium_at(losses, None, None, None)
    }

    /// Reinstatement premium for one year's losses at the given times (the
    /// fraction of the year elapsed, one per loss, in order). Pro rata as
    /// to time, the limit an event uses up is charged at `1 − t`; otherwise
    /// the times are ignored.
    pub fn reinstatement_premium_dated(&self, losses: &[f64], times: &[f64]) -> f64 {
        self.reinstatement_premium_at(losses, None, Some(times), None)
    }

    pub(super) fn reinstatement_premium_at(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        times: Option<&[f64]>,
        lf: Option<&[f64]>,
    ) -> f64 {
        if self.reinstatement_rates.is_empty() {
            return 0.0;
        }
        let used = |a: f64, b: f64| self.reinstated(a, b);
        if !self.pro_rata_time {
            return self.premium * used(0.0, self.layer_loss(losses, si, lf)) / self.limit;
        }
        let Some(times) = times else {
            return f64::NAN;
        };
        let mut before = 0.0;
        let total: f64 = self
            .cumulative_layer_loss(losses, si, lf)
            .into_iter()
            .zip(times)
            .map(|(after, t)| {
                let u = used(before, after) * (1.0 - t);
                before = after;
                u
            })
            .sum();
        self.premium * total / self.limit
    }

    /// Layer loss at 100%, after annual terms, up to and including each
    /// event in turn. `lf`, when given, scales each event's limit (an
    /// interlocking factor, [`Tower::interlocking`]).
    pub(super) fn cumulative_layer_loss(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        lf: Option<&[f64]>,
    ) -> Vec<f64> {
        let mut recovery = 0.0;
        self.recoveries(losses, si, lf)
            .into_iter()
            .map(|r| {
                recovery += r;
                self.after_terms(recovery)
            })
            .collect()
    }

    /// Annual layer loss at 100%, after annual terms.
    pub(super) fn layer_loss(&self, losses: &[f64], si: Option<&[f64]>, lf: Option<&[f64]>) -> f64 {
        self.after_terms(self.before_terms(losses, si, lf))
    }

    /// Annual layer loss at 100%, before annual terms.
    pub(super) fn before_terms(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        lf: Option<&[f64]>,
    ) -> f64 {
        self.recoveries(losses, si, lf).into_iter().sum()
    }

    /// Ceded loss per event for one year, in the order given.
    ///
    /// By default ([`LimitAllocation::EventOrder`]) annual terms are used
    /// up in that order (losses are taken as chronological): the
    /// deductible absorbs the first recoveries and the annual limit stops
    /// the last ones. Event `k` cedes the increase in annual ceded loss it
    /// causes. Under [`LimitAllocation::ProRata`] every event is scaled by
    /// the same factor instead. Either way the entries sum to
    /// [`ceded`](Self::ceded) up to rounding.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // Recoveries 3, 10, 7; deductible 4 and limit 15 leave 0, 9, 6.
    /// let layer = Layer::xol("L", 10.0, 5.0).unwrap()
    ///     .aggregate_deductible(4.0).unwrap()
    ///     .aggregate_limit(15.0).unwrap();
    /// assert_eq!(layer.ceded_by_event(&[8.0, 20.0, 12.0]), [0.0, 9.0, 6.0]);
    /// ```
    pub fn ceded_by_event(&self, losses: &[f64]) -> Vec<f64> {
        self.ceded_by_event_si(losses, None, None)
    }

    pub(super) fn ceded_by_event_si(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        lf: Option<&[f64]>,
    ) -> Vec<f64> {
        if self.allocation == LimitAllocation::ProRata {
            let recoveries = self.recoveries(losses, si, lf);
            let total: f64 = recoveries.iter().sum();
            // No recovery, nothing to share; the factor is 1 by convention.
            let factor = if total > 0.0 {
                self.after_terms(total) / total
            } else {
                1.0
            };
            return recoveries
                .into_iter()
                .map(|u| self.share * u * factor)
                .collect();
        }
        let mut before = 0.0;
        self.cumulative_layer_loss(losses, si, lf)
            .into_iter()
            .map(|cum| {
                let after = self.share * cum;
                let ceded = after - before;
                before = after;
                ceded
            })
            .collect()
    }

    /// Per-occurrence recovery at 100%, before annual terms, of a layer on
    /// the loss basis.
    pub(crate) fn recovery(&self, loss: f64) -> f64 {
        self.recovery_at(loss, None, 1.0)
    }

    /// Rate-weighted limits used between annual layer losses `a` and `b`
    /// (at 100%, after the annual terms).
    fn reinstated(&self, a: f64, b: f64) -> f64 {
        self.reinstatement_rates
            .iter()
            .enumerate()
            .map(|(k, rate)| {
                let floor = k as f64 * self.limit;
                rate * ((b - floor).clamp(0.0, self.limit) - (a - floor).clamp(0.0, self.limit))
            })
            .sum()
    }

    /// The reinstatement premium on an annual layer loss `layer_loss` at
    /// 100% after the annual terms, pro rata as to amount:
    /// `premium × Σ_k rates[k] × min(max(L − k × limit, 0), limit) / limit`.
    pub(crate) fn reinstatement_premium_on(&self, layer_loss: f64) -> f64 {
        if self.reinstatement_rates.is_empty() {
            return 0.0;
        }
        self.premium * self.reinstated(0.0, layer_loss) / self.limit
    }

    /// Per-occurrence recovery at 100%, before annual terms, for a loss on
    /// a risk with sum insured `si`, with the limit scaled by `lf` (1
    /// without interlocking).
    pub(crate) fn recovery_at(&self, loss: f64, si: Option<f64>, lf: f64) -> f64 {
        let amount = self.basis.amount(loss, si);
        if amount.is_nan() {
            // A surplus without a sum insured: `max` would turn NaN into 0.
            return f64::NAN;
        }
        if self.franchise {
            return if amount >= self.attachment {
                amount.min(self.limit * lf)
            } else {
                0.0
            };
        }
        if let Some(end) = self.disappearing {
            let d = self.attachment;
            let paid = if amount <= d {
                0.0
            } else if amount <= end {
                (amount - d) * (end / (end - d))
            } else {
                amount
            };
            return paid.min(self.limit * lf);
        }
        (amount - self.attachment).max(0.0).min(self.limit * lf)
    }

    /// Each event's recovery at 100%, before annual terms, in order: the
    /// occurrence terms, then an nth-event cover's count of qualifying
    /// events.
    pub(crate) fn recoveries(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        lf: Option<&[f64]>,
    ) -> Vec<f64> {
        let mut qualifying = 0u32;
        losses
            .iter()
            .enumerate()
            .map(|(e, &x)| {
                let s = si.map(|s| s[e]);
                let r = self.recovery_at(x, s, factor(lf, e));
                match self.nth_event {
                    Some(nth) if self.basis.amount(x, s) > nth.threshold => {
                        qualifying += 1;
                        if qualifying < nth.n { 0.0 } else { r }
                    }
                    Some(_) => 0.0,
                    None => r,
                }
            })
            .collect()
    }

    /// Annual terms applied to an annual recovery total at 100%: the
    /// deductible, the corridor, then the annual limit. Non-decreasing in
    /// `recovery`, so each event's share is its increase.
    pub(crate) fn after_terms(&self, recovery: f64) -> f64 {
        if recovery.is_nan() {
            return f64::NAN;
        }
        let after_aad = (recovery - self.aggregate_deductible).max(0.0);
        let after_corridor = match &self.corridor {
            Some(c) => after_aad - c.kept(after_aad),
            None => after_aad,
        };
        after_corridor.min(self.aggregate_limit)
    }

    /// Whether the layer has annual terms: a deductible, a corridor or an
    /// annual limit.
    pub fn has_annual_terms(&self) -> bool {
        self.aggregate_deductible > 0.0
            || self.corridor.is_some()
            || self.aggregate_limit.is_finite()
    }
}
