//! Reinsurance: excess-of-loss layers, quota shares, stop-losses and
//! surplus treaties, and towers of them applied to simulated events,
//! giving gross, ceded and net distributions.

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::monte_carlo::EventSet;

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
    /// A band of the annual layer loss the cedant keeps; see
    /// [`Layer::loss_corridor`].
    pub corridor: Option<Corridor>,
    /// Ceding commission paid back to the cedant on the year's premium:
    /// a flat rate or a sliding scale; see [`Layer::ceding_commission`].
    pub commission: Option<Commission>,
    /// Profit commission; see [`Layer::profit_commission`].
    pub profit_commission: Option<ProfitCommission>,
    /// Swing rating at 100% of the layer: the year's premium as a function
    /// of its ceded loss, in place of `premium`; see [`Layer::swing_rated`].
    pub swing: Option<LossSensitivePremium>,
}

/// A premium that depends on the year's loss `x`:
/// `clip(basic + lcm × x, minimum, maximum)`.
///
/// It is the premium of a swing-rated reinsurance layer (with `x` the
/// layer's ceded loss; see [`Layer::swing_rated`]) and of a
/// retrospectively rated policy (with `x` the account's loss, net of any
/// reinsurance that inures to it). `lcm` is the loss conversion factor,
/// often written as a loading such as `100/80` and, for a retro plan,
/// multiplied by the tax multiplier along with `basic`. As in `aggregate`,
/// the minimum defaults to `basic` and the maximum to no cap.
///
/// ```
/// use prospicio_aggregate::LossSensitivePremium;
///
/// // aggregate's retro example: basic 1000, factor 1.1, maximum 2500.
/// let retro = LossSensitivePremium::new(1000.0, 1.1, None, Some(2500.0)).unwrap();
/// assert_eq!(retro.premium(0.0), 1000.0);
/// assert!((retro.premium(500.0) - 1550.0).abs() < 1e-9);
/// assert_eq!(retro.premium(1500.0), 2500.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossSensitivePremium {
    pub basic: f64,
    /// Loss conversion factor applied to the loss.
    pub lcm: f64,
    pub minimum: f64,
    /// May be infinite (no cap).
    pub maximum: f64,
}

impl LossSensitivePremium {
    /// `minimum` defaults to `basic`, `maximum` to infinity. Fails unless
    /// `basic`, `lcm` and the minimum are finite and non-negative and the
    /// minimum is at most the maximum.
    pub fn new(basic: f64, lcm: f64, minimum: Option<f64>, maximum: Option<f64>) -> Result<Self> {
        if !(basic.is_finite() && basic >= 0.0) {
            return Err(invalid("basic", basic, "must be finite and non-negative"));
        }
        if !(lcm.is_finite() && lcm >= 0.0) {
            return Err(invalid("lcm", lcm, "must be finite and non-negative"));
        }
        let minimum = minimum.unwrap_or(basic);
        let maximum = maximum.unwrap_or(f64::INFINITY);
        if !(minimum.is_finite() && minimum >= 0.0) {
            return Err(invalid(
                "minimum",
                minimum,
                "must be finite and non-negative",
            ));
        }
        if maximum.is_nan() || maximum < minimum {
            return Err(invalid("maximum", maximum, "must be at least the minimum"));
        }
        Ok(Self {
            basic,
            lcm,
            minimum,
            maximum,
        })
    }

    /// The premium for a loss `x`.
    pub fn premium(&self, x: f64) -> f64 {
        (self.basic + self.lcm * x).clamp(self.minimum, self.maximum)
    }

    /// The terms for a placed share `s`: the currency amounts scaled by
    /// `s`, the factor kept (it applies to a loss that is already at the
    /// placed share).
    fn placed(&self, s: f64) -> Self {
        Self {
            basic: s * self.basic,
            lcm: self.lcm,
            minimum: s * self.minimum,
            maximum: s * self.maximum,
        }
    }
}

/// A ceding commission, as a fraction of the year's premium.
#[derive(Debug, Clone, PartialEq)]
pub enum Commission {
    /// The same rate whatever the losses.
    Flat(f64),
    /// A sliding scale: `(commission, loss_ratio)` anchors in increasing
    /// loss ratio, the commission linear between them and flat beyond the
    /// first and last (the maximum and minimum commission).
    SlidingScale(Vec<(f64, f64)>),
}

impl Commission {
    /// The commission rate at ceded loss ratio `lr`.
    pub fn rate(&self, lr: f64) -> f64 {
        match self {
            Commission::Flat(c) => *c,
            Commission::SlidingScale(anchors) => {
                let (first, last) = (anchors[0], anchors[anchors.len() - 1]);
                if lr <= first.1 {
                    return first.0;
                }
                if lr >= last.1 {
                    return last.0;
                }
                let k = anchors.partition_point(|a| a.1 <= lr);
                let ((c0, l0), (c1, l1)) = (anchors[k - 1], anchors[k]);
                c0 + (c1 - c0) * (lr - l0) / (l1 - l0)
            }
        }
    }
}

/// A profit commission: `share` of the reinsurer's profit after an
/// expense allowance, `share × max(1 − LR − allowance, 0)` of the year's
/// premium, with `LR` the ceded loss ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfitCommission {
    pub share: f64,
    /// The reinsurer's allowance for its expenses and margin, as a
    /// fraction of premium; it includes any ceding commission the contract
    /// deducts before profit.
    pub allowance: f64,
}

/// A loss corridor: of the annual layer loss at 100% after the annual
/// deductible, the cedant keeps `retained` of the part between `lower`
/// and `upper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corridor {
    pub lower: f64,
    pub upper: f64,
    /// Share of the band the cedant keeps, in `(0, 1]`.
    pub retained: f64,
}

impl Corridor {
    /// The part of an annual layer loss `x` the cedant keeps.
    fn kept(&self, x: f64) -> f64 {
        self.retained * (x - self.lower).clamp(0.0, self.upper - self.lower)
    }
}

/// What a layer's per-event recovery is figured on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Basis {
    /// The loss: `min(max(x − attachment, 0), limit)`.
    Loss,
    /// A surplus treaty with retention line `retention` and `lines` lines:
    /// a loss `x` on a risk with sum insured `SI` cedes the share
    /// `min(max(SI − retention, 0), lines × retention) / SI` of `x` (the
    /// layer's `attachment` and `limit`, an event limit, then apply to
    /// that). Needs each event's sum insured.
    Surplus { retention: f64, lines: f64 },
}

impl Basis {
    /// The amount a layer's per-event terms apply to, for a loss on a risk
    /// with sum insured `si` (NaN for a surplus without one).
    fn amount(self, loss: f64, si: Option<f64>) -> f64 {
        match self {
            Basis::Loss => loss,
            Basis::Surplus { retention, lines } => match si {
                Some(si) if si > 0.0 => loss * (si - retention).clamp(0.0, lines * retention) / si,
                _ => f64::NAN,
            },
        }
    }
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
            corridor: None,
            commission: None,
            profit_commission: None,
            swing: None,
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
        self.pro_rata_time = true;
        Ok(self)
    }

    /// Whether the layer needs each event's time (paid reinstatements pro
    /// rata as to time).
    pub fn needs_times(&self) -> bool {
        self.pro_rata_time
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
        if anchors.is_empty() {
            return Err(invalid("anchors", 0.0, "must not be empty"));
        }
        let mut anchors = anchors;
        for &(c, lr) in &anchors {
            if !(c.is_finite() && (0.0..=1.0).contains(&c)) {
                return Err(invalid("commission", c, "must be in [0, 1]"));
            }
            if !(lr.is_finite() && lr >= 0.0) {
                return Err(invalid("loss_ratio", lr, "must be finite and non-negative"));
            }
        }
        anchors.sort_by(|a, b| a.1.total_cmp(&b.1));
        for w in anchors.windows(2) {
            if w[1].1 == w[0].1 {
                return Err(invalid("loss_ratio", w[1].1, "must not repeat"));
            }
            if w[1].0 > w[0].0 {
                return Err(invalid(
                    "commission",
                    w[1].0,
                    "must not rise with the loss ratio",
                ));
            }
        }
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
    fn check_loss_ratio_terms(&self, name: &'static str) -> Result<()> {
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
    pub fn ceded(&self, losses: &[f64]) -> f64 {
        self.share * self.layer_loss(losses, None)
    }

    /// Ceded loss for one year's losses on risks with the given sums
    /// insured, one per loss.
    pub fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> f64 {
        self.share * self.layer_loss(losses, Some(sums_insured))
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
        self.reinstatement_premium_at(losses, None, None)
    }

    /// Reinstatement premium for one year's losses at the given times (the
    /// fraction of the year elapsed, one per loss, in order). Pro rata as
    /// to time, the limit an event uses up is charged at `1 − t`; otherwise
    /// the times are ignored.
    pub fn reinstatement_premium_dated(&self, losses: &[f64], times: &[f64]) -> f64 {
        self.reinstatement_premium_at(losses, None, Some(times))
    }

    fn reinstatement_premium_at(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        times: Option<&[f64]>,
    ) -> f64 {
        if self.reinstatement_rates.is_empty() {
            return 0.0;
        }
        // Rate-weighted limits used between layer losses `a` and `b`.
        let used = |a: f64, b: f64| -> f64 {
            self.reinstatement_rates
                .iter()
                .enumerate()
                .map(|(k, rate)| {
                    let floor = k as f64 * self.limit;
                    rate * ((b - floor).clamp(0.0, self.limit) - (a - floor).clamp(0.0, self.limit))
                })
                .sum()
        };
        if !self.pro_rata_time {
            return self.premium * used(0.0, self.layer_loss(losses, si)) / self.limit;
        }
        let Some(times) = times else {
            return f64::NAN;
        };
        let mut before = 0.0;
        let total: f64 = self
            .cumulative_layer_loss(losses, si)
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
    /// event in turn.
    fn cumulative_layer_loss(&self, losses: &[f64], si: Option<&[f64]>) -> Vec<f64> {
        let mut recovery = 0.0;
        losses
            .iter()
            .enumerate()
            .map(|(e, &x)| {
                recovery += self.recovery_at(x, si.map(|s| s[e]));
                self.after_terms(recovery)
            })
            .collect()
    }

    /// Annual layer loss at 100%, after annual terms.
    fn layer_loss(&self, losses: &[f64], si: Option<&[f64]>) -> f64 {
        let recovery: f64 = losses
            .iter()
            .enumerate()
            .map(|(e, &x)| self.recovery_at(x, si.map(|s| s[e])))
            .sum();
        self.after_terms(recovery)
    }

    /// Ceded loss per event for one year, in the order given.
    ///
    /// Annual terms are used up in that order (losses are taken as
    /// chronological): the deductible absorbs the first recoveries and the
    /// annual limit stops the last ones. Event `k` cedes the increase in
    /// annual ceded loss it causes, so the entries sum to
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
        self.ceded_by_event_si(losses, None)
    }

    fn ceded_by_event_si(&self, losses: &[f64], si: Option<&[f64]>) -> Vec<f64> {
        let mut before = 0.0;
        self.cumulative_layer_loss(losses, si)
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
        self.recovery_at(loss, None)
    }

    /// Per-occurrence recovery at 100%, before annual terms, for a loss on
    /// a risk with sum insured `si`.
    fn recovery_at(&self, loss: f64, si: Option<f64>) -> f64 {
        let amount = self.basis.amount(loss, si);
        if amount.is_nan() {
            // A surplus without a sum insured: `max` would turn NaN into 0.
            return f64::NAN;
        }
        (amount - self.attachment).max(0.0).min(self.limit)
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

/// A reinsurance programme: layers in inuring stages.
///
/// Layers in the same stage see the same losses: the gross losses in the
/// first stage, and in each later stage the losses net of every earlier
/// stage, event by event (see [`Layer::ceded_by_event`]). Within a stage,
/// layers that overlap would both pay.
#[derive(Debug, Clone, PartialEq)]
pub struct Tower {
    pub layers: Vec<Layer>,
    /// Stage of each layer, starting at 0 and non-decreasing.
    pub stages: Vec<usize>,
}

impl Tower {
    /// A tower of `layers` in one stage, all seeing the gross losses;
    /// fails if there are none or two share a name.
    pub fn new(layers: Vec<Layer>) -> Result<Self> {
        Self::inuring(vec![layers])
    }

    /// A tower whose stages inure in order: each stage's layers see the
    /// losses net of all earlier stages. Fails if any stage is empty or
    /// two layers share a name.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    ///
    /// // A 50% quota share inures to the benefit of a 5 xs 5 cover:
    /// // a 30 loss is 15 net of the quota share, so the cover pays 5.
    /// let tower = Tower::inuring(vec![
    ///     vec![Layer::quota_share("QS", 0.5).unwrap()],
    ///     vec![Layer::xol("5x5", 5.0, 5.0).unwrap()],
    /// ])
    /// .unwrap();
    /// assert_eq!(tower.ceded(&[30.0]), [15.0, 5.0]);
    /// ```
    pub fn inuring(stages: Vec<Vec<Layer>>) -> Result<Self> {
        if stages.is_empty() || stages.iter().any(Vec::is_empty) {
            return Err(invalid("layers", 0.0, "must not be empty"));
        }
        let stage_of = stages
            .iter()
            .enumerate()
            .flat_map(|(i, stage)| std::iter::repeat_n(i, stage.len()))
            .collect();
        let layers: Vec<Layer> = stages.into_iter().flatten().collect();
        for (i, layer) in layers.iter().enumerate() {
            if layers[..i].iter().any(|l| l.name == layer.name) {
                return Err(invalid("layers", i as f64, "repeats an earlier layer name"));
            }
        }
        Ok(Self {
            layers,
            stages: stage_of,
        })
    }

    /// Ceded loss of each layer, in tower order, for one year's losses.
    pub fn ceded(&self, losses: &[f64]) -> Vec<f64> {
        self.year(losses, None, None)
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    /// Ceded loss of each layer for one year's losses on risks with the
    /// given sums insured, one per loss. Every stage sees each risk's
    /// original sum insured.
    pub fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> Vec<f64> {
        self.year(losses, Some(sums_insured), None)
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    /// Whether any layer needs each event's sum insured.
    pub fn needs_sums_insured(&self) -> bool {
        self.layers.iter().any(Layer::needs_sums_insured)
    }

    /// Whether any layer needs each event's time.
    pub fn needs_times(&self) -> bool {
        self.layers.iter().any(Layer::needs_times)
    }

    /// Ceded loss and reinstatement premium of each layer for one year.
    fn year(&self, losses: &[f64], si: Option<&[f64]>, times: Option<&[f64]>) -> Vec<(f64, f64)> {
        let mut ceded = Vec::with_capacity(self.layers.len());
        let mut seen = losses.to_vec();
        let last_stage = self.stages.last().copied().unwrap_or(0);
        let mut i = 0;
        while i < self.layers.len() {
            let stage = self.stages[i];
            let end = i + self.stages[i..].iter().take_while(|&&s| s == stage).count();
            let mut stage_by_event = vec![0.0; seen.len()];
            for layer in &self.layers[i..end] {
                ceded.push((
                    layer.share * layer.layer_loss(&seen, si),
                    layer.reinstatement_premium_at(&seen, si, times),
                ));
                if stage < last_stage {
                    for (total, c) in stage_by_event
                        .iter_mut()
                        .zip(layer.ceded_by_event_si(&seen, si))
                    {
                        *total += c;
                    }
                }
            }
            if stage < last_stage {
                for (x, c) in seen.iter_mut().zip(&stage_by_event) {
                    *x -= c;
                }
            }
            i = end;
        }
        ceded
    }

    /// Applies the tower to every simulated year.
    ///
    /// The result has dimensions `["kind", "layer"]` and components
    /// `(gross, ground_up)`, `(ceded, <layer name>)` for each layer, and
    /// `(net, retained)`, kept joint per year, then
    /// `(reinstatement_premium, <layer name>)` for each layer with paid
    /// reinstatements, `(swing_premium, <layer name>)` for each
    /// swing-rated layer, `(ceding_commission, <layer name>)` for each
    /// layer with a flat or sliding commission and `(profit_commission,
    /// <layer name>)` for each with a profit commission. A fixed premium
    /// is the layer's `premium` and gets no component.
    /// `aggregate(&["kind"])` gives gross, total ceded and net (and the
    /// totals of the others); `net = gross - Σ ceded` in every year, so
    /// net is a loss, before any premium or commission.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower, simulate_events};
    /// use prospicio_prob::{Distribution, Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
    ///     10_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let tower = Tower::new(vec![
    ///     Layer::xol("5x5", 5e6, 5e6).unwrap(),
    ///     Layer::xol("15x10", 15e6, 10e6).unwrap(),
    /// ])
    /// .unwrap();
    /// let result = tower.apply(&events).unwrap();
    /// let by_kind = result.aggregate(&["kind"]).unwrap();
    /// assert_eq!(by_kind.n_components(), 3); // gross, ceded, net
    /// ```
    pub fn apply(&self, events: &EventSet) -> Result<PredictiveDistribution> {
        if self.needs_sums_insured() && !events.has_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty needs each event's sum insured: use \
                 EventSet::with_sums_insured or simulate from a risk profile"
                    .into(),
            ));
        }
        if self.needs_times() && !events.has_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need each event's time: use \
                 EventSet::with_times, with_uniform_times or with_seasonal_times"
                    .into(),
            ));
        }
        // The events' seed, stream scheme and samplers (none for years
        // from elsewhere).
        let provenance = events.provenance("reinsurance_tower");
        self.apply_years(
            (0..events.n_sims())
                .map(|i| (events.events(i), events.sums_insured(i), events.times(i))),
            events.n_sims(),
            provenance,
        )
    }

    /// Applies the tower to any predictive distribution, taking each
    /// simulation's total as one aggregate loss: an adverse development
    /// cover or loss portfolio transfer on a reserve bootstrap, a stop-loss
    /// or quota share on premium risk from a GLM or a collective model.
    /// Aggregate terms (stop-loss, quota share, annual deductible and
    /// limit) read as usual; an occurrence layer sees the year's total as
    /// one occurrence, so it acts as an aggregate excess of loss. To cover
    /// some components only, `aggregate` or select them first.
    ///
    /// The result has the components of [`apply`](Self::apply), and keeps
    /// the input's seed in its provenance.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // Reserves by origin in three simulations; an ADC of 50 xs 100.
    /// let reserve = PredictiveDistribution::from_draws(
    ///     vec!["origin".into()],
    ///     vec![vec![KeyValue::Int(2023)], vec![KeyValue::Int(2024)]],
    ///     vec![40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
    ///     Provenance::new("odp_bootstrap"),
    /// )
    /// .unwrap();
    /// let adc = Tower::new(vec![Layer::stop_loss("adc", 50.0, 100.0).unwrap()]).unwrap();
    /// let result = adc.apply_aggregate(&reserve).unwrap();
    /// let ceded = result.marginal(&vec![KeyValue::from("ceded"), KeyValue::from("adc")]).unwrap();
    /// assert_eq!(prospicio_prob::Empirical::draws(&ceded), [0.0, 30.0, 50.0]);
    /// ```
    pub fn apply_aggregate(&self, pd: &PredictiveDistribution) -> Result<PredictiveDistribution> {
        if self.needs_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty works risk by risk; an aggregate loss has no sum insured".into(),
            ));
        }
        if self.needs_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need event times; an aggregate loss has none"
                    .into(),
            ));
        }
        let totals = prospicio_prob::Empirical::draws(pd.total()).to_vec();
        let source = pd.provenance();
        // The input's seed, stream scheme and samplers: the tower draws
        // nothing itself.
        let provenance = Provenance::new("reinsurance_tower")
            .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
            .param("loss", format!("total of {}", source.model))
            .draws_from(source);
        self.apply_years(
            totals.chunks(1).map(|t| (t, None, None)),
            totals.len(),
            provenance,
        )
    }

    /// Indices of the layers that satisfy `f`.
    fn indices(&self, f: impl Fn(&Layer) -> bool) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&i| f(&self.layers[i]))
            .collect()
    }

    fn apply_years<'a>(
        &self,
        years: impl Iterator<Item = (&'a [f64], Option<&'a [f64]>, Option<&'a [f64]>)>,
        n_sims: usize,
        base: Provenance,
    ) -> Result<PredictiveDistribution> {
        let paid: Vec<bool> = self
            .layers
            .iter()
            .map(|l| !l.reinstatement_rates.is_empty())
            .collect();
        // Layers with each term that varies with the year's ceded loss.
        let swing: Vec<usize> = self.indices(|l| l.swing.is_some());
        let commission: Vec<usize> = self.indices(|l| l.commission.is_some());
        let profit: Vec<usize> = self.indices(|l| l.profit_commission.is_some());
        let n_components = self.layers.len()
            + 2
            + paid.iter().filter(|&&p| p).count()
            + swing.len()
            + commission.len()
            + profit.len();
        let mut draws = Vec::with_capacity(n_sims * n_components);
        for (losses, si, times) in years {
            let gross: f64 = losses.iter().sum();
            draws.push(gross);
            let year = self.year(losses, si, times);
            let ceded_total: f64 = year.iter().map(|(c, _)| c).sum();
            draws.extend(year.iter().map(|(c, _)| c));
            draws.push(gross - ceded_total);
            draws.extend(
                year.iter()
                    .zip(&paid)
                    .filter(|(_, p)| **p)
                    .map(|((_, rp), _)| rp),
            );
            let ceded = |i: usize| year[i].0;
            draws.extend(swing.iter().map(|&i| self.layers[i].premium_for(ceded(i))));
            draws.extend(
                commission
                    .iter()
                    .map(|&i| self.layers[i].ceding_commission_for(ceded(i))),
            );
            draws.extend(
                profit
                    .iter()
                    .map(|&i| self.layers[i].profit_commission_for(ceded(i))),
            );
        }

        let key = |kind: &str, layer: &str| -> ComponentKey {
            vec![KeyValue::from(kind), KeyValue::from(layer)]
        };
        let mut components = vec![key("gross", "ground_up")];
        components.extend(self.layers.iter().map(|l| key("ceded", &l.name)));
        components.push(key("net", "retained"));
        components.extend(
            self.layers
                .iter()
                .filter(|l| !l.reinstatement_rates.is_empty())
                .map(|l| key("reinstatement_premium", &l.name)),
        );
        for (kind, layers) in [
            ("swing_premium", &swing),
            ("ceding_commission", &commission),
            ("profit_commission", &profit),
        ] {
            components.extend(layers.iter().map(|&i| key(kind, &self.layers[i].name)));
        }

        let mut provenance = base;
        for (l, stage) in self.layers.iter().zip(&self.stages) {
            let mut terms = format!(
                "{} xs {}, share {}, aad {}, aal {}, stage {stage}",
                l.limit, l.attachment, l.share, l.aggregate_deductible, l.aggregate_limit
            );
            if let Basis::Surplus { retention, lines } = l.basis {
                terms += &format!(", surplus retention {retention}, lines {lines}");
            }
            if !l.reinstatement_rates.is_empty() {
                terms += &format!(
                    ", premium {}, reinstatement rates {:?}",
                    l.premium, l.reinstatement_rates
                );
                if l.pro_rata_time {
                    terms += ", pro rata as to time";
                }
            }
            if l.reinstatement_rates.is_empty() && l.premium > 0.0 {
                terms += &format!(", premium {}", l.premium);
            }
            if let Some(s) = &l.swing {
                terms += &format!(
                    ", swing basic {} lcm {} min {} max {} at 100%",
                    s.basic, s.lcm, s.minimum, s.maximum
                );
            }
            match &l.commission {
                Some(Commission::Flat(c)) => terms += &format!(", ceding commission {c}"),
                Some(Commission::SlidingScale(a)) => {
                    terms += &format!(", sliding scale (commission, loss ratio) {a:?}");
                }
                None => {}
            }
            if let Some(pc) = &l.profit_commission {
                terms += &format!(", profit commission {} after {}", pc.share, pc.allowance);
            }
            provenance = provenance.param(format!("layer:{}", l.name), terms);
        }
        PredictiveDistribution::from_draws(
            vec!["kind".into(), "layer".into()],
            components,
            draws,
            provenance,
        )
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate_events;
    use prospicio_prob::provenance::SIM_INDEX_SCHEME;
    use prospicio_prob::{Distribution, Lognormal, Poisson, Severity};

    #[test]
    fn the_seven_million_example() {
        // Retain 5m; 5m xs 5m; 15m xs 10m. A 7m loss: 5m kept, 2m to layer 2.
        let l2 = Layer::xol("5x5", 5e6, 5e6).unwrap();
        let l3 = Layer::xol("15x10", 15e6, 10e6).unwrap();
        assert_eq!(l2.ceded(&[7e6]), 2e6);
        assert_eq!(l3.ceded(&[7e6]), 0.0);
        // A 30m loss exhausts both layers and leaves 5m + 5m retained.
        assert_eq!(l2.ceded(&[30e6]), 5e6);
        assert_eq!(l3.ceded(&[30e6]), 15e6);
    }

    #[test]
    fn annual_terms() {
        let base = || Layer::xol("L", 10.0, 5.0).unwrap();
        let losses = [8.0, 20.0, 12.0]; // recoveries 3, 10, 7 = 20
        assert_eq!(base().ceded(&losses), 20.0);
        assert_eq!(
            base().aggregate_deductible(4.0).unwrap().ceded(&losses),
            16.0
        );
        assert_eq!(base().aggregate_limit(15.0).unwrap().ceded(&losses), 15.0);
        assert_eq!(base().reinstatements(0).unwrap().ceded(&losses), 10.0);
        assert_eq!(base().share(0.25).unwrap().ceded(&losses), 5.0);
        let all = base()
            .aggregate_deductible(4.0)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap()
            .share(0.5)
            .unwrap();
        assert_eq!(all.ceded(&losses), 7.5);
        assert_eq!(base().ceded(&[]), 0.0);
    }

    #[test]
    fn loss_corridors() {
        let base = || Layer::xol("L", 10.0, 5.0).unwrap();
        let losses = [8.0, 20.0, 12.0]; // recoveries 3, 10, 7 = 20
        // After the deductible of 4, 16; the corridor keeps half of 5..9,
        // so 14; the annual limit of 12 then caps it.
        let layer = base()
            .aggregate_deductible(4.0)
            .unwrap()
            .loss_corridor(5.0, 9.0, 0.5)
            .unwrap()
            .aggregate_limit(12.0)
            .unwrap();
        assert_eq!(layer.ceded(&losses), 12.0);
        assert_eq!(layer.clone().share(0.5).unwrap().ceded(&losses), 6.0);
        // The reinsurer still pays up to the full annual limit.
        let wide = base()
            .aggregate_deductible(4.0)
            .unwrap()
            .loss_corridor(5.0, 9.0, 0.5)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap();
        assert_eq!(wide.ceded(&losses), 14.0);
        // Used up in event order: after the deductible 0, 9, 16, after the
        // corridor 0, 7, 14, after the limit 0, 7, 12.
        assert_eq!(layer.ceded_by_event(&losses), [0.0, 7.0, 5.0]);
        assert!(layer.has_annual_terms());
        assert!(
            base()
                .loss_corridor(0.0, 1.0, 1.0)
                .unwrap()
                .has_annual_terms()
        );
        assert!(!base().has_annual_terms());

        // Reinstatement premiums follow the loss after the corridor:
        // recovery 2, of which the corridor keeps 1.
        let paid = || {
            Layer::xol("10x10", 10.0, 10.0)
                .unwrap()
                .paid_reinstatements(2.0, vec![1.0])
                .unwrap()
        };
        assert!((paid().reinstatement_premium(&[12.0]) - 0.4).abs() < 1e-12);
        let kept = paid().loss_corridor(0.0, 1.0, 1.0).unwrap();
        assert!((kept.reinstatement_premium(&[12.0]) - 0.2).abs() < 1e-12);

        assert!(base().loss_corridor(-1.0, 1.0, 1.0).is_err());
        assert!(base().loss_corridor(2.0, 2.0, 1.0).is_err());
        assert!(base().loss_corridor(1.0, f64::INFINITY, 1.0).is_err());
        assert!(base().loss_corridor(1.0, 2.0, 0.0).is_err());
        assert!(base().loss_corridor(1.0, 2.0, 1.5).is_err());
    }

    #[test]
    fn corridor_keeps_gross_equal_to_ceded_plus_net() {
        let tower = Tower::inuring(vec![
            vec![
                Layer::quota_share("QS", 0.4)
                    .unwrap()
                    .loss_corridor(2e6, 6e6, 0.5)
                    .unwrap(),
            ],
            vec![
                Layer::xol("5x5", 5e6, 5e6)
                    .unwrap()
                    .loss_corridor(2e6, 8e6, 1.0)
                    .unwrap()
                    .reinstatements(2)
                    .unwrap(),
            ],
        ])
        .unwrap();
        let result = tower.apply(&events()).unwrap();
        let mut kept = 0;
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
            // The QS cedes 40% of the year's loss less half its corridor.
            let gross = row[0];
            let want = 0.4 * (gross - 0.5 * (gross - 2e6).clamp(0.0, 4e6));
            assert!((row[1] - want).abs() <= 1e-6 * gross.max(1.0));
            kept += usize::from(gross > 2e6);
        }
        assert!(kept > 1_000);
    }

    #[test]
    fn rejects_bad_terms() {
        assert!(Layer::xol("L", 0.0, 1.0).is_err());
        assert!(Layer::xol("L", 1.0, -1.0).is_err());
        assert!(Layer::xol("L", 1.0, 0.0).unwrap().share(1.5).is_err());
        assert!(
            Layer::xol("L", 1.0, 0.0)
                .unwrap()
                .aggregate_limit(0.0)
                .is_err()
        );
        assert!(Tower::new(vec![]).is_err());
        let l = Layer::xol("L", 1.0, 0.0).unwrap();
        assert!(Tower::new(vec![l.clone(), l]).is_err());
    }

    #[test]
    fn premiums_from_quotes() {
        let xl = || Layer::xol("10x10", 10.0, 10.0).unwrap().share(0.4).unwrap();
        assert!((xl().rate_on_line(0.2).unwrap().premium - 0.8).abs() < 1e-12);
        assert!((xl().deposit_premium(3.0).unwrap().premium - 1.2).abs() < 1e-12);
        assert!((xl().premium_rate(0.05, 50.0).unwrap().premium - 1.0).abs() < 1e-12);
        assert!(
            Layer::quota_share("QS", 0.3)
                .unwrap()
                .rate_on_line(0.1)
                .is_err()
        );
        assert!(xl().rate_on_line(-0.1).is_err());
        assert!(xl().deposit_premium(f64::NAN).is_err());
        assert!(xl().premium_rate(0.1, f64::INFINITY).is_err());
    }

    #[test]
    fn commissions_on_hand_worked_years() {
        // A 50% quota share of a subject premium of 200: premium 100.
        let qs = || {
            Layer::quota_share("QS", 0.5)
                .unwrap()
                .premium_rate(1.0, 200.0)
                .unwrap()
        };
        let flat = qs().ceding_commission(0.3).unwrap();
        assert!((flat.ceding_commission_for(0.0) - 30.0).abs() < 1e-12);
        assert!((flat.ceding_commission_for(500.0) - 30.0).abs() < 1e-12);
        assert_eq!(flat.premium_for(500.0), 100.0);
        assert_eq!(flat.profit_commission_for(10.0), 0.0);

        // aggregate's worked example: 45% at 60%, 25% at 70%, 19% at 80%,
        // given out of order.
        let slide = qs()
            .sliding_scale(vec![(0.25, 0.70), (0.45, 0.60), (0.19, 0.80)])
            .unwrap();
        for (lr, c) in [
            (0.55, 0.45),
            (0.60, 0.45),
            (0.65, 0.35),
            (0.70, 0.25),
            (0.75, 0.22),
            (0.80, 0.19),
            (0.90, 0.19),
        ] {
            let got = slide.ceding_commission_for(100.0 * lr);
            assert!((got - 100.0 * c).abs() < 1e-10, "{lr}: {got}");
        }
        // A one-anchor scale is flat.
        let one = qs().sliding_scale(vec![(0.3, 0.6)]).unwrap();
        assert!((one.ceding_commission_for(90.0) - 30.0).abs() < 1e-12);
        // A sliding scale replaces a flat commission.
        let replaced = qs()
            .ceding_commission(0.1)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.6)])
            .unwrap();
        assert!((replaced.ceding_commission_for(10.0) - 30.0).abs() < 1e-12);

        // aggregate's profit commission example: 25% after 10%.
        let pc = qs().profit_commission(0.25, 0.10).unwrap();
        for (lr, want) in [(0.50, 0.10), (0.60, 0.075), (0.90, 0.0), (0.95, 0.0)] {
            assert!((pc.profit_commission_for(100.0 * lr) - 100.0 * want).abs() < 1e-10);
        }
        // With a flat commission too: each is figured on its own.
        let both = qs()
            .ceding_commission(0.3)
            .unwrap()
            .profit_commission(0.25, 0.4)
            .unwrap();
        assert!((both.ceding_commission_for(40.0) - 30.0).abs() < 1e-12);
        assert!((both.profit_commission_for(40.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn swing_and_retro_premiums() {
        // aggregate's swing example: basic 0, lcm 1, between 100 and 300.
        let terms = LossSensitivePremium::new(0.0, 1.0, Some(100.0), Some(300.0)).unwrap();
        assert_eq!(terms.premium(50.0), 100.0);
        assert_eq!(terms.premium(200.0), 200.0);
        assert_eq!(terms.premium(400.0), 300.0);
        // The minimum defaults to the basic premium, the maximum to no cap.
        let retro = LossSensitivePremium::new(1000.0, 1.1, None, None).unwrap();
        assert_eq!(retro.minimum, 1000.0);
        assert!((retro.premium(1e6) - 1_101_000.0).abs() < 1e-6);
        // At a placed share of 40%, the amounts scale and the factor does
        // not: clip(0.4 × 20 + 1.25 × ceded, 0.4 × 50, 0.4 × 300).
        let swing = LossSensitivePremium::new(20.0, 1.25, Some(50.0), Some(300.0)).unwrap();
        let layer = Layer::xol("L", 1000.0, 0.0)
            .unwrap()
            .share(0.4)
            .unwrap()
            .swing_rated(swing)
            .unwrap();
        assert!((layer.premium_for(0.0) - 20.0).abs() < 1e-12);
        assert!((layer.premium_for(40.0) - 58.0).abs() < 1e-12);
        assert!((layer.premium_for(1000.0) - 120.0).abs() < 1e-12);
        // A flat commission applies to the swing premium.
        let comm = layer.clone().ceding_commission(0.1).unwrap();
        assert!((comm.ceding_commission_for(40.0) - 5.8).abs() < 1e-12);

        assert!(LossSensitivePremium::new(-1.0, 1.0, None, None).is_err());
        assert!(LossSensitivePremium::new(1.0, f64::NAN, None, None).is_err());
        assert!(LossSensitivePremium::new(1.0, 1.0, Some(5.0), Some(4.0)).is_err());
        assert!(LossSensitivePremium::new(1.0, 1.0, Some(f64::INFINITY), None).is_err());
    }

    #[test]
    fn refuses_terms_that_do_not_combine() {
        let qs = || Layer::quota_share("QS", 0.5).unwrap();
        let priced = || qs().premium_rate(1.0, 200.0).unwrap();
        let swing = LossSensitivePremium::new(0.0, 1.0, None, None).unwrap();
        // A loss ratio needs a premium.
        assert!(qs().sliding_scale(vec![(0.3, 0.6)]).is_err());
        assert!(qs().profit_commission(0.2, 0.1).is_err());
        // Swing rating against loss-ratio terms and paid reinstatements, in
        // either order.
        assert!(
            priced()
                .swing_rated(swing)
                .unwrap()
                .sliding_scale(vec![(0.3, 0.6)])
                .is_err()
        );
        assert!(
            priced()
                .sliding_scale(vec![(0.3, 0.6)])
                .unwrap()
                .swing_rated(swing)
                .is_err()
        );
        assert!(
            priced()
                .profit_commission(0.2, 0.1)
                .unwrap()
                .swing_rated(swing)
                .is_err()
        );
        let xl = || Layer::xol("10x10", 10.0, 10.0).unwrap();
        assert!(
            xl().paid_reinstatements(1.0, vec![1.0])
                .unwrap()
                .swing_rated(swing)
                .is_err()
        );
        assert!(
            xl().swing_rated(swing)
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0])
                .is_err()
        );
        assert!(
            xl().paid_reinstatements(1.0, vec![1.0])
                .unwrap()
                .profit_commission(0.2, 0.1)
                .is_err()
        );
        assert!(
            xl().rate_on_line(0.1)
                .unwrap()
                .sliding_scale(vec![(0.3, 0.6)])
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0])
                .is_err()
        );
        // A flat commission combines with paid reinstatements and swing.
        assert!(
            xl().paid_reinstatements(1.0, vec![1.0])
                .unwrap()
                .ceding_commission(0.1)
                .is_ok()
        );
        assert!(
            xl().swing_rated(swing)
                .unwrap()
                .ceding_commission(0.1)
                .is_ok()
        );
        // Bad values.
        assert!(priced().ceding_commission(1.5).is_err());
        assert!(priced().sliding_scale(vec![]).is_err());
        assert!(
            priced()
                .sliding_scale(vec![(0.2, 0.6), (0.3, 0.7)])
                .is_err()
        );
        assert!(
            priced()
                .sliding_scale(vec![(0.3, 0.6), (0.2, 0.6)])
                .is_err()
        );
        assert!(priced().sliding_scale(vec![(1.2, 0.6)]).is_err());
        assert!(priced().profit_commission(1.5, 0.1).is_err());
        assert!(priced().profit_commission(0.2, -0.1).is_err());
    }

    #[test]
    fn tower_reports_contract_terms() {
        let swing = LossSensitivePremium::new(2e5, 1.25, Some(4e5), Some(3e6)).unwrap();
        let tower = Tower::inuring(vec![
            vec![
                Layer::quota_share("QS", 0.3)
                    .unwrap()
                    .premium_rate(1.0, 20e6)
                    .unwrap()
                    .sliding_scale(vec![(0.35, 0.5), (0.25, 0.7)])
                    .unwrap()
                    .profit_commission(0.2, 0.3)
                    .unwrap(),
            ],
            vec![
                Layer::xol("5x5", 5e6, 5e6)
                    .unwrap()
                    .swing_rated(swing)
                    .unwrap(),
            ],
        ])
        .unwrap();
        let result = tower.apply(&events()).unwrap();
        let kinds: Vec<String> = result
            .components()
            .iter()
            .map(|k| format!("{}/{}", k[0], k[1]))
            .collect();
        assert_eq!(
            kinds,
            [
                "gross/ground_up",
                "ceded/QS",
                "ceded/5x5",
                "net/retained",
                "swing_premium/5x5",
                "ceding_commission/QS",
                "profit_commission/QS"
            ]
        );
        let (qs, xl) = (&tower.layers[0], &tower.layers[1]);
        let mut slid = 0;
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
            assert_eq!(row[4], xl.premium_for(row[2]));
            assert_eq!(row[5], qs.ceding_commission_for(row[1]));
            assert_eq!(row[6], qs.profit_commission_for(row[1]));
            let lr = row[1] / 6e6;
            slid += usize::from(lr > 0.5 && lr < 0.7);
        }
        // The scale slides in some years, not only at its ends.
        assert!(slid > 1_000, "{slid}");
        assert!(
            result
                .provenance()
                .parameters
                .iter()
                .any(|(k, v)| k == "layer:QS" && v.contains("sliding scale"))
        );
    }

    fn events() -> EventSet {
        simulate_events(
            &Poisson::new(2.0).unwrap(),
            &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
            100_000,
            11,
        )
        .unwrap()
    }

    #[test]
    fn gross_equals_ceded_plus_net_every_year() {
        let tower = Tower::new(vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .reinstatements(1)
                .unwrap(),
            Layer::xol("15x10", 15e6, 10e6).unwrap().share(0.6).unwrap(),
        ])
        .unwrap();
        let result = tower.apply(&events()).unwrap();
        assert_eq!(result.n_components(), 4);
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        }
        let by_kind = result.aggregate(&["kind"]).unwrap();
        let kinds: Vec<_> = by_kind
            .components()
            .iter()
            .map(|k| k[0].to_string())
            .collect();
        assert_eq!(kinds, ["gross", "ceded", "net"]);
        assert_eq!(result.provenance().seed, Some(11));
    }

    #[test]
    fn mean_ceded_matches_the_exact_layer_value() {
        // With no annual terms, E[ceded] = E[N] × E[min(max(X - a, 0), l)].
        let sev = Lognormal::from_mean_cv(3e6, 1.5).unwrap();
        let layer = Layer::xol("5x5", 5e6, 5e6).unwrap();
        let exact = 2.0 * sev.layer(5e6, 5e6);
        let result = Tower::new(vec![layer]).unwrap().apply(&events()).unwrap();
        let ceded = result
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("5x5")])
            .unwrap();
        // Four standard errors.
        let se = ceded.std_dev() / (ceded.len() as f64).sqrt();
        assert!(
            (ceded.mean() - exact).abs() < 4.0 * se,
            "{} vs {exact}",
            ceded.mean()
        );
    }

    #[test]
    fn quota_share_and_stop_loss() {
        let qs = Layer::quota_share("QS", 0.25).unwrap();
        assert_eq!(qs.ceded(&[8.0, 4.0]), 3.0);
        assert!(Layer::quota_share("QS", 0.0).is_err());
        let sl = Layer::stop_loss("SL", 50.0, 100.0).unwrap();
        assert_eq!(sl.ceded(&[60.0]), 0.0);
        assert_eq!(sl.ceded(&[60.0, 70.0]), 30.0);
        assert_eq!(sl.ceded(&[200.0]), 50.0);
        assert!(Layer::stop_loss("SL", 50.0, f64::INFINITY).is_err());
    }

    #[test]
    fn ceded_by_event_uses_annual_terms_in_order() {
        let layer = Layer::xol("L", 10.0, 5.0)
            .unwrap()
            .aggregate_deductible(4.0)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap()
            .share(0.5)
            .unwrap();
        // Recoveries 3, 10, 7: the deductible takes 3 then 1, the limit
        // stops the last 5.
        assert_eq!(layer.ceded_by_event(&[8.0, 20.0, 12.0]), [0.0, 4.5, 3.0]);
        // The same losses in another order use the terms up differently
        // but cede the same annual total.
        assert_eq!(layer.ceded_by_event(&[20.0, 12.0, 8.0]), [3.0, 3.5, 1.0]);
        assert!(layer.ceded_by_event(&[]).is_empty());
        let big = Layer::xol("5x5", 5e6, 5e6)
            .unwrap()
            .aggregate_deductible(2e6)
            .unwrap()
            .reinstatements(1)
            .unwrap();
        let events = events();
        for sim in 0..events.n_sims() {
            let losses = events.events(sim);
            let split: f64 = big.ceded_by_event(losses).iter().sum();
            assert!((split - big.ceded(losses)).abs() <= 1e-6);
        }
    }

    #[test]
    fn later_stages_see_losses_net_of_earlier_ones() {
        // Stage 1: 10 xs 5 with a 15 annual limit; on [20, 20] it pays 10
        // then 5, leaving 10 and 15. Stage 2: a stop-loss of 20 xs 20 on
        // that net total of 25 pays 5. A layer in stage 1 alongside it
        // still sees gross.
        let tower = Tower::inuring(vec![
            vec![
                Layer::xol("A", 10.0, 5.0)
                    .unwrap()
                    .aggregate_limit(15.0)
                    .unwrap(),
                Layer::xol("B", 100.0, 18.0).unwrap(),
            ],
            vec![Layer::stop_loss("SL", 20.0, 20.0).unwrap()],
        ])
        .unwrap();
        assert_eq!(tower.stages, [0, 0, 1]);
        // B takes 2 + 2 of the gross, so stage 2 sees 8 + 13 = 21.
        assert_eq!(tower.ceded(&[20.0, 20.0]), [15.0, 4.0, 1.0]);
        assert!(Tower::inuring(vec![vec![], vec![Layer::xol("A", 1.0, 0.0).unwrap()]]).is_err());
        assert!(
            Tower::inuring(vec![
                vec![Layer::xol("A", 1.0, 0.0).unwrap()],
                vec![Layer::xol("A", 2.0, 0.0).unwrap()],
            ])
            .is_err()
        );
    }

    #[test]
    fn quota_share_inuring_to_a_layer_scales_it() {
        // A cession c inuring to l xs a is (1 - c) × (l / (1 - c)) xs
        // (a / (1 - c)) on gross, in every simulated year.
        let c = 0.4;
        let inuring = Tower::inuring(vec![
            vec![Layer::quota_share("QS", c).unwrap()],
            vec![
                Layer::xol("XL", 5e6, 5e6)
                    .unwrap()
                    .reinstatements(1)
                    .unwrap(),
            ],
        ])
        .unwrap();
        let scaled = Layer::xol("XL", 5e6 / (1.0 - c), 5e6 / (1.0 - c))
            .unwrap()
            .reinstatements(1)
            .unwrap()
            .share(1.0 - c)
            .unwrap();
        let events = events();
        for sim in 0..events.n_sims() {
            let losses = events.events(sim);
            let ceded = inuring.ceded(losses);
            assert!((ceded[0] - c * losses.iter().sum::<f64>()).abs() <= 1e-6);
            assert!(
                (ceded[1] - scaled.ceded(losses)).abs() <= 1e-6,
                "year {sim}"
            );
        }
        let result = inuring.apply(&events).unwrap();
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        }
    }

    #[test]
    fn reinstatement_premiums_pro_rata_as_to_time() {
        let amount = Layer::xol("10x10", 10.0, 10.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0, 0.5])
            .unwrap();
        let timed = amount.clone().pro_rata_as_to_time().unwrap();
        assert!(timed.needs_times() && !amount.needs_times());
        // Without times: NaN; times are ignored when only pro rata as to
        // amount.
        assert!(timed.reinstatement_premium(&[22.0]).is_nan());
        assert_eq!(
            amount.reinstatement_premium_dated(&[22.0, 12.0], &[0.25, 0.5]),
            amount.reinstatement_premium(&[22.0, 12.0])
        );
        // Recoveries 5 at t = 0.4, then 10 at t = 0.7, which uses the rest
        // of the first limit and 5 of the second.
        let rp = timed.reinstatement_premium_dated(&[15.0, 20.0], &[0.4, 0.7]);
        let want = 2.0 * (1.0 * 5.0 * 0.6 + (1.0 * 5.0 + 0.5 * 5.0) * 0.3) / 10.0;
        assert!((rp - want).abs() < 1e-12, "{rp} vs {want}");
        // At t = 0 it is the amount-only premium.
        assert_eq!(
            timed.reinstatement_premium_dated(&[15.0, 25.0], &[0.0, 0.0]),
            amount.reinstatement_premium(&[15.0, 25.0])
        );
        // The annual deductible absorbs the first event, which uses no limit.
        let aad = timed.clone().aggregate_deductible(5.0).unwrap();
        let rp = aad.reinstatement_premium_dated(&[15.0, 25.0], &[0.1, 0.5]);
        assert!((rp - 2.0 * 0.5 * 1.0).abs() < 1e-12);
        assert!(
            Layer::xol("free", 10.0, 10.0)
                .unwrap()
                .reinstatements(1)
                .unwrap()
                .pro_rata_as_to_time()
                .is_err()
        );

        // One loss a year exhausting the first limit, at a uniform time: the
        // premium averages 2 × E[1 − t] = 1.
        let n = 20_000;
        let events = EventSet::from_years(vec![vec![20.0]; n], 9).unwrap();
        let one = Layer::xol("10x10", 10.0, 10.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0])
            .unwrap()
            .pro_rata_as_to_time()
            .unwrap();
        let tower = Tower::inuring(vec![
            vec![Layer::quota_share("QS", 0.0001).unwrap()],
            vec![one],
        ])
        .unwrap();
        assert!(tower.apply(&events).is_err());
        assert!(tower.apply_aggregate(&events.totals().unwrap()).is_err());
        let pd = tower.apply(&events.clone().with_uniform_times()).unwrap();
        let rp = pd
            .marginal(&vec![
                KeyValue::from("reinstatement_premium"),
                KeyValue::from("10x10"),
            ])
            .unwrap();
        let se = (rp.variance() / n as f64).sqrt();
        assert!((rp.mean() - 1.0).abs() < 4.0 * se, "{} ± {se}", rp.mean());
        assert!((rp.variance() / (4.0 / 12.0) - 1.0).abs() < 0.05);

        // The same losses in the second half of the year only: t is uniform
        // on [0.5, 1], so the premium averages 2 × 0.25 = 0.5 with variance
        // 4 × 0.5² / 12.
        let pd = tower
            .apply(&events.with_seasonal_times(&[0.0, 1.0]).unwrap())
            .unwrap();
        let rp = pd
            .marginal(&vec![
                KeyValue::from("reinstatement_premium"),
                KeyValue::from("10x10"),
            ])
            .unwrap();
        let se = (rp.variance() / n as f64).sqrt();
        assert!((rp.mean() - 0.5).abs() < 4.0 * se, "{} ± {se}", rp.mean());
        assert!((rp.variance() / (4.0 * 0.25 / 12.0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn reinstatement_premiums() {
        let layer = Layer::xol("10x10", 10.0, 10.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0, 0.5])
            .unwrap();
        // Recoveries 5 + 10 = 15: the first limit and 5 of the second,
        // so 2 × (1.0 × 10 + 0.5 × 5) / 10 = 2.5.
        assert_eq!(layer.reinstatement_premium(&[15.0, 25.0]), 2.5);
        // Exhausted (30 = three limits): both reinstatements in full.
        assert_eq!(layer.reinstatement_premium(&[40.0, 40.0, 40.0]), 3.0);
        assert_eq!(layer.reinstatement_premium(&[5.0]), 0.0);
        // The share scales the loss, not the premium, which is already
        // for the placed share.
        let half = layer.clone().share(0.5).unwrap();
        assert_eq!(half.reinstatement_premium(&[15.0, 25.0]), 2.5);
        // The annual deductible comes off first.
        let aad = layer.clone().aggregate_deductible(5.0).unwrap();
        assert_eq!(aad.reinstatement_premium(&[15.0, 25.0]), 2.0);
        assert_eq!(
            Layer::xol("F", 10.0, 0.0)
                .unwrap()
                .reinstatements(2)
                .unwrap()
                .reinstatement_premium(&[30.0]),
            0.0
        );
        assert!(
            Layer::quota_share("QS", 0.5)
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0])
                .is_err()
        );
        assert!(
            Layer::xol("L", 1.0, 0.0)
                .unwrap()
                .paid_reinstatements(-1.0, vec![])
                .is_err()
        );
        assert!(
            Layer::xol("L", 1.0, 0.0)
                .unwrap()
                .paid_reinstatements(1.0, vec![f64::NAN])
                .is_err()
        );
    }

    #[test]
    fn tower_reports_reinstatement_premiums() {
        let tower = Tower::new(vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .paid_reinstatements(1e6, vec![1.0])
                .unwrap(),
            Layer::xol("15x10", 15e6, 10e6).unwrap(),
        ])
        .unwrap();
        let events = events();
        let result = tower.apply(&events).unwrap();
        let kinds: Vec<String> = result
            .components()
            .iter()
            .map(|k| k[0].to_string())
            .collect();
        assert_eq!(
            kinds,
            ["gross", "ceded", "ceded", "net", "reinstatement_premium"]
        );
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            // One reinstatement at 100%: premium × min(ceded, limit) / limit.
            let expected = 1e6 * row[1].min(5e6) / 5e6;
            assert!((row[4] - expected).abs() <= 1e-9 * expected.max(1.0));
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        }
    }

    #[test]
    fn surplus_cedes_by_sum_insured_and_inures_to_a_per_risk_xl() {
        // Retention 1m, 4 lines: capacity to 5m of sum insured.
        let surplus = Layer::surplus("surplus", 1e6, 4.0).unwrap();
        let si = [0.5e6, 2e6, 5e6, 10e6];
        let losses = [0.5e6, 2e6, 1e6, 10e6];
        // Cessions 0, 1/2, 4/5, 2/5.
        let by_risk: Vec<f64> = (0..4)
            .map(|i| surplus.ceded_with_sums_insured(&losses[i..=i], &si[i..=i]))
            .collect();
        for (got, want) in by_risk.iter().zip([0.0, 1e6, 0.8e6, 4e6]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // A 1m xs 1m per-risk XL sees each loss net of the surplus.
        let tower = Tower::inuring(vec![
            vec![surplus.clone()],
            vec![Layer::xol("1x1", 1e6, 1e6).unwrap()],
        ])
        .unwrap();
        let ceded = tower.ceded_with_sums_insured(&losses, &si);
        // Nets 0.5m, 1m, 0.2m, 6m: only the last reaches the XL, for 1m.
        assert!((ceded[0] - 5.8e6).abs() < 1e-6);
        assert!((ceded[1] - 1e6).abs() < 1e-6);
        assert!(tower.needs_sums_insured());
        assert!(tower.ceded(&losses)[0].is_nan());
    }

    #[test]
    fn surplus_needs_sums_insured_on_events() {
        let tower = Tower::new(vec![Layer::surplus("s", 1e6, 4.0).unwrap()]).unwrap();
        let events = EventSet::from_years(vec![vec![2e6], vec![]], 0).unwrap();
        assert!(tower.apply(&events).is_err());
        let events = events.with_sums_insured(vec![2e6]).unwrap();
        let result = tower.apply(&events).unwrap();
        let ceded = result
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("s")])
            .unwrap();
        assert_eq!(prospicio_prob::Empirical::draws(&ceded), [1e6, 0.0]);
        let pd = PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            vec![1.0, 2.0],
            Provenance::new("test"),
        )
        .unwrap();
        assert!(tower.apply_aggregate(&pd).is_err());
        assert!(Layer::surplus("s", 0.0, 4.0).is_err());
        assert!(Layer::surplus("s", 1.0, f64::INFINITY).is_err());
        let e = EventSet::from_years(vec![vec![5.0]], 0).unwrap();
        assert!(e.clone().with_sums_insured(vec![4.0]).is_err());
        assert!(e.with_sums_insured(vec![5.0, 6.0]).is_err());
    }

    #[test]
    fn apply_aggregate_keeps_the_input_streams_and_samplers() {
        let tower = Tower::new(vec![Layer::stop_loss("adc", 50.0, 100.0).unwrap()]).unwrap();
        let mut source = Provenance::new("odp_bootstrap").seed(4, SIM_INDEX_SCHEME);
        source.samplers = None; // read from a file saved before samplers were recorded
        let pd = PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            vec![90.0, 120.0],
            source.clone(),
        )
        .unwrap();
        let p = tower.apply_aggregate(&pd).unwrap().provenance().clone();
        assert_eq!(p.seed, Some(4));
        assert_eq!(p.stream_scheme.as_deref(), Some(SIM_INDEX_SCHEME));
        assert_eq!(p.samplers, None);
        assert!(p.shares_streams(&source));
    }

    #[test]
    fn apply_records_the_samplers_only_of_events_drawn_here() {
        let tower = Tower::new(vec![Layer::xol("5x5", 5.0, 5.0).unwrap()]).unwrap();
        let external = EventSet::from_years(vec![vec![7.0], vec![12.0]], 0).unwrap();
        let p = tower.apply(&external).unwrap().provenance().clone();
        assert_eq!(p.seed, Some(0));
        assert_eq!(p.stream_scheme.as_deref(), Some(SIM_INDEX_SCHEME));
        assert_eq!(p.samplers, None);
        let other = EventSet::from_years(vec![vec![1.0], vec![]], 0).unwrap();
        let q = tower.apply(&other).unwrap().provenance().clone();
        assert!(!p.replays_same_draws(&q));

        let drawn = simulate_events(
            &Poisson::new(1.0).unwrap(),
            &Lognormal::from_mean_cv(6.0, 1.0).unwrap(),
            20,
            0,
        )
        .unwrap();
        let d = tower.apply(&drawn).unwrap().provenance().clone();
        assert_eq!(
            d.samplers,
            Some(prospicio_prob::provenance::current_samplers())
        );
        assert!(d.shares_streams(&p) && !d.replays_same_draws(&p));
    }
}
