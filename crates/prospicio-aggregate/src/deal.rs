//! A deal's deductions and its ratios (#485): brokerage on a layer's
//! premium, flat or on a scale of the loss ratio, the other deductions a
//! reinsurer takes from the premium (a TPA fee, tax, reinsurance cost, a
//! fronting fee, a manager's fee), and, simulation by simulation, the
//! layer's loss ratio, each deduction, their total, the combined ratio,
//! the margin and the nominal margin before the fees kept outside it.
//!
//! With `P` the year's premium for the placed share, `L` the ceded loss,
//! and every deduction `d_k` divided by `P`:
//!
//! ```text
//! loss_ratio       = L / P
//! total_deductions = CC + PC + BK + Σ d_k
//! combined_ratio   = loss_ratio + total_deductions
//! margin           = 1 − combined_ratio
//! nmbf             = margin + Σ d_k outside NMBF
//! ```
//!
//! `CC` is the ceding commission (flat or sliding), `PC` the profit
//! commission and `BK` the brokerage. NMBF, the nominal margin before
//! fees, counts every deduction except those marked
//! [`outside_nmbf`](Deduction::outside_nmbf), typically a manager's fee.
//!
//! A [`DealScale`] reads a commission and a brokerage on the loss ratio of
//! several layers together, the sections of one deal, on top of their own;
//! [`Tower::deal_ratios`] then takes the deal's name and sums its sections.
//! See `docs/design/aggregate.md`, "Deal deductions and ratios".

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution};

use crate::pnl::{ExpenseBasis, Group, Layout, Ledger, Leg, LegKind, Source};
use crate::reinsurance::{CarryForward, Commission, Layer, Tower, checked_anchors};

/// Kinds a tower's result already uses, and the names of the deal ratios:
/// a deduction may not take one as its label.
const RESERVED: &[&str] = &[
    "gross",
    "ceded",
    "net",
    "reinstatement_premium",
    "swing_premium",
    "ceding_commission",
    "profit_commission",
    "brokerage",
    "loss_before_terms",
    "loss_ratio",
    "loss_ratio_before_terms",
    "total_deductions",
    "combined_ratio",
    "nmbf",
    "margin",
];

/// A deduction from a layer's premium: a rate of the year's premium or of
/// its ceded loss, or an amount, booked under its label.
///
/// ```
/// use prospicio_aggregate::Deduction;
///
/// let tax = Deduction::premium("tax", 0.03).unwrap();
/// assert!(tax.inside_nmbf);
/// let fee = Deduction::premium("manager_fee", 0.05).unwrap().outside_nmbf();
/// assert!(!fee.inside_nmbf);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Deduction {
    /// The label: the deduction's `kind` in a tower's result and its name
    /// among the deal ratios.
    pub label: String,
    /// What the value is a fraction of: the year's premium for the placed
    /// share, its ceded loss, or nothing (an amount).
    pub basis: ExpenseBasis,
    /// The rate, or the amount for [`ExpenseBasis::Fixed`].
    pub value: f64,
    /// Whether the nominal margin before fees counts it; true but for a
    /// fee kept outside it.
    pub inside_nmbf: bool,
}

impl Deduction {
    /// A deduction of `value` on `basis`, inside NMBF. Fails on an empty
    /// label, one a tower's result or the deal ratios already use, or a
    /// value that is not finite and non-negative.
    pub fn new(label: impl Into<String>, basis: ExpenseBasis, value: f64) -> Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(Error::Data("a deduction needs a label".into()));
        }
        if RESERVED.contains(&label.as_str()) {
            return Err(Error::Data(format!(
                "a deduction cannot be labelled {label:?}: a tower's result or the deal ratios use it"
            )));
        }
        if !(value.is_finite() && value >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "value",
                value,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self {
            label,
            basis,
            value,
            inside_nmbf: true,
        })
    }

    /// A rate of the year's premium.
    pub fn premium(label: impl Into<String>, rate: f64) -> Result<Self> {
        Self::new(label, ExpenseBasis::Premium, rate)
    }

    /// A rate of the year's ceded loss.
    pub fn loss(label: impl Into<String>, rate: f64) -> Result<Self> {
        Self::new(label, ExpenseBasis::Loss, rate)
    }

    /// An amount, the same every year.
    pub fn fixed(label: impl Into<String>, amount: f64) -> Result<Self> {
        Self::new(label, ExpenseBasis::Fixed, amount)
    }

    /// Kept outside the nominal margin before fees, as a manager's fee.
    pub fn outside_nmbf(mut self) -> Self {
        self.inside_nmbf = false;
        self
    }

    /// The amount for a year with premium `premium` and ceded loss `ceded`.
    pub fn amount(&self, premium: f64, ceded: f64) -> f64 {
        match self.basis {
            ExpenseBasis::Premium => self.value * premium,
            ExpenseBasis::Loss => self.value * ceded,
            ExpenseBasis::Fixed => self.value,
        }
    }
}

impl Layer {
    /// A flat brokerage: `rate` of the year's premium. It replaces any
    /// brokerage set before.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .brokerage(0.025).unwrap();
    /// assert!((qs.brokerage_for(70.0) - 2.5).abs() < 1e-12);
    /// ```
    pub fn brokerage(mut self, rate: f64) -> Result<Self> {
        if !(rate.is_finite() && (0.0..=1.0).contains(&rate)) {
            return Err(Error::InvalidParameter {
                name: "rate",
                value: rate,
                reason: "must be in [0, 1]",
            });
        }
        self.brokerage = Some(Commission::Flat(rate));
        Ok(self)
    }

    /// Brokerage on a scale of the loss ratio, from `(rate, loss_ratio)`
    /// anchors read as a sliding scale is ([`Commission::rate`]): linear
    /// between anchors, held at the first and last. The rate may rise or
    /// fall with the loss ratio. It reads the ceded loss ratio, or the one
    /// before annual terms with
    /// [`scales_before_annual_terms`](Self::scales_before_annual_terms), held
    /// inside a sliding-scale commission's range with
    /// [`brokerage_within_commission_range`](Self::brokerage_within_commission_range).
    /// Set the premium first; like a sliding scale it cannot be
    /// combined with swing rating or paid reinstatements. It replaces any
    /// brokerage set before.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// // 3% at a 50% loss ratio, 1% at 80% and beyond, on a premium of 100.
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .brokerage_scale(vec![(0.03, 0.5), (0.01, 0.8)]).unwrap();
    /// assert!((qs.brokerage_for(65.0) - 2.0).abs() < 1e-12);
    /// assert!((qs.brokerage_for(95.0) - 1.0).abs() < 1e-12);
    /// ```
    pub fn brokerage_scale(mut self, anchors: Vec<(f64, f64)>) -> Result<Self> {
        self.check_loss_ratio_terms("brokerage_scale")?;
        self.brokerage = Some(Commission::SlidingScale(checked_anchors(anchors, false)?));
        Ok(self)
    }

    /// Adds a deduction from the premium. Its label must differ from the
    /// layer's other deductions'. A premium-basis deduction reads the
    /// year's premium ([`premium_for`](Self::premium_for)), so set the
    /// premium first.
    ///
    /// ```
    /// use prospicio_aggregate::{Deduction, Layer};
    ///
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .deduction(Deduction::premium("tax", 0.03).unwrap()).unwrap()
    ///     .deduction(Deduction::loss("tpa", 0.02).unwrap()).unwrap();
    /// let [tax, tpa] = [0, 1].map(|k| qs.deduction_for(&qs.deductions[k], 70.0));
    /// assert!((tax - 3.0).abs() < 1e-12 && (tpa - 1.4).abs() < 1e-12);
    /// ```
    pub fn deduction(mut self, deduction: Deduction) -> Result<Self> {
        if self.deductions.iter().any(|d| d.label == deduction.label) {
            return Err(Error::Data(format!(
                "layer {:?} already has a deduction labelled {:?}",
                self.name, deduction.label
            )));
        }
        self.deductions.push(deduction);
        Ok(self)
    }

    /// The year's brokerage given its ceded loss: the rate (flat, or from
    /// the scale at the loss ratio `ceded / premium`, held inside a
    /// sliding-scale commission's range with
    /// [`brokerage_within_commission_range`](Self::brokerage_within_commission_range))
    /// times the year's premium; zero without one. A layer with
    /// [`scales_before_annual_terms`](Self::scales_before_annual_terms)
    /// reads its scale on another loss, which a tower's result gives.
    pub fn brokerage_for(&self, ceded: f64) -> f64 {
        self.brokerage_keyed(ceded, ceded)
    }

    /// [`brokerage_for`](Self::brokerage_for) with the scale read at
    /// `key / premium`.
    pub(crate) fn brokerage_keyed(&self, ceded: f64, key: f64) -> f64 {
        let Some(brokerage) = &self.brokerage else {
            return 0.0;
        };
        let premium = self.premium_for(ceded);
        if premium == 0.0 {
            return 0.0;
        }
        let mut lr = key / premium;
        if let (true, Some(Commission::SlidingScale(cc))) =
            (self.brokerage_within_commission_range, &self.commission)
        {
            lr = lr.clamp(cc[0].1, cc[cc.len() - 1].1);
        }
        brokerage.rate(lr) * premium
    }

    /// Keys both sliding scales, the commission's and the brokerage's, on
    /// the loss ratio before the annual terms: the placed share of the
    /// year's recoveries before the annual deductible, the corridor and the
    /// annual limit, over the premium. A cap on the loss ratio stated as an
    /// [`aggregate_limit`](Self::aggregate_limit) then caps the ceded loss
    /// but not what the scales read, as in the reference model of #477.
    /// Without it (the default, as `aggregate`) they read the ceded loss
    /// ratio. A tower's result then carries each year's
    /// `(loss_before_terms, <layer>)`. Flat terms, the profit commission and
    /// the deductions are unchanged. The grid methods
    /// ([`Tower::on_grid`], [`Tower::on_annual_grid`]) and the P&L ledger,
    /// which see only the ceded loss, refuse such a layer with annual terms
    /// and a sliding scale.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{Empirical, KeyValue};
    ///
    /// // A loss-ratio cap of 120% on a premium of 100, a commission sliding
    /// // from 30% at a 50% loss ratio to 10% at 150%.
    /// let cover = Layer::quota_share("cover", 1.0).unwrap()
    ///     .aggregate_limit(120.0).unwrap()
    ///     .premium_rate(1.0, 100.0).unwrap()
    ///     .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)]).unwrap()
    ///     .scales_before_annual_terms();
    /// let result = Tower::new(vec![cover]).unwrap().apply_annual(&[140.0]).unwrap();
    /// let at = |k: &str| {
    ///     let key: Vec<KeyValue> = vec![k.into(), "cover".into()];
    ///     Empirical::draws(&result.marginal(&key).unwrap())[0]
    /// };
    /// assert_eq!(at("ceded"), 120.0);
    /// assert_eq!(at("loss_before_terms"), 140.0);
    /// // The scale at 140%, not 120%: 30% − 0.2 × 0.9 = 12%.
    /// assert!((at("ceding_commission") - 12.0).abs() < 1e-12);
    /// ```
    pub fn scales_before_annual_terms(mut self) -> Self {
        self.scales_before_annual_terms = true;
        self
    }

    /// Holds the loss ratio a brokerage scale reads inside the
    /// sliding-scale commission's range of loss ratios, from its first
    /// anchor to its last, as the reference model of #477 does. Off by
    /// default; it changes nothing without a sliding-scale commission.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    ///
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .sliding_scale(vec![(0.3, 0.5), (0.1, 0.7)]).unwrap()
    ///     .brokerage_scale(vec![(0.03, 0.5), (0.01, 0.9)]).unwrap();
    /// // At a 90% loss ratio, 1%; held at the commission's 70%, 2%.
    /// assert!((qs.brokerage_for(90.0) - 1.0).abs() < 1e-12);
    /// let held = qs.brokerage_within_commission_range();
    /// assert!((held.brokerage_for(90.0) - 2.0).abs() < 1e-12);
    /// ```
    pub fn brokerage_within_commission_range(mut self) -> Self {
        self.brokerage_within_commission_range = true;
        self
    }

    /// The year's amount of `deduction` given its ceded loss.
    pub fn deduction_for(&self, deduction: &Deduction, ceded: f64) -> f64 {
        deduction.amount(self.premium_for(ceded), ceded)
    }
}

/// A scale read on the loss ratio of several layers of a tower together,
/// the sections of one deal: a sliding-scale ceding commission and a
/// brokerage on the sum of the sections' keys over the sum of their
/// premiums, paid on that summed premium, on top of whatever each section
/// pays on its own. Add one to a tower with [`Tower::deal_scale`].
///
/// A section's key is the one its own scales read: its ceded loss, or with
/// [`Layer::scales_before_annual_terms`] its loss before annual terms, so a
/// deal of such sections reads the reference model's `LR` of the deal.
///
/// ```
/// use prospicio_aggregate::{DealScale, Layer, Tower};
/// use prospicio_prob::{Empirical, KeyValue};
///
/// // Two sections of premium 100 each; the deal's commission slides from
/// // 35% at a 50% loss ratio to 15% at 90%.
/// let a = Layer::stop_loss("A", 50.0, 0.0).unwrap().premium_rate(1.0, 100.0).unwrap();
/// let b = Layer::stop_loss("B", f64::INFINITY, 50.0).unwrap().premium_rate(1.0, 100.0).unwrap();
/// let deal = DealScale::new("deal", ["A", "B"]).unwrap()
///     .sliding_scale(vec![(0.35, 0.5), (0.15, 0.9)]).unwrap();
/// let tower = Tower::new(vec![a, b]).unwrap().deal_scale(deal).unwrap();
/// // A loss of 160: A cedes 50 (50%), B 110 (110%), the deal 160 of 200.
/// let result = tower.apply_annual(&[160.0]).unwrap();
/// let key: Vec<KeyValue> = vec!["ceding_commission".into(), "deal".into()];
/// let cc = Empirical::draws(&result.marginal(&key).unwrap())[0];
/// // At 80%: 35% − 0.2 × 0.75 = 20% of 200. Each section on its own
/// // scale would get 35 + 15 = 50.
/// assert!((cc - 40.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DealScale {
    /// The deal's name: its `layer` in a tower's result.
    pub name: String,
    /// The layers whose loss ratios it reads together, by name.
    pub sections: Vec<String>,
    /// The sliding-scale ceding commission, if any.
    pub commission: Option<Commission>,
    /// Deficits and credits of the commission carried between the years of
    /// a term.
    pub commission_carry: Option<CarryForward>,
    /// The brokerage, flat or on a scale, if any.
    pub brokerage: Option<Commission>,
    /// Holds the brokerage key inside the commission scale's range.
    pub brokerage_within_commission_range: bool,
}

impl DealScale {
    /// A deal of at least two `sections`, layers of the tower it will join,
    /// with no scale yet. Fails on an empty name, fewer than two sections
    /// or a repeated one.
    pub fn new<S: Into<String>>(
        name: impl Into<String>,
        sections: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Data("a deal needs a name".into()));
        }
        let sections: Vec<String> = sections.into_iter().map(Into::into).collect();
        if sections.len() < 2 {
            return Err(Error::Data(format!(
                "deal {name:?} needs at least two sections; a scale on one section is the \
                 layer's own"
            )));
        }
        for (i, s) in sections.iter().enumerate() {
            if sections[..i].contains(s) {
                return Err(Error::Data(format!(
                    "deal {name:?} repeats the section {s:?}"
                )));
            }
        }
        Ok(Self {
            name,
            sections,
            commission: None,
            commission_carry: None,
            brokerage: None,
            brokerage_within_commission_range: false,
        })
    }

    /// The deal's sliding-scale ceding commission from `(commission,
    /// loss_ratio)` anchors, read as [`Layer::sliding_scale`] is: the
    /// commission must not rise with the loss ratio. It replaces any set
    /// before and drops a carry.
    pub fn sliding_scale(mut self, anchors: Vec<(f64, f64)>) -> Result<Self> {
        self.commission = Some(Commission::SlidingScale(checked_anchors(anchors, true)?));
        self.commission_carry = None;
        Ok(self)
    }

    /// Carries the deal commission's deficits, credits or both between the
    /// years of a term, on the deal's key, as
    /// [`Layer::sliding_scale_carry_forward`] does for a layer.
    pub fn carry_forward(
        mut self,
        deficit: bool,
        credit: bool,
        years: Option<u32>,
    ) -> Result<Self> {
        if self.commission.is_none() {
            return Err(Error::Data(
                "carry_forward needs the deal's sliding scale first".into(),
            ));
        }
        if !(deficit || credit) {
            return Err(Error::Data(
                "carry_forward must carry deficits, credits or both".into(),
            ));
        }
        if years == Some(0) {
            return Err(Error::InvalidParameter {
                name: "years",
                value: 0.0,
                reason: "must be at least 1 (or None, without limit)",
            });
        }
        self.commission_carry = Some(CarryForward {
            deficit,
            credit,
            years,
        });
        Ok(self)
    }

    /// A flat deal brokerage, `rate` of the sections' premium together.
    pub fn brokerage(mut self, rate: f64) -> Result<Self> {
        if !(rate.is_finite() && (0.0..=1.0).contains(&rate)) {
            return Err(Error::InvalidParameter {
                name: "rate",
                value: rate,
                reason: "must be in [0, 1]",
            });
        }
        self.brokerage = Some(Commission::Flat(rate));
        Ok(self)
    }

    /// A deal brokerage on a scale of the deal's loss ratio, from `(rate,
    /// loss_ratio)` anchors, read as [`Layer::brokerage_scale`] is.
    pub fn brokerage_scale(mut self, anchors: Vec<(f64, f64)>) -> Result<Self> {
        self.brokerage = Some(Commission::SlidingScale(checked_anchors(anchors, false)?));
        Ok(self)
    }

    /// Holds the loss ratio the deal's brokerage reads inside the deal
    /// commission's range, as [`Layer::brokerage_within_commission_range`].
    pub fn brokerage_within_commission_range(mut self) -> Self {
        self.brokerage_within_commission_range = true;
        self
    }

    /// The deal's brokerage on the sections' `premium` at the key `key`.
    pub(crate) fn brokerage_at(&self, premium: f64, key: f64) -> f64 {
        let Some(brokerage) = &self.brokerage else {
            return 0.0;
        };
        let mut lr = key / premium;
        if let (true, Some(Commission::SlidingScale(cc))) =
            (self.brokerage_within_commission_range, &self.commission)
        {
            lr = lr.clamp(cc[0].1, cc[cc.len() - 1].1);
        }
        brokerage.rate(lr) * premium
    }

    /// The deal's terms, for a result's provenance.
    pub(crate) fn terms_text(&self) -> String {
        let mut terms = format!("sections {:?}", self.sections);
        if let Some(Commission::SlidingScale(a)) = &self.commission {
            terms += &format!(", sliding scale (commission, loss ratio) {a:?}");
        }
        if let Some(cf) = &self.commission_carry {
            terms += &format!(
                ", carried deficits {} credits {} years {:?}",
                cf.deficit, cf.credit, cf.years
            );
        }
        match &self.brokerage {
            Some(Commission::Flat(b)) => terms += &format!(", brokerage {b}"),
            Some(Commission::SlidingScale(a)) => {
                terms += &format!(", brokerage scale (rate, loss ratio) {a:?}");
            }
            None => {}
        }
        if self.brokerage_within_commission_range {
            terms += ", brokerage within the commission's range";
        }
        terms
    }
}

impl Tower {
    /// Adds a deal's scales ([`DealScale`]), read on its sections'
    /// loss ratio together. A tower's results then carry
    /// `(ceding_commission, <deal>)` and `(brokerage, <deal>)` where the deal
    /// has them, and `(loss_before_terms, <deal>)`, the sum of the
    /// sections' keys, where a section reads its scales before annual
    /// terms; [`deal_ratios`](Self::deal_ratios) takes the deal's name.
    ///
    /// Fails if the deal has no scale, if its name is a layer's, another
    /// deal's or one a result uses, if a section is not a layer of the
    /// tower or is a section of another deal, or if a section has no
    /// positive fixed premium (it is swing rated, has paid reinstatements,
    /// or has no premium). The grid methods and the P&L ledger refuse a
    /// tower with a deal: they see one layer's ceded loss at a time.
    pub fn deal_scale(mut self, deal: DealScale) -> Result<Self> {
        if deal.commission.is_none() && deal.brokerage.is_none() {
            return Err(Error::Data(format!(
                "deal {:?} has no scale: set a sliding scale or a brokerage",
                deal.name
            )));
        }
        if self.layers.iter().any(|l| l.name == deal.name)
            || self.deals.iter().any(|d| d.name == deal.name)
            || ["ground_up", "retained"].contains(&deal.name.as_str())
        {
            return Err(Error::Data(format!(
                "the deal's name {:?} is taken: a layer, another deal or a result uses it",
                deal.name
            )));
        }
        for s in &deal.sections {
            let layer = self.layers.iter().find(|l| &l.name == s).ok_or_else(|| {
                Error::Data(format!(
                    "deal {:?}: the tower has no layer named {s:?}",
                    deal.name
                ))
            })?;
            if let Some(d) = self.deals.iter().find(|d| d.sections.contains(s)) {
                return Err(Error::Data(format!(
                    "layer {s:?} is already a section of deal {:?}",
                    d.name
                )));
            }
            layer.check_loss_ratio_terms("deal_scale")?;
        }
        self.deals.push(deal);
        Ok(self)
    }

    /// Fails if a layer reads a sliding scale (commission or brokerage)
    /// before annual terms it has, or the tower has a deal: `what` sees
    /// only each layer's ceded loss.
    pub(crate) fn check_scales_on_ceded(&self, what: &str) -> Result<()> {
        if let Some(d) = self.deals.first() {
            return Err(Error::Data(format!(
                "deal {:?} reads the loss ratio of several layers together, which {what} cannot \
                 see; use Tower::apply on simulated years",
                d.name
            )));
        }
        let sliding = |c: &Option<Commission>| matches!(c, Some(Commission::SlidingScale(_)));
        match self.layers.iter().find(|l| {
            l.scales_before_annual_terms
                && l.has_annual_terms()
                && (sliding(&l.commission) || sliding(&l.brokerage))
        }) {
            Some(l) => Err(Error::Data(format!(
                "layer {:?} reads its sliding scale before its annual terms, which {what} cannot \
                 see; use Tower::apply on simulated years",
                l.name
            ))),
            None => Ok(()),
        }
    }

    /// The reinsurer's P&L ledger of one layer or one deal
    /// ([`deal_scale`](Self::deal_scale)) over the annual gross losses
    /// `gross`: one [`Group::sell`] named `layer`, the premium received
    /// against the ceded loss, the commissions paid, the brokerage and each
    /// deduction ([`Layer::deduction`]), every leg at each atom of the
    /// source. The tower is applied to each atom as one annual loss
    /// ([`apply_annual`](Self::apply_annual)), so every term reads the
    /// annual total, scales keyed before annual terms and a deal's scales
    /// included, and the amounts are those [`deal_ratios`](Self::deal_ratios)
    /// divides by the premium. A deal books its sections' amounts with its
    /// own.
    ///
    /// The cedant's ledger ([`Book::ledger`](crate::pnl::Book::ledger))
    /// books the same tower as covers bought and leaves brokerage and
    /// deductions out, since they come out of what the reinsurer receives.
    ///
    /// The legs are `Premium` (kind premium), `Loss` (loss), `Ceding
    /// commission` and `Profit commission` (commission), `Brokerage` and each
    /// deduction by its label (expense). NMBF is not a ledger row:
    /// [`deal_ratios`](Self::deal_ratios) gives it.
    ///
    /// Fails on a joint source (occurrence covers), a layer the tower
    /// cannot apply to an annual loss (a surplus treaty, reinstatements pro
    /// rata as to time), a booked layer with paid reinstatements (they
    /// belong to an occurrence cover), an unknown layer or deal, or a
    /// ledger the kernel refuses.
    ///
    /// ```
    /// use prospicio_aggregate::pnl::Source;
    /// use prospicio_aggregate::{Deduction, Layer, Tower};
    ///
    /// // A cover of 120 xs 0 for a premium of 100 with a 20% commission,
    /// // 2.5% brokerage and 3% tax, on annual losses of 40 or 150.
    /// let cover = Layer::stop_loss("cover", 120.0, 0.0).unwrap()
    ///     .premium_rate(1.0, 100.0).unwrap()
    ///     .ceding_commission(0.2).unwrap()
    ///     .brokerage(0.025).unwrap()
    ///     .deduction(Deduction::premium("tax", 0.03).unwrap()).unwrap();
    /// let tower = Tower::new(vec![cover]).unwrap();
    /// let gross = Source::new(vec![40.0, 150.0], vec![0.5, 0.5]).unwrap();
    /// let ledger = tower.deal_ledger(&gross, "cover").unwrap();
    /// // 100 − 20 − 2.5 − 3 = 74.5 less the ceded 40 or 120.
    /// assert_eq!(ledger.result().values(), [34.5, -45.5]);
    /// assert!((ledger.row("Brokerage").unwrap().mean() + 2.5).abs() < 1e-12);
    /// ```
    pub fn deal_ledger(&self, gross: &Source, layer: &str) -> Result<Ledger> {
        if gross.second().is_some() {
            return Err(Error::Data(
                "a deal's ledger reads annual gross losses; a joint source carries occurrence \
                 covers"
                    .into(),
            ));
        }
        let booked: Vec<&Layer> = match self.deals.iter().find(|d| d.name == layer) {
            Some(d) => self
                .layers
                .iter()
                .filter(|l| d.sections.contains(&l.name))
                .collect(),
            None => self.layers.iter().filter(|l| l.name == layer).collect(),
        };
        if let Some(l) = booked.iter().find(|l| !l.reinstatement_rates.is_empty()) {
            return Err(Error::Data(format!(
                "layer {:?} has paid reinstatements, which apply to an occurrence cover; an \
                 annual loss has no occurrences",
                l.name
            )));
        }
        let result = self.apply_annual(gross.values())?;
        let a = self.deal_amounts(&result, layer)?;
        let column = |f: &dyn Fn(&YearAmounts) -> f64| a.amounts.iter().map(f).collect::<Vec<_>>();
        let mut group = Group::sell(layer)
            .consideration(Leg::values("Premium", column(&|y| y.premium)).kind(LegKind::Premium))
            .obligation(Leg::values("Loss", column(&|y| y.ceded)).kind(LegKind::Loss));
        for (k, name) in a.names.iter().enumerate() {
            let (label, kind) = match name.as_str() {
                "ceding_commission" => ("Ceding commission", LegKind::Commission),
                "profit_commission" => ("Profit commission", LegKind::Commission),
                "brokerage" => ("Brokerage", LegKind::Expense),
                other => (other, LegKind::Expense),
            };
            group = group.obligation(Leg::values(label, column(&|y| y.terms[k])).kind(kind));
        }
        Ledger::new(gross, vec![group], Layout::new(layer))
    }

    /// Each simulation's amounts of a layer or deal in a tower's result,
    /// as [`deal_ratios`](Self::deal_ratios) reads them.
    fn deal_amounts(&self, result: &PredictiveDistribution, layer: &str) -> Result<DealAmounts> {
        // What is read: the layer alone, or a deal's sections and its own
        // components.
        let deal = self.deals.iter().find(|d| d.name == layer);
        let sections: Vec<&Layer> = match deal {
            Some(d) => d
                .sections
                .iter()
                .map(|s| {
                    self.layers
                        .iter()
                        .find(|l| &l.name == s)
                        .expect("a deal's sections are layers of its tower")
                })
                .collect(),
            None => vec![
                self.layers
                    .iter()
                    .find(|l| l.name == layer)
                    .ok_or_else(|| {
                        Error::Data(format!("the tower has no layer or deal named {layer:?}"))
                    })?,
            ],
        };
        let dims = result.dims();
        if dims.len() < 2 || dims[0] != "kind" || dims[1] != "layer" {
            return Err(Error::Data(
                "deal ratios read a tower's result, with dimensions kind and layer first".into(),
            ));
        }
        let term = dims.iter().any(|d| d == "year");
        let keep: Vec<&str> = if term {
            vec!["kind", "layer", "year"]
        } else {
            vec!["kind", "layer"]
        };
        let pd = result.aggregate(&keep)?;
        let years: Vec<Option<KeyValue>> = if term {
            let mut seen: Vec<KeyValue> = Vec::new();
            for key in pd.components() {
                if !seen.contains(&key[2]) {
                    seen.push(key[2].clone());
                }
            }
            seen.into_iter().map(Some).collect()
        } else {
            vec![None]
        };
        let column = |kind: &str, name: &str, year: &Option<KeyValue>| -> Result<usize> {
            let mut key: ComponentKey = vec![KeyValue::from(kind), KeyValue::from(name)];
            key.extend(year.iter().cloned());
            pd.component_index(&key).ok_or_else(|| {
                Error::Data(format!(
                    "the result has no ({kind}, {name}) component: is it this tower's?"
                ))
            })
        };
        // Each term: its name, whether NMBF leaves it out, and the
        // `(kind, layer)` components it adds up.
        struct Term {
            name: String,
            outside: bool,
            parts: Vec<(String, String)>,
        }
        let mut terms: Vec<Term> = Vec::new();
        let mut add = |name: &str, outside: bool, kind: &str, of: &str| -> Result<()> {
            match terms.iter_mut().find(|t| t.name == name) {
                Some(t) => {
                    if t.outside != outside {
                        return Err(Error::Data(format!(
                            "the deduction {name:?} is inside NMBF in one section of {layer:?} \
                             and outside it in another"
                        )));
                    }
                    t.parts.push((kind.into(), of.into()));
                }
                None => terms.push(Term {
                    name: name.into(),
                    outside,
                    parts: vec![(kind.into(), of.into())],
                }),
            }
            Ok(())
        };
        let (cc, pc, bk) = ("ceding_commission", "profit_commission", "brokerage");
        for l in &sections {
            if l.commission.is_some() {
                add(cc, false, cc, &l.name)?;
            }
        }
        if deal.is_some_and(|d| d.commission.is_some()) {
            add(cc, false, cc, layer)?;
        }
        for l in &sections {
            if l.profit_commission.is_some() {
                add(pc, false, pc, &l.name)?;
            }
        }
        for l in &sections {
            if l.brokerage.is_some() {
                add(bk, false, bk, &l.name)?;
            }
        }
        if deal.is_some_and(|d| d.brokerage.is_some()) {
            add(bk, false, bk, layer)?;
        }
        for l in &sections {
            for d in &l.deductions {
                add(&d.label, !d.inside_nmbf, &d.label, &l.name)?;
            }
        }
        let keyed = sections.iter().any(|l| l.scales_before_annual_terms);
        // Per year: the ceded loss, the key, the swing premium and each
        // term's columns.
        struct Cols {
            ceded: Vec<usize>,
            before: Option<usize>,
            swing: Option<usize>,
            terms: Vec<Vec<usize>>,
        }
        let only = &sections[0];
        let cols = years
            .iter()
            .map(|y| {
                Ok(Cols {
                    ceded: sections
                        .iter()
                        .map(|l| column("ceded", &l.name, y))
                        .collect::<Result<_>>()?,
                    before: keyed
                        .then(|| column("loss_before_terms", layer, y))
                        .transpose()?,
                    swing: (deal.is_none() && only.swing.is_some())
                        .then(|| column("swing_premium", layer, y))
                        .transpose()?,
                    terms: terms
                        .iter()
                        .map(|term| {
                            term.parts
                                .iter()
                                .map(|(k, of)| column(k, of, y))
                                .collect::<Result<_>>()
                        })
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fixed: f64 = sections.iter().map(|l| l.premium).sum();
        let mut years_amounts = Vec::with_capacity(pd.n_sims() * years.len());
        for i in 0..pd.n_sims() {
            let row = pd.row(i).expect("a simulation within range");
            for c in &cols {
                years_amounts.push(YearAmounts {
                    premium: c.swing.map_or(fixed, |k| row[k]),
                    ceded: c.ceded.iter().map(|&k| row[k]).sum(),
                    before: c.before.map(|k| row[k]),
                    terms: c
                        .terms
                        .iter()
                        .map(|parts| parts.iter().map(|&k| row[k]).sum())
                        .collect(),
                });
            }
        }
        Ok(DealAmounts {
            term,
            years,
            keyed,
            names: terms.iter().map(|t| t.name.clone()).collect(),
            outside: terms.iter().map(|t| t.outside).collect(),
            amounts: years_amounts,
            provenance: pd.provenance().clone(),
        })
    }

    /// The deal ratios of one layer or one deal ([`deal_scale`](Self::deal_scale))
    /// in each simulation of `result`, a
    /// result of this tower ([`apply`](Self::apply),
    /// [`apply_aggregate`](Self::apply_aggregate),
    /// [`apply_annual`](Self::apply_annual) or
    /// [`apply_term`](Self::apply_term)): every amount over the year's
    /// premium for the placed share, the fixed premium or the swing-rated
    /// one (reinstatement premiums are not in it).
    ///
    /// The result has dimension `["ratio"]` (`["ratio", "year"]` for a
    /// term) and components `loss_ratio` (the ceded loss over the
    /// premium), `loss_ratio_before_terms` for a layer whose scales read it
    /// ([`Layer::scales_before_annual_terms`]), then `ceding_commission`,
    /// `profit_commission` and `brokerage` where the layer has them, each
    /// deduction by its label, `total_deductions`, `combined_ratio`,
    /// `nmbf` and `margin`; see the [module](crate::deal) for the
    /// formulas. The commissions are read from `result`, so a sliding
    /// scale or profit commission carried over a term counts as it was
    /// paid. Labels on the events are summed over.
    ///
    /// For a deal, every amount is summed over its sections, with the
    /// deal's own commission and brokerage added to theirs, over the
    /// sections' premium together; deductions with the same label add.
    /// A section's own ratios leave the deal's scales out.
    ///
    /// Fails if the tower has no layer or deal named `layer`, if `result` lacks
    /// one of its components (it is not this tower's), or if a year's
    /// premium is not positive.
    ///
    /// ```
    /// use prospicio_aggregate::{Deduction, Layer, Tower};
    /// use prospicio_prob::{Empirical, KeyValue};
    ///
    /// // A quota share of a book with subject premium 100, losses capped at
    /// // a 120% loss ratio, a sliding commission, 2.5% brokerage, 3% tax and
    /// // a 5% manager's fee outside NMBF.
    /// let cover = Layer::stop_loss("cover", 120.0, 0.0).unwrap()
    ///     .premium_rate(1.0, 100.0).unwrap()
    ///     .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)]).unwrap()
    ///     .brokerage(0.025).unwrap()
    ///     .deduction(Deduction::premium("tax", 0.03).unwrap()).unwrap()
    ///     .deduction(Deduction::premium("manager_fee", 0.05).unwrap().outside_nmbf()).unwrap();
    /// let tower = Tower::new(vec![cover]).unwrap();
    /// let result = tower.apply_annual(&[40.0, 70.0, 150.0]).unwrap();
    /// let ratios = tower.deal_ratios(&result, "cover").unwrap();
    /// let at = |k: &str| Empirical::draws(&ratios.marginal(&vec![KeyValue::from(k)]).unwrap()).to_vec();
    /// // At a 70% loss ratio: CC 20%, so TD = 20 + 2.5 + 3 + 5 = 30.5%.
    /// assert!((at("combined_ratio")[1] - 1.005).abs() < 1e-12);
    /// assert!((at("margin")[1] + 0.005).abs() < 1e-12);
    /// assert!((at("nmbf")[1] - 0.045).abs() < 1e-12);
    /// ```
    pub fn deal_ratios(
        &self,
        result: &PredictiveDistribution,
        layer: &str,
    ) -> Result<PredictiveDistribution> {
        let a = self.deal_amounts(result, layer)?;
        let mut ratio_names = vec!["loss_ratio"];
        if a.keyed {
            ratio_names.push("loss_ratio_before_terms");
        }
        ratio_names.extend(a.names.iter().map(String::as_str));
        ratio_names.extend(["total_deductions", "combined_ratio", "nmbf", "margin"]);
        let mut draws = Vec::with_capacity(a.amounts.len() * ratio_names.len());
        for (j, y) in a.amounts.iter().enumerate() {
            let premium = y.premium;
            if premium.is_nan() || premium <= 0.0 {
                return Err(Error::Data(format!(
                    "deal ratios need a positive premium; {layer:?} has {premium} in simulation {}",
                    j / a.years.len()
                )));
            }
            let lr = y.ceded / premium;
            draws.push(lr);
            if let Some(b) = y.before {
                draws.push(b / premium);
            }
            let (mut total, mut fees) = (0.0, 0.0);
            for (&x, &out) in y.terms.iter().zip(&a.outside) {
                let r = x / premium;
                draws.push(r);
                total += r;
                if out {
                    fees += r;
                }
            }
            let cr = lr + total;
            let margin = 1.0 - cr;
            draws.extend([total, cr, margin + fees, margin]);
        }
        let components: Vec<ComponentKey> = a
            .years
            .iter()
            .flat_map(|y| {
                ratio_names.iter().map(move |n| {
                    let mut key = vec![KeyValue::from(*n)];
                    key.extend(y.iter().cloned());
                    key
                })
            })
            .collect();
        let mut out_dims = vec!["ratio".to_string()];
        if a.term {
            out_dims.push("year".into());
        }
        let provenance = a.provenance.param("deal_ratios", layer);
        PredictiveDistribution::from_draws(out_dims, components, draws, provenance)
    }
}

/// A layer's or deal's amounts in a tower's result, simulation by
/// simulation and, over a term, year by year.
struct DealAmounts {
    term: bool,
    years: Vec<Option<KeyValue>>,
    keyed: bool,
    /// The commissions, brokerage and deductions, in order.
    names: Vec<String>,
    outside: Vec<bool>,
    /// Simulation-major, one per year.
    amounts: Vec<YearAmounts>,
    provenance: prospicio_prob::Provenance,
}

/// One year's amounts: the premium, the ceded loss, the scales' key where
/// it is not the ceded loss, and each term of [`DealAmounts::names`].
struct YearAmounts {
    premium: f64,
    ceded: f64,
    before: Option<f64>,
    terms: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Empirical, Grid, Provenance};

    fn draws(pd: &PredictiveDistribution, key: &[&str]) -> Vec<f64> {
        let key: ComponentKey = key.iter().map(|&k| KeyValue::from(k)).collect();
        Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
    }

    fn sd(x: &[f64]) -> f64 {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// The deal of #485: a sliding commission, flat brokerage and four
    /// legs on the premium, a manager's fee outside NMBF.
    fn deal(extra: Option<Deduction>) -> Tower {
        let mut layer = Layer::stop_loss("deal", 120.0, 0.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.35, 0.5), (0.15, 0.9)])
            .unwrap()
            .brokerage(0.025)
            .unwrap();
        for (label, rate) in [
            ("tpa", 0.01),
            ("tax", 0.03),
            ("ri_cost", 0.02),
            ("fronting", 0.005),
        ] {
            layer = layer
                .deduction(Deduction::premium(label, rate).unwrap())
                .unwrap();
        }
        layer = layer
            .deduction(Deduction::premium("ms_fee", 0.05).unwrap().outside_nmbf())
            .unwrap();
        if let Some(d) = extra {
            layer = layer.deduction(d).unwrap();
        }
        Tower::new(vec![layer]).unwrap()
    }

    const LOSSES: [f64; 6] = [20.0, 45.0, 60.0, 75.0, 95.0, 150.0];

    #[test]
    fn premium_legs_shift_the_total_without_spreading_it() {
        let tower = deal(None);
        let result = tower.apply_annual(&LOSSES).unwrap();
        let r = tower.deal_ratios(&result, "deal").unwrap();
        let [lr, cc, bk, td, cr, nmbf, margin] = [
            "loss_ratio",
            "ceding_commission",
            "brokerage",
            "total_deductions",
            "combined_ratio",
            "nmbf",
            "margin",
        ]
        .map(|k| draws(&r, &[k]));
        for i in 0..LOSSES.len() {
            // TD − CC − BK is the sum of the flat legs, fee included.
            assert!((td[i] - cc[i] - bk[i] - 0.115).abs() < 1e-12);
            assert!((cr[i] - lr[i] - td[i]).abs() < 1e-12);
            assert!((margin[i] - (1.0 - cr[i])).abs() < 1e-12);
            // Only the manager's fee sits outside NMBF.
            assert!((nmbf[i] - margin[i] - 0.05).abs() < 1e-12);
            assert!((bk[i] - 0.025).abs() < 1e-12);
            assert!((lr[i] - LOSSES[i].min(120.0) / 100.0).abs() < 1e-12);
        }
        assert!((sd(&td) - sd(&cc)).abs() < 1e-12);
        assert_eq!(draws(&r, &["tax"]), vec![0.03; LOSSES.len()]);
        // The amounts in the tower's result.
        assert_eq!(draws(&result, &["ms_fee", "deal"]), vec![5.0; LOSSES.len()]);
    }

    #[test]
    fn a_loss_leg_spreads_the_total() {
        let tower = deal(Some(Deduction::loss("claims_fee", 0.04).unwrap()));
        let result = tower.apply_annual(&LOSSES).unwrap();
        let r = tower.deal_ratios(&result, "deal").unwrap();
        let (td, cc, lr) = (
            draws(&r, &["total_deductions"]),
            draws(&r, &["ceding_commission"]),
            draws(&r, &["loss_ratio"]),
        );
        let fee = draws(&r, &["claims_fee"]);
        for i in 0..LOSSES.len() {
            assert!((fee[i] - 0.04 * lr[i]).abs() < 1e-12);
        }
        assert!((sd(&td) - sd(&cc)).abs() > 1e-3);
        let fixed = deal(Some(Deduction::fixed("audit", 2.0).unwrap()));
        let r = fixed
            .deal_ratios(&fixed.apply_annual(&LOSSES).unwrap(), "deal")
            .unwrap();
        assert_eq!(draws(&r, &["audit"]), vec![0.02; LOSSES.len()]);
    }

    /// A 120% loss-ratio cap as an annual limit on a whole-account quota
    /// share, and both scales.
    fn capped(before: bool) -> Tower {
        let mut layer = Layer::quota_share("deal", 1.0)
            .unwrap()
            .aggregate_limit(120.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
            .unwrap()
            .brokerage_scale(vec![(0.01, 0.5), (0.03, 1.5)])
            .unwrap();
        if before {
            layer = layer.scales_before_annual_terms();
        }
        Tower::new(vec![layer]).unwrap()
    }

    #[test]
    fn the_scales_read_the_loss_ratio_before_the_cap_when_asked() {
        let losses = [40.0, 140.0];
        let after = capped(false);
        let before = capped(true);
        let a = after.apply_annual(&losses).unwrap();
        let b = before.apply_annual(&losses).unwrap();
        assert_eq!(draws(&a, &["ceded", "deal"]), [40.0, 120.0]);
        assert_eq!(draws(&b, &["ceded", "deal"]), [40.0, 120.0]);
        assert_eq!(draws(&b, &["loss_before_terms", "deal"]), [40.0, 140.0]);
        assert!(
            a.component_index(&vec!["loss_before_terms".into(), "deal".into()])
                .is_none()
        );
        // Below the cap the two agree; above it the scale reads 140%.
        let rate = |lr: f64| 0.3 - 0.2 * (lr - 0.5);
        let (cca, ccb) = (
            draws(&a, &["ceding_commission", "deal"]),
            draws(&b, &["ceding_commission", "deal"]),
        );
        assert!((cca[0] - 30.0).abs() < 1e-12 && (ccb[0] - 30.0).abs() < 1e-12);
        assert!((cca[1] - 100.0 * rate(1.2)).abs() < 1e-12);
        assert!((ccb[1] - 100.0 * rate(1.4)).abs() < 1e-12);
        let bk = draws(&b, &["brokerage", "deal"]);
        assert!((bk[1] - 100.0 * (0.01 + 0.02 * 0.9)).abs() < 1e-12);
        let r = before.deal_ratios(&b, "deal").unwrap();
        assert_eq!(draws(&r, &["loss_ratio"]), [0.4, 1.2]);
        assert_eq!(draws(&r, &["loss_ratio_before_terms"]), [0.4, 1.4]);
        let ra = after.deal_ratios(&a, "deal").unwrap();
        assert_eq!(ra.components().len() + 1, r.components().len());
    }

    #[test]
    fn the_brokerage_key_is_held_in_the_commission_range_only_when_asked() {
        let tower = |held: bool| {
            let mut l = Layer::quota_share("deal", 1.0)
                .unwrap()
                .premium_rate(1.0, 100.0)
                .unwrap()
                .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
                .unwrap()
                .brokerage_scale(vec![(0.0, 0.3), (0.04, 1.3)])
                .unwrap();
            if held {
                l = l.brokerage_within_commission_range();
            }
            Tower::new(vec![l]).unwrap()
        };
        let losses = [10.0, 70.0, 130.0];
        let free = draws(
            &tower(false).apply_annual(&losses).unwrap(),
            &["brokerage", "deal"],
        );
        let held = draws(
            &tower(true).apply_annual(&losses).unwrap(),
            &["brokerage", "deal"],
        );
        let bk = |lr: f64| 100.0 * 0.04 * (lr - 0.3);
        for (got, want) in free.iter().zip([0.0, bk(0.7), bk(1.3)]) {
            assert!((got - want).abs() < 1e-12);
        }
        for (got, want) in held.iter().zip([bk(0.5), bk(0.7), bk(0.9)]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn a_term_gives_ratios_by_year_with_the_carry_on_the_key() {
        let layer = Layer::quota_share("deal", 1.0)
            .unwrap()
            .aggregate_limit(120.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
            .unwrap()
            .sliding_scale_carry_forward(true, false, None)
            .unwrap()
            .brokerage(0.02)
            .unwrap()
            .scales_before_annual_terms();
        let tower = Tower::new(vec![layer]).unwrap();
        let pd = |v: Vec<f64>| {
            PredictiveDistribution::from_draws(vec![], vec![vec![]], v, Provenance::new("test"))
                .unwrap()
        };
        let result = tower
            .apply_aggregate_term(&[pd(vec![180.0]), pd(vec![60.0])])
            .unwrap();
        let r = tower.deal_ratios(&result, "deal").unwrap();
        assert_eq!(r.dims(), ["ratio", "year"]);
        let at = |k: &str, y: i64| {
            let key: ComponentKey = vec![KeyValue::from(k), KeyValue::Int(y)];
            Empirical::draws(&r.marginal(&key).unwrap())[0]
        };
        assert_eq!(at("loss_ratio", 1), 1.2);
        assert_eq!(at("loss_ratio_before_terms", 1), 1.8);
        // Year 1 is beyond the scale's 150%: 30% of deficit carried, read
        // on the uncapped loss; year 2 reads 60% + 30%.
        assert!((at("ceding_commission", 1) - 0.1).abs() < 1e-12);
        assert!((at("ceding_commission", 2) - (0.3 - 0.2 * 0.4)).abs() < 1e-12);
        assert!((at("brokerage", 2) - 0.02).abs() < 1e-12);
    }

    #[test]
    fn the_grids_refuse_a_scale_read_before_annual_terms() {
        let annual = Grid::new(10.0, vec![0.25; 4]).unwrap();
        let err = capped(true).on_annual_grid(&annual).unwrap_err();
        assert!(err.to_string().contains("before its annual terms"));
        assert!(capped(false).on_annual_grid(&annual).is_ok());
        // Without annual terms the key is the ceded loss.
        let open = Layer::quota_share("deal", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
            .unwrap()
            .scales_before_annual_terms();
        assert!(
            Tower::new(vec![open])
                .unwrap()
                .on_annual_grid(&annual)
                .is_ok()
        );
    }

    #[test]
    fn refuses_bad_deductions_and_results() {
        assert!(Deduction::premium("", 0.1).is_err());
        assert!(Deduction::premium("brokerage", 0.1).is_err());
        assert!(Deduction::premium("nmbf", 0.1).is_err());
        assert!(Deduction::premium("tax", -0.1).is_err());
        assert!(Deduction::fixed("tax", f64::INFINITY).is_err());
        let l = Layer::quota_share("QS", 0.5)
            .unwrap()
            .deduction(Deduction::premium("tax", 0.03).unwrap())
            .unwrap();
        assert!(
            l.clone()
                .deduction(Deduction::loss("tax", 0.01).unwrap())
                .is_err()
        );
        assert!(l.clone().brokerage(1.5).is_err());
        // A brokerage scale needs a premium and refuses swing rating.
        assert!(l.clone().brokerage_scale(vec![(0.02, 0.5)]).is_err());
        let priced = l.premium_rate(1.0, 100.0).unwrap();
        assert!(priced.clone().brokerage_scale(vec![(1.2, 0.5)]).is_err());
        let scaled = priced
            .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
            .unwrap();
        let swing = crate::LossSensitivePremium::new(10.0, 1.0, None, None).unwrap();
        assert!(scaled.clone().swing_rated(swing).is_err());
        let tower = Tower::new(vec![scaled]).unwrap();
        let result = tower.apply_annual(&[20.0, 40.0]).unwrap();
        assert!(tower.deal_ratios(&result, "nope").is_err());
        let other = deal(None);
        assert!(
            other
                .deal_ratios(&other.apply_annual(&[20.0]).unwrap(), "deal")
                .is_ok()
        );
        assert!(other.deal_ratios(&result, "deal").is_err());
    }

    /// Two sections of premium 100 each, A capped at 50 and B above 50,
    /// each with its own scale, and a deal scale on both together.
    fn two_sections(deal: DealScale) -> Tower {
        let scale = vec![(0.35, 0.5), (0.15, 0.9)];
        let a = Layer::stop_loss("A", 50.0, 0.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(scale.clone())
            .unwrap()
            .deduction(Deduction::premium("tax", 0.03).unwrap())
            .unwrap();
        let b = Layer::stop_loss("B", f64::INFINITY, 50.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(scale)
            .unwrap()
            .brokerage(0.02)
            .unwrap()
            .deduction(Deduction::premium("tax", 0.03).unwrap())
            .unwrap();
        Tower::new(vec![a, b]).unwrap().deal_scale(deal).unwrap()
    }

    fn deal_scale() -> DealScale {
        DealScale::new("deal", ["A", "B"])
            .unwrap()
            .sliding_scale(vec![(0.2, 0.5), (0.1, 0.9)])
            .unwrap()
    }

    #[test]
    fn a_deal_scale_reads_its_sections_together() {
        let tower = two_sections(
            deal_scale()
                .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
                .unwrap(),
        );
        // A loss of 160: A at 50%, B at 110%, the deal at 80%.
        let result = tower.apply_annual(&[160.0, 40.0]).unwrap();
        let deal_cc = draws(&result, &["ceding_commission", "deal"]);
        // 20% − 0.1 × 0.75 = 12.5% of 200 at 80%; 20% of 200 at 20%.
        assert!((deal_cc[0] - 25.0).abs() < 1e-12);
        assert!((deal_cc[1] - 40.0).abs() < 1e-12);
        // The sections' own scales on each loss ratio alone: not the
        // deal's 80%.
        let own = [&["ceding_commission", "A"], &["ceding_commission", "B"]]
            .map(|k| draws(&result, k)[0]);
        assert!((own[0] - 35.0).abs() < 1e-12 && (own[1] - 15.0).abs() < 1e-12);
        let at_80 = 0.35 - 0.2 * 0.75;
        assert!((own[0] + own[1] - 2.0 * 100.0 * at_80).abs() > 1.0);
        // The deal's brokerage at 80%: 1% + 0.02 × 0.75 = 2.5%.
        assert!((draws(&result, &["brokerage", "deal"])[0] - 5.0).abs() < 1e-12);
        // The deal's ratios add the sections and the deal over 200.
        let r = tower.deal_ratios(&result, "deal").unwrap();
        let ratio = |k: &str| draws(&r, &[k])[0];
        assert!((ratio("loss_ratio") - 0.8).abs() < 1e-12);
        assert!((ratio("ceding_commission") - (35.0 + 15.0 + 25.0) / 200.0).abs() < 1e-12);
        assert!((ratio("brokerage") - (2.0 + 5.0) / 200.0).abs() < 1e-12);
        assert!((ratio("tax") - 0.03).abs() < 1e-12);
        // The commission's total is above the deal scale's 20% maximum,
        // as in the reference model's workbook.
        assert!(ratio("ceding_commission") > 0.2);
        let total = ratio("ceding_commission") + ratio("brokerage") + ratio("tax");
        assert!((ratio("total_deductions") - total).abs() < 1e-12);
        assert!((ratio("margin") - (1.0 - 0.8 - total)).abs() < 1e-12);
        // A section's own ratios leave the deal's scales out.
        let ra = tower.deal_ratios(&result, "A").unwrap();
        assert!((draws(&ra, &["ceding_commission"])[0] - 0.35).abs() < 1e-12);
        assert!(
            result
                .provenance()
                .parameters
                .iter()
                .any(|(k, _)| k == "deal:deal")
        );
    }

    #[test]
    fn a_deal_carries_its_own_deficit_over_a_term() {
        let tower = two_sections(deal_scale().carry_forward(true, false, None).unwrap());
        let pd = |v: Vec<f64>| {
            PredictiveDistribution::from_draws(vec![], vec![vec![]], v, Provenance::new("test"))
                .unwrap()
        };
        // Year 1 at 100%: 10% of 200 beyond the deal's 90%, carried. Year 2
        // at 50% reads 60%: 20% − 0.1 × 0.25.
        let result = tower
            .apply_aggregate_term(&[pd(vec![200.0]), pd(vec![100.0])])
            .unwrap();
        let at = |k: &str, l: &str, y: i64| {
            let key: ComponentKey = vec![KeyValue::from(k), KeyValue::from(l), KeyValue::Int(y)];
            Empirical::draws(&result.marginal(&key).unwrap())[0]
        };
        assert!((at("ceding_commission", "deal", 1) - 20.0).abs() < 1e-12);
        assert!((at("ceding_commission", "deal", 2) - 0.175 * 200.0).abs() < 1e-12);
        // The sections carry nothing: A at 50% in year 2 reads 35%.
        assert!((at("ceding_commission", "A", 2) - 35.0).abs() < 1e-12);
        let r = tower.deal_ratios(&result, "deal").unwrap();
        assert_eq!(r.dims(), ["ratio", "year"]);
    }

    #[test]
    fn a_deal_reads_the_sections_keys_before_annual_terms() {
        let section = |name: &str, before: bool| {
            let l = Layer::quota_share(name, 1.0)
                .unwrap()
                .aggregate_limit(60.0)
                .unwrap()
                .premium_rate(1.0, 100.0)
                .unwrap();
            if before {
                l.scales_before_annual_terms()
            } else {
                l
            }
        };
        let deal = DealScale::new("deal", ["A", "B"])
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 1.0)])
            .unwrap()
            .brokerage_scale(vec![(0.01, 0.5), (0.04, 2.0)])
            .unwrap()
            .brokerage_within_commission_range();
        let tower = Tower::new(vec![section("A", true), section("B", false)])
            .unwrap()
            .deal_scale(deal)
            .unwrap();
        // A loss of 150: each cedes 60; A's key is 150, B's 60.
        let result = tower.apply_annual(&[150.0]).unwrap();
        assert_eq!(draws(&result, &["loss_before_terms", "deal"]), [210.0]);
        // The deal at 105%, beyond the scale: 10% of 200.
        assert!((draws(&result, &["ceding_commission", "deal"])[0] - 20.0).abs() < 1e-12);
        // Its brokerage held at the commission's 100%: 1% + 0.03 / 3 = 2%,
        // not the 2.1% at 105%.
        assert!((draws(&result, &["brokerage", "deal"])[0] - 4.0).abs() < 1e-12);
        let r = tower.deal_ratios(&result, "deal").unwrap();
        assert!((draws(&r, &["loss_ratio"])[0] - 0.6).abs() < 1e-12);
        assert!((draws(&r, &["loss_ratio_before_terms"])[0] - 1.05).abs() < 1e-12);
        // The grids refuse a deal, naming it.
        let annual = Grid::new(10.0, vec![0.25; 4]).unwrap();
        let err = tower.on_annual_grid(&annual).unwrap_err();
        assert!(err.to_string().contains("deal \"deal\""));
    }

    #[test]
    fn refuses_bad_deals() {
        assert!(DealScale::new("", ["A", "B"]).is_err());
        assert!(DealScale::new("d", ["A"]).is_err());
        assert!(DealScale::new("d", ["A", "B", "A"]).is_err());
        let bare = DealScale::new("d", ["A", "B"]).unwrap();
        assert!(bare.clone().carry_forward(true, false, None).is_err());
        assert!(
            bare.clone()
                .sliding_scale(vec![(0.1, 0.5), (0.2, 0.9)])
                .is_err()
        );
        assert!(bare.clone().brokerage(1.5).is_err());
        let tower = two_sections(deal_scale());
        let base = Tower::new(tower.layers.clone()).unwrap();
        // No scale, a taken name, an unknown section, a section twice.
        assert!(base.clone().deal_scale(bare.clone()).is_err());
        assert!(
            base.clone()
                .deal_scale(
                    DealScale::new("A", ["A", "B"])
                        .unwrap()
                        .brokerage(0.01)
                        .unwrap()
                )
                .is_err()
        );
        assert!(
            base.clone()
                .deal_scale(
                    DealScale::new("d", ["A", "C"])
                        .unwrap()
                        .brokerage(0.01)
                        .unwrap()
                )
                .is_err()
        );
        assert!(
            tower
                .clone()
                .deal_scale(
                    DealScale::new("e", ["A", "B"])
                        .unwrap()
                        .brokerage(0.01)
                        .unwrap()
                )
                .is_err()
        );
        // A section without a premium.
        let unpriced = Tower::new(vec![
            Layer::xol("A", 10.0, 0.0).unwrap(),
            Layer::xol("B", 10.0, 10.0)
                .unwrap()
                .premium_rate(1.0, 10.0)
                .unwrap(),
        ])
        .unwrap();
        assert!(unpriced.deal_scale(bare.brokerage(0.01).unwrap()).is_err());
        // A label inside NMBF in one section and outside in the other.
        let mut mixed = tower.clone();
        mixed.layers[1].deductions[0].inside_nmbf = false;
        let result = mixed.apply_annual(&[100.0]).unwrap();
        assert!(mixed.deal_ratios(&result, "deal").is_err());
        assert!(mixed.deal_ratios(&result, "A").is_ok());
    }

    #[test]
    fn the_ledger_books_a_deal_atom_by_atom_as_its_ratios() {
        let tower = two_sections(deal_scale().brokerage(0.01).unwrap());
        let values = vec![20.0, 60.0, 160.0, 240.0];
        let gross = Source::new(values.clone(), vec![0.4, 0.3, 0.2, 0.1]).unwrap();
        let ledger = tower.deal_ledger(&gross, "deal").unwrap();
        let ratios = tower
            .deal_ratios(&tower.apply_annual(&values).unwrap(), "deal")
            .unwrap();
        // The result at each atom is the margin times the deals' premium.
        let margin = draws(&ratios, &["margin"]);
        for (r, m) in ledger.result().values().iter().zip(&margin) {
            assert!((r - 200.0 * m).abs() < 1e-9, "{r} {m}");
        }
        // Each leg is the ratio times the premium, with the sign of an
        // obligation.
        for (row, ratio) in [
            ("Loss", "loss_ratio"),
            ("Ceding commission", "ceding_commission"),
            ("Brokerage", "brokerage"),
            ("tax", "tax"),
        ] {
            let leg = ledger.row(row).unwrap().values();
            for (l, x) in leg.iter().zip(draws(&ratios, &[ratio])) {
                assert!((l + 200.0 * x).abs() < 1e-9, "{row}");
            }
        }
        assert_eq!(ledger.row("Premium").unwrap().values(), [200.0; 4]);
        // A section alone books its own terms.
        let a = tower.deal_ledger(&gross, "A").unwrap();
        assert!(a.row("Brokerage").is_none() && a.row("tax").is_some());
    }

    #[test]
    fn the_ledger_refuses_what_an_annual_loss_cannot_carry() {
        let tower = two_sections(deal_scale());
        let joint = Source::joint(vec![10.0, 20.0], vec![5.0, 10.0], vec![0.5, 0.5]).unwrap();
        assert!(tower.deal_ledger(&joint, "deal").is_err());
        let gross = Source::new(vec![10.0, 20.0], vec![0.5, 0.5]).unwrap();
        assert!(tower.deal_ledger(&gross, "nope").is_err());
        let reinstated = Tower::new(vec![
            Layer::xol("L", 10.0, 5.0)
                .unwrap()
                .paid_reinstatements(2.0, vec![1.0])
                .unwrap(),
        ])
        .unwrap();
        assert!(reinstated.deal_ledger(&gross, "L").is_err());
    }
}
