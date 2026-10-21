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
//! See `docs/design/aggregate.md`, "Deal deductions and ratios".

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution};

use crate::pnl::ExpenseBasis;
use crate::reinsurance::{Commission, Layer, Tower, checked_anchors};

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

impl Tower {
    /// Fails if a layer reads a sliding scale (commission or brokerage)
    /// before annual terms it has: `what` sees only the ceded loss.
    pub(crate) fn check_scales_on_ceded(&self, what: &str) -> Result<()> {
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

    /// The deal ratios of one layer in each simulation of `result`, a
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
    /// Fails if the tower has no layer named `layer`, if `result` lacks
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
        let l = self
            .layers
            .iter()
            .find(|l| l.name == layer)
            .ok_or_else(|| Error::Data(format!("the tower has no layer named {layer:?}")))?;
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
        let column = |kind: &str, year: &Option<KeyValue>| -> Result<usize> {
            let mut key: ComponentKey = vec![KeyValue::from(kind), KeyValue::from(layer)];
            key.extend(year.iter().cloned());
            pd.component_index(&key).ok_or_else(|| {
                Error::Data(format!(
                    "the result has no ({kind}, {layer}) component: is it this tower's?"
                ))
            })
        };
        // Per year: the ceded loss, the swing premium, CC, PC, BK and each
        // deduction's column.
        struct Cols {
            ceded: usize,
            before: Option<usize>,
            swing: Option<usize>,
            terms: Vec<usize>,
        }
        let mut names: Vec<&str> = Vec::new();
        if l.commission.is_some() {
            names.push("ceding_commission");
        }
        if l.profit_commission.is_some() {
            names.push("profit_commission");
        }
        if l.brokerage.is_some() {
            names.push("brokerage");
        }
        names.extend(l.deductions.iter().map(|d| d.label.as_str()));
        let outside: Vec<bool> = (0..names.len() - l.deductions.len())
            .map(|_| false)
            .chain(l.deductions.iter().map(|d| !d.inside_nmbf))
            .collect();
        let cols = years
            .iter()
            .map(|y| {
                Ok(Cols {
                    ceded: column("ceded", y)?,
                    before: l
                        .scales_before_annual_terms
                        .then(|| column("loss_before_terms", y))
                        .transpose()?,
                    swing: l
                        .swing
                        .is_some()
                        .then(|| column("swing_premium", y))
                        .transpose()?,
                    terms: names.iter().map(|k| column(k, y)).collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ratio_names = vec!["loss_ratio"];
        if l.scales_before_annual_terms {
            ratio_names.push("loss_ratio_before_terms");
        }
        ratio_names.extend(&names);
        ratio_names.extend(["total_deductions", "combined_ratio", "nmbf", "margin"]);
        let width = ratio_names.len() * years.len();
        let mut draws = Vec::with_capacity(pd.n_sims() * width);
        for i in 0..pd.n_sims() {
            let row = pd.row(i).expect("a simulation within range");
            for c in &cols {
                let premium = c.swing.map_or(l.premium, |k| row[k]);
                if premium.is_nan() || premium <= 0.0 {
                    return Err(Error::Data(format!(
                        "deal ratios need a positive premium; layer {layer:?} has {premium} in simulation {i}"
                    )));
                }
                let lr = row[c.ceded] / premium;
                draws.push(lr);
                if let Some(k) = c.before {
                    draws.push(row[k] / premium);
                }
                let (mut total, mut fees) = (0.0, 0.0);
                for (&k, &out) in c.terms.iter().zip(&outside) {
                    let r = row[k] / premium;
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
        }
        let components: Vec<ComponentKey> = years
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
        if term {
            out_dims.push("year".into());
        }
        let provenance = pd.provenance().clone().param("deal_ratios", layer);
        PredictiveDistribution::from_draws(out_dims, components, draws, provenance)
    }
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
}
