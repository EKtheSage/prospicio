//! A cover's quote: from its risk-loaded price to the premium the cedant
//! pays, and what a quoted premium gives the reinsurer.
//!
//! The risk premium `R` is the expected loss plus the margin for the
//! capital held, from [`risk_load::price`](crate::risk_load::price) or
//! [`price_grid`]. Internal expenses `e`
//! and brokerage `b` are fractions of the premium itself and a fixed
//! expense `f` an amount, so they gross up rather than add:
//!
//! `P = (R + f) / (1 − e − b)`.
//!
//! The ratios the market quotes are all on that premium: the rate on line
//! `P / limit` and its inverse the payback, the multiple `P / E[X]` and
//! the loss ratio `E[X] / P`. [`Quote::at`] reads a quoted premium the
//! other way: the reinsurer keeps `P (1 − e − b) − f` as its risk premium,
//! and the margin and return on capital follow with the same expected loss
//! and assets.
//!
//! The premium side ([`PremiumTerms`]) adds what a cover collects beyond
//! its deposit `D` and pays back from it, where it is linear in `D`:
//! expected reinstatement premium `κ D` (`κ` per unit of deposit, as a
//! tower's grid gives it), with internal expenses and premium tax on it
//! and, by default, brokerage too, and a flat ceding commission `c D`.
//! The reinsurer keeps
//!
//! `R = D [(1 + κ)(1 − e − τ) − b (1 + κ) − c] − f`
//!
//! (`b κ` dropped without brokerage on reinstatement premium), so the
//! technical deposit is `(R + f)` over the bracket. A sliding-scale
//! commission and a profit commission depend on the year's loss and so on
//! the premium: [`Quote::technical_for_layer`] solves for the deposit under
//! them, and [`Quote::at_with_commission`] reads them back at a quoted one.
//! A swing-rated premium follows the loss and has no deposit to solve for:
//! it is read back at its expected premium.
//!
//! Tax comes in two forms. A premium tax `τ` is a fraction of the premium,
//! grossed up as brokerage is. An income tax `t` is charged on the margin,
//! and the return a price asks for is read as after tax: the technical
//! quote raises the risk premium until `(1 − t)` times the margin over the
//! capital is the price's return on capital, and
//! [`Quote::return_on_capital`] is after tax. See `docs/design/pareto.md`.

use prospicio_core::{Error, Result};

use prospicio_aggregate::{Commission, Layer, Tower, TowerGrids};
use prospicio_prob::{Distortion, Grid};

use crate::risk_load::{PremiumRule, Price, price_grid};

/// The loads between the risk premium and the premium charged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loads {
    /// Internal expenses, a fraction of the premium.
    pub expenses: f64,
    /// Brokerage, a fraction of the premium.
    pub brokerage: f64,
    /// A fixed expense per contract, an amount.
    pub fixed: f64,
    /// Premium tax, a fraction of the premium (an excise or insurance
    /// premium tax); zero by default.
    pub premium_tax: f64,
    /// Income tax on the margin; zero by default. The return a price asks
    /// for is then after tax.
    pub income_tax: f64,
}

impl Loads {
    /// Expense and brokerage ratios with `expenses + brokerage < 1`, and a
    /// fixed expense; all non-negative. No tax.
    pub fn new(expenses: f64, brokerage: f64, fixed: f64) -> Result<Self> {
        for (name, v) in [
            ("expenses", expenses),
            ("brokerage", brokerage),
            ("fixed", fixed),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and non-negative",
                });
            }
        }
        Self {
            expenses,
            brokerage,
            fixed,
            premium_tax: 0.0,
            income_tax: 0.0,
        }
        .checked()
    }

    /// The same loads with a premium tax, a fraction of the premium in
    /// `[0, 1)`; expenses, brokerage and the tax must sum below 1.
    pub fn with_premium_tax(self, rate: f64) -> Result<Self> {
        check_rate("premium_tax", rate)?;
        Self {
            premium_tax: rate,
            ..self
        }
        .checked()
    }

    /// The same loads with an income tax on the margin, a rate in `[0, 1)`.
    pub fn with_income_tax(self, rate: f64) -> Result<Self> {
        check_rate("income_tax", rate)?;
        Ok(Self {
            income_tax: rate,
            ..self
        })
    }

    fn checked(self) -> Result<Self> {
        if self.keep() <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "brokerage",
                value: self.brokerage,
                reason: "expenses, brokerage and premium tax must leave some premium, sum below 1",
            });
        }
        Ok(self)
    }

    /// The share of each unit of premium left after expenses, brokerage and
    /// premium tax, `1 − e − b − τ`.
    pub fn keep(&self) -> f64 {
        1.0 - self.expenses - self.brokerage - self.premium_tax
    }
}

/// What a cover collects beyond its deposit and pays back from it, where
/// that is linear in the deposit; see the [module](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PremiumTerms {
    /// Expected reinstatement premium per unit of deposit, `κ`.
    pub reinstatement_rate: f64,
    /// Whether brokerage is charged on reinstatement premium; `true` by
    /// default, as the market charges it.
    pub brokerage_on_reinstatements: bool,
    /// A flat ceding commission, a fraction of the deposit.
    pub commission: f64,
}

impl Default for PremiumTerms {
    fn default() -> Self {
        Self::none()
    }
}

impl PremiumTerms {
    /// No reinstatement premium and no commission: the quote of the deposit
    /// alone.
    pub fn none() -> Self {
        Self {
            reinstatement_rate: 0.0,
            brokerage_on_reinstatements: true,
            commission: 0.0,
        }
    }

    /// Expected reinstatement premium `κ` per unit of deposit (non-negative)
    /// and a flat ceding commission on the deposit in `[0, 1)`.
    pub fn new(reinstatement_rate: f64, commission: f64) -> Result<Self> {
        if !(reinstatement_rate.is_finite() && reinstatement_rate >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "reinstatement_rate",
                value: reinstatement_rate,
                reason: "must be finite and non-negative",
            });
        }
        check_rate("commission", commission)?;
        Ok(Self {
            reinstatement_rate,
            commission,
            ..Self::none()
        })
    }

    /// The same terms with brokerage charged on the deposit only.
    pub fn without_brokerage_on_reinstatements(self) -> Self {
        Self {
            brokerage_on_reinstatements: false,
            ..self
        }
    }

    /// A layer's linear terms from its grid result: `κ` is the expected
    /// reinstatement premium (from [`TowerGrids`] or [`AnnualGrids`],
    /// `expected_reinstatement_premium`) over the layer's premium, which it
    /// is proportional to, and a flat ceding commission is taken as it is.
    /// A sliding-scale commission, a profit commission and a swing-rated
    /// premium are left out: read them back with
    /// [`Quote::at_with_commission`].
    ///
    /// [`TowerGrids`]: prospicio_aggregate::TowerGrids
    /// [`AnnualGrids`]: prospicio_aggregate::AnnualGrids
    pub fn from_layer(layer: &Layer, expected_reinstatement_premium: f64) -> Result<Self> {
        let rate = if expected_reinstatement_premium == 0.0 {
            0.0
        } else if layer.premium > 0.0 {
            expected_reinstatement_premium / layer.premium
        } else {
            return Err(Error::InvalidParameter {
                name: "expected_reinstatement_premium",
                value: expected_reinstatement_premium,
                reason: "is not zero but the layer has no premium to divide it by",
            });
        };
        let commission = match layer.commission {
            Some(Commission::Flat(c)) => c,
            _ => 0.0,
        };
        Self::new(rate, commission)
    }
}

/// A cover's premium with its build-up and ratios; see the
/// [module](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quote {
    /// The deposit premium the cedant pays upfront.
    pub premium: f64,
    /// Expected loss `E[X]`.
    pub expected_loss: f64,
    /// The risk premium the reinsurer keeps for the loss and the capital:
    /// all the premium collected less expenses, brokerage, premium tax and
    /// commissions.
    pub risk_premium: f64,
    /// Assets backing the loss.
    pub assets: f64,
    /// The loads.
    pub loads: Loads,
    /// The premium side linear in the deposit.
    pub terms: PremiumTerms,
    /// Expected commissions read back at this premium beyond the flat one
    /// in `terms` (a sliding scale, a profit commission), an amount; zero
    /// unless set by [`Quote::at_with_commission`].
    pub other_commission: f64,
    /// The cover's limit at the placed share, for the rate on line; `None`
    /// for a cover without one, such as a quota share.
    pub limit: Option<f64>,
}

impl Quote {
    /// The technical premium of a risk-loaded price: `(R + f) / (1 − e −
    /// b − τ)`, with `R` the price's premium (raised for an income tax).
    ///
    /// ```
    /// use prospicio_pricing::quote::{Loads, Quote};
    /// use prospicio_pricing::risk_load::Price;
    ///
    /// // A risk premium of 1.24 for an expected loss of 0.586 on a limit of 5.
    /// let price = Price { expected_loss: 0.586, premium: 1.24, assets: 7.778 };
    /// let q = Quote::technical(&price, Loads::new(0.03, 0.10, 0.0).unwrap(), Some(5.0)).unwrap();
    /// assert!((q.premium - 1.24 / 0.87).abs() < 1e-12);
    /// assert!((q.rate_on_line().unwrap() - q.premium / 5.0).abs() < 1e-15);
    /// assert!((q.brokerage() - 0.1 * q.premium).abs() < 1e-15);
    /// ```
    pub fn technical(price: &Price, loads: Loads, limit: Option<f64>) -> Result<Self> {
        Self::technical_with(price, loads, PremiumTerms::none(), limit)
    }

    /// The technical deposit with the premium side: the deposit at which
    /// all it collects, net of the loads and commissions, is the price's
    /// premium, `D = (R + f) / [(1 + κ)(1 − e − τ) − b (1 + κ) − c]`. With
    /// an income tax `t` the risk premium is first raised so that the
    /// margin after tax over the capital is the price's return on capital:
    /// `R = (ρ a + (1 − t) E[X]) / (1 − t + ρ)`, with `ρ` that return and
    /// `a` the assets; for a cost-of-capital price at `r` this is the same
    /// price at `r / (1 − t)`.
    ///
    /// ```
    /// use prospicio_pricing::quote::{Loads, PremiumTerms, Quote};
    /// use prospicio_pricing::risk_load::Price;
    ///
    /// let price = Price { expected_loss: 0.586, premium: 1.24, assets: 7.778 };
    /// // Half a reinstatement expected per unit of deposit, brokerage on it.
    /// let terms = PremiumTerms::new(0.5, 0.0).unwrap();
    /// let q = Quote::technical_with(&price, Loads::new(0.03, 0.10, 0.0).unwrap(), terms, None)
    ///     .unwrap();
    /// assert!((q.premium - 1.24 / (1.5 * 0.87)).abs() < 1e-12);
    /// assert!((q.premium_in_all() - 1.24 / 0.87).abs() < 1e-12);
    /// // An income tax keeps the return after tax at the price's.
    /// let taxed = Loads::new(0.03, 0.10, 0.0).unwrap().with_income_tax(0.21).unwrap();
    /// let t = Quote::technical_with(&price, taxed, terms, None).unwrap();
    /// assert!((t.return_on_capital() - price.return_on_capital()).abs() < 1e-12);
    /// assert!(t.premium > q.premium);
    /// ```
    pub fn technical_with(
        price: &Price,
        loads: Loads,
        terms: PremiumTerms,
        limit: Option<f64>,
    ) -> Result<Self> {
        check_limit(limit)?;
        let keep = keep_per_deposit(&loads, &terms)?;
        let t = loads.income_tax;
        let risk_premium = if t == 0.0 {
            price.premium
        } else {
            let rho = price.return_on_capital();
            (rho * price.assets + (1.0 - t) * price.expected_loss) / (1.0 - t + rho)
        };
        let premium = (risk_premium + loads.fixed) / keep;
        Self {
            premium,
            expected_loss: price.expected_loss,
            risk_premium,
            assets: price.assets,
            loads,
            terms,
            other_commission: 0.0,
            limit,
        }
        .funded(premium)
    }

    /// The technical deposit of a layer whose commissions follow its loss
    /// ratio: a sliding-scale ceding commission or a profit commission,
    /// which depend on the deposit through the loss ratio, so the deposit
    /// is solved for rather than read off in closed form. A one-year
    /// no-claims bonus enters at its expected rate per unit of deposit
    /// ([`Layer::expected_no_claims_bonus`]), with the commissions in
    /// `other_commission`; a bonus over several years is refused. With `C(D)` the
    /// expected sliding-scale and profit commission at a deposit `D`
    /// ([`Layer::expected_loss_sensitive_commissions`] over the layer's
    /// ceded grid), the deposit keeps the price's risk premium `R` (raised
    /// for an income tax as in [`technical_with`](Self::technical_with)):
    ///
    /// `D [(1 + κ)(1 − e − τ) − b (1 + κ) − c] − f − C(D) = R`.
    ///
    /// `κ` and a flat commission `c` are read off the layer as
    /// [`PremiumTerms::from_layer`] reads them, with brokerage on
    /// reinstatement premium if `brokerage_on_reinstatements`, and the
    /// limit for the rate on line is the layer's at the placed share. A
    /// layer with neither term is [`technical_with`](Self::technical_with)
    /// of those terms. `price` is the price of `ceded`.
    ///
    /// The commission per unit of deposit lies between the scale's lowest
    /// and highest rates, plus up to `share × (1 − allowance)` for a profit
    /// commission, so the root lies between `(R + f)` over the bracket less
    /// the lowest rate and `(R + f)` over the bracket less the highest
    /// rates. The kept amount need not rise with the deposit: within a
    /// scale's band it is `D (k − α) + β E[X]` for a band `α − β LR`, which
    /// falls as `D` rises when the band's intercept `α` exceeds what the
    /// loads leave, `k`, so a steep scale can meet the risk premium at more
    /// than one deposit. The solve returns the lowest: it scans the
    /// interval at 64 points for the first sign change and bisects it to a
    /// relative `1e-12`. Fails when the highest rates leave no premium (no
    /// deposit is sure to meet the risk premium), for a swing-rated layer
    /// (its premium follows its loss, so it has no deposit to solve for;
    /// `docs/design/pareto.md`) and for a scale read before annual terms.
    ///
    /// ```
    /// use prospicio_aggregate::Layer;
    /// use prospicio_pricing::quote::{Loads, Quote};
    /// use prospicio_pricing::risk_load::{price_grid, PremiumRule};
    /// use prospicio_prob::{Distortion, Grid};
    ///
    /// // A quota share's ceded loss: 0 to 100 in steps of 20.
    /// let ceded = Grid::new(20.0, vec![0.05, 0.15, 0.3, 0.3, 0.15, 0.05]).unwrap();
    /// let qs = Layer::quota_share("QS", 0.5).unwrap()
    ///     .premium_rate(1.0, 200.0).unwrap()
    ///     .sliding_scale(vec![(0.35, 0.50), (0.20, 0.70)]).unwrap();
    /// let p = price_grid(&ceded, &PremiumRule::cost_of_capital(0.1).unwrap(),
    ///                    &Distortion::tvar(0.99).unwrap()).unwrap();
    /// let loads = Loads::new(0.02, 0.0, 0.0).unwrap();
    /// let q = Quote::technical_for_layer(&p, loads, &qs, &ceded, 0.0, true).unwrap();
    /// // At the solved deposit the commission is the scale's, and the return
    /// // is the 10% the price asks for.
    /// let (cc, _) = qs.expected_loss_sensitive_commissions(q.premium, &ceded).unwrap();
    /// assert!((q.other_commission - cc).abs() < 1e-12);
    /// assert!((q.return_on_capital() - 0.1).abs() < 1e-9);
    /// ```
    pub fn technical_for_layer(
        price: &Price,
        loads: Loads,
        layer: &Layer,
        ceded: &Grid,
        expected_reinstatement_premium: f64,
        brokerage_on_reinstatements: bool,
    ) -> Result<Self> {
        if layer.swing.is_some() {
            return Err(Error::InvalidParameter {
                name: "swing",
                value: f64::NAN,
                reason: "a swing-rated layer's premium follows its loss, so it has no deposit \
                         to solve for; quote it at its expected premium with `at`",
            });
        }
        let mut terms = PremiumTerms::from_layer(layer, expected_reinstatement_premium)?;
        if !brokerage_on_reinstatements {
            terms = terms.without_brokerage_on_reinstatements();
        }
        let limit = layer.limit.is_finite().then_some(layer.limit * layer.share);
        let base = Self::technical_with(price, loads, terms, limit)?;
        let (low, high) = match &layer.commission {
            Some(Commission::SlidingScale(anchors)) => anchors
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(c, _)| {
                    (lo.min(c), hi.max(c))
                }),
            _ => (0.0, 0.0),
        };
        let profit = layer
            .profit_commission
            .map_or(0.0, |pc| pc.share * (1.0 - pc.allowance).max(0.0));
        // A no-claims bonus is linear in the deposit: its expected rate is
        // the same at every trial deposit, a flat commission at that rate
        // (#592, `docs/design/aggregate.md`, N6).
        let bonus = match &layer.no_claims_bonus {
            Some(_) => layer.expected_no_claims_bonus(1.0, ceded)?,
            None => 0.0,
        };
        let (low, high) = (low + bonus, high + bonus);
        let sliding = matches!(layer.commission, Some(Commission::SlidingScale(_)));
        if !sliding && layer.profit_commission.is_none() && layer.no_claims_bonus.is_none() {
            return Ok(base);
        }
        let keep = keep_per_deposit(&loads, &terms)?;
        let target = base.risk_premium + loads.fixed;
        if keep - high - profit <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "commission",
                value: high + profit,
                reason: "the highest sliding-scale and profit commission rates leave no \
                         premium after the loads: no deposit is sure to meet the risk premium",
            });
        }
        if target <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: target,
                reason: "the risk premium and fixed expense must be positive",
            });
        }
        let commission = |d: f64| -> Result<f64> {
            let (cc, pc) = layer.expected_loss_sensitive_commissions(d, ceded)?;
            Ok(cc + pc + bonus * d)
        };
        let gap = |d: f64| -> Result<f64> { Ok(d * keep - commission(d)? - target) };
        let (d_low, d_high) = (target / (keep - low), target / (keep - high - profit));
        let deposit = if gap(d_low)? >= 0.0 {
            d_low
        } else {
            // The first sign change on a scan of the bracket, then bisection.
            const SCAN: usize = 64;
            let step = (d_high - d_low) / SCAN as f64;
            let mut lo = d_low;
            let mut hi = d_high;
            for k in 1..=SCAN {
                let d = if k == SCAN {
                    d_high
                } else {
                    d_low + k as f64 * step
                };
                if gap(d)? >= 0.0 {
                    hi = d;
                    break;
                }
                lo = d;
            }
            for _ in 0..200 {
                if hi - lo <= 1e-12 * hi {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                if gap(mid)? >= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            0.5 * (lo + hi)
        };
        base.at_with_commission(deposit, commission(deposit)?)
    }

    /// The same cover at a quoted deposit: the expected loss, assets, loads
    /// and premium side kept, the risk premium what the quote leaves after
    /// them. Fails when that risk premium reaches the assets, leaving no
    /// capital.
    ///
    /// ```
    /// use prospicio_pricing::quote::{Loads, Quote};
    /// use prospicio_pricing::risk_load::{price, PremiumRule};
    /// use prospicio_prob::{Distortion, Sampled};
    ///
    /// let losses = Sampled::new(vec![0.0, 0.0, 2.0, 6.0]).unwrap();
    /// let p = price(&losses, &PremiumRule::cost_of_capital(0.1).unwrap(),
    ///               &Distortion::tvar(0.5).unwrap()).unwrap();
    /// let tech = Quote::technical(&p, Loads::new(0.03, 0.1, 0.0).unwrap(), None).unwrap();
    /// // At the technical premium the return is the 10% asked for.
    /// assert!((tech.at(tech.premium).unwrap().return_on_capital() - 0.1).abs() < 1e-12);
    /// // A higher quote returns more.
    /// assert!(tech.at(1.2 * tech.premium).unwrap().return_on_capital() > 0.1);
    /// ```
    pub fn at(&self, premium: f64) -> Result<Self> {
        self.at_with_commission(premium, 0.0)
    }

    /// The cover at a quoted deposit with further expected commissions read
    /// back at it, an amount beyond the flat commission in the terms: a
    /// sliding-scale or profit commission from a tower's grid run at that
    /// deposit (`expected_ceding_commission` and
    /// `expected_profit_commission`). For a swing-rated premium pass its
    /// expected premium (`expected_premium`) as the deposit.
    pub fn at_with_commission(&self, premium: f64, commission: f64) -> Result<Self> {
        for (name, v) in [("premium", premium), ("commission", commission)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and non-negative",
                });
            }
        }
        let keep = keep_per_deposit(&self.loads, &self.terms)?;
        Self {
            premium,
            risk_premium: premium * keep - self.loads.fixed - commission,
            other_commission: commission,
            ..*self
        }
        .funded(premium)
    }

    fn funded(self, premium: f64) -> Result<Self> {
        if self.risk_premium >= self.assets {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: premium,
                reason: "leaves a risk premium at or above the assets: no capital",
            });
        }
        Ok(self)
    }

    /// Expected reinstatement premium, `κ D`.
    pub fn reinstatement_premium(&self) -> f64 {
        self.terms.reinstatement_rate * self.premium
    }

    /// All the premium the cedant is expected to pay, the deposit plus the
    /// expected reinstatement premium, `(1 + κ) D`.
    pub fn premium_in_all(&self) -> f64 {
        self.premium + self.reinstatement_premium()
    }

    /// Internal expenses, `e (1 + κ) D + f`.
    pub fn expenses(&self) -> f64 {
        self.loads.expenses * self.premium_in_all() + self.loads.fixed
    }

    /// Brokerage, `b D`, and `b κ D` on reinstatement premium unless the
    /// terms exclude it.
    pub fn brokerage(&self) -> f64 {
        let base = if self.terms.brokerage_on_reinstatements {
            self.premium_in_all()
        } else {
            self.premium
        };
        self.loads.brokerage * base
    }

    /// Premium tax, `τ (1 + κ) D`.
    pub fn premium_tax(&self) -> f64 {
        self.loads.premium_tax * self.premium_in_all()
    }

    /// Commissions paid back to the cedant: the flat `c D` and any read
    /// back.
    pub fn commission(&self) -> f64 {
        self.terms.commission * self.premium + self.other_commission
    }

    /// The premium net of brokerage, what the reinsurer receives.
    pub fn net_premium(&self) -> f64 {
        self.premium_in_all() - self.brokerage()
    }

    /// The margin for the capital before income tax, `R − E[X]`.
    pub fn margin(&self) -> f64 {
        self.risk_premium - self.expected_loss
    }

    /// Income tax on the margin, `t (R − E[X])`; negative, a credit, on a
    /// negative margin.
    pub fn income_tax(&self) -> f64 {
        self.loads.income_tax * self.margin()
    }

    /// Capital, the assets less the risk premium.
    pub fn capital(&self) -> f64 {
        self.assets - self.risk_premium
    }

    /// Return on capital after income tax, `(1 − t)` times the margin over
    /// capital; the margin over capital without an income tax.
    pub fn return_on_capital(&self) -> f64 {
        (self.margin() - self.income_tax()) / self.capital()
    }

    /// Return on capital before income tax, margin over capital.
    pub fn pre_tax_return_on_capital(&self) -> f64 {
        self.margin() / self.capital()
    }

    /// Loss ratio `E[X] / ((1 + κ) D)` on all the premium.
    pub fn loss_ratio(&self) -> f64 {
        self.expected_loss / self.premium_in_all()
    }

    /// Multiple `(1 + κ) D / E[X]`, the inverse of the loss ratio.
    pub fn multiple(&self) -> f64 {
        self.premium_in_all() / self.expected_loss
    }

    /// Rate on line `D / limit` on the deposit, when the cover has a limit.
    pub fn rate_on_line(&self) -> Option<f64> {
        self.limit.map(|l| self.premium / l)
    }

    /// Payback `limit / D`: the loss-free years the deposit takes to pay
    /// one full limit.
    pub fn payback(&self) -> Option<f64> {
        self.limit.map(|l| l / self.premium)
    }
}

/// A whole tower quoted at once: each layer standalone from its own ceded
/// grid, and the total cession, every layer together, as one cover. Each
/// standalone price holds capital for its layer alone, so the layers'
/// premiums add up to more than the total's: the difference is the
/// [diversification](Self::diversification) of writing the tower as one
/// placement. No share of the total is allocated to the layers.
#[derive(Debug, Clone, PartialEq)]
pub struct TowerQuote {
    /// Each layer's standalone quote, in tower order.
    pub layers: Vec<Quote>,
    /// The quote of the total cession.
    pub total: Quote,
}

impl TowerQuote {
    /// The technical quote of each layer and of the total cession. Each
    /// layer is priced on `grids.ceded`, with the reinstatement rate and
    /// flat ceding commission its grid and terms give
    /// ([`PremiumTerms::from_layer`]) and its limit at the placed share for
    /// the rate on line. A layer with a sliding-scale or profit commission
    /// has its deposit solved with those commissions
    /// ([`Quote::technical_for_layer`]); a swing-rated layer is quoted on its
    /// flat terms, to be read back at its expected premium. The total is
    /// priced on `grids.total_ceded`, with the layers' reinstatement rates
    /// and flat commissions weighted by their technical deposits and no
    /// limit: the layers' sliding-scale and profit commissions, which read
    /// each layer's own loss ratio, are not carried to the total.
    ///
    /// Fails when the total cession is not one compound total
    /// (`grids.total_ceded` is `None`): several layers with annual terms
    /// (paid reinstatements, which cap each layer's year, aggregate limits
    /// or deductibles), or an aggregate cover after per-occurrence stages,
    /// cede a sum of dependent annual totals, which needs simulated events
    /// ([`Tower::apply`]) to price. A single layer with annual terms, or
    /// any number without, has a total.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_pricing::quote::{Loads, TowerQuote};
    /// use prospicio_pricing::risk_load::PremiumRule;
    /// use prospicio_prob::{Distortion, Grid, Poisson};
    ///
    /// let sev = Grid::new(1.0, vec![0.0, 0.4, 0.3, 0.2, 0.1]).unwrap();
    /// let tower = Tower::new(vec![
    ///     Layer::xol("1x1", 1.0, 1.0).unwrap(),
    ///     Layer::xol("2x2", 2.0, 2.0).unwrap(),
    /// ]).unwrap();
    /// let grids = tower.on_grid(&Poisson::new(3.0).unwrap(), &sev, 256).unwrap();
    /// let q = TowerQuote::technical(
    ///     &tower, &grids, &PremiumRule::cost_of_capital(0.1).unwrap(),
    ///     &Distortion::tvar(0.99).unwrap(), Loads::new(0.03, 0.1, 0.0).unwrap(), true,
    /// ).unwrap();
    /// // Writing the layers together needs less capital than one by one.
    /// assert!(q.diversification() > 0.0);
    /// let el: f64 = q.layers.iter().map(|l| l.expected_loss).sum();
    /// assert!((q.total.expected_loss - el).abs() < 1e-9);
    /// ```
    pub fn technical(
        tower: &Tower,
        grids: &TowerGrids,
        rule: &PremiumRule,
        assets: &Distortion,
        loads: Loads,
        brokerage_on_reinstatements: bool,
    ) -> Result<Self> {
        Self::from_parts(
            &tower.layers,
            &grids.ceded,
            &grids.expected_reinstatement_premium,
            grids.total_ceded.as_ref(),
            rule,
            assets,
            loads,
            brokerage_on_reinstatements,
        )
    }

    /// [`technical`](Self::technical) from the pieces of a tower's grid
    /// result: its layers, their ceded grids and expected reinstatement
    /// premiums, and the total cession (`None` fails, as there).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        layers: &[Layer],
        ceded: &[Grid],
        expected_reinstatement_premium: &[f64],
        total_ceded: Option<&Grid>,
        rule: &PremiumRule,
        assets: &Distortion,
        loads: Loads,
        brokerage_on_reinstatements: bool,
    ) -> Result<Self> {
        if ceded.len() != layers.len() || expected_reinstatement_premium.len() != layers.len() {
            return Err(Error::InvalidParameter {
                name: "ceded",
                value: ceded.len() as f64,
                reason: "must hold one ceded grid and reinstatement premium per layer",
            });
        }
        let total_grid = total_ceded.ok_or(Error::InvalidParameter {
            name: "total_ceded",
            value: f64::NAN,
            reason: "the total cession is not one compound total: several layers with \
                     annual terms (reinstatements, aggregate limits or deductibles), or an \
                     aggregate cover after per-occurrence stages, cede a sum of dependent \
                     annual totals; quote the layers one by one, or the total on simulated \
                     events",
        })?;
        let with_brokerage = |t: PremiumTerms| {
            if brokerage_on_reinstatements {
                t
            } else {
                t.without_brokerage_on_reinstatements()
            }
        };
        let quotes = layers
            .iter()
            .zip(ceded)
            .zip(expected_reinstatement_premium)
            .map(|((layer, grid), &rp)| {
                let price = price_grid(grid, rule, assets)?;
                if layer.swing.is_none() {
                    return Quote::technical_for_layer(
                        &price,
                        loads,
                        layer,
                        grid,
                        rp,
                        brokerage_on_reinstatements,
                    );
                }
                let terms = with_brokerage(PremiumTerms::from_layer(layer, rp)?);
                let limit = layer.limit.is_finite().then_some(layer.limit * layer.share);
                Quote::technical_with(&price, loads, terms, limit)
            })
            .collect::<Result<Vec<_>>>()?;
        let deposits: f64 = quotes.iter().map(|q| q.premium).sum();
        let weighted = |f: fn(&PremiumTerms) -> f64| {
            if deposits > 0.0 {
                quotes.iter().map(|q| f(&q.terms) * q.premium).sum::<f64>() / deposits
            } else {
                0.0
            }
        };
        let terms = with_brokerage(PremiumTerms::new(
            weighted(|t| t.reinstatement_rate),
            weighted(|t| t.commission),
        )?);
        let total =
            Quote::technical_with(&price_grid(total_grid, rule, assets)?, loads, terms, None)?;
        Ok(Self {
            layers: quotes,
            total,
        })
    }

    /// The layers' technical deposits added up less the total's: what
    /// quoting the tower as one placement saves the cedant.
    pub fn diversification(&self) -> f64 {
        self.layers.iter().map(|q| q.premium).sum::<f64>() - self.total.premium
    }
}

/// The risk premium kept per unit of deposit before the fixed expense,
/// `(1 + κ)(1 − e − τ) − b (1 + κ or 1) − c`.
fn keep_per_deposit(loads: &Loads, terms: &PremiumTerms) -> Result<f64> {
    let k = terms.reinstatement_rate;
    let brokered = if terms.brokerage_on_reinstatements {
        1.0 + k
    } else {
        1.0
    };
    let keep = (1.0 + k) * (1.0 - loads.expenses - loads.premium_tax)
        - loads.brokerage * brokered
        - terms.commission;
    if keep <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "commission",
            value: terms.commission,
            reason: "the loads and commission leave no premium",
        });
    }
    Ok(keep)
}

fn check_rate(name: &'static str, rate: f64) -> Result<()> {
    if !(rate.is_finite() && (0.0..1.0).contains(&rate)) {
        return Err(Error::InvalidParameter {
            name,
            value: rate,
            reason: "must be in [0, 1)",
        });
    }
    Ok(())
}

fn check_limit(limit: Option<f64>) -> Result<()> {
    if let Some(l) = limit {
        if !(l.is_finite() && l > 0.0) {
            return Err(Error::InvalidParameter {
                name: "limit",
                value: l,
                reason: "must be finite and positive",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk_load::{PremiumRule, price_grid};
    use prospicio_prob::{Distortion, Grid};

    fn grid() -> Grid {
        Grid::new(1.0, vec![0.6, 0.2, 0.1, 0.05, 0.03, 0.02]).unwrap()
    }

    #[test]
    fn zero_loads_leave_the_risk_premium() {
        let p = price_grid(
            &grid(),
            &PremiumRule::cost_of_capital(0.12).unwrap(),
            &Distortion::tvar(0.9).unwrap(),
        )
        .unwrap();
        let q = Quote::technical(&p, Loads::new(0.0, 0.0, 0.0).unwrap(), Some(5.0)).unwrap();
        assert_eq!(q.premium, p.premium);
        assert!((q.return_on_capital() - p.return_on_capital()).abs() < 1e-15);
    }

    #[test]
    fn the_loads_net_back_to_the_risk_premium() {
        let p = price_grid(
            &grid(),
            &PremiumRule::Distortion(Distortion::proportional_hazard(0.7).unwrap()),
            &Distortion::tvar(0.95).unwrap(),
        )
        .unwrap();
        let loads = Loads::new(0.05, 0.1, 0.02).unwrap();
        let q = Quote::technical(&p, loads, None).unwrap();
        let parts = q.expected_loss + q.margin() + q.expenses() + q.brokerage();
        assert!((parts - q.premium).abs() < 1e-12);
        assert!((q.premium * loads.keep() - loads.fixed - p.premium).abs() < 1e-12);
        assert_eq!(q.rate_on_line(), None);
        // The quoted premium round-trips, and a lower quote returns less.
        let back = q.at(q.premium).unwrap();
        assert!((back.risk_premium - q.risk_premium).abs() < 1e-12);
        assert!(q.at(0.9 * q.premium).unwrap().return_on_capital() < q.return_on_capital());
        // A quote that funds the assets leaves no capital.
        assert!(q.at(10.0 * q.premium).is_err());
    }

    #[test]
    fn loads_are_checked() {
        assert!(Loads::new(0.6, 0.4, 0.0).is_err());
        assert!(Loads::new(-0.1, 0.1, 0.0).is_err());
        let p = Price {
            expected_loss: 1.0,
            premium: 1.5,
            assets: 5.0,
        };
        assert!(Quote::technical(&p, Loads::new(0.0, 0.1, 0.0).unwrap(), Some(0.0)).is_err());
    }

    fn priced() -> Price {
        price_grid(
            &grid(),
            &PremiumRule::cost_of_capital(0.1).unwrap(),
            &Distortion::tvar(0.9).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn no_terms_and_no_tax_leave_the_quote_unchanged() {
        let p = priced();
        let loads = Loads::new(0.05, 0.1, 0.02).unwrap();
        let a = Quote::technical(&p, loads, Some(5.0)).unwrap();
        let b = Quote::technical_with(&p, loads, PremiumTerms::default(), Some(5.0)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.premium, (p.premium + 0.02) / 0.85);
        assert_eq!(a.premium_in_all(), a.premium);
        assert_eq!(a.return_on_capital(), a.pre_tax_return_on_capital());
    }

    #[test]
    fn the_premium_side_adds_up() {
        let p = priced();
        let loads = Loads::new(0.05, 0.1, 0.02)
            .unwrap()
            .with_premium_tax(0.03)
            .unwrap()
            .with_income_tax(0.21)
            .unwrap();
        for terms in [
            PremiumTerms::new(0.4, 0.15).unwrap(),
            PremiumTerms::new(0.4, 0.15)
                .unwrap()
                .without_brokerage_on_reinstatements(),
        ] {
            let q = Quote::technical_with(&p, loads, terms, Some(5.0)).unwrap();
            let parts = q.expected_loss
                + q.margin()
                + q.expenses()
                + q.brokerage()
                + q.premium_tax()
                + q.commission();
            assert!((parts - q.premium_in_all()).abs() < 1e-12);
            assert!((q.reinstatement_premium() - 0.4 * q.premium).abs() < 1e-15);
            // The return after tax is the price's; before tax it is higher.
            assert!((q.return_on_capital() - p.return_on_capital()).abs() < 1e-12);
            assert!((q.return_on_capital() - 0.1).abs() < 1e-12);
            assert!(q.pre_tax_return_on_capital() > 0.1);
            let after = q.margin() - q.income_tax();
            assert!((after - 0.79 * q.margin()).abs() < 1e-15);
            // The loss ratio is on all the premium, the rate on line on the deposit.
            assert!((q.loss_ratio() - q.expected_loss / (1.4 * q.premium)).abs() < 1e-12);
            assert!((q.rate_on_line().unwrap() - q.premium / 5.0).abs() < 1e-15);
            // A read-back commission lowers the return at the same deposit.
            let back = q.at_with_commission(q.premium, 0.01).unwrap();
            assert!((back.risk_premium - (q.risk_premium - 0.01)).abs() < 1e-12);
            assert!(back.return_on_capital() < q.return_on_capital());
            assert!((q.at(q.premium).unwrap().risk_premium - q.risk_premium).abs() < 1e-12);
        }
        // Brokerage on reinstatements costs a higher deposit.
        let with = Quote::technical_with(&p, loads, PremiumTerms::new(0.4, 0.0).unwrap(), None);
        let without = Quote::technical_with(
            &p,
            loads,
            PremiumTerms::new(0.4, 0.0)
                .unwrap()
                .without_brokerage_on_reinstatements(),
            None,
        );
        assert!(with.unwrap().premium > without.unwrap().premium);
    }

    #[test]
    fn an_income_tax_is_cost_of_capital_grossed_up() {
        // For a cost-of-capital price at r, the after-tax solve is the price
        // at r / (1 - t).
        let t = 0.25;
        let at = |r: f64| {
            price_grid(
                &grid(),
                &PremiumRule::cost_of_capital(r).unwrap(),
                &Distortion::tvar(0.9).unwrap(),
            )
            .unwrap()
        };
        let loads = Loads::new(0.0, 0.0, 0.0)
            .unwrap()
            .with_income_tax(t)
            .unwrap();
        let q = Quote::technical(&at(0.09), loads, None).unwrap();
        assert!((q.risk_premium - at(0.09 / (1.0 - t)).premium).abs() < 1e-12);
    }

    #[test]
    fn the_reinstatement_rate_is_read_off_the_layer() {
        let layer = |deposit: f64| {
            Layer::xol("xl", 10.0, 5.0)
                .unwrap()
                .paid_reinstatements(deposit, vec![1.0])
                .unwrap()
                .ceding_commission(0.1)
                .unwrap()
        };
        let (one, two) = (layer(1.0), layer(2.0));
        let (rp1, rp2) = (
            one.reinstatement_premium(&[9.0, 12.0]),
            two.reinstatement_premium(&[9.0, 12.0]),
        );
        assert!(rp1 > 0.0);
        assert!((rp2 - 2.0 * rp1).abs() < 1e-12);
        let k1 = PremiumTerms::from_layer(&one, rp1).unwrap();
        let k2 = PremiumTerms::from_layer(&two, rp2).unwrap();
        assert!((k1.reinstatement_rate - k2.reinstatement_rate).abs() < 1e-15);
        assert_eq!(k1.commission, 0.1);
        let none = Layer::xol("xl", 10.0, 5.0).unwrap();
        assert!(PremiumTerms::from_layer(&none, 0.5).is_err());
        assert!(PremiumTerms::new(0.0, 1.0).is_err());
        // More than one deposit of reinstatement premium is expected on a
        // heavily used layer with several paid reinstatements.
        assert!(PremiumTerms::new(1.5, 0.0).is_ok());
        assert!(
            Loads::new(0.0, 0.0, 0.0)
                .unwrap()
                .with_income_tax(1.0)
                .is_err()
        );
    }

    #[test]
    fn a_tower_is_quoted_layer_by_layer_and_as_one() {
        use prospicio_prob::Poisson;
        let sev = Grid::new(1.0, vec![0.0, 0.3, 0.3, 0.2, 0.1, 0.1]).unwrap();
        let freq = Poisson::new(2.0).unwrap();
        let rule = PremiumRule::cost_of_capital(0.12).unwrap();
        let assets = Distortion::tvar(0.99).unwrap();
        let loads = Loads::new(0.03, 0.1, 0.0).unwrap();
        // One layer: the total is the layer.
        let one = Tower::new(vec![
            Layer::xol("2x2", 2.0, 2.0)
                .unwrap()
                .paid_reinstatements(0.5, vec![1.0])
                .unwrap(),
        ])
        .unwrap();
        let g = one.on_grid(&freq, &sev, 256).unwrap();
        let q = TowerQuote::technical(&one, &g, &rule, &assets, loads, true).unwrap();
        assert!((q.total.premium - q.layers[0].premium).abs() < 1e-12);
        assert!(q.diversification().abs() < 1e-12);
        assert!(q.layers[0].terms.reinstatement_rate > 0.0);
        assert_eq!(q.layers[0].limit, Some(2.0));
        // Two layers without annual terms, one with a flat commission.
        let two = Tower::new(vec![
            Layer::xol("1x1", 1.0, 1.0)
                .unwrap()
                .deposit_premium(1.0)
                .unwrap()
                .ceding_commission(0.1)
                .unwrap(),
            Layer::xol("2x2", 2.0, 2.0).unwrap(),
        ])
        .unwrap();
        let g = two.on_grid(&freq, &sev, 256).unwrap();
        let q = TowerQuote::technical(&two, &g, &rule, &assets, loads, true).unwrap();
        let el: f64 = q.layers.iter().map(|l| l.expected_loss).sum();
        assert!((q.total.expected_loss - el).abs() < 1e-9);
        assert!(q.diversification() > 0.0);
        for l in &q.layers {
            assert!((l.return_on_capital() - 0.12).abs() < 1e-12);
        }
        assert!((q.total.return_on_capital() - 0.12).abs() < 1e-12);
        // The total's terms are the layers' weighted by their deposits.
        let d: f64 = q.layers.iter().map(|l| l.premium).sum();
        let k: f64 = q
            .layers
            .iter()
            .map(|l| l.terms.reinstatement_rate * l.premium)
            .sum();
        assert!((q.total.terms.reinstatement_rate - k / d).abs() < 1e-12);
        assert!((q.total.terms.commission - 0.1 * q.layers[0].premium / d).abs() < 1e-12);
        // Two layers with paid reinstatements: their annual totals are
        // dependent, so there is no total to quote on the grid.
        let reinstated = Tower::new(vec![
            Layer::xol("1x1", 1.0, 1.0)
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0])
                .unwrap(),
            Layer::xol("2x2", 2.0, 2.0)
                .unwrap()
                .paid_reinstatements(0.5, vec![1.0])
                .unwrap(),
        ])
        .unwrap();
        let g = reinstated.on_grid(&freq, &sev, 256).unwrap();
        assert!(TowerQuote::technical(&reinstated, &g, &rule, &assets, loads, true).is_err());
        // A stop-loss after a per-occurrence layer: no total either.
        let inuring = Tower::inuring(vec![
            vec![Layer::xol("2x2", 2.0, 2.0).unwrap()],
            vec![Layer::stop_loss("SL", 5.0, 4.0).unwrap()],
        ])
        .unwrap();
        let g = inuring.on_grid(&freq, &sev, 256).unwrap();
        assert!(TowerQuote::technical(&inuring, &g, &rule, &assets, loads, true).is_err());
    }

    /// A quota share's ceded loss, 0 to 100 in steps of 10.
    fn qs_ceded() -> Grid {
        Grid::new(
            10.0,
            vec![
                0.02, 0.04, 0.08, 0.14, 0.2, 0.2, 0.14, 0.09, 0.05, 0.03, 0.01,
            ],
        )
        .unwrap()
    }

    fn qs_price(ceded: &Grid) -> Price {
        price_grid(
            ceded,
            &PremiumRule::cost_of_capital(0.1).unwrap(),
            &Distortion::tvar(0.99).unwrap(),
        )
        .unwrap()
    }

    fn qs(premium: f64) -> Layer {
        Layer::quota_share("QS", 0.5)
            .unwrap()
            .premium_rate(1.0, 2.0 * premium)
            .unwrap()
    }

    #[test]
    fn a_sliding_scale_at_one_rate_is_a_flat_commission() {
        let ceded = qs_ceded();
        let p = qs_price(&ceded);
        let loads = Loads::new(0.03, 0.05, 0.5).unwrap();
        let slide = qs(100.0)
            .sliding_scale(vec![(0.3, 0.5), (0.3, 0.7)])
            .unwrap();
        let q = Quote::technical_for_layer(&p, loads, &slide, &ceded, 0.0, true).unwrap();
        let flat =
            Quote::technical_with(&p, loads, PremiumTerms::new(0.0, 0.3).unwrap(), None).unwrap();
        assert!((q.premium - flat.premium).abs() < 1e-10 * flat.premium);
        assert!((q.other_commission - 0.3 * q.premium).abs() < 1e-10);
        // A layer with neither term is the flat quote exactly.
        let plain = qs(100.0).ceding_commission(0.3).unwrap();
        let q = Quote::technical_for_layer(&p, loads, &plain, &ceded, 0.0, true).unwrap();
        assert_eq!(q, flat);
    }

    #[test]
    fn the_solved_deposit_earns_the_price_and_pays_its_own_commissions() {
        let ceded = qs_ceded();
        let p = qs_price(&ceded);
        let loads = Loads::new(0.02, 0.05, 0.0)
            .unwrap()
            .with_income_tax(0.21)
            .unwrap();
        for layer in [
            qs(100.0)
                .sliding_scale(vec![(0.35, 0.4), (0.25, 0.6), (0.2, 0.8)])
                .unwrap(),
            qs(100.0).profit_commission(0.25, 0.1).unwrap(),
            qs(100.0)
                .ceding_commission(0.2)
                .unwrap()
                .profit_commission(0.2, 0.3)
                .unwrap(),
        ] {
            let q = Quote::technical_for_layer(&p, loads, &layer, &ceded, 0.0, true).unwrap();
            // The commission is recomputed at the solved deposit, not read at
            // the layer's own premium.
            let (cc, pc) = layer
                .expected_loss_sensitive_commissions(q.premium, &ceded)
                .unwrap();
            assert!((q.other_commission - cc - pc).abs() < 1e-12);
            assert!((q.return_on_capital() - p.return_on_capital()).abs() < 1e-9);
            let again = q.at_with_commission(q.premium, cc + pc).unwrap();
            assert!((again.return_on_capital() - p.return_on_capital()).abs() < 1e-9);
            // The commissions make the deposit dearer than the flat terms.
            let flat = Quote::technical_with(
                &p,
                loads,
                PremiumTerms::from_layer(&layer, 0.0).unwrap(),
                None,
            )
            .unwrap();
            assert!(q.premium > flat.premium);
        }
    }

    #[test]
    fn a_no_claims_bonus_is_solved_with_the_deposit() {
        let ceded = qs_ceded();
        let p = qs_price(&ceded);
        let loads = Loads::new(0.03, 0.05, 0.5).unwrap();
        // Paid in every year, the bonus is a flat commission of its share.
        let always = qs(100.0).no_claims_bonus(0.1, 1, 1e9).unwrap();
        let q = Quote::technical_for_layer(&p, loads, &always, &ceded, 0.0, true).unwrap();
        let flat =
            Quote::technical_with(&p, loads, PremiumTerms::new(0.0, 0.1).unwrap(), None).unwrap();
        assert!((q.premium - flat.premium).abs() < 1e-10 * flat.premium);
        // Paid only when the year cedes at most 20 (probability 0.14).
        let some = qs(100.0).no_claims_bonus(0.1, 1, 20.0).unwrap();
        let q = Quote::technical_for_layer(&p, loads, &some, &ceded, 0.0, true).unwrap();
        let bonus = some.expected_no_claims_bonus(q.premium, &ceded).unwrap();
        assert!((bonus - 0.1 * 0.14 * q.premium).abs() < 1e-10 * q.premium);
        assert!((q.other_commission - bonus).abs() < 1e-10 * q.premium);
        assert!((q.return_on_capital() - p.return_on_capital()).abs() < 1e-9);
        // A bonus over several years needs the history and is refused.
        let look = qs(100.0).no_claims_bonus(0.1, 3, 20.0).unwrap();
        assert!(Quote::technical_for_layer(&p, loads, &look, &ceded, 0.0, true).is_err());
    }

    #[test]
    fn a_steep_scale_gives_the_lowest_deposit_that_meets_the_price() {
        let ceded = qs_ceded();
        let p = qs_price(&ceded);
        let loads = Loads::new(0.02, 0.0, 0.0).unwrap();
        // From 90% to nothing over ten points of loss ratio.
        let layer = qs(100.0)
            .sliding_scale(vec![(0.9, 0.5), (0.0, 0.6)])
            .unwrap();
        let q = Quote::technical_for_layer(&p, loads, &layer, &ceded, 0.0, true).unwrap();
        let keep = 1.0 - 0.02;
        let gap = |d: f64| {
            let (cc, _) = layer
                .expected_loss_sensitive_commissions(d, &ceded)
                .unwrap();
            d * keep - cc - p.premium
        };
        assert!(gap(q.premium).abs() < 1e-9);
        // No lower deposit from the bracket's foot meets the risk premium.
        let foot = p.premium / keep;
        for k in 0..2000 {
            let d = foot + (q.premium - foot) * k as f64 / 2000.0;
            assert!(gap(d) < 1e-9, "a lower deposit {d} meets the price");
        }
        // The kept amount falls with the deposit somewhere on the scale.
        let falls = (0..400).any(|k| {
            let d = foot * (1.0 + k as f64 / 100.0);
            gap(d * 1.01) < gap(d)
        });
        assert!(falls);
    }

    #[test]
    fn the_solve_refuses_what_it_cannot_quote() {
        let ceded = qs_ceded();
        let p = qs_price(&ceded);
        // A profit commission that can take all the premium the loads leave.
        let greedy = qs(100.0).profit_commission(1.0, 0.0).unwrap();
        let loads = Loads::new(0.05, 0.0, 0.0).unwrap();
        assert!(Quote::technical_for_layer(&p, loads, &greedy, &ceded, 0.0, true).is_err());
        // A swing-rated layer has no deposit to solve for.
        let swing = qs(100.0)
            .swing_rated(
                prospicio_aggregate::LossSensitivePremium::new(50.0, 1.1, None, None).unwrap(),
            )
            .unwrap();
        let err = Quote::technical_for_layer(&p, loads, &swing, &ceded, 0.0, true).unwrap_err();
        assert!(err.to_string().contains("swing"));
    }
}
