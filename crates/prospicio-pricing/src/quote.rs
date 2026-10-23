//! A cover's quote: from its risk-loaded price to the premium the cedant
//! pays, and what a quoted premium gives the reinsurer.
//!
//! The risk premium `R` is the expected loss plus the margin for the
//! capital held, from [`risk_load::price`](crate::risk_load::price) or
//! [`price_grid`](crate::risk_load::price_grid). Internal expenses `e`
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
//! technical deposit is `(R + f)` over the bracket. A swing-rated premium,
//! a sliding-scale commission and a profit commission depend on the
//! year's loss and so on the premium: they are read back at a quoted
//! premium ([`Quote::at_with_commission`]), not solved for.
//!
//! Tax comes in two forms. A premium tax `τ` is a fraction of the premium,
//! grossed up as brokerage is. An income tax `t` is charged on the margin,
//! and the return a price asks for is read as after tax: the technical
//! quote raises the risk premium until `(1 − t)` times the margin over the
//! capital is the price's return on capital, and
//! [`Quote::return_on_capital`] is after tax. See `docs/design/pareto.md`.

use prospicio_core::{Error, Result};

use prospicio_aggregate::{Commission, Layer};

use crate::risk_load::Price;

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
}
