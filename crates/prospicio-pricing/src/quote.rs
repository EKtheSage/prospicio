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
//! and assets. See `docs/design/pareto.md`.

use prospicio_core::{Error, Result};

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
}

impl Loads {
    /// Expense and brokerage ratios with `expenses + brokerage < 1`, and a
    /// fixed expense; all non-negative.
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
        if expenses + brokerage >= 1.0 {
            return Err(Error::InvalidParameter {
                name: "brokerage",
                value: brokerage,
                reason: "expenses and brokerage must leave some premium, sum below 1",
            });
        }
        Ok(Self {
            expenses,
            brokerage,
            fixed,
        })
    }

    /// The share of each unit of premium left after expenses and
    /// brokerage, `1 − e − b`.
    pub fn keep(&self) -> f64 {
        1.0 - self.expenses - self.brokerage
    }
}

/// A cover's premium with its build-up and ratios; see the
/// [module](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quote {
    /// The premium the cedant pays.
    pub premium: f64,
    /// Expected loss `E[X]`.
    pub expected_loss: f64,
    /// The risk premium the reinsurer keeps for the loss and the capital,
    /// `P (1 − e − b) − f`.
    pub risk_premium: f64,
    /// Assets backing the loss.
    pub assets: f64,
    /// The loads.
    pub loads: Loads,
    /// The cover's limit at the placed share, for the rate on line; `None`
    /// for a cover without one, such as a quota share.
    pub limit: Option<f64>,
}

impl Quote {
    /// The technical premium of a risk-loaded price: `(R + f) / (1 − e −
    /// b)`, with `R` the price's premium.
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
        check_limit(limit)?;
        Ok(Self {
            premium: (price.premium + loads.fixed) / loads.keep(),
            expected_loss: price.expected_loss,
            risk_premium: price.premium,
            assets: price.assets,
            loads,
            limit,
        })
    }

    /// The same cover at a quoted premium: the expected loss, assets and
    /// loads kept, the risk premium what the quote leaves after the loads.
    /// Fails when that risk premium reaches the assets, leaving no capital.
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
        if !(premium.is_finite() && premium >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: premium,
                reason: "must be finite and non-negative",
            });
        }
        let risk_premium = premium * self.loads.keep() - self.loads.fixed;
        if risk_premium >= self.assets {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: premium,
                reason: "leaves a risk premium at or above the assets: no capital",
            });
        }
        Ok(Self {
            premium,
            risk_premium,
            ..*self
        })
    }

    /// Internal expenses, `e P + f`.
    pub fn expenses(&self) -> f64 {
        self.loads.expenses * self.premium + self.loads.fixed
    }

    /// Brokerage, `b P`.
    pub fn brokerage(&self) -> f64 {
        self.loads.brokerage * self.premium
    }

    /// The premium net of brokerage, what the reinsurer receives.
    pub fn net_premium(&self) -> f64 {
        self.premium - self.brokerage()
    }

    /// The margin for the capital, `R − E[X]`.
    pub fn margin(&self) -> f64 {
        self.risk_premium - self.expected_loss
    }

    /// Capital, the assets less the risk premium.
    pub fn capital(&self) -> f64 {
        self.assets - self.risk_premium
    }

    /// Return on capital, margin over capital.
    pub fn return_on_capital(&self) -> f64 {
        self.margin() / self.capital()
    }

    /// Loss ratio `E[X] / P` on the premium charged.
    pub fn loss_ratio(&self) -> f64 {
        self.expected_loss / self.premium
    }

    /// Multiple `P / E[X]`, the inverse of the loss ratio.
    pub fn multiple(&self) -> f64 {
        self.premium / self.expected_loss
    }

    /// Rate on line `P / limit`, when the cover has a limit.
    pub fn rate_on_line(&self) -> Option<f64> {
        self.limit.map(|l| self.premium / l)
    }

    /// Payback `limit / P`: the loss-free years the premium takes to pay one
    /// full limit.
    pub fn payback(&self) -> Option<f64> {
        self.limit.map(|l| l / self.premium)
    }
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
}
