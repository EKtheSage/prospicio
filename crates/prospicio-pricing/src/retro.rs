//! Retrospective rating: the insurance charge and the insurance savings of
//! an aggregate loss distribution, the functions a Table M tabulates. See
//! `docs/design/primary-rating.md`.
//!
//! For aggregate losses `A` with mean `E` and an entry ratio `r` (losses
//! as a multiple of their expectation):
//!
//! - the **charge** `φ(r) = E[max(A / E − r, 0)]` is the expected loss
//!   above `r E`, as a share of `E`: what a maximum premium gives up;
//! - the **savings** `ψ(r) = E[max(r − A / E, 0)]` is the expected
//!   shortfall below `r E`: what a minimum premium collects.
//!
//! They are tied by `φ(r) − ψ(r) = 1 − r`, with `φ(0) = 1` and `ψ(0) = 0`.
//! Both come from the distribution's stop-loss, `φ(r) = stop_loss(r E) / E`,
//! so any [`Severity`] gives them: the aggregate on a
//! [`Grid`](prospicio_prob::Grid) from `prospicio_aggregate::fft` or
//! `panjer`, a fitted law, or a sample of simulated or observed totals
//! ([`InsuranceCharge::from_sample`], Brosius's construction).
//!
//! # A per-occurrence limit
//!
//! When each occurrence is limited (a loss limit in the plan), the
//! aggregate that enters the retrospective premium is the limited one,
//! `A*`, with mean `E*` below the unlimited `E`. Two conventions tabulate
//! its charge, and they are not interchangeable:
//!
//! - **The limited Table M** is the table of the limited aggregate on its
//!   own: entry ratios are multiples of `E*`, and the cost of the limit is
//!   charged apart, as an excess loss premium. Build it from the limited
//!   aggregate with [`InsuranceCharge::from_severity`] or
//!   [`from_sample`](InsuranceCharge::from_sample).
//! - **Table L** (Skurnick, 1974) keeps entry ratios as multiples of the
//!   unlimited `E` and puts the cost of the limit into the charge:
//!   `φ*(r) = E[max(A* / E − r, 0)] + k`, with `k = (E − E*) / E` the share
//!   of the losses the limit removes, and `ψ*(r) = E[max(r − A* / E, 0)]`.
//!   Then `φ*(r) − ψ*(r) = 1 − r` still, and `φ*(0) = 1`.
//!   [`InsuranceCharge::table_l`] turns the limited table into it.
//!
//! # Balancing a plan
//!
//! A retrospective premium is `R = T (b + c L)`, held between a minimum
//! `H` and a maximum `G`: `T` the tax multiplier, `c` the loss conversion
//! factor, `b` the basic premium and `L` the year's losses. The plan is
//! balanced when its expected premium is the guaranteed-cost premium
//! `T (e + E)`, with `e` the expenses and profit and `E` the expected
//! losses. [`RetroPlan::balance`] finds the basic premium that does it,
//! from the table's charge at the entry ratios where the minimum and the
//! maximum bind.
//!
//! ```
//! use prospicio_pricing::retro::InsuranceCharge;
//! use prospicio_prob::Gamma;
//!
//! // Exponential aggregate losses: φ(r) = exp(−r).
//! let table = InsuranceCharge::from_severity(Gamma::new(1.0, 250.0).unwrap()).unwrap();
//! assert!((table.charge(1.5) - (-1.5f64).exp()).abs() < 1e-14);
//! assert!((table.charge(1.5) - table.savings(1.5) - (1.0 - 1.5)).abs() < 1e-14);
//! ```

use std::fmt;
use std::sync::Arc;

use prospicio_core::{Error, Result};
use prospicio_prob::Severity;

/// The stop-loss `E[max(A − d, 0)]` of the aggregate losses, in money.
type StopLoss = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

/// The insurance charge `φ(r)` and savings `ψ(r)` of an aggregate loss
/// distribution, at any entry ratio: a Table M without the rows.
#[derive(Clone)]
pub struct InsuranceCharge {
    /// What an entry ratio multiplies: the mean of the aggregate, or the
    /// unlimited mean of a Table L.
    expected: f64,
    stop_loss: StopLoss,
    /// A Table L's `k = (E − E*) / E`; 0 otherwise.
    loss_elimination: f64,
}

impl fmt::Debug for InsuranceCharge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsuranceCharge")
            .field("expected", &self.expected)
            .field("loss_elimination", &self.loss_elimination)
            .finish_non_exhaustive()
    }
}

/// One row of a Table M: [`InsuranceCharge::table`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeRow {
    /// The entry ratio `r`.
    pub entry_ratio: f64,
    /// The insurance charge `φ(r)`.
    pub charge: f64,
    /// The insurance savings `ψ(r)`.
    pub savings: f64,
}

fn check_expected(expected: f64) -> Result<f64> {
    if expected.is_finite() && expected > 0.0 {
        Ok(expected)
    } else {
        Err(Error::InvalidParameter {
            name: "expected",
            value: expected,
            reason: "the expected aggregate loss must be positive and finite",
        })
    }
}

impl InsuranceCharge {
    /// The charge and savings of the aggregate loss distribution
    /// `aggregate`, from its stop-loss.
    ///
    /// On a [`Grid`](prospicio_prob::Grid) the stop-loss is exact for the
    /// grid, so the charge carries the grid's discretization error and
    /// adds none.
    ///
    /// Fails when the mean is not positive and finite.
    ///
    /// ```
    /// use prospicio_pricing::retro::InsuranceCharge;
    /// use prospicio_prob::Grid;
    ///
    /// // Losses of 0, 50, 100 or 250, equally likely: expected 100.
    /// let agg = Grid::new(50.0, vec![0.25, 0.25, 0.25, 0.0, 0.0, 0.25]).unwrap();
    /// let table = InsuranceCharge::from_severity(agg).unwrap();
    /// assert!((table.charge(1.0) - 0.375).abs() < 1e-15); // (250 − 100) / 4 / 100
    /// assert!((table.savings(1.0) - 0.375).abs() < 1e-15); // (100 + 50) / 4 / 100
    /// ```
    pub fn from_severity<S>(aggregate: S) -> Result<Self>
    where
        S: Severity + Send + Sync + 'static,
    {
        let expected = check_expected(aggregate.mean())?;
        Ok(Self {
            expected,
            stop_loss: Arc::new(move |d| aggregate.stop_loss(d)),
            loss_elimination: 0.0,
        })
    }

    /// The charge and savings of a sample of aggregate losses, each
    /// equally likely: simulated years, or the experience of a group of
    /// similar risks (Brosius, *Table M Construction*). The losses need
    /// not be sorted.
    ///
    /// Fails on an empty sample, a negative or non-finite loss, or a
    /// sample whose mean is zero.
    ///
    /// ```
    /// use prospicio_pricing::retro::InsuranceCharge;
    ///
    /// let table = InsuranceCharge::from_sample(&[250.0, 0.0, 100.0, 50.0]).unwrap();
    /// assert_eq!(table.expected(), 100.0);
    /// assert!((table.charge(0.5) - 0.625).abs() < 1e-15);
    /// assert!((table.savings(0.5) - 0.125).abs() < 1e-15);
    /// ```
    pub fn from_sample(losses: &[f64]) -> Result<Self> {
        if losses.is_empty() {
            return Err(Error::Data(
                "the sample of aggregate losses is empty".into(),
            ));
        }
        if let Some(&bad) = losses.iter().find(|x| !(x.is_finite() && **x >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "losses",
                value: bad,
                reason: "an aggregate loss must be non-negative and finite",
            });
        }
        let mut sorted = losses.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len() as f64;
        // above[k] = the sum of sorted[k..], summed from the small end of
        // the tail so the large losses do not swamp the small ones.
        let mut above = vec![0.0; sorted.len() + 1];
        for k in (0..sorted.len()).rev() {
            above[k] = above[k + 1] + sorted[k];
        }
        let expected = check_expected(above[0] / n)?;
        let stop_loss = move |d: f64| {
            // The losses above d: sorted[k..].
            let k = sorted.partition_point(|x| *x <= d);
            (above[k] - d * (sorted.len() - k) as f64) / n
        };
        Ok(Self {
            expected,
            stop_loss: Arc::new(stop_loss),
            loss_elimination: 0.0,
        })
    }

    /// This table of a limited aggregate as a Table L: entry ratios
    /// become multiples of `unlimited_expected`, the expected aggregate
    /// loss without the per-occurrence limit, and the charge takes on the
    /// share of the losses the limit removes,
    /// `k = (E − E*) / E` ([`loss_elimination`](Self::loss_elimination)).
    ///
    /// Call it on the table built from the limited aggregate. Fails when
    /// `unlimited_expected` is not finite or is below the limited mean (a
    /// limit cannot add losses), or when the table is already a Table L.
    ///
    /// ```
    /// use prospicio_pricing::retro::InsuranceCharge;
    ///
    /// // Four years' totals with each occurrence limited to 100, from
    /// // unlimited totals of 110, 200, 0 and 220 (expected 132.5).
    /// let limited = InsuranceCharge::from_sample(&[110.0, 100.0, 0.0, 200.0]).unwrap();
    /// assert_eq!(limited.expected(), 102.5);
    /// let table = limited.table_l(132.5).unwrap();
    /// assert!((table.loss_elimination() - 30.0 / 132.5).abs() < 1e-15);
    /// // At r = 1: the limited losses above 132.5 are 67.5 in one year of
    /// // four, and the limit removed 30 a year.
    /// assert!((table.charge(1.0) - (67.5 / 4.0 + 30.0) / 132.5).abs() < 1e-15);
    /// assert!((table.charge(1.0) - table.savings(1.0)).abs() < 1e-15);
    /// assert_eq!(table.charge(0.0), 1.0);
    /// ```
    pub fn table_l(mut self, unlimited_expected: f64) -> Result<Self> {
        if self.loss_elimination != 0.0 {
            return Err(Error::Data(
                "this table is already a Table L; build it from the limited aggregate".into(),
            ));
        }
        let limited = self.expected;
        // A limit that removes nothing may leave the two means a rounding
        // apart, on either side.
        if !(unlimited_expected.is_finite() && unlimited_expected >= limited * (1.0 - 1e-12)) {
            return Err(Error::InvalidParameter {
                name: "unlimited_expected",
                value: unlimited_expected,
                reason: "must be finite and at least the limited aggregate's mean",
            });
        }
        self.expected = unlimited_expected;
        self.loss_elimination = ((unlimited_expected - limited) / unlimited_expected).max(0.0);
        Ok(self)
    }

    /// What the entry ratios multiply: the expected aggregate loss `E`,
    /// and for a Table L the expected loss without the limit.
    pub fn expected(&self) -> f64 {
        self.expected
    }

    /// A Table L's `k = (E − E*) / E`, the share of the expected losses
    /// the per-occurrence limit removes; 0 for any other table.
    pub fn loss_elimination(&self) -> f64 {
        self.loss_elimination
    }

    /// The insurance charge `φ(r) = E[max(A / E − r, 0)]`: 1 at `r = 0`,
    /// falling to 0, and `1 − r` for a negative `r`. A Table L's charge
    /// adds [`loss_elimination`](Self::loss_elimination) and so falls to
    /// that, not to 0.
    pub fn charge(&self, entry_ratio: f64) -> f64 {
        if entry_ratio.is_nan() {
            return f64::NAN;
        }
        if entry_ratio <= 0.0 {
            return 1.0 - entry_ratio;
        }
        if entry_ratio == f64::INFINITY {
            return self.loss_elimination;
        }
        ((self.stop_loss)(entry_ratio * self.expected) / self.expected).max(0.0)
            + self.loss_elimination
    }

    /// The insurance savings `ψ(r) = E[max(r − A / E, 0)] = φ(r) + r − 1`:
    /// 0 at `r = 0` and below, rising toward `r − 1`.
    pub fn savings(&self, entry_ratio: f64) -> f64 {
        if entry_ratio.is_nan() {
            return f64::NAN;
        }
        if entry_ratio <= 0.0 {
            return 0.0;
        }
        (self.charge(entry_ratio) + entry_ratio - 1.0).max(0.0)
    }

    /// The table at the given entry ratios, one row each, in their order.
    ///
    /// ```
    /// use prospicio_pricing::retro::InsuranceCharge;
    /// use prospicio_prob::Gamma;
    ///
    /// let table = InsuranceCharge::from_severity(Gamma::new(1.0, 1.0).unwrap()).unwrap();
    /// let rows = table.table(&[0.0, 0.5, 1.0, 2.0]);
    /// assert_eq!(rows[0].charge, 1.0);
    /// assert!((rows[2].charge - (-1.0f64).exp()).abs() < 1e-15);
    /// assert!(rows.windows(2).all(|w| w[0].charge > w[1].charge && w[0].savings < w[1].savings));
    /// ```
    pub fn table(&self, entry_ratios: &[f64]) -> Vec<ChargeRow> {
        entry_ratios
            .iter()
            .map(|&entry_ratio| ChargeRow {
                entry_ratio,
                charge: self.charge(entry_ratio),
                savings: self.savings(entry_ratio),
            })
            .collect()
    }
}

/// A retrospective rating plan: the premium `R = T (b + c L)` for losses
/// `L`, no less than a minimum and no more than a maximum.
///
/// Amounts are in money. A plan with a loss limit takes `L` as the
/// losses with each occurrence limited, and is balanced against a
/// Table L ([`InsuranceCharge::table_l`]); its basic premium then carries
/// the cost of the limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetroPlan {
    expected_loss: f64,
    expense: f64,
    loss_conversion: f64,
    tax_multiplier: f64,
    minimum: f64,
    maximum: f64,
}

/// A balanced plan: [`RetroPlan::balance`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BalancedPlan {
    /// The basic premium `b`, before the tax multiplier.
    pub basic: f64,
    /// The entry ratio `r_H` at and below which the minimum premium
    /// applies. Negative when the minimum is below `T b`, so that it never
    /// binds.
    pub entry_ratio_minimum: f64,
    /// The entry ratio `r_G` at and above which the maximum premium
    /// applies; infinite for a plan without a maximum.
    pub entry_ratio_maximum: f64,
    /// The net insurance charge `φ(r_G) − ψ(r_H)` as a share of the
    /// expected losses: what the maximum gives up less what the minimum
    /// collects.
    pub net_insurance_charge: f64,
    /// The expected retrospective premium, the guaranteed-cost premium
    /// `T (e + E)`.
    pub expected_premium: f64,
}

impl RetroPlan {
    /// A plan with expected losses `expected_loss` (`E`, without any loss
    /// limit), expenses and profit `expense` (`e`, excluding the premium
    /// tax), a loss conversion factor (`c`), a tax multiplier (`T`), and a
    /// minimum and a maximum premium (`H` and `G`; `f64::INFINITY` for no
    /// maximum, 0 for no minimum).
    ///
    /// Fails unless `E`, `c` and `T` are positive and finite, `e` is
    /// non-negative and finite, and `0 ≤ H < G`.
    pub fn new(
        expected_loss: f64,
        expense: f64,
        loss_conversion: f64,
        tax_multiplier: f64,
        minimum: f64,
        maximum: f64,
    ) -> Result<Self> {
        let positive = |name: &'static str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name,
                    value,
                    reason: "must be positive and finite",
                })
            }
        };
        positive("expected_loss", expected_loss)?;
        positive("loss_conversion", loss_conversion)?;
        positive("tax_multiplier", tax_multiplier)?;
        if !(expense.is_finite() && expense >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "expense",
                value: expense,
                reason: "must be non-negative and finite",
            });
        }
        if !(minimum.is_finite() && minimum >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "minimum",
                value: minimum,
                reason: "must be non-negative and finite",
            });
        }
        if maximum.is_nan() || maximum <= minimum {
            return Err(Error::InvalidParameter {
                name: "maximum",
                value: maximum,
                reason: "must be above the minimum premium",
            });
        }
        Ok(Self {
            expected_loss,
            expense,
            loss_conversion,
            tax_multiplier,
            minimum,
            maximum,
        })
    }

    /// The guaranteed-cost premium `T (e + E)`, which a balanced plan
    /// collects on average.
    pub fn guaranteed_cost(&self) -> f64 {
        self.tax_multiplier * (self.expense + self.expected_loss)
    }

    /// The retrospective premium for a year's `losses` under the basic
    /// premium `basic`: `T (b + c L)`, between the minimum and the
    /// maximum.
    ///
    /// It is the premium of
    /// [`LossSensitivePremium`](prospicio_aggregate::LossSensitivePremium)
    /// with the tax multiplier folded into its basic premium and its loss
    /// conversion factor, `new(T b, T c, Some(H), Some(G))`, which is the
    /// form to apply to simulated years or a P&L.
    pub fn premium(&self, basic: f64, losses: f64) -> f64 {
        (self.tax_multiplier * (basic + self.loss_conversion * losses))
            .clamp(self.minimum, self.maximum)
    }

    /// The basic premium at which the plan's expected premium is the
    /// guaranteed-cost premium, under the aggregate distribution behind
    /// `charge`.
    ///
    /// With `r_H` and `r_G` the entry ratios where the minimum and the
    /// maximum bind, the plan balances when
    ///
    /// ```text
    /// r_G − r_H         = (G − H) / (c E T)
    /// φ(r_H) − φ(r_G)   = (e + E − H / T) / (c E)
    /// ```
    ///
    /// and then `b = H / T − c E r_H`, which is
    /// `e − (c − 1) E + c E (φ(r_G) − ψ(r_H))`. The left side of the second
    /// equation falls as `r_H` rises with the first held, so one bisection
    /// solves the pair.
    ///
    /// `charge` must be on the plan's expected losses: a Table M of the
    /// aggregate, or for a plan with a loss limit a Table L.
    ///
    /// Fails when the table's expected loss is not the plan's; when the
    /// minimum premium is at least the guaranteed-cost premium, or the
    /// maximum at most it (no basic premium balances such a plan); or when
    /// the charge is not finite.
    ///
    /// ```
    /// use prospicio_pricing::retro::{InsuranceCharge, RetroPlan};
    ///
    /// // Ten equally likely years; expected losses 100, expenses 30.
    /// let losses = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 120.0, 150.0, 200.0, 280.0];
    /// let table = InsuranceCharge::from_sample(&losses).unwrap();
    /// let plan = RetroPlan::new(100.0, 30.0, 1.1, 1.03, 80.0, 220.0).unwrap();
    /// let balanced = plan.balance(&table).unwrap();
    /// // The plan collects the guaranteed-cost premium on average.
    /// let mean: f64 =
    ///     losses.iter().map(|&l| plan.premium(balanced.basic, l)).sum::<f64>() / 10.0;
    /// assert!((mean - plan.guaranteed_cost()).abs() < 1e-10);
    /// assert!((balanced.expected_premium - 1.03 * 130.0).abs() < 1e-12);
    /// ```
    pub fn balance(&self, charge: &InsuranceCharge) -> Result<BalancedPlan> {
        let (e, big_e, c, t) = (
            self.expense,
            self.expected_loss,
            self.loss_conversion,
            self.tax_multiplier,
        );
        if (charge.expected() / big_e - 1.0).abs() > 1e-9 {
            return Err(Error::Data(format!(
                "the table's expected loss {} is not the plan's {big_e}; for a plan with a                  loss limit, pass a Table L on the unlimited expected loss",
                charge.expected()
            )));
        }
        let spread = (self.maximum - self.minimum) / (c * big_e * t);
        let target = (e + big_e - self.minimum / t) / (c * big_e);
        if target <= 0.0 {
            return Err(Error::Data(format!(
                "the minimum premium {} is at least the guaranteed-cost premium {}:                  no basic premium balances the plan",
                self.minimum,
                self.guaranteed_cost()
            )));
        }
        // φ is 1-Lipschitz, so φ(r) − φ(r + spread) never exceeds spread.
        if target >= spread {
            return Err(Error::Data(format!(
                "the maximum premium {} is at most the guaranteed-cost premium {}:                  no basic premium balances the plan",
                self.maximum,
                self.guaranteed_cost()
            )));
        }
        let gap = |r: f64| charge.charge(r) - charge.charge(r + spread);
        // At r = −spread the gap is spread, above the target; it falls to 0.
        let lo = if spread.is_finite() { -spread } else { -target };
        let mut hi = 1.0;
        while gap(hi) > target {
            hi *= 2.0;
            if hi >= 1e12 {
                return Err(Error::Data(
                    "the insurance charge does not fall far enough to balance the plan".into(),
                ));
            }
        }
        if !gap(lo).is_finite() {
            return Err(Error::Data("the insurance charge is not finite".into()));
        }
        let r_h = prospicio_math::roots::bisect(lo, hi, |r| gap(r) > target);
        let r_g = r_h + spread;
        Ok(BalancedPlan {
            basic: self.minimum / t - c * big_e * r_h,
            entry_ratio_minimum: r_h,
            entry_ratio_maximum: r_g,
            net_insurance_charge: charge.charge(r_g) - charge.savings(r_h),
            expected_premium: self.guaranteed_cost(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Gamma, Grid, Lognormal};

    #[test]
    fn a_sample_and_the_grid_of_its_values_agree() {
        let sample = InsuranceCharge::from_sample(&[0.0, 50.0, 100.0, 250.0]).unwrap();
        let grid = Grid::new(50.0, vec![0.25, 0.25, 0.25, 0.0, 0.0, 0.25]).unwrap();
        let grid = InsuranceCharge::from_severity(grid).unwrap();
        assert_eq!(sample.expected(), 100.0);
        for r in [0.0, 0.1, 0.5, 0.75, 1.0, 1.7, 2.5, 3.0] {
            assert!((sample.charge(r) - grid.charge(r)).abs() < 1e-15, "r = {r}");
            assert!(
                (sample.savings(r) - grid.savings(r)).abs() < 1e-15,
                "r = {r}"
            );
        }
        // Above the largest loss nothing is charged and the savings are r − 1.
        assert_eq!(sample.charge(2.5), 0.0);
        assert!((sample.savings(3.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn the_charge_less_the_savings_is_one_less_the_entry_ratio() {
        let laws: Vec<InsuranceCharge> = vec![
            InsuranceCharge::from_severity(Gamma::new(0.4, 10.0).unwrap()).unwrap(),
            InsuranceCharge::from_severity(Lognormal::new(3.0, 1.5).unwrap()).unwrap(),
            InsuranceCharge::from_sample(&[3.0, 0.0, 0.0, 41.0, 7.5, 7.5]).unwrap(),
        ];
        for t in &laws {
            assert_eq!((t.charge(0.0), t.savings(0.0)), (1.0, 0.0));
            let mut last = (1.0, 0.0);
            for k in 1..=60 {
                let r = k as f64 * 0.1;
                let (phi, psi) = (t.charge(r), t.savings(r));
                assert!((phi - psi - (1.0 - r)).abs() < 1e-12, "r = {r}");
                assert!((0.0..=last.0).contains(&phi) && psi >= last.1, "r = {r}");
                last = (phi, psi);
            }
        }
    }

    #[test]
    fn a_negative_entry_ratio_charges_everything() {
        let t = InsuranceCharge::from_severity(Gamma::new(2.0, 5.0).unwrap()).unwrap();
        assert_eq!((t.charge(-0.5), t.savings(-0.5)), (1.5, 0.0));
        assert!(t.charge(f64::NAN).is_nan() && t.savings(f64::NAN).is_nan());
    }

    #[test]
    fn bad_samples_are_refused() {
        assert!(InsuranceCharge::from_sample(&[]).is_err());
        assert!(InsuranceCharge::from_sample(&[1.0, -2.0]).is_err());
        assert!(InsuranceCharge::from_sample(&[1.0, f64::NAN]).is_err());
        assert!(InsuranceCharge::from_sample(&[0.0, 0.0]).is_err());
        assert!(
            format!("{:?}", InsuranceCharge::from_sample(&[2.0]).unwrap())
                .contains("expected: 2.0")
        );
    }

    #[test]
    fn table_l_charges_the_limit_and_keeps_the_identity() {
        // Occurrences by year, each limited to 100.
        let years: [&[f64]; 4] = [&[30.0, 80.0], &[200.0], &[], &[50.0, 50.0, 120.0]];
        let total = |limit: f64| -> Vec<f64> {
            years
                .iter()
                .map(|y| y.iter().map(|x| x.min(limit)).sum())
                .collect()
        };
        let (unlimited, limited) = (total(f64::INFINITY), total(100.0));
        let e = unlimited.iter().sum::<f64>() / 4.0;
        let e_limited = limited.iter().sum::<f64>() / 4.0;
        let k = (e - e_limited) / e;
        let table = InsuranceCharge::from_sample(&limited)
            .unwrap()
            .table_l(e)
            .unwrap();
        let table_m = InsuranceCharge::from_sample(&unlimited).unwrap();
        assert_eq!(table.expected(), e);
        assert!((table.loss_elimination() - k).abs() < 1e-15);
        assert_eq!((table.charge(0.0), table.savings(0.0)), (1.0, 0.0));
        for step in 1..=40 {
            let r = step as f64 * 0.05;
            let above: f64 = limited.iter().map(|a| (a - r * e).max(0.0)).sum::<f64>() / 4.0;
            let below: f64 = limited.iter().map(|a| (r * e - a).max(0.0)).sum::<f64>() / 4.0;
            assert!((table.charge(r) - (above / e + k)).abs() < 1e-14, "r = {r}");
            assert!((table.savings(r) - below / e).abs() < 1e-14, "r = {r}");
            assert!((table.charge(r) - table.savings(r) - (1.0 - r)).abs() < 1e-14);
            // The limit never lowers the charge at an entry ratio.
            assert!(table.charge(r) >= table_m.charge(r) - 1e-15, "r = {r}");
        }
        // Far up, only the limit's cost is left.
        assert!((table.charge(10.0) - k).abs() < 1e-15);
    }

    #[test]
    fn a_limit_that_removes_nothing_gives_table_m() {
        let losses = [3.0, 0.0, 12.0, 41.0];
        let m = InsuranceCharge::from_sample(&losses).unwrap();
        let l = m.clone().table_l(m.expected()).unwrap();
        assert_eq!(l.loss_elimination(), 0.0);
        for r in [0.0, 0.3, 1.0, 2.5] {
            assert_eq!(l.charge(r), m.charge(r));
        }
        // A rounding below the limited mean is taken as equal to it.
        let l = m.clone().table_l(m.expected() * (1.0 - 1e-14)).unwrap();
        assert_eq!(l.loss_elimination(), 0.0);
    }

    #[test]
    fn table_l_refuses_a_lower_unlimited_mean_and_a_second_call() {
        let m = InsuranceCharge::from_sample(&[10.0, 30.0]).unwrap();
        assert!(m.clone().table_l(19.0).is_err());
        assert!(m.clone().table_l(f64::NAN).is_err());
        assert!(m.clone().table_l(f64::INFINITY).is_err());
        let l = m.table_l(25.0).unwrap();
        assert!(l.table_l(30.0).is_err());
    }

    fn mean_premium(plan: &RetroPlan, basic: f64, losses: &[f64]) -> f64 {
        losses.iter().map(|&l| plan.premium(basic, l)).sum::<f64>() / losses.len() as f64
    }

    const YEARS: [f64; 10] = [
        0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 120.0, 150.0, 200.0, 280.0,
    ];

    #[test]
    fn a_balanced_plan_collects_the_guaranteed_cost_premium() {
        let table = InsuranceCharge::from_sample(&YEARS).unwrap();
        for (minimum, maximum) in [
            (80.0, 220.0),
            (0.0, 180.0),
            (120.0, f64::INFINITY),
            (60.0, 400.0),
        ] {
            let plan = RetroPlan::new(100.0, 30.0, 1.1, 1.03, minimum, maximum).unwrap();
            let b = plan.balance(&table).unwrap();
            assert!(
                (mean_premium(&plan, b.basic, &YEARS) - plan.guaranteed_cost()).abs() < 1e-9,
                "minimum {minimum}, maximum {maximum}"
            );
            // The textbook form of the basic premium.
            let textbook = 30.0 - 0.1 * 100.0 + 1.1 * 100.0 * b.net_insurance_charge;
            assert!(
                (b.basic - textbook).abs() < 1e-9,
                "minimum {minimum}, maximum {maximum}"
            );
            // The minimum and the maximum bind at the entry ratios given.
            let at = |r: f64| 1.03 * (b.basic + 1.1 * 100.0 * r);
            assert!((at(b.entry_ratio_minimum) - minimum).abs() < 1e-9);
            if maximum.is_finite() {
                assert!((at(b.entry_ratio_maximum) - maximum).abs() < 1e-9);
            } else {
                assert_eq!(b.entry_ratio_maximum, f64::INFINITY);
            }
        }
    }

    #[test]
    fn a_plan_that_never_binds_has_the_basic_premium_of_its_expenses() {
        // No minimum and a maximum above the largest year: b = e − (c − 1) E,
        // and the entry ratio of the minimum is negative.
        let table = InsuranceCharge::from_sample(&YEARS).unwrap();
        let plan = RetroPlan::new(100.0, 30.0, 1.1, 1.03, 0.0, 1_000.0).unwrap();
        let b = plan.balance(&table).unwrap();
        assert!((b.basic - 20.0).abs() < 1e-9);
        assert!(b.entry_ratio_minimum < 0.0 && b.net_insurance_charge.abs() < 1e-12);
    }

    #[test]
    fn a_plan_with_a_loss_limit_balances_against_table_l() {
        // The same ten years with each year's losses limited to 150 (one
        // occurrence a year, for the test): the plan's losses are the
        // limited ones and its expected losses the unlimited 100.
        let limited: Vec<f64> = YEARS.iter().map(|l| l.min(150.0)).collect();
        let table = InsuranceCharge::from_sample(&limited)
            .unwrap()
            .table_l(100.0)
            .unwrap();
        let plan = RetroPlan::new(100.0, 30.0, 1.1, 1.03, 80.0, 200.0).unwrap();
        let b = plan.balance(&table).unwrap();
        assert!((mean_premium(&plan, b.basic, &limited) - plan.guaranteed_cost()).abs() < 1e-9);
        // The limited Table M is on another expected loss, and is refused.
        let limited_m = InsuranceCharge::from_sample(&limited).unwrap();
        assert!(plan.balance(&limited_m).is_err());
    }

    #[test]
    fn a_continuous_aggregate_balances_by_its_own_integral() {
        // Exponential losses of mean 100: E[min(max(L, a), b)] in closed form.
        let table = InsuranceCharge::from_severity(Gamma::new(1.0, 100.0).unwrap()).unwrap();
        let plan = RetroPlan::new(100.0, 25.0, 1.15, 1.04, 90.0, 260.0).unwrap();
        let b = plan.balance(&table).unwrap();
        let (lo, hi) = (100.0 * b.entry_ratio_minimum, 100.0 * b.entry_ratio_maximum);
        let clipped = lo + 100.0 * ((-lo / 100.0).exp() - (-hi / 100.0).exp());
        let expected = 1.04 * (b.basic + 1.15 * clipped);
        assert!((expected - plan.guaranteed_cost()).abs() < 1e-9);
    }

    #[test]
    fn plans_that_cannot_balance_are_refused() {
        let table = InsuranceCharge::from_sample(&YEARS).unwrap();
        let plan = |minimum, maximum| RetroPlan::new(100.0, 30.0, 1.1, 1.03, minimum, maximum);
        // Guaranteed cost 133.9: a minimum at or above it, a maximum at or below it.
        assert!(plan(134.0, 300.0).unwrap().balance(&table).is_err());
        assert!(plan(50.0, 133.0).unwrap().balance(&table).is_err());
        assert!(plan(100.0, 100.0).is_err() && plan(-1.0, 100.0).is_err());
        assert!(plan(0.0, f64::NAN).is_err());
        assert!(RetroPlan::new(0.0, 30.0, 1.1, 1.03, 0.0, 200.0).is_err());
        assert!(RetroPlan::new(100.0, -1.0, 1.1, 1.03, 0.0, 200.0).is_err());
        assert!(RetroPlan::new(100.0, 30.0, 0.0, 1.03, 0.0, 200.0).is_err());
        assert!(RetroPlan::new(100.0, 30.0, 1.1, f64::INFINITY, 0.0, 200.0).is_err());
        // A table on another expected loss.
        let other = InsuranceCharge::from_sample(&[1.0, 3.0]).unwrap();
        assert!(plan(80.0, 220.0).unwrap().balance(&other).is_err());
    }

    #[test]
    fn the_premium_is_the_loss_sensitive_premium_with_the_tax_folded_in() {
        use prospicio_aggregate::LossSensitivePremium;
        let plan = RetroPlan::new(100.0, 30.0, 1.1, 1.03, 80.0, 220.0).unwrap();
        let basic = 25.0;
        let terms =
            LossSensitivePremium::new(1.03 * basic, 1.03 * 1.1, Some(80.0), Some(220.0)).unwrap();
        for losses in [0.0, 20.0, 48.0, 90.0, 150.0, 400.0] {
            assert!((plan.premium(basic, losses) - terms.premium(losses)).abs() < 1e-12);
        }
    }
}
