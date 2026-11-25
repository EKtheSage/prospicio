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
    expected: f64,
    stop_loss: StopLoss,
}

impl fmt::Debug for InsuranceCharge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsuranceCharge")
            .field("expected", &self.expected)
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
        })
    }

    /// The expected aggregate loss `E`, which the entry ratios multiply.
    pub fn expected(&self) -> f64 {
        self.expected
    }

    /// The insurance charge `φ(r) = E[max(A / E − r, 0)]`: 1 at `r = 0`,
    /// falling to 0, and `1 − r` for a negative `r`.
    pub fn charge(&self, entry_ratio: f64) -> f64 {
        if entry_ratio.is_nan() {
            return f64::NAN;
        }
        if entry_ratio <= 0.0 {
            return 1.0 - entry_ratio;
        }
        ((self.stop_loss)(entry_ratio * self.expected) / self.expected).max(0.0)
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
}
