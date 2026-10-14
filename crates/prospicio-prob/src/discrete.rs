//! A discrete distribution on any finite set of values, of either sign and
//! any spacing: the distribution of a row of a P&L ledger
//! (`docs/design/pnl.md`), as the Python package `aggregate`'s
//! `GridDistribution` on an irregular support.
//!
//! [`Grid`](crate::Grid) holds a non-negative loss on a regular step; a
//! ledger row (premium less loss less expenses, at each point of the loss's
//! grid) can be negative and lands anywhere. `Discrete` keeps the distinct
//! values in ascending order with their probabilities, rescaled to sum to 1.
//! Values within `1e-12` of each other (relative, or absolute near 0) are
//! one value: sums of the same cash flows in a different order differ in
//! the last bit, and must not split an atom.
//!
//! Quantiles and TVaR follow `aggregate`'s `make_var_tvar` exactly, so the
//! two agree number for number:
//!
//! - the cumulative probabilities `F` are the running sum, except when
//!   every mass equals the first to `aggregate`'s tolerance (within
//!   `2⁻⁵³ + 1e-5` relative), where `F_k = k / n` exactly;
//! - the lower quantile is the first value with `F ≥ p`, the upper the first
//!   with `F > p` (none at `p = 1`);
//! - [`tvar`](Discrete::tvar) is the upper tail, `E[X | X ≥ VaR]` with the
//!   atom at the quantile split, even when the values are a payoff whose
//!   adverse tail is the low one.

use prospicio_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};

/// The default return periods of [`Discrete::tail`]: `aggregate`'s
/// `DEFAULT_RETURN_PERIODS`.
pub const RETURN_PERIODS: [f64; 10] = [
    2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0,
];

/// A distribution on finitely many values; see the [module](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Discrete {
    /// Distinct values, ascending.
    values: Vec<f64>,
    /// Their probabilities, positive and summing to 1.
    probs: Vec<f64>,
    /// `F`, as `aggregate` builds it for quantiles.
    cumulative: Vec<f64>,
    /// `Σ_{j ≥ k} x_j p_j`.
    upper_sums: Vec<f64>,
}

/// One row of [`Discrete::tail`]: a return period read from both tails.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailRow {
    /// The non-exceedance probability `P`: `1 / T` in the lower tail,
    /// `1 − 1 / T` in the upper.
    pub p: f64,
    /// The return period `T` the row came from.
    pub period: f64,
    /// The lower quantile at `p`.
    pub var: f64,
    /// [`Discrete::tvar`] at `p` (the upper tail on every row).
    pub tvar: f64,
    /// `var` less the mean.
    pub excess: f64,
    /// `var` over the mean; `None` when the mean is within `1e-12` of 0.
    pub var_to_mean: Option<f64>,
}

/// Whether `a` and `b` are the same value to the ledger's tolerance.
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

impl Discrete {
    /// The distribution putting probability `probs[i]` on `values[i]`.
    /// Values may repeat (and come in any order); equal values (to `1e-12`)
    /// are merged at the smallest of them, and zero probabilities dropped.
    /// The probabilities are rescaled to sum to 1, unless they already do to
    /// `1e-12`.
    ///
    /// Fails if the lengths differ, a value or probability is not finite, a
    /// probability is negative, or no probability is positive.
    ///
    /// ```
    /// use prospicio_prob::{Discrete, Distribution};
    ///
    /// // A margin: 9 less the loss of two claims of 1, 2 or 4.
    /// let m = Discrete::new(
    ///     vec![7.0, 6.0, 5.0, 4.0, 3.0, 1.0],
    ///     vec![0.25, 0.3, 0.09, 0.2, 0.12, 0.04],
    /// )
    /// .unwrap();
    /// assert!((m.mean() - 5.2).abs() < 1e-12);
    /// assert_eq!(m.quantile(0.01).unwrap(), 1.0); // the 1-in-100 bad year
    /// ```
    pub fn new(values: Vec<f64>, probs: Vec<f64>) -> Result<Self> {
        if values.len() != probs.len() {
            return Err(Error::Data(format!(
                "{} values but {} probabilities",
                values.len(),
                probs.len()
            )));
        }
        if let Some(&v) = values.iter().find(|v| !v.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: v,
                reason: "must be finite",
            });
        }
        if let Some(&p) = probs.iter().find(|p| !(p.is_finite() && **p >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: p,
                reason: "must be finite and non-negative",
            });
        }
        let mut atoms: Vec<(f64, f64)> = values
            .into_iter()
            .zip(probs)
            .filter(|(_, p)| *p > 0.0)
            .collect();
        if atoms.is_empty() {
            return Err(Error::Data(
                "a discrete distribution needs a positive probability".into(),
            ));
        }
        atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(atoms.len());
        for (x, p) in atoms {
            match merged.last_mut() {
                Some(last) if same(last.0, x) => last.1 += p,
                _ => merged.push((x, p)),
            }
        }
        let total: f64 = merged.iter().map(|a| a.1).sum();
        // Probabilities already summing to 1 are kept as given, as
        // aggregate keeps them: rescaling them again moves a cumulative 0.1
        // to 0.09999999999999999, across a quantile's jump.
        let scale = if (total - 1.0).abs() <= 1e-12 {
            1.0
        } else {
            total
        };
        let (values, probs): (Vec<f64>, Vec<f64>) =
            merged.into_iter().map(|(x, p)| (x, p / scale)).unzip();
        Ok(Self::build(values, probs))
    }

    fn build(values: Vec<f64>, probs: Vec<f64>) -> Self {
        let n = values.len();
        // aggregate: np.isclose(p, p[0], atol=2**-53), rtol 1e-5.
        let equal = probs
            .iter()
            .all(|&p| (p - probs[0]).abs() <= 2f64.powi(-53) + 1e-5 * probs[0].abs());
        let cumulative: Vec<f64> = if equal {
            (1..=n).map(|k| k as f64 / n as f64).collect()
        } else {
            probs
                .iter()
                .scan(0.0, |s, p| {
                    *s += p;
                    Some(*s)
                })
                .collect()
        };
        let mut upper_sums = vec![0.0; n + 1];
        for k in (0..n).rev() {
            upper_sums[k] = upper_sums[k + 1] + values[k] * probs[k];
        }
        Self {
            values,
            probs,
            cumulative,
            upper_sums,
        }
    }

    /// The distinct values, ascending.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Their probabilities, summing to 1.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The probability of exactly `x` (to `1e-12`), 0 off the support.
    pub fn pmf(&self, x: f64) -> f64 {
        self.values
            .iter()
            .position(|&v| same(v, x))
            .map_or(0.0, |k| self.probs[k])
    }

    /// The upper quantile, the smallest value with `F > p`; `None` at
    /// `p = 1` (or above the last cumulative probability).
    pub fn quantile_upper(&self, p: f64) -> Result<Option<f64>> {
        check_probability(p)?;
        let k = self.cumulative.partition_point(|&f| f <= p);
        Ok(self.values.get(k).copied())
    }

    /// Tail value at risk at level `p` in `[0, 1)`: the mean of the upper
    /// `1 − p` of the distribution, the atom at the quantile taken in part,
    /// as `aggregate`'s `GridDistribution.tvar`. It is the upper tail even
    /// for a payoff, whose adverse tail is the low one. From the level of
    /// the second-largest value's cumulative probability up, it is the
    /// largest value.
    ///
    /// ```
    /// use prospicio_prob::Discrete;
    ///
    /// let d = Discrete::new(vec![0.0, 10.0], vec![0.5, 0.5]).unwrap();
    /// assert_eq!(d.tvar(0.0).unwrap(), 5.0);
    /// assert_eq!(d.tvar(0.5).unwrap(), 10.0);
    /// ```
    pub fn tvar(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let n = self.values.len();
        if n == 1 || p >= self.cumulative[n - 2] {
            return Ok(self.values[n - 1]);
        }
        let k = self.cumulative.partition_point(|&f| f <= p);
        Ok(((self.cumulative[k] - p) * self.values[k] + self.upper_sums[k + 1]) / (1.0 - p))
    }

    /// Skewness, `E[(X − μ)³] / σ³`; `None` when the variance is 0.
    pub fn skewness(&self) -> Option<f64> {
        let mean = self.mean();
        let sd = self.std_dev();
        (sd > 0.0).then(|| {
            self.values
                .iter()
                .zip(&self.probs)
                .map(|(x, p)| (x - mean).powi(3) * p)
                .sum::<f64>()
                / sd.powi(3)
        })
    }

    /// The return-period table of `aggregate`'s `return_period_frame`: each
    /// period `T` read from both tails, at `P = 1 / T` and `P = 1 − 1 / T`,
    /// in ascending `P` with the one row at the median kept once. A payoff
    /// (a P&L) is read off the low rows, a loss off the high ones.
    /// [`RETURN_PERIODS`] is `aggregate`'s ladder.
    ///
    /// Fails if a period is below 1 or not finite.
    pub fn tail(&self, periods: &[f64]) -> Result<Vec<TailRow>> {
        if let Some(&t) = periods.iter().find(|t| !(t.is_finite() && **t >= 1.0)) {
            return Err(Error::InvalidParameter {
                name: "periods",
                value: t,
                reason: "must be finite and at least 1",
            });
        }
        let mut rungs: Vec<(f64, f64)> = periods
            .iter()
            .flat_map(|&t| [(1.0 / t, t), (1.0 - 1.0 / t, t)])
            .collect();
        rungs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        rungs.dedup_by(|b, a| b.0 <= a.0);
        let mean = self.mean();
        rungs
            .into_iter()
            .map(|(p, period)| {
                let var = self.quantile(p)?;
                Ok(TailRow {
                    p,
                    period,
                    var,
                    tvar: self.tvar(p)?,
                    excess: var - mean,
                    var_to_mean: (mean.abs() > 1e-12).then(|| var / mean),
                })
            })
            .collect()
    }
}

/// The return period of a non-exceedance probability `p`: `1 / p` for a
/// payoff, whose adverse tail is the low one, `1 / (1 − p)` for a loss, as
/// `aggregate`'s `return_period_map`.
pub fn return_period(p: f64, payoff: bool) -> f64 {
    if payoff { 1.0 / p } else { 1.0 / (1.0 - p) }
}

impl Distribution for Discrete {
    fn mean(&self) -> f64 {
        self.values
            .iter()
            .zip(&self.probs)
            .map(|(x, p)| x * p)
            .sum()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();
        self.values
            .iter()
            .zip(&self.probs)
            .map(|(x, p)| (x - mean) * (x - mean) * p)
            .sum()
    }

    /// `P(X ≤ x)`, the running sum of the probabilities.
    fn cdf(&self, x: f64) -> f64 {
        let k = self.values.partition_point(|&v| v <= x);
        self.probs[..k].iter().sum::<f64>().min(1.0)
    }

    /// Summed from the top, so small tail probabilities keep their
    /// precision.
    fn survival(&self, x: f64) -> f64 {
        let k = self.values.partition_point(|&v| v <= x);
        self.probs[k..].iter().sum::<f64>().min(1.0)
    }

    /// The lower quantile, the smallest value with `F ≥ p` (`F` as in the
    /// [module](self)); the smallest value at `p = 0`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let k = self.cumulative.partition_point(|&f| f < p);
        Ok(self.values[k.min(self.values.len() - 1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_merged_sorted_and_rescaled() {
        let d = Discrete::new(
            vec![3.1, -2.0, 3.0999999999999996, 0.0],
            vec![1.0, 2.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(d.values(), &[-2.0, 3.0999999999999996]);
        assert_eq!(d.probs(), &[0.5, 0.5]);
        assert_eq!(d.pmf(3.1), 0.5);
        assert_eq!(d.pmf(0.0), 0.0);
    }

    #[test]
    fn quantiles_and_tvar_of_a_signed_distribution() {
        let d = Discrete::new(vec![-10.0, 0.0, 5.0], vec![0.2, 0.3, 0.5]).unwrap();
        assert_eq!(d.quantile(0.0).unwrap(), -10.0);
        assert_eq!(d.quantile(0.2).unwrap(), -10.0);
        assert_eq!(d.quantile_upper(0.2).unwrap(), Some(0.0));
        assert_eq!(d.quantile_upper(1.0).unwrap(), None);
        assert_eq!(d.quantile(1.0).unwrap(), 5.0);
        assert!((d.tvar(0.0).unwrap() - d.mean()).abs() < 1e-12);
        // Above 0.2 the tail is 0.3 at 0 and 0.5 at 5, over 0.8.
        assert!((d.tvar(0.2).unwrap() - 2.5 / 0.8).abs() < 1e-12);
        assert_eq!(d.tvar(0.6).unwrap(), 5.0);
        assert!((d.cdf(-0.5) - 0.2).abs() < 1e-15);
        assert!((d.survival(0.0) - 0.5).abs() < 1e-15);
        assert!(d.skewness().unwrap() < 0.0);
        assert!(d.tvar(1.5).is_err());
    }

    #[test]
    fn a_lone_atom_and_bad_input() {
        let d = Discrete::new(vec![-4.5], vec![2.0]).unwrap();
        assert_eq!(d.tvar(0.3).unwrap(), -4.5);
        assert_eq!(d.skewness(), None);
        assert!(Discrete::new(vec![1.0], vec![0.0]).is_err());
        assert!(Discrete::new(vec![1.0, 2.0], vec![0.5]).is_err());
        assert!(Discrete::new(vec![f64::NAN], vec![1.0]).is_err());
        assert!(Discrete::new(vec![1.0], vec![-1.0]).is_err());
    }

    #[test]
    fn the_tail_reads_both_ends() {
        let d = Discrete::new(vec![-1.0, 1.0, 2.0], vec![0.1, 0.6, 0.3]).unwrap();
        let rows = d.tail(&[2.0, 10.0]).unwrap();
        let ps: Vec<f64> = rows.iter().map(|r| r.p).collect();
        assert_eq!(ps, vec![0.1, 0.5, 0.9]);
        assert_eq!(rows[0].var, -1.0);
        assert_eq!(rows[0].period, 10.0);
        assert!((rows[0].excess - (-1.0 - 1.1)).abs() < 1e-12);
        assert_eq!(return_period(0.01, true), 100.0);
        assert!((return_period(0.99, false) - 100.0).abs() < 1e-9);
        assert!(d.tail(&[0.5]).is_err());
    }
}
