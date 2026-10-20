//! A severity read from a size-of-loss table: claim counts and loss totals
//! by size band, as Werner and Modlin's *Basic Ratemaking* chapter 11 prices
//! increased limits and deductibles from them (#338).
//!
//! A table gives each band's count and total, not how the claims spread
//! inside it, so the severity puts each band's claims at the band's average.
//! Its limited expected value is then exact at every band edge, which is
//! where the book reads it: `LEV(b) = (Σ losses below b + b · claims above
//! b) / claims`, the limited average severity. Between edges it is the
//! value of that discrete distribution, with a kink at each band's average.
//! The mean is the table's average severity; the variance and higher
//! moments, which see only the band averages, understate the spread.
//!
//! The limited moments are sums over the bands, so the severity works with
//! [`Severity::lev`] and everything built on it (increased limit factors,
//! loss elimination ratios, layers).

use prospicio_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Claim counts and loss totals by size band; see the [module](self).
///
/// ```
/// use prospicio_prob::{Distribution, Severity, SizeOfLoss};
///
/// // Werner and Modlin, Table 11.3: 5,000 claims up to 100K, 250K, 500K, 1M.
/// let t = SizeOfLoss::new(
///     &[0.0, 100e3, 250e3, 500e3, 1e6],
///     &[2_324.0, 1_923.0, 680.0, 73.0],
///     &[117_629_223.0, 307_599_929.0, 222_793_514.0, 43_047_470.0],
/// )
/// .unwrap();
/// assert!((t.lev(100e3) - 77_045.8446).abs() < 1e-3);
/// assert!((t.lev(250e3) / t.lev(100e3) - 1.5925).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SizeOfLoss {
    bounds: Vec<f64>,
    counts: Vec<f64>,
    losses: Vec<f64>,
    /// The bands with claims: their averages and probabilities.
    values: Vec<f64>,
    probs: Vec<f64>,
}

impl SizeOfLoss {
    /// `counts[k]` claims with `losses[k]` in total between `bounds[k]` and
    /// `bounds[k + 1]`. The bounds start at 0 or above and increase
    /// strictly; the last may be infinite (an open top band). Every band's
    /// average must lie within its bounds (a claim at the top bound, such
    /// as one capped at a policy limit, belongs to the band below it), and
    /// a band with no claims has no losses.
    pub fn new(bounds: &[f64], counts: &[f64], losses: &[f64]) -> Result<Self> {
        if counts.is_empty() || bounds.len() != counts.len() + 1 || losses.len() != counts.len() {
            return Err(Error::Data(format!(
                "{} counts need {} bounds and {} loss totals, not {} and {}",
                counts.len(),
                counts.len() + 1,
                counts.len(),
                bounds.len(),
                losses.len()
            )));
        }
        if bounds.iter().any(|b| b.is_nan())
            || bounds[0] < 0.0
            || bounds[..counts.len()].iter().any(|b| !b.is_finite())
            || bounds.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(Error::Data(
                "the bounds must start at 0 or above, be finite except the last, \
                 and increase strictly"
                    .into(),
            ));
        }
        if counts.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err(Error::Data(
                "the counts must be finite and non-negative".into(),
            ));
        }
        let total: f64 = counts.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data("the counts must have a positive total".into()));
        }
        let (mut values, mut probs) = (Vec::new(), Vec::new());
        for (k, (&n, &l)) in counts.iter().zip(losses).enumerate() {
            if !l.is_finite() || l < 0.0 {
                return Err(Error::Data(format!(
                    "the loss total of band {k} must be finite and non-negative, not {l}"
                )));
            }
            if n == 0.0 {
                if l != 0.0 {
                    return Err(Error::Data(format!(
                        "band {k} has no claims but losses of {l}"
                    )));
                }
                continue;
            }
            let (lo, hi) = (bounds[k], bounds[k + 1]);
            let mean = l / n;
            let slack = 1e-12 * if hi.is_finite() { hi } else { lo }.max(1.0);
            if mean < lo - slack || mean > hi + slack {
                return Err(Error::Data(format!(
                    "band {k}'s average {mean} lies outside its bounds {lo} to {hi}"
                )));
            }
            values.push(mean.clamp(lo, hi));
            probs.push(n / total);
        }
        Ok(Self {
            bounds: bounds.to_vec(),
            counts: counts.to_vec(),
            losses: losses.to_vec(),
            values,
            probs,
        })
    }

    /// The band bounds.
    pub fn bounds(&self) -> &[f64] {
        &self.bounds
    }

    /// The claim counts by band.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// The loss totals by band.
    pub fn losses(&self) -> &[f64] {
        &self.losses
    }

    /// The number of claims.
    pub fn claims(&self) -> f64 {
        self.counts.iter().sum()
    }

    fn points(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.values.iter().copied().zip(self.probs.iter().copied())
    }
}

impl Distribution for SizeOfLoss {
    fn mean(&self) -> f64 {
        self.losses.iter().sum::<f64>() / self.claims()
    }

    /// The variance of the band averages, which understates the claims'.
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.points().map(|(x, p)| (x - m) * (x - m) * p).sum()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.points()
            .filter(|&(v, _)| v <= x)
            .map(|(_, p)| p)
            .sum::<f64>()
            .min(1.0)
    }

    fn survival(&self, x: f64) -> f64 {
        self.points().filter(|&(v, _)| v > x).map(|(_, p)| p).sum()
    }

    /// The smallest band average with `F ≥ p`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let mut running = 0.0;
        for (v, q) in self.points() {
            running += q;
            if running >= p * (1.0 - 1e-15) {
                return Ok(v);
            }
        }
        Ok(*self.values.last().expect("a table has a band with claims"))
    }
}

impl Severity for SizeOfLoss {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.points().map(|(x, p)| x.min(limit) * p).sum()
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.points()
            .map(|(x, p)| (x - retention).max(0.0) * p)
            .sum()
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.points()
            .map(|(x, p)| {
                let y = (x - a).clamp(0.0, limit);
                y * y * p
            })
            .sum()
    }

    fn layer_third_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.points()
            .map(|(x, p)| (x - a).clamp(0.0, limit).powi(3) * p)
            .sum()
    }

    fn third_moment(&self) -> Option<f64> {
        Some(self.points().map(|(x, p)| x * x * x * p).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_11_3() -> SizeOfLoss {
        SizeOfLoss::new(
            &[0.0, 100e3, 250e3, 500e3, 1e6],
            &[2_324.0, 1_923.0, 680.0, 73.0],
            &[117_629_223.0, 307_599_929.0, 222_793_514.0, 43_047_470.0],
        )
        .unwrap()
    }

    #[test]
    fn the_lev_at_a_band_edge_is_the_tables_limited_average_severity() {
        let t = table_11_3();
        let want = (117_629_223.0 + 307_599_929.0 + 250e3 * (680.0 + 73.0)) / 5_000.0;
        assert!((t.lev(250e3) - want).abs() < 1e-8);
        assert!((t.lev(1e6) - t.mean()).abs() < 1e-8);
        assert!((t.lev(f64::INFINITY) - t.mean()).abs() < 1e-8);
        assert!((t.stop_loss(100e3) - (t.mean() - t.lev(100e3))).abs() < 1e-6);
        assert!((t.layer(150e3, 100e3) - (t.lev(250e3) - t.lev(100e3))).abs() < 1e-6);
        assert!((t.cdf(250e3) - (2_324.0 + 1_923.0) / 5_000.0).abs() < 1e-15);
    }

    #[test]
    fn an_open_top_band_and_empty_bands_are_allowed() {
        let t = SizeOfLoss::new(
            &[0.0, 10.0, 20.0, f64::INFINITY],
            &[2.0, 0.0, 1.0],
            &[10.0, 0.0, 50.0],
        )
        .unwrap();
        assert_eq!(t.lev(20.0), (10.0 + 20.0) / 3.0);
        assert_eq!(t.mean(), 20.0);
        assert_eq!(t.quantile(0.9).unwrap(), 50.0);
        assert_eq!(t.quantile(0.5).unwrap(), 5.0);
    }

    #[test]
    fn a_band_average_outside_its_bounds_is_an_error() {
        assert!(SizeOfLoss::new(&[0.0, 10.0, 20.0], &[1.0, 1.0], &[11.0, 15.0]).is_err());
        assert!(SizeOfLoss::new(&[0.0, 10.0], &[0.0], &[0.0]).is_err());
        assert!(SizeOfLoss::new(&[0.0, 10.0, 20.0], &[1.0, 0.0], &[5.0, 3.0]).is_err());
        assert!(SizeOfLoss::new(&[0.0, 10.0, 5.0], &[1.0, 1.0], &[5.0, 6.0]).is_err());
        // A claim at the top bound, as one capped at a policy limit, is in.
        assert!(SizeOfLoss::new(&[0.0, 10.0], &[2.0], &[20.0]).is_ok());
    }
}
