//! The ogive: the piecewise linear distribution function through tabulated
//! points `(x_k, F(x_k))`, the usual reading of grouped loss data (losses
//! uniform within each band). *Pricing Insurance Risk*, Exercise 50, reads
//! VaR and TVaR off one.
//!
//! On a band where `F` is linear the survival function is linear too, so
//! the limited moments are exact: the trapezoid rule for `∫ S` and
//! Simpson's rule for `∫ (x − a) S`, band by band.

use prospicio_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// A non-negative loss whose distribution function is linear between the
/// points `(x_k, F_k)`; see the [module](self).
///
/// # Example
///
/// ```
/// use prospicio_prob::{Distribution, Ogive, Severity};
///
/// // 40 claims below 1,000, 50 from 1,000 to 5,000 and 10 to 10,000.
/// let o = Ogive::from_counts(&[0.0, 1000.0, 5000.0, 10_000.0], &[40.0, 50.0, 10.0]).unwrap();
/// assert!((o.cdf(3000.0) - 0.65).abs() < 1e-15);
/// assert!((o.quantile(0.95).unwrap() - 7500.0).abs() < 1e-9);
/// // The worst 5% are uniform on 7,500 to 10,000.
/// assert!((o.tvar(0.95).unwrap() - 8750.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Ogive {
    /// Ascending band boundaries, the first at least 0.
    x: Vec<f64>,
    /// `F` at each boundary, non-decreasing from 0 to 1.
    f: Vec<f64>,
}

impl Ogive {
    /// The ogive through `(x_k, F_k)`: at least two points, `x` finite,
    /// non-negative and strictly increasing, `F` non-decreasing from `0` at
    /// the first point to `1` at the last (each to `1e-12`).
    pub fn new(x: Vec<f64>, f: Vec<f64>) -> Result<Self> {
        if x.len() != f.len() {
            return Err(Error::Data(format!(
                "{} boundaries but {} values of F",
                x.len(),
                f.len()
            )));
        }
        if x.len() < 2 {
            return Err(Error::Data("an ogive needs at least two points".into()));
        }
        if x.iter().any(|v| !v.is_finite()) || x[0] < 0.0 {
            return Err(Error::Data(
                "the boundaries must be finite and non-negative".into(),
            ));
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Data(
                "the boundaries must be strictly increasing".into(),
            ));
        }
        if f.iter().any(|v| !v.is_finite()) || f.windows(2).any(|w| w[1] < w[0]) {
            return Err(Error::Data("F must be finite and non-decreasing".into()));
        }
        let last = f.len() - 1;
        if f[0].abs() > 1e-12 || (f[last] - 1.0).abs() > 1e-12 {
            return Err(Error::Data(format!(
                "F must run from 0 to 1, not {} to {}",
                f[0], f[last]
            )));
        }
        let mut f = f;
        f[0] = 0.0;
        f[last] = 1.0;
        Ok(Self { x, f })
    }

    /// The ogive of grouped data: `counts[k]` losses (or any weights)
    /// between `bounds[k]` and `bounds[k + 1]`, spread uniformly.
    pub fn from_counts(bounds: &[f64], counts: &[f64]) -> Result<Self> {
        if bounds.len() != counts.len() + 1 {
            return Err(Error::Data(format!(
                "{} counts need {} bounds, not {}",
                counts.len(),
                counts.len() + 1,
                bounds.len()
            )));
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
        let mut f = Vec::with_capacity(bounds.len());
        let mut running = 0.0;
        f.push(0.0);
        for c in counts {
            running += c;
            f.push(running / total);
        }
        Self::new(bounds.to_vec(), f)
    }

    /// The band boundaries.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// `F` at the boundaries.
    pub fn f(&self) -> &[f64] {
        &self.f
    }

    /// The density, `ΔF / Δx` on the band holding `x` (the band to its
    /// right at a boundary), 0 outside.
    pub fn pdf(&self, x: f64) -> f64 {
        let n = self.x.len();
        if x < self.x[0] || x >= self.x[n - 1] {
            return 0.0;
        }
        let k = self.x.partition_point(|b| *b <= x);
        (self.f[k] - self.f[k - 1]) / (self.x[k] - self.x[k - 1])
    }

    /// `∫_lo^hi g(t) S(t) dt` with `g` a polynomial of degree at most 1,
    /// exact by Simpson's rule on each linear piece of `S`.
    fn integrate_survival(&self, lo: f64, hi: f64, g: impl Fn(f64) -> f64) -> f64 {
        let n = self.x.len();
        let hi = hi.min(self.x[n - 1]);
        if hi <= lo {
            return 0.0;
        }
        let mut points = vec![lo];
        points.extend(self.x.iter().copied().filter(|b| *b > lo && *b < hi));
        points.push(hi);
        points
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let m = 0.5 * (a + b);
                (b - a) / 6.0
                    * (g(a) * self.survival(a)
                        + 4.0 * g(m) * self.survival(m)
                        + g(b) * self.survival(b))
            })
            .sum()
    }

    /// `Σ ΔF_k · (∫ t^j over band k) / Δx_k`, the `j`-th raw moment.
    fn raw_moment(&self, j: i32) -> f64 {
        self.x
            .windows(2)
            .zip(self.f.windows(2))
            .map(|(x, f)| {
                let (a, b) = (x[0], x[1]);
                let jj = f64::from(j + 1);
                (f[1] - f[0]) * (b.powi(j + 1) - a.powi(j + 1)) / (jj * (b - a))
            })
            .sum()
    }
}

impl Distribution for Ogive {
    fn mean(&self) -> f64 {
        self.raw_moment(1)
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        (self.raw_moment(2) - m * m).max(0.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        let n = self.x.len();
        if x <= self.x[0] {
            return 0.0;
        }
        if x >= self.x[n - 1] {
            return 1.0;
        }
        let k = self.x.partition_point(|b| *b <= x);
        let (a, b) = (self.x[k - 1], self.x[k]);
        self.f[k - 1] + (self.f[k] - self.f[k - 1]) * (x - a) / (b - a)
    }

    fn survival(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// The smallest `x` with `F(x) ≥ p`: the lower end of the support at
    /// `p = 0`, and the left end of a band of zero probability at its `F`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(self.x[0]);
        }
        // The first boundary with F ≥ p; F rises on the band below it.
        let k = self.f.partition_point(|v| *v < p);
        let (a, b) = (self.x[k - 1], self.x[k]);
        let (fa, fb) = (self.f[k - 1], self.f[k]);
        Ok(a + (b - a) * (p - fa) / (fb - fa))
    }
}

impl Severity for Ogive {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.integrate_survival(0.0, limit, |_| 1.0)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.integrate_survival(retention, f64::INFINITY, |_| 1.0)
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let shift = a - attachment;
        // 2 ∫_a^(a+l) (x − a) S(x) dx, with a negative attachment carried
        // as the sure part below 0.
        let top = attachment + limit;
        2.0 * self.integrate_survival(a, top, |t| t - attachment)
            + if shift > 0.0 {
                shift.min(limit).powi(2)
            } else {
                0.0
            }
    }

    fn third_moment(&self) -> Option<f64> {
        Some(self.raw_moment(3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bands() -> Ogive {
        Ogive::from_counts(&[0.0, 1000.0, 5000.0, 10_000.0], &[40.0, 50.0, 10.0]).unwrap()
    }

    #[test]
    fn moments_are_those_of_uniform_bands() {
        let o = bands();
        let mean = 0.4 * 500.0 + 0.5 * 3000.0 + 0.1 * 7500.0;
        assert!((o.mean() - mean).abs() < 1e-9);
        // E[X²] of a uniform on (a, b) is (a² + ab + b²) / 3.
        let m2 =
            0.4 * 1e6 / 3.0 + 0.5 * (1e6 + 5e6 + 25e6) / 3.0 + 0.1 * (25e6 + 50e6 + 100e6) / 3.0;
        assert!((o.variance() - (m2 - mean * mean)).abs() < 1e-6);
        assert!((o.layer_second_moment(f64::INFINITY, 0.0) / m2 - 1.0).abs() < 1e-12);
        let m3 = o.third_moment().unwrap();
        assert!((o.layer_third_moment(f64::INFINITY, 0.0) / m3 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn limited_moments_and_identities() {
        let o = bands();
        // E[min(X, 2000)] = ∫_0^2000 S: 1000 · (1 + 0.6) / 2 + 1000 · (0.6 + 0.475) / 2.
        assert!((o.lev(2000.0) - (800.0 + 537.5)).abs() < 1e-9);
        for d in [0.0, 500.0, 1000.0, 2500.0, 9999.0, 20_000.0] {
            assert!((o.lev(d) + o.stop_loss(d) - o.mean()).abs() < 1e-9, "d {d}");
        }
        assert_eq!(o.stop_loss(10_000.0), 0.0);
        assert_eq!(o.lev(-1.0), -1.0);
        // Second moment of a layer against its definition on fine bands.
        let (l, a) = (3000.0, 800.0);
        let n = 200_000;
        let h = l / f64::from(n);
        let numeric: f64 = (0..n)
            .map(|i| {
                let t = (f64::from(i) + 0.5) * h;
                2.0 * t * o.survival(a + t) * h
            })
            .sum();
        assert!((o.layer_second_moment(l, a) / numeric - 1.0).abs() < 1e-8);
        let v = o.layer_variance(l, a);
        assert!(v > 0.0 && v.is_finite());
    }

    #[test]
    fn quantiles_invert_the_cdf_and_skip_empty_bands() {
        let o = Ogive::new(vec![0.0, 10.0, 20.0, 30.0], vec![0.0, 0.5, 0.5, 1.0]).unwrap();
        assert_eq!(o.quantile(0.0).unwrap(), 0.0);
        assert!((o.quantile(0.25).unwrap() - 5.0).abs() < 1e-12);
        assert_eq!(o.quantile(0.5).unwrap(), 10.0);
        assert!((o.quantile(0.75).unwrap() - 25.0).abs() < 1e-12);
        assert_eq!(o.quantile(1.0).unwrap(), 30.0);
        assert_eq!(o.cdf(15.0), 0.5);
        assert_eq!(o.pdf(15.0), 0.0);
        assert!((o.pdf(25.0) - 0.05).abs() < 1e-15);
        for p in [0.1, 0.3, 0.6, 0.9] {
            assert!((o.cdf(o.quantile(p).unwrap()) - p).abs() < 1e-12);
        }
        // TVaR in the top band: uniform on (VaR, 30).
        assert!((o.tvar(0.75).unwrap() - 27.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_tables() {
        assert!(Ogive::new(vec![0.0], vec![0.0]).is_err());
        assert!(Ogive::new(vec![0.0, 1.0], vec![0.0]).is_err());
        assert!(Ogive::new(vec![-1.0, 1.0], vec![0.0, 1.0]).is_err());
        assert!(Ogive::new(vec![0.0, 0.0], vec![0.0, 1.0]).is_err());
        assert!(Ogive::new(vec![0.0, 1.0, 2.0], vec![0.0, 0.6, 0.5]).is_err());
        assert!(Ogive::new(vec![0.0, 1.0], vec![0.1, 1.0]).is_err());
        assert!(Ogive::new(vec![0.0, 1.0], vec![0.0, 0.9]).is_err());
        assert!(Ogive::from_counts(&[0.0, 1.0], &[0.0]).is_err());
        assert!(Ogive::from_counts(&[0.0, 1.0, 2.0], &[1.0]).is_err());
        assert!(Ogive::from_counts(&[0.0, 1.0], &[-1.0]).is_err());
    }
}
