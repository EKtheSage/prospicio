//! The lognormal distribution.

use act_core::{Error, Result};
use act_math::special::{norm_cdf, norm_quantile};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Lognormal distribution: `ln X ~ Normal(meanlog, sdlog^2)`.
///
/// Parameterized as in SciPy (`s = sdlog`, `scale = exp(meanlog)`), R
/// (`meanlog`, `sdlog`) and actuar.
///
/// # Example
///
/// ```
/// use act_prob::{Distribution, Lognormal};
///
/// let d = Lognormal::from_mean_cv(1000.0, 0.5).unwrap();
/// assert!((d.mean() - 1000.0).abs() < 1e-9);
/// assert!((d.std_dev() - 500.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lognormal {
    meanlog: f64,
    sdlog: f64,
}

impl Lognormal {
    /// Lognormal with the given log-scale mean and standard deviation.
    pub fn new(meanlog: f64, sdlog: f64) -> Result<Self> {
        if !meanlog.is_finite() {
            return Err(Error::InvalidParameter {
                name: "meanlog",
                value: meanlog,
                reason: "must be finite",
            });
        }
        if !sdlog.is_finite() || sdlog <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sdlog",
                value: sdlog,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { meanlog, sdlog })
    }

    /// Lognormal with the given mean and coefficient of variation, the way
    /// severity assumptions are usually stated.
    pub fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        if !mean.is_finite() || mean <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "mean",
                value: mean,
                reason: "must be finite and positive",
            });
        }
        if !cv.is_finite() || cv <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "must be finite and positive",
            });
        }
        let sigma2 = cv.mul_add(cv, 1.0).ln();
        Self::new(mean.ln() - 0.5 * sigma2, sigma2.sqrt())
    }

    /// Log-scale mean.
    pub fn meanlog(&self) -> f64 {
        self.meanlog
    }

    /// Log-scale standard deviation.
    pub fn sdlog(&self) -> f64 {
        self.sdlog
    }
}

impl Distribution for Lognormal {
    fn mean(&self) -> f64 {
        (self.meanlog + 0.5 * self.sdlog * self.sdlog).exp()
    }

    fn variance(&self) -> f64 {
        let s2 = self.sdlog * self.sdlog;
        s2.exp_m1() * (2.0 * self.meanlog + s2).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        norm_cdf((x.ln() - self.meanlog) / self.sdlog)
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        norm_cdf((self.meanlog - x.ln()) / self.sdlog)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok((self.meanlog + self.sdlog * norm_quantile(p)).exp())
    }
}

impl Severity for Lognormal {
    /// `E[min(X, d)] = e^(mu + s^2/2) Phi((ln d - mu - s^2) / s) + d (1 - Phi((ln d - mu) / s))`.
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean();
        }
        let (mu, s) = (self.meanlog, self.sdlog);
        let z = (limit.ln() - mu) / s;
        self.mean() * norm_cdf(z - s) + limit * norm_cdf(-z)
    }

    /// `E[(X - d)+] = e^(mu + s^2/2) Phi((mu + s^2 - ln d) / s) - d Phi((mu - ln d) / s)`,
    /// with both terms small in the tail, so it keeps full relative
    /// precision where `mean() - lev(d)` would cancel.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        if retention == f64::INFINITY {
            return 0.0;
        }
        let (mu, s) = (self.meanlog, self.sdlog);
        let z = (retention.ln() - mu) / s;
        (self.mean() * norm_cdf(s - z) - retention * norm_cdf(-z)).max(0.0)
    }

    /// With `b = a + limit` and `Y` the layer loss,
    /// `E[Y^2] = E[(X - a)^2; a < X <= b] + limit^2 P(X > b)`, from the
    /// partial moments `E[X^k; a < X <= b] = e^(k mu + k^2 s^2 / 2)
    /// (Phi(z_b - k s) - Phi(z_a - k s))`. Differences of `Phi` are taken in
    /// the upper tail so they keep their precision for high layers.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let (mu, s) = (self.meanlog, self.sdlog);
        let a = attachment.max(0.0);
        let b = a + limit;
        let z = |x: f64| {
            if x <= 0.0 {
                f64::NEG_INFINITY
            } else {
                (x.ln() - mu) / s
            }
        };
        let (za, zb) = (z(a), z(b));
        // Phi(hi) - Phi(lo) for hi >= lo, from the side where both are small.
        let band = |lo: f64, hi: f64| {
            if lo > 0.0 {
                norm_cdf(-lo) - norm_cdf(-hi)
            } else {
                norm_cdf(hi) - norm_cdf(lo)
            }
        };
        let m0 = band(za, zb);
        let m1 = self.mean() * band(za - s, zb - s);
        let m2 = (2.0 * mu + 2.0 * s * s).exp() * band(za - 2.0 * s, zb - 2.0 * s);
        let inside = m2 - 2.0 * a * m1 + a * a * m0;
        let above = if b == f64::INFINITY {
            0.0
        } else {
            limit * limit * norm_cdf(-zb)
        };
        (inside + above).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_core::StreamRng;

    #[test]
    fn severity_identities_and_edges() {
        let d = Lognormal::new(7.0, 0.5).unwrap();
        for limit in [100.0, 1_000.0, 5_000.0] {
            let sum = d.lev(limit) + d.stop_loss(limit);
            assert!((sum - d.mean()).abs() < 1e-12 * d.mean());
        }
        assert_eq!(d.lev(0.0), 0.0);
        assert_eq!(d.lev(-5.0), -5.0);
        assert_eq!(d.lev(f64::INFINITY), d.mean());
        assert_eq!(d.stop_loss(f64::INFINITY), 0.0);
        assert_eq!(d.stop_loss(0.0), d.mean());
        assert_eq!(d.layer(f64::INFINITY, 0.0), d.mean());
        // Layer limits stack: 1000 xs 0 + 1000 xs 1000 = 2000 xs 0.
        let stacked = d.layer(1_000.0, 0.0) + d.layer(1_000.0, 1_000.0);
        assert!((stacked - d.layer(2_000.0, 0.0)).abs() < 1e-9);
    }

    #[test]
    fn stop_loss_keeps_precision_in_the_tail() {
        // Retention 1135 is about the 1 - 1e-12 quantile of Lognormal(0, 1).
        let d = Lognormal::new(0.0, 1.0).unwrap();
        let sl = d.stop_loss(1135.0);
        // mpmath at 40 digits, closed form and quadrature agree.
        let exact = 1.796211496502316e-10;
        assert!((sl / exact - 1.0).abs() < 1e-12, "{sl}");
        // The naive difference keeps only about 6 digits here (relative
        // error 9e-7), against 2e-14 for the direct formula.
        let naive = d.mean() - d.lev(1135.0);
        assert!((naive / exact - 1.0).abs() > 1e-8);
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Lognormal::new(0.0, 0.0).is_err());
        assert!(Lognormal::new(f64::NAN, 1.0).is_err());
        assert!(Lognormal::from_mean_cv(-1.0, 0.5).is_err());
    }

    #[test]
    fn quantile_edges() {
        let d = Lognormal::new(0.0, 1.0).unwrap();
        assert_eq!(d.quantile(0.0), Ok(0.0));
        assert_eq!(d.quantile(1.0), Ok(f64::INFINITY));
        assert_eq!(d.quantile(1.1), Err(Error::InvalidProbability(1.1)));
        assert_eq!(d.cdf(-1.0), 0.0);
    }

    /// Same draws are asserted in `python/tests` and `R/prospicio/tests`,
    /// proving all front ends share this kernel.
    const PINNED_SAMPLE: [f64; 3] = [1.0007760893701914, 1.6293872534754683, 1.0763869265482304];

    #[test]
    fn sample_is_pinned() {
        let d = Lognormal::new(0.0, 1.0).unwrap();
        assert_eq!(d.sample(&mut StreamRng::new(42, 3), 3), PINNED_SAMPLE);
    }

    #[test]
    fn sample_is_reproducible_and_centred() {
        let d = Lognormal::from_mean_cv(100.0, 0.3).unwrap();
        let a = d.sample(&mut StreamRng::new(1, 0), 200_000);
        let b = d.sample(&mut StreamRng::new(1, 0), 200_000);
        assert_eq!(a, b);
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        // Standard error of the mean is 30 / sqrt(200_000) ~ 0.067.
        assert!((mean - 100.0).abs() < 0.35, "mean {mean}");
    }
}
