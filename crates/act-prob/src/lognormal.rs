//! The lognormal distribution.

use act_core::{Error, Result};
use act_math::special::{norm_cdf, norm_quantile};

use crate::distribution::{Distribution, check_probability};

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

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok((self.meanlog + self.sdlog * norm_quantile(p)).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_core::StreamRng;

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

    /// Same draws are asserted in `python/tests` and `R/actuarialrs/tests`,
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
