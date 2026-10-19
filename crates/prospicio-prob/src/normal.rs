//! The normal distribution, on the whole real line: asset returns, log
//! returns and other outcomes of either sign. It is a [`Distribution`] but
//! not a [`Severity`](crate::Severity), whose losses are non-negative.

use prospicio_core::{Error, Result};
use prospicio_math::special::{mills_ratio, norm_cdf, norm_pdf, norm_quantile};

use crate::distribution::{Distribution, check_probability};

/// Normal distribution with mean `mu` and standard deviation `sigma`.
///
/// Its TVaR is `μ + σ φ(z_p) / (1 − p)` with `z_p = Φ⁻¹(p)` (*Pricing
/// Insurance Risk*, section 4.3.3): VaR plus a multiple of `σ` that depends
/// only on `p`.
///
/// # Example
///
/// ```
/// use prospicio_prob::{Distribution, Normal};
///
/// let x = Normal::new(0.0, 1.0).unwrap();
/// assert!((x.quantile(0.975).unwrap() - 1.959963984540054).abs() < 1e-12);
/// assert!((x.tvar(0.99).unwrap() - 2.665214220345808).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    /// Normal with mean `mu` and standard deviation `sigma > 0`.
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        if !mu.is_finite() {
            return Err(Error::InvalidParameter {
                name: "mu",
                value: mu,
                reason: "must be finite",
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma",
                value: sigma,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { mu, sigma })
    }

    /// The mean.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// The standard deviation.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The density.
    pub fn pdf(&self, x: f64) -> f64 {
        norm_pdf((x - self.mu) / self.sigma) / self.sigma
    }

    /// Tail value at risk at `p` in `[0, 1)`, `E[X | X ≥ VaR(p)]`:
    /// `μ + σ φ(z_p) / (1 − p)`, by the Mills ratio above the median so
    /// that it keeps its precision far in the tail.
    pub fn tvar(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p >= 1.0 {
            return Err(Error::InvalidProbability(p));
        }
        if p == 0.0 {
            return Ok(self.mu);
        }
        let z = norm_quantile(p);
        // φ(z) / (1 − p) = 1 / R(z) with R the Mills ratio (1 − Φ) / φ.
        let multiple = if z > 0.0 {
            1.0 / mills_ratio(z)
        } else {
            norm_pdf(z) / (1.0 - p)
        };
        Ok(self.mu + self.sigma * multiple)
    }
}

impl Distribution for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    fn std_dev(&self) -> f64 {
        self.sigma
    }

    fn cdf(&self, x: f64) -> f64 {
        norm_cdf((x - self.mu) / self.sigma)
    }

    fn survival(&self, x: f64) -> f64 {
        norm_cdf((self.mu - x) / self.sigma)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok(self.mu + self.sigma * norm_quantile(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_and_quantiles() {
        let x = Normal::new(10.0, 2.0).unwrap();
        assert_eq!(x.mean(), 10.0);
        assert_eq!(x.variance(), 4.0);
        assert!((x.cdf(10.0) - 0.5).abs() < 1e-15);
        assert!((x.cdf(12.0) + x.survival(12.0) - 1.0).abs() < 1e-15);
        assert!((x.quantile(x.cdf(13.1)).unwrap() - 13.1).abs() < 1e-10);
        assert_eq!(x.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(x.quantile(1.0).unwrap(), f64::INFINITY);
        assert!((x.pdf(10.0) - 0.5 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn tvar_is_var_plus_a_multiple_of_sigma() {
        // E[X | X ≥ VaR] by the density: ∫_z^∞ t φ(t) dt = φ(z).
        let x = Normal::new(5.0, 3.0).unwrap();
        for p in [0.01, 0.3, 0.5, 0.9, 0.99, 0.999_999] {
            let z = norm_quantile(p);
            let want = 5.0 + 3.0 * norm_pdf(z) / (1.0 - p);
            let got = x.tvar(p).unwrap();
            assert!(
                (got / want - 1.0).abs() < 1e-9,
                "p {p}: {got} against {want}"
            );
            assert!(got > x.quantile(p).unwrap());
        }
        assert_eq!(x.tvar(0.0).unwrap(), 5.0);
        assert!(x.tvar(1.0).is_err());
        assert!(x.tvar(-0.1).is_err());
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Normal::new(f64::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, f64::INFINITY).is_err());
    }
}
