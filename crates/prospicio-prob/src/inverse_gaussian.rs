//! The inverse Gaussian distribution.

use prospicio_core::{Error, Result};
use prospicio_math::special::{mills_ratio, norm_cdf, norm_pdf};

use crate::distribution::{Distribution, bisect_quantile, check_probability};
use crate::severity::{Moments, severity_from_moments};

/// Inverse Gaussian distribution with mean `μ` and shape `λ`, density
/// `√(λ / 2πx³) exp(-λ(x - μ)² / (2μ²x))`; variance `μ³/λ`. SciPy's
/// `invgauss(mu=μ/λ, scale=λ)`, actuar's `dinvgauss(mean, shape)`.
///
/// With `a = (x - μ)/μ √(λ/x)`, `b = (x + μ)/μ √(λ/x)` and the Mills ratio
/// `R = Φ(-·)/φ`, the distribution function is `Φ(a) + φ(a) R(b)`: the
/// factor `e^(2λ/μ)` of the textbook form cancels exactly against `φ(b)/φ(a)`,
/// so nothing overflows at small coefficients of variation and the survival
/// `φ(a) (R(a) - R(b))` keeps its precision in the tail.
///
/// Limited moments: `x f(x) / μ` is the density of `1/Y` with `Y`
/// inverse Gaussian `(1/μ, λ/μ²)`, which gives `E[X; X > u] = μ (Φ(-a) + φ(a) R(b))`,
/// and integrating the density's derivative by parts gives
/// `E[X²; X > u] = (μ²/λ)(E[X; X > u] + λ S(u) + 2u² f(u))`.
///
/// ```
/// use prospicio_prob::{Distribution, InverseGaussian, Severity};
///
/// let d = InverseGaussian::from_mean_cv(1000.0, 0.5).unwrap();
/// assert!((d.variance().sqrt() - 500.0).abs() < 1e-9);
/// assert!((d.lev(800.0) + d.stop_loss(800.0) - 1000.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussian {
    mean: f64,
    shape: f64,
}

impl InverseGaussian {
    /// Inverse Gaussian with mean `μ > 0` and shape `λ > 0`.
    pub fn new(mean: f64, shape: f64) -> Result<Self> {
        for (name, v) in [("mean", mean), ("shape", shape)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        Ok(Self { mean, shape })
    }

    /// Inverse Gaussian with the given mean and coefficient of variation:
    /// `λ = μ / cv²`.
    pub fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        if !(cv.is_finite() && cv > 0.0) {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "must be finite and positive",
            });
        }
        Self::new(mean, mean / (cv * cv))
    }

    /// Mean `μ`.
    pub fn mean_param(&self) -> f64 {
        self.mean
    }

    /// Shape `λ`.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// `(a, b)` at `x > 0`.
    fn ab(&self, x: f64) -> (f64, f64) {
        let r = (self.shape / x).sqrt();
        (
            (x - self.mean) / self.mean * r,
            (x + self.mean) / self.mean * r,
        )
    }

    /// `(F(x), S(x))` for `0 < x < ∞`, each without cancellation where it
    /// is the small one.
    fn probs(&self, x: f64) -> (f64, f64) {
        let (a, b) = self.ab(x);
        if a <= 0.0 {
            let f = norm_cdf(a) + norm_pdf(a) * mills_ratio(b);
            (f, 1.0 - f)
        } else {
            let s = norm_pdf(a) * (mills_ratio(a) - mills_ratio(b));
            (1.0 - s, s)
        }
    }

    /// `(E[X; X > u], S(u), u² f(u))` for `0 < u < ∞`.
    fn upper(&self, u: f64) -> (f64, f64, f64) {
        let (a, b) = self.ab(u);
        let pa = norm_pdf(a);
        let m1 = if a <= 0.0 {
            self.mean * (norm_cdf(-a) + pa * mills_ratio(b))
        } else {
            self.mean * pa * (mills_ratio(a) + mills_ratio(b))
        };
        let density = (self.shape / u).sqrt() / u * pa;
        (m1, self.probs(u).1, u * u * density)
    }
}

impl Moments for InverseGaussian {
    fn limited(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        if u == f64::INFINITY {
            return if j == 1 {
                self.mean
            } else {
                self.variance() + self.mean * self.mean
            };
        }
        let (a, b) = self.ab(u);
        let pa = norm_pdf(a);
        let (f, s) = self.probs(u);
        // E[X; X <= u] = μ (Φ(a) - φ(a) R(b)).
        let below = if a <= 0.0 {
            self.mean * pa * (mills_ratio(-a) - mills_ratio(b))
        } else {
            self.mean * (norm_cdf(a) - pa * mills_ratio(b))
        };
        if j == 1 {
            return below + u * s;
        }
        let u2f = u * (self.shape / u).sqrt() * pa;
        let m2 = self.mean * self.mean / self.shape * (below + self.shape * f - 2.0 * u2f);
        m2.max(0.0) + u * u * s
    }

    fn tail(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return self.limited(j, f64::INFINITY);
        }
        if u == f64::INFINITY {
            return 0.0;
        }
        let (m1, s, u2f) = self.upper(u);
        let upper = if j == 1 {
            m1
        } else {
            self.mean * self.mean / self.shape * (m1 + self.shape * s + 2.0 * u2f)
        };
        (upper - u.powi(j) * s).max(0.0)
    }

    fn pivot(&self, _j: i32) -> Option<f64> {
        Some(self.mean)
    }
}

severity_from_moments!(InverseGaussian);

impl Distribution for InverseGaussian {
    fn mean(&self) -> f64 {
        self.mean
    }

    /// `μ³ / λ`.
    fn variance(&self) -> f64 {
        self.mean.powi(3) / self.shape
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        self.probs(x).0
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x == f64::INFINITY {
            return 0.0;
        }
        self.probs(x).1
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(bisect_quantile(
            p,
            0.0,
            self.mean + self.std_dev(),
            f64::INFINITY,
            |x| self.cdf(x),
            |x| self.survival(x),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Severity;

    #[test]
    fn identities() {
        for cv in [0.05, 0.5, 1.0, 3.0] {
            let d = InverseGaussian::from_mean_cv(1000.0, cv).unwrap();
            for u in [100.0, 1000.0, 5000.0] {
                let x = d.quantile(d.cdf(u)).unwrap();
                if d.cdf(u) > 1e-300 && d.cdf(u) < 0.99 {
                    assert!((x / u - 1.0).abs() < 1e-10, "{cv} {u}");
                }
                let total = d.lev(u) + d.stop_loss(u);
                assert!((total / d.mean() - 1.0).abs() < 1e-12, "{cv} {u}");
            }
            let stacked = d.layer(500.0, 0.0) + d.layer(1500.0, 500.0);
            assert!((stacked / d.layer(2000.0, 0.0) - 1.0).abs() < 1e-12);
            let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
            assert!((m2 / (d.variance() + 1e6) - 1.0).abs() < 1e-12, "{cv}");
        }
        // Tiny cv: e^(2λ/μ) would overflow in the textbook form.
        let d = InverseGaussian::from_mean_cv(1.0, 0.02).unwrap();
        assert!((d.cdf(1.0) - 0.5).abs() < 0.05);
        assert!(d.cdf(1.1).is_finite() && d.survival(1.1) > 0.0);
        assert!(InverseGaussian::new(1.0, 0.0).is_err());
    }
}
