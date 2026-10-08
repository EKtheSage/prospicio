//! The inverse gamma distribution.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::special::{gamma_inc, gamma_upper, ln_gamma};

use crate::Gamma;
use crate::distribution::{Distribution, bisect_quantile, check_probability};
use crate::severity::{Moments, severity_from_moments};

/// Inverse gamma distribution with shape `α` and scale `θ`: `X = θ / G` for
/// `G` a unit-scale gamma with shape `α`, so `P(X <= x) = Q(α, θ/x)`, as
/// SciPy's `invgamma(a=α, scale=θ)` and actuar's `dinvgamma(shape, scale)`.
///
/// The tail is Pareto-like with index `α`: `E[X^j] = θ^j Γ(α - j) / Γ(α)`
/// exists only for `j < α`. Limited moments exist for every `α`:
/// `E[X^j; X <= u] = θ^j Γ(α - j, θ/u) / Γ(α)` with the upper incomplete
/// gamma, which [`gamma_upper`] extends to a non-positive first argument.
///
/// ```
/// use prospicio_prob::{Distribution, InverseGamma, Severity};
///
/// let d = InverseGamma::new(3.0, 2000.0).unwrap();
/// assert!((d.mean() - 1000.0).abs() < 1e-9);
/// assert!((d.variance() - 1e6).abs() < 1e-6);
/// assert!((d.lev(f64::INFINITY) - 1000.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    scale: f64,
}

impl InverseGamma {
    /// Inverse gamma with shape `α > 0` and scale `θ > 0`.
    pub fn new(shape: f64, scale: f64) -> Result<Self> {
        for (name, v) in [("shape", shape), ("scale", scale)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        Ok(Self { shape, scale })
    }

    /// Inverse gamma with the given mean and coefficient of variation:
    /// `α = 2 + 1/cv²` and `θ = mean (α - 1)`.
    pub fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        if !(cv.is_finite() && cv > 0.0) {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "must be finite and positive",
            });
        }
        let shape = 2.0 + 1.0 / (cv * cv);
        Self::new(shape, mean * (shape - 1.0))
    }

    /// Shape `α`.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale `θ`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `E[X^j]`, infinite for `j >= α`.
    fn raw_moment(&self, j: f64) -> f64 {
        if j >= self.shape {
            return f64::INFINITY;
        }
        self.scale.powf(j) * (ln_gamma(self.shape - j) - ln_gamma(self.shape)).exp()
    }
}

impl Moments for InverseGamma {
    fn limited(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        let jf = f64::from(j);
        if u == f64::INFINITY {
            return self.raw_moment(jf);
        }
        let z = self.scale / u;
        let s = self.shape - jf;
        let below = if s > 0.0 {
            self.raw_moment(jf) * gamma_inc(s, z).1
        } else {
            self.scale.powi(j) * gamma_upper(s, z) / ln_gamma(self.shape).exp()
        };
        below + u.powi(j) * gamma_inc(self.shape, z).0
    }

    /// `E[X^j] P(α - j, θ/u) - u^j P(α, θ/u)`.
    fn tail(&self, j: i32, u: f64) -> f64 {
        let jf = f64::from(j);
        if u <= 0.0 {
            return self.raw_moment(jf);
        }
        if u == f64::INFINITY {
            return 0.0;
        }
        let z = self.scale / u;
        let upper = self.raw_moment(jf) * gamma_inc(self.shape - jf, z).0;
        (upper - u.powi(j) * gamma_inc(self.shape, z).0).max(0.0)
    }

    fn pivot(&self, j: i32) -> Option<f64> {
        (f64::from(j) < self.shape).then(|| self.mean())
    }
}

severity_from_moments!(InverseGamma);

impl Distribution for InverseGamma {
    /// `θ / (α - 1)`, infinite for `α <= 1`.
    fn mean(&self) -> f64 {
        self.raw_moment(1.0)
    }

    /// `mean² / (α - 2)`, infinite for `α <= 2`.
    fn variance(&self) -> f64 {
        if self.shape <= 2.0 {
            return f64::INFINITY;
        }
        let m = self.mean();
        m * m / (self.shape - 2.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gamma_inc(self.shape, self.scale / x).1
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        gamma_inc(self.shape, self.scale / x).0
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
            self.scale,
            f64::INFINITY,
            |x| self.cdf(x),
            |x| self.survival(x),
        ))
    }

    /// `θ / G` for `G` drawn by the gamma's sampler (Marsaglia and Tsang),
    /// not inverse transform; see [`Distribution::sample`].
    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        let unit = Gamma::new(self.shape, 1.0).expect("a positive shape");
        unit.sample(rng, n)
            .into_iter()
            .map(|g| self.scale / g)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Severity;

    #[test]
    fn identities() {
        for shape in [0.6, 1.0, 1.5, 2.0, 3.5] {
            let d = InverseGamma::new(shape, 1000.0).unwrap();
            for u in [10.0, 1000.0, 50_000.0] {
                // Far in the tail, 1 - F(u) has too few digits to invert.
                if d.cdf(u) < 0.99 {
                    let x = d.quantile(d.cdf(u)).unwrap();
                    assert!((x / u - 1.0).abs() < 1e-11, "{shape} {u}");
                }
                if shape > 1.0 {
                    let total = d.lev(u) + d.stop_loss(u);
                    assert!((total / d.mean() - 1.0).abs() < 1e-12, "{shape} {u}");
                }
            }
            let stacked = d.layer(500.0, 0.0) + d.layer(1500.0, 500.0);
            assert!((stacked / d.layer(2000.0, 0.0) - 1.0).abs() < 1e-12);
            assert!(d.layer_second_moment(2000.0, 3000.0) > 0.0);
        }
        let d = InverseGamma::new(3.5, 1000.0).unwrap();
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!((m2 / (d.variance() + d.mean().powi(2)) - 1.0).abs() < 1e-12);
        assert!(InverseGamma::from_mean_cv(1000.0, 0.5).unwrap().variance() > 0.0);
        assert!(InverseGamma::new(0.0, 1.0).is_err());
    }
}
