//! The Weibull distribution.

use prospicio_core::{Error, Result};
use prospicio_math::special::{gamma_inc, ln_gamma};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Weibull distribution with shape `k` and scale `λ`:
/// `P(X > x) = exp(-(x/λ)^k)`, as SciPy's `weibull_min(c=k, scale=λ)` and
/// R's `dweibull(shape, scale)`.
///
/// Below shape 1 the tail is heavier than exponential (but every moment
/// exists); above 1 it is lighter. Layer moments use the regularized
/// incomplete gamma: with `z = (u/λ)^k`,
/// `E[X^j; X > u] = λ^j Γ(1 + j/k) Q(1 + j/k, z)`.
///
/// ```
/// use prospicio_prob::{Distribution, Severity, Weibull};
///
/// // Shape 1 is the exponential.
/// let e = Weibull::new(1.0, 2.0).unwrap();
/// assert!((e.mean() - 2.0).abs() < 1e-14);
/// assert!((e.stop_loss(3.0) - 2.0 * (-1.5f64).exp()).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull {
    shape: f64,
    scale: f64,
}

impl Weibull {
    /// Weibull with shape `k > 0` and scale `λ > 0`.
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

    /// Shape `k`.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale `λ`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `E[X^j]`.
    fn raw_moment(&self, j: f64) -> f64 {
        self.scale.powf(j) * ln_gamma(1.0 + j / self.shape).exp()
    }

    /// `E[X^j; X > u] - u^j S(u)` for `j` in 1 and 2: the tail part of
    /// `E[X^j] - E[min(X, u)^j]`.
    fn tail_moment(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return self.raw_moment(f64::from(j)) - u.powi(j);
        }
        if u == f64::INFINITY {
            return 0.0;
        }
        let z = (u / self.scale).powf(self.shape);
        let jf = f64::from(j);
        let q = gamma_inc(1.0 + jf / self.shape, z).1;
        (self.raw_moment(jf) * q - u.powi(j) * (-z).exp()).max(0.0)
    }
}

impl Distribution for Weibull {
    fn mean(&self) -> f64 {
        self.raw_moment(1.0)
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        self.raw_moment(2.0) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -(-(x / self.scale).powf(self.shape)).exp_m1()
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-(x / self.scale).powf(self.shape)).exp()
    }

    /// `λ (-ln(1 - p))^(1/k)`, with `ln(1 - p)` taken as `ln_1p(-p)`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok(self.scale * (-(-p).ln_1p()).powf(1.0 / self.shape))
    }
}

impl Severity for Weibull {
    /// `E[min(X, u)] = λ Γ(1 + 1/k) P(1 + 1/k, z) + u e^(-z)`.
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean();
        }
        let z = (limit / self.scale).powf(self.shape);
        self.mean() * gamma_inc(1.0 + 1.0 / self.shape, z).0 + limit * (-z).exp()
    }

    /// From the tail, so it does not cancel against the mean.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.tail_moment(1, retention)
    }

    /// LEV differences below the mean, stop-loss differences above it.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        if a <= self.mean() {
            self.lev(a + limit) - self.lev(a)
        } else {
            self.stop_loss(a) - self.stop_loss(a + limit)
        }
    }

    /// `t_2(a) - t_2(b) - 2a (t_1(a) - t_1(b))` with `b = a + limit`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let t = |j: i32, u: f64| self.tail_moment(j, u);
        (t(2, a) - t(2, b) - 2.0 * a * (t(1, a) - t(1, b))).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities() {
        let d = Weibull::new(0.7, 1000.0).unwrap();
        for u in [10.0, 1000.0, 20_000.0] {
            assert!((d.lev(u) + d.stop_loss(u) - d.mean()).abs() < 1e-10 * d.mean());
            let x = d.quantile(d.cdf(u)).unwrap();
            assert!((x / u - 1.0).abs() < 1e-12);
        }
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!((m2 / (d.variance() + d.mean().powi(2)) - 1.0).abs() < 1e-12);
        let stacked = d.layer(500.0, 0.0) + d.layer(500.0, 500.0);
        assert!((stacked - d.layer(1000.0, 0.0)).abs() < 1e-9);
        assert!(Weibull::new(0.0, 1.0).is_err());
    }
}
