//! The beta distribution, scaled to `[0, θ]`.

use prospicio_core::{Error, Result};
use prospicio_math::special::{beta_inc, ln_gamma};

use crate::distribution::{Distribution, bisect_quantile, check_probability};
use crate::severity::{Moments, severity_from_moments};

/// Beta distribution with shapes `a`, `b` on `[0, θ]`: `X / θ` is
/// Beta(`a`, `b`), as SciPy's `beta(a, b, scale=θ)` and actuar's
/// `dgenbeta(shape1=a, shape2=b, shape3=1, scale=θ)`. A bounded severity:
/// a loss as a share of a sum insured, a damage ratio.
///
/// `E[X^j] = θ^j B(a + j, b) / B(a, b)` and
/// `E[X^j; X <= u] = E[X^j] I_{u/θ}(a + j, b)`.
///
/// ```
/// use prospicio_prob::{Beta, Distribution, Severity};
///
/// // Beta(1, 1) on [0, 10] is uniform.
/// let d = Beta::new(1.0, 1.0, 10.0).unwrap();
/// assert!((d.mean() - 5.0).abs() < 1e-14);
/// assert!((d.lev(4.0) - (4.0 - 0.8)).abs() < 1e-14);
/// assert_eq!(d.quantile(1.0).unwrap(), 10.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
    scale: f64,
}

impl Beta {
    /// Beta with shapes `a > 0`, `b > 0` on `[0, θ]`, `θ > 0`.
    pub fn new(a: f64, b: f64, scale: f64) -> Result<Self> {
        for (name, v) in [("a", a), ("b", b), ("scale", scale)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        Ok(Self { a, b, scale })
    }

    /// First shape `a`.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Second shape `b`.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Scale `θ`, the top of the support.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `E[X^j]`.
    fn raw_moment(&self, j: f64) -> f64 {
        let (a, b) = (self.a, self.b);
        let ln = ln_gamma(a + j) + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(a + b + j);
        self.scale.powf(j) * ln.exp()
    }
}

impl Moments for Beta {
    fn limited(&self, j: i32, u: f64) -> f64 {
        let jf = f64::from(j);
        if u <= 0.0 {
            return 0.0;
        }
        if u >= self.scale {
            return self.raw_moment(jf);
        }
        let x = u / self.scale;
        self.raw_moment(jf) * beta_inc(self.a + jf, self.b, x) + u.powi(j) * self.survival(u)
    }

    /// `E[X^j] I_{1-u/θ}(b, a + j) - u^j S(u)`.
    fn tail(&self, j: i32, u: f64) -> f64 {
        let jf = f64::from(j);
        if u <= 0.0 {
            return self.raw_moment(jf);
        }
        if u >= self.scale {
            return 0.0;
        }
        let y = 1.0 - u / self.scale;
        let upper = self.raw_moment(jf) * beta_inc(self.b, self.a + jf, y);
        (upper - u.powi(j) * self.survival(u)).max(0.0)
    }

    fn pivot(&self, _j: i32) -> Option<f64> {
        Some(self.mean())
    }
}

severity_from_moments!(Beta);

impl Distribution for Beta {
    /// `θ a / (a + b)`.
    fn mean(&self) -> f64 {
        self.scale * self.a / (self.a + self.b)
    }

    /// `θ² a b / ((a + b)² (a + b + 1))`.
    fn variance(&self) -> f64 {
        let (a, b) = (self.a, self.b);
        self.scale * self.scale * a * b / ((a + b) * (a + b) * (a + b + 1.0))
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= self.scale {
            return 1.0;
        }
        beta_inc(self.a, self.b, x / self.scale)
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x >= self.scale {
            return 0.0;
        }
        beta_inc(self.b, self.a, 1.0 - x / self.scale)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(self.scale);
        }
        Ok(bisect_quantile(
            p,
            0.0,
            self.scale,
            self.scale,
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
        for (a, b) in [(0.5, 0.5), (2.0, 5.0), (8.0, 1.5)] {
            let d = Beta::new(a, b, 1000.0).unwrap();
            for u in [10.0, 300.0, 900.0] {
                let x = d.quantile(d.cdf(u)).unwrap();
                assert!((x / u - 1.0).abs() < 1e-11, "{a} {b} {u}");
                let total = d.lev(u) + d.stop_loss(u);
                assert!((total / d.mean() - 1.0).abs() < 1e-12, "{a} {b} {u}");
            }
            let stacked = d.layer(500.0, 0.0) + d.layer(1500.0, 500.0);
            assert!((stacked / d.layer(2000.0, 0.0) - 1.0).abs() < 1e-12);
            let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
            assert!((m2 / (d.variance() + d.mean().powi(2)) - 1.0).abs() < 1e-12);
            assert_eq!(d.layer(100.0, 1000.0), 0.0);
        }
        assert!(Beta::new(1.0, 1.0, 0.0).is_err());
    }
}
