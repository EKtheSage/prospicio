//! The loglogistic (Fisk) distribution.

use std::f64::consts::PI;

use prospicio_core::{Error, Result};
use prospicio_math::special::beta_lower;

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Loglogistic distribution with shape `α` and scale `θ`:
/// `F(x) = (x/θ)^α / (1 + (x/θ)^α)`, as SciPy's `fisk(c=α, scale=θ)` and
/// actuar's `dllogis(shape, scale)`. Its median is `θ`.
///
/// It is also the growth curve of Clark's LDF method, `G(t) = t^ω / (t^ω + θ^ω)`,
/// so a fitted curve is this distribution's [`cdf`](Distribution::cdf).
///
/// The tail is Pareto-like: `E[X^j]` is finite only for `j < α`, so the
/// mean is infinite for `α <= 1` and the variance for `α <= 2`. Limited and
/// layer moments are finite for every `α`. With `v = F(u)`,
/// `E[min(X, u)^j] = θ^j B(1 + j/α, 1 - j/α; v) + u^j (1 - v)`, using the
/// unnormalized incomplete beta, which for `j >= α` has a non-positive
/// second argument and is reached by recurrence.
///
/// ```
/// use prospicio_prob::{Distribution, Loglogistic, Severity};
///
/// // Shape 1: F(x) = x / (x + θ) and E[min(X, u)] = θ ln(1 + u/θ).
/// let d = Loglogistic::new(1.0, 2.0).unwrap();
/// assert!((d.cdf(3.0) - 0.6).abs() < 1e-15);
/// assert!((d.lev(3.0) - 2.0 * 2.5f64.ln()).abs() < 1e-13);
/// assert!(d.mean().is_infinite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loglogistic {
    shape: f64,
    scale: f64,
}

impl Loglogistic {
    /// Loglogistic with shape `α > 0` and scale `θ > 0`.
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

    /// Shape `α`.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale `θ`, the median.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `E[X^j] = θ^j (πj/α) / sin(πj/α)`, infinite for `j >= α`.
    fn raw_moment(&self, j: f64) -> f64 {
        if j >= self.shape {
            return f64::INFINITY;
        }
        let r = PI * j / self.shape;
        self.scale.powf(j) * r / r.sin()
    }

    /// `(F(u), S(u))`, each computed without cancellation.
    fn probs(&self, u: f64) -> (f64, f64) {
        let z = (u / self.scale).powf(self.shape);
        (z / (1.0 + z), 1.0 / (1.0 + z))
    }

    /// `E[min(X, u)^j]` for `u > 0`.
    fn limited_moment(&self, j: i32, u: f64) -> f64 {
        if u == f64::INFINITY {
            return self.raw_moment(f64::from(j));
        }
        let (v, s) = self.probs(u);
        let r = f64::from(j) / self.shape;
        self.scale.powi(j) * beta_lower(1.0 + r, 1.0 - r, v) + u.powi(j) * s
    }

    /// `E[X^j; X > u] - u^j S(u)`, the tail part of
    /// `E[X^j] - E[min(X, u)^j]`, for `j < α` and `u > 0`; it uses
    /// `E[X^j; X > u] = θ^j B(1 - j/α, 1 + j/α; S(u))`.
    fn tail_moment(&self, j: i32, u: f64) -> f64 {
        if u == f64::INFINITY {
            return 0.0;
        }
        let (_, s) = self.probs(u);
        let r = f64::from(j) / self.shape;
        (self.scale.powi(j) * beta_lower(1.0 - r, 1.0 + r, s) - u.powi(j) * s).max(0.0)
    }

    /// `∫_a^b x^(j-1) S(x) dx` for `0 <= a < b <= ∞`: limited-moment
    /// differences below `c = θ 2^(1/α)` (where `S <= 1/3` begins), and the
    /// tail series above it, so a far layer does not cancel.
    fn partial(&self, j: i32, a: f64, b: f64) -> f64 {
        let below = |u: f64| {
            if u <= 0.0 {
                0.0
            } else {
                self.limited_moment(j, u) / f64::from(j)
            }
        };
        let c = self.scale * 2f64.powf(1.0 / self.shape);
        if b <= c {
            return below(b) - below(a);
        }
        if a >= c {
            return self.tail_series(j, a, b);
        }
        below(c) - below(a) + self.tail_series(j, c, b)
    }

    /// `∫_a^b x^(j-1) S(x) dx` for `θ 2^(1/α) <= a < b`. With
    /// `r = (θ/x)^α <= 1/2`, it is `(θ^j/α) ∫ r^(-j/α) / (1 + r) dr` over
    /// `[r_b, r_a]`, integrated term by term in the geometric series of
    /// `1 / (1 + r)`; infinite when `b = ∞` and `j >= α`.
    fn tail_series(&self, j: i32, a: f64, b: f64) -> f64 {
        let ra = (self.scale / a).powf(self.shape);
        let rb = if b == f64::INFINITY {
            0.0
        } else {
            (self.scale / b).powf(self.shape)
        };
        let ln_ratio = (rb / ra).ln();
        let r = f64::from(j) / self.shape;
        let mut sum = 0.0;
        for k in 0..200 {
            // ∫_{r_b}^{r_a} r^(e - 1) dr with e = k + 1 - j/α.
            let e = f64::from(k) + 1.0 - r;
            let term = if e.abs() < 1e-12 {
                -ln_ratio
            } else if rb == 0.0 {
                if e > 0.0 {
                    ra.powf(e) / e
                } else {
                    f64::INFINITY
                }
            } else {
                -ra.powf(e) * (e * ln_ratio).exp_m1() / e
            };
            if !term.is_finite() {
                return f64::INFINITY;
            }
            sum += if k % 2 == 0 { term } else { -term };
            if k > 0 && term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        self.scale.powi(j) / self.shape * sum
    }
}

impl Distribution for Loglogistic {
    fn mean(&self) -> f64 {
        self.raw_moment(1.0)
    }

    fn variance(&self) -> f64 {
        if self.shape <= 2.0 {
            return f64::INFINITY;
        }
        let m = self.mean();
        self.raw_moment(2.0) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.probs(x).0
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        self.probs(x).1
    }

    /// `θ (p / (1 - p))^(1/α)`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok(self.scale * (p / (1.0 - p)).powf(1.0 / self.shape))
    }
}

impl Severity for Loglogistic {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.limited_moment(1, limit)
    }

    /// From the tail, so it does not cancel against the mean; infinite for
    /// `α <= 1`.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        if self.shape <= 1.0 {
            return f64::INFINITY;
        }
        self.tail_moment(1, retention)
    }

    /// `∫_a^(a+l) S(x) dx`, by the tail series in the tail.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.partial(1, a, a + limit)
    }

    /// `2 ∫_a^b (x - a) S(x) dx` with `b = a + limit`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let p1 = self.partial(1, a, b);
        if p1 == 0.0 {
            return 0.0;
        }
        (2.0 * (self.partial(2, a, b) - a * p1)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities() {
        for shape in [0.5, 0.8, 1.0, 1.5, 2.0, 4.0] {
            let d = Loglogistic::new(shape, 100.0).unwrap();
            for u in [1.0, 100.0, 20_000.0] {
                // Far in the tail, 1 - F(u) has too few digits to invert.
                if d.cdf(u) < 0.99 {
                    let x = d.quantile(d.cdf(u)).unwrap();
                    assert!((x / u - 1.0).abs() < 1e-12);
                }
                if shape > 1.0 {
                    let total = d.lev(u) + d.stop_loss(u);
                    assert!((total / d.mean() - 1.0).abs() < 1e-12, "{shape} {u}");
                }
            }
            let stacked = d.layer(500.0, 0.0) + d.layer(500.0, 500.0);
            assert!((stacked / d.layer(1000.0, 0.0) - 1.0).abs() < 1e-12);
            assert!(d.layer_second_moment(200.0, 300.0) > 0.0);
        }
        let d = Loglogistic::new(4.0, 3.0).unwrap();
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!((m2 / (d.variance() + d.mean().powi(2)) - 1.0).abs() < 1e-12);
        assert!(Loglogistic::new(0.0, 1.0).is_err());
    }

    /// Layer moments by the tail series and by limited moments agree.
    #[test]
    fn layer_second_moment_branches() {
        let d = Loglogistic::new(3.0, 50.0).unwrap();
        let (a, b) = (40.0, 90.0);
        let direct = d.lev(b) - d.lev(a);
        assert!((d.layer(b - a, a) / direct - 1.0).abs() < 1e-12);
        let m = |j: i32, u: f64| d.limited_moment(j, u);
        let direct = m(2, b) - m(2, a) - 2.0 * a * (m(1, b) - m(1, a));
        let tails = d.layer_second_moment(b - a, a);
        assert!((tails / direct - 1.0).abs() < 1e-11);
    }
}
