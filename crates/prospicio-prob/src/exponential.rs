//! The exponential distribution.

use prospicio_core::{Error, Result};
use prospicio_math::special::gamma_inc;

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Exponential distribution with mean `θ`: `P(X > x) = exp(−x/θ)`, as
/// SciPy's `expon(scale=θ)` and R's `dexp(rate = 1/θ)`.
///
/// It is memoryless, `X − a | X > a` is the same exponential, so every
/// layer moment is in closed form: the loss to `l xs a` is `S(a)` times
/// `min(X, l)`, and `E[min(X, l)^j] = j! θ^j P(j, l/θ)` with `P` the
/// regularized lower incomplete gamma.
///
/// ```
/// use prospicio_prob::{Distribution, Exponential, Severity};
///
/// let d = Exponential::new(2.0).unwrap();
/// assert!((d.survival(3.0) - (-1.5f64).exp()).abs() < 1e-15);
/// // Memoryless: the excess over 3 is the same exponential.
/// assert!((d.stop_loss(3.0) - 2.0 * (-1.5f64).exp()).abs() < 1e-15);
/// assert!((d.variance() - 4.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    mean: f64,
}

impl Exponential {
    /// Exponential with mean `θ > 0`.
    pub fn new(mean: f64) -> Result<Self> {
        if !(mean.is_finite() && mean > 0.0) {
            return Err(Error::InvalidParameter {
                name: "mean",
                value: mean,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { mean })
    }

    /// The mean `θ`, the parameter (the rate is `1/θ`).
    pub fn mean_param(&self) -> f64 {
        self.mean
    }

    /// `E[min(X, l)^j] = j! θ^j P(j, l/θ)` for `j` in 1 to 3.
    fn limited_moment(&self, j: i32, limit: f64) -> f64 {
        let factorial = [1.0, 1.0, 2.0, 6.0][j as usize];
        let full = factorial * self.mean.powi(j);
        if limit == f64::INFINITY {
            return full;
        }
        if limit <= 0.0 {
            return 0.0;
        }
        full * gamma_inc(f64::from(j), limit / self.mean).0
    }

    /// `E[min((X − a)+, l)^j] = S(a) E[min(X, l)^j]`.
    fn layer_moment(&self, j: i32, limit: f64, attachment: f64) -> f64 {
        if limit.is_nan() || limit <= 0.0 {
            return 0.0;
        }
        self.survival(attachment) * self.limited_moment(j, limit)
    }
}

impl Distribution for Exponential {
    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        self.mean * self.mean
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -(-x / self.mean).exp_m1()
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        (-x / self.mean).exp()
    }

    /// `−θ ln(1 − p)`, with `ln(1 − p)` taken as `ln_1p(−p)`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok(-self.mean * (-p).ln_1p())
    }
}

impl Severity for Exponential {
    /// `−θ expm1(−u/θ)`.
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean;
        }
        -self.mean * (-limit / self.mean).exp_m1()
    }

    /// `θ S(d)`.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean - retention;
        }
        self.mean * self.survival(retention)
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.layer_moment(1, limit, attachment.max(0.0))
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.layer_moment(2, limit, attachment.max(0.0))
    }

    fn layer_third_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.layer_moment(3, limit, attachment.max(0.0))
    }

    /// `6 θ³`.
    fn third_moment(&self) -> Option<f64> {
        Some(6.0 * self.mean.powi(3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gamma;
    use prospicio_core::StreamRng;

    #[test]
    fn matches_the_gamma_with_shape_one() {
        let e = Exponential::new(700.0).unwrap();
        let g = Gamma::new(1.0, 700.0).unwrap();
        for x in [1.0, 300.0, 700.0, 5000.0] {
            assert!((e.cdf(x) - g.cdf(x)).abs() < 1e-14);
            assert!((e.lev(x) / g.lev(x) - 1.0).abs() < 1e-12);
            assert!((e.stop_loss(x) / g.stop_loss(x) - 1.0).abs() < 1e-12);
            for (l, a) in [(500.0, 0.0), (1000.0, x), (f64::INFINITY, x)] {
                let (m1, n1) = (e.layer(l, a), g.layer(l, a));
                assert!((m1 / n1 - 1.0).abs() < 1e-11, "{l} {a}: {m1} {n1}");
                let (m2, n2) = (e.layer_second_moment(l, a), g.layer_second_moment(l, a));
                assert!((m2 / n2 - 1.0).abs() < 1e-10, "{l} {a}: {m2} {n2}");
                // The quadrature default against the closed form.
                let m3 = e.layer_third_moment(l, a);
                let n3 = g.layer_third_moment(l, a);
                assert!((m3 / n3 - 1.0).abs() < 1e-8, "{l} {a}: {m3} {n3}");
            }
            let q = e.quantile(e.cdf(x)).unwrap();
            assert!((q / x - 1.0).abs() < 1e-12);
        }
        assert_eq!(e.third_moment(), Some(6.0 * 700f64.powi(3)));
        assert!(Exponential::new(0.0).is_err());
        let draws = e.sample(&mut StreamRng::new(1, 0), 200_000);
        let m = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((m / 700.0 - 1.0).abs() < 0.01);
    }
}
