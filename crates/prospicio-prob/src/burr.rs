//! The Burr (type XII) distribution.

use prospicio_core::{Error, Result};
use prospicio_math::special::{beta_lower, ln_gamma};

use crate::distribution::{Distribution, check_probability};
use crate::severity::{Moments, severity_from_moments};

/// Burr (type XII) distribution with tail shape `α`, power `γ` and scale
/// `θ`: `P(X > x) = (1 + (x/θ)^γ)^(-α)`, as Klugman, Panjer and Willmot's
/// Burr, SciPy's `burr12(c=γ, d=α, scale=θ)` and actuar's
/// `dburr(shape1=α, shape2=γ, scale)`. With `α = 1` it is the
/// [`Loglogistic`](crate::Loglogistic); with `γ = 1`, the Lomax (Pareto II).
///
/// The tail index is `αγ`: `E[X^j] = θ^j Γ(1 + j/γ) Γ(α - j/γ) / Γ(α)`
/// exists only for `j < αγ`. With `z = (u/θ)^γ` and `v = z / (1 + z)`,
/// limited moments are
/// `E[min(X, u)^j] = α θ^j B(1 + j/γ, α - j/γ; v) + u^j S(u)` with the
/// unnormalized incomplete beta ([`beta_lower`]), finite for every `α`.
///
/// ```
/// use prospicio_prob::{Burr, Distribution, Severity};
///
/// // γ = 1 is the Lomax: S(x) = (1 + x/θ)^(-α), mean θ/(α - 1).
/// let d = Burr::new(3.0, 1.0, 2000.0).unwrap();
/// assert!((d.mean() - 1000.0).abs() < 1e-9);
/// assert!((d.survival(2000.0) - 0.125).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Burr {
    alpha: f64,
    gamma: f64,
    scale: f64,
}

impl Burr {
    /// Burr with tail shape `α > 0`, power `γ > 0` and scale `θ > 0`.
    pub fn new(alpha: f64, gamma: f64, scale: f64) -> Result<Self> {
        for (name, v) in [("alpha", alpha), ("gamma", gamma), ("scale", scale)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        Ok(Self {
            alpha,
            gamma,
            scale,
        })
    }

    /// Tail shape `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Power `γ`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Scale `θ`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `E[X^j]`, infinite for `j >= αγ`.
    fn raw_moment(&self, j: f64) -> f64 {
        let r = j / self.gamma;
        if r >= self.alpha {
            return f64::INFINITY;
        }
        let ln = ln_gamma(1.0 + r) + ln_gamma(self.alpha - r) - ln_gamma(self.alpha);
        self.scale.powf(j) * ln.exp()
    }

    /// `(z/(1+z), 1/(1+z), S)` with `z = (x/θ)^γ`, for `x > 0`.
    fn parts(&self, x: f64) -> (f64, f64, f64) {
        let z = (x / self.scale).powf(self.gamma);
        let w = 1.0 / (1.0 + z);
        (z * w, w, (-self.alpha * z.ln_1p()).exp())
    }
}

impl Moments for Burr {
    fn limited(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        if u == f64::INFINITY {
            return self.raw_moment(f64::from(j));
        }
        let (v, _, s) = self.parts(u);
        let r = f64::from(j) / self.gamma;
        self.alpha * self.scale.powi(j) * beta_lower(1.0 + r, self.alpha - r, v) + u.powi(j) * s
    }

    /// `α θ^j B(α - j/γ, 1 + j/γ; 1/(1+z)) - u^j S(u)`.
    fn tail(&self, j: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return self.raw_moment(f64::from(j));
        }
        if u == f64::INFINITY {
            return 0.0;
        }
        let (_, w, s) = self.parts(u);
        let r = f64::from(j) / self.gamma;
        let upper = self.alpha * self.scale.powi(j) * beta_lower(self.alpha - r, 1.0 + r, w);
        (upper - u.powi(j) * s).max(0.0)
    }

    fn pivot(&self, j: i32) -> Option<f64> {
        (f64::from(j) < self.alpha * self.gamma).then(|| self.mean())
    }
}

severity_from_moments!(Burr);

impl Distribution for Burr {
    fn mean(&self) -> f64 {
        self.raw_moment(1.0)
    }

    fn variance(&self) -> f64 {
        if 2.0 >= self.alpha * self.gamma {
            return f64::INFINITY;
        }
        let m = self.mean();
        self.raw_moment(2.0) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x / self.scale).powf(self.gamma);
        -(-self.alpha * z.ln_1p()).exp_m1()
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        self.parts(x).2
    }

    /// `θ ((1 - p)^(-1/α) - 1)^(1/γ)`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let z = (-(-p).ln_1p() / self.alpha).exp_m1();
        Ok(self.scale * z.powf(1.0 / self.gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Loglogistic, Severity};

    #[test]
    fn identities() {
        for (alpha, gamma) in [(0.5, 1.5), (1.0, 1.0), (1.5, 2.0), (2.0, 0.75), (3.0, 4.0)] {
            let d = Burr::new(alpha, gamma, 1000.0).unwrap();
            for u in [10.0, 1000.0, 50_000.0] {
                if d.cdf(u) < 0.99 {
                    let x = d.quantile(d.cdf(u)).unwrap();
                    assert!((x / u - 1.0).abs() < 1e-11, "{alpha} {gamma} {u}");
                }
                if alpha * gamma > 1.0 {
                    let total = d.lev(u) + d.stop_loss(u);
                    assert!(
                        (total / d.mean() - 1.0).abs() < 1e-12,
                        "{alpha} {gamma} {u}"
                    );
                }
            }
            let stacked = d.layer(500.0, 0.0) + d.layer(1500.0, 500.0);
            assert!((stacked / d.layer(2000.0, 0.0) - 1.0).abs() < 1e-12);
            assert!(d.layer_second_moment(2000.0, 3000.0) > 0.0);
        }
        let d = Burr::new(3.0, 4.0, 1000.0).unwrap();
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!((m2 / (d.variance() + d.mean().powi(2)) - 1.0).abs() < 1e-12);
        assert!(Burr::new(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn alpha_one_is_the_loglogistic() {
        for shape in [0.8, 1.5, 4.0] {
            let b = Burr::new(1.0, shape, 300.0).unwrap();
            let l = Loglogistic::new(shape, 300.0).unwrap();
            for u in [50.0, 300.0, 4000.0] {
                assert!((b.cdf(u) / l.cdf(u) - 1.0).abs() < 1e-14);
                assert!((b.lev(u) / l.lev(u) - 1.0).abs() < 1e-12, "{shape} {u}");
                let (lb, ll) = (b.layer(1000.0, u), l.layer(1000.0, u));
                assert!((lb / ll - 1.0).abs() < 1e-10, "{shape} {u}");
            }
        }
    }
}
