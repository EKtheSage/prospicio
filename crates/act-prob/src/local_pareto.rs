//! The log-affine local Pareto distribution: a local Pareto alpha that
//! rises linearly in the log of the amount (see `docs/design/pareto.md`).

use std::f64::consts::{LN_2, PI};

use act_core::Result;
use act_math::special::{norm_cdf, norm_pdf};

use crate::distribution::{Distribution, check_probability};
use crate::pareto::{invalid, power_integral, raw_integral};
use crate::severity::Severity;

/// Log-affine local Pareto: above the threshold `t` the local Pareto alpha
/// is `α(x) = α₀ (1 + γ ln(x/t))`, so with `L = ln(x/t)`
///
/// ```text
/// P(X > x) = exp(−α₀ L − ½ α₀ γ L²),   x ≥ t,
/// ```
///
/// and 1 below `t`. The tail thins out with size (lognormal-like) at a
/// rate set by two readable numbers: the alpha `α₀` at the threshold and
/// `δ = α₀ γ ln 2`, the rise in alpha each time the amount doubles.
/// `γ = 0` is the Pareto. Matches `pLALocPareto(x, t, alpha_0, gamma)` in
/// the R package LocalPareto.
///
/// In `L` the survival function is a Gaussian, so every layer moment
/// reduces to normal tail probabilities, evaluated through the Mills ratio
/// so that layers far in the tail keep full relative precision.
///
/// ```
/// use act_prob::{Distribution, LogAffinePareto, Severity};
///
/// // Alpha 1.5 at 1m, rising by 0.5 per doubling.
/// let d = LogAffinePareto::from_delta(1e6, 1.5, 0.5).unwrap();
/// assert!((d.local_alpha(2e6) - 2.0).abs() < 1e-14);
/// assert!(d.layer(4e6, 1e6) > 0.0 && d.mean().is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogAffinePareto {
    t: f64,
    alpha0: f64,
    gamma: f64,
}

impl LogAffinePareto {
    /// Threshold `t > 0`, alpha `α₀ > 0` at the threshold and `γ ≥ 0` (a
    /// falling alpha would turn negative far out).
    pub fn new(t: f64, alpha0: f64, gamma: f64) -> Result<Self> {
        if !t.is_finite() || t <= 0.0 {
            return Err(invalid("t", t, "must be finite and positive"));
        }
        if !alpha0.is_finite() || alpha0 <= 0.0 {
            return Err(invalid("alpha0", alpha0, "must be finite and positive"));
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(invalid("gamma", gamma, "must be finite and non-negative"));
        }
        Ok(Self { t, alpha0, gamma })
    }

    /// The same distribution from `δ = α₀ γ ln 2 ≥ 0`, the increase in
    /// the local alpha when the amount doubles.
    pub fn from_delta(t: f64, alpha0: f64, delta: f64) -> Result<Self> {
        if !alpha0.is_finite() || alpha0 <= 0.0 {
            return Err(invalid("alpha0", alpha0, "must be finite and positive"));
        }
        Self::new(t, alpha0, delta / (alpha0 * LN_2))
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    /// Local alpha at the threshold.
    pub fn alpha0(&self) -> f64 {
        self.alpha0
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// `δ = α₀ γ ln 2`.
    pub fn delta(&self) -> f64 {
        self.alpha0 * self.gamma * LN_2
    }

    /// The local Pareto alpha `−x S'(x) / S(x)` at `x ≥ t` (0 below).
    pub fn local_alpha(&self, x: f64) -> f64 {
        if x < self.t {
            return 0.0;
        }
        self.alpha0 * (1.0 + self.gamma * (x / self.t).ln())
    }

    /// `∫_a^b x^k S(x) dx` for `k ∈ {0, 1}` and `0 ≤ a ≤ b ≤ ∞`.
    fn integral(&self, k: i32, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let t = self.t;
        let below = if a < t {
            power_integral(k, a, b.min(t))
        } else {
            0.0
        };
        let lo = a.max(t);
        if lo >= b {
            return below;
        }
        let c = self.alpha0 * self.gamma;
        if c == 0.0 {
            return below + raw_integral(k, t, self.alpha0, lo, b);
        }
        // x = t e^L: t^(k+1) ∫ exp(β L − ½ c L²) dL with β = k + 1 − α₀.
        let beta = f64::from(k) + 1.0 - self.alpha0;
        let (l1, l2) = ((lo / t).ln(), (b / t).ln());
        below + t.powi(k + 1) * gaussian_integral(beta, c, l1, l2)
    }
}

/// `∫_{l1}^{l2} exp(β L − ½ c L²) dL` for `c > 0` and `l1 < l2 ≤ ∞`.
///
/// With `μ = β/c`, `σ = 1/√c`, `z = (L − μ)/σ` and `h(L)` the log of the
/// integrand, a normal tail `Q(z) = φ(z) R(z)` (`R` the Mills ratio)
/// turns `σ √(2π) e^(β²/2c) Q(z)` into `σ e^(h(L)) R(z)`, which stays
/// accurate when the integrand is tiny.
fn gaussian_integral(beta: f64, c: f64, l1: f64, l2: f64) -> f64 {
    let sigma = 1.0 / c.sqrt();
    let mu = beta / c;
    let h = |l: f64| beta * l - 0.5 * c * l * l;
    // σ e^h(L) R(|z|): the tail beyond L, away from the peak.
    let tail = |l: f64| -> f64 {
        if l.is_infinite() {
            return 0.0;
        }
        sigma * h(l).exp() * mills((l - mu).abs() / sigma)
    };
    let (z1, z2) = ((l1 - mu) / sigma, (l2 - mu) / sigma);
    if z1 >= 0.0 {
        // Both above the peak: Q(z1) − Q(z2).
        tail(l1) - tail(l2)
    } else if z2 <= 0.0 {
        // Both below the peak: Q(−z2) − Q(−z1).
        tail(l2) - tail(l1)
    } else {
        // Across the peak: 1 − Q(−z1) − Q(z2).
        sigma * (2.0 * PI).sqrt() * (0.5 * beta * mu).exp() - tail(l1) - tail(l2)
    }
}

/// The Mills ratio `R(z) = Q(z) / φ(z)` for `z ≥ 0`, by a continued
/// fraction where `φ` underflows.
fn mills(z: f64) -> f64 {
    if z < 26.0 {
        return norm_cdf(-z) / norm_pdf(z);
    }
    // R(z) = 1 / (z + 1 / (z + 2 / (z + 3 / (z + …)))).
    let mut r = 0.0;
    for n in (1..=60).rev() {
        r = f64::from(n) / (z + r);
    }
    1.0 / (z + r)
}

impl Distribution for LogAffinePareto {
    fn mean(&self) -> f64 {
        self.integral(0, 0.0, f64::INFINITY)
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        if m == f64::INFINITY {
            return f64::INFINITY;
        }
        2.0 * self.integral(1, 0.0, f64::INFINITY) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival(x)
    }

    fn survival(&self, x: f64) -> f64 {
        if x < self.t {
            return 1.0;
        }
        let l = (x / self.t).ln();
        (-self.alpha0 * l * (1.0 + 0.5 * self.gamma * l)).exp()
    }

    /// Solves `½ α₀ γ L² + α₀ L + ln s = 0` for `L ≥ 0` in the form
    /// `L = −2 ln s / (α₀ + √(α₀² − 2 α₀ γ ln s))`, which has no
    /// cancellation and covers `γ = 0`.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let log_s = (-p).ln_1p();
        let a = self.alpha0;
        let l = -2.0 * log_s / (a + (a * a - 2.0 * a * self.gamma * log_s).sqrt());
        Ok(self.t * l.exp())
    }
}

impl Severity for LogAffinePareto {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.integral(0, 0.0, limit)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.integral(0, retention, f64::INFINITY)
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.integral(0, a, a + limit)
    }

    /// `2 ∫_a^b (x − a) S(x) dx` with `b = a + limit`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        2.0 * (self.integral(1, a, b) - a * self.integral(0, a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pareto;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn gamma_zero_is_the_pareto() {
        let d = LogAffinePareto::new(1000.0, 2.5, 0.0).unwrap();
        let p = Pareto::new(1000.0, 2.5).unwrap();
        for (l, a) in [
            (4000.0, 1000.0),
            (1e5, 5e4),
            (f64::INFINITY, 2000.0),
            (500.0, 0.0),
        ] {
            assert!(close(d.layer(l, a), p.layer(l, a), 1e-14));
        }
        assert!(close(d.variance(), p.variance(), 1e-13));
        assert!(close(
            d.quantile(0.9).unwrap(),
            p.quantile(0.9).unwrap(),
            1e-14
        ));
    }

    #[test]
    fn layers_match_quadrature() {
        // Composite Simpson in L on a fine grid.
        let simpson = |d: &LogAffinePareto, k: i32, a: f64, b: f64| -> f64 {
            let (l1, l2) = ((a / d.t).ln(), (b / d.t).ln());
            let n = 20_000;
            let h = (l2 - l1) / f64::from(n);
            let f = |l: f64| {
                let x = d.t * l.exp();
                x.powi(k) * d.survival(x) * x
            };
            let mut s = f(l1) + f(l2);
            for i in 1..n {
                s += f(l1 + h * f64::from(i)) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            s * h / 3.0
        };
        for (alpha0, gamma) in [(0.5, 0.3), (1.0, 1.0), (2.0, 0.05), (0.2, 2.0), (3.0, 0.5)] {
            let d = LogAffinePareto::new(1000.0, alpha0, gamma).unwrap();
            for (a, b) in [(1000.0, 2000.0), (1500.0, 9000.0), (5e4, 2e5), (2e3, 2.1e3)] {
                let want = simpson(&d, 0, a, b);
                assert!(
                    close(d.layer(b - a, a), want, 1e-10),
                    "{alpha0} {gamma} {a} {b}"
                );
                let m2 = 2.0 * (simpson(&d, 1, a, b) - a * want);
                assert!(
                    close(d.layer_second_moment(b - a, a), m2, 1e-9),
                    "{alpha0} {gamma} {a} {b}"
                );
            }
        }
    }

    #[test]
    fn identities_and_round_trip() {
        let d = LogAffinePareto::from_delta(1000.0, 0.8, 0.6).unwrap();
        assert!(close(d.delta(), 0.6, 1e-15));
        for x in [500.0, 2000.0, 1e5] {
            assert!(close(d.lev(x) + d.stop_loss(x), d.mean(), 1e-12));
        }
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!(close(m2 - d.mean() * d.mean(), d.variance(), 1e-12));
        for p in [0.0, 0.3, 0.9, 0.999_999] {
            let x = d.quantile(p).unwrap();
            assert!(close(d.cdf(x), p, 1e-12) || p == 0.0);
        }
        // Every moment is finite once γ > 0, even with α₀ < 1.
        assert!(
            LogAffinePareto::new(1.0, 0.3, 0.1)
                .unwrap()
                .variance()
                .is_finite()
        );
        // Far in the tail (S ≈ 1e-29 at 1e8) a layer stays positive and
        // below limit · S(attachment).
        let far = d.layer(1e8, 1e8);
        assert!(far > 0.0 && far < 1e8 * d.survival(1e8), "{far}");
    }

    #[test]
    fn mills_ratio_is_continuous() {
        // R changes by about −1/z² per unit of z: 4e-14 over this step.
        let below = mills(26.0 - 1e-12);
        let above = mills(26.0);
        assert!(close(below, above, 1e-12), "{below} {above}");
        assert!(close(mills(0.0), (PI / 2.0).sqrt(), 1e-15));
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(LogAffinePareto::new(0.0, 1.0, 0.0).is_err());
        assert!(LogAffinePareto::new(1.0, 0.0, 0.0).is_err());
        assert!(LogAffinePareto::new(1.0, 1.0, -0.1).is_err());
        assert!(LogAffinePareto::from_delta(1.0, 1.0, -0.1).is_err());
    }
}
