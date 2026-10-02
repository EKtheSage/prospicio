//! The single-parameter (European) Pareto distribution, optionally
//! truncated: the standard large-loss severity in reinsurance pricing (see
//! `docs/design/pareto.md`).

use act_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Pareto distribution `Pareto(t, α)`: `P(X > x) = (t/x)^α` for `x ≥ t`,
/// and 1 below `t`. Also known as Pareto type I or single-parameter
/// Pareto; matches the R package Pareto (`pPareto(x, t, alpha)`).
///
/// With a `truncation` `T > t` it is the Pareto conditioned on `X < T`:
/// `P(X > x) = ((t/x)^α − (t/T)^α) / (1 − (t/T)^α)` for `t ≤ x < T`.
///
/// Layer integrals are in closed form for every `α > 0`, including the
/// logarithmic cases `α = 1` (limited mean) and `α = 2` (limited second
/// moment), which need no special handling: integrals of `x^{k−α}` are
/// evaluated as `a^{k+1−α} · L · exprel((k + 1 − α) L)` with
/// `L = ln(b/a)` and `exprel(z) = (e^z − 1)/z`.
///
/// ```
/// use act_prob::{Distribution, Pareto, Severity};
///
/// let p = Pareto::new(500.0, 2.0).unwrap();
/// // The R package's vignette example: layer 4000 xs 1000.
/// assert!((p.layer(4000.0, 1000.0) - 200.0).abs() < 1e-12);
/// assert!((p.mean() - 1000.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pareto {
    t: f64,
    alpha: f64,
    truncation: Option<f64>,
}

impl Pareto {
    /// Pareto with threshold `t > 0` and alpha `α > 0`.
    pub fn new(t: f64, alpha: f64) -> Result<Self> {
        if !t.is_finite() || t <= 0.0 {
            return Err(invalid("t", t, "must be finite and positive"));
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(invalid("alpha", alpha, "must be finite and positive"));
        }
        Ok(Self {
            t,
            alpha,
            truncation: None,
        })
    }

    /// The same Pareto conditioned on `X < truncation`, with
    /// `t < truncation < ∞`.
    pub fn truncated(self, truncation: f64) -> Result<Self> {
        if !truncation.is_finite() || truncation <= self.t {
            return Err(invalid(
                "truncation",
                truncation,
                "must be finite and above t",
            ));
        }
        Ok(Self {
            truncation: Some(truncation),
            ..self
        })
    }

    /// Threshold `t`.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// Pareto alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Truncation point, if any.
    pub fn truncation(&self) -> Option<f64> {
        self.truncation
    }

    /// `P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        if x < self.t {
            return 1.0;
        }
        let raw = (self.t / x).powf(self.alpha);
        match self.truncation {
            None => raw,
            Some(tr) if x >= tr => 0.0,
            Some(tr) => {
                // (t/x)^α − (t/T)^α = (t/x)^α (1 − (x/T)^α), without cancellation.
                let (_, one_minus_q) = self.truncated_mass(tr);
                raw * -(self.alpha * (x / tr).ln()).exp_m1() / one_minus_q
            }
        }
    }

    /// `(q, 1 − q)` with `q = (t/T)^α`, `1 − q` without cancellation.
    fn truncated_mass(&self, tr: f64) -> (f64, f64) {
        let log_q = self.alpha * (self.t / tr).ln();
        (log_q.exp(), -log_q.exp_m1())
    }

    /// `∫_a^b x^k S(x) dx` for `k ∈ {0, 1}` and `0 ≤ a ≤ b ≤ ∞`, with `S`
    /// the survival function (truncated if set).
    fn integral(&self, k: i32, a: f64, b: f64) -> f64 {
        match self.truncation {
            None => self.raw_integral(k, a, b),
            Some(tr) => {
                let (a, b) = (a.min(tr), b.min(tr));
                if a >= b {
                    return 0.0;
                }
                let (q, one_minus_q) = self.truncated_mass(tr);
                (self.raw_integral(k, a, b) - q * power_integral(k, a, b)) / one_minus_q
            }
        }
    }

    /// `∫_a^b x^k S(x) dx` for the untruncated Pareto.
    fn raw_integral(&self, k: i32, a: f64, b: f64) -> f64 {
        raw_integral(k, self.t, self.alpha, a, b)
    }
}

/// `∫_a^b x^k S(x) dx` for `S` the untruncated `Pareto(t, α)` survival
/// function, `k ∈ {0, 1}`, `0 ≤ a ≤ b ≤ ∞` and `α ≥ 0`.
pub(crate) fn raw_integral(k: i32, t: f64, alpha: f64, a: f64, b: f64) -> f64 {
    if a >= b {
        return 0.0;
    }
    let below = if a < t {
        power_integral(k, a, b.min(t))
    } else {
        0.0
    };
    let lo = a.max(t);
    if lo >= b {
        return below;
    }
    // t^α ∫_lo^b x^(k−α) dx = (t/lo)^α lo^(k+1) ∫_1^(b/lo) u^(k−α) du.
    let e = f64::from(k) + 1.0 - alpha;
    let scale = (t / lo).powf(alpha) * lo.powi(k + 1);
    let above = if b == f64::INFINITY {
        if e < 0.0 { scale / -e } else { f64::INFINITY }
    } else {
        let l = (b / lo).ln();
        scale * l * exprel(e * l)
    };
    below + above
}

impl Distribution for Pareto {
    /// `α t / (α − 1)` untruncated; infinite for `α ≤ 1`.
    fn mean(&self) -> f64 {
        match self.truncation {
            None if self.alpha > 1.0 => self.alpha * self.t / (self.alpha - 1.0),
            None => f64::INFINITY,
            Some(_) => self.integral(0, 0.0, f64::INFINITY),
        }
    }

    /// `α t² / ((α − 1)² (α − 2))` untruncated; infinite for `α ≤ 2`.
    fn variance(&self) -> f64 {
        match self.truncation {
            None if self.alpha > 2.0 => {
                let a = self.alpha;
                a * self.t * self.t / ((a - 1.0) * (a - 1.0) * (a - 2.0))
            }
            None => f64::INFINITY,
            Some(_) => {
                let m = self.mean();
                2.0 * self.integral(1, 0.0, f64::INFINITY) - m * m
            }
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival(x)
    }

    /// `t (1 − p)^(−1/α)` untruncated; `p = 1` gives `+∞` (or the
    /// truncation point).
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let s = match self.truncation {
            None => 1.0 - p,
            Some(tr) => {
                let (q, one_minus_q) = self.truncated_mass(tr);
                q + (1.0 - p) * one_minus_q
            }
        };
        if s <= 0.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.t * s.powf(-1.0 / self.alpha))
    }
}

impl Severity for Pareto {
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

/// `∫_a^b x^k dx` for `k ∈ {0, 1}`.
pub(crate) fn power_integral(k: i32, a: f64, b: f64) -> f64 {
    match k {
        0 => b - a,
        _ => 0.5 * (b - a) * (b + a),
    }
}

/// `(e^z − 1) / z`, and 1 at `z = 0`.
fn exprel(z: f64) -> f64 {
    if z == 0.0 { 1.0 } else { z.exp_m1() / z }
}

pub(crate) fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn layer_means_match_the_closed_forms() {
        // I(a, b) = t^α (b^(1−α) − a^(1−α)) / (1 − α), and t ln(b/a) at α = 1.
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0] {
            let p = Pareto::new(500.0, alpha).unwrap();
            let (a, b) = (1000.0f64, 5000.0f64);
            let want = if alpha == 1.0 {
                500.0 * (b / a).ln()
            } else {
                500f64.powf(alpha) * (b.powf(1.0 - alpha) - a.powf(1.0 - alpha)) / (1.0 - alpha)
            };
            assert!(close(p.layer(4000.0, 1000.0), want, 1e-13), "alpha {alpha}");
        }
        // Below the threshold every loss pays in full.
        let p = Pareto::new(500.0, 2.0).unwrap();
        assert_eq!(p.layer(300.0, 100.0), 300.0);
        assert!(close(
            p.lev(1000.0),
            500.0 + 500.0 * 500.0 * (1.0 / 500.0 - 1.0 / 1000.0),
            1e-14
        ));
    }

    #[test]
    fn severity_identities() {
        for p in [
            Pareto::new(1000.0, 2.5).unwrap(),
            Pareto::new(1000.0, 1.5)
                .unwrap()
                .truncated(20_000.0)
                .unwrap(),
            Pareto::new(1000.0, 0.8).unwrap().truncated(1e6).unwrap(),
        ] {
            for d in [500.0, 1000.0, 3000.0, 15_000.0] {
                assert!(
                    close(p.lev(d) + p.stop_loss(d), p.mean(), 1e-12),
                    "{p:?} at {d}"
                );
                let layer = p.layer(4000.0, d);
                assert!(close(
                    layer,
                    p.stop_loss(d) - p.stop_loss(d + 4000.0),
                    1e-10
                ));
            }
            // Second moment of the whole loss gives the variance.
            let m2 = p.layer_second_moment(f64::INFINITY, 0.0);
            assert!(
                close(m2 - p.mean() * p.mean(), p.variance(), 1e-10),
                "{p:?}"
            );
            assert!(close(
                p.layer_variance(f64::INFINITY, 0.0),
                p.variance(),
                1e-10
            ));
        }
    }

    #[test]
    fn untruncated_moments_and_infinite_cases() {
        let p = Pareto::new(100.0, 3.0).unwrap();
        assert!(close(p.mean(), 150.0, 1e-15));
        // E[X^2] = α t^2 / (α − 2) = 30,000; variance 30,000 − 150^2.
        assert!(close(p.variance(), 7500.0, 1e-15));
        assert!(close(
            p.layer_second_moment(f64::INFINITY, 0.0),
            3.0e4,
            1e-13
        ));
        let heavy = Pareto::new(100.0, 1.0).unwrap();
        assert_eq!(heavy.mean(), f64::INFINITY);
        assert_eq!(heavy.stop_loss(1000.0), f64::INFINITY);
        assert!(heavy.layer(1000.0, 1000.0).is_finite());
        assert_eq!(Pareto::new(100.0, 2.0).unwrap().variance(), f64::INFINITY);
        // α = 2: the limited second moment has a logarithm.
        let p2 = Pareto::new(1.0, 2.0).unwrap();
        // 2 ∫_1^e x · x^-2 dx + 2 ∫_0^1 x dx = 2 + 1.
        assert!(close(
            p2.layer_second_moment(std::f64::consts::E, 0.0),
            3.0,
            1e-14
        ));
    }

    #[test]
    fn cdf_quantile_round_trip() {
        for p in [
            Pareto::new(1000.0, 2.0).unwrap(),
            Pareto::new(1000.0, 0.7)
                .unwrap()
                .truncated(50_000.0)
                .unwrap(),
        ] {
            for q in [0.0, 0.1, 0.5, 0.9, 0.999] {
                let x = p.quantile(q).unwrap();
                assert!(
                    close(p.cdf(x), q, 1e-12) || (q == 0.0 && p.cdf(x) == 0.0),
                    "{p:?} {q}"
                );
            }
            assert_eq!(p.cdf(999.0), 0.0);
        }
        assert_eq!(
            Pareto::new(1.0, 2.0).unwrap().quantile(1.0).unwrap(),
            f64::INFINITY
        );
        let t = Pareto::new(1.0, 2.0).unwrap().truncated(10.0).unwrap();
        assert!(close(t.quantile(1.0).unwrap(), 10.0, 1e-15));
        assert_eq!(t.survival(10.0), 0.0);
    }

    #[test]
    fn large_alpha_does_not_overflow() {
        let p = Pareto::new(1e6, 80.0).unwrap();
        let l = p.layer(1e6, 2e6);
        assert!(l.is_finite() && l > 0.0 && l < 1e-15);
        assert!(close(p.mean(), 80.0 * 1e6 / 79.0, 1e-15));
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Pareto::new(0.0, 1.0).is_err());
        assert!(Pareto::new(1.0, 0.0).is_err());
        assert!(Pareto::new(1.0, f64::INFINITY).is_err());
        assert!(Pareto::new(10.0, 1.0).unwrap().truncated(5.0).is_err());
    }
}
