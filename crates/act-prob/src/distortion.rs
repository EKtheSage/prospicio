//! Distortion risk measures: `ρ(X) = ∫ g(S(x)) dx` for a concave
//! distortion `g` of the survival function, with `g(0) = 0`, `g(1) = 1`.
//!
//! On a discrete distribution the integral is a weighted sum of its
//! values: the value `x_k` gets weight `g(P(X >= x_k)) - g(P(X > x_k))`.
//! The weights depend only on ranks, which is what lets capital
//! allocation reuse them on a joint distribution (see
//! `docs/design/risk.md`).

use act_core::{Error, Result};
use act_math::special::{norm_cdf, norm_quantile};

use crate::distribution::check_probability;

/// A distortion of the survival function. Every variant is concave, so
/// every measure here is coherent.
///
/// | Variant | `g(s)` | Parameter |
/// |---|---|---|
/// | `Tvar(p)` | `min(s / (1 - p), 1)` | `p` in `[0, 1]` |
/// | `Wang(λ)` | `Φ(Φ⁻¹(s) + λ)` | `λ >= 0` |
/// | `ProportionalHazard(ρ)` | `s^ρ` | `ρ` in `(0, 1]` |
/// | `DualPower(β)` | `1 - (1 - s)^β` | `β >= 1` |
///
/// Each has a parameter value that gives the mean (`Tvar(0)`, `Wang(0)`,
/// `ProportionalHazard(1)`, `DualPower(1)`).
///
/// ```
/// use act_prob::Distortion;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// // TVaR at 50%: the mean of the top half.
/// assert_eq!(Distortion::tvar(0.5).unwrap().apply_sorted(&x), 3.5);
/// // Wang with λ = 0 is the mean.
/// assert!((Distortion::wang(0.0).unwrap().apply_sorted(&x) - 2.5).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distortion {
    Tvar(f64),
    Wang(f64),
    ProportionalHazard(f64),
    DualPower(f64),
}

impl Distortion {
    /// Tail value at risk at level `p`.
    pub fn tvar(p: f64) -> Result<Self> {
        check_probability(p)?;
        Ok(Self::Tvar(p))
    }

    /// Wang transform with market price of risk `lambda >= 0`.
    pub fn wang(lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(invalid("lambda", lambda, "must be finite and non-negative"));
        }
        Ok(Self::Wang(lambda))
    }

    /// Proportional hazard transform with `rho` in `(0, 1]`.
    pub fn proportional_hazard(rho: f64) -> Result<Self> {
        if !(rho > 0.0 && rho <= 1.0) {
            return Err(invalid("rho", rho, "must be in (0, 1]"));
        }
        Ok(Self::ProportionalHazard(rho))
    }

    /// Dual power transform with `beta >= 1`.
    pub fn dual_power(beta: f64) -> Result<Self> {
        if !beta.is_finite() || beta < 1.0 {
            return Err(invalid("beta", beta, "must be finite and at least 1"));
        }
        Ok(Self::DualPower(beta))
    }

    /// The distortion `g(s)` of a survival probability `s` in `[0, 1]`.
    pub fn g(&self, s: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        if s >= 1.0 {
            return 1.0;
        }
        match *self {
            Self::Tvar(p) if p >= 1.0 => 1.0,
            Self::Tvar(p) => (s / (1.0 - p)).min(1.0),
            Self::Wang(lambda) => norm_cdf(norm_quantile(s) + lambda),
            Self::ProportionalHazard(rho) => s.powf(rho),
            Self::DualPower(beta) => -((-s).ln_1p() * beta).exp_m1(),
        }
    }

    /// Weights for `n` equally likely values sorted ascending: value `i`
    /// (0-based) gets `g((n - i) / n) - g((n - i - 1) / n)`. They are
    /// non-negative and sum to 1.
    ///
    /// ```
    /// use act_prob::Distortion;
    ///
    /// // TVaR at 50% on four values: the top two, equally.
    /// assert_eq!(Distortion::tvar(0.5).unwrap().weights(4), [0.0, 0.0, 0.5, 0.5]);
    /// ```
    pub fn weights(&self, n: usize) -> Vec<f64> {
        let nf = n as f64;
        (0..n)
            .map(|i| self.g((n - i) as f64 / nf) - self.g((n - i - 1) as f64 / nf))
            .collect()
    }

    /// The risk measure of equally likely draws sorted ascending.
    ///
    /// For `Tvar(p)` this agrees with [`tvar_sorted`](crate::risk::tvar_sorted)
    /// up to rounding, including the fractional weight at the VaR. `sorted`
    /// must be non-empty and sorted ascending; this is checked only in
    /// debug builds.
    pub fn apply_sorted(&self, sorted: &[f64]) -> f64 {
        debug_assert!(!sorted.is_empty(), "draws must not be empty");
        debug_assert!(
            sorted.windows(2).all(|w| w[0] <= w[1]),
            "draws must be sorted ascending"
        );
        self.weights(sorted.len())
            .iter()
            .zip(sorted)
            .map(|(w, x)| w * x)
            .sum()
    }

    /// The risk measure of a discrete distribution with ascending `values`
    /// and their `probs` (which should sum to 1).
    ///
    /// Survival probabilities are summed from the top, so small tail
    /// probabilities keep their precision.
    ///
    /// ```
    /// use act_prob::Distortion;
    ///
    /// let ph = Distortion::proportional_hazard(0.5).unwrap();
    /// // P(X = 0) = 0.75, P(X = 1) = 0.25: ρ = g(0.25) = 0.5.
    /// assert_eq!(ph.apply_discrete(&[0.0, 1.0], &[0.75, 0.25]), 0.5);
    /// ```
    pub fn apply_discrete(&self, values: &[f64], probs: &[f64]) -> f64 {
        debug_assert_eq!(values.len(), probs.len());
        debug_assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "values must be sorted ascending"
        );
        let mut above = 0.0; // P(X > x_k)
        let mut g_above = 0.0;
        let mut total = 0.0;
        for (x, p) in values.iter().zip(probs).rev() {
            let at_or_above = above + p;
            let g_at_or_above = self.g(at_or_above);
            total += (g_at_or_above - g_above) * x;
            above = at_or_above;
            g_above = g_at_or_above;
        }
        total
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::tvar_sorted;

    const X: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

    fn all() -> Vec<Distortion> {
        vec![
            Distortion::tvar(0.7).unwrap(),
            Distortion::wang(0.5).unwrap(),
            Distortion::proportional_hazard(0.6).unwrap(),
            Distortion::dual_power(2.5).unwrap(),
        ]
    }

    #[test]
    fn tvar_matches_the_risk_module() {
        for i in 0..=100 {
            let p = i as f64 / 100.0;
            let d = Distortion::tvar(p).unwrap().apply_sorted(&X);
            let t = tvar_sorted(&X, p).unwrap();
            assert!((d - t).abs() <= 1e-12 * t, "p = {p}: {d} vs {t}");
        }
    }

    #[test]
    fn identity_parameters_give_the_mean() {
        for d in [
            Distortion::tvar(0.0).unwrap(),
            Distortion::wang(0.0).unwrap(),
            Distortion::proportional_hazard(1.0).unwrap(),
            Distortion::dual_power(1.0).unwrap(),
        ] {
            assert!((d.apply_sorted(&X) - 30.0).abs() < 1e-12, "{d:?}");
        }
    }

    #[test]
    fn weights_are_a_probability_vector_rising_to_the_tail() {
        for d in all() {
            let w = d.weights(1000);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{d:?}");
            assert!(w.iter().all(|&w| w >= 0.0));
            // Concave g: weights never fall towards the larger values.
            assert!(w.windows(2).all(|p| p[1] >= p[0] - 1e-15), "{d:?}");
        }
    }

    #[test]
    fn coherence_properties() {
        let mean = 30.0;
        for d in all() {
            let r = d.apply_sorted(&X);
            assert!(r >= mean && r <= 50.0, "{d:?}: {r}");
            // Translation and positive scaling.
            let shifted: Vec<f64> = X.iter().map(|x| 2.0 * x + 7.0).collect();
            assert!((d.apply_sorted(&shifted) - (2.0 * r + 7.0)).abs() < 1e-12);
        }
        // A larger parameter is more conservative.
        let w = |l| Distortion::wang(l).unwrap().apply_sorted(&X);
        assert!(w(0.2) < w(0.5) && w(0.5) < w(1.0));
        let ph = |r| Distortion::proportional_hazard(r).unwrap().apply_sorted(&X);
        assert!(ph(0.9) < ph(0.5));
        let dp = |b| Distortion::dual_power(b).unwrap().apply_sorted(&X);
        assert!(dp(1.5) < dp(3.0));
    }

    #[test]
    fn discrete_agrees_with_equal_weights_and_merges_ties() {
        let p = [0.2; 5];
        for d in all() {
            let a = d.apply_discrete(&X, &p);
            let b = d.apply_sorted(&X);
            assert!((a - b).abs() < 1e-12, "{d:?}");
        }
        // Tied draws and one atom with their total mass are the same.
        let d = Distortion::dual_power(2.0).unwrap();
        let ties = d.apply_sorted(&[1.0, 5.0, 5.0, 5.0]);
        let atom = d.apply_discrete(&[1.0, 5.0], &[0.25, 0.75]);
        assert!((ties - atom).abs() < 1e-15);
        // Closed form: g(0.75) × 5 + (1 - g(0.75)) × 1.
        assert!((atom - (0.9375 * 5.0 + 0.0625)).abs() < 1e-15);
    }

    #[test]
    fn g_closed_forms() {
        let s = 0.3;
        assert_eq!(Distortion::tvar(0.8).unwrap().g(s), 1.0);
        assert!((Distortion::tvar(0.4).unwrap().g(s) - 0.5).abs() < 1e-15);
        assert!((Distortion::proportional_hazard(0.5).unwrap().g(s) - s.sqrt()).abs() < 1e-15);
        assert!((Distortion::dual_power(2.0).unwrap().g(s) - 0.51).abs() < 1e-15);
        // Φ(Φ⁻¹(0.5) + 1) = Φ(1).
        assert!((Distortion::wang(1.0).unwrap().g(0.5) - norm_cdf(1.0)).abs() < 1e-15);
        assert_eq!(Distortion::tvar(1.0).unwrap().g(1e-300), 1.0);
        for d in all() {
            assert_eq!(d.g(0.0), 0.0);
            assert_eq!(d.g(1.0), 1.0);
        }
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Distortion::tvar(1.5).is_err());
        assert!(Distortion::wang(-0.1).is_err());
        assert!(Distortion::wang(f64::INFINITY).is_err());
        assert!(Distortion::proportional_hazard(0.0).is_err());
        assert!(Distortion::proportional_hazard(1.5).is_err());
        assert!(Distortion::dual_power(0.5).is_err());
    }
}
