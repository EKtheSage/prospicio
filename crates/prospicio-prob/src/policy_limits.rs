//! A severity capped by a mix of policy limits.

use std::sync::Arc;

use prospicio_core::{Error, Result};

use crate::dist::{Dist, SeverityDist};
use crate::distribution::{Distribution, bisect_quantile, check_probability};
use crate::severity::Severity;

/// The loss `min(X, L)` when each policy has its own limit `L`: a
/// cedant's losses censored at many limits, as a severity. `L` is
/// `limits[j]` with probability `weights[j]` (the share of losses under
/// that limit, from a limits profile), independent of `X`; a limit of
/// infinity is an uncapped policy.
///
/// Each quantity is the weighted sum of the base severity's limited
/// moments, so it is exact: `E[min(X, L, d)] = Σ w_j lev(min(d, L_j))`,
/// and a layer `l` xs `a` takes `Σ_{L_j > a} w_j layer(min(l, L_j − a), a)`.
/// The distribution function has an atom of `w_j S(L_j)` at each finite
/// limit, so a discretization ([`crate::Grid::rounding`]) puts the capped
/// losses' mass at the limit.
///
/// ```
/// use prospicio_prob::{Dist, Distribution, Pareto, PolicyLimits, Severity, SeverityDist};
///
/// let x = SeverityDist::try_from(Dist::from(Pareto::new(100.0, 2.0).unwrap())).unwrap();
/// let capped = PolicyLimits::new(x, vec![1000.0, 5000.0], vec![3.0, 1.0]).unwrap();
/// // E[min(X, L)] = t (2 − t/L) for alpha 2.
/// let want = 0.75 * 100.0 * (2.0 - 0.1) + 0.25 * 100.0 * (2.0 - 0.02);
/// assert!((capped.mean() - want).abs() < 1e-9);
/// // The atom at 1000: 3/4 of the losses that reach it.
/// assert!((capped.cdf(1000.0) - capped.cdf(999.999) - 0.75 * 0.01).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct PolicyLimits {
    base: SeverityDist,
    limits: Vec<f64>,
    /// Normalized to sum to 1.
    weights: Vec<f64>,
}

impl PolicyLimits {
    /// `base` capped at `limits[j]` with weight `weights[j]`: limits
    /// positive (infinity for no limit), weights non-negative with a
    /// positive sum, normalized to 1 (counts of policies or of losses do).
    pub fn new(base: SeverityDist, limits: Vec<f64>, weights: Vec<f64>) -> Result<Self> {
        if limits.is_empty() || limits.len() != weights.len() {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: weights.len() as f64,
                reason: "must have one entry per limit, and at least one",
            });
        }
        for &l in &limits {
            if l.is_nan() || l <= 0.0 {
                return Err(Error::InvalidParameter {
                    name: "limits",
                    value: l,
                    reason: "must be positive (infinity for no limit)",
                });
            }
        }
        for &w in &weights {
            if !(w.is_finite() && w >= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "weights",
                    value: w,
                    reason: "must be finite and non-negative",
                });
            }
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: total,
                reason: "must have a positive sum",
            });
        }
        let weights = weights.iter().map(|w| w / total).collect();
        Ok(Self {
            base,
            limits,
            weights,
        })
    }

    /// The severity before the limits.
    pub fn base(&self) -> &SeverityDist {
        &self.base
    }

    /// The policy limits.
    pub fn limits(&self) -> &[f64] {
        &self.limits
    }

    /// Each limit's weight, summing to 1.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `(w_j, L_j)` for each limit.
    fn each(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.weights
            .iter()
            .copied()
            .zip(self.limits.iter().copied())
    }
}

impl From<PolicyLimits> for Dist {
    fn from(p: PolicyLimits) -> Self {
        Self::PolicyLimits(Arc::new(p))
    }
}

impl Distribution for PolicyLimits {
    fn mean(&self) -> f64 {
        self.each().map(|(w, l)| w * self.base.lev(l)).sum()
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        self.layer_second_moment(f64::INFINITY, 0.0) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let f = self.base.cdf(x);
        self.each()
            .map(|(w, l)| if x >= l { w } else { w * f })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    fn survival(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 1.0;
        }
        let s = self.base.survival(x);
        self.each()
            .filter(|&(_, l)| l > x)
            .map(|(w, _)| w * s)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let top = self.limits.iter().copied().fold(0.0, f64::max);
        if p == 0.0 {
            return self.base.quantile(0.0);
        }
        if p == 1.0 && top.is_finite() {
            return Ok(top);
        }
        let start = if top.is_finite() {
            top
        } else {
            self.base.quantile(0.5)?.max(1e-300)
        };
        let x = bisect_quantile(p, 0.0, start, top, |x| self.cdf(x), |x| self.survival(x));
        // In an atom's range the answer is the limit itself; bisection
        // lands within its tolerance above it.
        Ok(self
            .limits
            .iter()
            .copied()
            .filter(|&l| l.is_finite() && l <= x && self.cdf(l) >= p)
            .fold(x, f64::min))
    }

    fn is_parallel_safe(&self) -> bool {
        self.base.is_parallel_safe()
    }
}

impl Severity for PolicyLimits {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.each()
            .map(|(w, l)| w * self.base.lev(limit.min(l)))
            .sum()
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.layer(f64::INFINITY, retention)
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.each()
            .filter(|&(_, l)| l > a)
            .map(|(w, l)| w * self.base.layer(limit.min(l - a), a))
            .sum()
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.each()
            .filter(|&(_, l)| l > a)
            .map(|(w, l)| w * self.base.layer_second_moment(limit.min(l - a), a))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gamma, Lognormal, Pareto};

    fn sev(d: impl Into<Dist>) -> SeverityDist {
        SeverityDist::try_from(d.into()).unwrap()
    }

    /// `∫ S` and `2 ∫ (x - a) S` over a layer by the midpoint rule.
    fn numeric(p: &PolicyLimits, a: f64, b: f64) -> (f64, f64) {
        let n = 200_000;
        let h = (b - a) / f64::from(n);
        let (mut m1, mut m2) = (0.0, 0.0);
        for i in 0..n {
            let x = a + (f64::from(i) + 0.5) * h;
            let s = p.survival(x);
            m1 += s * h;
            m2 += 2.0 * (x - a) * s * h;
        }
        (m1, m2)
    }

    #[test]
    fn layers_match_integration() {
        let g = sev(Gamma::new(2.0, 500.0).unwrap());
        let p =
            PolicyLimits::new(g, vec![800.0, 2000.0, f64::INFINITY], vec![2.0, 1.0, 1.0]).unwrap();
        for (a, b) in [
            (0.0, 300.0),
            (200.0, 1200.0),
            (1000.0, 2500.0),
            (0.0, 8000.0),
        ] {
            let (m1, m2) = numeric(&p, a, b);
            let got1 = p.layer(b - a, a);
            let got2 = p.layer_second_moment(b - a, a);
            assert!(
                (got1 - m1).abs() < 1e-6 * m1.max(1.0),
                "{a} {b}: {got1} {m1}"
            );
            // The jump at a limit inside a midpoint cell costs about 1e-6.
            assert!(
                (got2 - m2).abs() < 1e-5 * m2.max(1.0),
                "{a} {b}: {got2} {m2}"
            );
        }
        assert!((p.lev(1500.0) + p.stop_loss(1500.0) - p.mean()).abs() < 1e-9);
        assert!((p.lev(f64::INFINITY) - p.mean()).abs() < 1e-9);
    }

    #[test]
    fn quantiles_find_the_atoms() {
        let ln = sev(Lognormal::from_mean_cv(1000.0, 2.0).unwrap());
        let p = PolicyLimits::new(ln, vec![2000.0, 5000.0], vec![0.5, 0.5]).unwrap();
        // Inside the atom at 2000: (F(2000⁻), F(2000)].
        let (lo, hi) = (p.cdf(1_999.999_999), p.cdf(2000.0));
        assert!(hi - lo > 0.01);
        assert_eq!(p.quantile(0.5 * (lo + hi)).unwrap(), 2000.0);
        assert_eq!(p.quantile(hi).unwrap(), 2000.0);
        assert_eq!(p.quantile(1.0).unwrap(), 5000.0);
        for q in [0.1, 0.5, 0.97] {
            let x = p.quantile(q).unwrap();
            assert!(p.cdf(x) >= q - 1e-12, "{q}");
        }
        // One infinite limit is the base severity itself.
        let base = sev(Pareto::new(100.0, 2.5).unwrap());
        let open = PolicyLimits::new(base.clone(), vec![f64::INFINITY], vec![1.0]).unwrap();
        assert!((open.mean() - base.mean()).abs() < 1e-12);
        assert!((open.quantile(0.9).unwrap() / base.quantile(0.9).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_profiles() {
        let g = sev(Gamma::new(2.0, 500.0).unwrap());
        assert!(PolicyLimits::new(g.clone(), vec![], vec![]).is_err());
        assert!(PolicyLimits::new(g.clone(), vec![100.0], vec![1.0, 2.0]).is_err());
        assert!(PolicyLimits::new(g.clone(), vec![0.0], vec![1.0]).is_err());
        assert!(PolicyLimits::new(g.clone(), vec![100.0], vec![-1.0]).is_err());
        assert!(PolicyLimits::new(g, vec![100.0, 200.0], vec![0.0, 0.0]).is_err());
    }
}
