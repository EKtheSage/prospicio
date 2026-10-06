//! Finite mixtures of severities.

use act_core::{Error, Result};
use act_math::roots::bisect;

use crate::dist::SeverityDist;
use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// A severity drawn from component `i` with probability `w_i`: attritional
/// and large losses in one model, or a book of mixed risks.
///
/// Every quantity that is linear in the distribution (distribution
/// function, survival, limited expected values, layer moments) is the
/// weighted sum of the components'; the variance follows the law of total
/// variance; quantiles bisect the distribution function.
///
/// ```
/// use act_prob::{Distribution, Lognormal, Mixture, Pareto, Severity};
///
/// let m = Mixture::new(vec![
///     (0.9, Box::new(Lognormal::from_mean_cv(1e4, 1.0).unwrap()) as Box<dyn Severity + Send + Sync>),
///     (0.1, Box::new(Pareto::new(1e5, 2.0).unwrap())),
/// ])
/// .unwrap();
/// assert!((m.mean() - (0.9 * 1e4 + 0.1 * 2e5)).abs() < 1e-6);
/// // Above 1e5 only the large losses reach: 0.1 × 2e5.
/// assert!(m.layer(f64::INFINITY, 1e5) > 0.1 * 1e5);
/// ```
pub struct Mixture {
    weights: Vec<f64>,
    components: Vec<Box<dyn Severity + Send + Sync>>,
    /// The components as [`SeverityDist`]s, when built by
    /// [`Mixture::from_dists`]: what saving the mixture needs.
    dists: Option<Vec<SeverityDist>>,
}

impl std::fmt::Debug for Mixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mixture")
            .field("weights", &self.weights)
            .field("components", &self.components.len())
            .finish()
    }
}

impl Mixture {
    /// A mixture of `(weight, component)` pairs. Weights must be positive
    /// and sum to 1 within `1e-12`.
    pub fn new(parts: Vec<(f64, Box<dyn Severity + Send + Sync>)>) -> Result<Self> {
        if parts.is_empty() {
            return Err(Error::InvalidParameter {
                name: "components",
                value: 0.0,
                reason: "must not be empty",
            });
        }
        if let Some((w, _)) = parts.iter().find(|(w, _)| !(w.is_finite() && *w > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: *w,
                reason: "must be finite and positive",
            });
        }
        let total: f64 = parts.iter().map(|(w, _)| w).sum();
        if (total - 1.0).abs() > 1e-12 {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: total,
                reason: "must sum to 1",
            });
        }
        let (weights, components) = parts.into_iter().unzip();
        Ok(Self {
            weights,
            components,
            dists: None,
        })
    }

    /// A mixture of `(weight, component)` pairs of native severities. It
    /// behaves as [`Mixture::new`], and also keeps the components as
    /// [`SeverityDist`]s, so it can be saved ([`crate::Dist::to_json`]).
    pub fn from_dists(parts: Vec<(f64, SeverityDist)>) -> Result<Self> {
        let dists: Vec<SeverityDist> = parts.iter().map(|(_, d)| d.clone()).collect();
        let mut m = Self::new(
            parts
                .into_iter()
                .map(|(w, d)| (w, Box::new(d) as Box<dyn Severity + Send + Sync>))
                .collect(),
        )?;
        m.dists = Some(dists);
        Ok(m)
    }

    /// The components as [`SeverityDist`]s, when built by
    /// [`Mixture::from_dists`].
    pub fn dists(&self) -> Option<&[SeverityDist]> {
        self.dists.as_deref()
    }

    /// Component weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn sum(&self, f: impl Fn(&(dyn Severity + Send + Sync)) -> f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(w, c)| w * f(c.as_ref()))
            .sum()
    }
}

impl Distribution for Mixture {
    fn mean(&self) -> f64 {
        self.sum(|c| c.mean())
    }

    /// `Σ w (σ² + μ²) - (Σ w μ)²`.
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.sum(|c| c.variance() + c.mean() * c.mean()) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        self.sum(|c| c.cdf(x))
    }

    fn survival(&self, x: f64) -> f64 {
        self.sum(|c| c.survival(x))
    }

    fn is_parallel_safe(&self) -> bool {
        self.components.iter().all(|c| c.is_parallel_safe())
    }

    /// The smallest `x` with `F(x) >= p`, bracketed by the components'
    /// quantiles and bisected (on the survival function above the median).
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let qs = self
            .components
            .iter()
            .map(|c| c.quantile(p))
            .collect::<Result<Vec<_>>>()?;
        let lo = qs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = qs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo == hi {
            return Ok(lo);
        }
        let below = |x: f64| {
            if p <= 0.5 {
                self.cdf(x) < p
            } else {
                self.survival(x) > 1.0 - p
            }
        };
        Ok(bisect(lo, hi, below))
    }
}

impl Severity for Mixture {
    fn lev(&self, limit: f64) -> f64 {
        self.sum(|c| c.lev(limit))
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        self.sum(|c| c.stop_loss(retention))
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.sum(|c| c.layer(limit, attachment))
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.sum(|c| c.layer_second_moment(limit, attachment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gamma, Lognormal};

    #[test]
    fn mixture_of_one_is_the_component_and_quantiles_invert() {
        let g = Gamma::new(2.0, 50.0).unwrap();
        let m = Mixture::new(vec![(1.0, Box::new(g) as Box<dyn Severity + Send + Sync>)]).unwrap();
        assert_eq!(m.variance(), g.variance());
        let two = Mixture::new(vec![
            (0.3, Box::new(g) as Box<dyn Severity + Send + Sync>),
            (0.7, Box::new(Lognormal::new(6.0, 1.5).unwrap())),
        ])
        .unwrap();
        for p in [1e-6, 0.3, 0.5, 0.95, 1.0 - 1e-9] {
            let x = two.quantile(p).unwrap();
            let err = if p <= 0.5 {
                two.cdf(x) / p - 1.0
            } else {
                two.survival(x) / (1.0 - p) - 1.0
            };
            assert!(err.abs() < 1e-9, "{p} {err}");
        }
        assert!(Mixture::new(vec![(0.5, Box::new(g) as Box<dyn Severity + Send + Sync>)]).is_err());
    }
}
