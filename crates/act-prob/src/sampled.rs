//! The sampled representation: a distribution known only through draws.

use act_core::{Error, Result};

use crate::distortion::Distortion;
use crate::distribution::Distribution;
use crate::risk::{tvar_sorted, var_sorted};

/// Operations that are exact on draws: any statistic of the empirical
/// distribution.
///
/// Every result is exact for the draws and an estimate of the distribution
/// they came from. Operations that are exact only on a parametric or
/// discretized distribution (limited expected value, layers) are not
/// offered here; a caller who wants an estimate from draws asks for it
/// with [`Empirical::mean_of`].
pub trait Empirical: Distribution {
    /// The draws, in the order they were simulated. Draw `i` came from
    /// simulation `i`, so two distributions from the same simulations can
    /// be paired draw by draw.
    fn draws(&self) -> &[f64];

    /// The draws sorted ascending.
    fn sorted(&self) -> &[f64];

    /// Mean of `f(x)` over the draws.
    ///
    /// ```
    /// use act_prob::{Empirical, Sampled};
    ///
    /// let s = Sampled::new(vec![50.0, 150.0, 400.0]).unwrap();
    /// // Estimated limited expected value at 100.
    /// assert_eq!(s.mean_of(|x| x.min(100.0)), 250.0 / 3.0);
    /// ```
    fn mean_of(&self, f: impl Fn(f64) -> f64) -> f64 {
        let draws = self.draws();
        draws.iter().map(|&x| f(x)).sum::<f64>() / draws.len() as f64
    }

    /// Value at risk at level `p`; see [`var_sorted`].
    fn var(&self, p: f64) -> Result<f64> {
        var_sorted(self.sorted(), p)
    }

    /// Tail value at risk at level `p`; see [`tvar_sorted`].
    fn tvar(&self, p: f64) -> Result<f64> {
        tvar_sorted(self.sorted(), p)
    }

    /// Distortion risk measure of the draws; see
    /// [`Distortion::apply_sorted`].
    ///
    /// ```
    /// use act_prob::{Distortion, Empirical, Sampled};
    ///
    /// let s = Sampled::new(vec![3.0, 1.0, 4.0, 2.0]).unwrap();
    /// let tvar = Distortion::tvar(0.5).unwrap();
    /// assert_eq!(s.distortion(&tvar), s.tvar(0.5).unwrap());
    /// ```
    fn distortion(&self, d: &Distortion) -> f64 {
        d.apply_sorted(self.sorted())
    }
}

/// A distribution represented by `n` equally weighted draws.
///
/// Every [`Distribution`] method describes the empirical distribution of
/// the draws: `variance` divides by `n`, not `n - 1`, and `quantile` is
/// the inverse of the empirical distribution function (R `type = 1`).
/// `sample` resamples the draws with replacement.
///
/// # Example
///
/// ```
/// use act_core::StreamRng;
/// use act_prob::{Distribution, Empirical, Lognormal, Sampled};
///
/// let d = Lognormal::from_mean_cv(1000.0, 0.5).unwrap();
/// let s = Sampled::new(d.sample(&mut StreamRng::new(42, 0), 10_000)).unwrap();
/// assert!((s.mean() - 1000.0).abs() < 20.0);
/// assert!(s.tvar(0.99).unwrap() > s.var(0.99).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    draws: Vec<f64>,
    sorted: Vec<f64>,
    mean: f64,
    variance: f64,
}

impl Sampled {
    /// The empirical distribution of `draws`.
    ///
    /// Fails if `draws` is empty or holds a value that is not finite.
    pub fn new(draws: Vec<f64>) -> Result<Self> {
        if draws.is_empty() {
            return Err(Error::InvalidParameter {
                name: "draws",
                value: 0.0,
                reason: "must not be empty",
            });
        }
        if let Some(&bad) = draws.iter().find(|x| !x.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "draws",
                value: bad,
                reason: "must all be finite",
            });
        }
        let mut sorted = draws.clone();
        sorted.sort_by(f64::total_cmp);
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        // Two passes: the centred sum loses no precision to a large mean.
        let variance = draws.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        Ok(Self {
            draws,
            sorted,
            mean,
            variance,
        })
    }

    /// Number of draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Always `false`: a `Sampled` holds at least one draw.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The draws, giving up ownership.
    pub fn into_draws(self) -> Vec<f64> {
        self.draws
    }
}

impl Distribution for Sampled {
    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        self.variance
    }

    fn cdf(&self, x: f64) -> f64 {
        self.sorted.partition_point(|&d| d <= x) as f64 / self.sorted.len() as f64
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        var_sorted(&self.sorted, p)
    }
}

impl Empirical for Sampled {
    fn draws(&self) -> &[f64] {
        &self.draws
    }

    fn sorted(&self) -> &[f64] {
        &self.sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_core::StreamRng;

    fn sampled(x: &[f64]) -> Sampled {
        Sampled::new(x.to_vec()).unwrap()
    }

    #[test]
    fn rejects_empty_and_non_finite() {
        assert!(Sampled::new(vec![]).is_err());
        assert!(Sampled::new(vec![1.0, f64::NAN]).is_err());
        assert!(Sampled::new(vec![f64::INFINITY]).is_err());
    }

    #[test]
    fn keeps_simulation_order() {
        let s = sampled(&[3.0, 1.0, 2.0]);
        assert_eq!(s.draws(), [3.0, 1.0, 2.0]);
        assert_eq!(s.sorted(), [1.0, 2.0, 3.0]);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn moments_are_those_of_the_draws() {
        let s = sampled(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!(s.mean(), 5.0);
        assert_eq!(s.variance(), 4.0);
        assert_eq!(s.std_dev(), 2.0);
    }

    #[test]
    fn variance_survives_a_large_mean() {
        let s = sampled(&[1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0]);
        assert!((s.variance() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn cdf_and_quantile_are_inverse() {
        let s = sampled(&[40.0, 10.0, 30.0, 20.0]);
        assert_eq!(s.cdf(5.0), 0.0);
        assert_eq!(s.cdf(10.0), 0.25);
        assert_eq!(s.cdf(25.0), 0.5);
        assert_eq!(s.cdf(40.0), 1.0);
        for x in [10.0, 20.0, 30.0, 40.0] {
            assert_eq!(s.quantile(s.cdf(x)), Ok(x));
        }
        assert!(s.quantile(1.5).is_err());
    }

    #[test]
    fn sample_resamples_the_draws() {
        let s = sampled(&[1.0, 2.0, 3.0]);
        let r = s.sample(&mut StreamRng::new(9, 0), 1000);
        assert!(r.iter().all(|x| [1.0, 2.0, 3.0].contains(x)));
        assert_eq!(r, s.sample(&mut StreamRng::new(9, 0), 1000));
    }

    #[test]
    fn lognormal_tail_measures_converge() {
        use crate::Lognormal;
        use act_math::special::norm_cdf;

        let d = Lognormal::new(0.0, 0.5).unwrap();
        let s = Sampled::new(d.sample(&mut StreamRng::new(1, 0), 400_000)).unwrap();
        let p = 0.99;
        let q = d.quantile(p).unwrap();
        // Lognormal TVaR: E[X] * Phi(sdlog - z_p) / (1 - p).
        let z = (q.ln() - d.meanlog()) / d.sdlog();
        let tvar = d.mean() * norm_cdf(d.sdlog() - z) / (1.0 - p);
        let var_err = (s.var(p).unwrap() - q).abs() / q;
        let tvar_err = (s.tvar(p).unwrap() - tvar).abs() / tvar;
        assert!(var_err < 0.01, "VaR relative error {var_err}");
        assert!(tvar_err < 0.01, "TVaR relative error {tvar_err}");
    }
}
