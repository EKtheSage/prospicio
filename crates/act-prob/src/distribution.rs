//! The trait shared by every distribution representation.

use act_core::{Result, StreamRng};

/// A univariate loss distribution.
///
/// Only operations that are exact for every representation live here; see
/// `docs/design/distributions.md` for the representation-specific traits
/// that will join it.
pub trait Distribution {
    /// Expected value.
    fn mean(&self) -> f64;

    /// Variance.
    fn variance(&self) -> f64;

    /// Standard deviation.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `P(X <= x)`.
    fn cdf(&self, x: f64) -> f64;

    /// Smallest `x` with `cdf(x) >= p`.
    ///
    /// Fails with [`act_core::Error::InvalidProbability`] unless `p` is in
    /// `[0, 1]`.
    fn quantile(&self, p: f64) -> Result<f64>;

    /// `n` draws from stream `rng`, by inverse transform.
    ///
    /// Inverse transform keeps draws a pure function of `(seed, stream)` and
    /// preserves ordering under common random numbers.
    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                self.quantile(rng.next_open01())
                    .expect("next_open01 is always in (0, 1)")
            })
            .collect()
    }
}

/// Checks that `p` is a probability.
pub(crate) fn check_probability(p: f64) -> Result<()> {
    if (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(act_core::Error::InvalidProbability(p))
    }
}
