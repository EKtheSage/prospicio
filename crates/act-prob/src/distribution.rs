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

    /// `P(X > x)`. Representations with a direct form override the
    /// default `1 - cdf(x)`, which loses all precision far in the tail.
    fn survival(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

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

    /// Whether the distribution may be evaluated on several threads at
    /// once. True for every native family; false for a [`crate::Custom`]
    /// whose callbacks must stay on the calling thread (an R function), so
    /// the parallel simulations run single-threaded when they meet one.
    fn is_parallel_safe(&self) -> bool {
        true
    }
}

/// A boxed distribution (for example `Box<dyn Distribution>`) is one too,
/// so trait objects fit generic models.
impl<T: Distribution + ?Sized> Distribution for Box<T> {
    fn mean(&self) -> f64 {
        (**self).mean()
    }

    fn variance(&self) -> f64 {
        (**self).variance()
    }

    fn std_dev(&self) -> f64 {
        (**self).std_dev()
    }

    fn cdf(&self, x: f64) -> f64 {
        (**self).cdf(x)
    }

    fn survival(&self, x: f64) -> f64 {
        (**self).survival(x)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        (**self).quantile(p)
    }

    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        (**self).sample(rng, n)
    }

    fn is_parallel_safe(&self) -> bool {
        (**self).is_parallel_safe()
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
