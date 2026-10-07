//! The interface every model implements, whatever its engine.

use prospicio_core::Result;
use prospicio_prob::PredictiveDistribution;

use crate::design::Design;

/// A model specification that can be fitted: a GLM, a GAM, a network.
///
/// `fit` takes the design (with offset and weights) and the response, and
/// returns the fitted model. Specs are cheap values; fitting does the work.
pub trait Model {
    type Fitted: Fitted;

    /// Fits the model to `design` and response `y`.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<Self::Fitted>;
}

/// A fitted model: predictions, joint predictive distributions, scores.
pub trait Fitted {
    /// Expected response for each row of `design`.
    fn predict(&self, design: &Design) -> Result<Vec<f64>>;

    /// Joint predictive distribution of the response across the rows of
    /// `design`: `n_sims` simulations × rows, each simulation drawn from
    /// stream `sim` of `seed`, so results do not depend on thread count.
    /// Draws carry parameter and process uncertainty, and are joint because
    /// anything summed later (cells into reserves, policies into a
    /// portfolio) needs the joint draws, not per-row marginals.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution>;
}
