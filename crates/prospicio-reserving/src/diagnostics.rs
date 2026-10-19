//! MCMC diagnostics of a fit, kept outside the `bayes` feature so that the
//! diagonal backtest can carry them.

/// MCMC diagnostics of a fit, with a warning for each threshold of
/// Vehtari et al. (2021) that fails: R-hat above 1.01, bulk or tail ESS
/// below 100 per chain, any divergent transition. A fit with warnings is
/// still returned; check them before relying on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// The largest R-hat over the parameters; NaN when any parameter's is
    /// NaN (a constant draw).
    pub max_rhat: f64,
    /// The smallest bulk ESS over the parameters; NaN when any is NaN.
    pub min_ess_bulk: f64,
    /// The smallest tail ESS over the parameters; NaN when any is NaN.
    pub min_ess_tail: f64,
    /// Divergent transitions among the kept draws, over all chains.
    pub divergences: usize,
    /// One message per failed threshold, naming the parameter, its value
    /// and the threshold; empty when every threshold holds.
    pub warnings: Vec<String>,
}
