### Models
- `prospicio_bayes::elpd::psis` (and so `loo` and every Bayesian one-year
  view) no longer returns a NaN Pareto k. A tail it cannot fit, because a
  quarter or more of its exceedances are 0 (log ratios so far below the
  largest that `exp` underflows, or ties at the cutoff), now gets an
  infinite k and the raw weights, as a tail too short to fit already did.
  Before, the NaN k also set every tail weight to the largest one. In the
  one-year views those simulations are now moment-matched before they are
  refitted (#782, #814).
