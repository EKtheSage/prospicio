### Bindings

- Three R functions now call the Rust core instead of recomputing it
  (#838, #864): `limited_fluctuation_credibility()` gives 0 for a `NaN` count,
  as Python does (it gave `NaN`); `worst_var()` takes its quantiles from
  the core, with its checks on `p` (in `(0, 1)`) and `n` (positive); and
  `predict()` on a `hachemeister_fit` is the core's prediction.
  `random_search(seed = )` no longer changes the caller's random state.
  `partial_credibility()` in `prospicio_pricing::credibility` is the square
  root rule on its own.
