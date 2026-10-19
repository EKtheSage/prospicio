### Reserving

- **Breaking:** `CellForecast` has a new field `diagnostics:
  Option<Diagnostics>`, the sampler diagnostics of the fit behind the
  forecast (`None` for a model with no sampler); code that builds a
  `CellForecast` in full must set it. `Diagnostics` moves to
  `prospicio_reserving::Diagnostics` (outside the `bayes` feature), and
  `prospicio_reserving::bayes::Diagnostics` stays as a re-export.
- **Breaking:** `METRICS` grows from four to six names, `mean_log_density`
  and `total_percentile` after the old four, so code that sizes arrays by
  its length changes. `diagonal_backtest` scores both.
- `Backtest::diagnostics(model)`, `Backtest::warnings()` and
  `Backtest::diagonals()`: every split's diagnostics, the warnings as
  `(model, split, message)` and each split's held-out calendar period. A
  Bayesian fit with warnings is still scored.
- **Breaking:** `diagonal_backtest` returns an error naming two models
  with the same name, before any fit (it used to accept them);
  `NamedModel::new(name, model)` renames one. (#420)
- The diagonal backtest in Python and R (#342, #420). Python:
  `prospicio.reserving.diagonal_backtest(triangle, column, models,
  n_diagonals, premium=...)` with `BayesOdp`, `Meyers` or a new
  `GlmCandidate` (`GlmCandidate.odp()` for the ODP model), returning a
  `Backtest` with `scores()`, `mean()`, `diagnostics()`, `warnings`,
  `diagonals` and `log_densities`. R: `diagonal_backtest()` with the new
  settings objects `odp_glm_spec()`, `bayes_odp_spec()` and
  `meyers_spec()`, and the same tables as data frames. Both report each
  split's sampler diagnostics and the two new metrics.
