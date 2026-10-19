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
- `diagonal_backtest` returns an error naming two models with the same
  name, before any fit, and `NamedModel::new(name, model)` renames one.
