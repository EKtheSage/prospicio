### Reserving

- `BenchmarkCurve` fits Clark's log-logistic or Weibull growth curve to a
  benchmark pattern (percents developed or factors to ultimate, with
  optional weights by age) by weighted least squares on the percents
  developed, with Clark's `max_age` and an optional average-date shift
  (`origin_width`). `BenchmarkCurveFit` gives the parameters, the fitted
  values and residuals, the percent developed and factor to ultimate at any
  age, and `pattern(ages)`, a `SelectedPattern` for the chain ladder, Mack
  and the expected-loss methods whose level at the oldest age is the
  curve's tail. Matches `scipy.optimize.least_squares` (Python
  `BenchmarkCurve`, R `benchmark_curve()`, `benchmark_selected()`; #637).
