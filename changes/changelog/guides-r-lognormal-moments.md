### Distributions and probability

- R: `survival()`, `layer_variance()` and `layer_second_moment()` on
  `lognormal()`; a new generic `layer_second_moment(dist, limit, attachment)`
  on every family with `layer_variance()` (the Pareto family, gamma, Tweedie,
  Weibull, log-logistic, inverse gamma, inverse Gaussian, Burr, beta,
  truncated, policy limits, mixture, custom); `std_dev()`, the standard
  deviation of anything with a `variance()` (#327, #270).
- Python: `Lognormal.layer_second_moment(limit, attachment)` (#327).

### Aggregate and reinsurance

- R: `moment_approximation()` also takes a `compound_moments()` result
  alone, `moment_approximation(m, "gamma")`; calls with a count and a
  severity are unchanged (#327).
