### Distributions and probability

- The loglogistic and exponential in `fit_severity`, and the single
  Pareto and Riegel's generalized Pareto fitted from a threshold `t`
  through `fit_severity_with_threshold`, scored on the same likelihood as
  the body families so that all nine rank by AIC (`t` is not counted); an
  `Exponential` distribution with exact layer moments, saved as JSON; the
  log-likelihood, AIC and BIC of any Pareto, piecewise Pareto or
  generalized Pareto fit (`fit_statistics`); Python
  `fit_severity(..., t=)`, `SeverityFit.threshold` and `Exponential`, R
  `severity_fit(..., t =)` and `exponential_distribution()` (#800, #862).
