### Reserving

- Gesmann and Morris' hierarchical compartmental reserving model with one
  outstanding compartment (`prospicio_reserving::bayes::Compartmental`,
  behind the `bayes` feature): paid and outstanding fitted jointly with
  NUTS from a reporting rate, a payment rate and each origin's reported
  loss ratio and reserve robustness factor (correlated origin effects),
  with priors a required argument (`CompartmentalPriors`, and the paper's
  case study as `CompartmentalPriors::gesmann_morris()`), a prior
  predictive check (`prior_predictive`), the reserve to the square's last
  age as a joint `PredictiveDistribution`, the future paid and outstanding
  cells, PSIS-LOO split by type, and a warning when a draw's rates are in
  the swapped order (or `RateOrder::Ordered`). The likelihood, scale, paid
  response and fitted pair are settings; this release fits the paper's
  model 1 (normal on amounts, cumulative paid, paid and outstanding), at
  parity with the paper's code reconstructed in Stan and its printed
  posterior on company 337; the other values are an error saying they are
  not yet built. In Python as `prospicio.reserving.Compartmental`
  (`CompartmentalPriors.gesmann_morris()`, `fit(..., incurred=...)` or
  `outstanding=...`, `prior_predictive`) and in R as `compartmental()`
  (`gesmann_morris_priors()`, `compartmental_prior_predictive()`), with
  the settings as strings, the diagnostics as one warning, and
  `bayes_loo()` in R (#243).
