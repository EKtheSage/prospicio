### Breaking changes

- `credit_loss`, `recovered` and `net_after_default` are reserved names in
  `prospicio_aggregate`, like `brokerage` and `margin`: a deduction with one
  of these labels, in Rust or in a tower's JSON, is now refused (#980).

### Aggregate and reinsurance

- Reinsurer default on a tower's simulated years: `Panel` and `Reinsurer`
  in `prospicio_aggregate`, each reinsurer with a probability of default,
  a recovery rate, collateral and its signed lines on the layers.
  `Panel::apply` adds each layer's credit loss and recovery after default
  and the net loss after default to a tower's one-year result, and
  `Panel::by_reinsurer` gives each reinsurer's credit loss, exposure and
  default from the same draws. Defaults follow one Gaussian factor ranked
  on the year's gross loss (or a driver given), so they gather in the worst
  years without changing each reinsurer's probability of default; the
  result has the same bits on any number of threads. Rust only for now
  (#980).
