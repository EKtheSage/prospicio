### Reserving

- A `ulr_mean` column, the posterior mean ultimate over the exposure (a
  loss ratio only when the exposure is premium), per origin and in total,
  at the end of the long tables of `BayesBf`, `BayesBfNormalised` and
  Meyers' CRC, CSR and CAY, alone and over segments: Rust `to_long` and
  `totals`, Python `to_frame` and `totals_frame`, R `as.data.frame()` and
  `totals_frame()`. Meyers' is its `ultimate_mean` over the premium; the
  two Bayesian BFs also gain `ultimate_mean`, the latest value plus the
  posterior mean reserve. `ulr` stays the chain ladder's ultimate over the
  exposure, and the one-year views keep `opening_ulr` (#645, #648).
