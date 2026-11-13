### Bindings

- A mixed Poisson count's mixing names come from one table,
  `MixingFamily::NAMES` with `from_name` and `name` in
  `prospicio_prob::count_families`, which Python and R read: `"ig"` is
  `"inverse_gaussian"` in both, now also in R's `bonus_malus_portfolio()`,
  Python accepts `"sichel_skewness"` as R does, and an unknown name's error
  reads the same in both, `mixing must be "gamma", "inverse_gaussian",
  "ig", ..., got "lognormal"` (#834).
