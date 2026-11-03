### Reserving

- `DevelopmentPatterns::monthly` (Python `fit.patterns.monthly`, R
  `monthly_pattern_draws()`) is about fifty times faster under the Weibull
  and loglogistic readings (#717): the simulations are read in parallel (a
  new `parallel` feature of `prospicio-reserving`, on with `bayes` and in
  the bindings, off on the WASM path), and each simulation's curve is
  searched from the mean pattern's. 2,000 ODP bootstrap draws at 144
  months on an underwriting basis take about 0.3 s on 16 cores, from 17 s
  to 21 s (3.4 s on one thread). The monthly shares move by less than
  `1e-9` of a share, and they do not depend on the number of threads. The
  documentation says that bootstrap draws usually need
  `monotone="running_max"`.
