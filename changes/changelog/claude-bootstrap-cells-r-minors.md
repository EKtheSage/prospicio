### Reserving

- R's `odp_bootstrap()` and `mack_bootstrap()` refuse a `cells` that is not
  a single `TRUE` or `FALSE` (`1`, `"TRUE"`, `NA` used to keep no cells
  without a word), and the fit's `cells` is built once with the fit, as
  `reserves` is, rather than copied from Rust on every access (#979).
