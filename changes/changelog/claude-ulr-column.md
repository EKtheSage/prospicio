### Breaking changes

- The new `ulr` column below sits after each table's own columns in the
  Bayesian and Meyers frames, so the columns after it (`mean`,
  `std_dev`, `ultimate_mean` and so on) move one place. Code that reads
  these tables' columns by position or compares their exact names sees
  one more column (#636, #643).

### Reserving

- A `ulr` column, the ultimate over the exposure (a loss ratio only when
  the exposure is premium), in the long tables of the expected-loss
  methods, Clark's Cape Cod form, `BayesBf`, `BayesBfNormalised` and
  Meyers' CRC, CSR and CAY (on premium), per origin and in total: Rust
  `to_long` and `totals`, Python `to_frame` and `totals_frame`, R
  `as.data.frame()` and `totals_frame()`. The Bayesian tables' `ulr` is the
  chain ladder's ultimate over the exposure, as their `ultimate` column is.
  Their one-year views have `opening_ulr` instead: the opening ultimate
  (the latest value plus the posterior mean opening reserve) over the
  exposure, which differs from the lifetime fit's `ulr` for the same
  origin. `ReserveFit` gains a default `exposure()` method (#636, #643).
