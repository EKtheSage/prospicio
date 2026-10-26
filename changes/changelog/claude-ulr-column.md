### Reserving

- A `ulr` column, the ultimate over the exposure (a loss ratio only when
  the exposure is premium), in the long tables of the expected-loss
  methods, Clark's Cape Cod form, `BayesBf`, `BayesBfNormalised` and
  Meyers' CRC, CSR and CAY (on premium), per origin and in total: Rust
  `to_long` and `totals`, Python `to_frame` and `totals_frame`, R
  `as.data.frame()` and `totals_frame()`. The Bayesian tables' `ulr` is the
  chain ladder's ultimate over the exposure, as their `ultimate` column is,
  and their one-year views name it `opening_ulr`. `ReserveFit` gains a
  default `exposure()` method. Tables that list columns by position see
  one more column (#636, #643).
