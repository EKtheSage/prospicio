### Reserving

- `Compartmental` (models 1 and 2) in the Python and R diagonal backtest:
  `Compartmental.candidate(...)` and `compartmental_candidate()` go in the
  `models` of `diagonal_backtest`, and `outstanding_backtest` and
  `compartmental_outstanding_backtest()` score the held-out outstanding on
  the same fits. `CompartmentalCandidate` now owns its triangle and columns
  (no lifetime parameter, **breaking** for Rust code that named
  `CompartmentalCandidate<'_>`) and a clone shares its cache of fits.
