### Reserving

- Development factors averaged over the latest periods and without the
  highest and lowest link ratios: `n_periods`, `drop_high` and `drop_low`
  on `Development` (chainladder-python's names), for the chain ladder, the
  four expected-loss methods and Mack. `n_periods` counts calendar
  diagonals, and an age with fewer link ratios uses them all. The drops
  rank within the kept link ratios and are skipped at an age where they
  would leave none, as Friedland's medial average does. The exclusions
  enter sigma and Mack's standard errors too, at parity with
  chainladder-python 0.10.1 and R ChainLadder's `MackChainLadder(weights
  = W)`. `DevelopmentFit::n_links` gives the number of link ratios behind
  each factor. The bootstraps and the one-year views reject them for now.
  In Python and R as `n_periods`, `drop_high` and `drop_low` (#336, #339).
- A geometric average, `average = "geometric"`, for the chain ladder and
  the expected-loss methods; Mack and the bootstraps reject it, since it
  has no variance. **Breaking (Rust):** `Average` has a new variant and
  `Average::alpha` returns `Option<f64>`, `None` for it; `Error` has a new
  variant, `NotSupported`.
- A selected pattern of factors to ultimate (or percents developed) may
  now go below 1 and need not be monotone, for negative development such
  as Friedland's XYZ Insurer reported pattern (#336).
