### Models

- `elpd_lppd()` in R, the log pointwise predictive density that
  `elpd_loo()` and `elpd_waic()` correct, as `lppd` in Python (#977).
  `trend_factor()`, `credibility_weighted()`, `trended_present_rates()` and
  `ledger_stack()` in R now compute in Rust, with the same results.
