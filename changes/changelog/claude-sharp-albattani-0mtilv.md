### Breaking changes

- The posterior predictive checks (`prospicio_bayes::ppc::check`,
  `check_with`, `check_grouped`, `loo_pit`, and `BayesGlmFit`'s
  `posterior_predictive_check` and `loo_pit`; Python `ppc_check`,
  `loo_pit` and the fit's methods; R `ppc_check()`, `loo_pit()`,
  `bayes_ppc()`, `bayes_loo_pit()`) return an error where they returned a
  biased value: a replicate or observation that is not finite (a Bayesian
  GLM's replicate is NaN where a posterior draw's mean is invalid for the
  family, such as a mean at or below 0 under an identity link), or a
  statistic that is not finite (the sd of one value). The error says how
  many draws and observations are affected (#831, #852).

### Models

- `loo_pit` breaks ties with stream `u64::MAX` of its seed
  (`prospicio_bayes::ppc::TIE_STREAM`) instead of stream 0, which
  `BayesGlmFit::replicate` also gave its first draw; a seeded LOO-PIT of
  counts with ties changes once (#831, #852).

### Pricing

- `DevelopmentPattern::from_history` sums the claims in the order each
  first appears instead of a hash map's order, so its factors, and the
  IBNER and burning cost built on them, are the same to the bit on every
  run; a result may move once in its last bit (#830, #852).

### Reserving

- `Triangle::from_long` (and every reader built on it) refuses a long
  table whose periods span a dense grid of more than 2^24 cells and more
  than 1,000 cells a row, as one mistyped origin such as `9020-01` among
  monthly origins does, with the new `Error::TooLarge` naming the first
  and last origin, instead of trying to allocate tens of gigabytes (#832, #852).
  Rust: `prospicio_reserving::Error` has a new variant, so an exhaustive
  `match` on it needs an arm.

### Aggregate and reinsurance

- `YearLossTable::new` (Python `read_ylt`, R `read_ylt()`) refuses a year
  column whose largest year is above 2^24 and above 1,000 times the row
  count, such as trial ids read as years (#832, #852).
- The GIL is released during `EventLossTable.simulate`, `Hawkes.simulate`,
  `EventSet.grouped`, `eventual_ruin`, `wiener_hopf`,
  `Portfolio.price`, `Portfolio.calibrate`, `price_with_reinstatements`,
  `TowerQuote.technical` and `Tower.deal_ledger`, and in `worst_var` and
  `mars`, so other Python threads run while they compute (#847, #852).

### Distributions and probability

- The `risk::*_sorted` value-at-risk and TVaR functions return an error on
  an empty slice instead of panicking (#832, #852).

### Models

- `OnnxModel` returns an error, not a Python `PanicException`, for an ONNX
  file on which its parser panics, and refuses a constant whose data does
  not fill its declared shape, which was read padded with zeros (#832, #852).

### Excel

- A function whose result would have no rows or no columns shows a
  `#PROSPICIO!` error instead of an empty array (#832, #852).

### Aggregate and reinsurance

- `Tower::from_json` refuses a document with `format_version` 0, as the
  model artifacts already did; no writer produces one (#840, #852).

### Distributions and probability

- `Dist::from_json` refuses `format_version` 0, as the model artifacts
  already did (#840, #852).
