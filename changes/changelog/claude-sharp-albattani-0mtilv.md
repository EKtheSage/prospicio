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
  many draws and observations are affected, and in `check_grouped` the
  group; a group with fewer observations than its statistic needs (the
  sd of a group of one) is left out of the grouped result
  (`Statistic::min_observations`) (#831, #852).

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
  table whose origins and ages span a grid of more than 2^24 cells in a
  segment and more than 1,000 cells a row, as one origin far from the
  others (a mistyped `9020-01` among monthly origins) does, with the new
  `Error::TooLarge` naming the first and last origin and age, instead of
  trying to allocate tens of gigabytes; many sparse segments over the same
  periods are not affected (#832, #852).
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

- `OnnxModel` returns an error quoting the parser's reason, not a Python
  `PanicException`, for an ONNX file on which its parser panics, and refuses a constant whose data does
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

### Breaking changes

- Building from source needs Rust 1.92 for every crate (`rust-version`
  in `Cargo.toml`, checked by CI's new `msrv` job). The declared 1.85 was
  not true: let chains already needed 1.88, and `prospicio-nn` (Burn 0.21)
  1.92. The R package already asked for 1.92; Python users get wheels
  (#829, #852).

### Reserving

- The `io` feature reads Excel files with calamine 0.38 (was 0.35), which
  fixes its CFB/OLE2 parser's panics and unbounded allocations on a
  malformed legacy `.xls`, and quick-xml's and zip's advisories; the seven
  ignores in `deny.toml` are gone (#829, #852).

### Distributions and probability

- `PredictiveDistribution`'s Arrow IPC (the `arrow` feature) is on arrow 60
  (was 59); the file format is unchanged (#829, #852).

### Models

- R's `booster_fit(dispersion_model = TRUE)` no longer gives the dispersion
  booster the mean's `monotone` and `interaction_constraints`, as Python's
  never did: they describe the mean, and the dispersion often moves the
  other way. R dispersion models fitted with constraints change (#833, #852).
- Both boosting adapters draw their bootstrap rows and the dispersion
  model's folds from the Rust resampler (the new `bootstrap_rows`, and
  `k_fold`), so one seed gives the same rows and folds in Python and R. The
  bootstrap refits and dispersion models of both languages change once
  (#833, #852).

### Breaking changes

- Python and R: a GLM of the gamma, inverse Gaussian or Tweedie family
  given no `link` takes the log link, as Excel's `PRO.PRC.GLM` always did,
  instead of the canonical link (inverse, inverse squared, `power(1 - p)`);
  the Gaussian, Poisson, negative binomial and binomial keep theirs. This
  holds for every model that takes a family and a link (`Glm`, `Gam`,
  `ElasticNet`, `BayesGlm`, `DoubleGlm`, the networks' `scale="link"`).
  R's `glm()` and statsmodels default to the canonical link, so give
  `link` when porting such a model (`Family::default_link`; #835, #852).
