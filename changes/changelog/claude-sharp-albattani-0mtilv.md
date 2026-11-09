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
  many draws and observations are affected (#831).

### Models

- `loo_pit` breaks ties with stream `u64::MAX` of its seed
  (`prospicio_bayes::ppc::TIE_STREAM`) instead of stream 0, which
  `BayesGlmFit::replicate` also gave its first draw; a seeded LOO-PIT of
  counts with ties changes once (#831).

### Pricing

- `DevelopmentPattern::from_history` sums the claims in the order each
  first appears instead of a hash map's order, so its factors, and the
  IBNER and burning cost built on them, are the same to the bit on every
  run; a result may move once in its last bit (#830).
