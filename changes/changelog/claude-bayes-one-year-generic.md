### Reserving

- The one-year view of the Bayesian Bornhuetter-Ferguson,
  `BayesBf::one_year` (behind the `bayes` feature, settings in
  `BayesOneYear`): the claims development result over the coming year by
  Pareto smoothed importance sampling of the posterior, the flagged
  simulations refitted with NUTS, as for `BayesOdp`. Under a flat prior
  its standard deviation agrees with `BayesOdp`'s one-year view; under a
  tight one with the closed form of the strong-prior limit. In Python
  `BayesBf.one_year(triangle, column, exposure, k_threshold=0.7,
  refit="refit_flagged")` and in R `bayes_bf_one_year()`, returning the
  same `BayesOneYearFit` / `bayes_one_year_fit` as `BayesOdp`, whose `fit`
  is now the model's own lifetime fit (a `BayesBfFit` here); its frames
  add `exposure` and `prior_ultimate`.
- The Bayesian one-year view is written once for every Bayesian model,
  on `prospicio_bayes::elpd::psis`, and checked by a conformance suite
  run on each model. `BayesOdp`'s one-year view is unchanged, bit for
  bit. A simulation whose year has a log likelihood that is not a number
  is now an error, as a failed refit is.
