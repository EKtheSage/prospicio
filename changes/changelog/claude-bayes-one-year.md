### Reserving

- The one-year view of the Bayesian ODP chain ladder,
  `BayesOdp::one_year` and `BayesOdp::one_year_segments` (behind the
  `bayes` feature, settings in `BayesOneYear`): the claims development
  result over the coming year by Pareto smoothed importance sampling of
  the posterior draws, each simulated year reweighting them; a
  simulation whose Pareto k is above `k_threshold` (0.7) is refitted
  with NUTS on the extended triangle (`OneYearRefit::RefitFlagged`, the
  default; `ReweightOnly` and `RefitAll` too). Each simulation's Pareto
  k and the flagged share are reported; segments are independent or rank
  correlated. Its standard deviation agrees with the ODP bootstrap's
  one-year view on RAA and GenIns on average over seeds. A triangle with
  no cells in the coming year gives a zero CDR, Pareto k of 0 and no
  refits. In Python `BayesOdp.one_year` and
  `one_year_segments` (`k_threshold=0.7`, `refit="refit_flagged"`,
  `"reweight_only"` or `"refit_all"`) return `BayesOneYearFit` and
  `BayesOneYearFits`; in R `bayes_odp_one_year()` and
  `bayes_odp_one_year_segments()` return a `bayes_one_year_fit` and a
  `bayes_one_year_fits`, with `fit`, `cdr`, `pareto_k`, `refitted`,
  `flagged_share` and the frames. Refits that fail their MCMC
  diagnostics raise a warning (#252).
