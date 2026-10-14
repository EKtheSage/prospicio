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
  one-year view on RAA and GenIns. Rust only for now.
