### Reserving

- A new underwriting year's loss ratio from the Bayesian models:
  `BayesBfNormalisedFit::new_year` and `MeyersFit::new_year` (with
  `new_year_with_level` and `NewOriginLevel`) return a `NewYear` with the
  loss ratio, the ultimate and each simulation's development pattern, for
  pricing a year not yet written; Python `fit.new_year`, R
  `bayes_new_year()` (#598, #728).
