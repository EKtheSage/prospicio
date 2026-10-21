### Reserving

- The Bayesian Bornhuetter-Ferguson with a normalised development
  pattern, `BayesBfNormalised` (behind the `bayes` feature): the
  expected ultimate of each origin has the gamma prior (mean `apriori`
  times the exposure, coefficient of variation `cv`) and the pattern is a
  set of shares summing to 1, so `apriori` is the expected ultimate loss
  ratio and moves the reserve, where `BayesBf`'s (EVW's Model 3.1,
  unchanged) barely does. A flat prior is the Bayesian chain ladder; a
  tight prior at the chain ladder's own ultimates is the classical
  Bornhuetter-Ferguson reserve (at other loss ratios the pattern is
  re-estimated given the ultimates). It has the one-year view
  (`BayesBfNormalised::one_year`) through the generic engine. In Python
  `BayesBfNormalised` and `BayesBfNormalisedFit` (`pattern`,
  `expected_ultimate`) with `.one_year()`, and in R `bayes_bf_normalised()`,
  `bayes_bf_normalised_fit` and `bayes_bf_normalised_one_year()`; the fit
  of a one-year view is the new fit class (#410).
