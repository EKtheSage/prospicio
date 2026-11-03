### Breaking changes

- Rust: `prospicio_reserving::Error` has the variant `IndustryPrior`, so
  an exhaustive `match` on it needs an arm for it (#604).

### Reserving

- An industry prior for the normalised Bayesian Bornhuetter-Ferguson from
  other companies' triangles (`reserving-bayes.md` decision 28):
  `IndustryPrior::from_triangles` (one triangle and premium vector per
  company) and `from_segments` (one triangle with a segment per company)
  estimate the mean ultimate loss ratio, its spread between companies
  (`common_cv`) and within a company (`cv`), the chain ladder's ODP
  process part, and a pattern prior (`PatternPrior`, the mean and sd over
  companies of each company's log share ratios), with a `PriorReport` and
  a warning below 10 companies; `at_age` cuts the prior to a younger
  triangle with a tail factor; `to_bf_normalised_settings`, `fit` and
  `one_year` give and fit the `BayesBfNormalised`. No data ships with the
  library. `BayesBfNormalised::fit_with_pattern_prior` and
  `one_year_with_pattern_prior` replace the vague normal prior on each log
  share ratio by the pattern prior's; without one every result is
  unchanged. Python `IndustryPrior`, `PatternPrior` and
  `BayesBfNormalised(pattern_prior=)`; R `industry_prior()`,
  `pattern_prior()`, `industry_prior_at_age()`, `bayes_bf_industry()`,
  `bayes_bf_industry_one_year()` and `bayes_bf_normalised(pattern_prior =)`
  (#604).
