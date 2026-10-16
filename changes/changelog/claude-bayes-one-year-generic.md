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
  add `exposure` and `prior_ultimate` (#274).
- The Bayesian one-year view is written once for every Bayesian model,
  on `prospicio_bayes::elpd::psis`, and checked by a conformance suite
  run on each model. `BayesOdp`'s one-year view is unchanged, bit for
  bit. A simulation whose year has a log likelihood that is not a number
  is now an error, as a failed refit is (#274).
- Moment matching in the Bayesian one-year view, opt-in:
  `OneYearRefit::MomentMatchThenRefit` moves the posterior draws of a
  flagged simulation by affine maps to the moments its importance weights
  estimate (Paananen et al. 2021, as R's `loo::loo_moment_match`, with
  its split estimate) and refits with NUTS only a simulation whose Pareto
  k stays above the threshold; `BayesOneYearFit::moment_matched` (and
  `BayesOneYearSegment::moment_matched`) lists the simulations it
  rescued. `RefitFlagged` stays the default. In Python
  `refit="moment_match_then_refit"` and `.moment_matched`, in R the same
  `refit` value and `@moment_matched` (1-based). **Breaking (Rust)** for
  code that matches `OneYearRefit` exhaustively or builds
  `BayesOneYearFit` or `BayesOneYearSegment` in full (#274).
