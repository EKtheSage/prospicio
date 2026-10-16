### Reserving

- The one-year view of Gesmann and Morris' compartmental models 1 and 2,
  `Compartmental::one_year` (behind the `bayes` feature): the claims
  development result over the coming year by Pareto smoothed importance
  sampling of the posterior. An origin with no observed cell (and, in
  model 2, an age with none) has its effects in each posterior draw, so
  the year's cells update them. On company 337 about four in five of
  model 1's simulations are flagged and all of model 2's; for model 1
  moment matching rescues about 95% of them within a twentieth of an sd
  of their refits, for model 2 none, and each refit takes as long as a
  fit (for model 2 about 31 s of wall time with 8 running side by side).
  So, by the user's decision of 2026-10-16, model 1's view moment-matches
  its flagged simulations and refits with NUTS those it cannot rescue,
  while model 2's (any model with development effects,
  `Compartmental::one_year_refit`) reweights them only and says in the
  view's `approximate` that it is approximate (about 0.4 sd of the total
  off a refit per simulation); setting `refit` still refits. Both run at
  most `Compartmental::ONE_YEAR_SIMS` (2,000) one-year simulations by
  default; the lifetime view keeps `n_sims`. In Python
  `Compartmental.one_year(triangle, paid, premium, *, outstanding=None,
  incurred=None, k_threshold=0.7, refit=None, max_refits=None,
  n_sims=None)`, returning a `CompartmentalOneYearFit` (with
  `approximate`, and a warning when it is set); in R
  `compartmental_one_year()` and `compartmental_model_2_one_year()` (with
  `refit = NULL` and `one_year_sims = NULL`), returning a
  `compartmental_one_year_fit` (#303).
- A refit budget for every Bayesian one-year view:
  `BayesOneYear::max_refits` refits at most that many simulations (the
  first due for a refit, in order) and reweights the others, listing them
  in the view's new `unrefitted`; `None`, the default, refits every one as
  before, bit for bit. Over segments the budget applies to each segment. Python's `one_year` and `one_year_segments` of
  `BayesOdp`, `BayesBf` and `Meyers` take `max_refits=None` and their
  views have `unrefitted`; R's `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
  `meyers_one_year()` and `meyers_one_year_segments()` take
  `max_refits = NULL` and their views have `@unrefitted`. A warning names
  the simulations left unrefitted. **Breaking (Rust):** `BayesOneYear`,
  `BayesOneYearFit` and `BayesOneYearSegment` have a new public field, so
  a struct literal of them without `..` must set it (#303).
- **Breaking (Rust):** `BayesOneYear::refit` is an `Option<OneYearRefit>`,
  `None` by default for the model's own one-year default (moment matching
  for every model but `Compartmental` with development effects, so
  `BayesOdp`, `BayesBf` and `Meyers` behave as before); write
  `refit: Some(OneYearRefit::RefitFlagged)` where `refit:
  OneYearRefit::RefitFlagged` was. `BayesOneYearFit` has a new public
  field, `approximate` (#303).
