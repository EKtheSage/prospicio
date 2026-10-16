### Reserving

- The one-year view of Gesmann and Morris' compartmental models 1 and 2,
  `Compartmental::one_year` (behind the `bayes` feature): the claims
  development result over the coming year by Pareto smoothed importance
  sampling of the posterior, with moment matching and then a NUTS refit
  for the flagged simulations, as the other Bayesian models' views. An
  origin with no observed cell (and, in model 2, an age with none) has
  its effects in each posterior draw, so the year's cells update them. On
  company 337 about four in five of model 1's simulations are flagged and all
  of model 2's; for model 1
  matching rescues about 95% of them within a twentieth of an sd of their
  refits, for model 2 none, so model 2's default view refits nearly
  every simulation, each refit as long as a fit (about 31 s of wall time
  with 8 running side by side). In Python
  `Compartmental.one_year(triangle, paid, premium, *, outstanding=None,
  incurred=None, k_threshold=0.7, refit="moment_match_then_refit",
  max_refits=None)`, returning a `CompartmentalOneYearFit`; in R
  `compartmental_one_year()` and `compartmental_model_2_one_year()`,
  returning a `compartmental_one_year_fit` (#NNN).
- A refit budget for every Bayesian one-year view:
  `BayesOneYear::max_refits` refits at most that many simulations (the
  first due for a refit, in order) and reweights the others, listing them
  in the view's new `unrefitted`; `None`, the default, refits every one as
  before, bit for bit. Python's `one_year` and `one_year_segments` of
  `BayesOdp`, `BayesBf` and `Meyers` take `max_refits=None` and their
  views have `unrefitted`; R's `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
  `meyers_one_year()` and `meyers_one_year_segments()` take
  `max_refits = NULL` and their views have `@unrefitted`. A warning names
  the simulations left unrefitted. **Breaking (Rust):** `BayesOneYear`,
  `BayesOneYearFit` and `BayesOneYearSegment` have a new public field, so
  a struct literal of them without `..` must set it (#NNN).
