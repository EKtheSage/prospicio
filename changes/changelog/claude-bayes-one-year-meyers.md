### Reserving

- **Behaviour change:** moment matching is now the default fallback of the
  Bayesian one-year view. `OneYearRefit::default()` and
  `BayesOneYear::default()` are `MomentMatchThenRefit` (was
  `RefitFlagged`), and the `refit` argument of Python's `one_year` and
  `one_year_segments` and of R's `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()` and `bayes_bf_one_year()` defaults to
  `"moment_match_then_refit"`. The simulated years and their Pareto k are
  unchanged; a flagged simulation that matching rescues is no longer
  refitted, so a default run of `BayesOdp` or `BayesBf` takes seconds
  instead of minutes and its rows differ from before within the refits'
  accuracy. Pass `RefitFlagged` (`refit="refit_flagged"`) for the old
  behaviour (#280).
- The one-year view of Meyers' CRC, CSR and CAY models,
  `Meyers::one_year` and `Meyers::one_year_segments` (behind the `bayes`
  feature): the claims development result over the coming year by Pareto
  smoothed importance sampling of the posterior, reweighted on the
  unconstrained scale NUTS samples, with moment matching and then a NUTS
  refit for the flagged simulations. On group 353 nearly every simulation
  is flagged; matching rescues about nine in ten within a tenth of an sd
  of their refits. In Python `Meyers.one_year(triangle, column, premium,
  k_threshold=0.7, refit="moment_match_then_refit")` and
  `Meyers.one_year_segments`, in R `meyers_one_year()` and
  `meyers_one_year_segments()`, returning the same `BayesOneYearFit` /
  `bayes_one_year_fit` (its `fit` a `MeyersFit`) and `BayesOneYearFits` /
  `bayes_one_year_fits`, which now has `premium` for Meyers' views
  (`scale` is the ODP's only) (#280).
- **Behaviour change:** `Meyers::one_year` and `one_year_segments` run at
  most 2,000 one-year simulations by default (`Meyers::ONE_YEAR_SIMS`),
  not the model's `n_sims` (10,000 by default), since nearly every
  simulation of these models costs a moment match or a NUTS refit; the
  lifetime view keeps `n_sims`. A new `BayesOneYear::n_sims`
  (`Option<usize>`, `None` by default) sets the count for any Bayesian
  one-year view; `None` keeps `BayesOdp`'s and `BayesBf`'s own `n_sims`.
  In Python `n_sims=None` on `one_year` and `one_year_segments`, in R
  `one_year_sims = NULL` on `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
  `meyers_one_year()` and `meyers_one_year_segments()`. Adding the field
  breaks a `BayesOneYear { .. }` literal without `..Default::default()`
  (#280).
