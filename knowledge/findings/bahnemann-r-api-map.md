---
type: Finding
title: The Bahnemann guides in R, what the R package covers
description: Each call of the two Python guides that reproduce CAS Monograph 2 (Bahnemann) mapped onto the R package, with the R-only gaps met (no survival or layer variance on lognormal(), no layer_second_moment(), no std(), moment approximations only from a count and a severity; all closed since by #327) and the quirks (total() twice for a simulated sample); every printed number equal to the Python guides'.
tags: [severity, frequency, aggregate, r, python, api, bindings, guides, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T14:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T14:00:00-07:00 }
sources:
  - id: guide1
    resource: ../docs/guides/distributions-for-actuaries-claim-size.qmd
    title: The R guide to chapters 2, 5 and 6, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: guide2
    resource: ../docs/guides/distributions-for-actuaries-counts-aggregates.qmd
    title: The R guide to chapters 3 and 4 and section 6.4
  - id: namespace
    resource: ../R/prospicio/NAMESPACE
    title: The R package's exports on main, 2026-10-17
---

# Facts (2026-10-17)

* The Python calls of the Bahnemann guides and their R names:[^guide1][^guide2]
  `Gamma`, `Lognormal`, `GeneralizedPareto`, `Burr`, `Pareto`, `Beta`,
  `Truncated` are `gamma_distribution()`, `lognormal()`, `generalized_pareto()`,
  `burr_distribution()`, `pareto()`, `beta_distribution()`,
  `truncated_distribution()`; methods `lev`, `layer`, `stop_loss`,
  `survival`, `cdf`, `quantile` are S7 generics of the same names, and
  `sev.sample(n, seed)` is `draws(sev, n, seed)` (same generator, same draws);
  `fit_severity` is `severity_fit()`, a list with `parameters` and
  `standard_errors`; `ilf`, `loss_elimination_ratio`, `layer_cv` keep their
  names; `Poisson`, `Binomial`, `NegativeBinomial` are `poisson_count()`,
  `binomial_count()`, `negative_binomial_count()`, with
  `negative_binomial_count_from_mean_variance()`, `claim_count()`,
  `mixed_poisson_count()`, `geometric_count()`, `empirical_count()`;
  `Grid(step, probs)` is `grid_distribution()`, `Grid.rounding` and
  `Grid.local_moment` are `discretize(d, step, n, "rounding" | "local_moment")`;
  `panjer` and `fft` are `compound_distribution(..., method = "panjer" | "fft")`;
  `compound_moments()` is a named vector (mean, variance, third_cumulant, cv,
  skew); `CompoundMoments.approximate(kind)` is
  `moment_approximation(count, severity, kind)`, whose `$parameters` are the
  fitted shape, scale and shift, read by `approximation_cdf()`.
* Gaps against the Python package met by these guides:[^guide1][^guide2]
  `lognormal()` has no `survival()` and no `layer_variance()` (use
  `1 - cdf()` and `truncated_distribution(ln)`; Python's `Lognormal` likewise
  lacks `survival` and `layer_second_moment`); there is no
  `layer_second_moment()` in R at all (E[X^2; l] is
  `layer_variance + layer^2`); there is no `std()`; `moment_approximation()`
  takes a count and a severity, not the result of `compound_moments()`.
  All closed since by #327: `survival()`, `layer_variance()` and the new
  `layer_second_moment()` on `lognormal()`, `layer_second_moment()` on every
  family that has `layer_variance()`, `std_dev()` (Python's `std()`; named so
  as not to mask `stats::sd`), and `moment_approximation(compound_moments(...),
  kind)`; Python's `Lognormal.layer_second_moment` too. The guides keep their
  workarounds, which print the same numbers.
* Quirks: `compound_distribution()` defaults to Panjer's recursion; `total()`
  of `simulate_events()` is a predictive distribution and a second `total()`
  gives the `sampled` with a `cdf()`; R's `optim(method = "Nelder-Mead")` with
  `reltol = 1e-14` reaches the same minima as SciPy's to the four decimals the
  guides print (Ex 2.8, 2.10, 2.11, P2.16, P2.39, P2.43).[^guide1]
* Every number the two Python guides print comes out the same in R, compared
  number by number over all 337 printed lines, including the maximum
  likelihood fit on 20,000 seeded draws and the FFT aggregate limits.[^guide1][^guide2]

[^guide1]: The R guide to chapters 2, 5 and 6, every chunk's printed text checked by R/prospicio/tests/test-guides.R
[^guide2]: The R guide to chapters 3 and 4 and section 6.4
[^namespace]: The R package's exports on main, 2026-10-17
