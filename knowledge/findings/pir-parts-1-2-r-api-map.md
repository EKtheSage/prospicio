---
type: Finding
title: The Pricing Insurance Risk Parts I and II guides in R, what the R package covers
description: The calls of the Python guides pir-risk-measures.md and pir-portfolio-pricing.md mapped onto the R package; every printed number equal to Python's once the Rearrangement Algorithm starts from fixed rotations and the FFT round-off is printed as a bound; R gaps VaR() on grids (#373) and VaR() and TVaR() on parametric severities (#375).
tags: [risk-measures, var, tvar, distortion, pricing, classical-premium, ruin, r, python, api, bindings, guides]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:50:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:50:00-07:00 }
sources:
  - id: guide1
    resource: ../docs/guides/pir-risk-measures.qmd
    title: The R guide to Part I, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: guide2
    resource: ../docs/guides/pir-portfolio-pricing.qmd
    title: The R guide to Part II
  - id: dist
    resource: ../R/prospicio/R/distributions.R
    title: The R VaR() and TVaR() generics and their methods on main, 2026-10-17
---

# Facts (2026-10-17)

* Python to R, Part I:[^guide1] `Discrete(x, p)` is `discrete_distribution(x,
  p)`, `.quantile(p)` is `VaR()` or `quantile()`, `.quantile_upper(p)` is
  `quantile_upper()`; `Lognormal.from_mean_cv` is `lognormal_from_mean_cv()`
  (with `@meanlog`, `@sdlog`), `.lev(x)` and `.cdf(x)` are `lev()` and
  `cdf()`; `Grid.rounding(sev, step, n)[0]` is `discretize(sev, step, n,
  method = "rounding")`, `Grid.map(f)[0]` is `map_grid(grid, f)$grid`;
  `Portfolio.from_independent(names, grids)` is `capital_portfolio(list(...))`
  (`@totals`, `@probs`), `.assets_for_epd(r)` is `assets_for_epd(port, r)`;
  `Distortion.proportional_hazard(r)` is `distortion("proportional_hazard",
  r)`, `.measure(x)` is `risk_measure(x, d)`, `.g`, `.g_dual`, `.weights(n)`,
  `.mass` are `distortion_g()`, `distortion_g_dual()`, `distortion_weights()`,
  `distortion_mass()`, `Distortion.minimum` is `distortion_minimum()`.
* Part II:[^guide2] `book.price(d, p=p).total` is the `"total"` row of
  `natural_price(port, d, p = p)`; `price_stand_alone(d, p)` returns the units,
  `"total"` and `"sum"` rows of one data frame; `book.calibrate(family, p=,
  return_on_capital=)` is `calibrate_portfolio(port, family, p =,
  return_on_capital =)` with the parameter in `@param`;
  `classical_premium(principle, x, t)` and `calibrate_classical(principle, x,
  P)` take `x` first in R; `eventual_ruin(sev, rho, points).capital(p,
  interpolate=True)` is `eventual_ruin(sev, rho =, p =, points
  =)$capital_interpolated`, vectorised over `p` (one call per margin, which
  halves the guide's run time); `Distortion.convex(points)` is
  `distortion_convex(s, g)`; `Tower([Layer.stop_loss(n, l, a)]).on_grid(
  Count.fixed(1), sev, n).net` is `tower_on_grid(reinsurance_tower(list(
  aggregate_stop_loss(n, l, a))), fixed_count(1), sev, n)$net` (its help page
  lists only Poisson, negative binomial and binomial frequencies, but a fixed
  count works).
* Equal numbers need care in two places. The Rearrangement Algorithm's answer
  depends on its start; both guides start from column j rotated by 7, 13 or
  19 times j places (numpy `np.roll`, R index arithmetic) and sort ties
  stably, and print 353.8, 353.6 and 352.9 against the book's 352.8. The FFT
  round-off of a `tower_on_grid()` net differs between the builds on one
  machine (largest stray probability 7.0e-20 in the Python wheel, 7.7e-20 in
  the R package; exponential premium of Tame's net B 509.4 against 510.3),
  so both guides print bounds.
* R gaps:[^dist] `VaR()` has no method for `grid_distribution` (#373) nor for
  any parametric severity (`lognormal`, `pareto`, `gamma_distribution`,
  `weibull_distribution`, `burr_distribution`, `loglogistic_distribution`,
  `inverse_gamma_distribution`, `generalized_pareto`), and `TVaR()` none for
  those severities (#375, the R side of #366's `tvar`). `quantile()` and
  `lev()` work on all of them, so the guides use `quantile()` and
  `VaR + (mean - lev(VaR)) / (1 - p)`.

[^guide1]: The R guide to Part I, every chunk's printed text checked by R/prospicio/tests/test-guides.R
[^guide2]: The R guide to Part II
[^dist]: The R VaR() and TVaR() generics and their methods on main, 2026-10-17
