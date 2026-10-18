---
type: Finding
title: The Monograph 15 chapters 3 to 6 guides in R, what the R package covers
description: The calls of the Python guides capital-modeling-monograph-15-theory.md and -applications.md mapped onto the R package; every printed number equal in both bindings, and the one R gap (pentagon() has no risk discount).
tags: [pricing, distortions, natural-allocation, r, python, api, bindings, guides, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:30:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:30:00-07:00 }
sources:
  - id: theory
    resource: ../docs/guides/capital-modeling-monograph-15-theory.qmd
    title: The R guide for chapters 3 and 4, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: applications
    resource: ../docs/guides/capital-modeling-monograph-15-applications.qmd
    title: The R guide for chapters 5 and 6
  - id: namespace
    resource: ../R/prospicio/NAMESPACE
    title: The R package's exports on main, 2026-10-17
---

# Facts (2026-10-17)

* Python to R:[^theory][^namespace] `Portfolio(units, rows, probs)` is
  `capital_portfolio(matrix, probs =)` with the units as column names
  (`@totals`, `@probs`, `@expected`); `Portfolio.kappa(u)` is
  `portfolio_kappa(port, u)`; `calibrate(family, assets=, return_on_capital=)`
  is `calibrate_portfolio(port, family, assets =, return_on_capital =)`;
  `price(d, assets=)` is `natural_price(port, d, assets =)`, a data frame with
  one row per unit and a `"total"` row; `premium_bounds` is `premium_bounds()`
  (a data frame with `unit`, `lower`, `upper`); `Distortion.ccoc(r)`,
  `.tvar(p)`, `.proportional_hazard(rho)`, `.bitvar(p0, p1, w)` are
  `distortion("ccoc", r)`, `distortion("tvar", p)`,
  `distortion("proportional_hazard", rho)`, `distortion("bitvar", c(p0, p1,
  w))`; `d.g(s)` is `distortion_g(d, s)` (vectorized); `d.measure(Sampled(x))`
  is `risk_measure(sampled(x), d)`; `Sampled.quantile`, `.tvar` are
  `quantile()`, `TVaR()`; `Gamma(shape, scale)` is `gamma_distribution(shape,
  scale)`; `Pentagon.solve(...)` is `pentagon(...)`, a named vector.
* Every printed number in the two R guides equals the Python guides', including
  the grid-search optimum of section 5.5 and the calibrated parameters to eight
  decimals.[^theory][^applications]
* R gap: `pentagon()` returns the five amounts and three ratios but not the risk
  discount `M / (a - L)` that Python's `Pentagon.discount` returns (issue #365,
  Aggregate lane).[^theory]
* The grid search of section 5.5 (about 12,000 `natural_price()` calls) takes
  about 9 s in R against 0.3 s in Python, the per-call overhead of the R data
  frame.[^applications]

[^theory]: capital-modeling-monograph-15-theory.qmd, sections 5 and 8
[^applications]: capital-modeling-monograph-15-applications.qmd, section 6
[^namespace]: R/prospicio/NAMESPACE
