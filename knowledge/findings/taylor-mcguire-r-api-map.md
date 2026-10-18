---
type: Finding
title: The Taylor and McGuire guide in R, what the R package covers
description: The calls of the Python guide reserving-taylor-mcguire.md mapped onto the R package, with the condition for seeded simulations to agree across bindings (the same design columns in the same order) and the R gaps (no odp_glm(), residuals(), hatvalues() or prediction standard errors on a glm_model).
tags: [reserving, glm, r, python, api, bindings, guides, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:00:00-07:00 }
sources:
  - id: guide
    resource: ../docs/guides/reserving-taylor-mcguire.qmd
    title: The R guide, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: namespace
    resource: ../R/prospicio/NAMESPACE
    title: The R package's exports on main, 2026-10-17
---

# Facts (2026-10-17)

* Python to R:[^guide][^namespace] `Triangle.from_long` is `triangle(long,
  "year", "age", "paid")`; `ChainLadder().fit` is `chain_ladder()` (`@ldf`,
  `@reserve`, `@total_reserve`); `Mack().fit` is `mack()`
  (`@standard_error`, `@total_standard_error`); `OdpBootstrap(n_sims, seed).fit`
  is `odp_bootstrap(tri, "paid", n_sims =, seed =)` (`@scale`, `@reserves`);
  `Glm(family, "log", dispersion="pearson").fit(Design(...), y)` is
  `glm_fit(y ~ 0 + x1 + ..., data, family, link = "log", dispersion =
  "pearson", weights =)`; `GlmFit.covariance`, `.dispersion`, `.fitted`,
  `.std_errors` are `@covariance`, `@dispersion`, `@fitted`, `@std_errors`;
  `predict` is `predict(model, newdata)`; `predict_distribution(design, n, seed,
  "normal")` is `predict_distribution(model, newdata, n_sims =, seed =,
  parameters = "normal")`, and `PredictiveDistribution.draw_matrix()` is
  `draw_matrix(pd)` (one row per draw in both).
* Seeded simulations agree across the bindings when the design has the same
  columns in the same order: the guide builds both designs from one list of
  functions of (k, j), and the bootstrap tables are equal to the digit.
* R gaps, filed as #333 (Reserving: no `odp_glm()`, analytic MSEP, custom
  designs, residual diagnostics) and #334 (Models: a `glm_model` has no
  `residuals()`, `hatvalues()`, `predict(..., se.fit = TRUE)` or GCV).
