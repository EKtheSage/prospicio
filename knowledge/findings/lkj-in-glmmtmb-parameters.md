---
type: Finding
title: The LKJ density in glmmTMB's correlation parameters is a product of powers of the row norms
description: "With glmmTMB's us parameters (row r of the correlation's Cholesky factor is (theta_r, 1) / n_r, n_r = sqrt(1 + |theta_r|^2)), Stan's lkj_corr_cholesky density plus the Jacobian of theta_r to theta_r / n_r gives log p(theta) = -(d + 2 eta - 1) sum_r log n_r; each correlation then has the LKJ marginal variance 1 / (2 eta + d - 1)."
tags: [prospicio-bayes, glmm, lkj, nuts, glmmtmb, priors]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T18:00:00Z }
verified: { by: process:ci, at: 2026-10-21T18:00:00Z }
sources:
  - id: stan
    resource: https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html
    title: Stan Functions Reference, the LKJ Cholesky distribution, lkj_corr_cholesky
  - id: lkj
    resource: https://doi.org/10.1016/j.jmva.2009.04.008
    title: Lewandowski, Kurowicka and Joe (2009), Generating random correlation matrices based on vines and extended onion method, JMVA 100
  - id: test
    resource: ../crates/prospicio-bayes/src/glmm.rs
    title: the_correlations_have_the_lkj_marginals and the_posterior_is_the_exact_one_on_a_small_book
---

# Facts

* glmmTMB's `us` structure writes a `d × d` correlation matrix by the
  strictly lower entries `θ` of a unit lower triangle, row by row; row `r`
  normalized, `(θᵣ, 1) / nᵣ` with `nᵣ = √(1 + |θᵣ|²)`, is row `r` of the
  Cholesky factor `L`, so `L_rr = 1 / nᵣ`.
* Stan's `lkj_corr_cholesky(η)` density on the strictly lower entries of
  `L` is `Σ_{r = 1}^{d − 1} (d − r + 2η − 3) log L_rr` (rows from 0). The
  map `θᵣ ↦ θᵣ / nᵣ` on `ℝʳ` has Jacobian `(I − θᵣθᵣᵀ / nᵣ²) / nᵣ`, of
  determinant `nᵣ^−(r + 2)`. Together:
  `log p(θ) = −(d + 2η − 1) Σ_{r ≥ 1} log nᵣ`, the same power on every
  row; for two terms `p(θ) ∝ (1 + θ²)^−(η + ½)`.
* Under it each correlation has the LKJ marginal, a beta on `(−1, 1)` with
  both shapes `η − 1 + d/2`: mean 0, variance `1 / (2η + d − 1)`. A NUTS
  run on the prior alone reproduces it for `(d, η)` = (2, 1), (3, 1),
  (3, 2) and (4, 0.5); a missing Jacobian would not.
* So `BayesGlmm` samples in the Laplace fit's own parameters and starts
  at its estimates, with no second parameterization (such as Stan's
  canonical partial correlations) to keep in step with it.
* On one factor of eight levels (Poisson counts), the posterior means and
  sds of the intercept and the standard deviation from NUTS agree with
  those from integrating each level's effect on a grid to about 2%, once
  the grid reaches `σ = e`; a grid stopping at `σ = 1` cut the tail and
  understated the sds by 8%.

# Gaps

* The prior's effect on a correlation near ±1 with few levels is not
  studied; LKJ(1) is uniform, and a larger `η` shrinks toward
  independence.
