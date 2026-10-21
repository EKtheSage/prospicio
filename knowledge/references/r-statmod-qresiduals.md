---
type: Reference
title: R statmod's randomized quantile residuals
description: statmod's qresiduals, the reference for GlmFit::quantile_residuals; each family has its own function, the gamma's default dispersion is Pearson's but the inverse Gaussian's is not, and the Poisson's ignores the prior weights and any dispersion.
resource: https://cran.r-project.org/package=statmod
tags: [models, glm, residuals, diagnostics, parity, r]
status: stable
verified: { by: process:ci, at: 2026-10-21T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-21T12:00:00Z }
sources:
  - id: statmod
    resource: https://cran.r-project.org/package=statmod
    title: statmod 1.5.2 (GPL), Smyth and others; qresiduals and its qres.* functions
  - id: dunn-smyth
    resource: https://doi.org/10.1080/10618600.1996.10474708
    title: Dunn and Smyth (1996), Randomized quantile residuals, Journal of Computational and Graphical Statistics 5(3)
  - id: script
    resource: ../validation/scripts/r_quantile_residuals.R
    title: r_quantile_residuals.R, the reference values for prospicio's quantile residuals
---

# What qresiduals computes

`qresiduals(m, dispersion = NULL)` dispatches on the family to one function
each:

- `qres.binom`, `qres.pois`, `qres.nbinom`: `qnorm(runif(n, F(y − 1), F(y)))`,
  drawn with R's generator, so only the bounds, not the draws, can be matched.
  `qres.pois` uses `ppois(y, fitted)`: it ignores the prior weights (a rate
  with exposure weights is not a Poisson count) and any quasi-Poisson
  dispersion. prospicio's `Family::cdf_bounds` scales the lattice by `w / φ`
  instead, as `Family::draw` samples.
- `qres.gamma`: `qnorm(pgamma(w y / μ / φ, w / φ))`, with `φ` by default
  Pearson's `Σ w ((y − μ) / μ)² / df`, the same as `summary(m)$dispersion`.
- `qres.invgauss`: `pinvgauss` on the lower or upper tail for `y` below or
  above `μ`, keeping precision far out. Its default `φ` is
  `Σ w (y − μ)² / (μ² y) / df`, not Pearson's `/ μ³`, so it differs from
  `summary(m)$dispersion`; pass that dispersion to match a fit's.
- `qres.default` (the Gaussian): the deviance residual over `√φ`.
- `qres.tweedie`: `ptweedie` with the zero randomized on `(0, P(Y = 0))`;
  it needs the `tweedie` package.

# What prospicio matches

`validation/tests/models.rs::quantile_residuals_match_statmod`: the
Gaussian, gamma (log link) and inverse Gaussian (log link, started at the
gamma fit, without which R's IRLS diverges on quine) to 1e-10, 1e-6 and
1e-6 at `summary(m)$dispersion`; the Poisson, `glm.nb` and binomial bounds
`F(y − 1)` and `F(y)` to 1e-6, with every residual between their normal
scores.
