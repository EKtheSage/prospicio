---
type: Reference
title: scam 1.2-22's monotone P-splines, bs = "mpi" and "mpd"
description: How scam builds and fits shape-constrained smooths, the reference for prospicio_glm's monotone smooths; cumulative-sum basis centered by column means, exp increments with a first-difference penalty on their logs, a scaled penalty (sp times c), an edf from the Newton Hessian, no score at fixed sp, and a BFGS that stops about 1e-4 short of the GCV minimum.
resource: https://cran.r-project.org/package=scam
tags: [gam, monotone, smoothing, reference-implementation, r]
status: stable
verified: { by: process:ci, at: 2026-10-15T16:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-15T16:00:00Z }
sources:
  - id: scam
    resource: https://cran.r-project.org/package=scam
    title: scam 1.2-22 (GPL), Pya; smooth.construct.mpi.smooth.spec, smooth.construct.mpd.smooth.spec and scam.fit
  - id: paper
    resource: https://doi.org/10.1007/s11222-013-9448-7
    title: Pya and Wood (2015), Shape constrained additive models, Statistics and Computing 25, 543-559
  - id: script
    resource: ../validation/scripts/r_gam_monotone.R
    title: r_gam_monotone.R, the reference values for prospicio's monotone smooths
---

# The basis

* `s(x, bs = "mpi", k = 10)`: `k + 4` knots with the `k - 2` inner ones
  from `min(x)` to `max(x)`, equally spaced and not widened, cubic
  B-splines `X1` (`k` columns), and `X = (X1 Σ)[, -1]` with `Σ` lower
  triangular ones: column `j` is `Σ_{i≥j} Bᵢ`. The first column, a
  constant, goes with the intercept.[^scam]
* `"mpd"` uses `X1[, -1] %*% Sig[-1, -1]` with a lower triangle of `-1`:
  the `"mpi"` basis negated.[^scam]
* Both are centered by their column means over the data (`cmX`), not by
  mgcv's sum-to-zero constraint, and the coefficients are `exp(γ)` with a
  first-difference penalty on `γ` (`P = diff(diag(k - 1))`), so equal
  increments, a straight line, cost nothing.[^scam]

# The fit and its outputs

* scam scales each penalty: `smooth[[1]]$S[[1]]` is `c DᵀD` (3.0625 for
  `k = 10` on the test data), so `sp` in scam is `λ = c sp` for the raw
  `DᵀD`.[^script]
* `coefficients` hold `γ` on the log scale; `coefficients.t` the
  increments `exp(γ)`.[^scam]
* The edf is the trace of an influence matrix from the full Newton
  Hessian, which includes the `exp` curvature term; prospicio's
  `tr((JᵀWJ − E + S)⁻¹ JᵀWJ)` matches it at `1e-7` at fixed `sp`.[^script]
* With `sp` given, `gcv.ubre` is `NULL`. With `sp` selected, scam's BFGS
  on the GCV or UBRE score stops about `1e-4` (in edf) short of the
  minimum; `optimize` over scam's own fixed-sp fits finds a lower score,
  which is the tight target in prospicio's parity test.[^script]
* scam has no REML.[^scam]

[^scam]: `smooth.construct.mpi.smooth.spec`, `smooth.construct.mpd.smooth.spec` and `scam.fit` in the scam 1.2-22 sources, read with `deparse()`.
[^script]: `validation/scripts/r_gam_monotone.R` and `validation/tests/models.rs`.
