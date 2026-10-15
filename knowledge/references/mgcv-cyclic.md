---
type: Reference
title: mgcv 1.9.1's cyclic P-spline, bs = "cp"
description: How mgcv builds s(x, bs = "cp"), the reference for prospicio_glm's cyclic smooths; k + 1 unwidened knots over the period, functions wrapped by cSplineDes, a cyclic difference penalty, new data wrapped by cwrap, and smoothCon rescaling the penalty unless scale.penalty = FALSE.
resource: https://cran.r-project.org/package=mgcv
tags: [gam, cyclic, smoothing, reference-implementation, r]
status: stable
verified: { by: process:ci, at: 2026-10-15T14:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-15T14:00:00Z }
sources:
  - id: mgcv
    resource: https://cran.r-project.org/package=mgcv
    title: mgcv 1.9-1 (GPL), smooth.construct.cp.smooth.spec, cSplineDes, cwrap and Predict.matrix.cpspline.smooth
  - id: script
    resource: ../validation/scripts/r_gam_cyclic.R
    title: r_gam_cyclic.R, the reference values for prospicio's cyclic smooths
---

# The basis

* `s(x, bs = "cp", k = 10)` takes `k + 1` equally spaced knots from the
  period's start to its end: `knots = list(x = c(lo, hi))` when given,
  else `min(x)` and `max(x)`. Unlike `"ps"`, the range is not widened by
  0.1%.[^mgcv]
* `cSplineDes(x, knots, ord = 4)` puts `ord - 1` knots below the start,
  evaluates `k` B-splines, and for `x` within `ord - 1` intervals of the
  end adds the same functions at `x - period`: the functions that start
  below the period wrap onto its top. The `k` values sum to 1.[^mgcv]
* The penalty is `DᵀD` for the `m[2]`-th difference matrix taken around
  the circle (the first `m[2]` columns of the ordinary difference matrix
  folded onto the last), so only a constant is unpenalized; `gam`'s
  sum-to-zero constraint then removes it.[^mgcv]
* `Predict.matrix.cpspline.smooth` wraps new `x` outside the knots into
  the period with `cwrap`, so a prediction at `hi` equals one at `lo` and
  at `hi + 1` one at `lo + 1`.[^mgcv]

# Quirk met reproducing it

* `smoothCon(..., absorb.cons = FALSE)` rescales the penalty by default
  (`scale.penalty = TRUE`: 0.375 where the raw `DᵀD` has 6 on the
  diagonal, for `k = 10` on the test data); pass `scale.penalty = FALSE`
  to compare matrices. `gam`'s fits do not depend on the scaling.[^script]
* prospicio_glm matches the raw basis and penalty at `1e-12`, and GCV and
  REML fits (Gaussian, Poisson) at `1e-6`.[^script]

[^mgcv]: `R/smooth.r` in the mgcv 1.9-1 sources.
[^script]: `validation/scripts/r_gam_cyclic.R` and `validation/tests/models.rs`.
