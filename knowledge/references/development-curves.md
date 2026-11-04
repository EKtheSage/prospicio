---
type: Reference Implementation
title: Development curves over every age in R lm and nls
description: R's lm and nls (port) as the reference for prospicio_reserving::DevelopmentCurve on RAA and GenIns; Sherman's curve has an interior optimum on both, plain Gauss-Newton nls does not converge at a tight tolerance, and no printed fit is in the local references.
tags: [reserving, development, curve, tail, parity, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-03T00:00:00Z }
verified: { by: validation/tests/reserving_curves.rs, at: 2026-11-03T00:00:00Z }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 31
  - id: curve
    resource: ../crates/prospicio-reserving/src/curve.rs
    title: prospicio_reserving::curve
  - id: r-script
    resource: ../validation/scripts/reserving_curves_r.R
    title: R reference generator for development curves
---

# Use

Reference values for `prospicio_reserving::DevelopmentCurve` and the new
`CurveShape` variants, on the volume-weighted factors of RAA and GenIns
(R 4.5.3, ChainLadder 0.2.21 to read the triangles).[^r-script] `k` is the
1-based development index; the growth curves read the ages in
months.[^design]

* Exponential, inverse power and Hoerl: `lm(log(f - 1) ~ k)`,
  `~ log(k)` and `~ k + log(k)`, to 1e-10 and 1e-9.
* Sherman, `f - 1 = a (k + c)^(-b)`: `nls(..., algorithm = "port", lower =
  c(0, -Inf, -0.999999))` from the inverse power's line at `c = 0`, to
  1e-6 on the parameters and fitted factors.
* The log-logistic and Weibull growth curves, `f_k = G(t_(k+1)) /
  G(t_k)`: `optim` on the log parameters, then `nls` by the port algorithm.

# Findings

* Sherman's curve has an interior least-squares optimum on both books: RAA
  `a = 10.0824`, `b = 2.66828`, `c = 0.83389` (sum of squares 0.0017589
  against 0.4968 for the exponential and 0.0243 for Hoerl), GenIns
  `a = 5.49816`, `b = 2.19403`, `c = 0.43518`.[^curve]
* R's default Gauss-Newton `nls` with `tol = 1e-10` stops at its iteration
  limit on Sherman's curve; the port algorithm converges.[^r-script]
* On both books every factor is above 1.00001, so every factor enters
  every fit and all shapes share their nine points.
* The curve's 100-period tail differs widely by shape on the same factors:
  on RAA 1.0094 (exponential), 1.0157 (Hoerl), 1.102 (log-logistic), 1.101
  (inverse power) and 1.127 (Sherman); the Weibull gives 1.0002.
* Neither Friedland (2010), which mentions curve fitting for tails without
  a fitted example, nor any other local reference prints a fitted Hoerl or
  Sherman curve; Sherman's 1984 paper is not in `knowledge/references/`.

[^design]: Design note, reserving v0.2, decision 31
[^curve]: prospicio_reserving::curve
[^r-script]: R reference generator for development curves
