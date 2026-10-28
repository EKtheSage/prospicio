---
type: Finding
title: Hachemeister's iterative estimator at the origin has no stable fixed point on his own data, so its results hold to about 1e-6
description: "actuar's hache.origin on Hachemeister's five states with the trend [1, t] stops at tol = sqrt(eps) after 47 updates; run past it the collective wanders by about 1e-5 (3,000 updates at tol = 0, never converging, and tol = 1e-13 hits any iteration limit), because the between covariance's correlation is 0.9998; prospicio's same loop agrees with it to 1e-6 at the default stop, and the fit with the intercept at the barycentre, where every matrix is diagonal, agrees to 1e-9."
tags: [prospicio-pricing, credibility, hachemeister, actuar, regression]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-27T20:00:00Z }
verified: { by: process:ci, at: 2026-10-27T20:00:00Z }
sources:
  - id: test
    resource: ../validation/tests/pricing.rs
    title: regression_credibility_matches_actuar
  - id: script
    resource: ../validation/scripts/r_credibility_regression.R
    title: The actuar 3.3.7 reference, and why its origin rows are checked at 1e-6
  - id: note
    resource: ../docs/design/credibility.md
    title: Design note, decisions 9 to 12 (#561)
---

# Hachemeister's origin iteration on his own data

`actuar::cm(~state, hachemeister, regformula = ~time, regdata =
data.frame(time = 1:12))` fits each state's line `[1, t]` and shrinks it
towards the collective with full credibility matrices. The between
covariance comes from Hachemeister's pseudo-estimator
`A = Σ Z_i (b_i - β̂)(b_i - β̂)ᵀ / (I - 1)`, iterated with the `Z_i`
and the collective.

## Facts

- At actuar's default `tol = sqrt(eps)` the loop stops after 47 updates,
  with the collective at (1468.775, 32.0489) and
  `A = [[24154, 2700], [2700, 301.8]]`: a correlation of 0.9998, nearly
  singular. [^script]
- The iteration has no stable fixed point there. At `tol = 0` the
  collective's intercept wanders between about 1468.757 and 1468.799 over
  3,000 updates, and `tol = 1e-13` reaches any iteration limit. The
  steps contract by about 0.7 at first, then round-off drives them.
  [^script]
- prospicio runs the same loop (start `Z_i = I`, the same stopping rule
  and the same final update). On the full data it stops after the same 47
  updates, its collective within 1e-10 of actuar's, and every value
  agrees to about 1e-9; with four cells missing the gap is largest, about
  1e-7. The origin rows are checked at 1e-6.
  [^test]
- With the intercept at the barycentre (`adj.intercept = TRUE`) the
  matrices are diagonal and each diagonal is Bühlmann-Straub's closed form
  or its one-dimensional iteration. Both estimators agree with actuar to
  1e-9 or better, and the premiums at t = 13 to 1e-10. [^test]

## Use

The premiums at the origin are stable in their first six digits; read
anything finer as round-off. The adjusted fit is the better-conditioned
one.

[^script]: validation/scripts/r_credibility_regression.R, run by hand with actuar 3.3.7 and `echo = TRUE`.
[^test]: validation/tests/pricing.rs, `regression_credibility_matches_actuar`, run by CI.
