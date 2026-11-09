---
type: Finding
title: The validation verdict against aggregate 2.1.0
description: aggregate 2.1.0's valid flags a built aggregate when a severity or aggregate moment on the grid misses its exact value (mean at 1e-4, cv at 1e-3, skewness at 1e-2), when the convolution moved the mean by more than 1e-5 with no deficit (ALIASING), or when more than 1e-4 is lost off the grid (DEFECTIVE); prospicio's validate_grid reads the same terms off the grid audit and agrees flag by flag on six books whose verdicts sit well away from the bounds.
tags: [validation, fft, grid, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T23:00:00Z }
verified: { by: process:ci, at: 2026-11-09T23:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_validation.py
    title: validation/scripts/aggregate_validation.py (aggregate 2.1.0's Aggregate.valid)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, _validation.py (valid_aggregate, convolution_residual, pmf_deficit) and constants.py (Validation)
---

# Finding

* `valid_aggregate` reads its moment errors from `stats_df`: the severity's
  and the aggregate's mean relative error against `eps` (1e-4), their cv
  against `10 eps` and skewness against `100 eps`, skipping a theoretical
  value of 0 (a symmetric severity's skewness). These are exactly the six
  terms of `sharpen`'s score, which prospicio's grid audit matches (#317),
  so `validate_grid` sets a flag where a term exceeds 1.
* `ALIASING` is not a ratio of mean errors since 1.0.0a311: it is the
  convolution residual `|E[A] − E[N] E[X_grid]| / |E[N] E[X_grid]|` above
  `aliasing_eps` (1e-5), and only when the deficit is dust, since a lost
  tail is `DEFECTIVE`'s (above `deficit_materiality`, 1e-4).
* `Validation.passes` ignores `INFEASIBLE` and `REINSURANCE`, a reading
  about the grid and a marker of a cession.
* On six books (a gamma and a capped lognormal, each on a fine, a short and
  a coarse grid) the flags agree one for one: the fine grids pass, the
  coarse ones fail on the severity's shape and the aggregate's mean and
  cv, the short ones fail on the aggregate's moments and are
  defective.[^script]

[^script]: validation/scripts/aggregate_validation.py, run 2026-11-09.
