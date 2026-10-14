---
type: Finding
title: Tower fits reproduce their references but not a plausible shape between them
description: fit_references on the guides' references reproduces all four at 1e-11 yet puts alphas of 0.64 and 0.55 inside the 1 xs 1 and 3 xs 2 gaps (layer_loss(2, 2) = 2.12 against about 1.0 by exposure rating); bounding the alpha between reference points to (1.2, 3.0) lowers it to 1.58, but the two pieces match_tower places inside a limited layer still reach 0.71, since the bound is on the point-to-point alpha only.
tags: [pricing, tower-matching, pareto]
status: stable
verified: { by: process:ci, at: 2026-10-14T21:30:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-14T21:30:00Z }
sources:
  - id: tests
    resource: ../crates/prospicio-pricing/src/tower.rs
    title: the issue_211 unit tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/211
    title: Tower matching, warn on implausible fits
---

The references (layers 1 xs 1 at 1.706, 5 xs 5 at 0.675, unlimited xs 10
at 0.295, 2.8 losses a year above 1) are met exactly by `fit_references`,
whose severity has pieces from 1, 1.29, 2, 3.61, 5, 5.99 and 10 with
alphas 1.87, 0.64, 0.55, 1.81, 7.28, 1.46 and 3.29. None of the low alphas
is on the last piece, so the mean is finite: an alpha below 1 makes the
mean infinite only on the last, untruncated piece.

`fit_references_bounded(..., Some((lo, hi)))` adds the bound on the alpha
between consecutive reference points to the completion's linear program
(`g(a) (a/b)^hi <= g(b) <= g(a) (a/b)^lo` is linear in the frequencies).
With `(1.0, 4.0)` the 2 xs 2 layer costs 1.83, with `(1.2, 3.0)` 1.58, and
`(2.5, 3.0)` leaves no tower. Inside a limited layer `match_tower` still
fits two pieces with the layer's loss and end frequencies given, and those
can fall far either side of the point-to-point alpha (0.71 and 3.20 inside
2 to 5 with the tighter bounds). Bounding them would make the layer loss a
nonlinear constraint; `TowerModel::diagnostics` reports them instead.

The issue's second repro, one layer 5 xs 5 at 3.0 with 0.2 losses a year
above 5, is infeasible (the layer can cost at most 0.2 x 5 = 1.0): the fit
refuses it, and only its message (`references = NaN`) was wrong.
