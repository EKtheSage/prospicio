---
type: Finding
title: Deductible forms against aggregate 2.1.0
description: The disappearing deductible y xs d1 to d2 pays k A + B with A the layer d2 - d1 xs d1, k = d2/(d2 - d1) and B the layer y - d2 xs d2, so its first three moments are exact from a severity's layer moments; aggregate 2.1.0's analytic moments of the disappearing, franchise and ordinary forms are matched to 1e-9 once divided by P(X > d1), since aggregate's layered severity is conditional on reaching the attachment.
tags: [deductible, franchise, layer, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T21:00:00Z }
verified: { by: process:ci, at: 2026-11-09T21:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_disappearing.py
    title: validation/scripts/aggregate_disappearing.py (aggregate 2.1.0's stats_df sev ex1 to ex3)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, _severity.py (_retention_payment, _retention_detachment)
---

# Finding

* `aggregate`'s retention family: `r(x)` is `d1` up to `d1`, `d1 (d2 - x) /
  (d2 - d1)` on `(d1, d2]` and 0 above; the payment is `min(x - r(x), y)`,
  so `y` is the largest payment. `d2 = d1` is the franchise and `d2 = inf`
  the ordinary deductible.
* The payment is `k A + B` with `A = min((x - d1)+, d2 - d1)`,
  `k = d2/(d2 - d1)` and `B = min((x - d2)+, y - d2)`; when `y <= d2` it is
  `k min((x - d1)+, y/k)`. Since `B > 0` only when `A` equals `d2 - d1`,
  `E[A^i B^j] = (d2 - d1)^i E[B^j]` for `j >= 1`, and the first three moments
  are exact from three layer moments each. The mean simplifies to
  `k (E[X ∧ d2] - E[X ∧ d1]) + E[X ∧ y] - E[X ∧ d2]`.
* `aggregate`'s layered severity is conditional on the loss exceeding the
  attachment, its count being of claims that reach the layer: `30 xs 50
  franchise` on a lognormal reports a mean of 30 and second moment 900,
  every claim that reaches 50 paying the limit. Divided by `P(X > d1)`,
  prospicio's unconditional moments match its analytic moments (computed
  by numerical integration) to 1e-9 on five disappearing cases (limits
  below, between and above `d2`, infinite; lognormal and gamma), two
  franchise and one ordinary case.[^script]

[^script]: validation/scripts/aggregate_disappearing.py, run 2026-11-09.
