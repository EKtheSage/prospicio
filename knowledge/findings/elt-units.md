---
type: Finding
title: Event loss tables across units against aggregate 2.1.0
description: aggregate 2.1.0 reads an event loss table per unit and rolls a shared event's rows into one at the event's rate, summing means, maxima and correlated sds and adding independent sds in quadrature, so the portfolio's rate counts a shared event once; prospicio's UnitEventLossTables does the same and matches aggregate's (eltq's) analytic rate and severity moments to 1e-10 on two tables, rolled and unit by unit.
tags: [catastrophe, elt, units, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-10T00:00:00Z }
verified: { by: process:ci, at: 2026-11-10T00:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_elt_units.py
    title: validation/scripts/aggregate_elt_units.py (aggregate 2.1.0 with eltq 2.15.0)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, _loss_table.py (_elt_model, SHARED_EVENT_RULE rollup)
---

# Finding

* An ELT row is `event rate mean max sdi sdc`. Rows of one event on
  several units must carry the same rate (relative 1e-12; the first row's
  is canonical and rates are never summed or averaged), so the portfolio's
  rate counts a shared event once: two units with events {1, 2, 4} and
  {1, 3, 4} have rate 0.082, not 0.094.[^script]
* The default rule rolls a shared event into one row: means, maxima and
  correlated sds add, independent sds add in quadrature
  (`np.hypot.reduceat`). A rolled row of admissible betas is admissible
  (the rolled sd is at most the sum of the units', below the rolled
  beta's bound by Cauchy-Schwarz). The pended alternative couples the
  units by Burke's one-factor Gaussian copula.
* `aggregate` needs `eltq` (2.15.0 or later) for an ELT with secondary
  uncertainty; its analytic severity moments `ex1` to `ex3` and the rate,
  rolled and for each unit alone, match prospicio's beta moments to 1e-10
  on a two-unit and a three-unit table.

[^script]: validation/scripts/aggregate_elt_units.py, run 2026-11-09.
