---
type: Reference Implementation
title: chainladder-python's ValuationCorrelation and DevelopmentCorrelation
description: chainladder-python 0.10.1's ValuationCorrelation (total=True and total=False) gives R ChainLadder's z, E[Z] and Var[Z] on RAA, GenIns and ABC to 1e-12, and DevelopmentCorrelation gives R's T_k, weights, T, variance and 50% interval; checked in CI. Where it departs from R is a separate, unverified concept.
resource: https://github.com/casact/chainladder-python
tags: [reserving, mack, chain-ladder, assumptions, calendar-year, correlation, parity, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T09:00:00+00:00 }
verified: { by: process:ci, at: 2026-10-28T09:00:00+00:00 }
sources:
  - id: source
    resource: https://github.com/casact/chainladder-python/blob/master/chainladder/core/correlation.py
    title: chainladder-python 0.10.1, chainladder/core/correlation.py
  - id: script
    resource: ../validation/scripts/reserving_assumptions_python.py
    title: Generator of validation/reference/reserving_assumptions_python.csv
  - id: traps
    resource: chainladder-python-correlation-traps.md
    title: Where chainladder-python's ValuationCorrelation departs from R ChainLadder
---

# What agrees with R ChainLadder

On RAA, GenIns and ABC, to 1e-12:[^script]

* `ValuationCorrelation(tri, total=True)`: `z`, `z_expectation` and
  `z_variance` are R's `cyEffTest` `Z`, `E` and `Var`.
* `ValuationCorrelation(tri, total=False)`: `z`, `z_expectation` and
  `z_variance` by valuation are R's `test_table` rows, with one more row
  first, the diagonal of a single link ratio, always 0.
* `DevelopmentCorrelation(tri, p_critical=0.5)`: `t` (each `T_k`),
  `weights`, `t_expectation` (`T`), `t_variance` and
  `confidence_interval` are R's `dfCorTest` values and its `T_k`. It uses
  Mack's formula G4, `1 - 6 sum d^2 / ((I-k)^3 - (I-k))`, on average ranks,
  which equals R's Pearson on ranks only without ties.

`validation/tests/reserving_assumptions.rs` (`matches_chainladder_python`)
checks every row of `validation/reference/reserving_assumptions_python.csv`
against prospicio, so these statistics are confirmed in CI. The range,
`z_critical` and `p_critical` conventions, which differ from R's, are not
in that file and are described in a separate concept.[^traps]

[^script]: script, the rows of reserving_assumptions_python.csv.
[^traps]: traps, the range, z_critical and p_critical.
