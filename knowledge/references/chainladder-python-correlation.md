---
type: Reference Implementation
title: chainladder-python's ValuationCorrelation and DevelopmentCorrelation
description: chainladder-python 0.10.1's Mack assumption tests match R ChainLadder's statistics on RAA, GenIns and ABC, but ValuationCorrelation(total=True) centres its range on the sum of the variances instead of E[Z] and compares that sum with it, so z_critical is False on every triangle; total=False is Mack's 1997 per-diagonal binomial test; p_critical is 1 - ci.
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
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 27, item 8
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

# What does not

* `ValuationCorrelation(total=True)`'s `range` is
  `(sum(VarZ) + q sd, sum(VarZ) - q sd)` with `q = norm.ppf(0.5 - (1 -
  p_critical) / 2)`: centred on the summed variance instead of `E[Z]`
  (RAA: `(0.069, 7.888)` against R's `(8.966, 16.784)`), and `z_critical`
  compares the summed variance with that range, not `z`, so it is False
  on every triangle. On ABC, where R and prospicio reject (`Z = 8` outside
  `(12.22, 20.38)`), it reports no calendar effect.[^source]
* `total=False` gives each diagonal a two-sided binomial p-value,
  `min(1, 2 * binom.cdf(z, n, 0.5))`, against `p_critical` (Mack 1997's
  per-diagonal test); prospicio does not build it.[^design]
* `p_critical` is `1 - ci`: the defaults are 0.1 for the valuation test
  (90%) and 0.5 for the development test (50%), against R's 95% and
  50%.[^source]

[^source]: source, `ValuationCorrelation.__init__` and `DevelopmentCorrelation.__init__`.
[^script]: script, the rows of reserving_assumptions_python.csv.
[^design]: design, decision 27, item 8.
