---
type: Reference Implementation
title: Where chainladder-python's ValuationCorrelation departs from R ChainLadder
description: In chainladder-python 0.10.1, ValuationCorrelation(total=True) centres its range on the sum of the variances instead of E[Z] and compares that sum with it, so z_critical is False on every triangle (ABC included, where R rejects); total=False is Mack's 1997 per-diagonal binomial test; p_critical is 1 - ci with defaults 0.1 and 0.5. Read from the source and an offline run; no CI test checks these.
resource: https://github.com/casact/chainladder-python
tags: [reserving, mack, chain-ladder, assumptions, calendar-year, correlation, python, pitfalls]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T09:00:00+00:00 }
sources:
  - id: source
    resource: https://github.com/casact/chainladder-python/blob/master/chainladder/core/correlation.py
    title: chainladder-python 0.10.1, chainladder/core/correlation.py
  - id: agree
    resource: chainladder-python-correlation.md
    title: chainladder-python's ValuationCorrelation and DevelopmentCorrelation, the statistics that match R
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 27, item 8
---

# The departures

The statistics themselves match R ChainLadder.[^agree] These do not, and
none of them is a target in `reserving_assumptions_python.csv`, so CI does
not check them. They come from the source and from an offline run of
chainladder 0.10.1 on RAA, GenIns and ABC.

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
[^agree]: agree, the statistics checked in CI.
[^design]: design, decision 27, item 8.
