---
type: Reference Implementation
title: chainladder-python expected-loss estimators
description: chainladder-python 0.10.1's ExpectedLoss, BornhuetterFerguson, Benktander and CapeCod, the reference for act_reserving's expected-loss family; how its arithmetic and Cape Cod trend work.
resource: https://github.com/casact/chainladder-python
tags: [reserving, bornhuetter-ferguson, cape-cod, benktander, parity, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T12:00:00Z }
sources:
  - id: reserving-v02
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2
  - id: generator
    resource: ../validation/scripts/reserving_expected_loss_python.py
    title: Expected-loss reference generator
---

# Facts

* All four are one class: `ExpectedLoss` is `Benktander` with
  `n_iters = 0` and `BornhuetterFerguson` with `n_iters = 1`. The ultimate
  is the closed form `sum(p**k, k < n) * latest + p**n * expectation`,
  `p = 1 - 1/cdf` at the origin's latest age and `expectation =
  sample_weight * apriori`. It equals iterating
  `U = latest + p * U` from the expectation `n` times; act_reserving
  iterates and agrees to 1e-9 relative on every reference row, including
  `n_iters = 100`.[^generator]
* The exposure (`sample_weight`) is a separate triangle; its examples pass
  `premium.latest_diagonal`. act_reserving reads the latest observed value
  of a measure column of the same triangle instead.[^reserving-v02]
* `CapeCod` weights origin `j` in origin `i`'s apriori by
  `decay ** abs(i - j) * sample_weight[j] / cdf[j]` and averages
  `latest[j] * trend_factor[j] / (sample_weight[j] / cdf[j])`; the
  `detrended_apriori_` divides by origin `i`'s own trend factor, and the
  ultimate is Bornhuetter–Ferguson with apriori 1 on
  `sample_weight * detrended_apriori_`. `apriori_` is the trended value.
* The trend factor is `(1 + trend) ** (months / 12)`, `months` from the
  last month of the origin period to the triangle's `valuation_date`
  (`Triangle.trend(axis="origin")`), clipped at 0. For annual origins
  valued in December, origin `y` gets `(1 + trend) ** (valuation_year - y)`.
  Because each origin is detrended by its own factor, the ultimates do not
  depend on which valuation the trend runs to, only the `apriori_` do.
* `CapeCod` also takes `n_iters` (Benktander on the Cape Cod expectation);
  act_reserving's `CapeCod` is the default `n_iters = 1`.
* With `decay = 0` each origin's apriori is its own chain-ladder loss
  ratio, so Cape Cod returns the chain ladder; act_reserving tests this.
* None of the estimators validates its settings: a zero apriori, a decay
  above 1 or a trend of -1 or below are accepted. act_reserving rejects
  them.

[^reserving-v02]: Design note, reserving v0.2
[^generator]: Expected-loss reference generator
