---
type: Reference Implementation
title: chainladder-python expected-loss estimators
description: chainladder-python 0.10.1's ExpectedLoss, BornhuetterFerguson, Benktander and CapeCod, the reference for prospicio_reserving's expected-loss family; how its arithmetic and Cape Cod trend work.
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
  sample_weight * apriori`, summed over an explicit array of `n + 1`
  powers. It equals iterating `U = latest + p * U` from the expectation `n`
  times up to rounding; prospicio_reserving uses the same closed form, with `p^n`
  and the sum by repeated squaring, and agrees to 1e-9 relative on every
  reference row, including `n_iters = 100`.[^generator]
* Stepping `U = latest + p * U` one at a time is unsafe as a stopping
  rule: with negative development (`cdf < 1`, `p < 0`) the floating-point
  steps can end in a two-cycle, so "stop once nothing changes" never fires.
  With `cdf < 1/2`, `|p| > 1` and the ultimate diverges: on paid
  `[[100, 40], [100]]`, premium 100, `Benktander(apriori=0.5)` gives the
  second origin -35.9375 at `n_iters = 5` and 153.90625 at 6.
* The exposure (`sample_weight`) is a separate triangle; its examples pass
  `premium.latest_diagonal`. prospicio_reserving reads the latest observed value
  of a measure column of the same triangle instead.[^reserving-v02]
* On an incremental triangle, `premium.latest_diagonal` is the last
  increment. prospicio_reserving cumulates the exposure column with the losses,
  so a premium repeated on each age of an incremental triangle is counted
  once per age (premium `[250, 250]` at ages 12 and 24 gives exposure 500,
  against chainladder-python's 250).[^reserving-v02]
* An origin with no value on the valuation diagonal gets a NaN ultimate
  (from the chain ladder up) and is left out of `CapeCod`'s pooled loss
  ratio by a NaN-skipping sum. prospicio_reserving uses that origin's latest
  observed value and the cdf at its age, and pools it: on paid
  `[[100, 150, 165], [120], [130]]` with premium 300, 320, 340,
  chainladder-python's apriori is 0.582934 and the third ultimate 208.078,
  prospicio_reserving's 415 / 700 = 0.592857 and 209.407.[^reserving-v02]
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
  prospicio_reserving's `CapeCod` is the default `n_iters = 1`.
* With `decay = 0` each origin's apriori is its own chain-ladder loss
  ratio, so Cape Cod returns the chain ladder; prospicio_reserving tests this.
* None of the estimators validates its settings: a zero apriori, a decay
  above 1 or a trend of -1 or below are accepted. prospicio_reserving rejects
  them.

[^reserving-v02]: Design note, reserving v0.2
[^generator]: Expected-loss reference generator
