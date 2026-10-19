---
type: Reference Implementation
title: Averages over the latest periods and excluding high and low in chainladder-python and R ChainLadder
description: chainladder-python 0.10.1's Development(n_periods, drop_high, drop_low, preserve) and R ChainLadder 0.2.21's MackChainLadder(weights = W) are the parity references for prospicio_reserving's Development n_periods, drop_high and drop_low; n_periods counts calendar diagonals, the drops rank within them with ties by the starting value and are skipped where they would leave no link ratio, and the exclusions enter the factor, sigma and volume alike. Neither library has a geometric average.
resource: https://github.com/casact/chainladder-python
tags: [reserving, development, chain-ladder, mack, parity, python, r, averages]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-18T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 13, averages over the latest periods, excluding high and low, and geometric
  - id: python
    resource: ../validation/scripts/reserving_averages_python.py
    title: chainladder-python reference generator for the averages
  - id: r
    resource: ../validation/scripts/reserving_averages_r.R
    title: R ChainLadder reference generator for Mack under exclusion weights
  - id: test
    resource: ../validation/tests/reserving_averages.rs
    title: Validation test of the averages against chainladder-python, R and the books
---

# Use

`Development(n_periods, drop_high, drop_low)` in chainladder-python is the
reference for the factors, sigmas and chain-ladder ultimates of
`prospicio_reserving::Development` with `n_periods`, `drop_high` and
`drop_low`, on RAA and GenIns with the volume, simple and regression
averages, and for Bornhuetter-Ferguson and Cape Cod totals on the CAS
workers' compensation triangle.[^python] R ChainLadder's
`MackChainLadder(Triangle, weights = W)`, with `W` the 0/1 matrix of the
kept link ratios built by the script, is the reference for Mack's standard
errors under the same exclusions.[^r] Both agree with prospicio at a
relative 1e-9.[^test] Read and checked on 2026-10-18.[^design]

# Behaviour

* **`n_periods` is calendar time.** chainladder-python keeps the cells
  whose valuation is on or after the `n * offset + 1`-th latest valuation,
  where `offset` is the number of development periods in an origin period
  (`val_offset`). On a complete annual triangle that is the link ratios
  ending on the latest `n` diagonals, the same as the latest `n` origins at
  each age; on a sparse or mixed-grain triangle the two differ.
  `n_periods` at or past the number of diagonals uses every link ratio.
* **`w_` counts one cell too many.** Under `n_periods`, the weights `w_`
  include the starting cell on the latest diagonal, which has no link
  ratio. The regression ignores it, so the number of link ratios actually
  used is the count with a weight above 0 and a finite ratio. prospicio's
  `DevelopmentFit::n_links` is that count.
* **The drops.** Link ratios are ranked by value with ties broken by the
  starting cell's value (`lexsort`), within those `n_periods` keeps. With
  the default `preserve = 1`, an age where the drops would leave no link
  ratio keeps them all and the library warns ("Some exclusions have been
  ignored"). This matches the books: Friedland's medial average of two
  values is their simple average, of one value the value itself.
* **Sigma follows the exclusions.** The factor, sigma (with `kept - 1`
  degrees of freedom) and the volume behind the standard error all use the
  kept link ratios only, in both libraries.
* **Standard errors where sigma is filled in.** At an age whose sigma is
  interpolated, chainladder-python divides it by the square root of the
  first origin's earlier value, not by the volume of the kept link ratios.
  prospicio uses the volume, as R does, so its parity test compares
  standard errors only where sigma is estimated.
* **R falls back to Mack's sigma rule.** In some settings R's
  `MackChainLadder` cannot extrapolate sigma log-linearly and switches to
  Mack's rule with a warning; the R script leaves those cases out and
  lists them in its header.
* **`n_periods = 1`.** chainladder-python computes the factors only and
  warns that there are too few degrees of freedom. In prospicio every sigma
  is then unestimable: the chain ladder and the expected-loss methods fit,
  and Mack is an error.
* **No geometric average.** Neither library has one. prospicio's
  `Average::Geometric` is checked against Friedland's printed geometric
  average of the latest 4 years instead.

[^design]: Design note, reserving v0.2, decision 13, averages over the latest periods, excluding high and low, and geometric
[^python]: chainladder-python reference generator for the averages
[^r]: R ChainLadder reference generator for Mack under exclusion weights
[^test]: Validation test of the averages against chainladder-python, R and the books
