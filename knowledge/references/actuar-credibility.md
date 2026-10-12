---
type: Reference
title: R actuar's credibility models
description: actuar 3.3.7's cm(), the reference for prospicio_pricing::credibility; with one level its "Buhlmann-Gisler" and "Ohlsson" methods are the same unbiased Bühlmann–Straub estimator, the collective mean is credibility-weighted, a negative between variance zeroes every factor, and Hachemeister's data come with it.
resource: https://cran.r-project.org/package=actuar
tags: [pricing, credibility, buhlmann-straub, parity, r]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-12T03:00:00Z }
sources:
  - id: actuar
    resource: https://cran.r-project.org/package=actuar
    title: actuar 3.3.7 (GPL), Dutang, Goulet and Pigeon; cm(), bstraub() and hierarc() in R/cm.R, R/bstraub.R, R/hierarc.R
  - id: script
    resource: ../validation/scripts/r_credibility.R
    title: r_credibility.R, the reference values for prospicio's credibility
---

# How cm() fits one level

* `cm(~state, data, ratios = , weights = , method = )` hands a one-level
  model to `bstraub()`. Its `method` takes `"Buhlmann-Gisler"` (the
  default), `"Ohlsson"` or `"iterative"`; the first two differ only in a
  hierarchy (Bühlmann–Gisler truncates each node's variance estimate at 0
  before averaging, Ohlsson does not), and with one level both map to
  `bstraub(method = "unbiased")`.[^actuar]
* The within variance is `Σ w_ij (X_ij - X̄_i)² / (n - I)`, counting only
  groups with weight; the unbiased between variance is
  `w (Σ w_i (X̄_i - X̄)² - (I - 1) s²) / (w² - Σ w_i²)`. The iterative
  (Bichsel–Straub) estimator starts from it, stops at a relative change
  below `√ε` or 100 iterations, and is skipped when all weights are equal
  or the unbiased value is not positive (then `a = 0`).[^actuar]
* `summary()`'s "Collective premium" is the credibility-weighted mean
  `Σ Z_i X̄_i / Σ Z_i`, not the weight-averaged one; with `a <= 0` the
  factors are 0 and it is the weight-averaged mean.
* Missing cells must be `NA` in the same places in ratios and weights;
  without weights, missing ratios are refused.

# Hachemeister's data

`data(hachemeister)`: five US states, twelve quarters of average claim
amounts (`ratio.1` to `ratio.12`) and claim counts (`weight.1` to
`weight.12`). Weighted, unbiased: collective premium 1,683.713, between
variance 89,638.73, within variance 139,120,026, factors from 0.728
(state 4) to 0.985 (state 1). Equal weights (Bühlmann): 1,671.017,
72,310.02 and 46,040.47, factor 0.9496 for every state.
prospicio_pricing::credibility matches every value at `1e-12`.[^script]

[^actuar]: `R/bstraub.R` and `R/hierarc.R` in the actuar 3.3.7 sources.
[^script]: `validation/scripts/r_credibility.R` and `validation/tests/pricing.rs`.
