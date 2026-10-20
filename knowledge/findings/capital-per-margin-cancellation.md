---
type: Finding
title: "The capital per unit of margin, (1 - g) / (g - S), cancels near S = 1; under the CCoC it is exactly 1 / r"
description: "Natural allocation converts each layer's margin to capital at (1 - g(S)) / (g(S) - S). Where S is within a few ulps of 1 (the low totals of a fine grid, whose probabilities are tiny), both differences cancel, and for the CCoC g(S) = (r + S) / (1 + r) can round to exactly 1, which read as a layer with no capital. On the gross Cat/Non-Cat case of Pricing Insurance Risk the lifted Cat capital was 197.62748 at the calibrated r = 0.09999999999999992 and 197.62785 at r = 0.1; computing 1 - g from 1 - S (Distortion::complement) gives 197.6276725024 for both, equal to the margin over r to 2e-13, and the PH capital is stable to 1e-11 under a last-bit change of rho."
tags: [prospicio-pricing, prospicio-prob, natural-allocation, distortions, numerics, pricing-insurance-risk]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-20T23:30:00Z }
verified: { by: process:ci, at: 2026-10-20T23:30:00Z }
sources:
  - id: test
    resource: ../validation/tests/pricing.rs
    title: ccoc_capital_is_the_margin_over_the_return_near_s_of_one
  - id: unit-test
    resource: ../crates/prospicio-prob/src/distortion.rs
    title: the_complement_is_one_minus_g_and_keeps_its_accuracy_near_one
  - id: guide
    resource: ../docs/guides/pir-price-allocation-practice.md
    title: the gross Cat/Non-Cat case, Tables 15.36 and 15.39
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/330
---

# The capital per unit of margin near S = 1

The natural allocation (`docs/design/risk.md`) gives each layer's margin
capital at `(1 - g(S)) / (g(S) - S)`. Both differences go to 0 as `S` goes
to 1, so computed from `g(S)` the ratio keeps only as many digits as `1 - S`
has above the rounding of `g`. On a 2^16-point grid the lowest totals have
`S` within an ulp or two of 1.

- **The CCoC case.** `g(S) = (r + S) / (1 + r)`, so the ratio is exactly
  `1 / r` for every `S < 1`, and each unit's capital is its margin over `r`.
  Computed from `g`, a layer with `S = 1 - 1e-16` could have `g` round to 1
  and read as a layer with no capital. Which layers did so depended on the
  last bit of `r`: the gross Cat/Non-Cat lifted Cat capital was 197.62748 at
  the calibrated `r = 0.09999999999999992` and 197.62785 at `r = 0.1`
  (issue #330, found by the Guides lane in Python against R).
- **The fix.** `Distortion::complement(s)` computes `1 - g(s)` from `1 - s`
  (closed forms for the CCoC, PH, dual, exponential, Wang, beta, the linear
  and log-linear caps, LEP, linear yield, the piecewise-linear convex
  envelope, mixtures and minima), and the ratio is `G / ((1 - S) - G)`. The
  Cat capital is 197.6276725024 at both values of `r`, the margin over `r` to
  2e-13; PH is stable to 1e-11 under a last-bit change of `rho`. The guide
  prints 197.63 either way.
- **What it does not change.** Premiums, the total capital and the
  distortions with a slope of 0 at 1 (TVaR, dual, Wang), whose ratio goes to
  0 near `S = 1`. Non-Cat's negative lifted capital (-45.58) is real: the
  lifted shares of the margin in the bottom layers add to zero across units
  but not unit by unit.
