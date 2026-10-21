---
type: Finding
title: A reference model's Sortino ratio uses the SD of the losses alone and its Omega counts outcomes, which differ from the textbook ratios
description: "On the margin -3, -1, 0, 2, 4, 4 (mean 1), the textbook Sortino over the downside deviation sqrt(mean(min(x, 0)^2)) is 0.775 and the reference's, over the population SD of the negative values only, is 1.0; Omega as the ratio of partial moments is 2.5 and as the ratio of counts 1.5, the zero counting on neither side."
tags: [prospicio-prob, statistics, risk-ratios, sortino, omega, conventions]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T23:30:00Z }
verified: { by: process:ci, at: 2026-10-21T23:30:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/486
    title: "Issue #486: risk ratios on draws (Sharpe, Sortino, Omega, TVaR ratio)"
  - id: sortino
    resource: https://doi.org/10.3905/joi.3.3.59
    title: Sortino and Price (1994), Performance measurement in a downside risk framework, Journal of Investing 3(3)
  - id: test
    resource: ../crates/prospicio-prob/src/statistics.rs
    title: risk_ratios_under_each_convention_and_at_the_edges
---

# Sortino and Omega conventions

A reference simulation pricing model reports risk ratios on the
per-simulation margin. Its Sortino divides the mean by the population
standard deviation of the negative margins only, `sd(x[x < 0])`: the
spread of the losses about their own mean, not their size. The textbook
ratio (Sortino and Price) divides by the downside deviation about the
threshold `t`, `sqrt(mean(min(x - t, 0)^2))` over all simulations. Its
Omega counts, `#(x > 0) / #(x < 0)`; Keating and Shadwick's is the ratio
of partial moments, `mean(max(x - t, 0)) / mean(max(t - x, 0))`.

On `-3, -1, 0, 2, 4, 4` (mean 1):

| Ratio | Textbook | Reference |
|---|---|---|
| Sortino | `1 / sqrt(10/6) = 0.775` | `1 / sd(-3, -1) = 1.0` |
| Omega | `10 / 4 = 2.5` | `3 / 2 = 1.5` |

The reference's Sortino rises as the losses bunch together, whatever
their size: losses all equal give an SD of 0 and an infinite ratio, and
with one loss it is always infinite. With no loss it is undefined (NaN in
`RiskRatios`), where the textbook ratio is infinite.

`RiskRatios` (`prospicio_prob::statistics`) reports both, by
`Sortino::DownsideDeviation` (the default) or `BelowThresholdSd` and
`Omega::PartialMoments` (the default) or `Count`, with the Sharpe ratio on
the population SD and a TVaR ratio `(mean - t) / TVaR_q(-x)` (STARR) under
`TvarMethod`; the reference's TVaR ratio was not identified.
