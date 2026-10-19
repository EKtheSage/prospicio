---
type: Finding
title: The product-limit estimate survives a common bound from above, not per-loss ones
description: "Kaplan-Meier with left truncation (risk set r < x <= y) stays valid when every loss is also truncated from above at one bound U: it then estimates P(X > x | tau < X <= U), so a model is compared through (S(x) - S(U)) / (S(tau) - S(U)) and its quantiles at 1 - S(U) - s (S(tau) - S(U)). With bounds that differ by loss the risk sets mix populations and the estimate is biased; the right estimator is Efron and Petrosian's (1999) self-consistency NPMLE, not built, so prospicio's goodness_of_fit refuses such data while fit_severity's likelihood takes per-loss bounds."
tags: [prospicio-prob, severity, truncation, kaplan-meier, goodness-of-fit]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T16:00:00Z }
verified: { by: process:ci, at: 2026-10-19T16:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: goodness_of_fit and the test goodness_of_fit_conditions_on_a_common_bound
  - id: efron-petrosian
    resource: https://doi.org/10.1080/01621459.1999.10474187
    title: Efron and Petrosian (1999), Nonparametric methods for doubly truncated data, JASA 94
---

# The product-limit estimate survives a common bound from above, not per-loss ones

A loss truncated from above at `u` was kept only because it was at most
`u`: a study of claims up to 50,000, Bahnemann's P2.43. Its likelihood is
conditional on `r < X ≤ u`, so `fit_severity` replaces each `S(a)` in the
denominator, and in a censored loss's numerator, by `S(a) − S(u)`,
computed as `ln S(a) + ln(1 − S(u)/S(a))` so that two tiny tail
probabilities do not cancel.

The product-limit estimator is another matter. With left truncation its
risk set at `x` is the losses with `r < x ≤ y`.

- **One bound `U` for every loss.** All the losses come from the same
  population, `X` given `τ < X ≤ U`, and the estimator estimates
  `P(X > x | τ < X ≤ U)`. `goodness_of_fit` compares a model through
  `(S(x) − S(U)) / (S(τ) − S(U))` and reads its quantiles at
  `1 − S(U) − ŝ (S(τ) − S(U))`. The test draws a lognormal, keeps the
  draws at most 2,000, and finds a distance under 0.02 against the truth
  (over 0.2 without the bound).
- **Bounds that differ.** The risk set at a high `x` contains only the
  losses whose bound lies above `x`, but the estimate has already been
  pulled down by losses that could never have reached `x`. It is biased.
  The nonparametric estimator for doubly truncated data is Efron and
  Petrosian's self-consistency algorithm, which is not built, so
  `goodness_of_fit` refuses such data, while `fit_severity` takes it.
