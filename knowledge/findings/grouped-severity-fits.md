---
type: Finding
title: Grouped losses are fitted conditional on the range of the bands
description: "Counts in size bands (b_k, b_k+1] are multinomial with p_k = (S(b_k) - S(b_k+1)) / (S(b_0) - S(b_n)), which covers a deductible (b_0 > 0) and a truncation from above (finite b_n) at once. Grouped maximum likelihood and minimum chi-square both reach the Fisher information N sum grad p grad p' / p, so one covariance serves both. Empirical limited means are exact from band counts and totals, even under a policy limit at or above the last size, so the minimum limited-mean distance needs no midpoint approximation. Bahnemann's Ex 2.8 prints a minimum chi-square of 1.828 at parameters whose chi-square is 1.020; the minimum is 0.614 at (7.2857, 0.4570). prospicio's unit, Python and R tests reproduce Ex 2.7, 2.10, 2.11, P2.16, P2.39 and P2.43."
tags: [prospicio-prob, severity, grouped-data, chi-square, minimum-distance, bahnemann]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T17:00:00Z }
verified: { by: process:ci, at: 2026-10-19T17:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/grouped_fit.rs
    title: GroupedLosses, fit_grouped, fit_lev_distance and their tests
  - id: bahnemann
    resource: https://www.casact.org/sites/default/files/2021-02/02-Bahnemann.pdf
    title: Bahnemann, Distributions for Actuaries, CAS Monograph 2, sections 2.4 and 2.5
---

# Grouped losses are fitted conditional on the range of the bands

Grouped data are counts `O_k` in size bands `(b_k, b_{k+1}]` with no
individual amounts. prospicio treats them as multinomial with

`p_k = (S(b_k) − S(b_{k+1})) / (S(b_0) − S(b_n))`,

so a first bound above 0 is a deductible's truncation and a finite last
bound a truncation from above, with no separate option for either.
Bahnemann's Ex 2.11 (losses above a deductible) reproduces only under this
conditioning.

- **Two objectives, one covariance.** `fit_grouped` maximizes `Σ O_k ln p_k`
  or minimizes Pearson's `Σ (O_k − N p_k)² / (N p_k)`. Both are efficient
  for grouped data, with the inverse of `N Σ ∇p_k ∇p_kᵀ / p_k` as their
  asymptotic covariance, so `GroupedFit::covariance` is that inverse for
  either. On Ex 2.6's six bands the two lognormal fits differ in the fourth
  decimal (7.28508, 0.45640 against 7.28575, 0.45696).
- **Starts.** The losses placed at the bands' midpoints, the open top band
  censored at its lower bound, give a `fit_severity` start close enough
  that Nelder–Mead on the unconstrained scale converges for every family.
- **Exact limited means.** With each band's count and total, the empirical
  `E[X; x]` at a band bound is exact: the total below `x` plus `x` times the
  count above. It stays exact under a policy limit at or above the last
  size, so `fit_lev_distance` reads censored data as is. Bahnemann's P2.39
  uses midpoints instead of totals, and reproduces with them.
- **A printed fit that is not the minimum.** Ex 2.8 prints a minimum
  chi-square lognormal (7.274670, 0.442525) with 1.828; at those
  parameters the statistic is 1.020 (the fitted table itself reproduces),
  and the minimum is 0.614 at (7.2857, 0.4570), p-value 0.893 on 3 degrees
  of freedom. The test asserts the true minimum and that the printed
  parameters do worse.
