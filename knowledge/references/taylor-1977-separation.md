---
type: Reference
title: Taylor (1977), the separation method, and its worked example
description: Taylor's arithmetic separation method E[s_ij] = r_j lambda_{i+j} with sum r_j = 1, the backward recursion that solves it, its equality with a Poisson GLM on development and calendar factors (no origin factor, no offset), and the printed numbers of Example 1 (a motor account) that the reserving-inflation PRs reproduce.
resource: https://doi.org/10.1017/S0515036100011533
tags: [reserving, inflation, separation-method, calendar-year, glm, reference]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T12:00:00-08:00 }
verified: { by: process:ci, at: 2026-11-24T18:00:00-08:00 }
sources:
  - id: paper
    resource: https://doi.org/10.1017/S0515036100011533
    title: G. C. Taylor (1977), Separation of inflation and other effects from the distribution of non-life insurance claim delays, ASTIN Bulletin 9, 219-230
  - id: design
    resource: ../docs/design/reserving-inflation.md
    title: The design note that plans the separation method in prospicio-reserving (#979)
  - id: parity
    resource: ../validation/tests/reserving_separation.rs
    title: The CI test of Separation against Example 1 and the statsmodels Poisson GLM
---

# What it is

* Incremental payments of origin `i` in development period `j`, divided by
  a normaliser `n_i` (claims incurred in the origin period, which Taylor
  fixes at development period 0), are `s_ij` with `E[s_ij] = r_j
  lambda_{i+j}`: development proportions `r_j` summing to 1 over the
  triangle's ages and an index `lambda_h` per payment period.
* With column sums `v_j` and diagonal sums `d_h` of `s`, the estimates
  solve backward from the last period `k`: `lambda_h = d_h / (1 -
  sum_{j>h} r_j)`, then `r_h = v_h / sum_{m>=h} lambda_m`. The scanned
  PDF's OCR garbles equations (5) and (6); this form reproduces every
  printed number.
* Section 6 cites Verbeek (1972): the same estimates are the Poisson
  maximum-likelihood ones: the recursion's fitted values equal those of
  `glm(s ~ C(dev) + C(cal), Poisson)`, with no origin factor and no
  offset (the normaliser divides the data first), to 2.3e-13 on Example 1
  (offline) and to 1e-9 on GenIns and ABC in CI (`Separation`, against
  statsmodels 0.14.5).
* Future `lambda_h` are chosen by the user, not fitted (section 8); a
  constant rate gives a `(1 + K)^t` factor.
* The geometric method (least squares on `log s_ij`) is not in this paper.
  Its fit equals OLS of `log s_ij` on development and calendar factors
  (`SeparationKind::Geometric`, against statsmodels 0.14.5 to 1e-9 in
  CI); a full n by n triangle leaves `(n - 1)(n - 2) / 2` residual
  degrees of freedom, so the lognormal mean `exp(sigma^2 / 2)` needs
  n of at least 3.

# Example 1, a motor account (p. 228)

* `s_ij` by origin: 50.4, 28.2, 9.0, 4.8; 58.0, 29.2, 9.7; 59.5, 33.2;
  66.2 (the `n_i` are not printed).
* Printed: `r = 0.5835, 0.2878, 0.0866, 0.0421`; `lambda = 86.4, 98.9,
  102.0, 113.9`. The recursion gives 0.583476, 0.287767, 0.086615,
  0.042142 and 86.3789, 98.9391, 101.9984, 113.9.
* The 3 by 3 sub-triangle prints `r = 0.6101, 0.2980, 0.0921` and `lambda
  = 82.6, 94.9, 97.7`; the recursion gives 0.60993, 0.29796, 0.09212 and
  82.633, 94.946, 97.7, so the first proportion is off the print by 1.7e-4
  (Taylor's intermediate rounding, presumably).
* At 10% a year from `lambda_3` (`lambda_4..7 = 125.3, 137.8, 151.6,
  166.8`) the future cells print 5.3; 10.8, 5.8; 36.1, 11.9, 6.4.
* Example 2 (a pecuniary loss account) is too damaged in the OCR to use.
