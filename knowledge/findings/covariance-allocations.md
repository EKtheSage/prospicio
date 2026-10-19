---
type: Finding
title: "Two covariance allocations: the whole measure or only the margin"
description: "The covariance allocation is defined two ways. Scaling the variance's Euler allocation to rho(S) gives rho(S) Cov(X_j, S) / Var(S), which spreads the expected loss by covariance too; Kalkbrener's allocation with the standard deviation as auxiliary measure, the one Pricing Insurance Risk uses (Exercise 217), gives E[X_j] + (rho(S) - E[S]) Cov(X_j, S) / Var(S). They share the shares Cov(X_j, S) / Var(S) and agree only when E[S] = 0 or every mean is in proportion to its covariance; on the Simple Discrete Example at a premium of 80 the first unit gets 1.086 or 5.216."
tags: [prospicio-prob, capital-allocation, covariance, pricing-insurance-risk]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T07:00:00Z }
verified: { by: process:ci, at: 2026-10-19T07:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/capital.rs
    title: AllocationMethod::Covariance and CovarianceMargin
  - id: test
    resource: ../validation/tests/distributions.rs
    title: covariance_margin_matches_pricing_insurance_risk_exercise_217 and haircut_and_equal_risk_match_pricing_insurance_risk_exercises_205_206
  - id: book
    resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
    title: Mildenhall and Major, Pricing Insurance Risk (2022), Exercise 217
---

# Facts

* `Covariance`: `rho(S) Cov(X_j, S) / Var(S)`. It sums to `rho(S)` and
  looks at the whole distribution, but it spreads the expected loss by
  covariance, so a unit nearly independent of the total can get less than
  its own mean.
* `CovarianceMargin`: `E[X_j] + (rho(S) - E[S]) Cov(X_j, S) / Var(S)`,
  Kalkbrener's allocation with the standard deviation as auxiliary
  measure. It also sums to `rho(S)` (the covariances sum to `Var(S)`).
* On *Pricing Insurance Risk*'s Simple Discrete Example (X1 in {0, 8, 10},
  X2 in {0, 1, 90}, weights 2, 1, 1 of 4, independent), at a premium of
  80: `Cov(X1, S) = 20.75` against `Var(S)` of about 1,528, so the first
  allocation gives X1 1.086 (a quarter of its mean 4.5) and the second
  5.216, the book's value (Exercise 217).
* The haircut and equal risk allocations (sections 12.2.3 and 12.2.4)
  allocate an exogenous total by stand-alone measures at one level. On the
  same example the premium of 80 by TVaR is 8 and 72 as a haircut at 75%
  (stand-alone TVaRs 10 and 90) and 9.56 and 70.44 at the equal risk level
  0.6796; `haircut` and `equal_risk` reproduce both. The EPD level is the
  ratio of `E[(X - a)+]` to the mean, and since the stop loss of draws is
  piecewise linear the assets at a ratio are exact.
* The Shapley value needs only a cost of each sub-portfolio, so
  `shapley_with` takes any closure of the sub-portfolio's draws (the
  book's Exercise 212 uses a standard deviation premium).
