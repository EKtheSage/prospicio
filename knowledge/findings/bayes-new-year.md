---
type: Finding
title: A new underwriting year from the Bayesian BF is its prior unless a common level links it to the triangle, and Meyers' own prior on a new origin is too wide to price on
description: "BayesBfNormalisedFit::new_year draws U ~ Gamma(1/cv^2, mean s apriori premium) per posterior draw; without common_cv s is 1 and the new year is exactly the prior (GenIns, apriori 0.7, cv 0.25: mean 0.7014, cv 0.2496), with a vague common_cv the triangle sets the level (eight origins at 0.6 under apriori 0.9: new year 0.601). On a square triangle the pattern drops out and the new year is a gamma mixture over a one-dimensional posterior of s, which the simulation matches (cdf at five quantiles within 0.003). Meyers' models have no hyperparameter on the origin effects: the literal predictive N(0, 3.162^2) gives a log sd of 3.16, an exchangeable Student t level fitted to each draw's origins 0.26 to 0.29 on group 353; the held-out 1997 paid loss ratio lands at the 67th percentile."
tags: [bayes, reserving, bornhuetter-ferguson, meyers, pricing, new-year]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-03T12:00:00Z }
verified: { by: process:ci, at: 2026-11-03T12:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving, decision 27 (a new underwriting year) and open decision 19
  - id: test
    resource: ../validation/tests/reserving_bayes_new_year.rs
    title: Validation of the new year, run by CI's bayes job (slow-bayes)
  - id: code
    resource: ../crates/prospicio-reserving/src/bayes/new_year.rs
    title: prospicio_reserving::bayes::NewYear and the shared simulation
---

# What was found

* **The normalised BF learns a new year's level only through the common
  multiplier.** `apriori` and `cv` are settings and the origins' gamma
  priors are independent given `s` (decision 25), so a new origin's
  expected ultimate is `Gamma(1 / cv^2, mean s apriori premium)` given a
  posterior draw.[^note] Without `common_cv` (`s = 1`) the new year is
  exactly the prior whatever the triangle: on GenIns at `apriori` 0.7 and
  `cv` 0.25 the new year's loss ratio has mean 0.7014 and cv 0.2496 over
  100,000 simulations, and the same at `common_cv` 0.001.[^test] With
  `common_cv` 10 on eight origins of loss ratio 0.6 and a wrong `apriori`
  of 0.9, the new year's mean is 0.601 (the origins' posterior loss ratios
  average 0.598) and its cv 0.213, at least `cv` 0.2.
* **The gamma process on the ultimate is one gamma.** The cells' gammas
  have means `U p_d` and the common scale `phi`, and the shares sum to 1,
  so the ultimate given `U` is `Gamma(U / phi, scale phi)`, variance
  `phi U`; a unit test checks the new year's variance `(cv m)^2 + phi m`.
* **On a square triangle the new year is known exactly.** Every origin's
  shares sum to 1, so the pattern leaves the posterior of `(U, s)` and each
  `U_w` integrates out of its gamma: `p(s) ∝ s^(k-1) e^(-k s) prod_w (a /
  (m_w s))^a (a / (m_w s) + 1 / phi)^-(a + S_w / phi)`, `a = 1 / cv^2`,
  `k = 1 / common_cv^2`. At `cv` 0.3 and `common_cv` 0.5 the simulated new
  year's mean is 0.6102 against 0.6116 by quadrature and its sd 0.1943
  against 0.1951, and the exact cdf at its 5% to 95% quantiles is within
  0.0025 of the level. As `cv -> 0` the posterior of `s` tends to
  `Gamma(k + T / phi, rate k + M / phi)`; at `cv` 0.01 its mean is exact
  and its sd 4% narrower than the fit's, the origins' 1% freedom.
* **Meyers' prior on a new origin's effect is not usable for pricing.**
  `alpha_w ~ N(0, 3.162^2)` with no hyperparameter makes the literal
  predictive's log loss ratio sd 3.16 in CRC, CSR and CAY. Drawing the
  level from a Student t (`n - 1` degrees of freedom, scale `sqrt(1 + 1 /
  n)` times the sd of the draw's origin levels) gives 0.29 (CRC paid), 0.27
  (CSR paid) and 0.26 (CAY incurred) on group 353, with median loss ratios
  0.74, 0.70 and 0.73; both match the moments the posterior implies within
  0.001 in the mean and 1% in the variance.
* **A held-out year.** CRC on paid at calendar years up to 1996 puts the
  1997 origin's paid loss ratio at 120 months, 0.834, at the 67th
  percentile of its new year (median 0.741, 90% interval 0.452 to
  1.241).

[^note]: `docs/design/reserving-bayes.md`, decision 27 and open decision 19.
[^test]: `validation/tests/reserving_bayes_new_year.rs`.
