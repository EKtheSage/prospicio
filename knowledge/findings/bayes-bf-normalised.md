---
type: Finding
title: A Bayesian BF with a normalised pattern puts the prior on the ultimate, and its tight limit is the classical BF only at the chain ladder's loss ratio
description: BayesBfNormalised (means U_w p_d, the shares p_d summing to 1, a gamma prior on the ultimate U_w) makes apriori a real prior (on RAA with a premium the total reserve at cv 0.05 rises from 41,288 to 375,560 as apriori goes from 0.3 to 1.2, where BayesBf stays at 73,000). Its flat prior is the Bayesian chain ladder; its tight prior is the classical Bornhuetter-Ferguson reserve when the prior ultimates are the chain ladder's and otherwise differs (-8.6% to +28.4% for apriori 0.35 to 0.65 on GenIns), because the pattern is estimated with the ultimates.
tags: [bayes, reserving, bornhuetter-ferguson, priors, verrall, evw, xll]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T12:00:00Z }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_bf_normalised.rs
    title: The limits, the finding's table and the one-year view
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/bf_normalised.rs
    title: prospicio_reserving::bayes::BayesBfNormalised and its unit tests
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 23)
---

# Finding

* **The model.** Incremental values are over-dispersed Poisson with mean
  `U_w p_d`, `U_w = exp(alpha_w)` the origin's expected ultimate
  with a `Gamma(shape 1/cv^2, mean apriori x exposure_w)` prior and `p_d =
  exp(beta_d) / sum exp(beta)` the share of it paid at age `d` (`beta_1 =
  0`, a vague normal prior on the others). The prior is on the quantity the
  actuary's judgment is about, so `apriori` is the expected ultimate loss
  ratio. `BayesBf` (EVW's Model 3.1) puts it on `mu_w`, which is not the
  ultimate.[^model][^design]
* **The prior is real.** RAA with a premium of 40,000 in 1981 rising by
  2,000 a year, total mean reserve, `BayesBfNormalised` then `BayesBf`
  (10,000 simulations, seed 1, release build):[^test]

  | cv | apriori 0.3 | 0.6 | 1.2 |
  |---|---|---|---|
  | 0.5 | 48,381 / 61,905 | 70,521 / 62,661 | 105,494 / 62,229 |
  | 0.2 | 43,378 / 69,137 | 90,015 / 69,019 | 330,277 / 69,068 |
  | 0.05 | 41,288 / 73,049 | 102,733 / 73,026 | 375,560 / 73,171 |

  The chain ladder's is 52,135. The normalised model's reserve rises with
  `apriori` at every `cv` and follows it more closely as `cv` falls;
  `BayesBf`'s does not move with the level (the table of
  [bayes-bf-apriori-scale](bayes-bf-apriori-scale.md)).
* **The vague limit is the Bayesian chain ladder.** At `cv` infinite each
  origin's and the total's mean is within `0.1 s` and sd within 10% of
  `BayesOdp`'s on GenIns with premium and on RAA, and the mode's fitted
  means are the chain ladder's to 1e-4.
* **The tight limit is the classical BF only where the prior is the chain
  ladder's.** With the ultimates fixed at `m_w` the likelihood's shares are
  `p_d = S_d / (M_d + lambda)`, `lambda` making them sum to 1. At the chain
  ladder's own ultimates `lambda` is 0 and the reserve is
  `BornhuetterFerguson`'s to 1e-5 (the fit at `cv` 0.001 is 0.25% below it,
  18,633,756 against 18,680,856, the mean of the shares against their
  mode). At other loss ratios the pattern follows the ultimates and the
  reserve leaves the classical one, which keeps the chain ladder's pattern:
  GenIns with premium, `cv` 0.001, no process error, against
  `BornhuetterFerguson` at the same `apriori`: 0.35: -8.6%; 0.45: -1.1%;
  0.55: +10.2%; 0.65: +28.4% (the chain ladder's loss ratio is about
  0.45). A prior above the data's reads as slower development, so the
  reserve rises faster than the classical one's, which is linear in
  `apriori`.[^test]
* **EVW's triangle shows it plainly.** With EVW's prior ultimates as the
  exposure (`apriori` 1) the strong-prior limit is 8,987,282 (the shares'
  closed form applied to the future cells), against the classical
  Bornhuetter-Ferguson's 7,356,578, the chain ladder's 6,047,059 and
  `BayesBf`'s own limit of 6,644,047. The fit at `cv` 0.001 gives a mean
  about 2.5% below the limit (the mean of the shares against their mode;
  the Python and R binding tests check it within 4%). On GenIns with
  premium, `apriori` 0.4 against 0.6 moves the three newest origins'
  summed reserve by +50.3% under `BayesBfNormalised` and by 0.0% under
  `BayesBf` (`cv` 0.1, no process error).
* **Newton needed the shares' closed form.** From the raw pattern `S_d /
  M_d` the negative Hessian is indefinite when the prior ultimates are far
  above the data's, and the mode finder's ridge, scaled by the stiff prior
  curvature (`1/cv^2`), crawled and gave up after 200 iterations at `cv`
  1e-4 and `apriori` 0.6. Starting at `S_d / (M_d + lambda)` converges at
  `cv` down to 1e-6.[^model]
* **The one-year view** goes through the generic engine and passes its
  conformance suite; under a flat prior its CDR sd is `BayesOdp`'s and it
  narrows as `cv` falls.[^test]

[^model]: `crates/prospicio-reserving/src/bayes/bf_normalised.rs`
[^design]: `docs/design/reserving-bayes.md`, decision 23
[^test]: `validation/tests/reserving_bayes_bf_normalised.rs`
