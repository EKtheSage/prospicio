---
type: Finding
title: BayesBf's a-priori level barely moves the reserve; only the premium's shape across origins and the CV do
description: In EVW's Model 3.1 the pattern is not normalised, so scaling apriori by k is undone by scaling the pattern by 1/k; only the vague prior on the development effects sees the change. On RAA with a premium column, apriori 0.3, 0.6 and 1.2 give the same mean reserve within Monte Carlo error at CV 0.5, 0.2 and 0.05, while the CV moves it by 17%.
tags: [bayes, reserving, bornhuetter-ferguson, priors, evw, xll]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T06:30:00Z }
sources:
  - id: bf
    resource: ../crates/prospicio-reserving/src/bayes/bf.rs
    title: prospicio_reserving::bayes::BayesBf (the model and its priors)
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 9, apriori sets the row parameter)
  - id: normalised
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 23, the model with the prior on the ultimate)
---

# Finding

* `BayesBf`'s means are `mu_w gamma_d` with `mu_w ~ Gamma(shape 1/cv^2,
  mean apriori * exposure_w)` and `log gamma_d ~ N(0, development_sd^2)`,
  the pattern not normalised.[^bf] Substituting `mu_w = apriori * mu'_w`
  and `gamma_d = gamma'_d / apriori` leaves the likelihood unchanged and
  makes the prior on `mu'_w` a gamma with mean `exposure_w`, free of
  `apriori`; what is left is the prior on `log gamma'_d`, now
  `N(log apriori, development_sd^2)`, a shift of 0.7 against an SD of 10
  for a factor of 2. In the strong-prior limit the reserve
  `m_w * sum S_d / M_d` is exactly free of `apriori` (it cancels between
  `m_w` and `M_d`). The design note's "apriori sets the row parameter, not
  the ultimate" understates this: the level of `apriori` is nearly
  inert, and the prior acts through the premium's relative sizes and
  the CV.[^design]
* On RAA with a premium of 40,000 in 1981 rising by 2,000 a year (four
  chains of 1,000 and 1,000 draws, 10,000 simulations, seed 1, release
  build), the total mean reserve:

  | CV | apriori 0.3 | 0.6 | 1.2 |
  |---|---|---|---|
  | 0.5 | 62,582 | 62,197 | 62,016 |
  | 0.2 | 69,171 | 69,678 | 69,460 |
  | 0.05 | 73,186 | 72,836 | 73,309 |

  The chain ladder's is 52,135. The CV pulls the reserve toward the
  premium-weighted raw pattern, whatever the a-priori loss ratio.
* Consequence: a prior on the ultimate (premium times an a-priori loss
  ratio, with a CV), as a Bornhuetter-Ferguson user expects, needs a
  normalised pattern (Verrall 2004's model, or a prior on `mu_w sum
  gamma_d`), which `BayesBf` is not. The Excel add-in's `PRO.BAYES.BF` was
  held until the user chose between them (2026-10-19); it now wraps
  `BayesBfNormalised` (`docs/design/xll.md`, decision 38, 2026-10-22).
* Resolved 2026-10-21: the user chose a new model and left `BayesBf` at
  EVW parity. `BayesBfNormalised` has a normalised pattern and the gamma
  prior on the ultimate, and on the same RAA table its reserve follows
  `apriori` (cv 0.05: 41,288, 102,733 and 375,560 at apriori 0.3, 0.6 and
  1.2) where `BayesBf`'s stays at 73,000
  ([the new model's finding](bayes-bf-normalised.md)).[^normalised]
