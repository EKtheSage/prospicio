---
type: Finding
title: The layer share's uncertainty widens a layer's Bayesian BF interval, the tight prior reaches the closed-form limit, and a share error common to all origins narrows the posterior total
description: LayerSharePrior feeds a layer's ground-up loss ratio times its share to BayesBfNormalised; a tight prior reaches decision 23's closed-form limit on layer values and a larger share cv widens the predictive interval, one share error common to all origins (a shared multiplier, the default since #518) widens the total's prior threefold but narrows its posterior against independent priors, and the excess pattern cannot enter the model's pattern.
tags: [reserving, excess-development, bayesian, bornhuetter-ferguson, prior]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-22T18:00:00Z }
verified: { by: process:ci, at: 2026-10-22T18:00:00Z }
sources:
  - id: code
    resource: ../validation/tests/reserving_bayes_excess_prior.rs
    title: The Rust checks of the prior, its tight limit and the widening
  - id: python-tests
    resource: ../python/tests/test_reserving_bayes_excess_prior.py
    title: The Python binding's tests
  - id: r-tests
    resource: ../R/prospicio/tests/test-reserving-bayes-excess-prior.R
    title: The R binding's tests
  - id: design
    resource: ../docs/design/reserving-excess.md
    title: Section 3 and open decisions 22 to 27
---

# What was found

* The prior on a layer's expected ultimate for origin `i` is a gamma with
  mean `E_i * ELR * s_i` and `cv^2 = (1 + cv_elr^2)(1 + cv_share^2) - 1`,
  the moments of a product of independent variables, exact to 1e-14 in the
  tests. `BayesBfNormalised` needs no change: the exposure column is the
  ground-up exposure times the share (`layer_exposure`), `apriori` is the
  ground-up loss ratio and `cv` is the combined one.
* On a layer that is a quarter of GenIns's paid values, a tight prior (both
  cvs 0.0005) puts the total reserve's posterior mean within 0.15% (ELR
  0.45) and 0.73% (ELR 0.55) of the closed-form strong-prior limit of
  `reserving-bayes.md` decision 23 applied to the layer values.
* With the share's cv at 0.001, 0.3, 0.6 and 1.2 (ELR 0.35, cv 0.1, a prior
  below what the data say) the total reserve's sd is 308,000, 514,000,
  640,000 and 703,000 and its mean 3.69 million, 4.28 million, 4.54 million
  and 4.66 million: the wider the prior, the more the data pull the
  ultimates up and the wider the interval.
* `BayesBfNormalised` has no prior or fixed value for the development
  pattern (the vague `development_sd` only), so an `ExcessPattern` cannot
  inform it (design note, decision 22, decided by the user on 2026-10-21;
  the share's cv is an input, decision 24, decided the same day). The gamma
  priors of PR 4 are independent across origins although one share error is
  common to all of them (decision 23, decided by the user on 2026-10-22 as a
  single shared share multiplier).
* **A share error common to all origins (#518).** `BayesBfNormalised` takes an
  optional multiplier `s` (gamma, mean 1, cv the share's), sampled with the
  other parameters and carried in each draw as `log_share`; each origin's
  prior is a gamma with mean `m_i s` and cv `cv_elr`. The marginal moments are
  unchanged and the origins' priors are correlated, so the total's prior sd on
  the layer is 3.12 million, 6.21 million and 12.4 million at share cvs 0.3,
  0.6 and 1.2, against 1.04 million, 2.01 million and 3.97 million (ratios
  2.99, 3.10, 3.12; closed form, and a 400,000-draw simulation of the generative
  law agrees). **The posterior total is narrower, not wider**: its sd is
  497,000, 505,000 and 509,000 at share cvs 0.3, 0.6 and 1.2 (means over 20
  seeds; a single seed gave 500,829, 516,247 and 505,205) against 514,000,
  639,000 and 710,000 under independent priors (10 seeds; ratios 0.97, 0.79,
  0.72). The ten origins identify `s`
  (sd about 0.08), after which each origin is held to `m_i s` with the loss
  ratio's own cv, where independent priors give each origin the whole marginal
  cv and no way to learn from the others. Independent priors understate the
  total's prior interval and overstate its fitted (posterior) interval once the
  data identify `s`; where they cannot (few origins, or all immature) the
  common model's total stays wider. **The common total's sd is flat in the
  share's cv, with a small real rise.** The single-seed values above (501k,
  516k, 505k) looked non-monotone, but that was seed noise: one fit's total sd
  varies by 5.5k to 8.2k (about 1.5%) from seed to seed. A 20-seed experiment
  (run by hand on 2026-10-22, not in CI; the layer 0.25 of GenIns paid,
  `LayerSharePrior::new(0.35, 0.1, &[0.25], cv)`, `BayesBfNormalised`
  defaults of 4 chains with 1000 warmup and 1000 draws, target accept 0.95,
  10,000 simulations, seeds 1 to 20) gives the mean total sd by share cv 0.3,
  0.6, 0.9, 1.2 and 1.5 of 497.4k, 505.0k, 506.2k, 509.0k and 508.3k (Monte
  Carlo standard error of each mean 1.2k to 1.8k) and the posterior mean of
  `s` of 1.298, 1.308, 1.311, 1.311 and 1.311 (sd 0.073 to 0.075). The sd rises
  about 2% from cv 0.3 to 1.5, about five standard errors, then saturates;
  `s` rises slightly and saturates, and does not fall, so the gamma's mode at
  0 for a cv above 1 does not pull `s` down on this data. The per-origin
  priors (10 seeds) give 514k, 639k, 698k, 710k and 730k at the same cvs. A
  lognormal prior on `s` with the same mean and cv gave 496k, 505k, 503k, 502k
  and 504k, within 0.5% of the gamma, so the two are indistinguishable here
  and decision 26 (gamma) stands. No divergences and R-hat at most 1.007 in
  either. Not tested: data where every origin is immature, or very few
  origins, where the prior matters. Against a known share the common
  error still widens the posterior sd by 65% (304,000 to 501,000), and then
  stops growing with the share's cv.
* On a layer generated with a known wrong share (0.35 of paid where the prior
  says 0.25, so `s` should be 1.4; `apriori_cv` 0.1, `share_cv` 0.5) the
  posterior of `s` has mean 1.419 and sd 0.080 and the total expected
  ultimate is 1.422 times the prior's, against 1.349 for independent priors.
  A common cv of 1e-4 is the plain model (total mean 3.692 million, sd 303,464
  against 3.689 million and 303,919). The one-year view conforms with the
  multiplier (GenIns, apriori 0.5, `cv` 0.2, `common_cv` 0.5, seed 0: the
  gap to the refits 0.053 sd of the total where `k̂` is at most 0.7 and 0.199
  above it, all 35 flagged simulations moment matched).
