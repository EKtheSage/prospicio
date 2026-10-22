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
  500,829, 516,247 and 505,205 against 514,073, 640,309 and 703,356 under
  independent priors (ratios 0.97, 0.81, 0.72), 95% widths 1.97, 2.02 and 1.97
  million against 2.01, 2.48 and 2.73 million. The ten origins identify `s`
  (sd about 0.08), after which each origin is held to `m_i s` with the loss
  ratio's own cv, where independent priors give each origin the whole marginal
  cv and no way to learn from the others; the premise of #518, that
  independent priors understate the total's interval, holds for the prior, not
  for the posterior of a data-rich triangle. Against a known share the common
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
