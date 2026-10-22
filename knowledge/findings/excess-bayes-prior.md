---
type: Finding
title: The layer share's uncertainty widens a layer's Bayesian BF interval, and the tight prior reaches the closed-form limit
description: LayerSharePrior feeds a layer's ground-up loss ratio times its share to BayesBfNormalised with the cv of a product of independent variables; a tight prior reaches decision 23's closed-form limit on layer values and a larger share cv widens the predictive interval, while the excess pattern cannot enter the model's pattern and one share error is treated as independent across origins (the shared multiplier is a follow-up, #518).
tags: [reserving, excess-development, bayesian, bornhuetter-ferguson, prior]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T23:30:00Z }
verified: { by: process:ci, at: 2026-10-21T23:30:00Z }
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
    title: Section 3 and open decisions 22 to 24
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
  priors are independent across origins although one share error is common
  to all of them (decision 23, decided by the user on 2026-10-22 as a single shared share multiplier, to be built under issue #518), so until then the total's interval is understated.
