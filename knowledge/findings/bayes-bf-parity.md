---
type: Finding
title: The Bayesian Bornhuetter–Ferguson against Stan, EVW (2012) and its limits
description: BayesBf matches the same model in Stan on GenIns and on EVW's triangle at every one of 200 seeds within tolerances of at least 4.7 times its own Monte Carlo error, reproduces EVW's printed totals, and moves from BayesOdp's reserve (flat prior) to EVW's strong-prior limit (tight prior); a tight prior needed the mode finder fixed, and a flat prior beside a very vague pattern prior gives divergences.
tags: [bayes, reserving, bornhuetter-ferguson, stan, validation, evw]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T23:30:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_bf.rs
    title: The parity tests and the seed sweep
  - id: bf
    resource: ../crates/prospicio-reserving/src/bayes/bf.rs
    title: prospicio_reserving::bayes::BayesBf and its unit tests
  - id: laplace
    resource: ../crates/prospicio-reserving/src/bayes/laplace.rs
    title: The mode finder
  - id: script
    resource: ../validation/scripts/reserving_bayes_bf_stan.py
    title: The Stan reference generator
  - id: py
    resource: ../python/tests/test_reserving_bayes_bf.py
    title: The Python binding tests
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 9 and 10)
---

# Finding

* Stan parity, 344 rows of `reserving_bayes_bf_stan.csv` (four cases:
  GenIns with premium at apriori 0.5 and cv 0.2; EVW's triangle at cv 0.1
  with `development_sd` 10 and 100, and at cv infinite with 100). Our
  seed-to-seed sd over seeds 0 to 199, with `s` Stan's posterior sd: at
  most `0.021 s` on a mean (`beta[10]`), 2.75% on an sd (`beta[10]`) and
  `0.063 s` on a percentile (the 2002 reserve's 95%), alike in all four
  cases; no heavy tail (the largest error is 0.69 of a tolerance at 15%
  on the sds). Tolerances `0.1 s`, 15% and `0.3 s`; a first 10% on the
  sds was only 3.6 times the largest spread. No fit of the 1,000 in the
  sweep had a divergence or a diagnostic warning.[^test][^script]
* The limits, unit tests on GenIns:[^bf] a flat prior (cv infinite)
  gives `BayesOdp`'s reserves by origin and in total within `0.1 s` and
  10%, though the two vague priors differ; a tight prior (cv 0.001, with
  `development_sd` 100 and no process error) gives EVW's closed-form
  strong-prior mean and sd by origin and in total within `0.1 s` and 10%;
  with the prior ultimates set to the chain ladder's, that limit is the
  classical Bornhuetter–Ferguson reserve (equal, then, to the chain
  ladder's); at apriori 0.5 (above GenIns' chain-ladder loss ratio of
  about 0.45) the total mean reserve rises monotonically as cv falls
  through infinity, 0.5, 0.2, 0.05 and 0.001.
* A tight prior broke the mode finder: Newton compared `f` to accept a
  step, and where the gain a step promised was below the rounding of `f`
  (a weak direction, the last column's `beta`, beside the stiff prior on
  `mu`), it accepted or halved steps of about 2e-8 at random and never
  converged. Two fixes: `BayesBf` samples `u = alpha - log m`, so the
  prior terms are near 0 and `alpha` keeps its precision near
  `log m` (about 15, where a step rounds to 2e-15); and `mode` takes such
  steps in full and stops once they no longer halve. `BayesOdp`'s modes
  on GenIns and RAA, with and without priors, are bitwise
  unchanged.[^laplace]
* Limitation: with a flat prior on `alpha` (cv infinite) the direction
  `(alpha + k, beta - k)` is held only by the normal prior on `beta`, and
  its length is about `development_sd / sqrt(m)`. At `development_sd`
  1,000 (about 300) NUTS on the Laplace-whitened scale gave 195 to 247
  divergent transitions in 4,000 on GenIns and EVW's triangle, at target
  acceptance 0.95 and 0.99 alike; at 100 or 10 none, and none at any
  finite cv tried (1, 0.1). The diagnostics report it; the EVW
  comparisons therefore use `development_sd` 100.[^test]
* Through the bindings (release build), with 2,000 simulations and the
  default chains on EVW's triangle, over seeds 0 to 29: the cv 0.1 total
  mean sits +0.26% above EVW's printed 6,145,526 (sd 0.15%, the default
  `development_sd` 10's bias); a flat prior with `development_sd` 100
  -0.03% from EVW's 6,049,398 (sd 0.17%) and -0.27% from `BayesOdp`'s
  (sd 0.23%); cv 0.001 +0.27% from EVW's closed form 6,644,047 (sd
  0.16%); every seed within 0.7% of its target. The Python and R tests
  hold 1.5% (2% against `BayesOdp`). The eight Python tests take about
  5 s.[^py]
