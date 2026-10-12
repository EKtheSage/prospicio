---
type: Reference Implementation
title: The Bayesian ODP chain ladder in Stan
description: validation/scripts/stan/bayes_odp.stan, the same model as BayesOdp run in CmdStan 2.39.0, is the parity reference; the plug-ins are passed as data, and RAA needs adapt_delta 0.99 to sample without divergences.
resource: ../validation/scripts/stan/bayes_odp.stan
tags: [bayes, reserving, stan, odp, validation, reference]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T00:00:00Z }
sources:
  - id: stan
    resource: ../validation/scripts/stan/bayes_odp.stan
    title: The Stan program
  - id: script
    resource: ../validation/scripts/reserving_bayes_stan.py
    title: The generator
  - id: csv
    resource: ../validation/reference/reserving_bayes_stan.csv
    title: The committed reference summaries
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 3 and 10)
---

# What it is

* R ChainLadder and chainladder-python have no MCMC reserving model, so
  the reference for `prospicio_reserving::bayes::BayesOdp` is the same
  model written in Stan: `eta = c + a[w] + b[d]` with `a[1] = b[1] = 0`,
  `target += (dot_product(x, eta) - sum(exp(eta))) / phi` (the quasi
  likelihood, which accepts negative increments where `poisson_log` would
  not), normal priors, and in generated quantities each origin's reserve
  from `gamma_rng(mu / phi, 1 / phi)`, the Gamma with mean `mu` and
  variance `phi mu` that the bootstrap draws too.[^stan]
* `phi` and `c0` are data, computed by the generator as `BayesOdp`
  computes them (the volume-weighted chain ladder's fitted increments, the
  Pearson scale on `N - p` degrees of freedom, the log of the mean observed
  increment). RAA's `phi` equals R ChainLadder's `BootChainLadder` value,
  983.635027030873, to 1e-9, which the generator asserts.[^script]
* Settings: 4 chains of 2,000 warm-up and 25,000 draws, `metric =
  "dense_e"`, `adapt_delta = 0.99`, seed 20261012. The generator fails if
  any R-hat is above 1.005 or any transition diverges. CI reads the CSV
  and never runs Stan.[^script]

# Quirks

* With `adapt_delta` 0.8, RAA gave 15 divergent transitions in 100,000
  even with a dense metric. RAA's `b[10]` rests on one increment of 172
  against `phi = 984` (shape 0.17), so its posterior is skewed with a steep
  exponential side, which no fixed metric whitens. 0.99 removes them; the
  posterior is unchanged.[^design]
* The reference's tolerances are set from Stan's posterior sd `s`: means
  `0.1 s`, sds 8%, reserve percentiles `0.15 s`, the oldest origin's
  reserve an exact 0.[^csv]
