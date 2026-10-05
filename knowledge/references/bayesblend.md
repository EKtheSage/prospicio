---
type: Reference Implementation
title: BayesBlend (Bayesian stacking reference)
description: Ledger Investing's BayesBlend 0.0.8 (MIT), whose Stan models act_bayes::stacking follows, including partial pooling and adaptive priors.
resource: https://pypi.org/project/bayesblend/
tags: [bayes, stacking, pooling, stan]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T09:42:58Z }
sources:
  - id: wheel
    resource: https://pypi.org/project/bayesblend/0.0.8/
    title: bayesblend 0.0.8 wheel (models.py, stan_files/hierarchical_stacking.stan, hierarchical_stacking_pooling.stan)
    author: team:ledger-investing
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
---

# Obtaining it

`pip download bayesblend --no-deps` gives the wheel with the Stan files;
no CmdStan is needed to read them. Licence MIT, so its models may be
followed directly (unlike GPL R packages, which this project uses only
for reference values).[^wheel]

# The models

* No pooling: `α ~ N(alpha_loc, (alpha_scale δ)²)`,
  `β ~ N(beta_loc, (beta_scale δ)²)`, separately for discrete and
  continuous covariates (act-bayes uses one slope prior for both).
* Pooling (`partial_pooling=True`): `β_mj ~ N(μ_m, (σ_m δ)²)` per
  covariate group, `μ_m ~ N(μ, (tau_mu δ)²)`, `μ ~ N(0, (tau_mu_global δ)²)`,
  `σ_m ~ N⁺(0, tau_sigma²)`. A scale of 0 removes a level.[^wheel]
* Adaptive priors: `δ = N^λ`, `λ ~ Exponential(lambda_loc)`, default rate
  4; otherwise `δ = 1`.
* Defaults: every scale 1; `CMDSTAN_DEFAULTS` is just 4 chains, so no
  raised `adapt_delta`.[^wheel]

# Quirks

* BayesBlend warns that partial pooling with fewer than 3 distinct
  covariates may not perform well; in act-bayes that case shows up as a
  funnel and divergences (see
  [stacking funnel](/findings/stacking-pooling-funnel.md)).
* Covariates: continuous ones divided by twice their standard deviation
  (Gelman 2008), discrete ones dummy-coded; act-bayes leaves that to the
  caller and takes the number of leading dummy columns.[^models-design]

[^wheel]: bayesblend 0.0.8 wheel
[^models-design]: Design note, models
