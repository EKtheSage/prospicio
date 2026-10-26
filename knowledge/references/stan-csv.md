---
type: Reference Implementation
title: CmdStan's CSV output and posterior's diagnostics on it
description: CmdStan 2.36.0's sample CSV as read by prospicio_bayes::stan_csv (comment lines, warmup kept with save_warmup, sampler columns ending in __, column-major element names), and the conventions of posterior 1.7.0 and loo 2.10.1 that the external-draws parity reproduces (loo's r_eff defaults to 1, relative_eff on exp(log_lik)).
resource: https://mc-stan.org/docs/cmdstan-guide/stan_csv.html
tags: [bayes, mcmc, stan, diagnostics, parity, r]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-23T12:00:00Z }
verified: { by: process:ci, at: 2026-10-23T12:00:00Z }
sources:
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
  - id: fixture
    resource: ../validation/scripts/stan_csv_fixture.sh
    title: stan_csv_fixture.sh, the CmdStan run behind the fixture
  - id: reference
    resource: ../validation/scripts/r_stan_csv.R
    title: r_stan_csv.R, the posterior and loo reference
---

# Facts

* A CmdStan sample file is `#` comment lines holding the run's settings
  (`method = sample`, `num_warmup`, `save_warmup = true` or `1`, `thin`),
  one header line, then the draws; with `save_warmup` the warmup rows come
  first, then a `# Adaptation terminated` comment, then the kept draws,
  then timing comments. The warmup rows number `ceil(num_warmup / thin)`.[^fixture]
* Sampler columns end in `__` (`accept_stat__`, `divergent__`,
  `treedepth__`, ...); `lp__` also ends in `__` but posterior keeps it as
  a variable, so `prospicio_bayes::stan_csv` does too.[^models-design]
* Array elements are flattened column-major with dots: `theta.1.2` is
  posterior's `theta[1,2]`, and the first index runs fastest.[^models-design]
* `loo::loo.array` uses `r_eff = 1` unless one is passed; the reference
  passes `relative_eff(exp(log_lik))`, which is `posterior::ess_mean` of
  each observation's `exp(log_lik)` over the number of draws. Draws read
  from outside take that `r_eff` in `PosteriorDraws::loo`.[^reference]
* On the fixture (eight schools, centred, 4 chains of 250 draws after 150
  warmup, seed 20261023) R-hat, bulk and tail ESS, the quantiles and
  `loo`'s estimates, pointwise ELPD, `r_eff` and Pareto k match posterior
  1.7.0 and loo 2.10.1 at 1e-10 relative (184 values); the chains have 30,
  9, 4 and 33 divergences, and tau's R-hat is 1.15, so the fixture is a
  poorly mixed posterior by design.[^reference]
* `cmdstanpy.install_cmdstan` failed in the cloud session (the GitHub API
  answered 403); downloading the release tarball with curl and running
  `make build` worked.[^fixture]
