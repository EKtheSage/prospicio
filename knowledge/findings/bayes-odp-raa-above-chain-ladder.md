---
type: Finding
title: On RAA the Bayesian ODP's mean reserve is 4.4% above the chain ladder's
description: With the default vague priors (SD 10), CmdStan's posterior mean of RAA's total reserve is 54,444 against the chain ladder's 52,135 (+4.4%), and BayesOdp at its defaults gives 54,733 (within the reference's tolerance); on GenIns the gap is 0.7%. The mean of exp of an uncertain log-scale mean exceeds exp of its mean, and RAA's young origins and last development column are uncertain on the log scale.
tags: [bayes, reserving, odp, raa, chain-ladder, excel]
status: stable
verified: { by: process:ci, at: 2026-10-16T16:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-16T16:00:00Z }
sources:
  - id: reference
    resource: ../validation/reference/reserving_bayes_stan.csv
    title: CmdStan 2.39 fit of bayes_odp.stan on RAA and GenIns (4 x 25,000 draws)
  - id: test
    resource: ../crates/prospicio-xll/tests/bayes.rs
    title: PRO.BAYES.ODP against the Stan reference at its defaults
  - id: finding
    resource: bayes-odp-vs-odp-bootstrap.md
    title: The Bayesian ODP chain ladder against Stan, the chain ladder and the ODP bootstrap
---

# Facts

* The chain ladder's total reserve on RAA is 52,135. The same ODP model
  sampled in CmdStan (priors SD 10, Gamma process error) has a posterior
  mean total reserve of 54,444, 4.4% higher, with SD 18,772 and a 95%
  quantile of 89,037. `PRO.BAYES.ODP` at its defaults (seed 0) gives
  54,733, inside the reference's tolerance of four times its Monte Carlo
  error.
* On GenIns the gap is 0.7% ([the GenIns finding](bayes-odp-vs-odp-bootstrap.md)).
  The difference is convexity: a future cell's mean is `exp(c + a_w + b_d)`,
  and the posterior mean of an exponential exceeds the exponential of
  the posterior mean by about half the log-scale variance. RAA's young
  origins and its one-cell last development column are far less certain
  on the log scale than GenIns', so the posterior mean moves further
  from the chain ladder's plug-in estimate.
* With chains of 500 draws (seed 7) RAA's largest R-hat is 1.0107, just
  above the 1.01 threshold; at the defaults of 1,000 it is under it with
  no divergences on RAA and GenIns.
