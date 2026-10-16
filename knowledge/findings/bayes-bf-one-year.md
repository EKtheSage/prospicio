---
type: Finding
title: The Bayesian Bornhuetter-Ferguson's one-year view between the chain ladder and the strong-prior limit
description: Reweighting BayesBf's posterior by each simulated year with PSIS gives a claims development result whose sd matches BayesOdp's one-year view under a flat prior and the closed form of EVW's strong-prior limit under a tight one (GenIns, total ratios 1.010 and 1.008 over 40 seeds); in between the total's sd falls to 70% of the flat prior's at cv 0.2 and 50% at cv 0.05, and the generic engine's reweighting sits within 0.06 sd of refits where Pareto k is at most 0.7.
tags: [bayes, reserving, bornhuetter-ferguson, one-year, psis, importance-sampling, genins, conformance]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_bf_one_year.rs
    title: BayesBf's one-year view against BayesOdp's and the strong-prior limit, and the seed sweep
  - id: conformance
    resource: ../validation/tests/reserving_bayes_one_year_conformance.rs
    title: The release half of the one-year conformance suite, on BayesOdp and BayesBf
  - id: engine
    resource: ../crates/prospicio-reserving/src/bayes/one_year.rs
    title: The generic one-year engine (YearModel, OdpYear) and the unit half of the suite
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 17)
---

# Finding

`BayesBf::one_year` runs the one-year view of `BayesOdp` (decision 16)
on the Bayesian Bornhuetter-Ferguson's means `mu_w gamma_d`, through the
generic engine of decision 17.[^design] On GenIns with premium and
apriori 0.5, 10,000 reweighted simulations, over seeds 0 to 39:[^test]

* **A flat prior is the Bayesian chain ladder.** Each origin's sd of the
  claims development result over `BayesOdp`'s averages 0.992 to 1.008 and
  the total's 1.010 (seed-to-seed sd 0.027).
* **A tight prior is the strong-prior limit.** At cv 0.001, with
  `development_sd` 100 and no process error, each `mu_w` sits at its
  prior and each `gamma_d` is `Gamma(S_d / phi, M_d / phi)`; a year moves
  `E[gamma_d]` to `(S_d + A_d gamma_d) / (M_d + A_d)`, so the CDR is
  linear in the `gamma_d` and its sd has a closed form. The simulated sd
  over it averages 0.999 to 1.009 per origin and 1.008 in total (sd
  0.025), without refits.
* **The prior narrows the view.** With Gamma process error, the total's
  sd at cv 0.2 is 0.697 of the flat prior's (sd 0.018) and at cv 0.05
  0.502 (sd 0.014): the year then informs the development pattern and
  hardly the origins' levels.
* **The reweighting is as close to refits as for BayesOdp.** On 300
  simulations at cv 0.2, the root mean square gap of a simulation's total
  CDR, reweighted against refitted, is 0.056 sd where Pareto k is at most
  0.7 (BayesOdp: 0.035 on RAA, 0.044 on GenIns) and 0.20 above it, with
  about 13% of the simulations flagged.[^conformance]

Not yet run by CI.
