---
type: Finding
title: The Bayesian ODP's one-year view by PSIS reweighting, against the ODP bootstrap and refits
description: Reweighting BayesOdp's posterior by each simulated year with Pareto smoothed importance sampling gives a claims development result whose sd matches the ODP bootstrap's one-year view on RAA and GenIns (total ratio 1.001 and 1.006 over 80 seeds); 14% (RAA) and 26% (GenIns) of the simulations have Pareto k above 0.7, the reweighting under-reacts there and leaves the sd about 1% low, and refitting the flagged simulations with NUTS costs about 39 ms each.
tags: [bayes, reserving, odp, one-year, psis, importance-sampling, bootstrap, merz-wuthrich, raa, genins]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-14T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_one_year.rs
    title: The comparison with the bootstrap and the refits, the seed sweep and the report
  - id: engine
    resource: ../crates/prospicio-reserving/src/bayes/one_year.rs
    title: The one-year engine (PSIS, refits) and its result types
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: prospicio_reserving::bayes::BayesOdp::one_year
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 16, open decisions 12 to 14)
---

# Finding

The one-year view of `BayesOdp` draws each simulation's year (the cells
valued in the twelve months after the valuation) from the posterior
predictive, reweights every posterior draw by the year's quasi
likelihood (`phi`, `c0` and the priors unchanged), smooths the weights
with PSIS and takes the closing reserve as the weighted posterior mean of
the expected run-off after the year; the claims development result is
the opening posterior mean reserve less the year's payment and that
closing reserve. Simulations whose Pareto `k` is above 0.7 are refitted
with NUTS by default.[^design]

* **Against the ODP bootstrap's one-year view** (chain ladder
  re-reserving, 50,000 simulations): over seeds 0 to 79 at 10,000
  simulations, reweighting only, the total's sd ratio averages 1.001 on
  RAA (seed-to-seed sd 0.026) and 1.006 on GenIns (0.021), and each
  origin's 0.98 to 1.02, except RAA 1982 (0.909) and 1983 (0.947), whose
  reserves ride on the skewed last development effects, where the
  Bayesian lifetime reserve already differs from the bootstrap's. The two
  share the process model and, under vague priors, the parameter
  error.[^test]
* **Against Merz–Wüthrich**: the ratio of sds runs from 0.50 (RAA 1990)
  to 4.5 (RAA 1982) per origin, 0.60 (RAA) and 1.33 (GenIns) in total,
  as the ODP bootstrap's does: the gap is the ODP's process model against
  Mack's ([the bootstrap's finding](/findings/one-year-bootstrap-vs-merz-wuthrich.md)).
* **Pareto k is often high.** At 4,000 posterior draws, 13.6% of RAA's
  simulations and 26.4% of GenIns' have `k` above 0.7 (mean `k` 0.38
  and 0.56, largest about 6): a year doubles the data on the newest
  origin's effect and on the last development effect, each of which
  rests on one cell, so the posterior moves far in a few directions.
* **The reweighting under-reacts where k is high.** On 2,000
  simulations run both ways on the same years, the root mean square gap
  of a simulation's total CDR, reweighted against refitted, is about
  0.03 to 0.04 sd for `k` up to 0.5 (the two estimators' own Monte
  Carlo error), 0.06 for 0.5 to 0.7, 0.10 to 0.11 for 0.7 to 1 and 0.2
  to 0.3 above 1; it leans one way, so reweighting every simulation
  leaves the total's sd 1.0% (RAA) and 1.2% (GenIns) below refitting
  every one, and refitting only the flagged ones closes a fifth (GenIns) to two
  thirds (RAA) of that.
* **Cost** (release, 16 threads): reweighting 10,000 simulations takes
  0.3 to 1.2 s with the fit; a refit about 39 ms, so the default
  takes 54 s on RAA and 104 s on GenIns.
* **A year with no cells has no tail.** On a fully developed triangle
  (every origin at the last age) every log ratio is 0 and PSIS returns
  an infinite `k`, which flagged every simulation and refitted each one
  under the default; the year is now recognised as uninformative, with
  equal weights, `k` 0 and no refit (a unit test checks it).

Not yet run by CI.
