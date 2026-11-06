---
type: Finding
title: HierarchicalBf's one-year view flags every simulation, matching rescues none, and reweighting alone is poor
description: A year of every program's diagonal moves the whole posterior of the stacked panel, so on the three-program reference case every one-year simulation has a Pareto k above 6, moment matching rescues none, and each simulation costs a NUTS refit of the panel of about 1.2 s.
tags: [reserving, bayesian, hierarchical, one-year, psis, moment-matching]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-05T22:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-reserving/src/bayes/hierarchical_bf.rs
    title: HierarchicalBf::one_year and its YearModel
  - id: code
    resource: ../validation/tests/reserving_bayes_one_year_conformance.rs
    title: The conformance suite on the reference panel (HIERARCHICAL_REFITS)
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, decision 30, "As built (PR 17)"
---

Measured on the three-program Stan reference panel (EVW, GenIns, RAA with
premium; 2 chains of 200 warm-up and 200 kept draws, release build, 16
threads, another job sharing the machine).

**Every simulation is flagged.** The Pareto k of 40 simulations at seed 0
ran from 6.2 to 56.6, so the reweighting by the year's quasi likelihood of a
posterior with 72 parameters is useless on its own: the year's three
diagonals move the industry level, both spreads and the shared pattern at
once. Moment matching rescued 0 of 40 (and 0 of 20 at each of seeds 0 to 2),
so the default `MomentMatchThenRefit` refits every simulation.

**Reweighting alone is poor.** Over seeds 0 to 2 (20 simulations each) the
root mean square gap between a reweighted and a refitted simulation's total
was 1.10, 0.13 and 0.08 sd of the total, and the reweighted total's mean
had z 11.8, 4.3 and 8.2 against a mean of zero (the conformance
suite's bound for the other models is 5). The opening reserve is the posterior mean of
the lifetime reserve to within z 2.0 at the three seeds.

**Cost.** One fit takes about 5.1 s at that sampler, and a refit about 1.2 s
a simulation on 16 threads (40 simulations in 55 s with the fit), so the
default view runs at most 2,000 simulations
(`HierarchicalBf::ONE_YEAR_SIMS`), as Meyers' and the compartmental models'.
At the default sampler (4 chains of 1,000 and 1,000, `target_accept` 0.99) a
refit costs several times more. A light refit (warm start, shorter sampler,
as `Compartmental`'s) is not built; decision 30 lists it as a follow-up.

**Limit.** One program with a tight hyperprior and spreads near 0 gives
`BayesBfNormalised`'s one-year view within 0.1 sd in the total and origins'
means and 10% in their sds, both under `ReweightOnly`.
