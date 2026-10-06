---
type: Finding
title: Reinstatements pro rata as to time on simulated years
description: Pro rata as to time charges each event's used limit at the share of the year left after it; dating a year's i.i.d. losses with sorted uniform times in drawn order is exact, and one exhausting loss a year costs half the amount-only premium on average.
tags: [reinsurance, reinstatements, simulation, aggregate]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T22:30:00Z }
verified: { by: process:ci, at: 2026-10-06T22:15:18Z }
sources:
  - id: layer
    resource: ../crates/act-aggregate/src/reinsurance.rs
    title: act_aggregate::reinsurance (Layer::pro_rata_as_to_time, tests)
  - id: events
    resource: ../crates/act-aggregate/src/monte_carlo.rs
    title: act_aggregate::EventSet (with_times, with_uniform_times)
---

# Finding

* Pro rata as to time works event by event, not on the annual total. With
  `A_e` the layer loss at 100% after annual terms up to and including
  event `e` at time `t_e` (the fraction of the year elapsed), the premium
  is `premium / l × Σ_e (1 − t_e) Σ_k rate_k |[A_{e−1}, A_e] ∩ [k l, (k+1) l]|`.
  With every `t_e = 0` it is the amount-only premium. An annual
  deductible absorbs the first recoveries, so the events it absorbs use
  no limit and cost nothing.[^layer]
* A year's losses from a frequency-severity simulation are independent
  and identically distributed, so giving them `n` sorted uniform times in
  their drawn order has the same joint law as dating each at random and
  sorting. `with_uniform_times` does that from stream `2^63 + i` of the
  set's seed: the losses are unchanged and each year replays alone.[^events]
* One loss a year that uses up a whole limit, at a uniform time, costs
  `premium × rate × E[1 − t] = premium × rate / 2` on average, with
  variance `(premium × rate)^2 / 12`. 20,000 simulated years agree within
  four standard errors.[^layer]
* A compound distribution on a grid has no event times, so `on_grid` and
  `apply_aggregate` refuse a layer pro rata as to time.[^layer]

[^layer]: act_aggregate::reinsurance (Layer::pro_rata_as_to_time, tests)
[^events]: act_aggregate::EventSet (with_times, with_uniform_times)
