---
type: Finding
title: Stream scheme and samplers, two records in a result's provenance
description: Since 2026-10-08 a seeded result records the sampler versions (gamma = marsaglia-tsang/2026-10) apart from its stream scheme (chacha20/sim-index/v1); join's seed-reuse check compares seed and scheme only, replay also compares samplers, and results saved earlier carry no samplers, so whether their Gamma draws replay cannot be told from the file.
tags: [probability, rng, provenance, reproducibility, arrow, gamma]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T20:00:00-07:00 }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/provenance.rs
    title: Provenance::samplers, SAMPLERS, shares_streams, replays_same_draws and their tests
  - id: join
    resource: ../crates/prospicio-prob/src/portfolio.rs
    title: PredictiveDistribution::join and the test join_refuses_a_shared_seed_across_a_sampler_change
  - id: ipc
    resource: ../crates/prospicio-prob/src/ipc.rs
    title: Arrow IPC format version 1, the optional samplers key
  - id: fixture
    resource: ../validation/tests/predictive.rs
    title: The pyarrow fixture, written before the split, reads with samplers not recorded
  - id: rng
    resource: ../docs/design/rng.md
    title: Design note, RNG streams, stability policy, stability log and open question 4
  - id: derived
    resource: ../crates/prospicio-aggregate/src/monte_carlo.rs
    title: EventSet records samplers only for years drawn here; blends and towers record their inputs' samplers (predictive.rs, reinsurance.rs)
---

# Finding

A simulated result's draws depend on two separate things: which uniforms
each simulation draws (the stream scheme, `chacha20/sim-index/v1`) and how
the uniforms become draws (the samplers). The Gamma's change to
Marsaglia–Tsang on 2026-10-08 changed the second and not the first, which
showed that one label could not serve both uses:[^rng]

* `PredictiveDistribution::join` refuses two independent parts with the
  same seed and scheme, since they share uniforms. Bumping the scheme for
  a sampler change would let a part drawn before the change and one drawn
  after through on the same seed.
* Replaying a result needs the same samplers too, so a result that only
  names the scheme cannot say whether it replays.

Since 2026-10-08 (the same day, after the Gamma change) `Provenance` has
both:[^code]

* `stream_scheme`, unchanged.
* `samplers`: `(family, sampler id)` pairs sorted by family, the build's
  table `provenance::SAMPLERS` as `Provenance::seed` records it, today
  `[("gamma", "marsaglia-tsang/2026-10")]`. A family not listed draws by
  its first sampler (inverse transform for a distribution, the documented
  method for a copula's frailty). The table is every sampler the draws
  may have used, not only those they did use: a closure passed to
  `simulate` can draw from anything, and a list that missed a family
  would claim a replay that fails. So a GLM's Gamma noise, drawn by
  `Gamma::quantile`, is recorded under `gamma = marsaglia-tsang/2026-10`
  too.
* The `gamma` entry is the Gamma distribution's sampler. The Student t
  and Clayton copulas draw their Gamma variates with the same function,
  but they were Marsaglia–Tsang before 2026-10-08 as well (the function
  moved from `copula.rs` to `gamma.rs`), so they belong to the copulas'
  documented method and a record without `gamma` does not make them
  inverse transform.

`Provenance::seed` records the build's table, so it suits only draws this
build made. A result computed from draws it did not make records theirs
instead:[^derived] a tower applied to a distribution copies its source's
record (`Provenance::draws_from`); a blend, which draws only the choice of
model, keeps its models' samplers when they all agree and records none
otherwise; and an `EventSet` from `from_years` (a catastrophe model's
event loss table) records none, while one this build simulated
(`simulate_events`, a risk profile) records the table. Without this,
two event loss tables on the default seed 0 would have equal records and
`replays_same_draws` would call them the same draws.

`Provenance::shares_streams` compares seed and scheme and is what `join`
uses; `Provenance::replays_same_draws` also compares the samplers. Parts on
the same seed drawn before and after a sampler change, or one read from an
old file, are still refused by `join`.[^join]

# Saved files

The Arrow IPC format stays version 1: `samplers` is an optional key of the
provenance JSON (`[["gamma", "marsaglia-tsang/2026-10"]]` or `null`).
Readers look keys up by name and ignore the others, so an older reader
reads a new file, and a missing key reads as `None`, "not
recorded".[^ipc] The pyarrow fixture
`validation/reference/predictive_distribution_v1.arrow` has no key and
reads that way.[^fixture] Readers reject a `samplers` array whose
families are not strictly increasing, since the check compares the pairs
in order and a file from another writer could otherwise hold the same
samplers in another order and never match.[^ipc]

A result made before the split says `chacha20/sim-index/v1` whether its
Gamma draws came by inverse transform (before 2026-10-08) or by
Marsaglia–Tsang (on 2026-10-08, before the split). Nothing in the file
tells them apart, so `replays_same_draws` never matches a record without
samplers; to replay such a result, compare the draws themselves.

Not yet run by CI.

[^code]: crates/prospicio-prob/src/provenance.rs
[^join]: crates/prospicio-prob/src/portfolio.rs, `join_refuses_a_shared_seed_across_a_sampler_change`
[^ipc]: crates/prospicio-prob/src/ipc.rs, `provenance_without_samplers_reads_as_not_recorded`
[^fixture]: validation/tests/predictive.rs, `reads_pyarrow_fixture`
[^derived]: crates/prospicio-prob/src/predictive.rs, `blend_records_its_inputs_samplers_not_the_builds`; crates/prospicio-aggregate/src/monte_carlo.rs, `years_from_elsewhere_record_no_samplers`; crates/prospicio-aggregate/src/reinsurance.rs, `apply_records_the_samplers_only_of_events_drawn_here` and `apply_aggregate_keeps_the_input_streams_and_samplers`
[^rng]: docs/design/rng.md, stability policy, stability log and open question 4
