---
type: Finding
title: The tower's simulated paths ran 17 to 38 times faster on 16 threads with the same bits, once each year was a parallel row with reused buffers
description: "On 200,000 simulated years (1,000,000 for apply_annual), Tower::apply, ceded_events, apply_annual and apply_term took 54 ms to 452 ms before #970 and 3 ms to 13 ms after on 16 threads (Windows, release), 15 ms to 57 ms on one. Most of the single-thread cost was memory: a vector per layer per call for the recoveries, a stage split by event even when nothing read it, and in ceded_events a string copy and a sort per event to rebuild the labels. Every result stayed the same bits, since each year is a row of its own summed in the old order, so no seeded number moved."
tags: [aggregate, reinsurance, performance, rayon, prospicio-aggregate]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-23T12:00:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/risk-rs/issues/970
    title: "[P1] Aggregate: parallelize Tower::apply* / ceded_events and eliminate per-year heap allocations"
  - id: bench
    resource: ../crates/prospicio-aggregate/benches/tower.rs
    title: The tower benchmark, with a checksum of each result's bits
  - id: note
    resource: ../docs/design/aggregate.md
    title: Design note, "The tower's simulated years in parallel (#970)"
  - id: test
    resource: ../validation/tests/aggregate_tower_threads.rs
    title: the_tower_gives_the_same_bits_on_any_number_of_threads, run by CI's Rust job
---

# The tower's simulated paths in parallel

`cargo bench -p prospicio-aggregate --bench tower` on a 16-core Windows
machine, the median of five runs, the old and new binaries run one after
the other in two rounds (the larger of the two medians shown). The
events are Poisson(5) losses of a lognormal(12, 1.5) severity.

| Scenario | Before, 16 threads | After, 16 threads | Before, 1 thread | After, 1 thread |
|---|---|---|---|---|
| `apply`, two excess layers | 62.5 ms | 3.4 ms | 60.7 ms | 14.7 ms |
| `apply`, three inuring stages | 140.0 ms | 6.5 ms | 133.8 ms | 37.9 ms |
| `apply` on labelled events | 218.8 ms | 9.9 ms | 210.2 ms | 56.6 ms |
| `ceded_events`, two layers | 287.7 ms | 13.4 ms | 259.9 ms | 43.6 ms |
| `apply_annual`, 1,000,000 years | 451.7 ms | 12.0 ms | 386.9 ms | 54.5 ms |
| `apply_term`, 5 years of 40,000 paths | 130.9 ms | 5.4 ms | 94.6 ms | 24.0 ms |

The checksums of every result's bits were the same before and after, on
one thread and on sixteen; the test also checks a scoped layer,
reinstatements pro rata as to time, labels and a term on 1, 2, 3 and 8
threads, and gave the same bits as the code before #970.

Where the time went on one thread:

- `Layer::layer_loss`, `before_terms` and the reinstatement premium each
  collected the year's recoveries into a vector and summed it, two or
  three times per layer per year. Summing them as they are computed, in
  the same order, gives the same bits.
- Every stage built each layer's ceded loss by event, which only a later
  stage, a labelled result or `ceded_events` reads.
- `ceded_events` rebuilt its labels by copying each kept event's label
  strings and sorting all of them; keeping the original combinations'
  indices gives the same sorted set.

Two things to know when timing on this machine: other sessions' builds
and test runs share the cores, so a benchmark run alone can read several
times slower than it is, and a bench built from another checkout into the
same `CARGO_TARGET_DIR` overwrites this one's binary (the same file name),
so compare two versions from separate target directories.
