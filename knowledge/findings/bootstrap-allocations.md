---
type: Finding
title: The bootstraps' simulation loops allocate two or three times a simulation, so per-worker scratch buffers would gain nothing measurable
description: "On a 40 by 40 quarterly triangle at 10,000 simulations, the ODP and Mack bootstraps allocate 3.1 and 2.1 times a simulation and spend 0.25% and 0.2% of their instructions in the allocator; the one-year bootstrap allocates 343 times a simulation, almost all in refitting the chain ladder through its public fit, and spends 8.7% there, a ceiling well short of #972's 2x to 5x."
tags: [reserving, bootstrap, performance, allocation, prospicio-reserving]
status: draft
generated: { by: claude-code/cloud-session, at: 2026-11-27T09:00:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/1126
    title: "Reserving: reuse per-worker scratch buffers in the bootstrap loops, if a profile shows it pays"
  - id: bench
    resource: ../crates/prospicio-reserving/examples/bootstrap_bench.rs
    title: The bootstrap benchmark, with a counting allocator
---

# Finding

The benchmark builds a 40 by 40 quarterly cumulative triangle (820
cells) and fits each bootstrap with the defaults and seed 1, on a 4-core
cloud container, release build.[^bench]

| Fit, 10,000 simulations | 4 threads | 1 thread | allocations a simulation | bytes |
|---|---|---|---|---|
| `OdpBootstrap::fit` | 0.54 s | 2.01 s | 3.1 | 162 MB |
| `MackBootstrap::fit` | 0.61 s | 2.34 s | 2.1 | 34 MB |
| `OdpBootstrap::one_year`, chain ladder | 0.34 s | 1.09 s | 343 | 1,662 MB |
| `MackBootstrap::one_year`, chain ladder | 0.36 s | 1.13 s | 342 | 1,534 MB |

Callgrind on 300 simulations, one thread, counts the instructions spent
in the allocator (malloc, free, realloc and Rust's vector growth) as
0.25% of the ODP bootstrap's, 0.2% of Mack's and 8.7% of the one-year
bootstrap's; copies and fills (memcpy, memset) add 1.6%, 0.15% and 2.3%.

* The lifetime loops (`Simulation::run`, `run_off`) already write each
  simulation's draws into the output rows, so their two or three
  allocations a simulation (the pseudo triangle and its projection) cost
  nothing measurable: keeping them per Rayon worker cannot move the times.
* The one-year bootstrap's allocations come from refitting the method on
  each simulated next diagonal (`OneYearMethod::ultimate`, through
  `ChainLadder::fit_segment` and `Development::fit_segment`), which builds
  a whole fit with its vectors. Removing them would need a refit that
  writes into buffers it is given, for every method the one-year view
  takes; 8.7% of the instructions is the most it could save.
* #1056 had measured `sample_one` at 5% to 10% of the ODP bootstrap on
  RAA; neither profile supports #972's estimate of 2 to 5 times.[^issue]

[^bench]: The bootstrap benchmark, with a counting allocator
[^issue]: "Reserving: reuse per-worker scratch buffers in the bootstrap loops, if a profile shows it pays"
