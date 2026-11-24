---
type: Finding
title: Reserving hot paths, measured (#972)
description: Drawing one value with sample_one instead of sample(rng, 1)[0] saves 5 to 10% on the ODP, Mack and one-year bootstraps and the Bayesian ODP with output unchanged bit for bit; HierarchicalBf's hand-written gradient made its fit about 3 to 6 times faster and its one-year view with refits about 3 times faster, moving its seeded draws in the last bits.
tags: [reserving, bootstrap, bayes, autodiff, performance, benchmark, reproducibility]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T18:00:00-08:00 }
sources:
  - id: bench
    resource: ../validation/tests/reserving_perf_benchmark.rs
    title: The benchmark of #972 (ignored release tests with output hashes)
  - id: hb
    resource: ../crates/prospicio-reserving/src/bayes/hierarchical_bf.rs
    title: Model::log_density_gradient and the_hand_gradient_matches_the_dual
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/972
    title: Issue 972, one-element Vec allocations, Bayes AD and nested rayon
---

# Finding

Issue #972 estimated that removing the one-element `Vec` behind each
`sample(rng, 1)[0]` would make the bootstraps 2 to 5 times faster. It does
not: a Gamma draw by Marsaglia and Tsang (one normal by inverse transform
and one uniform from ChaCha20) costs far more than a small allocation. With
`Distribution::sample_one` (#1055, #1056), the bootstraps and the Bayesian
ODP ran 5 to 10% faster, and every output hash was unchanged.[^bench]

The large gain was `HierarchicalBf`'s gradient. On the three-program
reference panel (72 parameters) the chunked dual ran the whole density three
times with 32 tangents per leapfrog step. A hand-written reverse sweep does
it in one pass of plain floats, since each term depends on a few
parameters.[^hb]

| Benchmark (release, Windows, 16 threads, other builds running) | before | after | output |
|---|---|---|---|
| ODP bootstrap, lifetime, ABC, 100k sims, Gamma | 0.179 s | 0.167 s | same hash |
| ODP bootstrap, one year, ABC, 20k sims | 0.030 s | 0.028 s | same hash |
| Mack bootstrap, lifetime, ABC, 100k sims, Gamma | 0.172 s | 0.162 s | same hash |
| Bayesian ODP fit, RAA | 0.093 s | 0.086 s | same hash |
| `HierarchicalBf` fit, reference panel | 133 s (min 128) | 40 s (min 21) | moved |
| `HierarchicalBf` one year, 64 sims, refit all | 1,579 s (min 1,402) | 502 s (one run) | moved |

The times are medians of 5 runs (3 for the fit) with the two binaries
interleaved. Three other sessions were compiling on the same machine, so
the times are noisy; only the ratios mean anything.

# Consequences

The value of the hand-written gradient is the generic density's operation
for operation; the gradient agrees with the dual's to 1e-10 relative, not
bit for bit. NUTS trajectories therefore differ in their last bits, and so
do `HierarchicalBf`'s seeded draws and its pin in `bayes::pins`.

Checked and left unchanged: `laplace::positive_definite`'s second Cholesky
factors `h + 2 lambda I`, not the matrix just tested. The one-year view's
nested rayon (refits and moment matching inside a parallel loop over
simulations) runs on one work-stealing pool, and making it sequential would
need a flag in `prospicio-bayes`'s `nuts::sample`.

[^bench]: validation/tests/reserving_perf_benchmark.rs
[^hb]: Model::log_density_gradient in hierarchical_bf.rs
