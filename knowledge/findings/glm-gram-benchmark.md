---
type: Finding
title: Blocked, parallel X'WX makes a freMTPL2-sized GLM fit about 50 times faster
description: On a seeded 678,013-row, 60-column Poisson portfolio the row-by-row X'WX loop over column-major storage took 22 to 87 s a fit on a shared 16-thread Windows machine; fixed row blocks run with rayon, each reducing a tile of weighted rows with eight-sum dot products over contiguous column slices, take about 1 s, and the Bayesian GLM's NUTS run is 5 times faster with its log density in parallel row blocks.
tags: [glm, gam, bayes, performance, rayon, benchmark, determinism]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T10:00:00-08:00 }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/risk-rs/issues/969
    title: "#969: cache-thrashing X'WX assembly, per-iteration allocations and rayon in prospicio-glm and prospicio-bayes"
  - id: kernel
    resource: ../crates/prospicio-glm/src/gram.rs
    title: gram::weighted_gram
  - id: bench
    resource: ../crates/prospicio-glm/benches/glm.rs
    title: cargo bench -p prospicio-glm
  - id: bayes-bench
    resource: ../crates/prospicio-bayes/benches/bayes_glm.rs
    title: cargo bench -p prospicio-bayes
  - id: design
    resource: ../docs/design/models.md
    title: models.md, decision "Row sums run in fixed blocks (#969)"
---

# Facts

- `Design` stores columns. The 0.2 `working_system` looped over rows and,
  per row, over every column pair, so each row touched `p` separate arrays
  and nothing vectorized. `gram::weighted_gram` cuts the rows into blocks
  of 8192 (parallel) and tiles of 256 (a weighted copy of the tile's `p`
  columns, `8 p 256` bytes, stays in L2), then reduces each column pair
  with a dot product of eight running sums.
- The block partition depends only on `n`, and the block sums are added in
  block order, so a fit is bit-identical on one thread and on sixteen
  (tested: `gram::tests::is_the_same_on_one_thread_as_on_many`, and the
  Bayesian density's test under a one-thread pool). The sums are not the
  row-by-row ones: results move in the last bits from 0.2, so 0.3.0 lists
  them under Breaking changes. Against the old loop the Gram matrix agrees
  to 1e-12 of its largest entry.
- Timings, median (best) of three interleaved runs of the old and new
  binaries with `BENCH_QUICK=1`, release, 16 hardware threads, Windows 11,
  2026-11-23, while another session's test binary loaded the machine (so
  the old binary's times vary by a factor of 4):

| Case | 0.2 | blocked X'WX |
|---|---|---|
| Poisson GLM, 678,013 rows, 60 columns | 57.7 s (21.6 s) | 1.05 s (0.92 s) |
| 1000 Poisson GLMs, 100 rows, 20 columns | 4.1 s (1.8 s) | 0.58 s (0.31 s) |
| Poisson GAM with REML, 50,000 rows | 26.8 s (10.4 s) | 1.79 s (1.70 s) |
| Bayesian Poisson GLM, 4 chains x 600 draws, 20,000 rows, 20 columns | 7.0 s (6.3 s) | 1.34 s (1.28 s) |

- A triangle-sized fit (100 rows) is faster too: it fits in one block, so
  rayon adds one task, and the old loop's per-row column hopping was a cost
  even there.
- The Bayesian log density first stayed sequential inside each chain, with
  `Design::linear_predictor_into` (itself parallel) nested in the four
  parallel chains; that was no faster than 0.2. With the rows in parallel
  blocks of 4096 inside `logp`, the 12 cores the four chains leave idle
  take part, and the run is 5 times faster. Most of a step's cost is the
  per-row `exp` and log density, not the gradient.
- Benchmarks on this machine are noisy whenever other sessions build or
  test: compare old and new binaries interleaved, copied out of `target/`
  so a rebuild cannot replace them, and report the median and the best.
