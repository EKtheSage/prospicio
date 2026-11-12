---
type: Environment Fact
title: Where CI's time went, and what cut it
description: By 2026-10-19 a pull request's CI took 27 to 32 minutes when it touched the bindings or the Bayesian code; the debug tests at opt-level 0, the docs build re-running both test suites, cold caches on new branches and the conformance suite's Meyers and compartmental arms were most of it. An optimized dev profile took the debug tests from 437 s to 66 s; nextest added little on top.
tags: [environment, ci, cargo, testing, cache]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T23:00:00Z }
updated: { by: claude-code/cloud-session, at: 2026-11-11T00:00:00Z }
stale_after: 2027-04-19T00:00:00Z
sources:
  - id: runs
    resource: process:GitHub Actions runs of ci.yml, #63 to #1083 (2026-10-02 to 2026-10-19)
    title: CI run and job timings
  - id: local
    resource: process:cloud session claude/stoic-hawking-qc3qby, 2026-10-19 (4 cores)
    title: Local timings of cargo test, cargo nextest and the Bayesian release tests
  - id: ci
    resource: ../.github/workflows/ci.yml
    title: CI workflow
  - id: legs
    resource: process:GitHub Actions runs 42951521128 (a pull request, the baseline) and 42992047374, 42994556570, 42998161759, 43000423587 (workflow_dispatch on claude/sharp-albattani-0mtilv), 2026-11-11
    title: Timings of the python, r and rust jobs before and after the test legs (#848 to #851)
  - id: rust-cache
    resource: https://github.com/Swatinem/rust-cache/blob/v2/src/restore.ts
    title: Swatinem/rust-cache v2, restore and save
---

# Before (2026-10-19)

A pull request's wall time grew from about 4 minutes on 2026-10-02 to 11
to 13 minutes for one that skipped the `bayes` and `docs` jobs and 27 to 32
for one that ran them.[^runs]

* `rust`, which every event runs: about 10.5 of its 12 minutes were the
  debug `cargo test`; compiling took about a minute with a warm cache. The
  Rust tests doubled to 1,576 in eleven days.
* `docs`: 26 minutes in `cargo xtask docs --check`, of which the Python
  suite (4.6 min) and the R suite (5.4 min) repeated the `python` and `r`
  jobs.
* `bayes`: about 30 minutes, 11 of them the one-year conformance suite
  (`every_model_conforms`, 674 s), 7 the compartmental one-year file, and
  2.5 a release build from scratch.
* `r`: the install-from-tarball step took 9.4 minutes, a second release
  build and a second run of every R test.
* The weekly run shared its concurrency group with pushes to `main`
  (`ci-${{ github.ref }}`), so a merge cancelled it: on 2026-10-19 at
  38.9 minutes, before its full Bayesian suite or Windows finished.

# Caches

A branch's run can restore caches saved on its own branch or on the base
branch, not another branch's. A push to `main` ran only `rust`, so the
`python`, `r`, `bayes` and `fremtpl2` caches existed only per branch and a
new branch started cold ("No cache found" in `bayes`).[^runs]
Swatinem/rust-cache keys a cache by job id, runner OS, the `key` and
`shared-key` inputs, the toolchain and a hash of every `Cargo.toml` and
`Cargo.lock`. It saves only when the restore was not an exact match, and
its `cache-hit` output is true only on an exact match.[^rust-cache] Its
`lookup-only` input downloads nothing even on a partial match, so it is
not used: since 2026-10-19 a push to `main` restores each job's cache,
skips the build on an exact hit, and on a partial match (a new
`Cargo.lock` or toolchain) builds incrementally on what it restored, which
the post step then saves; the Pages build shares the docs job's cache
through `shared-key: docs-site`.[^ci]

# The debug tests

Measured locally on 4 cores, 1,206 tests:[^local]

| profile | `cargo test` | `cargo nextest`, 2 threads | 4 threads |
|---|---|---|---|
| dev at `opt-level = 0` | 437 s | 405 s | |
| dev at 1, dependencies at 2 | 66 s | 59 s | 38 s (+15 s doctests) |

The time was in a few simulation-heavy tests of the workspace's own crates
(the diagonal backtest on RAA 27 s, `tensor_gams_match_mgcv` 24 s, the
bootstrap tail reconciliations 22 s and 14 s at `opt-level = 0`), so
optimizing only the dependencies would not have been enough. At
`opt-level = 1` rebuilding the workspace's crates took 54 s, about what CI
spent compiling before; a cold build of everything took 487 s. Rust does
not reorder or contract floating-point operations at any optimization
level (no fast-math), so results stay IEEE-identical, and the suite passed
unchanged. nextest saved about 13 s more and
was left out.

# The Bayesian smoke run

Release, 4 cores, every `slow-bayes` file without `RISK_RS_FULL_BAYES`:
484 s, after the conformance suite's smoke run dropped Meyers' CRC and
compartmental model 1 (both checked by their own one-year files): the
suite took 56 s (`BayesOdp` RAA 19 s, GenIns 19 s, `BayesBf` 17 s). The
rest: `reserving_compartmental_one_year` 204 s (model 1's view at 500
simulations; its bounds come from a 20-seed sweep at that size),
`reserving_compartmental_gm2` 96 s, `reserving_meyers_one_year` 38 s,
`reserving_bayes_one_year_moment_match` 27 s and
`reserving_compartmental` 23 s.[^local]

* One test binary per crate (#825, 2026-11-10): `validation/tests/` (69
  files) and `crates/prospicio-xll/tests/` (49) built 105 test binaries,
  each linking the whole dependency tree. As modules of three binaries
  (`validation`, `bayes` behind `slow-bayes`, `worksheet`, with
  `fremtpl2` alone), found by each crate's `build.rs`, the rebuild of the
  two crates' tests on warm dependencies fell from 318 s to 28 s on the
  4-core cloud container (debug info off in both runs), with the same 652
  tests listed and the 186 Bayesian tests in the `bayes` binary split
  90/45/51 by the three CI legs' module filters. The cost was linking, not
  code generation.[^local]

# The binding tests in parallel, and on more runners (2026-11-11)

Before (#848 to #851), a pull request's `python` job took 19.5 minutes, 14.7
of them pytest (883 s, 3,756 tests, guides included), and its `r` job 13
minutes, 9.4 of them the R files one after another (564 s; the guides 99 s,
`test-reserving-compartmental.R` 221 s).[^legs]

* Four processes on the runner's four cores do not help the Bayesian
  files: their NUTS chains already run in parallel through rayon. With one
  rayon thread per process the compartmental R file took 624 s and the
  whole parallel R step 624 s; with the default thread count, 395 s and
  406 s. pytest on four xdist workers took 854 s with one thread and 739 s
  with the default, against 788 s serially without the guides: a
  hierarchical BF test took 323 s beside the other workers and about 128 s
  alone.[^legs]
* LightGBM and XGBoost (OpenMP) slowed eightfold beside other processes:
  `test-boosting.R` 96 s in the parallel batch, 12 s alone.[^legs]
* The light files do parallelize: pytest's 3,600 or so tests outside the
  NUTS-heavy files and the guides took 69 s on four workers with one rayon
  thread, the R files outside them about 30 s.[^legs]
* So the python and r jobs became matrices of legs on runners of their own,
  each building the package itself (about 3.5 minutes; runner minutes are
  free): `tests` (the light files in parallel, then a few NUTS-heavy files
  and the boosting tests serially; Python 143 s, R 147 to 256 s), `slow-1`
  and `slow-2` in Python (`test_reserving_compartmental.py`, 245 to 297 s;
  the hierarchical BF and backtest files, 110 s) and `slow` in R
  (`test-reserving-compartmental.R`, 222 to 245 s), and `guides` (Python 96
  to 116 s, R 131 to 169 s). A job's wall time is now its longest leg:
  about 9 minutes for Python (from 19.5) and 8.5 for R (from 13; about 10
  when the install-from-tarball step runs). The same leg varied by a
  quarter between runs, as runners do. xdist also needs every worker
  to collect the same tests in the same order: `test_docs.py` listed native
  classes' attributes in `vars()` order, which differs between processes,
  and now sorts them.[^legs][^ci]
* Line tables only in the `rust` job's test builds
  (`CARGO_PROFILE_DEV_DEBUG=line-tables-only`, #851): `cargo test` took
  303, 313 and 392 s on a warm cache against 483 s on the baseline pull
  request, after #825 had already cut 105 test binaries to three.[^legs]

[^runs]: CI run and job timings
[^local]: Local timings of cargo test, cargo nextest and the Bayesian release tests
[^ci]: CI workflow
[^rust-cache]: Swatinem/rust-cache v2, restore and save
[^legs]: Timings of the python, r and rust jobs before and after the test legs (#848 to #851)
