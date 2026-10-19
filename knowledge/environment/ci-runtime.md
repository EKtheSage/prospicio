---
type: Environment Fact
title: Where CI's time went, and what cut it
description: By 2026-10-19 a pull request's CI took 27 to 32 minutes when it touched the bindings or the Bayesian code; the debug tests at opt-level 0, the docs build re-running both test suites, cold caches on new branches and the conformance suite's Meyers and compartmental arms were most of it. An optimized dev profile took the debug tests from 437 s to 66 s; nextest added little on top.
tags: [environment, ci, cargo, testing, cache]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T23:00:00Z }
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
with `lookup-only` it checks for the key without downloading; on a miss
its post step still saves what the job built.[^rust-cache] So since
2026-10-19 a push to `main` looks each job's cache up and builds only on a
miss (a new `Cargo.lock` or toolchain), and the Pages build shares the
docs job's cache through `shared-key: docs-site`.[^ci]

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
spent compiling before; a cold build of everything took 487 s. Rust's
floating point is the same at every optimization level (no fast-math), so
the tests' exact values do not move. nextest saved about 13 s more and
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

[^runs]: CI run and job timings
[^local]: Local timings of cargo test, cargo nextest and the Bayesian release tests
[^ci]: CI workflow
[^rust-cache]: Swatinem/rust-cache v2, restore and save
