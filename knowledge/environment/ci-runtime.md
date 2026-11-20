---
type: Environment Fact
title: Where CI's time went, and what cut it
description: By 2026-10-19 a pull request's CI took 27 to 32 minutes when it touched the bindings or the Bayesian code; the debug tests at opt-level 0, the docs build re-running both test suites, cold caches on new branches and the conformance suite's Meyers and compartmental arms were most of it. An optimized dev profile took the debug tests from 437 s to 66 s; nextest added little on top.
tags: [environment, ci, cargo, testing, cache]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T23:00:00Z }
updated: { by: claude-code/cloud-session, at: 2026-11-20T00:00:00Z }
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
  - id: gating
    resource: process:cloud session claude/cool-shannon-fqpoy8-ci-967, 2026-11-20 (4 cores, Rust 1.100)
    title: Release builds of the bayes test binary with and without Burn and Arrow (#967)
  - id: build-once
    resource: process:cloud session claude/cool-shannon-fqpoy8-ci-965, 2026-11-20 (4 cores, Rust 1.100, uv 0.11, R 4.3)
    title: The wheel, the R binary package and the Bayesian test binaries built once and tested from the files (#965)
  - id: speedups
    resource: process:cloud session claude/cool-shannon-fqpoy8-ci-966, 2026-11-20 (4 cores, Rust 1.100, uv 0.11)
    title: What the build speedups of #966 measured, and which were already in effect
  - id: setup-uv
    resource: https://raw.githubusercontent.com/astral-sh/setup-uv/v6/action.yml
    title: astral-sh/setup-uv v6, inputs and defaults
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

# What each release job compiles (#967, 2026-11-20)

Cargo compiles every dev-dependency of a crate for each of its test
binaries, so the `bayes` job built Burn 0.21 and Arrow 60 and the
`fremtpl2` job built Arrow and nuts-rs, which neither reads. A
dev-dependency cannot be optional, so `prospicio-bayes` and `prospicio-nn`
are optional dependencies of `prospicio-validation` behind its features
`bayes` and `nn`, with `arrow` (`prospicio-prob/arrow`) and `io`
(`prospicio-reserving/io`); all four are on by default, so `cargo test`,
clippy and the documented commands build what they built before. CI's
release jobs turn the defaults off: `--no-default-features --features
slow-bayes` (no Burn, Arrow or calamine in `cargo tree`) and
`--no-default-features --features nn` (no nuts-rs or Arrow).[^ci]

* A cold release build of the `bayes` test binary took 136 s with the
  defaults off; turning them on again compiled for another 199 s, so the
  job's cold build falls from about 335 s to 136 s on four cores.[^gating]
* `required-features` on the `validation` test binary (all four) and on
  `fremtpl2` (`nn`) make a build without them skip the binary instead of
  failing to compile it.
* The freMTPL2 job keeps `prospicio-nn` without its default `onnx`
  feature, as before: the bindings build it with the ONNX import, but
  each job has a cache of its own, so the second feature set costs
  nothing and saves the job `onnx-ir`.

# Build once, test in legs (#965, 2026-11-20)

After #848 to #850 every leg of `python` (four), `r` (three) and `bayes`
(three) compiled the same crates in release, and on a push to `main` the
three `bayes` legs each built and raced to save one cache key. Now a
build job compiles once and the legs, which keep their names (`python`,
`r`, `bayes`, so the checks' names are unchanged), install or run its
artifact with no Rust toolchain and no cache:[^ci]

* `python-build`: `uv build --wheel` (maturin through PEP 517, release,
  into the workspace's `target/`) writes one abi3 wheel, 12.9 MB. A leg
  runs `uv sync --no-install-project`, `uv pip install --no-deps` on the
  wheel, and `uv run --no-sync pytest`, so uv never builds the project.
  pytest and its xdist workers import the installed wheel although the
  working directory holds the source package `prospicio/` without its
  extension; `python -c "import prospicio"` there imports the source
  package and fails, so a test that starts Python itself from `python/`
  would need another directory.[^build-once]
* `r-build`: `R CMD INSTALL --build` writes
  `prospicio_<version>_R_x86_64-pc-linux-gnu.tar.gz` (22.8 MB; a `.zip` on
  Windows), which `R CMD INSTALL` installs as a binary package in
  seconds. The install-from-tarball check, which compiles the crates
  again, stays in the build job.[^build-once]
* `bayes-build`: the two test binaries (`bayes`, 10.5 MB, and
  prospicio-reserving's unit tests, 14.9 MB), named by the `executable`
  field of `cargo test --no-run --message-format=json`. A leg runs
  `bayes-bin/parity <module>:: ...` from `validation/` and
  `bayes-bin/unit` from `crates/prospicio-reserving/`: `cargo test` runs a
  test binary in its package's directory, and the artifact does not keep
  the executable bit (`chmod +x`). The paths compiled into the binary
  (`env!("CARGO_MANIFEST_DIR")`) hold because every runner checks the
  repository out at the same path.[^build-once]
* nextest's archive and `--partition count:k/3`, which the issue
  proposed, would run each test in a process of its own, and 15 of the 27
  parity files share one fit between their tests through a `OnceLock`:
  each test would refit. The legs therefore stay the hand-balanced module
  lists of `[package.metadata.bayes-shards]`.

# The smaller speedups, and three that were already in effect (#966, 2026-11-20)

* One `changes` job reads a pull request's diff (a clone with the history
  but no file contents, `filter: blob:none`; the pack is 39 MB with them)
  and the other jobs read its outputs. Before, thirteen jobs each cloned
  the whole history to decide whether to skip themselves. A job that is no
  matrix and feeds no other job (`rust`, `msrv`, `docs`, `deal-report`,
  `fremtpl2`, `deny`) is skipped as a whole, which takes no runner and
  counts as passed; the matrix jobs and the build jobs they need skip
  their steps instead, so their checks keep the names
  `python (ubuntu-latest, tests)` and the like.[^ci]
* `docs` no longer runs for a pull request that changes only the bindings'
  tests (`python/tests/`, `R/prospicio/tests/`), which no generated file
  reads; `msrv` runs nightly, on manual runs and on a pull request that
  touches a `Cargo.toml` or `Cargo.lock`.[^ci]
* R packages come through `r-lib/actions/setup-r-dependencies` (pak, the
  library cached) with `dependencies: 'NA'` (hard dependencies only: the
  default, `"all"`, adds the named packages' Suggests), and each `r` leg
  installs what its files load: lightgbm and xgboost in `tests`, ggplot2
  and gt in `guides`, S7 alone in `slow`.[^ci]
* The Excel add-in on a pull request: `cargo xtask xll --thin-lto` built
  in 135 s from nothing (18.3 MB); the fat-LTO build then took 194 s for
  its last crate and link alone, on the dependencies the thin build had
  compiled (15.7 MB).[^speedups]
* Already in effect, so nothing to gain: (1) `cargo xtask generated` did
  not compile twice. `uv sync` builds the extension through PEP 517 and
  `maturin develop --release` then finds it built (`Finished` in 0.26 s);
  what `--no-install-project` saves is one wheel packed and installed, a
  few seconds, and only with `UV_NO_SYNC=1` on the later `uv run` calls,
  which would otherwise install the project themselves. (2) setup-uv v6
  caches by default on GitHub's runners (`enable-cache: auto`) and prunes
  the cache to wheels built from source, so the explicit `enable-cache`
  and `cache-dependency-glob` only narrow the key.[^setup-uv] (3) Rust
  links with lld by default on `x86_64-unknown-linux-gnu` since 1.90:
  the link line of a build here on 1.100 names `gcc-ld/ld.lld`, so mold
  or an lld override would change nothing on Linux.[^speedups]
* Not done: sccache. Swatinem/rust-cache already restores each job's
  dependencies; what sccache could add is the workspace's own crates, and
  a pull request can read only `main`'s cache entries, which a push to
  `main` writes only when a job's cache misses. It would also share the
  10 GB cache quota with rust-cache. It needs a measurement on CI before
  it is worth its moving parts.

[^speedups]: What the build speedups of #966 measured, and which were already in effect
[^setup-uv]: astral-sh/setup-uv v6, inputs and defaults
[^build-once]: The wheel, the R binary package and the Bayesian test binaries built once and tested from the files (#965)
[^gating]: Release builds of the bayes test binary with and without Burn and Arrow (#967)
[^runs]: CI run and job timings
[^local]: Local timings of cargo test, cargo nextest and the Bayesian release tests
[^ci]: CI workflow
[^rust-cache]: Swatinem/rust-cache v2, restore and save
[^legs]: Timings of the python, r and rust jobs before and after the test legs (#848 to #851)
