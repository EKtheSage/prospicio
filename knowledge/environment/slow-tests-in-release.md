---
type: Environment Fact
title: Bayesian tests run in release, outside cargo test
description: The NUTS parity tests sit behind the slow-bayes feature of prospicio-validation and prospicio-reserving's NUTS unit tests are ignored without its slow-tests feature, so a debug cargo test skips both and CI's bayes job runs them in release; how to run them locally and add one.
tags: [environment, ci, bayes, testing]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T00:00:00Z }
stale_after: 2027-04-13T00:00:00Z
sources:
  - id: ci
    resource: ../.github/workflows/ci.yml
    title: CI, bayes job
  - id: cargo
    resource: ../validation/Cargo.toml
    title: prospicio-validation, the slow-bayes feature and its [[test]] entries
  - id: agents
    resource: ../AGENTS.md
    title: AGENTS.md, Checks
  - id: reserving
    resource: ../crates/prospicio-reserving/Cargo.toml
    title: prospicio-reserving, the slow-tests feature
---

# The split

The parity files that sample with NUTS are the `reserving_bayes*`,
`reserving_meyers*` and `reserving_compartmental*` targets (thirteen on
2026-10-15, with `reserving_meyers_one_year`). When there were six, they took about 220 s of the rust job's
10 to 12 minutes in CI's debug `cargo test`, and about 17 s in a release
build (measured locally on 2026-10-13: 17.5 s wall, 15 s of test
time).[^ci]

Each has a `[[test]]` entry in `validation/Cargo.toml` with
`required-features = ["slow-bayes"]`, a feature of
`prospicio-validation` that also turns on
`prospicio-reserving/slow-tests`.[^cargo] Without the feature cargo skips
the target silently, so `cargo test` neither builds nor runs them. That list is the
one place: CI's `bayes` job reads it with Python's `tomllib` and passes one
`--test` per entry. The rust job lints them with
`cargo clippy --all-targets --features prospicio-validation/slow-bayes`.

# Running them

```
cargo test --release -p prospicio-validation --features slow-bayes --test 'reserving_bayes*' --test 'reserving_meyers*' --test 'reserving_compartmental*'
```

Cargo accepts glob patterns in `--test`. Running the whole package in
release with the feature (no `--test`) also runs every other parity file,
which CI's rust job already covers in debug; the `bayes` job selects the
Bayesian targets only. On Windows on 2026-10-13, `pricing`'s
`natural_allocation_matches_aggregate` failed 7 of 1581 cases in both
debug and release (the `discrete3` case's `tvar_threshold` and stand-alone
prices off by one grid step) while CI on Ubuntu passed it, so a local
`cargo test` on Windows can fail there for a reason unrelated to the
Bayesian tests.[^agents]

# The unit tests

The same split applies to `prospicio-reserving`'s unit tests in `bayes::*`
since 2026-10-15. As the one-year engine, its conformance suite and
compartmental model 2 grew, that crate's unit-test binary took 295 s of
the rust job's 17.6 minutes on CI's 4-core runner (debug, run 38813063744),
and 232 s single-threaded locally, 226 s of it in `bayes::`. The 29 tests
that fit with NUTS and took more than about a second carry
`#[cfg_attr(not(feature = "slow-tests"), ignore = "...")]`; the crate's
`slow-tests` feature (which turns on `bayes`) un-ignores them.[^reserving]
Without it the binary takes 12 s single-threaded in debug (282 run, 32
ignored: the 29 and three plain `#[ignore]`s); with it, in release, all
311 run in about 4 s. The attribute, unlike a `cfg` on the module, keeps
them compiled, so clippy lints them in every build. Cheap checks (errors,
layouts, gradients, the prior predictive, a short run per model, one
one-year fit) stay in the debug run. Meyers' one-year view (branch
claude/bayes-one-year-meyers) added three such tests and two cheap ones:
284 run and 35 ignored in debug with `bayes`, 319 in all.

It also adds to the `bayes` job's release parity tests (measured locally
on 2026-10-16, 16 threads). `reserving_meyers_one_year` takes 109 s: per
model, the default view of 500 simulations at the full sampler (13 to
26 s) and `RefitAll` on 60 of them (11 to 19 s); its bounds come from a
20-seed sweep at those sizes, so cutting them needs a new sweep. The
conformance suite's three Meyers arms took 89 s of its 128 s, about 50 s
of it in the one-thread replay of `MomentMatchThenRefit`, whose refits run
one after another (the lifetime fit is about 0.2 s and the 10,000
reweighted simulations 0.6 to 1.3 s). Replaying only the first 20 of the
60 simulations (matched and refitted ones in each model at seed 0) cut
the arms to 57 s and the suite to 94 s. A 4-core runner takes about 3 to
4 times as long on the parallel parts. On CI, before that cut (run
38843505221, 2026-10-16), the `bayes` job took 19 min 42 s, 17 min 39 s
of it the parity tests and 1 min 42 s the unit tests, against 10 min 54 s
for the rust job: the PR's slowest check.

Since the user's decision of 2026-10-16 a pull request runs a smoke
version of the slowest arms, Meyers' one-year views (CRC alone in
`reserving_meyers_one_year` and among the conformance suite's Meyers
arms), and the weekly and manual runs set `RISK_RS_FULL_BAYES` and run
all of them (`prospicio_validation::full_bayes`, read at run time so one
build compiles and clippy lints every arm). Measured locally on
2026-10-16 (release, 16 threads): `reserving_meyers_one_year` 28 s smoke
against 114 s full, the conformance suite 54 s against 102 s, and every
Bayesian parity file together 230 s against 390 s. At the roughly 2.5
times local seen on CI for these files, that is about 10 minutes of
parity tests on a pull request and about 16 in the weekly run. The
compartmental one-year views take the same gate.

```
cargo test --release -p prospicio-reserving --features slow-tests --lib
```

CI's `bayes` job runs this after the parity tests, and `slow-bayes` turns
`prospicio-reserving/slow-tests` on, so one flag covers both. A seed-pinned
test that is platform-specific stays plainly `#[ignore]`d and runs only by
name with `--ignored`. Measured locally on 2026-10-15 (Windows, 16
threads): the debug unit tests of `prospicio-reserving --features bayes`
fell from 61 s to 6.5 s wall, and from 233 s to 12 s single-threaded.

`prospicio-bayes` has one slow unit test of its own,
`stacking::tests::pooled_and_adaptive_fits_find_the_better_model` (26 s
single-threaded in debug, most of that crate's 29 s); it is not gated.

# When CI runs them

On a pull request that touches `prospicio-reserving`, `-bayes`, `-prob`,
`-core`, `-math`, `-glm` or `-models`, the root `Cargo.toml` or
`Cargo.lock`, `validation/Cargo.toml`, `validation/src/`, `validation/data/`,
`validation/scripts/stan/`, a validation test, script or reference file
named for `bayes` or `meyers`, or `ci.yml`; and on every weekly and manual
run. A push to `main` does not run them, like every job but the rust job:
the PR already did.

[^ci]: CI, bayes job
[^cargo]: prospicio-validation, the slow-bayes feature and its [[test]] entries
[^agents]: AGENTS.md, Checks
[^reserving]: prospicio-reserving, the slow-tests feature
