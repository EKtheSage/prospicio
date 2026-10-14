---
type: Environment Fact
title: Bayesian parity tests run in release, outside cargo test
description: The NUTS parity tests take about 220 s in a debug build and about 17 s in release, so they sit behind the slow-bayes feature of prospicio-validation and CI's bayes job runs them in release; how to run them locally and add one.
tags: [environment, ci, bayes, testing]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T00:00:00Z }
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
---

# The split

Six parity files sample with NUTS: `reserving_bayes`, `reserving_bayes_bf`,
`reserving_bayes_segments`, `reserving_meyers`, `reserving_meyers_cay` and
`reserving_meyers_segments`. In CI's debug `cargo test` they took about
220 s of the rust job's 10 to 12 minutes; in a release build they take
about 17 s (measured locally on 2026-10-13: 17.5 s wall, 15 s of test
time).[^ci]

Each has a `[[test]]` entry in `validation/Cargo.toml` with
`required-features = ["slow-bayes"]`, an empty feature of
`prospicio-validation`.[^cargo] Without the feature cargo skips the target
silently, so `cargo test` neither builds nor runs them. That list is the
one place: CI's `bayes` job reads it with Python's `tomllib` and passes one
`--test` per entry. The rust job lints them with
`cargo clippy --all-targets --features prospicio-validation/slow-bayes`.

# Running them

```
cargo test --release -p prospicio-validation --features slow-bayes --test 'reserving_bayes*' --test 'reserving_meyers*'
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
