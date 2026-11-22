---
type: Environment Fact
title: Building and testing the JavaScript package (WebAssembly)
description: What the wasm-bindgen build of crates/prospicio-wasm needs, how its tests run under Node, and the quirks found building it (the pinned tool, serde-wasm-bindgen ignoring unknown fields, Rayon without threads, the node test runner).
tags: [environment, bindings, wasm, javascript, ci]
status: stable
stale_after: 2027-05-20T00:00:00Z
generated: { by: claude-code/cloud-session, at: 2026-11-20T06:00:00Z }
sources:
  - id: design
    resource: ../docs/design/wasm.md
    title: Design note, the JavaScript package
  - id: crate
    resource: ../crates/prospicio-wasm/Cargo.toml
    title: The binding crate's manifest, with the wasm-bindgen pin
  - id: xtask
    resource: ../xtask/src/main.rs
    title: cargo xtask wasm
  - id: session
    resource: process:cloud-session, branch claude/wasm-package (#576)
    title: First build of the package, 2026-11-20
---

# Building and testing the JavaScript package

`crates/prospicio-wasm` is the `wasm-bindgen` front end over the crates in
the WASM scope (`docs/design/wasm.md`). These are the facts found building
it.

## What a machine needs

- `rustup target add wasm32-unknown-unknown`.
- The `wasm-bindgen` command-line tool at exactly the version of the
  `wasm-bindgen` crate, which the crate's manifest pins with `=`:
  `cargo install wasm-bindgen-cli --version <pin> --locked` (about three
  minutes). It installs `wasm-bindgen` and `wasm-bindgen-test-runner`. A
  tool at another version refuses the module with a schema-version error.
- Node 20 or later on the path. Nothing from npm: there is no
  `node_modules` and no bundler in the build.

## Commands

- `cargo test -p prospicio-wasm --target wasm32-unknown-unknown` runs the
  Rust tests under Node. `.cargo/config.toml` sets the runner for that
  target to `wasm-bindgen-test-runner`.
- `cargo xtask wasm` builds the release module (fat LTO, one codegen unit:
  1.53 MB, 0.51 MB gzipped on 2026-11-20), writes the `bundler`, `node` and
  `web` glue into `target/wasm/pkg` with a `package.json`, and runs
  `crates/prospicio-wasm/js/package.test.mjs` with `node --test`. The
  release build takes under a minute on 16 cores.
- `cargo clippy -p prospicio-wasm --target wasm32-unknown-unknown
  --all-targets -- -D warnings`. `-D warnings` reaches the workspace crates
  it depends on, built on their default features, which no other job
  lints: it found `push_combined` in `prospicio-reserving` unused without
  the `bayes` feature.
- CI's `wasm` job runs these three when the `changes` job's `wasm` output
  is true (the binding crate, a crate in the WASM scope or
  `prospicio-glm` and `prospicio-models`, which `prospicio-reserving`
  depends on unconditionally and so links into the package, the data the
  tests read, xtask, `.cargo/`, the manifests or ci.yml), on pull requests and on
  the nightly and manual runs. `cargo xtask check --changed` reads the same
  filter (`xtask/src/check.rs`) and runs them when it matches, skipping
  with a note when Node, the pinned `wasm-bindgen` or the target is missing.

## Quirks

- **`serde-wasm-bindgen` ignores unknown fields.** Reading a JavaScript
  object into a struct, it looks up the fields the struct declares and
  never visits the others, so `#[serde(deny_unknown_fields)]` does nothing
  and a mistyped option is silently the default. The crate checks the
  object's keys against each options struct's `Fields::FIELDS` first.
- **Whole numbers arrive as integers.** A JavaScript number that is a safe
  integer is given to serde as an `i64`, so an untagged enum of a year
  (`i32`) or text, and a `u64` seed, read from plain numbers.
- **Typed arrays are accepted as sequences.** A `Float64Array` or
  `Uint32Array` reads into a `Vec` like a plain array.
- **Rayon without threads.** `wasm32-unknown-unknown` has no threads;
  Rayon's global pool then runs on the calling thread. The ODP bootstrap's
  draws and a tower's cessions equal the native ones bit for bit, since
  each simulation has its own random stream.
- **A WASM test has no file system.** The tests read the parity files of
  `validation/reference/` with `include_str!`.
- **`node --test <directory>`** in Node 22 treats the directory as a
  module and fails; the task names the test file.
- **Node's glue is CommonJS.** `wasm-bindgen --target nodejs` writes
  `require` calls, and the package is `"type": "module"`, so `node/` gets
  its own `package.json` with `"type": "commonjs"`.
- **`wasm-opt` is not used.** Binaryen is not installed in the cloud
  container or in CI; the size above is without it.
