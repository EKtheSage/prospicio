---
type: Environment Fact
title: WebAssembly builds of the core crates
description: Which crates build for wasm32, how to run their tests under WASI with Node, and how Rayon behaves without threads.
tags: [environment, wasm, rayon, ci]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-07T00:30:00Z }
sources:
  - id: ci
    resource: ../.github/workflows/ci.yml
    title: CI, rust job (wasm32 check)
  - id: architecture
    resource: ../docs/architecture.md
    title: architecture.md, Open decisions (WASM scope)
---

# Facts

* `prospicio-core`, `prospicio-math`, `prospicio-prob`, `prospicio-aggregate`, `prospicio-pricing` and
  `prospicio-reserving` build for `wasm32-unknown-unknown` with their default
  features, unchanged (checked 2026-10-07; CI's rust job checks it on every
  PR).[^ci] Arrow stays behind prospicio-prob's `arrow` feature, so it is not on
  that path.
* Their tests run under WASI: `rustup target add wasm32-wasip1`, `cargo test
  --target wasm32-wasip1 -p prospicio-aggregate --lib --no-run`, then run the test
  `.wasm` with Node 22's `node:wasi` module (`new WASI({version:
  "preview1", args, preopens: {"/": "/"}})`). All of prospicio-aggregate's tests
  pass that way (about 160 s in a debug build, single-threaded), except the
  one that builds explicit 1-, 2- and 8-thread pools.[^ci]
* Without threads, Rayon's global pool runs parallel iterators on the
  calling thread, so `simulate_events` and the other `par_iter` paths give
  the same results, only serially. An explicit `ThreadPoolBuilder` with more
  threads fails, since it cannot spawn them; code on the WASM path must not
  build one.[^architecture]

[^ci]: CI, rust job (wasm32 check)
[^architecture]: architecture.md, Open decisions (WASM scope)
