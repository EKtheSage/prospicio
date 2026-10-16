---
type: Environment Fact
title: prospicio-nn in the Python bindings
description: The wheel links Burn through the bindings' default nn feature; it adds about 8 MB to the native module (18.3 MB without, 26.5 MB with, release, unstripped, Linux x86_64), raises the bindings' minimum Rust to 1.92, and Burn 0.21's fitted NdArray networks are Send and Sync, so frozen pyclasses hold them without a lock.
tags: [environment, python, bindings, burn, prospicio-nn, wheel]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T15:00:00Z }
stale_after: 2027-04-16T00:00:00Z
sources:
  - id: build
    resource: process:cloud session claude/stoic-hawking-qc3qby, 2026-10-16
    title: cargo build --release -p prospicio-python with and without --no-default-features
  - id: cargo
    resource: ../crates/prospicio-python/Cargo.toml
    title: The bindings' nn feature
---

# Facts (2026-10-16)

* `crates/prospicio-python` has an `nn` cargo feature, on by default, that
  links `prospicio-nn` (Burn 0.21 on the `ndarray` backend, `onnx-ir`).
  `cargo build --release -p prospicio-python --no-default-features`
  builds the module without it, and `prospicio.models` then simply lacks
  the network classes (the import is guarded).
* Size of `prospicio_native.abi3.so`, release, unstripped, Linux x86_64:
  18,271,560 bytes without `nn`, 26,540,520 bytes with it, about 8 MB
  more.
* Burn 0.21 needs Rust 1.92, so the bindings' `rust-version` is 1.92
  while the rest of the workspace keeps 1.85.
* `CannFit` and `AttentionCannFit` (Burn `Param<Tensor<NdArray<f64>>>`
  inside) and `OnnxModel` (an owned plan, `onnx-ir`'s `Rc` graph dropped
  at load) are `Send + Sync`: they sit in `#[pyclass(frozen)]` wrappers
  and run under `py.detach` with no `Mutex`.
