---
type: Environment Fact
title: prospicio-nn in the Python and R bindings
description: The wheel and the R package link Burn through the bindings' default nn feature; it adds about 8 MB to the native module (18.3 MB without, 26.5 MB with, release, unstripped, Linux x86_64), needs Rust 1.92 to build, and Burn 0.21's fitted NdArray networks are Send and Sync, so frozen pyclasses hold them without a lock; R's model.matrix names factor indicators regionB, renamed region[B] for the attention CANN's tokens.
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
* Burn 0.21 needs Rust 1.92, so a build with `nn` does too. The
  bindings' `rust-version` stays the workspace's 1.85 anyway: raised to
  1.92 it turns on clippy lints that the lower minimum suppresses
  (`collapsible_if` with let chains, `manual_is_multiple_of`) in every
  lane's binding files, which fails CI's `-D warnings`.
* `CannFit` and `AttentionCannFit` (Burn `Param<Tensor<NdArray<f64>>>`
  inside) and `OnnxModel` (an owned plan, `onnx-ir`'s `Rc` graph dropped
  at load) are `Send + Sync`: they sit in `#[pyclass(frozen)]` wrappers
  and run under `py.detach` with no `Mutex`.
* `crates/prospicio-r` has the same `nn` feature, on by default, and the R
  package's `SystemRequirements` asks for rustc >= 1.92. Without the
  feature an empty `nn` module registers nothing and the R functions fail
  when called. `bootstrap.R` picks up `prospicio-nn` on its own (an
  optional `workspace = true` dependency matches its pattern).
* R's `model.matrix()` names a factor's indicator columns `regionB`, not
  `region[B]`; the attention CANN groups tokens by the `name[level]` form,
  so `nn.R` renames them from the matrix's `assign` attribute and the
  terms' labels before fitting and predicting.
* The extendr wrappers in `R/prospicio/R/extendr-wrappers.R` are kept by
  hand: generate them with `.Call("wrap__make_prospicio_wrappers",
  use_symbols = TRUE, package_name = "prospicio", PACKAGE = "prospicio")`
  after an install and append the new module's functions and
  environments, keeping each `#' @export` before the `$` and `[[`
  methods (roxygen refuses them otherwise).
