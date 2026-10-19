---
type: Finding
title: Burn networks save as JSON by walking their parameters with module paths
description: Burn 0.21 visits a module's float parameters in field order and calls enter_module/exit_module around each field and Vec index (not around an Option), so a visitor that keeps a path stack names every weight hidden.0.weight, hidden.0.bias, out.weight; a ModuleMapper with the same stack replaces them in the same order, so prospicio-nn's fitted networks round-trip through plain JSON bit for bit (f64 on NdArray) without Burn's recorders, and a rebuild for loading goes through the same seed lock as a fit, since Burn's initializers draw from a global generator.
tags: [prospicio-nn, burn, artifacts, serialization, json]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T04:00:00Z }
verified: { by: process:ci, at: 2026-10-19T04:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-nn/src/artifact.rs
    title: Collect (ModuleVisitor) and Load (ModuleMapper)
  - id: burn
    resource: https://docs.rs/burn-core/0.21.0/burn_core/module/trait.ModuleVisitor.html
    title: burn-core 0.21 ModuleVisitor and ModuleMapper
---

# Facts

* Burn 0.21's derived `Module` visits float parameters in field order and
  calls `enter_module(name, container)` and `exit_module` around each
  struct field and each `Vec` element (the name is the index); an
  `Option` (a `Linear`'s bias) is not entered. A visitor that pushes and
  pops a path stack therefore names the CANN's weights `hidden.0.weight`
  (shape `[inputs, width]`), `hidden.0.bias`, ..., `out.weight`,
  `out.bias`, and the attention CANN's `embed.0.weight`, `cls`,
  `blocks.0.attention.query.weight`, ..., `norm.gamma`.
* `ModuleMapper::map_float` with the same stack sees the parameters in
  the same order, so loading can check each path and shape and replace
  the tensor (`Param::consume`, `Tensor::from_data`,
  `Param::from_mapped_value`). On the `NdArray<f64>` backend the values
  are `f64` and, written with serde_json's `float_roundtrip`, a loaded
  network predicts bit for bit as the saved one.
* Burn's parameter initializers draw from a generator global to the
  backend, and parameters initialize lazily. A network rebuilt for
  loading is built and materialized under the same lock as a fit
  (`seeded`), so that loading in one thread cannot move another
  thread's seeded initialization.
* Burn's own recorders (`BinBytesRecorder`, `NamedMpkBytesRecorder`) are
  available under its `std` feature but write an opaque byte format; the
  model artifact rule (readable without our code) favours the plain
  JSON weights, a few thousand numbers for a tabular network.
