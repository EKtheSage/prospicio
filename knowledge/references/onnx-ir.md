---
type: Reference Implementation
title: onnx-ir 0.21, Burn's ONNX parser, at run time
description: onnx-ir 0.21.0, the parser behind burn-import, reads an ONNX file at run time into typed nodes with their constants; it builds without protoc (about 35 s), fuses MatMul then Add into Linear, drops Identity and no-op Flatten, keeps Gemm, and holds Rc values, so prospicio_nn::onnx copies the graph into an owned plan.
resource: https://crates.io/crates/onnx-ir
tags: [onnx, burn, prospicio-nn, import, pytorch]
status: stable
verified: { by: process:ci, at: 2026-10-16T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-16T12:00:00Z }
sources:
  - id: crate
    resource: https://docs.rs/crate/onnx-ir/0.21.0
    title: onnx-ir 0.21.0 sources (src/pipeline.rs, src/ir, src/node)
  - id: tests
    resource: ../crates/prospicio-nn/tests/onnx.rs
    title: prospicio-nn's ONNX parity tests
---

# Facts

* `burn-import` turns an ONNX file into Rust source at build time, so it
  cannot load a user's file at run time. Its front end, `onnx-ir`, can:
  `OnnxGraphBuilder::new().parse_bytes(&bytes)` (or `parse_file`) gives
  an `OnnxGraph` of typed `Node`s (`Node::Gemm(GemmNode { config, inputs,
  outputs, .. })`) with graph `inputs` and `outputs`.[^crate]
* 0.21.0 depends on `burn-tensor ^0.21`, so it matches the workspace's
  Burn. Its `protobuf-codegen` dependency needs no `protoc` binary; a
  cold build of the crate took about 35 s in the cloud container. Burn
  0.22.0 (and onnx-ir 0.22.0) are out; the workspace stays on 0.21.[^crate]
* Weights and other constants come back through `Argument::value()`, a
  `burn_tensor::TensorData` (`to_f64_vec()` from `TensorDataExt`, shape
  by `data.shape.to_vec()`), for arguments whose `value_source` is
  `Static` or `Constant`; `Dynamic` arguments are computed values.[^crate]
* The parse simplifies the graph: `MatMul` followed by `Add` becomes
  `Linear` (`transpose_weight: false`, weight `[in, out]`), `Identity`
  and a `Flatten` that changes nothing are removed, and initializers
  become `Constant` nodes. A `Gemm` whose weight comes from such a
  constant stays a `Gemm` (with `alpha`, `beta`, `trans_a`, `trans_b`).
  Axes and shapes given as constant inputs move into the node's
  configuration (`SqueezeInput::Static`, `ReshapeInput::Static`,
  `ReduceConfig { dims, keepdims }`).[^tests]
* The graph holds `Rc` references, so it is neither `Send` nor `Sync`;
  `prospicio_nn::onnx::OnnxModel` copies it at load into its own steps
  and `f64` tensors.[^tests]

[^crate]: onnx-ir 0.21.0 sources
[^tests]: `crates/prospicio-nn/tests/onnx.rs` and `crates/prospicio-nn/src/onnx.rs`
