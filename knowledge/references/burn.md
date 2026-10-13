---
type: Reference Implementation
title: Burn 0.21's activations in f64
description: Burn 0.21, the deep learning framework behind prospicio-nn; its default sigmoid and hard_sigmoid compute in f32 even on an f64 backend, gelu's backward pass is the tanh approximation's with six-digit constants, and prospicio-nn uses neither.
resource: https://github.com/tracel-ai/burn
tags: [burn, prospicio-nn, autodiff, precision, f64, activations]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T12:00:00Z }
sources:
  - id: backend-activation
    resource: https://docs.rs/crate/burn-backend/0.21.0/source/src/backend/ops/activation.rs
    title: burn-backend 0.21.0, the default ActivationOps
  - id: ndarray-activation
    resource: https://docs.rs/crate/burn-ndarray/0.21.0/source/src/ops/activation.rs
    title: burn-ndarray 0.21.0, the NdArray overrides (relu only)
  - id: autodiff-activation
    resource: https://docs.rs/crate/burn-autodiff/0.21.0/source/src/ops/activation.rs
    title: burn-autodiff 0.21.0, the backward passes
  - id: tensor-activation
    resource: https://docs.rs/crate/burn-tensor/0.21.0/source/src/tensor/activation/base.rs
    title: burn-tensor 0.21.0, activations composed from tensor ops
  - id: nn
    resource: ../crates/prospicio-nn/src/lib.rs
    title: prospicio-nn (backends Cpu = NdArray<f64>, Train = Autodiff<Cpu>)
---

# What computes in f32

`Cargo.lock` pins every `burn-*` crate at 0.21.0. In `burn-backend`'s default
`ActivationOps` (`src/backend/ops/activation.rs`):

* `sigmoid` (lines 139 to 148) casts its input to `FloatDType::F32`, computes
  `exp(-log(exp(-x) + 1))` and casts back to the input's dtype. On
  `NdArray<f64>` the result carries f32 rounding, about 1e-7 relative.
* `hard_sigmoid` (lines 179 to 189) does the same cast.
* `sigmoid_backward` is `output * (1 - output) * grad` in the input dtype, but
  `burn-autodiff` feeds it the forward output, so the gradient inherits the
  f32 error. An autodiff benchmark on 2026-10-13 with `Autodiff<NdArray<f64>>`
  measured a sigmoid gradient off by 1.4e-7 relative, against about 1e-15
  when sigmoid is written out as `1 / (1 + exp(-x))` from f64 tensor ops.
* `silu` (`burn-tensor`, `x * sigmoid(x)`) inherits the same error.

`burn-ndarray` overrides only `relu`, so every other default above is what
runs on the CPU backend.

# What stays in the input's precision

No cast in `log_sigmoid` and its backward pass, `softmax`, `log_softmax`,
`relu`, `tanh` (the backend's `float_tanh`), `softplus` (composed as
`log(exp(beta x) + 1) / beta`, not with `log1p`, so it overflows for large
`beta x` and loses relative precision for very negative `beta x`, in any
dtype), `mish`, `exp`, `log` and `log1p`.

`gelu`'s forward pass is the exact erf form, but `gelu_backward` (lines 98 to
125) is the derivative of the tanh approximation, with constants rounded to
six digits (`0.0356774`, `0.797885`, `0.0535161`, `0.398942`). Its gradient
is therefore not the derivative of its forward pass, whatever the dtype; do
not expect an f64 gradient check on `gelu` to pass at tight tolerance.

# prospicio-nn

As of 2026-10-13, `prospicio-nn` trains on `Autodiff<NdArray<f64>>` and calls
only `relu` (`lib.rs`, `attention.rs`), `exp` in the deviance losses, and
burn-nn's `MultiHeadAttention` and `LayerNorm` (softmax inside, no cast). None
of the affected ops are used, so its bit-exact replay tests and its `1e-15`
zero-epoch checks are unaffected. A future smooth activation should be
written from f64 tensor ops (for a sigmoid, `1 / (1 + exp(-x))`, or
`exp(log_sigmoid(x))`) rather than taken from `burn::tensor::activation`,
and checked against finite differences or a closed form.
