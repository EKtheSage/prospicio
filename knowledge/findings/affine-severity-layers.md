---
type: Finding
title: A scaled and shifted severity keeps exact layer moments
description: "For Y = s X + b with s > 0 and b >= 0, S_Y(y) = 1 below b and S_X((y - b) / s) above it. A layer of Y from a to a + l splits at b: the part c below b has survival 1, and the substitution y = b + s t turns the part [lo, hi] above b into s times the layer of X of width (hi - lo) / s at (lo - b) / s. Its second moment 2 ∫ (y - a) S_Y is c^2 + s^2 m2 + 2 (lo - a) s m1, with m1 and m2 that layer of X's first two moments; the unit tests check both against midpoint sums to 1e-6."
tags: [prospicio-prob, severity, affine, layers, aggregate-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T14:40:00Z }
verified: { by: process:ci, at: 2026-10-19T14:40:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/affine.rs
    title: prospicio_prob::Affine and its unit tests
  - id: aggregate
    resource: https://aggregate.readthedocs.io/
    title: aggregate's DecL, scaled and shifted severities (10 * lognorm 1 + 20)
---

# Facts

* `Y = s X + b`: `F_Y(y) = F_X((y − b) / s)`, `E[Y] = s E[X] + b`,
  `Var Y = s² Var X`, `E[Y³] = s³ m₃ + 3 s² b m₂ + 3 s b² m₁ + b³`.
* A layer `l xs a` of `Y` is `(min(a + l, b) − a)⁺ + s · layer_X((hi − lo) /
  s, (lo − b) / s)` with `lo = max(a, b)` and `hi = a + l`: the first term
  is the part of the layer below the shift, where the loss is sure.
* The layer's second moment is `c² + s² m₂ + 2 (lo − a) s m₁`, `c` the
  part below `b` and `m₁`, `m₂` the moments of that layer of `X`, since
  `y − a = s (t − t₀) + (lo − a)` under `y = b + s t`.
* Draws are `s x + b` of the inner severity's draws, so `(seed, stream)`
  gives the same draws as the inner severity, transformed.

# Why

`aggregate`'s DecL allows `10 * lognorm 1 + 20`; before #318 prospicio
could only discretize and `Grid.map`, losing the continuous form.
