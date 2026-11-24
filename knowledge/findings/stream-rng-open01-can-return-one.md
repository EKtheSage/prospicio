---
type: Finding
title: StreamRng::next_open01 can return exactly 1.0
description: prospicio_core's StreamRng::next_open01 computes (k + 0.5) 2^-53 from the top 53 bits k of a draw; for k >= 2^52 the half step is rounded away (ties to even), and k = 2^53 - 1 rounds to 2^53, so the draw is exactly 1.0 with probability 2^-53; it is never 0.0. Code that needs a value strictly below 1, such as a normal quantile, must cap it until the core crate is fixed.
tags: [rng, streams, uniform, floating-point, reinsurer-default]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T00:00:00Z }
sources:
  - id: rng
    resource: ../crates/prospicio-core/src/rng.rs
    title: crates/prospicio-core/src/rng.rs, StreamRng::next_open01
  - id: panel
    resource: ../crates/prospicio-aggregate/src/reinsurance/panel.rs
    title: crates/prospicio-aggregate/src/reinsurance/panel.rs, the cap below 1 (#980, #1068)
---

# Finding

`StreamRng::next_open01` returns `(k as f64 + 0.5) * 2^-53` with `k` the
top 53 bits of a 64-bit draw. Its name and documentation promise a value
in the open interval (0, 1), and the smallest value, `0.5 * 2^-53`, is
indeed above 0. At the top it is not:

- For `k` in `[2^52, 2^53)` the spacing of doubles is 1, so `k + 0.5` is
  a tie and rounds to even: an even `k` keeps `k`, an odd `k` goes to
  `k + 1`. The half step is lost on the upper half of the range.
- For `k = 2^53 - 1` (odd) the sum rounds to `2^53`, and the result is
  exactly `1.0`. That happens with probability `2^-53` per draw.

A caller that takes `norm_quantile(u)` or `ln(1 - u)` of the draw gets an
infinity or a NaN in that case. `Panel` (#980, #1068) caps its rank at
`1 - EPSILON / 2` and treats a probability of default of 1 as a default
whatever the draw. A fix in `prospicio-core` (for example
`((k << 1) | 1) as f64 * 2^-54`, exact for every `k`) changes the bits of
every stream's uniforms, so it is a change of its own with a stability-log
entry in `rng.md`, not part of #980.

Checked by hand on 2026-11-24 (`(2^53 - 1 + 0.5) * 2^-53 == 1.0` in IEEE
754 double precision); no test in CI covers it yet.
