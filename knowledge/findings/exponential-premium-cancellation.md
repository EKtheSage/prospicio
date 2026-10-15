---
type: Finding
title: The exponential premium loses its loading to cancellation at a small k
description: log E[e^(kX)] / k, as written, loses the loading below k of about 1e-8 (at k = 1e-12 a lognormal's premium came out 0.025 below its mean); taken about the mean as mean + ln_1p(E[expm1(k (X - mean))]) / k it keeps mean + k var / 2 to 1e-14.
tags: [pricing, premium-principles, numerics]
status: stable
verified: { by: process:ci, at: 2026-10-14T23:30:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-14T23:30:00Z }
sources:
  - id: tests
    resource: ../crates/prospicio-pricing/src/classical.rs
    title: the exponential_keeps_a_small_loading unit test
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/214
    title: Small API consistency items found by the guides
---

The zero-utility premium `log E[e^(kX)] / k` tends to the mean as `k`
goes to 0. Computed directly, `E[e^(kX)]` is `1 + k E[X] + ...`, and the
logarithm of a number that close to 1, divided by `k`, keeps only
`eps / k` of precision. On 10,000 lognormal draws the premium less the
mean was `3.7e-8` at `k = 1e-8` (right), `-1.3e-5` at `1e-10` and `-0.025`
at `1e-12`, below the mean.[^issue]

`Principle::Exponential` now centres at the mean, so the loading is the
only thing left to compute: `mean + ln_1p(E[expm1(k d)]) / k` with
`d = X − mean` when every `|k d| ≤ 1`, otherwise
`mean + log E[e^(k d)] / k` with the largest exponent factored out. The
`k d` terms in the sum cancel to `k² var / 2`, and their round-off,
divided by `k`, is `eps × |d|` in the premium. On two equally likely
values 0 and 10 the premium is `5 + 12.5 k` to `1e-14` from `k = 1e-4`
down to `±1e-12`.[^tests]

[^issue]: Issue #214, the unconfirmed items
[^tests]: prospicio_pricing::classical tests
