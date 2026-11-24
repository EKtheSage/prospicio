---
type: Finding
title: The FFT aliasing audit needs one transform, and a severity a real one
description: The compound pmf on a buffer of L points is the pmf on 2L points folded onto its first half, so fft's aliasing_error (the L result against the 2L one) comes from the 2L transform alone; with a real-input transform for the severity the 2^20-point compound went from 874 ms to 313 ms, and the 2^14-point one from 8.8 ms to 1.9 ms
tags: [aggregate, fft, aliasing, performance, math]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-24T12:00:00Z }
verified: { by: process:ci, at: 2026-11-24T12:00:00Z }
sources:
  - id: fft
    resource: ../crates/prospicio-aggregate/src/fft.rs
    title: prospicio_aggregate::fft (compound_pair, compound_buffer; the tests one_transform_gives_both_buffers and the_real_transform_matches_the_complex_one)
  - id: math
    resource: ../crates/prospicio-math/src/fft.rs
    title: prospicio_math::fft (forward, inverse, real_forward, real_inverse, CACHED_LEN)
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/973
    title: "#973: consolidate rustfft into prospicio-math, reuse the planner and scratch, evaluate realfft and an optional aliasing check"
---

# Finding

`fft` measures aliasing by comparing the compound pmf on a circular buffer
of `L` points with the one on `2L` points [fft]. Until #973 it ran the whole
calculation at both lengths, three units of work for a result worth two
[issue]. The short result is in the long one:

```
p_L[k] = p_2L[k] + p_2L[k + L],   k = 0, ..., L - 1
```

The severity wrapped onto `L` points has as its transform every second
coefficient of the transform on `2L` points; the claim-count pgf acts on
each coefficient alone; and the inverse transform of every second
coefficient is the long result folded onto its first half. So the audit is
one pass over the `2L` buffer, and it stays a measurement: no option to
switch it off is needed. The fold is taken before round-off negatives are
clamped at 0, and agrees with a transform at `L` to `1e-15` with and without
wrapped mass, and with a severity longer than the short buffer
(`one_transform_gives_both_buffers`).

A severity is real, so its transform has `n / 2 + 1` distinct coefficients,
the rest being conjugates. A pgf has real coefficients, so
`P(conj z) = conj P(z)`, and the pgf need only be evaluated at those
`n / 2 + 1` points; the inverse is a complex-to-real transform. The result
matches the complex transform to `1e-15`
(`the_real_transform_matches_the_complex_one`). A count whose
`pgf_complex` broke the conjugate rule would give a different result than
before; none of `prospicio-prob`'s does, and no pgf can.

## Timings

One `fft` call, negative binomial count (`r = 40`, `p = 0.5`), lognormal
severity rounded to a step of 0.01, release build on the cloud container
(4 cores, single thread), before and after:

| Points | Before | After | Ratio |
|---|---|---|---|
| `2^14` | 8.80 ms | 1.93 ms | 4.6 |
| `2^17` | 86.5 ms | 30.3 ms | 2.9 |
| `2^20` | 874 ms | 313 ms | 2.8 |

The pgf evaluations (a complex power each) are most of the small grids'
time, which is why halving them gains more there.

## Plans per thread

`prospicio_math::fft` keeps each thread's `rustfft` and `realfft` planners
and one scratch buffer in a thread-local, for lengths up to `CACHED_LEN`
(`2^16`); a longer transform plans for itself and drops the plan [math]. A
plan of `n` points holds about `16 n` bytes of twiddle factors, so an
unbounded cache would leave 32 MB on every rayon thread after one
`2^21`-point transform, and planning is a small part of a transform that
long.

## Not yet moved

The other FFT sites still build a planner per call: `bivariate`, `renewal`
and `ruin` in `prospicio-aggregate`, `natural` in `prospicio-pricing`, `chf`
in `prospicio-prob` and `prospicio-bayes`. They are the Aggregate
lane's next item under #973.
