---
type: Finding
title: The FFT aliasing audit needs one transform, and a severity a real one
description: The compound pmf on a buffer of L points is the pmf on 2L points folded onto its first half, so fft's aliasing_error (the L result against the 2L one) comes from the 2L transform alone; with a real-input transform for the severity the 2^20-point compound went from 874 ms to 313 ms, and the 2^14-point one from 8.8 ms to 1.9 ms
tags: [aggregate, fft, aliasing, performance, math]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-24T18:00:00Z }
verified: { by: process:ci, at: 2026-11-24T12:00:00Z }
sources:
  - id: fft
    resource: ../crates/prospicio-aggregate/src/fft.rs
    title: prospicio_aggregate::fft (compound_pair, compound_buffer; the tests one_transform_gives_both_buffers and the_real_transform_matches_the_complex_one)
  - id: math
    resource: ../crates/prospicio-math/src/fft.rs
    title: prospicio_math::fft (forward, inverse, real_forward, real_inverse, Plan, RealPlan, CACHED_LEN)
  - id: natural
    resource: ../crates/prospicio-pricing/src/natural.rs
    title: prospicio_pricing::natural (Portfolio::from_independent, convolve)
  - id: bayes
    resource: ../crates/prospicio-bayes/src/lib.rs
    title: prospicio_bayes (autocovariance, behind ess_bulk and ess_tail)
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

## Plans

Planning a transform (computing its twiddle factors) costs two to four
times the transform. Both directions of the real transform, release build,
cloud container:

| Points | Plan | Transform |
|---|---|---|
| `2^10` | 25.5 us | 4.0 us |
| `2^14` | 392 us | 101 us |
| `2^17` | 3.46 ms | 0.98 ms |
| `2^20` | 30.9 ms | 13.4 ms |

So a plan is worth keeping at every length. `prospicio_math::fft`'s free
functions keep each thread's `rustfft` and `realfft` planners and one
scratch buffer in a thread-local, for lengths up to `CACHED_LEN` (`2^16`);
a longer transform through them plans for itself and drops the plan
[math]. A plan of `n` points holds about `16 n` bytes of twiddle factors,
so an unbounded cache would leave 32 MB on every rayon thread after one
`2^21`-point transform. A caller that transforms one length several times
takes a `Plan` or a `RealPlan` and owns it, at any length: the finite-time
ruin recursion (one transform a stage), the Wiener-Hopf factorization (seven)
and the 2-D transform's rows and columns. This section first said that
planning is a small part of a long transform, which the table shows to be
wrong; the handles came with the correction.

## The other FFTs of `prospicio-aggregate`

Moved in the second PR of #973, each against its old complex transform on
the same inputs:

| Calculation | Before | After |
|---|---|---|
| `fft_bivariate`, 512 by 512 points | 116 ms | 81 ms |
| `fft_bivariate`, 64 by 64 points | 0.95 ms | 0.78 ms |
| `finite_horizon`, 20 periods, 4,000 points | 6.7 ms | 2.8 ms |
| `finite_horizon_poisson`, 10 periods of 12, 4,000 points | 23.7 ms | 14.0 ms |
| `wiener_hopf`, 2,000 points | 0.61 ms | 0.52 ms |
| `renewal_count`, gamma waits over 10 years | 2.45 s | 2.19 s |

The bivariate compound and the finite-time recursion take real transforms
(half the pgf evaluations, half the spectrum); the Wiener-Hopf
factorization keeps its complex ones, since the logarithm of the
transform is not a real sequence's, and gains only the scratch buffer.
The renewal count's time is its loop over the powers of the wait's
transform, not the two transforms.

## `natural::Portfolio`

Moved in the third PR of #973. `Portfolio::from_independent` transforms
each unit's probabilities and its loss-weighted probabilities and inverts
two products a unit, `4 m + 1` transforms of one length for `m` units, all
of real sequences: real transforms on one kept plan, and the prefix and
suffix products on half the spectrum. Lognormal units on a step of 0.05,
release build, cloud container:

| Calculation | Before | After |
|---|---|---|
| 3 units of 256 points | 168 us | 55 us |
| 8 units of 4,096 points | 27.2 ms | 7.4 ms |
| 20 units of 16,384 points | 1.24 s | 0.56 s |
| the convolution of two 4,096-point grids | 542 us | 82 us |
| the convolution of two 16,384-point grids | 1.64 ms | 0.37 ms |

The convolution is the one behind the stand-alone prices' sum of two
units.

## The effective sample size

Moved in the fourth PR of #973. `prospicio-bayes` computes each chain's
autocovariances by FFT, once per chain and parameter for the bulk and the
tail effective sample sizes, always at the same length. It built a
planner for every one; with real transforms through the free functions
the thread's plan is reused. Release build, cloud container:

| Chain length | Before | After |
|---|---|---|
| 500 | 41.0 us | 4.6 us |
| 1,000 | 87.3 us | 10.2 us |
| 10,000 | 2.45 ms | 0.21 ms |

Most of the gain is the plan, as the table under "Plans" predicts.

## Not yet moved

`chf` in `prospicio-prob` still builds a planner per call. It is the
Aggregate lane's next item under #973, after #972.
