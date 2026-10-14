---
type: Finding
title: Renewal claim counts against aggregate
description: The Plancherel renewal count matches aggregate 1.0.1 at 1e-8 on its own wait grid, and is closer to the exact values (5e-11 against aggregate's 6e-9 at k = 0 and 1); packing the tilted wait and the tilted indicator into one complex FFT ruins it, because the indicator is up to e^15 times the wait.
tags: [aggregate, renewal, frequency, fft, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T23:30:00Z }
sources:
  - id: renewal
    resource: ../crates/prospicio-aggregate/src/renewal.rs
    title: prospicio_aggregate::renewal (renewal_count, renewal_count_pmf, zero_wait_batches, WaitGrid)
  - id: script
    resource: ../validation/scripts/aggregate_renewal.py
    title: validation/scripts/aggregate_renewal.py (targets from aggregate 1.0.1's renewal frequency)
  - id: aggregate
    resource: https://github.com/mynl/aggregate
    title: mynl/aggregate 1.0.1, aggregate/_renewal.py (renewal_count_pmf, geometric_batch_compose, wait_grid, wait_count_pmf)
---

# Finding

`aggregate` 1.0.1 counts the claims of a renewal process by horizon `T`
with no inverse FFT: `P(N = k) = F^{*k}(T) - F^{*(k+1)}(T)`, each term the
Plancherel inner product of the tilted wait's transform to the `k`-th power
with the transform of the tilted indicator of `[0, T]` [renewal]
[aggregate]. The tilt `e^{-θx}` with `θL = 20` damps the wrap of mass past
the grid, and puts `e^{θx}` on the indicator: up to `e^{15}` at `T`, which
sits at three quarters of the grid.

- **Two transforms, not one packed.** Two real sequences fit in one complex
  FFT (one as the real part, one as the imaginary part, split afterwards
  by symmetry), but here the indicator is up to `e^{15}` times the wait,
  and the split leaves the indicator's round-off on the wait's transform,
  which the `k`-th power then amplifies: exponential waits over 2 years
  gave probabilities summing to 40. Separate transforms match `aggregate`
  [renewal].
- **Accuracy.** On `aggregate`'s own wait grid the counts match at `1e-8`
  on the renewal cases of its test suite (exponential, gamma, fixed,
  clustered, defective and mixed waits) [script]. Where they differ,
  prospicio is the closer: `P(N = 0)` for exponential waits over 10 years
  is `e^{-10}` within 3e-11, `aggregate`'s within 6.1e-9, and likewise
  `P(N = 1)` and the gamma and mixture cases at `k = 0` (1e-9 and 1.5e-9
  for `aggregate`, under 1.1e-11 for prospicio). The difference is the
  round-off of the tilted sums: `numpy.fft.rfft` against `rustfft`'s full
  transform.
- **The grid.** `aggregate`'s `wait_grid` sizing (shape, accuracy and
  coverage rules on `2^16` to `2^24` points, or the exact lattice of
  discrete waits) is reproduced exactly, step, points and reading, so a
  parity test needs only the wait law. The lattice step needs Python's
  `Fraction.limit_denominator(10^9)`: the exact binary value of each wait,
  then its continued fraction.
- **Zero waits.** `aggregate`'s `geometric_batch_compose` counts the zero
  waits before the first positive one: `N = G_1 + … + G_{M+1} - 1`, not
  `G_1 + … + G_M`; waits of 0 or 1 with even odds over one period give
  `P(N = n) = n / 2^(n+1)` exactly.
