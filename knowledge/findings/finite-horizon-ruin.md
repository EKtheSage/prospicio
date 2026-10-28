---
type: Finding
title: Finite-horizon ruin by a convolution recursion
description: psi(u, t) by the period recursion on the grid is exact for a random walk, agrees with 200,000 simulated paths at three standard errors, and with Poisson claims split into sub-periods approaches the continuous eventual ruin from below, 0.03 short at 30 steps a period on a step of 0.05.
tags: [aggregate, ruin, finite-horizon, fft, monte-carlo]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-27T21:00:00Z }
sources:
  - id: ruin
    resource: ../crates/prospicio-aggregate/src/ruin.rs
    title: prospicio_aggregate::ruin (finite_horizon, finite_horizon_poisson, FiniteRuin)
  - id: validation
    resource: ../validation/tests/aggregate.rs
    title: validation/tests/aggregate.rs (finite_horizon_ruin_matches_monte_carlo)
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/562
    title: Issue #562, finite-horizon ruin probability
verified: { by: process:ci, at: 2026-10-27T21:00:00Z }
---

# Finding

`ψ(u, t)`, the chance that the surplus `u + t c - S_1 - ... - S_t` is
negative at the end of some period up to `t`, follows from
`ψ_t(u) = P(S > u + c) + Σ_{s <= u + c} p(s) ψ_{t-1}(u + c - s)` with one
FFT convolution a period. No reference implementation was found
(`aggregate` 1.2.0 has only eventual ruin), so it is checked three ways.

- **A random walk, exactly.** A claim of 2 with probability 0.4 and a
  premium of 1 is a walk of +1 or -1. Enumerating all `2^t` paths for
  `t <= 12` matches the recursion to 1e-13; at 2000 periods `ψ(u)`
  equals the eventual `(0.4/0.6)^(u+1)` to `1e-12` (the remainder is
  about `0.96^1000`).
- **Monte Carlo.** A gamma severity (mean 1, cv 0.5) rounded on a step of
  0.0025, Poisson(2) claims and a premium of 2.4 a period, over 10
  periods: 200,000 simulated paths (stream seed 20261027, 562) agree at
  `u` = 0, 1, 2, 4 and `t` = 1, 5, 10 within three standard errors.
- **The continuous limit.** With exponential claims of mean 1, rate 1 and
  premium 1.5, the continuous eventual ruin is `exp(-u/3) / 1.5`. Split
  into 30 Poisson steps a period over 60 periods on a step of 0.05, the
  recursion is below it at every `u` and within 0.03. The gap is the ruin
  between checks, which shrinks as the steps do, plus the rounding of the
  severity.

The grid must hold the surplus plus the tail: beyond the last point `ψ`
reads the last value, an upper bound. A test that reads `ψ(3)` on a
three-point grid gets `ψ(2)`.
