---
type: Finding
title: A growth curve fitted to a benchmark pattern reproduces Clark's ultimates in the chain ladder, and matches scipy's least squares
description: Clark's log-logistic or Weibull curve fitted by weighted least squares on the percents developed recovers Clark's own omega and theta from his percents to 1e-7; its pattern in the chain ladder, with no tail, gives Clark's LDF ultimates to 1e-7; on a ten-age benchmark the parameters agree with scipy.optimize.least_squares (MINPACK LM) to 1e-7 and the sum of squares to 1e-9.
tags: [reserving, growth-curve, clark, benchmark-pattern, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-26T12:00:00Z }
verified: { by: process:ci, at: 2026-10-26T12:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-reserving/src/benchmark_curve.rs
    title: prospicio_reserving::benchmark_curve, BenchmarkCurve and its unit tests
  - id: parity
    resource: ../validation/tests/reserving_benchmark_curve.rs
    title: Parity with scipy.optimize.least_squares, run in every cargo test
  - id: script
    resource: ../validation/scripts/reserving_benchmark_curve_scipy.py
    title: The scipy generator of the reference values
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, decision 20
---

# What was found

* **Clark's pattern carries over exactly.** Clark's LDF method develops a
  latest value as `latest * G(c(m)) / G(c(a))`, `c` the age from the
  average date of loss and `m` the `max_age`. That is `latest / p(a)` for
  the percent developed `p(a) = G(c(a)) / G(c(m))`. Fitting the same curve
  to Clark's own `p` at 12 to 96 months, with `origin_width = 12`,
  recovers his `omega` and `theta` to 1e-7 for both curves, with and
  without `max_age`, and the fitted pattern at the triangle's six ages in
  `ChainLadder` with no tail gives his ultimates to 1e-7. The chain
  ladder's tail is then the pattern's level at the oldest age (decision
  12), `1 / p(oldest)`, the curve's development past it.
* **Least squares parity.** On a ten-age benchmark (12 to 120 months), the
  Nelder-Mead fit on `(ln omega, ln theta)` to 1e-12 and scipy 1.18.1's
  `least_squares(method="lm", ftol = xtol = gtol = 1e-15)` find the same
  minimum: parameters within 1e-7 relative and the weighted sum of squares
  within 1e-9, for the log-logistic, a Weibull with `max_age = 240` and
  the shift, and a weighted log-logistic with the shift.
* **The start.** The straight line of `logit p` (log-logistic) or
  `ln(-ln(1 - p))` (Weibull) on `ln c(a)` is exact for an exact curve
  without `max_age`, so the search starts at or near the answer; Clark's
  own start (R's) is the fallback when fewer than two percents lie in
  (0, 1).
* **An untruncated curve stays below 100%.** A benchmark that reaches 1 at
  a finite age cannot be fitted there without `max_age`; with it the
  pattern is exactly 1 at `max_age`.
* **A benchmark that does not increase runs off to an edge.** On flat
  percents (0.5 at 12, 24 and 36 months) the least-squares search drives
  omega to about 1e-17, and on decreasing ones (0.5, 0.4, 0.3) theta
  overflows to `f64::MAX` with omega near 6e-4; both converge to a flat
  line near the mean. `fit` therefore returns an error when omega leaves
  [1e-6, 1e6] or `|ln theta|` passes 700.

# Where it is used

`BenchmarkCurve` in Rust, Python and R (`benchmark_curve()`), issue #637.
