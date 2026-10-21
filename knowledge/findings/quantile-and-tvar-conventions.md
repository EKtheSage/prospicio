---
type: Finding
title: Two sample quantile and TVaR conventions, and how far apart they are on a small sample
description: "prospicio's Sampled uses the inverted empirical cdf (R type 1) and the exact empirical TVaR (mass 1/n per draw); numpy's default and R's quantile(type = 7) interpolate linearly at h = p (n - 1), and the matching TVaR is the mean of that piecewise-linear quantile function over [p, 1]. On ten draws the two differ by tens of percent in the tail; on 100,000 draws by about one draw's spacing."
tags: [prospicio-prob, quantiles, tvar, numpy, conventions, statistics]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T20:00:00Z }
verified: { by: process:ci, at: 2026-10-21T20:00:00Z }
sources:
  - id: hyndman-fan
    resource: https://doi.org/10.2307/2684934
    title: Hyndman and Fan (1996), Sample quantiles in statistical packages, The American Statistician 50
  - id: numpy
    resource: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    title: numpy.quantile, the method argument (linear is the default)
  - id: test
    resource: ../crates/prospicio-prob/src/statistics.rs
    title: the unit tests of the statistics grid, and test_statistics_match_numpy_linear_and_the_integral_tvar in python/tests/test_risk.py
---

# Two sample quantile and TVaR conventions

A results table from another tool states a percentile and a TVaR of
simulated draws. Two conventions are common, and prospicio offers both
(`prospicio_prob::statistics`, issue #474):

| | quantile at `p` | TVaR at `p` |
|---|---|---|
| prospicio's default (`InvertedCdf`, `Exact`) | the smallest draw with `F(x) >= p`, R `type = 1` | the mean of the worst `1 - p` of the empirical distribution, splitting the draw at the VaR |
| `Linear` | `x[k] + f (x[k+1] - x[k])` with `h = p (n - 1)`, `k = floor(h)`, `f = h - k`: numpy's default, R `type = 7` | the mean of that piecewise-linear quantile function over `[p, 1]`, by trapezoids on the positions `0 .. n - 1` |

The linear TVaR at `p = 0` is not the mean of the draws: it is
`(sum - (x[0] + x[n-1]) / 2) / (n - 1)`, the trapezoid rule's weights.
On the ten draws `7, 0, 100, 3, 1, 0, 45, 7, 12, 20` (mean 19.5) it is
`145 / 9 = 16.11`, and TVaR 90 is 75.25 linear against 100 exact; the
25th percentile is 1.5 linear against 1 inverted. The difference shrinks
with `n` to about one spacing between neighbouring order statistics, so
it matters for parity on a small example and little on a large run.

`StatisticsGrid` names its columns `Mean`, `SD` (population),
`CoV`, `Min`, `Max`, `Variance`, `Percentile <q>`, `TVaR <q>`,
`Mean + <k> SD` and `Mean - <k> SD`, with the levels in percent as given,
so a table from another tool can be joined on the names.
