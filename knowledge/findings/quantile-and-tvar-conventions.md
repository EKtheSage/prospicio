---
type: Finding
title: Sample quantile and TVaR conventions (inverted cdf, linear, k-th largest year), and how far apart they are
description: "prospicio's Sampled uses the inverted empirical cdf (R type 1) and the exact empirical TVaR (mass 1/n per draw); numpy's default and R's quantile(type = 7) interpolate linearly at h = p (n - 1), and the matching TVaR is the mean of that piecewise-linear quantile function over [p, 1]. On ten draws the two differ by tens of percent in the tail; on 100,000 draws by about one draw's spacing."
tags: [prospicio-prob, quantiles, tvar, numpy, conventions, statistics, catastrophe]
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
    title: the unit tests of the statistics grid and of risk.rs, and test_statistics_match_numpy_linear_and_the_integral_tvar and test_the_kth_largest_year_and_the_whole_years_tvar in python/tests/test_risk.py
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

A third pair is the catastrophe pricer's (#582): the `k`-th largest of
`N` years and the mean of the `k` largest, `k = max(1, floor((1 - p) N))`
(`KthLargest`, `WholeYears`; `"kth_largest"`, `"whole_years"`). The
`k`-th largest is the upper quantile `inf {x : F(x) > p}`: at `N = 2000`
and `p = 0.99` it is the 20th worst year, where the inverted cdf gives the
21st, so it sits one order statistic above the default (equal only at the
largest draw). The whole-years TVaR equals the exact one whenever
`(1 - p) N` is whole (every round level at 2,000 years: 99, 99.5, 99.6,
99.8) and drops the fraction of a year the exact one keeps otherwise, so
it is no smaller. Computing `k` as `floor((1 - p) N)` in floating point
loses a year at levels such as `p = 0.9`, `N = 10` (`1 - 0.9` is
`0.09999999999999998`), so prospicio takes `k = N - rank(N, p)` with the
default VaR's comparison `r / N >= p`.

`StatisticsGrid` names its columns `Mean`, `SD` (population),
`CoV`, `Min`, `Max`, `Variance`, `Percentile <q>`, `TVaR <q>`,
`Mean + <k> SD` and `Mean - <k> SD`, with the levels in percent as given,
so a table from another tool can be joined on the names.

A bootstrap standard error of these statistics (`StatisticsGrid::bootstrap_se`)
needs no resampled copy of the columns. A replicate is a count of how
often each simulation was drawn; on a column sorted once, the cumulative
counts give the k-th order statistic by a binary search and a suffix sum
gives the tail beyond it, so both conventions' percentiles and TVaRs of a
replicate cost O(n) without a sort. The unit tests check this against the
replicate written out and sorted, to 1e-10, under all nine pairings of the
quantile and TVaR conventions. Every column of a replicate uses the same
draw, so the items' errors come from the same resamples. Holding a fixed
part of each simulation (`bootstrap_se_holding`: simulation i is
`fixed[i] + rest[s_i]`) breaks the sorted order, so that replicate is
written out and sorted, at O(n log n).
