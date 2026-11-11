---
type: Finding
title: Python's ALE order of factor levels agrees with R's cmdscale, and a dominant negative eigenvalue did not occur in the scaled distances
description: "On a five-level factor whose doubly centred squared distance matrix has eigenvalues -0.0675, 0, 0.0287, 0.1303 and 0.1765, Python's power iteration and R's cmdscale give ALEPlot 1.1's order p, q, s, t, r and its effects; over about 600,000 searched distance matrices (half-L1 distances of conditional frequencies, small integer metrics, graph metrics) the most negative eigenvalue never exceeded 0.69 of the largest positive one, so the iteration's suspected failure needs a matrix these distances did not produce."
tags: [explanations, ale, classical-scaling, python-parity, r-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-11T12:00:00Z }
verified: { by: process:ci, at: 2026-11-11T12:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/r_explain_parity.R
    title: validation/scripts/r_explain_parity.R (ALEPlot 1.1, partial dependence and ICE)
  - id: reference
    resource: ../validation/reference/explain_parity.csv
    title: validation/reference/explain_parity.csv
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/837
    title: "#837, a cross-language parity test for the explanation tools"
---

# Python's ALE order of factor levels agrees with R's cmdscale

ALEPlot orders a factor's levels by the first coordinate of classical
scaling of the summed distances between the levels' distributions of the
other columns. R's `ale()` calls `stats::cmdscale`; Python's power iteration
on `B = -J D^2 J / 2` converges to the eigenvalue of largest magnitude. The
two differ only when a negative eigenvalue is larger in magnitude than the
largest positive one.

## The parity frame

`validation/data/explain_factor_frame.csv` has five levels of `h`, 20 rows
each, the same 20 values of `x` in every level (so `x` adds no distance),
and a categorical `z` with counts per level (u, v, w, y):

| h | u | v | w | y |
|---|---|---|---|---|
| p | 6 | 3 | 6 | 5 |
| q | 6 | 3 | 1 | 10 |
| r | 0 | 10 | 5 | 5 |
| s | 5 | 10 | 1 | 4 |
| t | 0 | 9 | 1 | 10 |

`B`'s eigenvalues are -0.0675, 0, 0.0287, 0.1303 and 0.1765. ALEPlot 1.1,
R's `ale()` and Python's `ale()` all give the order p, q, s, t, r and the
same effects to 1e-10.

## Can the negative eigenvalue dominate?

A hill climb and random searches over about 600,000 matrices found no
dominant negative eigenvalue: the largest ratio of the most negative to
the largest positive eigenvalue was 0.47 for half-L1 distances of
conditional frequencies (the distance ALEPlot uses for a categorical
column), 0.69 for metrics with integer entries up to 4 on 5 to 7 points,
and 0.42 for shortest-path metrics on cycles with chords. The trace of `B`
is positive, so a dominant negative eigenvalue needs at least two positive
ones each smaller than it; none of these families produced one. The
power iteration's suspected failure is therefore not shown. Step 2 of
#837 replaces the iteration with an eigendecomposition in Rust anyway.
