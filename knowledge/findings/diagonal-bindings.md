---
type: Finding
title: A frozen chain ladder round-trips through ldf, ldf_kind and tail, and latest diagonals over segments reproduce each segment alone
description: ChainLadderFit.to_chain_ladder gives an age-to-age dict (R a vector named by age) and a float tail that any method takes as ldf, ldf_kind and tail with the same projection to the bit, because Tail::from(f) is the default TailConstant; LatestDiagonals fitted segment by segment with a source fit's segments equals each segment fitted alone; Python and R reproduce the Rust validation and the chainladder-python parity rows.
tags: [reserving, latest-diagonal, bindings, python, r, parity, segments]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-24T12:00:00Z }
verified: { by: process:ci, at: 2026-10-24T12:00:00Z }
sources:
  - id: python-tests
    resource: ../python/tests/test_reserving_latest_diagonal.py
    title: The Python tests of the latest-diagonal bindings
  - id: r-tests
    resource: ../R/prospicio/tests/test-reserving-latest-diagonal.R
    title: The R tests of the latest-diagonal bindings, the same numbers
  - id: rust-tests
    resource: ../validation/tests/reserving_diagonal_segments.rs
    title: The Rust tests of latest diagonals over segments
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: The design note, decision 19 (latest diagonals over segments, and the bindings)
---

# What was found

* A fit frozen by `to_chain_ladder()` reaches a method as plain arguments:
  Python's `ChainLadder` has `ldf` (a dict from each link's starting age to
  its factor), `ldf_kind == "ldf"` and `tail` (a float), and R's
  `to_chain_ladder(fit)` is `list(ldf, ldf_kind = "ldf", tail)`. Unpacked
  into `ChainLadder`, `BornhuetterFerguson`, `Benktander` or `CapeCod` (R:
  `do.call()`), they give the same ultimates as passing the frozen fit, to
  the bit, because the binding builds `Tail::from(f)`, the default
  `TailConstant`, which is the frozen fit's own tail. The tests compare the
  two routes on every row of `reserving_fit_apply_python.csv`.
* `development=` (Python) and `development =` (R) on a latest diagonal give
  each segment the source segment of its label, a mapped one, or a source
  of one segment for every segment; the factors equal the source's to the
  bit and the ultimates within 1e-14, as for each segment fitted alone.
* Cape Cod over segments trends every segment to one valuation, the latest
  of all: a segment valued a year earlier than the others has its trended
  apriori `(1 + trend)` times its own-valuation value, and the same
  detrended apriori and ultimates.
* Both languages reproduce the Rust values: the hand diagonals, half-year
  origins on an annual pattern, the rule past a pattern's end (102.53798721500054,
  1.0132158418990078, the rescale 1.0526315789473684), an accident-month
  pattern on accident years (ultimates 48 and 48, the factor 37/13), RAA's
  off-grid reading (1.5484201731845508), and every row of
  `reserving_latest_diagonal_python.csv` and `reserving_fit_apply_python.csv`
  in both arms.
* An accident-month pattern to 24 months (`t / 24` developed at month
  `t`) averaged onto accident years by `ldf_origin = "M"` runs to 35
  months, the last month of the year at 24: on a triangle with ages 12 to
  48 it selects only the link 12 to 24 (37/13), and the later factors stay
  the triangle's; on a diagonal an origin at 48 months is past its end.
  Every method that takes `ldf`, both bootstraps and their one-year views
  then equal the same call with `{12: 48/13, 24: 48/37}` as factors to
  ultimate, within 1e-12 (seeded draws within 1e-10).
* No new external fact: the chainladder-python references are those of
  decisions 16 and 17; segments are checked against each segment fitted
  alone, not against chainladder-python's groupby.
