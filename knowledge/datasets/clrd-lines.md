---
type: Dataset
title: CAS loss reserve database, six lines summed over companies
description: Paid-loss triangles of the six CLRD lines (comauto, medmal, othliab, ppauto, prodliab, wkcomp), 1988-1997 at 12 to 120 months, summed over companies by chainladder-python 0.10.1, with each origin's net earned premium; the multi-line data for the dependence between segments and for Meyers' models over segments.
resource: ../validation/data/clrd_lines.csv
tags: [reserving, triangle, clrd, multi-line, dependence, premium]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T21:00:00-07:00 }
sources:
  - id: cl
    resource: https://chainladder-python.readthedocs.io/
    title: chainladder-python 0.10.1, load_sample('clrd')
  - id: script
    resource: ../validation/scripts/clrd_lines.py
    title: The generator of clrd_lines.csv, with its checks
  - id: test
    resource: ../validation/tests/reserving_dependence.rs
    title: Validation test of SegmentDependence on these lines
---

# Facts

* `chainladder.load_sample('clrd').groupby('LOB').sum()`, columns
  `CumPaidLoss` and `EarnedPremNet`, exported as
  `lob,origin,development,paid,premium` (the observed upper triangles, 55
  cells per line, 330 rows) with `uv run --no-project --with
  chainladder==0.10.1 python validation/scripts/clrd_lines.py`.[^cl]
* `premium` was added on 2026-10-13, for `Meyers::fit_segments` and the
  Meyers backtest. `EarnedPremNet` is constant across ages for an origin in
  these sums, so every row of an origin holds the same value. The
  generator checks that the `paid` column it writes equals the one already
  committed, cell for cell, and that wkcomp's `paid` and `premium` are
  `clrd_wkcomp.csv`'s.[^script]
* Every line has the same origins (1988-1997) and ages (12-120), so they
  can be synchronized; the wkcomp rows are `clrd_wkcomp.csv`'s `paid`.
* Paired ODP residual correlations range from -0.15 (othliab, ppauto) to
  0.60 (prodliab, wkcomp); the strongest pairs are prodliab with wkcomp
  (0.60), comauto with ppauto (0.57), comauto with wkcomp (0.55) and
  ppauto with wkcomp (0.48), and every pair with medmal or othliab is
  within 0.26 of zero.[^test]
