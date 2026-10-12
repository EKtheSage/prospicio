---
type: Dataset
title: CAS loss reserve database, six lines summed over companies
description: Paid-loss triangles of the six CLRD lines (comauto, medmal, othliab, ppauto, prodliab, wkcomp), 1988-1997 at 12 to 120 months, summed over companies by chainladder-python 0.10.1; the multi-line data for the dependence between segments.
resource: ../validation/data/clrd_lines.csv
tags: [reserving, triangle, clrd, multi-line, dependence]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T21:00:00-07:00 }
sources:
  - id: cl
    resource: https://chainladder-python.readthedocs.io/
    title: chainladder-python 0.10.1, load_sample('clrd')
  - id: test
    resource: ../validation/tests/reserving_dependence.rs
    title: Validation test of SegmentDependence on these lines
---

# Facts

* `chainladder.load_sample('clrd').groupby('LOB').sum()`, column
  `CumPaidLoss`, exported as `lob,origin,development,paid` (the observed
  upper triangles, 55 cells per line, 330 rows) with `uv run --no-project
  --with chainladder==0.10.1`.[^cl]
* Every line has the same origins (1988-1997) and ages (12-120), so they
  can be synchronized; the wkcomp rows are `clrd_wkcomp.csv`'s `paid`.
* Paired ODP residual correlations range from -0.15 (othliab, ppauto) to
  0.60 (prodliab, wkcomp); the strongest pairs are prodliab with wkcomp
  (0.60), comauto with ppauto (0.57), comauto with wkcomp (0.55) and
  ppauto with wkcomp (0.48), and every pair with medmal or othliab is
  within 0.26 of zero.[^test]
