---
type: Dataset
title: MCL paid and incurred triangles
description: Quarg and Mack's (2004) 7x7 paid and incurred triangles, R ChainLadder's MCLpaid and MCLincurred and chainladder-python's mcl sample; the Munich chain ladder's reference example.
resource: ../validation/data/mcl.csv
tags: [reserving, triangle, munich, paid-incurred]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-02T18:00:00-08:00 }
verified: { by: process:ci, at: 2026-11-02T18:00:00-08:00 }
sources:
  - id: r
    resource: R package ChainLadder 0.2.21, datasets MCLpaid and MCLincurred
    title: Quarg and Mack (2004), Munich Chain Ladder, Blaetter der DGVFM 26, as R ChainLadder::MCLpaid and MCLincurred
  - id: python
    resource: chainladder-python 0.10.1, cl.load_sample("mcl")
    title: The same triangles in chainladder-python
  - id: test
    resource: ../validation/tests/reserving_munich.rs
    title: Validation test of the Munich chain ladder against R and chainladder-python
---

# Facts

* Origins 2001 to 2007, development 1 to 7 years, cumulative paid and
  incurred observed at the same cells, every cell positive.[^r]
* `validation/data/mcl.csv` holds both in long form (origin, development,
  paid, incurred), development in months (12 to 84).[^test]
* R's `MunichChainLadder(MCLpaid, MCLincurred, est.sigmaP = "Mack",
  est.sigmaI = "Mack")` gives lambdas 0.636021 (paid) and 0.436187
  (incurred) over 20 residual pairs, total ultimates 32,121.50 (paid) and
  32,719.51 (incurred).[^test]

# Consequences

* R's default `est.sigma = "log-linear"` falls back to Mack's rule on these
  triangles with a warning, so R's default numbers are its Mack arm's. See
  [the parity finding](../findings/munich-chain-ladder-parity.md).
