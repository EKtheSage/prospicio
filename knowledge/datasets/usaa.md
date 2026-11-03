---
type: Dataset
title: USAA paid and incurred triangles
description: R ChainLadder's USAApaid and USAAincurred, a 10x10 pair of cumulative paid and incurred triangles, the R parity data for the paid-incurred chain.
resource: ../validation/data/usaa.csv
tags: [reserving, triangle, paid-incurred]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-03T06:00:00+00:00 }
verified: { by: process:ci, at: 2026-11-03T06:00:00+00:00 }
sources:
  - id: r
    resource: R package ChainLadder 0.2.21, datasets USAApaid and USAAincurred
    title: R ChainLadder's USAA triangles
  - id: script
    resource: ../validation/scripts/reserving_paid_incurred_r.R
    title: Writes validation/data/usaa.csv
  - id: tests
    resource: ../validation/tests/reserving_paid_incurred.rs
    title: matches_r_chainladder, on_usaa_the_ultimates_lie_between_the_paid_and_incurred_chain_ladders
---

# Facts

* Ten origins and ten ages, paid and incurred, both squares' upper
  triangles fully observed; R's origins 0 to 9 are relabelled 2001 to
  2010 and its ages 1 to 10 are 12 to 120 months.[^script]
* Under R's convention the paid-incurred chain's total reserve is
  1,596,953.58 with a standard error of 110,976.85; each origin's
  ultimate lies between its paid and its incurred chain-ladder
  ultimate.[^tests]

[^r]: R ChainLadder
[^script]: Generator
[^tests]: Validation tests
