---
type: Dataset
title: Dahms' paid and incurred triangles
description: Dahms' (2008) 10x10 paid and incurred triangles, the example of Merz and Wuthrich's paid-incurred chain papers, as printed in their tail-factor paper; a second R parity set.
resource: ../validation/data/dahms.csv
tags: [reserving, triangle, paid-incurred]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-03T06:00:00+00:00 }
verified: { by: process:ci, at: 2026-11-03T06:00:00+00:00 }
sources:
  - id: tail
    resource: https://www.variancejournal.org/
    title: Merz and Wuthrich, Estimation of tail development factors in the paid-incurred chain reserving method, Variance 7(1), Tables 1 and 2
  - id: script
    resource: ../validation/scripts/reserving_paid_incurred_r.R
    title: Writes validation/data/dahms.csv
  - id: tests
    resource: ../validation/tests/reserving_paid_incurred.rs
    title: matches_r_chainladder and hertigs_model_matches_merz_and_wuthrichs_printed_values
  - id: table5
    resource: https://casact.org/sites/default/files/2021-07/Estimation-Tail-Development-Merz-Wuthrich.pdf
    title: Merz and Wuthrich, tail-factor paper, Tables 4 and 5 (Hertig's model, no tail factor)
---

# Facts

* Ten origins and ten ages, cumulative paid and incurred, typed from the
  tail-factor paper's Tables 1 and 2; origins 0 to 9 are relabelled 2001
  to 2010 and ages 0 to 9 are 12 to 120 months.[^tail][^script]
* Under R's convention the paid-incurred chain's total reserve is
  11,074,832.95 with a standard error of 392,409.36.[^tests]
* Hertig's paid-only model (Table 4's sigma, the last by Mack's rule)
  gives a total reserve of 9,772,084 with a msep^(1/2) of 1,519,464, as
  the tail-factor paper's Table 5 prints; prospicio's PIC with a very
  large tau reproduces every origin's value to the unit.[^table5][^tests]

[^tail]: Tail-factor paper
[^script]: Generator
[^tests]: Validation tests
[^table5]: Tail-factor paper, Table 5
