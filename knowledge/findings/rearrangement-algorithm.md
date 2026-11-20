---
type: Finding
title: The Rearrangement Algorithm's worst VaR depends on its start
description: "The Rearrangement Algorithm estimates the worst VaR at p of a sum with known marginals as the smallest row sum after repeatedly ordering each column of n top quantiles opposite to the sum of the others. It is a heuristic: on Pricing Insurance Risk's Exercise 59 (three lognormals with mean 10 and CVs 1, 2, 3, n = 40, p = 0.99) the rotated starts of the guide give 353.8, 353.6 and 352.9 against the book's 352.8 from an unstated random start, all about 45% above the comonotonic 242.5."
tags: [prospicio-prob, risk-measures, rearrangement-algorithm, var, pricing-insurance-risk]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T09:30:00Z }
verified: { by: process:ci, at: 2026-10-19T09:30:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/rearrangement.rs
    title: rearrange and worst_var
  - id: test
    resource: ../validation/tests/distributions.rs
    title: worst_var_matches_the_pricing_insurance_risk_guide
  - id: guide
    resource: ../docs/guides/pir-risk-measures.md
    title: The worst VaR of a sum, section 4
---

# Facts

* The lower matrix of Embrechts, Puccetti and Rüschendorf (2013) holds, for
  each marginal, its quantiles at `p + (1 - p) i / n` for `i = 0, ..., n - 1`
  (the infinite top quantile left out); its rearranged minimum row sum
  converges to the worst VaR from below as `n` grows.
* A sweep orders each column opposite to the sum of the others (the
  others' sum ascending, the column descending), in place, column by
  column; the algorithm stops when a sweep raises the minimum by less than
  a tolerance.
* The result depends on the start. On Exercise 59 the guide's starts,
  column `j` rotated down `7 j`, `13 j` and `19 j` places (numpy's
  `roll`), give 353.8, 353.6 and 352.9; the book prints 352.8. prospicio's
  `Start::Rotated` reproduces the guide's numbers, so Python and R agree.
* The worst VaR is about 45% above the comonotonic sum of the VaRs, 242.5:
  comonotonicity is not the worst case for VaR, which is not subadditive.
* Speed: a sweep over `d` columns costs `O(d^2 n)` additions when each
  column recomputes the row totals, plus a sort of `n` per column. Summing
  the columns before the current one once per sweep and only the rest per
  column, and sorting each column once, gives the same bits as the plain
  version (same additions in the same order, same stable tie order) and ran
  a seeded `d = 10`, `n = 10 000` problem (5 sweeps, release build) in 18 ms
  against 44 ms; the sorts of the others' sums then dominate. Totals updated
  incrementally (`total -= old; total += new`) would be faster but change the
  rounding and so, through ties, possibly the arrangement; not done.
