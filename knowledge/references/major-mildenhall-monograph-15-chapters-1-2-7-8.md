---
type: Reference
title: Major and Mildenhall, CAS Monograph 15, chapters 1, 2, 7 and 8 reproduced in prospicio
description: Iman-Conover (Examples 1 and 2, block Iman-Conover), capital by layer (Tables 8.8 to 8.11) and Bodoff's allocation (Table 8.12) of CAS Monograph 15 recomputed by the guides capital-modeling-monograph-15-iman-conover.md and capital-modeling-monograph-15-practice.md; which printed values are exact and match, which are one random sample, the sorted first column of prospicio's Iman-Conover output, and three conventions for the bottom layer's capital.
resource: https://github.com/casact/capital-modeling
tags: [pricing, capital-allocation, natural-allocation, iman-conover, copulas, dependence, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:30:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:30:00-07:00 }
sources:
  - id: monograph
    resource: https://github.com/casact/capital-modeling
    title: John Major and Stephen Mildenhall (2026), Introduction to Capital Modeling and Portfolio Management, CAS Monograph 15 (Quarto source)
  - id: chapter8
    resource: ../docs/guides/capital-modeling-monograph-15-iman-conover.md
    title: The guide for chapter 8
  - id: practice
    resource: ../docs/guides/capital-modeling-monograph-15-practice.md
    title: The guide for chapters 1, 2 and 7
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guides and checks every printed number in CI
---

# Exact values that match

* Example 2's regional totals (Table 8.4), by FFT with independent lines:
  means and CVs to the digits printed; 99th percentiles 16,236, 15,920,
  25,737 and 17,330 against 16,237, 15,920, 25,737 and 17,331. The last digit
  moves with the discretization step (a step of 0.5 gives 16,236.5, 15,920.5,
  25,737 and 17,329.5), so the gap is the grid, not the model.[^chapter8]
* Section 8.3 on InsCo with the calibrated Wang (lambda 0.343): every row of
  Table 8.10 (layer margins and capital per unit width, `iota`), and the totals
  0.709, 0.337, 5.919 (margin), 8.411, 8.691, 29.333 (capital) and returns
  8.4%, 3.9%, 20.2%. Tables 8.8 and 8.9 print `alpha S` and `beta g(S)`, not
  `alpha` and `beta`.[^chapter8]
* Bodoff (Table 8.12): `Portfolio.bodoff(assets=100)` gives 24.159, 29.756 and
  46.086, and the first layer 7.152, 9.694 and 5.154.[^chapter8]
* InsCo's Table 2.1 statistics (population standard deviation) and the pro
  formas of Tables 2.2 to 2.4: net income 11.6 and ending equity 39.4 for
  outcome 4.[^practice]

# Values from one random sample

* Examples 1 and 2 print statistics of one sample of 1,000 (Tables 8.1, 8.3,
  8.5 to 8.7). prospicio's own samples agree within sampling error: over 100
  samples and shuffles, every Table 8.3 value lies within the 5% to 95%
  range (for example GL-Property Pearson 0.752 to 0.791, book 0.772).
* The rank correlation of an Iman-Conover output with normal scores sits at
  `(6/pi) asin(r/2)` of the target `r` (-0.288 and 0.786 for -0.3 and 0.8),
  not at `r`; the linear correlation falls short of `r` for skewed marginals.
* Example 2's South region is not exactly Example 1's book: GL's severity CV
  is 4 in Example 2 and 5 in Example 1, although the text says the South
  portfolio was used in section 8.2.2.

# prospicio specifics

* `iman_conover` returns its first component sorted ascending (column 0 of the
  reference is not shuffled). Whole-sample statistics are unaffected; shuffle
  rows before subsetting. Normal scores only (no t `dof` as in `aggregate`);
  both filed as #378.
* Block Iman-Conover is `iman_conover` per group, `PredictiveDistribution.join`
  and `reorder_groups`.
* In a busy session the repeated `simulate` calls of the chapter 8 guide ran
  ten times slower than when the machine was idle; the guide's loops are 100
  repetitions to stay under the harness's 20 s.

# The bottom layer's capital (CCoC and PH)

* Equation 8.5 gives the layer where `S = 1` no capital. prospicio (and
  `aggregate` 1.0.1) convert that layer's margins (which sum to zero) at the
  limit `g'(1) / (1 - g'(1))`, finite for the CCoC and PH and infinite for
  Wang, dual power and TVaR. On InsCo the CCoC capital by unit is 5.419,
  6.082, 34.934 by equation 8.5, 2.261, 1.478, 42.696 in prospicio, and 4.874,
  5.288, 36.273 in the monograph's Table 8.11 (an earlier `aggregate`); PH
  7.765, 7.694, 30.975, 6.481, 5.597, 34.356 and 7.074, 6.565, 32.796. All
  add to 46.435. Filed as #379.[^chapter8] `aggregate` 1.2.0 gives the same
  values as 1.0.1, so the limit stays the default, and `bottom_layer="none"`
  (`BottomLayer::Zero`) gives equation 8.5's split, checked in
  `validation/tests/pricing.rs` for the CCoC and PH; Tables 8.11 and 9.8 are
  not reproduced.

[^chapter8]: capital-modeling-monograph-15-iman-conover.md, sections 2, 3, 8, 9 and 10
[^practice]: capital-modeling-monograph-15-practice.md, sections 2 and 5
