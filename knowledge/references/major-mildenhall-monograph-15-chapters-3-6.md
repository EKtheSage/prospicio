---
type: Reference
title: Major and Mildenhall, CAS Monograph 15, chapters 3 to 6 reproduced in prospicio
description: The InsCo tables of CAS Monograph 15's theory and applications chapters (capital adequacy, pricing and allocation, applications, choosing a distortion) recomputed by the guides capital-modeling-monograph-15-theory.md and capital-modeling-monograph-15-applications.md; which printed values match, the rounding and calibration-tolerance digits that do not, and two readings of VaR at a jump.
resource: https://github.com/casact/capital-modeling
tags: [pricing, risk-measures, distortions, capital-allocation, natural-allocation, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:00:00-07:00 }
sources:
  - id: monograph
    resource: https://github.com/casact/capital-modeling
    title: John Major and Stephen Mildenhall (2026), Introduction to Capital Modeling and Portfolio Management, CAS Monograph 15 (Quarto source)
  - id: theory
    resource: ../docs/guides/capital-modeling-monograph-15-theory.md
    title: The guide for chapters 3 and 4
  - id: applications
    resource: ../docs/guides/capital-modeling-monograph-15-applications.md
    title: The guide for chapters 5 and 6
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guides and checks every printed number in CI
---

# What matches

* Every InsCo table of chapters 3 to 6 reproduces with `Portfolio`,
  `Pentagon`, `Distortion` and `premium_bounds` on the ten equally likely
  years of Table 2.1: Tables 3.1 to 3.3, 4.2 to 4.11, 5.1 to 5.10 and 6.1 to
  6.3, to the digits printed apart from the cases below.[^theory][^applications]
* The portfolio optimization of section 5.5 needs nothing beyond
  `Portfolio.price` in a loop: Table 5.8 matches to four decimals, Tables 5.9
  and 5.10 to 1e-6 for units A and B and 2e-5 for unit C, and a grid search
  lands next to the book's Nelder-Mead optimum (EVA -1.34514 against
  -1.345119).[^applications]
* Chapters 3 to 6 print no lifted (capital-by-unit) allocation, so the
  CCoC/PH version difference of the chapter 9 guide does not arise.

# Digits that differ, and why

* Calibration tolerance in `aggregate`: Table 4.5's dual parameter 1.5951
  (prospicio 1.595152), Table 4.8's PH ratio 1.9033 at the worst year (1.9034)
  and TVaR ratio 0.3941 at the total of 36 (0.3940). prospicio's TVaR level
  0.27128713 is the exact solution; a level of 0.271283 and a PH rho of
  0.7205, which print the same, reproduce the book's digits and leave the
  premium at 53.565 to three decimals.[^theory]
* The same number rounded two ways in the book: Wang's unit C premium
  20.81858 is 20.818 in Tables 4.9 and 6.2 and 20.819 in Tables 5.1 and 5.2;
  the dual unit C premium is 20.321 in Table 6.2 and 20.322 in Table 9.8.
* Rounded intermediate values: the net-of-reinsurance premium 45.956
  (section 5.2) uses discount factors rounded to 0.86956 and 0.13043
  (unrounded 45.9565); the acquisition's 58.26 subtracts the rounded 53.57
  (unrounded 58.269), and its capital 186.46 is 186.466 truncated; Table
  5.6's predicted EVA -1.2403 sums four-decimal steps (unrounded -1.2404).

# VaR at a jump

* Section 3.4.2's algorithm returns the `floor(n p)`-th order statistic,
  the upper quantile when `n p` is a whole number, while the text defines VaR
  as the least quantile. On InsCo they differ at p = 0.7 (55 against 40) and
  0.9 (100 against 65); `Sampled.quantile` returns the least. The
  subadditivity example of section 3.2 (outcome probabilities 0.9, 0.05,
  0.05) fails subadditivity at 0.9 only under the algorithm's reading; under
  the least quantile it fails at any level just above 0.9.[^theory]

[^theory]: capital-modeling-monograph-15-theory.md, sections 3 and 8
[^applications]: capital-modeling-monograph-15-applications.md, sections 3, 5, 6 and 8
