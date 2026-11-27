---
type: Reference
title: NCCI's ABCs of Experience Rating
description: NCCI's public explainer of the workers compensation experience rating plan, the reference for prospicio_pricing::emod's split plan; its worksheet's summary reproduces to the dollar, while its class rows and claim lines do not add up to the summary and its rows round expected losses before the D-ratio.
resource: https://www.ceiwc.com/PDFs/ABCs-of-Experience-Rating-NCCI.pdf
tags: [pricing, experience-rating, workers-compensation, emod, parity]
status: draft
generated: { by: claude-code/cloud-session, at: 2026-11-27T10:00:00Z }
sources:
  - id: abcs
    resource: https://www.ceiwc.com/PDFs/ABCs-of-Experience-Rating-NCCI.pdf
    title: NCCI, ABCs of Experience Rating (12 pages; worksheet effective 01/01/2021), as posted by CEIWC
  - id: tests
    resource: ../validation/tests/pricing_emod_split.rs
    title: The split plan against the ABCs' Exhibits A, B, G and H
---

# What it is

NCCI's short public guide to the Experience Rating Plan, with one worked
modification (the "Any Insured" worksheet, Exhibits G and H) and small
examples of the claim split (Exhibit A) and the medical-only reduction
(Exhibit B). The plan manual and its tables are licensed; this guide is
free.[^abcs]

# What prospicio reproduces

* **Exhibit H's summary**, from its totals: `E = 176,190`,
  `E_p = 56,172`, `A_p = 68,584`, `A_e = 31,985`, `W = 0.14`,
  `B = 47,400` give the stabilizing value 150,615, the adjusted actual
  223,677 and the adjusted expected 223,590 to the dollar, and the
  modification 1.00 (1.000390 unrounded).[^tests]
* **Exhibit A**: a split point of 18,000 and a per-claim limit of 250,000
  split 500,000, 100,000 and 5,000 into (18,000, 232,000),
  (18,000, 82,000) and (5,000, 0) exactly.
* **Exhibit B**: a medical-only claim of 20,000 enters as 30% of its
  primary 18,000 and its excess 2,000: 5,400 and 600.
* **Exhibit G's class rows**: expected losses as the payroll in hundreds
  times the expected loss rate, to the dollar.

# Quirks

* Exhibit G rounds each row's expected losses to the dollar *before* it
  applies the D-ratio: class 8748's 5,101.515 prints 5,102, and its
  expected primary 1,684 is 5,102 x 0.33 = 1,683.66, where the unrounded
  5,101.515 x 0.33 is 1,683.4995. prospicio keeps every amount unrounded.
* Exhibit G's four class rows are one policy of the three-year period:
  they add up to expected losses 46,195 and expected primary 14,786, not
  the summary's 176,190 and 56,172. Only the summary checks the formula.
* The claim detail's incurred 101,344 and primary 69,359 are 775 above the
  summary's 100,569 and 68,584. The guide says the medical-only reduction
  shows in the summary only, but 70% off its type-6 lines would be far
  more than 775, so the gap is not explained by the page.
* The guide prints no algebraic formula; its worksheet is (actual primary
  + W x actual excess + stabilizing value) over (expected primary + W x
  expected excess + stabilizing value), with the stabilizing value
  `E_e (1 - W) + B`, which is the textbook
  `(A_p + W A_e + (1 - W) E_e + B) / (E + B)`.
* No multiple-claim accident rule is given beyond the state's per-claim
  limitation; prospicio's per-accident limit is its own reading
  (`primary-rating.md` decision (k)).

[^abcs]: NCCI, ABCs of Experience Rating (12 pages; worksheet effective 01/01/2021), as posted by CEIWC
[^tests]: The split plan against the ABCs' Exhibits A, B, G and H
