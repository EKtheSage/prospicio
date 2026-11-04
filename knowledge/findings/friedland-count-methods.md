---
type: Finding
title: Friedland's chapters 11 to 13 reproduce within the book's rounding, with selections rounded in some exhibits and not in others and one payment left out of chapter 12's XYZ total
description: Frequency-severity, disposal rates, case outstanding development and the Berquist-Sherman adjustments reproduce Friedland's exhibits; US Auto in chapter 12 and XYZ's disposal rates project with selections rounded to three decimals, XYZ in chapter 12 with unrounded ones, the XYZ data rounded to the $000 moves the pooled tail severity by 8 dollars and the disposal total by 6 thousand in 386 million, Exhibit II of chapter 12 omits 233 of 2000's 120-month payment, and chainladder-python's berqsherm sample has the same Incurred for Auto as for MedMal.
tags: [reserving, friedland, parity, frequency-severity, berquist-sherman, disposal]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-02T18:00:00Z }
verified: { by: process:ci, at: 2026-11-02T18:00:00Z }
sources:
  - id: friedland
    resource: https://www.casact.org/sites/default/files/old/studynotes_friedland_estimating.pdf
    title: Friedland, Estimating Unpaid Claims Using Basic Techniques, version 3, July 2010 (CAS); chapters 11 to 13
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decisions 33 and 34, the methods on claim counts and the Berquist-Sherman adjustments
  - id: test
    resource: ../validation/tests/reserving_counts.rs
    title: Validation test of the exhibits of chapters 11 to 13 and of chainladder-python's BerquistSherman, run in every cargo test
---

# What was found

`FrequencySeverity`, `Disposal`, `CaseOutstanding`, `CaseAdequacy`,
`SettlementRate` and `BerquistSherman` reproduce the printed exhibits of
chapters 11 to 13 of Friedland's text.[^friedland][^test] The design decisions
are in the note.[^note] What the exhibits do not say:

- **Rounding differs by exhibit.** US Auto in chapter 12 and XYZ's disposal
  rates in chapter 11 project with selections rounded to three decimals. XYZ
  in chapter 12 projects with unrounded ones. The test passes the printed
  values as given selections where the book rounds; the library does not
  round.
- **The XYZ data is rounded to the $000.** The paid claims behind chapter 11's
  disposal exhibits are printed in dollars elsewhere, so the pooled tail
  severity comes to 70,424 where the book has 70,432. The ultimates agree
  within 2 and the total within 6 ($000) of 385,676.
- **Chapter 12, Exhibit II leaves out a payment.** Accident year 2000's
  120-month payment, 0.501 x 465 = 233, is missing from the book's ultimate,
  so the test adds it to the printed 513,944.
- **Approach 2 rounds the frequency.** The book's selected 2007 frequency is
  1.92% rounded, which moves that year's ultimate by up to 0.25%.
- **`max_age` matters in the combined adjustment.** XYZ adjusts average case
  only to 84 months; older ages keep their unadjusted averages in the combined
  adjustment too, which is what gives the printed 489,258.
- **The chapter's average paid triangle is not paid over closed.** Exhibit I,
  sheet 5's average paid (402 at 12 months for 1969, trends of 7% to 14%) comes
  from the paper; the paid and closed triangles beside it give 3,788 and, over
  every age at once, 24%, about the 25% of the average case. A trend fitted on
  paid over closed does not reproduce the comparison the chapter makes.
- **GL Insurer's rates are rounded too.** Exhibit V projects with the medial
  rates rounded to three decimals; on them every ultimate agrees within 30
  dollars in 40 million, the rest being the counts' hidden decimals.

chainladder-python 0.10.1's `BerquistSherman` agrees with the two adjustments
separately on its `berqsherm` sample. That sample's Auto line has the same
Incurred as its MedMal line, so it is a test of arithmetic and not an example
to read as data.

[^friedland]: Friedland (2010), chapters 11 to 13 and their exhibits.
[^note]: `docs/design/reserving-v02.md`, decisions 33 and 34.
[^test]: `validation/tests/reserving_counts.rs`.
