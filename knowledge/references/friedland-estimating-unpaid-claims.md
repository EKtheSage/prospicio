---
type: Reference
title: Friedland's Estimating Unpaid Claims Using Basic Techniques
description: The CAS syllabus text on basic reserving; its exhibits multiply by factors to ultimate rounded to three decimals, hold claims with decimals behind whole-thousand printing, cap CDFs at 1 for Bornhuetter-Ferguson and Cape Cod on XYZ, and use an unrounded Cape Cod ratio; every printed ultimate of chapters 7 to 10 reproduces in prospicio to the unit or within that rounding.
resource: https://www.casact.org/sites/default/files/old/studynotes_friedland_estimating.pdf
tags: [reserving, chain-ladder, bornhuetter-ferguson, cape-cod, benktander, parity, guides]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T12:00:00Z }
sources:
  - id: friedland
    resource: https://www.casact.org/sites/default/files/old/studynotes_friedland_estimating.pdf
    title: Friedland, Estimating Unpaid Claims Using Basic Techniques, version 3, July 2010 (CAS); chapters 7 to 10 and 15
  - id: guide
    resource: ../docs/guides/reserving-friedland.md
    title: The guide that reproduces it, run by python/tests/test_guides.py
---

# What it is

The CAS text on basic techniques for unpaid claims: the development (chain
ladder), expected claims, Bornhuetter-Ferguson, Benktander, Cape Cod,
frequency-severity, case outstanding development and Berquist-Sherman
techniques, recoveries and claim adjustment expenses. Chapters 7 to 10 run on
two data sets: U.S. Industry Auto (accident years 1998 to 2007, $000) and XYZ
Insurer Auto BI (1998 to 2008, a trapezoid starting at calendar year 2000).
Exhibit numbers restart in every chapter.[^friedland]

# Conventions that matter for parity

- **Averages over the latest n years.** The book's simple, medial (latest 5
  without high and low; two values are averaged, one is itself), volume-weighted
  and geometric averages run over the latest diagonals. prospicio's
  `ChainLadder` averages all years; a volume-weighted latest-n average equals
  the chain ladder on the latest `n + 1` diagonals. All of the book's printed
  averages for U.S. Industry Auto, and XYZ's selected latest-2 volume-weighted
  factors, reproduce to three decimals.[^guide]
- **Rounded CDFs.** The selected age-to-age factors are three-decimal values,
  and the factors to ultimate are rounded again to three decimals before they
  multiply the latest diagonal. Chaining the selected factors gives U.S.
  reported ultimates 36,133 below the book; passing the rounded CDFs with
  `ldf_kind="cdf"` reproduces every ultimate exactly.[^guide]
- **CDFs below 1.** XYZ's reported CDFs are 0.992 at 96 and 108 months.
  `ldf_kind="cdf"` rejects a factor to ultimate below 1, so they go in as
  ratios of consecutive CDFs with `ldf_kind="ldf"`. For Bornhuetter-Ferguson
  and Cape Cod the book caps them at 1.00.[^friedland]
- **Data printed rounded.** The spreadsheets hold claims with decimals: XYZ's
  printed paid diagonal adds to 330,627 while its printed total is 330,629.
  XYZ ultimates therefore differ from prospicio's by 1 in a few years and by
  up to 2 in a total; U.S. Industry figures agree to the unit.[^guide]
- **Cape Cod ratio unrounded.** U.S. Industry's ratio prints as 69.5% but is
  0.69544; at exactly 69.5% the ultimate would be 17,431 lower. XYZ's adjusted
  Cape Cod applies the tort factor to the claims, the 3.425% trend as
  `CapeCod(trend=0.03425)` does (latest origin as reference), and on-level
  premium as exposure; the expected claims then go into a BF fit on the
  actual claims, reproducing 504,300.[^guide]
- **Benktander** is shown only on synthetic changing-conditions scenarios
  (chapter 9, Exhibits V and VI); its formula fed with the printed U.S. BF
  ultimates equals `Benktander(n_iters=2)`.[^guide]

# Not in prospicio

Latest-n, medial and geometric averages as options; frequency-severity
(chapter 11), case outstanding development (12), Berquist-Sherman adjustments
(13), ALAE and ULAE ratio methods (16, 17).

[^friedland]: Friedland (2010), chapters 7 to 10 and their exhibits.
[^guide]: `docs/guides/reserving-friedland.md`, checked by `python/tests/test_guides.py`.
