---
type: Reference
title: Werner and Modlin's Basic Ratemaking, reproduced in the ratemaking guides
description: What the two Werner and Modlin guides found beyond the parity suite; chapter 6's development example and Appendix A's development exhibit reproduce with prospicio.reserving once link ratios and factors are rounded as printed, Table 12.1's 99% standards use z = 2.575, the Bühlmann example runs through credibility_premiums, and chapter 11's size-of-loss figures are arithmetic prospicio has no function for.
resource: https://www.casact.org/sites/default/files/2021-03/5_Werner_Modlin.pdf
tags: [pricing, ratemaking, credibility, development, ilf, deductibles, guides, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T15:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T15:00:00-07:00 }
sources:
  - id: wm
    resource: https://www.casact.org/sites/default/files/2021-03/5_Werner_Modlin.pdf
    title: Werner and Modlin, Basic Ratemaking, 5th ed., May 2016 (CAS); chapters 5 to 7, 11 and 12, Appendices A to C
  - id: guide1
    resource: ../docs/guides/ratemaking-werner-modlin.md
    title: The guide to the steps (dates, parallelogram, trend, development, expenses, credibility)
  - id: guide2
    resource: ../docs/guides/ratemaking-werner-modlin-indications.md
    title: The guide to Appendices A to C end to end, chapter 11 and the gaps
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs both guides and checks every printed number in CI
---

# What it is

Two guides run Werner and Modlin's overall indication in prospicio's Python
API with the book's printed numbers beside prospicio's.[^guide1][^guide2] The
book's general rounding habits (cells rounded to the printed precision, half
up, and carried on) and its half-month dates are in
[werner-modlin-ratemaking.md](werner-modlin-ratemaking.md); this note keeps
what the guides found on top of the parity suite.

# Findings

* **Table 6.12 averages rounded link ratios.** The 27-39 arithmetic average
  prints 1.30: the four link ratios rounded to two places average 1.295.
  The exact link ratios average 1.2932, which rounds to 1.29. Table 6.13's
  factors to ultimate are rounded to two places (2.19 for 2.1879) before
  multiplying; that is what gives the printed total of 14,095 (14,093.7
  unrounded). The "ratio of total losses" row is the chain ladder's
  volume-weighted average (`ChainLadder()`).[^wm]
* **Appendix A's Loss Development exhibit** rounds each link ratio to four
  places; its all-year 15-27 average is 1.0704 that way and 1.0703 from the
  exact ratios. The selection is the straight average excluding the highest
  and lowest ratio, which `ChainLadder` does not offer; passed as a selected
  pattern (`ldf=..., ldf_kind="ldf"`) it reproduces the factors to ultimate
  1.0966, 1.0282, 1.0003, 0.9799, 1.0000.[^guide1]
* **The layout of the PDF scrambles two triangles** when its text is
  extracted: Table 6.11's AY 2004 to 2006 cells and Appendix A's triangle
  must be rebuilt from the printed link ratios (Table 6.11: 2004 at 27 months
  is 1,560; 2005 is 1,093, 1,651, 2,125; 2006 is 1,126, 1,662).
* **Table 12.1's 99% rows use z = 2.575.** The exact quantile 2.5758 gives
  2,654 claims at k = 5% where the book prints 2,652; the other five rows
  round to the printed standards with the exact quantile.[^wm]
* **The Bühlmann example of chapter 12** (21 observations, EVPV 2.00, VHM
  0.50, prior mean 225, observed 200: Z 0.84, 204) is reproduced by
  `credibility_premiums` with a tight normal prior on the collective mean;
  the function needs two groups, so a second arbitrary group is added. The
  book leaves the estimation of K out of scope; `buhlmann_straub` estimates
  it.[^guide1]
* **Chapter 11's size-of-loss tables** give counts and loss totals per band,
  and prospicio's `ilf` and `loss_elimination_ratio` need a severity, which a
  `Sampled` or `Discrete` is not. The figures (ILF 1.59, censored LAS 77,046,
  121,621 and 136,834, LER 0.135) are arithmetic; the censored LAS adds
  layers rounded to the dollar (121,620.5 unrounded).[^guide2]
* **Appendix C's -5% profit provision** is an underwriting provision offset
  by investment income in a long-tailed occurrence line; Appendix C's data is
  on occurrence policies.[^wm]

[^wm]: Werner and Modlin, *Basic Ratemaking*, 5th ed. (CAS, 2016).
[^guide1]: `docs/guides/ratemaking-werner-modlin.md`.
[^guide2]: `docs/guides/ratemaking-werner-modlin-indications.md`.
