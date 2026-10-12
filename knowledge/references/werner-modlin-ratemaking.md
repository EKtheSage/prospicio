---
type: Reference
title: Werner and Modlin's Basic Ratemaking
description: The CAS study note behind prospicio_pricing::ratemaking; its exhibits round at every step to the printed precision, its dates sit on a half-month grid, and a handful of printed figures need the book's rounded inputs to reproduce.
resource: https://www.casact.org/sites/default/files/2021-03/5_Werner_Modlin.pdf
tags: [pricing, ratemaking, trend, on-level, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-12T08:00:00Z }
sources:
  - id: wm
    resource: https://www.casact.org/sites/default/files/2021-03/5_Werner_Modlin.pdf
    title: Werner and Modlin, Basic Ratemaking, 5th ed., May 2016 (CAS); chapters 5 to 8, 12 and Appendices A to D
  - id: tests
    resource: ../validation/tests/pricing.rs
    title: The ratemaking parity tests (module ratemaking)
---

# What it is

The CAS study note on the overall rate indication, free on casact.org. Its
chapters 5 (premium), 6 (losses and LAE), 7 (expenses and profit) and 8
(the indication), with four worked indications in Appendices A (auto
property damage, loss ratio method), B (homeowners, pure premium method),
C (medical malpractice) and D (workers compensation), are the reference
for `prospicio_pricing::ratemaking`.[^wm]

# Rounding

The exhibits behave like spreadsheets that round each cell to its printed
precision, half up, and chain on the rounded values.[^wm]

* Cumulative rate level indices are rounded to 4 dp at each step: Appendix
  A's levels end at 1.1857 where the exact chain is 1.185524, and chapter
  5's 1.14345 prints as 1.1435. The average level of a year uses the
  rounded indices and is rounded to 4 dp; the on-level factor divides the
  rounded figures. Appendix A's 2012 factor is 1.2176 that way and 1.2175
  exactly; Table 5.19's policy-year average is 1.1464 with 1.1435 and
  1.1463 with 1.14345.
* 0.97375 does not round half up reliably in binary floating point
  (Appendix A 2012's average). The tests divide by the printed average
  once it is checked, rather than round it.
* Table 5.29 divides the latest average written premium by the average
  earned premium rounded to 740.00 (1.017568; 1.017562 unrounded), and
  multiplies the step factors rounded to 4 dp (1.0176 × 0.9838 = 1.0011,
  where the exact product is 1.001084).
* Trend fits are on the printed values: Table 6.14's frequencies at 4 dp
  reproduce every printed frequency trend; the unrounded ones move the
  8-point fit from −1.24% to −1.30% (printed −1.2%).
* Appendix C rounds trend lengths to 2 dp (6.83 for 6.8333; 1.089^6.83 =
  1.7902, 1.089^6.8333 = 1.7907).
* Further steps for the indication (next PR): Appendix B rounds the
  credibility to 79.5% before weighting and its non-modelled catastrophe
  load to 3 dp at each step ($29.11 printed, $29.28 exact); Appendix B's
  age-to-ultimate factors are chained with rounding and Appendix D's are
  exact products; expense ratios and trend selections are rounded to 0.1%
  before use.

# Dates

Dates are on a half-month or quarter grid, never actual/365: 6/30 and 7/1
are the same point, as are 12/31 and 1/1, and 11/15 is 10.5 months into the
year. Calendar-year premium is written on average half a term before the
year's midpoint (1/1 for annual policies, 4/1 for six-month), policy-year
losses occur on average half a term after it; the forecast's average
written date is the effective date plus half the rate-effective period,
and its average accident date adds half the term. Appendix C (annual
policies, effective 5/1/2016) trends to 5/1/2017; Appendix A (six-month,
1/1/2017) to 9/30/2017.[^wm]

# Book errors met

* The six-month calendar-year average level in chapter 5 is 1.12875
  exactly (shares 0.25 and 0.75), printed 1.1288; a transcription that gave
  1.128763 was wrong, not the book.
* Appendix A Premium Trend-3, footnote 7, reads "06/30/2017 to 06/30/2017";
  the period is 6/30/2015 to 6/30/2017 (the printed 2.0 and the text).
* Table 7.5 uses 2015 earned premium of 545,250,000 where Table 7.4 has
  530,000,000; the printed 1.4% follows the former.
* Appendix C's 2008 earned premium equals 2014's (18,265,093), probably a
  copy in the book; the tests take it as printed.

[^wm]: Werner and Modlin, *Basic Ratemaking*, 5th ed. (CAS, 2016).
