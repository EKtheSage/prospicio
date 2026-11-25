---
type: Finding
title: Werner and Modlin's exposure exhibits count months; a day count is within two days in the year
description: Chapter 4's earned exposures (Table 4.11, 0.25, 3.25 and 2.50 by calendar year) take a quarter of an annual policy as 0.25, where exact days give 92/365 = 0.2521 or 91/366 = 0.2486, so prospicio_pricing::earned's daily method matches them to 6e-3 and the written, policy year and in-force tables exactly; the book's expiration date is the last day in force and its in-force exposure is a snapshot on a date
tags: [pricing, earned-exposure, ratemaking, werner-modlin, dates]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-25T07:00:00Z }
verified: { by: process:ci, at: 2026-11-25T07:00:00Z }
sources:
  - id: book
    resource: https://www.casact.org/sites/default/files/old/studynotes_werner_modlin_ratemaking.pdf
    title: "Werner and Modlin, Basic Ratemaking, 5th ed. (CAS, 2016), chapter 4, Tables 4.7 to 4.22"
  - id: earned
    resource: ../crates/prospicio-pricing/src/earned.rs
    title: prospicio_pricing::earned (Date, Policy, Windows, earned_exposure, in_force)
  - id: test
    resource: ../validation/tests/pricing_earned_exposure.rs
    title: validation/tests/pricing_earned_exposure.rs (werner_modlin_chapter_4_annual_policies, werner_modlin_chapter_4_six_month_policies)
---

# Finding

Chapter 4 of *Basic Ratemaking* works one example through every kind of
exposure: six policies of one exposure each, effective 10/01/10, 01/01/11,
04/01/11, 07/01/11, 10/01/11 and 01/01/12, first as annual policies and
then as six-month policies of half an exposure [book].

| Table | What | CY or PY 2010 | 2011 | 2012 |
|---|---|---|---|---|
| 4.7 | calendar year written, annual | 1.00 | 4.00 | 1.00 |
| 4.11 | calendar year earned, annual | 0.25 | 3.25 | 2.50 |
| 4.9, 4.13 | policy year written and earned, annual | 1.00 | 4.00 | 1.00 |
| 4.18 | calendar year written, six-month | 0.50 | 2.00 | 0.50 |
| 4.19 | calendar year earned, six-month | 0.25 | 2.00 | 0.75 |
| 4.20, 4.21 | policy year written and earned, six-month | 0.50 | 2.00 | 0.50 |

In-force exposures as of 01/01/11, 06/15/11 and 01/01/12 are 2, 3 and 4
for the annual policies (Table 4.15) and 2, 2 and 2 houses for the
six-month ones (Table 4.22).

Three things about how the book counts:

- **Months, not days.** A quarter of an annual policy is 0.25. By exact
  days the last quarter of 2010 is 92 of 365 days, 0.2521, and the first
  quarter of 2012 is 91 of 366, 0.2486. `earned_exposure` with
  `EarningMethod::Daily` therefore matches the earned tables to within two
  days in the year (6e-3 in the tests) and no closer, while the written
  and policy year tables match exactly [earned] [test]. The book's
  half-month grid (`ratemaking::year_fraction`) would reproduce the earned
  tables exactly; a method on it is proposed for the next PR.
- **The expiration date is the last day in force.** The book writes a
  policy effective 10/01/10 as expiring 09/30/11. `Policy` takes the day
  after, 1 October 2011, so that a term is the difference of its two
  dates.
- **In force is a snapshot on a date**: an inception on or before the
  date and an expiration after it, with no regard to how long the policy
  runs. A policy effective on the date counts (Policy B on 01/01/11), and
  the book counts one house a policy even for a six-month term.
  `in_force(policies, date)` is that, and a window's `in_force` in the
  table is the snapshot on the window's last day.

"As of 12/31/12" is a valuation date of 1 January 2013 in `earned`, which
reads a valuation date as the close of the day before.
