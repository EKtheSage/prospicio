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
  and policy year tables match exactly [earned] [test]. On the book's
  half-month grid (`ratemaking::year_fraction`), which is
  `EarningMethod::HalfMonth`, the earned tables are exact too.
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

## Changes, cancellations and twenty-fourths

The book's records (chapter 3, Table 3.1) fix how a mid-term event is
written, and `earned` follows them [book] [earned] [test]:

- **A cancellation** is a second record of the unearned part, negative,
  dated the last day in force: Policy B (600 from 04/01/10, cancelled on
  December 31, 2010, after 75% of its term) has -0.25 exposure and -150
  dated 12/31/10. Policy D cancelled on March 31, 2012 contributes 1 and
  400 to calendar year 2011 and -0.25 and -100 to 2012, and all of it to
  policy year 2011. `Policy::cancelled` takes the first day not in force
  (1 January 2011 for Policy B) and books the return the day before.
- **A change** is stated as the new full-term amount: Policy C (1,000 from
  07/01/10, 1,200 full term from January 1, 2011) has records of -500 and
  +600 on 01/01/11, so 100 is written then, and 500 and 600 are earned in
  the two years. The in-force amount is the full-term amount in effect:
  chapter 5's Policy D is in force at 400 until December 31, 2011, and at
  800 from January 1, 2012, and pays 600.
- **Twenty-fourths** ("the 15th of the month rule") take every policy as
  written at mid-month: an annual policy earns 1/24 in its first month,
  2/24 in each of the next eleven and 1/24 in the thirteenth (Table 4.24:
  230, 210, ..., 10 earned in 2010 of 240 written in each month), and is
  not in force on the first of the month it is written in (Table 4.23:
  1,440, 2,880 and 1,440 in force on 07/01/10, 01/01/11 and 07/01/11).
