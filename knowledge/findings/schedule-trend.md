---
type: Finding
title: Trend factors from a calendar-year rate schedule, as the AP pricing demo computes them
description: Each accident year trends from y + 0.5 (losses) or y (premium) to inception + 1 or + 0.5 (RAD) or + 0.5 or + 0 (LOD), compounding each calendar year's rate over its overlap; an incomplete latest year starts at y + c/24 (premium y + c/24 - 0.5); a year not in the schedule takes the latest year's rate, gaps included; the issue's 8 cases, two full columns and the extension case are reproduced to 1e-12.
tags: [ratemaking, trend, pricing, excel, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-10T12:00:00Z }
verified: { by: process:ci, at: 2026-11-10T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-pricing/src/ratemaking.rs
    title: prospicio_pricing::ratemaking (ScheduleTrend, RateSchedule, cumulative_loss_ratio)
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/801
    title: "#801, trend factors to a treaty period from a calendar-year rate schedule"
  - id: xll
    resource: ../crates/prospicio-xll/tests/trend_schedule.rs
    title: PRO.PRC.TREND.FACTORS, PRO.PRC.LR.CUMULATIVE and PRO.REVERSE tests
---

# Trend factors from a calendar-year rate schedule

The AP pricing demo trends each accident year's losses and premium to a
treaty period with one rate per calendar year. Its workbook does it with
`EXP(SUMPRODUCT(LN(1 + rate) x overlap))` per row; `ScheduleTrend` gives
the same numbers.

## The points

With `i` the inception and `c` the months complete of an incomplete
latest year:

| | Loss | Premium |
|---|---|---|
| Target, RAD | `i + 1` | `i + 0.5` |
| Target, LOD | `i + 0.5` | `i` |
| Start, complete year `y` | `y + 0.5` | `y` |
| Start, incomplete latest year | `y + c/24` | `y + c/24 - 0.5` |

Loss and premium therefore cover the same length for a complete year
under RAD at the latest year, which is why the issue's 2025 loss and
premium factors agree (1.060710030249) while 2012's do not.

## What the vectors confirm

- The overlap rule: each calendar year `k` contributes
  `(1 + r_k)^len([k, k+1) ∩ [start, target])`; a start at or after the
  target gives 1 (the extension case's 2025 against a 2016 inception).
- The fill rule: a calendar year outside the schedule takes the latest
  scheduled year's rate, before the schedule as well as after it (the
  extension case's 2008 and 2010 at 7%). `BeforeSchedule::Earliest` is the
  option for the earliest rate before the schedule.
- The requester's day count puts 1 January 2026 at 2026.0027397260274;
  `PRO.YEARFRACTION`'s half-month grid puts it at 2026.0, so an Excel date
  as the inception gives the issue's "inception exactly 2026.0" values.

## The running loss ratio

`cumulative_loss_ratio` is the demo's `CUM_WEIGHTED_LR`, `0` while the
premium to date is `0`. The demo's printed values (0.61, 0.594705882353,
0.638120662040, ..., 0.637034303676 in row 14) need its `inputs.json`,
which is not in the repository, so they are not yet a test.
