---
type: Reference
title: Excel's NPV, IRR, XNPV and XIRR
description: Excel's NPV discounts its first value a full period, so NPV(r, x0..xn) is the value one period before x0; XNPV and XIRR discount by (1 + r)^-(d - d0)/365 from the first date, which is prospicio_pricing::returns with TimeUnit::Days on offsets from the first date; Microsoft's worked examples (XNPV 2086.65, XIRR 0.373362535, IRR 8.7% and -2.1%) are reproduced by prospicio_pricing::returns to 1e-9. The add-in dropped PRO.NPV, PRO.IRR and PRO.MOIC after 0.0.7 in favour of these built-ins.
resource: https://support.microsoft.com/office/xirr-function-de1242ec-6477-445b-b11b-a303ad9adc9d
tags: [pricing, returns, irr, npv, excel, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-03T12:00:00Z }
verified: { by: process:ci, at: 2026-11-03T12:00:00Z }
sources:
  - id: xirr
    resource: https://support.microsoft.com/office/xirr-function-de1242ec-6477-445b-b11b-a303ad9adc9d
    title: Microsoft, XIRR function (the example of five flows from 1 January 2008)
  - id: xnpv
    resource: https://support.microsoft.com/office/xnpv-function-1b42bbf6-370f-4532-a0eb-d67c16b664b7
    title: Microsoft, XNPV function (the same flows at 9%)
  - id: irr
    resource: https://support.microsoft.com/office/irr-function-64925eaa-9988-495b-b290-3ad0c163c1bc
    title: Microsoft, IRR function (-70,000 then 12,000 to 26,000 a year)
  - id: test
    resource: ../crates/prospicio-xll/tests/cashflow.rs
    title: excel_s_worked_examples, the check CI runs
---

# Timing

* `NPV(rate, v1, ..., vn)` is `Σ v_i (1 + rate)^-i` from `i = 1`: the
  first value is discounted a full period. A flow at time 0 is added
  outside it, `x0 + NPV(rate, x1..xn)`. `prospicio_pricing::returns::npv`
  on times 0, 1, 2, ... equals that sum, not `NPV` itself.
* `XNPV(rate, values, dates)` and `XIRR(values, dates)` discount each
  value by `(1 + rate)^-((d_i - d_1) / 365)`: actual days over 365 from
  the first date, whatever the dates' order. That is
  `returns::npv` and `returns::irr` on `TimeUnit::Days` offsets from the
  first date.
* `IRR(values)` takes yearly flows at 0, 1, 2, ....
* The Excel add-in had `PRO.NPV`, `PRO.IRR` and `PRO.MOIC` over these
  until 0.0.7; they were removed as duplicates of the built-ins
  (`docs/design/xll.md` decision 46). On a sheet, use `XNPV` and `XIRR`,
  and `BYROW` with a `LAMBDA` for a block of simulations.

# Worked examples, to more digits

The roots and values of the same sums, by bisection in double precision:

| Example | Microsoft prints | Value |
|---|---|---|
| XNPV(0.09, ...) | 2086.65 | 2086.6476020315363 |
| XIRR(...) | 0.373362535 | 0.37336253351883164 |
| IRR of six yearly flows | 8.7% | 0.0866309480365316 |
| IRR of the first five | -2.1% | -0.021244848273411054 |

Excel's own `XIRR` and `IRR` iterate from a guess (10% by default)
until the result is accurate within 0.000001 percent, so they agree with
the crate to about 1e-7, not to the bit. `prospicio_pricing::returns::irr`'s grid search (400 points
from -95% to 100%, the bracket nearest 0) finds the same roots, and
unlike `IRR` names a root outside its range rather than returning
`#NUM!`.
