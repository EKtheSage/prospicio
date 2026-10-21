---
type: Finding
title: A reference pricing model's deal deductions are flat on premium and its scales read the loss ratio before the cap, so the issue bodies' formulas held only in special cases
description: "In a reference simulation pricing model the TPA fee, tax, reinsurance cost and fronting fee are flat rates on the placed premium, so they shift the total deductions without spreading them (sd(TD) = sd(CC) with flat brokerage), and only the manager's fee sits outside NMBF. Its commission and brokerage scales read the loss ratio before the loss-ratio cap and corridor, held inside the commission scale's range, each section on its own loss ratio. NMBF = 1 - ULR - CC - BK holds only with the four legs at zero, and summing scales point by point only for a one-section deal."
tags: [prospicio-aggregate, reinsurance, deal-ratios, brokerage, sliding-scale, reference-model]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T23:30:00Z }
sources:
  - id: issue-485
    resource: https://github.com/EKtheSage/prospicio/issues/485
    title: "Issue #485: brokerage and other deduction legs"
  - id: issue-477
    resource: https://github.com/EKtheSage/prospicio/issues/477
    title: "Issue #477: annual contracts, sliding scales and brokerage"
  - id: test
    resource: ../crates/prospicio-aggregate/src/deal.rs
    title: the deal module's tests
---

# Deal deductions and the loss ratio the scales read

The user read a reference simulation pricing model and its workbooks to
answer #485 and #477. What they found, in prospicio's terms:

**The deductions.** With `P` the placed premium (subject premium times
the placed share):

- The TPA fee, tax, reinsurance cost and fronting fee are flat rates of
  `P`. They do not move with the loss ratio, so with flat brokerage the
  standard deviation of the total deductions equals that of the ceding
  commission.
- `CR = ULR + CC + BK + fee + TPA + tax + RI cost + fronting`, `TD = CR -
  ULR`, `NMBF = 1 - CR + fee` and `margin = NMBF - fee`. The manager's fee
  is the only deduction kept outside NMBF.
- The fronting fee was zero in every workbook examined, and one build
  leaves the reinsurance cost and fronting fee out of the combined ratio.
  A portfolio-level fronting fee is a cash-flow item, not in the ratio.

**The scales.**

- The commission and brokerage scales read the loss ratio before the
  loss-ratio cap and corridor (`LR`), not the capped `ULR`, and both read
  the same key by the same function: linear between anchors, held at the
  ends.
- The key is held inside the commission scale's range of loss ratios,
  and the brokerage key too.
- Each section's scale reads that section's loss ratio on its own premium,
  a deal-level scale reads the deal's, and the amounts add. In one
  workbook the mean commission (0.419) is above the deal scale's maximum
  (0.358), the section commission (0.268) making up the difference.
- The scale sets the brokerage in the ratios; the provisional brokerage
  feeds only the cash flows (#479).
- Every brokerage scale found is flat, so there is no numerical check of a
  sliding one.

**In prospicio** (`prospicio_aggregate::deal`): brokerage is a
`Commission` on the layer, flat or on a scale; each other leg is a
`Deduction` on the premium (the default), the ceded loss or an amount,
inside NMBF unless marked outside. `Layer::scales_before_annual_terms`
keys both scales on the placed share of the recoveries before the annual
deductible, corridor and annual limit, so a cap stated as an
`aggregate_limit` reproduces the reference's `LR` key; the default stays
the ceded loss ratio, as `aggregate` reads it.
`Layer::brokerage_within_commission_range` is the clamp, off by default.
A deal of several sections is one layer per section. A deal-level scale
read on the loss ratio of several sections together is not built.
`Tower::deal_ratios` gives each simulation's loss ratio (and the one
before terms), each deduction, the total, the combined ratio, NMBF and
the margin. The tests check `TD - CC - BK` equal to the flat legs,
`sd(TD) = sd(CC)`, `NMBF - margin` equal to the fee, a loss-basis leg
spreading the total, and the scales read at 140% under a 120% cap.
