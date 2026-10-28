---
type: Finding
title: A collateralised deal's funding under a paid pattern by simulation is not the funding under the average pattern
description: In the cash-flow engine, half the simulations paying over 24 months and half over 56 fund 46.09 on average at a 90% loss ratio, against 38.96 when every simulation pays by the average of the two patterns, 18% more; the investor's mean NPV at 10% is -31.73 against -31.20. Pattern uncertainty is a cost of its own and the mean pattern understates it.
tags: [pricing, cashflow, collateral, payment-pattern, reserving]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-28T12:00:00Z }
verified: { by: process:ci, at: 2026-10-28T12:00:00Z }
sources:
  - id: note
    resource: ../docs/design/time-value.md
    title: Design note, the time value of money in a price, decision 33, a paid pattern for each simulation
  - id: test
    resource: ../python/tests/test_pricing.py
    title: test_cashflow_paid_pattern_per_simulation, run by CI's Python job
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/600
    title: Issue 600, paid patterns that vary by simulation in the cash-flow engine
---

# What was found

`CashflowTerms::run_with_patterns` pays each simulation's ultimate loss by
its own monthly cumulative paid pattern (`PatternDraws`). On the deal the
tests build in full (premium 100, commutation at 30 months, a requirement
of 30% of unearned premium plus 120% of the reserve with a limit, the
shortfall funding rule, quarterly dates, a 12-month buffer), at an ultimate
loss ratio of 0.9 and a commission of 0.2:

| Paid patterns | Mean funded | Mean NPV at 10% |
|---|---|---|
| one simulation over 24 months, one over 56 | 46.086 | -31.735 |
| both at the average of the two | 38.959 | -31.196 |

Funding is the largest shortfall over the months, a maximum, so it is
convex in the path of payments: spreading the patterns raises the mean
funded even though the average pattern is the same. A deal priced on its
mean payment pattern understates the capital it needs and overstates the
investor's return when the pattern is uncertain, as a reserving model's
posterior says it is.

# How it is checked

The Python test pins the four numbers to 1e-9. The Rust and binding tests
also check that identical patterns equal to the terms' reproduce `run`
bit for bit, that a mixture is each pattern's run recombined, and that the
trace conserves money under a pattern of its own.

# Limits

Only the paid pattern varies; written and earned stay the terms'. The
patterns here are two ramps, not draws from a reserving fit; turning a
fit's posterior or bootstrap draws into monthly patterns is the Reserving
lane's part of #600.
