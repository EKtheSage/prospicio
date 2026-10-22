---
type: Finding
title: A collateralised deal's monthly accounts conserve money, and the issue's funding rule lets the trust run below zero
description: "In prospicio_pricing::cashflow the premium, adjustments and interest paid into the trust equal the investor's net flows, the fee, the claims, the expenses and the commutation payment, to 1e-9, under every rule; #479's funding rule (the rise in the requirement net of premium collected) never calls the claims paid, so on a test deal the trust falls to about -8.9 per 100 of premium. Which obligation the rise is measured from, what the fee's margin is, when the adjustment reads the premium and whether the commutation reserve is loaded are open until the user's reference run."
tags: [prospicio-pricing, cashflow, collateral, reinsurance, reference-model]
status: draft
generated: { by: claude-code/cloud-session, at: 2026-10-22T03:00:00Z }
sources:
  - id: issue-479
    resource: https://github.com/EKtheSage/prospicio/issues/479
    title: "Issue #479: a monthly cash-flow engine over simulations"
  - id: test
    resource: ../crates/prospicio-pricing/src/cashflow.rs
    title: the cash-flow module's tests
---

# A collateralised deal's monthly accounts

`prospicio_pricing::cashflow` runs a trust (collateral) account and an
investor's operating account month by month for each simulation's
ultimate loss ratio `u` and commission rate `c`
(`docs/design/time-value.md`, decisions 13 to 20).

**Conservation.** Over the term, what comes into the deal (the premium
collected, the adjustments and the trust's interest) equals what leaves
it: the investor's net flows, the performance fee, the claims paid, the
expenses and the commutation requirement paid to the cedant. Both
accounts end at zero. The tests check this to 1e-9 under five settings
that between them use every rule, at loss ratios from 0 to 2
(`Trace::conservation_error`); a leg booked twice or dropped breaks it.

**The issue's funding rule.** #479 calls capital on the rise in
`S_t − net_rate × cWP_{t−lag}`, the requirement net of the premium
collected. The requirement already nets the claims paid (`u E − C`), but
the obligation does not add them back, so between rises the trust pays
claims it was never funded for. On the test deal (premium 100 written over
12 months, earned over 36, paid over 48, `u = 0.9`, 70% collected two
months late, a quarterly call) the trust's balance falls to -8.885 at its
lowest; the shortfall rule keeps it at the requirement
(`the_obligation_rule_lets_the_trust_run_below_zero`). The
engine follows the issue and lets the balance go below zero, visible in
`Trace::collateral_balance`.

**Open until the reference run.** The engine offers each as a setting:

- the obligation the rise is measured from: the previous funding date's or
  the highest so far;
- the performance fee's margin: the profit, over the premium or over the
  capital;
- whether the adjustment `adj_rate × cWP_end − cc × cEP_end` reads the
  ultimate premium or that month's;
- whether the commutation reserve carries a load (the issue lists one among
  its negative controls).

No parity has been run against the reference model, whose numbers are in
the user's local reproduction, so nothing here is marked verified.
