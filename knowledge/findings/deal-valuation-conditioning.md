---
type: Finding
title: Conditioned on the paid to date, a bound deal's forward spread falls with the unpaid share, and truncation drops simulations late
description: In the cash-flow engine's valuation after binding, Bornhuetter-Ferguson conditioning keeps every simulation and its since-inception NPV's sd falls close to the unpaid share 1 - S of the paid pattern (0.88, 0.75, 0.62, 0.49, 0.38 of inception's at months 6, 12, 18, 24, 29 of a 48-month pattern); truncation keeps 100% of the simulations at month 6 and 80% at month 29 on the same path. Over 300 paths drawn from the priced distribution, the realised NPV's percentile among the conditioned forward NPVs is uniform.
tags: [pricing, cashflow, collateral, monitoring, ils]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-28T18:00:00Z }
verified: { by: process:ci, at: 2026-10-28T18:00:00Z }
sources:
  - id: note
    resource: ../docs/design/time-value.md
    title: Design note, the time value of money in a price, decision 34, a bound deal valued from its actual accounts
  - id: test
    resource: ../crates/prospicio-pricing/src/cashflow.rs
    title: the_forward_distribution_narrows_as_the_deal_runs_off and the_actual_percentile_is_uniform_over_paths, unit tests CI's Rust job runs
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/603
    title: Issue 603, deal monitoring after binding
---

# What was found

`CashflowTerms::valuation` values a collateralised deal from its actual
accounts at a month `v` and runs the rest of the term with each
simulation's ultimate conditioned on the actual paid `A`, with `S` the
paid pattern reached by the end of month `v - 1`. On the test deal
(premium 100, commutation at 30 months, a 48-month linear paid pattern,
the shortfall funding rule, a 12-month buffer, a fee), 4,000 simulated
loss ratios lognormal with log-mean -0.3 and log-sd 0.35, and one path at
a loss ratio of 0.9 taken as the actual:

| Valuation month | 0 | 6 | 12 | 18 | 24 | 29 |
|---|---|---|---|---|---|---|
| `1 - S` | 1 | 0.875 | 0.75 | 0.625 | 0.5 | 0.396 |
| BF: sd of the since-inception NPV at 5%, over inception's | 1 | 0.880 | 0.752 | 0.618 | 0.488 | 0.381 |
| Truncation: share of simulations kept | 1 | 1 | 0.9995 | 0.9855 | 0.920 | 0.802 |

Under Bornhuetter-Ferguson conditioning (`A + u_j P (1 - S)`) the spread
falls close to the unpaid share, as the loss still to pay is `u_j P (1 - S)`.
Truncation keeps each surviving simulation's whole ultimate `u_j P`, so its
spread stays wide, and it drops the simulations that have already paid
less than the actual, a fifth of them by month 29 here.

Over 300 paths drawn from the priced distribution, valued at month 12, the
actual's A vs E percentile at month 11 and the realised since-inception
NPV's percentile among the conditioned forward NPVs both pass a
Kolmogorov-Smirnov test of uniformity at 1%: the conditioned forward
distribution brackets the realised return as often as it says.

# How it is checked

Rust unit tests in `prospicio_pricing::cashflow` pin the table to 1e-4
(the BF row) and exactly (the truncation row) at their seeds, and run the
uniformity tests; CI's Rust job runs them. The same module checks that a
path valued at its own loss ratio gives the rest of its flows to 1e-9 under
both conditionings and every rule.

# Limits

The paid pattern is the terms' one pattern; a valuation with a paid
pattern by simulation (#600) is not built. The forward IRR and MOIC are
not tabulated here, only the NPV's spread.
