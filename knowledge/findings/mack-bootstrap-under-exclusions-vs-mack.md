---
type: Finding
title: Mack's bootstrap under n_periods and the drops reproduces Mack's standard error under the same settings
description: With the drops fixed on the observed triangle (only kept link ratios have residuals and enter the pseudo factors), MackBootstrap::fit under n_periods = 5 and under one high and one low dropped has the chain ladder's mean and Mack's standard error under the same settings on RAA and GenIns, every origin and the total within 2 Monte Carlo standard errors at 20,000 simulations, once the parameter error is scaled by the kept residuals' variance as in the plain case.
tags: [reserving, bootstrap, mack, lifetime, exclusions, n_periods, drop_high, drop_low]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-19T12:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-19T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 14, the bootstraps and the one-year views under the exclusions
  - id: test
    resource: ../validation/tests/reserving_bootstrap_exclusions.rs
    title: Validation test of Mack's bootstrap under exclusions against Mack
  - id: plain
    resource: mack-bootstrap-lifetime-vs-mack.md
    title: Mack's bootstrap, lifetime view, against Mack's standard errors and EVW Table 4
---

# Finding

The drops stay fixed on the observed triangle (decision 14, item 2), so
Mack's bootstrap describes the model that `Mack` computes with 0/1 weights
(R's `MackChainLadder(weights = W)`, at parity with it). Only the kept link
ratios have a residual, the small-sample factor uses the informative kept
links at the age, and the pseudo factors average the kept links only.[^test]
Then, as in [the plain case](mack-bootstrap-lifetime-vs-mack.md), after
centring and the one adjustment of scaling the parameter error by the
resampled residuals' variance `v = 1 - m^2` (here of the kept ones), the
bootstrap's mean is the chain ladder's and its standard deviation is
`sqrt(process^2 + v parameter^2)` of Mack's under the same settings. At
20,000 simulations (Gamma process, seed 20,261,007), every origin and the
total are within 2 Monte Carlo standard errors of the standard deviation
and 5 of the mean (the test's tolerance is 5 for both):

| Dataset and setting | Total sd | Mack | Gap in SE | Largest origin gap in SE |
|---|---|---|---|---|
| RAA, `n_periods = 5` | 22,510.5 | 22,235.5 | +1.98 | +1.87 (1990) |
| RAA, drop 1 high and 1 low | 16,724.8 | 16,607.8 | +0.94 | +1.32 (1985) |
| GenIns, `n_periods = 5` | 2,537,588 | 2,527,420 | +0.78 | +1.69 (2007) |
| GenIns, drop 1 high and 1 low | 2,282,386 | 2,303,372 | -1.86 | +1.50 (2010) |

Plain Mack's total (before the `v` adjustment) is 22,261.2, 16,617.4,
2,527,472 and 2,303,595. The mean total reserves are 61,883 against the
chain ladder's 61,792 (RAA window), 52,462 against 52,450 (RAA drops),
18,539,089 against 18,518,169 (GenIns window) and 18,649,659 against
18,666,211 (GenIns drops).[^test]

[^test]: validation/tests/reserving_bootstrap_exclusions.rs
[^design]: Design note, reserving v0.2, decision 14
