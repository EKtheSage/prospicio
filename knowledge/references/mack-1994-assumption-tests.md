---
type: Reference Implementation
title: Mack's assumption tests in Mack (1994) and R ChainLadder
description: Mack (1994) appendices G and H, the development-factor correlation and calendar-year effect tests, printed on the RAA triangle (not Taylor-Ashe), and R ChainLadder 0.2.21's dfCorTest and cyEffTest, which reproduce them with qnorm intervals instead of Mack's 2 and .67; the calendar test has little power on a 10 x 10 triangle.
resource: https://cran.r-project.org/package=ChainLadder
tags: [reserving, mack, chain-ladder, assumptions, calendar-year, correlation, parity, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T09:00:00+00:00 }
verified: { by: process:ci, at: 2026-10-28T09:00:00+00:00 }
sources:
  - id: paper
    resource: https://www.casact.org/sites/default/files/database/forum_94spforum_94spf101.pdf
    title: Mack (1994), Measuring the variability of chain ladder reserve estimates, CAS Forum Spring 1994, appendices G (pp. 157-161) and H (pp. 162-168)
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 27
  - id: script
    resource: ../validation/scripts/reserving_assumptions_r.R
    title: Generator of validation/reference/reserving_assumptions_r.csv
  - id: tests
    resource: ../validation/tests/reserving_assumptions.rs
    title: The parity tests CI runs
---

# Mack's example

Mack's 1994 paper works both tests on the triangle of its chapter 6, the
RAA data (origins 1981 to 1990), not on Taylor and Ashe's GenIns, which
issue #551 named.[^paper] Printed values:

* Correlation (appendix G, p. 161): `T_k = 4/21, -9/28, 3/7, -1/5, 2/5,
  -1/2, 1` for `k = 2, ..., 8` with weights `I - k - 1 = 7, ..., 1`;
  `T = .070`; `Var(T) = 1/28`; the 50% limits `+-.67 / sqrt(28) =
  +-.127`. Not rejected.
* Calendar year (appendix H, p. 167): `S_j` and `L_j` for `j = 2, ..., 9`
  are `(1,1), (3,0), (3,1), (1,3), (1,3), (2,4), (4,4), (4,4)`; `Z = 14`,
  `E(Z) = 12.875`, `Var(Z) = 3.9785`; the 95% range is
  `E(Z) +- 2 sd = (8.886, 16.864)`. Not rejected.
* His worked moments for `n = 5`: `E(Z_j) = 50/32`, `Var(Z_j) = 95/256`.

Mack ranks each column of factors over all its elements for the
calendar test (the median of the whole column), and for the correlation
test ranks column `k` again without its last element, so that both columns
of a pair are ranked over the same origins.[^paper]

# R ChainLadder 0.2.21

* `cyEffTest(Triangle, ci = 0.95)` and `dfCorTest(Triangle, ci = 0.5)`
  return `Z`, `E`, `Var`, `Range` and a `test_table` (S_j, L_j, Z_j, n, m,
  E_Zj, Var_Zj for `j = 2, ..., n - 1`), and `T_stat`, `Var`, `Range`.
  `dfCorTest` does not return its `T_k`; they are
  `cor(ata[, i], ata[, i + 1], method = "spearman", use =
  "pairwise.complete.obs")`, Pearson's on average ranks.[^script]
* Intervals are `qnorm(ci + (1 - ci) / 2)` standard deviations, so R's RAA
  range is `(8.9656, 16.7844)`, not Mack's `(8.886, 16.864)`; prospicio
  reproduces Mack's with `ci = 2 Phi(2) - 1`.[^design]
* The weights are hard-coded by position, `(n - 3):1`, and `Var(T) =
  1 / ((n - 2)(n - 3) / 2)`, both assuming a full square triangle;
  prospicio uses each pair's `n - 1` and `1 / sum(weights)`, equal on a
  full triangle.[^design]
* The functions assume a square triangle (`ata(Triangle)[1:n - 1, ]`, the
  anti-diagonals `S_L_Triangle[1:i, i:1]`). prospicio keys a diagonal by
  the valuation month of each link ratio's later cell.[^design]
* On the three datasets: RAA passes both; GenIns passes the calendar test
  (`Z = 12` in `(8.91, 16.09)`) and fails the correlation test at 50%
  (`T = -0.1636` against `+-0.1275`); ABC fails both (`Z = 8` outside
  `(12.22, 20.38)`, `T = 0.5439` against `+-0.1124`).[^tests]

# Power

On a 10 x 10 triangle the calendar test has little power: RAA and GenIns
with every payment of two calendar years raised by 50% or 100% (years 5
and 6, 4 and 6, 5 and 7 or 3, 5 and 7 after the first origin) still pass
in all ten configurations on RAA and seven of ten on GenIns, since their
link ratios vary more than the shock moves them. A 15-origin synthetic triangle with 5% noise fails with two shocked
years at 50%, which is the shock test CI runs.[^tests]

[^paper]: paper, appendices G and H.
[^script]: script, and R ChainLadder's source of `dfCorTest`.
[^design]: design, decision 27, items 2, 4 and 5.
[^tests]: tests, `matches_r_chainladder` and `an_injected_calendar_shock_fails_the_test`.
