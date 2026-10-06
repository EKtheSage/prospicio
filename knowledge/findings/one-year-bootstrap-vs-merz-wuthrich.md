---
type: Finding
title: The ODP one-year bootstrap against Merz-Wuthrich
description: Re-reserving on the ODP bootstrap gives a one-year CDR standard deviation 0.30 to 5.95 times Merz-Wuthrich's per origin; the gap is the ODP's process variance, not the re-reserving, which with Mack's own bootstrap reproduces Merz-Wuthrich within 0.2%.
tags: [reserving, one-year, cdr, bootstrap, odp, merz-wuthrich, solvency-ii]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-06T18:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 8
  - id: test
    resource: ../validation/tests/reserving_one_year_bootstrap.rs
    title: Validation test of the simulated one-year view
  - id: evw
    resource: https://openaccess.city.ac.uk/id/eprint/21270/
    title: England, Verrall and Wuthrich (2019), On the lifetime and one-year views of reserve risk, with application to IFRS 17 and Solvency II risk margins, Insurance - Mathematics and Economics 85
  - id: boumezoued
    resource: https://arxiv.org/abs/1107.0164
    title: Boumezoued, Angoua, Devineau and Boisseau (2011), One-year reserve risk including a tail factor - closed formula and bootstrap approaches
---

# Finding

`OdpBootstrap::one_year` with the volume-weighted chain ladder, 20,000
simulations from seed 20,261,006, against R ChainLadder's
`CDR(MackChainLadder(tri))` `CDR(1)S.E.` (log-linear last sigma):[^test]

| Dataset | Total ratio | Per-origin ratios | Lifetime ratio (bootstrap SD / Mack SE, total) |
|---|---|---|---|
| RAA | 0.458 | 0.30 (1990) to 4.90 (1982) | 0.70 |
| GenIns | 1.023 | 0.70 (2006) to 2.22 (2004) | 1.23 |
| ABC | 0.970 | 0.80 (1987) to 5.95 (1979) | 1.13 |

The totals of GenIns and ABC are close to Merz-Wuthrich only by
coincidence: per origin they are not, and RAA's total is less than half.
R's `odp_one_year()` and Python's `OdpBootstrap.one_year()` give the same
draws for the same seed, and their tests hold RAA's (R) and GenIns's
(Python) total to the same ratios.

# Why

* The ODP's process variance is `phi` times the mean increment, one `phi`
  for the triangle; Mack's is `sigma_k^2` times the cumulative value, one
  sigma per age. RAA 1990 (2,063 at 12 months) shows it: its next cell
  has mean `2063 * (2.999 - 1) = 4,125`, an ODP process SD of
  `sqrt(983.6 * 4,125) = 2,014` and Mack's `166.98 * sqrt(2063) = 7,585`.
  Times the factor to ultimate from 24 months (2.974), that is 5,990
  against 22,556, which is most of R's 23,610. The ODP gives 7,159 with
  parameter error.
* Old origins go the other way: Mack's extrapolated last sigmas are tiny
  (RAA 1982: 143), while the ODP still charges `phi` per unit of mean.
* The ODP's one-year share of its lifetime risk (one-year SD over
  `BootChainLadder` SD) is lower than Mack's: totals 0.61, 0.60 and 0.66
  against Merz-Wuthrich over Mack 0.94, 0.73 and 0.77. Mack front-loads
  the risk of a young origin into its first factor.
* The re-reserving itself is not the gap. The same loop (append the next
  diagonal, refit the volume-weighted chain ladder, `CDR = U0 - U1`) with
  England, Verrall and Wuthrich's bootstrap of Mack's model (their
  Appendix 1: residuals of the link ratios, Gamma next cell with mean
  `f* C` and variance `sigma^2 C`) on GenIns with Mack's last-sigma rule,
  200,000 simulations, gives 1,780,466 against Merz-Wuthrich's 1,778,968,
  every origin within 0.2%. This was measured with a scratch harness, not a
  CI test.

# Published numbers

* England, Verrall and Wuthrich (2019) bootstrap Mack's model, not the
  ODP. Their Table 2 (analytic: reserves, Mack RMSEP and Merz-Wuthrich
  RMSEP on Taylor-Ashe, Mack's rule for the last sigma, total 1,778,968)
  equals act-reserving and R to the unit; the validation test checks it.
  Their Table 4 (500,000 simulations of the Mack bootstrap) gives a
  one-year total of 1,778,428. They say the ODP partitions the lifetime
  risk into uncorrelated one-year views too, but publish no ODP
  numbers.[^evw]
* Boumezoued et al. (2011) re-reserve with a Mack-type bootstrap on the
  Merz-Wuthrich 2008 triangle: 81,074 simulated against 81,081
  analytic.[^boumezoued]
* No published one-year standard deviation of the ODP bootstrap on RAA,
  GenIns or ABC was found, so the validation test holds each SD to R's
  Merz-Wuthrich value times the ratio measured here, within five Monte
  Carlo standard errors of the SD (`sd * sqrt((kurtosis - 1) / (4 n))`,
  five of which are 2.5% to 4% of the SD at 20,000 simulations).

# Other measurements

* An origin with one cell left has a one-year view equal to its run-off.
  Its SD is 0.985 (RAA), 0.995 (GenIns) and 0.996 (ABC) times the lifetime
  bootstrap's: the one-year view projects from the observed latest value,
  `BootChainLadder` from the resampled one.
* The CDR's mean is slightly negative, -8.5% (RAA), -5.2% (GenIns) and
  -1.5% (ABC) of its SD: the resampled factors' bias, which also puts R's
  `BootChainLadder` mean reserve above the chain ladder's. The Mack
  bootstrap shows it too (-3.4% on GenIns).

See [R ChainLadder CDR](/references/r-chainladder-cdr.md).

[^test]: validation/tests/reserving_one_year_bootstrap.rs
[^evw]: England, Verrall and Wuthrich (2019), Tables 2 and 4, Appendix 1 and Section 7
[^boumezoued]: Boumezoued et al. (2011), Table 2
