---
type: Finding
title: The ODP one-year bootstrap against Merz-Wuthrich
description: Re-reserving on the ODP bootstrap, projecting from the pseudo latest value as the lifetime bootstrap does, gives a one-year CDR standard deviation 0.50 to 5.98 times Merz-Wuthrich's per origin; the gap is the ODP's process variance, not the re-reserving, which with Mack's own bootstrap reproduces Merz-Wuthrich within 0.6% on GenIns.
tags: [reserving, one-year, cdr, bootstrap, odp, merz-wuthrich, solvency-ii]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-06T22:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 8
  - id: test
    resource: ../validation/tests/reserving_one_year_bootstrap.rs
    title: Validation test of the simulated one-year view
  - id: unit
    resource: ../crates/act-reserving/src/one_year_bootstrap.rs
    title: Unit tests mack_bootstrap_rereserving_reproduces_merz_wuthrich and one_cell_left_is_the_lifetime_run_off
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

| Dataset | Total ratio | Per-origin ratios | One-year / lifetime (total) | Merz-Wuthrich / Mack (total) |
|---|---|---|---|---|
| RAA | 0.609 | 0.50 (1990) to 4.97 (1982) | 0.81 | 0.94 |
| GenIns | 1.362 | 0.73 (2006) to 2.26 (2004) | 0.80 | 0.73 |
| ABC | 1.133 | 0.84 (1983) to 5.98 (1979) | 0.77 | 0.77 |

One-year / lifetime is the one-year total SD over R `BootChainLadder`'s
total reserve SD (`reference/reserving_bootstrap_r.csv`). The validation
test pins these ratios (a seed-pinned regression of this implementation,
not a check against Merz-Wuthrich), and so do the R test (RAA total) and
the Python test (GenIns total).

# The projection base

Each simulation's next increment has mean `C*_latest * (f*_k - 1)`, from
the pseudo triangle's latest value, as the lifetime ODP bootstrap (England
2002, R `BootChainLadder`) projects; the increment is then added to the
observed latest value for the refit. England, Verrall and Wuthrich ask
that the forecasts come from the model "bootstrapped appropriately", and
for the ODP that means the pseudo latest value, which carries the
estimation error of the origin's level. Mack's model is conditional on the
observed latest value, so their Mack bootstrap projects from it.[^evw]

The first version projected from the observed latest value. That left the
level's estimation error out: RAA's total SD was 11,653 instead of 15,318
(-24%), and 1990's 7,159 instead of 11,840 (-40%). It also broke the
identity below, by 1.5% on RAA, which a 2% allowance in the test had
absorbed.

* An origin with one cell left has a one-year view distributed as its
  lifetime reserve. Its SD is 0.998 (RAA 1982), 1.012 (GenIns 2002) and
  1.001 (ABC 1978) times `BootChainLadder`'s, within the Monte Carlo
  tolerance; a unit test also checks it against the crate's own lifetime
  bootstrap.[^unit]

# Why the ODP is not Merz-Wuthrich

* The ODP's process variance is `phi` times the mean increment, one `phi`
  for the triangle; Mack's is `sigma_k^2` times the cumulative value, one
  sigma per age. RAA 1990 (2,063 at 12 months) shows it: its next cell
  has mean `2063 * (2.999 - 1) = 4,125`, an ODP process SD of
  `sqrt(983.6 * 4,125) = 2,014` and Mack's `166.98 * sqrt(2063) = 7,585`.
  Mack front-loads the risk of a young origin into its first factor.
* Old origins go the other way: Mack's extrapolated last sigmas are tiny
  (RAA 1982: 143), while the ODP still charges `phi` per unit of mean.
* The re-reserving itself is not the gap. The same loop (append the next
  diagonal, refit the volume-weighted chain ladder, `CDR = U0 - U1`) with
  England, Verrall and Wuthrich's bootstrap of Mack's model (their
  Appendix 1: scaled bias-adjusted residuals of the link ratios, Gamma
  next cell with mean `f* C` and variance `sigma^2 C`) on GenIns with the
  log-linear last sigma, 20,000 simulations, gives 0.994 to 1.000 times
  Merz-Wuthrich per origin and 0.995 in total. A unit test runs it in CI
  at five Monte Carlo standard errors (about 2.5%).[^unit] On RAA the Mack
  bootstrap's first pseudo factor has an SD of a third of the factor, so
  a Gamma mean can be negative; the test uses GenIns, as EVW do.

# Published numbers

* England, Verrall and Wuthrich (2019) bootstrap Mack's model, not the
  ODP. Their Table 2 (analytic: reserves, Mack RMSEP and Merz-Wuthrich
  RMSEP on Taylor-Ashe, Mack's rule for the last sigma, total 1,778,968)
  equals act-reserving and R to the unit; the validation test checks it.
  Their Table 4 (500,000 simulations of the Mack bootstrap) gives a
  one-year total of 1,778,428. They say simulation studies of the ODP
  partition the lifetime risk into uncorrelated one-year views too, but
  publish no ODP numbers.[^evw]
* Boumezoued et al. (2011) re-reserve with a Mack-type bootstrap on the
  Merz-Wuthrich 2008 triangle: 81,074 simulated against 81,081
  analytic.[^boumezoued]
* No published one-year standard deviation of the ODP bootstrap on RAA,
  GenIns or ABC was found.

# Other measurements

* The CDR's mean is slightly negative, -6.4% (RAA), -5.0% (GenIns) and
  -0.4% (ABC) of its SD: the resampled factors' bias, which
  also puts R's `BootChainLadder` mean reserve above the chain ladder's.
  The Mack bootstrap shows it too (-3.4% of the SD on GenIns).

See [R ChainLadder CDR](/references/r-chainladder-cdr.md).

[^test]: validation/tests/reserving_one_year_bootstrap.rs
[^unit]: crates/act-reserving/src/one_year_bootstrap.rs, unit tests
[^evw]: England, Verrall and Wuthrich (2019), Tables 2 and 4, Section 2.2, Section 7 and Appendix 1
[^boumezoued]: Boumezoued et al. (2011), Table 2
