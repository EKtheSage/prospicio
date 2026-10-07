---
type: Finding
title: The simulated one-year view against Merz-Wuthrich, ODP and Mack's process
description: Re-reserving on the ODP bootstrap gives a one-year CDR standard deviation 0.50 to 5.98 times Merz-Wuthrich's per origin; the same re-reserving under Mack's process (MackBootstrap) reconciles its standard deviation with Merz-Wuthrich within Monte Carlo error on RAA, GenIns and ABC, so the gap is the ODP's process model, not the re-reserving. EVW's uncentred residuals bias its mean CDR (-0.21 to +0.18 SD); centring the pool removes the bias. At a quarterly grain split from an annual triangle both models give about half the annual SD, the ODP through its scale, which falls with the degrees of freedom (36/171 on RAA), Mack through sigmas a quarter the size.
tags: [reserving, one-year, cdr, bootstrap, odp, mack, merz-wuthrich, solvency-ii]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-07T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 8 and its section on Mack's process
  - id: test
    resource: ../validation/tests/reserving_one_year_bootstrap.rs
    title: Validation test of the ODP one-year view
  - id: mack
    resource: ../validation/tests/reserving_one_year_mack.rs
    title: Validation test of the one-year view under Mack's process (the reconciliation and EVW Table 4)
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

The volume-weighted chain ladder without a tail, re-reserved one year
ahead, 20,000 simulations from seed 20,261,006, Gamma process, against R
ChainLadder's `CDR(MackChainLadder(tri))` `CDR(1)S.E.` (log-linear last
sigma):[^test][^mack]

| Dataset | Merz-Wuthrich total | ODP total SD | Mack's process total SD | ODP / MW total | Mack / MW total | ODP / MW per origin | Mack / MW per origin |
|---|---|---|---|---|---|---|---|
| RAA | 25,166 | 15,318 | 25,664 | 0.609 | 1.020 | 0.50 (1990) to 4.97 (1982) | 0.989 to 1.021 |
| GenIns | 1,774,014 | 2,416,241 | 1,774,804 | 1.362 | 1.000 | 0.73 (2006) to 2.26 (2004) | 0.995 to 1.007 |
| ABC | 117,161 | 132,727 | 116,286 | 1.133 | 0.993 | 0.84 (1983) to 5.98 (1979) | 0.987 to 1.007 |

Mack's process, `MackBootstrap::one_year` (England, Verrall and Wuthrich
2019, Appendix 1), is within five Monte Carlo standard errors of every
origin and total there, and with Mack's rule for the last sigma too; the
validation test checks all of them in CI. That is the reconciliation, of
the standard deviation: replace the ODP's process with Mack's and the same
re-reserving gives Merz-Wuthrich, so the ODP's gap is its process model.
The mean is another matter (below).[^mack]

The ODP's ratios are a seed-pinned regression in its validation test, not
a check against Merz-Wuthrich, and so are the R test (RAA total) and the
Python test (GenIns total). Its one-year total is 0.81 (RAA), 0.80
(GenIns) and 0.77 (ABC) of R `BootChainLadder`'s lifetime total; Merz-
Wuthrich's is 0.94, 0.73 and 0.77 of Mack's.

# Why the ODP is not Merz-Wuthrich

* The ODP's process variance is `phi` times the mean increment, one `phi`
  for the triangle; Mack's is `sigma_k^2` times the cumulative value, one
  sigma per age. RAA 1990 (2,063 at 12 months) shows it: its next cell
  has mean `2063 * (2.999 - 1) = 4,125`, an ODP process SD of
  `sqrt(983.6 * 4,125) = 2,014` and Mack's `166.98 * sqrt(2063) = 7,585`.
  Mack front-loads the risk of a young origin into its first factor, so
  the ODP's 1990 is half of Merz-Wuthrich's and Mack's process matches it.
* Old origins go the other way: Mack's extrapolated last sigmas are tiny
  (RAA 1982: 143), while the ODP still charges `phi` per unit of mean, so
  the ODP's 1982 is five times Merz-Wuthrich's and Mack's process matches.
* The ODP's parameter error differs too: it projects from the pseudo
  latest value, which carries the estimation error of the origin's level;
  Mack's model is conditional on the observed latest value, and its
  bootstrap projects from it.

# The projection base

Each ODP simulation's next increment has mean `C*_latest * (f*_k - 1)`,
from the pseudo triangle's latest value, as the lifetime ODP bootstrap
(England 2002, R `BootChainLadder`) projects; the increment is then added
to the observed latest value for the refit. England, Verrall and Wuthrich
ask that the forecasts come from the model "bootstrapped appropriately",
and for the ODP that means the pseudo latest value. Mack's model is
conditional on the observed latest value, so their Mack bootstrap projects
from it.[^evw]

The first version projected from the observed latest value. That left the
level's estimation error out: RAA's total SD was 11,653 instead of 15,318
(-24%), and 1990's 7,159 instead of 11,840 (-40%). It also broke the
identity below, by 1.5% on RAA, which a 2% allowance in the test had
absorbed.

* An origin with one cell left has an ODP one-year view distributed as
  its lifetime reserve. Its SD is 0.998 (RAA 1982), 1.012 (GenIns 2002) and
  1.001 (ABC 1978) times `BootChainLadder`'s, within the Monte Carlo
  tolerance; a unit test also checks it against the crate's own lifetime
  bootstrap.[^unit]

# Mack's process in detail

* With 200,000 simulations GenIns and ABC are within 0.4% of R per origin
  and in total, but RAA's three youngest origins (1988 to 1990) and its
  total come out 0.4% to 1.2% above R, 2.5 to 6.4 Monte Carlo standard
  errors.
  An inference, not verified: Merz-Wuthrich's formula is a linear
  approximation of the CDR's MSEP, and RAA's young factors are the most
  volatile (the first pseudo factor's SD is a third of the factor), where
  the neglected higher-order terms are largest. At 20,000 simulations five
  standard errors are 2.5% to 5%, so the test does not resolve it.
* The bootstrap's draws equal the hand-written harness of EVW's Appendix 1
  in the unit test bit for bit (GenIns, the first 300 of the harness's
  20,000 simulations, each on its own stream), so the type is that
  harness.[^unit]
* EVW's Table 4 (500,000 simulations, Mack's rule for the last sigma, total
  one-year SD 1,778,428) against `MackBootstrap` with 20,000: total
  1,779,997 (+0.09%), 2002 75,547 against 75,502, every origin within five
  combined standard errors.[^mack][^evw]
* The process shape does not move the SD: Gamma and normal agree to 0.1%
  to 0.7% per origin on the same draws; Gamma, lognormal, residuals and
  normal all give GenIns's total within Monte Carlo error. RAA's normal
  process sends 1990's next value below zero in some simulations; the
  Gamma and lognormal do so only when the pseudo first factor is negative.
* The mean CDR under Mack's process, with EVW's uncentred residuals, is
  -0.214 (RAA), -0.038 (GenIns) and +0.176 (ABC) times its SD (seed
  20,261,006, 20,000 simulations, log-linear last sigma; pinned in the
  validation test). Merz-Wuthrich's CDR has mean zero. The cause: each
  factor's residuals have a zero `C^(alpha / 2)`-weighted sum, not a zero
  mean, so the pool's mean is 0.1395 (RAA), 0.0135 (GenIns) and -0.0595
  (ABC) with mean square exactly 1, and `E[f*_k] = f_k + m sigma_k
  sum(C^(alpha / 2)) / sum(C^alpha)`. EVW's Appendix 1 does not centre
  either; their Table 4 expected reserve on Taylor-Ashe (GenIns, pool mean
  0.0135) is only slightly above the chain ladder's. The `Residuals`
  process draws from the same pool, so uncentred it adds a bias of its
  own (mean `f* C + m sd`, variance `(1 - m^2) sd^2`).[^mack][^evw]
* `MackBootstrap::centre_residuals` subtracts the pool's mean before
  resampling (off by default, as EVW). Centred, at the same seed and
  20,000 simulations, the total mean CDR is -0.011 (RAA), -0.003 (GenIns)
  and +0.007 (ABC) times its SD, within Monte Carlo error of zero, and the
  SD still reconciles: total 1.003, 1.000 and 0.994 times R's
  `CDR(1)S.E.`, per origin 0.989 to 1.005, 0.994 to 1.006 and 0.987 to
  1.007, every origin within five Monte Carlo standard errors. Those
  numbers were measured once, not in CI; a unit test checks on RAA (4,000
  simulations) that the uncentred mean is far below zero and the centred
  Gamma and `Residuals` means are within four standard errors of it.[^unit]
  A review's independent R harness of EVW's Appendix 1 found the same
  pool means and, centred, RAA -0.013 and ABC +0.001 times the SD.

# Development grain and lagging origins

Both models simulate every cell valued in the twelve months after the
segment's valuation, so any development grain works; an origin short of
the diagonal develops from its own latest cell, and only the year's cells
are appended. Measured on RAA, GenIns and ABC split into quarters (each
year's increment in four equal parts), 5,000 simulations, against the
annual triangle:[^test][^unit]

* The opening chain-ladder reserve is the annual one to rounding: with
  cells in whole years, a year's four quarterly volume-weighted factors
  average over the same origins and telescope to the annual factor.
* The one-year SD is about half the annual one: ODP 0.43 to 0.47 per
  origin and 0.44 to 0.47 in total, Mack's process 0.48 to 0.72 and 0.53
  to 0.55 (highest for the origin with one year left, whose last sigma is
  extrapolated). The two get there differently.
  * ODP: its process variance is the scale times the mean, linear in it,
    so four independent quarters at the annual scale would add up to the
    annual variance. The halving is the scale's. The split leaves the
    Pearson chi-square unchanged: a quarter's fitted increment is a
    quarter of the annual cell's, so its residual is half the annual one
    and four of them sum to its square (the only zero residuals are the
    corner cells, 8 against 2 annually). The quarterly scale is therefore
    the annual one times the ratio of degrees of freedom, exactly: 36/171
    = 0.211 on RAA (207.08 against 983.64) and GenIns, 45/210 = 0.214 on
    ABC. The process SD ratio is its square root, about 0.46, which the
    measured totals match.
  * Mack: the split's quarterly link ratios deviate from their factors by
    about a quarter of the annual ones, so the sigmas squared are about a
    sixteenth of the annual, and each quarter's variance with them; four
    independent quarters add to a quarter, half the SD. The first year's
    three quarterly links (2, 3/2 and 4/3 for every origin) have sigma
    zero and give no residuals: put in the pool as zeros, as they were at
    first, they cut its mean square to 0.854 (RAA 30 of 206) and the
    parameter error with it, though the measured ratios moved little
    (before: 0.49 to 0.70, 0.53 to 0.55).
  A split is smoother than real quarterly data; this checks the
  mechanics.
* At a quarterly grain Mack's process draws each cell of the year from
  the drawn one before, so a Gamma draw can come out near zero (around
  1e-200) and the next one's shape `m^2 / variance` underflow to zero.
  That panicked on quarterly RAA at 2,000 simulations (seed 3, a unit
  test now; a review saw 3 of seeds 0 to 9); the draw from a Gamma or
  lognormal out of range is now its limit, zero. An annual triangle draws each origin once, from its
  observed value, and cannot reach it.
* An exact pattern stays exact when split, so both models give a zero
  CDR; Mack's needs every sigma estimable (a lone link ratio's sigma
  cannot be interpolated from sigmas that are all zero: an error).
* On RAA, 1985 cut back a year to 60 months has a one-year SD 1.98 times
  (ODP) and 1.38 times (Mack) its SD on the full triangle, at 2,000
  simulations: its year reveals two years of development.
* Annual triangles draw bit for bit as before the change (297 hashed
  configurations).

# Published numbers

* England, Verrall and Wuthrich (2019) Table 2 (analytic: reserves, Mack
  RMSEP and Merz-Wuthrich RMSEP on Taylor-Ashe, Mack's rule for the last
  sigma, total 1,778,968) equals act-reserving and R to the unit; the ODP
  validation test checks it. Their Table 4 is checked against
  `MackBootstrap` (above). They say simulation studies of the ODP
  partition the lifetime risk into uncorrelated one-year views too, but
  publish no ODP numbers. Their Appendix 1 allows a Gamma or lognormal
  process, or resampling the residuals again.[^evw]
* Boumezoued et al. (2011) re-reserve with a Mack-type bootstrap on the
  Merz-Wuthrich 2008 triangle: 81,074 simulated against 81,081
  analytic.[^boumezoued]
* No published one-year standard deviation of the ODP bootstrap on RAA,
  GenIns or ABC was found.

# Other measurements

* The ODP CDR's mean is slightly negative, -6.4% (RAA), -5.0% (GenIns)
  and -0.4% (ABC) of its SD: the resampled factors' bias, which also puts
  R's `BootChainLadder` mean reserve above the chain ladder's.

See [R ChainLadder CDR](/references/r-chainladder-cdr.md).

[^test]: validation/tests/reserving_one_year_bootstrap.rs
[^mack]: validation/tests/reserving_one_year_mack.rs
[^unit]: crates/act-reserving/src/one_year_bootstrap.rs, unit tests
[^evw]: England, Verrall and Wuthrich (2019), Tables 2 and 4, Section 2.2, Section 7 and Appendix 1
[^boumezoued]: Boumezoued et al. (2011), Table 2
