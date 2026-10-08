---
type: Finding
title: The simulated one-year view against Merz-Wuthrich, ODP and Mack's process
description: Re-reserving on the ODP bootstrap gives a one-year CDR standard deviation 0.50 to 5.96 times Merz-Wuthrich's per origin; the same re-reserving under Mack's process (MackBootstrap) reconciles its standard deviation with Merz-Wuthrich within Monte Carlo error on RAA, GenIns and ABC, so the gap is the ODP's process model, not the re-reserving. EVW's uncentred residuals bias its mean CDR (-0.21 to +0.18 SD), and its SD too, by up to 1.3% on RAA's young origins at 200,000 simulations, which an exact formula for the bootstrap's moments explains (Merz-Wuthrich's linear approximation is worth at most 0.09%); centring the pool removes both, so MackBootstrap centres by default. At a quarterly grain split from an annual triangle both models give about half the annual SD, the ODP through its scale, which falls with the degrees of freedom (36/171 on RAA), Mack through sigmas a quarter the size.
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
    resource: ../crates/prospicio-reserving/src/one_year_bootstrap.rs
    title: Unit tests mack_bootstrap_rereserving_reproduces_merz_wuthrich and one_cell_left_is_the_lifetime_run_off
  - id: evw
    resource: https://openaccess.city.ac.uk/id/eprint/21270/
    title: England, Verrall and Wuthrich (2019), On the lifetime and one-year views of reserve risk, with application to IFRS 17 and Solvency II risk margins, Insurance - Mathematics and Economics 85
  - id: mw2008
    resource: https://www.casact.org/pubs/forum/08fforum/21merz_wuetrich.pdf
    title: Merz and Wuthrich (2008), Modelling the claims development result for solvency purposes, CAS E-Forum Fall 2008, Appendix A
  - id: boumezoued
    resource: https://arxiv.org/abs/1107.0164
    title: Boumezoued, Angoua, Devineau and Boisseau (2011), One-year reserve risk including a tail factor - closed formula and bootstrap approaches
---

# Finding

The volume-weighted chain ladder without a tail, re-reserved one year
ahead, 20,000 simulations from seed 20,261,006, Gamma process, against R
ChainLadder's `CDR(MackChainLadder(tri))` `CDR(1)S.E.` (log-linear last
sigma), measured with the Marsaglia-Tsang Gamma sampler (since
2026-10-08; by inverse transform the ODP totals were 0.609, 1.362 and
1.133 times Merz-Wuthrich's, per origin 0.50 to 5.98):[^test][^mack]

| Dataset | Merz-Wuthrich total | ODP total SD | Mack's process total SD | ODP / MW total | Mack / MW total | ODP / MW per origin | Mack / MW per origin |
|---|---|---|---|---|---|---|---|
| RAA | 25,166 | 15,478 | 25,453 | 0.615 | 1.011 | 0.50 (1990) to 5.06 (1982) | 0.991 to 1.014 |
| GenIns | 1,774,014 | 2,436,241 | 1,756,199 | 1.373 | 0.990 | 0.72 (2006) to 2.27 (2004) | 0.990 to 1.009 |
| ABC | 117,161 | 133,044 | 117,567 | 1.136 | 1.003 | 0.84 (1983) to 5.96 (1979) | 0.993 to 1.009 |

Mack's process, `MackBootstrap::one_year` (England, Verrall and Wuthrich
2019, Appendix 1), is within five Monte Carlo standard errors of every
origin and total there, and with Mack's rule for the last sigma too; the
validation test checks all of them in CI. That is the reconciliation, of
the standard deviation: replace the ODP's process with Mack's and the same
re-reserving gives Merz-Wuthrich, so the ODP's gap is its process model.
The mean is another matter (below).[^mack]

The ODP's ratios are a seed-pinned regression in its validation test, not
a check against Merz-Wuthrich, and so are the R test (RAA total) and the
Python test (GenIns total). Its one-year total is 0.82 (RAA), 0.81
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
(-24%), and 1990's 7,159 instead of 11,840 (-40%), both by the
inverse-transform Gamma sampler (before 2026-10-08). It also broke the
identity below, by 1.5% on RAA, which a 2% allowance in the test had
absorbed.

* An origin with one cell left has an ODP one-year view distributed as
  its lifetime reserve. Its SD is 1.017 (RAA 1982), 1.013 (GenIns 2002) and
  1.001 (ABC 1978) times `BootChainLadder`'s, within the Monte Carlo
  tolerance; a unit test also checks it against the crate's own lifetime
  bootstrap.[^unit]

# Mack's process in detail

* Uncentred (EVW's Appendix 1 as written, `centre_residuals: false`), with
  200,000 simulations GenIns and ABC are within 0.4% of R per origin and
  in total, but RAA's three youngest origins (1988 to 1990) and its total
  come out 0.4% to 1.3% above R, 2.2 to 7.0 Monte Carlo standard
  errors (three runs). The uncentred residuals cause it, not
  Merz-Wuthrich's linear approximation: see the next section. At 20,000
  simulations five standard errors are 2.5% to 5%, so the CI test does
  not resolve it.
* The uncentred bootstrap's draws equal the hand-written harness of EVW's
  Appendix 1 in the unit test bit for bit (GenIns, the first 300 of the
  harness's 20,000 simulations, each on its own stream), so with
  `centre_residuals: false` the type is that harness; the default,
  centred, draws other values from the same random numbers.[^unit]
* EVW's Table 4 (500,000 simulations, Mack's rule for the last sigma, total
  one-year SD 1,778,428) against `MackBootstrap` with 20,000 (seed
  20,261,006, the Marsaglia-Tsang Gamma sampler): centred (the default),
  total 1,761,459 (-0.95%), 2002 75,547 against 75,502, every origin and
  the total within 1.9 combined standard errors (2008 -1.83, 2005 +1.70,
  the total -1.89); uncentred, total 1,763,188 (-0.86%), within 1.8. By
  the inverse-transform sampler (before 2026-10-08) the same seed gave
  1,778,254 (-0.01%) centred and 1,779,997 (+0.09%) uncentred: both
  samplers are within Monte Carlo error of EVW, and the earlier near-exact
  total was not a closer match. GenIns's pool mean is small, so the
  one-year SD cannot tell centred and uncentred apart.[^mack][^evw]
* The process shape does not move the SD: Gamma and normal agreed to 0.1%
  to 0.7% per origin on the same uniforms, when the Gamma was drawn by
  inverse transform (before 2026-10-08; they now share only the pseudo
  factors); Gamma, lognormal, residuals and normal all give GenIns's
  total within Monte Carlo error. RAA's normal
  process sends 1990's next value below zero in some simulations; the
  Gamma and lognormal do so only when the pseudo first factor is negative.
* The mean CDR under Mack's process, with EVW's uncentred residuals, is
  -0.214 (RAA), -0.038 (GenIns) and +0.176 (ABC) times its SD (seed
  20,261,006, 20,000 simulations, log-linear last sigma, the
  inverse-transform Gamma sampler; pinned in the
  validation test until centring became the default). Merz-Wuthrich's CDR
  has mean zero. The cause: each
  factor's residuals have a zero `C^(alpha / 2)`-weighted sum, not a zero
  mean, so the pool's mean is 0.1395 (RAA), 0.0135 (GenIns) and -0.0595
  (ABC) with mean square exactly 1, and `E[f*_k] = f_k + m sigma_k
  sum(C^(alpha / 2)) / sum(C^alpha)`. EVW's Appendix 1 does not centre
  either, but their Table 4 lifetime expected reserve on Taylor-Ashe
  (GenIns) is the chain ladder's within Monte Carlo error (+0.02%), where
  the uncentred bootstrap gives +0.7%: their numbers agree with centred
  residuals ([the lifetime finding](/findings/mack-bootstrap-lifetime-vs-mack.md)). The `Residuals`
  process draws from the same pool, so uncentred it adds a bias of its
  own (mean `f* C + m sd`, variance `(1 - m^2) sd^2`).[^mack][^evw]
* `MackBootstrap::centre_residuals` subtracts the pool's mean before
  resampling. It is on by default (since branch
  claude/mack-centre-default): the mean CDR is then Merz-Wuthrich's zero,
  the lifetime mean reserve the chain ladder's, the SD Merz-Wuthrich's
  without RAA's 1.3% excess, and EVW's Table 4 lifetime expected reserves
  agree; `false` keeps EVW's Appendix 1 as written. Centred, at the same
  seed and 20,000 simulations, the total mean CDR is -0.0022 (RAA),
  +0.0031 (GenIns) and -0.0095 (ABC) times its SD, within 1.4 Monte Carlo
  standard errors of zero (pinned in the validation test, which checks
  them against the exact zero; -0.0108, -0.0035 and +0.0067 with the
  inverse-transform Gamma sampler before 2026-10-08), and the SD
  reconciles (measured with that sampler): total 1.003, 1.000
  and 0.994 times R's `CDR(1)S.E.`, per origin 0.989 to 1.005, 0.994 to
  1.006 and 0.987 to 1.007 (measured once), every origin within five Monte
  Carlo standard errors (the CI test). A unit test checks on RAA (4,000
  simulations) that the uncentred mean is far below zero and the centred
  Gamma and `Residuals` means are within four standard errors of it.[^unit]
  A review's independent R harness of EVW's Appendix 1 found the same
  pool means and, centred, RAA -0.013 and ABC +0.001 times the SD.

# RAA's gap at 200,000 simulations

The bootstrap's one-year SD has a closed form, and with it the gap is the
uncentred residuals' bias of the pseudo factors.[^mack]

* The exact moments. On an annual triangle, volume-weighted, no tail,
  each open origin's next value `Z_i = f*_k C_i + process` (`k` its
  latest age) is independent of the others: each origin uses its own
  pseudo factor, from its own resampled residuals, and its own process
  draw. The refitted factor at `k` is `(A_k + Z_i) / (S_k + C_i)`, `A_k`
  and `S_k` the older origins' sums at `k + 1` and `k`. So origin `l`'s
  closing ultimate, `Z_l` times the refitted factors of the older
  origins, is a product of independent factors each linear in one `Z`,
  and every covariance of the CDR is a product of means: it needs each
  `Z`'s mean `E[f*_k] C_i` and variance `Var[f*_k] C_i^2 + sigma_k^2 C_i`,
  nothing else. `exact_covariance` in the validation test computes it.
* (a) Merz-Wuthrich's approximation is not it. With their conditional
  resampling (`E[f*_k] = f_k`, `Var[f*_k] = sigma_k^2 / S_k`), the
  first-order (delta method) covariance equals R's `CDR(1)S.E.` to 1e-9
  relative, per origin and in total, on RAA, GenIns and ABC, which checks
  the model; the exact one is above it by at most 0.084% (RAA 1990,
  23,630.22 against 23,610.35; total 25,185.83 against 25,166.30, +0.078%),
  0.037% on GenIns and 0.0013% on ABC. Merz and Wuthrich (2008),
  Appendix A, (A.1), replace the product terms `prod(1 + a_j) - 1` by
  `sum(a_j)`, a lower bound, which is what the exact one undoes. A fast
  test pins it.[^mw2008]
* (b) The pseudo factors are. Resampling an uncentred pool with mean `m`
  and mean square 1 gives `E[f*_k] = f_k + m sigma_k sum(sqrt(C)) / S_k`
  and `Var[f*_k] = (1 - m^2) sigma_k^2 / S_k`. On RAA (`m = 0.1395`) the
  first factor is 14.3% high, the next two 3.3% and 2.6%, and 1990's
  closing ultimate's mean 15.7% above its opening. Higher means of the
  next values and refitted factors widen their products, and the exact SD
  becomes 0.48% (1988), 0.93% (1989), 1.20% (1990) and 1.29% (total)
  above R; older origins, where the smaller pool variance `1 - m^2`
  dominates, are 0.46% (1982) to 0.02% (1985) below. GenIns (`m =
  0.0135`) moves by at most +0.13%, ABC (`m = -0.0595`) by at most -0.13%.
  Centred (`E[f*_k] = f_k`, the same variance `(1 - m^2) sigma_k^2 /
  S_k`: centring shifts the pool without rescaling it), RAA's 1988 to
  1990 and total are 0.11% to 0.00% below R, and every origin of the
  three triangles is between 0.46% below (RAA 1982) and 0.04% above
  (GenIns 2010) R. Merz-Wuthrich's exact values are never below R, so
  the centred bootstrap is not their model either. The exact uncentred
  mean CDR is -0.204, -0.034 and +0.168 times its SD, and zero centred
  (the 20,000-simulation runs gave -0.214, -0.038, +0.176 uncentred, within
  1.5 of their Monte Carlo standard errors, about `1 / sqrt(20,000)`, and
  give -0.0022, +0.0031, -0.0095 centred with the Marsaglia-Tsang Gamma
  sampler, which the regression pins). A
  fast test pins these numbers and checks that, under Merz-Wuthrich's
  factors, the exact mean closing ultimates are the opening chain-ladder
  ones.
* The simulations agree with the exact values. At 200,000 simulations,
  seed 20,261,006 (Gamma, normal and centred Gamma) and seed 7
  (Gamma), every origin and total of the three triangles is within 2.71
  Monte Carlo standard errors of its exact SD (RAA uncentred within 1.35),
  while RAA's uncentred runs are up to 7.0 standard errors from R. An
  ignored test (`simulation_matches_the_exact_moments`, about four minutes
  with `--release` by inverse transform; 1.4 s with `--release` and 19 s
  in a debug build with the Marsaglia-Tsang sampler, and it still passes)
  checks it at four standard errors. Against Merz-
  Wuthrich's exact values instead, the centred run's RAA 1982 would be
  -3.41 standard errors: the centred pool keeps the variance `1 - m^2`.
* (c) The process shape cannot move the SD: the covariances need only the
  first two moments of each `Z`, and Gamma, lognormal and normal share
  them. Gamma and normal on the same seed are both within 1.11 standard
  errors of the exact SD on every RAA origin and the total.
* (d) The Monte Carlo standard error, `sd sqrt((kurtosis - 1) / (4 n))`,
  is not understated, but the evidence is thinner than the 124 z-scores
  against the exact SDs suggest. Only two runs per triangle are
  independent, Gamma uncentred on seeds 20,261,006 and 7: the normal and
  centred runs on seed 20,261,006 share its random numbers, and their
  z-scores repeat the Gamma run's (to 0.2 centred and 0.3 normal, except
  RAA's normal 1990 and total, 1.2 and 1.0). Over the two independent runs the 62
  z-scores have a root mean square of 0.85 and a maximum of 2.71 (ABC
  1986), and the origins and total of a run are correlated. That rules
  out an understated standard error, not a somewhat overstated one.

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
  sigma, total 1,778,968) equals prospicio-reserving and R to the unit; the ODP
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
[^unit]: crates/prospicio-reserving/src/one_year_bootstrap.rs, unit tests
[^evw]: England, Verrall and Wuthrich (2019), Tables 2 and 4, Section 2.2, Section 7 and Appendix 1
[^boumezoued]: Boumezoued et al. (2011), Table 2
[^mw2008]: Merz and Wuthrich (2008), Appendix A, (A.1)
