---
type: Finding
title: On the rebuilt panel, Meyers' CRC, CSR and CAY reach his conclusions on 16 of 20 cells, and BayesOdp tracks the ODP bootstrap
description: Meyers' CRC on paid and incurred, CSR on paid and CAY on incurred, fitted with his priors and the library's NUTS defaults to the 200-triangle rebuilt panel, give the same pass or fail at 5% as his published D in 16 of the 20 cells (4 lines and all 200, four arm and basis pairs). The four that differ are other liability incurred under CRC and CAY, personal auto paid under CSR and all 200 under CAY, each more calibrated here than his, except CSR personal auto. BayesOdp on paid scores only 87 of 200 triangles (113 have a non-positive column sum) and its percentiles match the ODP bootstrap's triangle by triangle.
tags: [reserving, meyers, calibration, crc, csr, cay, bayes-odp, nuts, schedule-p, ks-test]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-20T20:00:00-07:00 }
sources:
  - id: harness
    resource: ../validation/tests/reserving_calibration_meyers.rs
    title: calibration_study (ignored; run by hand), the Bayesian arms and their wiring test on group 353
  - id: panel
    resource: meyers-calibration-panel.md
    title: The panel, its differences from Table A.2, and Mack and the ODP bootstrap on it
  - id: note
    resource: ../docs/design/reserving-calibration.md
    title: Design note, decisions 1, 4, 5, 7 and 8, and "As built"
  - id: meyers-code
    resource: /references/meyers-monograph-8.md
    title: Meyers' CRC.R, CSR.R and CAY.R, their floor, ceiling and sampler settings
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), CAS Monograph 8, Figures 5.1, 5.2, 7.4 and 8.5
---

# Facts

Run on 2026-10-20 (Windows, 16 threads, release build) on the panel with
sha256 `075b3925...59d2` (the one of the Mack and ODP finding), seeds from
`seed_for`, `RISK_RS_CALIBRATION_ARMS=mack,odp,bayes_odp,crc,csr,cay`. Not
marked verified: no CI job runs the study, only the wiring test on group
353 (below). NUTS draws at a fixed seed differ between Windows and Linux,
so the last digits are this platform's. The D values are pinned by the
study since 2026-10-21 (below), and still not verified.

* **Settings.** `Meyers` and `BayesOdp` at their defaults: Meyers' priors
  (`logelr ~ N(-0.4, 3.162^2)`, origin and development effects
  `N(0, 3.162^2)`, `gamma ~ N(0, 0.05^2)`, `(1 - rho) / 2 ~ Beta(2, 2)`),
  four chains with 1,000 warm-up and 2,000 kept draws for Meyers' models
  and 1,000 kept draws for `BayesOdp`, target acceptance 0.95, 10,000
  predictive simulations, the premium of the panel (`earned_prem_net`). A
  fit with a divergence or R-hat above 1.05 is refitted once at target
  acceptance 0.99 and the refit is scored (both results are in
  `percentiles.csv`).
* **Bases** as Meyers' (Figures 5.1, 5.2, 7.4, 8.5): CRC on paid and on
  case incurred, CSR on paid, CAY on incurred; `BayesOdp` on paid, which
  Meyers has no counterpart for.
* **D against Meyers'** (percentage points; `*` fails at 5%, the bound
  `136 / sqrt(n)` is 19.2 for n = 50 and 9.6 for n = 200, larger when
  triangles were skipped; n in brackets when below 50 or 200, Meyers'
  own n is 50 and 200 throughout). The `BayesOdp` row rests on the 87
  triangles it can score (below) and compares only with the ODP
  bootstrap on the same 87, not with the other rows:

  | | CA | PA | WC | OL | All |
  |---|---|---|---|---|---|
  | CRC paid, here | 20.7* (48) | 43.0* | 32.8* | 21.6* (45) | 26.4* (193) |
  | CRC paid, Meyers | 22.8* | 41.7* | 30.5* | 25.3* | 25.5* |
  | CRC incurred, here | 8.6 | 11.8 | 17.3 | 10.4 | 10.3* |
  | CRC incurred, Meyers | 10.2 | 16.7 | 14.4 | 20.7* | 11.9* |
  | CSR paid, here | 5.5 (48) | 25.7* | 10.9 | 12.8 (45) | 4.6 (193) |
  | CSR paid, Meyers | 5.9 | 18.5 | 12 | 10.4 | 3.1 |
  | CAY incurred, here | 12.2 | 10.1 | 15.0 | 10.6 | 7.0 |
  | CAY incurred, Meyers | 8.3 | 15.9 | 12.5 | 20.8* | 10.8* |
  | BayesOdp paid, its own 87 only | 53.8* (14) | 57.8* (19) | 36.5* (40) | 27.2 (14) | 38.5* (87) |

  Sixteen of Meyers' 20 cells have his verdict. The four that do not:
  CRC incurred other liability (10.4 here, 20.7* his), CAY incurred other
  liability (10.6, 20.8*) and all 200 (7.0, 10.8*), all more calibrated
  here than his, and CSR paid personal auto (25.7* here, 18.5 his). The
  shape agrees with his conclusions: paid CRC fails in every line, while
  paid Mack and the ODP bootstrap pass in other liability (D 10.3 and 8.7
  in the panel's finding, his 8.5 and 6.8), so paid CRC's overestimate is
  a problem of its own, as Meyers also treats it; incurred CRC passes except
  in total; CSR on paid passes in every line but personal auto; CAY on
  incurred passes in every line and in total. What causes the four
  differences is not tested. Three sources are open: the panel (it misses
  26 of Table A.2's printed codes, 6 in personal auto and 9 in other
  liability, the differences sit in exactly those two lines for three of
  the four, and a swapped triangle moves a line's D by up to 2 points),
  the differences from Meyers' code (next bullets), and NUTS noise (the
  bullet on diagnostics).
* **Near the bound.** Four cells are within 2 points of their bound:
  CRC paid commercial auto (20.7 against 19.6) and other liability (21.6
  against 20.3), CRC incurred workers' compensation (17.3 against 19.2)
  and all 200 (10.3 against 9.6). All four have Meyers' verdict, but
  another platform's draws or the code differences below could flip any
  of them, so they should not be pinned as exact verdicts.
* **Differences from Meyers' code** (`knowledge/references/meyers-monograph-8.md`),
  each a like-for-like caveat on every comparison above:
  * *Floor.* His code raises every cell below 1 to 1 (`pmax(loss, 1)`)
    before the log; `Meyers` errors on a non-positive cell and the study
    skips the triangle, and cells between 0 and 1 are used as they are.
    The 7 skipped paid triangles (other liability 5, commercial auto 2)
    are where his floor acts, so paid D in those lines is on 45 and 48
    triangles against his 50: other liability paid (CRC 21.6* against
    his 25.3*, CSR 12.8 against 10.4) is not computed on his set. The
    floor cannot explain the incurred mismatches, which skip nothing.
  * *Ceiling.* His prediction takes `ceiling()` of each simulated
    ultimate, about +0.5 per origin and +4.5 on the total; the library
    does not. Against group 353's total sd of about 1,400 this is
    nothing, but the panel has a few tiny triangles, all in other
    liability: per arm 1 to 3 with a total sd below 45 (group 16373's is
    5.4 under CRC incurred, so +4.5 is 0.8 sd there). The ceiling can
    move those triangles' percentiles, and through them other
    liability's D, which holds two of the four mismatches. By how much is
    not tested: the study keeps no draws.
  * *Sampler.* His runs use target acceptance 0.9999, maximum tree depth
    50, 2,500 warm-up and 2,500 kept draws per chain, and thinning until
    R-hat is below 1.05; the study uses 0.95 (0.99 on a refit), the
    library's depth of 10, and 1,000 warm-up and 2,000 kept draws. These
    change how well the posterior is sampled, not the posterior, so they
    matter through the fits that still fail diagnostics (below).
* **Skipped triangles.** `Meyers` errors when a cell is not positive (the
  log-normal model, as Meyers' code floors at 1 and the library does not):
  7 triangles for CRC paid and CSR paid (other liability 5, commercial
  auto 2), none for incurred. `BayesOdp` errors when a development age's
  observed increments sum to zero or less: 113 of 200 triangles (commercial
  auto 36, personal auto 31, other liability 36, workers' compensation 10),
  mostly the late ages (84 to 120 months), that is small or
  fast-settling triangles, so the 87 are not a random sample of Meyers'
  200. Its D rests on those 87, and the common set of all arms is 85,
  whose D values are in `ks.csv` (set `common`); that set is almost
  entirely `BayesOdp`'s selection, so its rows are not a comparison of
  the arms. A run that floors the cells (the second run of the note's
  decision 5) is not built: decision 5 calls for it only when skips are
  many, the Meyers arms skip 7 of 200, and a floor on cells does nothing
  for `BayesOdp`'s skips, which come from column sums.
* **`BayesOdp` against the ODP bootstrap.** On the 87 triangles both score,
  the ODP bootstrap's percentile differs from `BayesOdp`'s by 0.5 points at
  the median (mean difference -0.9), and the ODP bootstrap's D on that set
  is 37.4 against 38.5. Both pile their percentiles low (25 of 87 below
  1): the same paid-bias Meyers finds for the ODP bootstrap and Mack.
* **NUTS diagnostics** (`percentiles.csv`, `diagnostics` and `refit`).
  First fits that failed (a divergence in every case; none had R-hat above
  1.05): CRC paid 41 of 193, CRC incurred 42 of 200, CSR paid 42 of 193,
  CAY incurred 40 of 200, `BayesOdp` 1 of 87. After the refit at 0.99 still
  failing (mostly 1 to 12 divergences among 8,000 draws): 9, 10, 10, 7 and
  0. The refit moves a percentile by 0.4 to 0.6 points at the median and
  at most 2.7. The still-failing fits are scored in the table above (the
  refit rule does not depend on the outcome, so they add noise, not
  selection). D with them left out (n in brackets where it drops):

  | | CA | PA | WC | OL | All |
  |---|---|---|---|---|---|
  | CRC paid | 21.0* (47) | 44.2* (49) | 32.8* | 26.9* (38) | 28.1* (184) |
  | CRC incurred | 9.5 (47) | 11.8 | 17.3 | 10.0 (43) | 10.8* (190) |
  | CSR paid | 5.5 (46) | 24.8* (49) | 10.9 | 10.6 (38) | 5.1 (183) |
  | CAY incurred | 11.2 (49) | 10.1 | 15.0 | 10.6 (44) | 7.0 (193) |

  No verdict changes; the largest moves are other liability paid under
  CRC (21.6* to 26.9*) and under CSR (12.8 to 10.6). Worst kept fits: max R-hat
  1.0082 (CRC paid), 1.0062, 1.0118 (CAY) and 1.0074 (`BayesOdp`); CSR
  paid has one pathological fit, commercial auto group 32875 (R-hat
  1.168, bulk ESS 17, 444 divergences), whose percentile is kept; the
  other kept fits have bulk ESS of at least 614.
* **Pinned (PR 5, 2026-10-21).** The user reviewed the KS scores and
  approved them; the study asserts every D above within a tolerance. A
  rerun on the default seed reproduced them to their rounding, and two
  salted runs measured how far the seed alone moves D, in points (largest
  difference between two of three runs, per line and on all 200): CRC
  1.73 and 0.81, CSR 1.48 and 1.21, CAY 1.23 and 0.62, `BayesOdp` 1.01 and
  0.19. The tolerances are twice those (CRC 3.5 and 1.7, CSR 3.0 and 2.5,
  CAY 2.5 and 1.3, `BayesOdp` 2.1 and 0.5; 1.0 and 0.5 floors). No
  verdict changed in either salted run. The study does not pin the verdict
  of CRC paid commercial auto and other liability, CRC incurred workers'
  compensation and all 200 (a D within the tolerance of its bound): the
  cells "Near the bound" above named. Not
  marked verified: no CI job runs the study, and the pins were recorded on
  Windows.
* **Cost.** The full study (1,600 fits: Mack, ODP bootstrap and the five
  Bayesian arm and basis pairs, 200 triangles) takes 230 s wall on 16
  threads. Serial (`RAYON_NUM_THREADS=1`, 3 triangles per line), the mean
  fit including refits and errors is 0.11 s for `BayesOdp` (successful fits
  about 0.3 s), 0.93 s for CRC paid, 1.12 s CRC incurred, 1.45 s CSR paid
  and 1.27 s CAY incurred. The slowest single fit is 56 s (a CRC paid fit
  that was refitted, under parallel load). The note's estimate of 5 to 13
  minutes for Meyers' four arms is an upper bound on a 16-thread machine.
* **Wiring.** The test `the_bayesian_arms_reproduce_meyers_on_group_353`
  runs the study's code path on Meyers' own triangle: CRC paid 51.42
  (Meyers 51.88), CRC incurred 75.04 (74.75), CSR paid 86.85 (86.26), CAY
  incurred 73.69 (73.24), `BayesOdp` paid 69.75 against the ODP
  bootstrap's 74.06 in the same test (asserted within 6 points). The gap
  is in the means: `BayesOdp`'s predictive mean is 39,376 against the
  bootstrap's 39,197 (sd 1,399 and 1,381), so the outcome 40,000 is 0.45
  sd above it against 0.58. The file's `[[test]]` entry requires
  `slow-bayes`, so CI's `bayes` job runs this test (about 2 s in release)
  and the file's unit tests, and never the study.

# Not yet shown

Compartmental is in `meyers-calibration-compartmental.md` (PR 4).
A run on the printed codes only, which would separate panel from
model for the four mismatching cells, and a run with Meyers' floor and
ceiling, which would separate his code's data handling from the model.
