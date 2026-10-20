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
so the last digits are this platform's. The D values are recorded, not
pinned.

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
  triangles were skipped):

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
  | BayesOdp paid, here | 53.8* (14) | 57.8* (19) | 36.5* (40) | 27.2 (14) | 38.5* (87) |

  Sixteen of Meyers' 20 cells have his verdict. The four that do not:
  CRC incurred other liability (10.4 here, 20.7* his), CAY incurred other
  liability (10.6, 20.8*) and all 200 (7.0, 10.8*), all more calibrated
  here than his, and CSR paid personal auto (25.7* here, 18.5 his). The
  shape agrees with his conclusions: paid CRC fails in every line, as the
  paid Mack and ODP bootstrap do; incurred CRC passes except in total;
  CSR on paid passes in every line but personal auto; CAY on incurred
  passes in every line and in total. Whether the four differences are the
  models' or the panel's is not tested: the panel misses 26 of Table A.2's
  printed codes (6 in personal auto, 9 in other liability), the
  differences sit in exactly those two lines for three of the four, and a
  swapped triangle moves a line's D by up to 2 points.
* **Skipped triangles.** `Meyers` errors when a cell is not positive (the
  log-normal model, as Meyers' code floors at 1 and the library does not):
  7 triangles for CRC paid and CSR paid (other liability 5, commercial
  auto 2), none for incurred. `BayesOdp` errors when a development age's
  observed increments sum to zero or less: 113 of 200 triangles (commercial
  auto 36, personal auto 31, other liability 36, workers' compensation 10),
  mostly the late ages (84 to 120 months). So its D rests on 87
  triangles, and the common set of all arms is 85, whose D values are in
  `ks.csv` (set `common`) and are dominated by `BayesOdp`'s selection. A run
  that floors the cells (the second run of the note's decision 5) is not
  built.
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
  at most 2.7. Leaving the still-failing fits out moves the D of an arm by
  at most 1.7 on all 200 (CRC paid, 26.4 to 28.1 on 193) and none changes a
  verdict in total; by line, other liability paid under CRC goes from 21.6*
  to 26.9* and under CSR from 12.8 to 10.6. Worst kept fits: max R-hat
  1.0082 (CRC paid), 1.0062, 1.0118 (CAY) and 1.0074 (`BayesOdp`); CSR
  paid has one pathological fit, commercial auto group 32875 (R-hat
  1.168, bulk ESS 17, 444 divergences), whose percentile is kept; the
  other kept fits have bulk ESS of at least 614.
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
  incurred 73.69 (73.24), `BayesOdp` paid 69.75 (the ODP bootstrap's 74).
  It runs with `--features slow-bayes` in release and is ignored without
  it.

# Not yet shown

Compartmental (PR 4), pinning the D values (after the user's review of the
run), and a run on the printed codes only, which would separate panel from
model for the four mismatching cells.
