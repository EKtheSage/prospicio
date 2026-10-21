---
type: Finding
title: On the rebuilt panel, Compartmental model 1 fails Meyers' percentile test in every line, with predictive intervals far too narrow; model 2 scores only 48 of 200 triangles
description: Gesmann and Morris' compartmental model 1, at the paper's preset priors (fitted to workers' compensation company 337) and the library's sampler, scores all 200 triangles of the rebuilt panel on paid and fails Meyers' uniformity test at 5% in every line and in total (D 49.3 commercial auto, 36.9 personal auto, 26.8 workers' compensation, 66.3 other liability, 39.3 on all 200, against bounds of 19.2 and 9.6), with a larger total D (on all 200) than Mack, the ODP bootstrap, CRC and CSR on paid (on 193 to 200). Its predictive total has a median coefficient of variation of 1.3% to 3.8% in three lines, and other liability sits near the top of the predictive distribution. Model 2, whose log-normal likelihood needs every observed amount positive, scores 48 of 200 triangles (27 in workers' compensation) and fails in total (20.5 against 19.6) and in workers' compensation.
tags: [reserving, compartmental, gesmann-morris, calibration, schedule-p, ks-test, nuts, preset-priors]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-20T10:00:00-07:00 }
sources:
  - id: harness
    resource: ../validation/tests/reserving_calibration_meyers.rs
    title: calibration_study (ignored; run by hand), the Compartmental arms and their wiring tests
  - id: panel
    resource: meyers-calibration-panel.md
    title: The panel, and Mack and the ODP bootstrap on it
  - id: bayes-arms
    resource: meyers-calibration-bayesian-arms.md
    title: BayesOdp, CRC, CSR and CAY on the same panel
  - id: note
    resource: ../docs/design/reserving-calibration.md
    title: Design note, decisions 1, 4, 5 and 6, and "As built (PR 4)"
  - id: gm
    resource: compartmental-gm1-parity.md
    title: The model 1 preset and its parity on company 337
---

# Facts

Run on 2026-10-20 (Windows, 16 threads, release build) on the panel with
sha256 `075b3925...59d2` (the one of the Mack and ODP finding), seeds from
`seed_for`. Not marked verified: no CI job runs the study, only two wiring
tests (below). NUTS draws at a fixed seed differ between Windows and
Linux, so the last digits are this platform's. The D values are pinned by
the study since 2026-10-20 (below), and still not verified. Both models ran on all 200 triangles in one run,
`RISK_RS_CALIBRATION_ARMS=compartmental,compartmental_2`, with the panel
copied to `target/calibration-pr4/` (`RISK_RS_CALIBRATION_PANEL`), so its
`percentiles.csv` (sha256 `62822bcf...fadb7`) and `ks.csv` sit beside it
and no later probe overwrites them. `target/` is not committed: every
figure below is reproduced by that command (about 33 minutes on 16
threads). The D values come from `ks.csv`; the medians, tail counts,
coefficients of variation (`sd / mean`), mean over outcome, diagnostics
and errors are computed from `percentiles.csv`'s `percentile`, `mean`,
`sd`, `outcome`, `diagnostics`, `refit` and `error` columns. A first run
of each model on the same seeds gave the same figures.

* **Settings.** `Compartmental::new(CompartmentalPriors::gesmann_morris())`
  (model 1: normal likelihood on amounts, cumulative paid, paid and
  outstanding jointly, target acceptance 0.99, four chains of 1,000 warm-up
  and 2,000 kept draws) and `Compartmental::model_2(
  CompartmentalPriors::gesmann_morris_2())` (log-normal on loss ratios,
  incremental paid, target acceptance 0.999, four chains of 2,000 warm-up
  and 8,000 kept draws), 10,000 simulations, the panel's `premium`
  (`earned_prem_net`), outstanding as case incurred less paid
  (`Outstanding::FromIncurred`). The outcome is the age-10 total of
  cumulative paid, scored like the ODP bootstrap: the share of simulated
  reserve totals at or below the outcome less what is paid. The oldest
  origin is already at age 10 and contributes no reserve. A fit with a
  divergence or R-hat above 1.05 is refitted once at a target acceptance a
  tenth of the way to 1 closer (0.999 for model 1, 0.9999 for model 2), not
  at the 0.99 of the other NUTS arms, since 0.99 is model 1's first fit.
  The paper's presets were fitted to workers' compensation company 337; on
  the other lines the arm tests the preset as much as the model.
* **Model 1, D** (percentage points; `*` fails at 5%; bound `136 / sqrt(n)`
  is 19.2 for n = 50 and 9.6 for n = 200). All 200 triangles score; there
  is no Meyers figure for it, so the columns below are the paid rows of the
  two earlier findings, all on Meyers' paid basis:

  | | CA | PA | WC | OL | All |
  |---|---|---|---|---|---|
  | Compartmental 1, here | 49.3* | 36.9* | 26.8* | 66.3* | 39.3* (200) |
  | Mack paid, here | 23.2* | 48.3* | 33.4* | 10.3 (49) | 24.6* (199) |
  | ODP paid, here | 25.6* | 48.8* | 31.3* | 8.7 | 26.3* |
  | CRC paid, here | 20.7* (48) | 43.0* | 32.8* | 21.6* (45) | 26.4* (193) |
  | CSR paid, here | 5.5 (48) | 25.7* | 10.9 | 12.8 (45) | 4.6 (193) |

  Each line has 50 triangles unless the cell gives n; Compartmental model
  1 scores all 200, the other arms a subset, so a comparison across rows
  is of D on slightly different sets (and bounds: 9.6 on 200, 9.8 on
  193). It fails in every line and in total. Its D is the largest of these
  arms in commercial auto, other liability and in total; in personal auto and
  workers' compensation its D is below those of Mack, the ODP bootstrap and
  CRC (which also fail there) and above CSR's. It is the only one of these
  arms whose other liability D is above 22.
* **Where the percentiles sit.** Median percentile 94.3 (CA), 17.2 (PA),
  62.9 (WC), 99.9 (OL), 89.2 (all). Below 5 / above 95: CA 1 / 25 of 50,
  PA 19 / 7, WC 7 / 15, OL 1 / 35, all 28 / 82 of 200; 50 percentiles are
  at least 99.9 and 6 at most 0.1. So the outcome lands in the tails far
  more than 10% of the time. The predictive total is narrow: the median
  coefficient of variation is 3.8% (CA), 1.3% (PA), 1.7% (WC) and 8.0%
  (OL). The median of predictive mean over outcome is 0.935 (CA), 1.009
  (PA), 0.994 (WC) and 0.839 (OL): in personal auto and workers'
  compensation the mean is right and the interval too narrow, in other
  liability the mean is low as well, in commercial auto both. This
  describes the fits; why the intervals are narrow is not tested.
* **Preset line against the others.** Workers' compensation, the line the
  preset was fitted to, has the lowest D (26.8*), against 36.9*, 49.3* and
  66.3*, but it still fails, a little below Mack, the ODP bootstrap and CRC on paid
  (31.3 to 33.4) and above CSR (10.9).
  The panel's workers' compensation group 337 (the code of the paper's
  company 337, whose triangle in the paper is not this square) scores 55.4 under model 1 and 99.7 under model 2 (its rows
  of the run's `percentiles.csv`).
  Whether the other lines' worse D is the preset, the line's data (case
  incurred can be below paid, and other liability triangles are small) or
  both is not separated: a run with priors fitted per line is not built.
* **Group 353.** On Meyers' group 353 commercial auto (the wiring test) the
  preset gives mean ultimate paid 35,803 (sd 1,212) against the ODP
  bootstrap's 39,197 (sd 1,381) and the outcome 40,000: percentile 100.00.
* **Model 2 scores 48 of 200.** The log-normal likelihood needs every
  observed response positive, and the library errors on the first
  violation: outstanding (case incurred less paid) at or below zero in 123
  triangles (other liability 42, commercial auto 39, personal auto 31,
  workers' compensation 11), an incremental paid at or below zero in 28
  (WC 12, PA 9, CA 5, OL 2), and one other error (OL). The 48 scored are 27
  workers' compensation, 10 personal auto, 6 commercial auto and 5 other
  liability, so they are not a sample of Meyers' 200. Model 1 errors on
  none. A floor on the cells (the second run of the note's decision 5)
  would put the other 152 in play and is not built. D on the 48: CA 27.3
  (n = 6), PA 38.2 (10), WC 29.4* (27, bound 26.2), OL 19.8 (5), all 20.5*
  (48, bound 19.6). Only workers' compensation and the total have enough
  triangles for a verdict, and the total is 0.9 above its bound, so it is
  not a clear failure.
* **Diagnostics.** Model 1: the first fit failed in 25 of 200 (a divergence
  in all 25, R-hat above 1.05 in one); all 25 were refitted at 0.999 (CA 7,
  PA 4, WC 12, OL 2); 3 still fail (commercial auto groups 44598, 2135 and
  31550: R-hat 1.1238, 1.0410 and 1.0319, 2 to 9 divergences; the worst
  has bulk ESS 24); over the kept fits the largest R-hat is 1.1238, there
  are 14 divergences in all and the smallest bulk and tail ESS are 24 and
  67. No fit flagged the rate-swap diagnostic. Model 2: no fit failed;
  largest R-hat 1.0161, no divergences, smallest bulk and tail ESS 264 and
  372.
* **Cost.** One model 1 fit alone takes 2.1 s (workers' compensation,
  first triangle) and one model 2 fit 59 s; the note's 5.5 s and 61 s were
  company 337's. The 200 model 1 fits take 528 s on 16 threads (about 2.6 s
  a fit, with refits); the 200 model 2 jobs take 1,329 s, of which 48 fit
  (the rest fail at once), the slowest single fit 1,126 s under load. Run
  alone, 48 model 2 fits would take about 48 minutes.
* **Pinned (PR 5, 2026-10-20).** The user reviewed the KS scores and
  approved them. A rerun on the default seed (PR 5, with model 2) gave the
  identical `ks.csv` and all 400 identical percentiles of this run, so the
  fits are deterministic on this platform. A run with another seed salt
  moved model 1's D by at most 1.32 points in a line (other liability; 0.45
  on all 200) between three seeds; model 2 was salted once, by at most 0.33
  points in a line (0.06 on all 48). The study asserts model 1's D within
  2.7 points per line and 0.9 on all 200, and model 2's within 2.0 and 1.0
  (a wider margin, from one salt on 5 to 27 triangles per line and fits
  that take minutes). Model 2 on all 48 (20.5 against a bound of 19.6) is
  within its tolerance of the bound, so its verdict is not pinned. Not
  marked verified: no CI job runs the study, and the pins were recorded on
  Windows. Model 2's tolerance is a floor above one salted difference, not
  a measured spread; the thread count changes no draw (model 1 at 3
  threads against 16, bit for bit); another platform is unverified.
* **Wiring.** `the_compartmental_arm_scores_a_synthetic_triangle` (cut-down
  sampler: the mean is an ultimate, not a reserve; reproducible by seed;
  the library's target acceptances) and `the_compartmental_arm_scores_group_353`
  (the predictive mean within 15% of the ODP bootstrap's, about 1.5 s in
  release) run in CI's `bayes` job. Company 337 is not a 10 by 10 square,
  so its parity targets stay in `validation/tests/reserving_compartmental*.rs`
  and are not reused here.

# Not yet shown

Whether the
narrow intervals come from the preset's priors or the model (a run with
priors fitted to each line, or to the panel); a run with a floor on the
cells that scores more of model 2's triangles; a run on the printed codes
only, which would separate panel from model.
