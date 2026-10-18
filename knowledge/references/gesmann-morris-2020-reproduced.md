---
type: Reference
title: Gesmann and Morris (2020), compartmental models 1 and 2, reproduced in the guide
description: The guide reserving-compartmental.md fits the paper's models 1 and 2 on company 337 with Compartmental; every printed m1fit row agrees within a tolerance of 0.15 printed sd on a mean and 10% on an sd (checked in CI), every m2fit mean within 0.08 printed sd and every sd within 4% but tau_dev_kp's (recorded, not run by CI). Facts kept here are what the paper does not print (the outcome's percentile, 92nd under model 1 and 99th under model 2; the test diagonal, model 1's outstanding misfit, and why model 2's fit and both one-year views are recorded rather than checked).
resource: Casualty Actuarial Society, CAS Research Papers (2020)
tags: [reserving, compartmental, bayes, gesmann-morris, clrd, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T20:00:00-07:00 }
sources:
  - id: paper
    resource: Casualty Actuarial Society, CAS Research Papers (2020)
    title: Gesmann and Morris (2020), "Hierarchical Compartmental Reserving Models", CAS Research Paper
  - id: guide
    resource: ../docs/guides/reserving-compartmental.md
    title: The guide that runs the reproduction (its checked blocks run in CI)
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guide and checks every printed line
  - id: code
    resource: gesmann-morris-2020.md
    title: The paper's brms code reconstructed in Stan, its printed posteriors and its text and code disagreements
---

# What it is

The guide `docs/guides/reserving-compartmental.md` (and its R twin, `.qmd`)
explains the compartmental model and fits the paper's two models on company
337 workers' compensation with `Compartmental` (training cells at calendar
years up to 1996, 1997 with premium only), the paper's printed posterior
summaries beside prospicio's.[^guide] The paper's text and plots are not
reproduced; the PDF is not in the repository, so the printed values are
those recorded in the reference note.[^code]

# Facts

* Model 1 at prospicio's default sampler (4 x 2,000 kept draws, seed 0, 2,000
  simulations) takes 5 to 9 s in a release build, so it runs in CI: all 18
  rows of the printed `m1fit` (9 parameters, mean and sd) agree within 0.15
  printed sd + 0.005 on a mean and 10% + 0.005 on an sd; the largest gap is
  `log_sigma_pd`, 6.671 against 6.66 (0.055 sd). The reserves and elpd_loo,
  which the paper does not print, are checked against the Stan
  reconstruction (total 173,341, sd 10,076; 1997 31,321, sd 6,125; elpd_loo
  -811.0).[^guide][^code]
* Model 2 takes about 61 s (4 x 8,000 kept draws at 0.999), so its fit is a
  recorded `no-run` block: every printed `m2fit` mean within 0.08 printed sd
  (largest `tau_dev_kp`, 0.076) and every sd within 4% except `tau_dev_kp`,
  0.511 against 0.43, as the Stan reconstruction's 0.518; the paper's own
  run had about 500 effective draws there.[^guide][^code]
* The outcome the paper's training set hides: paid at 120 months less the
  latest paid at the end of 1996 is 187,420. Model 1's total reserve is
  173,457 (sd 10,260; outcome at the 91.9th percentile), model 2's 151,861
  (sd 13,729; 99.2nd). The miss is 1996 and 1997, which paid 1.07 and 1.15
  of premium by 120 months against posterior ultimate loss ratios of 1.04
  and 0.67 (model 1) and 0.89 and 0.67 (model 2), while premium fell from
  119,427 (1993) to 48,052 (1997).[^guide]
* The test diagonal (calendar 1997) under model 1: 17 of 20 cells inside
  their 90% predictive intervals; 1997's paid (9,372) and outstanding
  (40,799) at 12 months lie above them. The normal likelihood's constant
  `sigma_os` (about 3,900) gives the old years' outstanding cells intervals
  below zero.[^guide]
* Model 1's two-rate curve fits paid closely but runs 1988's outstanding
  off too fast (24,944 actual at 48 months against 20,017 fitted, 13,137
  against 8,551 at 72).[^guide]
* elpd_loo: model 1 -810.7 with 3 of 90 Pareto k above 0.7 (largest
  0.96), model 2 203.8 with 9 of 90 (largest 1.30); the two are on
  different responses (amounts against loss ratios) and do not
  compare.[^guide]
* One-year views (recorded, Windows, 2,000 simulations): model 1 flagged
  81%, moment-matched 1,549, refitted 78, CDR sd 7,982, 0.78 of the
  lifetime sd, 503 s; model 2 flagged 100%, reweighted only with the
  approximate warning, CDR sd 11,379, 0.83 of the lifetime sd, 84 s.[^guide]
* The prior predictive check with the paper's preset puts 1988's paid at
  12 months at 0.18 to 0.39 of premium (90%) against 0.09 actual, and its
  outstanding at 36 months at 0.03 to 0.09 against 0.31: the conflict with
  the slow-paying book that pulls `k_p` to 0.43, visible before
  fitting.[^guide]
