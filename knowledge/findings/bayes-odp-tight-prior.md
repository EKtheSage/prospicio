---
type: Finding
title: A tight prior on BayesOdp's development effects lifts the late columns and the reserve
description: BayesOdp's normal priors are centred on 0, which for a development effect is the first column's level; the late columns rest on one or two cells, so even a moderate prior SD pulls their effects up toward 0, and on RAA the total mean reserve rises from 54,600 at SD 10 to 59,100 at 3, 74,000 at 1 and 115,000 at 0.2.
tags: [bayes, reserving, chain-ladder, priors, xll]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T06:30:00Z }
sources:
  - id: odp
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: prospicio_reserving::bayes::BayesOdp (the model and its priors)
  - id: xll
    resource: ../docs/design/xll.md
    title: Design note, the Excel add-in (decision 21, PRO.BAYES.ODP's prior_sd)
---

# Finding

* `BayesOdp` puts `N(0, sd^2)` priors on the origin effects `a_w` and the
  development effects `b_d` (corner constraints `a_1 = b_1 = 0`), so a
  development effect's prior mean is the first column's level. The
  likelihood holds the early columns, but the last ones rest on one or
  two cells, and a prior SD that looks vague on the log scale still
  binds them.[^odp]
* On RAA (four chains of 1,000 warm-up and 1,000 kept draws, 10,000
  simulations, seed 1, release build), with one SD for the intercept,
  origin and development effects as `PRO.BAYES.ODP`'s `prior_sd` sets
  them:[^xll]

  | prior SD | b[7] | b[9] | b[10] | total mean reserve |
  |---|---|---|---|---|
  | 10 | -1.21 | -2.76 | -5.06 | 54,614 |
  | 3 | -1.13 | -2.07 | -2.43 | 59,071 |
  | 1 | -0.84 | -1.01 | -0.86 | 74,037 |
  | 0.5 | -0.56 | -0.49 | -0.37 | 89,564 |
  | 0.2 | -0.22 | -0.15 | -0.09 | 115,291 |

  The chain ladder's reserve is 52,135. At SD 10 the posterior is close
  to the likelihood's (the remaining gap is the process error and the
  vague priors), at 3 the last column already moves by half its value.
* So the add-in's `prior_sd` help and README say to keep it wide; a
  prior that means to shrink should be centred on a pattern, which the
  model does not offer.[^xll]
