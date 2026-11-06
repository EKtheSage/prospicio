---
type: Finding
title: "On four-origin programs held out of the Schedule P panel, the industry prior steadies a weak triangle, gains little on the chain ladder, and its 90% intervals cover 38% to 74%"
description: "Each company of four Schedule P lines in turn, its accident years 1994 to 1997 at ages to 48 months, fitted with the prior of the other 49 cut by at_age(48), beside a vague BayesBfNormalised and the chain ladder, all on the same tail factor and scored on paid at 120 months. The prior's rmse of the ultimate is 0.135, 0.197, 0.088 and 0.287 (workers' compensation, commercial auto, personal auto, other liability) against the chain ladder's 0.141, 0.220, 0.085 and 0.276; the vague model's is 0.15 to 2.4 and its other liability value moves between Python and R. The prior's 90% intervals cover the outcome 38% to 59% of the time at 120 months and 50% to 74% at 48, no better than the vague model's. A pattern prior from peers of another pattern distorts the fit: in the ILS notebook, a slow peer pattern on a fast cedant moved the realised one-year result to the 100th percentile."
tags: [reserving, bayes, bornhuetter-ferguson, industry-prior, schedule-p, guides]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-05T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/reserving-industry-prior.md
    title: Sections 5 and 6, the blocks that print these tables
  - id: test
    resource: ../python/tests/test_guides.py
    title: test_recorded_blocks_of_the_industry_prior_guide (runs only with PROSPICIO_GUIDE_FETCH=1)
  - id: panel
    resource: industry-prior-calibration-panel.md
    title: The prior per line on the calibration panel
  - id: notebook
    resource: ../examples/ils_deal_analysis.ipynb
    title: The ILS deal analysis notebook, on synthetic peers
---

# What was found

Run on 2026-11-05 (Windows, release build of the Python bindings) from
`cas-schedule-p` 2026.6.13, on the 50 companies per line that
`screens.select_companies` chooses, paid losses, accident years 1988 to 1997.
Not marked verified: CI has no panel, so only a run with
`PROSPICIO_GUIDE_FETCH=1` compares these numbers. Workers' compensation paid
gave the recorded 0.6385, 0.1890 and 0.2927 of
[the calibration panel finding](industry-prior-calibration-panel.md).

The program is a company's accident years 1994 to 1997 at the end of 1997, ten
cells to 48 months. The prior is from the other 49 companies, cut with
`at_age(48)`. Sampler: 4 chains, 500 tuning and 500 kept draws, 2,000
simulations, seed the company's code. The tail factor is the prior's for all
three methods. The error is relative to the paid at 120 months.

| Line | Scored | Skipped | rmse chain ladder | rmse vague | rmse prior | Cover at 120, vague and prior | Cover at 48, vague and prior |
|---|---|---|---|---|---|---|---|
| Workers' comp | 48 | 2 | 0.141 | 0.158 | 0.135 | 0.42, 0.38 | 0.71, 0.73 |
| Commercial auto | 44 | 6 | 0.220 | 0.266 | 0.197 | 0.61, 0.59 | 0.64, 0.64 |
| Personal auto | 48 | 2 | 0.085 | 0.085 | 0.088 | 0.48, 0.42 | 0.58, 0.50 |
| Other liability | 43 | 7 | 0.276 | 1.842 | 0.287 | 0.51, 0.40 | 0.79, 0.74 |

* **The prior against the vague model.** It removes the runaway ultimates of a
  few weak triangles. On other liability the vague model's rmse rests on those
  few posteriors. Python and R agree on the prior's column and on the chain
  ladder's to the printed digits but print 1.842 and 2.426 for that cell
  (0.158 and 0.153 for workers' compensation), since those posteriors react to
  differences in the last digits of the draws.
* **The prior against the chain ladder.** The gain is 0.006 and 0.023 of rmse
  on two lines, and the prior is worse by 0.003 and 0.011 on the other two. In
  each line it beats the chain ladder on about half of the companies (25 of 48,
  25 of 44, 19 of 48, 17 of 43).
* **The tail.** The prior's factor from 48 to 120 months is 1.155, 1.118, 1.083
  and 1.335. The companies' own factors have 10th and 90th percentiles of 1.068
  and 1.308, 1.051 and 1.290, 1.028 and 1.170, and 1.074 and 1.813. A fixed tail
  sits inside all three methods, which limits how much any of them can differ.
* **The intervals.** A fixed tail takes cover at 120 months down to 0.38 to 0.61.
  Cover at 48 months, with no tail involved, is still only 0.50 to 0.79. The
  prior narrows the interval without moving the cover up. Calendar-year
  effects, which the ODP model does not have, are the likely cause (not tested
  here); compare Meyers' result for Mack and the ODP bootstrap in
  [the calibration panel finding](meyers-calibration-panel.md).
* **Skipped companies.** 17 of 200 thin triangles are refused: at 36 or 48
  months a column of one or two cells has increments that sum to zero or less.
  Paid development that stops early in a small company does this.
* **Peers must have the program's pattern.** The ILS notebook's first synthetic
  peers paid slowly (10% at 12 months) against a cedant at 39%. The peers' sd of
  0.1 on the log share ratios then held the fit near the wrong pattern: the
  realised one-year result went to the 100th percentile, from 65% with a vague
  pattern prior. With peers paying like the cedant it is at 75%. On Schedule P
  the sds are far wider (0.3 to 1.2), so this risk grows as peers get more alike.
* **New year.** With `common_cv`, `new_year` pools the cedant's level over all
  its years and has no trend, so on the notebook's rising loss ratios it gives
  0.453 where the last five years average 0.501 (at 2,000 simulations). The old
  lognormal read the last five years directly and gave 0.496, so the median IRR
  of the flat structure moved from 40% to 56%, about 2 points of it from paying on
  the underwriting basis. The notebook states this gap.
