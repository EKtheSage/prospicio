---
type: Finding
title: Reading 2,000 ODP bootstrap patterns month by month on an underwriting basis takes about 21 s with the Weibull reading and 0.05 s with ln-linear
description: "DevelopmentPatterns.monthly(144, basis=UnderwritingBasis()) on the sample submission's paid triangle (ages to 108 months) takes about 21 s for 2,000 ODP bootstrap draws (release build, Windows, 16 cores), the loglogistic reading about 20 s and the ln-linear reading 0.05 s, so a guide that must run in 20 s cannot pay each simulation on its own pattern with the default reading; the ln-linear mean share paid is 0.195 at 12 months against the Weibull's 0.150. monotone='refuse' errors on 77 of the 2,000 draws, where a pseudo factor below 1 makes the share fall, so the draws need 'running_max'."
tags: [guides, reserving, cashflow, pattern, performance, prospicio-reserving]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-29T20:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/submission-schema.md
    title: the submission schema and deal report guide
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/717
    title: "Reserving: DevelopmentPatterns.monthly with the Weibull reading is slow for thousands of draws"
---

# What was found

Measured while updating the guides for 0.0.7, with
`OdpBootstrap(n_sims=2000, seed=20261028)` on the paid triangle of
`examples/data/submission/triangle.csv` cut at 108 months, and
`boot.patterns.monthly(144, basis=..., monotone="running_max")`:[^guide]

| reading | seconds | share paid at 12 | at 24 | at 60 |
|---|---|---|---|---|
| ln-linear | 0.05 | 0.195 | 0.540 | 0.971 |
| weibull (default) | 21.0 | 0.150 | 0.586 | 0.973 |
| loglogistic | 20.4 | 0.142 | 0.596 | 0.973 |

* About 10 ms a draw for the fitted readings; the time looks
  single-threaded. A guide must run in under 20 s
  (`python/tests/test_guides.py`), so `submission-schema.md` keeps one
  chain-ladder pattern for every simulation and says that per-simulation
  patterns exist (#673). Issue #717 asks the Reserving lane to make the
  reading faster.[^issue]
* The ln-linear reading is fast, but the API docs flag it as putting an
  underwriting year too far developed at 12 months (see
  [ay-pattern-monthly-reading](/findings/ay-pattern-monthly-reading.md)),
  and here its mean share at 12 months is 30% above the Weibull's, so the
  guides do not use it.
* `monotone="refuse"` errors on 77 of the 2,000 draws: a pseudo factor
  below 1 makes the share fall over a period. `running_max` holds the share
  instead.
* The guide's loss ratios are in the bootstrap's simulation order (built
  from per-simulation sums of `boot.ultimate_draws(y)`), so the draws would
  pair row for row with the losses once the reading is fast.
