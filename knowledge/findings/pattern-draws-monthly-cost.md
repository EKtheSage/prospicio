---
type: Finding
title: Reading 2,000 ODP bootstrap patterns month by month on an underwriting basis took about 21 s with the Weibull reading; in parallel and from the mean pattern's curve it takes 0.3 s
description: "DevelopmentPatterns.monthly(144, basis=UnderwritingBasis()) on the sample submission's paid triangle cut at 108 months took 17 s to 21 s for 2,000 ODP bootstrap draws (release, Windows, 16 cores), the loglogistic reading 11 s to 20 s and ln-linear 0.05 s, nearly all of it the growth curve's search from eleven starts, on one thread. Since #717 (reserving-v02.md decision 29) the draws are read in parallel and each curve is searched from the mean pattern's: 0.3 s on 16 cores and 3.4 s on one, within 9.3e-10 of a share of the old result and the same with any number of threads. monotone='refuse' refuses 77 of the 2,000 draws, where a pseudo factor below 1 at the oldest ages puts the share above 1 at 96 months before it falls to 1 at 108 (1,344 on the uncut triangle to 120 months), so bootstrap draws need 'running_max'."
tags: [guides, reserving, cashflow, pattern, performance, prospicio-reserving]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-02T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/submission-schema.md
    title: the submission schema and deal report guide
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/717
    title: "Reserving: DevelopmentPatterns.monthly with the Weibull reading is slow for thousands of draws"
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 29, reading thousands of simulated patterns month by month quickly
  - id: test
    resource: ../crates/prospicio-reserving/src/pattern_draws.rs
    title: simulations_read_from_the_mean_curve_match_the_search_from_every_start, run by CI's Rust job
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

# Made fast (#717)

Profiled in Rust on the same draws (release, Windows, 16 cores), 21.6 s
of the Weibull reading's 22 s on the uncut triangle was the growth curve's
search (Nelder-Mead from eleven starts to a tolerance of `1e-12`, about
3,400 evaluations of the objective a draw, each 120 Weibull values); the
projection onto the exact monthly pattern took 0.3 s. `reserving-v02.md`
decision 29 reads the draws in parallel and searches each draw's curve
from the mean pattern's curve.[^note] On the triangle cut at 108 months
(the machine was shared, so the times vary by about a fifth):

| reading | before | after, one thread | after, 16 cores | Python, after |
|---|---|---|---|---|
| ln-linear, underwriting basis | 0.06 s | 0.05 s | 0.005 s | 0.016 s |
| Weibull, underwriting basis | 17.2 s | 3.4 s | 0.30 s | 0.35 s |
| loglogistic, underwriting basis | 11.1 s | 3.7 s | 0.22 s | 0.28 s |
| Weibull, origin basis | 19.4 s | 5.4 s | 0.31 s | |

* Threads alone give the old result bit for bit and cut the time about
  twelve times on 16 cores. The search from the mean pattern's curve (and
  decision 23's line), with the search from every start as its fallback,
  moves the monthly shares by at most `9.3e-10` of a share over the 2,000
  draws (`3.3e-10` Weibull on the underwriting basis, `5.4e-10` Weibull on
  the origin basis, `9.3e-10` loglogistic), since the curve is only the
  projection's prior. A unit test in `pattern_draws.rs` holds simulated
  draws to `1e-8` of the search from every start under three bases.[^test]
* Fitting distinct patterns once would save nothing: the bootstrap's 2,000
  patterns are products of continuous pseudo factors, all distinct.
* The refused draws: on the triangle cut at 108 months one or two cells
  set the oldest factors, and in 77 draws the bootstrap draws one below 1,
  so the share passes 1 at 96 months and falls back to 1 at 108. On the
  uncut triangle, where one cell sets the 108 to 120 month factor, 1,344
  of the 2,000 draws are refused.
