---
type: Finding
title: In the submission schema's quota share the underwriting basis moves the IRR, and paying each simulation on its own drawn pattern does not
description: "docs/guides/submission-schema.md now pays each of its 2,000 ODP bootstrap simulations on that simulation's drawn pattern, read month by month on an underwriting basis (boot.patterns.monthly(120, basis=UnderwritingBasis(), monotone=\"running_max\")). The median IRR of the flat structure went from 40.5% to 41.3% and the mean MOIC from 1.378 to 1.366. Taken step by step: the underwriting basis for the paid pattern (15.0% paid by month 12 against 7.3% from each accident date) adds 1.7 points of median IRR, earning on the basis's exposure (0.500 earned at month 12 against 13/24) takes 0.9 back, and the drawn patterns against their mean move the median IRR, mean MOIC and the 5th and 95th percentiles by at most 0.1 point. The monthly reading needs 120 months, the month the converted pattern reaches 1, though the deal commutes at 60; 77 of 2,000 draws pass 1 at 96 months and need running_max. The reading takes 0.9 s with #726 and about 20 s without it."
tags: [prospicio-pricing, prospicio-reserving, cashflow, payment-pattern, bootstrap, guides]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-03T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/submission-schema.md
    title: the submission schema guide, whose printed IRR and MOIC test_guides.py checks
  - id: tables
    resource: ../examples/data/deal_report/metrics.csv
    title: the committed sample tables, checked against the guide's output by test_guides.py
  - id: speed
    resource: https://github.com/EKtheSage/prospicio/pull/726
    title: the faster monthly reading of drawn patterns (#717)
---

# The submission schema's drawn patterns

Found while moving `docs/guides/submission-schema.md` to a pattern per
simulation (`[submission-pattern-draws]`). The deal is the guide's 20% quota
share of the sample cedant's 2026 year, priced on 2,000 ODP bootstrap
simulations of the paid triangle cut at 108 months (seed 20261028),
commuted at month 60. Numbers are from a Windows run with #726's code.

* **Setup.** `boot.patterns.monthly(MONTHS, basis=UnderwritingBasis(),
  monotone="running_max")` returns one row of cumulative paid shares per
  simulation, in the order of `boot.ultimate_draws`, so the guide's 2026
  loss ratios pair with it row by row. `MONTHS` must reach the month at
  which every converted pattern reaches 1: 108 + 12 + 12 - 12 = 120 here, so
  `monthly(60, ...)` raises. `terms.run(..., paid_patterns=...)` needs each
  row to have the terms' length and to end at the terms' last paid share, so
  the terms' written, earned and paid shares run for 120 months as well,
  with commutation still at 60. `monotone="refuse"` fails on 77 draws, the
  first falling from 1.00039 at 96 months to 1 at 108.
* **Effect, step by step** (median IRR and mean MOIC):

  | Step | Flat | Sliding scale |
  |---|---|---|
  | one pattern from each accident date, earned by `monthly_patterns` (before) | 40.54%, 1.3776 | 36.09%, 1.3512 |
  | the rows' mean pattern on the underwriting basis, same earned | 42.26%, 1.3661 | 37.56%, 1.3397 |
  | earned on the basis's exposure (written through each month) | 41.34%, 1.3659 | 36.76%, 1.3396 |
  | each simulation on its own row (after) | 41.33%, 1.3659 | 36.77%, 1.3396 |

  Paying faster lowers the outstanding loss, and so the collateral called,
  which raises the IRR and lowers the multiple. Earning on the exposure is
  half a month later than `monthly_patterns`' writing at the start of each
  month (0.500 against 13/24 at month 12), which holds more unearned
  premium. The drawn rows against their mean change the 5th and 95th
  percentiles of IRR and MOIC by at most 0.0007 and the share of
  simulations with a MOIC below 1 from 1.70% to 1.65% (flat): the
  bootstrap's patterns carry the factors' parameter uncertainty only, and
  most of the loss is paid by the commutation (97.3% at month 60). The
  committed `metrics.csv`'s probability of loss for the flat structure moved
  from 1.15% to 1.65%, almost all from the basis.
* **Time.** The monthly reading of the 2,000 rows takes 0.86 s on 16 cores
  with #726's code and the whole guide about 3 s. Without #726 the reading
  is about 20 s, which would break `test_guides.py`'s 20 s a guide, so the
  guide's change merges after #726.
