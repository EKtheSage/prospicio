---
type: Finding
title: The ILS notebook's hand-written earned pattern fell after month 24, and an accident-year pattern read from each accident date pays slowly
description: "Before #601 the ILS deal notebook built its monthly earned share with 1 - (2 - t)^2 / 2 past the first year, which falls after month 24 (0.5 at month 36, -3.5 at month 60). The unearned premium then grew, the 30% unearned-premium collateral grew with it, and the flat quota share called 65.8m of capital for a median IRR of 18%; with monthly_patterns the capital called is 21.2m and the median IRR 40% (2,000 simulations). monthly_patterns reads its payment pattern from each accident date, so a chain ladder's accident-year pattern passed to it pays 7.3% by month 12; the underwriting-year conversion of the fit's draws (DevelopmentPatterns.monthly with UnderwritingBasis) pays 15.0% and the old helper, the pattern from 1 January, 39.1%. The median IRR is 40.0%, 41.7% and 44.4% under the three. The exact conversion reads every draw and took 52 s for 2,000 draws."
tags: [prospicio-pricing, cashflow, payment-pattern, ils, python, r]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-02T12:00:00Z }
sources:
  - id: notebook
    resource: ../examples/ils_deal_analysis.ipynb
    title: the ILS deal analysis notebook
  - id: test
    resource: ../python/tests/test_examples.py
    title: test_ils_deal_analysis, which pins the monthly patterns and the deal's actual against expected
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/722
    title: a basis on the single-pattern route of monthly_patterns
  - id: pricing-test
    resource: ../python/tests/test_pricing.py
    title: test_an_accident_year_pattern_on_the_underwriting_basis
---

# The monthly patterns of the ILS notebook

Found while replacing the notebook's hand-written monthly patterns with
`monthly_patterns` (#601), for the `[ils-workaround-table]` item. The
numbers are for the flat structure at `PROSPICIO_EXAMPLE_SIMS=2000` and the
notebook's seed, on Windows.

* **The earned pattern.** The notebook earned a year of policies written
  evenly, each over twelve months, as `t^2 / 2` for `t <= 1` and
  `min(1, 1 - (2 - t)^2 / 2)` after. The second branch is right up to
  `t = 2` and falls after it: 0.5 at month 36 and -3.5 at month 60. The
  unearned premium (written less earned) therefore grew after month 24, and
  the requirement of 30% of it grew too. The fund called 65.8m of capital
  over the term, for a median IRR of 18.3% and a mean MOIC of 1.123.
  `monthly_patterns` earns the same writing correctly (each month written
  at its start, so 13/24 by month 12, and 1 from month 24): the capital
  called is 21.2m, the median IRR 40.0% and the mean MOIC 1.367. The R twin
  had the same expression. `test_ils_deal_analysis` now checks that the
  earned share never falls and is 1 from month 24.
* **The paid pattern's basis.** `monthly_patterns` reads its
  `PaymentPattern` from each loss's accident date. A chain ladder's pattern
  is an accident-year pattern, from the start of the year, so passing it
  lags every loss a second time. Three readings of the sample paid
  triangle, paid by months 12 and 24 and the median IRR (all with the
  corrected earned pattern):

  | Reading | Month 12 | Month 24 | Median IRR |
  |---|---|---|---|
  | accident-year pattern from 1 January (the old helper) | 39.1% | 72.4% | 44.4% |
  | the fit's draws on `UnderwritingBasis()`, averaged | 15.0% | 58.6% | 41.7% |
  | accident-year pattern from each accident date (`monthly_patterns`) | 7.3% | 39.6% | 40.0% |

  The underwriting basis is the right one for a year of policies, and the
  two shortcuts err on either side of it. Its only route is
  `fit.patterns.monthly(months, basis=UnderwritingBasis())`, which reads
  every simulation's pattern: 52 s for 2,000 draws, where the notebook's
  whole test takes about 40 s (#717). The notebook uses `monthly_patterns`
  and says that it pays slowly; #722 asks for the conversion of one
  deterministic pattern. Since #722, `monthly_patterns(written, chain,
  accident_period="year")` converts the chain ladder's own pattern with the
  deal's writing: 15.01% and 58.56% by months 12 and 24, the averaged
  draws' figures within 0.1 point, checked by
  `test_an_accident_year_pattern_on_the_underwriting_basis`. `docs/guides/submission-schema.md` passed
  `chain.cdf` to `monthly_patterns` the same way until it paid each
  simulation on its drawn pattern
  ([the submission schema's drawn patterns](submission-schema-pattern-draws.md)).
* **`deal_ave`** (#603) compares a deal's actual cumulative paid with the
  priced simulations' `u P S(t)`. With the actual paid on the same pattern
  at 1.25 times the priced mean loss ratio, the ratio to expected is 1.25
  at every month and the actual sits at the 95th percentile of the
  simulations from month 6 on.
