---
type: Finding
title: A cash-flow run's commission is not deducted from the premium, so a commission that differs by simulation needs one run per rate
description: "CashflowTerms.run(u, commission) uses the commission rate only for premium adjustments with less_commission and for the fee, so a ceding commission has to be taken out of the premium through collected_rate, a number fixed in the terms. A sliding scale makes the commission differ by simulation, so the notebook groups simulations by their commission rate (to 0.1 point) and runs the terms once for each group. A collateral rule on the reserve (ultimate loss times earned premium less paid) alone lets the investor receive money before posting any, which gives flows of mixed sign and an IRR near -50% beside a MOIC of 3.8; a rule on the outstanding loss (ultimate loss times premium less paid) makes the fund call capital from the start, as a collateralised deal does. The IRR is searched up to 500% (cashflows(upper=5.0)): the default upper bound of 100% left 42 flat and 6 sliding-scale simulations of 10,000, among the best years, without an IRR."
tags: [prospicio-pricing, cashflow, collateral, ils, python]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-23T12:00:00Z }
sources:
  - id: notebook
    resource: ../examples/ils_deal_analysis.ipynb
    title: the ILS deal analysis notebook
  - id: code
    resource: ../crates/prospicio-pricing/src/cashflow.rs
    title: the cash-flow engine
---

# Commission and collateral in the cash-flow engine

Found while writing `examples/ils_deal_analysis.ipynb` (a collateralised
20% quota share of a synthetic cedant).

* `CashflowTerms(premium, ..., collected_rate=...)` puts `collected_rate`
  times the written premium into the trust. `run(u, commission)` reads the
  commission only for a `PremiumAdjustment(..., less_commission=True)` and
  for the fee. A run with a 30.5% commission and `collected_rate=1` pays the
  cedant nothing: one simulation (loss ratio 0.52) gave the investor flows
  of 17.7m in total, where the premium of 36.0m less its commission and its
  loss leaves 6.4m.
* A flat commission is one `collected_rate`. A sliding scale gives each
  simulation its own, so the notebook runs the terms once for each distinct
  commission rate (rounded to 0.001) and puts the flows back in simulation
  order. The grouping is exact up to the rounding of the rate.
* With a requirement of 0.3 of unearned premium plus 1.2 of the reserve
  (`"reserve"`: ultimate loss times premium earned, less paid), the trust
  holds less than the premium collected for the first year. The investor
  receives money first and posts it later, and the IRR search found a root
  near -51% in every simulation beside a MOIC of 3.8. A requirement of 0.3
  of unearned premium plus 1.25 of the outstanding loss (`"outstanding"`)
  makes the fund call capital from the start (65.8m called in total over
  the term on a 36.0m premium, recycled as losses are paid) and gives a
  median IRR of 18% and a mean MOIC of 1.12 over 10,000 simulations.
* `DealBook.cashflows()` searches for the IRR between -95% and 100% by
  default, and a simulation whose IRR is above 100% comes back as NaN,
  like flows with no root (#670). At 10,000 simulations 42 flat and 6
  sliding-scale simulations, among the best years, had no root and were
  left out of the IRR percentiles. The notebook called
  `cashflows(upper=5.0)` (R `upper = 5`), and since #675 calls
  `cashflows(extend=True)` (R `extend = TRUE`) with byte-identical
  output: every simulation has an IRR, the
  largest 199%, and the flat structure's 95th percentile moves from 47.7%
  to 48.9%.

A per-simulation commission in `run`, and per-simulation payment patterns,
were gaps of the cash-flow engine. The second, #600, is closed:
`CashflowTerms.run(..., paid_patterns=)` takes a reserving fit's
`patterns.monthly(...)` (#673). The notebook keeps one pattern, since its
loss ratios are lognormal draws around the fit and not the fit's own
simulations.
