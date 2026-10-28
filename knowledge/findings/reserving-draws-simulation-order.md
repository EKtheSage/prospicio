---
type: Finding
title: A reserving fit's ultimates per simulation keep one order through towers, deal books and cash flows; Meyers' are its own draws and Compartmental's are paid
description: Row i of a fit's reserves is simulation i for every origin and segment, and a fit run again with the same seed (bootstraps and NUTS alike, on one machine) gives the same rows; UltimateDraws adds each origin's latest value to its reserve draw, so a tower on total_ultimate_draws by apply_annual cedes, simulation by simulation, what it cedes on the joint ultimates or on the reserves with the attachment lowered by the latest values. Meyers' models keep their simulated value at the oldest age, which the reserves plus the latest values match to 1e-12; Compartmental's ultimate is the paid at the square's last age. On RAA the ODP bootstrap's mean total ultimate is about 0.77% (1,650) above the chain ladder's, though each origin's is within four standard errors of it.
tags: [reserving, bootstrap, bayes, meyers, compartmental, predictive-distribution, simulation-order, reinsurance, cashflow]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
verified: { by: process:ci, at: 2026-10-28T12:00:00Z }
sources:
  - id: unit-tests
    resource: ../crates/prospicio-reserving/src/draws.rs
    title: The unit tests of UltimateDraws (layout by key, totals, same seed, join and reorder, the mean against the chain ladder)
  - id: tower-tests
    resource: ../validation/tests/reserving_loss_ratio_draws.rs
    title: Towers on the draws against the same towers on the fit's distributions
  - id: bayes-tests
    resource: ../validation/tests/reserving_bayes_loss_ratio_draws.rs
    title: The Bayesian fits' layout, long-table means and same-seed draws (slow-bayes)
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Reserving v0.2, decision 26
---

# Facts

* Every stochastic reserving fit returns its reserves as one
  `PredictiveDistribution` whose row `i` is simulation `i` for every
  origin and segment. `UltimateDraws` (`prospicio_reserving`) reads row
  `i` into index `i`: an origin's ultimate draws are its latest value
  plus its reserve column, and the totals are sums by index. Fitting
  again with the same seed gives identical draws: the bootstraps draw
  simulation `i` from `StreamRng::new(seed, i)`, and the NUTS fits
  (`BayesOdp`, both Bayesian BFs, Meyers' CRC, compartmental model 1,
  alone and over segments) repeated their draws exactly in the tests.
* A stop loss applied by `Tower::apply_annual` to a fit's
  `total_ultimate_draws` cedes, in every simulation, what
  `Tower::apply_aggregate` cedes on the fit's `ultimates()` and on its
  `reserves` with the attachment lowered by the sum of the latest values
  (RAA, ODP bootstrap, 5,000 simulations; 1e-9 relative). The same holds
  for a contract on one origin's loss ratio and for each line of a fit
  over segments, so the vectors keep their pairing through the tower.
* `join` keeps row `i` of each part; `reorder_groups` keeps each group's
  rows whole, so every column of a group's row stays the latest plus the
  reserve of one source simulation (which `reorder_groups_indexed`
  names), but row `i` of one group then meets another simulation of
  another group.
* Meyers' models simulate the cumulative value at the oldest age and keep
  it as `ultimates`; their `reserves` are those less the latest values,
  so the two agree to rounding (within 1e-12 relative on group 353), and
  `UltimateDraws` returns the simulated ultimates. Their mean matches the
  long table's `ultimate_mean` and the mean loss ratio on their own
  premium its `ulr_mean`.
* A compartmental model's reserve is the paid at the square's last age
  less the latest paid, so its ultimate (latest paid plus reserve) is on
  the paid basis and equals the last future paid cell of each origin in
  the same simulation (company 337, model 1); an origin with no paid yet
  (1997) has its reserve as its ultimate.
* On RAA the ODP bootstrap (gamma process, 4,000 simulations) has each
  origin's mean ultimate within four standard errors of the chain
  ladder's, but the total's mean is 214,799 against the chain ladder's
  213,122 (+0.79%); two runs of 40,000 simulations give +0.77% and
  +0.76% (about 1,650, 3.2% of the 52,135 reserve), so the bias is the
  bootstrap's and not Monte Carlo error. A check of "the mean equals the
  expected ultimate" therefore compares the total with the bootstrap's
  own larger run, not with the chain ladder.
* `BayesBfNormalisedFit::expected_ultimate()` is the posterior mean of
  the prior's `U_w`, without the process error and the cells already
  paid, so it is not the mean of the ultimate draws.
