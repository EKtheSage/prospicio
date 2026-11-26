---
type: Finding
title: The ODP bootstrap's mean total ultimate on RAA is about 0.8% above the chain ladder's
description: "Measured by hand, not by a CI test: on RAA the ODP bootstrap (gamma process) gives a mean total ultimate of 214,799 against the chain ladder's 213,122 at 4,000 simulations (+0.79%), and +0.77% and +0.76% in two runs of 40,000 (about 1,650, 3.2% of the 52,135 reserve), while each origin's mean ultimate is within four standard errors of the chain ladder's. The small upward bias of each origin adds up over the origins, so the total's mean is the bootstrap's expected ultimate and not the chain ladder's to Monte Carlo error."
tags: [reserving, bootstrap, odp, raa, bias]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
sources:
  - id: unit-test
    resource: ../crates/prospicio-reserving/src/draws.rs
    title: the_mean_ultimate_is_the_bootstraps_expected_ultimate_to_monte_carlo_error, which asserts only that the large run's mean total is within 2% of the chain ladder's
  - id: cells
    resource: ../validation/tests/reserving_bootstrap_cells.rs
    title: the_mean_cells_are_the_chain_ladders_on_genins, whose ODP allowance of 3% rests on the gap below
  - id: order
    resource: reserving-draws-simulation-order.md
    title: A reserving fit's ultimates per simulation keep one order
---

# Facts

* On RAA the ODP bootstrap (gamma process, 4,000 simulations, seed 5)
  has a mean total ultimate of 214,799 against the chain ladder's
  213,122 (+0.79%). Two runs of 40,000 simulations gave +0.77% and
  +0.76%, about 1,650 or 3.2% of the chain ladder's 52,135 reserve, so
  the gap is the bootstrap's and not Monte Carlo error.
* Each origin's mean ultimate is within four standard errors of the chain
  ladder's in the same run (asserted by the unit test); the small upward
  bias of each adds up over the origins in the total.
* A check that "the mean equals the expected ultimate" therefore compares
  the bootstrap's total with the bootstrap's own larger run, not with the
  chain ladder. The unit test does that and bounds the larger run's gap
  to the chain ladder below 2%; the figures above were measured by hand
  and are not pinned by any test.
* On GenIns (Taylor-Ashe, 10,000 simulations) the ODP bootstrap's mean
  reserve by origin is 1% to 2.3% above the chain ladder's (origin 2010:
  4,734,060 against 4,625,811), measured on one Windows run while
  building the bootstraps' cells (#979). The simulated future cells sum
  to those reserves, so each cell's mean carries the same gap; the cells'
  shares of their origin's mean reserve match the chain ladder's within
  4 standard errors. The Mack bootstrap's mean cells match the chain
  ladder within 4 standard errors with no allowance.
