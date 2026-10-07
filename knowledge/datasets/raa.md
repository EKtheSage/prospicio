---
type: Dataset
title: RAA casualty triangle
description: Mack's (1993) 10x10 cumulative paid triangle; has one negative increment, which shapes how ODP models must handle it.
resource: ../validation/data/raa.csv
tags: [reserving, triangle, odp, chain-ladder]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T15:54:30Z }
sources:
  - id: mack
    resource: R package ChainLadder, dataset RAA
    title: Mack (1993), Distribution-free calculation of the standard error of chain ladder reserve estimates, as R ChainLadder::RAA
  - id: tests
    resource: ../validation/tests/models.rs
    title: over_dispersed_poisson_fits_raa_with_its_negative_increment, mean_preserving_draws_centre_the_odp_reserve_on_the_chain_ladder
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/111
    title: Issue 111, four gaps found while building reserving
---

# Facts

* Origins 1981–1990, development 12–120 months, values in thousands.[^mack]
* Origin 1982 loses 103 between 72 and 84 months: the only negative
  incremental cell.[^issue]
* Chain Ladder total reserve (volume-weighted factors): 52,135.228.

# Consequences

* R's `quasipoisson` and, before #114, `prospicio_glm` refused the ODP GLM on
  RAA. A quasi-likelihood ODP needs only `V(μ) = μ` and `μ > 0`, so
  `Glm::over_dispersed_poisson()` now accepts negative responses and
  reproduces the Chain Ladder reserve to 1e-6.[^tests] See
  [negative increments](/findings/odp-negative-increments.md).
* Its log score is −∞ at the negative cell under any density on y ≥ 0
  (see [ODP log density](/findings/odp-continuous-density.md)).

[^mack]: Mack (1993)
[^issue]: Issue 111
[^tests]: RAA ODP tests
