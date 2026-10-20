---
type: Finding
title: R's glm() stops short of the coefficients on a non-canonical link
description: stats::glm tests convergence on the deviance alone, so with the gamma or Tweedie on the log link, where IRLS converges only linearly, it stops with coefficients about 1e-8 (relative) from the optimum even at epsilon = 1e-14; refitting from its own estimate until the coefficients settle gives a reference good to 1e-9.
tags: [glm, r, statmod, convergence, irls, gamma, tweedie, parity, reserving]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-20T12:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-20T12:00:00-07:00 }
sources:
  - id: script
    resource: ../validation/scripts/reserving_glm_r.R
    title: The reference script for the ODP GLM's families, with the refit loop
  - id: test
    resource: ../validation/tests/reserving.rs
    title: odp_glm_matches_r_glm, the parity test on GenIns and ABC
---

# Facts (2026-10-20)

* `glm(..., control = glm.control(epsilon = 1e-14))` with `Gamma(link =
  "log")` on GenIns' cross-classified design stops after 12 iterations,
  `converged = TRUE`, with `factor(origin)2002` at 0.317247704664739; refit
  from those coefficients it moves to 0.317247708852829, a relative 1.3e-8.
  The quasi-Poisson on the same design, whose log link is canonical, does
  not move (relative 1.5e-10 over every coefficient).[^script]
* The cause: R's test is `|dev - dev_old| / (|dev| + 0.1) < epsilon`, on
  the deviance only. On a non-canonical link IRLS (Fisher scoring)
  converges linearly, and near the optimum the deviance changes with the
  square of the coefficients' error, so a deviance step below 1e-14 leaves
  them about 1e-7 to 1e-8 off. statmod's `tweedie(var.power = 1.5,
  link.power = 0)` behaves the same.
* prospicio-glm's IRLS also tests the coefficients (relative change below
  `1e-3 sqrt(tolerance)`), but at the default tolerance `1e-10` a small
  coefficient can still end 1e-9 short; at `1e-14` both agree to 1e-9.[^test]
* So a parity reference for a non-canonical link refits until the
  coefficients' relative change is below 1e-13 (at most 20 refits, each at
  `epsilon = 1e-14`; a lower epsilon only makes glm warn that it did not
  converge), and the Rust side of the test fits at `tolerance = 1e-14`.
