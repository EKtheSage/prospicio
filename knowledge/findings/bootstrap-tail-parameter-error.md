---
type: Finding
title: The tail in the bootstraps, refitted or drawn, against Mack's tail risk
description: With a constant tail drawn from the lognormal with Mack's tail standard error and the step to ultimate given Mack's tail sigma, Mack's bootstrap reproduces R's MackChainLadder(tail = ...) standard errors on RAA, GenIns and ABC. A tail refitted on each simulation's pseudo factors has R's process error but its own parameter error, first order on ABC and wider on RAA (1.56 times on the oldest origin) and GenIns (1.12), where R's log-linear rule extrapolates far and is convex in the factors; R's tail.se is of the same order as the refit's first-order error, not equal to it. On the ODP's pseudo factors the refit moves much more (RAA's oldest origin's mean reserve +78%).
tags: [reserving, bootstrap, mack, odp, tail, parameter-error, lifetime]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T16:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 9, tails in the bootstraps' lifetime view
  - id: test
    resource: ../validation/tests/reserving_bootstrap_tail.rs
    title: Validation test of the bootstraps with a tail against R MackChainLadder(tail = ...)
  - id: reference
    resource: ../validation/reference/reserving_tails_r.csv
    title: R ChainLadder 0.2.21 MackChainLadder with tail = TRUE, tail = 1.05 and given tail.se and tail.sigma
  - id: unit
    resource: ../crates/prospicio-reserving/src/mack_bootstrap.rs
    title: Unit tests of MackBootstrap::fit with a tail (and bootstrap.rs for the ODP)
  - id: mack1999
    resource: https://doi.org/10.2143/AST.29.2.504622
    title: Mack (1999), The standard error of chain ladder reserve estimates, recursive calculation and inclusion of a tail factor, ASTIN Bulletin 29(2)
---

# Finding

Both bootstraps' lifetime views take a tail (decision 9). An estimated
tail (`Tail::Curve`, `Tail::Bondy`, `Tail::LogLinear`) is refitted on each
simulation's pseudo factors; a constant one is drawn from the lognormal
with the factor as mean and the tail's standard error (Mack's, given or
extrapolated, for `MackBootstrap`; only a given one for `OdpBootstrap`).
The step to ultimate has Mack's process variance `tail_sigma^2 C^(2 -
alpha)`, or for the ODP is one more increment with the scale's Gamma
process. Measured at 20,000 simulations, seed 20,261,008:[^test]

* **A constant tail reconciles with Mack.** Against R ChainLadder 0.2.21's
  `MackChainLadder(tri, tail = 1.05)` (the tail's sigma and standard error
  extrapolated, either rule for the last sigma, or given) every origin's
  and the total standard deviation is R's `Mack.S.E`, with the factors'
  parameter error scaled by the residuals' variance `v` as without a tail,
  within 3.2 Monte Carlo standard errors; RAA 1990 and RAA's total are the
  widest (1.04 times R under the Gamma), the nonlinearity of RAA's young
  factors. R's `tail = TRUE` reconciles the same way when the bootstrap's
  tail is the constant R's rule fits: R's `tail.se` is the log-linear
  extrapolation of the factors' standard errors at the tail's position,
  which depends on the factor only.[^reference]
* **A refitted tail has Mack's process error.** The Gamma and no-process
  runs on one seed share every pseudo factor and refitted tail, so their
  per-simulation difference is the process error: its standard deviation
  is R's `Mack.ProcessRisk` with `tail = TRUE` within 2.5 standard errors.
* **A refitted tail's parameter error is not R's.** On the oldest origin
  (the tail step alone) the refit's standard deviation is 0.998 times its
  first-order (delta-method) value on ABC, 1.56 on RAA and 1.12 on GenIns,
  and its mean is above the plug-in (RAA +13%, GenIns -2%, ABC -0.1%):
  R's rule multiplies 100 extrapolated factors, a convex function of the
  fitted line, so the tail's sampling distribution is skewed where the
  extrapolation is long. The first-order standard error of the refitted
  factor is 0.0051 against R's extrapolated `tail.se` of 0.0044 (RAA),
  0.0099 against 0.0083 (GenIns) and 0.00068 against 0.00089 (ABC). In
  total the refitted bootstrap's standard deviation is 1.003 (RAA), 1.072
  (GenIns) and 1.011 (ABC) times R's `Mack.S.E` with `tail = TRUE`.
* **The ODP's refit moves more.** The ODP's late pseudo factors rest on
  small increments with the triangle's single scale, so a refitted
  log-linear tail varies far more than on Mack's pseudo link ratios: RAA's
  oldest origin's mean reserve is 316 against the plug-in 178 (+78%) and
  its total mean 1.064 times the tailed chain ladder's (1.033 without a
  tail); GenIns 1.017 (1.011); ABC 0.999 (0.999). A constant 1.05 keeps
  the ratio as without a tail (RAA 1.029, GenIns 1.010, ABC 0.999).

So the constant tail is the one to use to reproduce Mack's analytic tail
risk, and the estimated tail gives the tail estimator's own sampling
error, which R's extrapolated `tail.se` only approximates in order of
magnitude.[^design]

[^test]: `validation/tests/reserving_bootstrap_tail.rs` checks the
    constant tail against R (Gamma and parameter error alone), the refit's
    process error, the oldest origin's refit against the delta method on
    ABC and above it on RAA and GenIns, and R's `tail.se` within a factor
    of 2 of the first-order error. The total and ODP ratios were measured
    once and are not in CI.
[^reference]: `validation/reference/reserving_tails_r.csv`, from
    `validation/scripts/reserving_tails_r.R`.
[^design]: `docs/design/reserving-v02.md`, decision 9.
