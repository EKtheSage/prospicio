---
type: Finding
title: Reinsurer default on a lognormal year has closed forms in one integral over the normal score
description: With the annual gross loss lognormal and the defaults ranked on it, every expected credit loss, default frequency and joint default of Panel's one-factor model is one integral over the gross loss's normal score, split at the layers' and collaterals' kinks; 300,000 simulated years of Panel::apply and by_reinsurer agree with all 30 reference values within 2.1 standard errors.
tags: [reinsurance, credit-risk, reinsurer-default, vasicek, quadrature, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T00:00:00Z }
verified: { by: process:ci, at: 2026-11-24T00:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_reinsurer_default.py
    title: validation/scripts/aggregate_reinsurer_default.py (NumPy 2.5.4, SciPy 1.18.1)
  - id: test
    resource: ../validation/tests/aggregate_reinsurer_default.rs
    title: validation/tests/aggregate_reinsurer_default.rs
  - id: design
    resource: ../docs/design/reinsurer-default.md
    title: docs/design/reinsurer-default.md (Validation)
---

# Finding

* With `G = exp(m + s x)` and the year's rank `U = Phi(x)`, Panel's
  conditional probability of default
  `Phi((Phi^{-1}(p) - sqrt(rho) z) / sqrt(1 - rho))` with `z = -Phi^{-1}(U)`
  is `Phi((Phi^{-1}(p) + sqrt(rho) x) / sqrt(1 - rho))`, a function of the
  same `x` as the layers' ceded losses, so `E[credit loss] = (1 - R)
  E[(E(x) - C)^+ p(x)]` is one integral against the normal density. Over
  `x` rather than `u` the integrand has no endpoint singularity; split at
  each layer's attachment and exhaustion and where the exposure meets the
  collateral, `scipy.integrate.quad` holds every self-check (the lognormal
  limited expected value, `E[p(x)] = p`, the bivariate normal forms) to
  1e-10, and the weekly rerun reproduces the file under the pinned and
  the newest releases.[^script]
* The joint default of two reinsurers, `Phi_2(Phi^{-1}(p_1),
  Phi^{-1}(p_2); sqrt(rho_1 rho_2))`, and the default given the top 1 % of
  years, `Phi_2(Phi^{-1}(p), -Phi^{-1}(0.99); sqrt(rho)) / 0.01`: SciPy
  1.18.1's `multivariate_normal.cdf` in two dimensions is deterministic
  and agrees with the quadrature of `E[p_1(x) p_2(x)]` to 1e-16.[^script]
* The wrong-way effect is large at these parameters (gross `meanlog = 3`,
  `sdlog = 0.8`, layer 40 xs 35, `p = 0.1`, `R = 0.4`, collateral 5): the
  expected credit loss is 0.983 with `rho = 0.5` against 0.217 with
  independent default, 4.5 times; the probability of default in the worst
  1 % of years is 0.792 instead of 0.1; two reinsurers at `p = 0.1, 0.2`
  and `rho = 0.3, 0.6` default together with probability 0.0457 against
  0.02 independently.[^script]
* 300,000 simulated years with the panel applied to `Tower::apply_annual`
  match all 30 values within 2.1 standard errors at seed 2026 (within
  1.6 at seeds 7 and 31). The test runs in about 1 s in
  a debug build.[^test]
* Panels applied with the same seed draw the same uniforms by position: a
  reinsurer first on the panel draws the same default uniforms in every
  case, so independent-default cases with the same `p` default in exactly
  the same years, and their simulated errors are correlated across rows
  (the documented stream layout, not a fault).[^test]

[^script]: validation/scripts/aggregate_reinsurer_default.py, run 2026-11-24.
[^test]: validation/tests/aggregate_reinsurer_default.rs, `reinsurer_default_matches_the_closed_forms`.
