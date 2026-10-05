---
type: Finding
title: The quasi-Poisson ODP can fit negative increments
description: A quasi-likelihood ODP needs only V(mu) = mu and mu > 0; negative responses are fitted with a quasi-deviance that is not a distance.
tags: [glm, odp, reserving, quasi-likelihood]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T15:54:30Z }
sources:
  - id: pr
    resource: https://github.com/EKtheSage/risk-rs/pull/114
    title: PR 114, ODP GLM fixes
  - id: family
    resource: ../crates/act-models/src/family.rs
    title: act_models::Family::unit_deviance
---

# Finding

* The quasi-score equations `Σ w (y − μ)/μ · x = 0` under a log link are
  well posed for any finite y as long as μ > 0; for the log link the
  objective `μ − y ln μ` is convex in η whatever the sign of y.
* R's `quasipoisson` still refuses negative y.

# Resolution

* A Poisson GLM with an estimated dispersion (`Glm::accepts`) takes
  negative y. Its unit deviance there is the quasi-deviance
  `2 (y ln(|y|/μ) − (y − μ))`: right derivative in μ, so the estimates are
  the quasi-likelihood ones, but no saturated model exists and it can be
  negative. The log-likelihood of such a row is NaN.[^family]
* IRLS starts a negative row at the overall mean.
* A Poisson with φ fixed at 1 still refuses negative y.[^pr]

See [RAA](/datasets/raa.md).

[^family]: act_models::Family::unit_deviance
[^pr]: PR 114
