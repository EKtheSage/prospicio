---
type: Finding
title: Normal parameter draws bias a log-link predictive mean upward
description: Drawing beta from N(beta_hat, Sigma) and exponentiating raises each mean by exp(x'Sigma x / 2); MeanPreserving draws remove it exactly.
tags: [glm, odp, reserving, predictive-distribution, bias]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T15:54:30Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/111
    title: Issue 111, item 2
  - id: pr
    resource: https://github.com/EKtheSage/prospicio/pull/114
    title: PR 114, ODP GLM fixes
  - id: code
    resource: ../crates/prospicio-glm/src/lib.rs
    title: prospicio_glm::ParameterDraws
---

# Finding

With `η = xᵀβ`, `β ~ N(β̂, Σ)` and a log link,
`E[exp(η)] = exp(xᵀβ̂ + xᵀΣx / 2) = μ̂ · exp(xᵀΣx / 2)`. The ODP GLM's
simulated reserve was 7.2% above the Chain Ladder on GenIns and 0.4% on
ABC, though its point estimate equals the Chain Ladder exactly.[^issue]

# Resolution

`GlmFit::predict_distribution_with(.., ParameterDraws::MeanPreserving)`
shifts each row's linear predictor by `−xᵀΣx/2` (log link; identity is
unbiased; other links are refused rather than approximated), keeping the
rows' dependence. `Normal` stays the default; `Fixed` draws process noise
only. On [RAA](/datasets/raa.md) mean-preserving draws centre the reserve
on the Chain Ladder within 4 Monte Carlo SE.[^pr][^code]

[^issue]: Issue 111, item 2
[^pr]: PR 114
[^code]: prospicio_glm::ParameterDraws
