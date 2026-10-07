---
type: Finding
title: Partial pooling over too few covariates gives a funnel
description: With one slope per pooled group, the group scale and the slope trade off; NUTS diverges. Three or more covariates mix well.
tags: [bayes, stacking, pooling, nuts, divergences]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T09:42:58Z }
sources:
  - id: stacking
    resource: ../crates/prospicio-bayes/src/stacking.rs
    title: prospicio_bayes::stacking tests
  - id: bayesblend
    resource: /references/bayesblend.md
    title: BayesBlend
---

# Finding

* Pooled hierarchical stacking with one discrete and one continuous
  covariate (one slope per group), adaptive priors, 2 × 400 draws: 108
  divergences in 800 and R̂ up to 1.06.
* Four regions as three dummies plus one continuous covariate (400
  observations, 2 × 600 draws), target acceptance 0.95: R̂ ≤ 1.006 and no divergences without adaptation,
  R̂ ≤ 1.015 and 8 divergences with it.[^stacking]
* This is the model's geometry (β = μ + σδβ′ with σ unidentified from a
  single slope), the same as BayesBlend's Stan model, which warns below
  three covariates.[^bayesblend]

# Practice

Pool only over three or more covariates, raise `target_accept`, and read
the divergences that `StackingFit` reports.

[^stacking]: prospicio_bayes::stacking tests
[^bayesblend]: BayesBlend
