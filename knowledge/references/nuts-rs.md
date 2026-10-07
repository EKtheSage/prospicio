---
type: Reference Implementation
title: nuts-rs (NUTS sampler)
description: nutpie's Rust core, nuts-rs 0.19 (MIT), the sampler behind prospicio_bayes::nuts, BayesGlm and stacking; how prospicio-bayes drives it.
resource: https://crates.io/crates/nuts-rs
tags: [bayes, mcmc, nuts, dependency]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T09:27:28Z }
sources:
  - id: nuts-module
    resource: ../crates/prospicio-bayes/src/nuts.rs
    title: prospicio_bayes::nuts
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
---

# How prospicio-bayes drives it

* One `CpuLogpFunc` adapter over the public `LogDensity` trait;
  `FlowParameters = ()`, `ExpandedVector = Vec<f64>`. Other crates sample
  through `prospicio_bayes::nuts::sample` without depending on nuts-rs.[^nuts-module]
* Chain `c` keys a ChaCha20 generator from `StreamRng(seed, c)`, so runs
  replay exactly; chains run in parallel on Rayon.[^nuts-module]
* A recoverable `LogpError` marks a position outside the support; the
  sampler counts it as a divergence and moves on.
* Starts are jittered by up to ±0.01 per coordinate.[^nuts-module]

# Facts

* A correlated normal, a half-line support and a gamma-Poisson posterior
  predictive (against the exact negative binomial) are its tests.[^models-design]
* No Python or R binding for user densities: a density written in the
  interpreter would be called back at every gradient.

[^nuts-module]: prospicio_bayes::nuts
[^models-design]: Design note, models
