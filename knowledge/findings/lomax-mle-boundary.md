---
type: Finding
title: A Lomax fit runs to the exponential when the losses' CV is at most 1
description: "The Lomax S(x) = (1 + x/theta)^(-alpha) fitted by maximum likelihood to untruncated losses solves alpha = n / sum ln(1 + y/theta) and (alpha + 1) sum y/(theta + y) = n. The Lomax's CV exceeds 1 wherever it is finite, and on losses whose sample CV is at most 1 the likelihood rises along alpha, theta -> infinity with theta/alpha fixed at the mean, the exponential limit, so the simplex stops at a large alpha with huge standard errors. prospicio's unit, Python and R tests check the two equations to 1e-7 on heavier losses, and R's optim on the same likelihood to 1e-4."
tags: [prospicio-prob, severity, lomax, pareto, maximum-likelihood]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T15:20:00Z }
verified: { by: process:ci, at: 2026-10-19T15:20:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: Family::Lomax and the test untruncated_lomax_solves_its_score_equations
  - id: bahnemann
    resource: https://www.casact.org/sites/default/files/2021-02/02-Bahnemann.pdf
    title: Bahnemann, Distributions for Actuaries, CAS Monograph 2, chapter 3
---

# A Lomax fit runs to the exponential when the losses' CV is at most 1

Bahnemann's Pareto is the Lomax, `S(x) = (1 + x/θ)^(−α)`, the Burr with
`γ = 1`; prospicio fits it as `Family::Lomax` and returns a `Dist::Burr`.

Without truncation or censoring its log-likelihood
`n ln α − n ln θ − (α + 1) Σ ln(1 + y/θ)` has the score equations

- `α = n / Σ ln(1 + y/θ)`, and
- `(α + 1) Σ y / (θ + y) = n`.

A Lomax with a finite variance has a coefficient of variation
`sqrt(α / (α − 2)) > 1`. When the losses' own CV is at most 1 the two
equations have no finite root: the likelihood keeps rising along
`α, θ → ∞` with `θ / α` near the mean, the exponential limit. The fit
then stops at a large `α` with very large standard errors, as a Burr fit
does at its Weibull boundary; read `converged` and the standard errors.

The tests use ten losses from 50 to 12,000 (CV well above 1): the fit
gives `α ≈ 0.886` and `θ ≈ 372.06`, which R's `optim` on the same
likelihood reproduces.
