---
type: Finding
title: Correlated Ornstein-Uhlenbeck factors and their integrals step exactly as one Gaussian draw, and a year-end shock regressed on the year's increments keeps its correlation
description: Two OU factors driven by Brownian motions of correlation ρ, with their integrals over a step, are jointly Gaussian with covariances ρ ∫ f g of their kernels on the step (σ e^(-av), σ B_a(v), or 1 for the increment); a monthly step makes the noises nearly collinear, so the factor is a semidefinite Cholesky. An annual shock drawn as its regression on the year's Brownian increments plus independent noise has the given correlation with each increment, and with each factor's annual innovation ρ E(a) / √E(2a).
tags: [esg, inflation, short-rate, simulation, correlation]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-28T02:00:00Z }
verified: { by: process:ci, at: 2026-11-28T02:00:00Z }
sources:
  - id: note
    resource: ../docs/design/esg-dfa.md
    title: Design note, an economic scenario generator and multi-year DFA, PR 2 as built
  - id: test
    resource: ../validation/tests/aggregate_esg_inflation.rs
    title: Validation test of the ESG's inflation, run in every cargo test
---

# What was found

Over a step of length `Δ`, an Ornstein-Uhlenbeck factor `dx = -a x dt +
σ dW` moves to `x e^(-aΔ) + ∫ σ e^(-a v) dW` and its integral by
`x B_a(Δ) + ∫ σ B_a(v) dW`, with `v` the time before the step's end and
`B_a(v) = (1 - e^(-av)) / a`.[^note] Each noise is a Wiener integral of a
kernel, so the noises of two factors with correlated motions are jointly
Gaussian, with covariance `ρ ∫_0^Δ f(v) g(v) dv`. With `E(c) = (1 -
e^(-cΔ)) / c`:

* factor and factor: `σ_f σ_g E(a + b)`;
* factor `a` and integral `b`: `σ_f σ_g (E(a) - E(a + b)) / b`;
* integral and integral: `σ_f σ_g / (ab) (Δ - E(a) - E(b) + E(a + b))`;
* the increment and a factor: `σ E(a)`; and an integral: `σ (Δ - E(a)) / a`.

For one factor these are the bivariate formulas the ESG used before, and
its draws are the same numbers. At a monthly step the increment, factor
and integral kernels are nearly linearly dependent (`1`, `1 - av`, `v`),
so the strict Cholesky of `prospicio-math` can refuse the covariance by
rounding; a factor that takes a non-positive pivot as a zero column is
exact to rounding.

An annual shock `ε` meant to have correlation `c_j` with each year's
Brownian increment `w_j` (unit variance, correlations `C_ww`) is
`ε = cᵀ C_ww⁻¹ w + √(1 - cᵀ C_ww⁻¹ c) z`; then `Cov(ε, w) = c`, and since a
factor's annual innovation `x(y) - e^(-a) x(y - 1)` covaries with `w_j`
as `σ E(a)` times their correlation, `Corr(ε, innovation) = c_j E(a) /
√E(2a)` (one-year `E`). The test checks both factors' innovations at 20,000
paths.[^test]

[^note]: The design note, PR 2 as built.
[^test]: `the_superimposed_excess_is_an_ar1_correlated_with_the_year` in the validation test.
