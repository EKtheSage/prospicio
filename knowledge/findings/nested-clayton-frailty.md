---
type: Finding
title: A nested Clayton copula's inner frailty is a tilted stable that plain rejection cannot draw
description: The inner frailty of a nested Clayton copula has Laplace transform exp(−V((1 + t)^α − 1)); rejection from the scaled stable accepts with probability e^−V, which collapses when the root θ is small, so the frailty is split into ⌈V⌉ parts each accepted with probability at least e^−1.
tags: [copulas, simulation, numerics]
status: stable
verified: { by: process:ci, at: 2026-10-14T23:30:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-14T23:30:00Z }
sources:
  - id: tests
    resource: ../crates/prospicio-prob/src/copula.rs
    title: the nested Archimedean unit tests
  - id: mcneil
    resource: https://doi.org/10.1080/00949650701255834
    title: McNeil (2008), Sampling nested Archimedean copulas
---

A nested Archimedean copula is sampled by McNeil's recursion: the root's
frailty `V`, then for each child an inner frailty whose Laplace transform
is `exp(−V ψ⁻¹(ψ_child(t)))`.[^mcneil] With the child's `θ` at least the
parent's (the sufficient nesting condition), `α = θ/θ_child ≤ 1`.

- **Gumbel.** `ψ⁻¹(ψ_child(t)) = t^α`, so the inner frailty is
  `V^(1/α) S` with `S` positive stable of index `α`: a closed form.
- **Clayton.** `ψ⁻¹(ψ_child(t)) = (1 + t)^α − 1`, the exponentially
  tilted stable. Drawing `S` from the scaled stable (Laplace transform
  `exp(−V t^α)`) and accepting with probability `e^−S` is exact, but its
  acceptance rate is `E[e^−S] = e^−V`. The root's frailty is a gamma
  with shape `1/θ`, so a root `θ` of 0.2 gives `V` near 5 and an
  acceptance near 0.7%.

Splitting the frailty into `m = ⌈V⌉` parts of size `V/m` and summing `m`
independent tilted stables keeps the Laplace transform (it multiplies)
and accepts each part with probability at least `e^−1`, so a draw costs
`O(V)`. At `V = 3.7`, `α = 0.5` the draws' Laplace transform matches
`exp(−V((1 + t)^α − 1))` within `4e-3` at `t = 0.3, 1, 4` on 100,000
draws.[^tests]

[^mcneil]: McNeil (2008)
[^tests]: prospicio_prob::copula tests
