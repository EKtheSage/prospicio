---
type: Finding
title: A density for the over-dispersed Poisson's log score
description: The ODP's lattice probability is -inf off the lattice; its continuation through Gamma, normalized by C(lambda), is a proper density in y.
tags: [glm, odp, log-score, stacking, numerics]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T15:54:30Z }
sources:
  - id: family
    resource: ../crates/prospicio-models/src/family.rs
    title: prospicio_models::Family::log_density, continuous_poisson_mass
  - id: scipy
    resource: ../validation/scripts/scipy_family_scores.py
    title: scipy_family_scores.py
---

# Finding

* The ODP's response is `(φ/w) N` with `N ~ Poisson(λ)`, `λ = wμ/φ`. Real
  increments are almost never multiples of φ/w, so the lattice log
  probability is −∞, and as a mass it is not comparable with continuous
  models' densities.
* `f(y) = (w/φ) λᴺ e^(−λ) / (Γ(N + 1) C(λ))`, `N = wy/φ`, with
  `C(λ) = ∫₀^∞ λˣ e^(−λ)/Γ(x + 1) dx`, is a proper density. On the lattice
  it is the count's probability times `w/(Cφ)`.[^family]

# Numerical facts about C(λ)

| λ | 1 − C(λ) |
|---|---|
| 0.001 | 0.847 |
| 1 | 0.166 |
| 5 | 2.2e-3 |
| 20 | 4.9e-10 |
| 40 | 3.8e-15 |

By Euler–Maclaurin the gap to 1 comes only from boundary terms in e^(−λ)
(the periodic remainder is of order e^(−2π²λ)), so C is taken as 1 from
λ = 40. Gauss–Legendre quadrature's own rounding grows with λ (about
2.6e-10 at λ = 10⁶), another reason to stop integrating there.
`scipy.integrate.quad` agrees to 1e-11.[^family][^scipy]

A negative y still has no density (−∞).

[^family]: prospicio_models::Family
[^scipy]: scipy_family_scores.py
