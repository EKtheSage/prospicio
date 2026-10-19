---
type: Finding
title: An ogive's limited moments are exact by the trapezoid and Simpson rules
description: "An ogive's distribution function is linear on each band, so its survival function is too: the trapezoid rule is exact for lev and stop_loss, Simpson's rule for a layer's second moment 2 ∫ (x - a) S, and the raw moments are those of uniform bands, E[X^j] = sum of dF (b^(j+1) - a^(j+1)) / ((j + 1)(b - a)). The lower quantile at the F of an empty band is that band's left end. The normal's TVaR mu + sigma phi(z_p) / (1 - p) is computed as mu + sigma / R(z_p) above the median, R the Mills ratio, which keeps its precision as p approaches 1."
tags: [prospicio-prob, ogive, grouped-data, normal, tvar, pricing-insurance-risk]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T13:00:00Z }
verified: { by: process:ci, at: 2026-10-19T13:00:00Z }
sources:
  - id: ogive
    resource: ../crates/prospicio-prob/src/ogive.rs
    title: prospicio_prob::Ogive and its unit tests
  - id: normal
    resource: ../crates/prospicio-prob/src/normal.rs
    title: prospicio_prob::Normal and its unit tests
  - id: book
    resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
    title: Mildenhall and Major, Pricing Insurance Risk (2022), section 4.3.3 and Exercise 50
---

# Facts

* On the band `[x_{k-1}, x_k]` an ogive's `F`, and so `S = 1 - F`, is
  linear. `∫ S` over any part of a band is exact by the trapezoid rule, and
  `∫ (x - a) S` (a quadratic) by Simpson's rule, so `lev`, `stop_loss` and
  `layer_second_moment` are exact up to rounding once the integration is
  split at the band boundaries. The unit tests check the layer's second
  moment against a 200,000-step midpoint sum to `1e-8`.
* The raw moments are those of a mixture of uniforms: `E[X^j] = Σ ΔF_k
  (x_k^(j+1) - x_{k-1}^(j+1)) / ((j + 1) Δx_k)`.
* The lower quantile, the smallest `x` with `F(x) ≥ p`, sits at the left
  end of a band with no probability when `p` equals that band's `F`
  (`F = 0, 0.5, 0.5, 1` at `0, 10, 20, 30`: `VaR(0.5) = 10`).
* With 40, 50 and 10 claims in 0–1,000, 1,000–5,000 and 5,000–10,000,
  `VaR(0.95) = 7,500` and `TVaR(0.95) = 8,750`: the worst 5% are uniform on
  7,500 to 10,000.
* The normal's TVaR is `μ + σ φ(z_p) / (1 − p)`; above the median the
  multiple is `1 / R(z_p)` with `R = (1 − Φ) / φ` the Mills ratio, which
  avoids dividing two numbers that both underflow toward 0 as `p → 1`.
  At `p = 0.99` it is `2.665214 σ` (`VaR = 2.326348 σ`).

# Why

*Pricing Insurance Risk* reads VaR and TVaR off grouped data (Exercise 50)
and states the normal's TVaR as a fixed multiple of `σ` (section 4.3.3);
the guide `pir-risk-measures.md` worked around both with plain Python
(issue #366).
