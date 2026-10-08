---
type: Finding
title: The inverse Gaussian without e^(2λ/μ), through the Mills ratio
description: The textbook inverse Gaussian cdf Φ(a) + e^(2λ/μ) Φ(-b) overflows once λ/μ passes about 355, and its survival cancels in the tail (SciPy's invgauss does not overflow; prospicio matches it at λ/μ = 400). Since e^(2λ/μ) φ(b) = φ(a) exactly, F = Φ(a) + φ(a) R(b) and S = φ(a)(R(a) - R(b)) with the Mills ratio R; the first and second limited moments follow in the same terms. The survival still loses about log10(x/μ) digits far out, so layers a hundred means out match mpmath to 1e-9, not 1e-12.
tags: [probability, inverse-gaussian, severity, numerics, layer-moments, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-08T22:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/inverse_gaussian.rs
    title: InverseGaussian, its distribution function and limited moments
  - id: mills
    resource: ../crates/prospicio-math/src/special.rs
    title: mills_ratio, direct below 10 and by continued fraction above
  - id: script
    resource: ../validation/scripts/mpmath_layer_moments.py
    title: Layer means and second moments by 30-digit integration of the survival
---

# The identity

With `a = (x - μ)/μ √(λ/x)` and `b = (x + μ)/μ √(λ/x)`, `b² - a² = 4λ/μ`,
so `e^(2λ/μ) φ(b) = φ(a)` exactly. Writing `Φ(-t) = φ(t) R(t)` with the
Mills ratio `R`, the textbook distribution function becomes

* `F(x) = Φ(a) + φ(a) R(b)`, and for `a > 0`, `S(x) = φ(a) (R(a) - R(b))`.

Nothing overflows. The textbook form needs `e^(2λ/μ)`, which is `inf`
above `λ/μ ≈ 355` (a coefficient of variation below about 0.053). SciPy's
`invgauss` avoids that too (finite and correct at `λ/μ = 2000`), and
prospicio matches it to `1e-11` at `λ/μ = 400` in the parity rows.[^code]

# Limited moments

`x f(x) / μ` is the density of `1/Y` for `Y` inverse Gaussian with mean
`1/μ` and shape `λ/μ²`, whose `a` is `-a` and whose `b` is `b`, so

* `E[X; X > u] = μ (Φ(-a) + φ(a) R(b))`.

Integrating the derivative of `x^(p-1) e^(-(αx + β/x)/2)` by parts links
the half-integer orders and gives

* `E[X²; X > u] = (μ²/λ) (E[X; X > u] + λ S(u) + 2u² f(u))`,

which at `u = 0` is `μ³/λ + μ²`.[^code]

# Precision

`R(a) - R(b)` cancels when `b - a = 2√(λ/x)` is small next to `a`, which
loses about `log10(x/μ)` digits. Against 30-digit integration, every layer
matches to `1e-12` except those attaching 100 means out (values near
`1e-174`), which are within `3e-10`; the reference rows there carry
`1e-9`.[^script]

[^code]: `crates/prospicio-prob/src/inverse_gaussian.rs` and `mills_ratio` in `crates/prospicio-math/src/special.rs`.
[^script]: `validation/scripts/mpmath_layer_moments.py`, rows for `inverse_gaussian`.
