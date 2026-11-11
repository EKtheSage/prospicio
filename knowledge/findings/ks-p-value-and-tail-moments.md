---
type: Finding
title: SciPy's expect fails on heavy tails, and its kstwo is three methods stitched together
description: "scipy.stats' dist.expect(lambda x: x, lb=t, conditional=True) returns a wrong lognormal tail mean on large losses (quad does not converge over [t, inf)), so the reference tail moments substitute the survival probability: E[g(X) | X > t] = ∫ g(isf(e^-v)) e^-v dv / S(t) from v = -ln S(t), in unit and then wider pieces up to v near 700, where e^-v underflows. The two-sided Kolmogorov-Smirnov tail kstwo.sf(d, n) is Ruben and Gambino's closed forms at n d ≤ 1 and n d ≥ n - 1, twice Smirnov's exact one-sided tail at d ≥ 0.5 or where n d² > 4 (n ≤ 140) or ≥ 2.2 (n > 140), Durbin's matrix by Marsaglia, Tsang and Wang between, and an asymptotic series for large n; prospicio_math::kolmogorov ports the exact parts (to 1e-10 on 161 SciPy points) and uses Kolmogorov's limit where the matrix would pass 401 rows. With a fitted severity the p-value is optimistic."
tags: [prospicio-math, prospicio-prob, severity, goodness-of-fit, scipy]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-11T18:00:00Z }
verified: { by: process:ci, at: 2026-11-11T18:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-math/src/kolmogorov.rs
    title: kolmogorov_sf, kolmogorov_limit_sf
  - id: tails
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: GoodnessOfFit::ks_p_value, tail_moments
  - id: test
    resource: ../validation/tests/severity_fit.rs
    title: the_kolmogorov_distribution_is_scipys, ks_p_values_and_tail_moments_match_scipy
  - id: scripts
    resource: ../validation/scripts/scipy_fit_tails.py
    title: the SciPy reference for the p-values and tail moments
---

# The KS p-value and tail moments of a fit

Issue #805 asked for the KS p-value, the tail mean and the tail CV beside
each fitted severity's KS distance.

## SciPy's `expect` on a heavy tail

`dist.expect(lambda x: x, lb=t, conditional=True)` integrates `x f(x)`
over `[t, ∞)` with `quad`, which on a lognormal fitted to large losses
stops before the integral converges and returns a tail mean about 20%
too small. The reference script `scipy_fit_tails.py` substitutes the
survival probability `q = S(x) = e^-v` instead,
`E[g(X) | X > t] = ∫ g(isf(e^-v)) e^-v dv / S(t)` from `v = -ln S(t)`, in
pieces up to `v` near 700, where `e^-v` underflows; on a lognormal it
agrees with the closed form to 1e-9 (`lognormal_check`). The Rust side
computes the moments exactly from the layer moments,
`E[X | X > t] = t + stop_loss(t) / S(t)`. A moment that diverges (a Pareto
with alpha at most 1 for the mean, at most 2 for the CV; a loglogistic
with shape at most 2 for the CV) is infinite in both.

## `kstwo`

`scipy.stats.kstwo.sf(d, n)` follows Simard and L'Ecuyer (2011). Ruben and
Gambino give closed forms at the ends: `P(D_n ≤ d) = n!/n^n (2nd - 1)^n` for
`1/2 ≤ n d ≤ 1`, and `P(D_n > d) = 2 (1 - d)^n` for `n d ≥ n - 1`. Where the
two one-sided events cannot both happen (`d ≥ 0.5`), or where their joint
probability is below double precision (`n d² > 4` for `n ≤ 140`, `≥ 2.2`
above), the tail is twice Smirnov's exact one-sided tail. Between, Durbin's
matrix of side `2⌈nd⌉ - 1` raised to the `n`-th power gives the
distribution function exactly. `kolmogorov_sf` ports these to 1e-10 on
161 SciPy points (`kolmogorov_scipy.csv`) and, where the matrix would
pass 401 rows, uses Kolmogorov's limit at `√n d` where SciPy uses Pelz and
Good's series; the two differ by about 1e-3 at most there.

The p-value is the classical one: the severity's parameters were fitted to
the same losses, so the true p-value is smaller, and the help texts say so.
`GoodnessOfFit` uses `n` the total weight, rounded.
