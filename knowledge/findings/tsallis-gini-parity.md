---
type: Finding
title: The Tsallis distortion and the Gini index against aggregate 2.1.0
description: prospicio's Tsallis distortion, the Gini index of every family, from_gini and the extended calibration set match aggregate 2.1.0 on 113 values (g to 1e-15, Gini to 1e-11, from_gini to 1e-12, InsCo's extended calibration within aggregate's premium error); the stable forms for g near alpha = 1 and for 1 - g near s = 1, and the exponential family's Gini near k = 0, are what keep them accurate.
tags: [distortion, tsallis, gini, calibration, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T18:00:00Z }
verified: { by: process:ci, at: 2026-11-09T18:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_distortion_gini.py
    title: validation/scripts/aggregate_distortion_gini.py (aggregate 2.1.0's Distortion, gini, from_gini, calibrate_distortions(extended=True))
  - id: code
    resource: ../crates/prospicio-prob/src/distortion.rs
    title: prospicio_prob::distortion (Tsallis, gini, from_gini, calibrate_all, Family::EXTENDED)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, spectral.py (TsallisDistortion, Distortion.gini, Distortion.from_gini) and _pricing.py (EXTENDED_CALIBRATION_DISTORTIONS)
---

# Finding

* `g(s) = (α s − s^α)/(α − 1)` cancels as `α → 1`. Written as `s − t/β`
  with `t = s·expm1(β ln s)` and `β = α − 1` (aggregate's `_g_alpha`) it is
  continuous through `α = 1`, where it is `s (1 − ln s)`, to 1e-8 at
  `α = 1 ± 1e-9`, and stays in `[s, 1]` for orders from 1e-6 to 1e6.[^code]
* `1 − g(s)` near `s = 1` is `α t²/2 (1 − (α − 2) t/3 + ...)` with
  `t = 1 − s`; the series (the factor `α − 1` cancels from every term)
  keeps its relative accuracy where `1 − g(s)` itself loses all of it,
  which matters for the capital per unit of margin `(1 − g)/(g − S)`.
* The Gini index `2 ∫ g − 1` has a closed form for every one-parameter
  family (aggregate's `p_equiv`) and for the capped, beta, TVaR-mixture and
  convex families. The exponential family's `2(1 − m/k)/m − 1`, `m = 1 −
  e^{−k}`, loses half its digits at `k = 1e-6`; `coth(k/2) − 2/k`, with
  its series `k/6 − k³/360 + k⁵/15120` below `k = 0.01`, does not. `Lep`,
  `LinearYield` and `Minimum` are integrated by adaptive Gauss–Legendre on
  dyadic panels toward 0, within 1e-12 of the closed forms where both
  exist and 1e-11 of aggregate's `quad`.
* `from_gini` inverts the one-parameter families; `ccoc`'s Gini is its
  discount `d = r/(1 + r)`, so `r = G/(1 − G)`.
* Tsallis's price falls as `α` grows without bound, so it is calibrated by
  bisection on `G = 1/(α + 1)` in `(0, 1)`. On Monograph 15's InsCo at a
  15% cost of capital it gives `α = 3.0691` (Gini 0.2458, between Wang's
  0.1915 and dual's 0.2293 by aggregate's ordering), as aggregate's
  `calibrate_distortions(..., extended=True)` does.[^script]

[^code]: crates/prospicio-prob/src/distortion.rs, unit tests.
[^script]: validation/scripts/aggregate_distortion_gini.py, run 2026-11-09.
