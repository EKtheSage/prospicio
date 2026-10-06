---
type: Reference
title: MBBEFD exposure curves (Bernegger 1997)
description: The MBBEFD class behind act_pricing::exposure; its four closed-form cases, the Swiss Re c curves, and numerical points found while building it.
resource: https://doi.org/10.2143/AST.27.1.563208
tags: [pricing, exposure-rating, mbbefd, property, reinsurance]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T04:00:00Z }
sources:
  - id: bernegger
    resource: https://doi.org/10.2143/AST.27.1.563208
    title: Bernegger (1997), The Swiss Re exposure curves and the MBBEFD distribution class, ASTIN Bulletin 27(1)
  - id: code
    resource: ../crates/act-pricing/src/exposure.rs
    title: act_pricing::exposure
  - id: parity
    resource: ../validation/scripts/mpmath_mbbefd.py
    title: mpmath_mbbefd.py
---

# The class

* Destruction rate `X = loss / MPL` on `[0, 1]`; `b ≥ 0`, `g ≥ 1`; a total
  loss has probability `1/g`.[^bernegger]
* Exposure curve `G(x) = E[min(X, x)] / E[X]`, with four closed forms:
  linear (`g = 1` or `b = 0`), `b = 1`, `bg = 1`, and the general
  `ln(((g − 1)b + (1 − gb)bˣ)/(1 − b)) / ln(gb)`.
* Swiss Re curves: `b = exp(3.1 − 0.15(1 + c)c)`, `g = exp((0.78 +
  0.12c)c)`; `c = 1.5, 2, 3, 4` are Y1–Y4, `c = 5` the Lloyd's curve, `c = 0`
  the straight line.[^bernegger]
* Layer `l xs a` on a risk with MPL `M`: share
  `G(min((a + l)/M, 1)) − G(min(a/M, 1))` of the risk's expected loss.

# Numerical points

* The general form cancels near `b = 1` and `bg = 1`; act-pricing switches
  to the special forms within 1e-10 of them. At a relative distance 1e-6 the
  general form still agrees with the limit to 1e-5.[^code]
* `G(0)` in the general case is `ln` of a ratio equal to 1, so it comes out
  as a rounding error near 0, not exactly 0; compare with a tolerance.
* The closed forms agree with 30-digit quadrature of the survival function
  to 1e-12 for c = 1.5–5 and one curve per case (56 values).[^parity]

[^bernegger]: Bernegger (1997)
[^code]: act_pricing::exposure
[^parity]: mpmath_mbbefd.py
