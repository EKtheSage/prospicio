---
type: Finding
title: Tweedie conversions against aggregate 2.1.0
description: A Tweedie law with 1 < p < 2 spelled from its dispersion, its cv or a Poisson-gamma compound gives the same eight quantities as aggregate 2.1.0's TweedieConversion to 1e-13 (p0 = exp(-lambda) to 1e-13 times lambda); the power implied by a mean, cv and dispersion is 2 + ln(cv^2 / phi) / ln(mean), in (1, 2) only when cv^2 / phi lies between mean^-1 and 1 (mean above 1) and undefined at a mean of 1.
tags: [tweedie, conversion, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T19:00:00Z }
verified: { by: process:ci, at: 2026-11-09T19:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_tweedie.py
    title: validation/scripts/aggregate_tweedie.py (aggregate 2.1.0's TweedieConversion.from_dispersion, from_cv, from_compound)
  - id: code
    resource: ../crates/prospicio-prob/src/tweedie.rs
    title: prospicio_prob::Tweedie (from_cv, from_cv_dispersion, from_compound, cv, severity_mean, severity_cv, p0)
---

# Finding

* The reproductive Tweedie `Tw_p(μ, φ)` has variance `φ μ^p`, so its
  coefficient of variation is `√φ μ^(p/2 - 1)` and `from_cv` sets
  `φ = cv² μ^(2-p)`. As a compound Poisson the count has mean
  `μ^(2-p) / ((2 - p) φ)`, the gamma shape is `α = (2 - p)/(p - 1)` (so the
  severity's cv is `α^(-1/2)`) and the severity mean is `μ / λ`.[^code]
* Ten laws from `p = 1.05` to `1.95` and means from 0.07 to 1e6, each
  spelled by `from_dispersion`, `from_cv` or `from_compound`, give
  aggregate's eight quantities to 1e-13. `p0 = e^(-λ)` carries the rounding
  of `λ` magnified by `λ` (2e-13 at `λ = 250`), so its tolerance is 1e-13
  times `λ`.[^script]
* Given the mean, cv and dispersion, `φ μ^p = (cv μ)²` fixes
  `p = 2 + ln(cv² / φ) / ln μ`. At `μ = 1` the variance is `φ` at every
  power and the power is not identified, so it is refused; otherwise
  `p` lies in `(1, 2)` only if `cv² / φ` lies between `μ^(-1)` and 1 (for
  `μ > 1`; between 1 and `μ^(-1)` below 1). The round trip through it
  returns every reference law's power to 1e-12.

[^code]: crates/prospicio-prob/src/tweedie.rs.
[^script]: validation/scripts/aggregate_tweedie.py, run 2026-11-09.
