---
type: Finding
title: Wilson-Hilferty is the shifted gamma's cube-root normal approximation
description: "The normal power approximation of a compound sum puts its p quantile at m + s (z + g (z^2 - 1)/6) and Wilson-Hilferty at m + s (2/g) ((1 + g z/6 - g^2/36)^3 - 1), with m, s and g the mean, standard deviation and skewness and z the normal quantile. Both are defined by their quantiles; their distribution functions invert them, and each has a lower end (m - s (3/(2g) + g/6) and m - 2s/g). Wilson-Hilferty is the cube-root normal approximation of the shifted gamma with the same three moments, within 0.3% of its quantiles to the 99.9th at skewness 0.5 (Poisson 10, gamma claims with mean 50 and cv 0.7)."
tags: [prospicio-aggregate, approximations, normal-power, wilson-hilferty, bahnemann]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T10:00:00Z }
verified: { by: process:ci, at: 2026-10-19T10:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/moments.rs
    title: Approximation::NormalPower and WilsonHilferty
  - id: book
    resource: https://www.casact.org/monograph/cas-monograph-no-2
    title: Bahnemann, Distributions for Actuaries (CAS Monograph 2), section 4.4
---

# Facts

* Normal power: `x_p = m + s (z + g (z^2 - 1) / 6)`. The quadratic in `z`
  turns at `z = -3/g`, so the distribution function, which inverts it as
  `Phi(-3/g + sqrt(9/g^2 + 1 + 6 y / g))` with `y = (x - m)/s`, is 0
  below `m - s (3/(2g) + g/6)`.
* Wilson–Hilferty: `x_p = m + s (2/g) ((1 + g z/6 - g^2/36)^3 - 1)`; its
  distribution function is
  `Phi((6/g) ((1 + g y/2)^(1/3) - 1 + g^2/36))`, 0 below `m - 2 s / g`.
* A gamma with shape `a` has skewness `2/sqrt(a)`, and Wilson and
  Hilferty's cube-root transformation of it is nearly normal; the formula
  above is that transformation for the shifted gamma with the same mean,
  variance and skewness. On Poisson 10 claims of a gamma with mean 50 and
  cv 0.7 (skewness 0.513) the two agree to 0.3% at the 50th to the 99.9th
  percentiles.
* Neither is in the Python package `aggregate`; both need a positive
  skewness here, as the shifted fits do.
