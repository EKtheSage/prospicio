---
type: Reference
title: pyvinecopulib 1.0.1, the reference for prospicio's vine copulas
description: The Python bindings of vinecopulib, against which prospicio_prob::vine is checked; its rotation and h-function conventions match prospicio's, but its C-vine order puts the root last and as each edge's second argument, and its inverse Rosenblatt transform runs through a D-vine's order backwards.
resource: https://vinecopulib.github.io/pyvinecopulib/
tags: [copulas, vines, reference-implementation]
status: stable
verified: { by: process:ci, at: 2026-10-14T23:59:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-14T23:59:00Z }
sources:
  - id: docs
    resource: https://vinecopulib.github.io/pyvinecopulib/
    title: pyvinecopulib documentation
  - id: tests
    resource: ../validation/tests/distributions.rs
    title: vines_match_pyvinecopulib
  - id: fit
    resource: ../validation/tests/distributions.rs
    title: pair_fits_and_the_student_pair_match_pyvinecopulib
---

pyvinecopulib 1.0.1 installs from PyPI (`uv run --with pyvinecopulib`)
and is the reference for `prospicio_prob::vine`
(`validation/scripts/pyvinecopulib_vines.py`).[^docs]

- **Pair copulas.** `Bicop(family, rotation, parameters)` with the
  rotations of `VineCopula` in R: 90 is `v − C(1 − u, v)`, 270 is
  `u − C(u, 1 − v)`. `hfunc2` is `∂C/∂v` (prospicio's `h2(u | v)`),
  `hfunc1` is `∂C/∂u` (`h1(v | u)`), and `hinv2`, `hinv1` their inverses.
  Transposing a rotated copula swaps 90 and 270.
- **D-vines.** `DVineStructure(order)` (1-based) has the same edges and
  argument order as prospicio's D-vine on the same order: each edge's
  first argument is its earlier variable. Their densities agree.
- **C-vines.** `CVineStructure(order)` puts the root of the first tree
  **last** in the order and makes it each edge's **second** argument,
  listing a tree's edges by the other variable. prospicio's C-vine on
  order `(a, b, c)` is pyvinecopulib's on `(c, b, a)`, with every pair
  transposed and each tree's edges reversed.
- **Inverse Rosenblatt.** `inverse_rosenblatt(w)` takes `w` by variable
  and conditions from the end of a D-vine's order, so it equals
  prospicio's transform of the same D-vine read backwards (pairs
  transposed). For a C-vine both start from the root.

- **Student t pairs.** `Bicop(BicopFamily.student, 0, [[rho], [nu]])`.
  Its pair functions round a non-integer `nu` to an integer (its bivariate
  t cdf is Genz's integer-`nu` formula), so prospicio's non-integer `nu`
  is checked against quadrature instead.[^fit]
- **Fitting.** `Bicop.fit` searches within fixed bounds (Clayton
  `(0, 28]`, Gumbel `[1, 50]`, Frank `[-35, 35]`, Joe `[1, 30]`, the t's
  `nu` `[2, 50]`), and `Bicop.select` by default chooses by AIC over the
  families given, trying the asymmetric families in the two rotations
  matching Kendall's tau; on a Clayton sample it can choose a
  rotated Joe, and `prospicio_prob::vine_fit::select_pair` agrees with it
  on all seven reference sets.[^fit]

On every family, each rotation and five points, and on three-dimensional
C- and D-vines with rotated (asymmetric) pairs, 512 values agree to
`1e-9` relative, inverses to `1e-7`.[^tests]

[^docs]: pyvinecopulib documentation
[^tests]: vines_match_pyvinecopulib
[^fit]: pair_fits_and_the_student_pair_match_pyvinecopulib
