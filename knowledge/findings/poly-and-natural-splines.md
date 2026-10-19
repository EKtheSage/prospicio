---
type: Finding
title: R's poly is a three-term recurrence and ns spans the restricted cubic spline
description: "R's poly(x, degree) columns are the monic discrete orthogonal polynomials of the training values, p1 = x - alpha1 and p(k+1) = (x - alpha(k+1)) p(k) - (n(k+1)/n(k)) p(k-1) with alpha the weighted means of x and n the squared norms, each divided by its norm; storing alpha and norm2 reproduces predict(poly) on new data. A natural cubic spline with knots t1 < ... < tK is spanned by x and the restricted cubic (truncated power) terms d_j - d_(K-1), d_j(x) = ((x - t_j)^3+ - (x - t_K)^3+) / (t_K - t_j), the same space as splines::ns with those interior and boundary knots, so fits and predictions agree with ns while coefficients differ."
tags: [prospicio-models, design, splines, orthogonal-polynomials, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T09:00:00Z }
verified: { by: process:ci, at: 2026-10-19T09:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-models/src/design.rs
    title: orthogonal_poly, poly_columns and spline_columns
  - id: test
    resource: ../validation/tests/models.rs
    title: orthogonal_polys_and_natural_splines_match_r
  - id: script
    resource: ../validation/scripts/r_terms_spline.R
    title: R's poly, predict(poly) and lm(y ~ ns(x, knots))
---

# Facts

* `poly(x, degree)` computes its columns by a QR decomposition, but they
  equal the monic orthogonal polynomials of the training values divided
  by their norms. With `norm2 = [1, n, n_1, ...]` and `alpha_k` the mean
  of `x` weighted by `p_(k-1)^2`, the recurrence
  `p_(k+1) = (x - alpha_(k+1)) p_k - (n_(k+1) / n_k) p_(k-1)` gives both the
  training columns and `predict(poly, newx)` to 1e-12.
* `splines::ns(x, knots)` builds a B-spline basis and projects out the
  second derivatives at the boundary knots with a Householder QR; its
  coefficients depend on LINPACK's signs.
* The restricted cubic spline basis (Harrell's), `x` and
  `(d_j - d_(K-1)) / (t_K - t_1)^2` for `j < K - 1`, spans the same natural
  cubic splines (linear beyond the boundary knots, `K - 1` columns with the
  intercept apart). `lm(y ~ ns(x, knots = c(10, 25)))` and a Gaussian GLM on
  an intercept and this basis agree to 1e-9 in fitted values and in
  predictions at 0, 20 and 50, inside and outside the range.
* The `(t_K - t_1)^2` scale keeps the cubic terms on the scale of `x`.
