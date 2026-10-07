---
type: Reference Implementation
title: chainladder-python ClarkLDF
description: chainladder-python 0.10.1's ClarkLDF, the secondary reference for prospicio_reserving's Clark methods; what it shares with R ChainLadder and where its outputs are not comparable.
resource: https://github.com/casact/chainladder-python
tags: [reserving, clark, growth-curve, parity, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T18:00:00Z }
sources:
  - id: generator
    resource: ../validation/scripts/reserving_clark_python.py
    title: Clark chainladder-python reference generator
  - id: r-clark
    resource: r-chainladder-clark.md
    title: R ChainLadder ClarkLDF and ClarkCapeCod
---

# Facts

* One class, `ClarkLDF(growth)`, fits the LDF method, or Cape Cod when
  `fit` gets a `sample_weight` (exposure). It has no `maxage`, no
  `origin.width` and no standard errors.[^generator]
* Ages are `age - offset` with a fixed offset of half the development
  grain (6 months for annual development) and from-ages
  `max(age - interval, 0)`: R's `adol` ages whenever the origin and
  development grains are equal.
* The LDF likelihood sets each origin's ultimate to
  `latest / G(latest age)` and optimizes only `(omega, theta)`. On rows
  observed from the first age this is exactly the profiled maximum of R's
  likelihood,[^r-clark] so `omega_`, `theta_` and `scale_` (with `nobs - 2 -
  n_origins` degrees of freedom) estimate R's quantities.
* scipy's `minimize` with bounds (L-BFGS-B, default tolerances) on the
  unscaled likelihood stops about 1e-4 (relative) short of the maximum on
  GenIns and RAA; the parity rows carry rel_tol 1e-3.
* `ldf_` holds `G(a_{k+1}) / G(a_k)` for the triangle's ages only, so
  `Chainladder().fit(ClarkLDF().fit_transform(tri))` develops to the last
  age: R's and prospicio_reserving's `maxage` equal to the last age (120 months
  here).
* For Cape Cod, `omega_`, `theta_` and `elr_` estimate R's. `scale_` does
  not: it uses `incremental_fits_`, which are built from the latest
  diagonal times the curve (the LDF fit) rather than from `elr * exposure`,
  and so gives 53 301 on GenIns where R's `SIGMA2` is 61 147. The parity
  file leaves it out.

[^generator]: Clark chainladder-python reference generator
[^r-clark]: R ChainLadder ClarkLDF and ClarkCapeCod
