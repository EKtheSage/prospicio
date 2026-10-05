---
type: Reference Implementation
title: R Pareto (tower matching reference)
description: The R package Pareto 2.4.5 (GPL), used only for reference values; where its tower matching departs from Riegel (2018).
resource: https://cran.r-project.org/package=Pareto
tags: [pareto, reinsurance, tower-matching, parity, r]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
sources:
  - id: pareto-design
    resource: ../docs/design/pareto.md
    title: Design note, Pareto-family severities
---

# Use

GPL: reference values only, never translated. Scripts:
`validation/scripts/r_pareto*.R`, `r_tower_matching.R`, `r_pml_curve.R`.

# Where it departs from the paper

* With the minimize rule it agrees with act-pricing to about 1e-8 on
  Riegel's Example 4.[^pareto-design]
* Its rule without minimization is not the paper's midpoint, so it is no
  reference for `SelectionRule::Midpoint`.[^pareto-design]
* On the tower 5m xs 5m, 15m xs 10m, ∞ xs 25m (losses 2.4m, 1.5m, 1.2m,
  f₁ = 1) its middle layer is not the minimum of the stated objective
  (spread 0.0703 where 0.0645 is attainable) and misses the layer's loss
  by 1.5e-5. act-pricing tests that tower for reproduction, scale
  invariance and local optimality instead.[^pareto-design]
* `fit_references` fills gaps differently from the package (a default
  alpha there, an analytic-centre completion here), so the two are not
  compared.[^pareto-design]

[^pareto-design]: Design note, Pareto-family severities
