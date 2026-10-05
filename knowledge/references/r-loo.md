---
type: Reference Implementation
title: R loo (PSIS-LOO and stacking reference)
description: The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC; its stacking_weights stops early.
resource: https://mc-stan.org/loo/
tags: [bayes, elpd, stacking, parity, r]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
sources:
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
---

# Facts

* `act_bayes::elpd` matches loo 2.6.0 (`validation/scripts/r_loo.R`):
  every estimate, pointwise ELPD to 1e-9, k̂ to 1e-6.[^models-design]
* `loo::stacking_weights` stops its optimizer early and agrees with the
  exact stacking optimum only to about 1e-3. The stacking reference is
  therefore SLSQP (as BayesBlend uses) polished by Newton's method, to
  1e-10 (`validation/scripts/stacking_weights.py`).[^models-design]

See [BayesBlend](/references/bayesblend.md).

[^models-design]: Design note, models
