---
type: Reference Implementation
title: R loo (PSIS-LOO and stacking reference)
description: The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC, and 2.10.1 for the LOO-PIT (E_loo of y_rep <= y under the PSIS weights; loo has no loo_pit of its own); its stacking_weights stops early.
resource: https://mc-stan.org/loo/
tags: [bayes, elpd, stacking, parity, r]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T11:00:00Z }
verified: { by: process:ci, at: 2026-10-16T11:00:00Z }
sources:
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
  - id: loo-pit
    resource: ../validation/scripts/r_loo_pit.R
    title: r_loo_pit.R, the LOO-PIT reference
---

# Facts

* `prospicio_bayes::elpd` matches loo 2.6.0 (`validation/scripts/r_loo.R`):
  every estimate, pointwise ELPD to 1e-9, k̂ to 1e-6.[^models-design]
* `loo::stacking_weights` stops its optimizer early and agrees with the
  exact stacking optimum only to about 1e-3. The stacking reference is
  therefore SLSQP (as BayesBlend uses) polished by Newton's method, to
  1e-10 (`validation/scripts/stacking_weights.py`).[^models-design]

* loo 2.10.1 has no `loo_pit` (that is in `rstantools` and `bayesplot`);
  the LOO-PIT is `E_loo(1{y_rep <= y}, psis(-log_lik, r_eff), type =
  "mean", log_ratios = -log_lik)$value`, which is how `rstantools::loo_pit`
  defines it for continuous data. `prospicio_bayes::ppc::loo_pit` matches
  it to 1e-9 on the normal model of `r_loo.R` with one replicated dataset
  per draw.[^loo-pit]
* In the cloud container loo is not preinstalled; `install.packages("loo",
  repos = "https://cloud.r-project.org")` into the user library installs
  2.10.1 with its dependencies in about a minute.[^loo-pit]

See [BayesBlend](/references/bayesblend.md).

[^models-design]: Design note, models
[^loo-pit]: `validation/scripts/r_loo_pit.R` and `loo_pit_matches_loo` in `validation/tests/models.rs`
