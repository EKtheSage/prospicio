---
type: Reference Implementation
title: statsmodels (GLM parity reference)
description: Python statsmodels, the reference for act-glm's GLMs, robust standard errors and Tweedie profiles; its conventions and quirks.
resource: https://www.statsmodels.org/
tags: [glm, parity, python, sandwich]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T16:34:15Z }
stale_after: 2027-10-05T00:00:00Z
sources:
  - id: models-design
    resource: ../docs/design/models.md
    title: Design note, models
  - id: robust
    resource: ../crates/act-glm/src/robust.rs
    title: act_glm::robust
  - id: scripts
    resource: ../validation/scripts/
    title: statsmodels_glm.py, statsmodels_glm_robust.py, statsmodels_tweedie_profile.py, statsmodels_fremtpl2.py
---

# Version

0.15.0 for every committed reference (`source` column of the CSVs).
Install with `pip install numpy statsmodels`; PyPI is reachable from the
cloud environment without allowlisting (see
[cloud network](/environment/cloud-network.md)).

# Conventions act-glm follows

* Log-likelihood and AIC as statsmodels reports them, with
  `var_weights` as prior weights; the Gaussian's log-likelihood uses the
  maximum-likelihood variance `deviance / n`.[^models-design]
* Quasi-Poisson: `fit(scale="X2")`, Pearson's dispersion.
* Sandwich bread: the inverse *observed* information, so non-canonical
  links match statsmodels (R's `sandwich` uses the expected information;
  the two agree for canonical links).[^models-design]

# Quirks

* statsmodels' GLM reports HC0 when asked for `cov_type="HC1"`, so HC1 is
  tested as HC0 rescaled by `n / (n - p)`.[^robust]
* Every reference is fitted with `tol=1e-14`, tighter than the
  default, so the parity tolerances (down to 1e-10) test act-glm rather
  than statsmodels' stopping rule.[^scripts]
* On [freMTPL2](/datasets/fremtpl2.md) (678k rows, 49 columns) a Poisson
  fit with the null model takes about 1 m 45 s.

[^models-design]: Design note, models
[^robust]: act_glm::robust
[^scripts]: statsmodels reference scripts
