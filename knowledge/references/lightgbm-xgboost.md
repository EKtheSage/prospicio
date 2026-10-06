---
type: Reference Implementation
title: LightGBM and XGBoost (boosting engines)
description: The engines behind actuarialrs.boosting; how offsets, starting scores and precision behave, and which wheels to install.
resource: https://lightgbm.readthedocs.io/
tags: [boosting, lightgbm, xgboost, python, offsets]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T03:30:00Z }
verified: { by: process:ci, at: 2026-10-06T03:31:35Z }
sources:
  - id: adapter
    resource: ../python/actuarialrs/boosting.py
    title: actuarialrs.boosting
  - id: tests
    resource: ../python/tests/test_boosting.py
    title: Boosting adapter tests
  - id: lgb
    resource: https://pypi.org/project/lightgbm/
    title: lightgbm 4.7.0 on PyPI
  - id: xgb
    resource: https://pypi.org/project/xgboost-cpu/
    title: xgboost-cpu 3.2.0 on PyPI
---

# Offsets and starting scores

* The offset (log exposure for a frequency) goes in as LightGBM's
  `init_score` and XGBoost's `base_margin`, on the link scale; predictions
  then need the offset added back (`raw_score=True` /
  `output_margin=True`) before the inverse link.[^adapter]
* With an offset given, the trees start from the offset alone: a log-link
  model of responses near 150 starts at a mean of 1 and 100 rounds at
  learning rate 0.05 leave it about 8% short. The adapter adds
  `log(Σ w y / Σ w e^offset)` (the identity link: the weighted mean of
  `y − offset`) to the offset and keeps it in the fit.[^tests]

# Precision

XGBoost computes margins in single precision: doubling every exposure
doubles its means only to about 3e-6 relative, where LightGBM's agree to
the 1e-9 the test asks.[^tests]

# Installing

* `xgboost-cpu` (5.6 MB manylinux x86-64 wheel) installs the same
  `xgboost` module as `xgboost`, whose Linux wheel bundles over 200 MB of
  GPU libraries; `python/pyproject.toml` uses the CPU wheel on Linux.[^xgb]
* `lightgbm` 4.7.0 is a 3.5 MB wheel.[^lgb]

[^adapter]: actuarialrs.boosting
[^tests]: Boosting adapter tests
[^lgb]: lightgbm on PyPI
[^xgb]: xgboost-cpu on PyPI
