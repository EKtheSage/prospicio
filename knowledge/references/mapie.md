---
type: Reference
title: MAPIE 0.9.1, the reference for prospicio's conformal intervals
description: MAPIE's split ("base") and CV+ ("plus") conformal quantiles are the ceil((1 - alpha)(n + 1))-th order statistic, taken as numpy's "lower" quantile at that rank over n, clipped to the largest score rather than infinite; its MapieQuantileRegressor (CQR) takes numpy's "higher" quantile at (1 - alpha)(1 + 1/n), sometimes one order statistic higher.
resource: https://mapie.readthedocs.io/
tags: [conformal-prediction, mapie, reference-implementation]
status: stable
verified: { by: process:ci, at: 2026-10-29T20:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-29T20:00:00Z }
sources:
  - id: docs
    resource: https://mapie.readthedocs.io/
    title: MAPIE documentation
  - id: test
    resource: ../validation/tests/models.rs
    title: conformal_intervals_match_mapie
---

MAPIE 0.9.1 (with scikit-learn 1.5.2) is the reference for
`prospicio_models::conformal` (`validation/scripts/conformal_mapie.py`).[^docs]

- **Split and CV+.** `ConformityScore.get_quantile` takes the level
  `ceil(alpha_ref (n + 1)) / n`, clipped to `[0, 1]`, and numpy's
  `"lower"` quantile there, which is exactly the `ceil((1 - alpha)(n +
  1))`-th smallest score. When that rank exceeds `n` the clip gives the
  largest score; prospicio gives an infinite bound, as the theory does
  (MAPIE's `allow_infinite_bounds=True`). CV+'s lower bound is the same
  rule on the negated values, the `floor(alpha (n + 1))`-th smallest.
- **CQR.** `MapieQuantileRegressor.predict` takes
  `np.quantile(scores, (1 - alpha)(1 + 1/n), method="higher")`. Its
  virtual index `x - x/n` with `x = (1 - alpha)(n + 1)` rounds up to the
  conformal rank or one past it, depending on the fractional part of
  `x`; at `n = 200`, `alpha = 0.1` the two agree, which is the parity
  case.
- **Folds.** With `cv=KFold(5)` (no shuffle) or `LeaveOneOut()`, MAPIE's
  fold models are the same scikit-learn fits on the same rows as a
  script's own, so their out-of-fold and test predictions can be handed to
  prospicio exactly.

On these, prospicio's intervals agree with MAPIE's to `1e-12`.[^test]

[^docs]: MAPIE documentation
[^test]: conformal_intervals_match_mapie
