---
type: Finding
title: "Permutation explanations: the interaction screen has no zero line, and partial dependence of an additive log-link GLM is its relativities"
description: "On a Poisson portfolio with a planted young-driver high-power effect, the interaction screen L_A + L_B - L_AB of a Poisson GLM without the interaction term is about -0.025 (20 shuffles), not 0, because the deviance is not additive on the log scale; with the term it is about +0.02. With one shuffle per split the planted pair's strength ranged from -0.034 to +0.062 over five seeds, so the screen needs repeats or folds and ranks pairs of one model rather than testing them against zero. A column the model does not use has importance exactly 0 and adds exactly 0 to any pair. Partial dependence ratios of a log-link GLM without interactions equal exp of its coefficients to 1e-12. permutation(3, 3) is [2, 0, 1], the monograph's shuffle of Tables 9.1 to 9.4 (importance 0.133, 100 on its scale)."
tags: [prospicio-models, explanations, permutation-importance, partial-dependence, interactions, monograph-16]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-20T22:00:00Z }
verified: { by: process:ci, at: 2026-10-20T22:00:00Z }
sources:
  - id: monograph
    resource: ../references/chalk-monograph-16.md
    title: CAS Monograph 16, chapter 9 (permutation importance, partial dependence, interactions)
  - id: python-tests
    resource: ../python/tests/test_models.py
    title: test_interaction_screen_finds_the_planted_pair and the other explanation tests
  - id: r-tests
    resource: ../R/prospicio/tests/test-models.R
    title: the same portfolio, shuffles and pinned values in R
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/355
---

# Permutation explanations

Found while building `permutation_importance`, `interaction_screen` and
`partial_dependence` (#355) in Python and R.

* **No zero line for the screen.** Monograph 16 reads `L_A + L_B - L_AB`
  as zero when neither feature interacts. That holds for a squared error
  on an additive model in expectation over shuffles. On a Poisson deviance
  the losses are not additive on the log scale: the test portfolio's
  additive GLM gives about -0.025 for (age, power) over 20 shuffles, the
  GLM with the planted term about +0.02. Compare pairs of one model with
  one another; do not test a pair against zero.
* **One shuffle is noisy.** With one shuffle the planted pair's strength
  over seeds 0 to 4 was -0.034, +0.062, -0.025, -0.002 and -0.025. Use
  `n_repeats` or folds; the monograph averages over seven folds.
* **Unused columns are exact zeros.** Shuffling a column the model never
  reads leaves every prediction unchanged, so its importance is exactly 0
  and each of its pairs has strength exactly 0, whatever the noise.
* **Partial dependence of an additive log-link GLM** is
  `exp(β)` times the baseline's mean prediction at each level, so the
  ratios are the relativities to `1e-12`, the identity behind the
  monograph's Tables 9.7 to 9.9.
* **The monograph's toy shuffle.** `permutation(3, 3)` is `[2, 0, 1]`
  (R `c(3, 1, 2)`), ages 40, 20, 30 as in Table 9.2, so the helpers
  reproduce Table 9.4's 0.133 and 100.
