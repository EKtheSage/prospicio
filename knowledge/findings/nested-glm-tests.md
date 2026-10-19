---
type: Finding
title: R's anova tests of nested GLMs scale by the larger model's dispersion
description: "R's anova(small, big, test = 'F') divides the deviance drop by q times the larger model's dispersion (Pearson's estimate by default) and refers it to F(q, n - p_big); test = 'Chisq' refers the drop over that dispersion to chi2(q). With one extra coefficient the F statistic is the square of its t statistic. R's BIC counts an estimated dispersion as a parameter while prospicio's GlmFit::bic, like its aic, counts only the coefficients, so they agree for the Poisson and binomial. Variance inflation factors are the diagonal of the inverse correlation matrix of the non-constant columns, 1 / (1 - R^2) of each on the others with an intercept."
tags: [prospicio-glm, glm, anova, f-test, bic, vif, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T08:00:00Z }
verified: { by: process:ci, at: 2026-10-19T08:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-glm/src/nested.rs
    title: f_test and likelihood_ratio_test
  - id: test
    resource: ../validation/tests/models.rs
    title: nested_glm_tests_match_r_anova
  - id: script
    resource: ../validation/scripts/r_glm_nested.R
    title: R's anova, BIC and lm-based VIF on quine
---

# Facts

* `anova(small, big, test = "F")`: `F = (D_small - D_big) / (q phi_big)`
  on `F(q, n - p_big)`, with `phi_big` the larger model's dispersion
  (Pearson's by default). On quine, `log(Days + 1)` on Eth and Sex against
  the full model: F = 1.2950554093611, p = 0.274969956093846 (q = 4,
  139 residual degrees of freedom).
* `test = "Chisq"`: the drop over `phi_big` on `chi2(q)`; p = 0.26930041235005
  on the same models, and the Poisson model of `Days` drops 178.38 in
  deviance, p = 1.66e-37.
* The upper tail of F is `I_{d2 / (d2 + q f)}(d2 / 2, q / 2)`, the
  regularized incomplete beta, so no F distribution is needed.
* With one extra coefficient the F statistic is that coefficient's squared
  t statistic and the p-values agree.
* R's `BIC` counts an estimated dispersion; prospicio's `bic`, as its
  `aic` (statsmodels' convention), counts the coefficients only. They agree
  for the Poisson: 2320.06887663732 on quine.
* The VIF of a column is the diagonal of the inverse of the correlation
  matrix of the non-constant columns, which is `1 / (1 - R^2)` of that
  column on the others with an intercept (`car::vif` for single-column
  terms).
* The area under the ROC curve by the trapezoid rule, ties counting half,
  is the Mann-Whitney probability that a positive outscores a negative
  (R `wilcox.test`'s statistic over the number of pairs), and for a 0/1
  response with unit exposure it is `(1 + normalized Gini) / 2`, the
  Gini computed with ties grouped (CAS Monograph 5, footnote 17); the
  unit tests check both identities.
