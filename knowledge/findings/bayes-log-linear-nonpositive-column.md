---
type: Finding
title: The log-linear Bayesian reserving models refuse a development column that sums to zero or less, which the chain ladder accepts
description: BayesOdp, BayesBf and BayesBfNormalised all refuse a triangle whose development column's observed increments sum to zero or less (one late cell below the previous one is enough), because every mean is a positive product and the likelihood then has no finite maximum; the chain ladder takes the same triangle with a factor below 1. Rebuilding the triangle from its long table without the age is the remedy at the first or the last age only (at the first it also removes the origins observed only there); at a middle age the rebuilt triangle keeps the age with no cells and the chain ladder refuses it; no Triangle method drops ages.
tags: [bayes, reserving, bornhuetter-ferguson, odp, negative-development, errors]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
verified: { by: process:ci, at: 2026-10-28T12:00:00Z }
sources:
  - id: normalised
    resource: ../crates/prospicio-reserving/src/bayes/bf_normalised.rs
    title: prospicio_reserving::bayes::BayesBfNormalised (prepare, column_error, origin_error and the test a_falling_late_column_names_its_age_and_the_remedy)
  - id: odp
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: prospicio_reserving::bayes::BayesOdp (the column and origin check of its model)
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 26 and open decision 18)
---

# Finding

* On a four-origin triangle whose oldest age, 48 months, has one cell 5
  below the cell at 36 months, the volume-weighted chain ladder fits with a
  factor of 155 / 160 = 0.9688 into 48 months, while `BayesOdp`, `BayesBf`
  and `BayesBfNormalised` each return `Error::Bayes` naming that age.[^normalised]
  Issue #610 said `BayesOdp` handled the triangle; it does not.[^odp]
* The cause is the model, not the sampler: the means are positive
  products `exp(a_w + b_d)`, so the score in `b_d` is
  `(S_d - sum_w exp(a_w + b_d)) / phi`, negative everywhere when the
  column's sum `S_d` is not positive. The likelihood rises as the age's
  share falls to 0 and has no finite maximum. Under `BayesBfNormalised`'s
  vague normal prior on `beta_d` the posterior is still proper, with a mode
  near `development_sd^2 S_d / phi` and so a share of about 0 set by the
  prior alone, which misstates an age whose movement is negative.[^design]
* An origin row is different: under a flat prior (`cv` infinite) it has no
  log-linear mean either, but under an informative prior its ultimate's
  posterior is a gamma of shape `1 / cv^2 + S_w / phi`, proper for any
  `cv` below `sqrt(-phi / S_w)`; the error prints that bound.[^normalised]
* No `Triangle` method drops an age (`select`, `select_columns`,
  `group_by` and `grain` do other things), so the remedy is to rebuild the
  triangle from its long table without the rows at that age (Rust
  `to_long` and `from_long`, Python `to_frame` and `from_frame`, R
  `as.data.frame()` and `triangle()`); the rebuilt triangle keeps the
  other origins' latest values and fits. A `Triangle` method is open
  decision 18.[^design]
* That rebuild works only at the first and the last age.
  `Triangle::from_long` fills every age from the first observed one to
  the last, so on a four-origin triangle whose middle age, 36 months, sums
  to -15, the triangle without the rows at 36 months still has ages 12,
  24, 36 and 48, with 36 empty, and the chain ladder refuses it (no origin
  observed at both 24 and 36 months), while it accepted the original
  triangle with a factor of 0.9531. At the first age the rebuild starts
  development at the next age and removes every origin observed only at
  the first (one of four on the test triangle), whose reserve then has to
  be estimated outside the model; the error names those origins.[^normalised]
