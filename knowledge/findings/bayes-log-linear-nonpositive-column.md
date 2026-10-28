---
type: Finding
title: The log-linear Bayesian reserving models refuse a development column that sums to zero or less, which the chain ladder accepts
description: BayesOdp, BayesBf and BayesBfNormalised all refuse a triangle whose development column's observed increments sum to zero or less (one late cell below the previous one is enough), because every mean is a positive product and the likelihood then has no finite maximum; the chain ladder takes the same triangle with a factor below 1. At the last age the remedy is Triangle::max_age at the previous age, which equals the rebuild from the long table on a cumulative triangle; at the first age it is the rebuild without the age's rows, which also removes the origins observed only there; at a middle age neither works (the rebuilt triangle keeps the age with no cells and the chain ladder refuses it).
tags: [bayes, reserving, bornhuetter-ferguson, odp, negative-development, errors]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T18:00:00Z }
verified: { by: process:ci, at: 2026-10-28T18:00:00Z }
sources:
  - id: normalised
    resource: ../crates/prospicio-reserving/src/bayes/bf_normalised.rs
    title: prospicio_reserving::bayes::BayesBfNormalised (prepare, column_error, origin_error and the test a_falling_late_column_names_its_age_and_the_remedy)
  - id: odp
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: prospicio_reserving::bayes::BayesOdp (the column and origin check of its model, last_age_remedy)
  - id: triangle
    resource: ../crates/prospicio-reserving/src/triangle.rs
    title: prospicio_reserving::Triangle::max_age and its max_age_* tests
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 18 and 26)
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
* At the last age the remedy is `Triangle::max_age` at the previous age
  (Python `tri.max_age(age)`, R `max_age(tri, age)`; decision 18, #667),
  which the three models' messages name: the triangle keeps the other
  origins' latest values and fits. On a cumulative triangle it equals the
  rebuild from the long table without the rows at that age (Rust
  `to_long` and `from_long`, Python `to_frame` and `from_frame`, R
  `as.data.frame()` and `triangle()`).[^triangle] On an incremental one the
  rebuild does not drop the age: `from_long` runs an incremental table's
  ages to the oldest origin's age at the valuation and fills the missing
  rows with zero increments, so the dropped age comes back with zeros;
  `max_age` cuts the axis itself.[^triangle]
* `max_age` keeps every age up to one, so it does not reach the first age
  or a middle one, and the rebuild from the long table works only at the
  first and the last age.
  `Triangle::from_long` fills every age from the first observed one to
  the last, so on a four-origin triangle whose middle age, 36 months, sums
  to -15, the triangle without the rows at 36 months still has ages 12,
  24, 36 and 48, with 36 empty, and the chain ladder refuses it (no origin
  observed at both 24 and 36 months), while it accepted the original
  triangle with a factor of 0.9531. At the first age the rebuild starts
  development at the next age and removes every origin observed only at
  the first (one of four on the test triangle), whose reserve then has to
  be estimated outside the model; the error names those origins.[^normalised]
