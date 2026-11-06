---
type: Finding
title: On freMTPL2 two-dimensional embeddings of VehBrand and Region beat a one-hot CANN on held-out policies, with the tutorial's network features
description: "With every fifth policy of freMTPL2freq held out and both networks starting at the Poisson GLM fitted on the rest (hidden layers 20, 15, 10; 15 epochs of Adam at 1e-2 on batches of 10,000), the test mean deviance is 0.319120 for the GLM, 0.317186 for a CANN on the indicator columns and 0.316367 with VehBrand and Region as two-dimensional embeddings (seed 1; 0.316843 and 0.316193 at seed 2). The networks take the tutorial's features, VehPower a number; on the GLM's own design, VehPower a factor, the embeddings were not better (0.317480 against 0.317189)."
tags: [prospicio-nn, embeddings, cann, fremtpl2, frequency]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-05T10:00:00Z }
verified: { by: process:ci, at: 2026-11-05T10:00:00Z }
sources:
  - id: test
    resource: ../validation/tests/fremtpl2.rs
    title: fremtpl2_embeddings_beat_the_one_hot_cann, run by CI's fremtpl2 job in release
  - id: note
    resource: ../docs/design/models.md
    title: Design note, models, "Tabular networks with embeddings and several heads"
  - id: tutorial
    resource: https://ssrn.com/abstract=3320525
    title: Schelldorfer and Wüthrich (2019), Nesting Classical Actuarial Models into Neural Networks
---

# What was found

`TabularNet` (#577) enters a factor as a learned vector of a few
dimensions instead of its indicator columns.[^note] The freMTPL2 run
follows Schelldorfer and Wüthrich's comparison:[^tutorial] every fifth
policy is held out, the Poisson GLM of Noll, Salzmann and Wüthrich is
fitted on the other 542,410, and both networks start at it (its linear
predictor and the log exposure as the offset, output layers at zero).[^test]

| Model | Test mean deviance, seed 1 | Seed 2 |
|---|---|---|
| GLM | 0.319120 | 0.319120 |
| CANN on the indicator columns | 0.317186 | 0.316843 |
| CANN with VehBrand and Region embedded in 2 dimensions | 0.316367 | 0.316193 |

* The networks' features are the tutorial's: Area, VehPower, BonusMalus
  and Density as numbers, VehAge, DrivAge and VehGas as indicators, and
  VehBrand and Region as indicators or embeddings. On the GLM's own design,
  where VehPower is a factor, the embeddings did not beat the indicators:
  0.318175 against 0.317565 after 4 epochs and 0.317480 against 0.317189
  after 15.
* The run takes about 60 s in a release build on a 16-core machine (the GLM
  and two networks of 15 epochs).

[^note]: `docs/design/models.md`, "Tabular networks with embeddings and several heads".
[^tutorial]: Schelldorfer and Wüthrich (2019), section 4.
[^test]: `validation/tests/fremtpl2.rs`, `fremtpl2_embeddings_beat_the_one_hot_cann`.
