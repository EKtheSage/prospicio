---
type: Finding
title: On the Chalk portfolio a held-out isotonic recalibration of a Poisson frequency pools to 0 at its low end, and CORP's miscalibration term is 0.012 for a calibrated GLM
description: "A Poisson GLM fitted on a third of the Chalk portfolio scores a CORP miscalibration term of 0.012 on another third (mean deviance 0.914, discrimination 0.020, uncertainty 0.923): the in-sample isotonic fit takes up noise, so the term is judged against a reference model, not zero. Bent to (p/m)^2.5 m the term is 0.047 with discrimination unchanged; an isotonic map fitted on the remaining third pools its lowest block to a frequency of 0, which makes a later claim's Poisson deviance infinite, and with a floor at 5% of the mean it brings the term to 0.007 and the score to 0.912."
tags: [models, calibration, isotonic, murphy-decomposition, chalk-portfolio, prospicio-models]
status: draft
verified: { by: process:ci, at: 2026-10-29T18:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-29T18:00:00Z }
sources:
  - id: design
    resource: ../docs/design/models.md
    title: models.md, "Auto-calibration"
  - id: test
    resource: ../validation/tests/models.rs
    title: recalibration_repairs_a_bent_model_on_the_chalk_portfolio
---

# Isotonic recalibration on the Chalk portfolio

The Chalk portfolio (`examples/data/chalk_portfolio.csv`) is split by row
number into thirds. A Poisson GLM on age, vehicle age and experience,
weighted by exposure, is fitted on the first.[^test]

- **The miscalibration term is not zero for a calibrated model.** On the
  last third the GLM's CORP decomposition of the mean Poisson deviance is
  score 0.914, miscalibration 0.012, discrimination 0.020, uncertainty
  0.923. The term is the score the isotonic fit on the same rows gains,
  and with a weak signal (discrimination a fiftieth of the uncertainty)
  much of it is noise, so models are compared on it rather than tested
  against zero.[^design]
- **A bent model.** Mapped to `(p / m)^2.5 m` (same ranking), the GLM's
  miscalibration rises to 0.047 and its discrimination stays 0.020.
- **A floor is needed.** An increasing isotonic regression of the claim
  frequency on the bent predictions, fitted on the middle third, pools
  its lowest block to 0. On the last third a policy in that block with a
  claim has an infinite Poisson deviance. With values clipped below at 5%
  of the mean frequency the recalibrated model scores 0.912 (better than
  the plain GLM), miscalibration 0.007, discrimination 0.018 (the steps
  merge some levels).[^test]

[^design]: models.md, "Auto-calibration"
[^test]: recalibration_repairs_a_bent_model_on_the_chalk_portfolio
