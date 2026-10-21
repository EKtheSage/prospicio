---
type: Finding
title: The derived excess pattern beats a sparse layer's own chain ladder when its model holds, and a scale shift understates high layers under shape drift
description: Over 500 simulated books per arm, the pattern derived from the ground-up triangle has 0.003 to 0.056 times the RMSE of the layer's own chain ladder on a sparse layer and 0.59 to 0.66 on a thin one, and no bias on a dense one, where its model holds (scale shift, capped ground-up column, Pareto shape drift fitted by ParetoShape); shape drift makes the scale shift understate high layers by 14% to 29%, smaller late claims understate the dense layer by 0.8%, and without the cap a small k is ruinous.
tags: [reserving, excess-development, simulation, credibility, shape-drift]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T22:00:00Z }
verified: { by: process:ci, at: 2026-10-21T22:00:00Z }
sources:
  - id: code
    resource: ../validation/tests/reserving_excess_simulation.rs
    title: The simulation study, run in every cargo test
  - id: module
    resource: ../crates/prospicio-reserving/src/excess.rs
    title: prospicio_reserving::excess, credibility_weights and ParetoShape
  - id: note
    resource: ../docs/design/reserving-excess.md
    title: Design note, "Validation targets" and decisions 11, 16 to 19
---

# What was found

The study (`docs/design/reserving-excess.md` PR 2, decisions 11 and 16)
simulates 500 books per arm with seed 192: ten accident years at ten
annual ages, Poisson claim counts, report lags of up to four years, and
each claim developing to ultimate at the oldest age, so the ground-up,
count and layer triangles and the true layer ultimates come from the same
claims. Every error is of the total layer ultimate over the ten years.

Arms: `scale` (lognormal, mean 30,000, log-sd 1.8, 400 claims a year, every
claim the ultimate one over `c_j`, `c` from 2.0 to 1); `capped` (the same
claims, the ground-up column limited at 250,000, so the shifts come from the
numerical solve); `shape` (the same mean development, the log-sd at the
first age 0.8 of the ultimate one, each claim by a quantile map); `late`
(the scale shift with claims reported after the first age half the size);
`pareto` (150 claims above 100,000 a year, the Pareto shape from 3.0 to 2.2,
fitted by `ParetoShape` with the true shapes on the losses excess of
100,000, and by the scale shift on the ground-up column). Layers: dense
(50k xs 50k; Pareto 150k xs 150k), thin (250k xs 250k; 400k xs 400k) and
sparse (1m xs 1m). The blend uses the layer claim counts, the cap at 0.1.

| Arm, layer | Own chain ladder fails | Derived: mean error / truth | Derived: RMSE ratio to own | Capped blend ratio, k = 1 to 200 |
|---|---|---|---|---|
| scale, dense | 0 | -0.05% (MC SE 0.16%) | 0.950 | 0.999 to 0.934 |
| scale, thin | 0 | 0.28% | 0.656 | 0.820 to 0.665 |
| scale, sparse | 284 | 1.6% | 0.055 | 0.068 to 0.056 |
| capped, dense | 0 | -0.01% (MC SE 0.15%) | 0.934 | 0.999 to 0.941 |
| capped, thin | 0 | 0.32% | 0.647 | 0.804 to 0.655 |
| capped, sparse | 284 | 1.7% | 0.056 | 0.068 to 0.057 |
| shape, dense | 0 | 0.69% (MC SE 0.16%) | 1.018 | 0.998 to 0.924 |
| shape, thin | 0 | -13.8% | 0.359 | 0.313 to 0.339 (0.289 at k = 20) |
| shape, sparse | 429 | -26.1% | 0.059 | 0.070 to 0.059 |
| late, dense | 0 | -0.79% (MC SE 0.14%) | 0.906 | 0.998 to 0.896 |
| late, thin | 2 | -0.58% | 0.608 | 0.831 to 0.611 |
| late, sparse | 328 | -0.2% (MC SE 1.4%) | 0.066 | 0.091 to 0.066 |
| pareto, dense | 0 | 0.25% (MC SE 0.14%) | 1.094 | 0.999 to 0.977 |
| pareto, thin | 0 | -0.19% | 0.594 | 0.738 to 0.599 |
| pareto, sparse | 330 | 1.8% | 0.003 | 0.003 |

The ratios are taken on the simulations where the own chain ladder fits (it
fails where every earlier value of an age is zero), which favours it. In
the `pareto` arm the scale shift on the ground-up column understates by
1.1% (dense), 18.9% (thin) and 28.7% (sparse).

* **Where the model holds** (scale, capped, Pareto by `ParetoShape`) the
  thresholds of decision 11 pass: the derived pattern's RMSE, and the
  capped blend's at every k, is below 0.8 times the own chain ladder's on
  the sparse layer; on the dense layer its mean error is within two Monte
  Carlo standard errors of zero. RMSE falls as k grows, so the derived
  pattern alone is best. The capped arm matches the uncapped one: the
  numerical solve recovers the shifts from capped data.
* **Shape drift** makes the scale shift understate high layers (14% thin,
  26% sparse; 19% and 29% for the Pareto arm's scale shift) and slightly
  overstate the dense layer. The blend with the layer's own factors
  recovers part of the bias and is best at k of 10 to 20. A Pareto
  shape-drift model with the right shapes is unbiased.
* **Late claims half the size** understate the dense layer by 0.8% and leave
  the sparse layer unbiased: the count factor overstates a high layer's
  development, the shift (ground-up over count factor) is understated, and
  the two cancel on the high layer. The note's first reading, an
  overstated factor, holds only for the count part.
* **The cap matters.** Without it, k = 1 on the sparse layer gives an RMSE of
  3.1 times the mean ultimate (scale arm) against 0.43 with the cap at 0.1.
* **The layer's own chain ladder overstates a thin layer** by 2.6% to 6.7%
  (ratio-estimator bias on few claims).

The test asserts the thresholds, the cap's effect, the derived pattern's ratio on the thin layer
below 0.8 where the model holds, and the direction of each bias (shape
arm and the Pareto arm's scale shift below zero on the thin and sparse
layers, the late arm below zero on the dense layer and within two MC SE
on the sparse one). The other numbers are the test's printed output,
deterministic by seed (debug and release agree), not asserted. The study
takes under a second in a debug build with rayon, so CI's `cargo test`
runs all 500 simulations of every arm.
