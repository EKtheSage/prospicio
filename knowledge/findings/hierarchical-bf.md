---
type: Finding
title: HierarchicalBf's posterior is improper past a falling-column bound, and a dense pilot scale biases few programs' standard deviations
description: A partially pooled pattern with one omega has no posterior when falling columns reach 1 / omega_scale in program scales, a dense pilot scale left EVW's sd 9% below Stan's on three programs, and thin programs' raw sds are not always below separate fits'.
tags: [reserving, bayesian, hierarchical, nuts, stan]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-05T12:00:00Z }
verified: { by: process:ci, at: 2026-11-05T12:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-reserving/src/bayes/hierarchical_bf.rs
    title: prospicio_reserving::bayes::hierarchical_bf, the model and its error rules
  - id: code
    resource: ../validation/tests/reserving_bayes_hierarchical_bf.rs
    title: The validation suite, with the Stan parity and the recovery study
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, decision 30, "As built"
---

Three facts from building `HierarchicalBf`, with the design note's decision
30 holding the full tables.

**A falling column can leave no posterior.** Under `beta_cd = beta_d + omega
delta_cd`, a program's column summing to `S_cd < 0` gains `|S_cd| / phi_c`
per unit of `omega delta_cd`. Raising `omega` and `delta_cd` together costs
only `1 / omega_scale` per unit of their product, so both sides are linear
and a large fall wins. One `omega` scales every deviation, so the bound is
the root of the sum of `(S_cd / phi_c)^2` over all programs' falling columns
against `1 / omega_scale`. An error refuses it. The prior alone does not
hold such a column, as the first version of the note assumed.

**A dense pilot scale misreads a funnel.** With three programs the pilot
sees the funnels of `tau` and `omega` and its covariance scales the main run
badly: 20 of 24 seeds warned and EVW's loss ratio sd was 9% below Stan's
(2.7 standard errors). Variances alone, a short pilot and `target_accept`
0.99 gave 1 of 24 and agreed with Stan within its noise. Centred forms and
longer pilots were worse. From about 10 programs the dense scale is better.
`Meyers` and `Compartmental` use the dense scale and were not tested.

**Thin programs do not always get narrower in raw amounts.** A separate fit
of three origins has one degree of freedom, so its scale ranges from 0.035
to 17.8 and its sd with it; the hierarchical sd was lower in only 62 of 120
programs. Per unit of the root of each fit's scale it was lower in 111 to
116 of 120 (a ratio of about 0.78), and the error of the mean was 0.66 to
0.75 of the separate fits'. Pooling's gain is real, and part of what a
separate fit shows as a small sd is an underestimated scale.
