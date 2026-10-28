---
type: Finding
title: A layer's attachment probability per year is one minus the Poisson zero of its excess count, below the expected count
description: On 200,000 simulated years with Poisson(2) claims of a lognormal of mean 3m and cv 1.5, 5m xs 5m with one reinstatement attaches in 26.5% of years, has an occurrence using its whole limit in 9.3% and uses up both limits in 0.51%, against 0.309 occurrences a year reaching it; the closed forms 1 − exp(−λ S(a)) = 26.7%, 1 − exp(−λ S(a + l)) = 9.4% and λ S(a) = 0.311 are within two standard errors.
tags: [aggregate, reinsurance, layer, cat, attachment, exhaustion, simulation]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-28T13:00:00Z }
verified: { by: process:ci, at: 2026-10-28T13:00:00Z }
sources:
  - id: note
    resource: ../docs/design/aggregate.md
    title: Design note, aggregate, "Layer statistics and prices by view", trial statistics on simulated years
  - id: test
    resource: ../crates/prospicio-aggregate/src/trials.rs
    title: agrees_with_the_tower_and_with_the_poisson_closed_forms, run in every cargo test
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/581
    title: Issue 581, attachment and exhaustion probabilities of a layer on simulated years
---

# What was found

`Layer::trial_statistics` counts a layer's exhibit on simulated years.
With Poisson(2) claims of a lognormal severity (mean 3m, cv 1.5), 200,000
years from seed 11, and 5m xs 5m with one free reinstatement:

| Statistic | Simulated | Standard error | Closed form |
|---|---|---|---|
| years attached | 0.2654 | 0.0010 | 1 − exp(−λ S(5m)) = 0.2672 |
| years with a full-limit occurrence | 0.0930 | 0.0006 | 1 − exp(−λ S(10m)) = 0.0939 |
| years with both limits used | 0.0051 | 0.0002 | (none) |
| occurrences reaching the layer a year | 0.3092 | 0.0012 | λ S(5m) = 0.3109 |

The event loss table's `occurrence_layer_statistics` gives `pr_attach =
S(a)`, a probability per occurrence (0.155 here), and the expected excess
count `λ S(a)`. Neither is the probability that a year touches the layer,
which is `1 − exp(−λ S(a))` under Poisson counts, 14% below the expected
count here. An exhibit that reads the expected count as a probability
overstates how often the layer is hit, more so the more often it is.

# How it is checked

The unit test asserts each simulated value within four standard errors of
its closed form and the ceded mean equal to a one-layer tower's;
the hand-counted tests pin every statistic on small event sets exactly.
