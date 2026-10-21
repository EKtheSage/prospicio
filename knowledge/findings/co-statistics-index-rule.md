---
type: Finding
title: An item read at another variable's percentile depends on how h = q (n - 1) is rounded, and Python's round is not Rust's
description: "A results table reports a margin or a commission at each percentile of the loss ratio: the item on the simulation o[i] of the driver's stable ascending order, i a rounding of h = q (n - 1). Python's and numpy's round take halves to even, Rust's f64::round away from zero. On five simulations at q = 62.5%, h = 2.5: halves to even and floor pick the third smallest driver, ceil and halves away from zero the fourth. The TVaR reading averages o[ceil(h):] whatever the rule."
tags: [prospicio-prob, statistics, co-metrics, results-table, conventions]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T23:30:00Z }
verified: { by: process:ci, at: 2026-10-21T23:30:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/487
    title: "Issue #487: co-metric statistics (an item read at another item's percentiles and tail)"
  - id: test
    resource: ../crates/prospicio-prob/src/statistics.rs
    title: co_statistics_pick_by_the_drivers_rank_under_each_rule
---

# An item read at another variable's percentile depends on the index rule

A reference simulation pricing model reports its commission, combined
ratio, margin, IRR and capital "at" each percentile and TVaR of the
underwriting loss ratio: the item on the simulation the loss ratio's
statistic selects, not the item's own percentile. With `o` the stable
argsort of the driver over `n` simulations and `h = q (n - 1)`:

- the mean reads the item on the first `o[i]` whose driver is at or above
  the driver's mean;
- a percentile reads the item on `o[i]`, `i` a rounding of `h`;
- a TVaR averages the item over `o[ceil(h):]`.

The rounding is a convention, and two languages disagree on it at every
half. Python's `round` and numpy's `round` take halves to even, as R's
`round` does; Rust's `f64::round` takes them away from zero
(`f64::round_ties_even` is the Python one). On five simulations at
`q = 62.5%`, `h = 2.5`: halves to even and `floor` pick index 2, `ceil`
and halves away from zero index 3. At `n = 10,000` the median is such a
case, `h = 4999.5`, so the 50th percentile's item depends on the rule.

`StatisticsGrid::co_statistics` takes the rule as `IndexRule`
(`RoundHalfEven`, the default, `Floor`, `Ceil`), keeps ties in simulation
order (a stable sort), and caps the driver's mean at its largest value,
since the mean of equal values can round above them and would otherwise
select nothing. Its TVaR ignores the rule, as the reference does.

These readings are not `marginal_expected_shortfall` or the co-TVaR of
`allocate`, which condition on the total and share weights among tied
totals; read at the total, the co-TVaR at 90% of a component over 1,001
simulations agrees with `allocate`'s within 1%.
