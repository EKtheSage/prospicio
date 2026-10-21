---
type: Finding
title: A contract stated on the loss ratio is a layer on money amounts
description: "A loss-ratio cap c on subject premium P is a stop loss of c * P xs 0 with premium_rate(1, P): the ceded loss ratio ceded / premium is then min(LR, c), the capped one, so a sliding scale on the layer reads the same ULR as the contract's min(LR, cap). Several sliding scales on the same loss-ratio grid add pointwise into one, since a sum of piecewise linear functions on the same anchors is piecewise linear on them. On 40, 70, 150 against P = 100, a cap of 120% and a scale from 30% at 50% to 10% at 90% cede 40, 70, 120 with commissions 30, 20, 10."
tags: [prospicio-aggregate, reinsurance, sliding-scale, stop-loss]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T11:00:00Z }
verified: { by: process:ci, at: 2026-10-21T11:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/reinsurance.rs
    title: prospicio_aggregate::Tower::apply_annual, its doctest and unit test
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/477
    title: "Issue #477: annual contracts on a vector of annual losses, terms as a percentage of premium"
---

# Facts

* `Tower::apply_annual(&[f64])` takes each simulation's annual loss as one aggregate loss, as `apply_aggregate` does for a predictive distribution's totals; its components and draws equal `apply_aggregate`'s on the same losses (unit test).
* A loss-ratio cap `c` on subject premium `P` is `Layer::stop_loss(name, c * P, 0)` with `premium_rate(1, P)`. The ceded loss ratio `ceded / P` is `min(LR, c)`, so `Layer::sliding_scale` keyed on it reads the capped loss ratio.
* A sliding scale is linear between its anchors and flat beyond the first and last; several on the same loss-ratio grid add pointwise into one. `Commission::rate` (Rust) and `ceding_commission_for` (Python, R) evaluate a scale.
* Brokerage on a loss-ratio scale is not a leg of a layer; it waits on the user's decision (lane slug `[brokerage-leg]`).
