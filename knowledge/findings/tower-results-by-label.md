---
type: Finding
title: A tower's ceded loss by peril or class is the sum of each layer's split by event, and depends on the limit allocation
description: "With labels on the events, Tower::apply sums each layer's ceded loss by event over the events of each label combination; gross and net by label follow, and the sums over labels give back the unlabelled result exactly in the tests. Under an annual limit the split by label follows the layer's allocation: 10 xs 5 with an annual limit of 15 on wind 15 then quake 20 cedes wind 10 and quake 5 in event order, but wind 7.5 and quake 7.5 pro rata. Premiums and commissions are annual and get empty labels."
tags: [prospicio-aggregate, reinsurance, labels, perils, allocation]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T11:00:00Z }
verified: { by: process:ci, at: 2026-10-21T11:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/reinsurance.rs
    title: Tower::apply on labelled events and its unit tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/475
    title: "Issue #475: peril and class labels on EventSet, carried through Tower.apply"
---

# Facts

* Each loss carries a value in every label dimension; the set stores the sorted distinct combinations and each loss's index into them.
* A layer's ceded loss by label is the sum of `Layer::ceded_by_event` over that label's events, through every inuring stage, so the by-label split of a later stage follows what the earlier stages left of each event.
* In event order a later event gets what is left of the annual limit, so a peril whose events come late in the year recovers less; pro rata every event is scaled by the same factor and the split by label follows the recoveries before the annual terms.
* Reinstatement and swing premiums and commissions depend on the year's ceded total only, so they are not split; they carry `""` in every label dimension.
* A stop loss (or any cover on the annual total) under `LimitAllocation::ProRata` gives each label `component × capped / uncapped`, the reference model's pro-rata convention for a capped total: 20 xs 30 on wind 50 and quake 10 cedes wind 50/3 and quake 10/3. Outside a tower, `PredictiveDistribution::scale_to_totals` does the same to any components and a total per simulation (#478).
* `merge` keeps labels from either set, filling `""` for a dimension a set lacks; `grouped` (an hours clause) drops them, since an occurrence may span perils.
