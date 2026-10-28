---
type: Finding
title: A layer scoped to a peril is not one peril's share of an unscoped layer
description: "With annual terms, a treaty covering one peril must see only that peril's losses: another peril's events would use up its annual limit and count toward its deductible. On quake 20 then wind 15 (one year) and wind 8, flood 30, wind 12 (another) under 10 xs 5 with an annual limit of 15, the unscoped layer's wind share is 5 in each year while the wind-scoped layer cedes 10 in each. The scoped layer on the whole set equals the plain layer on the filtered set, component by component, in one year and over a term."
tags: [prospicio-aggregate, reinsurance, labels, annual-terms, inuring]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-28T12:00:00Z }
verified: { by: process:ci, at: 2026-10-28T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/reinsurance.rs
    title: prospicio_aggregate::Layer::scope, Tower::year_by_event and their unit tests
  - id: filter
    resource: ../crates/prospicio-aggregate/src/monte_carlo.rs
    title: prospicio_aggregate::EventSet::filter
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/584
    title: "Issue #584: layers scoped to a peril or class"
---

# Facts

* A label split of an unscoped layer divides the layer's recovery by the labels of the events that produced it; every peril still uses the same annual deductible and limit. On quake 20 then wind 15 under 10 xs 5 with an annual limit of 15, quake recovers 10 first and wind gets the remaining 5.
* `Layer::scope("peril", &["wind"])` masks the other perils' losses for that layer only, so wind recovers 10 from 15; in the second year wind 8 and 12 recover 3 and 7, still 10 in all, where the unscoped layer's wind share is again 5 (3, then flood's 10, then 2 of wind's 7).
* The scoped layer on the whole set equals the plain layer on `EventSet::filter("peril", &["wind"])` for every component but gross and net, in one year and over a two-year term with a profit commission carried forward; a scope over every value equals no scope bit for bit.
* Losses out of a layer's scope pass to the next inuring stage whole: a wind-scoped 10 xs 5 in front of a 100 xs 20 stop loss leaves the stop loss 25 and 40 to see (quake 20 plus wind's net 5; wind's nets 5 and 5 plus flood 30), so it recovers 5 and 20.
* A scope is an error on events without labels, without the dimension or with none of its values, so a misspelt peril does not cede nothing silently; paths with no labelled events (the grids, annual losses, the P&L ledger) refuse a scoped layer.
* A scoped layer is written to the tower's JSON as `"scope": {"label": ..., "values": [...]}`, format version 7.
