---
type: Finding
title: An occurrence retro on merged deal cessions needs them grouped by event
description: "Tower::ceded_events gives each event's ceded loss to the named layers, after their annual terms and behind earlier stages, and the set adds to the layers' ceded totals in every simulated year. Merging two deals' ceded sets keeps each deal's cession of one event as its own occurrence: deals ceding 20 and 10 of one windstorm give a 15 xs 10 retro 10 that way, and 15 once the merged set is grouped by an hours clause into one occurrence of 30."
tags: [prospicio-aggregate, reinsurance, retrocession, events]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-28T14:00:00Z }
verified: { by: process:ci, at: 2026-10-28T14:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/reinsurance.rs
    title: prospicio_aggregate::Tower::ceded_events and its unit tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/590
    title: "Issue #590: a tower's ceded losses by event as an event set, for an occurrence retro"
---

# Facts

* On 100,000 simulated years (Poisson 2, lognormal mean 3m and cv 1.5) under a 30% quota share inuring to a 5m xs 5m with one reinstatement and a 10m xs 10m with an annual limit of 12m split pro rata, the ceded set of each layer, and of the two excess layers together, adds to the tower's ceded total in every year to 1e-6 relative.
* Events that cede nothing are left out, so the set's count in a year is the number of attaching events; the events keep their times and labels, and sums insured are dropped.
* The split by event is the one the labelled results use: under an annual limit in event order a late event gets what is left (10 xs 5 with an annual limit of 15 on 20, 8 and 15 cedes 10, 3 and 2), and a scoped layer cedes only from its perils.
* `EventSet::merge` of two deals' ceded sets keeps each deal's cession as its own occurrence: deal A ceding 20 of a windstorm at time 0.2 and deal B ceding 10 of it and 2 of a later quake give a 15 xs 10 retro 10 + 0 + 0 = 10; grouped with a 72-hour clause the windstorm is one occurrence of 30 and the retro recovers 15.
