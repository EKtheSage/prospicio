---
type: Finding
title: Splitting a layer's annual limit pro rata across events keeps the year's total and changes what an inuring stage sees
description: "With annual terms (an aggregate deductible and limit, reinstatements), a layer's year recovers after_terms(U) from per-event recoveries u_e summing to U. In event order the limit is used up as the events occur; pro rata each event cedes share * u_e * after_terms(U) / U (factor 1 when U = 0). The annual total is the same either way; the split by event differs, and so do the nets an inuring stage sees: 10 xs 5 with one reinstatement on 15, 20, 10 cedes 10, 10, 0 in event order and 8, 8, 4 pro rata, and a 5 xs 5 inuring behind it then recovers 10 against 8."
tags: [prospicio-aggregate, reinsurance, annual-terms, inuring]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T10:00:00Z }
verified: { by: process:ci, at: 2026-10-21T10:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/reinsurance.rs
    title: prospicio_aggregate::LimitAllocation, Layer::allocation and their unit tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/476
    title: "Issue #476: pro-rata allocation of the annual limit across events"
---

# Facts

* Event order (`LimitAllocation::EventOrder`, the default) uses the aggregate deductible and limit up in the order the events occur: a late event gets what is left.
* Pro rata (`LimitAllocation::ProRata`) scales every event's recovery by the same factor, `after_terms(U) / U`, so the per-event shares follow the per-event recoveries and do not depend on the events' order.
* The year's ceded total, and so every annual statistic of the layer alone, is the same under both.
* An inuring stage sees the net of the earlier stage by event, so its recovery can differ: on 15, 20, 10 the nets behind 10 xs 5 with one reinstatement are 5, 10, 10 in event order and 7, 12, 6 pro rata, and 5 xs 5 recovers 0 + 5 + 5 = 10 against 2 + 5 + 1 = 8.
* Pro rata is refused together with reinstatements pro rata as to time, whose charge depends on when the limit is used up.
* A tower's JSON writes `"allocation":"pro_rata"` only for a pro-rata layer, so older files read as event order.
