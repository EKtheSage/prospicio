---
type: Reference
title: ISO's general liability experience rating plan
description: The experience ratio plan behind prospicio_pricing::emod's ExperienceRatioPlan; its manual and tables are licensed and no free text works an exhibit through, so the library is checked by hand against the formula rather than against a printed worksheet.
resource: ../validation/tests/pricing_emod_ratio.rs
tags: [pricing, experience-rating, general-liability, emod]
status: draft
generated: { by: claude-code/cloud-session, at: 2026-11-27T14:00:00Z }
sources:
  - id: design
    resource: ../docs/design/primary-rating.md
    title: The design note's experience ratio plan and decision (l)
  - id: tests
    resource: ../validation/tests/pricing_emod_ratio.rs
    title: The plan's sums by hand, linearity in the credibility and monotonicity in the MSL
---

# What it is

ISO's Commercial General Liability Experience and Schedule Rating Plan
compares an actual experience ratio with an expected one:
`AER = (reported + expected unreported) / subject loss cost` and
`mod = 1 + Z (AER - EER) / EER`. Each occurrence enters at basic-limits
loss plus ALAE held to the maximum single limit; a year's expected
unreported losses are its subject loss cost times a detrend factor, `EER`
and the share of its losses still to be reported.[^design]

# No public exhibit

The plan manual, its subject loss costs, `EER`s, MSLs, credibilities and
unreported shares are licensed. The free material found while building
PR 8 of #981 (NCCI-style explainers, broker notes and CAS readings
searched in November 2026) states the formula but works no exhibit
through to a modification, so there is no printed figure to match. The
tests check the sums by hand and two properties that hold however the
tables read: the modification is linear in `Z` (1 at `Z = 0`), and never
falls as the MSL rises.[^tests] A licensed worksheet, if the user supplies
one, would become a parity test in the same file.

[^design]: `docs/design/primary-rating.md`, the experience ratio plan.
[^tests]: `validation/tests/pricing_emod_ratio.rs`.
