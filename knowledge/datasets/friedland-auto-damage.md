---
type: Dataset
title: Friedland's Auto Physical Damage and Auto Property Damage triangles
description: The reported and received salvage and subrogation, reported and paid ALAE, and reported and paid claims of Friedland's chapters 14 and 16, accident years 1998 to 2008, read from the PDF by position and checked against the printed diagonals.
resource: ../validation/data/friedland_auto_physical_damage.csv
tags: [reserving, triangle, salvage-subrogation, alae, friedland]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-04T12:00:00-08:00 }
verified: { by: process:ci, at: 2026-11-04T12:00:00-08:00 }
sources:
  - id: friedland
    resource: https://www.casact.org/sites/default/files/old/studynotes_friedland_estimating.pdf
    title: Friedland, Estimating Unpaid Claims Using Basic Techniques, version 3, July 2010 (CAS); chapter 14 Exhibit I and chapter 16 Exhibit I
  - id: test
    resource: ../validation/tests/reserving_expense.rs
    title: Validation test, which checks each column's printed diagonal
---

# Layout

Two long CSV files of 66 rows each, cumulative values in thousands of dollars.
`validation/data/friedland_auto_physical_damage.csv` has `origin`, `age`,
`reported_ss`, `received_ss`, `reported` and `paid`;
`validation/data/friedland_auto_property_damage.csv` has `origin`, `age`,
`reported_alae`, `paid_alae`, `reported` and `paid`.[^test]

[^test]: `validation/tests/reserving_expense.rs`, `the_data_are_the_printed_diagonals`.
