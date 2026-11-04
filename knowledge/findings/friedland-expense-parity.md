---
type: Finding
title: Recoveries, ALAE and ULAE against Friedland's printed exhibits
description: RatioDevelopment and Ulae reproduce Friedland's chapter 14 Exhibit I, chapter 16 Exhibit I and chapter 17 Exhibits I to IV to the unit or the printed rounding; negative increments pass through Triangle, the chain ladder and Mack, while a cumulative value below zero errors in Mack and the geometric average.
tags: [reserving, salvage-subrogation, alae, ulae, friedland, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-04T12:00:00-08:00 }
verified: { by: process:ci, at: 2026-11-04T12:00:00-08:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 37
  - id: test
    resource: ../validation/tests/reserving_expense.rs
    title: Validation test of recoveries, ALAE and ULAE
  - id: unit
    resource: ../crates/prospicio-reserving/src/expense.rs
    title: Unit tests of recoveries, ALAE and ULAE
  - id: friedland
    resource: https://www.casact.org/sites/default/files/old/studynotes_friedland_estimating.pdf
    title: Friedland, Estimating Unpaid Claims Using Basic Techniques, version 3, July 2010 (CAS); chapters 14, 16 and 17
---

# Facts

- Salvage and subrogation: ratio-method ultimate 44,934, recoverable 3,054, and
  3,216 and 2,760 by developing reported and received dollars.[^test]
- ALAE: ratio method 12,485 with 800 unpaid, additive 13,215 with 1,530 unpaid,
  all within two of the printed totals.[^test]
- ULAE: XYZ 27,787,500 and 21,105,000 (classical), 24,700,000 and 18,760,000
  (Kittel); New Small Insurer 457,975 and 236,761; PQR 41,587 and 29,891, and
  Conger and Nolibos' 16,767 and 12,795.[^test]
- The book's spreadsheets hold decimals behind whole-unit printing, so the tests
  pass printed factors to ultimate as selected patterns and allow a unit on a
  cell and two on a total.[^test]
- Mack on a cumulative value below zero errors ("sigma can neither be estimated
  nor interpolated"), the geometric average errors on a negative link ratio, and
  the volume and simple averages work.[^test]
- A ratio of exactly zero, a year with no recoveries yet, stops the simple average
  of `RatioDevelopment` ("a zero value gets an infinite weight"), as it stops the
  chain ladder; the volume average and additive factors take it.[^unit]

[^test]: `validation/tests/reserving_expense.rs`; decisions in `docs/design/reserving-v02.md`, decision 37.
[^unit]: The unit tests of `crates/prospicio-reserving/src/expense.rs`.
