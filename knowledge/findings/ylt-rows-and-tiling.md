---
type: Finding
title: A year loss table's rows that share an event slot are separate losses, and tiling needs a whole multiple of its years
description: "A vendor YLT lists one row per non-zero loss with its year, peril, event slot and class, so one event hitting two classes is two rows with the same year and slot. Read row by row, they are two losses of the year, and an occurrence layer sees each, not their sum: a 5 xs 5 layer on rows of 10 and 4 in one slot cedes 5, where one occurrence of 14 would cede 5 as well but 8 xs 6 cedes 4 against 8. So the duplicate check is by year, slot and the labels that may not repeat (peril, not class). Tiling a table of n years into N simulations gives simulation i year i mod n and is unbiased only when N is a multiple of n; otherwise the first N mod n years count once more, so the loader refuses it and resampling with replacement is the alternative."
tags: [prospicio-aggregate, year-loss-table, cat, event-set, provenance]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T14:00:00Z }
verified: { by: process:ci, at: 2026-10-21T14:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-aggregate/src/ylt.rs
    title: prospicio_aggregate::ylt::YearLossTable and its unit tests
  - id: design
    resource: ../docs/design/cat-inputs.md
    title: "Design note: catastrophe XL inputs, decisions 10 to 12"
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/484
    title: "Issue #484: a year loss table loader with tile or resample expansion"
---

# Facts

* `YearLossTable::events` orders a year's rows by slot, the file's order breaking ties, and keeps each row as one loss with its labels; rows sharing a slot are not summed into an occurrence.
* `check_unique(&["peril"])` rejects two rows of one year, slot and peril and accepts one slot hitting two classes; `check_unique(&[])` rejects any repeated year and slot.
* `Expansion::Tile` refuses a simulation count that is not a multiple of the table's years, since the remainder would weight the first years more; `Expansion::Resample { seed }` draws simulation `i`'s year from `StreamRng::new(seed, i)`.
* The table's years default to its largest year; a table whose last years have no loss needs `with_n_years`, or the tiling and every annual statistic are off by those empty years.
* The bindings hash the file's bytes (`ylt::hash_bytes`, blake3 through `InputHasher`, not a plain blake3 of the file); the events carry it as `input_hash` into results' provenance, and a merge with another hashed set hashes the pair in order.
