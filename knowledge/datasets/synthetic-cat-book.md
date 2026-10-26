---
type: Dataset
title: Synthetic property-catastrophe book, Larkspur Re
description: A seeded, generated catastrophe book for the cat-* guides (a fictional reinsurer, three cedents, four perils, an event loss table and 2,000 simulated years of dated events, five contracts); what the guides find on it about quantile conventions, the hours clause and the event table against the simulation.
resource: ../examples/data/make_cat_book.py
tags: [reinsurance, catastrophe, synthetic, yelt, elt, guides]
status: stable
verified: { by: process:ci, at: 2026-10-23T12:00:00Z }
generated: { by: claude-code/local-session, at: 2026-10-23T12:00:00Z }
sources:
  - id: generator
    resource: ../examples/data/make_cat_book.py
    title: make_cat_book.py, the seeded generator (numpy only, seed 20261023)
  - id: overview
    resource: ../docs/guides/cat-0-overview.md
    title: The track's overview, the story, the files and the notation
  - id: metrics
    resource: ../docs/guides/cat-2-risk-metrics.md
    title: cat-2-risk-metrics, the conventions and the event table against the simulation
  - id: tests
    resource: ../python/tests/test_guides.py
    title: test_cat_book_files_match_generator and the guides' printed numbers, checked in CI
---

# Facts

* Four CSV files in `examples/data/`, written by `make_cat_book.py` (seed
  20261023, numpy only): `cat_book_yelt.csv` (16,432 rows: year, slot,
  event, day, hour, peril, region, cedent, loss), `cat_book_elt_alder.csv`
  (299 events of one cedent: rate, mean, independent and correlated sd,
  exposure), `cat_book_cedents.csv` (three cedents) and
  `cat_book_contracts.csv` (five contracts: two occurrence layers for Alder,
  an occurrence layer and an aggregate cover for Bracken, a capped quota share
  for Corran). Money is in millions of a synthetic currency.
  `test_cat_book_files_match_generator` regenerates them and compares the
  rows, floats to 1.5e-4.[^tests]
* Simplifications in the generator, stated in its docstring and in the
  guides: each cedent's loss from an event is a beta draw with standard
  deviation `sd_independent + sd_correlated`, drawn independently per cedent,
  so the correlated part of secondary uncertainty is not correlated across
  cedents; and a YELT row is written only for a loss of at least 0.05, so the
  YELT's mean annual loss is slightly lower than an unfloored simulation's.
  Alder's simulated mean annual loss is 16.778 against its ELT's AAL of
  16.056.[^generator]
* Some events reach the YELT as two or three timed pieces within 60 hours.
  Under a 72-hour clause with the window fixed at the first piece, Alder's
  5,391 losses become 4,238 occurrences, 87 of which join two or more catalog
  events and 9 of those exceed 30; grouping is what brings the simulated
  occurrence exceedance (0.1150 at 30) in line with the ELT's (0.1136), where
  the raw pieces give 0.1070.[^metrics]
* VaR conventions on Alder's 2,000 annual totals at p = 0.99: the `k`-th
  largest year with `k = floor((1 - p) N) = 20` is 177.979, the inverted cdf
  (Hyndman and Fan type 1, `Sampled.var` and `statistics`' default) 176.824,
  the linear quantile (type 7, numpy's default) 176.836. The exact TVaR
  (`statistics`' default `tvar="exact"`) equals the mean of the worst 20 years,
  255.969, when `(1 - p) N` is an integer; the linear-quantile TVaR is
  246.942.[^metrics]
* An ELT's per-occurrence exceedance probability is per catalog event (rate
  2.3365 a year for Alder), a simulation's per grouped occurrence (2.119 a
  year): 0.0516 times 2.3365 / 2.119 is the simulation's 0.0569 for the
  attachment of a 40 xs 30 layer.[^metrics]

[^generator]: [make_cat_book.py](../examples/data/make_cat_book.py)
[^metrics]: [cat-2-risk-metrics.md](../docs/guides/cat-2-risk-metrics.md), sections 1, 2 and 5 to 7
[^tests]: [test_guides.py](../python/tests/test_guides.py)
