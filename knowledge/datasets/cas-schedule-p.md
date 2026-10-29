---
type: Dataset
title: The cas-schedule-p package's reserving mart
description: The pip package cas-schedule-p 2026.6.13 bundles the CAS Schedule P data model's Parquet tables; mart_reserving_model_training has 149,550 rows over six lines, accident years 1988 to 2007 and development ages 1 to 10, which prospicio.reserving.schedule_p reads as the calibration panel does.
resource: ../python/prospicio/reserving.py
tags: [reserving, triangle, clrd, schedule-p, dataset, loader]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-28T18:00:00-07:00 }
sources:
  - id: package
    resource: https://pypi.org/project/cas-schedule-p/2026.6.13/
    title: cas-schedule-p 2026.6.13 (publish 20260613_041006), read with pyarrow on 2026-10-28
  - id: repo
    resource: https://github.com/EKtheSage/cas-schedule-p-data-model
    title: EKtheSage/cas-schedule-p-data-model, the source of the package's data
  - id: note
    resource: ../docs/design/triangle.md
    title: Decision 6, triangles from files and the Schedule P loader
---

# Facts

* The package carries sixteen Parquet tables and a `manifest.json` that
  pins each by sha256; `cas_schedule_p.path(table)` gives a table's file and
  `read(table)` reads it with pyarrow. Its own import needs only the
  standard library; `read` needs pyarrow and `screens` needs duckdb and
  pandas.[^package]
* `mart_reserving_model_training`: 149,550 rows with columns
  `company_code` (text), `company_name`, `line_of_business`,
  `statement_year`, `accident_year`, `development_age` (1 to 10, years),
  `calendar_age`, `incurred_loss`, `cum_paid_loss`, `bulk_loss`,
  `earned_prem_net`, `earned_prem_direct`, `total_reserve`,
  `case_reserve`, `loss_ratio`, `paid_to_incurred_ratio` and
  `ata_factor_into_this_age`.[^package]
* Lines (`line_of_business`): `commercial_auto`,
  `private_passenger_auto`, `workers_compensation`, `other_liability`,
  `medical_malpractice` and `products_liability`. Accident years run from
  1988 to 2007, so the mart holds the later CAS release's years too;
  Meyers' panel and `schedule_p()` default to 1988 to 1997.[^package]
* Values are USD thousands and can be negative (a net earned premium of
  -20 appears in other liability).[^package]
* The CAS loss reserve database's own CSV files name the same measures
  `IncurLoss_*`, `CumPaidLoss_*`, `BulkLoss_*`, `EarnedPremDIR_*`,
  `EarnedPremCeded_*` and `EarnedPremNet_*`, with a suffix per line (`_C`
  for commercial auto), and `GRCODE`, `AccidentYear`, `DevelopmentYear`,
  `DevelopmentLag`. R's `schedule_p_triangle()` reads that layout and finds
  each column by its prefix; `validation/data/schedule_p_comauto_sample.csv`
  holds groups 353 and 8672 in it, written from the mart.[^note]
* `prospicio.reserving.schedule_p("ca", company=353, columns=["paid",
  "case_incurred", "premium"], lower=True)` equals
  `validation/data/clrd_comauto_353.csv` cell by cell, the check
  `calibration_panel.py` makes; the test skips where the package is not
  installed, which includes CI, so this is not marked verified.[^note]

[^package]: Read from the installed package with pyarrow, 2026-10-28.
[^note]: `docs/design/triangle.md`, decision 6; `python/tests/test_reserving_io.py`.
