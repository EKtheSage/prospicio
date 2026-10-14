---
type: Dataset
title: Gesmann and Morris' company 337, workers' compensation
description: The full 10 x 10 square of NAIC company 337 workers' compensation, 1988-1997 (paid, incurred with bulk and IBNR, case incurred, direct and net earned premium), from the CAS Schedule P data model release 20260613_041006; the papers' outstanding is incurred less paid and their premium is direct, which is what makes every cell of Morris' (2016) printed upper triangles round to the data (premium 104, outstanding 53 and paid 10, USD millions, for 1988 at age 1).
resource: ../validation/data/clrd_wkcomp_337.csv
tags: [reserving, triangle, clrd, schedule-p, compartmental, bayes, premium, outstanding]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T21:00:00-07:00 }
sources:
  - id: repo
    resource: https://github.com/EKtheSage/cas-schedule-p-data-model
    title: EKtheSage/cas-schedule-p-data-model, release 20260613_041006, table mart_reserving_model_training
  - id: script
    resource: ../validation/scripts/clrd_wkcomp_337.py
    title: The generator
  - id: morris
    resource: Casualty Actuarial Society, E-Forum Summer 2016
    title: Morris (2016), "Hierarchical Compartmental Models for Loss Reserving", sec. 3, the upper triangles of company 337
  - id: test
    resource: ../validation/tests/reserving_compartmental.rs
    title: The validation test that reads it
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving ("Data")
---

# Facts

* The source is the CAS loss reserve database (NAIC Schedule P, assembled
  for research by Meyers and Shi) as rebuilt by the user's public
  repository `EKtheSage/cas-schedule-p-data-model`: table
  `mart_reserving_model_training` of release `20260613_041006`, Parquet
  sha256 `8d18b0b12b7b1915a0ad34c30f90b56453555ee729786a29872b40baad903196`
  (the same file as group 353's), which matches the release's
  `manifest.json`.[^repo][^script]
* Selection: `company_code = '337'`, `line_of_business =
  'workers_compensation'`, accident years 1988-1997, `development_age` 1 to
  10: exactly 100 rows. Columns, USD thousands: `paid` is `cum_paid_loss`;
  `incurred` is `incurred_loss` (with bulk and IBNR); `case_incurred` is
  `incurred_loss - bulk_loss` (Meyers' convention); `premium` is
  `earned_prem_direct`; `premium_net` is `earned_prem_net`.[^script]
* The two outstanding definitions differ by the bulk reserve: at 1988 age 1
  paid is 9,558, incurred 62,679, bulk 24,619, so the papers' outstanding
  (incurred less paid) is 53,121 and the case reserve 28,502. Morris
  (2016) prints premium 104, outstanding 53 and paid 10 (USD millions) for
  that cell, which fits only incurred with bulk and IBNR and the direct
  premium (104,437; net 99,779). The generator asserts the cell, and every
  cell of Morris' printed upper triangles (calendar years up to 1997:
  premium, outstanding and cumulative paid, headed $'000s but in USD
  millions) to within 0.5 of the printed figure.[^script][^morris]
* Each origin has one direct and one net premium. Direct premium,
  1988-1997: 104,437, 88,883, 85,956, 99,339, 104,897, 119,427, 110,784,
  77,731, 63,646, 48,052.[^test]
* Every cell has incurred at or above paid and positive paid and case
  incurred. Incurred with bulk falls with age in most rows (1988: 62,679 at
  age 1 to 53,261 at age 10) as the bulk reserve runs off.[^script]
* The papers' training set is calendar years up to 1996 (`origin + age - 1
  <= 1996`): 45 cells of each type over accident years 1988-1996, 90
  observations, and no cell for 1997, whose premium alone enters the model.
  The 1997 diagonal is the test set.[^test]
* Generator: `uv run --no-project --python 3.14 --with duckdb==1.5.6
  python -I -X utf8 validation/scripts/clrd_wkcomp_337.py` (needs `gh`;
  the downloads go to a temporary directory).[^script]
