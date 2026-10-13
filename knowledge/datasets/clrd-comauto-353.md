---
type: Dataset
title: Meyers' illustrative insurer, group 353 Commercial Auto
description: The full 10 x 10 square of CAS loss reserve database group 353, Commercial Auto, 1988-1997 (paid, case incurred, net earned premium), taken from the CAS Schedule P data model release 20260613_041006; it matches Meyers' Tables 2.1 to 2.3 cell by cell except one misprint, and its age-10 sums are his outcomes, paid 40,000 and incurred 40,061.
resource: ../validation/data/clrd_comauto_353.csv
tags: [reserving, triangle, clrd, schedule-p, meyers, bayes, premium]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T17:30:00-07:00 }
sources:
  - id: repo
    resource: https://github.com/EKtheSage/cas-schedule-p-data-model
    title: EKtheSage/cas-schedule-p-data-model, release 20260613_041006, table mart_reserving_model_training
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), Stochastic Loss Reserving Using Bayesian MCMC Models, 2nd ed., CAS Monograph 8, Tables 2.1-2.3 (p. 4-5)
  - id: script
    resource: ../validation/scripts/clrd_comauto_353.py
    title: The generator
  - id: test
    resource: ../validation/tests/reserving_meyers.rs
    title: The validation test that reads it
---

# Facts

* The source is the CAS loss reserve database (NAIC Schedule P, assembled
  for research by Meyers and Shi) as rebuilt by the user's public
  repository `EKtheSage/cas-schedule-p-data-model`: table
  `mart_reserving_model_training` of release `20260613_041006`, Parquet
  sha256 `8d18b0b12b7b1915a0ad34c30f90b56453555ee729786a29872b40baad903196`,
  which matches the release's `manifest.json`
  (`tables.mart_reserving_model_training.sha256`).[^repo][^script]
* Selection: `company_code = '353'`, `line_of_business =
  'commercial_auto'`, accident years 1988-1997, `development_age` 1 to 10:
  exactly 100 rows. `paid` is `cum_paid_loss`; `incurred` is case incurred,
  `incurred_loss - bulk_loss`, as Meyers' `ins.line.data` computes it;
  `premium` is `earned_prem_net`, Meyers' `net_premium` (column 11 of the
  CAS files, `EarnedPremNet_C`). USD thousands.[^script]
* The premium is constant within each origin: 5812, 4908, 5454, 5165, 5214,
  5230, 4992, 5466, 5226, 4962. Table 2.1 labels it "Net Written Premium";
  the code uses net earned premium, and the values agree.[^monograph]
* Every cell matches Tables 2.2 and 2.3 as printed except one: Table 2.2
  prints 2004 for accident year 1995 at lag 6, where the database holds
  4116 (between the printed 3678 and 4117). The printed row's "Source"
  column reads 1997 instead of 2004, so the year slipped into the cell. The
  generator transcribes the tables and asserts every cell, with that one
  listed as a misprint.[^monograph][^script]
* Age-10 sums: paid 40,000 and incurred 40,061, Meyers' outcomes (Tables
  5.2 and 5.3); the generator asserts them, and so does the validation
  test.[^test]
* Every cell is positive (Meyers' code raises values below 1 to 1, which
  never applies here). Paid is not monotone: 1989 falls from 2531 to 2527
  at lag 10; incurred falls in several rows.
* The training triangle is the 55 cells at calendar years up to 1997
  (`origin + age - 1 <= 1997`), as Meyers' `cal < 11`.
* Generator: `uv run --no-project --python 3.14 --with duckdb==1.5.6
  python -I -X utf8 validation/scripts/clrd_comauto_353.py` (needs `gh`;
  the downloads go to a temporary directory). `-I` ignores `PYTHONUTF8`, so
  UTF-8 output needs `-X utf8`: printing a DuckDB relation on the Windows
  console's cp1252 fails without it.[^script]
