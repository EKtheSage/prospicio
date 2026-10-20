---
type: Dataset
title: Meyers' 200-triangle Schedule P panel (rebuilt, not stored)
description: The 50 triangles each of commercial auto, personal auto, workers' compensation and other liability that Meyers' screen selects, rebuilt at run time from the cas-schedule-p package for the calibration study; Table A.2 prints only 175 of the 200 codes (25 for commercial auto), so the panel is checked by Mack reproducing Meyers' published D statistics.
resource: ../docs/design/reserving-calibration.md
tags: [reserving, triangle, clrd, schedule-p, meyers, calibration, panel]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-20T12:00:00-07:00 }
sources:
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), Stochastic Loss Reserving Using Bayesian MCMC Models, 2nd ed., CAS Monograph 8, Appendix (Tables A.1 and A.2, p. 83-85), Figures 3.2 and 4.2
  - id: package
    resource: https://pypi.org/project/cas-schedule-p/2026.6.13/
    title: cas-schedule-p 2026.6.13, bundles the parquet and cas_schedule_p.screens.select_companies
  - id: repo
    resource: https://github.com/EKtheSage/cas-schedule-p-data-model
    title: EKtheSage/cas-schedule-p-data-model, the source of the package's data
  - id: note
    resource: ../docs/design/reserving-calibration.md
    title: The design note of the calibration study (accepted 2026-10-20)
---

# Facts

* The panel is 50 triangles in each of four lines (commercial auto, personal
  auto, workers' compensation, other liability), accident years 1988 to
  1997, 200 in all. It is **not stored in the repo**: the user decided
  (2026-10-20) that the study builds it at run time from the pip package
  `cas-schedule-p==2026.6.13` into `target/calibration/panel.csv`, which is
  gitignored.[^note][^package]
* Meyers' screen: premium above 20 and incurred loss above 4 (USD
  thousands), net earned premium CV1 and net-to-direct ratio CV2 under
  Table A.1's limits (CA 0.795 and 0.125, PA 1.003 and 0.125, WC 0.772 and
  0.300, OL 0.628 and 0.15), then the 50 smallest CV1 per line, with group
  38997 removed by hand.[^monograph]
* Table A.2 prints 175 codes, not 200: 25 for commercial auto (18 on p.84,
  8672 to 14974, and 7 on p.85, 15024 to 19780) and 50 each for the other
  three lines. That the 25 missing are lower-numbered CA codes is an
  inference; group 353, Meyers' illustrative CA insurer, is not among the
  printed CA codes.[^monograph]
* Meyers' D statistics (percentage points) that the rebuilt panel must
  reproduce through Mack, by line CA, PA, WC, OL and all 200: incurred
  16.4, 16.6, 27, 14.2, 15.4 (Figure 3.2, p.10); paid 24.6, 44.6, 30.4, 8.5,
  23.1 (Figure 4.2, p.15). The 5% bounds are 19.2 per line and 9.6 on all
  200.[^monograph]
* Built in the calibration study's PR 2 (2026-10-20) by
  `validation/scripts/calibration_panel.py` as `target/calibration/panel.csv`
  (20,000 rows, columns `line, group_code, origin, development, paid,
  incurred, case_incurred, premium, premium_direct`) with the diff
  `panel_check.csv` (each code's status and its rank in the screen) and
  `panel_meta.csv` (the package versions and the panel's sha256; the first
  run's is `075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2`,
  cas-schedule-p 2026.6.13, duckdb 1.5.6, pandas 3.0.6). It agrees with
  Table A.2 on 149 of the 175 printed codes (16 of 25 CA, 44 of 50 PA, 48 of
  50 WC, 41 of 50 OL), so at least 26 of its triangles are not Meyers', and
  Mack's D on it is within 3.7 points per line and 1.5 on all 200 of
  Meyers'; see [the finding](/findings/meyers-calibration-panel.md). The package needs the
  `screens` extra (`cas-schedule-p[screens]==2026.6.13`) and Python 3.11 or
  newer.
* Status stays draft: no CI job builds the panel, so nothing here is
  verified.

[^monograph]: Meyers (2019), the pages named.
[^package]: The package's `select_companies` rebuilds the screen from the appendix criteria, not Meyers' list.
[^note]: `docs/design/reserving-calibration.md`, decisions 2 and 3.
