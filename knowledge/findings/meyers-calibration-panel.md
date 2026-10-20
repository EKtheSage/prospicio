---
type: Finding
title: On the rebuilt 200-triangle panel, Mack and the ODP bootstrap reach Meyers' conclusions
description: The panel rebuilt from the cas-schedule-p package agrees with Table A.2 on 149 of its 175 printed codes, and the screen's next-ranked triangles stand in for the rest. Mack on incurred and paid reproduces Meyers' D statistics to within 3.7 points per line and 1.5 on all 200, with the same pass or fail on every line and in total; the ODP bootstrap on paid is 1.9 to 3.9 points above his, again with the same verdicts.
tags: [reserving, meyers, calibration, mack, odp-bootstrap, schedule-p, ks-test]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-20T12:00:00-07:00 }
sources:
  - id: harness
    resource: ../validation/tests/reserving_calibration_meyers.rs
    title: calibration_study (ignored; run by hand) and its unit tests
  - id: script
    resource: ../validation/scripts/calibration_panel.py
    title: Builds target/calibration/panel.csv and panel_check.csv from cas-schedule-p[screens]==2026.6.13
  - id: note
    resource: ../docs/design/reserving-calibration.md
    title: Design note, decisions 3, 5 to 8 and "As built"
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), CAS Monograph 8, Figures 3.2, 4.1 and 4.2, Tables A.1 and A.2
---

# Facts

Run on 2026-10-20 (Windows, release build, the ODP bootstrap at 10,000
simulations and seeds from `seed_for`); not marked verified, since no CI
job runs the study. The numbers are what `target/calibration/ks.csv` held.

* **The panel.** The package's screen gives 50 triangles per line (63
  commercial auto, 60 personal auto, 52 workers' compensation and 56 other
  liability pass it before the top-50 cut). Against Table A.2's printed
  codes it agrees on 16 of 25 for commercial auto (the other 25 of the 50
  are not printed), 44 of 50 for personal auto, 48 of 50 for workers'
  compensation and 41 of 50 for other liability: 149 of 175. Every printed
  code the screen did not choose but which passes it ranks 51st to 63rd, just
  past the cut (commercial auto 8672, 9466, 13528, 14311, 14508, 15024,
  18791, 19780; personal auto 43, 6807, 15024, 15997, 23876, 26808;
  workers' compensation 6807, 30589; other liability 1279, 3000, 11150,
  30139, 32875); five printed codes fail the screen (commercial auto 13420;
  other liability 11231, 13439, 14451, 16799). So the data differ from
  Meyers' by small premium or loss differences that move the CV1 ranking
  near the cut, and in five cases the screen itself. Group 353
  commercial auto's square equals `validation/data/clrd_comauto_353.csv`
  cell by cell and its Mack percentiles are 72.00 (paid) and 86.06
  (incurred), Meyers' 72.02 and 86.03.
* **Mack's D against Meyers'** (percentage points; `*` fails at 5%, bounds
  19.2 per line and 9.6 on all 200):

  | | CA | PA | WC | OL | All |
  |---|---|---|---|---|---|
  | Mack incurred, here | 17.8 | 14.7 | 24.7* | 16.2 | 16.5* |
  | Mack incurred, Meyers | 16.4 | 16.6 | 27* | 14.2 | 15.4* |
  | Mack paid, here | 23.2* | 48.3* | 33.4* | 10.3 (n = 49) | 24.6* (n = 199) |
  | Mack paid, Meyers | 24.6* | 44.6* | 30.4* | 8.5 | 23.1* |
  | ODP paid, here | 25.6* | 48.8* | 31.3* | 8.7 | 26.3* |
  | ODP paid, Meyers | 23.1* | 44.9* | 28.4* | 6.8 | 24.1* |

  The largest difference for Mack is 3.7 points (personal auto paid) per
  line and 1.5 on all 200; the verdicts, pass or fail, are Meyers' in every
  one of the fifteen cells. The ODP bootstrap is 1.9 to 3.9 points above his
  per line and 2.2 above on all 200.
* **One triangle is skipped by Mack on paid**: other liability group 17043
  (not in Table A.2, a tiny triangle with negative paid cells: an origin
  at -21 at 12 months) errors with "sigma can neither be estimated nor
  interpolated", so Mack paid has n = 49 in
  other liability and 199 in all. Mack incurred and the ODP score all 200.
* **Cost.** All 600 fits (Mack on two bases, the ODP bootstrap) take 1.6 s
  in release.

# Not yet shown

The Bayesian arms (`BayesOdp`, Meyers' CRC, CSR and CAY) and Compartmental
are PR 3 and PR 4. The D values here are recorded, not pinned.
