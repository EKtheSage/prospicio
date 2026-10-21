---
type: Finding
title: On the rebuilt 200-triangle panel, Mack and the ODP bootstrap reach Meyers' conclusions
description: The panel rebuilt from the cas-schedule-p package agrees with Table A.2 on 149 of its 175 printed codes; the other 26 printed codes fall just past the top-50 cut or fail the screen. Mack on incurred and paid reproduces Meyers' D statistics to within 3.7 points per line and 1.5 on all 200, with the same pass or fail on every line and in total; the ODP bootstrap on paid is 1.9 to 3.9 points above his, again with the same verdicts, an offset the swapped triangles alone can produce.
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
job runs the study. The numbers are what `target/calibration/ks.csv` and
`panel_check.csv` held; `panel_meta.csv` records the panel's sha256,
`075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2`, built
with cas-schedule-p 2026.6.13, duckdb 1.5.6, pandas 3.0.6 and Python
3.14.4. A rebuild with another hash is not the reviewed panel.

* **The panel.** The package's screen gives 50 triangles per line (63
  commercial auto, 60 personal auto, 52 workers' compensation and 56 other
  liability pass it before the top-50 cut; `panel_check.csv`, column
  `screen_pass`). Against Table A.2's printed codes it agrees on 16 of 25
  for commercial auto (its other 34 chosen codes are `unchecked`: Table
  A.2 prints 25 of that line's 50, so at least 9 of them differ from
  Meyers' and the rest cannot be told apart), 44 of 50 for personal auto,
  48 of 50 for workers' compensation and 41 of 50 for other liability: 149
  of 175, so at least 26 of the 200 triangles are not Meyers'. Every
  printed code the screen did not choose but which passes it ranks 51st
  to 63rd, just past the cut (`panel_check.csv`, column `rank`;
  commercial auto 8672, 9466, 13528, 14311, 14508, 15024, 18791, 19780;
  personal auto 43, 6807, 15024, 15997, 23876, 26808; workers'
  compensation 6807, 30589; other liability 1279, 3000, 11150, 30139,
  32875); five printed codes fail the screen (commercial auto 13420; other
  liability 11231, 13439, 14451, 16799). The codes chosen in their place
  do not sit at the cut: in personal auto, workers' compensation and
  other liability they rank from 2nd to 50th (other liability 33049 is
  2nd, 38733 8th). So the package's data and Meyers' differ by more than a
  reshuffle near the cut: the inference, not tested, is that some
  triangles pass the screen on the package's data and not on his (or were
  not in his data). Group 353
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
  per line and 2.2 above on all 200. Neither offset is evidence of a model
  difference: a swapped triangle can move a line's D by up to 2 points and
  the total's by 0.5, so the 26 known swaps alone allow up to about 13
  points on all 200 and more than the observed offsets in every line. A
  run on the printed codes only would separate panel from model; it is
  not built. The harness's acceptance tolerance (8 points per line, 4 on
  all 200) is likewise empirical, not implied by the panel difference.
* **One triangle is skipped by Mack on paid**: other liability group 17043
  (not in Table A.2, a tiny triangle with negative paid cells: an origin
  at -21 at 12 months) errors with "sigma can neither be estimated nor
  interpolated", so Mack paid has n = 49 in
  other liability and 199 in all. Mack incurred and the ODP score all 200.
* **Pinned (PR 5, 2026-10-21).** The user reviewed the KS scores and
  approved them, and the study (`PINS`, `TOLERANCES` in the harness) now
  asserts every D of this finding. A rerun on the default seed reproduced
  the D above to their rounding (0.1). Mack does not depend on the seed
  (two salted runs gave the same D to the digit), so its tolerance is 0.1
  point; the ODP bootstrap's D moved by at most 0.84 points in a line and
  0.21 on all 200 between three seeds, and its tolerance is 1.7 and 0.5.
  Still not marked verified: no CI job runs the study. Pins were recorded
  on Windows; the note's "As built (PR 5)" gives the tolerances, their
  derivation and how to re-pin.
* **Cost.** All 600 fits (Mack on two bases, the ODP bootstrap) take 1.6 s
  in release.

# Not yet shown

The Bayesian arms are in [their own finding](/findings/meyers-calibration-bayesian-arms.md);
Compartmental is in its own finding. The D values here are pinned by the
study but not verified (above).
