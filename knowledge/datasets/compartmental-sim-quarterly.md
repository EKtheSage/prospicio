---
type: Dataset
title: A quarterly-development triangle simulated from compartmental model 1
description: Ten annual accident years (2011-2020) with quarterly development to the end of 2020, 220 outstanding and 220 cumulative paid cells, simulated from Gesmann and Morris' model 1 with known parameters (k_er 3.3155, k_p 0.8187, population RLR 0.855 and RRF 0.724, sigma_os 1,500, sigma_pd 500), the only unequal-grain triangle in the repository, used for the compartmental model's unequal-grain parity and recovery tests.
resource: ../validation/data/compartmental_sim_quarterly.csv
tags: [reserving, triangle, compartmental, bayes, simulated, quarterly, grain]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-22T23:00:00-07:00 }
sources:
  - id: script
    resource: ../validation/scripts/compartmental_grain_simulate.py
    title: The generator (numpy, seed 20261022)
  - id: test
    resource: ../validation/tests/reserving_compartmental_grain.rs
    title: The validation test that reads it
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decision 13)
---

# Facts

* No public dataset has unequal origin and development grain, so this one
  is simulated.[^design]
* The model is the one of `validation/scripts/stan/compartmental_gm1.stan`
  on its own scale: `k_er = 3 e^(0.1 b_ker)`, `k_p = e^(0.1 b_kp)`,
  `RLR_i = 0.7 e^(0.2 (b_RLR + r_i1))`, `RRF_i = 0.8 e^(0.1 (b_RRF +
  r_i2))`, with `r_i = diag(sd) L z_i`; each cell is normal about premium
  times the closed-form outstanding or cumulative paid at `t` = age in
  months over 12.[^script]
* Truth: `b_ker` 1, `b_kp` -2, `b_RLR` 1, `b_RRF` -1, `tau_RLR` 0.4,
  `tau_RRF` 0.2, `rho` 0.3, `sigma_os` 1,500, `sigma_pd` 500, so `k_er` =
  3.3155 and `k_p` = 0.8187; each origin's `RLR` (0.773 to 0.920) and
  `RRF` (0.705 to 0.739) are in the file's header. Premium is 100,000 in
  2011, growing 2% a year.[^script]
* Shape: origin 2011 at 3 to 120 months (40 quarterly ages), 2020 at 3
  to 12 (4); 220 rows, columns `origin, development, paid, outstanding,
  premium`. With the cumulative paid drawn independently each quarter,
  paid can fall from one quarter to the next (model 1's normal
  likelihood allows it).[^script]
* The expected total reserve to 120 months given each origin's latest
  observed paid is 74,389 at the truth (the closed form at 120 months
  less the latest paid, from the header's parameters).[^script]
* sha256 `d9c97c9db2626143a8ccc03332fa533331999b8ae9379b32f14c138ba6f9fc49`;
  the generator reproduces it.[^script]

