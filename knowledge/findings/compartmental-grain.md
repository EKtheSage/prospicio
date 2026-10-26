---
type: Finding
title: Compartmental on quarterly development matches its Stan reference and recovers the truth of a simulated triangle; time in years keeps every equal-grain fit unchanged
description: With each cell's development time in years on every grain and the square in development periods, Compartmental model 1 on ten annual origins with quarterly development (simulated with known parameters) matches model 1 in Stan at all 2,620 reference-row comparisons of a 20-seed sweep (model 1's tolerances; the largest at 0.51 of its tolerance); every one of 31 true parameters lies in the central 95% of the default fit's posterior, and its total reserve is 74,834 (sd 1,900) against the truth's 74,389 and Stan's 74,873 (1,895). Fits on company 337, annual and relabelled as equal quarterly grain, models 1 and 2, are bit for bit what they were.
tags: [bayes, reserving, compartmental, stan, grain, quarterly, simulated, validation]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-22T23:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_grain.rs
    title: The parity, recovery and square tests and the seed sweep
  - id: csv
    resource: ../validation/reference/reserving_compartmental_grain_stan.csv
    title: Stan summaries of model 1 on the simulated quarterly triangle
  - id: script
    resource: ../validation/scripts/reserving_compartmental_grain_stan.py
    title: The reference generator
  - id: data
    resource: ../datasets/compartmental-sim-quarterly.md
    title: The simulated triangle and its truth
  - id: unit
    resource: ../crates/prospicio-reserving/src/bayes/compartmental.rs
    title: The unit tests on a quarterly triangle (square, times, increments, gradient, one-year conformance)
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decision 13)
---

# Findings (2026-10-22)

* **Time in years, not origin periods.** Before this change every cell's
  time was its age in months over 12 on every grain, equal or not. Taking
  time in origin periods instead would leave annual-origin fits unchanged
  but change every fit on an equal quarterly or monthly triangle and make
  the rates per quarter or per month, against the paper's preset (rates
  per year). Keeping years leaves every existing fit bit for bit
  unchanged: the seed-0 fits of company 337 (annual, and relabelled as
  quarterly origins with quarterly development), models 1 and 2, gave the
  same total reserve mean and sd and the same first and last posterior
  draw to the last bit before and after the change (total reserve
  136,753.30605254587 on the annual model 1 fit without 1997; 146,728.75995580832
  on the quarterly relabelling, whose rates per year differ since its
  ages are a quarter as long).[^unit][^design]
* **Parity.** All 2,620 reference-row comparisons of a 20-seed sweep pass, the largest at 0.51 of its tolerance (a parameter mean), and the total reserve is 74,834 (sd 1,900) at seed 0 against Stan's 74,873 (1,895); every seed passed every diagnostic threshold (max R-hat at most 1.0064, no divergent transition). The Stan program draws a reserve for 2011, observed at the square's last age, where decision 5 gives it 0, so the generator sets Stan's to 0 too (before that fix the five reserve rows of 2011 failed at every seed).[^test][^csv]
* **Recovery.** At seed 0 every true parameter lies within the central 95%
  of its posterior: the truth sits at the posterior's 68.5% for `k_er`
  (3.3155; posterior mean 3.2924), 32.1% for `k_p` (0.8187; 0.8214), 81.0%
  and 42.7% for the population `RLR` and `RRF`, 93.5% and 74.1% for
  `tau_RLR` and `tau_RRF`, 29.3% for `rho`, 52.4% and 71.4% for the two
  sigmas (1,500 and 500; 1,498 and 487), and between 5.5% (`RRF[2018]`)
  and 94.8% (`RLR[2018]`) for the twenty origin parameters. The total
  reserve is 74,834 (sd 1,900), 0.23 sd above the truth's expected
  74,389.[^test][^data]
* **Cost.** A default fit (4 chains of 1,000 warm-up and 2,000 draws,
  10,000 simulations) takes about 32 s in release on 440 cells, against
  about 5.5 s on company 337's 90; it passed every diagnostic at seed 0
  (max R-hat 1.005, bulk ESS 964). Stan took about 74 minutes for 4 chains of
  27,000 iterations (beside other builds), the 20-seed sweep about 8
  minutes.[^test][^script]
* **The one-year view on quarterly development** conforms for models 1
  and 2 on a five-origin quarterly triangle, with four quarterly cells of
  each type per open origin in the year. The light refit's own check
  (that a short warm-started refit samples the closing posterior) failed
  for model 2 when that triangle had only 2% noise on each cell, a
  near-noiseless posterior the short refit could not centre, and passes
  with 8%.[^unit]
