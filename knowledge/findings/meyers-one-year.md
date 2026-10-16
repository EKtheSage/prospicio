---
type: Finding
title: Meyers' one-year view flags nearly every simulation, and moment matching rescues most
description: In the one-year view of Meyers' CRC, CSR and CAY on group 353, 98.5% to 99.7% of the simulations have Pareto k above 0.7 because each year cell carries its origin's smallest variance; moment matching rescues 83% to 90% of them within about 0.08 sd of their NUTS refits (reweighting alone 0.19 to 0.21), and the one-year sd is about four fifths of the lifetime sd, as for the ODP bootstrap.
tags: [bayes, reserving, one-year, psis, importance-sampling, moment-matching, meyers, crc, csr, cay, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_meyers_one_year.rs
    title: Meyers' one-year view against refits and the ODP bootstrap on group 353, and the seed sweep
  - id: conformance
    resource: ../validation/tests/reserving_bayes_one_year_conformance.rs
    title: The release half of the one-year conformance suite, with Meyers' models
  - id: year
    resource: ../crates/prospicio-reserving/src/bayes/meyers/year.rs
    title: MeyersYear, the YearModel of CRC, CSR and CAY
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 19)
---

# Finding

Meyers' one-year view reweights the posterior draws on the unconstrained
scale NUTS samples (`u_d` for the steps of `sigma_d^2`, `atanh(rho)`),
which the segment fit now keeps beside its natural-scale posterior; a
year's values are the cumulative values of the cells valued in the next
twelve months.[^year]

Over seeds 0 to 19 on group 353 (CRC and CSR on paid, CAY on incurred, the
defaults; 500 simulations, the first 60 also refitted; mean and
seed-to-seed sd):[^test]

| | CRC paid | CSR paid | CAY incurred |
|---|---|---|---|
| flagged (k above 0.7) | 99.3% (0.3%) | 98.5% (0.5%) | 99.7% (0.3%) |
| rescued by moment matching | 90.4% (3.4%) | 90.0% (3.9%) | 83.3% (5.2%) |
| rms gap to the refit, matched (sd of the total) | 0.080 (0.029) | 0.075 (0.019) | 0.084 (0.027) |
| the same simulations, reweighted only | 0.212 (0.059) | 0.191 (0.037) | 0.211 (0.043) |
| total CDR sd over lifetime reserve sd | 0.769 (0.081) | 0.798 (0.071) | 0.825 (0.083) |

* Nearly every simulation is flagged, against 14% to 26% for the
  Bayesian ODP on RAA and GenIns. Each year cell is an origin's value at
  an older age than any it has, where Meyers' `sigma_d` is smallest
  (about 0.02 at the late paid ages), so it moves the origin's level far
  relative to its posterior spread, and nine such cells move the
  posterior beyond what reweighting can follow.
* Moment matching brings the rescued simulations about 2.5 times closer
  to their refits than reweighting alone, at every seed (largest ratio
  0.88), with the refits' spread (sd ratio within 0.01 of 1). The matched
  gap is about twice the ODP models' (0.036 to 0.058), and 10% to 17% of
  the flagged simulations stay above 0.7 and are refitted.
* The ODP bootstrap's one-year sd on the same paid triangle is 0.803
  (sd 0.008) of its lifetime sd; CRC's ratio over it is 0.957 (0.097).
  The models' sds differ, the share that emerges in the first year does
  not by much.
* In the release conformance suite (4 chains of 300 warm-up and 300
  kept draws, 60 refits) the reweighted gap of the flagged simulations is
  0.33 to 0.36, at most 0.52 over 20 seeds, above the ODP models' bound of
  0.5; Meyers' models take 0.65.[^conformance]

At 10,000 simulations (release, 16 threads) the default takes 249 s
(CRC), 411 s (CSR) and 396 s (CAY), most of it in the 9% to 13% of the
simulations it refits; `RefitFlagged` would take about 6.7 times as long
(171 to 276 s per 1,000 simulations), `ReweightOnly` 3 to 8 s.[^test]

Checked by CI at seed 0 (rescued at least 0.75, 0.74 and 0.62; matched
gap below 0.20, 0.16 and 0.20 and below the reweighted gap), not yet
`verified` in this bundle's sense until it has run there.
