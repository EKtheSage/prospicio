---
type: Finding
title: The compartmental one-year view needs centred effects for moment matching, and model 2 defeats it
description: In the one-year view of Gesmann and Morris' compartmental models on company 337, about four in five (model 1) and all (model 2) simulations have Pareto k above 0.7; with every origin and development effect stored centred, moment matching rescues about 95% of model 1's within about 0.05 sd of their NUTS refits (about 1 in 10 on the non-centred scale NUTS samples), and none of model 2's, whose refits each take as long as a fit (about 31 s of wall time with 8 running side by side), so model 2's default view reweights only, with a warning that it is approximate.
tags: [bayes, reserving, one-year, psis, importance-sampling, moment-matching, compartmental, gesmann-morris, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_one_year.rs
    title: The compartmental one-year view against refits on company 337, the seed sweep and the model 2 comparison
  - id: year
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/year.rs
    title: CompartmentalYear, the YearModel of models 1 and 2
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 20, open decision 15)
---

# Finding

The compartmental model's one-year view adds to each posterior draw the
effects the year's cells inform for the first time (the papers' 1997,
with no observed cell, and in model 2 the development effects at 120
months), drawn from their N(0, 1) prior, which is their posterior given
the observed cells.[^year] The year's 20 cells (every origin's next
diagonal of paid and outstanding) move the posterior far: on company 337
82% (sd 1.9%) of model 1's simulations and all of model 2's have Pareto
`k` above 0.7.[^test]

**Centring.** Moment matching's affine maps act on the stored scale. On
the non-centred scale NUTS samples (`z`, with an effect `u = τ z`) the
year's cells pin `u`, a curved ridge `z = u / τ` in `(log τ, z)`, and
matching rescued about 1 in 10 of model 1's flagged simulations in a
probe (2 of 33 with only 1997's effects centred). With every effect
centred (`u` stored, the log Jacobian `-log τ` per effect added to the
log posterior, which the conformance suite's score check confirms) it
rescues nearly all: the classic rule that a centred parameterization
suits effects the data inform well.[^design]

**Model 1** (seeds 0 to 19, 500 simulations, the first 40 refitted;
mean, sd):[^test]

| | model 1 |
|---|---|
| rescued by matching | 95.5% (1.7%), lowest 91.1% |
| rms gap to the refit, matched (sd of the total) | 0.051 (0.025), largest 0.150 |
| the same simulations reweighted only | 0.156 (0.037) |
| total CDR sd over the lifetime reserve sd | 0.804 (0.031) |

At 10,000 simulations (release, 16 threads, seed 0) model 1's default
view takes 1,957 s: matching rescues 7,746 simulations and the 369 it
cannot are refitted at about 5 s of wall time each; refitting every
flagged simulation would take about 11 hours.[^test]

**Model 2** (seeds 0 to 2, 200 simulations moment-matched without
refits, the first 8 refitted): every simulation is flagged (median `k`
about 1.5) and matching rescues none, on either scale; even the first
shift raises `k`. Its posterior has 95 parameters with several effect
sds near 0, far from the near-normal shape affine maps can follow.
Reweighting alone is 0.39 sd (0.20 to 0.73) off the refits, and the
total CDR sd is 0.82 (0.78 to 0.84) of the lifetime reserve's. So
moment matching would refit nearly every simulation; a refit at model
2's sampler (4 chains of 2,000 warm-up and 8,000 kept draws) took about
31 s of wall time with 8 running side by side, so 10,000 simulations
would take days.[^test] The user decided on 2026-10-16 (open decision 15
of the design note) that model 2's default view (any `Compartmental`
with development effects) reweights only, with a warning that it is
approximate (`BayesOneYearFit::approximate`), that refits stay a choice
(`refit`, with `max_refits` as a budget), and that every compartmental
view runs at most 2,000 one-year simulations by default
(`Compartmental::ONE_YEAR_SIMS`).[^design] The budget
applies to each segment of `one_year_segments`. A simulation beyond it
is kept between the engine's two passes as its reweighted closing
reserve, one value per origin: keeping its weights instead (one per
posterior draw) held `n_sims` times `n_draws` values, a peak of 1,239 MB
against 24 MB for `BayesOdp` on GenIns (4 chains of 4,000 draws, 10,000
simulations, `RefitAll` with a budget of 0; release build, Windows,
measured on 2026-10-16), and about 2.6 GB at model 2's 32,000 draws.

Model 1's checks run in CI at seed 0 (rescued at least 0.88, matched
gap below 0.20 and below the reweighted gap, the CDR sd between 0.68
and 0.93 of the lifetime sd), not yet `verified` until they have run
there. A pull request checks all but the gap, whose 40 `RefitAll`
refits took 199 s of the test's 368 s locally (release, 16 threads,
2026-10-16); the weekly and manual runs (`RISK_RS_FULL_BAYES`) check
the gap too and model 2's default view end to end.

Over segments (`Compartmental::one_year_segments`, branch
claude/compartmental-one-year-segments) each segment's view is the single
triangle's view on the segment's own seeds. One local run (release,
Windows, seed 0, model 1 at its default sampler, reweighting only, 500
simulations) of company 337 under two names gave total claims development
result sds of 8,021 and 8,748 against 7,872 for `one_year` of the
triangle alone (ratios 1.02 and 1.11, from other posterior samples and
other years), and rank-uncorrelated copies. The view took 22 s for the two
segments in the pull request's smoke run (34 s with the single-triangle
comparison) and 225 s in the full run, which also moment-matches 100
simulations a segment under a budget of 2 refits and views model 2's
default over two segments (reweighted, warned as approximate).[^test]

**Update 2026-10-21.** A warm-started, shorter refit (`light_refit`) brings
model 2's view within about 0.03 sd of full refits, at about a thirtieth
of their cost; see [the light refit
finding](compartmental-gm2-light-refit.md). Model 2 still reweights only by
default (open decision 17 of the design note).
