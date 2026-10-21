---
type: Finding
title: A warm-started, shorter refit puts compartmental model 2's one-year view within 0.03 sd of full refits, at about a thirtieth of their cost
description: On company 337, refitting model 2's one-year simulations with NUTS scaled by the opening posterior and started at its mean, 4 chains of 150 warm-up and 250 kept draws, gives totals within 0.029 to 0.035 sd of full refits on three seeds (reweighting only is 0.25 to 0.52 off) at about 0.6 to 0.9 s of wall time a refit against 30 s on 16 threads; 2,000 simulations take 31 minutes there. Measured for model 2 only. Two chains or fewer than about 400 iterations a chain lose it.
tags: [bayes, reserving, one-year, compartmental, gesmann-morris, clrd, nuts, warm-start]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_one_year.rs
    title: model_2_light_refit, light_runtime and the gated comparison with full refits
  - id: year
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/year.rs
    title: CompartmentalYear::refit, the light branch
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 24, open decision 17)
---

# Finding

Issue #394 asked for compartmental model 2's one-year view to come within
about 0.05 sd of the total of a full refit. Moment matching rescues none of
model 2's flagged simulations and a full refit takes about 30 s, so the view
reweighted only (0.25 to 0.52 sd off).[^design] The first of the issue's
ideas worked, so the others (why matching fails, a Laplace or variational
proposal) were not tried.

**The refit.** `BayesOneYear::light_refit` scales NUTS by the mean and
covariance of the opening posterior's draws (with 1997's and the new ages'
effects drawn, on the sampled `z` scale; regularised as Stan's dense
metric), starts at the mean, and runs a short warm-up and few draws. The fit's
refit instead runs a pilot sampler and then 2,000 warm-up and 8,000 kept
draws a chain.[^year]

**Accuracy** (`RefitAll`, the full sampler as reference; rms over the
simulations of the difference of the totals in units of the total's sd,
about 11,000; chains, warm-up, draws):[^test]

| seed, years | reweighting only | 4,150,250 | 4,300,500 | 4,100,150 | 4,50,100 | 4,100,100 | 2,200,300 | 2,300,500 |
|---|---|---|---|---|---|---|---|---|
| 0, 16 | 0.522 | 0.035 | 0.020 | | | | | |
| 1, 32 | 0.252 | 0.029 | 0.025 | 0.059 | 0.059 | 0.076 | 0.047 | 0.045 |
| 2, 24 | 0.297 | 0.033 | 0.038 | | | | | |

Seed 0 also gave 0.015 at 4,500,1000. The mean gap stays within 0.02 of 0
and the ratio of the totals' sds within 1.5% of 1. About 0.03 is the floor
of the comparison (two full refits differ by their own Monte Carlo error),
and 4,150,250 is the shortest run on it. It was chosen on seed 1, so its
0.029 there is in sample; seeds 0 and 2 (0.035, 0.033) were not used to
choose it. Everything here is model 2: model 1 uses the light sampler only
for what moment matching cannot rescue, and those refits were not compared
with full refits.

**Cost** (release, wall time on one 16-thread machine; a time per refit
is a parallel run's wall time over its refits, so throughput, and scales
with the hardware): a refit takes about 0.6 to 0.9 s against 30 s for a full one (1,047 s for 32 full refits and the fit,
on seed 1). 2,000 simulations, every one flagged and refitted, took 1,872 s
(the fit about 60 to 120 s of it), 29 refits warned, mean total claims
development result 145 and sd 11,058; reweighting only takes about 85 s.
With `target_accept` 0.95 in place of 0.999 a refit took 0.35 s and with 0.8
0.20 s, at the same gap (0.036 at seed 1) but with sporadic divergences (a
quarter of the refits).[^test]

**Default (user, 2026-10-21).** The user accepted open decision 17 (b): with
`refit` unset, model 2's view refits its flagged simulations with this light
refit (`RefitFlagged`, no approximate warning), at about 31 minutes for the
default 2,000 simulations on 16 threads; `light_refit=False` (Rust
`no_light_refit`) goes back to reweighting only, with the warning. An explicit
`refit` keeps the fit's own sampler unless `light_refit` is given. Model 1 is
unchanged.

**Warnings.** The fit's diagnostics on a light refit's 95 parameters warned
on every refit (effect sds near 0 mix slowly in a short run); on the total
closing reserve at R-hat 1.01 and 100 effective draws a chain they warned
on 14 of 32, whose totals were within 0.03 sd of the full refits'. So a
light refit warns on its closing reserve at R-hat 1.05 or fewer than 100
effective draws, or a divergence (29 of 2,000).[^year]

Commands (release; add `RISK_RS_FULL_BAYES=1` for the gated comparison):

```
LIGHT_CACHE=<prefix> LIGHT_SIMS=32 LIGHT_SEED=1 LIGHT="4,150,250;4,100,150" \
  cargo test --release -p prospicio-validation --features slow-bayes \
  --test reserving_compartmental_one_year -- --ignored model_2_light_refit --nocapture
LIGHT_SIMS=2000 cargo test --release -p prospicio-validation \
  --features slow-bayes --test reserving_compartmental_one_year \
  -- --ignored light_runtime --nocapture
```

