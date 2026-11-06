---
type: Finding
title: HierarchicalBf's one-year view is right under refits, and its reweighting fails on panels whose programs stop at different ages
description: The refit-only mean of the claims development result is zero (z -0.03 over 1,100 simulations) and the year's likelihood is the model's; the reference panel's Pareto k above 6 goes with EVW stopping at 6 of 10 ages, and a control with copies of GenIns and RAA shows k rising when one program stops short of the grid, though not to EVW's size; on programs with the same ages k is mostly below 0.7 and reweighting is within 0.05 sd of a refit at 1,000 draws.
tags: [reserving, bayesian, hierarchical, one-year, psis, moment-matching]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-06T12:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-reserving/src/bayes/hierarchical_bf.rs
    title: HierarchicalBf::one_year and its YearModel
  - id: code
    resource: ../validation/tests/reserving_bayes_one_year_conformance.rs
    title: The conformance suite on the reference panel (HIERARCHICAL_REFITS, HIERARCHICAL_CUT_REFITS)
  - id: measurements
    resource: ../validation/tests/reserving_bayes_hierarchical_bf_one_year.rs
    title: The ignored measurements of this finding
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, decision 30, "As built (PR 17)"
---

Measured on the three-program Stan reference panel (EVW, GenIns, RAA with
premium; 2 chains of 200 warm-up and 200 kept draws unless stated, release
build, 16 threads, another job sharing the machine). The tests are the
ignored ones of `validation/tests/reserving_bayes_hierarchical_bf_one_year.rs`
(`refit_only_mean_of_the_cdr`, `pareto_k_by_panel_and_draws`,
`pareto_k_with_every_program_cut_to_six_ages`,
`pareto_k_when_one_program_stops_short`, `reweighting_against_refits`,
`scales_and_diagnostics_by_panel`) and the conformance sweep. None of them
runs in a pull request's CI jobs, so the finding is not marked verified.

**The view is right.** Under refits only (`RefitAll`, which is what the
default does on the reference panel) the mean of the claims development
result is zero by the tower property, and the measurement agrees. Over 1,100
simulations (seeds 0 and 1 at 200, seeds 2 and 3 at 350) the total's mean is
-72,000 with standard error 2.82 million, z -0.03 (0.03 sd of the total; per
seed z 0.74, 0.67, -0.72, -0.09). By program: EVW z 0.00, GenIns -1.15
(mean -61,700, se 53,500), RAA 0.76 (mean 145, se 192; one seed alone has z
2.25). The measurement can see the reweighted view's bias, which is 0.12 sd
of the total (z 11.8 at 10,000 simulations), because its standard error is
0.03 sd. Every refit reported sampler warnings at this short sampler (low
effective sample size, as the fit's own at 200 draws).

**The year's likelihood is the model's.** A unit test
(`the_year_likelihood_is_the_stated_likelihood_of_the_year`) computes each
program's year log likelihood from the formulas alone (its own `phi_c`, the
mean `exp(alpha) p_d` with `p` the softmax of `beta + omega delta`) at six
draws and finds `year_log_likelihood` equal to 1e-10, and equal to the
difference of the posterior with and without the year's cells. A second
(`the_extended_data_are_the_opening_cells_and_the_year_and_the_year_is_drawn_from_the_draw`)
shows that a refit's data are the opening cells plus exactly the year's
cells (none of them already observed, `phi_c` unchanged) and that the year is
drawn from the draw's own mean and its program's `phi_c` (mean and variance
`phi mu` over 40,000 draws, within 3% and 8%). No term in theta is missing.

**Why every simulation is flagged.** It is not the number of programs, cells
or draws. The median Pareto k of 60 simulations (seed 0) by panel, at 200 and
at 2,000 draws per chain:

| panel | 200 draws | 2,000 draws |
|---|---|---|
| EVW alone, GenIns alone, RAA alone | 0.11, 0.46, 0.34 | 0.03, 0.31, 0.15 |
| GenIns and RAA | 0.52 | 0.41 |
| GenIns twice (a copy) | 0.65 | 0.49 |
| EVW and RAA | 9.2 | 3.1 |
| EVW and GenIns | 46 | 30 |
| all three | 10.3 | 6.9 |
| all three cut to 6 ages | 0.47 | 0.40 |
| EVW and GenIns cut to 6 ages | 0.49 | 0.29 |

(`BayesBfNormalised` on each triangle alone: 0.08, 0.67, 0.46 at 200 draws.)
Every panel with EVW and another program is flagged. Cutting every program
to 6 ages gives median k of 0.47 and 0.49 for the three programs and for EVW
with GenIns, but it also removes cells and shrinks the panel, so by itself it
does not show the cause. The reference panel's grid has 10 ages and EVW
stops at 6, so the next diagonal of EVW's older origins falls at ages 7 to
10, where its shares are held by the prior alone (the deviations of an
unobserved age).

*A control.* `pareto_k_when_one_program_stops_short` changes only whether a
program stops short of the grid, with the same data otherwise (a second copy
of GenIns or of RAA stands in for the other program). Median k and the share
of 60 simulations above 0.7, seed 0:

| panel (ages kept) | 200 draws: median k, share above 0.7 | 2,000 draws |
|---|---|---|
| GenIns 10 + GenIns 10 | 0.65, 0.45 | 0.49, 0.10 |
| GenIns 10 + GenIns 6 | 1.06, 0.82 | 0.78, 0.62 |
| GenIns 6 + GenIns 10 | 1.03, 0.85 | 0.66, 0.40 |
| GenIns 6 + GenIns 6 | 0.34, 0.12 | 0.31, 0.05 |
| RAA 10 + RAA 10 | 0.50, 0.18 | 0.34, 0.05 |
| RAA 10 + RAA 6 | 0.71, 0.50 | 0.47, 0.25 |
| RAA 6 + RAA 6 | 0.26, 0.03 | 0.21, 0.00 |
| GenIns 10 + RAA 10 | 0.52, 0.27 | 0.41, 0.03 |
| GenIns 10 + RAA 6 | 0.96, 0.85 | 0.66, 0.40 |
| GenIns 6 + RAA 10 | 0.80, 0.60 | 0.56, 0.28 |
| GenIns 6 + RAA 6 | 0.29, 0.05 | 0.21, 0.00 |
| EVW 6 + GenIns 10 | 46, 1.00 | 30, 1.00 |
| EVW 6 + GenIns 7 | 1.10, 0.85 | 0.75, 0.52 |
| EVW 6 + GenIns 6 | 0.49, 0.07 | 0.29, 0.02 |

Two equal programs at full length are at the level of the single ones, and so
are both cut to 6 ages (the cut alone does not raise k, and lowers it). A
program that stops at 6 beside one at 10 raises the median k by 0.2 to 0.4
and the flagged share from 0.18 to 0.50 for RAA, from 0.45 to 0.82 for
GenIns, and from 0.27 to 0.85 for GenIns with RAA (at 200 draws; the
2,000-draw column shows the same ordering). The control supports the stated
cause: a program whose next diagonal falls at ages it has not observed
raises k, even when the data are copies of one another. It does not explain
the size. EVW beside GenIns has a median k of 46 (30 at 2,000 draws),
against at most 1.06 for any of the copies, and EVW with GenIns cut to 7
ages, one age short, has a median of 1.10 but a maximum of 54. What else
about EVW makes the weights so concentrated (its premium is the prior
ultimates, and its cells are the smallest of the three) is not tested. That
matching cannot repair a diffuse prior-only direction is a conjecture: no
test isolates it. The fits are sound (2,000 draws: rhat at most 1.01, at
most 3 divergences in any panel).

**Reweighting where k is low.** On the programs alone and on the stacked
panels cut to 6 ages, the root mean square gap between a reweighted and a
refitted simulation's total (in sd of the total, 40 simulations, seeds 0 and
1, 200 draws) is 0.02 to 0.03 for EVW, 0.10 to 0.12 for GenIns, 0.08 to 0.11
for RAA and 0.06 to 0.10 for the programs cut to 6 ages, over the simulations
with k at most 0.7, and the mean gap is within 0.06 sd. At 1,000 draws per
chain the three programs cut to 6 ages give 0.038 and 0.042: most of the
200-draw gap is the refit's own sampling noise.

**The full reference panel.** At 200 draws the reweighted total's mean has z
11.8, 4.3 and 8.2 against zero (seeds 0 to 2, 10,000 simulations) and the
gap to a refit is 1.10, 0.13 and 0.08 sd of the total (20 simulations) and
0.80 and 0.10 (40 simulations, seeds 0 and 1). At 2,000 draws the gap on the
total is 0.046 sd (20 simulations, seed 0) although k stays above 3; the
total is dominated by EVW, so this says little about GenIns and RAA. The
opening reserve is the posterior mean of the lifetime reserve to within z
2.0 at three seeds.

**Conformance.** The full panel's entry (`HIERARCHICAL_REFITS`) has check
1's mean-z bound at 25 where the other entries have 5 or 6, set from three
seeds (observed mean z up to 11.8); a reweighted total biased by 0.12 sd of
the total passes it, so the entry is a regression guard and not a
conformance pass. The conformance evidence is the second entry, on the three
programs cut to 6 ages at 2 chains of 1,000 draws has the other models'
bounds (`HIERARCHICAL_CUT_REFITS`: unflagged gap 0.08, flagged gap 0.25, mean
z and opening z 6.0). Over seeds 0 to 5 the mean z is -0.60 (sd 1.13, |z| at
most 2.58), the opening z -0.31 (sd 1.15), the unflagged gap 0.045 (largest
0.055), the flagged gap at most 0.115, and 0 to 6 of 40 simulations are
flagged. It takes about 4 minutes a seed.

**Cost.** One fit takes about 5.1 s at the 200-draw sampler, and a refit
about 1.2 s a simulation on 16 threads (40 simulations in 55 s with the
fit), so the default view runs at most 2,000 simulations
(`HierarchicalBf::ONE_YEAR_SIMS`), as Meyers' and the compartmental models'.
At the default sampler (4 chains of 1,000 and 1,000, `target_accept` 0.99) a
refit costs several times more. A light refit (warm start, shorter sampler,
as `Compartmental`'s) is not built; decision 30 lists it as a follow-up.

**Limit.** One program with a tight hyperprior and spreads near 0 gives
`BayesBfNormalised`'s one-year view within 0.1 sd in the total and origins'
means and 10% in their sds, both under `ReweightOnly`.

**Open.** A panel whose programs stop at different ages gets the expensive
view. Whether to refuse it, warn, or leave the ages beyond a program's last
out of its year is the user's decision and is not taken here.
