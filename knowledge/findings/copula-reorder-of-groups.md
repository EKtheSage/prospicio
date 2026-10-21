---
type: Finding
title: Reordering pre-simulated groups by a copula's ranks gives them that copula's joint tails, which Iman-Conover's normal scores do not
description: "Sorting each group's rows by its total and giving them the ranks of a column of copula draws makes the totals' rank pattern exactly the copula's. On 3,000 simulations of two groups, the number in both top 1% was 4 under Iman-Conover at r = 0.5, 11 under a t copula with 2 degrees of freedom at the same r, and 18 under Gumbel theta = 2, whose upper tail coefficient 2 - 2^(1/2) = 0.586 predicts about 17.6."
tags: [prospicio-prob, copula, iman-conover, dependence, portfolio, tail-dependence]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T23:00:00Z }
verified: { by: process:ci, at: 2026-10-21T23:00:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/482
    title: "Issue #482: t, Gumbel and Clayton dependence in reorder_groups on pre-simulated groups"
  - id: test
    resource: ../crates/prospicio-prob/src/portfolio.rs
    title: a_copula_reorder_gives_the_totals_the_copulas_ranks
---

# Reordering pre-simulated groups by a copula's ranks

A reference simulation pricing model simulates each class first (an
attritional draw plus large events) and sets the dependence between
classes afterwards, on whole rows. `reorder_groups` does that by
Iman-Conover on normal scores, which gives a Gaussian copula's ranks
approximately. The model also admits t, Gumbel and Clayton dependence.

`PredictiveDistribution::reorder_groups_by_copula` draws `n` rows of
uniforms from any `Copula` (draw `i` on stream `i` of the seed, as the
bindings' `sample`), sorts each group's rows by the group's total and
gives simulation `i` the row whose rank is that of `u[i][g]`. The totals
then have exactly the rank pattern of the copula draws: their Spearman's
rho is the sample rho of the draws (0.68 for Clayton or Gumbel with
`theta = 2`) and their joint extremes are the copula's. Each group keeps
its own draws and its internal structure, since rows move whole.

Rows with equal totals (a layer with no loss in most years) take their
ranks in row order, so the achieved Spearman's rho with ties at their
average rank (`group_spearman`) can differ a little from the copula's.

Tail counts on 3,000 simulations of two groups, both in their top 1%
(30 rows each; 0.3 expected under independence):

| Dependence | Joint top 1% |
|---|---|
| Iman-Conover, normal scores, `r = 0.5` | 4 |
| t copula, `r = 0.5`, 2 degrees of freedom | 11 |
| Gumbel, `theta = 2` (upper tail coefficient 0.586) | 18 |

Iman-Conover with t scores (`Scores::StudentT`) also fattens the joint
tails, but the result's ranks come from a rotation of shuffled t scores,
not from a t copula's draws, so its tail counts only approximate the t
copula's; the copula reorder is the one to use when the model names the
copula.
