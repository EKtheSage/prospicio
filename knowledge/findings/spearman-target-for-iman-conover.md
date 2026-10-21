---
type: Finding
title: A Spearman target needs converting before Iman-Conover, or every achieved rho comes out low
description: "Iman-Conover rotates normal scores to a Pearson correlation r; the ranks then have Spearman's rho near (6/pi) asin(r/2), slightly below r between 0 and 1. A model that states a Spearman matrix rho must pass r = 2 sin(pi rho / 6). Without it a reference pricing model's 21 pairs each came out about 0.007 low, 9 standard errors."
tags: [prospicio-prob, iman-conover, spearman, dependence, portfolio]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T21:00:00Z }
verified: { by: process:ci, at: 2026-10-21T21:00:00Z }
sources:
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/481
    title: "Issue #481: Spearman input and a long pairwise table for reorder_groups"
  - id: kruskal
    resource: https://doi.org/10.1080/01621459.1958.10501481
    title: Kruskal (1958), Ordinal measures of association, JASA 53 (rho_S = (6/pi) asin(r/2) for the bivariate normal)
  - id: test
    resource: ../crates/prospicio-prob/src/portfolio.rs
    title: a_spearman_target_is_met_and_a_pearson_one_falls_short
---

# A Spearman target needs converting before Iman-Conover

`PredictiveDistribution::reorder_groups` takes the correlation `r` of
the normal scores it rotates to. For a bivariate normal, Spearman's rho
is `(6/π) asin(r/2)`, a little below `r` on `(0, 1)`: at `r = 0.7` it is
0.683, at `r = 0.3` it is 0.287. A model whose matrix is a Spearman
target (a reference simulation pricing model stores one, as a long table
of pairs) gets every achieved rho up to about 0.02 low unless it passes
`r = 2 sin(π ρ / 6)`. `CorrelationKind::Spearman` on
`reorder_groups_with` does the conversion; the converted matrix must
still be positive definite, which a valid Spearman matrix near the
boundary may not be.

`group_spearman` checks the result: Spearman's rho between the group
totals with ties at their average rank (`scipy.stats.spearmanr`, R
`cor(method = "spearman")`). Ties matter for group totals that are often
zero (a layer with no loss in most years): with first-in-order ranks
instead of average ranks the rho would depend on the row order.

On 20,000 simulations of three groups with targets 0.5, 0.3 and 0.7,
the converted run lands within 0.01 of every target; the unconverted one
gives about 0.683 for the 0.7 pair.
