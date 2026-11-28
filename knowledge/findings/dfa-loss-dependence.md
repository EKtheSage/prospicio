---
type: Finding
title: Reordering a year's simulated losses across the economic paths by a copula's draws gives them the copula's rank dependence on the economy and keeps their distribution exactly
description: Sort the paths by the year's index growth, pair them with copula draws sorted by their first coordinate, and give each path the loss whose rank is that draw's second coordinate's rank; the losses' empirical distribution is unchanged and their Spearman correlation with the growth is the copula's, (6 / π) asin(ρ / 2) for a Gaussian, within 0.03 at 10,000 paths.
tags: [esg, dfa, copula, dependence, simulation]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-28T04:00:00Z }
verified: { by: process:ci, at: 2026-11-28T04:00:00Z }
sources:
  - id: note
    resource: ../docs/design/esg-dfa.md
    title: Design note, an economic scenario generator and multi-year DFA, decision 7 as built
  - id: test
    resource: ../validation/tests/aggregate_dfa_dependence.rs
    title: Validation test of the projection's loss dependence, run in every cargo test
---

# What was found

The economic paths are fixed (the scenarios) and so are the year's
simulated losses, as a set. Dependence between them is put in by choosing
which path gets which loss: the reordering of Iman and Conover, with a
copula's draws as the reference in place of scores. For year `t`:

1. sort the paths by the year's index growth `I_t / I_(t-1)`;
2. draw `n` pairs `(u1, u2)` from the copula and sort them by `u1`;
3. give the path with the `i`-th smallest growth the loss whose rank among
   the year's losses is the rank of the `i`-th pair's `u2`.

The year's losses are a permutation of themselves, so every statistic of
their distribution is unchanged, and the ranks of (growth, loss) are the
ranks of the copula's draws, so the rank correlation is the copula's up to
sampling error.

# What it means here

A copula on the ranks needs no marginal model of the losses or of the
index: the losses can come from any simulation. The reordering is by year,
so it ties each year's losses to that year's growth and not to the path's
other years; a dependence between a path's years in the input is not kept.
A Gaussian with `ρ = 0.6` gave Spearman correlations within 0.03 of `(6 /
π) asin(0.3) = 0.582` in each of three years at 10,000 paths.
