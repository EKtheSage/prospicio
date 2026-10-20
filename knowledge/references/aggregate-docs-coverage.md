---
type: Reference
title: aggregate 1.2.0 documentation against prospicio, a coverage map
description: Feature by feature, what the aggregate 1.2.0 documentation (Getting Started, Overview, DecL, pipelines) offers, where prospicio matches it, and the gaps found by a name search of the Python stub; the starting point for the aggregate guides.
resource: https://aggregate.readthedocs.io/en/latest/
tags: [aggregate, parity, coverage, guides]
status: stable
generated: { by: claude-code/guides-session, at: 2026-10-17T12:00:00Z }
sources:
  - id: docs
    resource: https://aggregate.readthedocs.io/en/latest/
    title: aggregate 1.2.0 documentation (Sphinx, BSD-3 package by Stephen Mildenhall)
  - id: tour
    resource: https://github.com/EKtheSage/prospicio/blob/main/docs/guides/prospicio-for-aggregate-users.md
    title: prospicio for aggregate users, the guide that runs the Twenty Minutes session
---

# What was compared

The four parts of the 1.2.0 documentation: Getting Started, the Aggregate
Overview (twenty-minute session, underwriter and recipes, the aggregate,
reinsurance, portfolio and P&L pipelines, exhibits and charts, grid selection,
numerical issues, the 1.0 feature ledger), the DecL language reference and the
API reference. Against prospicio 0.0.3 as seen in
`python/prospicio/prospicio_native.pyi`.

A name search of the stub finds what exists; absence in the stub is the
evidence for a gap, so each gap below is a claim for the owning lane to confirm.

# Matched (checked by running, in the guide)

The Twenty Minutes session reproduces to the digit: the casualty book, a
mixture severity, a mixed gamma count, the increased limits table, an
occurrence tower net, the 99.6% VaR of a two-unit portfolio and its pentagon.
The 99.9% quantile differs by one unit (33,170 against 33,169): a quantile
convention at a bucket boundary.

# Matched by name (not run against aggregate)

| aggregate | prospicio |
|---|---|
| zero-truncated and zero-modified counts, logarithmic, Neyman A, mixed Poisson | `Count` constructors |
| occurrence and aggregate reinsurance, reinstatements, swing, slide, profit commission, corridor | `Layer`, `Tower`, `Layer.with_*` |
| `bivariate`, `clash` | `fft_bivariate`, `JointGrid`, `Clash` |
| `pnl`, `xpnl`, peel, expenses | `Ledger`, `Book`, `Expense`, `pnl_program` |
| distortions, calibration, `Bounds`, natural allocation, priority ladder | `Distortion`, `calibrate`, `Portfolio.premium_bounds`, `kappa`, `priority_kappa` |
| `wait` and `wiener_hopf` | renewal counts, `wiener_hopf` |
| `approximate` | `CompoundMoments.approximate` |
| `tweedie` | `Tweedie` |

# Gaps

1. **Mixing shared across the units of a portfolio.** In aggregate, units that
   share a `mixed` frequency share its mixing variable and are correlated.
   prospicio's `Portfolio.from_independent` is independent; `Clash` covers two
   lines with one event count, nothing covers more.
2. **A grid audit.** aggregate's `sharpen` re-runs a build over probe bucket
   sizes and scores them, and the tail report grades the tail. prospicio reports
   `aliasing_error`, `tail_mass` and `mean_error` and sizes by `recommend_grid`
   for an unlimited severity; no probe of a chosen step. Closed by #317:
   `audit_grid` and `audit_portfolio_grid` probe and score as `sharpen` does,
   to its digits on its books (the tail report is not ported).
3. **Automatic grids for a limited severity or a portfolio.** `fft_auto` sizes
   for the severity it is given; with a limit the user picks the step.
   Closed by #317: `recommend_grid` sizes a limited severity by its own
   quantiles, `recommend_portfolio_grid` sizes several units, and the audit
   moves a step that is too fine or too coarse.
4. **Shifted and scaled severities.** `10 * lognorm 1 + 20` has no wrapper;
   `Grid.map` does it on a grid, losing the continuous form.
5. **A DecL front end, the recipe base, `%%agg`.** A design decision, not a
   defect: prospicio is a Python and R API. If wanted, a parser for a subset of
   DecL is a large item.
6. **Exact rational compounds (`nttagg`), the `cantor` severity, the exhibit and
   chart documents, `decl_help`.** Small or presentational.

# Not compared

The API reference page by page, `Bounds` and `PricingBounds` numerics, the P&L
routes' accuracy and the bivariate `massive` mode; the guide notes where it
stops.
