---
type: Finding
title: The Pricing Insurance Risk allocation guides in R, what the R package covers
description: The calls of pir-price-allocation.md and pir-price-allocation-practice.md mapped onto the R package; every printed number equal to Python's, and one gap, VaR() on a grid_distribution (quantile() works; #373).
tags: [pricing, capital-allocation, r, python, api, bindings, guides, gaps]
status: stable
generated: { by: claude-code/guides-session, at: 2026-10-17T23:59:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:59:00-07:00 }
sources:
  - id: guide
    resource: ../docs/guides/pir-price-allocation.qmd
    title: The R twin of the classical, natural and Bodoff allocation guide
  - id: practice
    resource: ../docs/guides/pir-price-allocation-practice.qmd
    title: The R twin of the case-study and reinsurance guide
  - id: test
    resource: ../R/prospicio/tests/test-guides.R
    title: The harness that runs the R chunks and checks every printed number
---

# Facts (2026-10-17)

| Python | R |
|---|---|
| `Portfolio(units, rows, probs)` | `capital_portfolio(matrix, probs =)` |
| `Portfolio.from_independent` | `capital_portfolio(list(name = grid, ...))` |
| `Portfolio.price(d, assets, p, allocation)` | `natural_price(port, d, assets, p, allocation)`, a data frame with a `total` row |
| `Portfolio.calibrate` | `calibrate_portfolio()` |
| `Portfolio.kappa`, `.assets`, `.totals`, `.probs` | `portfolio_kappa()`, `portfolio_assets()`, `port@totals`, `port@probs` |
| `Portfolio.bodoff`, `.epd`, `.priority` | `bodoff_allocation()`, `epd_ratio()`, `priority_ladder()` |
| `Portfolio.premium_bounds`, `.price_stand_alone` | `premium_bounds()`, `price_stand_alone()` |
| `prospicio.risk.capital(pd, d, method)` | `capital_allocation(pd, d, method)$by_component` |
| `Sampled.tvar`, `Grid.tvar`, `Grid.stop_loss`, `Grid.variance` | `TVaR()`, `stop_loss()`, `variance()` |
| `Grid.quantile` | `quantile(grid, p)`; `VaR()` has no `grid_distribution` method (#373) |

Every number the two R guides print equals the Python guides', including the
linear allocation at `a - 1e-9` and the scaled-down cover of Example 302.
