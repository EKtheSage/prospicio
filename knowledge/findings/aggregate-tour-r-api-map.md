---
type: Finding
title: The aggregate twenty-minute tour in R, what the R package covers
description: Each call of the Python guide prospicio-for-aggregate-users.md mapped onto the R package's functions, with the R-only quirks met (Panjer as compound_distribution's default, map_grid trimming the grid, 1-based probabilities) and the one gap, a portfolio's assets at a probability; every printed number equal to the Python guide's except the aliasing error's rounding noise.
tags: [aggregate, r, python, api, bindings, guides, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/prospicio-for-aggregate-users.qmd
    title: The R guide, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: namespace
    resource: ../R/prospicio/NAMESPACE
    title: The R package's exports on main, 2026-10-17
---

# Facts (2026-10-17)

* The Python calls of the aggregate tour and their R names:[^guide][^namespace]
  `fft(count, sev, n)` is `compound_distribution(count, sev, n, method = "fft")`
  (the default method is `"panjer"`), its report is the grid's `@report`;
  `Grid.local_moment(d, step, n)` is `discretize(d, step, n, "local_moment")`;
  `Grid.map(f)` is `map_grid(grid, f)`, a list with `grid` and `on_points`;
  `Count.empirical` is `empirical_count()`; `Count.mixed_poisson` is
  `mixed_poisson_count()`; `Lognormal.from_mean_cv` is
  `lognormal_from_mean_cv()`; `Tower([...]).on_grid` is
  `tower_on_grid(reinsurance_tower(list(xol_layer(...))), ...)`, with
  `Layer.stop_loss` as `aggregate_stop_loss()`;
  `Portfolio.from_independent(names, grids)` is
  `capital_portfolio(list(name = grid, ...))`; `Portfolio.price(d, p =)` is
  `natural_price(port, d, p =)`, a data frame with a `"total"` row;
  `Distortion.ccoc(r)` is `distortion("ccoc", r)`; `Pentagon.solve` is
  `pentagon()`, a named vector; `simulate_events(...).totals()` is
  `total(simulate_events(...))`.
* Gap: the R package has no exported equivalent of Python's
  `Portfolio.assets(p)`. `natural_price(port, d, p = p)` reports the same
  assets in its `"total"` row.[^guide]
* `map_grid()` returns a grid that ends at the largest mapped value, so a
  limited severity's `@probs` is shorter than the points it was discretized
  on; adding grids of different limits needs padding (Python's guide loops
  over the shorter list and is unaffected).[^guide]
* Every number the Python guide prints comes out the same in R (casualty
  mean 24,534.1, sd 2,561.7, 99.9% 33,170; the simulated quantiles by seed;
  ILFs; the net of the Book; the pentagon), except the FFT's aliasing error,
  rounding noise of 6.1e-19 in R on Windows against 8.5e-19 in the Python
  guide, which the R guide checks only as below 1e-15.[^guide]

[^guide]: The R guide, every chunk's printed text checked by R/prospicio/tests/test-guides.R
[^namespace]: The R package's exports on main, 2026-10-17
