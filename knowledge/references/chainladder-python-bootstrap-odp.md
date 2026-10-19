---
type: Reference Implementation
title: chainladder-python's BootstrapODPSample under n_periods and the drops
description: chainladder-python 0.10.1's BootstrapODPSample(n_periods, drop_high, drop_low) is the parity reference for the ODP bootstrap's deterministic steps under link ratio exclusions in prospicio_reserving; its first age takes the weight of link 0, its scale divides by the full degrees of freedom while summing only the kept residuals, its zero residuals are dropped and its pool centred, hat_adj defaults to True against a docstring that says False, and its resampled triangles are not refitted with the exclusions downstream.
resource: https://github.com/casact/chainladder-python
tags: [reserving, bootstrap, odp, parity, python, exclusions, scale]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-19T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 14, the bootstraps and the one-year views under the exclusions
  - id: script
    resource: ../validation/scripts/reserving_bootstrap_exclusions_python.py
    title: chainladder-python reference generator for the ODP bootstrap under exclusions
  - id: test
    resource: ../validation/tests/reserving_bootstrap_exclusions_odp.rs
    title: Validation test of the ODP bootstrap under exclusions against chainladder-python
---

# Use

`BootstrapODPSample(n_periods, drop_high, drop_low)` is the reference for
the fitted incremental values, the cells that have a residual, the unscaled
residuals and the scale `phi` of `prospicio_reserving::OdpBootstrap` with
`n_periods`, `drop_high` and `drop_low`, on RAA and GenIns under
`n_periods = 3`, under one high and one low dropped, and under both. Those
are compared exactly (relative 1e-9); the simulated mean reserve is compared
within Monte Carlo error.[^test] It has no Mack bootstrap and no one-year
view, and R ChainLadder's `BootChainLadder` takes no weights. Read from the
0.10.1 wheel (`adjustments/bootstrap.py`) and checked on
2026-10-19.[^design]

# Behaviour

* **The fit.** `Development(n_periods, drop_high, drop_low).fit_transform`,
  then `Chainladder().fit`: the fitted incremental values are the back-cast
  `full_expectation_` from each origin's latest value with the factors
  under the exclusions. prospicio does the same.[^script]
* **The first age takes the weight of link 0.** `w_` has one column fewer
  than the triangle, and the code prepends its first column, so the first
  age's cell of origin `o` has a residual only if link `(o, 0)` is kept.
  Dropping that link drops two residuals. prospicio always keeps the first
  age's cells and gives a later cell a residual when the link it closes is
  kept. The parity check compares every other cell and lists this one.
* **The scale's denominator is not reduced.** `phi` is the sum of the
  squared kept residuals over `N - p`, with `N` all observed cells and `p`
  the design matrix's columns (`origins + ages - 1`). With no exclusions
  this is the usual `phi`. Under exclusions it is biased low: on RAA with
  `n_periods = 3`, chainladder-python's `scale_` is 322.4 against 983.6
  plain, because it also loses the first-age residuals of the origins whose
  link 0 is dropped. prospicio's `phi` keeps those residuals and is 694.0
  there, the reference's own sum over the same cells.[^script]
* **Zero residuals are dropped and the pool is centred.** The resampling
  pool is the finite non-zero standardized residuals less their mean.
  prospicio keeps zeros and corner cells in the pool and does not centre
  it, as R's `BootChainLadder` does. With the native pool, the simulated
  means on RAA differ from prospicio's by up to 16%, so the reference
  script replaces the pool with prospicio's to compare means.
* **`hat_adj` defaults to `True`, the docstring says `False`.** The
  signature has `hat_adj=True`, the documented default is `False`, and the
  docstring's `scale_` examples do not tell them apart (983.64 either way).
  With `hat_adj=True` the hat matrix is built from the full triangle's
  design matrix, whatever the exclusions. The reference uses
  `hat_adj=False`, which applies no factor (prospicio's residuals carry
  `sqrt(N / (N - p))`).
* **The resampled triangles are not refitted with the exclusions.**
  `resampled_triangles_` is a plain triangle, so a downstream
  `Chainladder().fit(sims)` averages every link ratio of each sample unless
  the user adds a `Development` with the same settings, and then each
  sample ranks its own drops again. prospicio keeps the observed
  triangle's kept links in every simulation (decision 14, item 2). The
  reference script refits with the observed exclusions as fixed positions
  (its `drop` argument) and records the re-ranked mean for the
  record.[^script]
* **`random_state`.** `None` is `np.random.RandomState(None)`, so
  unseeded. An integer, 0 included, seeds it in 0.10.1: three runs with the
  same integer gave the same triangles. Sums of samples need `nansum`,
  since the unobserved cells are NaN.

# Heavy tails

RAA under `n_periods = 3` with both drops leaves ten link ratios and a pool
of twenty residuals, a single link at almost every age and a newest origin
of one cell. A pseudo factor can divide by a near-zero value, and the
reserve has outliers of 1e8 on a median of 4e4. Its mean has no usable
standard error, so the test compares medians.[^test]

[^script]: validation/scripts/reserving_bootstrap_exclusions_python.py
[^test]: validation/tests/reserving_bootstrap_exclusions_odp.rs
[^design]: Design note, reserving v0.2, decision 14
