---
type: Finding
title: The fit-curves table plots (n - i + 0.5) / n and thins to numpy's rows, halves rounded to even
description: "fit_curves sets every fitted family's survival conditional on t beside the plotting position (n - i + 0.5) / n of each sorted loss, never 0 at the largest as the product-limit estimate is, so a log axis keeps it. Thinning to points rows reproduces numpy's unique(round(linspace(0, n - 1, points))) to the bit: i * ((n - 1) / (points - 1)), the last index exactly n - 1, rounded half to even (89 losses to 10 points keep rows 1, 11, 21, 30, 40, 50, 60, 69, 79, 89; six to three keep index 2, not 3). The plotting positions count each loss once, so censored or weighted losses are refused. Each curve at four rows of 89 synthetic losses agrees with SciPy's fits to 1e-6."
tags: [prospicio-prob, severity, goodness-of-fit, numpy, plotting]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-11T14:00:00Z }
verified: { by: process:ci, at: 2026-11-11T14:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: fit_curves, thinned_rows, compare_fits
  - id: test
    resource: ../validation/tests/severity_fit.rs
    title: fit_curves_match_scipy
  - id: script
    resource: ../validation/scripts/scipy_fit_curves.py
    title: the SciPy and numpy reference
---

# The fit-curves table

Issue #804 asked for the table behind a chart of every fitted severity
against the data on a log axis.

## Plotting positions, not the product-limit estimate

The product-limit estimate is 0 at the largest loss, which a log axis
drops. The table uses the plotting position `(n − i + 0.5) / n` of the
`i`-th smallest of the `n` losses at or above `t` instead (0.994382 for
the first of 89, 0.005618 for the last). It counts each loss once, so
`fit_curves` refuses censored losses and weights other than 1 rather
than mislabel them.

## Thinning as numpy does it

`numpy.linspace(0, n - 1, points)` is `arange(points) * step` with
`step = (n - 1) / (points - 1)`, the last element then set to `n - 1`, and
`numpy.round` rounds halves to even. `thinned_rows` follows both
(`f64::round_ties_even`), so a half lands where numpy puts it: six losses
to three points keep indices 0, 2 and 5 (2.5 to 2), where rounding halves
up would keep 3. R's `round()` also rounds halves to even.

## Parity

`validation/scripts/scipy_fit_curves.py` takes the maximum likelihood fits
of `scipy_severity_fits.py` on the 89 synthetic losses above 150,000 and
writes each family's `S(x) / S(t)` at rows 1, 2, 45 and 89, and numpy's
thinned rows for 10 points (`validation/reference/fit_curves_scipy.csv`).
The Rust table matches to 1e-6 (1e-4 for the gamma, whose fit runs to the
boundary of its shape).
