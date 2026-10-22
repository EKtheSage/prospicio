---
type: Finding
title: The excess-layer bindings take the pattern through ldf, and exposure curves need no route of their own
description: In Python and R an ExcessPattern gives a method's ldf, ldf_kind, ldf_weight and tail, so the chain ladder, Mack and the expected-loss methods take it without a new parameter; the exposure-curve route of decision 2 is the curve's own layer_share passed as a share; the Python layer class is ExcessLayer because the native module is flat; both languages reproduce the Rust parity targets.
tags: [reserving, excess-development, bindings, python, r, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T23:00:00Z }
verified: { by: process:ci, at: 2026-10-21T23:00:00Z }
sources:
  - id: python-tests
    resource: ../python/tests/test_reserving_excess.py
    title: The Python tests of the excess-layer bindings
  - id: r-tests
    resource: ../R/prospicio/tests/test-reserving-excess.R
    title: The R tests of the excess-layer bindings, the same numbers
  - id: code
    resource: ../validation/tests/reserving_excess.rs
    title: The Rust parity tests the bindings reproduce
  - id: design
    resource: ../docs/design/reserving-excess.md
    title: The design note, API section and decisions 20 and 21 (decided by the user on 2026-10-22 as built)
---

# What was found

* The derived pattern reaches a method as arguments, not as an object:
  `ExcessPattern.selected(weight)` is `{"ldf", "ldf_kind": "ldf",
  "ldf_weight"}` and `tail` is the layer's tail, so `ChainLadder(tail=p.tail,
  **p.selected(w))` (R: `do.call(chain_ladder, c(list(tri, "layer", tail =
  p@tail), excess_selected(p, w)))`) fits a layer triangle with an empty first
  age, which its own factors cannot. Mack, `ExpectedLoss`,
  `BornhuetterFerguson` and `CapeCod` take the same arguments.
* Exposure curves need no new route (decision 2 (a)): `Mbbefd.layer_share`
  and `TabulatedCurve.layer_share` in Python and `exposure_layer_share()` in R
  give the share that `layer_exposure` takes.
* The Python native module is flat (`#[pymodule]` with exported classes), so
  a class named `Layer` cannot be added next to `prospicio.reinsurance.Layer`:
  the reserving layer is `ExcessLayer` (R `excess_layer()`).
* Both languages give the Rust values: Bloom and Patrik Year One 17.2,
  4112.5/3010 and 23.5 and Year Three 13.0797; Pinto and Gogol 5.0; Bear
  3.8835; and the credibility weights 0.5 and 0.6 without counts and 3/13 and
  3/8 with them on a four-origin layer triangle whose ratios equal the pattern.
