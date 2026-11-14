---
type: Finding
title: An event loss table on aggregate 2.1.0's FFT grid
description: aggregate 2.1.0 lowers an ELT to a compound Poisson whose severity is the rows' losses by rate, bucketed by eltq's nearest-node survival method when the table has secondary uncertainty and by the mean-preserving linear scatter of dhistogram atoms when it has means alone; prospicio's EventLossTable::severity_grid reproduces both, to 1e-13 for means and 1e-7 through eltq, whose interpolation it computes exactly (1e-12 against SciPy's beta).
tags: [catastrophe, elt, fft, aggregate, eltq, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-14T00:00:00Z }
verified: { by: process:ci, at: 2026-11-14T00:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_elt_grid.py
    title: validation/scripts/aggregate_elt_grid.py (aggregate 2.1.0 with eltq 2.15.0 and SciPy 1.18.1)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, _loss_table.py (_lower_elt), _severity.py (SeverityElt.quantize), _aggregate_compute.py (discretize)
---

# Finding

* `agg NAME elt [...]` lowers to `poisson` with mean `Λ = Σ rate` and a
  severity: an ELT3, an ELT4 or an ELT6 under `no2u` becomes
  `dhistogram` atoms at the event means (duplicate means merged), placed
  by `dsev_bucket='linear'` (the default): each atom split between its two
  neighbouring nodes so the mean is exact, an atom within `1e-12` of the
  largest atom of a node snapped onto it, an atom at or above the top node
  on it.[^script]
* An ELT6 with secondary uncertainty, even when every row's sds are zero,
  becomes an `EltModel` that `eltq.quantize_arrays(method='survival',
  overflow='top_bucket')` buckets: node `k` takes `((k - 1/2) h, (k + 1/2)
  h]`, the mass above `(n - 1/2) h` goes onto the top node, and an atom
  exactly on a bucket edge is split evenly between the two nodes. eltq
  interpolates each row's survival between log-spaced nodes (100 a decade,
  12 a standard deviation near a bell-shaped beta's mean), so its buckets
  differ from the exact beta's by up to about 1.4e-5 relative, 1e-8 in a
  normalized bucket; the exact rule matches SciPy's beta to 1e-12.
* With `padding=2` aggregate transforms on four times the grid, the buffer
  prospicio's `fft` returns, so the annual totals agree to 1e-13 once
  prospicio's lumped tail is taken off the last node; with the default
  `padding=1` they differ by the wrap-around of mass above twice the grid
  (about 1e-9 on the test table).
* `occurrence ceded to 1000 xs 500` on the table of means, on a grid whose
  nodes include the layer's attachment and limit, equals
  `Tower::on_grid`'s ceded annual loss to 1e-13.

[^script]: validation/scripts/aggregate_elt_grid.py, run 2026-11-14.
