---
type: Finding
title: The dyadic lattice rule against aggregate 2.1.0
description: aggregate 2.1.0 finds a discrete severity's lattice by searching powers of two from the largest atom down 16 levels at a tolerance of 1e-12 of it, takes the gcd of the multiples, and prefers the largest nice bucket dividing the step (0.25 for 0.75, 1 for 3, 50 for 150); a lattice finer than 2^-16 of the largest atom does not count (123456 and 7 have none), decimal atoms never have one; prospicio's port matches on 16 atom sets.
tags: [grid, lattice, discrete, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T23:30:00Z }
verified: { by: process:ci, at: 2026-11-09T23:30:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_lattice.py
    title: validation/scripts/aggregate_lattice.py (aggregate 2.1.0's _dyadic_lattice_step and _lattice_candidates)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, utilities.py and _bucket_window.py (warn_lattice_grid)
---

# Finding

* Every float is a dyadic rational, so "on a lattice" needs a depth and a
  tolerance: `aggregate` searches `2^-r` from the coarsest at or below the
  largest atom `M` to `2^-16 M`, and accepts the first where every atom is
  within `1e-12 M` of a multiple. The step is the gcd of the multiples:
  `[12, 15]` gives 3, `[0.375, 1.125]` 0.375, `[-6, 9, 15]` 3.[^script]
* The depth makes a hit mean something: `[123456, 7]` has no lattice,
  since 1 is finer than `2^-16` of 123456; `[7, 1e-7]` and decimal atoms
  (`[0.1, 0.2]`, thirds) have none either.
* The preferred bucket is the largest nice one dividing the step (a power
  of two for a fractional step, `{1, 2, 4, 5, 8} × 10^k` for a whole one):
  0.75 to 0.25, 3 to 1, 150 to 50, 1000 stays. The exact step is used only
  when the nice one needs a larger grid; when neither fits, a coarser bucket
  is used and `aggregate` warns with the `log2` needed (book `dsev [1000
  3000 7000]` with 50 claims needs `log2` 9 at bucket 1000 and gets 2000 at
  `log2` 8).

[^script]: validation/scripts/aggregate_lattice.py, run 2026-11-09.
