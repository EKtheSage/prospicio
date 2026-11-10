---
type: Finding
title: Discrete inputs aggregate 2.1.0 refuses
description: aggregate 2.1.0 refuses dsev probabilities that fail numpy's isclose(sum, 1) (|sum - 1| above 1e-8 + 1e-5, so 0.99999 builds and 0.99998 does not) and a fractional fixed count, but not dfreq probabilities, whose shortfall becomes a defective count; prospicio's Discrete now refuses as dsev does, with Discrete::normalized for weights, and its EmpiricalCount refuses a sum more than 1e-9 from 1.
tags: [discrete, counts, validation, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T20:00:00Z }
verified: { by: process:ci, at: 2026-11-09T20:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_discrete_inputs.py
    title: validation/scripts/aggregate_discrete_inputs.py (which dsev sums aggregate 2.1.0 builds)
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, parser.py (the probability check of a dsev clause, np.isclose(total, 1.0))
---

# Finding

* `aggregate` 2.1.0 checks a `dsev` clause's probabilities with
  `np.isclose(total, 1.0)` at numpy's defaults, `|total - 1| <= 1e-8 +
  1e-5`, and raises "dsev probabilities sum to 0.98; they must sum to 1, to
  within a relative 1e-5". Of `dsev [1 2] [0.5 q]`, `q` = 0.49999,
  0.499991 and 0.5000099 build and 0.500011, 0.49998, 0.48, 0.52 and 1.5
  are refused. A sum within the bound is rescaled to 1.[^script]
* `2.5 claims ... fixed` is refused: "a fixed frequency needs a whole
  number of claims".
* `dfreq [1 2] [.5 .49]` builds, and its density sums to 0.99: the
  shortfall is a defective count, not an error. prospicio's
  `EmpiricalCount` refuses a sum more than 1e-9 from 1, stricter than
  `aggregate`, and is unchanged.
* prospicio's `Discrete::new` applies the `dsev` bound exactly, so the nine
  sums above build or not as in `aggregate`; `Discrete::normalized` (Python
  `normalize=True`, R `normalize = TRUE`) divides weights or counts by
  their total on purpose.

[^script]: validation/scripts/aggregate_discrete_inputs.py, run 2026-11-09.
