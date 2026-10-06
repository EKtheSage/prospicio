---
type: Finding
title: chainladder-python's Bondy exponent stops short of the optimum
description: TailBondy fits b with scipy least_squares at default tolerances, which stop on a 1e-8 relative cost change; b is off by up to 4e-6, the tail's results by up to 3e-5.
tags: [reserving, tail, bondy, parity, optimization, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T12:00:00Z }
sources:
  - id: tail
    resource: ../crates/act-reserving/src/tail.rs
    title: act_reserving::tail, bondy_exponent
  - id: py-script
    resource: ../validation/scripts/reserving_tails_python.py
    title: chainladder-python reference generator for tails
---

# Finding

* `TailBondy(earliest_age=...)` minimizes `sum (ln f_j - c b^j)^2` with
  `scipy.optimize.least_squares` from `(0.5, ln f_0)` at the default
  tolerances. It ends with status 2 (`ftol`: the cost changed by less than
  1e-8 of itself), before `b` has converged.
* With `earliest_age=36`, chainladder-python 0.10.1 returns b = 0.6147430802
  (RAA), 0.5651242627 (GenIns), 0.6494360303 (ABC); the optimum, from
  `least_squares` at tolerances of 1e-15 or a bounded scalar minimization
  of the profiled cost, is 0.6147431242, 0.5651263072, 0.6494358690.
* The tail raises the last fitted factor to `b / (1 - b)` and Mack's tail
  sigma is read off at the tail's position, so on GenIns the tail's share
  of each result is off by up to 3e-5 (relative).

# Resolution

* `act_reserving` profiles out `c` (for a given `b` the best `c` is
  `A / B`) and finds `b` exactly by a grid and bisection on the sign of
  the derivative.[^tail]
* The generalized Bondy parity rows are checked to a relative 1e-4; the
  classic Bondy (b stays at 1/2) and every other tail to 1e-9.[^py-script]

[^tail]: act_reserving::tail, bondy_exponent
[^py-script]: chainladder-python reference generator for tails
