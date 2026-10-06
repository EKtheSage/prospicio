---
type: Reference Implementation
title: Tail factors in R ChainLadder and chainladder-python
description: R ChainLadder 0.2.21's tail = TRUE rule and tail_SE, and chainladder-python 0.10.1's TailConstant, TailCurve and TailBondy; the quirks act_reserving::Tail reproduces or avoids.
resource: https://cran.r-project.org/package=ChainLadder
tags: [reserving, tail, mack, parity, r, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T12:00:00Z }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 3
  - id: tail
    resource: ../crates/act-reserving/src/tail.rs
    title: act_reserving::tail
  - id: r-script
    resource: ../validation/scripts/reserving_tails_r.R
    title: R reference generator for tails
  - id: py-script
    resource: ../validation/scripts/reserving_tails_python.py
    title: chainladder-python reference generator for tails
---

# Use

Reference values for `act_reserving::Tail` and Mack with a tail, on RAA,
GenIns and ABC.[^r-script][^py-script] Read from the sources:
`ChainLadder:::tailfactor`, `tail_SE`, `MackChainLadder`; Python
`inspect.getsource` of `chainladder.tails`.

# R ChainLadder quirks

* `tailfactor` (`tail = TRUE`) tests `f[n-2] * f[n-1] > 1.0001` with
  `n = length(f)`: the third- and second-last factors, not the last two.
  It then regresses `ln(f - 1)` on the 1-based index over the factors
  above 1 and multiplies the 100 extrapolated factors after the last one
  above 1. A product above 2 is printed and reset to 1.[^tail]
* `MackChainLadder` adds the tail step, and `tail_SE`, only when the tail
  factor is above 1. A tail below 1 is stored in `$f` but neither scales
  the ultimates nor adds risk; given `tail.se` and `tail.sigma` are then
  ignored. On RAA, `tail = 0.98` gives a total ultimate of 213122.2 and
  `Total.Mack.S.E` 26880.74, the same as no tail. `act_reserving` follows
  chainladder-python here instead (below), so that Mack's ultimates are
  the chain ladder's.
* `tail_SE` finds the tail's position where the line through `ln(f - 1)`
  reaches `ln(tail - 1)` and reads lines through `ln(f.se)` and
  `ln(sigma)` there. The tail's process term is
  `tail.sigma^2 C^(2 - alpha)` at the oldest age, its parameter term
  `C^2 tail.se^2`; every origin, the oldest too, gets them.

# chainladder-python quirks

* `TailBase._get_tail_stats` matches R's `tail_SE` when every factor is
  above 1. It drops a factor at or below 1 by setting its `ln(f - 1)` to
  NaN while keeping its `x` in the regression's mean, so with such a factor
  it departs from R; `act_reserving` drops the point cleanly, as R's `lm`
  does. A tail below 1 scales the ultimates and is moved to 1.001 for the
  position (`_get_tail_weighted_time_period`), so it carries a non-zero
  sigma and standard error; a tail of exactly 1 carries none.
  `act_reserving` does the same: on RAA, `TailConstant(0.98)` with
  `MackChainladder` gives a total ultimate of 208859.78, a total standard
  error of 26343.49 and a tail sigma of 0.11275.[^tail]
* `TailConstant` spreads the factor as `1 + x decay^k` with `x` the root
  of `a x^2 + b x - ln(tail)` (`a`, `b` sums over 1000 periods); the last
  factor makes up the difference. `_apply_decay` tests `if attach_idx:`,
  so an `attachment_age` at or before the youngest age (index 0) is
  ignored and the tail attaches at the oldest age; `TailCurve` has no such
  test. `act_reserving` attaches at the youngest age for both, replacing
  every estimated factor (on RAA with 1.05 at age 12 the ultimate of 1990
  is 2166 against about 19322 in Python); a unit test records it.
* Ages are read by position, `int(age / grain - 1)`: `TailCurve`'s
  `fit_period=(start, end)` is the slice `[int(start/grain - 1),
  int(end/grain - 1))` and `TailBondy`'s `earliest_age` is
  `ddims[int(age/grain) - 1]`. An age off the grid therefore means the
  age at or before it (`fit_period=(30, 102)` is `(24, 96)` on an annual
  grain). `act_reserving` maps both to the last age at or before the given
  one, which is the same on ages that are multiples of the grain from one
  grain on; on a triangle whose ages start elsewhere (3, 15, 27 months)
  Python's positions point at other ages, and `act_reserving` uses the
  ages themselves. `attachment_age` is read by value in both (first age
  at or after for `TailConstant` and `TailCurve`, last at or before for
  `TailBondy`). `TailCurve` leaves out factors at or below 1.00001.
* `TailBondy` keeps the factor from its attachment age to the next
  (`TailConstant` and `TailCurve` replace it), builds the fitted factors
  from the observed factor at `earliest_age` rather than the fitted one,
  and computes an unused `tail` (`exp(exp(c) b^(n-1))`). Comparing
  `attachment_age` with a default `earliest_age` of `None` raises a
  `TypeError`. Its exponent stops short of the optimum: see
  [Bondy least squares](/findings/bondy-least-squares-stop.md).
* `MackChainladder` on a tailed pattern uses the tail-replaced factors in
  its projection and recursions but the estimated sigmas and standard
  errors, and the cdf at the oldest age as the tail step.

On RAA, GenIns and ABC every factor is above 1, so R's `tail = TRUE` and
`TailCurve()` give the same tail, sigma and standard error to 1e-12.[^design]

[^design]: Design note, reserving v0.2, decision 3
[^tail]: act_reserving::tail
[^r-script]: R reference generator for tails
[^py-script]: chainladder-python reference generator for tails
