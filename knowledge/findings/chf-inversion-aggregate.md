---
type: Finding
title: aggregate 2.1.0's characteristic-function engine, reproduced with and without the tilt
description: prospicio_prob::chf reproduces AggregateCHF's untilted grids on eleven laws and its tilted stable grids on eleven more to round-off (every step and left edge exactly, the masses to 1e-13); its masses are the rounding discretization's bucket probabilities, rule A's bucket walks down the ladder when the dropped frequencies exceed 1e-6 (a Tweedie 1.5 100 2 at log2 12 lands on 0.125), it tilts only laws with a power tail, and on a light law a tilt amplifies the truncation ringing by e^(theta x) instead of removing anything; under the automatic tilt on a stable law (tau* of 9 to 15 e-folds) the round-off at the heavy end is amplified by up to 2.3e6, so sums of the masses agree to 1e-10, not 1e-13.
tags: [aggregate, characteristic-function, fft, tilting, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-13T12:00:00Z }
verified: { by: process:ci, at: 2026-11-13T12:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_chf.py
    title: validation/scripts/aggregate_chf.py, the reference generator pinned to aggregate==2.1.0
  - id: test
    resource: ../validation/tests/probability_chf.rs
    title: validation/tests/probability_chf.rs, chf_inversion_matches_aggregate
  - id: tilt-test
    resource: ../validation/tests/probability_chf_tilt.rs
    title: validation/tests/probability_chf_tilt.rs, chf_tilted_stable_matches_aggregate, against validation/scripts/aggregate_chf_tilt.py
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0, characteristic_function.py (AggregateCHF, invert_characteristic_function, window_edges) and utilities.py (round_bucket)
---

# Finding

* `AggregateCHF` inverts `phi(s) sinc(s h / 2)` with one `irfft` on the
  frequencies `t_j = 2 pi j / (n h)`, evaluated at `s = -t_j + i theta`; the
  result is the bucket probability `P(x_k - h/2 < X <= x_k + h/2)`, not a
  density sample, so it equals the rounding discretization up to the
  dropped frequencies and the wrapped mass. On a gamma(4, 25) at log2 12
  each mass equals the gamma's own bucket probability to 1e-12.[^test]
* Rule A takes the Chernoff edges `inf_s (K(s) - ln eps) / s` at
  `eps = (1 - 0.99999) / 2` a side (a log ladder of 217 values of `s`, then
  a bounded Brent step), rounds `(x_R - x_L) / n` up the ladder
  `{1, 2, 4, 5, 8} * 10^k` (a power of two below 1), splits the slack
  between the sides, and snaps the left edge to a multiple of the bucket
  from 0 or the atom. A support bound within 5% of the window replaces the
  Chernoff edge.[^aggregate]
* When `|phi(pi / h)|` (the atom taken out) is above 1e-6, the bucket walks
  down the ladder, at most twenty rungs, while that lowers the larger of
  the truncation and the Chernoff mass outside the shortened window. A
  Tweedie with p = 1.5, mean 100 and dispersion 2 at log2 12 ends on 0.125
  with a truncation of 1.8e-6, still above the flag; a gamma of shape 0.5
  ends on 1/64 with a truncation of 0.022 (its density has a pole at 0).
* `aggregate` tilts only a law with a power tail (the totally skewed
  stable laws, and the two halves of a skewed one); `tilt=` a number on a
  light law is a `ValueError`. A tilted light law gives the same masses in
  the body, but the ringing of the dropped frequencies is multiplied by
  `e^(theta x_k)`: on the gamma(4, 25) with `tau = 2` the far-tail masses
  are 7.4 times the untilted ones' 1e-12, so a tilt there buys nothing.
* Every quantity matched with the same arithmetic: the steps and left
  edges exactly, the masses to 1e-13 (numpy's pocketfft against rustfft),
  the truncation to 1e-9 relative and the Chernoff outside masses to 1e-6
  relative (scipy's bounded Brent stops at an absolute 1e-5 in `s`, on a
  flat minimum).[^script]
* Under the automatic tilt `AggregateCHF` picks, for a stable law with
  `beta = 1` (or `-1` by reflection), `tau* = 1/2 ln(rho / (eps kappa))`
  with `rho = q^(alpha+1) zeta(alpha+1, 1+q)` and `eps = 2^-52 +
  |phi(pi/h)|`: 10.9 at `alpha = 1.5`, 14.6 at 0.7 and 12.6 at 1 on log2
  12. The bucket is capped at `min(gamma/10, pi gamma/(52 ln 2)^(1/alpha))`
  rounded down the ladder (1/16 at unit scale), the window runs from the
  support bound or the 1e-40 Chernoff edge, and the mass beyond it is left
  out: 1.0e-4 at `alpha = 1.5`, 0.042 at 0.7.[^tilt-test]
* The tilt multiplies the inverse transform's round-off by `e^(theta x)`,
  up to `e^tau` at the heavy end, so prospicio's and aggregate's masses
  (rustfft against pocketfft) agree to 1e-11 there and their sums over
  4,096 buckets to 8e-11 at `alpha = 0.7`; the tilts and edge errors
  agree to 1e-9 relative. Untilted, the same law's modelled edge error is
  11.9 (a relative error of order one near the right edge), tilted
  1.1e-5.
