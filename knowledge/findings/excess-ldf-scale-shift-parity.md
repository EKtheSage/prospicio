---
type: Finding
title: Bloom and Patrik's excess factors are a pure scale shift of the ground-up factors, and ExcessPattern reproduces them to the digit
description: With the ultimate severity as four point masses and the ground-up factors 3.60 and 1.25, the layer factor at age j is the ultimate layer over the layer of the severity divided by the ground-up factor to ultimate; ExcessPattern gives 17.2, 4112.5/3010 and 23.5 for Year One and the printed factors of Years Two and Three, and Sahasrabuddhe's 14.352 and Pinto and Gogol's layer identity hold. The capped-column solve has no source example and is checked by round trip.
tags: [reserving, excess-development, scale-shift, parity, bloom-patrik]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T18:00:00Z }
verified: { by: process:ci, at: 2026-10-21T18:00:00Z }
sources:
  - id: code
    resource: ../validation/tests/reserving_excess.rs
    title: The parity tests of ExcessPattern against the examples
  - id: module
    resource: ../crates/prospicio-reserving/src/excess.rs
    title: prospicio_reserving::excess and its unit tests
  - id: note
    resource: ../docs/design/reserving-excess.md
    title: Design note, "Validation targets" and decisions 8, 12 to 16
  - id: bloom-patrik
    resource: https://casact.org/abstract/methodology-avoiding-pitfalls-excess-loss-development
    title: Bloom and Patrik, CAS E-Forum Fall 2013, pp. 4-8
---

# What was found

With `X_j = X_inf / c_j` and counts fixed, the layer `l xs a` has factor to
ultimate `CDF_layer(j) = [n_inf / n_j] (LEV(b) - LEV(a)) c_j / (LEV(c_j b) -
LEV(c_j a))`, where `c_j` is the ground-up factor to ultimate over the count
factor. For Bloom and Patrik's four claims (ultimate 450, 1,687.5, 1,125 and
2,250; a `Grid` of step 112.5 whose `lev` is exact at any limit) and the
unlimited layer xs 350, `c = [4.5, 1.25, 1]` gives layer values 175, 3,010 and
4,112.5 and:

| Target (source) | Printed | Computed |
|---|---|---|
| Year One 12:24 (Bloom and Patrik p. 5) | 17.20 | 17.2 (1e-11) |
| Year One 24:36 | 1.37 | 4112.5 / 3010 = 1.36628 (1e-11) |
| Year One 12:ult | 23.50 | 23.5 (1e-11) |
| Year Two 12:24, 24:36, 12:ult | 14.77, 1.36, 20.06 | 14.768, 1.3583, 20.06 |
| Year Three 12:24, 24:36, 12:ult | 13.08, 1.35, 17.67 | 13.0797, 1.3511, 17.6719 |
| Limited to 350, 12:24 and ultimate (p. 7) | 1.33, 1,400 | 1.3333, 1,400 |
| Excess CDF from limited, unlimited and ELF (section 4) | 17.54 | 17.536 (unrounded path; rounded inputs give 17.59) |
| Pinto and Gogol section 5 (p. 248) | 2.891, 2.697, 2.437, 2.144 | within 1e-3 (their inputs have three decimals) |
| Sahasrabuddhe Example 1, E4 (p. 18) | 14.352 | 14.3519 |

The paper prints two decimals, so its values are matched to the rounding and
the exact ones are computed from the claims in the test. Sahasrabuddhe's
number checks the rebasing identity through `layer_at_shift` (an exponential
at scale 280,000 shifted by 280000/28138), not the solve.

# What has no source example

* The numerical solve of `c_j` for a capped ground-up column (`c / LEV(c u) =
  r / LEV(u)`, monotone in `c`): no source prints one. It is checked by round
  trip, a triangle built from the scales 3, 1.5, 1.1 and 1 on a lognormal
  capped at 200,000, which gives the scales back to 1e-9; an uncapped reading
  of the same data understates the development.
* Sahasrabuddhe's Example 3 (column 9, 0.856) is not pinned: it needs the
  example's trend indices and selected ratios.

# A correction to the note's illustration

The lognormal illustration (sigma 1.5; layer factors 1.275, 1.545, 1.761,
1.982 for 1m xs 0, 750k xs 250k, 2m xs 1m and 7m xs 3m) holds when the mean
is 50,000 at the earlier age and 65,000 at ultimate (a shift of 1.3); a mean
of 50,000 at ultimate gives 1.2815, 1.5853, 1.8101 and 2.0379. Both are
tests.
