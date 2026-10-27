---
type: Finding
title: An annual accident-year pattern read at monthly ages as a Weibull corrected to its own yearly averages converts to an underwriting year within 0.2% where decision 18's reading is 31% off
description: An accident year's share at age a is the average of its months, P(a) = (1/12) sum p(a - q), so a reading between annual ages is a choice of p; the Weibull fitted through that average and moved to the nearest monotone monthly pattern whose averages are the given values reproduces them exactly and converts a simulated book's underwriting-year shares within 0.2% at 12 to 60 months (decision 18's ln-linear reading 1.3116 and 0.9155 at 12 and 24), at worst 1.5% off on a slow loglogistic truth; on CLRD and Friedland patterns the exact monotone readings agree within 2% to 12% at 12 months and the current reading sits 19% to 86% above them.
tags: [reserving, development, pattern, underwriting-year, growth-curve]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-27T18:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 25, reading an annual accident-year pattern at monthly ages before the underwriting-year conversion
  - id: script
    resource: ../validation/scripts/ay_monthly_reading.py
    title: The study's script, every candidate against the simulated book, four synthetic truths, the CLRD lines and Friedland's selections, run by hand with the Python venv
---

# What was found

Decision 21 converts a pattern of `s`-month accident periods to an
underwriting basis as `P_UY(a) = sum_j v(j) P_s(a - j)` and reads `P_s`
between its given ages by decision 18 item 2: linear from 0 below the
first age, `ln`-linear between ages. On an annual accident-year pattern
that reading is wrong in the first two years, because an accident year's
share at age `a` is the even-exposure average of its months, `P_12(a) =
(1/12) sum_{q<12} p(a - q)` with `p(t) = 0` at or below 0: at six months
only six of the year's months have begun and they are younger, so `P_12(6)
<= (6/12) p(6)`, well below half of `P_12(12)`.[^note]

The study (`validation/scripts/ay_monthly_reading.py`) expands the given
annual values to `P_12` at every month by each candidate reading and feeds
the result to decision 21's exact route, then compares the converted
underwriting-year share with the book's own.[^script]

* **The relationship pins nothing between annual ages.** `p(a) - p(a - 12)
  = 12 (P_12(a) - P_12(a - 1))` recovers `p` only when `P_12` is given at
  every month; at annual ages each year is one constraint on twelve
  unknowns. The least a reading must satisfy: `p` not decreasing where the
  given pattern is not, `0 <= p <= level`, and the given values
  reproduced exactly as averages of `p`. Interpolating `P_12` alone (PCHIP
  through `(0, 0)`) is worse than the current reading at 12 months
  (1.35), and knot families solved to reproduce the averages (`p` linear
  or `ln`-linear between the given ages) oscillate: exact at the given
  ages and not monotone on every truth.[^script]
* **The Weibull corrected to exactness is the best reading found.**
  Clark's Weibull `L G(t) / G(E)` with `(omega, theta)` fitted so that
  its twelve-month averages match the given values in least squares, then
  its monthly increments moved as little as possible to a monotone `p`
  whose averages are the given values exactly, converts the book reported
  along `1 - 0.9^t` with ratios 0.9989, 1.0004, 0.9999, 1.0000, 1.0000 at
  12 to 60 months (the current reading: 1.3116, 0.9155, 0.9866, 0.9967,
  0.9991), is exact on Weibull truths, and is 1.0153 at 12 months on a
  slow loglogistic truth (omega 1.5, theta 36). The loglogistic corrected
  the same way is 0.9308 at 12 months on the book and 0.9303 on a fast
  Weibull truth: its heavy tail fits a pattern that reaches 100% at a
  finite age badly. The exact monotone `p` of least roughness with no
  curve is 1.08 to 1.10 at 12 months on fast patterns, since least
  roughness flattens the first year. Decision 23's mid-year shift
  (`origin_width = 12`) is 0.59 to 0.88 at 12 months and is not a
  substitute for the averaging relation.[^script]
* **Every exact candidate keeps decision 21's exact cases.** A pattern
  given at every month comes back unchanged (ratios 1 to 3e-16); at
  quarterly ages, where the current reading is within 1.3%, the corrected
  Weibull is exact; with three given ages it is 0.978, 1.008, 0.999
  against 1.312, 0.916, 0.982; a level below 1 changes nothing. The
  current reading's error depends on the basis: 1.40 at 12 months under
  seasonal writing and 1.17 under six-month policies, where the corrected
  Weibull is 0.9986 and 0.9994.[^script]
* **Industry patterns bound what the data cannot say.** With no truth,
  the exact monotone readings (least roughness, loglogistic and Weibull
  corrected) give the underwriting-year share at 12 months as 0.146 to
  0.164 on CLRD private passenger auto paid (current reading 0.195),
  0.087 to 0.094 on workers compensation (0.122), 0.0102 to 0.0111 on
  medical malpractice (0.0207), 0.0300 to 0.0305 on other liability
  (0.0459) and 0.0155 to 0.0160 on Friedland's XYZ paid selection
  (0.0230): a band of 2% to 12% between them, and the current reading 19%
  to 86% above its top, 5% to 13% below its bottom at 24 months, 1% to 4%
  below at 36, 0.3% to 2% below at 48 and about 1% below at 60.
  On Friedland's XYZ reported selection, whose factors at 108 and 120
  months are below 1, the corrected curves reproduce the given values
  with the development downward in those spans; the uncorrected Weibull
  smooths the dip away and misses the given values by 0.03.[^script]
* **Cost.** The prototype's scipy fit and SLSQP projection take 0.1 to 0.3
  s on 120 monthly ages; a Rust active-set solve on the 120 increments
  with about ten equality rows needs no new dependency
  (`prospicio_math::linalg::solve`). A least roughness on the log
  increments was tried and gave no evidence: SLSQP did not converge in 15
  of 17 runs.[^script]

Nothing in CI runs the study; the numbers are the script's on 2026-10-27
with numpy 2.5.3 and scipy 1.18.1. No reference implementation reads an
annual pattern at monthly ages through the averaging relation
(chainladder-python 0.10.1's `DevelopmentConstant` keeps the triangle's
grain, decision 18 item 10).

[^note]: Design note, reserving v0.2, decision 25, reading an annual accident-year pattern at monthly ages before the underwriting-year conversion
[^script]: The study's script, every candidate against the simulated book, four synthetic truths, the CLRD lines and Friedland's selections, run by hand with the Python venv
