---
type: Reference
title: Mildenhall and Major, Pricing Insurance Risk, Parts I and II, reproduced in prospicio
description: The printed numbers of chapters 3 to 11 (risk measures and portfolio pricing) recomputed by the guides pir-risk-measures.md and pir-portfolio-pricing.md; what matches, the grids and conventions needed, and the printed values that do not match, with their causes (a wrong lognormal mu, TVaR of stop-lossed units, a factor of 10 in Table 9.15, Esscher and exponential on a lognormal).
resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
tags: [pricing, risk-measures, distortions, var, tvar, classical-premium, ruin, guides, reference, parity]
status: stable
generated: { by: claude-code/guides-session, at: 2026-10-17T23:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:00:00-07:00 }
sources:
  - id: book
    resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
    title: Stephen J. Mildenhall and John A. Major (2022), Pricing Insurance Risk, Theory and Practice, Wiley
  - id: guide1
    resource: ../docs/guides/pir-risk-measures.md
    title: Part I guide (chapters 3 to 7)
  - id: guide2
    resource: ../docs/guides/pir-portfolio-pricing.md
    title: Part II guide (chapters 8 to 11)
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guides and checks every printed number in CI
---

# Settings that reproduce the book

* Case studies on `aggregate`'s grid: step 1/64, 2^16 points, severity rounded
  on 2^17 points and renormalized by the mass below the top (the
  `discretize` of the case studies guide). Tame at a 99.99% VaR standard (the
  book's Tables 7.2, 9.x and 11.x; the case studies guide uses 99.9%),
  Cat/Non-Cat at 99.9%, cost of capital 0.10.
* Net units: Tame B net of 12.91 xs 56.17 aggregate, Cat net of 79.64 xs 41.11,
  made exactly with `Grid.map`. A `Tower.on_grid` net carries FFT round-off of
  about 1e-19 where the probability should be zero, which turns the
  exponential premium of net B from 50.42 into about 510 (issue #367). The
  round-off's digits differ between builds (7.0e-20 and 509.4 from the Python
  wheel, 7.7e-20 and 510.3 from the R package on the same machine), so the
  guides print bounds.
* The hurricane table (Table 4.3) reproduces on a step of 20 (USD millions)
  with 2^21 points; the book's aggregate VaRs are multiples of 20.
* Classical principles: the book's ten are `classical_premium` names
  `expected_value`, `var`, `variance`, `standard_deviation`, `semi_variance`,
  `exponential`, `esscher`, `dutch`, `fischer` (q = 2), plus the net premium,
  calibrated on the gross total to the CCoC premium.
* Blend (Exercise 197): `Distortion.convex` on (s, (s + nu) / (1 + nu)) for
  s = 0, 0.01, 0.03, 0.05, 0.1, 0.2 and nu = 0.02 to 0.10, plus (1, 1). The
  concave hull lifts the s = 0.03 point (0.0673 to 0.0713); Tables 11.7 and
  11.8 still reproduce to four figures.

# Matched

Tables 3.2 to 3.4, Exercises 38, 49 and 63, Tables 4.2, 4.3, 4.4, 4.6 and 4.7
(all but the cells below), Example 83, Exercise 183, Table 7.2, Tables 9.2,
9.3, 9.5 (seven principles), 9.6, 9.11 to 9.13, Table 10.3, Tables 11.1 and
11.3, 11.5 (Tame and Cat/Non-Cat), 11.7 and 11.8 (premiums, to four figures).

# Not matched, and why

* **Example 54's mu = 21.315** does not fit a lognormal with mean USD 5
  billion and sigma 2.58 (mu = 19.005 in dollars, 5.189 in millions); the
  table itself (median 179) uses 19.005.
* **TVaR of a stop-lossed unit** (Table 4.6 net B at 90%, 95%, 97.5%: 58, 59,
  59 against 56.5, 56.8, 57.5; Table 4.7 net Cat at 95%, 97.5%, 99%: 64, 75,
  81 against 45, 49, 62). Inconsistent with the book's own VaR rows: with VaR
  flat on the cover's attachment, TVaR is the attachment plus
  E[(X - attachment)+] / (1 - p), and E is 0.032 for net B. The net totals
  agree, so only these unit cells are affected.
* **Table 9.15 (natural scale)** is 10 times Eq. (9.2) with delta = 0.1 in
  every cell (ratio 9.98 to 10.00); it equals u / (r E[X]).
* **Table 9.5's Esscher and exponential rows** (Cat/Non-Cat): a lognormal has
  no exponential moment, so these depend on where the grid stops (Cat's
  exponential premium at loading 0.014 is 35.5 on a grid to 1,024 and 25.2 to
  512). The book's numbers are its grid's.
* **Rearrangement Algorithm** (Table 4.5): 352.9 to 353.8 from three fixed
  starts (column j rotated by 7, 13 or 19 times j places, the same in the
  Python and R guides), 352.9 to 353.5 from random shuffles, against 352.8; a
  heuristic whose answer depends on a start the book does not give.
* **Table 4.4 as extracted** sets the 0.8 row's difference one line low.

# Gaps found

Issue #366 (Probability: rearrangement algorithm, `tvar` on parametric
severities, a normal, an ogive) and #367 (Aggregate: FFT fuzz in
`Tower.on_grid`, an option pricing model).
