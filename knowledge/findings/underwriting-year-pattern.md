---
type: Finding
title: An accident-period pattern converts exactly to an underwriting-year pattern by the year's exposure over accident months, and an annual accident-year pattern only approximately in the first two years
description: P_UY(a) = sum_t e(t) p(a - t) with e the writing convolved with even earning (the parallelogram's (2t + 1)/288 under the default) links a simulated book's accident-month or monthly accident-year chain ladder pattern to its underwriting-year one to 1e-12 and matches the pricing crate's MonthlyPatterns::from_annual to 1e-13; an accident-year pattern given only at annual ages, read by decision 18's rule, puts the underwriting year 31% too far developed at 12 months and 8% too little at 24.
tags: [reserving, development, pattern, underwriting-year, exposure]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-26T18:00:00Z }
verified: { by: process:ci, at: 2026-10-26T18:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 20, an accident-period pattern on an underwriting basis
  - id: test
    resource: ../validation/tests/reserving_uy_pattern.rs
    title: Validation test of the underwriting-basis conversion, run in every cargo test
---

# What was found

`SelectedPattern::to_underwriting_basis` turns an accident-period pattern
into the pattern of an underwriting period (`reserving-v02.md`, decision
20).[^note] With policies written at an even rate through each month and
earning evenly over `m` months, a policy's earning by accident month is
`1/(2m)`, `m - 1` months of `1/m` and `1/(2m)`, and the period's exposure
`e` is the monthly writing shares convolved with it. The underwriting
period's share developed at age `a` from its start is

```text
P_UY(a) = sum_t e(t) p(a - t)
```

on percents developed, with `p` an accident month's share at its own age.
Under the default (even writing over 12 months, 12-month policies) `e(t) =
(2t + 1)/288` for `t` up to 11 and `(47 - 2t)/288` from 12 to 23, the
triangular density integrated by month, with mean 11.5 months; a linear
`p(t) = t/T` gives `P_UY(a) = (a - 11.5)/T` once every accident month has
begun, and `a(a + 1)(2a + 1)/(1728 T)` in the first year.

An accident-year pattern is decision 18's even-exposure average of twelve
accident months, so it converts with no deconvolution whenever the
exposure is an even spread over twelve months of some weights: when the
policy term is a whole number of years, or the writing is even within the
year. Six-month policies written seasonally have no such weights, and the
conversion asks for an accident-month pattern.

# Evidence

All in the validation test,[^test] run in every `cargo test`:

- The hand closed forms above to 1e-14, the result ending 23 months after
  an accident-month pattern's end (12 after an accident-year pattern's) at
  exactly the pattern's level.
- With the accident-month pattern `p(k) = paid_by((k - 1/2)/12)` and the
  written increments spread half and half over two months,
  `MonthlyPatterns::from_annual` of the pricing crate gives the same
  shares to 1e-13.
- A book of three policies a month from 2000 to 2019, each accident month
  reported along `1 - 0.9^t`: with even writing and 12- and 6-month
  policies, the accident-year triangle's chain ladder pattern (monthly
  development) converts to the underwriting-year triangle's to 1e-12, and
  projects each underwriting year to its ultimate to 1e-12; with seasonal
  writing and 12-, 6- and 18-month policies, the accident-month pattern
  converts to it to 1e-12.
- The same book's triangles valued annually: the accident-year pattern at
  12, 24, ... months is read between its ages by decision 18's rule
  (linear from 0 below 12 months, `ln CDF` linear between), and the
  converted share over the book's own is 1.3116 at 12 months, 0.9155 at
  24, 0.9866 at 36, 0.9967 at 48, 0.9991 at 60, 0.9996 at 72 and exact from
  84. An accident year at six months has much less than half of its full
  year's share, which the linear rule overstates.

# Consequences

Give an accident-month pattern, or an accident-year pattern at every
month, where the underwriting year's first two years matter; an annual
accident-year pattern is close from about 36 months. No reference
implementation converts a reserving pattern between accident and
underwriting periods (chainladder-python 0.10.1 has no such argument).

[^note]: `docs/design/reserving-v02.md`, decision 20.
[^test]: `validation/tests/reserving_uy_pattern.rs`.
