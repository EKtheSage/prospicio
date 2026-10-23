---
type: Finding
title: An accident-month pattern averaged over the months of an accident year projects even exposure exactly, and the tail past a pattern's end is best spread by the origin grain
description: Averaging percents developed over an origin period's months, P(a) = (1/n) sum p(a - q s), gives the exact ultimate of every origin up to the pattern's end when the exposure is even through the period, an accident year in progress included; between the end and the end plus the period it defers the tail's development. A tail spread from the pattern's end at one step per origin period gives an annual book chainladder-python's 1.02538 one year past the end, where a monthly step gives 1.00116. No reference implementation does either.
tags: [reserving, development, pattern, latest-diagonal, tail]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-23T18:00:00Z }
verified: { by: process:ci, at: 2026-10-23T18:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 18, a pattern on another origin basis and ages past its end
  - id: test
    resource: ../validation/tests/reserving_pattern_basis.rs
    title: Validation test of the averaged origin basis, patterns off the triangle's ages and origins past the pattern's end, run in every cargo test
---

# What was found

`SelectedPattern::averaged_over_origins` turns an accident-month pattern
`p` into the pattern of origin periods of `n` sub-periods of `s` months
(`reserving-v02.md`, decision 18).[^note] An origin at age `a` has its
`q`-th youngest sub-period at age `a - q s`, so with the exposure even
through the period its share reported is

```text
P(a) = (1/n) * sum_{q=0}^{n-1} p(a - q s)
```

on percents developed, with `p` 0 at an age of 0 or less, linear from 0 to
its first age, `ln CDF` linear between its ages and held at its level past
its end. The result runs from the pattern's first age to its end `E` plus
`(n - 1) s` months.

* **Exact to the pattern's end.** Each month reaches ultimate the same way
  and the average is linear, so for origins up to `E` the chain ladder with
  `P` gives each origin the sum of its months' ultimates, whatever the
  tail. On a synthetic book of accident months reported along `1 - 0.9^t`
  to 60 months, summed to accident years and quarters, every origin
  projects to 12 (or 3) times the monthly ultimate to 1e-12, on the plain
  triangle path and through the latest diagonal, and an accident year at
  June (6 months) projects to its twelve months' ultimate.[^test]
* **The closed forms.** `p(t) = t/24` averages to `(a - 1)/24` for accident
  quarters and to 13/48 and 37/48 at 12 and 24 months for accident years;
  `p(t) = t/60` gives `(a - 5.5)/60` from 12 months; `p(t) = 1 - 0.9^t` gives
  `1 - 0.9^a (0.9^-12 - 1) / (12 (0.9^-1 - 1))`, 0.46182215236075 at 12
  months and 0.84800267994690 at 24.[^test]
* **Deferred after the end.** Between `E` and `E + (n - 1) s`, `P` holds
  the level for the sub-periods past the end, so a tail's development is
  taken to happen only at the period's last age: the ultimate is slightly
  overstated there.
* **Uneven exposure is not exact.** With monthly ultimates `U_j`, an origin
  projects to `sum U_j p(a - j) / P(a)`, not `sum U_j`; the test pins the
  miss.[^test]
* **The spread past the end steps by the origin grain.** An origin older
  than the pattern's end takes what is left of a tail `T` spread from the
  end as chainladder-python's constant tail at decay 0.5. Stepping by the
  origin period, `T = 1.05` leaves 1.0253798721500054 one year past the end
  of an annual book (chainladder-python's 1.02538 at 132 months on RAA) and
  1.0132158418990078 two years past it, and an accident quarter one quarter
  past the end gets the same 1.0253798721500054. Stepping by the month
  would leave 1.0011602413037541 one year past the end, so the spread would
  depend on the pattern's grain.[^test]

No reference implementation averages a pattern over origin bases or reads
one off a triangle's ages: chainladder-python 0.10.1's
`DevelopmentConstant(patterns, style, callable_axis, groupby)` has no
origin-basis argument and keeps the triangle's grain
([chainladder-python's DevelopmentConstant](/references/chainladder-python-development-constant.md)).

[^note]: Design note, reserving v0.2, decision 18, a pattern on another origin basis and ages past its end
[^test]: Validation test of the averaged origin basis, patterns off the triangle's ages and origins past the pattern's end, run in every cargo test
