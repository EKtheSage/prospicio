---
type: Finding
title: A factor to ultimate at one age is the tail from that age, and on a book observed at one age it is the whole projection
description: A cumulative selected pattern with one value fixes no link, so read by decision 12's rules it is the tail exactly when its age is the oldest and no tail is given. On a triangle with one origin at one age, or a latest diagonal at one age, that is the only age, and the value projects the book alone; on a full triangle it is the same fit as a constant tail of that factor. At any other age, with a tail or blended it would select nothing.
tags: [reserving, development, pattern, latest-diagonal, tail]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-26T12:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 20, a cumulative pattern with one value
  - id: test
    resource: ../validation/tests/reserving_latest_diagonal.rs
    title: Validation test of a latest diagonal with a selected pattern, issue 632's cases, run in every cargo test
  - id: errors
    resource: ../validation/tests/reserving_pattern_basis.rs
    title: Validation test of the plain path's pattern errors and the one value as a constant tail on RAA, run in every cargo test
---

# What was found

A cumulative pattern fixes the links between the ages it gives, and its
level at its last age fixes none (`reserving-v02.md`, decision 12).[^note]
A pattern with one value is only a level. With no tail and the value at
the triangle's oldest age, the level is the tail, so `{a: CDF(a)}` gives
`pattern_tail = CDF(a)` with no parameter error and `pattern_rescale = 1`
(decision 20).

* **A book observed at one age.** One origin at 12 months with 200 paid
  and `{12: 3.0}` to ultimate projects to 600 in the chain ladder; with
  1,500 of premium and an ELR of 0.7 Bornhuetter-Ferguson gives 900,
  Benktander's two iterations 800 and Cape Cod 600, the same as the full
  pattern from 12 to 60 months, which decision 18 folds past the oldest
  age into the same tail.[^test]
* **A longer pattern that ends at the only age.** On a triangle
  observed at one age a pattern's earlier ages fix no link, so a pattern
  that ends at that age is its level there, the same tail: the pattern
  from 12 to 48 months on an origin at 48 months with 800 paid gives 896,
  as `{48: 1.12}` does.[^test]
* **A full triangle.** `{120: 1.05}` on RAA gives every origin the
  ultimate of the chain ladder with a constant tail of 1.05 to 1e-15.[^errors]
* **Elsewhere it is inert.** At an age before the oldest, with a given
  tail (on a latest diagonal also `PastPatternEnd::Tail` or
  `NoDevelopment`, even with an origin older than the value, where the
  level is rescaled to the tail spread past it) or at a weight below 1,
  nothing of the value would be left, so each is an error.

[^note]: Design note, reserving v0.2, decision 20, a cumulative pattern with one value
[^test]: Validation test of a latest diagonal with a selected pattern, issue 632's cases, run in every cargo test
[^errors]: Validation test of the plain path's pattern errors and the one value as a constant tail on RAA, run in every cargo test
