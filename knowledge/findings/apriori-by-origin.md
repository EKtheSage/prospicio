---
type: Finding
title: An a-priori loss ratio per origin must be keyed by origin, since a fit's origins are not the rows given
description: The expected loss, Bornhuetter-Ferguson and Benktander methods take one loss ratio per origin (Apriori), and each origin of such a fit equals that origin fitted alone at its own loss ratio, to the bit, on every path; a vector by position would not line up with the user's rows because a latest diagonal's fit takes its rows in any order and fits its origins oldest first, and a triangle segment is trimmed to the origins it observes; on latest diagonals values in order are an error unless the rows are oldest first.
tags: [reserving, expected-loss, bornhuetter-ferguson, benktander, latest-diagonal, segments, one-year, bindings]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-26T12:00:00Z }
verified: { by: process:ci, at: 2026-10-26T12:00:00Z }
sources:
  - id: rust-tests
    resource: ../crates/prospicio-reserving/src/expected_loss.rs
    title: The unit tests of Apriori (identities, errors, labels, segments)
  - id: diagonal-tests
    resource: ../crates/prospicio-reserving/src/latest_diagonal.rs
    title: The unit test of a diagonal given newest first
  - id: one-year-tests
    resource: ../crates/prospicio-reserving/src/one_year_bootstrap.rs
    title: The unit test of the one-year view's closed form with one loss ratio per origin
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: The design note, decision 22 (an a-priori loss ratio per origin)
---

# What was found

* The three methods' arithmetic is per origin, so a loss ratio per origin
  changes nothing else: each origin of a fit with `Apriori::ByOrigin` (or
  `InOrder`) equals the same origin fitted alone with that origin's loss
  ratio for every origin, bit for bit, for `n_iters` 0, 1, 2 and 5 on RAA.
* A fit's origins are not the rows a user gives. `LatestDiagonal` accepts
  origins in any order and `fit_diagonal` fits them oldest first (a
  diagonal that skips an origin is `Error::EmptyOrigin` before any a priori
  is read); `Triangle::segments` trims each segment to the origins its
  column observes. So one value per origin is matched by origin period
  (`Period`, grain included), and values in order are read as oldest first
  over the triangle's (or the diagonals') origins, then keyed. A diagonal
  given newest first lines up only when keyed.
* A diagonal's exposure is read in the order of its rows, so values in
  order next to it would go to the wrong origins on rows not oldest first.
  On latest diagonals values in order are therefore `Error::Apriori` unless
  the rows, segment after segment, are the origins oldest first, each once;
  on such rows they equal the keyed fit.
* One map serves segments that start in different years: the map is
  checked once against every origin of the triangle (or the diagonals
  together), and each segment takes its own origins' values. An exact
  match per segment would reject any map on a triangle whose lines span
  different years.
* In the one-year view the re-reserving method keeps each origin's own
  value in every simulated year: on a triangle with no noise the claims
  development result is the closed form `(f - 1) (a[o] E / cdf - latest)`
  per origin, and a keyed and an in-order a priori draw the same.

# Where it is checked

The Rust unit tests above (run by CI's `cargo test`), and the same
identities through Python (`python/tests/test_reserving.py`,
`python/tests/test_reserving_latest_diagonal.py`) and R
(`R/prospicio/tests/test-reserving.R`,
`R/prospicio/tests/test-reserving-latest-diagonal.R`).
