---
type: Finding
title: Casualty exposure counts come from a curve's slope, and recorded claims respect attachment and limit
description: In prospicio_pricing::casualty, a policy's expected ground-up occurrences above t are premium x ELR / (share x [LEV(a+l) - LEV(a)]) x S(t), and S is an ILF or ELF table's slope just above t, so the table's constant cancels; the claims a policy records at or above t are counts x S(max(t, a)) when t <= a + l and none above, which is what a count test against trended, limit-capped claims must use. A ground-up Pareto from the threshold makes the frequency form on these counts equal to the ILF form.
tags: [casualty, exposure-rating, pareto, ilf, counts, pricing]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T03:00:00Z }
verified: { by: process:ci, at: 2026-10-19T03:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-pricing/src/casualty.rs
    title: CasualtyProfile::occurrences_above, expected_claims, count_test, sweep
  - id: test
    resource: ../validation/tests/pricing.rs
    title: clark_casualty::the_anchor_sweep_and_checks
  - id: note
    resource: ../docs/design/casualty-exposure.md
    title: Design note, decisions 4 to 9
---

# Facts

* A policy paying `s · min((X − a)+, l)` per occurrence expects
  `s · [LEV(a + l) − LEV(a)]` per ground-up occurrence, so its expected
  loss `premium × ELR` implies `premium × ELR / (s · [LEV(a + l) −
  LEV(a)])` ground-up occurrences a year, and `P(X > t)` times that above
  `t`. On an ILF or ELF table `LEV` is known only up to a constant, and
  `P(X > t)` is the table's slope just above `t` (the right derivative of
  the linear interpolation); the constant cancels between the two, so
  scaling every ILF by 2 leaves the counts unchanged. At or beyond a
  table's last limit there is no slope and the count is refused.
* On a ground-up Pareto from the threshold, `rate_by_frequency` with
  these counts and the same Pareto as the excess severity equals `rate`,
  the ILF form, to 1e-12, for excess policies above the threshold.
* The claims a policy records at or above a ground-up size `t` are the
  occurrences above its attachment, capped at `a + l`: `counts ·
  S(max(t, a))` when `t ≤ a + l` (a claim closed at the limit counts at
  the limit) and none above. A Poisson count test of trended claims is
  only fair against expectations built the same way; ground-up counts
  overstate it for an excess policy below its attachment and for any
  size above a policy's limit.
* For a Pareto from 1M, a 5M xs 5M layer of a 10M policy costs `1M ln 2`
  (693K) per occurrence at α = 1 and `1M (1/5 − 1/10)` (100K) at α = 2,
  a factor of 6.9, so the sweep with no stated central alpha does not
  set a price (decision 5).
