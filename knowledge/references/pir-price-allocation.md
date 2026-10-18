---
type: Reference
title: Mildenhall and Major, Pricing Insurance Risk, Part III (price allocation), reproduced in prospicio
description: What the guides pir-price-allocation.md and pir-price-allocation-practice.md found recomputing chapters 12 to 15 and 19 of Pricing Insurance Risk; the book's linear allocation at an atom at the asset level (prospicio matches it at a - 1e-9), the two covariance allocations, Example 302's net view, and the CCoC row of Table 15.36, which is not a natural allocation.
resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
tags: [pricing, capital-allocation, natural-allocation, distortions, bodoff, guides, reference, parity]
status: stable
generated: { by: claude-code/guides-session, at: 2026-10-17T23:30:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:30:00-07:00 }
sources:
  - id: book
    resource: https://www.wiley.com/en-us/Pricing+Insurance+Risk%3A+Theory+and+Practice-p-9781119755678
    title: Stephen J. Mildenhall and John A. Major (2022), Pricing Insurance Risk, Theory and Practice, Wiley
  - id: guide
    resource: ../docs/guides/pir-price-allocation.md
    title: The guide that recomputes the small examples of chapters 12 to 15
  - id: practice
    resource: ../docs/guides/pir-price-allocation-practice.md
    title: The guide that recomputes Tables 13.3, 15.36 and 15.39 (Cat/Non-Cat, gross) and Example 302
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guides and checks every printed number in CI
---

# What matched

- Chapter 12 on the Simple Discrete Example (16 equally likely rows reproduce
  its 1/16 probabilities): Tables 12.5 and 12.8 (TVaR 75% = 94.5, coTVaR 4.5 and
  90, Merton-Perold 4.5 and 84.5, Shapley 7.25 and 87.25; at a premium of 80
  the TVaR level 0.6975 and Shapley 7.55 and 72.45), the haircut (8, 72), equal
  risk TVaR (level 0.6796, 9.56 and 70.44), Kalkbrener (5.216, 74.784),
  Exercise 219's equal priority (533, 400, 267).
- Chapter 15: Example 258 (PH 0.5: 6.2048, 45.1838), Example 263 (lifted at 80),
  Tables 15.14 and 15.15 (capital 3.5274 and 45.0839; returns 41.24%, 49.76%,
  and 41.51%, 52.39% at 80; margins -0.5423 and 2.4365 at 11), all thirty
  numbers of Table 15.20 (largest gap 4.6e-5), Bodoff's Thought Experiment 1
  (Tables 15.21 to 15.31, PH and Wang lifted and linear), Example 235.
- Table 13.3 gross Cat/Non-Cat: 16 of 18 margins to the printed digits; Cat's
  scaled TVaR and scaled EPD margins are 12.845 and 15.675 against 12.85 and
  15.68, on the rounding boundary.
- Tables 15.36 and 15.39, gross: PH, Wang, dual and TVaR lifted premiums,
  capital and assets within 0.06; PLC assets 113.2 and 154.0. The book's PH
  capital (103.2) is prospicio's 103.25, not `aggregate` 0.30.1's 103.52.
- Example 302: 51.1560, 1.2613 (gross view), 49.8986, 1.2574 (A/B), 1.2497 (net
  view), best share 2/3 at the A/B premium.

# Conventions and findings

- **The linear allocation at an atom at the asset level.** The book's linear
  natural allocation conditions on `X ∧ a`, so an outcome with total exactly
  `a` is lumped with the defaults. prospicio (as `aggregate` 1.0.1) splits the
  default weight only over totals above `a`, so the outcome at `a` is paid as
  in full. Both are members of the natural allocation set; they differ only
  when `a` is a total with positive probability (always so for `assets(p)` on
  a small discrete book). `price(..., assets=a - 1e-9, allocation="linear")`
  gives the book's numbers (Example 235, Table 15.20, Tables 15.30 and 15.34).
  The lifted allocation agrees at `a` itself.
- **Two covariance allocations.** prospicio's `capital(..., "covariance")` is
  `rho(S) Cov(X_j, S) / Var(S)`; the book's (Kalkbrener, Exercise 217, and the
  var/cov premium of Table 15.31) is `E[X_j] + (rho(S) - E[S]) Cov / Var`.
  Table 15.31 uses the covariances of the unlimited losses with the limited
  expected losses.
- **Table 15.36's CCoC row is not a natural allocation.** Under the CCoC each
  outcome's weight is at least `1 / 1.1` of its probability, so every natural
  allocation charges a unit at least `E[X_i(a)] / 1.1`; Non-Cat's 79.99 gives
  72.72, and the book's lifted Non-Cat premium is 68.11 (Cat 47.04, Non-Cat
  assets -50.62). prospicio gives 39.72 for Cat, `aggregate` 0.30.1 40.51: the
  lifted CCoC is version dependent, as for Monograph 15.
- **Book parameter rounding.** The PH parameter of Exercise 276, 0.82156,
  prices the total at 30.7361; calibrated to the CCoC premium 30.73636 it is
  0.821554. The Wang 0.209615 matches.
- **The net view of a cover** (Example 302) needs the weights of the net
  order: a `Portfolio` whose units are the net pieces plus the cover scaled by
  1e-6 keeps that order, and its allocation divided by 1e-6 is the book's.
- **Speed.** `Portfolio.premium_bounds` on the Cat/Non-Cat grid (about 70,000
  totals) did not finish in two minutes; on four totals it is instant.
