---
type: Finding
title: The one-OS compartmental means are invariant under a rate swap
description: The map (k_er, k_p, RLR, RRF) -> (k_p, k_er, RLR k_er / k_p, RRF k_p / k_er) leaves every outstanding and paid mean of the one-OS model unchanged (to 1e-12 at 200 random points), so with both columns observed the likelihood still has two modes and only the priors tell them apart; the closed form stays finite through k_er = k_p with the kernel g(t) = t e^(-k_p t) (1 - e^(-x)) / x, x = (k_er - k_p) t.
tags: [reserving, compartmental, bayes, identification, numerics]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T21:00:00-07:00 }
sources:
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/compartmental.rs
    title: prospicio_reserving::bayes::Compartmental and its unit tests
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decision 2, "Identification")
  - id: scalar
    resource: ../crates/prospicio-reserving/src/bayes/scalar.rs
    title: exprel_neg, the kernel's series branch
---

# Finding

* With `g(t) = (e^(-k_p t) - e^(-k_er t)) / (k_er - k_p)`, the one-OS means
  are `OS_i(t) = Π_i RLR_i k_er g(t)` and `PD_i(t) = Π_i RLR_i RRF_i [(1 -
  e^(-k_p t)) - k_p g(t)]`. `g` is symmetric in the two rates, `RLR_i k_er`
  is kept by the swap, and so is `RLR_i RRF_i`, so the swap leaves both
  means unchanged for every origin and age. The unit test
  `the_means_are_invariant_under_the_rate_swap` checks it to 1e-12 relative
  at 200 random points (rates 0.05 to 6, ages 0.1 to 15).[^model][^design]
* So "outstanding breaks the symmetry", as the sources say, holds only with
  `RLR` and `RRF` fixed. On company 337 the paper's priors exclude the
  mirror mode by about 150 nats of rate prior alone; no posterior draw had
  `k_er <= k_p` in the reference (100,000 draws) or at our defaults. With
  rate priors centred on the mirror image the fit lands there and warns
  (`the_rate_swap_warns`); `RateOrder::Ordered` (`k_er = k_p + e^delta`,
  the same priors restricted to `k_er > k_p`) gives the paper's posterior on
  company 337 within the Stan rows' tolerances.[^model]
* From paid alone only `RLR × RRF` is identified (the paid mean depends on
  `RLR_i RRF_i` and the rates, which swap with no penalty), so the model
  requires the outstanding column.[^design]
* The kernel: written as `t e^(-k_p t) exprel(-x)` with `x = (k_er - k_p)
  t` and `exprel(-x) = (1 - e^(-x)) / x`, it is exact for every difference,
  with a series branch below `|x| = 1e-2`, where the quotient's dual
  derivative would lose about `2 eps / |x|`. The density and its dual
  gradient match central differences through `k_er = k_p`.[^scalar][^model]
