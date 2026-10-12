---
type: Finding
title: The Bayesian ODP chain ladder against Stan, the chain ladder and the ODP bootstrap
description: With vague priors BayesOdp matches the same model in Stan on GenIns and RAA (every summary within 0.70 of its tolerance) and on GenIns gives a mean reserve 0.8% above the chain ladder and 0.1% below the ODP bootstrap, with an sd 2.0% below the bootstrap's.
tags: [bayes, reserving, odp, bootstrap, stan, genins, raa]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T00:00:00Z }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes.rs
    title: The parity and sanity tests
  - id: odp
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: prospicio_reserving::bayes::BayesOdp
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 3, 5 and 10)
---

# Finding

* Against Stan (190 summaries: 19 parameters' means and sds, each
  origin's and the total's reserve mean, sd and 5/50/95% percentiles,
  `phi` and `c0`), `BayesOdp` at its defaults (4 chains of 1,000 draws,
  10,000 simulations, seed 0) uses at most 0.70 of any tolerance; the
  largest is RAA's 1982 reserve sd (676 against 640, 8% allowed).[^test]
* GenIns total reserve: Stan mean 18,851,576, sd 2,984,047; `BayesOdp`
  18,834,037 and 2,937,848; the chain ladder 18,680,856; the ODP bootstrap
  (20,000 simulations, seed 1) 18,851,366 and 2,998,662. England and
  Verrall's result that vague priors give back the bootstrap holds closely:
  the bootstrap's residual scaling `sqrt(n/(n-p))` (about 1.24 here) did not
  open a gap in the sd beyond 2%.[^test]
* With the priors off, the posterior mode is the chain ladder: the fitted
  means agree to 1e-8 relative on RAA and GenIns. With sd-`s` priors the
  mode moves by `O(1/s^2)`: the shift at sd 100 is 100 times the shift at
  sd 1,000 within 1%.[^odp]
* RAA at the default target acceptance 0.8 has 3 divergent transitions in
  4,000 draws, reported as a warning; its `b[10]` posterior is skewed (one
  cell, shape `x/phi` 0.17). GenIns has none (min bulk ESS 5,563).[^design]
* `loo()` on GenIns: elpd -749.9 (se 6.0), p_loo 16.1, largest Pareto k
  0.90: the corner cells that alone fix a parameter are influential. RAA
  has a negative increment, where the normalised ODP density is minus
  infinity, so `loo()` is an error there.[^test]
* Speed: a default GenIns fit takes about 80 ms in a release build. The
  pointwise density in `loo()` integrates a normalising constant per draw
  and cell, 5.1 s serially for 4,000 x 55, 0.56 s in parallel.[^design]

Not yet run by CI.
