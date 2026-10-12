---
type: Finding
title: The Bayesian ODP chain ladder against Stan, the chain ladder and the ODP bootstrap
description: With vague priors BayesOdp matches the same model in Stan on GenIns and RAA at every one of 30 seeds, within tolerances sized at 4 times its own Monte Carlo error, which on RAA's young origins is about twice GenIns'; on GenIns it gives a mean reserve 0.7% above the chain ladder and 0.3% below the ODP bootstrap, with an sd 2.5% below the bootstrap's.
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
  - id: bindings
    resource: ../python/tests/test_reserving_bayes.py
    title: The Python binding's tests (R's are R/prospicio/tests/test-reserving-bayes.R)
---

# Finding

* `BayesOdp`'s Monte Carlo error at its defaults (4 chains of 1,000
  draws, 10,000 simulations), measured as the seed-to-seed sd of each
  summary over seeds 0 to 19, is at most `0.018 s` on a mean, 2.4% on an
  sd and `0.066 s` on a reserve percentile on GenIns (`s` Stan's posterior
  sd), and `0.03 s`, 4.6% and `0.083 s` on RAA. RAA's largest are its
  young origins' reserves, which ride on the skewed `b[9]` and `b[10]`:
  the 1982 reserve is one cell on `b[10]` (posterior sd 3.6), and its sd
  ranged over about 4% of Stan's 640 between seeds. The sd error of about
  1.2% first assumed holds only for GenIns' parameters.[^design]
* Against Stan (190 summaries: 19 parameters' means and sds, each
  origin's and the total's reserve mean, sd and 5/50/95% percentiles,
  `phi` and `c0`), with tolerances of at least 4 such errors (GenIns
  `0.1 s`, 10%, `0.3 s`; RAA `0.15 s`, 20%, `0.4 s`), every row passes at
  each of seeds 0 to 29, using at most 0.80 of a tolerance. Averaged over
  the seeds, every summary is within 0.2 of a tolerance of Stan's, so the
  samplers agree on the posterior. The first tolerances (`0.1 s`, 8%,
  `0.15 s`) failed RAA at 10 of 20 seeds.[^test]
* GenIns total reserve: Stan mean 18,851,576, sd 2,984,047; `BayesOdp`
  (seed 0) 18,803,484 and 2,924,291; the chain ladder 18,680,856; the ODP
  bootstrap (20,000 simulations, seed 1) 18,851,366 and 2,998,662.
  England and Verrall's result that vague priors give back the bootstrap
  holds closely: the bootstrap's residual scaling `sqrt(n/(n-p))` (about
  1.24 here) did not open a gap in the sd beyond 2.5%.[^test]
* With the priors off, the posterior mode is the chain ladder: the fitted
  means agree to 1e-8 relative on RAA and GenIns. With sd-`s` priors the
  mode moves by `O(1/s^2)`: the shift at sd 100 is 100 times the shift at
  sd 1,000 within 1%.[^odp]
* RAA at NUTS's usual target acceptance 0.8 gave divergent transitions
  in 4,000 draws at 7 of seeds 0 to 7 (0 to 6 each), reported as a
  warning; its `b[10]` posterior is skewed (one cell, shape `x/phi` 0.17).
  At 0.95, `BayesOdp`'s default, seeds 0 to 29 gave none on RAA or GenIns,
  with max R-hat at most 1.006 and the summaries unchanged within Monte
  Carlo error.[^design]
* `loo()` on GenIns (seed 0): elpd -750.5 (se 6.1), p_loo 16.7, largest
  Pareto k 0.80: the corner cells that alone fix a parameter are
  influential. RAA
  has a negative increment, where the normalised ODP density is minus
  infinity, so `loo()` is an error there.[^test]
* Speed: a default GenIns fit takes about 0.1 s in a release build. The
  pointwise density in `loo()` integrates a normalising constant per draw
  and cell, 5.1 s serially for 4,000 x 55, 0.56 s in parallel.[^design]
* Short chains do not pass the diagnostics on GenIns (measured at target
  acceptance 0.8): 2 chains of 300
  draws (300 warm-up) gave R-hat above 1.01 for some parameter at seeds
  0, 1 and 3 (largest 1.014 to 1.023, the bulk ESS still above 770), and 4
  chains of 500 at seed 1 (b[6] 1.013). The defaults passed at all three
  seeds (largest R-hat 1.002 to 1.005), so the binding tests that expect no
  warning use the defaults, and 2 chains of 10 kept draws give a sure ESS
  warning.[^bindings]

Not yet run by CI.
