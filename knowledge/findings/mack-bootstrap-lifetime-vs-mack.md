---
type: Finding
title: Mack's bootstrap, lifetime view, against Mack's standard errors and EVW Table 4
description: Bootstrapping Mack's model to the last age (MackBootstrap::fit, EVW 2019 Appendix 1) reproduces Mack's analytic standard error on RAA, GenIns and ABC within Monte Carlo error only with centred residuals, once the parameter error is scaled by the resampled residuals' variance 1 - m^2; uncentred, the pool's mean biases the mean reserve (RAA +17%, GenIns +0.7%, ABC -0.8%). EVW's Table 4 expected reserves agree with the centred bootstrap, not the uncentred one, so centring is the default. Gamma draws were slow at large shapes (ABC) until the Gamma sampler became Marsaglia-Tsang.
tags: [reserving, bootstrap, mack, lifetime, evw, residuals, performance]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-07T18:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 8, section on Mack's lifetime view
  - id: test
    resource: ../validation/tests/reserving_mack_bootstrap.rs
    title: Validation test of the lifetime view against R MackChainLadder and EVW Table 4
  - id: unit
    resource: ../crates/prospicio-reserving/src/mack_bootstrap.rs
    title: Unit tests of MackBootstrap::fit
  - id: evw
    resource: https://openaccess.city.ac.uk/id/eprint/21270/
    title: England, Verrall and Wuthrich (2019), On the lifetime and one-year views of reserve risk, with application to IFRS 17 and Solvency II risk margins, Insurance - Mathematics and Economics 85
---

# Finding

`MackBootstrap::fit` simulates every future cumulative value to the last
age from the observed latest one, each from the one before, with pseudo
factors from resampled link-ratio residuals (parameter error) and a
process draw of mean `f* C` and variance `sigma^2 C^(2 - alpha)` (EVW
2019, Appendix 1, steps 7(a) to (g)). Against R ChainLadder's
`MackChainLadder` (volume weighting, either rule for the last sigma):[^test]

* **Centred residuals reconcile.** At 50,000 simulations (Gamma) the
  standard deviation is 0.991 to 1.007 times Mack's standard error per
  origin and 0.994 to 1.002 in total on RAA, GenIns and ABC, every origin
  within three Monte Carlo standard errors; the mean reserve is the chain
  ladder's within 2.2 standard errors of the mean.
* **The pool's variance.** The resampled residuals have mean `m` and mean
  square 1, so variance `v = 1 - m^2`: RAA 0.981, GenIns 0.9998, ABC
  0.996. Every pseudo factor's variance is `v sigma^2 / S_k`, so the
  bootstrap's parameter error is `sqrt(v)` times Mack's. Parameter error
  alone (`MackProcess::None`) on RAA came out 0.8% to 1.2% below R's
  parameter risk at every origin but 1990 (2.7 to 3.8 standard errors at
  50,000), which `sqrt(0.981) = 0.990` explains; the CI test compares with
  `sqrt(process^2 + v parameter^2)`.
* **Mack's formula is linear.** After the `v` adjustment, RAA 1990's
  parameter error is above Mack's: its first pseudo factor's standard
  deviation is a third of the factor, and the variance of a product of
  random factors exceeds the sum of their relative variances that Mack's
  formula keeps. Centred, the pseudo factors are independent with mean
  `f_k` and variance `v sigma_k^2 / S_k`, so the exact parameter variance
  is `C^2 (prod(f_k^2 + v sigma_k^2 / S_k) - prod f_k^2)` against Mack's
  `C^2 prod f_k^2 sum(v sigma_k^2 / (f_k^2 S_k))`. On RAA (log-linear
  sigma, whose linear form reproduces R's 7,275 for 1990) the exact value
  is 1.0066 times `sqrt(v)` times Mack's for 1990, 1.0022 for 1989 and at
  most 1.0009 earlier. The 1.0% to 1.5% first measured at 50,000
  simulations overstated it; the gap is within two Monte Carlo standard
  errors.
* **Uncentred residuals do not reconcile.** With EVW's pool as it is, the
  pool's mean (RAA 0.14) biases every pseudo factor and the bias compounds
  over an origin's remaining factors: total mean reserve 1.17 (RAA), 1.007
  (GenIns) and 0.992 (ABC) times the chain ladder's, and RAA's standard
  deviation 1.09 times Mack's (it grows with the mean); RAA 1990's mean is
  1.28 times.
* **Other weightings.** Centred, `alpha = 0` (simple average) and
  `alpha = 2` (regression) agree with R's `MackChainLadder(alpha = ...)`
  on GenIns and ABC within 3.5 standard errors at 50,000, but RAA's
  youngest origins come out up to 7.4% above (`alpha = 0`, 1990; its
  variance is proportional to `C^2`, so the linear approximation fails
  sooner) and 1983 and 1984 1.5% and 1.8% below (`alpha = 2`). With
  `alpha = 0` the residuals of each factor sum to zero unweighted, so the
  pool's mean is zero and centring changes nothing. Measured once, not in
  CI.

# EVW's Table 4 points to centred residuals

EVW's Table 4 (Taylor–Ashe, i.e. GenIns, Mack's rule for the last sigma,
500,000 simulations) gives expected reserves within Monte Carlo error of
the chain ladder's: total 18,684,738 against 18,680,856 (+0.02%), and no
origin more than two of their standard errors away.[^evw] The
centred bootstrap matches every origin's expected reserve and standard
deviation within five combined standard errors (20,000 simulations: total
18,703,619 and 2,458,884 against their 2,448,700). The uncentred one, which
is what their Appendix 1 describes, puts the total at 18,816,241, eleven
combined standard errors above theirs. So their implementation most
likely resampled zero-mean residuals, by centring or otherwise; the paper
does not say. An inference. Their one-year standard deviations (Table 4)
are matched either way, since GenIns's pool mean is only 0.0135 and the
standard deviation barely moves; the bias shows in the mean, which
compounds over the lifetime but not over one year.[^test][^evw]

`MackBootstrap::centre_residuals` is therefore on by default (decision 8,
since branch claude/mack-centre-default), in Rust, Python and R: centred,
the bootstrap matches the chain ladder and EVW's Table 4 in the mean and
Mack and Merz-Wuthrich in the standard deviation, while uncentred the
mean reserve is off by +17% (RAA), +0.7% (GenIns) and -0.8% (ABC) and the
one-year standard deviation up to 1.3% wide on RAA. `centre_residuals:
false` keeps EVW's Appendix 1 as written. The validation tests now run
the default; the uncentred total against Table 4 is checked
explicitly.[^test]

# Performance

Until 2026-10-08 `Gamma::sample` inverted the Gamma cdf, which was slow at
large shapes. A late cell's shape `f^2 C / sigma^2` runs to the thousands
on ABC, so ABC's lifetime view at 2,000 simulations took 30 s in a debug
build, against about 1 s for RAA or GenIns, and the validation test
simulated ABC with the lognormal of the same mean and variance. Since the
sampler is Marsaglia and Tsang's ([the Gamma sampler](gamma-sampler.md)),
ABC's lifetime view with the Gamma at 20,000 simulations takes 0.02 s in
a release build, against 39 s before, and the validation test runs the
Gamma on all three triangles under both rules for the last sigma: every
origin's and the total standard deviation within 2.4 Monte Carlo standard
errors of Mack's (ABC total 1.011 and 1.010 times), the mean within 1.9 of
the chain ladder's.[^test] The lognormal is still unsuitable for RAA: at
shapes below one its heavy tail makes the sample kurtosis, and so the
standard deviation's estimated standard error, unreliable (RAA 1990 came
out 6% low, 3.3 estimated standard errors, at 20,000; 0.8% low at
200,000).

# Unit-test facts

* An origin with one cell left draws the same value in the lifetime and
  one-year views from the same seed (same pseudo factors, then its cell
  first), so RAA 1982's lifetime reserve plus its one-year CDR is its
  opening reserve in every simulation.[^unit]
* An exact pattern (every sigma zero) runs off to the chain ladder's
  reserve in every simulation, for the Gamma and the residuals
  process.[^unit]

[^test]: validation/tests/reserving_mack_bootstrap.rs
[^unit]: crates/prospicio-reserving/src/mack_bootstrap.rs, unit tests
[^evw]: England, Verrall and Wuthrich (2019), Tables 2 and 4 and Appendix 1
