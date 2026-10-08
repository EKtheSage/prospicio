---
type: Reference
title: Mildenhall's aggregate package, Pricing Insurance Risk and CAS Monograph 15
description: The Python aggregate package (1.0.1) as the reference for distortions, calibration and natural allocation; where the Pricing Insurance Risk and Monograph 15 examples live, and the API and tolerance quirks met reproducing them.
resource: https://github.com/mynl/aggregate
tags: [pricing, risk-measures, distortions, capital-allocation, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-08T15:00:00Z }
sources:
  - id: aggregate
    resource: https://github.com/mynl/aggregate
    title: mynl/aggregate, Stephen Mildenhall's aggregate distribution and pricing package (1.0.1, 2026-10-07)
  - id: monograph
    resource: https://github.com/casact/capital-modeling
    title: casact/capital-modeling, the Quarto source of CAS Monograph 15 (Major and Mildenhall, Introduction to Capital Modeling and Portfolio Management, 2026)
  - id: pir
    resource: https://pypi.org/project/aggregate/0.30.1/
    title: aggregate 0.30.1 on PyPI, whose extensions/ reproduce the exhibits of Pricing Insurance Risk (Mildenhall and Major, Wiley 2022)
  - id: script
    resource: ../validation/scripts/aggregate_distortions.py
    title: aggregate_distortions.py
---

# Which source for what

* **aggregate 1.0.1** (Python 3.12 to 3.14) is the reference for the
  distortions, their calibration and the natural allocation: `Distortion`,
  `Portfolio.calibrate_distortions`, `Portfolio.price(p, d, allocation=)`.[^aggregate]
* **Pricing Insurance Risk** exhibits do not reproduce on 1.0: its
  changelog says much of their machinery was removed in the 1.0 alphas.
  They need **aggregate 0.30.1** in its own environment, whose
  `extensions/` hold the case studies (`case_studies.py`, `discrete.py`,
  `tame.py`, `cnc.py`, `hs.py`), the Bodoff exhibit and the figure code.[^pir]
* **CAS Monograph 15** is free, and its full Quarto source, with the
  scripts behind its tables (`basic_example_script.py`, `070_script.py`),
  is on GitHub. The CAS site itself was blocked from the cloud
  environment; GitHub and PyPI were not.[^monograph]

# Quirks

* The monograph's scripts call `calibrate_distortions(Ps=[1], COCs=[.15])`,
  an older signature; 1.0.1 takes `calibrate_distortions(0.15, p=1)` (the
  return first, then the capital standard `p` or the assets `a`).[^monograph]
* `Distortion` takes its parameters by keyword: `Distortion('ccoc', r=)`,
  `('ph', a=)`, `('wang', lam=)`, `('dual', b=)`, `('tvar', p=)`,
  `('bitvar', p0=, p1=, w1=)` with `w1` the weight on `TVaR(p1)`,
  `('clin', r0=, slope=)`, `('cll', r0=, b=)`, `('lep', r0=, r=)`,
  `('ly', r0=, r=)`, `('beta', a=, b=)`, `('wtdtvar', ps=, wts=)`.
  `Distortion.price(ser, a=, kind='ask')` returns a `Price` named tuple;
  the asset level must be one of the series' outcomes.
* Calibration stops Newton at a premium error, reported as `error` in
  `distortion_df`: on InsCo (`P = 53.565`, ι = 15%, `p = 1`) it is
  `3.3e-10` for PH, `1.3e-8` for Wang, `3.4e-7` for dual and `7.6e-6` for
  TVaR. prospicio's bisection reaches the premium to rounding, so its
  parameters agree with aggregate's only within that error over the
  price's slope (`3.2e-6` in TVaR's `p`).[^script]
* InsCo's calibrated parameters at 15% and `p = 1`: CCoC `r = 0.15`, PH
  `0.720479`, Wang `0.342731`, dual `1.595151`, TVaR `0.271287`; the
  monograph's totals are 22, 28, 36, 40 (four times), 55, 65 and 100.
* Natural allocation (`Portfolio.price(p, d, allocation=)`) reproduces in
  closed form; prospicio matches it at `1e-10` on InsCo and the PIR
  Discrete case. Details that matter for a port:
  * `price(p)` reads `p > 1` as an asset level, otherwise the lower `p`
    quantile.
  * Capital is allocated layer by layer with ratio `(1 - g) / (g - S)`;
    where `g(S) = 1` it uses `g'(1) / (1 - g'(1))`, and 0 when `g'(1)` is
    NaN (Wang).
  * Under the linear allocation the unit margin jumps at each total; on
    aggregate's unit grid the jump lands in the layer below the total.
  * A sample-built portfolio's tied totals take the unweighted mean of
    the tied rows (`np.mean`), not the probability-weighted one.
  * `S` at the largest total must be exactly 0: aggregate gets 0 from
    `max(1 - cumsum(p), 0)`; a residue of `1e-16` there would add the
    CCoC mass times the maximum to the price.
  * 1.0.1 has no `epd` kind in `var_dict`; `priority_epd_df` gives EPD
    by unit, and refuses a portfolio built from a sample.
* `AllocationBounds(port, a=).bounds([P])` takes its BiTVaR knots from the
  cumulative probabilities of `X ∧ a`; a level inside the atom at `a`
  would price the total the same but split the linear allocation
  differently, and is not searched.
* `pedagogy.ClassicalPremium`'s Fischer principle reads `self.p`, which
  nothing sets, so it raises unless the caller sets it. Its semi-variance
  principle adds `θ E[(X - μ)₊²]`, not the root.
* Counts: `build('agg A n claims dsev [1] <frequency>')` gives the count's
  distribution as the aggregate's. `poisson zm p0` fixes the *realized*
  mean (the base mean is solved); a trailing `!` fixes the base mean
  instead. `mixed delaporte cv c` and `mixed sig cv c` take the cv of the
  whole mixing variable, fixed part `c` included. `neymana θ` with `n`
  claims has `n / θ` clusters. `pascal cv k` came out with mean 6.0104 for
  6 claims on a 1024-point grid, so it was left out of the parity.
* **Severities.** aggregate takes any SciPy continuous family by name,
  so its Burr is `burr12(c, d)` (`burr` is Burr III), the inverse Gaussian
  is `invgauss(mu, scale)` with mean `mu * scale` and shape `scale`, and
  the inverse gamma is `invgamma(a, scale)`. `sev_lb` and `sev_ub` with
  the default `sev_conditional=True` condition the severity on
  `lb < X <= ub` (the cdf is rescaled by `F(ub) - F(lb)`), and a splice
  is a mixture of such pieces; prospicio's `Truncated` and
  `Truncated::splice` follow that. SciPy 1.18 matches prospicio's new
  families to `1e-10` or better on quantiles (taken by `isf` above the
  median) and `1e-11` on the cdf.
* **Grid sizing.** With no `bs`, `Aggregate.update` sizes the grid at
  `log2 = 16` from three-moment shifted lognormal and gamma fits at
  `bucket_sizing_p = 0.99999`, floors it by a single big jump
  `ES - mu_X + q_X(1 - (1 - p*)/E[N])` with `p* = 1 - 1e-12` and the
  severity tail floored at `1e-14`, and rounds the bucket up on
  `round_bucket`'s ladder `{1, 2, 4, 5, 8} × 10^k` (powers of two below
  1). Its `q_X` there is a method-of-moments quantile, not the
  severity's (a Lomax with alpha 2.5 gets 0.58m where the true quantile is
  20.9m), and it refuses an infinite-variance book without an explicit
  `bs` (`InfiniteVarianceError`). `est_m` is the grid's mean, not the
  exact one (49,970.5 for a book whose mean is 50,000).
* **`normalize=True` thins the tail.** By default `aggregate` rescales the
  severity cut at the grid top instead of keeping that mass: on a Poisson
  20 Lomax (alpha 2.5, scale 100) book at bucket 1 the 0.9999 quantile is
  14,401 where finer and longer grids converge on 14,494.5;
  `normalize=False` gives 14,495, which is what prospicio's rounding
  (lumping the mass on the last point) gives.
