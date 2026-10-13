---
type: Reference
title: Mildenhall's aggregate package, Pricing Insurance Risk and CAS Monograph 15
description: The Python aggregate package (1.0.1) as the reference for distortions, calibration, natural allocation and contract terms; where the Pricing Insurance Risk and Monograph 15 examples live, and the API and tolerance quirks met reproducing them.
resource: https://github.com/mynl/aggregate
tags: [pricing, risk-measures, distortions, capital-allocation, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T00:45:00Z }
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
  - id: terms
    resource: ../validation/scripts/aggregate_contract_terms.py
    title: aggregate_contract_terms.py
  - id: oep
    resource: ../validation/scripts/aggregate_oep.py
    title: aggregate_oep.py
  - id: moments
    resource: ../validation/scripts/aggregate_moments.py
    title: aggregate_moments.py
  - id: reins
    resource: ../validation/scripts/aggregate_reinsurance.py
    title: aggregate_reinsurance.py
  - id: natural
    resource: ../validation/scripts/aggregate_natural.py
    title: aggregate_natural.py
  - id: ruin
    resource: ../validation/scripts/aggregate_ruin.py
    title: aggregate_ruin.py
  - id: picks
    resource: ../validation/scripts/aggregate_picks.py
    title: aggregate_picks.py
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
    by unit, and refuses a portfolio built from a sample (below).
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
* **Contract terms** live in `aggregate.contract_terms` (`SwingTerms`,
  `RetroTerms`, `SlideTerms`, `ProfitCommissionTerms`, `CorridorTerms`,
  `ReinstatementTerms`), each a map `phi` of one loss quantity; DecL
  writes them `swing basic b lcm m [min lo] [max hi]`, `slide c1 at lr1
  and c2 at lr2 ...`, `pc <share> after <allowance>` and `corridor`.[^terms]
  * Swing and retro are `clip(basic + lcm x, minimum, maximum)`, the
    minimum defaulting to `basic` and the maximum to no cap. For swing,
    `x` is the layer's ceded loss at the placed share, and the
    underwriter (`Underwriter._make_feature_terms`) scales `basic`, the
    minimum and the maximum by the share but not `lcm`.
  * A sliding scale's anchors are `(commission, loss_ratio)` pairs,
    interpolated with flat ends (`numpy.interp`); commissions must not rise
    with the loss ratio. The profit commission is `share (1 - LR -
    allowance)+` of premium: it does not deduct the ceding commission, so
    a contract that does puts it in the allowance.
  * The loss ratio is ceded loss over a fixed ceded premium (`deposit`,
    `rol` or `rate`), so slide, pc and corridor need one and refuse swing;
    a program carries at most one variable feature and none with
    reinstatements; `cede` (a flat commission) needs a premium and
    cannot sit with `slide`. `deposit` and `rol` are quoted for 100% of
    the layer (`share x amount`, `share x rol x limit`).
  * Its loss-ratio corridor (`corridor <share> po <width> xs <attach>`)
    acts on the ceded loss ratio after the share; prospicio's
    `Layer::loss_corridor` acts on the layer loss at 100% after the
    annual deductible and before the annual limit, so they agree only
    without an annual limit.
  * prospicio matches `phi` for swing, retro, slide and pc at 475 losses
    through every kink to `1e-12` relative.[^terms]

# The occurrence PML, and what 1.0.1 does not have

* `aggregate.utilities.oep(agg, p, freq=0)` is closed form from the
  severity and the expected count: the loss at the severity quantile
  `1 + ln(1 − p) / λ`, with `S_sev`, `F_sev`, the occurrence return period
  `1 / (λ S_sev)` and the annual one `1 / p`. It never reads the
  aggregate's grid, so `bs` and `log2` do not matter, and it refuses
  non-Poisson and zero-modified counts. prospicio's `occurrence_pml` takes
  any count through its pgf and matches `oep` on three Poisson books at
  `1e-10`.[^oep]
* Found while listing the gaps against 1.0.1's source (2026-10-12):
  `Aggregate.update` has no `tilt` option; `sev_calc='moment'` raises
  `NotImplementedError` (prospicio's `Grid::local_moment` goes further);
  the FFT engine never uses Panjer's recursion; and point-by-point
  densities need `normalize=False`, because the default renormalizes the
  severity cut at the grid's top and thins the tail.

# Compound moments and approximate

* `Aggregate.stats_df['mixed']` holds the analytic `ex1` to `ex3`, mean,
  cv and skewness of frequency, severity and aggregate, from raw moments.
  Against 30-digit closed forms it is 6.5e-10 off a Weibull severity's
  mean and 2e-9 off the skewness (its severity moments are numerical), and
  2.5e-10 off the skewness at a Poisson mean of 1,000 (raw moments
  converted to central ones cancel).[^moments]
* `Aggregate.approximate(kind)` fits `stats_df['empirical']`, the moments
  of the computed grid, when the aggregate has been updated, not the
  analytic ones; on a gamma book at `bs=1` its normal fit's location is
  500.000005 for a mean of 500. The formulas are
  `_fits.approximate_from_mcvsk`, which takes `(m, cv, skew)` directly; the
  scipy parameters are `(shape, loc, scale)`, the lognormal's `shape` being
  `σ` and `scale` `e^μ`.[^moments]

# Reinsurance statistics and prices

* The reinsurance frames live on the `Aggregate`: `reins_stats_df`
  (statistics by view, `occ` and `agg` blocks), `reins_price_df(distortion)`
  (`L`, `M`, `P`, `LR` and capital columns per view; it needs a distortion
  or `calibrate_distortions` first) and `reins_density_df`, the densities
  (`p_sev_gross`, `p_agg_gross`, `p_agg_ceded_occ`, `p_agg_net_occ`,
  `p_agg_ceded`, `p_agg_net`), not `density_df`. DecL places a share as
  `200 po 400 xs 100` (part of), not `so`.[^reins]
* `reins_stats_df`'s layer columns are conditional on a loss to the layer
  (frequency `E[N] pr_attach`, severity over `pr_attach`); attach is
  exclusive, detach inclusive. Its `agg` rows are the moments of the FFT
  density, not analytic: against prospicio's cumulants on the same grid
  they are 1e-9 off the mean and cv and up to 1.3e-7 off the skewness of a
  rare layer. Its gross severity skewness is 4e-11 to 4e-10 off the
  grid's.[^reins]
* A layer at a share below 1 is mapped onto the grid with the share
  applied, so a ceded 0.5 splits between 0 and 1: its `pr_loss` for 50% of
  400 xs 100 is 0.4% below `1 − G_N(F(100))`, and its cv and skewness are
  smeared (1e-5 to 1e-4). prospicio scales the full layer's cumulants
  exactly.[^reins]
* Distortion prices are sensitive to the far tail: the proportional
  hazard at 0.7 raises FFT round-off of `1e-16` enough to move `P` by
  about 1e-8 relative, and `aggregate`'s own `1 − cumsum(p)` and sum from
  the top differ by as much. With `normalize=False`, prospicio's
  `Tower::on_grid` matches its densities at `1e-14`.[^reins]

# The priority ladder

* `Portfolio.priority_epd_df(assets=)` (or `p=`) and
  `priority_kappa(unit, assets=)` need the independent combine of the
  units: a portfolio from `create_from_sample` (InsCo) raises `ValueError`
  ("the units' transforms do not reproduce density_df p_total"). So the
  ladder is checked on `build`'s independent books, the Discrete case and
  a three-unit one.[^natural]
* The asset level is snapped to the nearest grid point
  (`.attrs['assets']`); `mean` is the represented mean of the grid, not the
  declared one; the `total` group repeats `E[X ∧ a]` under all three
  rules.[^natural]
* `priority_kappa` returns `(junior, senior, p_total)` on the whole grid,
  `NaN` where `p_total` is below machine epsilon, and FFT noise of about
  `5e-15` (including small negatives) where the exact value is 0.[^natural]
* Senior and junior are two-tier readings (unit against the rest pooled);
  1.0.1 has no ordered k-tier schedule.[^natural]

# Eventual ruin

* `Aggregate.pollaczeck_khinchine(rho, cap=, excess=, stop_loss=,
  kind=)` returns `(ruin, find_u, mean, density)`: `ruin` a Series of
  `ψ(u)` on the severity grid's index, `density` the equilibrium severity
  `bs S(x) / mean` with `S` strictly above `x`. It needs a Poisson
  frequency; the rate cancels.[^ruin]
* `ψ` is one FFT of twice the grid with wrapping; on the books checked
  (2^16 points) it agrees with a compound FFT at four and eight times the
  grid to `1e-12`, so the wrap is negligible there.[^ruin]
* `find_u(p)` with `kind='index'` is the smallest grid point with
  `ψ < p` (strict); `kind='interp'` interpolates linearly between the
  bracketing points.[^ruin]
* `eventual_ruin` reports the Lundberg exponent from the private
  `_lundberg_exponent(p_sev, xs, rho)`, by brentq on `(1e-12, 700 / max x)`
  of the whole severity (not the capped one), and a 1,000-path simulated
  check of the same discretized model.[^ruin]

# Layer loss picks

* `Aggregate.picks(attachments, layer_loss_picks)` returns the adjusted
  severity probabilities (`_picks_work`, with `debug=True` a namedtuple of
  `layers`, `exact`, `density`, `audit`); the picks are annual, divided by
  the aggregate's `n`. Attachments off the grid raise a `ValueError` naming
  a compatible bucket.[^picks]
* The adjusted survival is `min(1, ω + (S − S(a)) w)` per layer, bins
  right-closed with 0 in the bottom one; the layer integrals are left sums
  `bs Σ S` over `[a_{i−1}, aᵢ)`. Continuity at the attachments makes the
  two conventions agree.[^picks]
* When the cap at 1 makes the bottom layer miss its pick by more than
  `1e-3` (per claim) it rebuilds the bottom layer as 1 then a straight line,
  with the line's length rounded to the grid, and warns "Revised layer 1
  achieved": 704.90 for a pick of 700 on a gamma (mean 100, cv 1.5) with
  10 claims and a step of 1. A smaller miss is kept silently.[^picks]
* Infeasible picks (a negative weight) only warn, and the returned
  probabilities are negative.[^picks]

[^terms]: aggregate_contract_terms.py
[^moments]: aggregate_moments.py
[^oep]: aggregate_oep.py
[^reins]: aggregate_reinsurance.py
[^natural]: aggregate_natural.py
[^ruin]: aggregate_ruin.py
[^picks]: aggregate_picks.py
