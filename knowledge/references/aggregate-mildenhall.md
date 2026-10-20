---
type: Reference
title: Mildenhall's aggregate package, Pricing Insurance Risk and CAS Monograph 15
description: The Python aggregate package (1.0.1) as the reference for distortions, calibration, natural allocation and contract terms; where the Pricing Insurance Risk and Monograph 15 examples live, and the API and tolerance quirks met reproducing them.
resource: https://github.com/mynl/aggregate
tags: [pricing, risk-measures, distortions, capital-allocation, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T07:30:00Z }
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
  - id: bounds
    resource: ../validation/scripts/aggregate_bounds.py
    title: aggregate_bounds.py
  - id: counts
    resource: ../validation/scripts/aggregate_counts.py
    title: aggregate_counts.py
  - id: clash
    resource: ../validation/scripts/aggregate_clash.py
    title: aggregate_clash.py
  - id: renewal
    resource: ../validation/scripts/aggregate_renewal.py
    title: aggregate_renewal.py
  - id: pnl
    resource: ../docs/design/pnl.md
    title: docs/design/pnl.md (the P&L ledger's design note and its reference cases)
  - id: pirguide
    resource: ../docs/guides/pricing-insurance-risk-case-studies.md
    title: docs/guides/pricing-insurance-risk-case-studies.md (the PIR case studies run in prospicio beside aggregate 0.30.1)
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
  * `aggregate` 1.2.0 (installed from PyPI on 2026-10-20) reproduces all
    1,581 values of `validation/reference/natural_aggregate.csv` written by
    1.0.1: its natural allocation, bottom layer included, did not change.
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
* **`sharpen` (1.2.0) audits a grid after the FFT.** Each cell scores six
  terms `|emp/exact - 1| / (k eps)` with `k` = 1, 10, 100 for the mean,
  cv and skewness of the severity and of the aggregate (`eps = 1e-4`),
  combined by a power mean with `p = 2`; a term with an exact value at or
  below `1e-12` in magnitude, or not finite, is skipped. From the center it
  doubles `bs` until the score stops improving, then halves, at most four
  times each way, on rows `d_log2 = 0` and `-1`; it takes the best score
  among cells with no `pmf_deficit` (`1 - sum(p)` above `1e-12`), scores
  within 1.25 of the best counting as tied and ties going to the smaller
  `log2`, then the smaller move, and searches `d_log2 = +1` only when that
  misses `good_enough = 0.5`. Its `sharpen_df` is indexed by
  `(d_bs, d_log2)`. On `100 claims 1000 xs 0 sev lognorm 100 cv 1.5
  poisson` from `bs = 1`, `log2 = 13` it moves to `bs = 4` (score 0.0353);
  on `50 claims sev gamma 10 cv 0.5 poisson` from 0.5 and 10 it grows to
  `bs = 1`, `log2 = 11` (score 0.722, nothing sound reaching the target).
  prospicio's `audit_grid` scores the cells that hold all their mass to
  aggregate's six digits and selects the same cell; the cells that lose
  mass agree on that mass (2.5e-4 at `bs = 2` above) but not on their
  scores, which depend on where each puts what falls off the grid; they
  are never chosen while a sound cell exists, and the walk through them
  can differ by a cell (`validation/scripts/aggregate_sharpen.py`).
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

# Total cession and aggregate covers

* With several occurrence layers `reins_price_df`'s `ceded` row and
  `reins_stats_df`'s `Ceded` column are their total cession. The `Ceded`
  column's `freq` is the ground-up count and its `sev` the ceded loss per
  ground-up claim (zeros included); `pr_attach` and `pr_loss` are at the
  lowest attachment, `lol` is over the summed limits. Its per-claim cv and
  skewness come off the ceded severity density, 1.5e-11 and 2.6e-9 off the
  points' moments.[^reins]
* An aggregate cover's statistics are the `agg` block's `layer.1`:
  `pr_attach = pr_loss = P(S > a)`, `pr_detach = P(S ≥ a + l)`, read off
  the annual FFT density (3e-14 absolute round-off; a rare cover's skew of
  32 is 1.4e-6 off the grid's exact moments). In DecL, `occurrence net
  of` puts the aggregate cover on the occurrence net, `occurrence ceded to`
  on the occurrence cession.[^reins]
* The reference books' severity is limited (`1000 xs 0`, rounded onto
  `cap + 1` points of step 1) while the annual total runs on `2^14`. A
  caller that rounds the severity and compounds it on one length (the
  XLL's `PRO.AGG.COVER`, `step` and `points`) reproduces the book by one
  more occurrence layer, unlimited above the cap at 100%: the net is then
  the capped severity's exactly, and the `agg` block's statistics match
  at the reference's tolerances.[^reins]

# Pricing bounds

* `aggregate.bounds.PricingBounds(x, ys, a=)` takes pmf Series (or
  Aggregates, Portfolios); `.bounds(P)` is a frame on `(P, risk)` with
  `lower`, `upper`, `width`; `.bitvars(P)` the attaining BiTVaRs. P must be
  in `[E[X], max X]` of the capped X.[^bounds]
* Its vertices are `p = 0` and the breakpoints below 1 (with tail at
  least `s_floor = 1e-14`), deduplicated on "first vertex attaining each
  `TVaR_p(X)`". So where X's top atom is larger than Y's, the `p → 1`
  vertex `(max X, max Y)` is lost and the upper bound is too low: InsCo's
  total and X3 capped at 65, at P = 65, give 52 where `TVaR_1` attains
  64.[^bounds]
* `Bounds(x, P)` searches 256 levels (refined near `p*`, found by brentq to
  `xtol = 2^-17`) and an s-grid of 513; its `max_envelope` interpolates
  linearly in s and can exceed the exact envelope by 2.4e-4 at a kink (s =
  0.1 on InsCo).[^bounds]
* `GridDistribution` lives in `aggregate._grid_distribution`, not the
  package root.[^bounds]

# TVaR thresholds and named counts

* `tvar_threshold(p, kind)` (on `GridDistribution`, and `Portfolio`
  through its total) bisects `TVaR(t) − VaR(p)` on `[0, 1]` with scipy's
  `bisect` (`xtol = 2e-12`, so 0.859375 comes back as 0.859375000001819).
  It raises when the VaR is at or below the mean (no sign change; at
  `p = 0` it returns 0) and returns 1 when the VaR is the largest value,
  where every `t` past the last value's cumulative start
  qualifies.[^natural]
* Its grid TVaR is the coherent `(1/(1 − p)) ∫_p^1 VaR_u du`, returning the
  largest point once `p` reaches the last point's cumulative
  start.[^natural]
* `fixed` is `N = n` with pgf `zⁿ` (meaningless for a fractional `n`);
  `bernoulli` reads the mean `n < 1` as the chance of one claim;
  `geometric` is on `0, 1, …` with `P(N = 0) = 1 / (1 + n)`, a negative
  binomial with `r = 1`.[^counts]

# Stand-alone prices, the comonotonic allocation, reinstatements

* `Portfolio.price_stand_alone(dist, p)` prices each unit with
  `Aggregate.price(p, dist)`: assets at the unit's own `q(p)`, `L` and `P`
  read from `density_df`'s `exa` and `exag` there; `total` comes from
  `pricing_at`, and `sum` adds `L, M, P, Q` and re-derives the
  ratios.[^natural]
* `make_comonotonic_allocations` runs Denuit et al.'s (2025) algorithm on
  every grid point up to `q(1)`, including points with zero probability,
  where `exeqa_` is 0 and the result shows odd values (−2.15 for X1 at a
  total of 7 in the three-unit Discrete case). Run on the positive-mass
  totals only, the values there are the same to 3e-14.[^natural]
* Reinstatement premiums are priced only inside the P&L builders
  (`_pnl_builders.build_reinstatement_source`), from
  `ReinstatementTerms` (pro rata as to amount, not as to time) on the
  unlimited occurrence recovery; a DecL build without `pnl` leaves no
  `reinstatement_terms` on the object, and `reins_stats_df` ignores the
  clause. `E[h(R)]` on `reins_density_df`'s `p_agg_ceded_occ` gives the
  expected reinstatement premium.[^reins]

# Clash and the bivariate aggregate

* `clash NAME na nb nc claims <layers sev A> <layers sev B> [freq]` builds a
  `aggregate.bivariate.BivariateAggregate` (no frequency clause: Poisson;
  `mixed gamma c` gives a negative binomial with mixing cv `c`). Its
  `clash` attribute holds `na, nb, nc, n0, pa, pb`; `update(bs=(h_A, h_B),
  log2=(l_A, l_B))` pins each axis (a scalar `log2` is a total budget split
  by measured support), with `padding=1` doubling each axis for the FFT.
  `density` is the truncated joint (row for A), `marginals` the two sums,
  `total` a `GridDistribution` property (`x`, `p`), `moments(k)` the raw
  mixed moments, `corr` a property.[^clash]
* `update_work` zeroes cells below `1e-15` (`_clip_density_fuzz`); patch
  `aggregate.bivariate._clip_density_fuzz` to a no-op for targets that
  keep the tail mass (see the finding on clash parity).[^clash]
* Copula coupling (`bivariate` with a `Copula`: independent, normal,
  Gumbel, Clayton, FGM, shuffles) builds the per-event joint as the
  discrete-Sklar rectangle mass `Copula.rectangle_pmf(G1, G2)`; `pnl`
  axes are rejected there.[^clash]

# Renewal counts

* `agg NAME T years sev ... wait LAW` (or `dwait [x] [p]`, with `!` for a
  defective wait) builds a `FrequencyRenewal` in `aggregate._renewal`; the
  count is `frequency.freq_df.p`, the grid `wait_bs`, `wait_log2`,
  `wait_lattice`, with `wait_p0`, `wait_defect` and `kmax` (the count of
  positive waits, before the zero waits go back). `build(..., update=False)`
  is enough for the count. `wiener_hopf(rho)` is the renewal ruin, beside
  `pollaczeck_khinchine`. The suite's cases are the `Y.Renewal.*` lines of
  `agg/_test_suite.agg`.[^renewal]

# The P&L ledger

* `PnL` (`_pnl.py`) evaluates every leg at each atom of its source and
  groups the signed values into a `GridDistribution` with irregular,
  possibly negative support; it never needs a signed FFT grid. The source's
  probabilities are rescaled to sum to 1, so a ledger over an FFT density
  with mass `1 − 2e-5` shows the rescaled means.[^pnl]
* The scenario columns (`κ`) slice the atoms whose grand result equals its
  lower quantile by exact float equality (`_scenario_ladder`); the walk
  (`xpnl`) and consolidated (`pnl`) faces of one program differ in the last
  bit (`3.1` against `3.0999999999999996`, a cumulative `0.75` against
  `0.7499999999999999`) and so in `κ75`.[^pnl]
* `GridDistribution`'s `tvar` is the upper tail even on a payoff, so on a
  P&L row it sits near the mean; `return_period` is the only accessor that
  reads the payoff sign (`T = 1 / p`). Its `make_var_tvar` reads equal
  masses as `k / n` exactly (equal meaning within `2⁻⁵³ + 1e-5`
  relative, `np.isclose`'s default `rtol`) and builds on the atoms with
  mass only; `tvar` on a lone atom raises an `IndexError` (it reads
  `F[-2]`). `prospicio_prob::Discrete` matches it on 544 values
  (`aggregate_discrete.py`).[^pnl]
* `PnL` divides the source's probabilities by their total once and its
  rows' distributions keep them as they are: dividing again (even by a
  total of `0.9999999999999999`) moves a cumulative that equals a ladder
  level, as 0.1 on four outcomes of 0.4, 0.3, 0.2, 0.1, across the
  quantile's jump. `prospicio_aggregate::pnl::Ledger` matches the kernel
  on 1,704 values (`aggregate_pnl_ledger.py`).[^pnl]
* A leg's `bs` puts only that row's own distribution on a regular grid
  (a linear, mean-keeping scatter from `floor(lo / bs) bs`; a constant
  gets a span of `(|lo| + 1) 1e-9`): `economic_marginal_df` reads the
  grid, while `summary_df`'s quantiles, the moments and `κ` stay on the
  exact values. A `PnL` built with `role=` names its one group after the
  ledger, the label of its `economic_ratios_df` row. Rebucketed legs, the
  ratios and `stack_marginal_pnls` match on 324 values
  (`aggregate_pnl_ratios.py`).[^pnl]
* In the builders' DecL, `deposit 1 cede 10%` on a cover is a premium of
  1 and a 10% ceding commission on it, not a 10% share. A DecL ledger's
  source is the engine's whole grid, zeros included, and `PnL` normalizes
  by numpy's pairwise sum over it, which can differ in the last bit from
  a sum over the atoms with mass alone: on example 1 one total is exactly
  1 and the other `0.9999999999999999`, and `κ75` moves by an atom. The
  consolidated face (`pnl`) reads the net marginal off
  `reins_density_df`, rebucketed onto the grid and renormalized, so its
  `κ75` on example 3b sits across the same kind of tie. The walk names a
  tier of several layers after its first declared label, and a peel
  skips zero-share layers. `prospicio_aggregate::pnl::Book` matches the
  builders on 2,401 values (`aggregate_pnl_builders.py`).[^pnl]
* The occurrence walk runs on `occ_bivariate(views=('gross', 'ceded'))`:
  each claim goes to `(x, c(x))` on the exact common lattice when the
  budget affords it (`bs` 1, 64 × 32 cells for example 4), compounded by
  a 2-D FFT, with cells below `1e-15` of the total mass zeroed
  (`_clip_density_fuzz`); its total stays `0.9999999999999893` and `PnL`
  divides by it. An aggregate cover behind it recovers on
  `max(l − c, 0)`, and a one-group span followed by another group earns
  only its `Net of occurrence` block. The consolidated face reads its net
  marginal off the 1-D engine, about `2e-12` from the joint's in its
  standard deviation. In DecL the count goes between the two covers:
  `occurrence net of ... poisson aggregate net of ...`.
  `Book::occurrence_ledger` matches on 1,566 values
  (`aggregate_pnl_occurrence.py`).[^pnl]
* Variable features (`swing basic b lcm m [min] [max]`, `slide c at lr
  and ...`, `pc s after a`, `corridor s po w xs a`) decorate one aggregate
  layer, which the engine then drops, so the walk's feature step is the
  generic `ceded agg` and its commission legs `sliding commission` and
  `profit commission`. The corridor works on the ceded loss ratio:
  `LR − s clip(LR − a, 0, w)` times the ceded premium. `retro <collar>
  premium` is `pnl` only (`xpnl` refuses it, and any reinsurance with
  it), its premium expenses taken on the collar's basic. Partial
  placement is spelled `1.5 po 3 xs 5`. On the consolidated face the
  variable builder books a loss-basis expense stochastic on the gross,
  the guaranteed-cost builder at its expectation. `pnl_program`'s ladder
  (`_pnl_technical`) rounds each deposit at 100% placement before the
  premium, both half to even (whole units above 100, cents below).
  `premium_style='rate'` respells each priced deposit as
  `f"{d / base:.8g}"`, the aggregate tier's base the booked premium less
  the occurrence premiums as resolved; an engine stating its own premium
  (`premium at lr`) replaces `E_net / c_net`, and with no cession the
  program says `derive premium`.
  prospicio matches on 2,342 values (`aggregate_pnl_variable.py`).[^pnl]
* Reinstatements decorate a single occurrence layer (`deposit 2
  reinstatements 1 at 100%`, `1 free and 1 at 50%`, `no reinstatements`;
  a base premium clause is required). The ledger runs on the `(L, R)`
  joint, `R` the layer's unlimited annual cession, with the recovery
  `A(R)` capped at the annual limit and the premium `D + h(R)`, `h` pro
  rata as to amount; an aggregate cover behind it recovers on
  `max(L − A(R), 0)`. Both faces are over the joint (the consolidated
  one books a loss-basis expense stochastic on `L`), and `xpnl ... peel`
  is refused. `Book::occurrence_ledger` matches on 1,121 values
  (`aggregate_pnl_reinstatements.py`).[^pnl]
* With two or more occurrence layers, `xpnl ... peel` takes the
  stitched route (`_peel_stitched`): every row is `sign × X + shift` of
  its own marginal, a compound of a per-claim map of the severity (one
  FFT on the padded grid, truncated and normalized by its mass) or a
  pushforward of the retained aggregate; the `Impact` row is the
  differences of the grand and gross results' statistics (its SD can be
  negative), and a loss-basis expense is constant. Its `κ` columns are a
  Palm ladder conditioned through the claims, or through the retained
  aggregate once an aggregate cover exists, so they depend on the
  program (the gross loss row's `κ01` reads −16 on one and −11.55 on
  another). `Book::occurrence_peel` matches the 495 other values
  (`aggregate_pnl_stitched.py`).[^pnl]
* `PnL.stats_df` is the engine's, not the ledger's; the ledger's sheet is
  `economic_df`. `pnl_program` returns DecL text.[^pnl]

# The Pricing Insurance Risk case studies in 0.30.1

* `extensions/case_studies.py` prices each case at a 10% return on
  capital (`roe=0.10`) with assets at `reg_p`: 1 for Discrete, 0.999 for
  Cat/Non-Cat and Hu/SCS, and 0.9999 in `tame.py` for Tame. The grid is
  `bs=1, log2=8` (Discrete), `bs=1/64, log2=16` (Tame, CNC) and
  `bs=1/4, log2=19` (Hu/SCS), with `padding=1, remove_fuzz=True`. The
  net book is priced with the distortions calibrated on the gross one.
  The net covers are read off the gross distributions: CNC's
  `79.64 xs 41.11` is the cat unit's 90% to 99.5% quantiles on the grid
  (41.109375, 120.75), Hu/SCS's `372.4 xs 40.25` the hurricane
  severity's 95% to 99.5% quantiles (`isf(0.05)`, `isf(0.005)`).[^pir]
* The test suite's Hu/SCS programs (`M.PIR.5`, `M.PIR.6`) write the
  lognormal scale as `exp(-1.9**2 / 2)`; 0.30.1's DecL parser reads
  `-1.9**2` as `(-1.9)**2`, so SCS events get a mean of 36.97 instead of
  1 and the 99.9% assets land on the grid's top. `hs.py` builds the
  program in Python and gets mean 1; write the scale as a number.[^pir]
* 0.30.1's `Portfolio.price(p, d, allocation='linear')` on a continuous
  book adds up to slightly less than the premium it calibrated: 115.1488
  against 115.1511 on gross CNC under the CCoC, 321.3136 against
  321.3140 on gross Hu/SCS; its lifted allocation reaches the calibrated
  premium. prospicio's linear units add up to the calibrated premium and
  match 0.30.1's linear premiums to 0.003 (CNC) and 0.0012
  (Hu/SCS).[^pirguide]
* With `aggregate`'s default `normalize=True`, a severity is put on the
  grid by rounding and renormalized by the mass below the top.
  prospicio reproduces it exactly from `Grid.rounding(sev, h, 2 n)`,
  keeping the first `n` cells and dividing by their sum: the 99.9% assets
  (267.203125, 2583.0), the premiums and the calibrated parameters then
  agree with 0.30.1 to the calibration's error. With `Grid.rounding`'s
  own convention (tail mass on the last point) Hu/SCS's assets are
  2584.75.[^pirguide]
* The lifted allocation on these books differs from 0.30.1 under the
  CCoC in the premiums too, not only the capital (gross CNC NonCat
  75.43 against 74.64), and under PH in the capital only (48.80 against
  48.54); Wang, dual and TVaR agree to about 0.004.[^pirguide]
* An aggregate cover whose attachment falls between grid points (CNC's
  41.11 on a 1/64 grid) leaves the net cat mean 0.0014 apart: 17.7856 in
  prospicio's `Tower` (mean exact on the grid), 17.7870 in 0.30.1.[^pirguide]
* prospicio's `Portfolio.calibrate("wang", ...)` took about 9 s on the
  2^19-point Hu/SCS portfolio (the other families 0.1 to 0.4 s).
  Calibrating on the total capped at the assets, a one-unit portfolio of
  about 10,000 points, gives the same parameter in 0.15 s.[^pirguide]
  Since #331 `calibrate` merges equal values first, and the capped values
  at the assets are most of the 525,955 totals: Wang takes 0.25 s on the
  full portfolio (release build, 2026-10-20), the other families 0.02 to
  0.03 s, with the capped portfolio's parameters to about 1e-15.

[^pnl]: pnl.md
[^renewal]: aggregate_renewal.py
[^clash]: aggregate_clash.py
[^terms]: aggregate_contract_terms.py
[^moments]: aggregate_moments.py
[^oep]: aggregate_oep.py
[^reins]: aggregate_reinsurance.py
[^natural]: aggregate_natural.py
[^ruin]: aggregate_ruin.py
[^picks]: aggregate_picks.py
[^bounds]: aggregate_bounds.py
[^counts]: aggregate_counts.py
[^pir]: aggregate 0.30.1 extensions
[^pirguide]: pricing-insurance-risk-case-studies.md
