---
type: Reference
title: aggregate 1.0.1's reinsurance pricing examples, what its numbers mean and where prospicio matches them
description: The reference file behind examples/reinsurance_pricing.ipynb, six contracts built in aggregate 1.0.1 (tower, program and share, aggregate limit with reinstatements, stop loss and deductible, variable features, distortion pricing), how aggregate defines each number, and the quirks met matching them with prospicio (skewness deficit, share re-bucketing, tail lumping, calibration error).
resource: https://github.com/mynl/aggregate/blob/v0.22.0/docs/2_user_guides/2_x_re_pricing.rst
tags: [reinsurance, pricing, aggregate, parity, fft, contract-terms, distortions]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T16:00:00Z }
verified: { by: process:ci, at: 2026-10-22T16:00:00Z }
sources:
  - id: docs
    resource: https://github.com/mynl/aggregate/blob/v0.22.0/docs/2_user_guides/2_x_re_pricing.rst
    title: aggregate's Reinsurance Pricing guide at tag v0.22.0 (removed from master after 0.22)
  - id: pipeline
    resource: https://github.com/mynl/aggregate/blob/master/docs/2_aggregate_overview/pipeline-reinsurance.rst
    title: aggregate's reinsurance pipeline notes, the reporting frames of 1.0.1
  - id: script
    resource: ../validation/scripts/aggregate_re_pricing.py
    title: aggregate_re_pricing.py, which writes validation/reference/re_pricing_aggregate.csv
  - id: reference
    resource: ../validation/reference/re_pricing_aggregate.csv
    title: re_pricing_aggregate.csv, 544 rows (example, quantity, value, note) from aggregate 1.0.1
  - id: notebook
    resource: ../examples/reinsurance_pricing.ipynb
    title: examples/reinsurance_pricing.ipynb
  - id: test
    resource: ../python/tests/test_examples.py
    title: test_reinsurance_pricing in python/tests/test_examples.py
---

# What the reference file holds

* Six examples, 544 rows: `tower_gross`, `tower_program`,
  `agg_limit_reinst`, `stop_loss` and `stop_loss_aad`, `variable_retro`,
  `variable_swing`, `variable_pc`, `variable_slide`, and
  `risk_pricing_gross` and `risk_pricing_ceded`. The books and clauses of
  Reinsurance Functions and Bear and Nemlick's Treaties 1, 2, 4, 5 and 6 come
  from the guide; the reinstatement schedules, the swing terms, the half
  placements and the risk pricing are designed (marked in the script).[^docs]
* Every build uses `normalize=False`, and the grid (`bs`, `log2`) is recorded
  per example. Regenerate with `aggregate==1.0.1` in its own environment
  (`validation/scripts/aggregate_re_pricing.py`, 15 seconds).[^script]
* `examples/reinsurance_pricing.ipynb` compares 343 of them with prospicio, all
  inside their stated tolerances: 321 independent comparisons and 22
  consistency checks that hold by construction (a calibrated price equal to its
  target premium, the CCoC return on capital, a maximum reinstatement premium).
  CI runs the notebook (`test_reinsurance_pricing`).[^notebook][^test]
* The guide's Treaty 1 table gives the subject premiums as 3,000 and 6,000,
  but its code uses `[9000 3000]` at `[0.04 0.03]`, which gives the 450 (3.75%
  of 12,000) its Treaty 4 text states; the script and notebook follow the
  code.[^docs]

# How aggregate defines the numbers

* **Grid and discretisation.** Rounding with survival differences (mass at
  `b_i` is `P(b_i - bs/2 < X <= b_i + bs/2)`) is `Grid::rounding` to `1e-11`.
  A policy limit `L xs 0` is a severity censored at `L`: with `L / bs + 1`
  points `Grid::rounding` lumps everything beyond `L` on `L`, which is
  aggregate's censoring. A layer at an attachment or share that falls between
  buckets is split linearly between the two neighbours (`reins_bucket='linear'`),
  which keeps the layers consistent with the discretized severity (they still
  add up to the gross on the grid); `Tower::on_grid` does the same, so its
  `on_points` flag is `false` there with no effect on agreement. Rounding also
  drops claims below half a bucket to 0: on the ReTester book (bucket 4) 10.3%
  of claims, so a layer attaching at 0 shows `pr_attach` 0.897.
* **Statistics.** `*_mean/_sd/_cv/_skew` are of the annual aggregate density,
  not per claim; quantiles are the lower quantile on the grid. `pr_attach` is
  the per-claim probability read from the grid (`1 - cdf(att)`), `pr_detach`
  is `P(X >= att + limit)`, and `cond_sev_mean` is the layer's mean given the
  claim reaches it (`layer / pr_attach`), the only conditional number.[^pipeline]
* **`exposure at rate`.** The expected loss of each class is subject premium
  times rate, and the claim count is solved from it with the limited expected
  severity. The Lomax is `Burr(alpha, 1, scale)` (`300 * pareto q - 300`
  limited to 700). The classes share one gamma mixing variable, so they are
  one mixed Poisson, a negative binomial with `Var/E = 1 + cv^2 n`, over a
  severity mixture weighted by the class counts. The exact CV is
  `sqrt(n E[X^2] + cv^2 n^2 E[X]^2) / (n E[X])`.
* **`net of`.** `aggregate net of 360 xs 0` removes the first 360 of each
  year's loss, so its density is the loss *above* 360 (the reinsurer's side);
  its `insurance_charge` (`after / before` the deductible) is the excess ratio,
  the Table M charge at entry ratio 360 / 450, not the loss elimination ratio; read gross, ceded and net of an
  occurrence program from `reins_density_df` (`p_agg_gross`, `p_agg_ceded_occ`,
  `p_agg_net_occ`).
* **Reinstatements** are pro rata as to amount only; `ReinstatementTerms`
  prices them on the annual layer loss before the annual limit. The
  reinstatement premium, recovery and probability match `Tower::on_grid`'s
  `expected_reinstatement_premium` to `3e-9`. The script's
  `pr_any_reinstatement` is `P(R > 0)`, any loss to the layer, even under a
  schedule whose first reinstatement is free. The designed deposit (1,200, a
  rate on line of 171%) is not a price: schedule A's loss ratio is 33%.
* **Sliding scale docs.** The guide's code builds the schedule with
  `make_ceder_netter([(0.25, .2, .35), (0.5, .1, .55)])`, a first slope of
  0.25 where its text says 0.75; the code's commission is 35.74% against
  30.34% for the stated terms. The reference file records both
  (`slide_commission_rate_mean_docs_code`); prospicio matches the stated terms.

# Quirks met matching it

* **The skew is polluted by the density's deficit.** `xsden_to_meancvskew`
  (through `xsden_to_mwrangler`) puts `1 - sum(p)`, an FFT rounding deficit of
  `1e-15` to `5e-14`, at the implied maximum loss `xs[-1] + bs` (262,144 on a
  `2**16` grid of step 4), which adds `deficit * 262144^3` (tens to hundreds)
  to a third moment. For a light-tailed cover this moves the quoted skew in
  the fifth digit: a 250 xs 750 stop loss quotes 0.5895211 where the density
  has 0.5894734 (`8.1e-5`). The reference file keeps both (`*_skew` and
  `*_skew_central`, with `*_density_deficit`); the notebook compares
  prospicio's skewness with `*_skew_central`, to `2e-5`.
* **FFT noise at the top of the grid moves skewness.** prospicio's FFT leaves
  noise out to the grid's top, and cubed distance makes it count. The 250 xs
  250 layer's skewness differs from aggregate's density by about `2e-6`
  (observed `1.5e-6` Windows, `1.8e-6` Linux, on 2026-10-22: the noise depends
  on the platform), and by under `1e-13` once the probability beyond 10,000 is
  dropped. It was `1.2e-5` when this note was first written; the round-off
  removed from `Tower::on_grid`'s grids by #367 is the likely cause of the
  change, an inference that no test checks. The worst skewness difference is
  the 50% stop-loss share's `9.5e-6`, where aggregate's re-bucketing adds to
  it; the 80% share's is `1.5e-6`. Means and standard deviations agree to
  better than `1e-7`. What CI checks (`test_reinsurance_pricing`, #269): the
  layer's difference lies between `5e-7` and `2.5e-6` and the prose says
  "about 2e-6", the `1e-13`, the two share figures as the prose prints them,
  the 50% share being the worst, and the `1e-7` on means and standard
  deviations. The per-platform values are observations, not checked.
* **A share off the grid is re-bucketed, adding variance.** An 80% share of
  1,000 xs 1,000 has ceded values at multiples of 3.2 on a step-4 grid;
  aggregate splits them, prospicio keeps a step `0.8 * 4`. Means agree to
  `1e-11`, the standard deviation to `3e-6` (`7.4e-6` for a 50% stop-loss
  share), and quantiles by at most half a bucket. Every on-grid quantile is the
  same grid point in both: the cdf at each compared quantile is at least `6.7e-8`
  from the probability, so none is near a tie.
* **The tail beyond the grid.** aggregate with `normalize=False` drops the
  mass beyond the grid; prospicio puts it on the last point. A stop loss
  at 2,800 on the Treaty 2 grid (top 8,192, mass `5.7e-10`) differs by `5.7e-7`
  relative, which is why that quantity has a `1e-5` tolerance.
* **The bucket.** aggregate sizes `bs = 4` for the ReTester book with its
  policy limit of 5,000 (and `0.0625`, `0.5` for the Treaty 1 and 6 books);
  `recommend_grid` on the unlimited lognormal gives 2, because it does not see
  a limit. The notebook passes aggregate's `bs` in every section.
* **Calibration stops at a premium error.** `calibrate_distortions` reports
  `error` (4.3e-6 for the gross view's TVaR, 5e-13 for PH), so the calibrated
  parameter differs from prospicio's bisected one by that over the slope. The
  notebook compares the distortion function `g(s)` at `s = 0.05` and `0.5` with
  aggregate's own parameter, and the fixed-parameter and unlimited prices, all
  at `1e-6`; the TVaR `g` differs by `4.9e-9` (gross), because its premium is
  steep in the parameter. A calibrated price of `min(X, a)` equals the target
  premium by construction, so it is only a convergence check. A premium of
  `min(X, a)` is the ask at assets `a`, not the unlimited premium.
* **The assets are a grid point.** `a = q(0.99)` is the lower quantile (5,440
  gross, 1,824 ceded), so `F(a)` is 0.990015 and 0.990037, not 0.99; CCoC
  `P = (E[min(X, a)] + r a) / (1 + r)`.
* **Not matched.** The bid price (the dual distortion applied through
  `Distortion.price`), `gini_p_*`, the standard deviation of the reinstatement
  premium, the loss corridor (the two define it differently when an annual limit
  exists, see `aggregate-mildenhall.md`), cat and casualty exposure examples
  and multi-line limit profiles.

[^docs]: 2_x_re_pricing.rst
[^script]: aggregate_re_pricing.py
[^notebook]: examples/reinsurance_pricing.ipynb
[^test]: test_reinsurance_pricing
[^pipeline]: pipeline-reinsurance.rst
