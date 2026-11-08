# prospicio-xll

prospicio's reserving, pricing and capital methods as Excel worksheet
functions: an XLL add-in for 64-bit Excel on Windows. Design and decisions
are in `docs/design/xll.md`.

## Get it

- **Release:** each version's GitHub release has
  `prospicio-xll-<version>.zip` attached.
- **CI:** every run of the **XLL** workflow keeps a `prospicio-xll`
  artifact with `prospicio.xll`, the test workbook, this README and a
  SHA-256 (*Actions → XLL → a run → Artifacts*).

## Install

1. Copy `prospicio.xll` to a local folder. If it came from a download or
   an email, right-click it, choose Properties and tick **Unblock**.
   Excel refuses an XLL that is marked as coming from the internet.
2. In Excel, go to File > Options > Add-ins > Manage: Excel Add-ins > Go >
   Browse. Pick `prospicio.xll` and keep it ticked.
3. Type `=PRO.VERSION()` in a cell. Insert Function (fx) lists the
   functions in four categories, "Prospicio – Reserving", "– Pricing",
   "– Capital" and "– Common", and shows each argument's help.

A workbook opened before the add-in was loaded shows `#NAME?`. Press
Ctrl+Alt+F9 to recalculate.

## How the functions behave

- **Spilling:** type a function in one cell. In Excel 365 its table spills
  from there. Older Excel needs it entered as an array formula
  (Ctrl+Shift+Enter) over the result's size.
- **Errors:** a bad argument returns `#PROSPICIO! <reason>` in one cell.
- **Ranges broadcast:** where an argument takes a number or a range,
  a column against a row gives a table. For example,
  `=PRO.PRC.LAYER(B1:E1, A2:A4, "pareto", 500, 2)` gives limits across and
  attachments down.
- **Simulations** take `sims` (default 10,000, at most 1,000,000) and
  `seed` (default 0). The same seed gives the same table.
- **Long runs don't block Excel:** the bootstraps, the Bayesian fits and
  `PRO.CAP.COPULA.SIM` run in the background. Their cell shows
  `#GETTING_DATA` until the result arrives, and you can keep working.
  Cells with the same inputs share one run, and a recalculation that
  changes none of a function's inputs answers at once from the result
  kept. Editing while a run goes on interrupts the recalculation but not
  the run: the next recalculation picks up its result.
- **No duplicates of Excel:** use Excel's own `NPV`, `XNPV`, `IRR` and
  `XIRR` for cash flows in cells.

## Names and groups

Each function's name says its group, so typing the prefix lists the
group's functions as you type:

| Prefix | Group | For example |
|---|---|---|
| `PRO.RES.` | Reserving: a triangle or a latest diagonal to a reserve and its range | `PRO.RES.MACK`, `PRO.RES.ODP` |
| `PRO.PRC.` | Pricing: ratemaking, layers, exposure and experience rating, treaty terms | `PRO.PRC.ILF`, `PRO.PRC.BURNINGCOST` |
| `PRO.CAP.` | Capital: aggregate and cat losses, risk measures, allocation, the P&L | `PRO.CAP.AGG.STATS`, `PRO.CAP.ALLOCATE` |
| `PRO.SEV.`, `PRO.EVT.`, `PRO.TRI.` and two others | Common: severities, tails, triangle helpers | `PRO.SEV.FIT`, `PRO.TRI.LATEST` |

**Older workbooks:** before 0.0.9 the functions had no group (`PRO.MACK`,
`PRO.ILF`), and those names now show `#NAME?`. Replace each with its new
name (Ctrl+H, Find what `PRO.MACK(`, Replace with `PRO.RES.MACK(`): the
new name is the old one with `RES.`, `PRC.` or `CAP.` after `PRO.`, as
the tables below show. `docs/design/xll.md`, decision 47, lists every
pair.

## Start here

These functions cover most everyday work:

| Task | Functions |
|---|---|
| Reserve a triangle | `PRO.RES.CL`, `PRO.RES.MACK`, `PRO.RES.BF`, `PRO.RES.CAPECOD` |
| Reserve ranges | `PRO.RES.ODP`, `PRO.RES.MACK.CDR` (one-year view) |
| Price a layer | `PRO.PRC.LAYER`, `PRO.PRC.ILF`, `PRO.SEV.FIT` |
| Exposure rate | `PRO.PRC.EXPOSURE.SHARE`, `PRO.PRC.PROFILE` |
| Experience rate | `PRO.PRC.BURNINGCOST`, `PRO.PRC.PARETO.EXTRAPOLATE` |
| Rate indication | `PRO.PRC.ONLEVEL`, `PRO.PRC.TREND.FIT`, `PRO.PRC.LR.EXPERIENCE`, `PRO.PRC.INDICATED.CHANGE` |
| Aggregate loss | `PRO.CAP.AGG.STATS`, `PRO.CAP.AGG.LAYERS` |
| Price with a risk load | `PRO.PRC.TOWER.PRICE`, `PRO.PRC.QUOTE` |

The tables below list every function by group. Signatures show optional
arguments in brackets.

## Reserving (`PRO.RES.`)

### Chain ladder and Mack

| Function | Returns |
|---|---|
| `PRO.RES.CL(triangle, [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Chain ladder exhibit: latest, CDF, ultimate and reserve per origin, and a total |
| `PRO.RES.CL.LDF(...)`, `PRO.RES.CL.CDF(...)` | Age-to-age or age-to-ultimate factors, one per column; the last LDF is the tail |
| `PRO.RES.CL.ULTIMATE(...)`, `PRO.RES.CL.RESERVE(...)` | One column, one value per origin |
| `PRO.RES.CL.FACTORS(triangle, [average], [tail], [selected], [kind], [weight], [sigma], [pattern_ages])` | Per link: LDF, CDF, Mack sigma and the factor's standard error |
| `PRO.RES.MACK(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma], [pattern_ages])` | Mack exhibit: process, parameter and total SE and CV per origin |

### Expected loss methods

| Function | Returns |
|---|---|
| `PRO.RES.EL(triangle, exposure, apriori, ...)` | Expected loss method: exposure times the a priori loss ratio |
| `PRO.RES.BF(triangle, exposure, apriori, ...)` | Bornhuetter–Ferguson |
| `PRO.RES.BENKTANDER(triangle, exposure, apriori, [iterations], ...)` | Benktander (iterated BF) |
| `PRO.RES.CAPECOD(triangle, exposure, [trend], [decay], ...)` | Cape Cod (Stanard–Bühlmann) |

The `...` is the chain ladder's options: `[average], [tail], [origins],
[selected], [kind], [weight], [pattern_ages]`.

### Ranges and the one-year view

| Function | Returns |
|---|---|
| `PRO.RES.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins], [average])` | ODP bootstrap: CL reserve, mean, SD, CV and percentiles per origin and in total |
| `PRO.RES.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins], [average])` | The bootstrap's draws: a row per simulation, a column per origin and the total |
| `PRO.RES.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack's bootstrap, laid out as `PRO.RES.ODP` |
| `PRO.RES.MACK.BOOT.SIMS(...)` | Its draws, laid out as `PRO.RES.ODP.SIMS` |
| `PRO.RES.MACK.CDR(...)`, same arguments as `PRO.RES.MACK` | Merz–Wüthrich: CDR(1) SE and run-off SE per origin |
| `PRO.RES.MACK.CDR.YEARS(...)` | CDR SE by future calendar year |
| `PRO.RES.ODP.CDR(...)`, `PRO.RES.MACK.BOOT.CDR(...)` | Each bootstrap's simulated one-year CDR: mean, SD and percentiles |

### Other reserving models

| Function | Returns |
|---|---|
| `PRO.RES.CLARK(triangle, [exposure], [curve], [max_age], [origins])` | Clark's growth curve: the LDF method, or Cape Cod with `exposure`; reserve and SEs |
| `PRO.RES.CLARK.PARAMS(triangle, [exposure], [curve], [max_age])` | Its fitted parameters |
| `PRO.RES.ODP.GLM(triangle, [origins])` | The ODP model as a quasi-Poisson GLM: reserve, SEs, RMSEP and CV per origin |
| `PRO.RES.ODP.GLM.PARAMS(triangle, [origins])` | Its coefficients, SEs and dispersion |
| `PRO.RES.ODP.GLM.RESIDUALS(triangle, [by], [origins])` | Its residuals by cell, or grouped by `"origin"`, `"development"` or `"calendar"` |
| `PRO.RES.DEV.EXHIBIT(triangle, [average], [tail], [selected], [kind], [weight], [pattern_ages])` | Friedland's development exhibit: six averages of the link ratios, then the selected factors, CDF and percent developed |
| `PRO.RES.SELECT(ultimates, weights, [ages], [reported], [paid], [origins], [methods])` | The selected ultimate per origin, weighting several methods' ultimates by origin or by age, with IBNR and unpaid |
| `PRO.RES.CURVE(triangle, [shape], [average], [fit_period], [keep_observed_until], [tail_periods])` | A development curve fitted over every age: observed, fitted and selected factor per link, and the curve's tail |
| `PRO.RES.CURVE.STATS(triangle, [shapes], [average], [fit_period], [tail_periods])` | The shapes compared: parameters, sum of squares, R-squared, adjusted R-squared and tail |
| `PRO.RES.MUNICH(paid, incurred, [sigma], [tail_paid], [tail_incurred], [origins])` | Munich chain ladder: paid and incurred developed together; latest and ultimate P/I, paid reserve, incurred IBNR and the separate chain ladders |
| `PRO.RES.MUNICH.PARAMS(paid, incurred, [sigma], [tail_paid], [tail_incurred])` | Its `q`, `rho`, factors and sigmas by period, and the two `lambda`s |
| `PRO.RES.PIC(paid, incurred, [sigma], [tau], [extrapolation], [convention], [phi_mean], [phi_sd], [psi_mean], [psi_sd], [origins])` | Merz and Wüthrich's paid-incurred chain: one ultimate from both, its weight on incurred, reserve, SE and CV |
| `PRO.RES.PIC.PARAMS(paid, incurred, ...)` | Its `sigma`, `tau` and the posterior of `Phi` and `Psi` by period |
| `PRO.RES.FREQSEV(amounts, counts, ultimate_counts, [average], [tail], [selected], [kind], [origins])` | Frequency-severity (Friedland's approach #1): ultimate counts times the severity developed by the chain ladder |
| `PRO.RES.FREQSEV.TRENDED(amounts, counts, exposure, frequency, severity, [base], [frequency_trend], [severity_trend], [exposure_factors], [frequency_factors], [severity_factors], [origins])` | Frequency-severity (approach #2): exposure times a trended frequency, times a trended severity |
| `PRO.RES.DISPOSAL(closed, paid, ultimate_counts, [rates], [severities], [tail_severity_from], [trend], [base], [factors], [origins], [output])` | The disposal rate method: ultimate and unpaid per origin, or the selections and projected squares |
| `PRO.RES.CASE.OS(paid, incurred, [paid_on_case], [case_on_case], [paid_to_ultimate], [origins], [output])` | Case outstanding development: ultimate, IBNR and unpaid per origin, or the ratios and projected squares |
| `PRO.RES.TREND.BY.AGE(amounts, [counts])` | The exponential trend across origins at each age of a severity triangle, and pooled: rate, R-squared, points |
| `PRO.RES.BS.CASE(incurred, paid, open, trend, [max_age], [output])` | Berquist-Sherman case adequacy adjustment: the adjusted incurred triangle |
| `PRO.RES.BS.SETTLEMENT(paid, closed, ultimate_counts, [rates], [interpolation], [output])` | Berquist-Sherman settlement rate adjustment: the adjusted paid or closed count triangle |
| `PRO.RES.BS(incurred, paid, reported_counts, closed, ultimate_counts, trend, [max_age], [rates], [interpolation], [output])` | Both adjustments: the adjusted incurred, paid or closed count triangle |
| `PRO.RES.RATIO(numerator, denominator, ultimate_claims, [average], [tail], [selected], [kind], [factors], [ultimate_ratios], [origins], [output])` | Friedland's ratio approach for salvage and subrogation or ALAE: the ratio developed to ultimate (multiplicative or additive) and applied to ultimate claims |
| `PRO.RES.ULAE(paid_ulae, paid, case_outstanding, ibnr, [technique], [ratio], [weights], [reported], [closed], [pure_ibnr], [unclosed], [ultimate], [output])` | Unpaid ULAE by the classical technique, Kittel's refinement or Conger and Nolibos' generalized approach |
| `PRO.RES.BAYES.ODP(triangle, [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins])` | Bayesian ODP chain ladder by MCMC: reserves, posterior or diagnostics |
| `PRO.RES.BAYES.BF(triangle, exposure, apriori, [cv], [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins], [common_cv], [new_premium], [pattern_mean], [pattern_sd])` | Bayesian Bornhuetter–Ferguson by MCMC: reserves, ultimates, posterior, diagnostics or a new year's loss ratio |
| `PRO.RES.INDUSTRY.PRIOR(triangles, companies, premiums, [min_developed], [subtract_process], [age], [output])` | The industry prior from several companies' triangles: the apriori and its CVs, the pattern's mean and SD by age, or a report |

### A latest diagonal (bordereau)

| Function | Returns |
|---|---|
| `PRO.RES.DIAG.CL(origins, ages, latest, pattern, ...)` | Chain ladder on one latest value per origin, with a selected pattern |
| `PRO.RES.DIAG.EL(origins, ages, latest, exposure, apriori, pattern, ...)`, `PRO.RES.DIAG.BF(...)` | Expected loss and BF on a latest diagonal |
| `PRO.RES.DIAG.BENKTANDER(origins, ages, latest, exposure, apriori, pattern, [iterations], ...)` | Benktander on a latest diagonal |
| `PRO.RES.DIAG.CAPECOD(origins, ages, latest, exposure, pattern, [trend], [decay], ...)` | Cape Cod on a latest diagonal |
| `PRO.RES.PATTERN.AVERAGE(pattern_ages, pattern, pattern_origin, [grain], [kind])` | A pattern averaged from a finer origin basis (accident months) to a coarser one |

The `...` is `[pattern_ages], [kind], [tail], [grain], [valuation],
[past_end], [pattern_origin]`.

### Triangle helpers

| Function | Returns |
|---|---|
| `PRO.TRI.INCREMENTAL(triangle)`, `PRO.TRI.CUMULATIVE(triangle)` | Incremental from cumulative, and back |
| `PRO.TRI.LINKRATIOS(triangle)` | Age-to-age link ratios |
| `PRO.TRI.LATEST(triangle)` | The latest diagonal, a column |

These are common functions, without a group, since pricing reads triangles too.

### Reserving arguments

- `triangle`: cumulative values, one row per origin (oldest first), one
  column per age. Leave cells blank or `#N/A` where unobserved.
- `average`: `"volume"` (default), `"simple"`, `"regression"` or
  `"geometric"` (chain ladder only). It can also be a row
  `kind, n_periods, drop_high, drop_low`: `{"volume",5,1,1}` averages the
  latest 5 calendar periods without the highest and lowest at each age.
  The ODP bootstrap functions take the same row as their last argument,
  with the kind `"volume"` or blank.
- `tail`: omit for none, a factor such as `1.05`, or `"exponential"`,
  `"inverse_power"`, `"bondy"` or `"log-linear"` (R ChainLadder's
  `tail = TRUE`).
- `ultimates`, `weights`: `PRO.RES.SELECT` takes the methods' ultimates
  side by side (a column each, blank where one does not project) and
  either a row of weights for every origin, a table of weights like
  `ultimates`, or, with each origin's `ages`, a table whose first column
  is the age from which its row holds (`{12,0,1;48,1,0}`: the second
  method before 48 months, the first from then on). Each origin's
  weights must sum to 1.
- `rates`, `severities`, `paid_on_case`, `case_on_case`: how the claim
  count methods select at each age. Blank for the average of every
  origin, a number `n` or `"latest n"` for the latest `n`, `"medial n"`
  (one dropped each side) or `"medial nxk"`, `"diagonal"` for the latest
  diagonal's (the settlement adjustment's default), or a row of values
  under the triangle's columns. `base` names an origin by its label in
  `origins`; `ultimate_counts` is a column, one per origin, such as
  `PRO.RES.CL.ULTIMATE` on a count triangle gives.
- `shape`: a development curve, `"exponential"` (default),
  `"inverse_power"`, `"hoerl"`, `"sherman"`, `"loglogistic"` or
  `"weibull"`. `fit_period` is one or two ages in months (the first fitted,
  the age the fit stops before); `keep_observed_until` keeps the observed
  factors before an age in months. To develop on the curve, pass its
  `Selected` factors (all but the last) as `PRO.RES.CL`'s `selected`, with
  `TRANSPOSE` if they are a column, and its last as `tail`.
- `paid`, `incurred`: the Munich chain ladder and the paid-incurred chain
  take two triangles of the same size, observed at the same cells. The
  Munich `sigma` is one setting or two cells, paid then incurred. The PIC
  estimates `sigma` and `tau` unless you give a row of them (one per
  period, and one fewer for `tau`); its priors are non-informative unless
  you give a mean and an SD row for a side; `convention` `"r"` reproduces
  R ChainLadder's `PaidIncurredChain` exactly, the default `"model"` is
  the model's own posterior (about 0.5% apart on USAA).
- `origins`: labels for the first column (1, 2, ... if omitted).
- `selected`, `kind`, `weight`, `pattern_ages`: your own factors.
  `selected` is a row with a cell per column (a blank keeps the
  estimate). `kind` is `"ldf"` (default), `"cdf"` or `"percent"`.
  `weight` (0 to 1, default 1) blends the selection with the estimate.
  `pattern_ages` gives each value's age in months, so a monthly pattern
  can be applied to an annual triangle.
- `sigma`: how Mack's last sigma is filled in, `"log-linear"` (default)
  or `"mack"`.
- `exposure`: premium per origin, positive. `apriori`: the expected loss
  ratio, one value or one per origin.
- `curve` (Clark): `"loglogistic"` (default) or `"weibull"`. `max_age`:
  the age in months where development stops.
- `percentiles`: a row or column such as `{0.75,0.95,0.995}`.
  `process`: `"gamma"` (default) or `"none"`. Mack's bootstrap also takes
  `"lognormal"`, `"normal"` and `"residuals"`.
- A negative CDR is adverse: the CDR is the opening ultimate less the
  re-estimated one.

**Latest diagonal.** `origins` is a year (`2024`) or an Excel date, floored
to `grain` (`"year"` by default, or `"semiannual"`, `"quarter"`,
`"month"`). Give either `ages` (months, one per origin) or `valuation`
(one Excel date), not both. `latest` uses 0 for nothing reported, never a
blank. `pattern` holds selected values, such as a row of `PRO.RES.CL.LDF`. A
blank in it is skipped, so the factor before it spans the gap; give 1 for
no development. `pattern_origin` is the basis the pattern was measured on
(`"month"` for accident months). `past_end` handles origins older than
the pattern: `"reject"` (default, an error), `"none"` (a factor of 1) or
`"tail"`. To apply one triangle's chain ladder to another book, pass its
`PRO.RES.CL.LDF` row as `pattern` and the last cell as `tail`.

**Bayesian models.** Check `output` `"diagnostics"` first: the largest
R-hat should be under 1.01, with no divergences. `draws` is per chain
(default 1000, after as many warm-up draws), with 4 `chains`. Keep
`prior_sd` (default 10) wide: a tight prior pulls the late columns toward
the first. In `PRO.RES.BAYES.BF`, each origin's ultimate has a gamma prior
with mean `apriori × exposure` and CV `cv` (default 0.1). A small `cv`
approaches `PRO.RES.BF`, and `"flat"` gives `PRO.RES.BAYES.ODP`. `common_cv` adds
a shared error in the a priori loss ratio. The last eight fits are kept,
so the different outputs of one fit sample only once. `output`
`"new_year"` with `new_premium` gives the next origin's loss ratio and
ultimate. `pattern_mean` and `pattern_sd` put a prior on the percents
developed, one per column with the first blank. `PRO.RES.INDUSTRY.PRIOR`
takes the companies' triangles stacked, each company's rows together and
labelled in `companies`, with their premiums; chain its summary into
`apriori`, `cv` and `common_cv`, and its pattern's percents developed into
`pattern_mean` and `pattern_sd`.

## Pricing (`PRO.PRC.`)

### Layers and Pareto rating

| Function | Returns |
|---|---|
| `PRO.PRC.LEV(limit, family, p1, p2, [p3])` | Limited expected value at each limit |
| `PRO.PRC.LAYER(limit, attachment, family, p1, p2, [p3])` | Expected loss to the layer per ground-up loss |
| `PRO.PRC.LAYER.VAR(...)`, `PRO.PRC.LAYER.CV(...)`, same arguments | The layer loss's variance, and its CV given a loss above the attachment |
| `PRO.PRC.ILF(limit, basic_limit, family, p1, p2, [p3])` | Increased limit factor |
| `PRO.PRC.LER(deductible, family, p1, p2, [p3])` | Loss elimination ratio |
| `PRO.PRC.PARETO.EXTRAPOLATE(from_limit, from_attachment, to_limit, to_attachment, alpha, [truncation])` | One layer's loss extrapolated to another |
| `PRO.PRC.PARETO.ALPHA(limit1, attachment1, loss1, limit2, attachment2, loss2, [truncation])` | Alpha implied by two layers' losses |
| `PRO.PRC.PARETO.ALPHA.FREQLAYER(threshold, frequency, limit, attachment, loss, [truncation])` | Alpha from a frequency and a layer |
| `PRO.PRC.PARETO.ALPHA.FREQS(threshold1, frequency1, threshold2, frequency2, [truncation])` | Alpha from two frequencies |

- `family` and `p1`, `p2`, `p3`: a severity, as in
  [Severities and tails](#severities-and-tails).
- `limit`, `attachment`, `basic_limit`: a number, a range or
  `"unlimited"`.

### Exposure rating

| Function | Returns |
|---|---|
| `PRO.PRC.EXPOSURE.CURVE(x, curve, p1, [p2], [p3])` | Exposure curve G(x) |
| `PRO.PRC.EXPOSURE.SHARE(limit, attachment, mpl, curve, p1, [p2], [p3])` | A layer's share of a risk's expected loss |
| `PRO.PRC.EXPOSURE.QUANTILE(p, curve, p1, [p2], [p3])` | Destruction rate at each probability |
| `PRO.PRC.EXPOSURE.STATS(curve, p1, [p2], [p3])` | b, g, mean destruction rate and total-loss probability |
| `PRO.PRC.SWISSRE.C(rate)` | Swiss Re c for a mean destruction rate |
| `PRO.PRC.SWISSRE.C.LAYER(limit, attachment, mpl, share)` | Swiss Re c for a layer's share |
| `PRO.PRC.PROFILE(sums_insured, expected_losses, limit, attachment, curve, p1, [p2], [p3], [lower], [upper], [spread], [retention], [lines])` | Exposure-rated risk profile by band, with a total |

- `curve`: `"swiss_re"` (c: 1.5, 2, 3, 4 for Y1 to Y4, 5 for Lloyd's),
  `"mbbefd"` (b, g) or `"tabulated"` (ranges of x and G(x) from 0 to 1,
  optionally `"linear"` or `"log-log"`). In `PRO.PRC.EXPOSURE.CURVE` and
  `PRO.PRC.EXPOSURE.QUANTILE`, a row of c against a column of x gives one
  column per curve.
- `mpl`: maximum possible loss or sum insured, a number or a range.
- `PRO.PRC.PROFILE`: one sum insured and expected loss per band. `lower` and
  `upper` spread each band's sums insured, `spread` is `"uniform"`
  (default), `"tilted"` or `"auto"`, and `retention` and `lines` put a
  surplus treaty underneath.

### Experience rating and ratemaking

| Function | Returns |
|---|---|
| `PRO.PRC.BURNINGCOST(accident, amounts, limits, attachments, years, premiums, [development], [trend], [trend_to], [threshold], [agg_deductible], [agg_limit])` | As-if layer losses and burning cost by year, for one or more layers |
| `PRO.PRC.ONLEVEL(effective, changes, from, to, [period], [term], [applies], [given])` | On-level factor of each period (parallelogram method) |
| `PRO.PRC.RATELEVEL(...)`, same arguments | One period's premium by rate level |
| `PRO.PRC.AVGDATE(from, to, [period], [term], [basis])` | Average written or earned date of each period, for trend lengths |
| `PRO.PRC.TREND.FIT(times, values, [kind])` | Exponential or linear trend: annual change, intercept, slope, R² |
| `PRO.PRC.LR.EXPERIENCE(earned_premium, losses, [on_level], [premium_trend], [development], [loss_trend], [lae])` | Loss ratio method: projected premium, losses and loss ratio by year |
| `PRO.PRC.PP.EXPERIENCE(exposure, losses, [development], [loss_trend], [lae])` | Pure premium method by year |
| `PRO.PRC.INDICATED.CHANGE(loss_ratio, variable, profit, [fixed_ratio])` | Indicated change, loss ratio method |
| `PRO.PRC.INDICATED.RATE(pure_premium, variable, profit, [fixed_per_exposure])` | Indicated rate, pure premium method |
| `PRO.PRC.PERMISSIBLE.LR(variable, profit, [fixed_ratio])` | Permissible loss ratio |
| `PRO.PRC.EXPENSES(ratios, [fixed_shares])` | Fixed and variable expense ratios |
| `PRO.PRC.TRENDED.PRESENT(prior_indicated, prior_implemented, trend)` | Trended present rates complement |
| `PRO.PRC.CRED.STANDARD(p, k, [kind], [cv])` | Full-credibility standard in claims |
| `PRO.PRC.CRED.Z(n, standard)` | Partial credibility, square root rule |
| `PRO.PRC.CRED.WEIGHTED(estimate, credibility, complement)` | Credibility-weighted estimate |
| `PRO.PRC.BUHLMANN(ratios, [weights], [estimator], [groups])` | Bühlmann–Straub credibility by group |
| `PRO.PRC.BUHLMANN.PARAMS(ratios, [weights], [estimator])` | Its structure parameters |
| `PRO.PRC.GLM(data, response, terms, [family], [link], [exposure], [weights], [p])` | A GLM on a table: coefficients with SEs, and fit statistics |
| `PRO.PRC.GLM.PREDICT(data, response, terms, newdata, ...)` | The fitted mean on each new row |
| `PRO.PRC.GLM.RELATIVITIES(data, response, terms, ..., [level])` | Relativities of a log-link GLM, with confidence intervals |

- Dates (`effective`, `from`, `to`, `times`) are Excel dates or year
  fractions (`2011.5` is 1 July 2011). A period is `[from, to)`, so
  calendar year 2011 is `2011, 2012`.
  `PRO.YEARFRACTION` (under [Other](#other)) shows where a date lands.
- `period` is `"calendar"` (default) or `"policy"`, and `term` is the
  policy term in years (default 1). `applies` is `"written"` (default) or
  `"in_force"`. `given` `"levels"` reads `changes` as cumulative levels.
- Credibility: `ratios` has one row per group and one column per period,
  blank where missing. `weights` has the same shape. `estimator` is
  `"unbiased"` (default) or `"iterative"`.
- The GLM functions are for modest tables. Fit rating plans on full
  policy data in Python or R.

### Treaty terms, the quote and the payout

| Function | Returns |
|---|---|
| `PRO.PRC.TOWER.PRICE(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], limits, attachments, [shares], [agg_deductibles], [agg_limits], [cover_limit], [cover_retention], distortion, parameter, [step], [points])` | A tower priced by a distortion |
| `PRO.PRC.LAYER.TERMS(count_mean, [dispersion], family, p1, p2, [p3], layer_limit, layer_attachment, premium, [reinstatements], [commissions], [loss_ratios], [profit_share], [profit_allowance], [share], [policy_limit], [step], [points])` | A layer's expected premium, reinstatement premium and commissions |
| `PRO.PRC.SWING(loss, basic, lcm, [minimum], [maximum], [share])` | Swing-rated (retro) premium at each loss |
| `PRO.PRC.SLIDING.SCALE(loss, premium, commissions, loss_ratios)` | Sliding-scale commission at each ceded loss |
| `PRO.PRC.PROFIT.COMMISSION(loss, premium, share, allowance)` | Profit commission at each ceded loss |
| `PRO.PRC.SLIDING.SCALE.TERM(losses, premium, commissions, loss_ratios, [carry], [carry_years])` | Sliding scale over a multi-year term, carrying deficits and credits |
| `PRO.PRC.PROFIT.COMMISSION.TERM(losses, premium, share, allowance, [carry_years])` | Profit commission over a multi-year term, carrying deficits |
| `PRO.PRC.QUOTE(expected_loss, risk_premium, [assets], [expenses], [brokerage], [fixed], [limit], [quoted])` | Technical premium with expenses and brokerage, and the return at a quoted premium |
| `PRO.PRC.PAYOUT.PATTERN(times, values, [kind], [shift], [rate])` | Payment pattern: paid, outstanding and discount factor by time |
| `PRO.PRC.PAYOUT.STATS(times, values, rate, [kind], [shift])` | Discount factor, duration and capital-years |
| `PRO.PRC.PAYOUT.PRICE(expected_loss, assets, coc, times, values, rate, [kind], [shift])` | A cost-of-capital price restated over the payout |

- `count_mean`, `dispersion`, `step` and `points` are as in
  [Aggregate and cat](#aggregate-and-cat); a distortion's `family` is as in
  [Risk measures, allocation and dependence](#risk-measures-allocation-and-dependence).
- Contract terms are at 100%, with losses ceded at the placed share. A
  sliding scale is two increasing ranges, commissions and loss ratios,
  linear between them and flat beyond. In `PRO.PRC.LAYER.TERMS`, paid
  `reinstatements` (a row of rates) also set the annual limit. They
  cannot be combined with a sliding scale or a profit commission.
- Multi-year terms: `losses` has one term per row and one year per column.
  A carried amount is used oldest first, and dropped after `carry_years`.
- `PRO.PRC.QUOTE` grosses up for expenses and brokerage:
  `(risk_premium + fixed) / (1 - expenses - brokerage)`.
- Payout `times` are years from inception. `values` are `"shares"` paid
  at each time (default), `"paid"` to date, or `"factors"` to ultimate
  (such as a paid `PRO.RES.CL.CDF` row). `shift` moves the pattern later.

## Capital (`PRO.CAP.`)

### Aggregate and cat

| Function | Returns |
|---|---|
| `PRO.CAP.AGG.STATS(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], [probabilities], [step], [points])` | Annual aggregate loss: moments, VaR and TVaR |
| `PRO.CAP.AGG.LAYERS(count_mean, [dispersion], family, p1, p2, [p3], limits, attachments, [shares], [step], [points])` | Statistics of each occurrence layer |
| `PRO.CAP.AGG.COVER(count_mean, [dispersion], family, p1, p2, [p3], limits, retentions, [shares], [occ_limits], [occ_attachments], [occ_shares], [subject], [step], [points])` | Aggregate covers (stop loss), net of or on occurrence layers |
| `PRO.CAP.AGG.PML(count_mean, [dispersion], family, p1, p2, [p3], [return_periods], [step], [points])` | Occurrence and aggregate PMLs by return period |
| `PRO.CAP.ELT.STATS(rates, means, [sd_independent], [sd_correlated], exposures)` | Event loss table: event count, expected annual loss, SD, CV |
| `PRO.CAP.ELT.OEP(rates, means, [sd_independent], [sd_correlated], exposures, [return_periods], [amounts])` | Its OEP curve with secondary uncertainty |
| `PRO.CAP.YLT.STATS(years, losses, [n_years], [probabilities], [quantile], [times], [hours], [negative])` | Year loss table: average annual loss, SD, VaR, TVaR |
| `PRO.CAP.YLT.OEP(years, losses, [return_periods], [amounts], [n_years], [quantile], [times], [hours], [negative])` | Its occurrence and aggregate exceedance curves |

- `count_mean` is claims a year. `dispersion` is Var N / E N (default 1,
  Poisson). A mixing CV c gives `1 + c² × count_mean`. The grid is sized
  automatically unless you give `step` and `points`.

### Risk measures, allocation and dependence

| Function | Returns |
|---|---|
| `PRO.CAP.DISTORTION(s, family, parameter, [r0])` | A distortion's g(s) |
| `PRO.CAP.DISTORTION.PRICE(values, [probs], family, parameter, [r0])` | Price of outcomes or draws: mean, margin, loss ratio |
| `PRO.CAP.DISTORTION.CALIBRATE(values, [probs], premium, [families], [r0])` | Each family's parameter that gives a target premium |
| `PRO.CAP.ALLOCATE(draws, family, parameter, [method], [r0])` | A distortion risk measure allocated to units (Euler, covariance, Shapley and others) |
| `PRO.CAP.ALLOCATE.TOTAL(draws, total, [method], [measure], [level])` | A premium or capital allocated by the haircut or equal-risk rule |
| `PRO.CAP.COPULA.SIM(family, parameter, [sims], [dim], [seed], [nu])` | Uniforms simulated from a copula, one row per simulation |
| `PRO.CAP.COPULA.CDF(points, family, parameter, [nu])` | A copula's distribution function at each point |
| `PRO.CAP.IMAN.CONOVER(draws, correlation, [seed], [dof], [shuffle])` | Simulated columns reordered to a rank correlation |

- Distortion families use `aggregate`'s names: `ccoc` (r), `ph` (rho),
  `wang` (lambda), `dual` (beta), `tvar` (p), `exp` (k), and the capped
  `clin`, `cll`, `lep` and `ly`, which take an intercept `r0`. Families
  with several parameters take them as one range: `bitvar` (`p0, p1, w`),
  `beta` (`a, b`), `wtdtvar` (levels and weights).
- Without `probs`, `values` are equally likely draws, so a column of
  `PRO.RES.ODP.SIMS` prices directly. To price a loss backed by assets `a`,
  cap the outcomes first with `=MIN(range, a)`.
- Copulas: `gaussian` and `t` take one correlation or a correlation
  matrix (the t also takes `nu`). `clayton`, `gumbel`, `frank` and `joe`
  take `theta`. Turn uniforms into losses with `PRO.SEV.QUANTILE`.

### P&L and cash flows

| Function | Returns |
|---|---|
| `PRO.CAP.PNL(values, [probs], premium, [expenses], [covers], [face], [output], [levels], [expense_at_mean])` | A book's P&L ledger with the annual covers it buys |
| `PRO.CAP.CASHFLOW(u, terms, patterns, requirement, [commission], [output], [rate], [commutation_requirement])` | A collateralised deal on each simulated loss ratio: IRR, MOIC, NPV and its monthly flows |

- `PRO.CAP.CASHFLOW` reads a deal from blocks of rows, a label and its values
  (`premium`, `commutation_month`, ...). `output` is `"simulations"`
  (default), `"flows"` or `"trace"`.

## Common (no group)

### Severities and tails

| Function | Returns |
|---|---|
| `PRO.SEV.CDF(x, family, p1, p2, [p3])`, `PRO.SEV.SURVIVAL(...)` | Distribution or survival function at each x |
| `PRO.SEV.QUANTILE(p, family, p1, p2, [p3])` | Quantile (VaR) at each probability |
| `PRO.SEV.STATS(family, p1, p2, [p3])` | Mean, SD and CV |
| `PRO.SEV.FIT(losses, family, [output], [t], [truncation], [thresholds], [censored], [weights], [upper])` | Maximum likelihood fit to large losses, with goodness of fit |
| `PRO.SEV.FIT.GROUPED(bounds, counts, family, [method], [output])` | A fit to counts in size bands |
| `PRO.SEV.FIT.LEV(sizes, lev, family, [weights], [output])` | A fit by minimum distance to empirical limited expected values |
| `PRO.EVT.THRESHOLDS(data, [thresholds])` | Mean excess and GPD shape and scale at each threshold, with SEs |
| `PRO.EVT.FIT(data, threshold, [probabilities])` | GPD fitted above a threshold, and the tail's VaR and TVaR |
| `PRO.EVT.HILL(data, [k])` | Hill's tail index from the k largest values, and its alpha |

- `family` and `p1`, `p2`, `p3`:
  - `"pareto"` (t, alpha, optional truncation)
  - `"lognormal"` (meanlog, sdlog) and `"lognormal_cv"` (mean, CV)
  - `"gamma"` (shape, scale) and `"gamma_cv"` (mean, CV)
  - `"weibull"`, `"loglogistic"` and `"inverse_gamma"` (shape, scale)
  - `"inverse_gaussian"` (mean, shape)
  - `"burr"` (alpha, gamma, scale)
  - `"lomax"` (alpha, scale), Bahnemann's Pareto
  - `"gpd"` (xi, beta)
  - `"piecewise_pareto"` (a range of thresholds, a range of alphas, an
    optional truncation)
  - `"gen_pareto"` (t, alpha_ini, alpha_tail)
- `PRO.SEV.FIT`: `family` is a body family (`lognormal`, `gamma`,
  `weibull`, `burr`, `lomax`) fitted by maximum likelihood, or a Pareto
  family fitted from the threshold `t`. `censored` is TRUE where a policy
  limit capped the loss, and `upper` caps the losses kept. `output` is
  `"params"` (default; the parameters line up with `p1`, `p2`, `p3`, so
  they feed `PRO.PRC.LAYER` directly), `"gof"` (P-P and Q-Q pairs) or
  `"compare"` (all five body families side by side, `family` left blank).
  Check the upper Q-Q pairs before trusting a body family in the tail.
- `PRO.SEV.FIT.GROUPED`: `bounds` has one more cell than `counts`. A
  blank last bound opens the top band.
- EVT: pick the lowest threshold above which the shape and modified scale
  are stable in `PRO.EVT.THRESHOLDS`, then fit there with `PRO.EVT.FIT`.
  The Pareto alpha is `1 / xi`.

### Other

| Function | Returns |
|---|---|
| `PRO.VERSION()` | The add-in's version |
| `PRO.YEARFRACTION(date)` | An Excel date on Werner and Modlin's half-month grid |

## Build and test

```sh
cargo xtask xll        # target/xll/prospicio.xll, then the fake-Excel smoke test
```

On Windows this uses the MSVC target (`rustup target add
x86_64-pc-windows-msvc`). Elsewhere it uses the GNU target with MinGW
(`rustup target add x86_64-pc-windows-gnu`, `apt install
gcc-mingw-w64-x86-64`). The smoke test runs when `x86_64-w64-mingw32-gcc`
is found, under Wine off Windows.

`python3 harness/make_workbook.py prospicio-xll-test.xlsx` (needs
openpyxl) writes a test workbook that calls every function with a check
beside it: a published value, R's or Python's result, or an identity. A
check reads TRUE when the function agrees.
