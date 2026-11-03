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
3. Type `=PRO.VERSION()` in a cell. The functions are in the `prospicio`
   category of Insert Function (fx), which also shows each argument's help.

A workbook opened before the add-in was loaded shows `#NAME?`. Press
Ctrl+Alt+F9 to recalculate.

## How the functions behave

- **Spilling:** type a function in one cell. In Excel 365 its table spills
  from there. Older Excel needs it entered as an array formula
  (Ctrl+Shift+Enter) over the result's size.
- **Errors:** a bad argument returns `#PROSPICIO! <reason>` in one cell.
- **Ranges broadcast:** where an argument takes a number or a range,
  a column against a row gives a table. For example,
  `=PRO.LAYER(B1:E1, A2:A4, "pareto", 500, 2)` gives limits across and
  attachments down.
- **Simulations** take `sims` (default 10,000, at most 1,000,000) and
  `seed` (default 0). The same seed gives the same table.
- **No duplicates of Excel:** use Excel's own `NPV`, `XNPV`, `IRR` and
  `XIRR` for cash flows in cells.

## Start here

These functions cover most everyday work:

| Task | Functions |
|---|---|
| Reserve a triangle | `PRO.CL`, `PRO.MACK`, `PRO.BF`, `PRO.CAPECOD` |
| Reserve ranges | `PRO.ODP`, `PRO.MACK.CDR` (one-year view) |
| Price a layer | `PRO.LAYER`, `PRO.ILF`, `PRO.SEV.FIT` |
| Exposure rate | `PRO.EXPOSURE.SHARE`, `PRO.PROFILE` |
| Experience rate | `PRO.BURNINGCOST`, `PRO.PARETO.EXTRAPOLATE` |
| Rate indication | `PRO.ONLEVEL`, `PRO.TREND.FIT`, `PRO.LR.EXPERIENCE`, `PRO.INDICATED.CHANGE` |
| Aggregate loss | `PRO.AGG.STATS`, `PRO.AGG.LAYERS` |
| Price with a risk load | `PRO.TOWER.PRICE`, `PRO.QUOTE` |

The tables below list every function by area. Signatures show optional
arguments in brackets.

## Reserving

### Chain ladder and Mack

| Function | Returns |
|---|---|
| `PRO.CL(triangle, [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Chain ladder exhibit: latest, CDF, ultimate and reserve per origin, and a total |
| `PRO.CL.LDF(...)`, `PRO.CL.CDF(...)` | Age-to-age or age-to-ultimate factors, one per column; the last LDF is the tail |
| `PRO.CL.ULTIMATE(...)`, `PRO.CL.RESERVE(...)` | One column, one value per origin |
| `PRO.CL.FACTORS(triangle, [average], [tail], [selected], [kind], [weight], [sigma], [pattern_ages])` | Per link: LDF, CDF, Mack sigma and the factor's standard error |
| `PRO.MACK(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma], [pattern_ages])` | Mack exhibit: process, parameter and total SE and CV per origin |

### Expected loss methods

| Function | Returns |
|---|---|
| `PRO.EL(triangle, exposure, apriori, ...)` | Expected loss method: exposure times the a priori loss ratio |
| `PRO.BF(triangle, exposure, apriori, ...)` | Bornhuetter–Ferguson |
| `PRO.BENKTANDER(triangle, exposure, apriori, [iterations], ...)` | Benktander (iterated BF) |
| `PRO.CAPECOD(triangle, exposure, [trend], [decay], ...)` | Cape Cod (Stanard–Bühlmann) |

The `...` is the chain ladder's options: `[average], [tail], [origins],
[selected], [kind], [weight], [pattern_ages]`.

### Ranges and the one-year view

| Function | Returns |
|---|---|
| `PRO.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins], [average])` | ODP bootstrap: CL reserve, mean, SD, CV and percentiles per origin and in total |
| `PRO.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins], [average])` | The bootstrap's draws: a row per simulation, a column per origin and the total |
| `PRO.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack's bootstrap, laid out as `PRO.ODP` |
| `PRO.MACK.BOOT.SIMS(...)` | Its draws, laid out as `PRO.ODP.SIMS` |
| `PRO.MACK.CDR(...)`, same arguments as `PRO.MACK` | Merz–Wüthrich: CDR(1) SE and run-off SE per origin |
| `PRO.MACK.CDR.YEARS(...)` | CDR SE by future calendar year |
| `PRO.ODP.CDR(...)`, `PRO.MACK.BOOT.CDR(...)` | Each bootstrap's simulated one-year CDR: mean, SD and percentiles |

### Other reserving models

| Function | Returns |
|---|---|
| `PRO.CLARK(triangle, [exposure], [curve], [max_age], [origins])` | Clark's growth curve: the LDF method, or Cape Cod with `exposure`; reserve and SEs |
| `PRO.CLARK.PARAMS(triangle, [exposure], [curve], [max_age])` | Its fitted parameters |
| `PRO.ODP.GLM(triangle, [origins])` | The ODP model as a quasi-Poisson GLM: reserve, SEs, RMSEP and CV per origin |
| `PRO.ODP.GLM.PARAMS(triangle, [origins])` | Its coefficients, SEs and dispersion |
| `PRO.ODP.GLM.RESIDUALS(triangle, [by], [origins])` | Its residuals by cell, or grouped by `"origin"`, `"development"` or `"calendar"` |
| `PRO.BAYES.ODP(triangle, [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins])` | Bayesian ODP chain ladder by MCMC: reserves, posterior or diagnostics |
| `PRO.BAYES.BF(triangle, exposure, apriori, [cv], [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins], [common_cv])` | Bayesian Bornhuetter–Ferguson by MCMC: reserves, ultimates, posterior or diagnostics |

### A latest diagonal (bordereau)

| Function | Returns |
|---|---|
| `PRO.DIAG.CL(origins, ages, latest, pattern, ...)` | Chain ladder on one latest value per origin, with a selected pattern |
| `PRO.DIAG.EL(origins, ages, latest, exposure, apriori, pattern, ...)`, `PRO.DIAG.BF(...)` | Expected loss and BF on a latest diagonal |
| `PRO.DIAG.BENKTANDER(origins, ages, latest, exposure, apriori, pattern, [iterations], ...)` | Benktander on a latest diagonal |
| `PRO.DIAG.CAPECOD(origins, ages, latest, exposure, pattern, [trend], [decay], ...)` | Cape Cod on a latest diagonal |
| `PRO.PATTERN.AVERAGE(pattern_ages, pattern, pattern_origin, [grain], [kind])` | A pattern averaged from a finer origin basis (accident months) to a coarser one |

The `...` is `[pattern_ages], [kind], [tail], [grain], [valuation],
[past_end], [pattern_origin]`.

### Triangle helpers

| Function | Returns |
|---|---|
| `PRO.TRI.INCREMENTAL(triangle)`, `PRO.TRI.CUMULATIVE(triangle)` | Incremental from cumulative, and back |
| `PRO.TRI.LINKRATIOS(triangle)` | Age-to-age link ratios |
| `PRO.TRI.LATEST(triangle)` | The latest diagonal, a column |

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
  `"inverse_power"` or `"bondy"`.
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
blank. `pattern` holds selected values, such as a row of `PRO.CL.LDF`. A
blank in it is skipped, so the factor before it spans the gap; give 1 for
no development. `pattern_origin` is the basis the pattern was measured on
(`"month"` for accident months). `past_end` handles origins older than
the pattern: `"reject"` (default, an error), `"none"` (a factor of 1) or
`"tail"`. To apply one triangle's chain ladder to another book, pass its
`PRO.CL.LDF` row as `pattern` and the last cell as `tail`.

**Bayesian models.** Check `output` `"diagnostics"` first: the largest
R-hat should be under 1.01, with no divergences. `draws` is per chain
(default 1000, after as many warm-up draws), with 4 `chains`. Keep
`prior_sd` (default 10) wide: a tight prior pulls the late columns toward
the first. In `PRO.BAYES.BF`, each origin's ultimate has a gamma prior
with mean `apriori × exposure` and CV `cv` (default 0.1). A small `cv`
approaches `PRO.BF`, and `"flat"` gives `PRO.BAYES.ODP`. `common_cv` adds
a shared error in the a priori loss ratio. The last eight fits are kept,
so the different outputs of one fit sample only once.

## Severity and layers

| Function | Returns |
|---|---|
| `PRO.SEV.CDF(x, family, p1, p2, [p3])`, `PRO.SEV.SURVIVAL(...)` | Distribution or survival function at each x |
| `PRO.SEV.QUANTILE(p, family, p1, p2, [p3])` | Quantile (VaR) at each probability |
| `PRO.SEV.STATS(family, p1, p2, [p3])` | Mean, SD and CV |
| `PRO.LEV(limit, family, p1, p2, [p3])` | Limited expected value at each limit |
| `PRO.LAYER(limit, attachment, family, p1, p2, [p3])` | Expected loss to the layer per ground-up loss |
| `PRO.LAYER.VAR(...)`, `PRO.LAYER.CV(...)`, same arguments | The layer loss's variance, and its CV given a loss above the attachment |
| `PRO.ILF(limit, basic_limit, family, p1, p2, [p3])` | Increased limit factor |
| `PRO.LER(deductible, family, p1, p2, [p3])` | Loss elimination ratio |
| `PRO.SEV.FIT(losses, family, [output], [t], [truncation], [thresholds], [censored], [weights], [upper])` | Maximum likelihood fit to large losses, with goodness of fit |
| `PRO.SEV.FIT.GROUPED(bounds, counts, family, [method], [output])` | A fit to counts in size bands |
| `PRO.SEV.FIT.LEV(sizes, lev, family, [weights], [output])` | A fit by minimum distance to empirical limited expected values |
| `PRO.PARETO.EXTRAPOLATE(from_limit, from_attachment, to_limit, to_attachment, alpha, [truncation])` | One layer's loss extrapolated to another |
| `PRO.PARETO.ALPHA(limit1, attachment1, loss1, limit2, attachment2, loss2, [truncation])` | Alpha implied by two layers' losses |
| `PRO.PARETO.ALPHA.FREQLAYER(threshold, frequency, limit, attachment, loss, [truncation])` | Alpha from a frequency and a layer |
| `PRO.PARETO.ALPHA.FREQS(threshold1, frequency1, threshold2, frequency2, [truncation])` | Alpha from two frequencies |
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
- `limit`, `attachment`, `basic_limit`: a number, a range or
  `"unlimited"`.
- `PRO.SEV.FIT`: `family` is a body family (`lognormal`, `gamma`,
  `weibull`, `burr`, `lomax`) fitted by maximum likelihood, or a Pareto
  family fitted from the threshold `t`. `censored` is TRUE where a policy
  limit capped the loss, and `upper` caps the losses kept. `output` is
  `"params"` (default; the parameters line up with `p1`, `p2`, `p3`, so
  they feed `PRO.LAYER` directly), `"gof"` (P-P and Q-Q pairs) or
  `"compare"` (all five body families side by side, `family` left blank).
  Check the upper Q-Q pairs before trusting a body family in the tail.
- `PRO.SEV.FIT.GROUPED`: `bounds` has one more cell than `counts`. A
  blank last bound opens the top band.
- EVT: pick the lowest threshold above which the shape and modified scale
  are stable in `PRO.EVT.THRESHOLDS`, then fit there with `PRO.EVT.FIT`.
  The Pareto alpha is `1 / xi`.

## Exposure rating

| Function | Returns |
|---|---|
| `PRO.EXPOSURE.CURVE(x, curve, p1, [p2], [p3])` | Exposure curve G(x) |
| `PRO.EXPOSURE.SHARE(limit, attachment, mpl, curve, p1, [p2], [p3])` | A layer's share of a risk's expected loss |
| `PRO.EXPOSURE.QUANTILE(p, curve, p1, [p2], [p3])` | Destruction rate at each probability |
| `PRO.EXPOSURE.STATS(curve, p1, [p2], [p3])` | b, g, mean destruction rate and total-loss probability |
| `PRO.SWISSRE.C(rate)` | Swiss Re c for a mean destruction rate |
| `PRO.SWISSRE.C.LAYER(limit, attachment, mpl, share)` | Swiss Re c for a layer's share |
| `PRO.PROFILE(sums_insured, expected_losses, limit, attachment, curve, p1, [p2], [p3], [lower], [upper], [spread], [retention], [lines])` | Exposure-rated risk profile by band, with a total |

- `curve`: `"swiss_re"` (c: 1.5, 2, 3, 4 for Y1 to Y4, 5 for Lloyd's),
  `"mbbefd"` (b, g) or `"tabulated"` (ranges of x and G(x) from 0 to 1,
  optionally `"linear"` or `"log-log"`). In `PRO.EXPOSURE.CURVE` and
  `PRO.EXPOSURE.QUANTILE`, a row of c against a column of x gives one
  column per curve.
- `mpl`: maximum possible loss or sum insured, a number or a range.
- `PRO.PROFILE`: one sum insured and expected loss per band. `lower` and
  `upper` spread each band's sums insured, `spread` is `"uniform"`
  (default), `"tilted"` or `"auto"`, and `retention` and `lines` put a
  surplus treaty underneath.

## Experience rating and ratemaking

| Function | Returns |
|---|---|
| `PRO.BURNINGCOST(accident, amounts, limits, attachments, years, premiums, [development], [trend], [trend_to], [threshold], [agg_deductible], [agg_limit])` | As-if layer losses and burning cost by year, for one or more layers |
| `PRO.ONLEVEL(effective, changes, from, to, [period], [term], [applies], [given])` | On-level factor of each period (parallelogram method) |
| `PRO.RATELEVEL(...)`, same arguments | One period's premium by rate level |
| `PRO.AVGDATE(from, to, [period], [term], [basis])` | Average written or earned date of each period, for trend lengths |
| `PRO.YEARFRACTION(date)` | An Excel date on Werner and Modlin's half-month grid |
| `PRO.TREND.FIT(times, values, [kind])` | Exponential or linear trend: annual change, intercept, slope, R² |
| `PRO.LR.EXPERIENCE(earned_premium, losses, [on_level], [premium_trend], [development], [loss_trend], [lae])` | Loss ratio method: projected premium, losses and loss ratio by year |
| `PRO.PP.EXPERIENCE(exposure, losses, [development], [loss_trend], [lae])` | Pure premium method by year |
| `PRO.INDICATED.CHANGE(loss_ratio, variable, profit, [fixed_ratio])` | Indicated change, loss ratio method |
| `PRO.INDICATED.RATE(pure_premium, variable, profit, [fixed_per_exposure])` | Indicated rate, pure premium method |
| `PRO.PERMISSIBLE.LR(variable, profit, [fixed_ratio])` | Permissible loss ratio |
| `PRO.EXPENSES(ratios, [fixed_shares])` | Fixed and variable expense ratios |
| `PRO.TRENDED.PRESENT(prior_indicated, prior_implemented, trend)` | Trended present rates complement |
| `PRO.CRED.STANDARD(p, k, [kind], [cv])` | Full-credibility standard in claims |
| `PRO.CRED.Z(n, standard)` | Partial credibility, square root rule |
| `PRO.CRED.WEIGHTED(estimate, credibility, complement)` | Credibility-weighted estimate |
| `PRO.BUHLMANN(ratios, [weights], [estimator], [groups])` | Bühlmann–Straub credibility by group |
| `PRO.BUHLMANN.PARAMS(ratios, [weights], [estimator])` | Its structure parameters |
| `PRO.GLM(data, response, terms, [family], [link], [exposure], [weights], [p])` | A GLM on a table: coefficients with SEs, and fit statistics |
| `PRO.GLM.PREDICT(data, response, terms, newdata, ...)` | The fitted mean on each new row |
| `PRO.GLM.RELATIVITIES(data, response, terms, ..., [level])` | Relativities of a log-link GLM, with confidence intervals |

- Dates (`effective`, `from`, `to`, `times`) are Excel dates or year
  fractions (`2011.5` is 1 July 2011). A period is `[from, to)`, so
  calendar year 2011 is `2011, 2012`.
- `period` is `"calendar"` (default) or `"policy"`, and `term` is the
  policy term in years (default 1). `applies` is `"written"` (default) or
  `"in_force"`. `given` `"levels"` reads `changes` as cumulative levels.
- Credibility: `ratios` has one row per group and one column per period,
  blank where missing. `weights` has the same shape. `estimator` is
  `"unbiased"` (default) or `"iterative"`.
- The GLM functions are for modest tables. Fit rating plans on full
  policy data in Python or R.

## Aggregate, cat and treaty terms

| Function | Returns |
|---|---|
| `PRO.AGG.STATS(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], [probabilities], [step], [points])` | Annual aggregate loss: moments, VaR and TVaR |
| `PRO.AGG.LAYERS(count_mean, [dispersion], family, p1, p2, [p3], limits, attachments, [shares], [step], [points])` | Statistics of each occurrence layer |
| `PRO.AGG.COVER(count_mean, [dispersion], family, p1, p2, [p3], limits, retentions, [shares], [occ_limits], [occ_attachments], [occ_shares], [subject], [step], [points])` | Aggregate covers (stop loss), net of or on occurrence layers |
| `PRO.AGG.PML(count_mean, [dispersion], family, p1, p2, [p3], [return_periods], [step], [points])` | Occurrence and aggregate PMLs by return period |
| `PRO.TOWER.PRICE(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], limits, attachments, [shares], [agg_deductibles], [agg_limits], [cover_limit], [cover_retention], distortion, parameter, [step], [points])` | A tower priced by a distortion |
| `PRO.LAYER.TERMS(count_mean, [dispersion], family, p1, p2, [p3], layer_limit, layer_attachment, premium, [reinstatements], [commissions], [loss_ratios], [profit_share], [profit_allowance], [share], [policy_limit], [step], [points])` | A layer's expected premium, reinstatement premium and commissions |
| `PRO.SWING(loss, basic, lcm, [minimum], [maximum], [share])` | Swing-rated (retro) premium at each loss |
| `PRO.SLIDING.SCALE(loss, premium, commissions, loss_ratios)` | Sliding-scale commission at each ceded loss |
| `PRO.PROFIT.COMMISSION(loss, premium, share, allowance)` | Profit commission at each ceded loss |
| `PRO.SLIDING.SCALE.TERM(losses, premium, commissions, loss_ratios, [carry], [carry_years])` | Sliding scale over a multi-year term, carrying deficits and credits |
| `PRO.PROFIT.COMMISSION.TERM(losses, premium, share, allowance, [carry_years])` | Profit commission over a multi-year term, carrying deficits |
| `PRO.ELT.STATS(rates, means, [sd_independent], [sd_correlated], exposures)` | Event loss table: event count, expected annual loss, SD, CV |
| `PRO.ELT.OEP(rates, means, [sd_independent], [sd_correlated], exposures, [return_periods], [amounts])` | Its OEP curve with secondary uncertainty |
| `PRO.YLT.STATS(years, losses, [n_years], [probabilities], [quantile], [times], [hours], [negative])` | Year loss table: average annual loss, SD, VaR, TVaR |
| `PRO.YLT.OEP(years, losses, [return_periods], [amounts], [n_years], [quantile], [times], [hours], [negative])` | Its occurrence and aggregate exceedance curves |

- `count_mean` is claims a year. `dispersion` is Var N / E N (default 1,
  Poisson). A mixing CV c gives `1 + c² × count_mean`. The grid is sized
  automatically unless you give `step` and `points`.
- Contract terms are at 100%, with losses ceded at the placed share. A
  sliding scale is two increasing ranges, commissions and loss ratios,
  linear between them and flat beyond. In `PRO.LAYER.TERMS`, paid
  `reinstatements` (a row of rates) also set the annual limit. They
  cannot be combined with a sliding scale or a profit commission.
- Multi-year terms: `losses` has one term per row and one year per column.
  A carried amount is used oldest first, and dropped after `carry_years`.

## Capital, risk loads and cash flows

| Function | Returns |
|---|---|
| `PRO.DISTORTION(s, family, parameter, [r0])` | A distortion's g(s) |
| `PRO.DISTORTION.PRICE(values, [probs], family, parameter, [r0])` | Price of outcomes or draws: mean, margin, loss ratio |
| `PRO.DISTORTION.CALIBRATE(values, [probs], premium, [families], [r0])` | Each family's parameter that gives a target premium |
| `PRO.ALLOCATE(draws, family, parameter, [method], [r0])` | A distortion risk measure allocated to units (Euler, covariance, Shapley and others) |
| `PRO.ALLOCATE.TOTAL(draws, total, [method], [measure], [level])` | A premium or capital allocated by the haircut or equal-risk rule |
| `PRO.COPULA.SIM(family, parameter, [sims], [dim], [seed], [nu])` | Uniforms simulated from a copula, one row per simulation |
| `PRO.COPULA.CDF(points, family, parameter, [nu])` | A copula's distribution function at each point |
| `PRO.IMAN.CONOVER(draws, correlation, [seed], [dof], [shuffle])` | Simulated columns reordered to a rank correlation |
| `PRO.PNL(values, [probs], premium, [expenses], [covers], [face], [output], [levels], [expense_at_mean])` | A book's P&L ledger with the annual covers it buys |
| `PRO.QUOTE(expected_loss, risk_premium, [assets], [expenses], [brokerage], [fixed], [limit], [quoted])` | Technical premium with expenses and brokerage, and the return at a quoted premium |
| `PRO.PAYOUT.PATTERN(times, values, [kind], [shift], [rate])` | Payment pattern: paid, outstanding and discount factor by time |
| `PRO.PAYOUT.STATS(times, values, rate, [kind], [shift])` | Discount factor, duration and capital-years |
| `PRO.PAYOUT.PRICE(expected_loss, assets, coc, times, values, rate, [kind], [shift])` | A cost-of-capital price restated over the payout |
| `PRO.CASHFLOW(u, terms, patterns, requirement, [commission], [output], [rate], [commutation_requirement])` | A collateralised deal on each simulated loss ratio: IRR, MOIC, NPV and its monthly flows |

- Distortion families use `aggregate`'s names: `ccoc` (r), `ph` (rho),
  `wang` (lambda), `dual` (beta), `tvar` (p), `exp` (k), and the capped
  `clin`, `cll`, `lep` and `ly`, which take an intercept `r0`. Families
  with several parameters take them as one range: `bitvar` (`p0, p1, w`),
  `beta` (`a, b`), `wtdtvar` (levels and weights).
- Without `probs`, `values` are equally likely draws, so a column of
  `PRO.ODP.SIMS` prices directly. To price a loss backed by assets `a`,
  cap the outcomes first with `=MIN(range, a)`.
- Copulas: `gaussian` and `t` take one correlation or a correlation
  matrix (the t also takes `nu`). `clayton`, `gumbel`, `frank` and `joe`
  take `theta`. Turn uniforms into losses with `PRO.SEV.QUANTILE`.
- `PRO.QUOTE` grosses up for expenses and brokerage:
  `(risk_premium + fixed) / (1 - expenses - brokerage)`.
- Payout `times` are years from inception. `values` are `"shares"` paid
  at each time (default), `"paid"` to date, or `"factors"` to ultimate
  (such as a paid `PRO.CL.CDF` row). `shift` moves the pattern later.
- `PRO.CASHFLOW` reads a deal from blocks of rows, a label and its values
  (`premium`, `commutation_month`, ...). `output` is `"simulations"`
  (default), `"flows"` or `"trace"`.

## Other

| Function | Returns |
|---|---|
| `PRO.VERSION()` | The add-in's version |

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
