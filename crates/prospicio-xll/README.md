# prospicio-xll

prospicio's reserving and layer pricing methods as Excel worksheet functions: an XLL add-in
for 64-bit Excel on Windows. Design and decisions: `docs/design/xll.md`.

## Get it

- From CI: the **XLL** workflow's run on `main` (or on a PR) has a
  `prospicio-xll` artifact with `prospicio.xll`, the test workbook, this
  README and a SHA-256 (*Actions → XLL → a run → Artifacts*).
- From a release: a version's GitHub release carries
  `prospicio-xll-<version>.zip` once the owner publishes it.

## Build

```sh
cargo xtask xll        # target/xll/prospicio.xll, then the fake-Excel smoke test
```

On Windows this uses the MSVC target (`rustup target add
x86_64-pc-windows-msvc`); elsewhere the GNU target with MinGW
(`rustup target add x86_64-pc-windows-gnu`, `apt install
gcc-mingw-w64-x86-64`). The smoke test runs when `x86_64-w64-mingw32-gcc`
is found, under Wine off Windows.

`python3 harness/make_workbook.py prospicio-xll-test.xlsx` (needs
openpyxl) writes a test workbook that calls every one of the add-in's
functions with a check beside it (a published value, R or Python's, or an
identity): RAA with R ChainLadder's totals, then a sheet per area (Layers,
ODP and its draws, Mack bootstrap and its draws, Expected loss, One-year,
Exposure, Ratemaking, Term, Bayesian and others). A check reads TRUE when
the function agrees.

## Install in Excel

1. Copy `prospicio.xll` to a local folder. If it came from a download or
   an email, right-click it, Properties, tick **Unblock**: Excel refuses
   XLLs marked as from the internet.
2. File > Options > Add-ins > Manage: Excel Add-ins > Go > Browse, pick
   `prospicio.xll`, keep it ticked.
3. Type `=PRO.VERSION()` in a cell. Functions are in the `prospicio`
   category of Insert Function (fx).

A workbook opened before the add-in was loaded shows `#NAME?`; press
Ctrl+Alt+F9 to recalculate.

## Functions

| Function | Returns |
|---|---|
| `PRO.CL(triangle, [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Chain ladder exhibit with a total row |
| `PRO.CL.LDF(triangle, [average], [tail], [selected], [kind], [weight], [pattern_ages])` | Age-to-age factors, one per column; the last is the tail |
| `PRO.CL.CDF(triangle, [average], [tail], [selected], [kind], [weight], [pattern_ages])` | Age-to-ultimate factors, one per column |
| `PRO.CL.ULTIMATE(...)`, `PRO.CL.RESERVE(...)` | One value per origin |
| `PRO.CL.FACTORS(triangle, [average], [tail], [selected], [kind], [weight], [sigma], [pattern_ages])` | LDF, CDF, sigma and standard error per link, and the tail |
| `PRO.MACK(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma], [pattern_ages])` | Mack exhibit: process, parameter and total SE and CV, with a total |
| `PRO.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins], [average])` | ODP bootstrap exhibit: CL reserve, mean, SD, CV and percentiles, with a total |
| `PRO.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins], [average])` | The bootstrap's draws: a row per simulation, a column per origin and the total (at most 100000 rows) |
| `PRO.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap exhibit, laid out as `PRO.ODP` |
| `PRO.MACK.BOOT.SIMS(triangle, [sims], [seed], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap's draws, laid out as `PRO.ODP.SIMS` |
| `PRO.MACK.CDR(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma], [pattern_ages])` | Merz-Wuthrich one-year view: CDR(1) SE and run-off SE |
| `PRO.MACK.CDR.YEARS(...)`, same arguments | CDR SE by future calendar year |
| `PRO.ODP.CDR(triangle, [sims], [seed], [percentiles], [tail], [process], [origins], [average])` | ODP bootstrap's simulated one-year CDR (negative is adverse) |
| `PRO.MACK.BOOT.CDR(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap's simulated one-year CDR |
| `PRO.CLARK(triangle, [exposure], [curve], [max_age], [origins])` | Clark's LDF method (or Cape Cod with exposure): reserve and SEs |
| `PRO.CLARK.PARAMS(triangle, [exposure], [curve], [max_age])` | Clark's fitted parameters |
| `PRO.ODP.GLM(triangle, [origins])` | The ODP model as a quasi-Poisson GLM: reserve, process and parameter SE, RMSEP and CV per origin, with a total |
| `PRO.ODP.GLM.PARAMS(triangle, [origins])` | Its coefficients and SEs (term, level), then Pearson's dispersion |
| `PRO.ODP.GLM.RESIDUALS(triangle, [by], [origins])` | Its residuals by cell, or grouped by `"origin"`, `"development"` or `"calendar"` |
| `PRO.GLM(data, response, terms, [family], [link], [exposure], [weights], [p])` | A GLM on a table with a header row: each coefficient's estimate, SE, statistic, p-value and exp(estimate), then deviance, dispersion, log-likelihood, AIC, BIC and the degrees of freedom (decision 41) |
| `PRO.GLM.PREDICT(data, response, terms, newdata, [family], [link], [exposure], [weights], [p])` | The fitted model's mean on each row of a table of new rows |
| `PRO.GLM.RELATIVITIES(data, response, terms, [family], [link], [exposure], [weights], [p], [level])` | A log-link GLM's relativities: each factor level against its base, a numeric term per unit, with a confidence interval |
| `PRO.TRI.INCREMENTAL(triangle)`, `PRO.TRI.CUMULATIVE(triangle)` | Incremental from cumulative, and back |
| `PRO.TRI.LINKRATIOS(triangle)` | Age-to-age link ratios |
| `PRO.TRI.LATEST(triangle)` | The latest diagonal |
| `PRO.EL(triangle, exposure, apriori, [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Expected loss method exhibit |
| `PRO.BF(triangle, exposure, apriori, [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Bornhuetter-Ferguson exhibit |
| `PRO.BENKTANDER(triangle, exposure, apriori, [iterations], [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Benktander exhibit |
| `PRO.CAPECOD(triangle, exposure, [trend], [decay], [average], [tail], [origins], [selected], [kind], [weight], [pattern_ages])` | Cape Cod exhibit |
| `PRO.DIAG.CL(origins, ages, latest, pattern, [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])` | Chain ladder on a latest diagonal (a bordereau) with a selected pattern |
| `PRO.DIAG.EL(origins, ages, latest, exposure, apriori, pattern, [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])` | Expected loss method on a latest diagonal |
| `PRO.DIAG.BF(origins, ages, latest, exposure, apriori, pattern, [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])` | Bornhuetter-Ferguson on a latest diagonal |
| `PRO.DIAG.BENKTANDER(origins, ages, latest, exposure, apriori, pattern, [iterations], [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])` | Benktander on a latest diagonal |
| `PRO.DIAG.CAPECOD(origins, ages, latest, exposure, pattern, [trend], [decay], [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])` | Cape Cod on a latest diagonal |
| `PRO.PATTERN.AVERAGE(pattern_ages, pattern, pattern_origin, [grain], [kind])` | A pattern averaged from one origin basis (accident months) to a coarser one |
| `PRO.VERSION()` | Version |
| `PRO.SEV.CDF(x, family, p1, p2, [p3])`, `PRO.SEV.SURVIVAL(...)` | Distribution or survival function at each x |
| `PRO.SEV.QUANTILE(p, family, p1, p2, [p3])` | Quantile (VaR) at each probability |
| `PRO.SEV.STATS(family, p1, p2, [p3])` | Family, mean, SD, CV |
| `PRO.LER(deductible, family, p1, p2, [p3])` | Loss elimination ratio of a deductible |
| `PRO.EVT.THRESHOLDS(data, [thresholds])` | Mean excess and the GPD's shape and modified scale at each threshold, with SEs |
| `PRO.EVT.FIT(data, threshold, [probabilities])` | GPD fitted above a threshold, with SEs, and the tail's VaR and TVaR |
| `PRO.EVT.HILL(data, [k])` | Hill's tail index from the k largest values, and its Pareto alpha |
| `PRO.PAYOUT.PATTERN(times, values, [kind], [shift], [rate])` | A payment pattern by time: share paid, paid to date, outstanding, discount |
| `PRO.PAYOUT.STATS(times, values, rate, [kind], [shift])` | Discount factor, duration and discounted capital years |
| `PRO.PAYOUT.PRICE(expected_loss, assets, coc, times, values, rate, [kind], [shift])` | A cost-of-capital price restated over the payout |
| `PRO.QUOTE(expected_loss, risk_premium, [assets], [expenses], [brokerage], [fixed], [limit], [quoted])` | Technical premium with its build-up and ratios, and the cover at quoted premiums |
| `PRO.DISTORTION(s, family, parameter, [r0])` | A distortion's g(s) |
| `PRO.DISTORTION.PRICE(values, [probs], family, parameter, [r0])` | A distortion's price of outcomes or draws, with mean, margin, loss ratio |
| `PRO.DISTORTION.CALIBRATE(values, [probs], premium, [families], [r0])` | Each family's parameter that gives a target premium |
| `PRO.COPULA.CDF(points, family, parameter, [nu])` | A copula's distribution function at each point (Gaussian in 2-D, Archimedean in any) |
| `PRO.COPULA.SIM(family, parameter, [sims], [dim], [seed], [nu])` | Uniforms simulated from a copula, one row a simulation |
| `PRO.IMAN.CONOVER(draws, correlation, [seed], [dof], [shuffle])` | Simulated columns reordered to a rank correlation, each keeping its values |
| `PRO.SWING(loss, basic, lcm, [minimum], [maximum], [share])` | Swing-rated or retro premium at each loss |
| `PRO.SLIDING.SCALE(loss, premium, commissions, loss_ratios)` | Sliding-scale ceding commission at each ceded loss |
| `PRO.PROFIT.COMMISSION(loss, premium, share, allowance)` | Profit commission at each ceded loss |
| `PRO.PROFIT.COMMISSION.TERM(losses, premium, share, allowance, [carry_years])` | Profit commission in each year of a term, deficits carried forward |
| `PRO.SLIDING.SCALE.TERM(losses, premium, commissions, loss_ratios, [carry], [carry_years])` | Sliding-scale commission in each year of a term, deficits and credits carried forward |
| `PRO.LAYER.TERMS(count_mean, [dispersion], family, p1, p2, [p3], layer_limit, layer_attachment, premium, [reinstatements], [commissions], [loss_ratios], [profit_share], [profit_allowance], [share], [policy_limit], [step], [points])` | A layer's premium, reinstatement premium and commissions in expectation |
| `PRO.SEV.FIT(losses, family, [output], [t], [truncation], [thresholds], [censored], [weights], [upper])` | Maximum likelihood fit to large losses: parameters with SEs, fit against the data, or the body families compared |
| `PRO.SEV.FIT.GROUPED(bounds, counts, family, [method], [output])` | A body family fitted to counts in size bands (maximum likelihood or minimum chi-square): parameters, SEs, chi-square test, or expected counts |
| `PRO.SEV.FIT.LEV(sizes, lev, family, [weights], [output])` | A body family fitted by the minimum distance to empirical limited expected values |
| `PRO.PARETO.EXTRAPOLATE(from_limit, from_attachment, to_limit, to_attachment, alpha, [truncation])` | Pareto extrapolation from one layer to others |
| `PRO.PARETO.ALPHA(limit1, attachment1, loss1, limit2, attachment2, loss2, [truncation])` | Alpha between two layers |
| `PRO.PARETO.ALPHA.FREQLAYER(threshold, frequency, limit, attachment, loss, [truncation])` | Alpha between a frequency and a layer |
| `PRO.PARETO.ALPHA.FREQS(threshold1, frequency1, threshold2, frequency2, [truncation])` | Alpha between two frequencies |
| `PRO.LEV(limit, family, p1, p2, [p3])` | Limited expected value per limit |
| `PRO.LAYER(limit, attachment, family, p1, p2, [p3])` | Expected loss to the layer per ground-up loss |
| `PRO.LAYER.VAR(limit, attachment, family, p1, p2, [p3])` | Variance of the layer loss per ground-up loss |
| `PRO.LAYER.CV(limit, attachment, family, p1, p2, [p3])` | CV of the layer loss given a loss above the attachment |
| `PRO.ILF(limit, basic_limit, family, p1, p2, [p3])` | Increased limit factor |
| `PRO.EXPOSURE.CURVE(x, curve, p1, [p2], [p3])` | Exposure curve G(x) |
| `PRO.EXPOSURE.SHARE(limit, attachment, mpl, curve, p1, [p2], [p3])` | Share of a risk's expected loss in a layer |
| `PRO.EXPOSURE.QUANTILE(p, curve, p1, [p2], [p3])` | Destruction rate at each probability |
| `PRO.EXPOSURE.STATS(curve, p1, [p2], [p3])` | b, g, mean rate, total loss probability |
| `PRO.SWISSRE.C(rate)` | Swiss Re c of a mean destruction rate |
| `PRO.SWISSRE.C.LAYER(limit, attachment, mpl, share)` | Swiss Re c of a layer's share |
| `PRO.CRED.STANDARD(p, k, [kind], [cv])` | Full credibility standard in claims |
| `PRO.CRED.Z(n, standard)` | Square root rule |
| `PRO.BUHLMANN(ratios, [weights], [estimator], [groups])` | Buhlmann-Straub by group |
| `PRO.BUHLMANN.PARAMS(ratios, [weights], [estimator])` | Buhlmann-Straub structure parameters |
| `PRO.YEARFRACTION(date)` | Excel date as a half-month year fraction |
| `PRO.ONLEVEL(effective, changes, from, to, [period], [term], [applies], [given])` | On-level factor of each period (parallelogram method) |
| `PRO.RATELEVEL(effective, changes, from, to, [period], [term], [applies], [given])` | One period's premium by rate level |
| `PRO.AVGDATE(from, to, [period], [term], [basis])` | Average written or earned date of each period |
| `PRO.TREND.FIT(times, values, [kind])` | Exponential or linear trend fit |
| `PRO.LR.EXPERIENCE(earned_premium, losses, [on_level], [premium_trend], [development], [loss_trend], [lae])` | Loss ratio method experience |
| `PRO.PP.EXPERIENCE(exposure, losses, [development], [loss_trend], [lae])` | Pure premium method experience |
| `PRO.INDICATED.CHANGE(loss_ratio, variable, profit, [fixed_ratio])` | Indicated change (loss ratio method) |
| `PRO.INDICATED.RATE(pure_premium, variable, profit, [fixed_per_exposure])` | Indicated rate (pure premium method) |
| `PRO.PERMISSIBLE.LR(variable, profit, [fixed_ratio])` | Permissible loss ratio |
| `PRO.TRENDED.PRESENT(prior_indicated, prior_implemented, trend)` | Trended present rates complement |
| `PRO.CRED.WEIGHTED(estimate, credibility, complement)` | Credibility-weighted estimate |
| `PRO.EXPENSES(ratios, [fixed_shares])` | Fixed and variable expense ratios |
| `PRO.BURNINGCOST(accident, amounts, limits, attachments, years, premiums, [development], [trend], [trend_to], [threshold], [agg_deductible], [agg_limit])` | Burning cost by accident year |
| `PRO.AGG.STATS(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], [probabilities], [step], [points])` | Annual aggregate loss: moments, VaR, TVaR |
| `PRO.AGG.LAYERS(count_mean, [dispersion], family, p1, p2, [p3], limits, attachments, [shares], [step], [points])` | Per-occurrence layer statistics |
| `PRO.AGG.COVER(count_mean, [dispersion], family, p1, p2, [p3], limits, retentions, [shares], [occ_limits], [occ_attachments], [occ_shares], [subject], [step], [points])` | Aggregate covers (a stop loss) on the annual total, net of occurrence layers or on their ceded total |
| `PRO.TOWER.PRICE(count_mean, [dispersion], family, p1, p2, [p3], [limit], [attachment], limits, attachments, [shares], [agg_deductibles], [agg_limits], [cover_limit], [cover_retention], distortion, parameter, [step], [points])` | A tower's views priced by a distortion on the annual grid |
| `PRO.AGG.PML(count_mean, [dispersion], family, p1, p2, [p3], [return_periods], [step], [points])` | Occurrence and aggregate PML curves by return period |
| `PRO.BAYES.ODP(triangle, [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins])` | Bayesian ODP chain ladder by MCMC (NUTS): reserves, posterior, diagnostics |
| `PRO.BAYES.BF(triangle, exposure, apriori, [cv], [output], [draws], [chains], [seed], [sims], [percentiles], [prior_sd], [process], [origins], [common_cv])` | Bayesian Bornhuetter-Ferguson by MCMC (NUTS), a gamma prior on each ultimate: reserves, ultimates, posterior, diagnostics |
| `PRO.PROFILE(sums_insured, expected_losses, limit, attachment, curve, p1, [p2], [p3], [lower], [upper], [spread], [retention], [lines])` | Exposure-rated risk profile by band, with a total |

- `triangle`: cumulative values only, one row per origin (oldest first),
  one column per age; blank or `#N/A` where unobserved.
- `average`: `"volume"` (default), `"simple"`, `"regression"` or
  `"geometric"` (the chain ladder only: Mack's model has no variance for
  it), or a row `kind, n_periods, drop_high, drop_low`, any cell blank:
  `{"volume",5,1,1}` averages the link ratios of the latest 5 calendar
  periods with the highest and lowest at each age left out. The ODP
  bootstrap functions take the same row as their last argument, with the
  kind `"volume"` or blank.
- `tail`: omitted for none, a factor such as `1.05`, or `"exponential"`,
  `"inverse_power"` or `"bondy"`.
- `origins`: labels for the exhibit's first column; omitted for 1, 2, ...
- `selected`: selected factors, a row with a cell per triangle column (a
  blank keeps the estimate); `kind`: `"ldf"` (default), `"cdf"` or
  `"percent"`; `weight`: 0 to 1 on the selection, 1 if omitted;
  `pattern_ages`: each selected value's age in months, so a monthly (or
  any other) pattern reaches the triangle's 12, 24, ... columns, read
  between its own ages (12, 24, ... by cell if omitted). The chain-ladder
  functions, Mack, the one-year view and the four expected-loss methods
  take all four, last and in this order (after `sigma` where there is
  one). With `pattern_ages`, an age-to-age factor runs to the next age
  given, as on the diagonal functions. For a pattern of accident months
  on accident years, average it with `PRO.PATTERN.AVERAGE` first and pass
  its result with `kind` `"percent"`.
- `sigma`: Mack's last sigma, `"log-linear"` (default) or `"mack"`.
- `exposure`: premium per origin, a row or column of positive values;
  `apriori`: the expected loss ratio, such as `0.75`, or a row or column
  of them, one per origin (oldest first on a triangle; on the diagonal
  functions, in the order of `origins`).
- `curve`: Clark's growth curve, `"loglogistic"` (default) or
  `"weibull"`; `max_age`: months development stops at, omitted for none.
- `sims` (default 10000, at most 1000000), `seed` (default 0; the same seed
  gives the same table), `percentiles` (a row or column such as
  `{0.75,0.95,0.995}`), `process` (`"gamma"` or `"none"`; Mack's
  bootstrap also takes `"lognormal"`, `"normal"` and `"residuals"`).
- The diagonal functions (`PRO.DIAG.*`) project one latest value per
  origin, as a bordereau holds it, with a pattern chosen elsewhere.
  `origins`: a year such as `2024` or an Excel date per origin, floored to
  `grain` (`"year"`, the default, `"semiannual"`, `"quarter"` or
  `"month"`); give `ages` (months, one per origin) or `valuation` (one
  Excel date), not both; `latest`: 0 for nothing reported, never blank.
  `pattern`: the selected values, such as a row of `PRO.CL.LDF`, read as
  `kind` at `pattern_ages` (12, 24, ... if omitted). A blank value is
  skipped, and an age-to-age factor runs to the next age given, so a
  blank widens the factor before it (unlike `selected` on `PRO.CL`, where
  a blank keeps the estimate); give 1 for no development. `pattern_origin`:
  the origin basis the pattern was measured on (`"month"` for accident
  months), averaged to `grain`; prefer a constant tail with it.
  `past_end`: an origin older than the pattern's end is an error
  (`"reject"`, the default), takes a factor of 1 (`"none"`, which
  also rescales a cumulative level below 100% for every origin) or the
  tail spread one step per origin period (`"tail"`). To apply a
  triangle's own chain ladder to another book, pass the first cells of
  its `PRO.CL.LDF` row as `pattern` and its last cell as `tail`.
  `PRO.PATTERN.AVERAGE` shows the averaged pattern that
  `pattern_origin` uses.

- `family` and `p1`, `p2`, `p3`: `"pareto"` (t, alpha, optional
  truncation), `"lognormal"` (meanlog, sdlog), `"lognormal_cv"` (mean,
  CV), `"gamma"` (shape, scale), `"gamma_cv"` (mean, CV), `"weibull"`
  (shape, scale), `"loglogistic"` (shape, scale), `"inverse_gamma"`
  (shape, scale), `"inverse_gaussian"` (mean, shape), `"burr"` (alpha,
  gamma, scale), `"lomax"` (alpha, scale: `S(x) = (1 + x/scale)^(-alpha)`,
  Bahnemann's Pareto), `"gpd"` (xi, beta), `"piecewise_pareto"` (a range of
  thresholds, a range of alphas, optionally a truncation point alone or
  with `"lp"`, the last piece only and the default, or `"wd"`, the whole
  distribution, in a second cell), `"gen_pareto"` (Riegel's: t,
  alpha_ini, alpha_tail).
- `curve` and `p1`, `p2`, `p3` (exposure curves): `"swiss_re"` (c, such
  as 1.5, 2, 3 or 4 for Swiss Re's Y1 to Y4, 5 for Lloyd's), `"mbbefd"`
  (b, g) or `"tabulated"` (a range of x from 0 to 1, a range of G(x) from
  0 to 1, and optionally `"linear"` or `"log-log"` interpolation).
  `PRO.EXPOSURE.CURVE` and `PRO.EXPOSURE.QUANTILE` tabulate several
  curves at once: a column of x (or u) against a row of c (or of b or g,
  the other a number or a row as long) gives one column per curve, and a
  row against a column one row per curve;
  `mpl`: a risk's maximum possible loss or sum insured, a number or a
  range that broadcasts against the limits and attachments.
- Credibility: `ratios` has one row per group and one column per period,
  blank where missing; `weights` (exposures) has the same shape, omitted
  for Buhlmann's model; `estimator` is `"unbiased"` (default) or
  `"iterative"` (Bichsel-Straub), as R's `actuar::cm`.
- Ratemaking dates (`effective`, `from`, `to`, `times`) are Excel dates
  or year fractions (`2011.5` is 1 July 2011); a period is `[from, to)`,
  so calendar year 2011 is `2011, 2012`. `period` is `"calendar"`
  (default) or `"policy"`, `term` the policy term in years (1 if
  omitted), `applies` `"written"` (default) or `"in_force"`, and `given`
  `"levels"` reads `changes` as cumulative levels.
- `PRO.BAYES.ODP`: `output` is `"reserves"` (default), `"params"` or
  `"diagnostics"`; check the diagnostics first (largest R-hat under
  1.01, no divergences, warnings `none`). `draws` per chain (1000) after
  as many warm-up draws, `chains` (4), `prior_sd` (10) on the log scale.
  Keep `prior_sd` wide: the priors are centred on 0, which for a
  development effect is the first column's level, and the late columns
  rest on one or two cells, so a tight prior pulls them up toward it. On
  RAA the total mean reserve is about 54,600 at 10, 59,100 at 3 and
  74,000 at 1 (the chain ladder's is 52,135).
  The last eight fits are kept, so the three views of one fit sample
  once.
- `PRO.BAYES.BF`: each origin's expected ultimate has a gamma prior with
  mean `apriori` times `exposure` and CV `cv` (0.1 if omitted), and the
  pattern is a share of the ultimate at each age. A small `cv` holds the
  a priori firmly and approaches `PRO.BF`; `"flat"` lets the data decide
  and gives `PRO.BAYES.ODP`'s reserves. `output` is `"reserves"`
  (default), `"ultimates"` (prior and posterior expected ultimate per
  origin), `"params"` (`alpha[w]`, the log ultimates, and `beta[d]`, the
  pattern's log share ratios to the first age) or `"diagnostics"`.
  `prior_sd` (10) is the pattern's prior SD; `common_cv`, if given, adds
  one multiplier common to every origin's prior ultimate with that CV
  (a shared error in the a priori loss ratio). The same draws, chains
  and seed arguments as `PRO.BAYES.ODP`, and its own last eight fits
  kept.
- `PRO.SEV.FIT`: `family` is `lognormal`, `gamma`, `weibull`, `burr`,
  `lomax` (fitted by maximum likelihood, each loss conditioned on exceeding its
  reporting threshold) or `pareto`, `piecewise_pareto`, `gen_pareto`
  (the R package Pareto's estimators, from the threshold `t`; a row of
  thresholds for the piecewise Pareto). `censored` is TRUE where a policy
  limit capped the loss. `output` `"params"` (default) lists the
  parameters in the order `p1`, `p2`, `p3` take them, so the estimates
  feed `PRO.SEV.CDF` and `PRO.LAYER`, then the log-likelihood, AIC, BIC
  (body families), the largest distance to the product-limit estimate
  whether the fit converged and a note (a Burr at its Pareto limit, a
  fit with no standard errors); `"gof"` gives the product-limit and the
  fit's survival and quantile at each loss (P-P and Q-Q); `"compare"`
  fits the five body families side by side (leave `family` blank). Read
  the upper Q-Q pairs before trusting a body family in the tail. `upper`
  truncates the body families from above: a loss is conditioned on lying
  at or below it (losses kept only up to a cap), one value or one per
  loss.
- `PRO.SEV.FIT.GROUPED`: `bounds` has one more cell than `counts`; a
  first bound above 0 is a deductible and a finite last bound a
  truncation from above, and a last bound left blank (or `"inf"`) opens
  the top band. `PRO.SEV.FIT.LEV` reads empirical limited means, exact
  from band counts and totals even under a policy limit at or above the
  last size: `(total below x + x * count above x) / count`.
- EVT: choose the threshold with `PRO.EVT.THRESHOLDS` (the lowest one
  above which the shape `xi` and the modified scale `beta - xi u` stay
  within their SEs and the mean excess is a straight line), then fit
  there with `PRO.EVT.FIT`; `xi` and `beta` are `PRO.SEV`'s `"gpd"` for
  the exceedances. A VaR or TVaR below the tail (`p` under `1 -
  P(exceed)`) says so in its cell, and the TVaR is infinite for `xi` of
  1 or more. The GPD takes `xi` (shape) where Pareto users read the
  alpha `1 / xi`.
- Payout: `times` are years from inception; `values` are shares paid
  at each time (`kind` `"shares"`, summing to 1), the share paid to date
  (`"paid"`, the last 1) or factors to ultimate (`"factors"`, such as a
  paid `PRO.CL.CDF` row, the last 1: add a tail as one more time).
  `shift` moves the pattern later (0.5 for losses occurring evenly over
  a year when the ages run from the accident date). Rates are annual
  effective. In `PRO.PAYOUT.PRICE` the cost of capital is a spread over
  the discount rate and capital runs off with the outstanding share; a
  distortion price over the payout is its premium times the discount
  factor.
- `PRO.QUOTE`: expenses and brokerage are shares of the premium itself,
  so they gross up, `(R + fixed) / (1 - expenses - brokerage)`; the risk
  premium `R` is a risk-loaded price such as `PRO.TOWER.PRICE`'s. Each
  quoted premium gets a column: what it leaves after the loads is the
  reinsurer's risk premium, and the margin and return on capital follow
  (capital needs `assets`; the rate on line and payback need `limit`).
- Distortions take `aggregate`'s names: `ccoc` (r), `ph` (rho), `wang`
  (lambda), `dual` (beta), `tvar` (p), `exp` (k), and with an intercept
  `r0` the capped `clin` (slope), `cll` (b), `lep` and `ly` (r). A
  family with several parameters takes them in one range: `bitvar`
  (`p0, p1, w`), `beta` (`a, b`), `wtdtvar` (two columns, levels and
  weights), and a capped family `r0` with its parameter, as
  `PRO.TOWER.PRICE` needs. These are not calibrated. Outcomes
  may be in any order; without `probs` they are equally likely draws, so
  a column of `PRO.ODP.SIMS` prices directly. To price a loss backed by
  assets `a`, cap the outcomes at `a` first (`=MIN(range, a)`): the
  premium then lies between the capped mean and `a`, the range
  `PRO.DISTORTION.CALIBRATE` solves in.
- Terms of several years: `losses` holds one term a row (a simulated
  path, or one history) and one year a column, in order; the result has
  the same shape. A carried amount is used oldest first and, with
  `carry_years`, dropped after that many later years. The premium is the
  same each year. Simulated years come from any sheet source, for
  example `PRO.ODP.SIMS` or a column of annual losses per year.
- Copulas: `gaussian` and `t` take one correlation for every pair or a
  square correlation matrix (the t also `nu`), `clayton`, `gumbel`,
  `frank` and `joe` their `theta`. `PRO.COPULA.SIM`'s row `i` is stream
  `i` of the seed, as the Python and R packages draw it; turn its
  uniforms into losses with `PRO.SEV.QUANTILE` column by column. The
  Gaussian's cdf is exact in two dimensions and within about `1e-5` in
  more (Genz's method), the Archimedean ones' in any, and the t has none
  in closed form. `PRO.IMAN.CONOVER` reorders simulations
  already on the sheet (`PRO.ODP.SIMS`, a column per line) and leaves
  each column's values as they are.
- Contract terms: amounts are at 100% and losses ceded at the placed
  share, as in `aggregate`. A sliding scale is two ranges, commission
  rates and their loss ratios (increasing), linear between and flat
  beyond. In `PRO.LAYER.TERMS` the deposit `premium` is at 100%; paid
  `reinstatements` (a row of rates) also set the annual limit to the
  limit times one more than their number, and cannot be combined with a
  sliding scale or a profit commission (the crate refuses it).
- Aggregates: `count_mean` claims a year with `dispersion` Var N / E N
  (1, Poisson, if omitted; a mixing CV c is `1 + c^2 * count_mean`); the
  grid is sized automatically unless `step` and `points` are given.
- `PRO.PROFILE`: one sum insured and expected loss per band; `p1` (and
  `p2`) may give each band its own Swiss Re c (or MBBEFD b, g); `lower`
  and `upper` spread a band's sums insured between bounds (`spread`:
  `"uniform"`, the default, `"tilted"` to keep the band's sum insured
  as the mean, or `"auto"`); `retention` and `lines` put a surplus
  treaty under the per-risk layer.
- `limit`, `attachment`, `basic_limit`: a number, a range or
  `"unlimited"`. A column of attachments against a row of limits gives a
  table, e.g. `=PRO.LAYER(B1:E1, A2:A4, "pareto", 500, 2)`.

Type a function in one cell: in Excel 365 its table spills from there
(older Excel needs it entered as an array formula, Ctrl+Shift+Enter, over
the result's size). A bad argument returns `#PROSPICIO! <reason>` in one
cell.
