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
openpyxl) writes a test workbook: RAA with every function in place and R
ChainLadder's totals beside them, plus Layers, ODP, ODP draws, Mack bootstrap, Expected loss (GenIns with premium), One-year and Triangle helpers sheets.

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
| `PRO.CL(triangle, [average], [tail], [origins], [selected], [kind], [weight])` | Chain ladder exhibit with a total row |
| `PRO.CL.LDF(triangle, [average], [tail], [selected], [kind], [weight])` | Age-to-age factors, one per column; the last is the tail |
| `PRO.CL.CDF(triangle, [average], [tail], [selected], [kind], [weight])` | Age-to-ultimate factors, one per column |
| `PRO.CL.ULTIMATE(...)`, `PRO.CL.RESERVE(...)` | One value per origin |
| `PRO.CL.FACTORS(triangle, [average], [tail], [selected], [kind], [weight], [sigma])` | LDF, CDF, sigma and standard error per link, and the tail |
| `PRO.MACK(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma])` | Mack exhibit: process, parameter and total SE and CV, with a total |
| `PRO.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins])` | ODP bootstrap exhibit: CL reserve, mean, SD, CV and percentiles, with a total |
| `PRO.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins])` | The bootstrap's draws: a row per simulation, a column per origin and the total (at most 100000 rows) |
| `PRO.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap exhibit, laid out as `PRO.ODP` |
| `PRO.MACK.BOOT.SIMS(triangle, [sims], [seed], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap's draws, laid out as `PRO.ODP.SIMS` |
| `PRO.MACK.CDR(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma])` | Merz-Wuthrich one-year view: CDR(1) SE and run-off SE |
| `PRO.MACK.CDR.YEARS(...)`, same arguments | CDR SE by future calendar year |
| `PRO.ODP.CDR(triangle, [sims], [seed], [percentiles], [tail], [process], [origins])` | ODP bootstrap's simulated one-year CDR (negative is adverse) |
| `PRO.MACK.BOOT.CDR(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])` | Mack bootstrap's simulated one-year CDR |
| `PRO.CLARK(triangle, [exposure], [curve], [max_age], [origins])` | Clark's LDF method (or Cape Cod with exposure): reserve and SEs |
| `PRO.CLARK.PARAMS(triangle, [exposure], [curve], [max_age])` | Clark's fitted parameters |
| `PRO.TRI.INCREMENTAL(triangle)`, `PRO.TRI.CUMULATIVE(triangle)` | Incremental from cumulative, and back |
| `PRO.TRI.LINKRATIOS(triangle)` | Age-to-age link ratios |
| `PRO.TRI.LATEST(triangle)` | The latest diagonal |
| `PRO.EL(triangle, exposure, apriori, [average], [tail], [origins])` | Expected loss method exhibit |
| `PRO.BF(triangle, exposure, apriori, [average], [tail], [origins])` | Bornhuetter-Ferguson exhibit |
| `PRO.BENKTANDER(triangle, exposure, apriori, [iterations], [average], [tail], [origins])` | Benktander exhibit |
| `PRO.CAPECOD(triangle, exposure, [trend], [decay], [average], [tail], [origins])` | Cape Cod exhibit |
| `PRO.VERSION()` | Version |
| `PRO.SEV.CDF(x, family, p1, p2, [p3])`, `PRO.SEV.SURVIVAL(...)` | Distribution or survival function at each x |
| `PRO.SEV.QUANTILE(p, family, p1, p2, [p3])` | Quantile (VaR) at each probability |
| `PRO.SEV.STATS(family, p1, p2, [p3])` | Family, mean, SD, CV |
| `PRO.LER(deductible, family, p1, p2, [p3])` | Loss elimination ratio of a deductible |
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
| `PRO.PROFILE(sums_insured, expected_losses, limit, attachment, curve, p1, [p2], [p3], [lower], [upper], [spread], [retention], [lines])` | Exposure-rated risk profile by band, with a total |

- `triangle`: cumulative values only, one row per origin (oldest first),
  one column per age; blank or `#N/A` where unobserved.
- `average`: `"volume"` (default), `"simple"` or `"regression"`.
- `tail`: omitted for none, a factor such as `1.05`, or `"exponential"`,
  `"inverse_power"` or `"bondy"`.
- `origins`: labels for the exhibit's first column; omitted for 1, 2, ...
- `selected`: selected factors, a row with a cell per triangle column (a
  blank keeps the estimate); `kind`: `"ldf"` (default), `"cdf"` or
  `"percent"`; `weight`: 0 to 1 on the selection, 1 if omitted.
- `sigma`: Mack's last sigma, `"log-linear"` (default) or `"mack"`.
- `exposure`: premium per origin, a row or column of positive values;
  `apriori`: the expected loss ratio, such as `0.75`.
- `curve`: Clark's growth curve, `"loglogistic"` (default) or
  `"weibull"`; `max_age`: months development stops at, omitted for none.
- `sims` (default 10000, at most 1000000), `seed` (default 0; the same seed
  gives the same table), `percentiles` (a row or column such as
  `{0.75,0.95,0.995}`), `process` (`"gamma"` or `"none"`; Mack's
  bootstrap also takes `"lognormal"`, `"normal"` and `"residuals"`).

- `family` and `p1`, `p2`, `p3`: `"pareto"` (t, alpha, optional
  truncation), `"lognormal"` (meanlog, sdlog), `"lognormal_cv"` (mean,
  CV), `"gamma"` (shape, scale), `"gamma_cv"` (mean, CV), `"weibull"`
  (shape, scale), `"loglogistic"` (shape, scale), `"inverse_gamma"`
  (shape, scale), `"inverse_gaussian"` (mean, shape), `"burr"` (alpha,
  gamma, scale), `"gpd"` (xi, beta), `"piecewise_pareto"` (a range of
  thresholds, a range of alphas, optionally a truncation point alone or
  with `"lp"`, the last piece only and the default, or `"wd"`, the whole
  distribution, in a second cell), `"gen_pareto"` (Riegel's: t,
  alpha_ini, alpha_tail).
- `curve` and `p1`, `p2`, `p3` (exposure curves): `"swiss_re"` (c, such
  as 1.5, 2, 3 or 4 for Swiss Re's Y1 to Y4, 5 for Lloyd's), `"mbbefd"`
  (b, g) or `"tabulated"` (a range of x from 0 to 1, a range of G(x) from
  0 to 1, and optionally `"linear"` or `"log-log"` interpolation);
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
