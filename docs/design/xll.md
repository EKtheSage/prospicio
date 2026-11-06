# Design note: the Excel add-in (XLL)

Status: **Prototype, tested by the user in Excel 365 (2026-10-15); built and kept by CI (2026-10-16)** · Lane: Excel (cloud) · Depends on: `triangle.md`, `reserving-v02.md` (the functions it wraps)

## Goal

prospicio's methods as worksheet functions in desktop Excel, from the same
Rust kernel as the Python and R packages: a triangle selected on a sheet
in, an exhibit that spills next to it out. No Python, R or server on the
user's machine; one file to load.

## What exists (prototype)

`crates/prospicio-xll` builds `prospicio.xll`, a 64-bit Windows DLL that
speaks the Excel C API (XLOPER12). Its 114 functions are in four groups
(decision 47): reserving (`PRO.RES.`), pricing (`PRO.PRC.`), capital
(`PRO.CAP.`) and the common functions every group uses, each group a
category of its own in Insert Function. The user manual,
`crates/prospicio-xll/README.md`, lists them by area (reserving,
severity and layers, exposure rating, experience rating and ratemaking,
aggregate and treaty terms, capital) with their arguments; the
catalogue in `crates/prospicio-xll/src/lib.rs` is the source of truth.
This section records what each group wraps and how it is checked.

`average` is `"volume"` (default), `"simple"` or `"regression"`; `tail` is
omitted (none), a number (a constant factor to ultimate at the oldest age)
or `"exponential"`, `"inverse_power"` or `"bondy"`. `sigma` is how Mack's
last sigma is filled in: `"log-linear"` (default, R's
`MackChainLadder(est.sigma = "log-linear")` and chainladder-python's) or
`"mack"` (Mack 1993). `selected` is a row with a cell per triangle column,
a blank keeping the estimated factor, as age-to-age factors (`kind`
`"ldf"`, the default), factors to ultimate (`"cdf"`) or percents developed
(`"percent"`), blended `weight * selected + (1 - weight) * estimated` (a
`SelectedPattern` on `Development`, `reserving-v02.md` decision 12). On
RAA, `sigma = "mack"` matches R's sigmas, their standard errors and the
total Mack SE, and the selections match chainladder-python's
`DevelopmentConstant` (`tests/mack_options.rs`).

`PRO.RES.ODP` wraps `OdpBootstrap::fit` (R's `BootChainLadder`): `sims`
defaults to 10,000 (at most 1,000,000), `seed` to 0, `percentiles` to
75%, 90%, 95% and 99.5%, `process` to `"gamma"` (`"none"` for parameter
error only); `tail` is read as for the chain ladder.

`PRO.RES.MACK.BOOT` wraps `MackBootstrap::fit` with centred residuals (the
crate's default, which reconciles with Mack's standard errors); `process`
is `"gamma"` (default), `"lognormal"`, `"normal"`, `"residuals"` or
`"none"`.

The one-year view: `PRO.RES.MACK.CDR` and `PRO.RES.MACK.CDR.YEARS` wrap
`MackFit::claims_development_result` (Merz and Wuthrich), and on RAA
every reserve, `CDR(1)S.E.`, Mack S.E. and `CDR(k)S.E.` R's
`CDR(MackChainLadder(RAA), dev = "all")` gives matches
(`tests/one_year.rs`). `PRO.RES.ODP.CDR` and `PRO.RES.MACK.BOOT.CDR` wrap the
bootstraps' `one_year` with the chain ladder (the bootstrap's tail, and
for Mack its average and sigma rule) as the re-reserving method; the CDR
is the opening ultimate less the re-estimated one, so a negative CDR is
adverse, and the default percentiles are 0.5%, 5%, 95% and 99.5%. The
re-reserving with the expected-loss methods is not exposed yet.

`PRO.RES.CLARK` wraps `ClarkLdf` without `exposure` and `ClarkCapeCod` with
it (R's `ClarkLDF` and `ClarkCapeCod`, `adol = TRUE`); `curve` is
`"loglogistic"` (default) or `"weibull"`, `max_age` an age in months
(omitted or `"inf"` for none). Against R run to convergence (`optim=tight`
in `validation/reference/reserving_clark_r.csv`, rel_tol 1e-5), every
parameter and every origin's and total's ultimate, reserve and standard
error match on RAA (LDF) and GenIns with premium (Cape Cod), both curves,
both `max_age` settings (`tests/clark.rs`).

The expected-loss methods wrap `ExpectedLoss`, `BornhuetterFerguson`,
`Benktander` and `CapeCod`. `exposure` (premium) is a row or column with
one positive value per triangle row, put on the triangle as an exposure
measure; each returns Origin, Exposure, Latest, CDF, A priori (the loss
ratio applied; Cape Cod's detrended estimate), Ultimate and Reserve, and a
total row. On GenIns with R's `?ClarkCapeCod` premium all 28 totals in
`validation/reference/reserving_expected_loss_python.csv`
(chainladder-python 0.10.1) match.

The diagonal functions (decision 39) wrap `LatestDiagonal` and the
methods' `fit_diagonal` (`reserving-v02.md` decisions 15 to 18).
`origins` is a year (a whole number below 10,000) or an Excel date per
origin, floored to `grain` (`"year"`, the default, `"semiannual"`,
`"quarter"` or `"month"`); exactly one of `ages` (months, one per
origin) and `valuation` (one Excel date) is given. `latest` is the
latest cumulative value per origin, 0 for nothing reported, and a blank
is an error. `pattern` holds the selected values, read as `kind`
(`"ldf"`, the default, `"cdf"` or `"percent"`), at `pattern_ages`
(12, 24, ... by cell if omitted); it is fully selected, so there is no
`weight`, and no `average` or `sigma`, since a diagonal has no link
ratios. `tail` is read as for the chain ladder, at decay 0.5.
`pattern_origin` names the origin basis the pattern was measured on
(`"month"` for a pattern of accident months) and averages it to
`grain` (`SelectedPattern::averaged_over_origins`); an estimated tail on
the averaged pattern is fitted on its monthly factors, so a constant tail
is the safer choice there. `past_end` says how an origin older than the
pattern's end is developed: `"reject"` (the default, an error),
`"none"` (a factor of 1, with no tail; a cumulative level other than
100% is then rescaled to 100% for every origin) or `"tail"` (the tail
spread from the pattern's end at one step per origin period). The
exhibit lists the origins oldest first over every period, whatever the
input order, labelled by year for accident years and as text
(`2024Q3`) for finer grains. The hand diagonal of
`validation/tests/reserving_latest_diagonal.rs` under every method,
chainladder-python 0.10.1's `DevelopmentConstant` on GenIns (the chain
ladder) and the CAS workers' compensation triangle (Bornhuetter-Ferguson,
Benktander, Cape Cod) in
`validation/reference/reserving_latest_diagonal_python.csv` at 1e-9, the
past-end, finer-grid and averaged-pattern cases of the Reserving lane's
validation, and each exhibit against the direct `prospicio_reserving`
fit to the bit are checked in `tests/latest_diagonal.rs`.

The severity and pricing functions build a `prospicio_prob::Dist` (as a
`SeverityDist`, so every family has exact limited moments) and wrap
`prospicio_pricing::layer` (`ilf`, `layer_cv`, `loss_elimination_ratio`)
and the `Distribution` and `Severity` traits. `family` and its parameters:
`"pareto"` (`t`, alpha, optional truncation as `p3`), `"lognormal"`
(meanlog, sdlog, as Excel's `LOGNORM.DIST`), `"lognormal_cv"` (mean, CV),
`"gamma"` (shape, scale), `"gamma_cv"` (mean, CV), `"weibull"` (shape,
scale), `"loglogistic"` (shape, scale), `"inverse_gamma"` (shape, scale),
`"inverse_gaussian"` (mean, shape), `"burr"` (alpha, gamma, scale as
`p3`), `"gpd"` (xi, beta). The sixth argument of the layer functions,
first `truncation`, is now `p3`: same position and meaning for the Pareto
(decision 17). On the gamma, log-logistic, lognormal and Weibull every
LEV, layer and stop loss in `validation/reference/severity_mpmath.csv`
matches at its tolerance (`tests/severities.rs`). The Pareto helpers wrap `prospicio_pricing::layer`'s
`pareto_extrapolation` and three alpha solvers; every case in
`validation/reference/pareto_rating_r.csv` (R package Pareto 2.4.5, 58,
truncated ones included) matches (`tests/pareto_helpers.rs`). TVaR is not offered for
a parametric severity: neither the R nor the Python package has it
(theirs is on grid, sampled and predictive distributions). Limits and attachments are a number, a range or `"unlimited"`;
blank cells give blank results.

**Checks.** `cargo test -p prospicio-xll` runs the worksheet functions on
RAA through the same catalogue Excel calls, against `prospicio-reserving`
and R ChainLadder's totals. `cargo xtask xll` builds the add-in and, when
MinGW (and off Windows, Wine) are found, runs `harness/fake_excel.c`: a
host that exports `MdCallBack12` as Excel does, loads the XLL, checks the
registrations and calls the functions with real XLOPER12 arguments (RAA's
Mack totals against R to 1e-6, an error, 2000 calls and frees, and a
2 x 2 Pareto layer table against the R package Pareto).
`cargo test` checks `PRO.RES.ODP` against `OdpBootstrap` draw for draw and
its RAA total (mean, SD, 75%, 95% and 99.5% percentiles, gamma and
parameter-only) against R ChainLadder's `BootChainLadder` targets in
`validation/reference/reserving_bootstrap_r.csv`, and the harness repeats
the mean and SD through XLOPER12 values. It also checks the pricing functions against the crates, the R
package Pareto's vignette (4000 xs 1000 under Pareto(500, 2) is 200) and
mpmath's lognormal targets (`validation/reference/severity_mpmath.csv`).
`harness/make_workbook.py` writes a workbook with RAA and every function
in place, with R's totals beside them, and a Layers sheet, for a test in
real Excel.

## Decisions

1. **A native XLL, beside the WASM/Office.js add-in, not instead of it.**
   An XLL runs in desktop Excel on Windows only, but in process, with no
   web runtime, on all of Excel's calculation threads, and offline. The
   Office.js add-in (`roadmap.md`, WASM) remains the route to Excel on the
   web and the Mac. Both are thin front ends over the same crates.
2. **64-bit Excel only.** The default install since Office 2019; a 32-bit
   build is extra work nobody has asked for.
3. **The Excel C API by hand, no XLL framework crate.** The surface we need
   (XLOPER12 values and arrays, `xlfRegister`, `xlGetName`, `xlFree`,
   `xlAutoOpen`/`xlAutoFree12`) is a few hundred lines in `src/xll.rs`.
   Everything else (`sheet`, `reserving`, the `FUNCTIONS` catalogue) is
   portable Rust tested on any host.
4. **Values in, values out, thread safe** (the simulations and MCMC fits
   asynchronous instead, decision 48). Every argument and result is
   type `Q` (values, never a reference) and every function is registered
   thread safe (`$`); no function reads the sheet or keeps state. Results
   are allocated per call and freed in `xlAutoFree12`.
5. **Arrays spill.** A table comes back as an array: in Excel 365 it
   spills from the one cell the formula is typed in; in older versions it
   is entered as an array formula (Ctrl+Shift+Enter) over the result's
   size. The test workbook saves its formulas as 365 dynamic arrays
   (`cm="1"` and `xl/metadata.xml`, patched in after openpyxl, which
   writes only legacy array formulas).
6. **The triangle is the values only.** One row per origin (oldest first),
   one column per development age; blank, empty text or `#N/A` where
   unobserved, and nothing observed after a blank in a row. Labels stay on
   the sheet; `origins` passes origin labels into the exhibit. Ages are
   read as development periods, so yearly, quarterly and monthly
   triangles all work; link labels say 12-24 and on.
7. **Errors come back as text** (confirmed, decision 11). A bad argument returns one cell
   `#PROSPICIO! <reason>`, since Excel's `#VALUE!` cannot say why. A panic
   is caught and reported the same way; it never unwinds into Excel.
8. **Built from Linux or Windows.** `cargo xtask xll` uses
   `x86_64-pc-windows-msvc` on Windows and `x86_64-pc-windows-gnu` with
   MinGW's linker elsewhere; the result needs only system DLLs.
9. **Distribution: a local file for now** (user, 2026-10-15): unsigned,
   built by `cargo xtask xll`, handed over for testing. A CI artifact or a
   release asset, and code signing, come after the user has tried it.

10. **The prefix is `PRO.`** (user, 2026-10-15): short to type. Every
    function keeps it, since renaming breaks saved workbooks. Decision 47
    adds a group after it, and the old names are gone in 0.0.9.
11. **Errors stay text** (user, 2026-10-15): `#PROSPICIO! <reason>`, which
    says why; `ISERROR` does not see it (test with `LEFT(x, 11) =
    "#PROSPICIO!"`).
12. **A severity is a family name and two parameters**, not an object
    held in a cell: every pricing function takes `family, p1, p2,
    [truncation]`, so the parameters stay in cells the sheet can change
    and audit, and no state lives between calls (decision 4). Families
    are added by name; the Pareto and the lognormal come first (user,
    2026-10-15).
13. **Limits and attachments broadcast as Excel's arithmetic does.** Two
    ranges of one size combine cell by cell; a side of length 1 stretches
    to the other's, so a column of attachments against a row of limits is
    a layer table from one formula. `"unlimited"` (or `"inf"`) is no
    limit, a blank cell gives a blank, and a value Excel cannot hold (an
    infinite CV) is left blank.
14. **The ODP bootstrap is reproducible and bounded.** The same `seed`
    gives the same table on every recalculation and machine (the
    workspace's RNG contract, `rng.md`), so a sheet does not move when
    Excel recalculates; the function is not volatile. `sims` is capped at
    1,000,000 so a recalculation cannot hang Excel (20,000 on RAA take
    about 0.2 s). The total's percentiles are of the simulated total, not
    the sum of the origins' percentiles; percentiles are the inverse of
    the empirical distribution function (R `type = 1`), as `Sampled`'s.
    `PRO.RES.ODP.SIMS` returns the draws themselves, the same ones for the
    same `sims`, `seed`, `tail` and `process`, so a sheet can chart them or
    compute its own statistics; it is capped at 100,000 rows (about 1 s and
    35 MB of cells in Excel), under the sheet's 1,048,576 rows.

15. **Distribution: CI builds it, a draft release carries it** (user,
    2026-10-16, "the next items"). `.github/workflows/xll.yml` builds the
    add-in on Ubuntu with MinGW and runs the fake-Excel smoke test under
    Wine on every PR and `main` push that touches its inputs, and keeps
    the `.xll`, the test workbook, the README and a SHA-256 as the
    `prospicio-xll` artifact (30 days). A version tag attaches the same
    bundle to that tag's GitHub release, created as a **draft** when there
    is none, so nothing is public until the owner publishes it. Its own
    workflow file, so `ci.yml` and `release.yml` are unchanged. The
    artifact can be downloaded from the run's page for testing instead of
    a file passed by hand.
16. **Signing when a certificate is given, not before.** With the
    repository secrets `XLL_SIGNING_PFX` (a base64 PKCS#12 code-signing
    certificate) and `XLL_SIGNING_PASSWORD`, the workflow signs the XLL
    with `osslsigncode` (Authenticode, SHA-256, a DigiCert timestamp) and
    checks the signature; without them it ships unsigned, as now. The step
    was checked with a self-signed certificate: the signed XLL verifies
    and passes the smoke test. The secrets fit a certificate from an
    organization's own CA; a publicly trusted one lives in an HSM and
    needs a signing service's step instead (Open, below). A signature lets
    an organization trust the publisher by policy instead of each user
    ticking Unblock; it does not by itself lift Excel's block on add-ins
    from the internet.

17. **New arguments go last.** An option added to an existing function is
    appended after its present arguments, so every saved formula still
    means what it did (Excel passes an omitted argument as missing).
    `call` treats arguments past those given as omitted, and
    `tests/catalogue.rs` parses `src/xll.rs` to check that every export
    takes exactly its catalogue entry's arguments.
18. **An exposure curve is named as a severity is** (decision 12):
    `curve, p1, [p2], [p3]`, with `"swiss_re"` (`c`), `"mbbefd"` (`b`,
    `g`) and `"tabulated"` (a range of `x`, a range of `G(x)` and the
    interpolation), so a sheet keeps the parameters in cells.
    `PRO.PRC.EXPOSURE.SHARE` broadcasts its limits, attachments and MPLs
    against each other (decision 13), so a column of a risk profile's sums
    insured gives each band's share of one layer, and a row of layers
    against it a table. A severity's own curve, `LEV(x MPL) / LEV(MPL)`,
    is `PRO.PRC.LEV` over `PRO.PRC.LEV` and needs no function of its own.
    `PRO.PRC.PROFILE` takes one curve, or a Swiss Re `c` or an MBBEFD `b` and
    `g` per band (a range with one cell per band), as Python's
    `RiskProfile` takes one curve or a list. A band spreads its sums
    insured between `lower` and `upper` when it has both; the spread is
    uniform unless `spread` says `"tilted"` or `"auto"` (Python's default
    too, without its warning: the exhibit's mean SI column shows where a
    band's mean moved).
19. **A date is a year fraction or an Excel date.** Ratemaking works in
    year fractions on Werner and Modlin's half-month grid (`2011.5` is 1
    July 2011, `ratemaking.md`), but a sheet holds dates. Every date
    argument takes either: a number of 10,000 or more is an Excel date
    (10,000 is 18 May 1927) and is put on the grid as `year_fraction`
    puts a calendar date (30 June is 1 July), anything smaller is a year
    fraction. No ratemaking year reaches 10,000 and no Excel date of
    interest falls below it, so the two never mix. `PRO.YEARFRACTION`
    shows the conversion. `PRO.PRC.ONLEVEL`'s `given` reads its changes as
    the cumulative levels an exhibit prints (`RateHistory::from_levels`),
    so a sheet can reproduce a book's rounded factors. An accident date
    is the exception: `PRO.PRC.BURNINGCOST` puts an Excel date at its exact
    decimal year (day of the year over the days in it), since the grid
    would move 31 December into the next accident year.
20. **An aggregate is a count by mean and dispersion and a severity by
    family** (decision 12): `count_mean, [dispersion]` picks the Panjer
    class (dispersion 1 Poisson, above negative binomial, below binomial),
    so `aggregate`'s mixing CV `c` is the dispersion `1 + c² E N`. The
    FFT grid is sized as `prospicio_aggregate::fft_auto` sizes it ground
    up; for a finite layer the severity's grid splits the layer's top
    into about 4,096 steps and the total runs to its mean plus 12 SDs
    (at most 2^20 points). `step` and `points` override both, rounding the
    severity onto `points` buckets with the mass above lumped at the top,
    as `aggregate` caps it, so a sheet can reproduce `aggregate`'s
    numbers. Ground-up moments are exact when finite (the grid's mean is
    shown beside them); a layer's are the grid's; VaR and TVaR are the
    grid's. The count is unwrapped from `PanjerClass` into its concrete
    family, whose third cumulant is closed form
    (`knowledge/findings/panjer-class-third-cumulant.md`).
21. **MCMC on a sheet: one function, three views, kept fits** (user,
    2026-10-16, "how you do Bayesian MCMC methods"). `PRO.RES.BAYES.ODP` runs
    the Reserving lane's `BayesOdp` (England and Verrall's ODP chain
    ladder, NUTS on the posterior whitened at its mode, chains on worker
    threads) at its defaults: four chains of 1,000 warm-up and 1,000 kept
    draws, target acceptance 0.95, normal priors of SD 10 on the log
    scale, Gamma process error, 10,000 simulations. Its `output` picks the
    view of one fit: the reserve exhibit (`PRO.RES.ODP`'s layout), each
    parameter's posterior (mean, SD, 5/50/95% quantiles, R-hat, bulk and
    tail ESS) or the diagnostics (largest R-hat, smallest ESS,
    divergences, the plugged-in scale, and the library's warnings as text,
    so a sheet sees a fit that did not converge). One function rather than
    three keeps the arguments in one place. The last eight fits are kept,
    keyed on every input: a fit is a pure function of its inputs and seed
    (decision 14), so a kept fit is exactly what a refit gives and
    decision 4 still holds for the results; a sheet showing all three
    views samples once. A default fit on RAA takes about 0.25 s in a
    release build, a kept one 30 µs. Draws are capped at 20,000 per chain
    and chains at 8. `prior_sd` is one SD for the intercept, origin and
    development effects, all centred on 0; a development effect of 0 is
    the first column's level, so a tight `prior_sd` lifts the late,
    thinly observed columns toward it and the reserve with them (on RAA
    the last effect's posterior mean is -5.1 at 10, -2.4 at 3 and -0.9
    at 1, the total mean reserve 54,600, 59,100 and 74,000). The help and
    the README say to keep it wide (user, 2026-10-19;
    `knowledge/findings/bayes-odp-tight-prior.md`). Other Bayesian models follow the same pattern
    when wanted: the BF with a normalised pattern does (decision 38);
    Meyers' and the compartmental models do not yet.
22. **PML curves by return period, occurrence beside aggregate** (user,
    2026-10-16, the parity plan's next items). `PRO.CAP.AGG.PML` takes the
    count and severity as `PRO.CAP.AGG.STATS` does (decision 20) and return
    periods (2 to 1,000 years if omitted; a value below 1 is read as the
    annual probability, as `aggregate`'s `oep` takes it). Each row gives
    the return period, the probability `p`, the occurrence PML (the loss
    with `1 − G_N(F(x)) = p`, exact for any Panjer count through its pgf,
    `prospicio_aggregate::pml`), the aggregate PML (the ground-up total's
    VaR at `1 − p` on the FFT grid, sized as in decision 20 or by `step`
    and `points`), the severity's exceedance at the occurrence PML and
    the occurrence return period `1 / (E N · S(x))`. A probability at or
    above `P(N ≥ 1)` is an error naming that ceiling. The two curves sit
    side by side because that is the comparison a cat or per-risk
    exhibit makes; a layer's occurrence PML is the curve clipped to the
    layer, so it takes no layer arguments.
23. **An aggregate cover sits on a chosen annual total** (user,
    2026-10-19, the parity plan's next item). `PRO.CAP.AGG.COVER` takes the
    count and severity as `PRO.CAP.AGG.STATS` does (decision 20), then the
    covers as `PRO.CAP.AGG.LAYERS` takes layers: annual `limits` (a row,
    `"unlimited"` allowed), `retentions` and `shares`, each one value or
    one per limit, one column per cover. Optional per-occurrence layers
    (`occ_limits`, `occ_attachments`, `occ_shares`) inure to the covers:
    by default the covers see the annual total net of them, as
    `aggregate`'s `occurrence net of ... aggregate ceded to ...` does, and
    with `subject` `"ceded"` they see the layers' ceded total
    (`occurrence ceded to`), an annual deductible and limit on an excess
    of loss treaty. The rows are
    `aggregate`'s `reins_stats_df` for its `agg` block (Pr attach, Pr
    detach, the ceded mean, SD, CV and skewness, loss on line) under the
    subject total's mean. The subject is on the grid `PRO.CAP.AGG.STATS`
    sizes ground up (the net or ceded total is lighter, so the gross
    grid covers it); `step` and `points` set it, the severity rounded onto
    as many points as the total. A cover's retention in loss-ratio terms
    is a sheet product (premium times ratio) rather than an argument.
    Against `aggregate` 1.0.1, books R and T (the policy limit `1000 xs
    0` given as one more occurrence layer, unlimited above 1,000, so the
    net is the capped severity's) match at the reference's tolerances.
24. **A tower is priced view by view on the grid** (user, 2026-10-19,
    the parity plan's last item). `PRO.PRC.TOWER.PRICE` wraps
    `Tower::on_grid` and `TowerGrids::price`: the count and severity as
    in decision 20, with the gross per-claim `limit` and `attachment` of
    `PRO.CAP.AGG.STATS` (a policy limit, `aggregate`'s `1000 xs 0`); one
    stage of per-occurrence layers (`limits`, `attachments`, `shares`,
    each layer's annual `agg_deductibles` and `agg_limits`); and an
    optional stop loss on the annual total net of them (`cover_limit`,
    `cover_retention`) as a second stage. One row per view (gross, each
    layer, the layers' total when there are several, the cover, net),
    each priced on its own by the distortion, as `reins_price_df` does:
    loss, SD, CV, premium, margin, loss ratio, and for a layer or the
    cover the rate on line (premium over share times the smaller of the
    occurrence and annual limit). The distortion is one of the five
    standard families or the exponential by `aggregate`'s short name and
    a single parameter; the two-parameter families (`clin`, `cll`, `lep`,
    `ly`, bi-TVaR) are left out until asked. Where the layers' total or
    the net is not one compound (a layer with annual terms) the row says
    so in place of numbers. A layer whose boundaries miss the grid's
    points keeps an exact mean and a shape smeared by up to a step;
    `step` and `points` put them on points. On `aggregate`'s books C, R,
    S and T under PH 0.7 every view's loss and premium matches
    `reins_price_df` at the reference's tolerances. Capital allocation
    and the natural allocation stay out (they want a portfolio object).
25. **A severity is fitted from a column of losses** (user, 2026-10-19,
    the parity plan's sixth group). `PRO.SEV.FIT(losses, family,
    [output], [t], [truncation], [thresholds], [censored], [weights])`
    fits a body family (lognormal, gamma, Weibull, Burr) by
    `fit_severity`, each loss conditioned on exceeding its reporting
    threshold and a censored loss (TRUE where a policy limit capped it)
    known only to exceed its value, or a Pareto family (the Pareto, the
    piecewise Pareto from a row of thresholds, Riegel's generalized
    Pareto) by the R package Pareto's estimators. The per-loss columns
    are one value or a range of the losses' shape, and a row whose loss is
    blank is skipped, so a long selected column works. `"params"` lists the
    parameters in the order `p1`, `p2`, `p3` take them, with standard
    errors from the observed information, then the log-likelihood, AIC and
    BIC (body families), the largest distance to the product-limit
    estimate, convergence and a note; the estimates feed `PRO.SEV.CDF`,
    `PRO.PRC.LAYER` and the rest. `"gof"` is the product-limit estimate beside
    the fit's survival and quantile at each loss (P-P and Q-Q), and
    `"compare"` the four body families side by side. The note flags a fit
    with no standard errors (a boundary of the family), and names a Burr
    at its Pareto limit (`alpha → 0`, `gamma → ∞`), which on Pareto-like
    large losses has the highest likelihood of the four and must not be
    picked on likelihood alone (`knowledge/findings/burr-fit-pareto-limit.md`).
    The Pareto fits match R Pareto 2.4.5 (`pareto_fit_r.csv`, with
    thresholds, censoring, weights and both truncations) at its
    tolerances.
26. **A GPD tail is chosen, fitted and read in three functions** (user,
    2026-10-19, the sixth group). `PRO.EVT.THRESHOLDS(data,
    [thresholds])` is `threshold_diagnostics`: per threshold the count
    above, the mean excess with its standard error, and the GPD's shape,
    scale and modified scale `beta - xi u` with standard errors, the
    numbers behind R `ismev`'s `mrl.plot` and `gpd.fitrange`, so a sheet
    chart replaces the plots. Thresholds default to the data's 50% to 95%
    quantiles in steps of 5% (R's type 7). `PRO.EVT.FIT(data, threshold,
    [probabilities])` fits the GPD to the exceedances over the threshold
    the user chose (choosing stays the user's, as in the crate), with the
    observed information's standard errors, and gives the
    peaks-over-threshold tail's VaR and TVaR (0.99, 0.995, 0.999 if
    omitted). A probability below the tail gets a message in its cell, not
    an error for the table. `PRO.EVT.HILL(data, [k])` gives Hill's index
    with its threshold and the Pareto alpha `1 / xi`. The fitted `xi` and
    `beta` are `PRO.SEV`'s `"gpd"` for the exceedances. Against `ismev`
    1.43 on its nine cases every quantity matches at the reference's
    tolerances.
27. **A payment pattern is a row of times and a row of values** (user,
    2026-10-19, the sixth group), over `prospicio_pricing::payout`
    (`time-value.md`). `kind` reads the values as shares paid at each
    time (summing to 1), the share paid to date or factors to ultimate,
    so a paid `PRO.RES.CL.CDF` row feeds the pattern directly; a tail goes in
    as one more time, never placed silently (`time-value.md` decision 2).
    `PRO.PRC.PAYOUT.PATTERN` shows the pattern with the share outstanding and
    the discount at each time, `PRO.PRC.PAYOUT.STATS` its discount factor,
    duration and discounted capital years, and `PRO.PRC.PAYOUT.PRICE` the
    cost-of-capital price over the payout from the expected loss, the
    one-year assets and the cost of capital: `P = E d + coc (a - P) A`.
    The one-period price comes back for one payment within the year at a
    zero rate. A distortion price over the payout is its premium times the
    discount factor (`time-value.md` decision 6), a sheet product, so it
    has no function of its own.
28. **A quote starts from a risk premium on the sheet** (user,
    2026-10-19, the sixth group). `PRO.PRC.QUOTE` wraps
    `prospicio_pricing::quote`: the technical premium
    `(R + fixed) / (1 - expenses - brokerage)` of a risk premium `R`
    (a `PRO.PRC.TOWER.PRICE` premium, or any risk-loaded price), its build-up
    (expenses, brokerage, the premium net of brokerage, the margin), the
    capital and its return when `assets` is given, and the market's
    ratios (loss ratio, multiple, and rate on line and payback when
    `limit` is given). The function takes the price's parts rather than
    pricing a distribution itself, so the quote applies to any price the
    sheet holds. Each value of `quoted` adds a column with the same
    cover at that premium (`Quote::at`): what the quote leaves after the
    loads is the reinsurer's risk premium. A quote that leaves no capital
    is an error. The guide `reinsurance-price.md`'s middle layer (risk
    premium 1.24, expected loss 0.586, limit 5, 3% expenses, 10%
    brokerage) comes back at its printed 1.425, 28.5%, payback 3.5,
    multiple 2.43 and loss ratio 41.1%.
29. **A distortion is a family name and a parameter; outcomes are a
    range** (user, 2026-10-19, the sixth group), over
    `prospicio_prob::distortion`. `PRO.CAP.DISTORTION(s, family, parameter,
    [r0])` gives `g(s)` over a range of `s`, `PRO.CAP.DISTORTION.PRICE` the
    risk measure of outcomes with probabilities, or of equally likely
    draws when `probs` is omitted, so a column of simulations prices
    directly, and `PRO.CAP.DISTORTION.CALIBRATE` each family's parameter
    that gives a target premium (the five standard families of
    Mildenhall and Major unless named). Families take `aggregate`'s
    short names, as `PRO.PRC.TOWER.PRICE` does (decision 24), and here also
    the four capped families with their intercept `r0`; the bi-TVaR,
    beta and weighted TVaR, which take more than one parameter, came
    later (decision 31). A family that cannot reach the premium gets a
    message in its row. Assets enter by capping the outcomes on the
    sheet, as `calibrate` documents, rather than as an argument. Against
    `aggregate` 1.0.1, `g(s)` of nine families and the InsCo prices and
    calibrations match at the reference's tolerances.
30. **Contract terms are evaluated at a loss and in expectation** (user,
    2026-10-19, the sixth group's last item), over
    `prospicio_aggregate::Layer`. `PRO.PRC.SWING` (a swing-rated or retro
    premium), `PRO.PRC.SLIDING.SCALE` (a ceding commission from anchors given
    as two ranges, rates and their increasing loss ratios) and
    `PRO.PRC.PROFIT.COMMISSION` take a range of losses and give the term at
    each, as `aggregate`'s `contract_terms` does, so a sheet can tabulate
    or chart a term against the loss. `PRO.PRC.LAYER.TERMS` gives a layer's
    terms in expectation over its annual loss on the grid
    (`Tower::on_grid`): the expected ceded loss, premium, reinstatement
    premium, ceding and profit commissions, the premium net of them, the
    margin and the loss ratio. Amounts are at 100% and losses at the placed
    share, as in `aggregate`; paid reinstatements set the annual limit and
    cannot be combined with a sliding scale or a profit commission, which
    the crate refuses. Multi-year terms with carried deficits
    (`Tower::apply_term`) need simulated years and are left out until
    asked. Against `aggregate` 1.0.1 the 475 values of
    `contract_terms_aggregate.csv` and the expected reinstatement premiums
    of books RI2 and RI10 match at the reference's tolerances.
31. **A distortion with several parameters takes them in one range**
    (user, 2026-10-19, the seventh group). The `parameter` argument of
    `PRO.CAP.DISTORTION`, `PRO.CAP.DISTORTION.PRICE` and `PRO.PRC.TOWER.PRICE` is a
    number for a one-parameter family, three cells `p0, p1, w` for the
    bi-TVaR (`bitvar`), two `a, b` for the beta, and two columns (levels
    | weights) or two rows for the weighted TVaR (`wtdtvar`), in
    `aggregate`'s order; no function gains an argument. The capped
    families keep `r0` beside the parameter and also take the two
    together in one range, which is how `PRO.PRC.TOWER.PRICE`, without an
    `r0` argument, reaches them. The families with several parameters
    are not calibrated: a target premium fixes one number, not two or
    three, and `PRO.CAP.DISTORTION.CALIBRATE` says so. Against `aggregate`
    1.0.1 the 48 `g(s)` values of the bi-TVaR, beta and weighted TVaR in
    `distortion_aggregate.csv` match at its tolerances, and CCoC at `r`
    prices as the bi-TVaR `(0, 1, r / (1 + r))` on a sheet and in every
    view of a tower.
32. **Copulas are rebuilt from their parameters in each call** (user,
    2026-10-19, the seventh group), over `prospicio_prob::copula`.
    `PRO.CAP.COPULA.CDF(points, family, parameter, [nu])` gives `C(u)` at
    each row of `points`; `PRO.CAP.COPULA.SIM(family, parameter, [sims],
    [dim], [seed], [nu])` simulates uniforms, row `i` from stream `i` of
    the seed as `copula::simulate` draws them, so a sheet agrees with the
    Python and R packages draw for draw; `PRO.CAP.IMAN.CONOVER(draws,
    correlation, [seed], [dof], [shuffle])` reorders simulated columns
    to a rank correlation with normal or t scores. The Gaussian and t
    take one correlation for every pair or a square matrix; the four
    Archimedean families take `theta`. The marginals are left to the
    sheet (`PRO.SEV.QUANTILE` on the uniforms) rather than taken as
    arguments, so any family, a fitted one included, joins without the
    copula functions knowing it. Nested Archimedean copulas and vines,
    which are a tree of parameters rather than a range, are left out
    until asked. Against mpmath at 30 digits the 39 Gaussian and
    Archimedean cdf values of `copula_mpmath.csv` (one Archimedean point
    per family in three dimensions) match at the reference's
    tolerances; the simulated uniforms are the crate's, and the
    Iman–Conover result is `iman_conover`'s.
33. **A term of several years is a row of losses** (user, 2026-10-19,
    the seventh group's last item), over
    `prospicio_aggregate::Layer::profit_commissions_over` and
    `ceding_commissions_over`. `PRO.PRC.PROFIT.COMMISSION.TERM(losses,
    premium, share, allowance, [carry_years])` and
    `PRO.PRC.SLIDING.SCALE.TERM(losses, premium, commissions, loss_ratios,
    [carry], [carry_years])` read one term from each row of `losses`,
    its columns the years in order, and give each year's commission in
    the same shape, deficits (and for the sliding scale credits, both by
    default) carried forward as `aggregate.md` defines them: used oldest
    first, dropped after `carry_years` later years, or kept until used
    up. The years' losses come from the sheet, so the paths may be
    independent years or share a trend, as `Tower::apply_term` allows,
    and the add-in keeps no simulation (decision 4). The premium is the
    same in every year. `aggregate` has no multi-year term, so the
    checks are the crate's hand-worked cases and, on 400 three-year
    paths, `Tower::apply_aggregate_term`'s commissions year by year.
34. **Link-ratio exclusions go in the `average` argument** (user,
    2026-10-20, "move forward"; the eighth group), over
    `Development::n_periods`, `drop_high` and `drop_low` and the geometric
    average (`reserving-v02.md` decision 13). `average` is still a kind,
    `"volume"`, `"simple"`, `"regression"` or now `"geometric"`, or a row
    `kind, n_periods, drop_high, drop_low`: `{"volume",5,1,1}` is the
    five-year volume average excluding the high and low link ratio. The
    sixteen chain-ladder, Mack, expected-loss and Mack bootstrap functions
    that take `average` get the exclusions without a new argument, and a
    formula written before keeps its meaning. The ODP bootstrap
    functions (`PRO.RES.ODP`, `.SIMS`, `.CDR`), which had no `average`, take
    the same row as a last argument, its kind the volume average; the
    one-year view re-reserves each year on the same link ratios. The
    geometric average has no Mack variance, so Mack's functions refuse
    it with the crate's message. Against chainladder-python 0.10.1 the
    570 factors and ultimates of `reserving_averages_python.csv` on RAA
    and GenIns match, and against R ChainLadder's `MackChainLadder(weights
    = W)` the standard errors of `reserving_averages_r.csv`, an infinite
    one as a blank cell.
35. **The ODP GLM is three functions on a triangle** (the eighth
    group), over `OdpGlm::fit`: `PRO.RES.ODP.GLM(triangle, [origins])` is the
    exhibit of Taylor and McGuire's Table 5-4 (latest, ultimate, the
    reserve, which is the chain ladder's, the process and parameter
    standard errors, RMSEP and CV per origin and in total);
    `PRO.RES.ODP.GLM.PARAMS` the coefficients with their standard errors and
    Pearson's dispersion; `PRO.RES.ODP.GLM.RESIDUALS(triangle, [by],
    [origins])` a row per observed increment (actual, fitted, actual to
    fitted, Pearson and deviance residuals, leverage, standardized
    residuals) or, with `by` "origin", "development" or "calendar", the
    count, mean and standard deviation of each group's standardized
    deviance residuals. The model is fixed by the triangle (an origin and
    a development factor), so unlike a general GLM, left out below, it
    wants no design table and no fit kept between calls. Coefficients
    are rows of term and level, the intercept first and then the origins
    and development periods from the second (each first is the
    reference), not the crate's coding names, whose levels are in string
    order and whose origins are the add-in's internal years. Development
    and calendar periods are numbered from 1; a cell fitted exactly
    (leverage 1) has blank standardized residuals. The checks: R's
    `glm(..., family = quasipoisson)` coefficients, standard errors,
    dispersion and reserves on GenIns and ABC, and Taylor and McGuire's
    printed Table 5-4 and Chapter 6 residual summaries and heat map.
36. **The newer severity fits extend `PRO.SEV.FIT` and add two functions**
    (the eighth group, over `prospicio_prob`'s #328 work). The Lomax
    (Bahnemann's Pareto, a Burr with `gamma = 1`) is a body family of
    `PRO.SEV.FIT` and of every severity function (`"lomax"`, alpha and
    scale), and `"compare"` fits five body families. A last argument
    `upper` truncates the body families from above (`truncated_above`);
    the Pareto families ignore it in the crate, so the add-in refuses it
    for them. Grouped losses and the minimum limited-mean distance take
    other data than a column of losses, so they are functions of their own:
    `PRO.SEV.FIT.GROUPED(bounds, counts, family, [method], [output])` by
    `fit_grouped` (maximum likelihood or minimum chi-square, conditional
    on the bands' range; parameters with standard errors, the
    chi-square, its degrees of freedom and p-value, or each band's
    expected count) and `PRO.SEV.FIT.LEV(sizes, lev, family, [weights],
    [output])` by `fit_lev_distance`. Excel holds no infinity, so a last
    bound left blank or `"inf"` opens the top band, and the open band's
    upper bound is blank in the table. The checks are Bahnemann's Ex 2.8,
    2.10 and P2.16 as the crate reproduces them, the Lomax and the
    truncated likelihood against `fit_severity` and the closed form.
37. **Exposure curves tabulate side by side, and the test workbook calls
    every function** (user, 2026-10-21, after checking 0.0.4 in Excel).
    `PRO.PRC.EXPOSURE.CURVE` and `PRO.PRC.EXPOSURE.QUANTILE` took one curve, so a
    row of c beside a column of x was refused; a curve comparison is the
    common exhibit, so a row (or column) of parameters now gives one curve
    each, across the other direction from the x. Both directions the same
    way is ambiguous and is an error, and a tabulated curve stays one
    curve (its parameters are ranges by nature). `harness/make_workbook.py`
    now calls all 96 functions (97 with decision 38's, 103 with decision 39's), each with a TRUE/FALSE check beside it;
    the twelve it left out are on the sheets of their area and a new
    "Mack draws" sheet, and the two P95 checks on the ODP sheet read the
    P95 column, not P90.
38. **The Bayesian BF puts its prior on the ultimate** (user, 2026-10-22,
    "go ahead", after `BayesBfNormalised` merged; decision 21's pattern).
    `PRO.RES.BAYES.BF(triangle, exposure, apriori, [cv], [output], ...)` runs
    the Reserving lane's `BayesBfNormalised`, not `BayesBf`: its pattern
    is normalised, so each origin's gamma prior has mean `apriori` times
    exposure and is on the ultimate a Bornhuetter-Ferguson user means,
    where `BayesBf`'s level is absorbed by its pattern
    (`knowledge/findings/bayes-bf-apriori-scale.md`). The arguments
    follow `PRO.RES.BF`'s (triangle, exposure, apriori first) and then
    `PRO.RES.BAYES.ODP`'s sampler arguments in its order; `cv` is the prior's
    CV (0.1, the library's default) or `"flat"` for a flat prior on the
    log ultimate, and `prior_sd` is the SD of the pattern's log share
    ratios. `common_cv`, last, is the library's multiplier common to all
    origins (`reserving-bayes.md` decision 25), omitted for none. The
    library's excess-layer prior (`LayerSharePrior`, `reserving-bayes.md`
    decision 25) needs no function of its own: it is this model with the
    layer's exposure (premium times the layer's share of the ground-up
    loss) as `exposure`, the ground-up loss ratio as `apriori`, its CV as
    `cv` and the share's CV as `common_cv` (or, with an error of its own
    per origin, the combined CV `sqrt((1 + cv^2)(1 + share_cv^2) - 1)` as
    `cv`, `common_cv` omitted). Its
    views are `PRO.RES.BAYES.ODP`'s three plus `"ultimates"` (exposure,
    latest, prior and posterior expected ultimate, posterior loss ratio),
    the comparison a BF exhibit makes. The fits are kept as decision 21's,
    eight per model. The tests check every view against the library's
    fit at the same seed, the flat limit against `PRO.RES.BAYES.ODP` within
    Monte Carlo error and the tight limit at the chain ladder's ultimates
    against `PRO.RES.BF`'s reserve. `BayesBf` (EVW's Model 3.1) stays
    without a sheet form; its one use, EVW parity, is a library check.
39. **A latest diagonal is projected by five functions, one per method**
    (#544, the Excel part; `reserving-v02.md` decisions 15 to 18).
    `PRO.RES.DIAG.CL`, `PRO.RES.DIAG.EL`, `PRO.RES.DIAG.BF`, `PRO.RES.DIAG.BENKTANDER`
    and `PRO.RES.DIAG.CAPECOD` follow the triangle functions, one function
    per method, not one function with a `method` argument: each method
    has its own arguments (`apriori`, `iterations`, `trend` and
    `decay`), and decision 21's single function is for views of one fit,
    not for different methods. The required arguments come first
    (`origins, ages, latest`, then `exposure` and `apriori` where the
    method takes them, then `pattern`), the method's own optional
    arguments right after `pattern`, then the seven shared optional
    arguments in one order: `pattern_ages, kind, tail, grain, valuation,
    past_end, pattern_origin`. `ages` and `valuation` are two arguments,
    exactly one given, rather than one argument read by decision 19's
    rule, since an age of 10,000 months is not a date. The pattern is
    two ranges, values and ages, as `PRO.PRC.PAYOUT.PATTERN` takes times and
    values; its ages default to 12, 24, ... by cell, as `selected`
    reads them, so a row of `PRO.RES.CL.LDF` feeds in directly. A blank
    value skips its pair; a value with a blank age is an error. An
    age-to-age factor runs from its age to the next age given (decision
    16 and item 4 of decision 18 in `reserving-v02.md`), so a skipped
    value widens the factor before it: 1.5, blank, 1.2 under kind
    `"ldf"` is 1.5 from 12 to 36 months, not a factor of 1 from 24 to
    36 (and not `selected`'s blank, which keeps the estimate). A blank
    `latest` is an error (0 is a valid latest value, and a blank is
    more often a range drawn too long). Origins are numbers (years or
    dates); labels such as `"2024Q1"` are not read. The exhibit's
    origin column is the year for accident years, so a lookup against a
    column of years finds it, and the period's text for finer grains;
    there is no `origins` labels argument, since the rows are the
    crate's order, not the input's. The tail's decay is not exposed and
    stays at 0.5, as on every sheet tail. The crate's error for an
    origin past the pattern's end names its Rust settings, so the
    add-in keeps its opening (the origin, its age and the pattern's
    ages) and replaces its ending with the sheet's `past_end` and
    `tail`, matched on the crate's text and pinned by a test, so a
    change of wording in the reserving crate fails the add-in's tests
    on purpose; the two settings errors (`"none"` with a tail,
    `"tail"` without one) are checked in the add-in before the fit,
    on the same conditions as the crate. `PRO.RES.PATTERN.AVERAGE(pattern_ages,
    pattern, pattern_origin, [grain], [kind])` shows decision 18's
    conversion on the sheet (Age, Percent developed and CDF for every
    month of the result); fed back as `pattern_ages` and `pattern` with
    kind `"percent"`, it gives exactly what `pattern_origin` gives.
    Its `grain` is optional and defaults to `"year"`, as on the diagonal
    functions. A `pattern_origin` equal to `grain` is an error there
    (the crate returns the pattern unchanged, and an exhibit that only
    repeats its input is more likely a slip), while on the diagonal
    functions the same pair uses the pattern as given. The grain word
    for six-month origins is `"semiannual"`, and `"semi-annual"` is
    accepted for it (the user's decision of 2026-10-24); `"half"` and
    `"semester"` are not grain words, since no released function took
    a grain word before this decision (the grain arguments are new
    here), so nothing older keeps them as aliases. A tail of exactly 1
    is read as no tail, as the crate reads it, so `past_end` `"none"`
    accepts it and `"tail"` rejects it. The other three details
    (`PRO.RES.PATTERN.AVERAGE`'s `grain` defaulting to `"year"`, a
    `pattern_origin` equal to `grain` being an error there, a tail of
    exactly 1 counting as no tail) were accepted by the user on
    2026-10-25 as built.
    Decision 17's frozen pattern needs no function: `PRO.RES.CL.LDF`
    already gives what `ChainLadderFit::to_chain_ladder` holds (the
    factors in use, then the tail factor), and the pattern stays in
    cells (decision 4). Its first n - 1 cells as `pattern` (ages 12 to
    108 by default) and its last as `tail` give
    `to_chain_ladder().fit_diagonal(&LatestDiagonal::from_triangle(..))`
    to the bit; the whole row with no tail gives the same ultimates for
    every origin up to the pattern's end (to about 1e-14 with a tail),
    since a factor at the oldest age runs one more period. A valuation
    date gives a full triangle's ages, `PRO.TRI.LATEST` its latest
    values. Pattern ages on `PRO.RES.CL`, `PRO.RES.MACK` and the expected-loss
    functions on a triangle, so that a monthly pattern reaches them
    (decision 18 item 4), are decision 40.
40. **Selected patterns on every triangle function, at their ages, and an
    a priori per origin** (#635 and #680, 0.0.7; the user's decisions of
    2026-10-26 and 2026-10-28, and of 2026-10-29: one PR for both). The
    expected-loss functions `PRO.RES.EL`, `PRO.RES.BF`, `PRO.RES.BENKTANDER` and
    `PRO.RES.CAPECOD` take `selected`, `kind` and `weight` as the chain-ladder
    functions read them, and the thirteen functions that take `selected`
    (`PRO.RES.CL`, `.LDF`, `.CDF`, `.ULTIMATE`, `.RESERVE`, `.FACTORS`,
    `PRO.RES.MACK`, `PRO.RES.MACK.CDR`, `.CDR.YEARS` and the four) take
    `pattern_ages`, each selected value's age in months. New arguments are
    appended, so every sheet written before keeps its meaning, and the tail
    of every list is the same: `selected, kind, weight, pattern_ages`, with
    `sigma` before `pattern_ages` where a function already had it after
    `weight` (`PRO.RES.CL.FACTORS`, `PRO.RES.MACK` and the one-year view). That
    is decision 39's order (`pattern_ages, kind, ...`) only in part: a
    triangle function's `kind` and `tail` came first, and moving them
    would break sheets. Without `pattern_ages` a value sits at 12, 24, ...
    by cell, as before; with it, the crate reads the pattern on its own
    ages where they are not the triangle's (`reserving-v02.md` decision 18
    item 4): an age-to-age factor runs to the next age given, a
    cumulative pattern is `ln`-linear between its ages, and one past the
    oldest age folds into the tail. A blank value skips its age, as on the
    diagonal functions, and a value with a blank age is an error. The
    triangle's columns stay 12, 24, ... months (decision 4's reading of
    ages as development periods), so `pattern_ages` is in months against
    an annual triangle; a quarterly triangle would need its period length
    as an argument, which waits until a user asks. `pattern_origin` is not
    added to the triangle functions: `PRO.RES.PATTERN.AVERAGE` gives the same
    averaged pattern in cells, which then goes in as `selected` with kind
    `"percent"` at its ages. `apriori` on `PRO.RES.EL`, `PRO.RES.BF`,
    `PRO.RES.BENKTANDER` and their `PRO.RES.DIAG.*` twins is one number or a row
    or column with one loss ratio per origin (`Apriori` of #634): oldest
    first on a triangle, one per row; on a diagonal in the order the
    `origins` argument gives them, keyed by their periods, since those
    rows may come in any order. `PRO.RES.CAPECOD` estimates its own level and
    takes none. Checks (`tests/selected_patterns.rs`): every
    chainladder-python 0.10.1 `DevelopmentConstant` case of the four
    methods on the CAS workers' compensation triangle (32 runs, the
    patterns to ultimate as percents too) per origin and in total;
    `pattern_ages` of 12, 24, ... equal to omitting it, to the bit, on all
    thirteen functions; a monthly cumulative pattern through RAA's annual
    factors equal to the annual selection to 1e-12 on all thirteen; a
    pattern at 6, 18, ..., 114 months through `PRO.RES.CL.ULTIMATE`,
    `PRO.RES.MACK` and `PRO.RES.BF` equal to the crate's own fits to the bit; the
    a priori per origin against the BF formula and the crate, and a
    diagonal's against any order of its origins. The fake Excel checks
    the last argument through the C API.

41. **GLMs on a sheet** (#578 group 1, 0.0.7; the user's decision of
    2026-10-29 to work on #578 after #635 and #680). `PRO.PRC.GLM`,
    `PRO.PRC.GLM.PREDICT` and `PRO.PRC.GLM.RELATIVITIES` fit a `prospicio-glm`
    GLM to a table whose first row names its columns, as an Excel table
    or a pasted extract has it; trailing blank rows are dropped, so a
    whole-column range works. `response` names a column, and `terms` is a
    row or column of entries: a column's name (a numeric column is a
    numeric term, a text column a factor), `name=level` for a factor with
    that base level (otherwise the first level in sorted order, as R and
    statsmodels choose it), and `a:b` for an interaction. The model is
    refitted on each call rather than kept between calls: a handle to a
    fit held by the add-in would go stale when the sheet changes, and the
    fits a sheet needs take milliseconds (600 rows). `family` is
    `poisson` (the default), `quasipoisson` (Poisson with Pearson's
    dispersion), `gamma`, `gaussian`, `inverse_gaussian`, `binomial` (a
    proportion with the trials as `weights`), `tweedie` (with `p` its
    power) or `negative_binomial` (with `p` its theta); `link` defaults
    to the log, the logit for the binomial and the identity for the
    Gaussian. `exposure` names a column whose log is the offset, so it is
    allowed only on the log link; `weights` names the prior weights.
    Every function takes the same arguments in the same order, with
    `newdata` after `terms` for the prediction and `level` (0.95) last
    for the relativities, so one argument list is copied between them. A
    level or column in `newdata` the training data lacks is an error
    naming it. The relativities list a factor's base level first at 1,
    labelled `(base)`, so the table reads as a rating manual's. Checks
    (`tests/glm.rs`): every statsmodels 0.15.0 case of
    `validation/reference/glm_statsmodels.csv` (ten families and links on
    `validation/data/glm_policies.csv`, 144 values: coefficients, SEs,
    deviance, null deviance, dispersion, log-likelihood and AIC) at the
    file's tolerances; predictions against the coefficients by hand;
    relativities and their intervals against exp(estimate), and a chosen
    base level as the reciprocal; the errors. The fake Excel fits a
    six-row table through the C API, and the test workbook's GLM sheet
    checks the Poisson fit against statsmodels.

42. **Cash flows and returns on a sheet** (#578 group 2, 0.0.7).
    `PRO.NPV`, `PRO.IRR` and `PRO.MOIC` were removed after 0.0.7
    (decision 46); what follows on them is kept as the record.
    `PRO.NPV`, `PRO.IRR` and `PRO.MOIC` wrapped
    `prospicio_pricing::returns`; `PRO.CAP.CASHFLOW` wraps
    `prospicio_pricing::cashflow` (`time-value.md`, decisions 13 to 20).
    Flows are a row or column, one series, or a block with one
    simulation a row, which gives a column of results. `times` default
    to 0, 1, 2, ..., so the first flow is undiscounted, unlike Excel's
    `NPV`, which discounts it a period; `unit` reads them as years,
    months (twelfths) or days (actual/365), or as Excel dates, days from
    the first over 365, which is `XNPV` and `XIRR` exactly. `PRO.IRR`
    keeps the crate's grid search and its defaults (decision 9 there:
    −95% to 100% a year, the root nearest 0), with `lower`, `upper` and
    `extend` as arguments; a single series without a root is an error
    that says whether one lies above or below the range, which Excel's
    `#NUM!` does not, and in a block such a row is blank; `detail` TRUE
    gives the search's flags. `PRO.CAP.CASHFLOW` reads a deal from three
    blocks rather than dozens of arguments: `terms`, rows of a label and
    its values with the labels the Python keyword names (`premium`,
    `commutation_month`, `collected_rate`, `lag`, `funding_dates`,
    `funding_rule`, `annual_yield`, `fee_breakpoints`, ...; case,
    spaces, `-` and `_` alike), an `adjustment` row per premium
    adjustment (month, rate, less commission, base), a set of months as
    `monthly`, `quarterly`, `n`, `n | first` or `months | m1 | m2 ...`,
    and `irr_lower`, `irr_upper` and `irr_extend` for the search;
    `patterns`, three columns of cumulative shares written, earned and
    paid by month; and `requirement` (and `commutation_requirement`),
    rows of a base and its load with `floor` and `limit | trigger | cap`
    rows. An unknown or repeated label is an error that lists the
    labels. `output` is `"simulations"` (the default: each simulation's
    loss ratio, IRR, NPV at `rate` when given, MOIC since inception,
    funded, opening, released, distributed, fee, commutation and
    whether it paid the investor first), `"flows"` (the investor's
    monthly flows) or `"trace"` (every account line of the first
    simulation, then the commutation payment, the fee and the
    conservation error). Paid patterns that differ by simulation (#600)
    and valuations after binding (#603) are left until a user asks.
    Checks (`tests/cashflow.rs`): Microsoft's worked examples of `XNPV`,
    `XIRR` and `IRR` to 1e-9 (`knowledge/references/excel-npv-irr.md`);
    closed forms, roots above the range with and without `extend`, flows
    of one sign; a deal with every rule set through the terms block
    equal to the same deal built in the crate, to the bit, in its flows,
    totals, IRR, NPV, MOIC and trace, and the crate's three-month
    example; the errors. The fake Excel checks `XIRR`'s example, `MOIC`
    and the three-month deal through the C API, and the test workbook's
    Cash flows sheet puts `XNPV`, `XIRR` and `IRR` beside the add-in's.

43. **The P&L ledger on a sheet** (#578 group 3, 0.0.7). `PRO.CAP.PNL`
    wraps `prospicio_aggregate::pnl::Book::ledger` (`pnl.md`, PR 4): a
    book's premium, its expenses and the annual covers it buys, as
    signed legs over a distribution of annual gross losses. The gross is
    atoms in cells, `values` and `probs` (equally likely when omitted),
    so the ledger reads a grid's points, a year loss table's years or a
    model's simulations alike, rather than rebuilding a compound
    distribution from a count and a severity as `PRO.CAP.AGG.*` does: those
    functions' grid can be passed in. `expenses` are rows of `label |
    premium | loss | fixed`, a rate on the gross premium, a rate on the
    gross loss and an amount, several in one row making one mixed leg
    (`aggregate`'s `expense`), a blank label the crate's name; `covers`
    is a table with a header row (`name`, `limit`, `attachment`,
    `share`, `premium` as a deposit, `commission` flat, `profit_share`,
    `profit_allowance`), each row an annual stop loss on the gross,
    lowest first. `face` is the walk (the default, `xpnl`: the gross
    book, then each cover, the running net and the covers' impact), the
    consolidated net position (`pnl`) or either peel. `output` is the
    sheet (the default: each row's step, side, line, mean, SD, CV and
    skewness, then a scenario column per level, each row's mean given
    the result at that lower quantile, which foot down the sheet; the
    levels `aggregate`'s nine unless given), the card (each block's
    consideration, obligation and margin with marginal 1%, 50% and 99%
    quantiles) or the ratios (premium, loss, expense, result, LR, ER,
    CR, their means at each atom, and the shares of the first block).
    `expense_at_mean` books a loss-basis expense at the expected gross
    loss on the consolidated face, as `aggregate`'s guaranteed-cost
    builder does. Not yet: occurrence covers over the joint gross and
    ceded loss, a deal's deductions (`Tower::deal_ledger`), reinstatement
    premiums, retro rating, relabelled legs and `pnl_program`'s premium
    ladder, each when a user asks. Checks (`tests/pnl.rs`): every
    `aggregate` 1.0.1 case of `validation/reference/pnl_builders_aggregate.csv`
    the arguments can state (eleven ledgers: one and two covers on the
    walk and both peels, the consolidated face, premium, loss, fixed and
    mixed expenses; 1,967 values of the sheet and the card) at the
    file's tolerances, leaving out the consolidated κ75 that the Rust
    parity test checks as a tie; the ratios against the crate's ledger;
    equally likely years; a profit commission; the errors. The fake
    Excel checks the crate's example (a net result of 4.54) through the
    C API, and the test workbook's P&L sheet puts `aggregate`'s Example 3
    beside the add-in's.

44. **Year and event loss tables on a sheet** (#578 group 4, 0.0.7).
    `PRO.CAP.YLT.STATS` and `PRO.CAP.YLT.OEP` read a year loss table as columns,
    each row's year (from 1) and loss, through
    `prospicio_aggregate::ylt::YearLossTable` (its checks: losses finite
    and non-negative, or with `negative` stored as negative numbers,
    years from 1, `n_years` at least the largest), tiled once into its
    years. `times` (each loss's fraction of its year) with `hours` applies
    the hours clause (`EventSet::grouped`), which joins the losses within
    a window into one occurrence: it changes the counts and the largest
    occurrences, not the totals. A row blank in every column is the empty
    end of a selected range and is skipped; a row blank in some is an
    error. `quantile` names the rule of `prospicio.risk.statistics`, its
    TVaR going with it: `"inverted_cdf"` (R type 1, the exact TVaR; the
    default, as `Sampled`'s), `"linear"` (R type 7) or `"kth_largest"`
    (the k-th worst year, the mean of the k worst). STATS gives the
    counts, the total's moments (population, dividing by the years), its
    extremes and the largest occurrence, then VaR and TVaR at each
    probability. OEP gives, by return period, the occurrence and
    aggregate PMLs (the quantiles at `1 - p` of each year's largest
    occurrence and total) with their TVaRs, or with `amounts` the share
    of years whose largest occurrence or total is above each amount and
    its return period; a return period past the table's years is an
    error, and the defaults (2 to 1000 years, as `PRO.CAP.AGG.PML`) drop
    those. `PRO.CAP.ELT.STATS` and `PRO.CAP.ELT.OEP` read an event loss table
    (`prospicio_aggregate::cat::EventLossTable`: rates, means, the two
    standard deviations, summed, a beta on `[0, exposure]` per event,
    omitted deviations read as 0); STATS adds the compound Poisson SD
    `sqrt(Σ rate (mean² + σ²))`, OEP gives the occurrence PML by return
    period with the exceedance frequency at it and its occurrence return
    period, or the frequency and OEP at amounts. Not yet: labels (a
    peril or a class) and an hours clause by label, resampling a table to
    more simulations, and towers applied to the years (`Tower::apply`),
    each when a user asks. Checks (`tests/cat.rs`): Alder Mutual's tables
    (`examples/data/cat_book_*.csv`) against every number the guide
    `cat-2-risk-metrics.md` prints, which CI checks against the Python
    package (the average annual loss, SD, VaR and TVaR under the three
    rules, the OEP and AEP at 30, 70 and 110, raw and under a 72-hour
    clause, the occurrences it leaves, the PMLs at 10 to 250 years, and
    the ELT's rate, expected annual loss, OEP and occurrence PMLs); the
    ELT's SD against 200,000 simulated years; negative losses, padded
    ranges and the errors. The fake Excel checks a small table's average
    annual loss and a one-event table's OEP through the C API, and the
    test workbook's Cat sheet holds Alder's tables with the guide's
    numbers beside the add-in's.

45. **Capital allocation on a sheet** (#578 group 5, 0.0.7).
    `PRO.CAP.ALLOCATE` and `PRO.CAP.ALLOCATE.TOTAL` read `draws`, a block with a
    column per unit and a row per equally likely simulation, an optional
    header row of names (a first row with any text), into a
    `PredictiveDistribution` with one component per unit; rows blank in
    every cell are skipped. The units must add up to the portfolio.
    `PRO.CAP.ALLOCATE` takes a distortion as `PRO.CAP.DISTORTION` does (`family`,
    `parameter`, `r0`) and a `method` of
    `prospicio_prob::capital::AllocationMethod`: `"euler"` (the default;
    `"natural"` too, the name of *Pricing Insurance Risk*),
    `"covariance"`, `"covariance_margin"`, `"proportional"`,
    `"marginal"` or `"shapley"` (up to 12 units). It returns a row per
    unit (its stand-alone measure, allocation, share of the portfolio's
    measure and diversification), the units' totals, and the portfolio's
    measure with the diversification benefit; the marginal's totals fall
    short of the portfolio's. `PRO.CAP.ALLOCATE.TOTAL` shares an amount given
    from outside by `PredictiveDistribution::haircut` (each unit's
    stand-alone `measure` at `level`, 0.99 unless given, scaled to the
    total) or `equal_risk` (the one level at which the stand-alone
    measures add up to the total, reported in the last row), on TVaR
    (the default), VaR or EPD. Not yet: the lifted and natural
    allocations of a priced book by layer (`aggregate`'s `price` with a
    distortion over units, which `PRO.PRC.TOWER.PRICE` covers for a tower)
    and allocation at a stated probability for VaR alone, each when a
    user asks. Checks (`tests/allocation.rs`): all 100 cases of
    `validation/reference/allocation_numpy.csv` (four units over 400
    simulations with tied totals; TVaR at 90% and 75%, proportional
    hazard and dual power; the portfolio's and each stand-alone measure,
    Euler, covariance, proportional, marginal and Shapley from
    definitions in numpy); *Pricing Insurance Risk*, Exercises 205 and
    206 (a premium of 80 as a haircut at 75%, 8 and 72, and at the equal
    risk level 0.6796, 9.56 and 70.44); the crate's example, names and
    the errors. The fake Excel checks the crate's example through the C
    API, and the test workbook's Allocation sheet puts the exercises and
    numpy's values beside the add-in's.

46. **No functions that duplicate Excel's own** (user, 2026-11-03).
    The add-in does not expose a function that an Excel built-in already
    does, or that is a one-line formula over Excel's arithmetic with no
    actuarial rule in it. `PRO.NPV` and `PRO.IRR` were Excel's `NPV`,
    `XNPV`, `IRR` and `XIRR` (`unit` `"dates"` was `XNPV` and `XIRR`
    exactly; without `times`, `PRO.NPV` was `x0 + NPV(rate, x1..xn)`).
    Their extras (a block of simulations a row at a time, a grid search
    that names a root outside its range) are covered by `BYROW` with a
    `LAMBDA` in Excel 365 and by `PRO.CAP.CASHFLOW`, which reports each
    simulation's IRR, NPV and MOIC itself. Excel has no `MOIC`, but
    `PRO.MOIC` was `1 + released / (funded + opening)`, a formula a cell
    writes directly. All three are removed (a breaking change after
    0.0.7). Microsoft's worked examples stay a CI check, now on
    `prospicio_pricing::returns`, the discounting and root search
    `PRO.CAP.CASHFLOW` reports with (`tests/cashflow.rs`). Not removed:
    severities whose family Excel also has (`LOGNORM.DIST`,
    `GAMMA.DIST`, `WEIBULL.DIST`), since `PRO.SEV.*` takes the same
    family names and parameters as the layer functions; `PRO.PRC.TREND.FIT`
    (Excel's `LOGEST` fits the same curve but does not give the annual
    change or read year fractions); and the short ratemaking formulas
    (`PRO.PRC.CRED.Z`, `PRO.PRC.CRED.WEIGHTED`, `PRO.PRC.PERMISSIBLE.LR`,
    `PRO.PRC.INDICATED.*`, `PRO.PRC.TRENDED.PRESENT`), which encode Werner and
    Modlin's definitions rather than duplicate a built-in. A new function
    is checked against Excel's function list before it is added.
47. **Functions are grouped by the question they answer** (user,
    2026-11-05). At 114 functions one alphabetical list was hard to keep
    track of, so each name now carries its group after `PRO.`: Excel's
    autocomplete lists a group's functions when its prefix is typed, and
    each group is a category of its own in Insert Function
    ("Prospicio – Reserving" and so on).
    - **`PRO.RES.`, reserving** (32): what the reserve is, from a triangle
      or a latest diagonal, by any method, with its uncertainty: the chain
      ladder, Mack, the bootstraps, the ODP GLM, the expected-loss
      methods, Clark, the Bayesian ODP and BF, the one-year view, the
      diagonal methods and averaged patterns.
    - **`PRO.PRC.`, pricing** (48): what a policy or a cover costs:
      ratemaking and the indication, credibility, GLMs, layers and ILFs,
      exposure and experience rating, the quote, contract terms, a tower's
      price on the grid and the payout's cost of capital.
    - **`PRO.CAP.`, capital** (18): the distribution of the year's total
      and what it implies: aggregate distributions, PMLs and covers on the
      total, event and year loss tables, copulas and Iman-Conover,
      distortions as risk measures, allocation, the P&L ledger and a
      deal's cash flows.
    - **Common, no group** (16): what every group reads, keeping its name:
      `PRO.SEV.CDF`, `PRO.SEV.SURVIVAL`, `PRO.SEV.QUANTILE`, `PRO.SEV.STATS`, `PRO.SEV.FIT`, `PRO.SEV.FIT.GROUPED`, `PRO.SEV.FIT.LEV`, `PRO.EVT.FIT`, `PRO.EVT.HILL`, `PRO.EVT.THRESHOLDS`, `PRO.TRI.INCREMENTAL`, `PRO.TRI.CUMULATIVE`, `PRO.TRI.LINKRATIOS`, `PRO.TRI.LATEST`, `PRO.VERSION`, `PRO.YEARFRACTION`.

    A function that could go in two goes where the user runs it: the
    bootstraps are reserving (run on a triangle) though capital reads
    their draws; layer pricing is pricing; aggregate distributions are
    capital though pricing uses them for aggregate covers. The word after
    `PRO.` decides the category (`Function::category`), and
    `tests/catalogue.rs` fails on a name in no group.

    **The old names are removed at once** (user, 2026-11-05). #746 first
    kept every old name registered, hidden from Insert Function, for one
    release; the user chose to drop them in 0.0.9 instead, before the
    add-in has users beyond its testers. A formula with an old name shows
    `#NAME?`; the new name is the old one with `RES.`, `PRC.` or `CAP.`
    after `PRO.`, and the table below is the find-and-replace list
    (`tests/catalogue.rs` checks that it covers every grouped function).

    **The add-in is built with link-time optimization.** Simulations,
    MCMC and models stay on the sheet (user, 2026-11-05), so the add-in
    stays one file holding every engine, and its size is not a reason to
    leave a function out. `cargo xtask xll` builds with `lto = "fat"`
    and one codegen unit, which drops unused code across crates: 16.3 MB
    to 14.0 MB on the GNU target (2026-11-05, with the old names; 16.2
    MB without LTO and without them), for a few more minutes of release
    build.
    The wrappers are 5% of the code; the rest is the engines
    (`rustfft`'s scalar, SSE and AVX kernels 16%, the Bayesian stack with
    the GLMs and the thread pool about 17%, `prospicio-reserving` 12%).

    | Old name | New name | Group |
    |---|---|---|
    | `PRO.CL` | `PRO.RES.CL` | Reserving |
    | `PRO.CL.CDF` | `PRO.RES.CL.CDF` | Reserving |
    | `PRO.CL.FACTORS` | `PRO.RES.CL.FACTORS` | Reserving |
    | `PRO.CL.LDF` | `PRO.RES.CL.LDF` | Reserving |
    | `PRO.CL.RESERVE` | `PRO.RES.CL.RESERVE` | Reserving |
    | `PRO.CL.ULTIMATE` | `PRO.RES.CL.ULTIMATE` | Reserving |
    | `PRO.MACK` | `PRO.RES.MACK` | Reserving |
    | `PRO.MACK.CDR` | `PRO.RES.MACK.CDR` | Reserving |
    | `PRO.MACK.CDR.YEARS` | `PRO.RES.MACK.CDR.YEARS` | Reserving |
    | `PRO.MACK.BOOT` | `PRO.RES.MACK.BOOT` | Reserving |
    | `PRO.MACK.BOOT.CDR` | `PRO.RES.MACK.BOOT.CDR` | Reserving |
    | `PRO.MACK.BOOT.SIMS` | `PRO.RES.MACK.BOOT.SIMS` | Reserving |
    | `PRO.ODP` | `PRO.RES.ODP` | Reserving |
    | `PRO.ODP.CDR` | `PRO.RES.ODP.CDR` | Reserving |
    | `PRO.ODP.SIMS` | `PRO.RES.ODP.SIMS` | Reserving |
    | `PRO.ODP.GLM` | `PRO.RES.ODP.GLM` | Reserving |
    | `PRO.ODP.GLM.PARAMS` | `PRO.RES.ODP.GLM.PARAMS` | Reserving |
    | `PRO.ODP.GLM.RESIDUALS` | `PRO.RES.ODP.GLM.RESIDUALS` | Reserving |
    | `PRO.EL` | `PRO.RES.EL` | Reserving |
    | `PRO.BF` | `PRO.RES.BF` | Reserving |
    | `PRO.BENKTANDER` | `PRO.RES.BENKTANDER` | Reserving |
    | `PRO.CAPECOD` | `PRO.RES.CAPECOD` | Reserving |
    | `PRO.CLARK` | `PRO.RES.CLARK` | Reserving |
    | `PRO.CLARK.PARAMS` | `PRO.RES.CLARK.PARAMS` | Reserving |
    | `PRO.BAYES.ODP` | `PRO.RES.BAYES.ODP` | Reserving |
    | `PRO.BAYES.BF` | `PRO.RES.BAYES.BF` | Reserving |
    | `PRO.DIAG.CL` | `PRO.RES.DIAG.CL` | Reserving |
    | `PRO.DIAG.EL` | `PRO.RES.DIAG.EL` | Reserving |
    | `PRO.DIAG.BF` | `PRO.RES.DIAG.BF` | Reserving |
    | `PRO.DIAG.BENKTANDER` | `PRO.RES.DIAG.BENKTANDER` | Reserving |
    | `PRO.DIAG.CAPECOD` | `PRO.RES.DIAG.CAPECOD` | Reserving |
    | `PRO.PATTERN.AVERAGE` | `PRO.RES.PATTERN.AVERAGE` | Reserving |
    | `PRO.AVGDATE` | `PRO.PRC.AVGDATE` | Pricing |
    | `PRO.ONLEVEL` | `PRO.PRC.ONLEVEL` | Pricing |
    | `PRO.RATELEVEL` | `PRO.PRC.RATELEVEL` | Pricing |
    | `PRO.TREND.FIT` | `PRO.PRC.TREND.FIT` | Pricing |
    | `PRO.TRENDED.PRESENT` | `PRO.PRC.TRENDED.PRESENT` | Pricing |
    | `PRO.LR.EXPERIENCE` | `PRO.PRC.LR.EXPERIENCE` | Pricing |
    | `PRO.PP.EXPERIENCE` | `PRO.PRC.PP.EXPERIENCE` | Pricing |
    | `PRO.INDICATED.CHANGE` | `PRO.PRC.INDICATED.CHANGE` | Pricing |
    | `PRO.INDICATED.RATE` | `PRO.PRC.INDICATED.RATE` | Pricing |
    | `PRO.PERMISSIBLE.LR` | `PRO.PRC.PERMISSIBLE.LR` | Pricing |
    | `PRO.EXPENSES` | `PRO.PRC.EXPENSES` | Pricing |
    | `PRO.CRED.STANDARD` | `PRO.PRC.CRED.STANDARD` | Pricing |
    | `PRO.CRED.Z` | `PRO.PRC.CRED.Z` | Pricing |
    | `PRO.CRED.WEIGHTED` | `PRO.PRC.CRED.WEIGHTED` | Pricing |
    | `PRO.BUHLMANN` | `PRO.PRC.BUHLMANN` | Pricing |
    | `PRO.BUHLMANN.PARAMS` | `PRO.PRC.BUHLMANN.PARAMS` | Pricing |
    | `PRO.GLM` | `PRO.PRC.GLM` | Pricing |
    | `PRO.GLM.PREDICT` | `PRO.PRC.GLM.PREDICT` | Pricing |
    | `PRO.GLM.RELATIVITIES` | `PRO.PRC.GLM.RELATIVITIES` | Pricing |
    | `PRO.LEV` | `PRO.PRC.LEV` | Pricing |
    | `PRO.LAYER` | `PRO.PRC.LAYER` | Pricing |
    | `PRO.LAYER.CV` | `PRO.PRC.LAYER.CV` | Pricing |
    | `PRO.LAYER.VAR` | `PRO.PRC.LAYER.VAR` | Pricing |
    | `PRO.ILF` | `PRO.PRC.ILF` | Pricing |
    | `PRO.LER` | `PRO.PRC.LER` | Pricing |
    | `PRO.EXPOSURE.CURVE` | `PRO.PRC.EXPOSURE.CURVE` | Pricing |
    | `PRO.EXPOSURE.SHARE` | `PRO.PRC.EXPOSURE.SHARE` | Pricing |
    | `PRO.EXPOSURE.QUANTILE` | `PRO.PRC.EXPOSURE.QUANTILE` | Pricing |
    | `PRO.EXPOSURE.STATS` | `PRO.PRC.EXPOSURE.STATS` | Pricing |
    | `PRO.SWISSRE.C` | `PRO.PRC.SWISSRE.C` | Pricing |
    | `PRO.SWISSRE.C.LAYER` | `PRO.PRC.SWISSRE.C.LAYER` | Pricing |
    | `PRO.PROFILE` | `PRO.PRC.PROFILE` | Pricing |
    | `PRO.BURNINGCOST` | `PRO.PRC.BURNINGCOST` | Pricing |
    | `PRO.PARETO.EXTRAPOLATE` | `PRO.PRC.PARETO.EXTRAPOLATE` | Pricing |
    | `PRO.PARETO.ALPHA` | `PRO.PRC.PARETO.ALPHA` | Pricing |
    | `PRO.PARETO.ALPHA.FREQLAYER` | `PRO.PRC.PARETO.ALPHA.FREQLAYER` | Pricing |
    | `PRO.PARETO.ALPHA.FREQS` | `PRO.PRC.PARETO.ALPHA.FREQS` | Pricing |
    | `PRO.QUOTE` | `PRO.PRC.QUOTE` | Pricing |
    | `PRO.LAYER.TERMS` | `PRO.PRC.LAYER.TERMS` | Pricing |
    | `PRO.PROFIT.COMMISSION` | `PRO.PRC.PROFIT.COMMISSION` | Pricing |
    | `PRO.PROFIT.COMMISSION.TERM` | `PRO.PRC.PROFIT.COMMISSION.TERM` | Pricing |
    | `PRO.SLIDING.SCALE` | `PRO.PRC.SLIDING.SCALE` | Pricing |
    | `PRO.SLIDING.SCALE.TERM` | `PRO.PRC.SLIDING.SCALE.TERM` | Pricing |
    | `PRO.SWING` | `PRO.PRC.SWING` | Pricing |
    | `PRO.TOWER.PRICE` | `PRO.PRC.TOWER.PRICE` | Pricing |
    | `PRO.PAYOUT.PATTERN` | `PRO.PRC.PAYOUT.PATTERN` | Pricing |
    | `PRO.PAYOUT.STATS` | `PRO.PRC.PAYOUT.STATS` | Pricing |
    | `PRO.PAYOUT.PRICE` | `PRO.PRC.PAYOUT.PRICE` | Pricing |
    | `PRO.AGG.STATS` | `PRO.CAP.AGG.STATS` | Capital |
    | `PRO.AGG.LAYERS` | `PRO.CAP.AGG.LAYERS` | Capital |
    | `PRO.AGG.COVER` | `PRO.CAP.AGG.COVER` | Capital |
    | `PRO.AGG.PML` | `PRO.CAP.AGG.PML` | Capital |
    | `PRO.ELT.STATS` | `PRO.CAP.ELT.STATS` | Capital |
    | `PRO.ELT.OEP` | `PRO.CAP.ELT.OEP` | Capital |
    | `PRO.YLT.STATS` | `PRO.CAP.YLT.STATS` | Capital |
    | `PRO.YLT.OEP` | `PRO.CAP.YLT.OEP` | Capital |
    | `PRO.COPULA.CDF` | `PRO.CAP.COPULA.CDF` | Capital |
    | `PRO.COPULA.SIM` | `PRO.CAP.COPULA.SIM` | Capital |
    | `PRO.IMAN.CONOVER` | `PRO.CAP.IMAN.CONOVER` | Capital |
    | `PRO.ALLOCATE` | `PRO.CAP.ALLOCATE` | Capital |
    | `PRO.ALLOCATE.TOTAL` | `PRO.CAP.ALLOCATE.TOTAL` | Capital |
    | `PRO.DISTORTION` | `PRO.CAP.DISTORTION` | Capital |
    | `PRO.DISTORTION.PRICE` | `PRO.CAP.DISTORTION.PRICE` | Capital |
    | `PRO.DISTORTION.CALIBRATE` | `PRO.CAP.DISTORTION.CALIBRATE` | Capital |
    | `PRO.PNL` | `PRO.CAP.PNL` | Capital |
    | `PRO.CASHFLOW` | `PRO.CAP.CASHFLOW` | Capital |
48. **Simulations and MCMC run asynchronously** (user, 2026-11-05, "work
    on the async functions"). Every function with a `sims` argument and
    every MCMC fit (`jobs::ASYNC`: `PRO.RES.ODP`, `.ODP.SIMS`, `.ODP.CDR`,
    `.MACK.BOOT`, `.MACK.BOOT.SIMS`, `.MACK.BOOT.CDR`, `.BAYES.ODP`,
    `.BAYES.BF` and `PRO.CAP.COPULA.SIM`) is an Excel 2010 asynchronous
    function: registered as `>` (no result) with its `Q` arguments and a
    last `X`, the handle Excel waits on, and not thread safe (`$`), since
    the call only starts the run. It returns at once; the cell shows
    `#GETTING_DATA` while the run goes on a thread of its own, and the
    result reaches Excel through `xlAsyncReturn`, the one callback Excel
    takes from any thread. A new function that simulates or samples joins
    the list, and `tests/catalogue.rs` fails when an export and the list
    disagree.
    - **One run per inputs.** `jobs::submit` keys a call on its function
      and every argument (each number written exactly): a call joins the
      run of the same inputs already going, so a sheet of cells that
      differ only in their view waits on one run; and the last 32 results
      (at most 4 million cells together) are kept, so a recalculation
      that changes nothing upstream answers at once (about a millisecond
      in the fake Excel, against 0.2 s for 20,000 ODP simulations). Each
      function is a pure function of its inputs and seed (decisions 14
      and 21), so a kept result is what a rerun gives: no number changes.
      The Bayesian fits stay kept as well (decision 21), so a fit's three
      views still sample once between them.
    - **An interrupted recalculation loses nothing.** Typing while Excel
      waits interrupts the recalculation and Excel drops its handles. The
      run goes on and its result is kept, so the next recalculation
      answers from it; delivering to a dropped handle is refused by Excel
      and ignored. So the add-in registers no calculation events. A run is
      not cancelled: one already going runs to its end (at most about
      10 s for a bootstrap's 1,000,000 simulations, decision 14).
    - **The add-in stays loaded.** From the first asynchronous call it is
      pinned (`GetModuleHandleExW` with `GET_MODULE_HANDLE_EX_FLAG_PIN`)
      until Excel exits, so removing it in the Add-ins dialog cannot free
      code that a run is still executing.
    - **Memory.** A result handed to `xlAsyncReturn` is not marked
      `xlbitDLLFree`: Excel copies it during the call, and the add-in
      frees it as soon as the call returns.
    - **Fit handles are not needed yet.** A handle (a cell holding a name
      for a fit that other functions read) was the other half of the
      plan. With runs shared and results kept, cells that read one fit
      already sample once, so a handle would add a kind of value the
      sheet has never had (decision 4) for no saving. It stays open for
      a model whose results do not fit one function's views.
    - **Sources.** Microsoft's C API pages were not reachable from the
      cloud session, so the constants (`xlAsyncReturn` = 16 | xlSpecial,
      `xltypeBigData` = xltypeStr | xltypeInt) and the type text (`>`,
      the arguments, `X`) come from Excel-DNA's source
      (`knowledge/references/excel-async-udfs.md`). The fake Excel checks
      the protocol as Excel-DNA reads it: nine asynchronous registrations,
      the call returning before its result, the handle coming back with a
      result not marked `xlbitDLLFree`, and the same inputs answered from
      the kept result. To confirm in Excel 365: `#GETTING_DATA` and then
      the result, and an edit during a long run.

## Parity plan

The user's direction (2026-10-16): move toward parity with the Python and
R packages. Not every one of R's 386 exports has a sheet form: a fitted
model object, a NUTS sampler or a P&L ledger is not a range of cells. The
order, one small PR each, those already done struck through:

1. Reserving: ~~chain ladder, Mack~~, ~~ODP bootstrap and its draws~~,
   ~~Mack's bootstrap and its draws~~; ~~Mack's options (the sigma rule,
   selected factors)~~; ~~the expected-loss methods (expected loss,
   Bornhuetter-Ferguson, Benktander, Cape Cod) on a premium column~~; ~~the
   one-year view (Merz-Wuthrich, the bootstraps' CDR)~~; ~~Clark's LDF and
   Cape Cod~~; ~~triangle helpers (link ratios, incremental and cumulative,
   the latest diagonal)~~.
2. Severities: ~~the Pareto and the lognormal's LEV, layer, CV, ILF~~;
   ~~more families (gamma, Weibull, Burr, log-logistic, inverse gamma and
   Gaussian, generalized Pareto) and their cdf, quantile, mean and SD,
   the loss elimination ratio~~; ~~the piecewise Pareto and Riegel's
   generalized Pareto~~; ~~the Pareto helpers (alpha between layers or
   frequencies, extrapolation)~~; ~~the layer variance~~.
3. Pricing: ~~exposure curves (MBBEFD, Swiss Re, tabulated) and a
   layer's share, the Swiss Re calibrations~~; ~~a risk profile's layer
   loss and surplus cession~~;
   ~~credibility (Buhlmann-Straub, limited fluctuation)~~; ~~ratemaking
   (trend fits and lengths, the rate level, the experience methods and
   the indicated rate)~~; ~~burning cost~~.
4. ~~Aggregate: a compound distribution of a count and a severity, its
   moments, VaR, TVaR and layer statistics~~.
5. Aggregate, continued (user, 2026-10-16): ~~the occurrence PMLs~~;
   ~~aggregate covers on the annual total~~; ~~towers on a grid~~.
6. Fitting and pricing inputs (user, 2026-10-19, "work on next step 2";
   the order chosen by this lane): ~~severity fits to large losses
   (body and Pareto families, with thresholds and censoring)~~; ~~EVT
   threshold diagnostics, the GPD fit and Hill's estimator~~; ~~payment
   patterns and discounting~~; ~~a cover's quote (technical premium with
   expenses and brokerage)~~; ~~distortions calibrated to a target and
   a layer priced by one~~; ~~contract terms on an aggregate (sliding
   scale, profit commission, reinstatement premiums)~~.
7. The nearest left out (user, 2026-10-19, "more Excel functions"):
   ~~the distortions with several parameters (bi-TVaR, beta, weighted
   TVaR)~~; ~~copulas (a copula's cdf at points, uniforms simulated from
   one, and Iman–Conover's reordering of simulated columns to a rank
   correlation, all on ranges that hold no state)~~; ~~multi-year
   contract terms (a profit commission or sliding scale over a term of
   simulated years, deficits and credits carried forward)~~.
8. What the other lanes have built since (user, 2026-10-20, "move
   forward"; the order chosen by this lane): ~~link-ratio exclusions and
   the geometric average in `average`~~; ~~the ODP GLM with its analytic
   prediction error and residual diagnostics~~; ~~the newer severity
   fits (the Lomax, truncation from above, grouped losses and
   minimum-distance fits)~~.
9. A latest diagonal (#544, user, 2026-10-24): ~~a bordereau's
   diagonal projected with a selected pattern by the chain ladder and
   the expected-loss methods, patterns averaged over origin periods,
   origins past the pattern's end~~ (decision 39).

Left out unless asked: model fitting and prediction (GLM, GAM, boosting;
Bayesian reserving models other than the ODP chain ladder, decision 21),
vine copulas, the P&L ledger, catastrophe event
sets. These want a table of data and an object kept between calls, which
an XLL function (decision 4: no state) does not hold.

## Open (for the user)

- **Next functions:** the parity plan's nine groups are done; further
  functions as the user asks (nested
  Archimedean copulas and vines, and the expected multi-year commission
  on the grid, are the nearest left out).
- **A code-signing certificate**, if the add-in goes beyond a few
  testers (decision 16). A certificate from the organization's own CA,
  trusted on its machines by group policy, comes as a PKCS#12 file and
  works with the two secrets as they are. A publicly trusted certificate
  (OV or EV) must keep its key on a hardware token or a cloud HSM since
  June 2023, so it cannot be a file in a secret; it would need a cloud
  signing service's own step (Azure Trusted Signing, DigiCert KeyLocker,
  SSL.com eSigner) in place of `osslsigncode`.
