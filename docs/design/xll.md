# Design note: the Excel add-in (XLL)

Status: **Prototype, tested by the user in Excel 365 (2026-10-15); built and kept by CI (2026-10-16)** · Lane: Excel (cloud) · Depends on: `triangle.md`, `reserving-v02.md` (the functions it wraps)

## Goal

prospicio's methods as worksheet functions in desktop Excel, from the same
Rust kernel as the Python and R packages: a triangle selected on a sheet
in, an exhibit that spills next to it out. No Python, R or server on the
user's machine; one file to load.

## What exists (prototype)

`crates/prospicio-xll` builds `prospicio.xll`, a 64-bit Windows DLL that
speaks the Excel C API (XLOPER12). Functions, all in the `prospicio`
category of Insert Function:

| Function | Returns |
|---|---|
| `PRO.CL(triangle, [average], [tail], [origins])` | Origin, Latest, CDF, Ultimate, Reserve per origin, and a total row |
| `PRO.CL.LDF(triangle, [average], [tail])` | Selected age-to-age factors, one per triangle column; the last is the tail factor |
| `PRO.CL.CDF(triangle, [average], [tail])` | Age-to-ultimate factors, one per triangle column |
| `PRO.CL.ULTIMATE(...)`, `PRO.CL.RESERVE(...)` | One column, one row per origin |
| `PRO.CL.FACTORS(triangle, [average], [tail])` | Per link: LDF, CDF, Mack sigma and the factor's standard error; then the tail |
| `PRO.MACK(triangle, [average], [tail], [origins])` | Latest, Ultimate, Reserve, Process SE, Parameter SE, Mack SE and CV per origin, and the total |
| `PRO.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins])` | ODP bootstrap: CL reserve, simulated mean, SD, CV and percentiles per origin, and the total |
| `PRO.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins])` | The bootstrap's simulated reserves: one row per simulation, a column per origin and the total |
| `PRO.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins])` | Mack's bootstrap (England, Verrall and Wuthrich), laid out as `PRO.ODP` |
| `PRO.MACK.BOOT.SIMS(triangle, [sims], [seed], [average], [tail], [process], [origins])` | Its simulated reserves, laid out as `PRO.ODP.SIMS` |
| `PRO.VERSION()` | The prospicio version the add-in was built from |
| `PRO.LEV(limit, family, p1, p2, [truncation])` | Limited expected value `E[min(X, limit)]`, per limit |
| `PRO.LAYER(limit, attachment, family, p1, p2, [truncation])` | Expected loss to `limit` xs `attachment` per ground-up loss |
| `PRO.LAYER.CV(limit, attachment, family, p1, p2, [truncation])` | CV of the layer loss given a loss above the attachment |
| `PRO.ILF(limit, basic_limit, family, p1, p2, [truncation])` | Increased limit factor `LEV(limit) / LEV(basic_limit)` |

`average` is `"volume"` (default), `"simple"` or `"regression"`; `tail` is
omitted (none), a number (a constant factor to ultimate at the oldest age)
or `"exponential"`, `"inverse_power"` or `"bondy"`. Mack's sigma is
extrapolated log-linearly, as R's `MackChainLadder(est.sigma =
"log-linear")`.

`PRO.ODP` wraps `OdpBootstrap::fit` (R's `BootChainLadder`): `sims`
defaults to 10,000 (at most 1,000,000), `seed` to 0, `percentiles` to
75%, 90%, 95% and 99.5%, `process` to `"gamma"` (`"none"` for parameter
error only); `tail` is read as for the chain ladder.

`PRO.MACK.BOOT` wraps `MackBootstrap::fit` with centred residuals (the
crate's default, which reconciles with Mack's standard errors); `process`
is `"gamma"` (default), `"lognormal"`, `"normal"`, `"residuals"` or
`"none"`.

The pricing functions wrap `prospicio_pricing::layer` (`ilf`, `layer_cv`)
and `prospicio_prob::Severity` (`lev`, `layer`). `family` is `"pareto"`
(`p1` the threshold `t`, `p2` alpha; `truncation` optional), `"lognormal"`
(`meanlog`, `sdlog`, as Excel's `LOGNORM.DIST`) or `"lognormal_cv"` (mean
and CV). Limits and attachments are a number, a range or `"unlimited"`;
blank cells give blank results.

**Checks.** `cargo test -p prospicio-xll` runs the worksheet functions on
RAA through the same catalogue Excel calls, against `prospicio-reserving`
and R ChainLadder's totals. `cargo xtask xll` builds the add-in and, when
MinGW (and off Windows, Wine) are found, runs `harness/fake_excel.c`: a
host that exports `MdCallBack12` as Excel does, loads the XLL, checks the
registrations and calls the functions with real XLOPER12 arguments (RAA's
Mack totals against R to 1e-6, an error, 2000 calls and frees, and a
2 x 2 Pareto layer table against the R package Pareto).
`cargo test` checks `PRO.ODP` against `OdpBootstrap` draw for draw and
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
4. **Values in, values out, thread safe.** Every argument and result is
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
    function keeps it, since renaming breaks saved workbooks.
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
    `PRO.ODP.SIMS` returns the draws themselves, the same ones for the
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

## Parity plan

The user's direction (2026-10-16): move toward parity with the Python and
R packages. Not every one of R's 386 exports has a sheet form: a fitted
model object, a NUTS sampler or a P&L ledger is not a range of cells. The
order, one small PR each, those already done struck through:

1. Reserving: ~~chain ladder, Mack~~, ~~ODP bootstrap and its draws~~,
   ~~Mack's bootstrap and its draws~~; Mack's options (the sigma rule,
   selected factors); the expected-loss methods (expected loss,
   Bornhuetter-Ferguson, Benktander, Cape Cod) on a premium column; the
   one-year view (Merz-Wuthrich, the bootstraps' CDR); Clark's LDF and
   Cape Cod; triangle helpers (link ratios, incremental and cumulative,
   the latest diagonal).
2. Severities: ~~the Pareto and the lognormal's LEV, layer, CV, ILF~~;
   more families (gamma, Weibull, Burr, log-logistic, inverse gamma,
   generalized Pareto, piecewise Pareto) and their cdf, quantile, mean,
   VaR and TVaR; the Pareto helpers (alpha between layers or
   frequencies, extrapolation), the loss elimination ratio, the layer
   variance, the stop loss.
3. Pricing: exposure curves (MBBEFD, Swiss Re) and a layer's share;
   credibility (Buhlmann-Straub, limited fluctuation); ratemaking (trend,
   rate level, the indicated rate); burning cost.
4. Aggregate: a compound distribution of a count and a severity, its
   moments, VaR, TVaR and layer statistics.

Left out unless asked: model fitting and prediction (GLM, GAM, boosting,
Bayesian models), copulas and vines, the P&L ledger, catastrophe event
sets. These want a table of data and an object kept between calls, which
an XLL function (decision 4: no state) does not hold.

## Open (for the user)

- **Next functions:** the parity plan above. More severity families (gamma, Weibull, Burr, the piecewise
  Pareto) and the Pareto helpers (`pareto_extrapolation`,
  `alpha_between_layers`) are each a few lines when wanted.
- **A code-signing certificate**, if the add-in goes beyond a few
  testers (decision 16). A certificate from the organization's own CA,
  trusted on its machines by group policy, comes as a PKCS#12 file and
  works with the two secrets as they are. A publicly trusted certificate
  (OV or EV) must keep its key on a hardware token or a cloud HSM since
  June 2023, so it cannot be a file in a secret; it would need a cloud
  signing service's own step (Azure Trusted Signing, DigiCert KeyLocker,
  SSL.com eSigner) in place of `osslsigncode`.
