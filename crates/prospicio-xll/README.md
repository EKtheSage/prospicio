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
ChainLadder's totals beside them, plus Layers, ODP, ODP draws, Mack bootstrap, Expected loss (GenIns with premium) and One-year sheets.

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
| `PRO.EL(triangle, exposure, apriori, [average], [tail], [origins])` | Expected loss method exhibit |
| `PRO.BF(triangle, exposure, apriori, [average], [tail], [origins])` | Bornhuetter-Ferguson exhibit |
| `PRO.BENKTANDER(triangle, exposure, apriori, [iterations], [average], [tail], [origins])` | Benktander exhibit |
| `PRO.CAPECOD(triangle, exposure, [trend], [decay], [average], [tail], [origins])` | Cape Cod exhibit |
| `PRO.VERSION()` | Version |
| `PRO.LEV(limit, family, p1, p2, [truncation])` | Limited expected value per limit |
| `PRO.LAYER(limit, attachment, family, p1, p2, [truncation])` | Expected loss to the layer per ground-up loss |
| `PRO.LAYER.CV(limit, attachment, family, p1, p2, [truncation])` | CV of the layer loss given a loss above the attachment |
| `PRO.ILF(limit, basic_limit, family, p1, p2, [truncation])` | Increased limit factor |

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

- `family`: `"pareto"` (`p1` = threshold t, `p2` = alpha, optional
  `truncation`), `"lognormal"` (meanlog, sdlog) or `"lognormal_cv"` (mean,
  CV).
- `limit`, `attachment`, `basic_limit`: a number, a range or
  `"unlimited"`. A column of attachments against a row of limits gives a
  table, e.g. `=PRO.LAYER(B1:E1, A2:A4, "pareto", 500, 2)`.

Type a function in one cell: in Excel 365 its table spills from there
(older Excel needs it entered as an array formula, Ctrl+Shift+Enter, over
the result's size). A bad argument returns `#PROSPICIO! <reason>` in one
cell.
