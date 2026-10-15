# prospicio-xll

prospicio's reserving methods as Excel worksheet functions: an XLL add-in
for 64-bit Excel on Windows. Design and decisions: `docs/design/xll.md`.

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
ChainLadder's totals beside them.

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
| `PRO.CL(triangle, [average], [tail], [origins])` | Chain ladder exhibit with a total row |
| `PRO.CL.LDF(triangle, [average], [tail])` | Age-to-age factors, one per column; the last is the tail |
| `PRO.CL.CDF(triangle, [average], [tail])` | Age-to-ultimate factors, one per column |
| `PRO.CL.ULTIMATE(...)`, `PRO.CL.RESERVE(...)` | One value per origin |
| `PRO.CL.FACTORS(triangle, [average], [tail])` | LDF, CDF, sigma and standard error per link, and the tail |
| `PRO.MACK(triangle, [average], [tail], [origins])` | Mack exhibit: process, parameter and total SE and CV, with a total |
| `PRO.VERSION()` | Version |

- `triangle`: cumulative values only, one row per origin (oldest first),
  one column per age; blank or `#N/A` where unobserved.
- `average`: `"volume"` (default), `"simple"` or `"regression"`.
- `tail`: omitted for none, a factor such as `1.05`, or `"exponential"`,
  `"inverse_power"` or `"bondy"`.
- `origins`: labels for the exhibit's first column; omitted for 1, 2, ...

A bad argument returns `#PROSPICIO! <reason>` in one cell.
