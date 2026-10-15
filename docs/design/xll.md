# Design note: the Excel add-in (XLL)

Status: **Prototype, for testing by the user** · Lane: Excel (cloud) · Depends on: `triangle.md`, `reserving-v02.md` (the functions it wraps)

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
| `PRO.VERSION()` | The prospicio version the add-in was built from |

`average` is `"volume"` (default), `"simple"` or `"regression"`; `tail` is
omitted (none), a number (a constant factor to ultimate at the oldest age)
or `"exponential"`, `"inverse_power"` or `"bondy"`. Mack's sigma is
extrapolated log-linearly, as R's `MackChainLadder(est.sigma =
"log-linear")`.

**Checks.** `cargo test -p prospicio-xll` runs the worksheet functions on
RAA through the same catalogue Excel calls, against `prospicio-reserving`
and R ChainLadder's totals. `cargo xtask xll` builds the add-in and, when
MinGW (and off Windows, Wine) are found, runs `harness/fake_excel.c`: a
host that exports `MdCallBack12` as Excel does, loads the XLL, checks the
registrations and calls the functions with real XLOPER12 arguments (RAA's
Mack totals against R to 1e-6, an error, 2000 calls and frees).
`harness/make_workbook.py` writes a workbook with RAA and every function
in place, with R's totals beside them, for a test in real Excel.

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
   spills from the formula's cell; in older versions it is entered as an
   array formula over the result's size.
6. **The triangle is the values only.** One row per origin (oldest first),
   one column per development age; blank, empty text or `#N/A` where
   unobserved, and nothing observed after a blank in a row. Labels stay on
   the sheet; `origins` passes origin labels into the exhibit. Ages are
   read as development periods, so yearly, quarterly and monthly
   triangles all work; link labels say 12-24 and on.
7. **Errors come back as text.** A bad argument returns one cell
   `#PROSPICIO! <reason>`, since Excel's `#VALUE!` cannot say why. A panic
   is caught and reported the same way; it never unwinds into Excel.
8. **Built from Linux or Windows.** `cargo xtask xll` uses
   `x86_64-pc-windows-msvc` on Windows and `x86_64-pc-windows-gnu` with
   MinGW's linker elsewhere; the result needs only system DLLs.
9. **Distribution: a local file for now** (user, 2026-10-15): unsigned,
   built by `cargo xtask xll`, handed over for testing. A CI artifact or a
   release asset, and code signing, come after the user has tried it.

## Open (for the user)

- **The prefix.** `PRO.` is short to type; `PROSPICIO.` cannot clash
  with another add-in. Changing it later breaks saved workbooks.
- **Text errors or `#VALUE!`.** Text explains itself but is not an Excel
  error, so `ISERROR` does not see it.
- **Next functions** (user's order, 2026-10-15): layer pricing (Pareto or
  lognormal layer loss, ILF), then an ODP bootstrap summary.
- **Distribution after testing:** CI artifact or GitHub release asset;
  signing so that Excel's block on XLLs from the internet can be lifted
  by policy instead of per file.
