# The JavaScript package (WebAssembly)

**Status:** built, first version (#576; branch `claude/wasm-package`). Decisions 1 to 9 are as built and proposed for the user's acceptance; decision 10 (the npm name) is the user's; decision 11 (publishing) is as built; the open decisions are at the end.

**Lane:** JavaScript. **Crate:** `crates/prospicio-wasm`. **Package:** `prospicio` on npm (decision 10).

## Purpose

The pricing and reserving engines in a browser page, in Node or in an Office.js Excel add-in, with no server and no XLL (which works only in desktop Excel on Windows). The crates in the WASM scope (`docs/architecture.md`: `prospicio-core`, `-math`, `-prob`, `-aggregate`, `-pricing`, `-reserving`) already build for `wasm32-unknown-unknown`; this crate is the JavaScript API over them, its packaging and its tests.

The user parked #576 on 2026-11-07 (the XLL already reaches Excel users, and a browser package needs enterprise work to deploy) and took it up again on 2026-11-20 for the 0.2.1 milestone.

## What is built

One module file per lane, as in the Python and R bindings; `lib.rs` only registers them.

| Module | JavaScript API |
|---|---|
| `reserving.rs` | `Triangle.fromLong`, `shape`, `columns`, `keyNames`, `origins`, `development`, `valuation`, `isCumulative`, `toIncremental`, `toCumulative`, `toLong`; `chainLadder`, `mack`, `bornhuetterFerguson`, `odpBootstrap` |
| `prob.rs` | `Dist.fromJson`, `toJson`, `family`, `mean`, `variance`, `standardDeviation`, `cdf`, `survival`, `quantile`, `lev`, `layer`, `tvar` |
| `aggregate.rs` | `Count.poisson`, `negativeBinomial`, `binomial`; `fft` (a `Grid` with `step`, `probs`, `mean`, `quantile`, `tvar`, `tailMass`, `aliasingError`); `simulateEvents`, `EventSet.fromYears`; `Tower.fromJson`, `toJson`, `apply`, `applyAnnual`; `oep`, `occurrencePml` |
| `pricing.rs` | `Distortion.fromName`, `g`, `apply`; `price`, `priceAtCostOfCapital` |
| `predictive.rs` | `Predictive` (simulated reserves, a tower's cessions): `dims`, `components`, `nSims`, `draws`, `total`, `mean`, `standardDeviation`, `var`, `tvar`, `summary`, `aggregate` |

`cargo xtask wasm` builds the release module once and writes three sets of glue with the `wasm-bindgen` command-line tool into `target/wasm/pkg`: `bundler/` (the package's default export, for webpack or Vite), `node/` (CommonJS, picked by Node through the `node` export condition) and `web/` (ES module with an `init()` for a page with no bundler, imported as `prospicio/web`). The module is 1.53 MB (0.51 MB gzipped) with fat LTO and one codegen unit.

## Decisions (as built)

1. **Thin wrappers only.** No number is computed in this crate: `Predictive.summary` reads each component's mean and standard deviation from the core's marginals, a fit's tables are the core's `FitTable`s. This is the rule for the Python, R and Excel front ends (`xll.md` decision 65).
2. **Names.** Functions, methods and option fields are camelCase (`chainLadder`, `nSims`). The columns of a result table keep the core's names (`standard_error`, `ulr`): they are the column names of the Python and R data frames, so a guide's table reads the same in every language.
3. **Tables are objects of columns.** A fit returns `{ byOrigin, totals, development }`, each an object with one array per column: key columns and `origin` as arrays of strings, numeric columns as `Float64Array`. This is the shape Arquero, Observable Plot and a hand-written loop all take, and it copies each column once.
4. **Arrays.** Numeric results are `Float64Array`. Going in, a plain array or any typed array is accepted (the options are read with `serde-wasm-bindgen`); `null` or `NaN` marks a missing measure.
5. **Errors.** A core error is thrown as a JavaScript `Error` whose `message` is the core's text (`JsError`). An unknown option name is an error that lists the fields: `serde-wasm-bindgen` reads the fields a struct declares and ignores any other key, so `deny_unknown_fields` alone let `{ nSim: 500 }` through as the defaults; each options struct therefore declares its field names (`convert::Fields`) and the keys are checked first.
6. **Distributions and towers come from their JSON documents.** `Dist.fromJson` and `Tower.fromJson` take the versioned documents that Rust, Python and R save (`risk_rs.distribution`, `risk_rs.tower`), as text or as an object. One format, already versioned and tested, in place of a constructor per family and per layer term in a fourth language. A claim count has no saved format yet, so `Count` has three constructors.
7. **Reproducibility.** A simulation takes `seed` and gives the draws the other front ends give: the tests compare the ODP bootstrap's draw matrix and a tower's cessions with the core called directly, bit for bit. There are no threads in `wasm32-unknown-unknown`; Rayon's global pool runs on the calling thread there, so the result is the same and only slower. A seed is a JavaScript number, a whole number up to 2^53.
8. **The tool is pinned.** The `wasm-bindgen` crate is pinned with `=` in the crate's manifest, and `cargo xtask wasm` and CI read that pin and refuse or install the command-line tool at another version: the two must match exactly or the tool rejects the module.
9. **Tests in two layers.** `crates/prospicio-wasm/tests/*.rs` run under Node with `wasm-bindgen-test` (`cargo test -p prospicio-wasm --target wasm32-unknown-unknown`; the runner is set in `.cargo/config.toml`) and check numbers: GenIns chain ladder and Mack against the R ChainLadder rows, the lognormal's limited expected values against the mpmath rows, the FFT aggregate and the occurrence PML against the `aggregate` rows, each with the row's own tolerance (the reference files are read with `include_str!`, since a WASM test has no file system), and the bootstrap, Bornhuetter-Ferguson, a given grid and a tower against the core, exactly. `crates/prospicio-wasm/js/package.test.mjs` runs on the built package as a JavaScript program does (`node --test`, from `cargo xtask wasm`) and checks what only JavaScript sees: names, argument and result types, thrown errors, the package's entry points.
10. **The npm package is `prospicio`.** Decided by the user on 2026-11-20: the same name as the PyPI package, the R package and the umbrella crate, unscoped, so it needs no organization on npm. The name was free on the registry that day (`npm view prospicio` returned 404). It is one constant in `xtask` (`WASM_PACKAGE`).
11. **Published by the release workflow, with no stored token.** `release.yml` builds and tests the package on every run (`npm-package`: `cargo xtask wasm`, then `npm pack --dry-run`) and, on a version tag, publishes that artifact from the `npm` job by npm's trusted publishing (OIDC: `id-token: write`, the `npm` environment, Node 24 for npm 11.5.1 or later), which also attaches a provenance attestation. The PyPI and crates.io jobs now need `npm-package` too, so a release whose JavaScript package does not build publishes nothing anywhere. npm accepts a trusted publisher only on a package that exists, so the first version is published by hand from the run's artifact, and the `npm` job is skipped until the owner sets the repository variable `NPM_PUBLISH` to `true` (`docs/release.md`, one-time setup, step 4). A new trusted publisher expires unless it publishes within two days, so it is set up just before a release.

## Validation

| Check | Result |
|---|---|
| GenIns, chain ladder reserve and ultimate by origin and in total; Mack's standard error by origin and in total | 33 rows of `reserving_chainladder_r.csv` within 1e-9 |
| Lognormal `lev` and `layer` | 48 rows of `severity_mpmath.csv` within 1e-11 to 1e-12 |
| Poisson-lognormal aggregate by FFT on the grid sized for it: mean and four quantiles | 5 rows of `sizing_aggregate.csv` |
| Occurrence PML, Poisson-lognormal | the `loss` rows of `oep_aggregate.csv` within 1e-10 |
| ODP bootstrap draws (400 simulations, seed 7), Bornhuetter-Ferguson, FFT masses on a given grid, simulated events and a two-layer tower's cessions | equal to the core, bit for bit |

## Open decisions (the user's)

- **O3. Size.** `wasm-opt -Oz` (Binaryen) usually takes 10% to 20% off a module; it is one more tool for CI to install. Shipping one `.wasm` for the three targets instead of three copies would cut the package from 4.6 MB to 1.6 MB unpacked, if the three glue files can share it.
- **O4. TypeScript types of the options.** The generated `.d.ts` types every class and function, but an options object is `any`. Hand-written interfaces (`typescript_custom_section`) would give editors the field names.
- **O5. What comes next in the API.** Candidates, by what a browser tool or an Office.js add-in would call: the expected-loss family beside Bornhuetter-Ferguson, selected patterns and averages on the chain ladder, the one-year view, `Tower.onGrid` with layer statistics and prices, exposure curves, and custom functions for Office.js (a manifest and `CustomFunctions.associate` over these exports).
- **O6. Threads.** A build with Web Workers (`wasm-bindgen-rayon`) needs cross-origin isolation headers on the page and a nightly toolchain; not proposed until a simulation in the browser is too slow.
