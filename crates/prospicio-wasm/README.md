# prospicio for JavaScript (WebAssembly)

The prospicio Rust core compiled to WebAssembly: actuarial reserving,
aggregate loss, reinsurance and risk-loaded pricing in a browser page, in
Node or in an Office.js Excel add-in, with no server. Every number comes
from the same Rust code as the Python and R packages and the Excel add-in.

```js
import { Triangle, mack, odpBootstrap } from "@prospicio/wasm";

const tri = Triangle.fromLong({
  origin: [2020, 2020, 2020, 2021, 2021, 2022], // a year, or "YYYY-MM"
  development: [12, 24, 36, 12, 24, 12],        // age in months
  value: [100, 150, 165, 110, 170, 120],
});
const fit = mack(tri, "value");
fit.totals.reserve[0];         // the chain ladder reserve
fit.totals.standard_error[0];  // Mack's standard error
const boot = odpBootstrap(tri, "value", { nSims: 10000, seed: 1 });
boot.reserves.tvar(0.99);      // on the simulated total
```

In a browser page with no bundler, import the `web` entry and wait for the
module:

```js
import init, { Dist } from "@prospicio/wasm/web";
await init();
```

## What is in it

| Area | Functions |
|---|---|
| Reserving | `Triangle.fromLong`, `chainLadder`, `mack`, `bornhuetterFerguson`, `odpBootstrap` |
| Distributions | `Dist.fromJson` (the JSON document Rust, Python and R save), `cdf`, `quantile`, `lev`, `layer`, `tvar` |
| Aggregate loss | `Count.poisson`, `Count.negativeBinomial`, `fft`, `simulateEvents`, `oep`, `occurrencePml` |
| Reinsurance | `Tower.fromJson`, `apply`, `applyAnnual` |
| Pricing | `Distortion.fromName`, `price`, `priceAtCostOfCapital` |

## Conventions

- Names are camelCase. The columns of a result table keep the core's names
  (`standard_error`), the column names of the Python and R data frames.
- A table is an object of columns, one array per column. Numeric columns
  are `Float64Array`s; a plain array or a typed array is accepted going in.
- An error is thrown as an `Error` whose `message` is the Rust core's. An
  unknown option name is an error, not ignored.
- Simulations run on one thread and are reproducible by `seed`: the same
  seed gives the same draws as in Rust, Python and R.
- An object of a class (`Triangle`, `Dist`, `Tower`, ...) holds memory in
  the module; call `.free()` when you are done with one in a long-running
  page.

## Building

`cargo xtask wasm` builds this package into `target/wasm/pkg` and runs its
tests under Node. It needs the `wasm32-unknown-unknown` target and the
`wasm-bindgen` command-line tool at the version this crate pins; the task
prints the install command when it is missing.

Design decisions: `docs/design/wasm.md`.
