# JavaScript lane: done

The lane's finished items, oldest first, from [../javascript.md](../javascript.md). Append-only: a PR appends its Done bullets at the end and never edits, reorders or reflows existing ones. One bullet per line; this file merges with git's union driver (.gitattributes).

## Done

- #576 the JavaScript package over the WASM-scope crates (`docs/design/wasm.md`, decisions 1 to 9 proposed; branch `claude/wasm-package`; the user took #576 up again on 2026-11-20 for 0.2.1): `crates/prospicio-wasm` with `wasm-bindgen`, one module per lane (`Triangle.fromLong`, `chainLadder`, `mack`, `bornhuetterFerguson`, `odpBootstrap`; `Dist.fromJson` with `cdf`, `quantile`, `lev`, `layer`, `tvar`; `Count`, `fft`, `simulateEvents`, `Tower.fromJson` and `apply`, `oep`, `occurrencePml`; `Distortion.fromName`, `price`, `priceAtCostOfCapital`; `Predictive`), tables as objects of `Float64Array` columns under the core's column names, core errors thrown as `Error`s and an unknown option an error; `cargo xtask wasm` builds the npm package (bundler, Node and web glue; 1.53 MB, 0.51 MB gzipped) and runs its JavaScript tests; the Rust tests run under Node against the R ChainLadder, mpmath and `aggregate` parity rows and equal the core bit for bit on the bootstrap's draws and a tower's cessions; CI's `wasm` job; publishing to npm is not built (open decisions O1 and O2)
