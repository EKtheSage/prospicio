### JavaScript (WebAssembly)

- A JavaScript package over the WASM-scope crates, `crates/prospicio-wasm`
  (`wasm-bindgen`), built by `cargo xtask wasm` for bundlers, Node and a
  plain browser page (#576): `Triangle.fromLong`, `chainLadder`, `mack`,
  `bornhuetterFerguson` and `odpBootstrap`; `Dist.fromJson` with `cdf`,
  `quantile`, `lev`, `layer` and `tvar`; `Count`, `fft`, `simulateEvents`,
  `Tower.fromJson` with `apply`, `oep` and `occurrencePml`;
  `Distortion.fromName`, `price` and `priceAtCostOfCapital`. Tables are
  objects of `Float64Array` columns under the core's column names, a core
  error is a thrown `Error`, and a seed gives the same draws as in Rust,
  Python and R. Not yet published to npm.
