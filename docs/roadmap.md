# Roadmap: milestones and the plan for each crate

2026-10-06. Where the project stands against the release plan in
[`architecture.md`](architecture.md#roadmap-and-releases), and what each
crate should do next. Lanes and owners are in [`AGENTS.md`](../AGENTS.md);
each lane's details are in its note under `docs/design/`. Update this file
when a milestone closes or the plan changes.

For an end-to-end use of the library, see
the notebook [`examples/one_year_view.ipynb`](../examples/one_year_view.ipynb),
saved with its results and charts. It covers
reserve risk, premium risk from GLMs and property per-risk reinsurance,
then joins them and sets and allocates capital.

## Milestones

| Stage | Scope (from `architecture.md`) | Status |
|---|---|---|
| Phase 0: Foundation | Workspace, core traits, design notes, RNG streams, Python and R skeletons, parity harness | Done |
| v0.1: Reserving core | Distributions, `Triangle`, Chain Ladder, Mack, ODP bootstrap with a joint `PredictiveDistribution` | Done; parity on RAA, GenIns and ABC |
| v0.2: Reserving breadth | BF, Cape Cod, Benktander, ELR, Clark, tails, one-year view (Merz–Wüthrich) | Done (#131–#136) |
| v0.3: Aggregate and reinsurance | Frequency-severity; FFT, Panjer, Monte Carlo; XOL, stop loss, reinstatements, towers, gross/ceded/net | Done, and more: surplus treaties, inuring stages, exact towers on the grid, risk profiles, exposure curves, reinstatements pro rata as to time |
| v0.4: Risk and capital | VaR, TVaR, CoTVaR, distortions, copulas, Iman–Conover, allocation, EVT tails | Done except vine and nested Archimedean copulas |
| ◆ Gate | Bootstrap reserve and tower results feed capital allocation end to end | Shown by `examples/one_year_view.ipynb`, whose code CI runs (`python/tests/test_examples.py`) |
| v0.5: GLM | IRLS, Tweedie, NB, regularization, GLM reserving, R parity | Done (statsmodels and glmnet parity, freMTPL2) |
| v0.6: GAM and pricing | P-splines, tensor smooths, GCV/REML; rate indication, ILF, MBBEFD, credibility | Part done: P-splines with GCV/UBRE, ILF, layer rating, MBBEFD and exposure curves. Still to do: REML, tensor smooths, rate indication, trend and on-level, credibility |
| v0.7: Bayesian | Specs and diagnostics, sampling with nutpie, Bayesian reserving and credibility | Part done: NUTS (nuts-rs), Bayesian GLM, stacking, ELPD. Still to do: Bayesian reserving and credibility |
| v0.8: Claim-level reserving | Event histories, payment and closure hazards, severity, ultimate | **Parked** (2026-10-06): large, and the design needs the user's thinking first. No lane works on it until the user picks it up |
| v0.9: Integrations | LightGBM and XGBoost adapters, PyTorch through ONNX, WASM build | Part done: the boosting adapters (Python and R). Still to do: ONNX import, WASM |
| ◆ Gate | API review and deprecation pass | — |
| v1.0: Stable core APIs | Semver for distributions, reserving, aggregate, reinsurance, risk, capital | — |

Work ran ahead of the plan's order: models (v0.5–v0.7) moved in parallel
with aggregate (v0.3–v0.4), because each has its own session. The rest of
the plan keeps the order above.

## Plan by crate

Each row lists what the crate holds now, what comes next in order, and
what it is aiming for by v1.0.

### `prospicio-core` (shared)

- **Now:** errors, periods and grains, `StreamRng` (ChaCha20 streams).
- **Next:** nothing planned. Changes come only as small PRs on their own
  (AGENTS.md, "Shared files"), for example if WASM needs a change to the
  RNG.
- **v1.0:** frozen; the stream scheme is part of reproducibility, so it
  changes only with a version bump.

### `prospicio-math` (Probability lane)

- **Now:** quadrature, linear algebra, one-dimensional minimizers (Brent,
  golden section), Nelder–Mead, root finding, special functions, splines.
- **Next, on demand:** log-determinants and Hessians for REML smoothing
  (`prospicio-glm`), and Sobol sequences for quasi-Monte Carlo (v1.x).
- **v1.0:** only the numerics other crates use. No general-purpose
  library.

### `prospicio-prob` (Probability lane)

- **Now:**
  - Distributions in three representations (parametric, `Grid`,
    `Sampled`), plus counts, the Pareto family, mixtures, `Custom` and
    the `Dist` enum with JSON save and load.
  - `PredictiveDistribution`: join, reorder, blend, Arrow IPC.
  - Risk measures and distortions, copulas (Gaussian, t, Archimedean),
    Iman–Conover, capital allocation (Euler, covariance, Shapley and
    others), EVT (GPD, POT, Hill).
- **Next:**
  1. Nested Archimedean and vine copulas (`docs/design/risk.md`).
  2. Threshold diagnostics for EVT.
  3. Signed grids for net cash flow and P&L, when capital needs them.
- **v1.0:** a stable distribution and `PredictiveDistribution` API that
  every other crate builds on.

### `prospicio-aggregate` (Aggregate lane)

- **Now:**
  - Panjer, FFT, Monte Carlo event sets (with sums insured and times),
    the collective model.
  - Layers on the loss or surplus basis, with annual terms and
    reinstatements pro rata as to amount and time.
  - Inuring towers, applied to events, to aggregates, or exactly on the
    grid.
- **Next:**
  1. Seasonal event times, from a density over the year.
  2. Loss corridors and other contract features listed in
     `architecture.md`.
  (Done: towers saved and loaded as JSON, so a programme can be stored
  and replayed.)
- **v1.0:** a reinsurance programme described once and applied to any
  loss source.

### `prospicio-pricing` (Aggregate lane)

- **Now:**
  - Layer rating, ILFs and Pareto extrapolation, tower matching.
  - Exposure curves (MBBEFD, tabulated, from a severity) and risk
    profiles with spread sums insured.
  - Risk-loaded prices from simulated losses.
- **Next:**
  1. Rate indication: trend, on-level premium and loss development into
     an indicated rate change. This is the v0.6 pricing scope.
  2. Credibility (limited fluctuation, Bühlmann, Bühlmann–Straub). It may
     become `prospicio-credibility` once it has a Bayesian side.
  3. Log-log interpolation for tabulated curves, and a tilted spread
     within bands.
- **v1.0:** treaty and primary pricing from data to a priced programme.

### `prospicio-reserving` (Reserving lane, local session)

- **Now:**
  - `Triangle`, Chain Ladder, Mack, ODP bootstrap and ODP GLM, segment
    fits, the diagonal backtest.
  - v0.2 methods: expected loss, BF, Benktander, Cape Cod, tails, Clark,
    Merz–Wüthrich.
- **Next (from that lane):**
  1. A simulated one-year view: re-reserving on the ODP bootstrap for any
     method.
  2. Bayesian reserving with `prospicio-bayes` (v0.7).
  Claim-level reserving (v0.8) is parked; see the milestones.
- **v1.0:** every reserve as a joint `PredictiveDistribution`, ready for
  capital.

### `prospicio-models` (Models lane)

- **Now:** families, links, `Terms` to `Design`, metrics, resampling,
  tuning, `compare`, stacking, monitoring, simulation from means.
- **Next:**
  1. Decide the model artifact format (`docs/design/models.md`).
  2. Survival models (Kaplan–Meier, Cox, parametric), for lapse and
     claim-closure models. Not urgent while claim-level reserving is
     parked.
- **v1.0:** one model protocol for every engine, native or delegated.

### `prospicio-glm` (Models lane)

- **Now:** GLM by IRLS, Tweedie with power profiling, elastic net with a
  cross-validated path, robust covariance, GAM with P-splines (GCV/UBRE).
- **Next:**
  1. REML smoothing.
  2. Tensor smooths.
  3. Cyclic and monotone smooths. This completes the GAM scope of v0.6.
- **v1.0:** mgcv's useful part for Tweedie, Poisson and gamma.

### `prospicio-nn` (Models lane, opt-in)

- **Now:** CANN and the attention CANN on Burn, with random search and
  early stopping.
- **Next:**
  1. Importing PyTorch models through ONNX (v0.9).
  2. Tabular MLPs with embeddings, and multi-task claim models.
- **v1.0:** networks behind the shared model protocol, in every front
  end.

### `prospicio-bayes` (Models lane, opt-in)

- **Now:** NUTS on nuts-rs, Bayesian GLM, Bayesian and hierarchical
  stacking, ELPD (PSIS-LOO, WAIC), MCMC diagnostics.
- **Next:**
  1. Bayesian chain ladder and compartmental reserving, with the
     Reserving lane.
  2. Hierarchical credibility.
  3. Posterior predictive checks.
- **v1.0:** our own model specs and diagnostics, with sampling delegated.

### `prospicio-python`, `prospicio-r`, `python/`, `R/` (each lane its own module)

- **Now:** every crate's API in both languages; docs generated from the
  doc comments (great-docs, pkgdown).
- **Next:**
  1. The WASM build (v0.9). The scope is decided (core, math, prob,
     aggregate with reinsurance, pricing, reserving) and CI checks that
     they build for `wasm32-unknown-unknown`. Next is a JavaScript API
     over them (`wasm-bindgen`) for a browser page or an Excel add-in.
  2. (Done: the docs sites are published to GitHub Pages from `main`.)
- **v1.0:** API review and deprecation pass, then semver.

## Waiting on the user

- Claim-level reserving (v0.8): parked until the user has thought through
  its design.

Decided 2026-10-07 (`architecture.md`, "Open decisions"): the name
(prospicio for the repository, the Rust crates and the Python and R
packages), the licence (MIT OR Apache-2.0), IP ownership (Ethan Kang) and
the WASM scope; the docs are published to GitHub Pages. Still open:

- the first release to PyPI and crates.io, which reserves the names. The
  release workflow and the umbrella crate `prospicio` are in place; the
  one-time registry setup in `docs/release.md` needs the user's accounts,
  then a pushed tag `v0.0.1` publishes. CRAN comes later.
