# Roadmap: milestones and the plan for each crate

2026-10-09. Where the project stands against the release plan in
[`architecture.md`](architecture.md#roadmap-and-releases), and what each
crate should do next. Lanes and owners are in [`AGENTS.md`](../AGENTS.md);
each lane's details are in its note under `docs/design/`. Update this file
when a milestone closes or the plan changes.

For an end-to-end use of the library, see
the notebook [`examples/one_year_view.ipynb`](../examples/one_year_view.ipynb),
saved with its results and charts. It covers
reserve risk, premium risk from GLMs and property per-risk reinsurance,
then joins them and sets and allocates capital. The notebook
[`examples/reinsurance_pricing.ipynb`](../examples/reinsurance_pricing.ipynb)
prices towers, aggregate limits, reinstatements, stop losses, variable
premiums and commissions, and risk-adjusted premiums, each beside the
numbers of Mildenhall's `aggregate`. The notebook
[`examples/submission_evaluation.ipynb`](../examples/submission_evaluation.ipynb)
evaluates a synthetic per-risk excess of loss submission end to end, from
the broker's raw files to a decision on each layer, and compares its
estimates with the generator's true parameters. The [guides](guides/README.md) explain
the ideas in prose for actuaries, starting with
[what a reinsurance price is](guides/reinsurance-price.md); CI runs their code.
Next for the guides, waiting on the user: when to price with a distortion
and when with a capital standard, when to rate with an exposure curve and
when from experience, and which pricing method fits which scenario.

## Milestones

| Stage | Scope (from `architecture.md`) | Status |
|---|---|---|
| Phase 0: Foundation | Workspace, core traits, design notes, RNG streams, Python and R skeletons, parity harness | Done |
| v0.1: Reserving core | Distributions, `Triangle`, Chain Ladder, Mack, ODP bootstrap with a joint `PredictiveDistribution` | Done; parity on RAA, GenIns and ABC |
| v0.2: Reserving breadth | BF, Cape Cod, Benktander, ELR, Clark, tails, one-year view (Merz–Wüthrich) | Done (#131–#136) |
| v0.3: Aggregate and reinsurance | Frequency-severity; FFT, Panjer, Monte Carlo; XOL, stop loss, reinstatements, towers, gross/ceded/net | Done, and more: surplus treaties, inuring stages, exact towers on the grid, risk profiles, exposure curves, reinstatements pro rata as to time, loss corridors, contract terms (ceding, sliding-scale and profit commissions, swing and retro rating) |
| v0.4: Risk and capital | VaR, TVaR, CoTVaR, distortions, copulas, Iman–Conover, allocation, EVT tails | Done except vine and nested Archimedean copulas |
| ◆ Gate | Bootstrap reserve and tower results feed capital allocation end to end | Shown by `examples/one_year_view.ipynb`, whose code CI runs (`python/tests/test_examples.py`) |
| v0.5: GLM | IRLS, Tweedie, NB, regularization, GLM reserving, R parity | Done (statsmodels and glmnet parity, freMTPL2) |
| v0.6: GAM and pricing | P-splines, tensor smooths, GCV/REML; rate indication, ILF, MBBEFD, credibility | Part done: P-splines with GCV/UBRE, ILF, layer rating, MBBEFD and exposure curves, classical credibility (limited fluctuation, Bühlmann–Straub), rate indication with trend and on-level premium. Still to do: REML, tensor smooths |
| v0.7: Bayesian | Specs and diagnostics, sampling with `nuts-rs` (nutpie's Rust core), Bayesian reserving and credibility | Part done: NUTS (nuts-rs), Bayesian GLM, stacking, ELPD, the Bayesian ODP chain ladder (`BayesOdp`, Rust, Python and R), Meyers' CRC, CSR and CAY (`Meyers`, Rust, Python and R), `BayesOdp` and `Meyers` over segments (Rust, Python and R) and in the diagonal backtest (Rust), the Bayesian Bornhuetter–Ferguson (`BayesBf`, Rust, Python and R). Still to do: the rest of Bayesian reserving (`docs/design/reserving-bayes.md`) and credibility |
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
  golden section), Nelder–Mead, root finding, special functions (with the
  incomplete beta and gamma past their usual domain, the exponential
  integral and the Mills ratio), splines.
- **Next, on demand:** log-determinants and Hessians for REML smoothing
  (`prospicio-glm`), and Sobol sequences for quasi-Monte Carlo (v1.x).
- **v1.0:** only the numerics other crates use. No general-purpose
  library.

### `prospicio-prob` (Probability lane)

- **Now:**
  - Distributions in three representations (parametric, `Grid`,
    `Sampled`), plus counts, the Pareto family, the inverse gamma,
    inverse Gaussian, Burr and scaled beta, mixtures, truncation and
    splicing, `Custom` and the `Dist` enum with JSON save and load.
    Third raw moments on `Severity` and `Counting`; `aggregate`'s named
    fixed, Bernoulli and geometric counts.
  - `PredictiveDistribution`: join, reorder, blend, Arrow IPC; provenance
    records the stream scheme and, apart from it, the sampler versions
    (`docs/design/rng.md`).
  - Risk measures and distortions, copulas (Gaussian, t, Archimedean)
    with their distribution functions, Iman–Conover, capital allocation (Euler, covariance, Shapley and
    others), EVT (GPD, POT, Hill); TVaR of a grid and `tvar_threshold`.
  - `Discrete`, a distribution on finitely many values of either sign,
    for the P&L ledger's rows (`docs/design/pnl.md`).
- **Next:**
  1. Nested Archimedean and vine copulas (`docs/design/risk.md`).
  2. Threshold diagnostics for EVT.
  3. Signed regular grids for rebucketed P&L legs (`pnl.md`, PR 3).
- **v1.0:** a stable distribution and `PredictiveDistribution` API that
  every other crate builds on.

### `prospicio-aggregate` (Aggregate lane)

- **Now:**
  - Panjer, FFT (with the grid sized automatically, as `aggregate`),
    Monte Carlo event sets (with sums insured and times), the collective
    model.
  - Layers on the loss or surplus basis, with annual terms and
    reinstatements pro rata as to amount and time.
  - Inuring towers, applied to events, to aggregates, or exactly on the
    grid.
  - Contract terms on a layer, with `aggregate`'s definitions: premium
    as a rate on line, a rate or a deposit; flat and sliding-scale ceding
    commissions, profit commissions, swing rating; retro premiums.
    (Also done: towers saved as JSON, seasonal event times, loss
    corridors.)
  - The occurrence exceedance (OEP) curve and occurrence PMLs for any
    count, exact and simulated; compound moments to third order and the
    moment approximations (normal, gamma, lognormal, shifted gamma and
    lognormal); per-occurrence layer statistics and each view of a tower
    priced under a distortion, the grid towers checked against
    `aggregate` point by point; the senior and junior priority ladder
    for capital, exact under any dependence from scenarios; eventual ruin
    by Pollaczeck–Khinchine; a severity reshaped to layer loss picks;
    pricing bounds for other risks and the envelope of the distortions;
    the total cession of a stage with several layers, priced, and
    statistics of total cessions and aggregate covers; each unit priced
    alone against the book, the comonotonic allocation, and the expected
    reinstatement premium checked against `aggregate` (tier 2 complete);
    a tower on an annual loss grid and risk-loaded prices on a grid at any
    capital standard (the Guides lane's #206 and #204); a cover's quote,
    from risk premium to technical premium and back from a market quote
    (#197); experience rating from large losses, trend to the burning
    cost, with each claim developed to ultimate, count IBNR, FX and an
    index clause (#190, `docs/design/experience-rating.md`), and as-if
    history under each year's program (#191); the time value of money in
    a price, with capital held over the payout (#195,
    `docs/design/time-value.md`); catastrophe event loss tables with
    secondary uncertainty, year loss tables, merged event sets and the
    hours clause (#196, `docs/design/cat-inputs.md`); two lines hit by the
    same events, by a 2-D compound FFT, `aggregate`'s clash model, and
    the two lines' losses in an event coupled by a copula; renewal claim
    counts, with any law for the waits between claims, and their eventual
    ruin by Wiener–Hopf.
- **Next:**
  1. The last large item parked from `aggregate` parity
     (`docs/design/aggregate.md`): the P&L ledger, planned in six PRs in
     `docs/design/pnl.md`, starting with a signed discrete distribution.
  2. Deficit and credit carry-forward in profit commissions and sliding
     scales, once multi-year simulation exists.
- **v1.0:** a reinsurance programme described once and applied to any
  loss source.

### `prospicio-pricing` (Aggregate lane)

- **Now:**
  - Layer rating, ILFs and Pareto extrapolation, tower matching.
  - Exposure curves (MBBEFD, tabulated, from a severity) and risk
    profiles with spread sums insured.
  - Risk-loaded prices from simulated losses.
  - Classical credibility: limited fluctuation standards and the
    Bühlmann–Straub model (`docs/design/credibility.md`). It may become
    `prospicio-credibility` once it has a Bayesian side.
  - Ratemaking: premium at the current rate level by the parallelogram
    method, trend fits, factors and lengths, and the overall rate
    indication by the loss ratio and pure premium methods with credibility
    against a complement (`docs/design/ratemaking.md`). This finishes the
    v0.6 pricing scope.
- **Next:**
  1. Log-log interpolation for tabulated curves (the tilted spread within
     bands is done).
- **v1.0:** treaty and primary pricing from data to a priced programme.

### `prospicio-reserving` (Reserving lane, local session)

- **Now:**
  - `Triangle`, Chain Ladder, Mack, ODP bootstrap and ODP GLM, segment
    fits, the diagonal backtest.
  - v0.2 methods: expected loss, BF, Benktander, Cape Cod, tails, Clark,
    Merz–Wüthrich.
  - The simulated one-year view, re-reserving on the ODP or Mack bootstrap
    for any method, weighting, tail or grain (#150, #156, #157), and
    `MackBootstrap`'s run-off view with centred residuals by default,
    reconciled with Mack and Merz–Wüthrich (#159, #160, #163).
  - Tails in both bootstraps, lifetime and one-year view: an estimated
    tail refitted on each simulation's pseudo factors, a constant one drawn
    from the lognormal with its standard error, and process error past the
    oldest age (Mack's tail sigma, the ODP's scale); reconciled with R's
    `MackChainLadder(tail = ...)` (`reserving-v02.md`, decision 9), and
    combined with every dependence between segments. In the one-year view
    the tail step is one more development period, closing the origins it
    falls on in the coming year, checked against targets built from R's
    `CDR(1)S.E.` (scaled by the tail) and `Mack.S.E` with the tail; R's
    `CDR` has no tail (decision 11).
  - Joint reserves across lines, for capital: `SegmentDependence` on both
    bootstraps' multi-segment fits, lifetime and one-year (synchronized
    residuals, Kirschner, Kerley and Isaacs 2008, or a Spearman rank
    correlation by Iman–Conover), into `capital` (`reserving-v02.md`,
    decision 10).
  - The Bayesian chain ladder, `BayesOdp`, behind the `bayes` feature
    (`reserving-bayes.md`, PR 1): the ODP model with log-linear effects
    and the chain ladder's scale, sampled with NUTS, reserves as a
    `PredictiveDistribution`, at parity with the same model in Stan on
    GenIns and RAA, in Python (`BayesOdp`) and R (`bayes_odp()`).
  - Meyers' CRC and CSR, `Meyers` (`reserving-bayes.md`, PR 3): log-normal
    cumulative values with premium, his priors, at parity with his own
    Stan code and his printed results on group 353 Commercial Auto, in
    Python (`Meyers`) and R (`meyers()`).
  - `BayesOdp` over segments (`fit_segments`, independent or rank
    correlated) and in the diagonal backtest (`reserving-bayes.md`, PR 2;
    `fit_segments` in Python (`BayesOdp.fit_segments`) and R
    (`bayes_odp_segments()`)).
  - The Bayesian Bornhuetter–Ferguson, `BayesBf` (`reserving-bayes.md`,
    PR 4): England, Verrall and Wüthrich's gamma priors on the origins'
    row parameters, at parity with the same model in Stan and with
    their printed totals, in Python (`BayesBf`) and R (`bayes_bf()`).
  - Meyers' CAY, `MeyersModel::Cay` (`reserving-bayes.md`, PR 5): CRC
    with the previous origin's residual at the same age in each mean, at
    parity with his own `CAY.R` Stan code and his printed results on
    group 353 incurred, in Rust, Python (`Meyers("cay")`) and R
    (`meyers(model = "cay")`).
  - Meyers' models over segments and in the diagonal backtest
    (`reserving-bayes.md`, PR 6): `Meyers::fit_segments` with
    independent or rank-correlated segments, and `TriangleModel for
    Meyers`, which rebuilds cumulative values and premium from the
    frame and simulates each origin's path jointly; `fit_segments` in
    Python (`Meyers.fit_segments`) and R (`meyers_segments()`).
- **Next (from that lane):** Bayesian reserving with `prospicio-bayes`
  (v0.7), by the PRs of `reserving-bayes.md`:
  in the user's order (2026-10-13, open decision 9): the compartmental
  models of `reserving-compartmental.md`, whose PR A (the `Scalar` trait
  and an in-house dual type, checked against every hand-written gradient;
  internal) is done, with PR B (`Compartmental`, Gesmann and Morris'
  model 1) next (the 200-triangle calibration study can now use
  `TriangleModel for Meyers`);
  then the one-year view by PSIS reweighting (open decision 6).
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
  1. Bayesian reserving, with the Reserving lane, which builds it in
     `prospicio-reserving`'s `bayes` feature on this crate's NUTS
     (`docs/design/reserving-bayes.md`): the Bayesian ODP chain ladder
     (PR 1), Meyers' CRC and CSR (PR 3) and CAY (PR 5) are in, and the
     Bayesian Bornhuetter–Ferguson (PR 4) too, and both `BayesOdp` and
     `Meyers` over segments and in the backtest (PRs 2 and 6), with no
     change here;
     compartmental reserving follows in a later note.
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

## Order of work (set by the user 2026-10-09)

1. Contract terms in `prospicio-aggregate` (done).
2. The R package self-contained, so `remotes::install_github` and
   r-universe can build it without a checkout of the whole repository
   (done for `install_github`: `R/prospicio/bootstrap.R` copies the crates
   into the tarball, and CI installs it that way. r-universe needs a
   registry repository from the user and has not been tried. CRAN still
   needs the third-party crates vendored).
3. Rate indication and credibility in `prospicio-pricing` (credibility
   done: limited fluctuation and Bühlmann–Straub; rate indication done,
   with premium at the current rate level and trend).
4. Release v0.0.2 (done 2026-10-12, at the user's call before rate
   indication, which moves to v0.0.3). Its contents are in `CHANGELOG.md`.
   For each later release, coordinated between the sessions (decided
   2026-10-12):
   - One release PR changes the workspace version in the root
     `Cargo.toml` (`workspace.package.version` and the `prospicio-*`
     entries in `[workspace.dependencies]`) and moves `CHANGELOG.md`'s
     "Unreleased" items into the new version's section. Nothing else goes
     in that PR.
   - Each feature PR adds a fragment in `changes/changelog/`, which the
     release PR folds into "Unreleased" in `CHANGELOG.md`, so the changelog
     says which release a feature ships in.
   - Open PRs either merge before the release PR or wait for the next
     version; neither session changes versions or the release workflow
     outside that PR.
   - After the release PR merges, the user (or the local session at the
     user's word) runs the Release workflow by hand on `main` (a dry
     run), then pushes the tag (`docs/release.md`); the cloud session
     cannot push tags.
5. Bayesian reserving (Meyers, compartmental, Clark growth curves,
   Gaussian processes), with `prospicio-bayes`: the local session
   (Reserving lane) takes it, so the cloud session stays out of
   `prospicio-reserving` and `prospicio-bayes`. Credibility in item 3 is
   the classical kind (limited fluctuation, Bühlmann, Bühlmann–Straub);
   hierarchical and Bayesian credibility stay with v0.7.

## Waiting on the user

- Claim-level reserving (v0.8): parked until the user has thought through
  its design.

Decided 2026-10-07 (`architecture.md`, "Open decisions"): the name
(prospicio for the repository, the Rust crates and the Python and R
packages), the licence (MIT OR Apache-2.0), IP ownership (Ethan Kang) and
the WASM scope; the docs are published to GitHub Pages. Still open:

- ~~the first release to PyPI and crates.io~~: done 2026-10-08, v0.0.1
  (PyPI `prospicio` and the 11 crates; `knowledge/environment/publishing.md`).
  CRAN comes later.
