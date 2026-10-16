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
| v0.2: Reserving breadth | BF, Cape Cod, Benktander, ELR, Clark, tails, one-year view (Merz–Wüthrich) | Done (#131–#136); selected development patterns (`reserving-v02.md` decision 12, in Rust, Python and R) |
| v0.3: Aggregate and reinsurance | Frequency-severity; FFT, Panjer, Monte Carlo; XOL, stop loss, reinstatements, towers, gross/ceded/net | Done, and more: surplus treaties, inuring stages, exact towers on the grid, risk profiles, exposure curves, reinstatements pro rata as to time, loss corridors, contract terms (ceding, sliding-scale and profit commissions, swing and retro rating) |
| v0.4: Risk and capital | VaR, TVaR, CoTVaR, distortions, copulas, Iman–Conover, allocation, EVT tails | Done, with nested Archimedean and C- and D-vine copulas (#261, #262) |
| ◆ Gate | Bootstrap reserve and tower results feed capital allocation end to end | Shown by `examples/one_year_view.ipynb`, whose code CI runs (`python/tests/test_examples.py`) |
| v0.5: GLM | IRLS, Tweedie, NB, regularization, GLM reserving, R parity | Done (statsmodels and glmnet parity, freMTPL2) |
| v0.6: GAM and pricing | P-splines, tensor smooths, GCV/REML; rate indication, ILF, MBBEFD, credibility | Done: P-splines, tensor product, cyclic and monotone smooths with GCV, UBRE or REML (#263), ILF, layer rating, MBBEFD and exposure curves, classical credibility (limited fluctuation, Bühlmann–Straub), rate indication with trend and on-level premium |
| v0.7: Bayesian | Specs and diagnostics, sampling with `nuts-rs` (nutpie's Rust core), Bayesian reserving and credibility | Part done: NUTS (nuts-rs), Bayesian GLM, stacking, ELPD, the Bayesian ODP chain ladder (`BayesOdp`, Rust, Python and R), Meyers' CRC, CSR and CAY (`Meyers`, Rust, Python and R), `BayesOdp` and `Meyers` over segments (Rust, Python and R) and in the diagonal backtest (Rust), the Bayesian Bornhuetter–Ferguson (`BayesBf`, Rust, Python and R), Bayesian hierarchical credibility (Rust, Python and R), Gesmann and Morris' compartmental models 1 and 2 (`Compartmental`, Rust, Python and R), the one-year view of `BayesOdp` and `BayesBf` by PSIS reweighting, one generic engine with a conformance suite that every Bayesian model with a one-year view passes (`BayesOdp`, `BayesBf`; Rust, Python and R). Still to do: the rest of Bayesian reserving (`docs/design/reserving-bayes.md`) |
| v0.8: Claim-level reserving | Event histories, payment and closure hazards, severity, ultimate | **Parked** (2026-10-06): large, and the design needs the user's thinking first. No lane works on it until the user picks it up |
| v0.9: Integrations | LightGBM and XGBoost adapters, PyTorch through ONNX, WASM build | Part done: the boosting adapters (Python and R), ONNX import (`prospicio_nn::onnx`, Rust). Still to do: WASM |
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
  integral and the Mills ratio), splines, symmetric eigenvalues (Jacobi,
  for REML's penalty determinants).
- **Next, on demand:** Sobol sequences for quasi-Monte Carlo (v1.x).
- **v1.0:** only the numerics other crates use. No general-purpose
  library.

### `prospicio-prob` (Probability lane)

- **Now:**
  - Distributions in three representations (parametric, `Grid`,
    `Sampled`), plus counts, the Pareto family, the inverse gamma,
    inverse Gaussian, Burr and scaled beta, mixtures, truncation and
    splicing, a severity under a mix of policy limits, `Custom` and the
    `Dist` enum with JSON save and load; standard errors of the Pareto
    fits, and maximum likelihood fits of the lognormal, gamma, Weibull and
    Burr with reporting thresholds and censoring.
    Third raw moments on `Severity` and `Counting`; `aggregate`'s named
    fixed, Bernoulli and geometric counts.
  - `PredictiveDistribution`: join, reorder, blend, select (one part of a
    result, such as a tower's cession, before joining it), Arrow IPC; provenance
    records the stream scheme and, apart from it, the sampler versions
    (`docs/design/rng.md`).
  - Risk measures and distortions, copulas (Gaussian, t, Archimedean,
    nested Archimedean, C- and D-vines)
    with their distribution functions, Iman–Conover, capital allocation (Euler, covariance, Shapley and
    others), EVT (GPD, POT, Hill, and threshold diagnostics: the mean
    excess and the GPD's shape and modified scale at each threshold with
    standard errors, as R's `ismev`); TVaR of a grid and `tvar_threshold`.
  - `Discrete`, a distribution on finitely many values of either sign,
    for the P&L ledger's rows (`docs/design/pnl.md`).
- **Next:** open.
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
    Over a multi-year term (`Tower::apply_term`), profit commission
    deficits and sliding scale deficits and credits carried forward,
    to extinction or for a number of years. (Also done: towers saved as
    JSON, seasonal event times, loss corridors.)
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
- **Next:** open. (Done: deficit and credit carry-forward in profit
  commissions and sliding scales over multi-year terms,
  `Tower::apply_term`, `docs/design/aggregate.md`; the P&L ledger, the
  last large item parked from `aggregate` parity: `docs/design/pnl.md`,
  PRs 1 to 6b and follow-ups 7a to 7c.)
- **v1.0:** a reinsurance programme described once and applied to any
  loss source.

### `prospicio-pricing` (Aggregate lane)

- **Now:**
  - Layer rating, ILFs and Pareto extrapolation, tower matching.
  - Exposure curves (MBBEFD, tabulated, from a severity, with the Swiss
    Re `c` calibrated to a mean rate or a layer's share) and risk
    profiles with spread sums insured.
  - Risk-loaded prices from simulated losses.
  - Classical credibility: limited fluctuation standards and the
    Bühlmann–Straub model (`docs/design/credibility.md`). Its Bayesian,
    hierarchical side is in `prospicio-bayes`; no `prospicio-credibility`
    crate is needed yet.
  - Ratemaking: premium at the current rate level by the parallelogram
    method, trend fits, factors and lengths, and the overall rate
    indication by the loss ratio and pure premium methods with credibility
    against a complement (`docs/design/ratemaking.md`). This finishes the
    v0.6 pricing scope.
  - Log-log interpolation of tabulated exposure curves, where it keeps
    the curve concave.
- **Next:** open.
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
  - Gesmann and Morris' hierarchical compartmental model with one
    outstanding compartment, model 1 (`reserving-compartmental.md` PR B):
    `Compartmental` fits paid and outstanding jointly with
    NUTS on a density written once over dual numbers, with priors a
    required argument (`CompartmentalPriors::gesmann_morris()` the
    paper's case study), a prior predictive check, the likelihood, scale,
    paid response and fitted pair as settings (other values an error
    until their PR), a warning when a draw's rates are swapped or
    `RateOrder::Ordered`; at parity with the paper's brms code
    reconstructed in Stan and its printed posterior on company 337; in
    Python (`Compartmental`, `CompartmentalPriors.gesmann_morris()`) and
    R (`compartmental()`, `gesmann_morris_priors()`).
  - Gesmann and Morris' model 2 (`reserving-compartmental.md` PR C):
    the log-normal likelihood, loss ratios to premium and incremental
    paid as built settings (any combination), origin effects on the rates
    and development-year effects set in the priors
    (`CompartmentalPriors::gesmann_morris_2()` the paper's preset), new
    development years drawn from the population; at parity with the
    paper's model 2 code reconstructed in Stan and its printed posterior
    on company 337; in Python (`Compartmental.model_2`,
    `CompartmentalPriors.gesmann_morris_2()`) and R
    (`compartmental_model_2()`, `gesmann_morris_2_priors()`).
  - The one-year view of `BayesOdp` by Pareto smoothed importance
    sampling (`reserving-bayes.md` PR 8, decision 16): each simulated
    year reweights the posterior draws, and a simulation whose Pareto
    `k` is above 0.7 is refitted with NUTS; one segment or several
    (independent or rank correlated), in Rust, Python and R; its claims development
    result agrees with the ODP bootstrap's one-year view on RAA and
    GenIns.
  - One one-year engine for every Bayesian model (`reserving-bayes.md`
    PR 9, decision 17): a model implements one interface (`YearModel`:
    draw the year, its log likelihood, the reserves before and after it,
    a refit and the log posterior) and gets the view, written once, on
    `prospicio_bayes::elpd::psis`; a conformance suite with no external
    reference runs on every implementor. `BayesBf::one_year` is the
    second model (Python `BayesBf.one_year`, R `bayes_bf_one_year()`):
    under a flat prior it agrees with `BayesOdp`'s view, under a tight
    one with the closed form of the strong-prior limit. Moment matching
    (decision 18, opt-in `MomentMatchThenRefit`, in Rust, Python and R)
    moves the posterior draws of a flagged simulation by affine maps
    before falling back to a refit: on RAA and GenIns it rescues nearly
    every flagged simulation, as accurately as a refit, at about a
    twenty-fifth of the cost.
  - Selected development patterns (`reserving-v02.md`, decision 12;
    branch `claude/selected-patterns`, in Rust, Python and R):
    `SelectedPattern` on `Development` as age-to-age factors, factors
    to ultimate or percents developed, partial or full, blended with
    the estimate by a weight, for the chain ladder, the expected-loss
    family, Mack with Merz–Wüthrich and both bootstraps. Selected ages
    carry no parameter error (a blend `(1 - w)` of it), so a pattern
    lowers the reported uncertainty; a cumulative pattern's level is
    rescaled to the tail. Parity with chainladder-python's
    `DevelopmentConstant` and with R's `MackChainLadder` on the
    triangle's own factors.
- **Next (from that lane):** Bayesian reserving with `prospicio-bayes`
  (v0.7), by the PRs of `reserving-bayes.md`:
  in the user's order (2026-10-13, open decision 9): the compartmental
  models of `reserving-compartmental.md`, whose PR A (the `Scalar` trait
  and an in-house dual type, checked against every hand-written gradient;
  internal), PR B (`Compartmental`, Gesmann and Morris' model 1, in
  Rust, Python and R) and PR C (model 2, in Rust, Python and R) are
  done (the 200-triangle calibration study can now use `TriangleModel
  for Meyers`);
  then the same one-year view for `Meyers` and `Compartmental`
  through the generic interface (decision 17), once the user sets the
  defaults for their refit cost (moment matching, built in decision
  18, may make that cost rare).
  Claim-level reserving (v0.8) is parked; see the milestones.
- **v1.0:** every reserve as a joint `PredictiveDistribution`, ready for
  capital.

### `prospicio-models` (Models lane)

- **Now:** families, links, `Terms` to `Design`, metrics, resampling,
  tuning, `compare`, stacking, monitoring, simulation from means.
  The model artifact format is decided: versioned JSON with one
  envelope and per-format versions (`docs/design/models.md`).
- **Next:**
  1. Survival models (Kaplan–Meier, Cox, parametric), for lapse and
     claim-closure models. Not urgent while claim-level reserving is
     parked.
- **v1.0:** one model protocol for every engine, native or delegated.

### `prospicio-glm` (Models lane)

- **Now:** GLM by IRLS, Tweedie with power profiling, elastic net with a
  cross-validated path, robust covariance, GAM with P-splines, tensor
  product, cyclic and monotone smooths (GCV/UBRE and REML; monotone ones
  GCV/UBRE, as scam). This completes the GAM scope of v0.6.
- **Next:** open.
- **v1.0:** mgcv's useful part for Tweedie, Poisson and gamma.

### `prospicio-nn` (Models lane, opt-in)

- **Now:** CANN and the attention CANN on Burn, with random search and
  early stopping; networks trained elsewhere imported from ONNX for
  inference (`prospicio_nn::onnx`, v0.9).
- **Next:**
  1. Tabular MLPs with embeddings, and multi-task claim models.
  2. Bindings for `prospicio-nn` (CANN and ONNX import) in Python and R,
     most likely to be added (user, 2026-10-16); the timing is open,
     since they put Burn in the wheel and the R package.
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
  2. (Done: hierarchical credibility, `prospicio_bayes::credibility`,
     Bühlmann–Straub's model and Jewell's hierarchy sampled with the means
     integrated out; parity with `actuar` at given structure parameters
     and with grid integration of the posterior;
     `docs/design/credibility.md`.)
  3. (Done: posterior predictive checks, `prospicio_bayes::ppc`: test
     statistics with posterior predictive p-values, within groups too,
     and the LOO-PIT at parity with R loo; on `BayesGlmFit` and for any
     model's replicates; `docs/design/models.md`.)
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

### `prospicio-xll` (Excel lane)

- **Now:** a prototype XLL for 64-bit desktop Excel (`docs/design/xll.md`):
  the chain ladder and Mack with standard errors as worksheet functions,
  built by `cargo xtask xll`, a local file, tested by the user in Excel 365;
  layer pricing (LEV, layer loss, layer CV, ILF) on a Pareto or lognormal
  severity; the ODP bootstrap exhibit; built, smoke-tested and kept as an
  artifact by CI, attached to a draft release on a version tag; the
  bootstrap's draws as a table; Mack's bootstrap; the expected-loss
  methods (expected loss, BF, Benktander, Cape Cod); Mack's sigma rule and
  selected factors; the one-year view (Merz-Wuthrich, both bootstraps);
  Clark's LDF and Cape Cod; triangle helpers; severities over every
  parametric family; the Pareto helpers; the piecewise and generalized
  Pareto, the layer variance; exposure curves (MBBEFD, Swiss Re,
  tabulated) and their calibrations; an exposure-rated risk profile;
  classical credibility; ratemaking (rate level, trend and the
  indication).
- **Next:**
  1. Parity with the Python and R packages where a function has a sheet
     form (`xll.md`, parity plan): burning cost, then aggregates.
  2. Signing, once the owner has a code-signing certificate (`xll.md`,
     decision 16).

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
