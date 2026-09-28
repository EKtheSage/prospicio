# Actuarial Modeling Platform — Architecture v2

2026-09-26 · @Ethan

> Snapshot of the [Architecture v2 doc](https://claude.ai/code/artifact/355749a6-b01b-4cc4-92d2-b6d458fb6b76) as of 2026-09-27. Update this file when the plan changes.

## Vision and core principle

Build a composable actuarial modeling runtime: a Rust core that owns the numerical algorithms, exposed through Python and R, with one shared set of primitives feeding reserving, aggregate risk, reinsurance, capital, pricing and claim-level modeling.

**Core principle:** implement mathematical and modeling capabilities once, then compose them into actuarial applications.

**Rust earns its place in three ways, and the plan is prioritized around them:**

1. **Simulation speed** — bootstrap, Monte Carlo, FFT aggregation, towers applied to millions of events, claim-level models. Deterministic methods on small triangles gain nothing from Rust.
2. **One kernel, many front ends** — Python, R, and a WASM build that can run inside the browser and an Office.js Excel add-in with no server.
3. **Reproducibility** — deterministic parallel results, which model governance requires.

**Rust for the math we own, protocols for everything else.** Where a mature engine exists (samplers, gradient boosting, deep learning), the platform defines the interface and delegates the engine.

## What changed from v1

v2 keeps the v1 layering, principle and target scope; it changes priorities, build-vs-integrate choices, and makes the load-bearing abstractions explicit.

| Area | v1 | v2 |
| --- | --- | --- |
| First release | Triangle + deterministic Chain Ladder | Triangle + Chain Ladder + Mack + ODP bootstrap returning a joint `PredictiveDistribution` |
| Phase order | GLM, GAM, pricing before capital | Aggregate, reinsurance, risk measures and capital pulled forward; GLM/GAM/pricing after |
| Bayesian | Native Gibbs, MH, HMC, NUTS | Own model specs and diagnostics; samplers via nutpie / BridgeStan |
| GAM | Aim at mgcv's useful portion | P-splines, tensor smooths, GCV/REML for Tweedie/Poisson/Gamma only |
| Tree boosting | Rust adapters for LightGBM/XGBoost | Adapters in the Python/R layer, emitting shared prediction objects |
| Neural | Burn-native training | PyTorch in Python implementing the shared protocols; Burn optional later |
| Crates | ~25 crates planned | ~5 crates at start; split when compile time or dependency weight forces it |
| Core abstractions | Named, not designed | Distribution representations, joint predictive distributions, Triangle model, RNG streams designed in Phase 0 |
| Front ends | Python + R | Python first; R skeleton in Phase 0, parity later; WASM target added |
| Reserving scope | Ultimate-view methods | Adds one-year view / CDR (Merz–Wüthrich) |
| Validation | "Testing framework" | Parity tests against reference packages on standard datasets as a release gate |
| Relationship to chainladder-python | Unstated | Explicit decision required (see Open decisions) |

## Layered architecture

Five layers, each depending only on the layers below it; the three-way split from v1 (math primitives, modeling engines, actuarial applications) is unchanged.

```text
Front ends          Python (PyO3 + maturin) · R (extendr) · WASM (browser, Excel add-in)
                    Thin wrappers only: no algorithms live here
        │
        ▼
Actuarial domains   Reserving · Aggregate · Reinsurance · Capital · Credibility · Pricing · Claim-level · EVT
                    Compose engines and risk math; hold no numerical code of their own
        │
        ▼
Modeling engines    Native in Rust: GLM · GAM · Survival
                    Delegated behind shared protocols: Bayesian samplers · GBDT · Neural nets
        │
        ▼
Risk mathematics    Distributions · PredictiveDistribution · Dependence · Risk measures · Transforms · Processes
  (the hub)         Every domain reads and writes these objects
        │
        ▼
Numerical engine    Linear algebra (faer) · Optimization · Integration · FFT · Special functions
                    Stream-indexed RNG · Rayon outer-loop parallelism
```

A shared evaluation subsystem (Section: Evaluation and validation) sits beside the stack and scores any model that emits the common prediction objects, native or delegated.

**Rule:** domain crates never implement their own numerics. Chain Ladder bootstrap produces a `PredictiveDistribution`; quantiles and TVaR come from the shared risk engine, not reserving code.

## Build vs integrate policy

Build natively where the math is actuarial, the implementation is tractable, and Rust delivers speed or portability; integrate everywhere else.

| Capability | Decision | Rationale |
| --- | --- | --- |
| Distributions, LEV, stop-loss, limited/excess | Build | Core of everything; small, well-specified |
| FFT aggregation, Panjer, Monte Carlo | Build | Simulation-heavy; clear Rust win |
| Reinsurance contract algebra, towers | Build | Domain logic applied to millions of events |
| Copulas, risk measures, distortions, allocation | Build | Actuarial core; no coherent existing library |
| Triangle, deterministic and stochastic reserving | Build | Needed for the shared kernel and WASM target |
| GLM (IRLS, Tweedie, offsets, regularization) | Build | Tractable; foundation for GAM and GLM reserving |
| GAM | Build, scoped | P-splines, tensor smooths, GCV/REML for Tweedie/Poisson/Gamma; not mgcv parity |
| MCMC samplers (NUTS/HMC) | Integrate | nutpie (Rust NUTS) or BridgeStan; own model specs and diagnostics |
| Gradient boosting | Integrate | LightGBM/XGBoost via their Python/R bindings; adapters emit shared objects |
| Neural networks | Integrate | PyTorch in Python behind shared protocols; revisit Burn once the core is stable |
| Dataframes | Integrate | Arrow as interchange; Polars optional at the edges, never the numerical core |

**Test for any new build decision:** does a mature, well-maintained engine already exist, and would owning it change what users can do? If yes and no, integrate.

## Core abstractions

Four designs are load-bearing and get written up and reviewed in Phase 0, before domain code: distribution representations, the joint predictive distribution, the Triangle, and RNG streams.

### Distribution representations

One trait family, three concrete representations, each closed under different operations:

| Representation | Examples | Exact operations | Converts to |
| --- | --- | --- | --- |
| Parametric | Lognormal, Pareto, GPD, mixtures | CDF, quantile, moments, LEV, limit/excess, scaling | Discretized (grid), Sampled |
| Discretized | FFT grids, Panjer output | Convolution (sum), layer, stop-loss, quantile on grid | Sampled |
| Sampled | Bootstrap draws, posterior draws, simulation output | Any operation, approximately; empirical quantile/TVaR | Discretized (histogram) |

Operations that are not exact for a representation must convert explicitly, never silently. Discretization carries its grid step and truncation point as metadata so error is auditable.

### PredictiveDistribution (joint by default)

The object every model returns and every risk measure consumes.

- Stores **joint** draws or a joint structure across its components (origin years, LOBs, layers). The 99.5% of a total reserve is not the sum of per-origin 99.5%s.
- Exposes `mean`, `variance`, `cdf`, `quantile`, `sample`, `var`, `tvar`, plus `marginal(key)` and `aggregate(keys)`.
- Carries provenance: model, parameters, seed, stream ids, package version.

```text
Bootstrap Chain Ladder ─┐
Bayesian reserve ───────┤
Freq-sev aggregate ─────┼──► PredictiveDistribution ──► risk measures, towers, capital
Claim-level model ──────┘
```

### Triangle

Designed as its own note before Phase 2, matching the semantics users know from chainladder-python:

- Four axes: index × column × origin × development.
- Grain and valuation date as first-class metadata; partial periods and irregular shapes supported.
- Transformations: incremental ↔ cumulative, accident ↔ calendar view, grain changes, long ↔ wide.
- Arrow in, Arrow out; internal storage is a dense Rust array with a mask, not a dataframe.

### RNG streams

Counter-based or stream-indexed generators (e.g. Philox, or ChaCha with stream ids), one stream per simulation index. Results are bit-identical regardless of thread count, and any single simulation can be replayed from `(seed, stream id)`.

## Language boundary

The fast path is a closed set of Rust-native objects; user-defined Python/R callbacks are a supported but explicitly slow path.

### Python (primary)

- PyO3 + maturin; wheels for Linux, macOS and Windows from day one.
- Zero-copy in and out via Arrow and NumPy buffers; Polars/pandas accepted at the edge.
- Release the GIL inside every Rust call; parallel work never calls back into Python.
- Custom distributions or losses defined in Python run single-threaded and are flagged as such in diagnostics.
- Objects pickle (for multiprocessing and caching) and follow sklearn estimator conventions where they fit (`fit`, `predict`, `get_params`).

### R (skeleton first, parity later)

- extendr, with vendored crates and offline builds to meet CRAN policy; pin a minimum rustc.
- Phase 0 proves the same kernel serves both languages; feature parity waits until the Python API is stable (targeted around v0.5).
- R wrappers stay idiomatic (functions + S3 methods) and contain no algorithms.

### WASM

- The core crates avoid OS-dependent dependencies so reserving, distributions and aggregate build for `wasm32`.
- Target use: client-side calculation in browser tools and Office.js custom functions, removing the need for a hosted service for standard methods.
- Rayon-parallel paths fall back to single-threaded under WASM.

## Workspace and crate plan

Start with five crates plus bindings; split a crate only when compile time, dependency weight, or independent release cadence forces it.

```text
actuarial-rs/
├── Cargo.toml
├── crates/
│   ├── act-core/          errors, traits, arrays, masks, periods, RNG streams
│   ├── act-math/          linear algebra, optimization, integration, FFT,
│   │                      special functions, root finding
│   ├── act-prob/          distributions (3 representations), PredictiveDistribution,
│   │                      dependence, risk measures, distortions, transforms
│   ├── act-reserving/     Triangle, deterministic + stochastic methods, CDR
│   ├── act-aggregate/     freq-sev, FFT/Panjer/MC, reinsurance contracts, towers
│   ├── act-python/        PyO3 bindings
│   └── act-r/             extendr bindings
├── python/
├── R/
└── validation/            reference datasets + parity suites
```

**Expected later splits** (not created until needed): `act-glm` (with GAM), `act-capital`, `act-claims`, `act-survival`, `act-credibility`, `act-evt`, `act-stochastic`. Delegated engines (samplers, GBDT, neural) never become required Rust dependencies.

**Feature flags** keep heavy paths optional: `default = ["reserving", "aggregate"]`, with `glm`, `capital`, `claims`, `bayes-bridge`, `polars`, `wasm` opt-in.

**Prefix:** internal crates use `act-*` until the public name is chosen; see Open decisions.

## Domain scope

The v1 target scope stands; domains are listed here in build order, with v2 additions marked.

| Domain | Scope | Added in v2 |
| --- | --- | --- |
| Reserving, classical | Chain Ladder, Mack, BF, Cape Cod, Benktander, ELR, Clark, Munich CL, paid-incurred, tails | — |
| Reserving, stochastic | ODP bootstrap, Mack uncertainty, GLM reserving, Bayesian reserving, simulation | One-year view: CDR, Merz–Wüthrich |
| Aggregate | Freq-sev; FFT, Panjer, Monte Carlo, importance sampling; full distribution, layers, stop-loss | Discretization error reported with results |
| Reinsurance | QS, surplus, per-risk/per-occurrence XOL, agg XOL, stop loss, corridors, AAD/AAL, reinstatements; layers, towers, inuring, gross/ceded/net | Contract terms as data, so towers serialize and replay |
| Risk measures and transforms | VaR, TVaR, CoVaR, CoTVaR, MES, spectral, entropic, distortion (Wang, PH, dual power); Esscher, exponential tilt | — |
| Dependence | Gaussian, t, Clayton, Gumbel, Frank, Joe; later vines, empirical, factor; Iman-Conover, comonotonic | — |
| Capital | Portfolio aggregation, diversification, allocation (Euler, TVaR, covariance, Shapley, Merton-Perold, Bodoff) | Reserve-risk input from one-year view |
| Claim-level reserving | Event-based histories; reporting delay, payment, severity, closure, reopen, ultimate | Aggregates to joint PredictiveDistribution |
| Credibility | Limited fluctuation, Bühlmann, Bühlmann-Straub, empirical Bayes, hierarchical | — |
| Pricing | Rate indication, trend, on-level, ILF, deductibles, MBBEFD and exposure curves, predictive models | — |
| Extreme value | GEV, GPD, POT, Hill, threshold diagnostics | — |
| Stochastic processes | Poisson family, renewal, marked point, Hawkes, Brownian, jump diffusion; martingale reserve diagnostic | Diagnostic tied to CDR literature |

**Martingale diagnostic:** a well-calibrated reserve estimate M_t = E[U | F_t] should satisfy E[M_{t+1} | F_t] = M_t. This is the same property the claims development result tests, so it is built alongside the one-year view rather than as a separate idea.

## Modeling engines

Every engine, native or delegated, implements one model protocol so comparison is uniform:

```python
model.fit(X, y, exposure=..., offset=..., weight=...)
model.predict(X)
model.predict_distribution(X)   # -> PredictiveDistribution
model.score(X, y)
model.diagnostics()
```

| Engine | Where it lives | Scope |
| --- | --- | --- |
| GLM | Rust | Gaussian, Poisson, Gamma, Tweedie, NB, Binomial, Inverse Gaussian; identity/log/logit/probit/inverse/power links; IRLS, weights, offsets, exposure, elastic net, robust covariance |
| GAM | Rust (on GLM) | B/P-splines, tensor, cyclic, monotone; penalized IRLS; GCV and REML smoothing selection; Tweedie/Poisson/Gamma families first |
| Survival | Rust | Kaplan-Meier, Cox, parametric, competing risks, multi-state |
| Bayesian | Specs + diagnostics in Rust; sampling via nutpie or BridgeStan | Hierarchical severity, Bayesian CL, compartmental reserving, credibility; R-hat, ESS, divergences, PPC, WAIC/LOO |
| Gradient boosting | Python/R adapters over LightGBM, XGBoost | Poisson/Gamma/Tweedie/quantile objectives, monotone constraints, exposure via offsets |
| Neural | Python (PyTorch) behind the protocol | Tabular MLP with embeddings, multi-task claim models, sequence models for claim trajectories |

Delegated engines are thin: they convert inputs, call the engine, and wrap outputs in shared objects. They add no evaluation logic of their own.

## Evaluation and validation

No release ships unless its methods reproduce reference implementations on standard datasets within stated tolerances.

### Shared evaluation subsystem

| Family | Metrics |
| --- | --- |
| Regression | RMSE, MAE, deviance, log-likelihood |
| Pricing | Gini, Lorenz, lift, double lift, calibration, actual vs expected |
| Probabilistic | Log score, CRPS, PIT, coverage, quantile calibration |
| Survival | Concordance, Brier score, calibration |
| Bayesian | WAIC, LOO, posterior predictive checks |
| Reserving | Back-testing on held-out diagonals, CDR distribution checks, martingale diagnostic |
| Capital | VaR backtests, ES calibration, tail exceedance tests |

`compare(models, validation)` produces one table across GLM, GAM, GBDT, neural and Bayesian models.

### Parity suite (release gate)

| Area | Reference implementations | Datasets |
| --- | --- | --- |
| Reserving | chainladder-python, ChainLadder (R) | RAA, GenIns, ABC, CAS Loss Reserve Database |
| Distributions | SciPy, actuar | Analytic moments and quantiles |
| Aggregate | aggregate (Python), actuar | Published examples; analytic compound Poisson cases |
| GLM/GAM | statsmodels, glum, mgcv | freMTPL2 and similar open pricing data |
| Bayesian | Stan / PyMC outputs of the same spec | Meyers monograph models |

Tolerances are recorded per test; bootstrap and simulation tests compare distributions (moments, quantiles, KS) under fixed seeds rather than exact values.

## Parallelism and reproducibility

Parallelize the outermost loop only, and make every result reproducible from a seed regardless of thread count.

- **Rayon on the outer loop:** bootstrap replicates, Monte Carlo paths, MCMC chains, CV folds, hyperparameter search, portfolio scenarios.
- **Inner math single-threaded:** no Polars or BLAS thread pools inside a Rayon task. Configure faer and any BLAS to one thread within parallel regions.
- **Deterministic streams:** simulation i always draws from stream i (see RNG streams), so 1 thread and 64 threads give identical output.
- **Provenance on every result:** seed, stream scheme, package version and input hash travel with the PredictiveDistribution for audit and model-governance documentation.

## User-facing API surface

Users see seven namespaces regardless of how many internal crates exist; each object has exactly one home.

| Namespace | Contents |
| --- | --- |
| `distributions` | Parametric, discretized, sampled; `PredictiveDistribution` |
| `models` | GLM, GAM, survival, Bayesian specs, GBDT and neural adapters, `compare` |
| `reserving` | Triangle, deterministic and stochastic methods, CDR |
| `aggregate` | Freq-sev models, FFT/Panjer/MC |
| `reinsurance` | Contracts, layers, towers, gross/ceded/net |
| `risk` | Risk measures, distortions, transforms, copulas and dependence |
| `capital` | Portfolio models, diversification, allocation |

Pricing and credibility join as `pricing` and `credibility` when those phases land. Copulas live in `risk` only (v1 listed a separate `dependence` namespace); Wang lives in `risk` only.

```python
import actuarialrs as ar

tri = ar.reserving.Triangle.from_arrow(df)
boot = ar.reserving.ODPBootstrap(n_sims=10_000, seed=42).fit(tri)

reserve = boot.predict_distribution()      # joint across origins
reserve.aggregate().quantile(0.995)

losses = ar.aggregate.FreqSev(
    frequency=ar.distributions.Poisson(3.0),
    severity=ar.distributions.Pareto(alpha=1.8, scale=1e6),
).simulate(n_sims=100_000, seed=42)        # event-level losses

tower = ar.reinsurance.Tower([
    ar.reinsurance.XOL(5e6, excess=5e6),
    ar.reinsurance.XOL(15e6, excess=10e6),
])
result = tower.apply(losses)               # gross / ceded / net
ar.risk.TVaR(0.99)(result.net)
```

```r
library(actuarialrs)

tri  <- triangle(df)
boot <- odp_bootstrap(tri, n_sims = 10000, seed = 42)
quantile(aggregate(boot), 0.995)
```

## Roadmap and releases

The first release is v0.1 reserving core with a joint bootstrap distribution; aggregate, reinsurance and capital follow before GLM, GAM and pricing.

Reserving and risk ship before pricing, with four gates on the way to v1.0.

| Stage | Scope |
| --- | --- |
| **Phase 0 — Foundation** | Workspace, core traits, four design notes, RNG streams, PyO3 + extendr skeletons, parity harness |
| ◆ *Gate* | One Rust distribution object called from both Python and R |
| **v0.1 — Reserving core (first release)** | Distributions, Triangle, Chain Ladder, Mack, ODP bootstrap returning a joint PredictiveDistribution |
| ◆ *Gate* | Parity with chainladder-python and ChainLadder (R) on RAA, GenIns, ABC |
| **v0.2 — Reserving breadth** | BF, Cape Cod, Benktander, ELR, Clark, tails, one-year view (CDR, Merz–Wüthrich) |
| **v0.3 — Aggregate and reinsurance** | Freq-sev, FFT, Panjer, Monte Carlo; XOL, stop loss, reinstatements, towers, gross/ceded/net |
| **v0.4 — Risk and capital** | VaR, TVaR, CoTVaR, distortions, copulas, Iman-Conover, capital allocation, EVT tails |
| ◆ *Gate* | Bootstrap reserve and tower results feed capital allocation end to end |
| **v0.5 — GLM** | IRLS, Tweedie, NB, regularization, GLM reserving; R feature parity |
| **v0.6 — GAM and pricing** | P-splines, tensor smooths, GCV/REML; rate indication, ILF, MBBEFD, credibility |
| **v0.7 — Bayesian** | Model specs and diagnostics; nutpie / BridgeStan sampling; Bayesian reserving and credibility |
| **v0.8 — Claim-level reserving** | Event histories, payment and closure hazards, severity, ultimate; aggregate to a joint distribution |
| **v0.9 — Integrations** | LightGBM / XGBoost and PyTorch adapters on the shared protocol; WASM build |
| ◆ *Gate* | API review and deprecation pass |
| **v1.0 — Stable core APIs** | Semver guarantees for distributions, reserving, aggregate, reinsurance, risk, capital |
| **v1.x — Advanced** | Hawkes, renewal processes, vines, QMC, importance sampling, spectral measures, ILS structures |

Each release is a vertical slice exposed in Python the same day it lands in Rust. A gate is passed only when its criterion is demonstrated in the parity suite or an end-to-end example, not by inspection.

**Why this order:** v0.1 proves the language bridge, RNG streams, parallelism and the joint PredictiveDistribution at once, and is visibly faster than existing tools. v0.3–v0.4 cover the simulation-heavy work where Rust wins most and where casualty ILS portfolio work needs support first.

## Failure modes to avoid

- **Building everything at once.** The architecture anticipates the full scope; implementation stays one vertical slice at a time.
- **Domain code with its own numerics.** Reserving never implements its own quantiles; everything routes through shared risk math.
- **Marginal-only distributions.** Any object that will be aggregated keeps joint draws or joint structure.
- **Rebuilding mature engines.** No home-grown NUTS, boosting library or deep-learning framework.
- **Polars as the numerical core.** Arrow/Polars at the boundary; dense Rust arrays inside.
- **Required heavy dependencies.** Chain Ladder never compiles ML infrastructure.
- **Nested parallelism.** One Rayon pool, outer loop only.
- **Separate Python and R implementations.** Rust owns every algorithm.
- **Over-splitting crates early.** Five crates until the pain of one crate is real.
- **Namespace drift.** Every object has exactly one user-facing home.

## Open decisions

| Decision | Options | Needed by |
| --- | --- | --- |
| Relationship to chainladder-python | (a) standalone rewrite with Triangle parity; (b) Rust kernel that chainladder-python can call as an optional backend; (c) both, starting with (b) | Before Phase 2 design note |
| Public name | Defer branding; reserve the chosen name on crates.io, PyPI and CRAN as soon as it is picked | Before v0.1 publish |
| Working prefix | `actuarial-rs` repo, `act-*` crates, `actuarialrs` Python/R package | Now |
| License | MIT/Apache-2.0 dual (Rust convention); confirm compatibility with CRAN distribution | Before first public commit |
| IP ownership | Confirm with employer that open-source work in this domain is personal IP | Before first public commit |
| Bayesian backend | nutpie (Rust-native NUTS) vs BridgeStan (Stan models) vs both | Before v0.7 |
| Neural backend | PyTorch-only vs adding Burn for WASM/embedded inference | Before v0.9 |
| WASM scope | Which crates guarantee `wasm32` builds; single-threaded fallback policy | Before v0.3 |

**Next step:** write the distribution-representation and PredictiveDistribution design note; the Triangle and RNG notes depend on it.
