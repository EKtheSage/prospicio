# Design note: Bayesian reserving (Bayesian chain ladder, Meyers' models)

Status: **Partly decided; PR 1 in progress** (proposed 2026-10-11; on 2026-10-12 the user decided open decisions 1, 2 and 5 as recommended and 7 for the CAS data model repository, and took the recommendation on open decisions 3 and 4 for now; decisions 1 to 6 and 10 below are settled. PR 1's Rust core is built on `claude/bayes-odp`: the `bayes` feature, `BayesOdp` and the Stan reference on GenIns and RAA; its bindings (decision 13) are still to come. The rest stay Proposed) · Lane: Reserving, with the Models lane for `prospicio-bayes` · Depends on: `triangle.md`, `reserving-v02.md` (exposure, tails, the bootstraps, dependence between segments), `models.md` (`prospicio-bayes`, NUTS, diagnostics), `predictive-distribution.md`, `docs/roadmap.md` (v0.7 row; `prospicio-bayes` Next, item 1)

## Goal

A Bayesian chain ladder in `prospicio-reserving` that gives the same thing
the bootstraps give, a joint `PredictiveDistribution` of reserves by
origin with parameter and process error, but from a posterior sampled with
NUTS instead of resampled residuals. It sits beside `OdpBootstrap` and
`MackBootstrap`: same inputs (a `Triangle` and a column), same output type,
so totals, quantiles, segments, dependence and capital allocation work on
it unchanged. Two things it adds that the bootstraps cannot:

* **priors**: an actuary's view of the development pattern or of the
  expected loss ratio enters as a prior (a Bayesian Bornhuetter–Ferguson),
  instead of as a separate deterministic method;
* **models the bootstraps do not cover**: Meyers' changing settlement rate
  (CSR) and correlated accident year (CAY) models. Meyers (2019, CAS
  Monograph 8, sec. 5 to 7) tests them on 200 Schedule P triangles: his
  baseline, the correlated chain ladder on cumulative values (CRC), is not
  better calibrated than Mack or the ODP bootstrap (paid CRC predicts too
  high, as the ODP does; incurred CRC still understates variability). CSR
  fixes the paid failure and CAY the incurred one. CRC is the common base
  of both, not a calibration gain in itself.

Out of scope here, kept for later notes: compartmental reserving (Gesmann
and Morris 2020), hierarchical growth curves across companies (Guszcza
2008; Zhang, Dukic and Guszcza 2012), joint models across lines with
shared parameters or copulas (Zhang and Dukic 2013; Meyers 2019, ch. 8),
and claim-level reserving (v0.8, parked).

## Pull requests

Each PR holds one model or one capability and lands in Rust, Python and R
together, as in `reserving-v02.md` decision 7.

| PR | Branch | Holds | Base |
|---|---|---|---|
| 0 | `bayes-reserving-design` | this note | `main` |
| 1 | `claude/bayes-odp` | the `bayes` feature of `prospicio-reserving` (decision 2); `BayesOdp` (decisions 3 to 6); Stan reference on GenIns and RAA; bindings | PR 0 |
| 2 | `claude/bayes-segments` | `BayesOdp::fit_segments` with `SegmentDependence` and per-segment seeds (decision 11); `TriangleModel` for `BayesOdp` in the diagonal backtest (decision 8) | PR 1 |
| 3 | `claude/bayes-meyers` | `Meyers` with `Crc` and `Csr` on cumulative values with premium (decision 7); Meyers' group 353 data; reference from Meyers' own Stan code | PR 1 |
| 4 | `claude/bayes-bf` | informative priors on the origin parameters: the Bayesian Bornhuetter–Ferguson (decision 9) | PR 1 |
| 5 | later | `Cay` and the paid-incurred `Ipi`; then compartmental reserving in its own note | PR 3 |

PRs 2 to 4 touch the same binding files
(`crates/prospicio-python/src/reserving.rs`,
`crates/prospicio-r/src/reserving.rs`, `R/prospicio/R/reserving.R`,
`python/prospicio/reserving.py`) and the generated stub, `NAMESPACE` and
`man/`. Whichever merges second merges `main` and regenerates them; it
does not merge them by hand.

## References

| Model | Primary reference | Secondary |
|---|---|---|
| Bayesian ODP chain ladder | the same model in Stan, run with CmdStan 2.39.0 (decision 10) | England and Verrall (2002, BAJ 8(3)), (2006, AAS 1(2)); our `OdpBootstrap` and chain ladder on the same triangle |
| Bayesian Bornhuetter–Ferguson | the same model in Stan | Verrall (2004, NAAJ); England, Verrall and Wüthrich (2012, AAS 6(2), 258-283) |
| CRC, CSR (later CAY, IPI) | Meyers' R/Stan code, `meyers-appendix.zip` (CAS), run with CmdStan 2.39.0 | Meyers (2019), CAS Monograph 8, 2nd ed.: group 353 Commercial Auto, data in Tables 2.1-2.3, parameter summaries in Tables 5.1 and 7.1, totals in Tables 5.2 (CRC paid), 5.3 (CRC incurred) and 7.2 (CSR paid), elpd_loo in Tables 6.4 and 7.3 |
| Diagnostics | `prospicio_bayes` (`rhat`, `ess_bulk`, `ess_tail`), already at parity with R `posterior` | |
| Model comparison | `prospicio_bayes::elpd::loo`, already at parity with R `loo` 2.6.0 | Meyers' elpd_loo for CSR (49.76) and CRC (47.80), Tables 6.4 and 7.3; only between models of the same response (decision 5) |

R `ChainLadder` and chainladder-python have no MCMC reserving model, so
the reference is a Stan program of the same model, not a package. R
`ChainLadder::PaidIncurredChain` (Merz and Wüthrich 2010) is a conjugate
Bayesian paid-incurred model with a closed form; it is a candidate exact
reference if a paid-incurred model is taken up (not checked).

Data:

* `validation/data/genins.csv` and `validation/data/raa.csv`, as in v0.1;
* new for PR 3: `validation/data/clrd_comauto_353.csv`, Meyers' illustrative
  insurer, group 353 Commercial Auto, 1988-1997 at 12 to 120 months,
  columns `paid`, `incurred` and `premium` (net earned premium), the full
  10×10 square so that the outcome can be checked. Source (decided
  2026-10-12 by the user): the CAS loss reserve database as rebuilt by
  [EKtheSage/cas-schedule-p-data-model](https://github.com/EKtheSage/cas-schedule-p-data-model),
  table `mart_reserving_model_training` from release `20260613_041006`
  (public; the generator pins the publish id and checks the Parquet's
  sha256 against the release's `manifest.json`). `company_code = '353'`,
  `line_of_business = 'commercial_auto'`, accident years 1988-1997, ages
  1-10 (100 cells); `paid` is `cum_paid_loss`, `incurred` is case incurred
  `incurred_loss - bulk_loss`, `premium` is `earned_prem_net`, all in USD
  thousands. The generator asserts the outcomes Meyers prints before
  writing (age-10 sums: paid 40,000, incurred 40,061; both checked
  2026-10-12); a cell-by-cell check against Tables 2.1-2.3 is still to do.
  The data is the NAIC Schedule P extract the CAS publishes for research
  (Meyers and Shi); the CSV header and the knowledge entry credit it. The
  repo's existing `clrd_*.csv` are industry sums and do not hold it.

## Decisions

### 1. Which model ships first

Decided 2026-10-12 (the user, open decision 1: option (a)). Four families were compared:

| Candidate | Fits the roadmap ("Bayesian chain ladder") | Reference parity | Sampler difficulty |
|---|---|---|---|
| England–Verrall Bayesian ODP: incremental ODP, log-linear origin and development effects | exactly: the chain ladder as a Bayesian model; vague priors give back the ODP bootstrap's answer | a Stan program of the same likelihood; our chain ladder and `OdpBootstrap` as sanity checks; GenIns and RAA already in the repo | easy: the log likelihood is concave in the parameters, the gradient is the GLM's, the posterior is close to normal |
| Meyers CRC and CSR: lognormal on cumulative values, premium, a decreasing σ by age, a settlement-rate trend in CSR | a chain ladder with an expected loss ratio; CSR goes past the chain ladder | Meyers' own Stan code and printed results for group 353 | moderate: about 30 parameters, σ ordering and the γ trend; Stan samples it with a diagonal metric, as ours does |
| Guszcza and ZDG hierarchical growth curves | not a chain ladder; a growth-curve loss-ratio model | no open reference code for ZDG; Gesmann's Stan reproduction of Guszcza (2008) | harder: nonlinear, hierarchical, funnel geometry |
| Gesmann–Morris compartmental | a different model family | brms output in the paper; no fixed seed or code to rerun locally without brms | hardest: `k_er` and `k_p` swap roles and are singular when equal; ODE variants need numerical gradients |

**Ship the Bayesian ODP first, then CRC and CSR.** The ODP is the model the
roadmap names, the cheapest to get right with a hand-written gradient, and
it can be checked three ways (a Stan run, the chain ladder, the ODP
bootstrap). It also builds everything the later models reuse: the feature
flag, the density and its finite-difference test, the predictive step,
diagnostics in the fit, the Stan reference pipeline and the bindings.
CRC and CSR come second. CRC alone gives no calibration gain over the
bootstraps (Meyers 2019, sec. 5); it is the base that CSR and CAY extend,
and CSR is what improves calibration on paid data. On incurred data the
model that improves calibration is CAY, which PR 5 holds, so until then
the plan offers no better-calibrated incurred model than Mack. Moving CAY
into PR 3 is open decision 1 (d). Meyers' code gives a parity target for
all of them. Growth curves and compartmental
models wait: neither has a reproducible reference, and the compartmental
model's identification problems need informative priors that a library
default should not choose.

### 2. Where the code lives

Decided 2026-10-12 (the user, open decision 2: option (a)). `prospicio-reserving` is in the WASM scope (AGENTS.md "Checks",
`docs/architecture.md`, `ci.yml` line 38, default features), and heavy
engines "never become required Rust dependencies"
(`docs/architecture.md`). `prospicio-bayes` pulls in `nuts-rs`, Rayon and
rustfft, and whether `nuts-rs` builds for `wasm32` is not verified. So
`prospicio-reserving` cannot take a required dependency on it. Options:

| Option | Ownership | WASM | Access to reserving internals | Cost |
|---|---|---|---|---|
| A. A `reserving` module in `prospicio-bayes`, depending on `prospicio-reserving` | Models lane's crate; Reserving would edit it | clean: `prospicio-bayes` is outside the scope | public API only (`NextYear`, `Resample`, `SharedPool`, `SegmentDependence::reorder` and `rank_correlate` are `pub(crate)`); PRs 1 to 4 need none of them | a lane crossing on every PR |
| B. An optional feature `bayes` and a module `bayes` in `prospicio-reserving` | Reserving lane's own crate | the default-feature check is unchanged | full: one crate | a feature to keep compiled in CI |
| C. A new crate `prospicio-reserving-bayes` | Reserving lane, new crate | clean | public API only, as A | one more crate to version, publish and re-export |

**Option B**, for lane ownership and release simplicity: the work stays in
the Reserving lane's crate, so no PR crosses into the Models lane's, and it
adds no crate to version, publish and re-export. Access to internals is not
the reason: PRs 1 to 4 use only public items (`SegmentDependence`,
`TriangleModel`, `PredictiveDistribution::reorder_groups`). Internal access
would matter only for a later one-year view (decision 12), or to reuse
`SegmentDependence::reorder` and `rank_correlate` (both `pub(crate)`)
instead of reimplementing them, which B allows and A and C do not without
widening the public API.

```toml
# crates/prospicio-reserving/Cargo.toml
[dependencies]
prospicio-bayes = { workspace = true, optional = true }
rayon = { workspace = true, optional = true }   # added in PR 1, for loo()

[features]
bayes = ["dep:prospicio-bayes", "dep:rayon"]

# crates/prospicio/Cargo.toml (the umbrella)
bayes = ["dep:prospicio-bayes", "prospicio-reserving/bayes"]
```

There is no dependency cycle: `prospicio-bayes` depends on core, glm,
math, models and prob, not on reserving. The Python and R binding crates
already depend on `prospicio-bayes` and on `prospicio-reserving` (without
the feature). PR 1 changes their `Cargo.toml` to turn on
`prospicio-reserving/bayes`, so wheels and the R package gain no new
native dependency. The umbrella's `bayes` feature exists today as
`["dep:prospicio-bayes"]`; PR 1 adds `"prospicio-reserving/bayes"` to it.

CI: the WASM check selects packages with `-p`, so it builds
`prospicio-reserving` with default features and stays as it is. The
workspace-wide `cargo clippy --all-targets` and `cargo test` unify features
over every member, so once the binding crates and `prospicio-validation`
turn on `prospicio-reserving/bayes`, the feature is compiled, linted and
tested with no new job (an inference about Cargo's feature unification, to
be confirmed in PR 1 by a deliberate lint failure inside the module). PR 1
also adds `cargo check -p prospicio-reserving --no-default-features` and
`--features bayes` to the local check list in AGENTS.md if the inference
fails.

Built in PR 1 (2026-10-12). The inference holds: with an unused variable
put in `bayes/odp.rs` on purpose, the workspace-wide `cargo clippy
--all-targets -- -D warnings` failed on it, because `prospicio-validation`
turns the feature on; so AGENTS.md "Checks" needs no new line. `cargo
check --target wasm32-unknown-unknown -p prospicio-reserving` still builds
without `prospicio-bayes` (`cargo tree` shows neither it nor nuts-rs). The
feature also turns on an optional `rayon` (already in the build through
nuts-rs), which runs `loo()`'s pointwise densities in parallel.

Lane ownership: the model code is the Reserving lane's, in
`crates/prospicio-reserving/src/bayes/`, `validation/tests/reserving_bayes*.rs`
and this note. PR 1 also touches shared files, named in its description:
`crates/prospicio/Cargo.toml` (the umbrella `bayes` feature, a release
file under AGENTS.md "Releases"), `crates/prospicio-python/Cargo.toml` and
`crates/prospicio-r/Cargo.toml` (the feature on `prospicio-reserving`),
`crates/prospicio-reserving/Cargo.toml`, and the validation crate's
`Cargo.toml` if its tests need the feature turned on there.
`prospicio-bayes` stays the Models lane's. If PR 1 needs a
change there (for example a public way to split one seed between the
chains and the predictive step, or a dense mass matrix, decision 5), it
goes in a small PR of its own with the user's go-ahead, and merges first.

### 3. The Bayesian ODP model

Decided 2026-10-12 (with open decision 1); built in PR 1 as written here,
except that the row and column check returns `Error::Bayes` (decision 5).
Incremental values `X_wd` for origin `w = 1..n` and development
`d = 1..m`, observed on the upper triangle:

```text
η_wd = c + a_w + b_d,   a_1 = 0,  b_1 = 0           (corner constraints, as the ODP GLM)
μ_wd = exp(η_wd)
log p(X | θ) = Σ_observed (X_wd η_wd − μ_wd) / φ     (scaled Poisson quasi likelihood)
```

`φ` is the chain ladder's Pearson scale, the same `scale` that
`OdpBootstrap` reports (degrees of freedom `N − p`), plugged in and not
sampled, as England and Verrall do (decision 4 and open decision 3).

Priors, all normal and independent:

| Parameter | Prior | Default |
|---|---|---|
| `c` | `N(c0, intercept_sd²)`, `c0` the log of the mean fitted increment of the chain ladder | `intercept_sd = 10` |
| `a_w`, `w ≥ 2` | `N(0, origin_sd²)` | `origin_sd = 10` |
| `b_d`, `d ≥ 2` | `N(0, development_sd²)` | `development_sd = 10` |

The prior on `c` is centred because the log of an increment is about 13
on GenIns: an `N(0, 10²)` prior would sit 1.3 sd away and pull. Centring
on the chain ladder's scale is the only data-dependent choice, documented
as such, and with sd 10 on the log scale it is vague. The names follow
`BayesGlm` (`intercept_sd`, `prior_sd`) where the meaning is the same.

Why this likelihood: the quasi log likelihood `(x η − e^η)/φ` has second
derivative `−e^η/φ < 0` for any `x`, so with normal priors the log
posterior is strictly concave: one mode, no funnels, and NUTS samples it
well. Negative increments are allowed cell by cell; the posterior is proper
whenever the priors are (they always are). The mode without priors (the
MLE) exists only when every development column's and every origin's
observed increments sum to a positive value (necessary; Kuang, Nielsen and
Nielsen 2009 give the exact conditions for the log-linear model). The
origin condition is not implied by the column one: the newest origin has
one cell, so a negative or zero `X_n1` sends `a_n` to `−∞`. The chain
ladder has no such condition (it accepts factors below 1 and negative
column sums, common in incurred data), so a triangle the chain ladder fits
can have no log-linear mean. With a non-positive row or column sum, the
posterior of that `a_w` or `b_d` is driven by its `N(0, 10²)` prior alone
and drifts far negative, where `OdpBootstrap` gives a signed fit.
`BayesOdp` therefore checks both sums before sampling and returns
`Error::Bayes` naming the origin or age (decision 5); tests use a
triangle with a negative development column and one with a negative
newest-origin cell. Where the MLE exists it is the chain ladder (the ODP
GLM's MLE is the chain ladder), which is the first test.

Gradient, by hand (there is no autodiff in `prospicio-bayes`):
`∂/∂η_wd = (X_wd − μ_wd)/φ`, summed into `c`, `a_w` and `b_d`, plus the
priors' `−(θ − m)/s²`. A finite-difference test in the pattern of
`glm.rs:599` (`the_gradient_matches_a_finite_difference`) checks it.

**Preconditioning.** The mass matrix of `prospicio-bayes` is diagonal only,
and corner constraints make `c`, `a_w` and `b_d` strongly correlated. The
density therefore samples `z` with `θ = θ̂ + L z`, where `θ̂` is the
posterior mode (Newton on the concave log posterior, started at the chain
ladder's GLM fit) and `L L' = H⁻¹`, the inverse of the negative Hessian
there. The map is linear, so the log Jacobian is a constant, the gradient
is `L' ∇_θ`, and the posterior in `z` is close to standard normal. Draws
are mapped back with `PosteriorDraws::transform`, which keeps the chains
so R-hat and ESS are reported on `c`, `a_w` and `b_d`. Chains start at
`z = 0` with the driver's ±0.01 jitter.

As built: `θ = (c, a_2..a_n, b_2..b_m)`, with `p = n + m − 1` entries (19
on GenIns and RAA). Newton starts from the logs of the chain ladder's
fitted increments (`U_w γ_d`, the ODP GLM's MLE) when all are positive, and
from `(c0, 0, …)` otherwise; it solves `H δ = ∇f` by Cholesky, halves the
step until `f` rises, and stops when `max |δ| < 1e-12` (or below `1e-8`
when a full step no longer raises `f`, the rounding floor), failing after
100 iterations. `H = R Rᵀ` with `R` lower triangular and `L = (Rᵀ)⁻¹`.
Sanity tests, priors switched off by an internal flag: the mode's `μ` is
the chain ladder's fitted increment to 1e-8 relative on RAA and GenIns,
from `(c0, 0, …)`; the prior's shift of the mode falls as `1/sd²` (ratio of
the shifts at sd 100 and 1,000 within 1% of 100 on GenIns); the gradient
in `θ` and in `z` matches central differences.

RAA's last development effect rests on one small cell (an increment of
172 against `φ = 984`, shape `x/φ ≈ 0.17`), so its posterior is strongly
skewed and the curvature varies: at the default target acceptance 0.8 our
sampler gives 3 divergent transitions in 4,000 draws (a warning) and Stan
15 in 100,000 with a dense metric. The summaries still agree (decision
10); the Stan reference runs at `adapt_delta = 0.99`, which removes them.

### 4. The predictive distribution

Decided 2026-10-12; built in PR 1 as written. The process step is the
bootstrap's own, moved into `bootstrap::process_draw` (Gamma with shape
`μ/φ` and scale `φ`, Stan's `gamma_rng(μ/φ, 1/φ)`), so the three models
draw it alike. Provenance also records `c0` and the `seed`. Reserves come from `PosteriorDraws::predictive`, which picks a
posterior draw uniformly for each simulation and calls the model to add
process error, giving a `PredictiveDistribution` with the same layout as
`OdpBootstrapFit::reserves`: dimension `origin`, one component per origin,
the reserve of each origin as the sum of its future increments.

* Future cells: `w + d > n + 1` up to the oldest observed age `m`. No tail
  (decision 12).
* Process error: the bootstrap's `ProcessDistribution`, reused as is:
  `Gamma` (mean `μ_wd`, variance `φ μ_wd`, the default, as
  `OdpBootstrap` and R's `process.distr = "gamma"`) or `None` (parameter
  error only, R's `ParamDist`). Sharing the enum means one setting reads
  the same in all three ODP-family models.
* `n_sims` (default 10,000) is independent of the number of posterior
  draws (default 4×1,000).
* Provenance records the method (`bayes_odp`), the priors, `φ`, the
  sampler settings, `posterior_draws` and `posterior_seed` (the latter two
  added by `predictive` already).

### 5. Settings, the fit and diagnostics

Decided 2026-10-12 (failed diagnostics: open decision 4 (a)); built in PR 1
with these changes from the sketch below, which is kept as proposed:

* `fit_segments` and `BayesOdpFits` wait for PR 2. `BayesOdpFit` also has
  `summary()`, `loo()`, `to_long()`, `totals()` and `development_table()`
  itself (one segment), and keeps the observed cells privately for `loo()`.
* **Seeds.** The chains and the simulations each use a seed derived from
  `seed`: `StreamRng::new(seed, u64::MAX − tag).next_u64()`, tag 0 for the
  chains and 1 for the simulations. `prospicio-bayes` keys chain `c` from
  stream `c` of its seed and `PredictiveDistribution::simulate` gives
  simulation `i` stream `i` of its seed, so with two derived seeds no
  simulation shares a stream with a chain, and nothing in
  `prospicio-bayes` had to change. PR 2 gives segment `k` the tags `2 + 2k`
  and `3 + 2k`. `Sampler::seed` is ignored (documented on the field), so one
  number replays a fit; a test checks that changing it changes nothing.
  Replay is exact whatever the thread count (a test runs one fit in a
  one-thread pool).
* **Errors.** A non-finite or non-positive prior sd and `n_sims == 0` are
  `InvalidSetting`. A non-positive row or column sum is a new
  `Error::Bayes(String)`, not `InvalidSetting`, whose fields (`name:
  &'static str`, `value: f64`) cannot name the origin; the message reads
  "origin 2023: the observed increments sum to -130, so no log-linear mean
  exists". `Error::Bayes` takes a `String` (not `&'static str`) for that
  reason; PR 2 reuses it for `Synchronized`. The gap, too-few-cells and
  unknown-column errors are the chain ladder's and the bootstrap's
  preparation's own (`Error::Bootstrap`, which reads "bootstrap: …").
  Sampler errors arrive as `Error::Core`.
* **Diagnostics** come from the summary of the natural-scale draws: a
  warning per parameter with R-hat above 1.01 (or NaN) or bulk or tail ESS
  below `100 × chains`, naming the parameter, its value and the threshold,
  and one for any divergence. Defaults on GenIns: max R-hat 1.002, min bulk
  ESS 5,563, min tail ESS 2,379, no divergence; on RAA: max R-hat 1.003,
  min bulk ESS 2,306, 3 divergences (decision 3).
* **`loo()`** uses `Family::Poisson.log_density(x, μ, φ, 1)` with
  `r_eff` per cell from `ess_mean`, as `BayesGlmFit::loo` does. That density
  is minus infinity at a negative increment, so `loo()` returns
  `Error::Bayes` on a triangle with one (RAA). Each term integrates the
  continuous Poisson's normalising constant: 4,000 draws × 55 cells take
  5.1 s serially in a release build, 0.56 s with the draws in parallel
  (Rayon); in a debug build 36 s serially and about 4 s in parallel. On
  GenIns: elpd −749.9 (se 6.0),
  p_loo 16.1, largest Pareto k 0.90 (the corner cells that alone fix a
  parameter are influential).

```rust
#[cfg(feature = "bayes")]
pub mod bayes {
    pub struct BayesOdp {
        pub intercept_sd: f64,          // 10
        pub origin_sd: f64,             // 10
        pub development_sd: f64,        // 10
        pub process: ProcessDistribution, // Gamma
        pub sampler: prospicio_bayes::Sampler, // 4 chains, 1000 tune, 1000 draws, target 0.8, depth 10
        pub n_sims: usize,              // 10_000
        pub seed: u64,                  // the chains and the simulations
        pub dependence: SegmentDependence, // decision 11; fit() ignores it
    }

    impl BayesOdp {
        pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<BayesOdpFit>;
        pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<BayesOdpFits>;
    }

    pub struct BayesOdpFit {
        pub chain_ladder: ChainLadderFit,      // the plug-in fit: latest values, factors, table columns
        pub scale: f64,                        // φ
        pub posterior: PosteriorDraws,         // c, a[2..n], b[2..m] by name, on the natural scale
        pub diagnostics: Diagnostics,
        pub reserves: PredictiveDistribution,  // dimension origin, as OdpBootstrapFit
    }

    pub struct Diagnostics {
        pub max_rhat: f64,
        pub min_ess_bulk: f64,
        pub min_ess_tail: f64,
        pub divergences: usize,
        pub warnings: Vec<String>,  // empty when every threshold holds
    }
}
```

* `BayesOdpFit` implements `ReserveFit` (it has a `ChainLadderFit`), so
  `to_long`, `totals` and `development_table` work as for the bootstraps.
* One `seed`, as the bootstraps have one: the chains use streams of it as
  `Sampler::seed` does, and the predictive step uses a seed derived from
  it so that no simulation shares a stream with a chain. How the derived
  seed is made is fixed in PR 1 after reading how `prospicio-bayes`
  numbers its streams; if a clean split needs a change to
  `prospicio-bayes`, that is a Models-lane PR (decision 2). Draws replay
  exactly for a seed, whatever the thread count, since each chain keys its
  own generator.
* Thresholds for `warnings`, from Vehtari et al. (2021): R-hat above 1.01,
  bulk or tail ESS below 100 per chain, any divergence. `fit` returns `Ok`
  with the warnings; it fails only when the sampler returns no usable
  draws. The bindings raise them as a Python `UserWarning` and an R
  `warning()`. The alternative, an error on a failed diagnostic, is open
  decision 4.
* `summary()` returns the posterior summary of `prospicio_bayes`
  (mean, sd, quantiles, R-hat, bulk and tail ESS) per parameter, and
  `loo()` the PSIS-LOO of the pointwise log likelihood. The pointwise term
  is a normalised density, not the quasi likelihood of decision 3 (which
  is not normalised in `x`, so its elpd is not a log predictive density):
  `prospicio_models::Family::log_density` (`family.rs:344`) at `μ_wd` and
  `φ`, the density `backtest.rs` already scores the ODP with. elpd is
  compared only between models of the same response, as Meyers compares
  CRC with CSR (Tables 6.4 and 7.3): a Bayesian ODP (incremental values)
  with another incremental model, CRC with CSR (cumulative values). Across
  responses, for example `BayesOdp` against CRC, the comparison is the
  diagonal backtest's held-out scoring (decision 8), not elpd.
* Errors: the chain ladder's own (for example an unknown column); the
  existing `Error::InvalidSetting { name, value, expected }` for a
  non-positive or non-finite prior standard deviation and for a triangle
  whose observed increments sum to a non-positive value in some origin or
  development column (decision 3); sampler errors from `prospicio-bayes`,
  which are already `prospicio_core::Error`, through the existing
  `Error::Core`, so the structured error is kept; and a new
  `Error::Bayes(&'static str)`, in the style of `Error::Bootstrap`, only
  for a setting the model does not support (for example `Synchronized`,
  decision 11).

### 6. Dispersion

Decided 2026-10-12 (open decision 3 (a), the recommendation, for now);
built in PR 1: `BayesOdpFit::scale` equals `OdpBootstrapFit::scale`
exactly (a unit test), and the Stan reference's own computation agrees with
R's `BootChainLadder` on RAA (983.635027030873) to 1e-9. `φ` is plugged in. Under a quasi likelihood, `φ` has no proper
likelihood of its own (the scaled Poisson is not normalised in `φ`), so a
prior on it does not give a valid posterior; England and Verrall plug it
in for the same reason, and it keeps the Bayesian ODP comparable with the
ODP bootstrap, which uses the same `φ`. A model with sampled dispersion
needs a proper likelihood (a gamma or a Tweedie on the increments) and is
a separate model, not a setting. Open decision 3.

### 7. Meyers' CRC and CSR (PR 3)

Proposed. Cumulative values `C_wd` (paid or incurred) and premium `P_w`
from a measure column, read as exposure is read in `reserving-v02.md`
decision 1 (the latest cumulative value per origin; `InvalidExposure`
for a missing or non-positive one).

```text
CRC:  μ_wd = log P_w + logelr + a_w + β_d,      a_1 = 0,  β_m = 0
CSR:  μ_wd = log P_w + logelr + a_w + β_d (1 − γ)^(w−1)
      σ_d² = Σ_{i=d}^{m} s_i,   s_i ∈ (0, 1)   (σ decreases with age)
      C_wd ~ lognormal(μ_wd, σ_d)
```

Priors as Meyers' Stan code has them: `logelr ~ N(−0.4, ·)`,
`a_w ~ N(0, ·)`, `β_d ~ N(0, ·)`, `γ ~ N(0, 0.05)`, `s_i ~ U(0, 1)`. The
monograph prints the scales as "10"; whether that is a variance or a
standard deviation, and the exact transform Meyers uses to keep `s_i` away
from 0 (an inverse gamma mapped through a CDF), are read from the
appendix code in PR 3, and the prior scales are copied exactly. For
`s_i`, Meyers' transform and a logit with its log Jacobian both give a
`U(0, 1)` prior and so the same posterior; we implement the logit (the
simpler unconstrained map for our sampler), and parity is on posterior
summaries, which is all parity can be with a different sampler (decision
10).

Output: `ultimates` (simulated `C_wm` per origin: cells are conditionally
independent given the parameters, so only the last age is drawn) and
`reserves` (ultimate less the latest value of the fitted column: an
outstanding reserve on paid data, IBNR on incurred data), both
`PredictiveDistribution` with dimension `origin`.

```rust
pub enum MeyersModel { Crc, Csr }   // Cay and Ipi later (PR 5)
pub struct Meyers { pub model: MeyersModel, /* priors, sampler, n_sims, seed, dependence */ }
impl Meyers { pub fn fit(&self, triangle: &Triangle, column: &str, premium: &str) -> Result<MeyersFit>; }
```

Meyers' published targets for group 353 Commercial Auto, 10,000 draws:
CRC paid total 40,121 (SE 2,487, Table 5.2); CRC incurred 39,147 (SE
1,642, Table 5.3); CSR paid 37,597 (SE 2,401, Table 7.2) with `γ` 0.0446
(sd 0.0282, Table 7.1); the outcomes, paid 40,000 and incurred 40,061;
elpd_loo CSR 49.76, CRC 47.80 (Tables 6.4 and 7.3). These are a secondary
check (decision 10).

Notes carried into PR 3: no development past age `m`, since `β_m = 0`
fixes the last age as ultimate; `a_1 = 0` and `β_m = 0` are needed for
identification; the σ ordering is a modelling constraint (variability
falls as claims settle), not an identification one; CAY's recursion (`ρ` times the
previous origin's residual) makes the likelihood depend on observed lagged
cells and is why CAY waits for PR 5.

### 8. The diagonal backtest

Proposed (PR 2). `BayesOdp` implements `TriangleModel`
(`backtest.rs:40`), whose doc already names "a Bayesian model": `forecast`
fits on the train rows and returns `n_sims` joint posterior predictive
draws of the test rows, with `log_density` from the normalised density of
decision 5. `fit_rows` keeps every train row, negative increments
included, since the quasi likelihood accepts them (the Poisson GLM's drops
them), subject to decision 3's row and column check. This gives
Meyers-style calibration on held-out diagonals beside the bootstraps and
the GLM.

`Meyers` does not implement `TriangleModel` in PR 2 or PR 3. The trait's
rows carry incremental values (`frame.rs:31`), while CRC and CSR model
cumulative values with premium. An adapter would have to rebuild
cumulatives from the frame's increments, read premium from its `exposure`
column, simulate each origin's whole cumulative path jointly (differences
of independently drawn cumulative cells would inflate the variance and
often be negative) and leave `log_density` as `None` (the density of a
difference of lognormals has no closed form). That adapter, or a backtest
on cumulative cells for these models, is a later PR.

### 9. Informative priors: the Bayesian Bornhuetter–Ferguson

Proposed (PR 4). The BF form of England, Verrall and Wüthrich (2012) puts
the prior on each origin's ultimate. In the log-linear form of decision 3
that is a prior on `c + a_w` centred on `log(apriori × exposure ×
share of the ultimate that is incremental)`, with a user-set coefficient
of variation per origin. A tight prior reproduces BF; a vague one the
Bayesian ODP. The exact mapping, and whether to parameterise by ultimate
(`x_w y_d` with `Σ y_d = 1`) instead, is settled in PR 4 against a Stan
program of the paper's model. Exposure is a measure column, as in
`reserving-v02.md` decision 1.

### 10. Validation

Decided 2026-10-12 (the user, open decision 5: option (a), CmdStan through cmdstanpy under `uv`). There is no package to match, so the reference is the same
model written in Stan and sampled long enough that the reference's own
Monte Carlo error is negligible.

* **Reference programs**: `validation/scripts/stan/bayes_odp.stan`
  (decision 3's model with `target += (x .* eta - exp(eta)) / phi`, which
  accepts negative and non-integer increments where `poisson_log` would
  not), and in PR 3 Meyers' `crc.stan` and `csr.stan` copied unchanged
  from the CAS appendix (licence and attribution checked when copied).
* **Generator**: `validation/scripts/reserving_bayes_stan.py`, run with
  `uv run --no-project --with cmdstanpy==<pinned> --with pandas` against
  CmdStan 2.39.0 (built locally at `~/.cmdstan/cmdstan-2.39.0`). Fixed
  seed, 4 chains, 2,000 warm-up and 25,000 draws each, so a mean's MCSE is
  under 1% of a posterior sd. The script prints the CmdStan version, the
  `.stan` file's SHA-256 and the seed into the `source` column, and fails
  if any reference R-hat exceeds 1.005 or any divergence occurs. `φ` and
  `c0` are computed by our chain ladder and passed in as data, so both
  sides use the same plug-ins.
* **Reference CSV**: `validation/reference/reserving_bayes_stan.csv` in
  the usual `dataset,method,quantity,arg,expected,abs_tol,rel_tol,source`
  format. Rows: posterior mean and sd of every parameter; mean, sd and the
  5th, 50th and 95th percentiles of each origin's reserve and of the total
  (process `Gamma` is drawn in Stan's generated quantities the same way).
  CI reads the committed CSV and never runs Stan.
* **Tolerances**, per quantity with posterior sd `s` (Stan's), following
  `BayesGlm`'s grid parity (`models.md`: means within 0.1 sd, sds within
  8%): means within `0.1 s`; sds within 8%; reserve percentiles within
  `0.15 s` (tail percentiles carry more Monte Carlo error). Our side runs
  the defaults (4×1,000 draws, 10,000 simulations), whose MCSE is about
  `0.02 s` on the means. Different samplers never give identical draws,
  so parity is on summaries, never draws.
* **Sanity checks, not parity** (unit tests in PR 1): with the priors
  switched off (Newton on the likelihood alone) the mode equals the chain
  ladder's GLM fit to 1e-8. With priors of sd 1,000 it cannot: a prior
  shifts a parameter by about `(θ − m)/(sd² I)` for Fisher information
  `I`, about 1e-6 on GenIns's last development parameter (one cell,
  information about 1.3), so that test checks instead that the difference
  shrinks as `1/sd²` between sd 100 and 1,000; the
  posterior mean reserve on GenIns is within 2% of the chain ladder's and
  of `OdpBootstrap`'s mean, and its sd within 10% of the bootstrap's (the
  England and Verrall result that the two agree under vague priors); exact
  replay for a seed; the gradient against finite differences.
* **Meyers' printed numbers** (PR 3): our CRC and CSR totals within 1% and
  SEs within 5% of the published ones, `γ`'s mean within 0.1 sd. Looser
  than the Stan parity because the monograph's runs used other seeds and an
  older Stan; the CmdStan run of Meyers' own code is the tight check.
* **Runtime**: a GenIns fit is expected to take well under a second; PR 1
  measures it and keeps the CI tests under a few seconds in total.

Done for PR 1 (2026-10-12). `validation/tests/reserving_bayes.rs` checks
190 rows of `reserving_bayes_stan.csv` (19 parameters, mean and sd; 10
origins and the total, mean, sd and three percentiles; `φ` and `c0`), on
GenIns and RAA, with the defaults (seed 0). As built:

* The reference runs with `metric = "dense_e"` (same posterior, better
  efficiency) and `adapt_delta = 0.99` (RAA's skewed `b[10]` gave 15
  divergences in 100,000 at 0.8, decision 3); seed 20261012; reference max
  R-hat 1.0001 on GenIns and 1.0002 on RAA, no divergence. CmdStan's `make`
  is RTools 4.5's, which cmdstanpy is told through `MAKE`; the model is
  compiled under `target/stan/`. About 60 s with a compiled model.
* Every row passes; the largest share of its tolerance any row uses is
  0.70 (RAA's 1982 reserve sd, 676 against 640 with 8% allowed). By kind,
  the largest shares: parameter means 0.23 (GenIns) and 0.34 (RAA), sds
  0.44 and 0.62, reserve means 0.22 and 0.27, reserve sds 0.37 and 0.70,
  95th percentiles 0.49 and 0.66.
* GenIns totals: Stan mean 18,851,576 and sd 2,984,047; ours 18,834,037 and
  2,937,848; the chain ladder 18,680,856; `OdpBootstrap` (20,000
  simulations, seed 1) 18,851,366 and 2,998,662. So the posterior mean is
  0.8% above the chain ladder and 0.1% below the bootstrap, and the sd 2.0%
  below the bootstrap's (the sanity bounds are 2% and 10%). The bootstrap's
  residual scaling `sqrt(n/(n−p))` did not open a large gap.
* Runtime: a default GenIns fit (4×1,000 draws after 4×1,000 warm-up,
  10,000 simulations) takes about 80 ms in a release build and 1.1 s in a
  debug one; the whole test file about 7 s in a debug build (one test
  thread), most of it `loo()`. The parity closure computes parameter means
  and sds from the draws: calling `summary()` (R-hat and ESS too) for each
  of the 76 cases took about a minute in a debug build.

### 11. Segments and dependence

Proposed (PR 2). `fit_segments` fits each segment's own posterior (the
same model, independently), and the output has the shape of
`OdpBootstrapFits`: dimensions the key names plus `origin`, with
`to_long`, `totals` and `development_table`. `dependence` is the existing
`SegmentDependence`:

* `Independent` (default): each segment's simulations as drawn, with a
  distinct seed per segment. `PosteriorDraws::predictive` gives simulation
  `i` the stream `StreamRng::new(seed, i)` (through
  `PredictiveDistribution::simulate`), and each segment calls it on its
  own, so the same `seed` in every segment would give every segment the
  same uniforms in simulation `i` (the same posterior-draw pick and the
  same Gamma uniforms), a strong spurious positive dependence. Segment `k`
  (in key order) therefore uses a seed derived from `(seed, k)` for both
  its chains and its predictive step, by the same derivation as decision
  5's split, fixed in PR 1. The bootstraps avoid the problem differently,
  by drawing all segments of simulation `i` from one stream in index order
  (`dependence.rs`), which is not available when each segment samples its
  own posterior. A test checks that the rank correlation between segment
  totals under `Independent` is about 0 (within `3/√n_sims`);
* `RankCorrelation { spearman }`: whole simulations of each segment
  reordered by Iman–Conover on the segment totals through
  `reorder_groups`, exactly as for the bootstraps, since it needs only a
  `PredictiveDistribution` per segment;
* `Synchronized`: an error (`Error::Bayes`, decision 5), since it shares
  residual positions and a posterior has none.

Dependence learned from the data through shared parameters (a joint
posterior across lines, Zhang and Dukic 2013, Meyers 2019 ch. 8) is a
later model, not a setting.

### 12. Tails and the one-year view

Proposed. **No tail in PRs 1 to 4**: the ODP model has no development past
the oldest observed age, and Meyers fixes `β_m = 0`, so the settings have
no `tail` field yet. A
later option is a constant tail with a lognormal prior on `log(tail − 1)`,
which gives the tail's parameter error the way `tail_std_err` does in
`reserving-v02.md` decision 9.

**No one-year view in PRs 1 to 4.** The claims development result of a
Bayesian model needs, per simulated next diagonal, the posterior given
that diagonal: refitting by MCMC per simulation (10,000 NUTS runs) is too
slow for a default. Two routes for later, open decision 6: reweighting the
existing posterior draws by the next diagonal's likelihood with Pareto
smoothed importance sampling (the `loo` machinery already exists, and
`k̂` says when it fails), or a Laplace approximation of the updated
posterior for the ODP model, whose posterior is close to normal.

### 13. Bindings

Proposed. Every model lands in Python and R in its own PR, in the style of
`OdpBootstrap` and `odp_bootstrap()`:

* Python: `prospicio.reserving.BayesOdp(intercept_sd=10, origin_sd=10,
  development_sd=10, process="gamma", chains=4, tune=1000, draws=1000,
  n_sims=10000, seed=0, dependence=None)` with `fit(triangle, column)` and
  `fit_segments`, returning `BayesOdpFit` with `reserves`,
  `chain_ladder`, `scale`, `summary()` (a frame), `diagnostics`, `loo()`
  and the same frames as the bootstrap fits.
* R: `bayes_odp(triangle, column = NULL, intercept_sd = 10, origin_sd = 10,
  development_sd = 10, process = c("gamma", "none"), chains = 4, tune =
  1000, draws = 1000, n_sims = 10000, seed = 0, dependence = NULL)`
  returning a `bayes_odp_fit` with the same fields and `as.data.frame()`,
  `totals_frame()`, `development_frame()`, `segment()`, `summary()`.
* The sampler settings are flattened into arguments as `BayesGlm`'s are;
  the posterior summary reuses whatever `BayesGlmFit` exposes in each
  language (checked in PR 1) rather than a second wrapper of
  `PosteriorDraws`.
* Docs regenerated with `cargo xtask docs`. No custom-density binding,
  as `models.md` decided.

### 14. Docs and knowledge

Proposed. Each PR updates, in itself: the Reserving row of AGENTS.md, the
v0.7 row of `docs/roadmap.md` and the `prospicio-bayes` Next list (item 1
is this work), `docs/architecture.md` if a roadmap row changes, and this
note's status. `knowledge/` gains: the Stan reference pipeline and the
local CmdStan facts (environment), Meyers' group 353 data and its
agreement with Tables 2.1-2.3 (datasets), and whether the Bayesian ODP and
the ODP bootstrap agree on GenIns and by how much (findings), with
`verified` once a CI test confirms each.

## Open decisions for the user

Each has a recommendation. Decided 2026-10-12 by the user: 1 (a), 2 (a),
5 (a) and 7 (the CAS data model repository); 3 (a) and 4 (a), the
recommendations, for now, so that PR 1 could be built. The others wait on the user
and will be revisited; nothing past PR 1 is built until they
are settled.

1. **First model.** (a) Bayesian ODP, then CRC and CSR; (b) CRC and CSR
   first; (c) both in one PR; (d) as (a), with CAY moved from PR 5 into
   PR 3. *Recommended: (a)*, decision 1: the roadmap names the chain
   ladder, it is the easiest to sample with a hand-written gradient, and it
   checks against both a Stan run and our bootstrap. CRC itself is not
   better calibrated than the bootstraps; CSR is the paid improvement, and
   the incurred one (CAY) arrives only in PR 5 under (a). Choose (d) if a
   better-calibrated incurred model matters sooner.
2. **Where the code lives.** (a) A `bayes` feature and module in
   `prospicio-reserving`; (b) a `reserving` module in `prospicio-bayes`;
   (c) a new `prospicio-reserving-bayes` crate. *Recommended: (a)*,
   decision 2: the Reserving lane's own crate, WASM check unchanged, and
   no crate added to the release. PRs 1 to 4 need no internals; (a) only
   helps a later one-year view or reuse of the `pub(crate)` dependence
   helpers.
3. **Dispersion.** (a) Plug in the chain ladder's Pearson `φ`; (b) sample
   `φ` with a prior under the quasi likelihood; (c) a separate proper
   model (gamma or Tweedie increments) with sampled dispersion, later.
   *Recommended: (a) now, (c) if wanted*, decision 6; (b) is not a valid
   posterior. Decided 2026-10-12: (a) for now.
4. **Failed diagnostics.** (a) Return the fit with warnings (and raise
   them in Python and R); (b) return an error. *Recommended: (a)*, so a
   user can inspect a poor fit; the warnings name the parameter and the
   threshold. Decided 2026-10-12: (a) for now.
5. **Reference draws.** (a) The same model in Stan through CmdStan 2.39.0
   and cmdstanpy under `uv`, summaries committed; (b) R rstan or brms
   (not installed locally); (c) published numbers only. *Recommended:
   (a)*, decision 10, with Meyers' printed numbers as a looser second
   check in PR 3.
6. **One-year view.** (a) Not now; (b) importance reweighting of the
   posterior by the next diagonal (PSIS, with `k̂` as the guard); (c) a
   Laplace update for the ODP model only; (d) MCMC refit per simulation.
   *Recommended: (a) now, then (b)* in its own PR after PR 3, decision 12.
7. **Meyers' data source.** Decided 2026-10-12 by the user:
   `EKtheSage/cas-schedule-p-data-model` (see Data above), which holds the
   full 10×10 squares that chainladder-python's `clrd` (upper triangles
   only) lacks. Its company screen for the 200-triangle panel
   (`cas_schedule_p.screens.select_companies`) is rebuilt from Meyers'
   appendix criteria, not his own list, so a claim about the panel needs a
   check against his appendix first; the group 353 checks do not need it.
8. **Bayesian Bornhuetter–Ferguson.** (a) In scope as PR 4; (b) later.
   *Recommended: (a)*: it is the main thing priors buy an actuary, and it
   reuses PR 1's model.
9. **Compartmental reserving.** (a) A separate design note after PR 3;
   (b) part of this plan. *Recommended: (a)*: its identification needs
   informative priors and its ODE variants need numerical gradients,
   which deserve their own decisions.
10. **Bindings with each model.** (a) Rust, Python and R in the same PR,
    as `reserving-v02.md` decision 7; (b) Rust first, bindings after.
    *Recommended: (a)*.
