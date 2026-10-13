# Design note: Bayesian reserving (Bayesian chain ladder, Meyers' models)

Status: **Partly decided; PR 1 merged (#188); PR 3 built; PR 4 built** (proposed 2026-10-11; on 2026-10-12 the user decided open decisions 1, 2 and 5 as recommended and 7 for the CAS data model repository, and took the recommendation on open decisions 3 and 4 for now; decisions 1 to 6 and 10 below are settled. PR 1 (`claude/bayes-odp`, merged as #188) built the `bayes` feature, `BayesOdp`, the Stan reference on GenIns and RAA, and the Python and R bindings (decision 13, for one segment). PR 3 is built on `claude/bayes-meyers` (decision 7): `Meyers` with CRC and CSR in Rust, Meyers' group 353 data, his transcribed Stan code as the reference, and the Python and R bindings (decision 13, for one segment). PR 4 is built on `claude/bayes-bf` (decision 9, at the user's request): `BayesBf`, EVW's triangle, the Stan reference, and the Python and R bindings (decision 13, for one segment). The rest stay Proposed) · Lane: Reserving, with the Models lane for `prospicio-bayes` · Depends on: `triangle.md`, `reserving-v02.md` (exposure, tails, the bootstraps, dependence between segments), `models.md` (`prospicio-bayes`, NUTS, diagnostics), `predictive-distribution.md`, `docs/roadmap.md` (v0.7 row; `prospicio-bayes` Next, item 1)

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
| Bayesian Bornhuetter–Ferguson | the same model in Stan | England, Verrall and Wüthrich (2012, AAS 6(2), 258-283): Table 1 data, totals in Tables 2 and 3 (decision 9); Verrall (2004, NAAJ) |
| CRC, CSR (later CAY, IPI) | Meyers' R/Stan code, `meyers-appendix.zip` (CAS), transcribed into current Stan syntax (no licence is stated, and the old array syntax no longer compiles) and run with CmdStan 2.39.0 | Meyers (2019), CAS Monograph 8, 2nd ed.: group 353 Commercial Auto, data in Tables 2.1-2.3, parameter summaries in Tables 5.1 and 7.1, totals in Tables 5.2 (CRC paid), 5.3 (CRC incurred) and 7.2 (CSR paid), elpd_loo in Tables 6.4 and 7.3 |
| Diagnostics | `prospicio_bayes` (`rhat`, `ess_bulk`, `ess_tail`), already at parity with R `posterior` | |
| Model comparison | `prospicio_bayes::elpd::loo`, already at parity with R `loo` 2.6.0 | Meyers' elpd_loo for CSR (49.76) and CRC (47.80), Tables 6.4 and 7.3; only between models of the same response (decision 5) |

R `ChainLadder` and chainladder-python have no MCMC reserving model, so
the reference is a Stan program of the same model, not a package. R
`ChainLadder::PaidIncurredChain` (Merz and Wüthrich 2010) is a conjugate
Bayesian paid-incurred model with a closed form; it is a candidate exact
reference if a paid-incurred model is taken up (not checked).

Data:

* `validation/data/genins.csv` and `validation/data/raa.csv`, as in v0.1;
* new for PR 4: `validation/data/evw2012.csv`, England, Verrall and
  Wüthrich's (2012) Table 1 (Wüthrich and Merz 2008, Tables 2.2-2.4): 10×10
  incremental paid, cumulated, with their prior ultimates `m_i` as a
  `prior` column, origins relabelled 2001 to 2010; and
  `validation/data/genins_premium.csv` (already in the repo for Clark);
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
  2026-10-12) and every cell against Tables 2.1-2.3 as printed: all agree
  but one misprint, Table 2.2's 2004 for 1995 at lag 6 (the data holds
  4116; the row's "Source" year slipped into the cell). Built in PR 3 as
  `validation/scripts/clrd_comauto_353.py` (DuckDB 1.5.6 under `uv`,
  `gh release download`).
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
skewed and the curvature varies: at NUTS's usual target acceptance 0.8
our sampler gave divergent transitions in 4,000 draws at 7 of seeds 0 to 7
(0 to 6 each, a warning), and Stan 15 in 100,000 with a dense metric.
`BayesOdp`'s default target acceptance is therefore 0.95: seeds 0 to 29
then give no divergence on RAA or GenIns, the summaries are unchanged
within Monte Carlo error, and a fit costs little more (about 0.1 s on
GenIns in a release build). The Stan reference runs at
`adapt_delta = 0.99` for the same reason.

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
  and one for any divergence. The fields `max_rhat`, `min_ess_bulk` and
  `min_ess_tail` are NaN when any parameter's value is (a constant draw),
  so they never look healthier than the warnings. Defaults (seed 0) on
  GenIns: max R-hat 1.003, min bulk ESS 4,550, min tail ESS 2,368, no
  divergence; on RAA: max R-hat 1.003, min bulk ESS 2,125, min tail ESS
  1,625, no divergence (decision 3).
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
        pub sampler: prospicio_bayes::Sampler, // 4 chains, 1000 tune, 1000 draws, target 0.95, depth 10
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

Built in PR 3 (2026-10-12; "As built" below); the sketch is kept as
proposed. Cumulative values `C_wd` (paid or incurred) and premium `P_w`
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

**As built** (`crates/prospicio-reserving/src/bayes/meyers.rs`):

```rust
pub enum MeyersModel { Crc, Csr }              // default Crc
pub struct Meyers {
    pub model: MeyersModel,
    pub logelr_mean: f64,     // -0.4
    pub logelr_sd: f64,       // 3.162
    pub origin_sd: f64,       // 3.162
    pub development_sd: f64,  // 3.162
    pub gamma_sd: f64,        // 0.05, CSR only
    pub sampler: Sampler,     // 4 chains, 1,000 warm-up, 2,000 draws, target 0.95, depth 10
    pub n_sims: usize,        // 10,000
    pub seed: u64,
}
impl Meyers { pub fn fit(&self, triangle: &Triangle, column: &str, premium: &str) -> Result<MeyersFit>; }
pub struct MeyersFit {
    pub chain_ladder: ChainLadderFit,
    pub premium: Vec<f64>,
    pub posterior: PosteriorDraws,           // logelr, alpha[2..n], beta[1..m-1], gamma, sigma[1..m]
    pub diagnostics: Diagnostics,
    pub ultimates: PredictiveDistribution,   // dimension origin
    pub reserves: PredictiveDistribution,    // ultimates less the latest values, draw by draw
}
// summary(), loo(), to_long(), totals(), development_table(); impl ReserveFit
```

* **Priors, read from the code.** The monograph's "10" is a standard
  deviation of `sqrt(10)`, which the code rounds to 3.162 (the errata
  confirm `sqrt(10)`); the defaults copy 3.162. Meyers' `a_ig ~
  inv_gamma(1, 1)` is declared `<upper=100000>` and mapped by
  `gamma_cdf(1 / a_ig | 1, 1) = 1 - exp(-1 / a_ig)`, which makes each step
  `s_i` uniform on `(1 - exp(-1e-5), 1)`, not `(0, 1)`. We sample
  `s_i = s0 + (1 - s0) logistic(u_i)` with the same floor `s0`; without it
  the late σ on incurred data came out 10 to 20% below Stan's and the
  sampler diverged. His step labels run in reverse of the monograph's
  (`sig2[10]` takes `a_ig[10]`, `sig2[10 - i]` adds `a_ig[i]`); the steps
  are iid, so the σ posterior is the same, and parity compares `sigma[d]`.
* **Premium** is a measure column of the same triangle, read as `CapeCod`
  reads exposure (`expected_loss::origin_exposure`: the latest value per
  origin, `InvalidExposure` when missing or not positive).
* **Errors.** `InvalidSetting` for a prior sd that is not finite and
  positive, a non-finite `logelr_mean` and `n_sims == 0`; `Error::Bayes`
  for a cell that is not positive (Meyers' code raises values below 1 to
  1 silently; the message names the origin and age), for no more cells than
  parameters, and from the sampler's scale; `MultipleSegments`, gaps and
  unknown columns as for the other models.
* **Sampling scale (a change from the sketch).** `BayesOdp` whitens at the
  posterior mode. Here the joint mode is not a usable centre: on CRC paid
  Newton creeps toward `sigma[10]` near 5e-4 (`u_10` near -15) against a
  posterior mean of 0.020, along the funnel where the late σ and the late
  residuals shrink together, and did not converge in 200 iterations. So
  `Meyers` runs a pilot (the same chains, `tune` warm-up and `max(tune,
  100)` kept draws) on a scale whitened at a least-squares start (the
  negative Hessian there, by central differences of the analytic
  gradient, with a ridge where it is not positive definite), then samples
  on the scale of the pilot draws' mean and covariance, regularised as
  Stan's dense metric (`n/(n + 5) S + 1e-3 (5/(n + 5)) I`). The pilot
  takes stream `u64::MAX` of the chains' derived seed, so `seed` still
  replays a fit exactly whatever the threads. The mode finder and the
  whitening moved to `bayes/laplace.rs`, shared with `BayesOdp` (whose
  numbers are unchanged).
* **Defaults.** Target acceptance 0.95: at 0.8 the paid fits gave 1 to 56
  divergent transitions in 8,000 draws at nearly every one of 30 seeds.
  2,000 kept draws per chain, not 1,000: with 1,000 the late σ had an R-hat
  above 1.01 (at most 1.016) in 5 of 120 fits (4 cases x 30 seeds); with
  2,000 all 120 pass every threshold, with no divergence.
* **Prediction.** Given a draw, each origin's ultimate is
  `exp(log P_w + logelr + alpha_w + sigma_m z)` (`beta_m = 0`, so neither
  β nor γ enters), and an origin observed at age `m` keeps its value, as
  Meyers does; no `ceiling()` (his adds about +4.5 to the total). The
  reserves are the same simulations less the latest values. Cells are
  independent given the parameters, so no path is drawn; a path would
  matter only for cash flows. No tail and no one-year view (decision 12);
  no `dependence` field until segments (decision 11), so `Meyers` departs
  from `BayesOdp`'s struct there. `Meyers` does not implement
  `TriangleModel` (decision 8).
* **`loo()`** is Meyers' pointwise `normal_lpdf(log C | mu, sigma_d)`,
  comparable only between CRC and CSR fits of the same column, with
  `r_eff` per cell from `ess_mean`. Pareto k reaches about 1 on group 353
  (also in Stan), so elpd carries PSIS noise.
* **Results on group 353** (total ultimate mean and sd; ours at the
  defaults, seed 0; Stan of Meyers' code, 4 x 25,000 draws; Meyers'
  printed):

  | Case | Ours | Stan | Meyers |
  |---|---|---|---|
  | CRC paid | 40,119 (2,538) | 40,136 (2,497) | 40,121 (2,487), Table 5.2 |
  | CRC incurred | 39,080 (1,541) | 39,132 (1,616) | 39,147 (1,642), Table 5.3 |
  | CSR paid | 37,612 (2,487) | 37,537 (2,391) | 37,597 (2,401), Table 7.2 |
  | CSR incurred | 38,168 (1,588) | 38,165 (1,567) | not printed |

  elpd_loo ours 47.90, 71.42, 49.82, 71.12; Stan 48.04, 71.22, 49.56,
  71.26; Meyers 47.80 (CRC paid), 70.97 (CRC incurred), 49.76 (CSR paid).
  CSR paid `gamma` ours 0.0447 (sd 0.0286), Stan 0.0450 (0.0280), Meyers
  0.0446 (0.0282); `logelr` ours -0.3954, Stan -0.3961, Meyers -0.3956.
* **Runtime.** A default fit takes 0.3 s (CRC) to 0.5 s (CSR) in a
  release build and 6 to 8 s in a debug one (the pilot is about 40% of
  it); `validation/tests/reserving_meyers.rs` takes about 34 s in a debug
  build, most of it the four default fits.

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

Built in PR 4 (2026-10-12, in Rust, Python and R; "As built" below, with two corrections
to the proposal); the proposal is kept as written:

Proposed (PR 4). The BF form of England, Verrall and Wüthrich (2012) puts
the prior on each origin's ultimate. In the log-linear form of decision 3
that is a prior on `c + a_w` centred on `log(apriori × exposure ×
share of the ultimate that is incremental)`, with a user-set coefficient
of variation per origin. A tight prior reproduces BF; a vague one the
Bayesian ODP. The exact mapping, and whether to parameterise by ultimate
(`x_w y_d` with `Σ y_d = 1`) instead, is settled in PR 4 against a Stan
program of the paper's model. Exposure is a measure column, as in
`reserving-v02.md` decision 1.

**As built** (`crates/prospicio-reserving/src/bayes/bf.rs`):

```rust
pub struct BayesBf {
    pub apriori: f64,          // 1.0, as BornhuetterFerguson
    pub cv: f64,               // 0.1 (EVW's a = 100); f64::INFINITY is flat
    pub development_sd: f64,   // 10
    pub process: ProcessDistribution, // Gamma
    pub sampler: Sampler,      // 4 chains, 1,000 warm-up, 1,000 draws, target 0.95
    pub n_sims: usize,         // 10,000
    pub seed: u64,
}
impl BayesBf { pub fn fit(&self, triangle: &Triangle, column: &str, exposure: &str) -> Result<BayesBfFit>; }
pub struct BayesBfFit {
    pub chain_ladder: ChainLadderFit,
    pub scale: f64,                     // phi, as BayesOdp's
    pub exposure: Vec<f64>,
    pub prior_ultimate: Vec<f64>,       // apriori * exposure
    pub posterior: PosteriorDraws,      // alpha[1..n], beta[1..m]
    pub diagnostics: Diagnostics,
    pub reserves: PredictiveDistribution, // dimension origin
}
// summary(), loo(), to_long(), totals(), development_table(); impl ReserveFit
```

* **The model is EVW's Model 3.1 on the log scale (correction 1).** The
  sketch's prior on `c + a_w` cannot be a prior on the ultimate: under
  the corner constraints `exp(c + a_w)` is the first increment, and the
  ultimate is `exp(c + a_w) Σ exp(b_d)`, which also moves with the
  pattern. EVW instead write the means `mu_w gamma_d` with an
  unnormalised pattern and put the prior on `mu_w`. As built:
  `eta_wd = alpha_w + beta_d`, the quasi likelihood and plug-in `phi` of
  decision 3, `mu_w = exp(alpha_w) ~ Gamma(shape 1/cv², mean m_w)` with
  `m_w = apriori × exposure_w` (the exposure as `BornhuetterFerguson`
  reads it), and `beta_d ~ N(0, development_sd²)`, EVW's vague pattern
  prior made proper. No corner constraint: the gamma prior, or with
  `cv = ∞` the normal prior on `beta`, fixes the scale shared by `mu` and
  `gamma`. Every term is concave, so the posterior has one mode. One
  `cv` for all origins (EVW's `a_i ≡ a`); a vector is a later option.
* **The tight limit is not the classical BF (correction 2).** As `cv →
  0`, `mu_w → m_w` and the pattern's posterior is `gamma_d ~ Gamma(S_d /
  phi, M_d / phi)` (EVW sec. 3.3; `S_d` the observed increments of column
  `d`, `M_d` the prior ultimates of the origins observed there), so the
  mean reserve tends to `m_w Σ S_d / M_d` over the future cells: Mack's
  (2008) BF with the raw pattern, not `(1 − 1/cdf_w) m_w`, which takes the
  chain ladder's pattern. EVW say so (their sec. 1.1): priors on the rows
  reshape the pattern even under vague column priors. On their triangle
  the limit is 6,644,047 and the classical BF 7,356,578. The two agree
  when `m_w` are the chain ladder's ultimates (the chain ladder's column
  equations are `gamma_d Σ mu_w = S_d`); a test checks that case. A BF
  that keeps the chain ladder's pattern exactly (Verrall 2004) is a
  different model, not built.
* **Limits as tests** (unit tests on GenIns with premium): `cv = ∞` gives
  `BayesOdp`'s reserves by origin and in total within `0.1 s` and 10%;
  `cv = 0.001` (with `development_sd` 100 and no process error) gives the
  closed-form limit's mean and sd by origin and in total within `0.1 s`
  and 10%, and with `m_w` the chain ladder's ultimates the classical BF
  reserve, to 1e-9 in the closed form and within `0.1 s` in the fit;
  between them the total mean reserve moves monotonically with `cv`
  (infinite, 0.5, 0.2, 0.05, 0.001).
* **`development_sd`** defaults to 10, as `BayesOdp`'s. Against EVW's
  flat pattern prior it lifts a one-cell last column with `S / phi` near 1
  by about 6.5% (0.27% on EVW's total at `a = ∞`); at 100 by 3e-5. With
  `cv = ∞` the direction `(alpha + k, beta − k)` is held only by this
  prior, about `development_sd / √m` long: at 1,000 our sampler gave 195
  to 247 divergent transitions in 4,000 (target acceptance 0.95 or 0.99),
  at 100 or 10 none. The diagnostics report it.
* **Sampling scale.** As `BayesOdp`, NUTS on a scale whitened at the mode
  (decision 3), on `(u, beta)` with `u_w = alpha_w − log m_w`: centred on
  the prior, `u` keeps full precision under a tight prior, where `alpha`
  near 15 rounds steps to 2e-15. Newton starts at the strong-prior mode
  (`u = 0`, the raw pattern). The mode finder (`laplace.rs`) now takes in
  full a step whose promised gain is below the rounding of the log
  posterior and stops once such steps stop halving; before, a tight prior
  made it creep by about 2e-8 a step for 200 iterations. `BayesOdp`'s
  modes on GenIns and RAA are bitwise unchanged.
* **Errors.** `InvalidSetting` for `apriori` not finite and positive,
  `cv` not positive (NaN included; infinity is allowed), `development_sd`
  not finite and positive, `n_sims == 0`; `InvalidExposure` from the
  exposure column; `Error::Bayes` for a development column whose
  increments sum to zero or less (as `BayesOdp`) and for an origin whose
  increments sum to `−phi / cv²` or less, where the gamma posterior of its
  `mu` is improper (under `cv = ∞`, any non-positive sum, as `BayesOdp`):
  an informative prior keeps a young origin with a zero or slightly
  negative first cell, which `BayesOdp` refuses.
* **Prediction and LOO** as `BayesOdp` (decisions 4 and 5): each future
  cell's Gamma process error around `exp(alpha_w + beta_d)`, no tail; LOO
  on the normalised ODP density, so it compares with `BayesOdp`'s on the
  same triangle. No `dependence` field until segments (decision 11), as
  `Meyers`.
* **Runtime.** A default fit takes about 0.1 s in a release build and 1
  to 2 s in a debug one; `validation/tests/reserving_bayes_bf.rs` takes
  about 7.5 s in a debug build (five fits), the unit tests about 2.5 s.
* **Bindings** (decision 13): Python `BayesBf` and R `bayes_bf()`, for
  one segment, as `BayesOdp`'s.

### 10. Validation

Decided 2026-10-12 (the user, open decision 5: option (a), CmdStan through cmdstanpy under `uv`). There is no package to match, so the reference is the same
model written in Stan and sampled long enough that the reference's own
Monte Carlo error is negligible.

* **Reference programs**: `validation/scripts/stan/bayes_odp.stan`
  (decision 3's model with `target += (x .* eta - exp(eta)) / phi`, which
  accepts negative and non-integer increments where `poisson_log` would
  not), and in PR 3 `validation/scripts/stan/meyers_crc.stan` and
  `meyers_csr.stan`: Meyers' programs from the CAS appendix, transcribed
  with provenance (no licence is stated, so not copied; the changes are
  syntax only, `array[]` for the removed `real x[10]`, `gamma_cdf(y | 1,
  1)`, the size 10 as data `n`, and generated quantities for σ and the
  predictive ultimates without his `ceiling()`).
* **Generator**: `validation/scripts/reserving_bayes_stan.py`, run with
  `uv run --no-project --python 3.14 --with cmdstanpy==1.3.0 --with
  pandas==3.0.6 --with numpy==2.5.4` (versions recorded in the CSV
  header) against
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
* **Tolerances**, per quantity with posterior sd `s` (Stan's), sized from
  our own Monte Carlo error, since Stan's is negligible: each is at least 4
  times the seed-to-seed sd of that row over seeds 0 to 19 of the defaults
  (4×1,000 draws, 10,000 simulations). GenIns: means within `0.1 s`, sds
  within 10%, reserve percentiles within `0.3 s` (measured error at most
  `0.018 s`, 2.4% and `0.066 s`). RAA: `0.15 s`, 20% and `0.4 s` (at most
  `0.03 s`, 4.6% and `0.083 s`). RAA's errors are about twice GenIns'
  because its young origins' reserves ride on the skewed `b[9]` and
  `b[10]`: the 1982 reserve is one cell on `b[10]` (posterior sd 3.6). The
  first tolerances (`0.1 s`, 8%, `0.15 s`, from `BayesGlm`'s grid parity
  in `models.md`) assumed an sd error of about 1.2%, which holds for
  GenIns' parameters but not for RAA's reserves, and failed RAA at 10 of
  20 seeds. Different samplers never give identical draws, so parity is on
  summaries, never draws.
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
  As built, each tolerance is at least 4 times the combined Monte Carlo
  error of our defaults and of his single run: totals 1% (combined error
  0.13%) as planned; total SEs 50% (our seed-to-seed error is at most
  4.3%, but heavy-tailed through the 1997 reserve's log-normal on the
  one-cell `alpha[10]`: 35% off at one seed in 200); `logelr` and `γ` means within 0.15 printed sd (combined
  0.027 sd) and their sds within 15% (combined 3.4%); elpd_loo within 2
  (each run's about 0.36, PSIS with Pareto k near 1). At 5% our CRC
  incurred SE (1,541 against 1,642) failed though Stan's (1,616) is within
  1.6% of Meyers'.

Done for PR 3 (2026-10-12). `validation/tests/reserving_meyers.rs` checks
464 rows of `reserving_meyers_stan.csv` (every parameter's mean and sd;
each origin's and the total reserve's mean, sd and three percentiles,
except the youngest origin's sd; the total ultimate's mean and sd;
elpd_loo) for CRC and CSR on paid and
incurred, and Meyers' 15 printed numbers. Generator
`validation/scripts/reserving_meyers_stan.py` (cmdstanpy 1.3.0, pandas
3.0.6, numpy 2.5.4, arviz 0.23.4 for PSIS-LOO), seed 20261012, 4 chains of
2,000 warm-up and 25,000 draws, dense metric, `adapt_delta` 0.99,
`max_treedepth` 12, Meyers' initial values; max R-hat 1.0004, no
divergence; a rerun reproduced every value. Our Monte Carlo error at the
defaults over seeds 0 to 199, with `s` Stan's posterior sd: at most
`0.026 s` on a mean, 3.6% on a parameter sd, 2.6% on the reserve sd of
origins 1989 to 1996, 4.3% on the total reserve's and total ultimate's
sds, `0.063 s` on a percentile and 0.43 on elpd_loo. The 1997 reserve's
sd is heavy-tailed: it rides on the one-cell `alpha[10]` through a
log-normal, and its seed-to-seed sd (up to 7.5%) understates the tail. A
first sizing from 20 seeds put it at 4.8% and set 20% (50% on CRC paid);
a review's 60-seed sweep found two failures, and at 200 seeds it failed
at 12 (CRC incurred +63% at seed 170). So the parity has no row for the
youngest origin's reserve sd (its 5%, 50% and 95% percentiles check its
spread), and the total sds, which inherit a milder tail (CRC incurred
+37% at seed 170), get 50%. Tolerances: means `0.15 s`, parameter sds
15%, origin reserve sds 20%, total sds 50%, percentiles `0.35 s`,
elpd_loo 2.0, each at least 4.2 times its family's largest spread. Seeds
0 to 199 pass every row (Stan's and Meyers' printed), using at most 0.86
of a tolerance (CRC incurred elpd_loo against Meyers' at seed 122); seed
0 uses at most 0.41. The sweep is the ignored test `seed_sweep` (release
build, `MEYERS_SEEDS` seeds; 200 seeds take about 7 minutes).
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
* Every row passes at each of seeds 0 to 29 (a release-build sweep, not
  in CI), using at most 0.80 of any tolerance (RAA's 1982 reserve sd).
  Mean over the seeds, our summaries sit within 0.2 of a tolerance of
  Stan's, so the two samplers agree on the posterior.
* GenIns totals: Stan mean 18,851,576 and sd 2,984,047; ours (seed 0)
  18,803,484 and 2,924,291; the chain ladder 18,680,856; `OdpBootstrap`
  (20,000 simulations, seed 1) 18,851,366 and 2,998,662. So the posterior
  mean is 0.7% above the chain ladder and 0.3% below the bootstrap, and
  the sd 2.5% below the bootstrap's (the sanity bounds are 2% and 10%).
  The bootstrap's residual scaling `sqrt(n/(n−p))` did not open a large
  gap.
* Runtime: a default GenIns fit (4×1,000 draws after 4×1,000 warm-up,
  10,000 simulations) takes about 0.1 s in a release build and 1.5 s in a
  debug one; the whole test file about 7 s in a debug build (one test
  thread), most of it `loo()`. The parity closure computes parameter means
  and sds from the draws: calling `summary()` (R-hat and ESS too) for each
  of the 76 cases took about a minute in a debug build.

Done for PR 4 (2026-10-12). `validation/tests/reserving_bayes_bf.rs`
checks 344 rows of `reserving_bayes_bf_stan.csv` (generator
`validation/scripts/reserving_bayes_bf_stan.py`, model
`validation/scripts/stan/bayes_bf.stan`, the PR 1 pipeline: seed
20261012, 4 chains of 2,000 warm-up and 25,000 draws, dense metric,
`adapt_delta` 0.99, started at the strong-prior mode; max R-hat 1.0002,
no divergence): every parameter's mean and sd where `cv` is finite (under
`cv = ∞` only `alpha + beta` is identified), each origin's and the total
reserve's mean, sd and three percentiles, and `phi`. Four cases: GenIns
with premium (`genins_premium.csv`), apriori 0.5, `cv` 0.2, a stated prior
a little above its chain-ladder loss ratio (about 0.45); EVW's triangle at
their `a = 100` (`cv` 0.1) with `development_sd` 10 and 100, and at their
`a = 0` (`cv = ∞`) with 100. Our Monte Carlo error at the defaults over
seeds 0 to 199, with `s` Stan's posterior sd: at most `0.021 s` on a mean,
2.75% on an sd and `0.063 s` on a percentile, alike in all four cases (no
heavy tail, unlike Meyers' one-cell youngest origin). Tolerances `0.1 s`,
15% and `0.3 s`, each at least 4.7 times the largest spread; a first 10%
on the sds was 3.6 times and was raised. Seeds 0 to 199 pass every row,
using at most 0.69 of a tolerance (seed 0 at most 0.38); no fit in the
sweep had a divergence or a warning. The sweep is the ignored test
`seed_sweep` (release build, `BF_SEEDS` seeds; 200 seeds take about 2.5
minutes).

EVW's printed totals (their Gibbs runs of 1,000,000 draws, flat pattern
prior; ours at `development_sd` 100, seed 0; Stan): `a = 100` theirs
6,145,526 (MSEP^(1/2) 422,526), ours 6,146,087 (415,078), Stan 6,145,195
(422,771); `a = 0` theirs 6,049,398 (430,160), ours 6,050,384 (426,478),
Stan 6,051,957 (429,728); `a = ∞`, closed form, theirs 6,644,047
(395,012), ours from the data 6,644,048 (395,010; their `phi` is rounded),
a fit at `cv` 0.001 6,639,993 (399,035). Checked: means within 0.5% and
MSEPs within 5% (our seed-to-seed sd over 200 seeds at most 0.10% and
1.1%, largest errors 0.29% and 3.5%), the closed form within 2, and the
classical BF on the same `m_i`, 7,356,578, as a fixed value.

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

Proposed; built for `BayesOdp` in PR 1 (2026-10-12) with these changes from
the sketch below, which is kept as proposed:

* **One segment.** `fit_segments`, the `dependence` argument and Python's
  `segment(...)` wait for PR 2 with the Rust side. R's `segment()` takes a
  `bayes_odp_fit` as it takes any single-segment fit without keys: with no
  condition it returns the fit, and a condition on a key is an error.
* **Arguments.** Python `BayesOdp(intercept_sd=10, origin_sd=10,
  development_sd=10, process="gamma", chains=4, tune=1000, draws=1000,
  n_sims=10000, seed=0, target_accept=0.95, max_depth=10)` and R
  `bayes_odp(triangle, column = NULL, ...)` with the same arguments in the
  same order: `BayesGlm`'s other two sampler settings come last, so the
  order of the sketch holds. `Sampler::seed` is not an argument (`seed`
  drives both streams).
* **The fit.** Both languages have `chain_ladder`, `origins`,
  `development`, `scale`, `reserves` (a `PredictiveDistribution` by
  origin), `diagnostics` (a dict or list of `max_rhat`, `min_ess_bulk`,
  `min_ess_tail`, `divergences` and `warnings`), `parameter_names` (Python)
  and `draws` (one row per draw, chain by chain; R a matrix with the names
  as columns). The long tables are Python's `to_frame()`, `totals_frame()`
  and `development_frame()` and R's `as.data.frame()`, `totals_frame()` and
  `development_frame()`, with the `mean` and `std_dev` of the posterior
  predictive reserve, as the bootstraps'.
* **Summary and LOO** reuse what `BayesGlmFit` exposes: the same columns
  (`parameter`, `mean`, `sd`, `q05`, `q50`, `q95`, `rhat`, `ess_bulk`,
  `ess_tail`), as a pandas frame from Python's `summary()` (the reserving
  fits' tables are frames) and as R's `@summary`, which `summary(fit)` also
  returns; PSIS-LOO is Python's `Elpd` from `loo()` and R's `bayes_loo()`,
  whose result has the shape it has for a `bayes_glm_model`. This touched
  two Models-lane files, each by a few lines: `PyElpd::from_loo` in
  `crates/prospicio-python/src/models.rs` (the fields are private to that
  module, and `BayesGlmFit.loo` now uses it too) and the `model` argument's
  documentation of `bayes_loo()` in `R/prospicio/R/models.R`.
* **Warnings.** A fit with failed diagnostics raises one Python
  `UserWarning` or one R `warning()` that lists every message, and is
  returned.
* **`Meyers` (PR 3)** follows the same pattern, for one segment: Python
  `Meyers(model="crc", logelr_mean=-0.4, logelr_sd=3.162, origin_sd=3.162,
  development_sd=3.162, gamma_sd=0.05, chains=4, tune=1000, draws=2000,
  n_sims=10000, seed=0, target_accept=0.95, max_depth=10)` with
  `fit(triangle, column, premium)`, and R `meyers(triangle, column,
  premium, model = c("crc", "csr"), ...)` with the same arguments in the
  same order (`column` and `premium` are required, as in `cape_cod()`).
  The fit (`MeyersFit`, R `meyers_fit`) has `BayesOdpFit`'s fields less
  `scale`, plus `model`, `premium` and `ultimates` (a second
  `PredictiveDistribution` by origin); its long tables add `premium`,
  `ultimate_mean` and `ultimate_std_dev`. LOO is on Meyers' log scale, so
  the documentation limits comparisons to CRC against CSR on one column.
  The warnings say `Meyers:` and `meyers:`. Tests:
  `python/tests/test_reserving_meyers.py` and
  `R/prospicio/tests/test-reserving-meyers.R` fit group 353 paid with the
  default chains for CRC and CSR (no warning; totals within 2% and elpd
  within 2 of Meyers' printed values, gamma within 0.01), replay an
  incurred fit by seed, warn on 2 × 10 kept draws, and check the errors
  (settings, a missing premium column, a zero cell, too few cells).
* **`BayesBf` (PR 4)** follows it too, for one segment: Python
  `BayesBf(apriori=1.0, cv=0.1, development_sd=10, process="gamma",
  chains=4, tune=1000, draws=1000, n_sims=10000, seed=0,
  target_accept=0.95, max_depth=10)` with `fit(triangle, column,
  exposure)`, and R `bayes_bf(triangle, column, exposure, apriori = 1,
  cv = 0.1, ...)` with the same arguments in the same order (`column` and
  `exposure` are required, as in `bornhuetter_ferguson()`). A flat prior
  is `cv=float("inf")` in Python and `cv = Inf` in R. The fit (`BayesBfFit`,
  R `bayes_bf_fit`) has `BayesOdpFit`'s fields plus `exposure` and
  `prior_ultimate`, which its long tables add too; the parameters are
  `alpha[1..n]` and `beta[1..m]`. The warnings say `BayesBf:` and
  `bayes_bf:`. Tests: `python/tests/test_reserving_bayes_bf.py` and
  `R/prospicio/tests/test-reserving-bayes-bf.R` fit EVW's triangle with
  the default chains (no warning, EVW's `phi`, the total within 1.5% of
  EVW's printed 6,145,526, the summary, the frames and LOO), check the
  two limits (a flat prior within 2% of `BayesOdp` and 1.5% of EVW's
  6,049,398, cv 0.001 within 1.5% of EVW's closed form 6,644,047, the
  cv 0.1 reserve between them), replay a fit by seed, warn on 2 × 10 kept
  draws, and check the errors (settings, a missing column, an origin
  whose increments sum to -5: refused under a flat prior, fitted under
  an informative one, with no LOO). Over seeds 0 to 29 these means sat
  within 0.7% of their targets with sds of 0.15% to 0.23%, so each
  tolerance is at least 6 sds beyond the mean offset.
* **Tests**: `python/tests/test_reserving_bayes.py` and
  `R/prospicio/tests/test-reserving-bayes.R` fit GenIns with the default
  chains (no warning, the Stan-matched scale, the reserves near the chain
  ladder, the summary, the frames and LOO), replay a fit by seed, warn on
  2 × 10 kept draws, and check the errors (settings, a negative origin
  sum, several segments, LOO on RAA). The parity with Stan stays in
  `validation/tests/reserving_bayes.rs`.

The sketch, as proposed. Every model lands in Python and R in its own PR,
in the style of `OdpBootstrap` and `odp_bootstrap()`:

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
   reuses PR 1's model. Built as (a) on 2026-10-12 at the user's request
   (decision 9).
9. **Compartmental reserving.** (a) A separate design note after PR 3;
   (b) part of this plan. *Recommended: (a)*: its identification needs
   informative priors and its ODE variants need numerical gradients,
   which deserve their own decisions.
10. **Bindings with each model.** (a) Rust, Python and R in the same PR,
    as `reserving-v02.md` decision 7; (b) Rust first, bindings after.
    *Recommended: (a)*.
