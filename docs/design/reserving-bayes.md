# Design note: Bayesian reserving (Bayesian chain ladder, Meyers' models)

Status: **Partly decided; PR 1 merged (#188); PR 3 merged (#200); PR 2 merged (#207); PR 4 merged (#208); PR 5 (CAY) built; PR 6 (Meyers over segments and in the backtest) built; PR 8 (the one-year view by PSIS) built; PR 9 (one generic one-year view, `BayesBf`'s) built; PR 13 (the light refit of the compartmental one-year view, decision 24, #394) built, its default open decision 17** (proposed 2026-10-11; on 2026-10-12 the user decided open decisions 1, 2 and 5 as recommended and 7 for the CAS data model repository, and took the recommendation on open decisions 3 and 4 for now; decisions 1 to 6 and 10 below are settled. PR 1 (`claude/bayes-odp`, merged as #188) built the `bayes` feature, `BayesOdp`, the Stan reference on GenIns and RAA, and the Python and R bindings (decision 13, for one segment). PR 3 (`claude/bayes-meyers`, merged as #200, decision 7) built: `Meyers` with CRC and CSR in Rust, Meyers' group 353 data, his transcribed Stan code as the reference, and the Python and R bindings (decision 13, for one segment). PR 2 (`claude/bayes-segments`, merged as #207) built decisions 8 and 11 ( `BayesOdp::fit_segments` and `BayesOdp` in the diagonal backtest, with `fit_segments` in Python and R, decision 13). PR 4 (`claude/bayes-bf`, merged as #208; decision 9, at the user's request) built `BayesBf`, EVW's triangle, the Stan reference, and the Python and R bindings (decision 13, for one segment). PR 5 is built on `claude/bayes-cay` (decision 7, "CAY"): `MeyersModel::Cay`, Meyers' `CAY.R` transcribed as the Stan reference, his printed results, and the Python and R bindings (`Meyers("cay", rho_shape=2.0)`, `meyers(model = "cay", rho_shape = 2)`; decision 13). PR 6 is built in Rust on `claude/bayes-meyers-segments`, stacked on PR 5 (decisions 8 and 11, "Meyers"): `Meyers::fit_segments`, `Meyers::dependence` and `TriangleModel for Meyers` for CRC, CSR and CAY, with premium added to the CLRD lines, and `fit_segments` in Python and R (`Meyers(..., dependence=, spearman=).fit_segments`, `meyers_segments()`; decision 13). PR 8 is built in Rust on `claude/bayes-one-year` (decision 16): `BayesOdp::one_year` and `one_year_segments`, the claims development result by PSIS reweighting with the flagged simulations refitted by NUTS, with the Python and R bindings (`BayesOdp.one_year` and `one_year_segments`, `bayes_odp_one_year()` and `bayes_odp_one_year_segments()`; decision 13). PR 9 is built on `claude/bayes-one-year-generic`, stacked on the Models-lane PR #273 that made `prospicio_bayes::elpd::psis` public (decision 17, open decisions 12 to 14 decided by the user on 2026-10-15): one interface every Bayesian model's one-year view implements, the view written once over it, a conformance suite run on every implementor, `BayesOdp` moved onto it bit for bit, `BayesBf::one_year` with its Python and R bindings (`BayesBf.one_year`, `bayes_bf_one_year()`; decision 13), and moment matching before refitting (decision 18, the opt-in `MomentMatchThenRefit`, in Rust, Python and R). PR 10 is built on `claude/bayes-one-year-meyers`, stacked on PR 9 and the CI slow-tests PR #279 (decision 19, the user's decisions of 2026-10-15 on open decisions 12 and 14): moment matching is the default fallback of the one-year view, and `Meyers` (CRC, CSR and CAY) has the one-year view through `YearModel`, with `one_year_segments`, in Rust, Python and R. PR 11 is built on `claude/bayes-one-year-compartmental`, stacked on PR 10 (decision 20): `Compartmental` (models 1 and 2) has the one-year view through `YearModel`, and every model's view takes a refit budget (`max_refits`), in Rust, Python and R; the user decided open decision 15 on 2026-10-16: model 1's view moment-matches by default, model 2's reweights only with a warning that it is approximate, both run at most 2,000 one-year simulations by default, and pull requests run model 1 of the compartmental one-year validation only. `Compartmental::one_year_segments` is built on `claude/compartmental-one-year-segments` (`reserving-compartmental.md` decision 8, "Built (one-year view over segments)"), in Rust, Python and R. The rest stay Proposed) · Lane: Reserving, with the Models lane for `prospicio-bayes` · Depends on: `triangle.md`, `reserving-v02.md` (exposure, tails, the bootstraps, dependence between segments), `models.md` (`prospicio-bayes`, NUTS, diagnostics), `predictive-distribution.md`, `docs/roadmap.md` (v0.7 row; `prospicio-bayes` Next, item 1)

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
| 5 | `claude/bayes-cay` | `MeyersModel::Cay` (decision 7, "CAY"), in Rust, Python and R | PR 3 |
| 6 | `claude/bayes-meyers-segments` | Meyers' CRC, CSR and CAY with `fit_segments` and `TriangleModel` (open decision 9; decisions 8 and 11, "Meyers"), in Rust, with `fit_segments` in Python and R | PR 5 |
| 7 | next | compartmental reserving, in its own design note (open decision 9) | |
| 8 | `claude/bayes-one-year` | the one-year view by PSIS reweighting, `BayesOdp` first, with segments (open decision 6, decision 16); Rust, then the bindings | PR 1 |
| 9 | `claude/bayes-one-year-generic` | one generic one-year view for every Bayesian model (`YearModel`, decision 17) on the public PSIS, its conformance suite, and `BayesBf::one_year`; moment matching before refitting, opt-in (decision 18); in Rust, Python and R | #273 (PR 8 merged) |
| 10 | `claude/bayes-one-year-meyers` | moment matching the default one-year fallback; `Meyers::one_year` and `one_year_segments` for CRC, CSR and CAY through `YearModel` (decision 19); in Rust, Python and R | PR 9, #279 |
| 11 | `claude/bayes-one-year-compartmental` | `Compartmental::one_year` for models 1 and 2 through `YearModel` (decision 20), a refit budget (`max_refits`) for every model's view; in Rust, Python and R | PR 10 |
| 12 | `claude/backtest-bindings` | the diagonal backtest in Python and R, each fit's diagnostics in the forecast and the backtest, two more metrics (decision 21, #342) | `main` |
| 13 | `claude/gm2-one-year-accuracy` | `BayesOneYear::light_refit`: a warm-started, shorter sampler for the refits of `Compartmental`'s one-year view, model 2's within 0.03 sd of full refits (decision 24, #394); in Rust, Python and R | `main` |
| later | | the paid-incurred `Ipi` | PR 5 |

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
| CRC, CSR, CAY (later IPI) | Meyers' R/Stan code, `meyers-appendix.zip` (CAS; `CRC.R`, `CSR.R`, `CAY.R`), transcribed into current Stan syntax (no licence is stated, and the old array syntax no longer compiles) and run with CmdStan 2.39.0 | Meyers (2019), CAS Monograph 8, 2nd ed.: group 353 Commercial Auto, data in Tables 2.1-2.3, parameter summaries in Tables 5.1, 7.1 and 8.1, totals in Tables 5.2 (CRC paid), 5.3 (CRC incurred), 7.2 (CSR paid) and 8.2 (CAY incurred), elpd_loo in Tables 6.4, 7.3 and 8.3 |
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
200 iterations (this note said 100; the code has always had 200). PR 2
adds a second rounding floor: a full step that lowers `f` by no more than
`1e-12 |f|` stops too. CLRD's private passenger auto summed over
companies needed it: its `b[10]` rests on one small cell against
increments near `10⁷`, rounding kept its Newton step near `1e-6`, the
halving reached a step that no longer moved `θ`, and the method ran out
of iterations. GenIns and RAA draws are unchanged bit for bit. `H = R Rᵀ` with `R` lower triangular and `L = (Rᵀ)⁻¹`.
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

* `fit_segments` and `BayesOdpFits` came in PR 2 (decision 11). `BayesOdpFit` also has
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
pub enum MeyersModel { Crc, Csr, Cay }         // default Crc; Cay in PR 5
pub struct Meyers {
    pub model: MeyersModel,
    pub logelr_mean: f64,     // -0.4
    pub logelr_sd: f64,       // 3.162
    pub origin_sd: f64,       // 3.162
    pub development_sd: f64,  // 3.162
    pub gamma_sd: f64,        // 0.05, CSR only
    pub rho_shape: f64,       // 2, CAY only (PR 5): (1 - rho) / 2 ~ Beta(2, 2)
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
  `TriangleModel` (decision 8). (PR 6 adds both: decisions 8 and 11,
  "Meyers"; in the backtest a path is drawn, since increments are.)
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

**CAY** (PR 5, built on `claude/bayes-cay`, 2026-10-13, in Rust, Python
and R). The model is Meyers' correlated accident
year model (2019, sec. 8, "The Correlated Accident Year Model"), read from
his `CAY.R` in the CAS appendix (lines 160-211 hold the Stan program,
lines 317-328 the prediction) and checked against the monograph's
equations:

```text
mu_1d = log P_1 + logelr + beta_d
mu_wd = log P_w + logelr + alpha_w + beta_d + rho (log C_(w-1)d - mu_(w-1)d),   w > 1
log C_wd ~ N(mu_wd, sigma_d^2),   sigma_d^2 = s_d + ... + s_m as in CRC
rho = 1 - 2 r,   r ~ Beta(2, 2)   (so rho in (-1, 1), prior density proportional to 1 - rho^2)
logelr ~ N(-0.4, 3.162^2),   alpha_w, beta_d ~ N(0, 3.162^2),   alpha_1 = 0,  beta_m = 0
```

`log C_wd - mu_wd` is the cell's own innovation, so down each age's column
`log C` is a moving average of order 1: the correlation of neighbouring
origins is `rho / (1 + rho^2)`, of origins further apart 0 (Meyers' eq.
8.3, credited to John Major). Meyers fits it to incurred data, where CRC's
tails are too light; with `rho = 0` it is CRC. There is no `gamma`: Meyers
combines CAY's correlation with CSR's speed-up only across paid and
incurred, in IPI (sec. 9), which stays later.

* **A third `MeyersModel` variant, not its own type.** CAY shares CRC's
  data (cumulative values and premium), priors, variance ladder and floor,
  parameter layout (one extra scalar where CSR has `gamma`), sampler path
  (the pilot-run scale), output (`MeyersFit` with ultimates and reserves
  by origin), PSIS-LOO on the same log scale and long tables. What differs
  is a recursion in the likelihood and in the prediction, about 100 lines
  inside the private model. A separate type would duplicate `fit`, the fit
  type, the long tables and, next, the bindings and `fit_segments`, for
  one parameter. So `MeyersModel::Cay` it is, with `rho` in the `gamma`
  slot of the posterior names, and one new prior field,
  `Meyers::rho_shape` (default 2, Meyers' `Beta(2, 2)`; both shapes of a
  symmetric beta on `(1 - rho) / 2`), used only by CAY as `gamma_sd` is
  only by CSR. The bindings follow: `model="cay"` and `rho_shape` next
  to `gamma_sd` (below).
* **Sampling.** `rho = tanh(v)` with its log Jacobian, so the prior on
  `(1 - rho) / 2` is exactly `Beta(a, a)`: `a (log logistic(2v) + log
  logistic(-2v))` in `v`. The gradient is hand-written: the residuals run
  forward in origin order, and `d f / d mu_i` runs backward down each
  age's column (`h_i = r_i / sigma_d^2 - rho h_(next i)`), as reverse-mode
  autodiff would; it is checked against central differences on the
  unconstrained and the whitened scales, as for CRC and CSR (no autodiff
  library is added; one is being evaluated separately).
* **Data requirement.** Each observed cell after the first origin needs the
  previous origin's cell at the same age (Meyers sorts the cells by age,
  then origin, and takes cell `i - 1`). Any triangle whose latest ages do
  not increase with the origin has that; one that does not is an
  `Error::Bayes` naming the cell (CRC and CSR still take it).
* **Prediction.** Given a draw, the ultimates are drawn down age `m` origin
  by origin: each mean is `log P_w + logelr + alpha_w + rho` times the
  previous origin's residual at age `m`, observed for a closed origin
  (Meyers starts from the oldest origin's observed value) and drawn
  otherwise. As for CRC and CSR, no `ceiling()` (his adds about +4.5 to
  the total). Ages are independent given the parameters, so age `m` alone
  is drawn.
* **`loo()`.** Meyers' pointwise term, `normal_lpdf(log C | mu, sigma_d)`
  with `mu` holding the previous origin's observed residual, so each term
  is conditional on that cell, as in his code. It is on the same log scale
  as CRC's and CSR's, so CAY compares with them on the same column.
* **Validation** (decision 10, "Done for PR 5"): parity with the
  transcribed `CAY.R` under CmdStan 2.39.0 and with Meyers' Tables 8.1 to
  8.3. Results on group 353 incurred (total ultimate mean and sd; ours at
  the defaults, seed 0; Stan 4 x 25,000 draws; Meyers printed):

  | | Ours | Stan | Meyers |
  |---|---|---|---|
  | Total ultimate | 39,178 (1,834) | 39,142 (1,862) | 39,193 (1,859), Table 8.2 |
  | `rho` | 0.1686 (sd 0.2075) | 0.1729 (0.2069) | 0.1709 (0.2071), Table 8.1 |
  | `logelr` | -0.3947 (0.0144) | -0.3947 (0.0150) | -0.3945 (0.0150), Table 8.1 |
  | elpd_loo | 69.05 | 68.47 | 68.65, Table 8.3 |

  CRC incurred gives 39,080 (1,541) here and 39,132 (1,616) in Stan:
  CAY's total sd is about 15% higher, as Meyers finds (his Table 5.3
  against 8.2), and his elpd_loo favours CRC on this triangle (70.97
  against 68.65), as ours does (71.42 against 69.05).
* **Runtime.** A default CAY fit takes 0.5 s in a release build, as CSR;
  `validation/tests/reserving_meyers_cay.rs` takes about 7 s in a debug
  build, most of it the one default fit.
* **Bindings** (decision 13): no new type. Python `Meyers(model="cay",
  ..., gamma_sd=0.05, rho_shape=2.0, chains=4, ...)` and R
  `meyers(triangle, column, premium, model = c("crc", "csr", "cay"), ...,
  gamma_sd = 0.05, rho_shape = 2, chains = 4, ...)`: `rho_shape` sits
  after `gamma_sd`, so positional calls past `gamma_sd` change (none in
  the repo's docs or tests). `MeyersFit` and `meyers_fit` are unchanged;
  `model` reads `"cay"` and `rho` takes `gamma`'s slot in
  `parameter_names`, the draws and the summary. Tests in
  `python/tests/test_reserving_meyers.py` and
  `R/prospicio/tests/test-reserving-meyers.R`: a default fit on group 353
  incurred with no warning, `rho` in slot 20 (1-based) and inside
  (-1, 1), the provenance (`meyers_cay` with `rho_shape`, no
  `gamma_sd`), Meyers' printed `rho` (within 0.05), `logelr` (0.01),
  total ultimate (2%) and elpd_loo (2.5), the missing previous-origin
  cell error (CRC takes the same triangle) and an invalid `rho_shape`.

### 8. The diagonal backtest

Built in PR 2 (Rust, 2026-10-12) as proposed below, with these choices
the proposal left open:

* **Levels and scale.** `forecast` fits the model of decision 3 on the
  train rows' increments, with an origin and a development effect for
  each level the train rows have (renumbered), and `φ` the Pearson scale
  of the train rows at the likelihood's mode on `N − p` degrees of
  freedom. That mode is the ODP GLM's MLE, which is the chain ladder's
  fit when the train rows form a triangle, so on a whole triangle `φ` is
  `BayesOdpFit::scale` (a unit test, to 1e-8). A test row whose origin or
  age has no train row is `Error::Bayes` (the backtest already leaves
  such rows out for every model), as are too few train rows and decision
  3's row and column check.
* **Seeds and size.** The chains and the simulations take seeds derived
  from the backtest's `seed` as `fit` does from `BayesOdp::seed`, and the
  backtest's `n_sims` replaces `BayesOdp::n_sims`. The diagnostics are
  not reported (the trait has no place for them; reversed on 2026-10-19
  by decision 21: `CellForecast::diagnostics` carries them and
  `Backtest::diagnostics` and `warnings` report them).
* **Outputs.** `mean` is each test cell's posterior mean of `exp(η)`; the
  draws add `process` to a posterior draw; the log density is the log of
  the mean over every posterior draw of the normalised density
  (`Family::Poisson.log_density`), the log predictive density, as
  `GlmCandidate` gives it with its parameter draws. The name is
  `"bayes_odp"`; two settings of `BayesOdp` in one backtest share it (a
  wrapper with a name is left for when it is needed).
* **Results** (`validation/tests/reserving_bayes_segments.rs`, latest
  three diagonals, tolerances at least 4 seed-to-seed sds over seeds 0 to
  69, every row within at most 0.65 of its tolerance at each of them): on
  the CLRD comauto and wkcomp lines `BayesOdp` and the ODP GLM agree (A/E
  within 0.002 per split, CRPS within 3% and 5%, mean log density within
  0.2). On GenIns the Bayesian means are about 1% higher (A/E lower by
  0.007, 0.013 and 0.010 per split on average over seeds, and by 0.004 to
  0.019 across seeds and splits: the posterior mean of `exp(η)` over the
  young origins' wide posteriors against the GLM's MLE) and its mean log
  density lower by 0.013 on average (sd 0.0044 over seeds, 0.0051 over
  seeds 40 to 69 alone; the test allows 0.025 around 0.013). Both models
  under-cover the CLRD lines' held-out cells alike (across seeds, 90%
  intervals hold 0.69 to 0.73 of comauto's and 0.41 to 0.46 of wkcomp's),
  a property of the ODP model on lines summed over companies, not of the
  sampler.

**Meyers** (built in PR 6, Rust, 2026-10-13). `Meyers` implements
`TriangleModel` for CRC, CSR and CAY, named `meyers_crc`, `meyers_csr` and
`meyers_cay`. The adapter the proposal below sketched, with one change:
the log density is not `None` in the diagonal backtest.

* **Cumulative values.** Each origin's train rows must run from the first
  age without a gap, and their running sums are its cumulative values
  (exact for integer cells), which must be positive. The train origins
  must have no gap between them (CAY's previous origin is the one before).
  The model's ages run from the first to the oldest train age, which takes
  `beta_m = 0`, and the fit is `Meyers::fit` on those cells: on a whole
  triangle with its premium as exposure, the backtest's draws of the next
  cell of an origin closed one age short are the fit's reserve draws,
  simulation by simulation (a unit test).
* **Premium** is the frame's `exposure` column (`TriangleFrame::new`'s
  exposure per origin). A frame built without exposure has 1 there; the
  model fits, but Meyers' `logelr` prior is then far from the data, so the
  doc asks for the premium.
* **Paths.** A test row must be after its origin's latest train row, at an
  age with train rows (`Error::Bayes` otherwise). Each simulation takes one
  posterior draw and simulates each origin's cumulative value from the age
  after its latest train row to its oldest test row; a test increment is
  the difference of consecutive values on that path, the first from the
  latest train value, so an origin's increments sum to its last value less
  its latest train value (a unit test: the 1997 origin's nine future cells
  sum to the fit's reserve, mean and sd). In CAY the paths run age by age
  and, within an age, origin by origin, each mean taking `rho` times the
  previous origin's residual, observed for a train cell and drawn
  otherwise.
* **Mean and log density.** The mean is the posterior mean of the
  log-normal `E[C_wd]` less the latest train value (less `E[C_w,d-1]` past
  a path's first age). In the diagonal backtest every test row follows its
  origin's latest train row, so a test increment is `C_wd` less a known
  value and its density is the log-normal density of `C_wd` at the
  increment plus that value: the log density is the log of its mean over
  every posterior draw, the log predictive density as for `BayesOdp`. In
  CAY a previous origin's cell that is not a train cell is integrated out
  (variance `(1 + rho^2) sigma_d^2`). Only when a test row is further out
  (its increment a difference of two random log-normals, with no closed
  density) is the log density `None`. A unit test checks the mean and the
  log density against the fit's posterior, written out, to `1e-9`.
* **Seeds and size** as for `BayesOdp`: the chains (with the pilot run)
  and the simulations on seeds derived from the backtest's `seed`, the
  backtest's `n_sims`, no diagnostics reported (until decision 21 of
  2026-10-19, which reports each split's). So a failed fit was
  silent, and on small held-out triangles the sampler settings matter:
  on group 353's 28- and 36-cell training triangles the default target
  acceptance 0.95 gave a fit with R-hat above 1.05 at 20 of 220 seeds
  (up to 1.18 and 978 divergences), and those fits made the scores'
  largest outliers (up to 9 seed-to-seed sds). At 0.99 such a fit comes
  at 4 of 300 seeds (R-hat up to 1.36 at seed 317). The test uses 0.99
  on group 353 (CLRD's summed lines need no change), and its seed sweep
  refits each split's training triangle with `Meyers::fit` on the
  backtest's seed (the same chains) to find failed fits and count their
  rows instead of checking them. Use a higher target acceptance than the
  fit's when a backtest holds out a small triangle's diagonals; the fits'
  diagnostics now come through `CellForecast` (decision 21).
* **Results** (`validation/tests/reserving_meyers_segments.rs`, latest
  three diagonals; each checked row's target is its mean over seeds 0 to
  119 and its tolerance at least 6 seed-to-seed sds, as the rows' tails
  are a little heavier than normal, and every row passes at the fresh
  seeds 200 to 379; with every fit converged the largest distance over
  seeds 0 to 119 and 200 to 379 is 5.3 sds. On incurred each model's
  own CRPS and log density are checked, not ratios to CRC's, which moved
  with CRC's chains). Meyers' models are not near the ODP models here,
  and the test checks each against its own sweep mean. On group 353 they
  forecast the held-out diagonals well above the outcomes: A/E 0.30 to
  0.36 on the earliest split (seven train diagonals) and 0.65 to 0.90 on
  the latest (CRC 0.66, CAY 0.65, CSR 0.90, the paid correction Meyers
  finds for CSR), against the ODP GLM's 0.82 to 0.97. With diagonals held
  out, the late `sigma_d` (whose steps have Meyers' `U(0, 1)` prior) and
  the young origins' levels rest on few cells, and the mean of a
  log-normal grows with both: on the seven-diagonal triangle `sigma[1]`
  has posterior mean 0.73 and `alpha[7]` sd 0.83. Meyers' intervals cover
  every held-out cell of group 353 and CLRD comauto and 0.92 to 1 of
  wkcomp's, where the ODP models cover 0.41 to 0.46 of wkcomp's; so on
  wkcomp Meyers' mean log density beats the GLM's by 0.54 to 0.60 a cell,
  on comauto it trails by 0.36 to 0.46 and on group 353 paid by 1.28 to
  1.39. On group 353 incurred (where the GLM's IRLS does not converge)
  CSR's and CAY's log densities are within 0.07 of CRC's. This is the
  diagonal view only; Meyers' own calibration test is of the ultimate,
  from the whole upper triangle (the 200-triangle study, built: `reserving-calibration.md`).

The proposal:

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
on cumulative cells for these models, is a later PR. (Built in PR 6,
"Meyers" above, with a closed log density in the diagonal backtest.)

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
* **`apriori` sets the row parameter, not the ultimate.** Because the
  pattern is not normalised, `m_w = apriori × exposure_w` is the prior
  mean of `mu_w`, EVW's row parameter (their "prior ultimate"), and the
  model's expected ultimate is `mu_w Σ_d gamma_d`: under a tight prior
  `m_w Σ_d S_d / M_d` over all columns. On EVW's triangle that sum is
  0.885, so an `apriori` read as a loss ratio gives an implied ultimate
  loss ratio about 11.5% lower. The method follows EVW; the docs of
  `apriori` and `prior_ultimate` in Rust, Python and R say so, and the
  names stay EVW's.
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

Done for PR 5 (2026-10-13). `validation/tests/reserving_meyers_cay.rs`
checks 129 rows of `reserving_meyers_cay_stan.csv` for CAY on incurred
(every parameter's mean and sd; each origin's and the total reserve's
mean, sd and three percentiles, except the youngest origin's sd; the same
for the sum of the reserves of 1988 to 1996; the correlation of each
neighbouring pair of origins' reserves from 1989/1990 to 1995/1996; the
total ultimate's mean and sd; elpd_loo) and Meyers' 7 printed numbers
(Tables 8.1 to 8.3: the total ultimate and its sd, `logelr` and `rho`
with their sds, elpd_loo 68.65). Generator
`validation/scripts/reserving_meyers_cay_stan.py` (PR 3's toolchain and
data handling), Stan file `validation/scripts/stan/meyers_cay.stan`, seed
20261013, 4 chains of 2,000 warm-up and 25,000 draws, dense metric,
`adapt_delta` 0.99, `max_treedepth` 12, Meyers' `init.CAY`; max R-hat
1.0002, no divergence, Pareto k up to 0.92; a rerun reproduced every
value. The transcription agrees with Meyers' printed run within its Monte
Carlo error: total 39,142 against 39,193 (0.13%), sd 1,862 against 1,859,
`rho` 0.1729 against 0.1709, p_loo 15.72 against 15.64. Our Monte Carlo
error at the defaults over seeds 0 to 199, with `s` Stan's posterior sd:
at most `0.025 s` on a mean (`sigma[7]`), 3.3% on a parameter sd
(`logelr`), 2.4% on an origin's reserve sd (1990), 1.9% on the sd of the
1988 to 1996 sum, `0.063 s` on a percentile (95% of 1997), 0.019 on a
neighbour correlation (1995/1996) and 0.43 on elpd_loo. The total sds
have a seed-to-seed sd of 5.9% but the same heavy tail as CRC's, from the
1997 log-normal: +66% at seed 38 (+37% at most for CRC and CSR), so they
get 75% (50% for CRC and CSR) and the 1988 to 1996 sum (10%) checks the
joint spread instead. Tolerances: means `0.15 s`, parameter sds 15%,
origin reserve sds 20%, the sum's sd 10%, total sds 75%, percentiles
`0.35 s`, correlations 0.08, elpd_loo 2.0, each at least 4.1 times its
row's spread. Meyers' printed rows: total 1% (combined error with his run
0.11%), total sd 75%, `logelr` and `rho` means 0.15 printed sd (combined
0.024 sd), their sds 20% (combined 3.8%), elpd_loo 2.5 (combined 0.61).
Seeds 0 to 199 pass every row, using at most 0.88 of a tolerance (the
total sds at seed 38); seed 0 uses at most 0.48. The defaults passed every
diagnostic at 199 of the 200 seeds (seed 73: one divergent transition in
8,000 draws, which the fit reports as a warning); the largest R-hat was
1.0073. What the parity cannot see: on these data `sigma[10]` is near
0.013, so CAY's recursion in the prediction moves the neighbour
correlations by at most 0.03 and the total sd by about 0.1% (a run with
`rho` set to 0 in the prediction alone passed every row), under 2 Monte
Carlo errors. The unit test `cay_draws_the_oldest_age_as_a_moving_average`
checks it instead, at `sigma_m = 0.5` and `rho = 0.8`: origin 2's mean
takes `rho` times origin 1's observed residual, the later origins have sd
`sigma_m sqrt(1 + rho^2)`, lag-one correlation `rho / (1 + rho^2)` and no
lag-two correlation, each within 1% on 200,000 draws; with `rho` set to 0
it fails. The sweep is `seed_sweep` in the same file (release build; 200
seeds take about 3 minutes).
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

Built in PR 2 (Rust, 2026-10-12) as proposed below:

* **API.** `BayesOdp::fit_segments(triangle, column) -> BayesOdpFits`
  with `segments: SegmentFits<BayesOdpSegment>` (each segment's
  `chain_ladder`, `scale`, `posterior`, `diagnostics`, `summary()` and
  `loo()`, no reserves of its own) and the joint `reserves`, plus
  `to_long`, `totals`, `development_table` and `segment(keys)`, as
  `OdpBootstrapFits`. Segments are fitted one after another (each fit
  already runs its chains in parallel); a failure names its segment.
* **Seeds.** Segment `k` (in index order) uses tags `2 + 2k` (chains)
  and `3 + 2k` (simulations) of decision 5, so segment 0's draws differ
  from `fit` of that segment alone, as in the bootstraps.
* **`RankCorrelation`** goes through the bootstraps' own
  `SegmentDependence::reorder` with `BayesOdp::seed` (stream `n_sims` of
  that seed, which no derived seed uses).
* **`Synchronized`** is `Error::Bayes`, checked before any sampling.
  So is a bad `RankCorrelation` matrix (not `S × S` for the triangle's
  segments, not symmetric with a unit diagonal, an entry out of
  `[-1, 1]`, or `2 sin(π ρ / 6)` not positive definite): it is
  `Error::InvalidSetting` before the first segment samples, through the
  same check the bootstraps run when they reorder (which now also reports
  a matrix that is not positive definite as `InvalidSetting`).
* **Provenance** of the joint reserves: the settings, `segments`,
  `dependence`, and each segment's `c0` and `scale` as a list.
* **`Meyers` has no `fit_segments`** in PR 2: this decision names only
  `BayesOdp`, and `Meyers` has no `dependence` field (decision 7). The
  same pattern fits it when a user needs Meyers over segments; that is a
  later PR.

**Meyers** (built in PR 6, Rust, 2026-10-13), the same pattern for CRC,
CSR and CAY:

* **API.** `Meyers` gains `dependence: SegmentDependence` (default
  `Independent`; `fit` ignores it), and
  `Meyers::fit_segments(triangle, column, premium) -> MeyersFits`, each
  segment's premium from its own `premium` column as `fit` reads it.
  `MeyersFits` holds `segments: SegmentFits<MeyersSegment>` (each
  segment's `chain_ladder`, `premium`, `posterior`, `diagnostics`,
  `model()`, `summary()` and `loo()`) and the joint `ultimates` and
  `reserves` (dimensions the key names and `origin`), with `to_long`,
  `totals` (both with `premium` and the ultimates' moments as
  `ultimate_mean` and `ultimate_std_dev`), `development_table` and
  `segment(keys)`.
* **Seeds** as for `BayesOdp`: segment `k` takes tags `2 + 2k` and
  `3 + 2k`; its pilot run (decision 7) takes stream `u64::MAX` of the
  segment's chains seed, as `fit`'s takes that of its own, so `fit`'s
  draws are unchanged bit for bit.
* **`RankCorrelation`** reorders the reserves and the ultimates
  together: each simulation's row holds every segment's reserves and then
  its ultimates, and Iman–Conover ranks a segment by the total over both,
  twice its total reserve plus its total latest value, which ranks the
  simulations as the reserves alone do. Each row's ultimates less its
  reserves stay the latest values exactly (a unit test).
* **`Synchronized`** and a bad matrix fail before any sampling, after the
  settings' own check, as for `BayesOdp`.
* **Provenance** of both joint distributions: the settings, `segments`,
  `dependence`, `posterior_draws`, each segment's total premium as a list,
  and `output` (`ultimates` or `reserves`).
* **Results** (`validation/tests/reserving_meyers_segments.rs`, seeds 0
  to 59, defaults): twin lines (CLRD wkcomp under two names, now with net
  earned premium, CRC) under `Independent` have a rank correlation of
  their totals with sd 0.0083 over seeds (largest 0.022), about
  `1/sqrt(n_sims)`, within the `4.5/sqrt(n_sims)` the test allows, and
  mean totals within 0.47% (sd 0.15%). `RankCorrelation` (CSR) on
  comauto, ppauto and wkcomp reaches each target within 0.0081 (sd at
  most 0.0030; the test allows 0.0076 to 0.013, 4 sds).
* **Results** (`validation/tests/reserving_bayes_segments.rs`, seed sweep
  over 0 to 69): twin lines (CLRD wkcomp under two names) under
  `Independent` have a rank correlation of their totals with sd 0.010
  over seeds, as `1/√n_sims` predicts for independent draws (largest
  0.027); the test allows `4.5/√n_sims`, not the `3/√n_sims` proposed
  below, which is about 3 such sds and would fail about one seed in 370.
  `RankCorrelation` on comauto, ppauto and wkcomp reaches each target
  within 0.009 (sd at most 0.0029; the test allows 0.015).

The proposal:

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

Built in PR 8 for `BayesOdp` (decision 16, 2026-10-14): PSIS
reweighting, the flagged simulations refitted. PR 9 (decision 17) made the
engine generic and gave `BayesBf` the same view. Still no tail: when a
model gains one, its tail step joins the year as in the bootstraps'
one-year view (`reserving-v02.md` decision 11), an origin at the last age
closing on its simulated tail increment.

### 13. Bindings

Proposed; built for `BayesOdp` in PR 1 (2026-10-12) with these changes from
the sketch below, which is kept as proposed:

* **One segment.** `fit_segments`, the `dependence` argument and Python's
  `segment(...)` wait for PR 2 with the Rust side. R's `segment()` takes a
  `bayes_odp_fit` as it takes any single-segment fit without keys: with no
  condition it returns the fit, and a condition on a key is an error.
* **Segments (PR 2).** `fit` and `bayes_odp()` stay single-segment, so
  their seeds and results do not change; segments have their own entry
  point, as in Rust, because segment `k` draws on other derived seeds.
  Python `BayesOdp(..., dependence="independent", spearman=None)` (the
  two new arguments last) with `fit_segments(triangle, column)` returning
  `BayesOdpFits`: `chain_ladder`, `keys`, `index`, `origins`,
  `development`, the joint `reserves` (dimensions the keys and `origin`),
  `to_frame()`, `totals_frame()`, `development_frame()`,
  `diagnostics_frame()` (one row per segment) and `segment(**keys)`, and,
  for a single-segment fit only (an error naming `segment(...)`
  otherwise, as `OdpBootstrapFit.scale`), `scale`, `diagnostics`,
  `parameter_names`, `draws`, `chains`, `summary()` and `loo()`. R
  `bayes_odp_segments(triangle, column = NULL, ..., dependence =
  c("independent", "rank_correlation", "synchronized"), spearman = NULL)`
  with `bayes_odp()`'s arguments in the same order, returning a
  `bayes_odp_fits` whose `diagnostics` is a data.frame (one row per
  segment, the warnings joined by `"; "`) and whose `scale`, `summary`,
  `draws` and `bayes_loo()` need one segment; `as.data.frame()`,
  `totals_frame()`, `development_frame()` and `segment()` work as for the
  bootstraps. `"synchronized"` is accepted by the constructor, as the
  bootstraps take it, and is the Rust `Error::Bayes` at the fit. Failed
  diagnostics raise one warning that prefixes each message with its
  segment's label. The diagonal backtest has no binding in either
  language, so `TriangleModel for BayesOdp` stays Rust-only (reversed
  on 2026-10-19 by decision 21). Tests:
  `python/tests/test_reserving_bayes_segments.py` and
  `R/prospicio/tests/test-reserving-bayes-segments.R` fit comauto and
  wkcomp from `clrd_lines.csv` with the default chains (no warning, the
  frames, one segment's scale equal to `fit`'s and its posterior close),
  twin lines' rank correlation within `4.5/√n_sims` of 0 and a rank
  correlation of 0.5 within 0.03 at 4,000 simulations, replay by seed,
  a warning that names both segments on 2 × 10 kept draws, and the errors
  (dependence settings, `"synchronized"`, a missing column).
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
* **`Meyers` over segments (PR 6)** follows `BayesOdp`'s segments: `fit`
  and `meyers()` stay single-segment. Python `Meyers(...,
  dependence="independent", spearman=None)` (the two new arguments last,
  as `BayesOdp`'s) with `fit_segments(triangle, column, premium)`
  returning `MeyersFits`: `model`, `chain_ladder`, `keys`, `index`,
  `origins`, `development`, `premium` (each segment's origins in turn),
  the joint `ultimates` and `reserves` (dimensions the keys and
  `origin`), `to_frame()`, `totals_frame()`, `development_frame()`,
  `diagnostics_frame()` and `segment(**keys)`, and, for a single-segment
  fit only, `diagnostics`, `parameter_names`, `draws`, `chains`,
  `summary()` and `loo()`. R `meyers_segments(triangle, column, premium,
  model = c("crc", "csr", "cay"), ..., dependence = c("independent",
  "rank_correlation", "synchronized"), spearman = NULL)` with `meyers()`'s
  arguments in the same order, returning a `meyers_fits` with the same
  properties as `meyers_fit` plus `diagnostics` as a data.frame (one row
  per segment), and `summary`, `draws` and `bayes_loo()` for one segment
  only. The warnings say `Meyers:` and `meyers_segments:` and prefix each
  message with its segment's label. Python's segment warnings and
  diagnostics frame and R's diagnostics table now share one helper with
  `BayesOdpFits`; the diagonal backtest still has no binding (it has one
  since decision 21). Tests:
  `python/tests/test_reserving_meyers_segments.py` and
  `R/prospicio/tests/test-reserving-meyers-segments.R` fit CLRD comauto
  and wkcomp (paid, net earned premium) with the default chains (no
  warning, the frames, each simulation's ultimates less reserves equal to
  the latest values, one segment's part of the joint draws, its
  posterior against `fit` alone), twin lines within `4.5/√n_sims` of
  uncorrelated, a CSR rank correlation of 0.5 within 0.03 with the
  ultimates ranked as the reserves, CAY with replay by seed, a warning
  that names both segments on 2 × 10 kept draws, and the errors. The
  tolerances against `fit` alone come from seeds 0 to 59 at 4,000
  simulations, each at least 4 sds and 1.25 times the largest deviation:
  `logelr`'s posterior mean 0.001 (sd 0.0001), the mean total reserve 1%
  (sd 0.18%), elpd_loo 2.5 (sd 0.52, largest 1.42), and twin lines' mean
  totals 1% (sd 0.19%); no seed warned.
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
* **The one-year view (PR 8, decision 16)**: Python
  `BayesOdp.one_year(triangle, column, k_threshold=0.7,
  refit="refit_flagged")` and `one_year_segments` with the same arguments
  (`refit` is `"refit_flagged"`, `"reweight_only"` or `"refit_all"`,
  `OneYearRefit`'s variants), returning `BayesOneYearFit` and
  `BayesOneYearFits`; R `bayes_odp_one_year()` and
  `bayes_odp_one_year_segments()` with `bayes_odp()`'s (and
  `bayes_odp_segments()`'s) arguments in the same order and
  `k_threshold = 0.7, refit = c("refit_flagged", "reweight_only",
  "refit_all")` last, returning a `bayes_one_year_fit` and a
  `bayes_one_year_fits`. The settings are arguments of the method rather
  than of `BayesOdp`, as `BayesOneYear` is a separate argument in Rust,
  and the class names carry no model, as `BayesOneYearFit<F>` does not,
  so `Meyers` and the others can return them too. The single fit has
  `fit` (the `BayesOdpFit`, `bayes_odp()`'s for the same settings),
  `chain_ladder`, `origins`, `development`, `latest`,
  `opening_ultimate`, `opening_reserve`, `cdr`, `pareto_k`, `refitted`,
  `refit_warnings`, `k_threshold`, `flagged_share` and the frames
  (`totals_frame()` adds `scale` and `flagged_share`); the fit over
  segments has the layout of `OneYearFit` plus `diagnostics` and
  `diagnostics_frame()`, with `scale`, `pareto_k`, `refitted`,
  `refit_warnings` and `flagged_share` for one segment only. `refitted`
  is 0-based in Python and 1-based in R. Failed diagnostics of the fit
  raise the usual warning, and refits that failed theirs one more
  (`"2 of 1357 refits"`, each segment's label first). Tests:
  `python/tests/test_reserving_bayes_one_year.py` and
  `R/prospicio/tests/test-reserving-bayes-one-year.R` run GenIns with
  the default chains and `"reweight_only"` (no warning, the fields and
  frames, the lifetime fit equal to `fit`'s, the oldest origin's CDR 0,
  the mean within 5 standard errors of 0, the flagged share between 0.1
  and 0.45, the sd within 12% of the ODP bootstrap's one-year view, a
  loose check because the validation test pins it), replay by seed, the
  refitted simulations on a four-origin triangle (those above
  `k_threshold`, or all; a refitted row is `"refit_all"`'s and the others
  `"reweight_only"`'s), the refit warning on 2 × 20 kept draws, two
  segments (one segment's fields, the frames, a rank correlation of 0.8
  within 0.05) and the errors.
* **The one-year view of `BayesBf` (PR 9, decision 17)**: Python
  `BayesBf.one_year(triangle, column, exposure, k_threshold=0.7,
  refit="refit_flagged")` and R `bayes_bf_one_year()` with `bayes_bf()`'s
  arguments in the same order and `k_threshold` and `refit` last. They
  return the same `BayesOneYearFit` and `bayes_one_year_fit`, whose `fit`
  is the model's own lifetime fit (`BayesBfFit`, `bayes_bf_fit`); the
  frames add `exposure` and `prior_ultimate`. In Python the class holds
  either model's view; in R the extendr object says which (`model()`) and
  hands out `fit()` or `bf_fit()`. Tests: the two files above add a
  four-origin triangle with premium (the lifetime fit equal to `fit`'s
  draw for draw, the oldest origin's CDR 0, the frames, the errors).
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

Proposed. Each PR updates, in itself: the Reserving lane's status file (`docs/lanes/reserving.md`), the
v0.7 row of `docs/roadmap.md` and the `prospicio-bayes` Next list (item 1
is this work), `docs/architecture.md` if a roadmap row changes, and this
note's status. `knowledge/` gains: the Stan reference pipeline and the
local CmdStan facts (environment), Meyers' group 353 data and its
agreement with Tables 2.1-2.3 (datasets), and whether the Bayesian ODP and
the ODP bootstrap agree on GenIns and by how much (findings), with
`verified` once a CI test confirms each.

### 15. Gradients: an in-house dual type, num-dual as the fallback

Decided 2026-10-13 by the user. Each new Bayesian model writes its log
density once, generic over a small `Scalar` trait, and gets its gradient
from an in-house forward-mode dual type in `bayes/` (no dependency, about
130 lines). `num-dual` is the fallback behind the same trait (it needs Rust
1.89 at 0.15, or 0.13 for 1.87). The hand-written gradients of `BayesOdp`,
`Meyers` and `BayesBf` stay as the fast path, and each gains a test against
the dual gradient. Laplace Hessians stay as central differences of the
exact gradient.

Why, from a benchmark of Meyers' CRC (29 parameters, 55 cells, release
build; `knowledge/findings/autodiff-benchmark.md`): every option matched the
hand-written gradient to about 1e-15, but per gradient the in-house dual
took 2.5 µs (4 to 5 times the hand-written 0.57 µs), `num-dual` 7 to 17 µs
at a fixed size and 50 µs at a runtime size, candle about 55 µs and Burn
about 100 µs. Burn and candle add over 110 crates each (candle compiles C
through `onig_sys`), and Burn 0.21's `sigmoid` drops to f32 on an f64
backend. tch needs libtorch, which CRAN cannot vendor, and
`std::autodiff` is nightly only. A generic RK4 solver differentiated with
duals matched a closed-form one-compartment model to 9e-10, so the
compartmental models need no ODE crate unless they turn out stiff
(`diffsol` then, behind its own feature).

The cost of chunked passes when the parameter count is known only at run
time, measured 2026-10-13 (`reserving-compartmental.md` decision 4, PR A):
32 tangents a pass, a gradient of Meyers' CRC costs 2.3 to 2.5 times the
hand-written one, within the 10 times accepted, and the in-house dual
matches every hand-written gradient here to 1e-12 relative. Revisit with reverse
mode (Burn, `ad_trait`, or `std::autodiff` once stable) for hierarchical
models with thousands of parameters.

### 16. The one-year view by PSIS reweighting (PR 8)

Built in Rust on `claude/bayes-one-year` (2026-10-14), for `BayesOdp`
first, as the user decided on 2026-10-13 (open decision 6 (b), with a
refit per flagged simulation as the fallback), with the Python and R
bindings in the same PR (decision 13); `Meyers`, `BayesBf` and
`Compartmental` follow, each in its own PR, through the same engine.

**API** (module `bayes`):

```rust
pub enum OneYearRefit { ReweightOnly, RefitFlagged /* default */, RefitAll }

pub struct BayesOneYear {
    pub k_threshold: f64,    // 0.7
    pub refit: OneYearRefit, // RefitFlagged (MomentMatchThenRefit since decision 19)
    pub n_sims: Option<usize>, // added in decision 19: None, the model's one-year default
}

impl BayesOdp {
    pub fn one_year(&self, triangle: &Triangle, column: &str, settings: &BayesOneYear)
        -> Result<BayesOneYearFit>;
    pub fn one_year_segments(&self, triangle: &Triangle, column: &str, settings: &BayesOneYear)
        -> Result<BayesOneYearFits>;
}

pub struct BayesOneYearFit<F = BayesOdpFit> {
    pub fit: F,                     // fit()'s posterior and lifetime reserves, draw for draw
    pub opening_ultimate: Vec<f64>, // latest + opening_reserve
    pub opening_reserve: Vec<f64>,  // posterior mean reserve per origin
    pub cdr: PredictiveDistribution, // dimension origin
    pub pareto_k: Vec<f64>,         // per simulation
    pub refitted: Vec<usize>,       // the simulations refitted with NUTS
    pub refit_warnings: usize,      // refits whose diagnostics warn
    pub k_threshold: f64,
}
// flagged_share(), to_long(), totals(); BayesOneYearFits<S = BayesOdpSegment>
// holds SegmentFits<BayesOneYearSegment<S>> and the joint cdr, with
// to_long(), totals() and segment(keys), as OneYearFits.
```

The settings are a separate argument, not fields of `BayesOdp`, so the
lifetime model is unchanged and the same struct serves every Bayesian
model. The number of simulations and the seed are the model's own
(`n_sims`, `seed`), as for the bootstraps' `one_year`.

**Method.** The year is the cells valued in the twelve months after the
segment's valuation, found as the bootstraps find them (`YearCells`, no
tail): one per open origin with an annual grain, up to four with a
quarterly one. Simulation `i`, on stream `i` of its seed:

1. picks a posterior draw `θ_j` uniformly and draws the year's
   increments `X` from it with the model's process error (Gamma, mean
   `μ = exp(η)`, variance `φ μ`, or none);
2. weights every posterior draw `θ_s` by `exp(Σ_year (X η_s − e^{η_s}) /
   φ)`, the quasi likelihood of the year with `φ`, `c0` and the priors
   left at their opening values, so the weighted draws target exactly the
   posterior the model would have on the extended triangle with the same
   plug-ins;
3. smooths the log weights with PSIS (relative efficiency 1, tail
   `M = min(0.2 S, 3 √S)`) and records `k̂`;
4. computes origin `w`'s claims development result

   ```text
   CDR_w = E[R_w | D0] − ( X_w + E[R'_w | D0, X] )
   ```

   where `R_w = Σ_{d > latest_w} μ_wd(θ)` is the expected run-off at the
   opening, `X_w` the origin's increments in the year, and
   `R'_w = Σ_{d > last_w} μ_wd(θ)` the expected run-off after the year;
   `E[· | D0]` is the plain mean over the posterior draws and
   `E[· | D0, X]` the smoothed, self-normalised weighted mean. So the
   opening and closing estimates are both posterior means of the expected
   remaining payments, the Bayesian counterpart of the bootstraps'
   re-reserving with the chain ladder (whose opening and closing
   estimates are the chain ladder's on the observed and the extended
   triangle). By the tower property its mean is zero, up to Monte Carlo
   and importance sampling error. An origin at the last age has no cell
   and a CDR of 0; one whose remaining cells all fall in the year has its
   whole run-off, `E[R_w | D0] − X_w`, as in the bootstraps.
5. When `k̂` is above `k_threshold` (or not finite) and `refit` is
   `RefitFlagged`, or always under `RefitAll` (except for a year with no
   information, below), `E[R'_w | D0, X]` comes
   instead from a NUTS run on the triangle extended by `X` (the same
   model, priors, `c0` and `φ`, the model's `sampler`, Newton started at
   the opening posterior mean, seed drawn on the simulation's own stream
   right after `X`, so every setting of `refit` sees the same years and
   the same refit seeds).

Under a lagging origin (one whose latest cell is before the valuation),
every cell up to the end of the year is drawn and conditioned on; the
ODP bootstrap leaves the cells before the valuation unobserved and
appends only the year's. Each cell's mean is a single `exp(η_wd)`, so a
sum over unobserved cells would need a likelihood of its own; this is
the simpler choice and the difference only touches triangles with a
ragged diagonal.

**Seeds.** The one-year simulations use the first value of stream
`u64::MAX` of the predictive simulations' derived seed (decision 5; the
segment's over segments, decision 11), which those simulations never
reach, so `one_year` keeps `fit`'s posterior and lifetime reserves bit
for bit (a test) and replays for a seed whatever the threads (a test).

**PSIS.** `prospicio_bayes::elpd`'s PSIS (Zhang and Stephens' GPD fit,
shrunk towards 0.5) was private to its crate, which the Models lane owns,
so PR 8's engine carried a copy. The user chose open decision 13 (a) on
2026-10-15: #273 made it public as `prospicio_bayes::elpd::psis(log_ratios,
r_eff)`, and PR 9 deleted the copy and calls it with `r_eff` 1. The two
were the same algorithm line for line (the tail length
`ceil(3 sqrt(S / r_eff))` is `ceil(3 sqrt(S))` at `r_eff` 1), and the
seed-pinned views of decision 17 are unchanged bit for bit. The copy's
test against `elpd::loo` went with it; #273 tests `psis` against `loo`.
A log ratio that is NaN or `+inf` is now an error of the simulation,
reported as a refit failure is, where the copy went on silently (a
unit test).

**The engine and other models.** `bayes/one_year.rs` knows no model: a
model implements `YearModel` (decision 17). `closing` takes the year,
because Meyers' cumulative models condition the remaining development
on the new cells; the ODP's does not, so it is computed once per
posterior draw. `BayesOneYearFit<F>` and `BayesOneYearFits<S>` are
generic over the lifetime fit, as `OneYearFit<B>` is over the bootstrap.

**Segments.** `one_year_segments` fits each segment as `fit_segments`
does (its own chains and simulation seeds, decision 11), runs each
segment's one-year simulations on its own seed, and joins them by row:
`Independent` as drawn, `RankCorrelation` by Iman–Conover on the
segments' total CDRs (the bootstraps' `reorder`), `Synchronized` an
`Error::Bayes` before any sampling. Unit tests: twin GenIns lines are
uncorrelated (rank correlation within `4.5/√n_sims`) and a target
Spearman 0.7 is met within 0.05. No tails: `BayesOdp` has none
(decision 12).

**Results** (`validation/tests/reserving_bayes_one_year.rs`; defaults,
seed 0, 10,000 simulations; the ODP bootstrap's one-year view with the
chain ladder, 50,000 simulations, seed 1; Merz–Wüthrich from `Mack`).
Standard deviation of the claims development result:

| RAA origin | opening reserve | reweight only | refit flagged | ODP bootstrap | Merz–Wüthrich |
|---|---|---|---|---|---|
| 1982 | 202 | 650 | 650 | 708 | 143 |
| 1983 | 716 | 961 | 967 | 1,033 | 570 |
| 1984 | 1,775 | 1,407 | 1,416 | 1,427 | 378 |
| 1985 | 2,906 | 1,540 | 1,554 | 1,551 | 1,299 |
| 1986 | 3,799 | 1,788 | 1,787 | 1,792 | 1,668 |
| 1987 | 5,595 | 2,147 | 2,146 | 2,146 | 1,186 |
| 1988 | 11,093 | 3,353 | 3,352 | 3,419 | 4,691 |
| 1989 | 10,864 | 4,614 | 4,614 | 4,565 | 4,707 |
| 1990 | 17,105 | 11,790 | 11,825 | 11,869 | 23,610 |
| total | 54,055 | 14,993 | 15,117 | 15,439 | 25,166 |

| GenIns origin | opening reserve | reweight only | refit flagged | ODP bootstrap | Merz–Wüthrich |
|---|---|---|---|---|---|
| 2002 | 98,613 | 113,080 | 113,080 | 115,069 | 71,835 |
| 2003 | 475,537 | 198,882 | 198,198 | 197,831 | 104,446 |
| 2004 | 716,185 | 174,527 | 173,647 | 177,064 | 78,739 |
| 2005 | 992,248 | 199,804 | 199,538 | 202,938 | 234,801 |
| 2006 | 1,427,226 | 235,636 | 232,324 | 229,568 | 318,170 |
| 2007 | 2,185,602 | 314,976 | 313,570 | 319,287 | 360,812 |
| 2008 | 3,948,315 | 607,024 | 600,873 | 589,044 | 629,452 |
| 2009 | 4,305,394 | 778,667 | 781,948 | 790,766 | 588,493 |
| 2010 | 4,692,920 | 1,745,247 | 1,750,157 | 1,778,970 | 1,029,850 |
| total | 18,842,040 | 2,351,573 | 2,367,332 | 2,427,934 | 1,774,014 |

* **Against the ODP bootstrap**: the same process and, under vague
  priors, about the same parameter error, so the two should agree, and
  they do within Monte Carlo error. Over seeds 0 to 79 (reweight only)
  the total's sd ratio averages 1.001 on RAA (seed-to-seed sd 0.026) and
  1.006 on GenIns (0.021), each origin's 0.98 to 1.02, except RAA's two
  oldest open origins: 1982 at 0.909 (sd 0.046) and 1983 at 0.947
  (0.028). They ride on the skewed `b[10]` and `b[9]` (decision 10), where
  the Bayesian lifetime reserve already differs from the bootstrap's;
  1982's one-year view is its whole run-off, so this is the lifetime
  model difference, not the reweighting.
* **Against Merz–Wüthrich**: not expected to agree. Merz–Wüthrich assumes
  Mack's process (variance proportional to the cumulative value), the
  ODP a constant `φ` times the mean, so per origin the ratio runs from
  0.50 (RAA 1990) to 4.5 (RAA 1982) and in total 0.60 (RAA) and 1.33
  (GenIns), as the ODP bootstrap's does (0.505, 5.06, 0.615 and 1.373,
  `knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`), where
  `MackBootstrap` reconciles the re-reserving with Merz–Wüthrich.
* **Flagged share**: `k̂` above 0.7 in 13.6% of the simulations on RAA
  (seed-to-seed sd 0.47 points) and 26.3% on GenIns (0.65 points); mean
  `k̂` 0.38 and 0.56, largest about 6. The year doubles the information
  on the newest origin's `a_n` and the last development effect, which
  rest on one cell each, so the posterior moves far for a few
  parameters, and over 4,000 draws the weights are heavy-tailed.
* **The importance sampling error**, from 2,000 simulations at seed 0
  run both ways on the same years (reweight only against refit all),
  each simulation's total CDR gap as a root mean square in units of the
  total's sd: RAA 0.029, 0.060, 0.113 and 0.301 for `k̂` in
  (−∞, 0.5], (0.5, 0.7], (0.7, 1] and above 1; GenIns 0.040, 0.059,
  0.100 and 0.198. Below 0.5 the gap is about the two estimators' own
  Monte Carlo error (the refit's posterior mean from 4,000 draws). The
  gap leans one way: the reweighting under-reacts to an outlying year,
  so reweighting every simulation gives the total a 1.0% (RAA) and 1.2%
  (GenIns) lower sd than refitting every one (15,054 against 15,199;
  2,399,585 against 2,428,321), and refitting only the flagged ones
  closes two thirds of that on RAA and a fifth on GenIns (15,149 and
  2,405,963; on one seed, so the split is rough). On
  the 10,000 simulations above, refitting the flagged raises the total's
  sd by 0.8% and 0.7%. The tail quantiles move by the same order (GenIns
  99.5% of the 2,000: 5,607,525 reweighted, 5,471,518 refitted where
  flagged, 5,459,202 all refitted).
* **Runtime** (release, 16 threads): reweighting only, about 0.6 s on
  RAA and 0.3 to 1.2 s on GenIns for 10,000 simulations, the fit
  included; with the default, the flagged simulations' refits cost about
  39 ms each, so 54 s on RAA (1,357 refits) and 104 s on GenIns (2,725).
  Refitting all 2,000 took 78 s and 69 s. No refit at the defaults had a
  diagnostic warning on either triangle at 10,000 simulations; at 2,000
  refitted RAA simulations, 2 did (the kind is not recorded; RAA's
  skewed `b[10]` makes a divergence the likely one). In a debug build the
  four unit tests (1,200 draws, at most 2,000 simulations, up to 40
  refits) take about 23 s in all, and the validation test below 285 s
  (9 s in release).

**Validation in CI** (release, `slow-bayes`, about 8 s): seed 0,
reweight only, 10,000 simulations, 28 rows. Each origin's and the
total's sd against the bootstrap's, as a ratio with target 1: RAA
tolerances 0.30 (1982), 0.18 (1983), 0.12, 0.12, 0.09, 0.10, 0.11, 0.12,
0.15 and 0.12 (total); GenIns 0.12 (2002), 0.08 to 0.11 for the rest and
0.10 for the total. The seed-to-seed sd of a ratio is 1.3% to 3.4% (4.9%
for RAA 1982), so each tolerance is at least the row's mean offset from 1
plus 4 such sds over seeds 0 to 79 (a first sizing from seeds 0 to 39
failed RAA 1985 at a fresh seed, 1.088 against 0.09, and was widened);
the two RAA rows with the skewed parameters carry their offset honestly
rather than a target moved to match it. The total's mean CDR within 5
Monte Carlo standard errors of 0 (seed-to-seed sd of the z score 0.84 to
1.01). The flagged share within 0.03 of 0.136 (RAA) and 0.035 of 0.264
(GenIns), 4 sds 0.022 and 0.030. On 100 simulations refitted all, the
root mean square gap below the threshold under 0.08 sd (RAA) and 0.10
(GenIns), above it under 0.7 and 0.5 (mean plus 4 sds over seeds 0 to
19: about 0.05, 0.08, 0.59 and 0.25; the flagged rows rest on about 13
and 26 simulations each, so they are loose). At seed 0 every row uses at
most 0.47 of its tolerance; over seeds 0 to 79 at most 0.79 (RAA 1983).
Unit tests in `bayes/odp.rs`: the posterior and lifetime
reserves are `fit`'s draw for draw, the oldest origin's CDR is 0, the
second oldest's sd is its lifetime reserve's within 10%, every origin's
mean CDR is within 4 standard errors of 0, the total is narrower than
the lifetime total, replay whatever the threads, `RefitFlagged`'s rows
are `RefitAll`'s for the flagged simulations and `ReweightOnly`'s for the
others, a fully developed triangle, segments and the errors.

**A year with no information.** When the coming year has no cells (every
origin at the last age, for example a closed run-off line among
segments) or every log ratio is equal, the weights are equal and exact:
`k̂` is set to 0 and the simulation is never refitted, whatever `refit`
says. PSIS itself would give an infinite `k̂` for want of a tail, and
under the default every simulation would run a NUTS refit of an
unchanged triangle and report a flagged share of 1. The CDR is then 0.

**Defaults.** `k_threshold` 0.7: the threshold of Vehtari et al. (2024)
for a few thousand draws (`elpd::loo` uses `min(1 − 1/log10 S, 0.7)`, 0.7
at 4,000). `RefitFlagged`: the user's decision, and the one setting
whose tail is not biased by the reweighting's under-reaction. Its cost,
a minute or two per triangle at the defaults, is the price; a user who
wants a quick look sets `ReweightOnly` (about 1% low on the sd here) or
fewer simulations. Whether a cheaper first fallback should come before
the refit was open decision 12; on 2026-10-15 the user kept
`RefitFlagged` as the default for now and asked for moment matching, (b),
to be tried (next, decision 17).

### 17. One one-year view for every Bayesian model (PR 9)

Built in Rust, Python and R on `claude/bayes-one-year-generic`
(2026-10-15), stacked on #273. The user decided on 2026-10-15 that every
Bayesian model gets the one-year view, made automatic (open decision
14): one interface every Bayesian fit implements, one generic view
written once over it, and a conformance suite run on every implementor,
which a new Bayesian model must pass before it ships (AGENTS.md,
"Checks").

**The interface** (`bayes/one_year.rs`, crate-private, since the models
and their posteriors are):

```rust
pub(crate) trait YearModel: Sync {
    fn n_origins(&self) -> usize;
    fn n_year(&self) -> usize;
    /// The year's values given a posterior draw, with process error.
    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]);
    /// The log importance ratio: the change the year makes to log_posterior.
    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64;
    fn paid(&self, year: &[f64], paid: &mut [f64]);
    /// Each origin's expected reserve given theta, at the opening and after the year.
    fn opening(&self, theta: &[f64], reserve: &mut [f64]);
    fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]);
    fn closing_uses_year(&self) -> bool { true }
    /// NUTS on the triangle extended by the year.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)>;
    /// The log posterior of a draw (as stored) given the triangle extended
    /// by `year` (empty: the observed triangle), for moment matching and
    /// the conformance suite.
    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64;
}

pub(crate) fn simulate<M: YearModel>(model: &M, posterior: &PosteriorDraws,
    settings: &BayesOneYear, n_sims: usize, seed: u64) -> Result<YearDraws>;
pub(crate) fn assemble<F: ReserveFit>(fit: F, year: YearDraws, segment: &Segment,
    settings: &BayesOneYear, provenance: Provenance) -> Result<BayesOneYearFit<F>>;
```

The engine takes the closing reserve per posterior draw from the model
and weights it itself (self-normalised smoothed weights, or the refit's
plain mean), so a model never sees the weights; `assemble` builds the
`BayesOneYearFit` (the CDR by origin, the opening ultimate and reserve,
`k̂`, the refits) once for every model. A model of over-dispersed
Poisson increments with a log-linear mean implements the smaller
`OdpCells` (`eta(theta, w, d)`, `phi`, the log posterior and a refit
with extra cells) and gets its `YearModel` from the shared `OdpYear`,
which finds the year's cells (`YearCells`, no tail), draws them with the
process error and scores them: `BayesOdp` and `BayesBf` are each about
50 lines of `OdpCells`. Moment matching (open decision 12 (b)) is the
reason for `log_posterior`: it moves the draws by an affine map and
needs the target's density at the moved draws. It is built in decision
18 (`bayes/moment_match.rs`, the opt-in `MomentMatchThenRefit`), which
evaluates `log_posterior` at the moved draws.

**The stored scale is unconstrained.** `theta` is a draw on the scale
`PosteriorDraws` stores it, and moment matching's affine maps act on
that scale, so it must be unconstrained (every point of `R^p` a valid
parameter) and `log_posterior` must be the density of the draws on it
(with a Jacobian when the scale is not an affine map of the sampling
scale). On a constrained scale a shift or a scaling can push a standard
deviation below 0 or a correlation past 1; the ratio is then NaN, taken
as a zero weight, and matching quietly falls back to refits. A Jacobian
left out cancels in conformance check 1 (the year's change to the log
posterior), so the unit suite has a fifth check, the score identity:
over the posterior draws the finite-difference score of
`log_posterior(theta, [])` has mean zero, every parameter within 6
standard errors. `BayesOdp` and `BayesBf` pass with the largest
`|z|` 1.1 to 1.7; a unit test shows a log posterior that leaves out
the Jacobian of draws stored as `exp` failing it at `z` 18.

**Carried with each posterior draw.** Reweighting can update only what
the posterior draws hold. Anything a model's predictive step draws afresh
per simulation that the year's cells would inform must therefore be part
of `theta`, drawn once per posterior draw and stored with it (augmented
draws), not drawn inside `draw_year`; only process error that the year
does not inform may be. The models, checked:

* `BayesOdp`: nothing to carry. Every origin and every age has an
  observed cell (`prepare` requires it), so `c`, each `a_w` and each
  `b_d` is in `theta`; the process error is per cell and, given
  `theta`, independent of the other cells; `phi` and `c0` are plug-ins
  held fixed (decision 16).
* `BayesBf`: nothing to carry, for the same reasons: each `alpha_w` and
  `beta_d` is in `theta`, the prior ultimates and `phi` are fixed. Its
  draws are stored as `(alpha, beta)` and sampled as `(u, beta)` with
  `u_w = alpha_w - log m_w`, a shift, so the log posterior needs no
  Jacobian and a refit starts Newton at the opening mean moved to `u`.
* `Meyers` (built in PR 10, decision 19, as described there; the CAY
  cells this paragraph worried about are always observed or in the year):
  every parameter is in `theta`, but CRC and
  CAY's means hold `rho` times the previous origin's residual at the same
  age, so a year cell's residual informs the next origin's cells at that
  age given `theta`: `closing` must condition on the year
  (`closing_uses_year`), and the year's log likelihood is the
  conditional density of the year's cells given the observed ones. CAY's
  residual chain also runs through unobserved cells of earlier origins,
  whose residuals `draw_paths` draws per simulation; where such a cell
  precedes a year cell, its residual is informed by the year and must be
  carried with the draw (or integrated out in the likelihood). The
  ultimates' lognormal noise past the last age is process error. Its
  stored draws are natural-scale values (`rho = tanh(v)` for CAY, each
  `sigma_d` the square root of variances built from logistics), not the
  unconstrained sampling scale, so it must store or expose that scale
  (with `log_posterior` on it) before it implements `YearModel`.
* `Compartmental` (not in this PR): `origin_values` draws fresh
  `z1`, `z2` (and a rate's `z`) for an origin with no observed cell, and
  `dev_multipliers` fresh effects for an age with no observed cell. A
  year that adds the first cell of such an origin or age informs them,
  so both must become part of the stored draw before its `YearModel`.
  The per-cell normal or log-normal noise is process error. Its stored
  draws are natural-scale values (exponentials of the sampled ones, and
  rates), so, as for `Meyers`, it must store or expose its unconstrained
  sampling scale before it implements `YearModel`.

`Meyers` and `Compartmental` waited on the user for the defaults of
their refit cost (a Meyers or compartmental NUTS run takes seconds, not
39 ms, so `RefitFlagged` at 10,000 simulations would take hours). The
user decided on 2026-10-15 that moment matching is their default, as
every model's (decision 18); `Meyers` is built in decision 19,
`Compartmental` is next.

**`BayesBf::one_year`**: `(triangle, column, exposure, settings)`,
returning `BayesOneYearFit<BayesBfFit>`, with `fit`'s posterior and
lifetime reserves draw for draw (a unit test) and the one-year
simulations on the seed `BayesOdp` uses (`one_year_seed` of the
predictive stream). Its tables add the fit's `exposure` and
`prior_ultimate` (`BayesOneYearFit`'s long tables now forward the
lifetime fit's origin columns). No segments: `BayesBf` has no
`fit_segments`, so there is no `one_year_segments` either. Bindings:
Python `BayesBf.one_year(triangle, column, exposure, k_threshold=0.7,
refit="refit_flagged")` and R `bayes_bf_one_year()` with `bayes_bf()`'s
arguments then `k_threshold` and `refit`, returning the same
`BayesOneYearFit` / `bayes_one_year_fit` as `BayesOdp`, whose `fit` is
now the model's own lifetime fit (`BayesBfFit` / `bayes_bf_fit`).

**Results** (`validation/tests/reserving_bayes_bf_one_year.rs`; GenIns
with premium, apriori 0.5, defaults otherwise, `ReweightOnly`, 10,000
simulations; about 2.4 s in release). Each origin's and the total's sd of
the CDR as a ratio, mean and seed-to-seed sd over seeds 0 to 39:

| row | target | origins 1 to 9 (mean) | total, mean (sd) | tolerances |
|---|---|---|---|---|
| flat prior (`cv` infinite) over `BayesOdp`'s view | 1 | 0.992 to 1.008 | 1.010 (0.027) | 0.09 to 0.15 |
| tight prior (`cv` 0.001, `development_sd` 100, no process) over the closed form | 1 | 0.999 to 1.009 | 1.008 (0.025) | 0.07 to 0.11 |
| total sd at `cv` 0.2 over the flat prior's | 0.697 | | 0.697 (0.018) | 0.08 |
| total sd at `cv` 0.05 over the flat prior's | 0.502 | | 0.502 (0.014) | 0.06 |

The closed form is the strong-prior limit's (EVW, sec. 3.3): each `mu_w`
at its prior `m_w`, each `gamma_d` independently `Gamma(S_d / phi, M_d /
phi)`. The year adds `A_d gamma_d` to column `d` (`A_d` the prior
ultimates of the origins whose year cell is there), which moves the
posterior mean of `gamma_d` to `(S_d + A_d gamma_d) / (M_d + A_d)`, so
origin `w`'s CDR is linear in the `gamma_d`, with coefficient `-m_w` at
its year's column and `-m_w A_d / (M_d + A_d)` after it, and its variance
is the sum of the squared coefficients times `phi S_d / M_d^2`; the mean
is 0. The reweighting meets it without refits (this conjugate posterior
moves little per year). Under a flat prior the view is the Bayesian
chain ladder's, as the lifetime reserves are (decision 9); the prior
narrows the view, by 30% at `cv` 0.2 and 50% at 0.05, because the year
then moves the development pattern and hardly the origins' levels. The
total's mean CDR is within 5 standard errors of 0 under both limits
(z scores' seed-to-seed sd 1.03 and 1.13). Every tolerance is the row's
mean offset plus at least 4 seed-to-seed sds; seed 0 uses at most 0.58
of any.

**The conformance suite.** No external reference; each property holds
for any correct model, so a model that fails one has a bug.

* Unit half (`one_year::conformance::check`, debug, on GenIns, run in
  `bayes/odp.rs` and `bayes/bf.rs` with `cv` 0.2 and flat): the year's
  log likelihood equals the change it makes to `log_posterior`, for years
  drawn from 8 posterior draws and compared across 16 others (to 1e-10
  relative; a 1% error in the scale fails it); the total CDR's mean is
  within 4.5 standard errors of 0 over 2,000 reweighted simulations; a
  seed replays bit for bit on one thread against many and another seed
  differs; a fully developed square gives a zero CDR, `k̂` 0 and no
  refit under every `refit`; the score of `log_posterior` has mean zero
  over the posterior draws (every parameter within 6 standard errors).
  About 2 s for the three in a debug build.
* Release half (`validation/tests/reserving_bayes_one_year_conformance.rs`,
  about 30 s, on `BayesOdp` RAA and GenIns and `BayesBf` GenIns at
  `cv` 0.2): the mean CDR within 5 standard errors over 10,000
  simulations; the opening reserve's total within 5 standard errors of
  the lifetime reserve's mean; replay; on 300 simulations, `RefitAll`
  sees `ReweightOnly`'s years and `RefitFlagged` gives `RefitAll`'s rows
  for the flagged and `ReweightOnly`'s for the others; the reweighting's
  root mean square gap to the refits below 0.08 sd where `k̂ <= 0.7` and
  0.5 above; and the square without a year under every `refit`. Over
  seeds 0 to 19: unflagged gaps 0.035 (RAA), 0.044 (GenIns) and 0.056
  (`BayesBf`), sds 0.004; flagged 0.235, 0.160 and 0.199, sds 0.055,
  0.018 and 0.042 (about 40, 80 and 38 of the 300 simulations flagged);
  every seed passes, the largest values 2.35 and 2.18 for the z scores
  and 0.063 and 0.380 for the gaps.

**Bit for bit.** Before the refactor a unit test pinned `BayesOdp`'s
views (an FNV hash of every CDR draw, of `k̂` and of the opening reserve,
and the first values) for `ReweightOnly` on GenIns (2,000 simulations),
`RefitAll` on 40 simulations and `one_year_segments` on two lines. The
generic engine and the public PSIS reproduce all three exactly, and
`reserving_bayes_one_year.rs` passes unchanged.

### 18. Moment matching before refitting (PR 9)

Built in Rust, Python and R on `claude/bayes-one-year-generic`
(2026-10-15), as open decision 12 (b): the user kept `RefitFlagged` as
the default and asked for moment matching to be tried and measured.

**The method** (`bayes/moment_match.rs`), the general form of Paananen,
Piironen, Bürkner and Vehtari (2021, "Implicitly adaptive importance
sampling", *Statistics and Computing* 31, 16) as R's
`loo::loo_moment_match` implements it (read from `loo`'s source, not
from memory):

1. For a flagged simulation, start from the opening draws `θ_s`, their
   smoothed normalised weights and `k̂`.
2. Try, in order: a shift to the weighted mean; a shift and a scaling
   to the weighted mean and marginal variances (`loo`'s `S / (S - 1)`
   correction); a shift and the map `L_w L⁻¹` to the weighted mean and
   covariance (`L` and `L_w` the Cholesky factors of the plain and the
   weighted covariance, `cov.wt`'s `1 - Σ w²` divisor; the identity when
   either is not positive definite). Each moves the current draws; the
   log ratio of a moved draw `T(θ_s)` is `log p(T(θ_s) | y, year) - log
   p(θ_s | y)`, the year's target taken as `log_posterior(x, [])` plus
   `year_log_likelihood(x, year)` (equal up to a constant, which the
   conformance suite checks). The map is affine, so the proposal's
   Jacobian is a constant and drops out of the self-normalised weights,
   as in `loo`.
3. Keep a step only if PSIS gives a lower `k̂`; after a kept step start
   again at the shift; stop when none lowers it, when `k̂` is at most the
   threshold, or after 30 kept steps (`loo`'s `max_iters`). A ratio PSIS
   cannot take (`+inf`) stops matching; a NaN ratio is a zero weight.
4. If `k̂` came to at most the threshold, the closing reserve is `loo`'s
   split estimate (its default): the first half of the draws moved by
   the whole map `T`, the second half left, each weighted by the target
   over the mixture `½ p(x | y) + ½ p(T⁻¹(x) | y) / |det T|` (here the
   Jacobian matters) and Pareto-smoothed, and the closing reserve is the
   weighted mean of the model's closing reserve at those draws.
   Otherwise the simulation is refitted with NUTS on the seed every
   setting draws, so its row is `RefitAll`'s.

All maps are lower triangular (diagonal or `L_w L⁻¹`), so `T⁻¹` is a
forward substitution and `log |det T|` the sum of the log diagonal. The
opening draws' log posterior is computed once per segment; each step
costs one pass of the log posterior over the 4,000 draws.
`OdpCells::log_posterior` skips cloning the model when the year is empty.

**Settings.** `OneYearRefit::MomentMatchThenRefit`, built opt-in; the
user made it the default on 2026-10-15 (open decision 12), in PR 10:
`OneYearRefit::default()`, `BayesOneYear::default()` and the bindings'
`refit` argument, so a default view of `BayesOdp` or `BayesBf` takes
seconds where `RefitFlagged` took minutes. The years and their `k̂` do
not change; a rescued simulation's row does, by the matching's gap to its
refit. `RefitFlagged` stays available. `pareto_k` stays each simulation's `k̂` before
matching, so `flagged_share` means the same under every setting; the new
`moment_matched` (on `BayesOneYearFit` and `BayesOneYearSegment`) lists
the simulations matching rescued, and each flagged simulation is in it
or in `refitted`. Python `refit="moment_match_then_refit"` and
`.moment_matched`; R the same `refit` value and `@moment_matched`
(1-based). Moment matching draws no random numbers, so the years and the
other settings' views are unchanged bit for bit (the pinned test of
decision 17 passes on the Windows machine that captured it; its hashes
are platform-specific, since another libm changes the NUTS draws, so it
is ignored in CI and run with `--ignored` as a local refactor guard).

**Results** (`validation/tests/reserving_bayes_one_year_moment_match.rs`,
release, 16 threads; `BayesOdp` on RAA and GenIns and `BayesBf` on GenIns
with premium, apriori 0.5 and cv 0.2; 2,000 simulations under every
setting on the same years; mean and seed-to-seed sd over seeds 0 to 19):

| | `BayesOdp` RAA | `BayesOdp` GenIns | `BayesBf` GenIns |
|---|---|---|---|
| flagged (`k̂` > 0.7) | 268 (14) | 531 (15) | 255 (17) |
| rescued by matching | 100% | 99.99% (min 99.8%) | 100% |
| rms gap to the refit, rescued, matched (sd of the total) | 0.058 (0.008) | 0.036 (0.007) | 0.038 (0.007) |
| the same simulations reweighted only | 0.240 (0.045) | 0.155 (0.012) | 0.186 (0.021) |
| total sd over `RefitAll`'s: `MomentMatchThenRefit` | 0.999 (0.006) | 1.001 (0.008) | 0.999 (0.008) |
| `RefitFlagged` | 0.999 (0.005) | 1.000 (0.005) | 0.999 (0.008) |
| `ReweightOnly` | 1.004 (0.013) | 1.004 (0.009) | 1.004 (0.011) |
| wall time, `ReweightOnly` | 0.23 s | 0.23 s | 0.22 s |
| wall time, `RefitFlagged` | 7.7 s | 13.6 s | 6.9 s |
| wall time, `MomentMatchThenRefit` | 0.50 s | 0.73 s | 0.48 s |

The rescued simulations' gap to their refits is the size of the
unflagged simulations' gap between reweighting and refitting (0.035 to
0.056 sd, decision 17), so matching makes a flagged simulation as good as
an unflagged one; their spread is the refits' (sd ratio 1.000 to 1.002)
and their mean gap within 0.006 sd of zero. Matching costs about 1 ms of
wall time per flagged simulation against 25 to 28 ms for a refit, so a
run costs two to three times `ReweightOnly` and a fifteenth to a nineteenth
of `RefitFlagged`. The total's sd is within Monte Carlo error of
`RefitAll`'s under every setting at 2,000 simulations; reweighting alone
is noisier per simulation, not visibly biased at this size. The wall
times were taken while other builds shared the machine, so they are
rough; their ratios are steady across seeds.

The test, at 1,000 simulations and seed 0, also checks exactly that
matching sees `ReweightOnly`'s `k̂`, that each flagged simulation is
matched or refitted but not both, and that a refitted row is
`RefitFlagged`'s and an unflagged one `ReweightOnly`'s; the release
conformance suite checks the same for every model on 300 simulations
against `RefitAll`, that matching matches at least one of them and
replays bit for bit on one thread against the default pool (its own
parallel steps nest inside the simulations'; for Meyers' models, whose
refits on one thread run one after another, the first 20 simulations of
60), and a triangle without a year matches and refits
nothing. A unit test on a conjugate normal model (six parameters, `k̂`
about 5) shows matching landing within 0.05 of the exact posterior mean
on every parameter (0.029 at its seed) where reweighting is 0.25 off, and another that each step's moved draws
have the weighted moments. Bounds: `rescued` at least 0.99 and the gap
below 0.11 (RAA), 0.07 (GenIns, `BayesBf`), each at least the sweep's
mean plus 4 sds.

**Recommendation.** Matching is as accurate as refitting on these
triangles and about 25 times cheaper per flagged simulation, and it
leaves the refit as the fallback, so it could be the default. Decided
2026-10-15 by the user: it is (PR 10). For `Meyers` the posteriors are
less close to normal under an affine map: matching rescues 83% to 90% of
the flagged simulations, within about 0.08 sd of their refits (decision
19); the compartmental models are measured when their `YearModel` exists.

### 19. Meyers' one-year view (PR 10)

Built in Rust, Python and R on `claude/bayes-one-year-meyers`
(2026-10-15), stacked on PR 9 (#274) and the CI slow-tests PR (#279). The
user decided on 2026-10-15 that moment matching becomes the default
fallback (open decision 12, decision 18) and that `Meyers` and
`Compartmental` get their one-year views through `YearModel` with it as
their default too (open decision 14); this PR does `Meyers` (CRC, CSR
and CAY), `Compartmental` follows in PR 11 (decision 20).

**The scale.** `MeyersSegment` keeps a second copy of its posterior, on
the unconstrained scale `Model::log_posterior` takes
(`theta = (logelr, alpha, beta, gamma | v, u_1..u_m)`, `rho = tanh(v)`,
each step of `sigma_d^2` a logistic of `u_d`), which is
`whitened.theta(z)`, an affine map of the scale NUTS samples, so the log
posterior on it needs no Jacobian. `sample` returns both copies; the
natural-scale posterior of `MeyersFit` and `MeyersSegment` is unchanged,
bit for bit (it is now computed from the stored unconstrained draws by the
same `natural`). `MeyersFit` does not keep the second copy, a
`MeyersSegment` does (the views over segments need it). The unconstrained
names are `v` and `u[d]`. The conformance suite's score check passes on
all three models.

**The year** (`bayes/meyers/year.rs`, `MeyersYear`). Its values are the
cumulative values `C_wd` of the cells valued in the twelve months after
the valuation (`YearCells`, no tail), origin by origin. Given a draw,
`log C_wd` is normal with mean `mu_wd` and variance `sigma_d^2`, the
cells independent in CRC and CSR. The year's log likelihood is the sum of
those normal densities without `-log C` (constant in the draw), which is
exactly the change the year's cells make to `Model::log_posterior` (a
unit test, and conformance check 1). The opening reserve given a draw is
each open origin's expected ultimate less its latest value,
`exp(mu_wm + sigma_m^2 / 2) - C_w`, whose posterior mean is the lifetime
reserve's (checked against the lifetime simulation at a fixed draw, and
conformance check 2 of the release half). The closing reserve is the
same after the year, from the year's last value of the origin; an origin
the year takes to the oldest age has none. For CRC and CSR the expected
ultimate does not depend on the year given the draw, so a simulation's
claims development result is the change in its posterior mean,
`E[U | D0] - E[U | D0, C]` (a unit test checks the year's payment and
closing reserve add up to the opening reserve draw by draw).

**CAY.** Each mean takes `rho` times the previous origin's residual
(innovation) at the same age, so each age's cells are a moving average of
order 1 down the origins. A year cell's residual follows from the draw,
the observed cells and the year's earlier cells, since its previous
origin's cell at that age is observed or in the year (an older origin's
cell is valued twelve months earlier; `MeyersYear::new` keeps the check
as a guard). The year's log likelihood walks each age's chain of
residuals from the first origin, observed cells and then the year's, and
`draw_year` draws the year's residuals along the same chains. At the
oldest age the expected ultimate takes `rho` times the previous origin's
residual where that origin is there by the end of the year (observed or
in the year), and integrates it out otherwise, which makes the variance
`(1 + rho^2) sigma_m^2`, as the lifetime simulation (`draw_ultimates`)
implies. Nothing the year informs is drawn afresh per simulation: every
parameter is in the draw, the observed and the year's residuals follow
from it, and the residual that is integrated out belongs to a cell the
year does not reach (decision 17, "carried with each posterior draw").

**API.** `Meyers::one_year(triangle, column, premium, settings)` returns
`BayesOneYearFit<MeyersFit>` with `fit`'s posterior and lifetime
predictive draw for draw (a unit test) and the simulations on the seed
the other models use (`one_year_seed` of the predictive stream);
`Meyers::one_year_segments` returns `BayesOneYearFits<MeyersSegment>`,
each segment on its own seeds as in `fit_segments`, `Independent` or
`RankCorrelation` (`Synchronized` an error), as `BayesOdp`'s. A refit
samples the extended model as `fit` does (least-squares start, pilot run,
the same sampler). Bindings: Python `Meyers.one_year(triangle, column,
premium, k_threshold=0.7, refit="moment_match_then_refit")` and
`Meyers.one_year_segments`, R `meyers_one_year()` and
`meyers_one_year_segments()` with `meyers()`'s arguments, returning
`BayesOneYearFit` / `bayes_one_year_fit` (its `fit` a `MeyersFit` /
`meyers_fit`) and `BayesOneYearFits` / `bayes_one_year_fits`, which now
holds either model's segments: `premium` for Meyers, `scale` for the ODP
(each an error for the other).

**Nearly every simulation is flagged.** On group 353 (CRC and CSR on
paid, CAY on incurred, the defaults) 98.5% to 99.7% of the simulations
have `k̂` above 0.7, against 14% to 26% for `BayesOdp` on RAA and
GenIns. Each year cell is the latest value of its origin at an older age,
whose `sigma_d` is the smallest the origin has seen (Meyers' variances
fall with age, to about 0.02 at the late ages on paid), so it moves its
origin's level `alpha_w` far relative to its posterior spread; nine such
moves in one year leave almost no posterior draw with a large weight.
Reweighting alone is then 0.19 to 0.21 sd off the refits per simulation
(below), so without moment matching the view would need a refit for
nearly every simulation.

**Results** (`validation/tests/reserving_meyers_one_year.rs`, release,
16 threads; the defaults; 500 simulations under `ReweightOnly` and the
default on the same years, the first 60 under `RefitAll`; mean and
seed-to-seed sd over seeds 0 to 19):

| | CRC paid | CSR paid | CAY incurred |
|---|---|---|---|
| flagged (`k̂` > 0.7) | 99.3% (0.3%) | 98.5% (0.5%) | 99.7% (0.3%) |
| rescued by matching | 90.4% (3.4%) | 90.0% (3.9%) | 83.3% (5.2%) |
| rms gap to the refit, rescued, matched (sd of the total) | 0.080 (0.029) | 0.075 (0.019) | 0.084 (0.027) |
| the same simulations reweighted only | 0.212 (0.059) | 0.191 (0.037) | 0.211 (0.043) |
| rescued simulations' sd over their refits' sd, matched | 0.993 (0.027) | 0.994 (0.020) | 0.998 (0.024) |
| the same, reweighted only | 1.003 (0.066) | 1.016 (0.050) | 1.039 (0.071) |
| total CDR sd over the lifetime reserve sd | 0.769 (0.081) | 0.798 (0.071) | 0.825 (0.083) |

Matching rescues five in six to nine in ten of the flagged simulations
and brings them about 2.5 times closer to their refits than reweighting
alone; at every seed the matched gap is below the reweighted one (the
largest ratio 0.88). The matched gap is about twice the ODP models'
(0.036 to 0.058, decision 18), so a rescued Meyers simulation is less
exact than a rescued ODP one, though the rescued simulations' spread is
the refits' (sd ratio within 0.01 of 1). The 10% to 17% that matching
cannot rescue are refitted under the default.

Against the ODP bootstrap (`OdpBootstrap::one_year`, chain ladder) on the
same paid triangle: the models differ (log-normal cumulative values with
a level from premium, against over-dispersed Poisson increments with a
free level per origin), so their sds differ, and only the share of the
lifetime uncertainty that emerges in the first year is comparable. The
bootstrap's one-year sd is 0.803 (sd 0.008) of its lifetime sd and CRC's
0.769 (0.081), a ratio of 0.957 (0.097): the same order, as expected,
and the test bounds the ratio between 0.56 and 1.35.

The release conformance suite runs on all three (decision 17;
`validation/tests/reserving_bayes_one_year_conformance.rs`, with 60 refit
simulations and 4 chains of 300 warm-up and 300 kept draws, about a sixth
of the default's cost, for CI's sake): over seeds 0 to 19 the mean and
opening z scores stay within 2.8 and 2.5, and the reweighted gap of the
flagged simulations is 0.33 to 0.36 (sd 0.05 to 0.07, largest 0.52),
for which Meyers' models take a bound of 0.65 (the ODP models' 0.5);
with almost no unflagged simulation, the unflagged gap is checked only
over at least 10 of them, so not for Meyers.

**Cost.** At 10,000 simulations on group 353 (release, 16 threads, seed 0,
`validation/tests/reserving_meyers_one_year.rs`, ignored `runtime`), the
default takes 249 s (CRC), 411 s (CSR) and 396 s (CAY): matching rescues
9,014, 8,872 and 8,618 simulations at about 10 ms of wall time each, and
the 903, 975 and 1,347 it cannot rescue are refitted at about 0.17 to
0.27 s of wall time each, which is most of the run. `RefitFlagged` takes 171 s,
276 s and 260 s for 1,000 simulations (it refits 99% of them), so about
28, 46 and 43 minutes for 10,000: the default is about 6.7 times faster.
`ReweightOnly` takes 3 to 8 s and is 0.19 to 0.21 sd off per simulation.
Against the ODP models (decision 18: a few seconds by default) a Meyers
view is still minutes, because nearly every simulation is flagged and one
in six to ten needs a refit; fewer simulations or a lighter sampler for
the refits are the levers. The user chose fewer simulations on
2026-10-16 (below).

**Fewer one-year simulations by default** (the user's decision of
2026-10-16). `BayesOneYear` gains `n_sims: Option<usize>`, the number of
one-year simulations, `None` by default; `Some(0)` is an
`InvalidSetting`. `None` takes the model's one-year default:
`BayesOdp` and `BayesBf` keep their own `n_sims` (10,000 by default, so
their views are unchanged bit for bit), and `Meyers` takes its `n_sims`
capped at `Meyers::ONE_YEAR_SIMS`, 2,000, so a default Meyers view runs
2,000 simulations instead of 10,000 while its lifetime view keeps 10,000.
A model whose `n_sims` is below 2,000 runs that many, as before, and
`Some(n)` runs `n` for any model. Simulation `i` stays on stream `i`, so
any count gives the first simulations of a larger one (unit tests in
Rust, Python and R). The time is proportional to the count (each
simulation's matching or refit costs the same), so the default views
take about a fifth of the 10,000-simulation times above: about 50 s
(CRC), 82 s (CSR) and 79 s (CAY) on 16 threads, estimated from those
runs. The Monte Carlo error of the total's sd at 2,000 simulations is
about 1.6% (sd / sqrt(2 n)), against 0.7% at 10,000. The provenance of
every Bayesian one-year view records the count as `one_year_sims`.
Bindings: Python's `one_year` and `one_year_segments` take `n_sims=None`
(`BayesOdp`, `BayesBf`, `Meyers`); R's `bayes_odp_one_year()`,
`bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
`meyers_one_year()` and `meyers_one_year_segments()` take
`one_year_sims = NULL`, since their `n_sims` is the model's. The same
decision gives `Compartmental` (models 1 and 2) the same 2,000 default
(`Compartmental::ONE_YEAR_SIMS`, PR 11, decision 20).

**CI: a smoke run on pull requests** (the user's decision of
2026-10-16). The Meyers arms made the bayes job the PR's slowest check
(19 min 42 s on CI before the conformance replay cut). A pull request now
runs a smoke version: CRC alone in
`reserving_meyers_one_year.rs` and among the Meyers arms of the
conformance suite, at the same sizes and bounds (seed 0, calibrated by
the sweeps above). The weekly and manual (`workflow_dispatch`) runs set
`RISK_RS_FULL_BAYES` (`prospicio_validation::full_bayes`) and run CSR
and CAY too. The gate is an environment variable, not a feature, so one
build compiles and clippy lints every arm; only what runs depends on it.
The compartmental one-year validation takes the same gate (decision 20:
model 1 without its refit comparison on a pull request, model 2 only in
the full run). Since the user's decision of 2026-10-19 a pull request's
conformance suite runs `BayesOdp` and `BayesBf` alone: Meyers' CRC and
compartmental model 1, whose refits took most of its 11 minutes on CI,
join only in the full run, and their own one-year files still check them
on every pull request (`knowledge/environment/ci-runtime.md`).
Measured locally (release, 16 threads, 2026-10-16):
`reserving_meyers_one_year` takes 28 s in the smoke run and 114 s in the
full one, the conformance suite 54 s and 102 s, and the Bayesian parity
files together 230 s and 390 s. On CI, whose 4-core runner took about
2.5 times the local time for these files before, that is about 10
minutes of parity tests on a pull request and about 16 in the weekly
run, plus about 2 minutes of unit tests and the build.

Every tolerance in the validation file is the sweep's mean minus (floors)
or plus (bounds) at least 4 seed-to-seed sds, and every seed passes.

### 20. The compartmental models' one-year view (PR 11)

Built in Rust, Python and R on `claude/bayes-one-year-compartmental`
(2026-10-15), stacked on PR 10 (#280). The user's decisions of
2026-10-15 (open decisions 12 and 14) give `Compartmental` (models 1 and
2, `reserving-compartmental.md`) the one-year view through `YearModel`
with moment matching as its default; open decision 15 below, what the
refits of model 2 cost, the user decided on 2026-10-16 (the last
paragraphs of this decision).

**The scale and the fresh effects** (decision 17). `CompartmentalFit`
keeps a second copy of its posterior on the unconstrained scale of its
log density (`whitened.theta(z)`, as `Meyers`), beside the natural-scale
`posterior`, which is unchanged bit for bit (hashes of the posterior, the
reserves and the paid cells of model 1 and model 2 fits match the base
branch). The fit draws an origin with no observed cell (the papers' 1997)
and, with development effects (model 2), an age with none from the
population in each simulation; the year's cells inform them, so the view
takes the posterior of the same model with a level for every origin of
the premium column and every age of the square (`Model::full`). Those
levels enter only through their N(0, 1) priors, so that posterior is the
fit's with independent standard normal `z` appended to each draw; draw
`d` takes its `z` from stream `d` of a seed derived from the view's (see
"Seeds" in the module docs). The opening reserve's posterior mean is the
fit's mean reserve within Monte Carlo error (a unit test).

**Every effect centred.** On the `z` scale moment matching rescued only
about 1 in 10 of model 1's flagged simulations on company 337: the year's
cells pin an origin's effect `u = τ z`, which on the `z` scale is a
curved ridge (`z = u / τ`) that affine maps cannot follow. The view
therefore stores every origin and development effect centred (`u = τ z`;
an origin's pair `u₁ = τ₁ z₁`, `u₂ = τ₂ (ρ z₁ + sqrt(1 - ρ²) z₂)`), a
smooth one-to-one map whose log Jacobian (`-(log τ₁ + log τ₂ + log
sqrt(1 - ρ²))` per origin pair, `-log τ` per other effect) is added to
the log posterior; conformance check 5 (the score identity) confirms it.
Centring only the effects the year informs fresh (1997) did not help (2
of 33 flagged rescued in a probe); centring all of them rescues nearly
all on model 1. On model 2 no scale helped (below).

**The year** (`bayes/compartmental/year.rs`, `CompartmentalYear`). Its
values are amounts: the cumulative paid of the cells valued in the twelve
months after the valuation, origin by origin, then their outstanding,
each type from its latest observed cell (an origin that lags catches up;
1997 starts at 12 months). The year's log likelihood is each cell's term
of the model's density on the fit's response (amount or loss ratio,
cumulative or incremental paid, normal or log-normal), `-r² / (2σ²) -
log σ`, so the year's change to the log posterior (conformance check 1).
The reserve given a draw is the expected paid to the square's last age
less the paid so far (`fit`'s reserve, decision 5 of
`reserving-compartmental.md`): under cumulative paid the mean at the last
age less the latest (or the year's) paid, under incremental paid the sum
of the expected increments after it, a log-normal response's mean being
its median times `e^(σ²/2)`. Under incremental paid the reserve after the
year does not depend on the year's values, so it is computed once per
draw. A refit samples the model with the year's cells added, as `fit`
samples (data start, pilot run, the fit's sampler), and its draws are
centred the same way.

**API.** `Compartmental::one_year(triangle, paid, outstanding, premium,
settings)` returns `BayesOneYearFit<CompartmentalFit>` (its tables by the
premium column's origins, with `premium` and `latest_paid`, as
`CompartmentalFit`'s); `one_year_segments` follows `fit_segments`, which came
in PR E of `reserving-compartmental.md` (its decision 8), and returns a
`CompartmentalOneYearFits` ("Built (one-year view over segments)" there;
it is not a `BayesOneYearFits`, which needs a chain ladder). Python
`Compartmental.one_year(triangle, paid, premium, *, outstanding=None,
incurred=None, k_threshold=0.7, refit=None, max_refits=None,
n_sims=None)` returns a new `CompartmentalOneYearFit`; R
`compartmental_one_year()` and `compartmental_model_2_one_year()` (with
`refit = NULL`, `max_refits = NULL` and `one_year_sims = NULL`) return a
`compartmental_one_year_fit`. The defaults are the user's of 2026-10-16
(below).

**A refit budget** (for every model). `BayesOneYear::max_refits` (default
`None`, unlimited, as before bit for bit; the ODP seed-pinned test and the
conformance suite confirm it) refits only the first that many simulations
due for a refit, in simulation order, and reweights the others with their
own smoothed weights, listing them in the view's `unrefitted` (and the
segment's), with a warning in Python and R. The engine now runs in two
passes: the first draws each year, smooths its weights and matches it;
the second refits the simulations due one within the budget. The refit
seed is still drawn in the first pass, so a simulation's refit is the
same with or without a budget. Under a budget the first pass also keeps
each such simulation's reweighted closing reserve (one value per origin),
not its weights (one per posterior draw, which would hold `n_sims` times
`n_draws` values until the second pass: 1.2 GB on GenIns at 16,000 draws
and 10,000 simulations, 24 MB with the closing reserves); without one
every such simulation is refitted and nothing is kept. Over segments the
budget applies to each segment (`one_year_segments` runs the engine once
per segment), so `k` segments refit at most `k` times `max_refits`.
Python and R take `max_refits` in every `one_year` and
`one_year_segments`.

**Results, model 1** (`validation/tests/reserving_compartmental_one_year.rs`,
release, 16 threads; company 337, the defaults; 500 simulations under
`ReweightOnly` and the default on the same years, the first 40 under
`RefitAll`; mean and seed-to-seed sd over seeds 0 to 19):

| | model 1 |
|---|---|
| flagged (`k̂` > 0.7) | 82.0% (1.9%) |
| rescued by matching | 95.5% (1.7%), lowest 91.1% |
| rms gap to the refit, rescued, matched (sd of the total) | 0.051 (0.025), largest 0.150 |
| the same simulations reweighted only | 0.156 (0.037) |
| rescued simulations' sd over their refits' sd, matched | 0.992 (0.017) |
| the same, reweighted only | 1.019 (0.033) |
| total CDR sd over the lifetime reserve sd | 0.804 (0.031) |

Matching rescues about 19 in 20 of the flagged simulations and brings
them about three times closer to their refits than reweighting alone,
at every seed (the gap is heavy-tailed: one seed at 0.150, the next
0.072). About four fifths of the lifetime uncertainty emerges in the
first year, as for Meyers' models (0.77 to 0.83) and the ODP bootstrap.
The test's bounds are the sweep's mean ± 4 sds (the gap's 0.20, above
mean + 4 sds), and every seed passes.

**Results, model 2** (`model_2_comparison`, ignored; the defaults, 200
simulations under `ReweightOnly` and under moment matching with
`max_refits` 0, the first 8 under `RefitAll`; seeds 0 to 2): every
simulation is flagged (median `k̂` about 1.5) and matching rescues none,
on the `z` scale or centred, with or without 1997 in the triangle; the
first shift already raises `k̂`. The posterior has 95 parameters on the
full layout, several effect sds near 0 and a slowly mixing `k_p`
(`reserving-compartmental.md` decision 5), far from the near-normal
shape an affine map can follow. Reweighting alone is 0.39 sd (0.20,
0.24 and 0.73) off the refits per simulation, and the total CDR sd is
0.82 (0.78 to 0.84) of the lifetime reserve's under it. So moment
matching, model 1's default, would refit nearly every simulation of
model 2; its default view reweights only (below).

**Cost** (`runtime`, ignored; release, 16 threads, seed 0). Model 1 at
10,000 simulations: `ReweightOnly` 41 s (81% flagged); the default 1,957
s (33 minutes): matching rescues 7,746 simulations and 369 are refitted
at about 5 s of wall time each, most of the run. Refitting all 8,100 or
so flagged simulations would take about 11 hours, so the default is
about 20 times faster than `RefitFlagged`. Model 2: `ReweightOnly` 118 s
at 10,000 simulations; matching without refits (`max_refits` 0) 183 s
at 1,000, about 0.12 s a simulation after the 61 s fit, and rescues
none; a refit takes about 31 s of wall time with 8 side by side, so
moment matching would refit all 10,000 simulations in about 3 to 4 days.
That was open decision 15.

The release conformance suite runs on both models (4 chains of 300
warm-up and 300 kept draws; 30 refit simulations for model 1 and 6 for
model 2, whose matching is not required to rescue any; check 6, a
triangle with no year, runs in the unit half only, since a compartmental
fit needs a future cell). Over seeds 0 to 9 (ten, for their cost: about
3 and 3 minutes a seed locally) the mean z stays within 1.6, and the
reweighted gap of the flagged simulations is 0.255 (sd 0.040) for model
1 and 0.403 (sd 0.111, largest 0.644) for model 2, bounded at 0.45 and
0.9. The opening z score there counts only the lifetime simulations'
Monte Carlo error, not the opening reserve's own over the 1,200 draws,
each with its own 1997 effects, so it is wider than the other models'
(-1.03, sd 1.43, and -0.77, sd 1.33, lowest -3.3), and the compartmental
arms bound it at 7 rather than 5 (the unit test
`the_opening_reserve_is_the_fits` checks it with both errors). The suite
passes each model its refit mode and its number of simulations
explicitly, so the defaults below do not change what it checks.

**The user's decisions of 2026-10-16** (open decision 15, and the fewer
simulations of decision 19):

* *Model 2 reweights only by default.* `BayesOneYear::refit` is now an
  `Option<OneYearRefit>`, `None` by default for the model's own default:
  `MomentMatchThenRefit` for every model (so `BayesOdp`, `BayesBf`,
  `Meyers` and model 1 are unchanged, their provenance too), and
  `ReweightOnly` for `Compartmental` with development effects in its
  priors (model 2; `Compartmental::one_year_refit`). Moment matching
  rescues none of model 2's flagged simulations, so the default skips it
  (it would cost about 0.12 s a simulation for nothing) as well as the
  refits. Such a view sets the new `BayesOneYearFit::approximate` to a
  message (the view is approximate, about 0.4 sd of the total claims
  development result off a refit per simulation on company 337, and how
  to refit), and Python and R warn with it. An explicit `refit`, any
  mode, is used as it is and sets no message, so refits stay a choice,
  with `max_refits` as their budget. A sentinel variant of `OneYearRefit`
  would have touched fewer lines, but every `match` on the enum and every
  binding's parser would need an arm for a value that is not a mode; the
  `Option` mirrors `n_sims`. It is a breaking change for a Rust struct
  literal (`refit: Some(..)`).
* *Fewer simulations.* `Compartmental::ONE_YEAR_SIMS` is 2,000: with
  `BayesOneYear::n_sims` `None` the view runs the model's `n_sims` capped
  at 2,000, as `Meyers` (decision 19), so model 2's default view takes
  about the fit (61 s) plus 2,000 reweightings (about 24 s at its
  10,000-simulation rate) and model 1's about a fifth of its 1,957 s.
  Python takes `n_sims=None` and R `one_year_sims = NULL`.
* *CI.* With `RISK_RS_FULL_BAYES` unset (a pull request) the
  conformance suite ran model 1 alone of the compartmental arms (none
  since 2026-10-19, decision 19's note above), and
  `reserving_compartmental_one_year.rs` checks model 1's `mean_z`,
  `rescued` and `life_ratio` at the same sizes and bounds but skips its
  `RefitAll` comparison (40 refits, 199 s of the test's 368 s locally),
  so not the matched gap. The weekly and manual runs check everything,
  and model 2's default view end to end
  (`the_default_view_of_model_2_reweights`: 2,000 simulations, none
  matched or refitted, the warning, the provenance). The lighter refit
  sampler of option (c) stays open for later.

### 21. The diagonal backtest in Python and R, with each fit's diagnostics (PR 12)

Status: **Built** (2026-10-19), branch `claude/backtest-bindings`,
issue #342, in Rust, Python and R. It reverses "the diagonal backtest has
no binding, so `TriangleModel for BayesOdp` stays Rust-only" (decisions 11
and 13) and the same line of `reserving-compartmental.md` decision 8, at
the user's request of 2026-10-19.

Meyers' validation is retrospective: fit on the upper triangle and score
the forecast against what came in later. `diagonal_backtest`
(`backtest.rs`) does that on held-out calendar diagonals for any
`TriangleModel` (`GlmCandidate`, `BayesOdp`, `Meyers`), but only from
Rust, and a Bayesian fit that did not converge is scored as if it had: the
forecast has no place for the sampler's diagnostics (decision 8, "The
diagnostics are not reported"; `knowledge/findings/meyers-segments-backtest.md`).

1. **Diagnostics travel with the forecast.** The `Diagnostics` struct of
   `bayes/mod.rs` (max R-hat, min bulk and tail ESS, divergences,
   warnings) moves to a module outside the `bayes` feature, so that
   `backtest.rs` (in the WASM scope) can name it, and `bayes::Diagnostics`
   stays a re-export at its old path. `CellForecast` gets `diagnostics:
   Option<Diagnostics>`: `None` for a model with no sampler (the GLM),
   the fit's diagnostics for `BayesOdp` and `Meyers` (for `Meyers` over
   several segments in one frame, not a case the backtest has: one
   frame is one segment). Adding a public field to `CellForecast` is a
   breaking change for anyone who builds one, marked in the changelog.

2. **The backtest reports them and does not stop.** A fit whose
   diagnostics have warnings is still scored, as `BayesOdp::fit` returns
   such a fit with its warnings (decision 5): a study over many triangles
   wants every split and a list of the doubtful ones, not an abort on the
   first. `Backtest::diagnostics(model)` gives one `Option<Diagnostics>`
   per split, and `Backtest::warnings()` every warning as `(model, split,
   message)`. A model that fails to fit is still an error naming the
   model, as now.

3. **Two more metrics.** `METRICS` gains `mean_log_density`, the mean over
   the split's scored rows of the model's held-out log density (NaN for a
   model that gives none), and `total_percentile`, the share of the
   simulated diagonal totals at or below the actual total: the
   probability integral transform of the diagonal, whose uniformity over
   many triangles is the calibration test of the 200-triangle study
   (#392). Meyers' own statistic is the percentile of the outcome's
   ultimate, from a fit on the whole upper triangle; in the bindings it is
   the cdf of a fit's predictive total at the known outcome
   (`Sampled.cdf`, `cdf()`), which needs no new function and is
   documented next to the backtest. `METRICS` grows from 4 to 6 names; its
   order of the first four is unchanged.

4. **Names.** Models are named by `TriangleModel::name` and two models
   with the same name are an error, checked before any fit, so two
   `BayesOdp` settings are told apart. `NamedModel { name, model }`
   wraps any `TriangleModel` under a name of the caller's. It owns its model,
   `Box<dyn TriangleModel + Send>` (`NamedModel::new(name, model)`), so that
   a binding can hold a list of models of different kinds without a
   lifetime. `Backtest::diagonals()` gives each split's held-out calendar
   period index, the label of a split in the bindings' tables.

5. **Python.** `prospicio.reserving.diagonal_backtest(triangle, column,
   models, n_diagonals, *, premium=None, n_sims=1000, seed=0,
   interval=0.9)`. `models` is a dict from name to model or a list of
   models (each named by its kind, an error on duplicates). A model is a
   `BayesOdp` or `Meyers` settings object as for its own `fit`, or a
   `GlmCandidate(terms, glm)` of `prospicio.models.Terms` and
   `prospicio.models.Glm` on the frame's columns (`origin`,
   `development`, `calendar`, `exposure`, as `TriangleFrame` names them);
   `GlmCandidate.odp()` is the ODP model (intercept, origin and
   development factors, the over-dispersed Poisson). `premium` is the
   exposure column of the frame, one value per origin, and `Meyers`
   needs it. The result, `Backtest`, has `models`, `metrics`,
   `n_splits`, `diagonals` (the held-out calendar periods), `scores()`
   (a long table: model, diagonal, metric, value), `mean()` (model by
   metric), `excluded`, `scored_rows`, `log_densities` (a dict by model,
   `None` where the model gives none, the input of the stacking
   functions), `diagnostics()` (a table: model, diagonal, max R-hat, min
   bulk and tail ESS, divergences, number of warnings) and `warnings`.
   The sampling runs without the GIL.

6. **R.** `diagonal_backtest(triangle, column, models, n_diagonals,
   premium = NULL, n_sims = 1000, seed = 0, interval = 0.9)` beside
   `meyers()` and `bayes_odp()`. R fits its models from flat arguments
   and has no settings objects, so `models` is a named list of new S7
   settings objects: `bayes_odp_spec()` and `meyers_spec()` take the
   settings arguments of `bayes_odp()` and `meyers()` (everything but the
   data) with the same defaults, and `odp_glm_spec()` is the ODP GLM. R
   has no builder of `Terms` (its GLMs take formulas), so a custom GLM
   candidate waits for one; Python has it now. The tables come out as
   data frames with the same columns as in Python.

   The Python `GlmCandidate` reads the `Terms` and `Glm` inside
   `prospicio.models.Terms` and `Glm`, whose Rust fields are private to
   `crates/prospicio-python/src/models.rs` (the Models lane): this PR
   makes the two `inner` fields `pub(crate)`, a visibility change only.

7. **Checks.** The new metrics against their definitions on a GLM
   candidate (Rust); diagnostics present for `BayesOdp` and `Meyers` and
   absent for the GLM; a forced failure (too few draws for the ESS
   thresholds) reported as warnings and still scored; duplicate names an
   error; the bindings' tables equal to the Rust `Backtest` on RAA with
   the ODP GLM (fast) and one Bayesian model at small settings, and the
   Python and R results equal to each other for the same seed.

`Compartmental` joins the Rust backtest in `reserving-compartmental.md`
PR E through `CompartmentalCandidate` (its decision 8, "As built"). The
candidate borrowed its triangle while the bindings' `NamedModel` owns its
model, so it now owns its data and a clone shares its fits; the Python
(`Compartmental.candidate`) and R (`compartmental_candidate()`) entry
points, and the second table of the held-out outstanding, were built on
`claude/compartmental-backtest-bindings` (`reserving-compartmental.md`
decision 8, "Built (PR E, backtest bindings)").

**As built** (2026-10-19). What changed from the proposal while building
it:

1. `NamedModel` owns its model (`Box<dyn TriangleModel + Send>`, made by
   `NamedModel::new(name, model)`) instead of borrowing one, so that the
   bindings can hold a list of models with no lifetime.
   `Backtest::diagonals()` gives each split's held-out calendar period.
2. `mean_log_density` is the plain mean, so one row with a log density of
   minus infinity makes its split's value minus infinity. `Meyers` gives
   no log density, so its `mean_log_density` is NaN.
3. The backtest's `n_sims` and `seed` replace those of each model's
   settings, as for `BayesOdp` before. R's `bayes_odp_spec()` and
   `meyers_spec()` therefore leave them out, and Python's docstring says
   that the settings' own are ignored.
4. R's `premium` is a numeric vector with one value per origin or the
   name of a column of the triangle, as `meyers()` reads it. Both
   bindings raise before any fit when a `Meyers` model has no premium.
5. The diagnostics table has a row for every model and split in both
   languages, missing values for a model with no sampler. The diagonals
   are period labels in both. Scored rows are 0-based row numbers of the
   frame in Python and 1-based in R, each language's convention.
6. Python's `GlmCandidate(terms, glm, name="glm")` and
   `GlmCandidate.odp()` (named "odp"); R's `odp_glm_spec()` is named
   "odp" as well. One ODP GLM backtest on RAA is pinned in Rust, Python
   and R to the same numbers.
7. `bayes_odp()` and `meyers()` in R validate their settings through
   the same helpers as the specs (`bayes_odp_settings`,
   `meyers_settings`), with no change in their behaviour.

### 22. `Synchronized` stays an error for the Bayesian models (issue #393)

Status: **Decided** (2026-10-21, by the user), branch
`claude/reserving-214-393`. It keeps decisions 5 and 11 and adds the
reasons and the message.

`SegmentDependence::Synchronized` is an `Error::Bayes` for `BayesOdp`,
`Meyers` and `Compartmental` in `fit_segments` and `one_year_segments`,
checked before any sampling. `BayesBf` fits one segment and has no
dependence setting; if it gets segments, the same rule applies. Of the
three Bayesian analogues the issue weighed, none is built:

1. **Pairing draw `i` of each segment** carries no dependence. Every
   segment has its own chains on its own seed, so draw `i` of one is
   unrelated to draw `i` of another, and pairing them by index gives
   independence under another name.
2. **Sharing the predictive step's variates cell by cell** (synchronized
   process noise) couples the process error only, not the parameter error,
   which is most of a Bayesian reserve's spread. It would also only line up
   cells that share an origin and an age, and Meyers' and the compartmental
   models draw paths, not cells. It looks like a dependence and understates
   it.
3. **A joint posterior with shared parameters** (Zhang and Dukic 2013;
   Meyers 2019, chapter 8) is a different model, as decision 11 says, not
   a setting. It remains a later model; nothing in this decision rules it
   out.

The bootstrap's `Synchronized` resamples the residuals of the same
positions in every segment, and a posterior has no residuals to resample.
`RankCorrelation` (Iman-Conover on the segments' totals) is the supported
dependence. The message every model returns, from one function
(`bayes::synchronized_error`), starts with "Synchronized dependence", says
a posterior has no residuals to resample and points to `RankCorrelation`
(or `Independent`); a unit test pins its words. No parity test with the
bootstrap's `Synchronized` is needed, since nothing is built.

### 24. A light refit for the compartmental one-year view (issue #394)

Built in Rust, Python and R on `claude/gm2-one-year-accuracy`
(2026-10-21). Issue #394 asked for model 2's one-year view to come within
about 0.05 sd of the total of a full refit on company 337 at a budget that
lets it be the default, or for the note to record why not. Decision 20
and open decision 15 left model 2 reweighting only (0.25 to 0.52 sd off the
refits' totals on the years below) because moment matching rescues none of
its flagged simulations and a refit takes as long as a fit. The first of
the issue's ideas, a shorter refit sampler warm-started from the fit,
meets the target, so the other two (finding out why matching fails, a
Laplace or variational proposal) were not tried, as the stop rule of the
task said. Moment matching is untouched.

**The setting.** `BayesOneYear::light_refit: Option<LightRefit>`, `None`
by default, which is the model's own: the fit's own sampler for every refit
of every model but model 2, whose default is now the light refit (the
user's decision of 2026-10-21, below). Every other existing view is
unchanged bit for bit, provenance included (the provenance gains
`light_refit` only when it is set). `LightRefit { chains, tune, draws }`
defaults to 4 chains of 150 warm-up and 250 kept draws. `Compartmental`
uses it and records it in the provenance; the other models ignore it and do
not record it, though a malformed value is still an error for every model
(the Python and R bindings expose it on `Compartmental` only). A refit with it:

* takes the opening posterior's draws (the fit's, with the fresh effects of
  1997 and the new ages drawn, which `simulate` already holds) back on the
  sampled `z` scale, and scales its sampler by their mean and covariance,
  regularised as Stan regularises its dense metric (`Whitened::from_draws`),
  where the fit's refit first runs a pilot sampler of `tune` iterations on
  a Laplace scale and scales by that;
* starts at that mean, adapts for `tune` iterations and keeps `draws` per
  chain, with the fit's `target_accept` and `max_depth` and the refit's
  seed as before;
* warns (`refit_warnings`) when the total closing reserve of its draws has an
  R-hat above 1.05 or fewer than 100 effective draws, or any chain
  diverged. The fit's bounds (1.01, 100 per chain) on the 95 parameters
  warned on every light refit (the effect sds near 0 mix slowly in a short
  run) and 1.01 and 100 per chain on the reserve alone warned on 14 of 32,
  whose totals were within 0.03 sd of the full refits'; the reserve is what
  the view uses of a refit.

**The default, by the user's decision of 2026-10-21 (open decision 17,
(b)).** With `refit` `None`, model 2 refits its flagged simulations
(`RefitFlagged`; moment matching would only add 0.12 s a simulation for
nothing) with `LightRefit::default()`, and the view is not approximate, so
no warning. This replaces open decision 15's reweight-only default for model
2. `BayesOneYear::no_light_refit` (Python and R `light_refit=False`) turns
it off: model 2 then reweights only, with the approximate warning of
decision 20 (its text now says the light refit was turned off). A refit
mode the settings choose is used as it is, with the fit's own sampler
unless `light_refit` is given, so every explicit `refit` of the earlier
decisions behaves as before. A `light_refit` given with `refit` `None` is a
choice to refit, as before. Model 1 keeps moment matching and uses the light
sampler for what matching cannot rescue only when it is given. Nothing else
changed: the view still runs at most 2,000 simulations by default, so
model 2's default view takes about 31 minutes at its default count on 16
threads (a smaller `n_sims` or a `max_refits` budget shortens it), against
85 s reweighted. `Compartmental::one_year_refit` returns `RefitFlagged` for
model 2. Python: `light_refit=None` (the model's default), `False` (off),
`True` or `(chains, tune, draws)` on `Compartmental.one_year` and
`one_year_segments`; R: `light_refit = NULL`, `FALSE`, `TRUE` or
`c(chains, tune, draws)` on `compartmental_one_year()`,
`compartmental_model_2_one_year()` and `compartmental_one_year_segments()`.
CI's pull requests run no model-2 view at the default count: the weekly and
manual run's `the_default_view_of_model_2_refits_lightly` uses 20
simulations and `the_view_of_model_2_without_the_light_refit_reweights`
runs the 2,000 reweighted.

**Checked on every PR's `bayes` job** (`light_refit_conforms`, run with
the one-year conformance suite on models 1 and 2 with a small sampler):
one year drawn from the opening posterior, refitted by the fit's sampler
and by the default light one, gives the opening's column names both ways,
closing reserves whose means agree within 0.3 of their sd and whose sds
agree within 25%, and light draws whose score under `log_posterior` with
that year has mean zero. Skipping the map back to the centred scale fails
it on both models. The accuracy against full refits on company 337 below
runs only weekly and on demand (`RISK_RS_FULL_BAYES`).

**Results** (company 337 as in decision 20, model 2 at its defaults, the
first years of the simulations of the seed; `RefitAll` with the full
sampler is the reference; the gap is the rms over simulations of the
difference of the totals, in units of the total's sd over 1,000
reweighted simulations (about 11,000); `model_2_light_refit`, ignored):

| seed, years | reweighting only | 4,150,250 | 4,300,500 | 4,100,150 | 4,50,100 | 2,200,300 |
|---|---|---|---|---|---|---|
| 0, 16 | 0.522 | 0.035 | 0.020 | | | |
| 1, 32 | 0.252 | 0.029 | 0.025 | 0.059 | 0.059 | 0.047 |
| 2, 24 | 0.297 | 0.033 | 0.038 | | | |

(chains, warm-up, draws). The mean gap is within 0.02 of zero and the
ratio of the sds of the totals within 1.5% of 1 in every row. Two full
refits differ by their own Monte Carlo error, so about 0.03 is the floor of
the comparison: longer light refits (4,300,500; 4,500,1000 at 0.015 on seed
0) do not get closer, shorter ones (4,100,150 and below) fall to 0.06 to
0.08. The default, 4,150,250, is the shortest run at the floor. It was
chosen on seed 1, whose years also report it, so 0.029 is in sample; seeds
0 and 2 (0.035 and 0.033) were not used to choose it. 2 chains are at the
bound of the target (0.045 to 0.047). These figures are model 2 on
company 337 only: model 1 uses the light sampler only for the simulations
moment matching cannot rescue (about 5% on company 337), and those refits
were not compared with full refits.

**Cost** (`light_runtime`, ignored; release, 16 threads). Every time here
is wall time on one 16-thread machine, and a time per refit is the wall
time of a parallel run over its refits (throughput, not one refit's
latency), so it scales with the hardware. A refit takes about 0.5 to 0.9 s
(32 refits of the default: 15 s beyond the fit on seed 1; full refits take
about 30 s each in the same conditions, about 30 times the 0.9 s of the
2,000-simulation run); 2,000 simulations, the default
count, took 1,872 s (31 minutes): fit 61 to 120 s, every simulation flagged
and refitted, 29 refits warned, mean total claims development result 145
(sd 11,058). The default view of model 2 (reweighting only) takes about
85 s, so a light refit of every simulation is about 20 times as long as
that view and about a thirtieth of refitting in full (about 17 hours at
2,000 simulations). It is as long as model 1's default view at 10,000
simulations (33 minutes) and about four times model 1's at 2,000.

**Proposal, for the user (open decision 17), decided.** The target is met
at a cost of about 30 minutes for the default 2,000 simulations, 20 times
the reweighted view. The user decided on 2026-10-21 to make it model 2's
default (open decision 17 below).

**A cheaper sampler setting, measured and not built.** Lowering the refit's
`target_accept` from model 2's 0.999 to 0.95 (a temporary override; 400
simulations at the default light sampler for the times, seed 1 and 32
years for the gap) cut a refit from 0.64 s to 0.35 s of wall time, and to
0.8 cut it to 0.20 s. The gap to the full refits was 0.036 at both
settings (0.029 at 0.999) and the total's sd within 0.5% of the full
refits', but with a few divergences (1 to 25 among 1,000 kept draws, in
about a quarter of the refits), which the warning above counts: 122 and
386 of 400 refits warned, against 4. Not built: the light refit uses the
fit's own `target_accept`, and the saving of 1.8 times is open decision
17's choice (d).

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
   posterior. Decided 2026-10-12: (a) for now. Confirmed 2026-10-13 by the user: (a), the plug-in, is kept.
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
   Decided 2026-10-13 by the user: (b), PSIS reweighting, as the next
   capability. Its weakness is a diagonal that moves the posterior far (a
   young origin, an outlying diagonal): `k̂` above 0.7 flags it. A refit
   per simulation (d) stays exact there but costs one NUTS run per
   simulation (about 0.1 s each, so 1,000 s for 10,000 simulations on one
   thread); a hybrid that refits only the flagged simulations is the
   proposed fallback. Built for `BayesOdp` in PR 8 (decision 16,
   2026-10-14) with that hybrid as the default (`RefitFlagged`); 14% to
   26% of the simulations are flagged on RAA and GenIns, so the default
   costs a minute or two per triangle (new open decision 12).
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
   Decided 2026-10-13 by the user: (a), and all three next models are
   wanted, in this order of PRs: Meyers' CAY for incurred data; Meyers'
   CRC and CSR with `fit_segments` and `TriangleModel` (for the
   200-triangle calibration study); then the compartmental design note.
   CAY is built as PR 5; the second, for CRC, CSR and CAY alike, is
   built as PR 6 (decisions 8 and 11, "Meyers"), with `fit_segments`
   in Python and R. The third is
   `docs/design/reserving-compartmental.md` (decided 2026-10-13).
10. **Bindings with each model.** (a) Rust, Python and R in the same PR,
    as `reserving-v02.md` decision 7; (b) Rust first, bindings after.
    *Recommended: (a)*.
11. **Draws from other samplers.** Noted 2026-10-13 by the user for the
    longer term, not planned: taking external draws (arviz
    `InferenceData`, Stan CSV, nutpie traces) into our R-hat, ESS and LOO.
    Today only our own NUTS draws (nuts-rs, the Rust core of nutpie) reach
    them; `docs/design/models.md` and the roadmap are corrected to say so.
12. **A cheaper first fallback for the one-year view.** Raised
    2026-10-14 by PR 8 (decision 16). At the defaults 14% (RAA) to 26%
    (GenIns) of the simulations have `k̂` above 0.7, and refitting them
    costs a minute or two per triangle; reweighting alone is about 1% low
    on the sd. (a) Keep `RefitFlagged` as the only fallback; (b) before
    refitting, retry a flagged simulation with the posterior draws moved
    by the affine map from the opening Laplace approximation to the
    updated one (importance sampling with a transformed proposal, the
    moment matching of Paananen et al. 2021, which `loo` offers as
    `loo_moment_match`), refitting only when its `k̂` is still high: the
    ODP's log posterior and Newton mode are at hand, and one such
    reweighting costs about one 4,000-draw pass of the log posterior,
    some ms, against about 39 ms for a refit, but it needs each model's
    log posterior in the engine and is not measured yet; (c) refit with a
    shorter sampler (the refits need only a posterior mean). *Recommended:
    (a) now, with (b) measured on RAA and GenIns before the bindings
    expose the settings*, since (b) would change the defaults' cost and
    the flagged share the user sees.
    Decided 2026-10-15 by the user: keep `RefitFlagged` as the default for
    now, and try (b); the engine's `YearModel::log_posterior` (decision 17)
    is its hook. Built and measured in PR 9 (decision 18) as the opt-in
    `MomentMatchThenRefit`, in `loo`'s general form (affine steps to the
    weighted mean, variances and covariance, then the split estimate)
    rather than from the Laplace approximation: on RAA and GenIns it
    rescues 99.99% to 100% of the flagged simulations, within 0.04 to 0.06
    sd of their refits (reweighting alone: 0.16 to 0.24), at about 1 ms
    each against about 26 ms for a refit. Open for the user: make it the
    default (recommended). Decided 2026-10-15 by the user: moment matching
    is the default fallback, for every model with a one-year view and for
    `Meyers` and `Compartmental` as they get theirs (PR 10, decisions 18
    and 19).
13. **One PSIS for both crates.** Raised 2026-10-14 by PR 8. The one-year
    engine carries a copy of `prospicio_bayes::elpd`'s private PSIS (a
    test checks the two agree on `k̂`). (a) Make `psis` public in
    `prospicio-bayes` (a Models-lane PR of a few lines) and delete the
    copy; (b) keep the copy. *Recommended: (a)*. Decided 2026-10-15 by the
    user: (a). `psis` is public (#273) and PR 9 deleted the copy
    (decision 16, "PSIS").
14. **The order of the one-year views to come.** Raised 2026-10-14 by PR
    8. The engine takes `Meyers` (whose `closing` depends on the year's
    cumulative values), `BayesBf` and `Compartmental` through
    `YearModel`. *Recommended: `BayesBf` first (the ODP's likelihood, so
    the same code with its priors), then `Meyers` (CRC, CSR, CAY), then
    `Compartmental`*, after this PR's bindings. Decided 2026-10-15 by the user: every Bayesian model gets the
    one-year view, made automatic: one interface every Bayesian fit
    implements, one generic view, and a conformance suite run on every
    implementor, with a line in AGENTS.md's "Checks" that a new Bayesian
    model must pass it. Built in PR 9 (decision 17) with `BayesBf`;
    `Meyers` and `Compartmental` follow once the user sets the defaults
    for their refit cost. Decided 2026-10-15 by the user: moment matching
    is their default too (open decision 12); `Meyers` is built in PR 10
    (decision 19), `Compartmental` (models 1 and 2) in PR 11 (decision
    20), which leaves the cost of model 2's refits to open decision 15.
15. **The refit cost of the compartmental one-year view.** Raised
    2026-10-16 by PR 11 (decision 20). Decided 2026-10-16 by the user:
    model 2's default view reweights only (no refits and no moment
    matching, which rescues none), with a warning that it is approximate
    (about 0.4 sd of the total off a refit), and refits stay a choice;
    model 1 keeps moment matching; both run at most 2,000 one-year
    simulations by default (a form of (d)); pull requests run model 1 of
    the compartmental one-year validation only, the weekly and manual runs
    all of it (decision 20, last paragraphs). (c) stays open for later.
    *Superseded for model 2's default on 2026-10-21 by open decision 17:
    its view refits the flagged simulations with the light refit, and
    reweights only (with the warning) when `light_refit` is turned off.*
    The question as raised: Moment matching rescues about 95%
    of model 1's flagged simulations, so its default view takes
    about 33 minutes at 10,000 simulations (369 refits); it rescues none of model 2's, so
    model 2's default view refits nearly every simulation, about 31 s of
    wall time each with 8 side by side: days at 10,000 simulations. (a)
    Keep the unlimited default, as decided, and leave `max_refits` to the
    user (the docs say what it costs); (b) a default budget for model 2
    (say `max_refits` of a few hundred), the rest reweighted and listed
    in `unrefitted` (0.39 sd off per simulation); (c) a lighter refit
    sampler for the one-year view (no pilot run, the opening posterior's
    scale as the metric, a short warm-up and fewer draws, since a refit
    needs only the posterior mean of the reserve), measured against full
    refits before it becomes a default; (d) fewer simulations for model
    2's view. *Recommended: (c), measured, then (b) with what is left*;
    until the user decides, (a) stands.
17. **Whether model 2's one-year view refits with the light sampler by
    default.** Decided 2026-10-21 by the user ("ok" to the recommendation):
    (b), the light refit of the flagged simulations is model 2's default
    (`RefitFlagged`, no approximate warning, about 31 minutes at the default
    2,000 simulations); `light_refit=False` reweights only; (c) and (d) not
    taken (decision 24). The question as raised: raised 2026-10-21 by
    decision 24 (#394), which builds the
    `light_refit` setting open decision 15 (c) asked for and measures it
    against full refits of model 2 on company 337: the gap of the totals is
    0.03 sd (0.029 to 0.035 for the default on three seeds; reweighting only
    0.25 to 0.52; the target was 0.05), at about 0.6 to 0.9 s of wall time
    a refit on 16 threads, so 2,000 simulations take about 31 minutes there
    against 85 s reweighted. Until the user decided, model 2 still
    reweighted only, with its warning. (a) Keep that and leave `light_refit` a choice (the docs and
    the warning say so and what it costs). (b) Make the light refit of the
    flagged simulations model 2's default (`light_refit` on by default,
    `RefitFlagged`; no approximate warning; about 31 minutes at 2,000
    simulations, less with a smaller `n_sims`). (c) As (b) with a smaller
    default count for model 2, say 500 simulations (about 8 minutes, the
    sd of the total from 500 simulations known to about 3%), or with a
    `max_refits` budget, the rest reweighted and listed. (d) As (b) or
    (c) with `target_accept` 0.95 for the light refits, about 1.8 times
    faster at the same gap but with a few divergences in a quarter of the
    refits and a different warning rule; not built. *Recommended: (b) if
    half an hour is acceptable for a default view, else (c)*; the change is
    one line in `Compartmental::one_year_defaults` and a documented
    change of CI's `the_default_view_of_model_2_reweights`, which would
    then run the light refits, so about 30 minutes in the weekly run
    (built so that the weekly run's default-view test uses 20 simulations).
