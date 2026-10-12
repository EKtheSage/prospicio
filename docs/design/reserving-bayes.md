# Design note: Bayesian reserving (Bayesian chain ladder, Meyers' models)

Status: **Proposed** (2026-10-11; every decision below waits on the user, see "Open decisions for the user") · Lane: Reserving, with the Models lane for `prospicio-bayes` · Depends on: `triangle.md`, `reserving-v02.md` (exposure, tails, the bootstraps, dependence between segments), `models.md` (`prospicio-bayes`, NUTS, diagnostics), `predictive-distribution.md`, `docs/roadmap.md` (v0.7 row; `prospicio-bayes` Next, item 1)

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
  (CSR) and correlated accident year (CAY) models, which Meyers (2015,
  2019) shows to be better calibrated than Mack and the ODP bootstrap on
  200 Schedule P triangles.

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
| 2 | `claude/bayes-segments` | `BayesOdp::fit_segments` with `SegmentDependence`; `TriangleModel` for the diagonal backtest (decision 8) | PR 1 |
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
| CRC, CSR (later CAY, IPI) | Meyers' R/Stan code, `meyers-appendix.zip` (CAS), run with CmdStan 2.39.0 | Meyers (2019), CAS Monograph 8, 2nd ed.: group 353 Commercial Auto, Tables 2.1-2.3, 5.1, 7.1 and the totals quoted in decision 7 |
| Diagnostics | `prospicio_bayes` (`rhat`, `ess_bulk`, `ess_tail`), already at parity with R `posterior` | |
| Model comparison | `prospicio_bayes::elpd::loo`, already at parity with R `loo` 2.6.0 | Meyers' elpd_loo for CSR (49.76) and CRC (47.80) |

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
  10×10 square so that the outcome can be checked. Source: the company-level
  CAS loss reserve database, as shipped in chainladder-python
  (`load_sample('clrd')`, `GRCODE == 353`, `LOB == 'comauto'`). The
  generator checks it against Meyers' Tables 2.1-2.3 before writing it
  (that they agree is not yet verified). The repo's existing
  `clrd_*.csv` are industry sums and do not hold it.

## Decisions

### 1. Which model ships first

Proposed. Four families were compared:

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
CRC and CSR come second because they are what a user would choose a
Bayesian model *for* (better calibrated on Meyers' 200 triangles), and
Meyers' code gives a parity target. Growth curves and compartmental
models wait: neither has a reproducible reference, and the compartmental
model's identification problems need informative priors that a library
default should not choose.

### 2. Where the code lives

Proposed. `prospicio-reserving` is in the WASM scope (AGENTS.md "Checks",
`docs/architecture.md`, `ci.yml` line 38, default features), and heavy
engines "never become required Rust dependencies"
(`docs/architecture.md`). `prospicio-bayes` pulls in `nuts-rs`, Rayon and
rustfft, and whether `nuts-rs` builds for `wasm32` is not verified. So
`prospicio-reserving` cannot take a required dependency on it. Options:

| Option | Ownership | WASM | Access to reserving internals | Cost |
|---|---|---|---|---|
| A. A `reserving` module in `prospicio-bayes`, depending on `prospicio-reserving` | Models lane's crate; Reserving would edit it | clean: `prospicio-bayes` is outside the scope | none: `NextYear`, `Resample`, `SharedPool` are `pub(crate)` | a lane crossing on every PR |
| B. An optional feature `bayes` and a module `bayes` in `prospicio-reserving` | Reserving lane's own crate | the default-feature check is unchanged | full: one crate, so the one-year view and dependence internals are reachable | a feature to keep compiled in CI |
| C. A new crate `prospicio-reserving-bayes` | Reserving lane, new crate | clean | none, as A | one more crate to version, publish and re-export |

**Option B.** It keeps the work in the Reserving lane's crate, it is the
only option that can reach the `pub(crate)` machinery of the bootstraps
(the one-year view and `SegmentDependence`, decisions 8 and 11) without
widening `prospicio-reserving`'s public API, and it adds no crate to the
release.

```toml
# crates/prospicio-reserving/Cargo.toml
[dependencies]
prospicio-bayes = { workspace = true, optional = true }

[features]
bayes = ["dep:prospicio-bayes"]

# crates/prospicio/Cargo.toml (the umbrella)
bayes = ["dep:prospicio-bayes", "prospicio-reserving/bayes"]
```

There is no dependency cycle: `prospicio-bayes` depends on core, glm,
math, models and prob, not on reserving. The Python and R binding crates
already depend on `prospicio-bayes` and turn on `prospicio-reserving/bayes`,
so wheels and the R package gain no new native dependency.

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

Lane ownership: everything in this note is the Reserving lane's, in
`crates/prospicio-reserving/src/bayes/`, `validation/tests/reserving_bayes*.rs`
and this note. `prospicio-bayes` stays the Models lane's. If PR 1 needs a
change there (for example a public way to split one seed between the
chains and the predictive step, or a dense mass matrix, decision 5), it
goes in a small PR of its own with the user's go-ahead, and merges first.

### 3. The Bayesian ODP model

Proposed. Incremental values `X_wd` for origin `w = 1..n` and development
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
well. Negative increments are allowed, as in the chain ladder; the
posterior is proper whenever the priors are (they always are), and the
mode exists without priors under the chain ladder's own condition (each
development column's observed sum positive). With vague priors the mode is
the chain ladder (the ODP GLM's MLE is the chain ladder), which is the
first test.

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

### 4. The predictive distribution

Proposed. Reserves come from `PosteriorDraws::predictive`, which picks a
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

Proposed.

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
  `loo()` the PSIS-LOO of the pointwise log likelihood, so a Bayesian ODP
  and a CRC can be compared on the same triangle as Meyers does.
* Errors: the chain ladder's own (a non-positive development column, an
  unknown column), the existing `Error::InvalidSetting { name, value,
  expected }` for a
  non-positive or non-finite prior standard deviation, and a new
  `Error::Bayes(String)` for a sampler error from `prospicio-bayes` or a
  setting the model does not support.

### 6. Dispersion

Proposed. `φ` is plugged in. Under a quasi likelihood, `φ` has no proper
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
appendix code in PR 3 and copied exactly, since parity is against that
code. On the sampler's unconstrained scale `s_i` is a logit with its log
Jacobian.

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
CRC paid total 40,121 (SE 2,487); CRC incurred 39,147 (SE 1,642); CSR
paid 37,597 (SE 2,401) with `γ` 0.0446 (sd 0.0282); the outcome 40,000;
elpd_loo CSR 49.76, CRC 47.80. These are a secondary check (decision 10).

Notes carried into PR 3: no development past age `m`, since `β_m = 0`
fixes the last age as ultimate; `a_1 = 0` and `β_m = 0` and the σ
ordering are needed for identification; CAY's recursion (`ρ` times the
previous origin's residual) makes the likelihood depend on observed lagged
cells and is why CAY waits for PR 5.

### 8. The diagonal backtest

Proposed (PR 2). `BayesOdp` and `Meyers` implement `TriangleModel`
(`backtest.rs:40`), whose doc already names "a Bayesian model": `forecast`
fits on the train rows and returns `n_sims` joint posterior predictive
draws of the test rows. `fit_rows` keeps every train row, negative
increments included, since the quasi likelihood accepts them (the Poisson
GLM's drops them). This gives Meyers-style calibration on held-out
diagonals beside the bootstraps and the GLM.

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

Proposed. There is no package to match, so the reference is the same
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
* **Sanity checks, not parity** (unit tests in PR 1): with priors of sd
  1,000 the posterior mode equals the chain ladder's GLM fit to 1e-8; the
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

### 11. Segments and dependence

Proposed (PR 2). `fit_segments` fits each segment's own posterior (the
same model, independently), and the output has the shape of
`OdpBootstrapFits`: dimensions the key names plus `origin`, with
`to_long`, `totals` and `development_table`. `dependence` is the existing
`SegmentDependence`:

* `Independent` (default): each segment's simulations as drawn;
* `RankCorrelation { spearman }`: whole simulations of each segment
  reordered by Iman–Conover on the segment totals through
  `reorder_groups`, exactly as for the bootstraps, since it needs only a
  `PredictiveDistribution` per segment;
* `Synchronized`: an error (`Error::Bayes`), since it shares residual
  positions and a posterior has none.

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

Each has a recommendation; nothing is built until the user picks.

1. **First model.** (a) Bayesian ODP, then CRC and CSR; (b) CRC and CSR
   first; (c) both in one PR. *Recommended: (a)*, decision 1: the roadmap
   names the chain ladder, it is the easiest to sample with a hand-written
   gradient, and it checks against both a Stan run and our bootstrap.
2. **Where the code lives.** (a) A `bayes` feature and module in
   `prospicio-reserving`; (b) a `reserving` module in `prospicio-bayes`;
   (c) a new `prospicio-reserving-bayes` crate. *Recommended: (a)*,
   decision 2: the Reserving lane's own crate, WASM check unchanged, and
   access to the one-year and dependence internals without widening the
   public API.
3. **Dispersion.** (a) Plug in the chain ladder's Pearson `φ`; (b) sample
   `φ` with a prior under the quasi likelihood; (c) a separate proper
   model (gamma or Tweedie increments) with sampled dispersion, later.
   *Recommended: (a) now, (c) if wanted*, decision 6; (b) is not a valid
   posterior.
4. **Failed diagnostics.** (a) Return the fit with warnings (and raise
   them in Python and R); (b) return an error. *Recommended: (a)*, so a
   user can inspect a poor fit; the warnings name the parameter and the
   threshold.
5. **Reference draws.** (a) The same model in Stan through CmdStan 2.39.0
   and cmdstanpy under `uv`, summaries committed; (b) R rstan or brms
   (not installed locally); (c) published numbers only. *Recommended:
   (a)*, decision 10, with Meyers' printed numbers as a looser second
   check in PR 3.
6. **One-year view.** (a) Not now; (b) importance reweighting of the
   posterior by the next diagonal (PSIS, with `k̂` as the guard); (c) a
   Laplace update for the ODP model only; (d) MCMC refit per simulation.
   *Recommended: (a) now, then (b)* in its own PR after PR 3, decision 12.
7. **Meyers' data source.** (a) chainladder-python's company-level
   `clrd`, checked against Tables 2.1-2.3; (b) R `raw`; (c) typed from the
   monograph's tables. *Recommended: (a)*: reproducible by script, and the
   check against the printed tables catches a mismatch.
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
