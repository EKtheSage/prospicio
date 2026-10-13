# Design note: compartmental reserving (Gesmann and Morris)

Status: **Proposed** (2026-10-13; nothing built; every decision below is
Proposed until the user settles the open decisions at the end) · Lane:
Reserving, with the Models lane for `prospicio-bayes` · Depends on:
`reserving-bayes.md` (decisions 2, 5, 10, 11, 13 and 15, and open decision
9, which asked for this note), `models.md` (`prospicio-bayes`, NUTS,
diagnostics), `predictive-distribution.md`, `reserving-v02.md` (exposure,
dependence between segments), `docs/roadmap.md` (v0.7 row)

## Goal

A hierarchical compartmental reserving model in `prospicio-reserving`,
behind the `bayes` feature, that fits paid and outstanding losses jointly
from one set of interpretable parameters and gives the same output as the
other Bayesian models: a joint `PredictiveDistribution` of reserves by
origin, with parameter and process error, sampled with NUTS. What it adds
over `BayesOdp`, `BayesBf` and `Meyers`:

* **paid and outstanding in one model.** Claims flow from exposure (EX)
  into outstanding (OS) as they are reported and from outstanding into
  paid (PD) as they settle. The fit uses both columns at once, so the
  paid projection and the case reserve tell one consistent story instead
  of two separate chain ladders;
* **parameters an actuary can read and set a prior on**: a reported loss
  ratio `RLR`, a reserve robustness factor `RRF` (paid over reported, so
  the ultimate loss ratio is `RLR × RRF`), a reporting rate `k_er` and a
  payment rate `k_p`. A prior on any of them is a statement about the
  business, not about an abstract development factor;
* **pooling across accident years**: `RLR` and `RRF` vary by origin as
  correlated random effects, so immature origins borrow strength from
  mature ones.

The models are Gesmann and Morris (2020), building on Morris (2016). They
are named after pharmacokinetic compartment models, and the papers fit
them with brms and Stan.

Out of scope here, kept for later PRs or notes: the multistage exposure
with a gamma-pdf source (Gesmann and Morris 2020, Eq. 4), a reporting rate
that grows with development time (Morris 2016, sec. 4.1.1; Gesmann's
PK/PD post), market-cycle multipliers (Gesmann and Morris' model 3), AR(1)
residuals and calendar-year shocks (Morris 2016, sec. 4.2), hierarchies
across companies (Zhang, Dukic and Guszcza 2012) and joint models across
lines.

## Pull requests

Each PR holds one capability and lands in Rust, Python and R together, as
in `reserving-v02.md` decision 7 and `reserving-bayes.md` open decision 10.

| PR | Branch | Holds | Base |
|---|---|---|---|
| 0 | `compartmental-design` | this note; a pointer from `reserving-bayes.md` open decision 9 | `main` |
| A | `claude/bayes-dual` | the `Scalar` trait and the in-house dual type (decision 4); the existing densities made generic, with oracle tests against their hand-written gradients; the chunked-pass cost measurement that `reserving-bayes.md` decision 15 still owes. No new model, no binding change | `main` |
| B | `claude/bayes-compartmental` | `Compartmental` with one OS compartment and Gesmann and Morris' model 1 likelihood (decisions 1 to 3, 5 to 7); company 337's data; the Stan reference; Python and R bindings | PR A |
| C | `claude/bayes-compartmental-lognormal` | model 2's likelihood (log-normal on outstanding and incremental paid loss ratios) and its development-year random effects (decision 2) | PR B |
| D | `claude/bayes-compartmental-two-os` | the two-OS model (fast and slow settlement), the generic RK4 solver as its oracle (decision 3) | PR B |
| E | `claude/bayes-compartmental-segments` | `Compartmental::fit_segments` with `SegmentDependence`, and `TriangleModel` for the diagonal backtest (decision 8) | PR B, and the Meyers segments branch merged |
| later | | multistage exposure on the RK4 solver; market cycles; AR(1) residuals (open decision 8) | PR D |

PRs B to E touch the same binding files as the other Bayesian models
(`crates/prospicio-python/src/reserving.rs`,
`crates/prospicio-r/src/reserving.rs`, `R/prospicio/R/reserving.R`,
`python/prospicio/reserving.py`) and the generated stub, `NAMESPACE` and
`man/`. Whichever merges second merges `main` and regenerates them; it
does not merge them by hand.

## References

| What | Primary reference | Secondary |
|---|---|---|
| One-OS model, model 1 (Gaussian on amounts) | Gesmann and Morris (2020), "Hierarchical Compartmental Reserving Models", CAS Research Paper: Eq. 1 and 2, sec. 5.2.1, Stan code in sec. 7.2.2, transcribed into current Stan syntax and run with CmdStan 2.39.0 | Gesmann's blog post "Hierarchical compartmental reserving models" (8 January 2018): nls, nlme and brms fits of company 337 |
| Model 2 (log-normal on loss ratios, incremental paid) | the same paper, sec. 3.3 and 5.2, Stan code in sec. 7.2.3, transcribed | its printed posteriors (standardised intercepts; decision 7) |
| Two-OS model | the same paper, Eq. 3 (closed form) and sec. 2.3 | our RK4 solve of the ODE system |
| Background | Morris (2016), "Hierarchical Compartmental Models for Loss Reserving", CAS E-Forum Summer 2016 (nlme, OpenBUGS) | the CAS blog "Unboxing hierarchical compartmental reserving models" (2022); IFoA GIRO "Introducing compartmental reserving" |
| Diagnostics and model comparison | `prospicio_bayes` (`rhat`, `ess_bulk`, `ess_tail`, `elpd::loo`), already at parity with R `posterior` and `loo` | |

No package in R `ChainLadder` or chainladder-python fits these models, so,
as in `reserving-bayes.md` decision 10, the reference is the same model
written in Stan. The published Stan code is brms output on rstan 2.19.2
(sec. 7.3) and uses syntax removed since Stan 2.33 (`real[]`,
`integrate_ode_rk45`); it is transcribed with provenance into plain Stan
(no licence is stated, so it is not copied), not run as published. brms,
rstan, cmdstanr and the `raw` package are not installed locally (open
decision 6).

Data (new in PR B): `validation/data/clrd_wkcomp_337.csv`, the papers'
illustrative company, NAIC company 337 workers' compensation, accident
years 1988 to 1997, ages 1 to 10, the full 10×10 square, from the same
source `reserving-bayes.md` chose for group 353
([EKtheSage/cas-schedule-p-data-model](https://github.com/EKtheSage/cas-schedule-p-data-model),
table `mart_reserving_model_training`, release pinned and its sha256
checked against the release's `manifest.json`). The generator
`validation/scripts/clrd_wkcomp_337.py` follows
`validation/scripts/clrd_comauto_353.py` with `company_code = '337'` and
`line_of_business = 'workers_compensation'`. Columns, USD thousands:

| Column | Mart source | Note |
|---|---|---|
| `paid` | `cum_paid_loss` | |
| `incurred` | `incurred_loss` | includes bulk and IBNR |
| `case_incurred` | `incurred_loss - bulk_loss` | Meyers' convention, as in `clrd_comauto_353.csv` |
| `premium` | `earned_prem_direct` | the papers' premium |
| `premium_net` | `earned_prem_net` | Meyers' convention |

The two outstanding definitions differ. Morris (2016) prints, for 1988 at
age 1, premium 104, OS 53 and paid 10 (USD millions), and the mart gives
`incurred_loss - cum_paid_loss` = 53,121 and `earned_prem_direct` =
104,437 there (queried locally with DuckDB, 2026-10-13). So the papers'
OS includes bulk and IBNR, and their premium is direct, unlike Meyers'
case incurred and net premium. The generator asserts that cell (paid
9,558, incurred 62,679, bulk 24,619, case reserve 28,502, direct premium
104,437) and Morris' printed table row by row before writing. The repo's
`clrd_wkcomp.csv` and `clrd_lines.csv` have no outstanding column and
cannot be used.

## Decisions

### 1. Which model ships first

Proposed. Four candidates:

| Candidate | Reference | Solver | Sampler difficulty |
|---|---|---|---|
| Gesmann and Morris one-OS, model 1: Gaussian on OS and cumulative PD amounts, origin effects on `RLR` and `RRF` | printed priors and Stan code (sec. 7.2.2); company 337 in the mart; the blog's brms fit | closed form (Eq. 2) | moderate: 4 rates and ratios, 3 hierarchy parameters, 2 effects per origin, 2 sigmas; tight priors make it identified (decision 2) |
| Gesmann and Morris one-OS, model 2: log-normal on OS and incremental paid loss ratios, origin and development effects on every parameter | printed priors, Stan code (sec. 7.2.3) and printed posteriors | closed form | harder: about 30 more random effects, funnels, and the paper's text and code disagree on one prior |
| Gesmann and Morris two-OS | the closed form (Eq. 3); no printed fit we can rerun | closed form, but the authors advise a numerical solver | harder: two payment rates that can swap |
| Morris (2016): nlme and OpenBUGS, AR(1) residuals, a 1996 shock | printed estimates only; OpenBUGS is not available | closed form | harder: AR(1) and the shock are model-specific |

**Ship model 1 first (PR B), then model 2's likelihood (PR C), then the
two-OS model (PR D).** Model 1 is the smallest model with the paper's own
priors and code, it needs no ODE solver, and every later variant reuses
its density, its predictive step and its data. It has a known flaw
(sec. 5.3.1 of the paper: a Gaussian on amounts predicts negative OS and
negative paid increments), which is why model 2 follows directly; model 1
first keeps PR B to one new likelihood and the parity check to the paper's
simplest program. Choosing model 2 first is open decision 1 (b).

### 2. The model, its priors and identification

Proposed. For origin `i` with premium `Π_i`, development time `t` in
years (ages 1 to 10, as the papers), and rates `k_er` (reporting) and
`k_p` (payment):

```text
dEX/dt = −k_er EX                       EX(0) = Π_i
dOS/dt =  k_er RLR_i EX − k_p OS        OS(0) = 0
dPD/dt =  k_p RRF_i OS                  PD(0) = 0
```

Closed form (Gesmann and Morris, Eq. 2), written with one kernel `g` that
removes the singularity at `k_er = k_p`:

```text
Δ    = k_er − k_p
g(t) = (e^{−k_p t} − e^{−k_er t}) / Δ = t e^{−k_p t} · (−expm1(−Δ t)) / (Δ t)
OS(t) = Π_i RLR_i k_er g(t)
PD(t) = Π_i RLR_i RRF_i [ (1 − e^{−k_p t}) − k_p g(t) ]
```

The second line is exact for every `Δ`, and `(−expm1(−x))/x` takes a
series branch for `|x| < 1e-5` (`1 − x/2 + x²/6`), so the density and its
dual gradient are finite and smooth through `k_er = k_p`, where
`g(t) = t e^{−k t}`. The PD form follows from the paper's by
`k_er (1 − e^{−k_p t}) − k_p (1 − e^{−k_er t}) = Δ (1 − e^{−k_p t}) −
k_p Δ g(t)`. The ultimate loss ratio is `PD(∞)/Π_i = RLR_i RRF_i`, and OS
peaks at `log(k_p/k_er)/(k_p − k_er)`, the inflection of paid.

**Hierarchy** (model 1, as the paper's code): the rates are shared, `RLR`
and `RRF` vary by origin with correlated log-normal effects.

```text
log RLR_i = log RLR + u_i1,   log RRF_i = log RRF + u_i2
u_i = diag(τ) L z_i,   z_i ~ N(0, I₂),   L L' = Ω,   Ω = [[1, ρ], [ρ, 1]]
```

**Sampler scale.** Every positive parameter is non-centred log-normal,
`θ = m e^{s o}` with `o ~ N(0, 1)`, as the paper (Stan's default inits in
(−2, 2) break positive parameters on their natural scale). The sampler
sees `o_ker`, `o_kp`, `o_RLR`, `o_RRF`, `log τ₁`, `log τ₂`, `atanh ρ`, the
`z_i`, and `log σ_OS`, `log σ_PD`, with the log Jacobians of the last
four maps. With 10 origins that is 29 parameters, the size of Meyers' CRC.

**Likelihood, model 1** (the paper's sec. 3.1 and code 7.2.2): normal on
the observed OS amounts and cumulative PD amounts, each type with its own
σ: `y_OS,it ~ N(OS_i(t), σ_OS²)`, `y_PD,it ~ N(PD_i(t), σ_PD²)`.

**Likelihood, model 2** (PR C): log-normal on the OS loss ratio and the
incremental paid loss ratio, `log(y/Π_i) ~ N(log(μ/Π_i), σ_type²)` with
`μ` the OS level or `PD(t) − PD(t − 1)`, and random effects by origin and
by development year on every parameter. One solve gives all ages, so the
increment is a difference of one closed form at two ages, not of two
solves as in the paper's `claimsprocess`. A non-positive observed
increment or OS has no log; PR C returns `Error::Bayes` naming the cell,
as `BayesOdp` does for a non-positive row sum.

**Default priors.** The paper's case study, model 1 (sec. 5.2.1, code
7.2.2), as the preset `CompartmentalPriors::gesmann_morris()`:

| Parameter | Prior | Origin effect sd prior |
|---|---|---|
| `k_er` | `3 e^{0.1 o}` | none |
| `k_p` | `1 e^{0.1 o}` | none |
| `RLR` | `0.7 e^{0.2 o}` | `τ₁ ~ student_t(10, 0, 0.2)`, half |
| `RRF` | `0.8 e^{0.1 o}` | `τ₂ ~ student_t(10, 0, 0.1)`, half |
| `ρ` | `LKJ(1)`, uniform on (−1, 1) for two effects | |
| `σ_OS`, `σ_PD` | `student_t(1, 0, 1000)`, half (half-Cauchy) | |

The σ prior is on amounts, so its scale depends on the currency unit; PR
B checks the unit of the paper's data (code 7.2.1) and states the preset's
unit. Model 2's preset (PR C) takes the code's values, sd priors 0.3 on
`k_er` and `k_p`, 0.7 on `RLR`, 0.5 on `RRF`, and `σ ~ N(log 0.2, 0.2)` on
the log link; the text says a median of 0.1, so the code wins for parity
and `knowledge/` records the disagreement. Whether a fit may run on
defaults at all is open decision 2: `reserving-bayes.md` decision 1 said
these priors are not ones "a library default should choose".

Each positive parameter's prior is a `PositivePrior`, `LogNormal { median,
sd }` (the paper's form) or `Gamma { shape, rate }` (the blog's form),
both mapped to the non-centred scale through their quantile function, so
the blog's priors (gamma (4, 5) on `RLR` and `RRF`, (3, 2) on `k_er`, (3,
4) on `k_p`) can be expressed for the second check in decision 7.

**Identification.** Four facts, the first derived for this note, checked
numerically at the paper's example values (`k_er` 1.7, `k_p` 0.5, `RLR`
0.8, `RRF` 0.95; OS and PD agree to rounding at ages 1, 2, 5 and 10,
2026-10-13) and tested in PR B:

* **The OS and PD means are exactly invariant under a swap.** The map
  `(k_er, k_p, RLR_i, RRF_i) → (k_p, k_er, RLR_i k_er/k_p, RRF_i k_p/k_er)`
  leaves `OS_i(t)` and `PD_i(t)` unchanged for every `t` and every `i`:
  `g` is symmetric in the two rates, `RLR_i k_er` is kept, and so is
  `RLR_i RRF_i`. So even with both OS and paid observed the likelihood has
  two modes; the sources' remark that "OS breaks the symmetry" holds only
  with `RLR` and `RRF` fixed. What separates the modes is the prior: with
  the paper's medians 3 and 1 and log sd 0.1, `k_er < k_p` is 7.8 sd away
  (`log 3 / √0.02`), so the swapped mode has negligible prior mass. A unit
  test checks the invariance to 1e-12 at random points.
* **From paid alone only the product `RLR × RRF` is identified**, and the
  rates swap with no penalty, so a paid-only fit needs both an ordering
  and a prior on the expected loss ratio. PR B requires the outstanding
  column.
* **The swap's fixed point `k_er = k_p` is a removable singularity**,
  handled by the kernel `g` above.
* **Hierarchical shrinkage** pulls young origins' `RLR_i` to the mean
  (the paper's reason for model 3's cycle indices); `τ₁`'s prior controls
  how hard.

How to guard the swap by default is open decision 4: keep the paper's
independent priors and warn when any draw has `k_er ≤ k_p`, or sample an
ordered pair `k_er = k_p + e^δ`. Under the paper's priors the warning
never fires and the two give the same posterior within Monte Carlo error;
they differ only when a user widens the rate priors.

### 3. The ODE strategy

Proposed. **Closed form wherever one exists**, a generic fixed-step solver
otherwise, and an ODE crate only for a stiff system:

* **One OS** (models 1 and 2): the closed form of decision 2. No solver.
* **Two OS** (PR D): Gesmann and Morris' Eq. 3. The authors advise a
  solver because the closed form loses precision when two rates are close;
  PR D writes it, like `g`, in `expm1` kernels with series branches, and
  tests it against the RK4 solve below to 1e-8 relative at every observed
  age, over a grid that includes equal rates.
* **Anything else** (multistage gamma-pdf exposure, Eq. 4; a reporting
  rate that grows with time): a generic classical RK4 in
  `crates/prospicio-reserving/src/bayes/ode.rs`, generic over `Scalar`
  (decision 4), so the gradient comes from duals through the solver, as in
  the benchmark of `reserving-bayes.md` decision 15 (9e-10 against the
  closed form at 1.6 to 2.2 times a plain solve).
  * **Fixed steps, fixed per fit.** The number of steps per year is
    chosen once from the prior (`n = max(100, ⌈20 k_max⌉)` with `k_max`
    the 99.9th prior percentile of the fastest rate) and does not change
    with the parameters, so the density is a smooth function of them.
    Adaptive step control makes the gradient discontinuous: Gesmann's
    PK/PD post reports `integrate_ode_rk45` at about 9 hours a chain and
    235 divergences, against minutes and none for a closed form.
  * **Accuracy target**: cumulative PD and OS within 1e-8 relative of the
    closed form at every observed age, over the prior's central 99.8%, on
    the one-OS and two-OS models. RK4's global error is of order
    `(k h)⁴`, so `k h ≤ 0.05` gives about 1e-9 relative.
  * **Observed ages fall on steps** (integer years times `n`), so no
    interpolation is needed.
  * **A singular source** (the gamma-pdf exposure has `t^{d_r − 1}`,
    singular at 0 when `d_r < 1`) is kept regular as the paper's code
    keeps it, `d_r = 1 + 0.1 e^{0.5 o} > 1`.
* **`diffsol` behind its own feature** only if a model needs `k h` above
  RK4's stability limit (about 2.8) at a practical step count, that is,
  rates that differ by orders of magnitude. None of the planned models
  does.

### 4. Gradients: the `Scalar` trait and the dual type (PR A)

Proposed, following `reserving-bayes.md` decision 15 (decided). PR A adds,
with no new model:

* **`crates/prospicio-reserving/src/bayes/scalar.rs`**: `pub(crate) trait
  Scalar`, implemented for `f64` and the dual type: `Copy`, the four
  arithmetic operators with `Self` and with `f64`, `from_f64`, `value`,
  and the elementary functions the densities need (`exp`, `expm1`, `ln`,
  `ln_1p`, `sqrt`, `powf`, `tanh`, `lgamma` where a prior needs it, and
  the normal log cdf for `Meyers`' `gamma_cdf`-style terms if used). The
  trait is crate-private until a second crate needs it; moving it to
  `prospicio-bayes` would be a small shared PR of its own.
* **`crates/prospicio-reserving/src/bayes/dual.rs`**: a forward-mode
  `Dual<const N: usize>` (value and `N` tangents, about 130 lines as in
  the benchmark) and a `gradient` driver that, for a runtime parameter
  count `p`, runs `⌈p / N⌉` chunked passes, each seeding `N` tangents.
  `N` is fixed in the crate (the measurement below picks 8, 16 or 32).
* **The densities made generic.** `BayesOdp`, `BayesBf` and `Meyers`
  (CRC, CSR, CAY) each get one `log_density<S: Scalar>` that is the single
  source of the value; the `f64` instance must give the existing value bit
  for bit (the existing parity suites run unchanged), and the hand-written
  gradients stay as the fast path.
* **Oracle tests.** For each of those densities, the chunked dual
  gradient matches the hand-written gradient to 1e-12 relative at 20
  random points per model (drawn from the posterior's scale, not only the
  mode); each elementary function's dual derivative matches its analytic
  derivative, including the series branches; and the one-compartment
  closed form's dual gradient matches central differences to 1e-7. The
  hand-written gradients are the oracle because each already passes a
  finite-difference test and a Stan parity suite.
* **The measurement decision 15 owes**: time per value and gradient
  (release, median) of the chunked dual at `N` = 4, 8, 16 and 32 against
  the fixed-size dual and the hand-written gradient, at `p` = 29 (Meyers'
  CRC and the one-OS model) and `p` = 60 (model 2's size). Recorded in
  `knowledge/findings/autodiff-benchmark.md`. Acceptance: a gradient at
  `p` = 29 within 10 times the hand-written one; otherwise PR B writes the
  one-OS gradient by hand (the closed form makes that practical) and keeps
  duals for the oracle and the RK4 models.

Laplace Hessians stay central differences of the exact gradient
(`bayes/laplace.rs`, `Posterior::negative_hessian`), as decided.

### 5. The fit: whitening, pilot run and predictive

Proposed. The fit follows `Meyers::fit` (`meyers.rs`), which already
handles hierarchical and funnel geometry:

1. a regularised Laplace whitening (`Whitened::regularised`) at the start
   point, all `o`, `z` and `atanh ρ` at 0, `log τ` and `log σ` at their
   prior medians;
2. a pilot NUTS run, then `Whitened::from_draws`;
3. the main run, with `LogDensity` returning `None` on a non-finite value
   (an overflowing exponential, a solver failure), which `prospicio-bayes`
   counts as a divergence;
4. the predictive: for each draw, the mean OS and PD at each future cell
   of the square, plus process error from the likelihood (normal on
   amounts in model 1; log-normal increments in model 2), giving a joint
   `PredictiveDistribution` of each origin's reserve.

Defaults: 4 chains, 1,000 tuning and 1,000 kept draws each,
`target_accept = 0.99` and `max_depth = 12` (the paper uses
`adapt_delta` 0.99 and `max_treedepth` 15; `Meyers` uses 12, and a
deeper tree is a setting), `n_sims = 10_000`, `seed = 0`. Seeds as
`reserving-bayes.md` decision 5: derived streams for the chains, the pilot
(`pilot_seed`) and the simulations, exact replay whatever the thread
count.

**The reserve** is future paid to the last age of the square (age 10 for
company 337), as `Meyers` and the paper's validation; the model's own
ultimate `Π_i RLR_i RRF_i` is reported alongside as a mean, without
process error past the square. A reserve to ultimate is open decision 5.

### 6. The API

Proposed:

```rust
#[cfg(feature = "bayes")]
pub mod bayes {
    pub enum Outstanding<'a> {
        Column(&'a str),        // an outstanding column
        FromIncurred(&'a str),  // incurred minus paid
    }

    pub enum CompartmentalLikelihood { Normal, LogNormal }   // model 1, model 2 (PR C)

    pub struct CompartmentalPriors {
        pub k_er: PositivePrior, pub k_p: PositivePrior,
        pub rlr: PositivePrior,  pub rrf: PositivePrior,
        pub rlr_effect_sd: HalfPrior, pub rrf_effect_sd: HalfPrior,
        pub lkj_eta: f64,
        pub sigma: [HalfPrior; 2],       // OS, PD
    }

    pub struct Compartmental {
        pub priors: CompartmentalPriors,     // open decision 2
        pub likelihood: CompartmentalLikelihood,
        pub rate_order: RateOrder,           // open decision 4
        pub sampler: prospicio_bayes::Sampler,
        pub n_sims: usize,
        pub seed: u64,
    }

    impl Compartmental {
        pub fn fit(&self, triangle: &Triangle, paid: &str,
                   outstanding: Outstanding<'_>, premium: &str)
                   -> Result<CompartmentalFit>;
        pub fn prior_predictive(&self, triangle: &Triangle, premium: &str,
                                n_sims: usize) -> Result<CompartmentalPaths>;
    }
}
```

* **Inputs.** One triangle holding the paid column, the outstanding (or
  incurred) column and premium by origin, read through `origin_exposure`
  as `Meyers::fit` reads it. Whether OS includes bulk and IBNR is the
  user's choice of column; the docs show both on company 337 and say which
  the papers use.
* **`CompartmentalFit`**: `posterior` (`PosteriorDraws` on the natural
  scale: `k_er`, `k_p`, `RLR`, `RRF`, `τ`, `ρ`, the σs and every origin's
  `RLR_i` and `RRF_i`), `diagnostics` (`bayes::Diagnostics`, thresholds
  as before, warnings not errors, as `reserving-bayes.md` open decision
  4), `summary()`, `loo()` (pointwise over both the OS and the paid
  cells, with the elpd also split by type), `reserves` (the joint
  `PredictiveDistribution` by origin, paid), `paid` and `outstanding`
  (predictive means and quantiles for each future cell, as long frames),
  `ultimate_loss_ratio` by origin, `to_long()`, `totals()`.
* **`prior_predictive`** draws OS and paid paths from the priors alone,
  so a user can check a prior set before fitting, as the paper does in
  sec. 3.4 to test identifiability.
* **Errors.** A missing column or premium, a non-positive premium, and a
  triangle with no outstanding cell are `Error::Bayes` naming the origin
  or column; bad prior settings are `InvalidSetting`.
* **Segments and the backtest** wait for PR E (decision 8).

### 7. Validation

Proposed, on the pattern of `reserving-bayes.md` decision 10.

* **Reference programs**: `validation/scripts/stan/compartmental_gm1.stan`
  (PR B) and `compartmental_gm2.stan` (PR C), the paper's models 1 and 2
  written as plain Stan from its brms code (sec. 7.2.2 and 7.2.3): the
  same closed form, priors, non-centred effects and `lkj_corr_cholesky`,
  with provenance in the header and only the syntax changed. PR D adds
  `compartmental_two_os.stan` with `ode_rk45` in place of
  `integrate_ode_rk45`.
* **Generator**: `validation/scripts/reserving_compartmental_stan.py`
  under `uv` with cmdstanpy 1.3.0 against CmdStan 2.39.0 (from PowerShell
  or cmd, with RTools' make; `knowledge/environment/local-cmdstan.md`).
  Fixed seed, 4 chains of 2,000 warm-up and 25,000 draws,
  `adapt_delta` 0.99, `max_treedepth` 15. It fails on any R-hat above
  1.005 or any divergence, and writes the CmdStan version, the `.stan`
  file's SHA-256 and the seed into `source`.
* **Training data as the paper**: calendar years up to 1996 (sec. 5.1,
  code 7.2.1), so the 1997 diagonal is the test set and the full square
  the validation. PR B builds that triangle from the square by dropping
  the 1997 diagonal (the backtest's own helper if it exposes one; a
  training CSV from the generator otherwise).
* **Reference CSV**: `validation/reference/reserving_compartmental_stan.csv`
  in the usual format. Rows: mean and sd of every natural-scale
  parameter, including each origin's `RLR_i` and `RRF_i`; each origin's
  and the total reserve's mean, sd and 5th, 50th and 95th percentiles;
  the 1997 diagonal's predictive means; elpd_loo. CI reads the committed
  CSV and never runs Stan.
* **Tolerances** from a seed sweep, as PR 3 of `reserving-bayes.md`: our
  defaults over seeds 0 to 199; each tolerance at least 4 times the
  seed-to-seed sd of its row (and of the combined errors where a printed
  number is the target). A row whose seed-to-seed error is heavy-tailed
  (the youngest origin's reserve sd was one for `Meyers`) is dropped and
  checked through its percentiles, and the note says so.
* **Printed numbers, looser** (Monte Carlo targets, brms 2.10.0 on rstan
  2.19.2, so never bit-exact): for model 2 (PR C) the standardised
  intercepts `oRLR` 1.44 (sd 0.26), `oRRF` −1.36 (0.42), `oker` −5.40
  (0.81), `okp` −8.49 (0.36), within 0.15 printed sd for the means and
  15% for the sds. For model 1 (PR B) the paper prints no table we can
  match; the blog's brms means (`k_er` 1.55, `RLR` 1.03, `k_p` 0.45, `RRF`
  0.67; origin effect sds 0.25 and 0.11) are a second check under the
  blog's gamma priors, if PR B confirms its data split and likelihood
  match model 1; otherwise that check is dropped and the note says so.
  The final reserve (mean 123 against an actual 130, USD millions) is
  model 3's and is recorded, not checked.
* **Sanity tests** (unit, no reference): the swap invariance and the
  equal-rate limit (decision 2); the closed form against RK4 (decision 3);
  the dual gradient against central differences; exact replay for a seed
  and in a one-thread pool; the posterior mean ultimate loss ratio on
  company 337 against the chain ladder's on incurred (a report, not a
  bound, since the models differ).

### 8. Segments and the backtest (PR E)

Proposed. `Compartmental::fit_segments(triangle, paid, outstanding,
premium)` with `SegmentDependence` (independent or rank-correlated;
`Synchronized` is `Error::Bayes`) and per-segment seeds, as `BayesOdp`'s
decision 11, and `TriangleModel for Compartmental` in the diagonal
backtest. Both follow the Meyers segments branch
(`claude/bayes-meyers-segments`, not merged), which solves the same
problem for a model that needs premium and a second column; PR E starts
after it merges. Python and R get `fit_segments` in PR E; the backtest
stays Rust-only, as for `BayesOdp`.

### 9. Bindings

Proposed, in the same PR as each model (`reserving-bayes.md` open
decision 10 (a)):

* **Python**: `Compartmental(priors, likelihood="normal", chains=4,
  tune=1000, draws=1000, n_sims=10000, seed=0, target_accept=0.99,
  max_depth=12)` with `fit(triangle, paid, premium, outstanding=None,
  incurred=None)` (exactly one of the last two), returning a fit with
  `posterior`, `diagnostics`, `summary()`, `loo()`, `reserves`,
  `to_frame()`, `totals_frame()`, `paid_frame()` and `outstanding_frame()`;
  priors as `CompartmentalPriors(...)` with
  `CompartmentalPriors.gesmann_morris()`, and `prior_predictive(...)`.
* **R**: `compartmental(triangle, paid, premium, outstanding = NULL,
  incurred = NULL, priors = compartmental_priors_gm(), likelihood =
  c("normal", "lognormal"), ...)` with the same sampler arguments in the
  same order, returning a `compartmental_fit` with `as.data.frame()`,
  `totals_frame()`, `summary()` and `bayes_loo()`, and
  `compartmental_prior_predictive()`.
* Failed diagnostics raise one warning in each language, as the other
  Bayesian fits. Tests: `python/tests/test_reserving_compartmental.py` and
  `R/prospicio/tests/test-reserving-compartmental.R` fit company 337 with
  the default chains (no warning, the frames, totals close to the Rust
  values), replay by seed, and the errors. Docs regenerated with
  `cargo xtask docs`.

### 10. Docs and knowledge

Proposed. Each PR updates in itself: the Reserving row of AGENTS.md, the
v0.7 row and the Reserving section of `docs/roadmap.md`,
`docs/architecture.md` if a roadmap row changes, `CHANGELOG.md`, and this
note's status. `knowledge/` gains: company 337's data and its agreement
with Morris' printed table, with both OS definitions (datasets, PR B); the
swap invariance of the one-OS means and the stable kernel `g` (findings,
`verified` once PR B's test runs in CI); the paper's text and code
disagreement on model 2's σ prior and the transcription notes for its
brms code (references); the chunked-dual measurement (PR A, in the
existing autodiff finding). A user guide page on company 337 (paid and OS
fitted jointly, the prior predictive check) comes with PR C, once both
likelihoods exist.

## Risks

* **Prior sensitivity.** The posterior leans on tight priors; on a book
  unlike company 337 the paper's preset may be wrong, and the fit will
  still look healthy. The prior predictive check and open decision 2 are
  the mitigation.
* **Sampler geometry.** Funnels in the origin effects (worse in model 2,
  with development effects too) and the swap. The non-centred effects,
  the pilot whitening and `target_accept` 0.99 are the mitigation; a
  reference run with divergences at `adapt_delta` 0.99 would mean
  tightening, and the note would record it.
* **The reference is a transcription.** A brms detail missed in the
  transcription (its default intercept centring, its `sd` priors'
  truncation) shifts both sides together and hides from the parity test;
  the printed posteriors in decision 7 are the guard. If they disagree
  beyond the looser tolerance, open decision 6 (b) is the fallback.
* **Model misfit.** The papers state the limits: a non-steady exposure is
  absorbed into `k_er`; cohorts are assumed homogeneous, so large-claim
  books fit poorly; short histories are not credible. A change in case
  reserve strength (company 337 in 1996 and 1997) makes incurred data
  misleading. The docs say so and the backtest of PR E measures it.
* **Dual cost.** If PR A's measurement fails its target, PR B's gradient
  is written by hand (decision 4), which adds a finite-difference test
  but no risk to the result.

## Open decisions for the user

Each has a recommendation. All are Proposed; nothing past PR A is built
until 1 to 4 are settled.

1. **First model.** (a) Gesmann and Morris' model 1 (Gaussian, origin
   effects on `RLR` and `RRF`) in PR B, then model 2 in PR C, then the
   two-OS model in PR D; (b) model 2 first, the paper's preferred
   likelihood; (c) Morris (2016) with AR(1) residuals. *Recommended: (a)*,
   decision 1: the smallest model with the paper's own code and priors,
   no solver, and everything later builds on it. Choose (b) if a model
   that cannot predict negative OS matters for the first release.
2. **Default priors.** (a) Priors are a required argument, with the
   paper's case study as a named preset (`gesmann_morris()`) and the docs
   showing a prior predictive check; (b) the preset as a silent default;
   (c) data-centred defaults (`RLR × RRF` at the chain ladder's incurred
   ultimate loss ratio), documented as using the data twice.
   *Recommended: (a)*: the rates and ratios are line-specific, and
   `reserving-bayes.md` decision 1 already judged that a library should
   not choose them silently. Parity tests use the preset.
3. **Data and the OS definition.** (a) Company 337 workers' compensation
   from the CAS data model mart (the source decided for group 353), with
   both OS definitions and both premiums in the CSV; parity on the papers'
   OS (with bulk and IBNR) and direct premium; (b) the `raw` R package's
   `wkcomp` (not installed; a download). *Recommended: (a)*: same source,
   pinned release and checks as group 353, and Morris' printed table
   agrees with it.
4. **Guarding the rate swap.** (a) The paper's independent rate priors,
   with a warning when any draw has `k_er ≤ k_p`; (b) an ordered pair
   `k_er = k_p + e^δ` by default; (c) (a) by default with (b) as a
   `rate_order` setting. *Recommended: (c)*: (a) keeps the parity with
   the paper exact, the warning never fires under its priors, and (b) is
   there for a user who widens them.
5. **Reserve horizon.** (a) Future paid to the last age of the square,
   with the model's ultimate reported as a mean; (b) also a reserve to
   ultimate with process error past the square. *Recommended: (a) now*,
   as `Meyers` and the paper's validation; (b) on request.
6. **Reference route.** (a) The paper's brms code transcribed into plain
   Stan and run with CmdStan 2.39.0 under cmdstanpy, as Meyers' code was;
   (b) install brms and cmdstanr in R 4.5.3 to run the published code as
   is (package downloads, needs the user's approval); (c) printed numbers
   only. *Recommended: (a)*, with the printed posteriors as a looser
   check, and (b) only if the transcription misses them.
7. **Gradients for the closed-form models.** (a) Chunked duals
   (`reserving-bayes.md` decision 15), with a hand-written gradient only if
   PR A's measurement misses its target; (b) hand-written from the start,
   duals as the oracle. *Recommended: (a)*: one generic density per model
   is the point of decision 15, and the closed form keeps (b) cheap if
   needed.
8. **Extensions.** (a) Multistage exposure, market cycles, AR(1) residuals
   and calendar shocks each as a later PR on request; (b) plan them now.
   *Recommended: (a)*: none has a reference we can rerun yet, and each is
   a model of its own.
