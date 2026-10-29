# Design note: compartmental reserving (Gesmann and Morris)

Status: **Decided** (2026-10-13; the user settled the open decisions at
the end, all as recommended, with decision 1 widened: see there). PR A
built (decision 4); PR B built (decisions 2, 5, 6, 7 and 9, each with its
"Built" paragraph: the Rust core and its Python and R bindings); PR C
built (decisions 1, 2, 4, 5, 6, 7 and 9, each with its "Built (PR C)"
paragraph: the Rust core and its Python and R bindings); PR F (the
one-year view, decision 11) built, with the defaults the user decided on
2026-10-16; PR E (segments and the backtest, decision 8) built, with the
user's decisions of 2026-10-19, and its one-year view over segments
(`Compartmental::one_year_segments`, decision 8) built after it; PRs G
and H (paid and incurred, and one response alone, decision 12) built;
PR I (unequal origin and development grain, decision 13) built; PR D not
yet · Lane:
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
| A | `claude/bayes-dual` | the `Scalar` trait and the in-house dual type (decision 4); a generic density beside each existing one, with oracle tests against the hand-written gradients; the chunked-pass cost measurement that `reserving-bayes.md` decision 15 still owes. No new model, no binding change | `main`, after the Meyers segments branch merges (it changes `meyers.rs` by about 1,500 lines) |
| B | `claude/bayes-compartmental` | `Compartmental` with one OS compartment and Gesmann and Morris' model 1 likelihood (decisions 1 to 3, 5 to 7); company 337's data; the Stan reference; Python and R bindings | PR A |
| C | `claude/bayes-compartmental-lognormal` | model 2's likelihood (log-normal on outstanding and incremental paid loss ratios) and its development-year random effects (decision 2) | PR B |
| D | `claude/bayes-compartmental-two-os` | the two-OS model (fast and slow settlement), the generic RK4 solver as its oracle (decision 3) | PR B |
| E | `claude/bayes-compartmental-segments` | `Compartmental::fit_segments` with `SegmentDependence`, and `CompartmentalCandidate` for the diagonal backtest with a held-out outstanding table (decision 8); Python and R `fit_segments`. **Built** | PR B, and the Meyers segments branch merged |
| F | `claude/bayes-one-year-compartmental` | the one-year view of models 1 and 2 through `YearModel` (decision 11; `reserving-bayes.md` decision 20), a refit budget for every model's view; Python and R bindings | the Meyers one-year PR (#280) |
| G | `claude/compartmental-paid-incurred` | `Fitted::PaidAndIncurred`: incurred replaces outstanding as the second response (decision 12); its own Stan reference; Rust, Python and R. **Built** | `main` |
| H | `claude/compartmental-one-response` | `Fitted::PaidOnly` and `Fitted::IncurredOnly`, which require `RateOrder::Ordered` (decision 12); their own Stan references; Rust, Python and R. **Built** | PR G |
| I | `claude/compartmental-grain` | unequal origin and development grain: the development grain finer than the origin grain, time in years, the square in development periods (decision 13); its own Stan reference on a simulated triangle; Rust, Python and R. **Built** | PR H |
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
| One-OS model, model 1 (Gaussian on amounts) | Gesmann and Morris (2020), "Hierarchical Compartmental Reserving Models", CAS Research Paper: Eq. 1 and 2, sec. 5.2.1, brms code in sec. 7.2.2 (R, with the closed form as Stan function snippets), reconstructed as the Stan program brms generates and run with CmdStan 2.39.0 | its printed posterior (`m1fit`, sec. 7.2.2; decision 7) |
| Model 2 (log-normal on loss ratios, incremental paid) | the same paper, sec. 3.3 and 5.2, brms code in sec. 7.2.3, reconstructed the same way | its printed posterior (`m2fit`, sec. 7.2.3; decision 7) |
| Two-OS model | the same paper, Eq. 3 (closed form) and sec. 2.3 | our RK4 solve of the ODE system |
| An earlier fit of company 337 | Gesmann's blog post "Hierarchical compartmental reserving models" (8 January 2018): nls, nlme and brms fits of company 337 | not a check (decision 7 says why) |
| Background | Morris (2016), "Hierarchical Compartmental Models for Loss Reserving", CAS E-Forum Summer 2016 (nlme, OpenBUGS) | the CAS blog "Unboxing hierarchical compartmental reserving models" (2022); IFoA GIRO "Introducing compartmental reserving" |
| Diagnostics and model comparison | `prospicio_bayes` (`rhat`, `ess_bulk`, `ess_tail`, `elpd::loo`), already at parity with R `posterior` and `loo` | |

No package in R `ChainLadder` or chainladder-python fits these models, so,
as in `reserving-bayes.md` decision 10, the reference is the same model
written in Stan. The paper prints brms R code with Stan function
snippets, not Stan programs, fitted with brms 2.10.0 on rstan 2.19.2 (sec.
7.3 is only the session information). So the reference reconstructs, with
provenance and without brms installed, the Stan program brms would
generate from that code (no licence is stated, so nothing is copied). The
reconstruction must reproduce these brms conventions: the nonlinear
parameters `o` with `normal(0, 1)` priors and no intercept centring
(`nl = TRUE`); the group sds as half Student-t (class `sd`, bounded at 0)
on the `o` scale; group effects non-centred, `r = (diag(sd) L z)'` with
`z ~ std_normal()` and `L ~ lkj_corr_cholesky(1)`, correlated only within
an `|ID|` term (model 1: `oRLR` and `oRRF` by accident year; model 2 adds
uncorrelated accident-year effects on `oker` and `okp` and uncorrelated
development-year effects on all four); one sd prior per `nlpar` shared by
its groups; and `sigma ~ 0 + deltaf` on the log link, so the `sigma` priors
are priors on `log σ` (class `b`, unbounded). brms, rstan, cmdstanr and the
`raw` package are not installed locally (open decision 6).

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

Built (PR B): as above. The generator asserts the 1988 cell and every cell
of Morris' printed upper triangles (calendar years up to 1997: premium,
outstanding and cumulative paid, headed $'000s but in USD millions) to
within 0.5 of the printed figure; all 55 of each pass, with the papers'
outstanding and direct premium only (`knowledge/datasets/clrd-wkcomp-337.md`).

## Decisions

### 1. Which model ships first

Decided 2026-10-13 by the user: model 1 first, then model 2, then the
two-OS model, **with the API flexible from PR B on**. The user asked that
the likelihood and the response be settings, not separate models: the
likelihood (normal, log-normal, and later others such as gamma), the
response scale (amounts, or loss ratios to premium), the paid response
(cumulative or incremental), and the pair fitted (paid with outstanding,
or paid with incurred, or one alone). PR B builds model 1 as one point of
that grid, with the settings in place and the other values returning
`Error::Bayes` ("not yet") until their PR adds them; PR C fills in the
log-normal, loss ratios and incremental paid. Each value that ships gets
its own Stan reference.

**Built (PR C, Rust core).** The log-normal likelihood, loss ratios and
incremental paid are built, and every combination of the three settings
is allowed (8 in all), with the same density code: the response of a cell
is its outstanding, cumulative or incremental paid, divided by premium on
the loss-ratio scale, and the likelihood normal on it or on its log. Each
value has its Stan reference (normal, amounts and cumulative paid in
`compartmental_gm1.stan`; log-normal, loss ratios and incremental paid in
`compartmental_gm2.stan`); the mixed combinations have no reference of
their own and are checked by a unit test against the model written out in
`f64` with the paper's closed form, for both presets' hierarchies (1e-12).
`Fitted` takes paid and outstanding, since PR G paid and incurred, and since PR H paid alone or incurred alone with ordered rates (decision 12). The hierarchy is not a
setting of its own but follows from the priors (decision 6):
`CompartmentalPriors::gesmann_morris_2()` is model 2's, and
`Compartmental::model_2(priors)` sets model 2's likelihood, scale, paid
response and sampler.

The analysis that led here, as proposed: Four candidates:

| Candidate | Reference | Solver | Sampler difficulty |
|---|---|---|---|
| Gesmann and Morris one-OS, model 1: Gaussian on OS and cumulative PD amounts, origin effects on `RLR` and `RRF` | printed priors, brms code and posterior (sec. 7.2.2); company 337 in the mart | closed form (Eq. 2) | moderate: 27 parameters on the paper's training triangle; the data pull the rates far from their tight priors (decision 2) |
| Gesmann and Morris one-OS, model 2: log-normal on OS and incremental paid loss ratios, origin and development effects on every parameter | printed priors, brms code (sec. 7.2.3) and printed posterior | closed form | harder: 87 parameters on the paper's training triangle (decision 2), funnels, and the paper's text and code disagree on two priors |
| Gesmann and Morris two-OS | the closed form (Eq. 3); no printed fit we can rerun | closed form, but the authors prefer a numerical solver | harder: a cascade of two OS compartments whose symmetries PR D derives |
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
series branch for `|x| < 1e-2` (`Σ (−x)^k/(k+1)!` to `k = 6`; PR A's
`exprel_neg` in `bayes/scalar.rs`), so the density and its
dual gradient are finite and smooth through `k_er = k_p`. The threshold is
set by the dual derivative, not the value: the quotient's derivative loses
about `2ε/|x|` relative, 3e-12 at the `1e-5` first proposed here with a
three-term series, 2e-14 at `1e-2`, where seven terms keep the series'
value and derivative exact to rounding. At `k_er = k_p`,
`g(t) = t e^{−k t}`. The PD form follows from the paper's by
`k_er (1 − e^{−k_p t}) − k_p (1 − e^{−k_er t}) = Δ (1 − e^{−k_p t}) −
k_p Δ g(t)`. The ultimate loss ratio is `PD(∞)/Π_i = RLR_i RRF_i`, and OS
peaks at `log(k_p/k_er)/(k_p − k_er)`, the inflection of paid.

**Hierarchy** (model 1, as the paper's code): the rates are shared, `RLR`
and `RRF` vary by origin with correlated effects on the standardised `o`
scale, which the preset's multipliers then scale.

```text
o_RLR,i = b_RLR + u_i1,   RLR_i = 0.7 e^{0.2 o_RLR,i}
o_RRF,i = b_RRF + u_i2,   RRF_i = 0.8 e^{0.1 o_RRF,i}
u_i = diag(τ) L z_i,   z_i ~ N(0, I₂),   L L' = Ω,   Ω = [[1, ρ], [ρ, 1]]
b ~ N(0, 1) for each of b_RLR, b_RRF, b_ker, b_kp
```

So `log RLR_i = log 0.7 + 0.2 (b_RLR + u_i1)`, and `τ₁`, `τ₂` are sds on
the `o` scale: an effect sd of `τ₁` on `o` is `0.2 τ₁` on `log RLR` and
`0.1 τ₂` on `log RRF`. The paper's printed `sd(oRLR)` 0.64 and `sd(oRRF)`
0.74 are on this scale. The paper's text (sec. 5.2.1) writes the effect
differently, `RLR[i] ~ μ_RLR + u_RLR` with `u` multivariate normal and the
`StudentT(10, 0, 0.2)⁺` sd on that `u`; the code wins for parity, as for
model 2's σ (below), and `knowledge/` records both disagreements
(decision 10).

**Sampler scale.** Every positive parameter is non-centred log-normal,
`θ = m e^{s o}` with `o ~ N(0, 1)`, as the paper (Stan's default inits in
(−2, 2) break positive parameters on their natural scale). The sampler
sees `b_ker`, `b_kp`, `b_RLR`, `b_RRF`, `log τ₁`, `log τ₂`, `atanh ρ`, the
`z_i` of each origin with an observed cell, and `log σ_OS`, `log σ_PD`,
with the log Jacobians of the `τ` and `ρ` maps (the σ priors are on
`log σ` already, so `log σ` needs none). On the paper's training triangle
(9 origins with data, below) that is 4 + 2 + 1 + 18 + 2 = 27 parameters,
close to Meyers' CRC (29).

**Model 2's size** (PR C): accident-year effects on all four `o` (9 levels
each; `oRLR` and `oRRF` correlated, `oker` and `okp` not) and
development-year effects on all four (9 levels each, uncorrelated): 72
`z`, 8 sds, 1 correlation, 4 intercepts and 2 `log σ`, 87 parameters (95
with 10 levels of each). PR A benchmarks at that size (decision 4).

**Origins and development years with no observed cell.** The paper's
training triangle (calendar years before 1997) has no cell for accident
year 1997, and brms reports 9 accident-year levels and 90 observations; in
model 2 development year 10 has no training cell either (9 levels). Their
effects are not identified by the data. Decision: such an origin (or, in
model 2, development year) has no `z` in the sampler; in the predictive
step each draw gives it a new effect from the population, `z ~ N(0, I)`
scaled by that draw's `τ` and `L`, so its reserve and its predicted cells
(1997 at age 1 in the test diagonal) carry the hierarchy's uncertainty.
brms leaves this to `sample_new_levels`, whose default reuses a randomly
chosen existing level's effect per draw; the Stan reference instead does
exactly our rule in generated quantities, so the two sides match. The fit
accepts an origin with no observed cell as long as it has a positive
premium and at least one origin has data (unlike `Meyers::fit`, which
requires every origin from the first age), and reports which origins and
development years were drawn from the population.

**Likelihood, model 1** (the paper's sec. 3.1 and code 7.2.2): normal on
the observed OS amounts and cumulative PD amounts, each type with its own
σ: `y_OS,it ~ N(OS_i(t), σ_OS²)`, `y_PD,it ~ N(PD_i(t), σ_PD²)`.

**Likelihood, model 2** (PR C): log-normal on the OS loss ratio and the
incremental paid loss ratio, `log(y/Π_i) ~ N(log(μ/Π_i), σ_type²)` with
`μ` the OS level or `PD(t) − PD(t − 1)` (cumulative `PD(1)` at age 1), and
random effects by origin and by development year on every parameter. So
each cell `(i, t)` has its own `(k_er, k_p, RLR, RRF)`, and, as the paper's
`claimsprocess` does, the increment evaluates the closed form twice with
that cell's parameters, `PD(t; θ_it) − PD(t − 1; θ_it)`. It is not a
difference of one per-origin curve at consecutive ages, nor `PD(t − 1)`
at development year `t − 1`'s parameters; both give a different
likelihood. The predictive step uses the same rule. A non-positive observed
increment or OS has no log; PR C returns `Error::Bayes` naming the cell,
as `BayesOdp` does for a non-positive row sum.

**Default priors.** The paper's case study, model 1 (sec. 5.2.1, code
7.2.2), as the preset `CompartmentalPriors::gesmann_morris()`:

| Parameter | Prior | Origin effect sd prior (on `o`) |
|---|---|---|
| `k_er` | `3 e^{0.1 o}` | none |
| `k_p` | `1 e^{0.1 o}` | none |
| `RLR` | `0.7 e^{0.2 o}` | `τ₁ ~ student_t(10, 0, 0.2)`, half |
| `RRF` | `0.8 e^{0.1 o}` | `τ₂ ~ student_t(10, 0, 0.1)`, half |
| `ρ` | `LKJ(1)`, uniform on (−1, 1) for two effects | |
| `log σ_OS`, `log σ_PD` | `student_t(1, 0, 1000)` (Cauchy) on `log σ`, unbounded | |

The σ prior is brms' class `b` prior on `sigma ~ 0 + deltaf` with the log
link, so it is a Cauchy(0, 1000) on `log σ`, effectively flat, and not a
half-Cauchy on σ; the text (sec. 5.2.1) says the same, `log(σ[δ]) ~
StudentT(1, 0, 1000)`. It adds no Jacobian. The printed posterior is `log
σ_OS` 8.25 and `log σ_PD` 6.66 (σ about 3,800 and 780 in the data's
units, USD thousands). Model 2's preset (PR C) takes the code's values, sd
priors 0.3 on `o_ker` and `o_kp`, 0.7 on `o_RLR`, 0.5 on `o_RRF`, and
`log σ ~ N(log 0.2, 0.2)`; the text says a median of 0.1, so the code wins
for parity and `knowledge/` records the disagreement. Whether a fit may
run on defaults at all is open decision 2: `reserving-bayes.md` decision 1
said these priors are not ones "a library default should choose".

Each positive parameter's prior is a `PositivePrior::LogNormal { median,
sd }` (the paper's form, `median e^{sd o}`). There is no gamma variant: its
only use was the blog's priors, whose check decision 7 drops, and it would
need a gamma quantile function with its dual derivative in `Scalar`. A
gamma or other prior can be added later as its own small change if a user
needs one.

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
  with `RLR` and `RRF` fixed. What separates the modes is the prior, but
  not because the posterior sits near the prior's centre. On company 337
  the paper's model 1 posterior has `o_ker` −5.40 and `o_kp` −8.49 (so
  `k_er` about 1.75 and `k_p` about 0.43 against prior medians 3 and 1),
  5.4 and 8.5 prior sds from the centre, and `o_RLR` 1.44 and `o_RRF`
  −1.36 about 1.4 sd off: the data overrule the preset, and the fit is a
  prior-data conflict, not a prior-identified fit. The swap is still
  excluded, by the joint penalty at the posterior: the swapped point needs
  `o_ker` about −19.4 and `o_kp` about 5.6, about 150 nats more prior
  penalty on the rates alone (`(19.4² + 5.6² − 5.4² − 8.49²)/2`), plus the
  shifts `RLR_i k_er/k_p` and `RRF_i k_p/k_er` impose on `o_RLR` (about +7)
  and `o_RRF` (about −14). A unit test checks the invariance to 1e-12 at
  random points.
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
independent priors and warn when a draw's population rates have `k_er ≤
k_p`, or sample an ordered pair `k_er = k_p + e^δ`. In model 1 the rates
are shared, so the guard is on them. In model 2 the rates vary by cell,
and a cell with `k_er ≤ k_p` is not a swap of the whole model (each cell
is invariant on its own), so the guard applies to the population
intercepts (`e^{0.1 b}` times the medians), and the fit reports, as a
diagnostic and not a warning, the share of draws and cells with
`k_er ≤ k_p`. Under the paper's priors the warning is expected never to
fire (above), and the two give the same posterior within Monte Carlo
error; they differ only when a user widens the rate priors.

**Built (PR B, Rust core).** As proposed: the 27 parameters on the
paper's training triangle in the order above, the density written once
over `Scalar` (`bayes/compartmental.rs`) and its gradient from one chunked
dual pass (`dual::CHUNK` = 32), with no hand-written gradient; the Laplace
Hessian is the default central differences of that exact gradient. The
closed form `one_os` moved from PR A's test support into the model. With
these additions and departures:

* **Priors.** `PositivePrior::LogNormal { median, sd }` only, as decided;
  `HalfPrior` is `StudentT { nu, scale }` or `Normal { scale }`, on `τ` with
  the log map's Jacobian; `RealPrior` is `StudentT { nu, location, scale }`
  or `Normal { mean, sd }`. The LKJ shape `lkj_eta` is general: for two
  effects its density in `atanh ρ`, with the Jacobian, is
  `eta log(1 - ρ²)`. A unit test checks the density at one point against
  the model written out in `f64` with the paper's form of the closed form
  (1e-12), and the dual gradient against central differences at 20 points
  around the posterior, for both rate orders.
* **The swap.** The invariance holds to 1.1e-14 at 200 random points
  (`the_means_are_invariant_under_the_rate_swap`;
  `knowledge/findings/compartmental-rate-swap.md`). Under
  `RateOrder::Free` the fit counts the draws with `k_er <= k_p`
  (`CompartmentalFit::rate_swaps`) and adds a diagnostics warning when
  there are any; none in 200 seeds on company 337. `RateOrder::Ordered`
  samples `delta` with `k_er = k_p + e^delta` and keeps the prior on
  `k_er`'s `o` with the Jacobian `delta - log k_er`, so the prior is the
  preset's restricted to `k_er > k_p`; on company 337 it gives the same
  posterior means within the Stan rows' tolerances. A test with rate
  priors centred on the mirror image lands there and warns.
* **Origins with no cell** have no `z`, as decided; each simulation draws
  one new effect per such origin, shared by its cells, and
  `CompartmentalFit::population_origins` lists them. Its mean ultimate loss
  ratio is exact given a draw (log-normal moments of the effect).

**Built (PR C, Rust core).** Model 2 as proposed, read again from the
paper's code (sec. 7.2.3) and its printed `m2fit`, which the earlier
notes had not quoted in full:

* **The hierarchy is in the priors.** `CompartmentalPriors` gains
  `ker_effect_sd` and `kp_effect_sd` (`Option<HalfPrior>`: uncorrelated
  origin effects on the rates) and `development_effect_sd` (`[Option<
  HalfPrior>; 4]`: uncorrelated development-year effects on `k_er`, `k_p`,
  `RLR` and `RRF`); `None` is no effect, and `gesmann_morris()` has none,
  so model 1 is unchanged. Each cell `(i, t)` then has `θ_p = m_p e^{s_p
  (b_p + u_ip + v_tp)}`, and `RLR` and `RRF` keep their correlated origin
  pair. `gesmann_morris_2()` is the code's preset: model 1's medians and
  multipliers, every sd half `student_t(10, 0, ·)` with 0.3 for `k_er` and
  `k_p`, 0.7 for `RLR` and 0.5 for `RRF`, shared by the parameter's two
  groups as brms shares one sd prior per nlpar, and `normal(log 0.2, 0.2)`
  on each `log σ`.
* **The layout** on the paper's training triangle is the 87 parameters
  proposed: the 4 intercepts, `log τ_RLR`, `log τ_RRF`, `atanh ρ`, the 6
  other `log τ` (origin `k_er`, `k_p`; development `k_er`, `k_p`, `RLR`,
  `RRF`), the 18 origin pairs, 9 `z` for each rate's origin effect, 9 for
  each development effect, and the two `log σ`. The posterior adds, beside
  model 1's names, `k_er[o]` and `k_p[o]` (each origin's rates at a zero
  development effect, as `RLR[o]` and `RRF[o]` are), `dev_ker[a]`,
  `dev_kp[a]`, `dev_RLR[a]` and `dev_RRF[a]` (each development age's
  effect on the `o` scale, `a` in months), `tau_ker`, `tau_kp` and
  `tau_dev_*`.
* **Levels with no cell.** Development year 10 has no training cell; each
  simulation draws one new effect per parameter for it, shared by every
  origin, as decided, and 1997 one new set of origin effects (four
  normals, the pair correlated). The ultimate loss ratio of an origin is
  `RLR_i RRF_i` at a zero development effect.
* **The swap** is invariant cell by cell, so `rate_swaps` (and its
  warning) still counts draws whose population rates have `k_er <= k_p`,
  and `CompartmentalFit::cell_rate_swaps` is the share of draws and
  observed cells with `k_er <= k_p`, a diagnostic. On company 337 both are
  0 at every seed of the sweep (decision 7).
* **Errors.** Under the log-normal likelihood a response that is not
  positive (an outstanding of 0, a paid that does not grow) is
  `Error::Bayes` naming the cell; under incremental paid a paid cell whose
  previous age has no paid is `Error::Bayes` naming both ages.

### 3. The ODE strategy

Proposed. **Closed form wherever one exists**, a generic fixed-step solver
otherwise, and an ODE crate only for a stiff system:

* **One OS** (models 1 and 2): the closed form of decision 2. No solver.
* **Two OS** (PR D): Gesmann and Morris' Eq. 3, a cascade, not two
  parallel compartments: `OS₁` fills at `k_er RLR EX` and drains at
  `k_p1 + k_p2`, its `k_p1` share paid and its `k_p2` share moved into
  `OS₂`, which pays at `k_p2`; `dPD/dt = RRF (k_p1 OS₁ + k_p2 OS₂)`. Its
  closed form has three exponents, `k_er`, `k_p1 + k_p2` and `k_p2`, with
  denominators that vanish when two of them coincide (`k_er = k_p1 +
  k_p2`, `k_er = k_p2`) and at `k_p1 = 0`. The authors prefer numerical
  solvers "to reduce algebraic computation and minimize risk of error";
  the cancellation near those coincidences is our own observation. PR D
  writes the closed form, like `g`, in `expm1` kernels with series
  branches, re-derives its symmetries (the one-OS swap does not carry over
  as is) with a unit test of each, and tests it against the RK4 solve
  below to 1e-8 relative at every observed age, over a grid that includes
  the coinciding rates.
* **Anything else** (multistage gamma-pdf exposure, Eq. 4; a reporting
  rate that grows with time): a generic classical RK4 in
  `crates/prospicio-reserving/src/bayes/ode.rs`, generic over `Scalar`
  (decision 4), so the gradient comes from duals through the solver, as in
  the benchmark of `reserving-bayes.md` decision 15 (9e-10 against the
  closed form at 1.6 to 2.2 times a plain solve).
  * **Fixed steps, fixed per fit.** The number of steps per year is
    chosen once from the prior (`n = max(200, ⌈40 k_max⌉)` with `k_max`
    the 99.9th prior percentile of the fastest rate) and does not change
    with the parameters, so the density is a smooth function of them.
    Gesmann's PK/PD post reports `integrate_ode_rk45` at about 9 hours a
    chain and 235 divergences, and says he does not know why the
    integrator was slow; that adaptive step control makes the density a
    non-smooth function of the parameters, and so hurts NUTS, is our
    inference, not his.
  * **Accuracy target**: cumulative PD and OS within 1e-8 relative of the
    closed form at every observed age, over the prior's central 99.8%, on
    the one-OS and two-OS models. RK4's local error is of order `(k h)⁵`
    and its global relative error grows roughly as `k t (k h)⁴ / 120`, so
    it depends on the horizon (10 years) and on the slow rate, through the
    late-age OS, and not only on the fastest rate. Measured (2026-10-13,
    classical RK4 on the one-OS model, worst relative error of OS and PD
    over ages 1 to 10): the earlier rule `n = max(100, ⌈20 k_max⌉)` gave
    6.5e-9 at the 99.9th prior percentile rates (4.09, 1.36) but 1.28e-8
    at (10, 3) with `n` = 200, missing the target; the rule above gives
    4.0e-10 and 8.0e-10 there, and 3.1e-11 at the paper's posterior rates
    (1.75, 0.43). PR D tests the target over the prior's grid and adjusts
    the constant if a point misses.
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

Built (PR A; the outcome at the end of this decision), following
`reserving-bayes.md` decision 15 (decided). PR A adds, with no new model:

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
* **The existing densities stay as they are.** `Posterior::log_posterior`
  in `odp.rs`, `bf.rs` and `meyers.rs` computes the value and the gradient
  in one pass, and it stays the fast path, unchanged, so the sampler makes
  no extra pass and the parity suites run as before. Beside it, each of
  `BayesOdp`, `BayesBf` and `Meyers` (CRC, CSR, CAY) gets a test-only
  generic `log_density<S: Scalar>` (`#[cfg(test)]`), used only by the
  oracle tests. The value is then written twice for these three models;
  the oracle test catches any drift between the two, which is the point
  of it, and the new models (PR B onward) have only the generic density.
  The generic value need match the existing one only to 1e-12 relative,
  not bit for bit, so it need not copy each function's accumulation order.
  PR A starts after the Meyers segments branch
  (`claude/bayes-meyers-segments`, which changes `meyers.rs` by about
  1,500 lines) merges, so the two do not conflict.
* **Oracle tests.** For each of those densities, the chunked dual
  gradient matches the hand-written gradient, and the generic value the
  hand-written value, to 1e-12 relative at 20 random points per model
  (drawn from the posterior's scale, not only the mode); each elementary
  function's dual derivative matches its analytic derivative, including
  the series branches; and the one-compartment closed form's dual
  gradient matches central differences to 1e-7. The hand-written
  gradients are the oracle because each already passes a
  finite-difference test and a Stan parity suite.
* **The measurement decision 15 owes**: time per value and gradient
  (release, median) of the chunked dual at `N` = 4, 8, 16 and 32 against
  the fixed-size dual and the hand-written gradient, at `p` = 27 (the
  one-OS model on the paper's training triangle), `p` = 29 (Meyers' CRC)
  and `p` = 90 (model 2's 87 parameters, decision 2). Recorded in
  `knowledge/findings/autodiff-benchmark.md`. Acceptance: a gradient at
  `p` = 29 within 10 times the hand-written one; otherwise PR B writes the
  one-OS gradient by hand (the closed form makes that practical) and keeps
  duals for the oracle and the RK4 models. The `p` = 90 time sizes model
  2's cost for PR C, which applies the same rule.

**Built (PR A, 2026-10-13; branch `claude/bayes-dual`).** As proposed,
with these outcomes and departures:

* **`Scalar`** has the operators with `Self` and `f64`, `+=` and `-=`,
  `from_f64`, `value`, and `exp`, `exp_m1`, `ln`, `ln_1p`, `sqrt`, `tanh`
  and `powi` (Meyers' CSR speed-up is an integer power). `powf`, `lgamma`
  and the normal log cdf are left out: no density written so far needs
  them (the half Student-t and Cauchy priors need only `ln_1p`, the LKJ(1)
  Jacobian `ln`), and each is a small addition with its dual derivative
  when one does. Beside the trait, `scalar.rs` holds the generic
  `exprel_neg` (decision 2's kernel, with the wider series branch
  explained there), `softplus` and `logistic`, which branch on `value()`
  so the dual follows the value's branch.
* **`Dual<N>`** computes each value exactly as `f64` does (a quotient
  divides, it does not multiply by a reciprocal), so the generic value
  in duals equals it bit for bit, and `tanh`'s derivative is
  `1 / cosh²`, since `1 − tanh²` loses digits as `|x|` grows (2e-14 at
  `x = −4`). `gradient::<N>(theta, grad, f)` runs `⌈p / N⌉` passes and
  returns the value. **`N` = 32** (`dual::CHUNK`).
* **The generic densities** implement a test-only trait,
  `GenericDensity::log_density<S: Scalar>`, in `odp.rs`, `bf.rs` and
  `meyers.rs`; the test support (the oracle check, the closed form, two
  synthetic compartmental densities and the benchmark) is in
  `bayes/autodiff.rs`, `#[cfg(test)]`. No non-test line of the three
  models changed, so the sampler's output is unchanged.
* **Oracle results**, largest relative error over 20 points `centre + L z`
  on each model's whitening scale (the Laplace scale at the mode for
  `BayesOdp` and `BayesBf`, the pilot's scale at the start for `Meyers`),
  value / gradient against its largest component / gradient
  componentwise: `BayesOdp` GenIns 0 / 3.7e-16 / 1.0e-14 and RAA 0 /
  3.8e-16 / 3.3e-14; `BayesBf` GenIns at cv 0.2 0 / 2.4e-16 / 4.2e-14 and
  flat 0 / 4.5e-16 / 7.1e-15; `Meyers` CRC 1.4e-15 / 3.6e-16 / 3.5e-14,
  CSR 1.6e-14 / 1.1e-15 / 2.5e-13, CAY 1.0e-14 / 9.3e-16 / 2.5e-14. Chunks
  of 8 give the same gradient as one pass, bit for bit. Every elementary
  function's dual derivative matches its analytic one to 1e-14 (1e-13 for
  `exprel_neg` on both sides of its branch), and the closed form's dual
  gradient matches central differences to 7.1e-9 at its worst, through
  `k_er = k_p` and inside the series branch.
* **The measurement** (release, median µs per value and gradient; rustc
  1.98.1, Windows, Intel Core Ultra 9 285H; times vary about 15% between
  runs). `p` = 27 and 90 are the one-OS models 1 and 2 on a synthetic
  triangle shaped like company 337's training triangle (its data arrive in
  PR B), with model 2's 87 parameters padded by three standard normals to
  90; neither has a hand-written gradient, so their ratios are to the f64
  value:

  | `p` | value | hand-written | N = 4 | N = 8 | N = 16 | N = 32 | fixed size |
  |---|---|---|---|---|---|---|---|
  | 27 (model 1) | 0.35 | | 5.23 | 4.31 | 3.73 | 3.25 (9.2× value) | 2.84 |
  | 29 (Meyers CRC, group 353 paid) | 0.52 | 1.11 | 5.60 | 3.54 | 3.04 | 2.77 (2.5× hand) | 3.55 |
  | 90 (model 2) | 2.42 | | 100.3 | 63.9 | 48.5 | 43.5 (18× value) | 58.4 |

  **Accepted**: at `p` = 29 the chunked gradient costs 2.3 to 2.5 times
  the hand-written one at `N` = 32 (2.9 to 3.2 times at 8, 4.7 to 5.8 at
  4) over three runs, well within 10 times, so PR B writes only the
  generic density. `N` = 32 was the fastest or within noise of it in
  every row: one pass up to 32 parameters, and at 90 three passes
  against six at 16 and twelve at 8. A fixed size of 90 tangents is
  slower than three passes of 32. Model 2 at 90 costs about 18 times its
  own value, 43 to 54 µs a gradient; PR C applies the rule above to it.
  Run the benchmark with `cargo test --release -p prospicio-reserving
  --features bayes --lib autodiff_benchmark -- --ignored --nocapture`.

Laplace Hessians stay central differences of the exact gradient
(`bayes/laplace.rs`, `Posterior::negative_hessian`), as decided.

**Built (PR C).** The rule above, measured on the real model 2 density
(company 337's training triangle, release, median µs; the
`model_2_gradient_benchmark` test in `bayes/compartmental.rs`, run while
other fits loaded the machine): the value 4.5 to 5.6, the chunked dual
pass (`N` = 32, three passes) 67 to 74, 13 to 15 times the value. A
hand-written gradient costs at least the value, so the chunked pass is
within 10 times of any hand-written one that costs 1.5 times the value or
more, and the rule alone would keep it. The fit, though, takes deep
trees (decision 5), so PR C also writes a faster gradient for the models
with effects by cell, still from the one generic density's pieces: the
prior terms without a `z` in one dual pass over their at most 15 inputs
(`Dual<16>`), the `z` terms by hand, each cell's term in a `Dual<5>` over
its four `o` values and its `log σ`, and the chain rule from the `o`
values to `theta` by hand (`o = b + u + v`, each effect `τ z`, and the
`RRF` origin effect through `ρ`), summed by level. It costs 10 µs, 1.8
times the value and about 7 times less than the chunked pass, and it
matches the chunked pass over the generic density (the oracle) to 2e-15
on the value and 6e-16 of the largest component on the gradient at 20
points around model 2's posterior, under both rate orders, with a partial
hierarchy and with amounts under the normal likelihood. Model 1's
hierarchy keeps the chunked pass, unchanged, so its fits are as before.

### 5. The fit: whitening, pilot run and predictive

Proposed. The fit follows `Meyers::fit` (`meyers.rs`), which already
handles hierarchical and funnel geometry:

1. a start point from the data, not the prior centre: a nonlinear
   least-squares fit of the population curve (shared `k_er`, `k_p`, `RLR`,
   `RRF`, no origin effects) mapped to the four `b`; `z` and `atanh ρ` at
   0; `log τ` at the prior median; `log σ` at the log of each type's
   residual sd from that fit. On company 337 the posterior sits 5 to 9
   prior sds from `o = 0` on the rates (decision 2), and the flat prior on
   `log σ` has no useful median (`log σ = 0` is 7 to 8 units of `log σ`
   from the posterior's 8.25 and 6.66), so a prior-centred start would leave the
   pilot run far to travel. If the least-squares fit fails, the start
   falls back to `o = 0`;
2. a regularised Laplace whitening (`Whitened::regularised`) at that
   point, a pilot NUTS run, then `Whitened::from_draws`;
3. the main run, with `LogDensity` returning `None` on a non-finite value
   (an overflowing exponential, a solver failure), which `prospicio-bayes`
   counts as a divergence;
4. the predictive: for each draw, the mean OS and PD at each future cell
   of the square, plus process error from the likelihood (normal on
   amounts in model 1; log-normal increments in model 2), giving a joint
   `PredictiveDistribution` of each origin's reserve. An origin or
   development year with no observed cell gets a new effect per draw from
   the population (decision 2).

A test fits company 337 from the default start and checks convergence (no
divergence, every R-hat below 1.01), and a second test does the same from
`o = 0`, so the fallback start is known to work.

**Built (PR B, Rust core).** As proposed, with these outcomes:

* **The start** is a weighted least-squares fit of the population curve on
  `log (k_er, k_p, RLR, RRF)` by the Newton of `laplace::mode`, each type
  weighted by the inverse of its residual variance, twice from the prior
  medians; both representatives of the swap are mapped to `theta` and the
  one with the higher log posterior is kept. `log τ` starts at 0.6745 times
  the half prior's scale (about its median) and `log σ` at the log of each
  type's residual RMS. On company 337 it gives `k_er` 2.64 and `k_p` 0.405
  (`b_ker` -1.28, `b_kp` -9.03) and `log σ` 8.46 and 8.69: without origin
  effects paid fits worse. The prior centre is the fallback and a setting
  (`CompartmentalStart::PriorCentre`); from either start the default fit
  at seed 0 converges with no warning (validation tests). Under one
  response (PR H, decision 12) the least squares is not identified, so the
  data start is kept only when its log posterior is above the prior
  centre's.
* **Defaults**: 4 chains of 1,000 warm-up and **2,000** kept draws (not
  1,000), `target_accept` 0.99, `max_depth` 10. With 1,000 kept draws 9 of
  30 seeds failed a threshold (R-hat just above 1.01 or ESS below 400 on
  `tau_RRF`, `rho`, `RLR[1991]`); with 2,000, 6 of 200 still do, each just
  over (R-hat at most 1.014, one tail ESS of 391, one divergent
  transition); 2,000 warm-up draws as well gave 1 of 40. The sweep does not
  count draws at the maximum depth: `prospicio-bayes` does not report tree
  depths (Models lane); Stan had none at its depth 15.
* **The predictive** draws every future cell of the square, paid and
  outstanding, normal about its mean with its type's σ and independent
  given the draw; the reserve is each origin's paid at the last age less
  its latest paid, 0 for an origin observed there. The square has
  `max(number of origins, observed ages)` ages on equal grains, and as
  many origin periods as there are origins, in development periods, when
  the development grain is finer (decision 13).
* **Runtime** on company 337 (rustc 1.98.1, Windows, Intel Core Ultra 9
  285H, 4 chains in parallel): about 5.5 s a default fit in release
  (median over the 200 seeds), including the pilot run and 10,000
  simulations, and about 430 s in a debug build (measured while CmdStan
  ran beside it), with the same draws as in release; so the default fits
  run only in the release `slow-bayes` suite, and the unit tests use 2
  chains of 100 warm-up and 100 kept draws (about 67 s for all of them in
  debug).

Defaults (as built; the **Defaults** decision above records why): 4
chains, 1,000 tuning and 2,000 kept draws each, `target_accept = 0.99`
(the paper's `adapt_delta`) and `max_depth = 10` (`Sampler::default()`'s
depth, which `Meyers` also keeps, with `target_accept` 0.95). The paper's
`max_treedepth` 15 is used for the Stan reference only. The plan was 1,000
kept draws and a seed sweep that counts draws at the maximum depth; the
sweep showed 1,000 kept draws too few, and it could not count tree depths
because `prospicio-bayes` does not report them. A deeper tree is a setting.
`n_sims = 10_000`, `seed = 0`. Seeds as
`reserving-bayes.md` decision 5: derived streams for the chains, the pilot
(`pilot_seed`) and the simulations, exact replay whatever the thread
count.

**The reserve** is future paid to the last age of the square (age 10 for
company 337), as `Meyers` and the paper's validation; the model's own
ultimate `Π_i RLR_i RRF_i` is reported alongside as a mean, without
process error past the square. A reserve to ultimate is open decision 5.

**Built (PR C, Rust core).** Model 2 fits as model 1 does (the data
start, the pilot, the main run), with these outcomes:

* **The start.** The population least squares runs on the likelihood's
  scale (on the logs of the responses under the log-normal), with each
  `log τ` at about its prior median; from it, and from the prior centre,
  the default fit at seed 0 converges with no warning (validation tests).
* **The sampler** is `Compartmental::model_2`'s, not `new`'s: 4 chains of
  2,000 warm-up and 8,000 kept draws, target acceptance 0.999, tree depth
  at most 10. Model 1's sampler left divergent transitions and R-hats
  above 1.01 on most seeds; what fails is `k_p`: its effect sd by accident
  year has a wide posterior (the paper's `sd(okp)` 2.23 (1.35)) and `b_kp`
  is weakly identified against the mean of the effects, so `k_p` mixes
  slowly (the paper's own fit has a bulk ESS of 499 on `okp` in 4,000
  draws), and the effect sds whose posteriors pile up near 0 bring
  divergences at lower target acceptance. Measured with the per-cell
  gradient (decision 4), seeds of the gm2 sweep, other runs beside them:

  | Warm-up, kept draws, target acceptance | Seeds | With a warning | With a divergence | Time a fit |
  |---|---|---|---|---|
  | 1,000, 2,000, 0.95 | 4 | 3 | 3 | 38 to 50 s |
  | 1,000, 2,000, 0.99 (model 1's) | 8 | 7 | 4 | 19 to 44 s |
  | 1,000, 2,000, 0.999 | 8 | 5 | 0 | 28 to 38 s |
  | 1,000, 4,000, 0.995 | 8 | 4 | 4 | 28 to 40 s |
  | 1,000, 4,000, 0.999 | 10 | 5 | 1 | 36 to 54 s |
  | 2,000, 4,000, 0.999 | 200 | 41 | 1 | median 81 s (3 sweeps and Stan beside it) |
  | 2,000, 8,000, 0.999 (model 2's) | 200 | 5 | 4 | median 126 s (4 sweeps beside it) |

  With 4,000 kept draws 41 of 200 seeds warned, almost all on `k_p`:
  R-hat 1.010 to 1.018 on `k_p`, `b_kp`, `tau_kp` or `tau_dev_kp`, or a
  tail ESS below 400 on every `k_p[o]`; doubling the kept draws took the
  smallest bulk ESS from about 700 to about 1,600. With 8,000 the 5 seeds
  that warn have one divergent transition each (4) or a tail ESS of 398
  and 399 on two `dev_kp` (1), the rate of model 1's defaults (6 of 200).
* **Runtime** on company 337 (rustc 1.98.1, Windows, Intel Core Ultra 9
  285H, 4 chains in parallel): a default model 2 fit takes about 61 s in
  release on its own (52 s from the prior centre), including the pilot run
  and 10,000 simulations; the release validation file with both fits about
  2 minutes. Model 1's fits are unchanged (its gm1 suite at seed 0 gives
  the same total reserve, 173,390 (10,167), and elpd_loo -810.75). The
  debug unit tests use short runs (2 chains of 60 warm-up and 60 kept
  draws for the model 2 fit): all compartmental unit tests take about 61 s
  in debug, with the oracle and written-out tests of model 2.
* **The predictive.** Each simulation draws the future increments,
  log-normal about the cells' mean increments (normal under the normal
  likelihood, with σ times the premium on the loss-ratio scale), and sums
  them onto each origin's latest paid; outstanding cells are drawn the
  same way. Development year 10's effects are drawn once per simulation
  and shared by the origins, 1997's origin effects once per simulation.

### 6. The API

Decided 2026-10-13 (the shape; names may change in PR B):

```rust
#[cfg(feature = "bayes")]
pub mod bayes {
    pub enum Outstanding<'a> {
        Column(&'a str),        // an outstanding column
        FromIncurred(&'a str),  // incurred minus paid
    }

    pub enum CompartmentalLikelihood { Normal, LogNormal }   // model 1, model 2 (PR C)

    // Decision 1 (the user, 2026-10-13): the response is a setting too.
    pub enum Scale { Amount, LossRatio }                     // PR B: Amount; PR C: LossRatio
    pub enum PaidResponse { Cumulative, Incremental }        // PR B: Cumulative; PR C: Incremental
    pub enum Fitted { PaidAndOutstanding, PaidAndIncurred, PaidOnly, IncurredOnly }
    // PR B ships PaidAndOutstanding; the others return Error::Bayes until added.

    pub struct CompartmentalPriors {
        pub k_er: PositivePrior, pub k_p: PositivePrior,
        pub rlr: PositivePrior,  pub rrf: PositivePrior,
        pub rlr_effect_sd: HalfPrior, pub rrf_effect_sd: HalfPrior, // τ, on the o scale
        pub lkj_eta: f64,
        pub log_sigma: [RealPrior; 2],   // OS, PD; on log σ (Student-t or normal)
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
  scale: `k_er`, `k_p`, `RLR`, `RRF`, the σs and every origin's `RLR_i`
  and `RRF_i`; and, on the scale the paper prints so they compare with
  it, the intercepts `b_RLR`, `b_RRF`, `b_ker`, `b_kp`, the effect sds
  `tau_RLR` and `tau_RRF` on the `o` scale, `rho`, and `log_sigma_os`,
  `log_sigma_pd`), `diagnostics` (`bayes::Diagnostics`, thresholds
  as before, warnings not errors, as `reserving-bayes.md` open decision
  4), `summary()`, `loo()` (pointwise over both the OS and the paid
  cells, with the elpd also split by type), `reserves` (the joint
  `PredictiveDistribution` by origin, paid), `paid` and `outstanding`
  (predictive means and quantiles for each future cell, as long frames),
  `ultimate_loss_ratio` by origin, `to_long()`, `totals()`.
* **`prior_predictive`** draws OS and paid paths from the priors alone,
  so a user can check a prior set before fitting, as the paper does in
  sec. 3.4 to test identifiability. Under `RateOrder::Ordered` it draws
  the rate intercepts exactly from their priors restricted to
  `k_er > k_p` (a half-plane in their standard normals: the projection
  across it from a truncated normal, by the inverse of its upper tail),
  and it is `Error::Bayes` when the rate priors put less than `1e-6` of
  their mass on `k_er > k_p`.
* **Errors.** A missing column or premium, a non-positive premium, and a
  triangle with no outstanding cell, and a triangle where no origin has an
  observed cell, are `Error::Bayes` naming the origin or column; an origin
  with no observed cell is accepted and drawn from the population
  (decision 2); bad prior settings are `InvalidSetting`.
* **Segments and the backtest** are PR E (decision 8).

**Built (PR B, Rust core).** The shape above, with these names and
additions: `Outstanding::{Column, FromIncurred}`;
`CompartmentalLikelihood::{Normal, LogNormal}`, `Scale::{Amount,
LossRatio}`, `PaidResponse::{Cumulative, Incremental}` and
`Fitted::{PaidAndOutstanding, PaidAndIncurred, PaidOnly, IncurredOnly}`,
each value but model 1's an `Error::Bayes` that says it is not yet built;
`RateOrder::{Free, Ordered}`; `CompartmentalStart::{Data, PriorCentre}`
(decision 5). `Compartmental` has no `Default`: `Compartmental::new(priors)`
takes the priors and sets the rest. `CompartmentalFit` holds `origins`,
`premium`, `latest_paid`, `population_origins`, `posterior` (the names of
decision 6, with `RLR[o]` and `RRF[o]` by origin label), `diagnostics`,
`rate_swaps`, `reserves`, `paid` and `outstanding` (each a
`PredictiveDistribution` over the future cells, dimensions `origin` and
`age`, from the same simulations as the reserves, with `paid_table()` and
`outstanding_table()` as long frames of mean, sd and 5/50/95%),
`ultimate_loss_ratio`, `summary()`, `loo()` (a `CompartmentalLoo`: the
`Loo` over the outstanding cells then the paid cells, and the elpd of
each), `to_long()` and `totals()`. `prior_predictive` returns
`CompartmentalPaths { outstanding, paid }`, the mean paths of every origin
at every age of the square, without the likelihood's noise (the preset's
flat prior on `log σ` would swamp them); the `Compartmental` doc example
shows the check. A bad prior setting is `InvalidSetting` naming it.

**Built (PR C, Rust core).** `CompartmentalLikelihood::LogNormal`,
`Scale::LossRatio` and `PaidResponse::Incremental` are built (decision 1),
`Fitted`'s other values still `Error::Bayes` (PR G builds `PaidAndIncurred`). `CompartmentalPriors` gains
the three effect fields of decision 2 (a struct literal now needs them;
`..CompartmentalPriors::gesmann_morris()` keeps model 1's hierarchy) and
`CompartmentalPriors::gesmann_morris_2()`. `Compartmental::model_2(priors)`
is `new(priors)` with model 2's likelihood, scale and paid response and
its sampler (decision 5). `CompartmentalFit` gains `cell_rate_swaps:
Option<f64>`. On the loss-ratio scale the predictions and reserves stay
amounts; with incremental paid the `paid` predictive is still cumulative
(the latest paid plus the simulated increments), so the reserve is still
the paid at the last age less the latest. `prior_predictive` draws the
new effects too, and with incremental paid and effects by cell its paid
path is the sum of the cells' mean increments. The provenance's model is
`compartmental_gm1` or `compartmental_gm2` when the settings and the
hierarchy are the paper's, `compartmental` otherwise. The Python and R
bindings take the new effects and model 2's settings (decision 9, "Built
(PR C, bindings)").

### 7. Validation

Proposed, on the pattern of `reserving-bayes.md` decision 10.

* **Reference programs**: `validation/scripts/stan/compartmental_gm1.stan`
  (PR B) and `compartmental_gm2.stan` (PR C), the paper's models 1 and 2
  reconstructed as the Stan program brms generates from its code (sec.
  7.2.2 and 7.2.3), reproducing the brms conventions listed under
  References (the `o` scale, half Student-t sds, `|ID|` correlation,
  non-centring, priors on `log σ`), with provenance in the header. Its
  generated quantities draw the effects of an origin or development year
  with no observed cell from the population, as decision 2. PR D adds
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
  code 7.2.1, `cal_year < max(accident_year)`), 90 observations over 9
  accident years with data, so the 1997 diagonal is the test set and the
  full square the validation. Accident year 1997 has no training cell
  (decision 2). PR B builds that triangle from the square by dropping
  the 1997 diagonal (the backtest's own helper if it exposes one; a
  training CSV from the generator otherwise).
* **Reference CSV**: `validation/reference/reserving_compartmental_stan.csv`
  in the usual format. Rows: mean and sd of every natural-scale
  parameter, including `RLR_i` and `RRF_i` for each of the 9 origins with
  data, and of the `o`-scale quantities the paper prints (decision 6);
  each origin's (1997's from its population draw)
  and the total reserve's mean, sd and 5th, 50th and 95th percentiles;
  the 1997 diagonal's predictive means; elpd_loo. CI reads the committed
  CSV and never runs Stan.
* **Tolerances** from a seed sweep, as PR 3 of `reserving-bayes.md`: our
  defaults over seeds 0 to 199; each tolerance at least 4 times the
  seed-to-seed sd of its row (and of the combined errors where a printed
  number is the target). A row whose seed-to-seed error is heavy-tailed
  (the youngest origin's reserve sd was one for `Meyers`) is dropped and
  checked through its percentiles, and the note says so. Because the
  data pull the rates 5 to 9 prior sds from the preset (decision 2), the
  rate rows are expected to be sensitive to the tight rate priors: a small
  mismatch in how the reference encodes them moves the posterior more than
  in a prior-dominated fit. PR B therefore also checks the reference
  against the printed `m1fit` (below) before it trusts any rate row.
* **Printed numbers, looser** (Monte Carlo targets, brms 2.10.0 on rstan
  2.19.2, so never bit-exact), each mean within 0.15 printed sd and each
  sd within 15%:
  * model 1 (PR B), `m1fit` in sec. 7.2.2: intercepts `oRLR` 1.44 (sd
    0.26), `oRRF` −1.36 (0.42), `oker` −5.40 (0.81), `okp` −8.49 (0.36);
    group sds `sd(oRLR)` 0.64 (0.15), `sd(oRRF)` 0.74 (0.22); their
    correlation 0.54 (0.26); `log σ_OS` 8.25 (0.16), `log σ_PD` 6.66
    (0.20);
  * model 2 (PR C), `m2fit` in sec. 7.2.3: intercepts `oRLR` 1.54 (0.44),
    `oRRF` −1.45 (0.69), `oker` −1.31 (1.08), `okp` −5.07 (1.75); its
    accident-year sds (`oRLR` 0.69, `oRRF` 0.73, `oker` 0.26, `okp` 2.23)
    and correlation (0.37), and its development-year sds (0.35, 1.11,
    0.27, 0.45). The sds with a posterior that piles up near 0 (lower 95%
    bound 0.01 to 0.04) are checked through the seed sweep's spread, not
    the 15% rule, and the note will say which.

  The blog's brms fit is not a check, and cannot be one: it trains on
  calendar years up to 1997 (110 observations, 10 levels, not 90 and 9);
  it normalises every amount by the oldest accident year's premium, not
  each origin's; its origin effects are additive on the natural scale
  (`RLR ~ 1 + (1 | accident_year)`) and uncorrelated; its priors are gamma
  on `RLR`, `RRF`, `k_er` and `k_p` directly; and its σs (about 0.034 and
  0.006) are on that normalised scale. None of this can be expressed with
  `CompartmentalPriors`. The final reserve (mean 123 against an actual
  130, USD millions) is model 3's and is recorded, not checked.
* **Sanity tests** (unit, no reference): the swap invariance and the
  equal-rate limit (decision 2); the closed form against RK4 (decision 3);
  the dual gradient against central differences; exact replay for a seed
  and in a one-thread pool; the posterior mean ultimate loss ratio on
  company 337 against the chain ladder's on incurred (a report, not a
  bound, since the models differ).

**Built (PR B, Rust core).** `compartmental_gm1.stan` and
`reserving_compartmental_stan.py` as proposed (seed 20261014, 4 chains of
2,000 warm-up and 25,000 draws, `adapt_delta` 0.99, `max_treedepth` 15;
about 20 minutes; max R-hat 1.0002, no divergence, no draw at the maximum
depth or with `k_er <= k_p`; with `--reuse` it rereads its draws). The
reconstruction matches the printed `m1fit` with every mean within 0.04
printed sd and every sd within 2.5%, so the rate rows were trusted. The
training triangle is built in `validation/tests/reserving_compartmental.rs`
from the square (calendar years up to 1996, and 1997 with its premium
only). Results at seed 0, ours / Stan / printed (mean, sd): `b_ker` -5.395
(0.833) / -5.392 (0.830) / -5.40 (0.81); `b_kp` -8.485 / -8.488 / -8.49;
`b_RLR` 1.446 / 1.442 / 1.44; `b_RRF` -1.357 / -1.358 / -1.36; `tau_RLR`
0.643 / 0.642 / 0.64; `tau_RRF` 0.739 / 0.737 / 0.74; `rho` 0.533 / 0.534 /
0.54; `log_sigma_os` 8.249 / 8.252 / 8.25; `log_sigma_pd` 6.671 / 6.671 /
6.66; total reserve 173,390 (10,167) / 173,341 (10,076); 1997 reserve
31,376 (6,171) / 31,321 (6,125); elpd_loo -810.7 / -811.0
(`knowledge/findings/compartmental-gm1-parity.md`).

Tolerances, from the sweep over seeds 0 to 199 (largest seed-to-seed sd in
each family, `s` the Stan posterior sd): parameter means `0.15 s` (spread
`0.032 s`), parameter sds 15% (2.3%), reserve means `0.15 s` (`0.016 s`),
reserve sds 15% (1.4%), reserve percentiles `0.3 s` (`0.042 s`), test
means `0.15 s` (`0.025 s`), elpd_loo 1.5 (0.35): every row at least 4.25
times its spread, and all 170 rows pass at every seed (at most 0.73 of a
tolerance). No row is dropped: the 1997 reserve is normal given a draw and
its sd is not heavy-tailed. The test-diagonal means are the posterior
means of the cells' expected values, not of the predictive cells, whose
noise (`σ_OS` about 3,900) is larger than the tolerance on a small cell.
The printed rows get 0.2 printed sd on a mean (not 0.15) and 15% on an sd:
the combined error of our run, the paper's (4 x 1,000 draws, bulk ESS from
1,300) and the rounding to two decimals is at most 0.043 printed sd and
3.6%, and 0.15 would be only 3.5 times it. Not built yet: the report of
the posterior mean ultimate loss ratio against the chain ladder's on
incurred; the CSV checks each origin's mean `RLR_i RRF_i` against Stan
instead.

**Built (PR C, Rust core).** `compartmental_gm2.stan`, reconstructed from
the paper's code as printed (read again from the published PDF: the
paper's `claimsprocess` subtracts the paid closed form at `t - 1` with the
cell's own parameters, and the response is the loss ratio to direct
premium), and `reserving_compartmental_gm2_stan.py` with the gm1 script's
conventions (seed 20261015, 4 chains of 2,000 warm-up and 25,000 draws,
`max_treedepth` 15). At the paper's `adapt_delta` 0.99 the run left 5
divergent transitions in 100,000 kept draws, so, as the Risks section
says, the step was tightened: at 0.999 it has none, max R-hat 1.0004, no
draw at the maximum depth and none with population `k_er <= k_p` (about
92 minutes). Against the printed `m2fit` every mean is within 0.08
printed sd and every sd within 3.4% except `tau_dev_kp`'s (0.518 against
0.43, heavy-tailed), so the rate rows were trusted. The CSV holds the same
row families as gm1's, with `k_er[o]`, `k_p[o]` and the development
effects `dev_*[a]` added, the test diagonal's expected outstanding and
incremental paid amounts, and elpd_loo (arviz, Pareto k up to 1.04 on a
few cells). For the two levels with no training cell the validation test
averages the test cells' means over 25 new effects per posterior draw.

Results at seed 0 (`Compartmental::model_2`), ours / Stan / printed
(mean, sd): `b_ker` -1.294 (1.064) / -1.271 (1.056) / -1.31 (1.08);
`b_kp` -5.112 (1.692) / -5.064 (1.722) / -5.07 (1.75); `b_RLR` 1.539 /
1.534 / 1.54; `b_RRF` -1.456 / -1.446 / -1.45; `tau_RLR` 0.699 / 0.695 /
0.69; `tau_RRF` 0.731 / 0.743 / 0.73; `tau_ker` 0.263 / 0.260 / 0.26;
`tau_kp` 2.175 (1.315) / 2.211 (1.337) / 2.23 (1.35); `rho` 0.358 / 0.357 /
0.37; `tau_dev_RLR` 0.352 / 0.361 / 0.35; `tau_dev_RRF` 1.124 / 1.119 /
1.11; `tau_dev_ker` 0.274 / 0.272 / 0.27; `tau_dev_kp` 0.483 (0.511) /
0.481 (0.518) / 0.45 (0.43); `log_sigma_os` -1.787 / -1.787 / -1.78;
`log_sigma_pd` -1.887 / -1.887 / -1.89; total reserve 152,387 (13,909) /
152,242 (13,938); 1997 reserve 30,322 (6,876) / 30,296 (6,801); elpd_loo
203.75 / 204.29 (`knowledge/findings/compartmental-gm2-parity.md`).

Tolerances, from the sweep over seeds 0 to 199 with model 2's sampler
(largest seed-to-seed sd in each family, `s` the Stan posterior sd):
parameter means `0.2 s` (spread `0.036 s`, `tau_dev_kp`; `0.017 s` off
`k_p`), parameter sds 10% (1.4%), ultimate loss ratio, reserve and test
means `0.1 s` (`0.011 s`, `0.017 s`, `0.014 s`), reserve sds 10% (1.6%),
reserve percentiles `0.2 s` (`0.037 s`), elpd_loo 2.0 (0.45): every row at
least 4.4 times its spread, and all 282 rows pass at every seed (at most
0.93 of a tolerance, the printed `tau_dev_kp` mean). `k_p` mixes slowly,
and the sds of its development effects and origin values (`tau_dev_kp`,
`dev_kp[a]`, `k_p[o]`) are heavy-tailed from seed to seed, 4% to 14% over
seeds 0 to 39 (18% with 4,000 kept draws), from the long right tail of
`tau_dev_kp`'s posterior; those 19 sd rows are dropped, as decided for
heavy-tailed rows, and their means kept (the sds of `k_p`, `b_kp` and
`tau_kp` spread about 1% and stay, checked on seeds 0 to 39). The printed
rows get 0.3 printed sd on a mean and 20% on an sd, at least 4.3 times the
combined error of our run, the paper's (bulk ESS down to 470) and the
rounding, without `tau_dev_kp`'s sd (0.43 printed, 0.518 in Stan). The
elpd_loo row is PSIS-LOO with Pareto k above 0.7 on a few cells (up to
1.04 in arviz, 1.3 in ours), so it checks the computation more than it
measures the fit.

### 8. Segments and the backtest (PR E)

Status: **Built** (the user's decisions of 2026-10-19), in three stages on
`claude/bayes-compartmental-segments`: `fit_segments` (Rust), the
diagonal backtest (Rust), then Python and R `fit_segments`. See "As
built" at the end of this decision. The template
is the Meyers segments and backtest work (`reserving-bayes.md` decisions
8, 11 and 13), which solves the same problem for a model that needs a
premium column and a second response.

* **`fit_segments`.** `Compartmental` gains `dependence:
  SegmentDependence` (default `Independent` in `new` and `model_2`; `fit`
  ignores it), and `Compartmental::fit_segments(triangle, paid,
  outstanding: Outstanding, premium) -> CompartmentalFits`, with `fit`'s
  arguments, each segment read as `fit` reads a single-segment triangle
  (its own premium column and outstanding cells). `CompartmentalFits`
  holds `segments: SegmentFits<CompartmentalSegment>` and the joint
  `reserves`, `paid` and `outstanding` as `PredictiveDistribution`s whose
  dimensions are the key names, then `origin` (reserves) or `origin` and
  `age` (the cells), with `to_long`, `totals`, `paid_table`,
  `outstanding_table` and `segment(keys)`. A `CompartmentalSegment` is
  the segment's `CompartmentalFit` (posterior, diagnostics, rate swaps,
  `summary()`, `loo()`), whose three predictive distributions are the
  segment's part of the joint ones, after any reordering. The tables
  are built from the segments' own `to_long`, `totals` and cell tables
  with the key columns prefixed, since `SegmentFits::to_long` needs a
  chain-ladder projection that this model does not have.
* **Seeds** as for `Meyers`: segment `k` takes `Stream::SegmentChains(k)`
  (tag `2 + 2k`, with its pilot run) and `Stream::SegmentPredictive(k)`
  (tag `3 + 2k`), so no two segments share a random number and `fit`'s
  draws are unchanged bit for bit. A one-segment `fit_segments` is `fit`
  with segment 0's streams.
* **Dependence.** `Independent` leaves the draws as simulated.
  `RankCorrelation` reorders each segment's reserves together with its
  future paid and future outstanding cells: one row per simulation holds
  every segment's reserves, paid cells and outstanding cells, Iman–Conover
  ranks a segment by its total reserve alone, and the whole row of
  that segment moves, so each simulation's reserve stays the paid at the
  last age less the latest paid exactly (a unit test). `Synchronized` is
  `Error::Bayes` (a posterior has no residuals to share) and a bad matrix
  is `InvalidSetting`, both before any sampling. A failure in a segment
  names it (`Error::InSegment`). Provenance as `MeyersFits`: the settings,
  `segments`, `dependence`, `posterior_draws`, each segment's total
  premium, and `output`.
* **Backtest input.** The shared `TriangleFrame` and `TriangleModel` do not
  change. A wrapper candidate, `CompartmentalCandidate`, holds the
  `Compartmental` settings, the triangle and the names of the paid,
  outstanding (or incurred) and premium columns, and implements
  `TriangleModel`: it reads outstanding at the training rows' origin and
  age from its triangle and premium per origin, the held-out diagonal
  never. The paid increments are scored in the standard
  `diagonal_backtest` table, comparable with the ODP GLM, `BayesOdp` and
  `Meyers`.
* **Outstanding table.** A function in the reserving crate reuses the
  backtest's split and scoring code and scores the held-out outstanding
  diagonal as a second table: the same metrics (mean cell CRPS, central
  interval coverage, A/E on the diagonal total, CRPS of the total, log
  density where closed form) over the outstanding cells of the held-out
  diagonals. The held-out outstanding is always withheld from training.
* **Scope.** Model 1 is what pull-request CI runs; model 2 runs only
  under `full_bayes()` (`RISK_RS_FULL_BAYES`). `one_year_segments` came in
  a later PR (below). Python and R get `fit_segments` with `dependence` and
  `spearman` settings; the backtest stays Rust-only, as for `BayesOdp`
  at the time (since reversed for `BayesOdp` and `Meyers` by
  `reserving-bayes.md` decision 21, which gives the backtest Python and
  R entry points). The user's decision of 2026-10-19 for this PR kept
  `CompartmentalCandidate` out of them: it borrowed its triangle, while
  the bindings' `NamedModel` owns its model, so wiring it in was a
  follow-up, built on `claude/compartmental-backtest-bindings` (see "As
  built").
* **Checks.** `Compartmental` still implements `YearModel` and passes the
  one-year conformance suite; the unit tests that sample with NUTS are
  gated by `slow-tests`, and the parity and backtest validation lives in
  `validation/tests/reserving_compartmental*` (`slow-bayes`).

**Built (PR E, backtest).** `CompartmentalCandidate` (in
`bayes/compartmental/backtest.rs`) holds `model: Compartmental`, the
`&Triangle` and the paid, outstanding and premium names, and implements
`TriangleModel` (name `compartmental_gm1`, `compartmental_gm2` or
`compartmental`). Its outstanding table is the method
`outstanding_backtest(cells, n_diagonals, n_sims, seed, interval)`, which
returns a `Backtest` (one model, `<name>_outstanding`) from the backtest's
own split (`scored_splits`, shared with `diagonal_backtest`) and scoring
(`score`) code. Details beyond the decision: the candidate keeps the fit
of every split (keyed by its training rows and seed), so the paid and the
outstanding table of one backtest share one NUTS run per split however
many diagonals it holds; a row the likelihood rejects (log-normal: the first non-positive
response of an origin and every later row of it) leaves `fit_rows`, and a
non-positive outstanding is left out of the fit; the mean of a row is the
mean of its predictive draws; the log density is closed for a paid row that
follows its origin's latest training row and for every outstanding cell of
an origin and age with an observed cell. The rows scored are the held-out
rows whose origin and age the candidate fits, as for the paid table, so on
company 337 the two tables score the same cells. Model 2 cannot be
backtested on the 9-origin training triangle (fewer cells than parameters
once a diagonal is held out); its arm uses the 55-cell upper triangle to
1997 and holds out the 1997 diagonal. Since `reserving-bayes.md`
decision 21 both tables carry its six metrics, each split's diagonal and
the fit's sampler diagnostics (`CellForecast::diagnostics`). No deviation
from the decision.

**Built (PR E, bindings).** As decided, with these outcomes:

* **Python** (`prospicio.reserving`): `Compartmental(...,
  dependence="independent", spearman=None)` as the new last arguments
  (and on `Compartmental.model_2`), with the getters `dependence` and
  `spearman`, and `fit_segments(triangle, paid, premium, *,
  outstanding=None, incurred=None)`, which returns a
  `CompartmentalFits`: `keys`, `index`, `origins`, `premium`,
  `latest_paid`, the joint `reserves`, `paid` and `outstanding`
  (`PredictiveDistribution`s whose first dimensions are the keys),
  `rate_swaps` (one count a segment), `diagnostics_frame()`,
  `to_frame()`, `totals_frame()`, `paid_frame()`,
  `outstanding_frame()` and `segment(**keys)`; the posterior of one
  segment (`diagnostics`, `parameter_names`, `draws`, `chains`,
  `summary()`, `loo()`) needs a single-segment fit, as for
  `MeyersFits`. Failed diagnostics and rate swaps raise one
  `UserWarning` with each message prefixed by its segment's label.
  `python/tests/test_reserving_compartmental_segments.py` fits two
  copies of company 337 at the default chains.
* **R**: `compartmental_segments(triangle, paid, premium, priors,
  outstanding = NULL, incurred = NULL, <compartmental()'s settings>,
  dependence = c("independent", "rank_correlation", "synchronized"),
  spearman = NULL)`, returning a `compartmental_fits` with
  `as.data.frame()`, `totals_frame()`, `summary()`, `print()`,
  `bayes_loo()`, `segment()` and the properties of the Python class
  (`paid_table` and `outstanding_table` included). Model 2 is
  `compartmental_segments()` with `compartmental_model_2()`'s
  likelihood, scale and paid response. `R/prospicio/tests/
  test-reserving-compartmental-segments.R` mirrors the Python test.
* **Not done.** The backtest had no `Compartmental` entry in Python or R
  in this PR: the candidate borrowed its triangle and the bindings'
  `NamedModel` owns its model, and the user's decision of 2026-10-19 for
  this PR kept the backtest Rust-only; the entry points are built in the
  follow-up below. `one_year_segments` came in a later PR ("Built (one-year
  view over segments)"). Model 2's segment
  and backtest arms run only under `RISK_RS_FULL_BAYES`.
* **Result.** On company 337 the backtest forecasts the held-out
  outstanding diagonals well and the paid increments badly, and cannot
  separate a 1996 case reserve strength shift from faster settlement:
  `knowledge/findings/compartmental-segments-backtest.md`.

**Built (PR E, backtest bindings).** Branch
`claude/compartmental-backtest-bindings`, the follow-up the user's
decision of 2026-10-19 left (`reserving-bayes.md` decision 21). As decided
there, with this design:

* **The candidate owns its data.** `CompartmentalCandidate` no longer has a
  lifetime: the triangle is copied once into an `Arc<Triangle>`, the
  column names are strings and the outstanding source is an owned form of
  `Outstanding`. `new` keeps its arguments (a borrowed triangle and
  columns, now copied), and `triangle()`, `paid()`, `outstanding()` and
  `premium()` stay read-only accessors, so the settings are still fixed
  after construction.
* **A clone shares the fits.** The cache of fits is behind an
  `Arc<Mutex<..>>`, and `CompartmentalCandidate` is `Clone`: a clone has
  the same settings and the same cache. A binding keeps its object, boxes
  a clone in a `NamedModel`, and the object's `outstanding_backtest`
  reuses the fits the backtest sampled (the reverse order too). A unit
  test boxes a clone in a `NamedModel`, runs the paid backtest and then
  the original's outstanding backtest, and checks that the cache holds one
  fit, the same `Arc`, after both. The cache keeps every fit until the
  candidate and its clones are dropped: a bound of four, evicting the
  oldest, refitted every split of a five-diagonal backtest, since both
  tables walk the splits oldest first; a test with five diagonals checks
  that the outstanding table reuses all five fits. This was chosen over an
  `Arc<CompartmentalCandidate>` implementing `TriangleModel` because a
  clone is a plain value a binding can hold in a frozen Python object or
  an R pointer.
* **Python**: `Compartmental.candidate(triangle, paid, premium, *,
  outstanding=None, incurred=None)` returns a `CompartmentalCandidate`
  (`name`, `paid`, `premium`, `outstanding`, `outstanding_from`), which
  `diagonal_backtest`'s `models` takes in a list or a dict next to
  `GlmCandidate`, `BayesOdp` and `Meyers`, and
  `CompartmentalCandidate.outstanding_backtest(n_diagonals, *, n_sims=1000,
  seed=0, interval=0.9)` returns the second table as a `Backtest` named
  `{name}_outstanding`.
* **R**: `compartmental_candidate(triangle, paid, premium, priors,
  outstanding = NULL, incurred = NULL, <compartmental()'s settings>)` and
  `compartmental_model_2_candidate()` return a
  `compartmental_backtest_model` for the `models` list of
  `diagonal_backtest()`, and `compartmental_outstanding_backtest(candidate,
  n_diagonals, n_sims, seed, interval)` returns the second table. Unlike
  the other model specs the candidate holds its triangle (the backtest's
  table of paid increments carries neither the outstanding nor the
  premium), through an external pointer to the Rust candidate, which is how
  the two tables find the same fits.
* **The triangle is checked.** A candidate reads its own triangle, so a
  backtest whose `triangle` or `column` differs from the candidate's is an
  error in both bindings (otherwise the cells of the frame and the fit
  would disagree without a message).
* **Tests.** Company 337 model 1 at 2 chains of 300 warm-up and 300 kept
  draws, 400 simulations and seed 3, three diagonals: 5, 6 and 7 scored
  cells, finite metrics, the two tables' diagnostics equal (one fit) and
  the outstanding table in under half the paid table's time, the paid A/E
  within a quarter of the Rust validation's targets and the ODP GLM's equal
  to them (`python/tests/test_reserving_compartmental_backtest.py`,
  `R/prospicio/tests/test-reserving-compartmental-backtest.R`, about 10 s
  each). Model 2 is left to the Rust validation: its sampler is too slow
  for a binding test.

**Built (one-year view over segments).** Branch
`claude/compartmental-one-year-segments`, the follow-up the sections above
left (`reserving-bayes.md` decisions 16 to 20, "Segments"). As decided
there for `BayesOdp` and `Meyers`, with these particulars:

* **`Compartmental::one_year_segments(triangle, paid, outstanding,
  premium, settings)`** returns a `CompartmentalOneYearFits`: `segments:
  SegmentFits<BayesOneYearSegment<CompartmentalSegment>>` (each segment's
  `CompartmentalFit` with its opening ultimate and reserve, Pareto `k̂`,
  `moment_matched`, `refitted`, `unrefitted` and `refit_warnings`, the same
  `BayesOneYearSegment` the other models use), the joint `cdr` (dimensions
  the key names and `origin`, components over each segment's premium
  origins in turn), and `approximate`. It is not a `BayesOneYearFits`,
  whose tables and `ReserveFit` need a chain-ladder projection that this
  model does not have (the reason `one_year` returns a
  `BayesOneYearFit<CompartmentalFit>` with tables of its own); `to_long`,
  `totals` and `segment(keys)` are built from the segments' premium,
  latest paid, opening reserve and the joint `cdr`, stacked under the key
  columns as `CompartmentalFits`' are.
* **One engine call a segment, on `fit_segments`' seeds.** `one_year` and
  `one_year_segments` share one function (`year_streams`: a fit on the
  chains stream it is given, then the opening draws, the simulations and
  the refits on the view seed that the predictive stream derives), so a
  segment's view is by construction `one_year` of its own triangle on
  `Stream::SegmentChains(k)` and `Stream::SegmentPredictive(k)`, and a
  one-segment triangle's is segment 0's (unit tests; `one_year` itself is
  bit for bit what it was). The posteriors are `fit_segments`' (a unit
  test).
* **Defaults per segment.** The refit default, the 2,000-simulation cap and
  the approximate warning are `one_year`'s: the model's refit mode when the
  settings leave it `None` (model 1 moment-matches, model 2 reweights
  only), `approximate` set once for the view since the model is the same
  in every segment, and `max_refits` a budget of each segment, so `k`
  segments refit at most `k` times the budget (`reserving-bayes.md`,
  "A refit budget").
* **Dependence.** As `Meyers::one_year_segments`: `Independent` as drawn;
  `RankCorrelation` by Iman-Conover on the segments' total claims
  development results (`SegmentDependence::reorder_ranges`, seeded by the
  model's seed, the rows of each segment's results moved whole), so the
  segments' results reach the Spearman matrix and each keeps its
  simulations as a set; `Synchronized` is `Error::Bayes` and a bad matrix
  `InvalidSetting`, both before any sampling. A segment's Pareto `k̂`,
  refit lists and fit are not reordered (they describe the simulations as
  drawn); only the joint `cdr` is, as for `Meyers`.
* **Python** (`prospicio.reserving`): `Compartmental.one_year_segments(
  triangle, paid, premium, *, outstanding=None, incurred=None,
  k_threshold=0.7, refit=None, max_refits=None, n_sims=None)`, returning a
  new `CompartmentalOneYearFits` (`keys`, `index`, `origins`, `premium`,
  `latest_paid`, `opening_ultimate`, `opening_reserve`, `cdr`,
  `approximate`, `k_threshold`, `diagnostics_frame()`, `to_frame()`,
  `totals_frame()`, `segment(**keys)`, and for a single-segment view
  `pareto_k`, `refitted`, `moment_matched`, `unrefitted`,
  `refit_warnings`, `flagged_share`, `diagnostics`, `parameter_names`,
  `draws`, `chains`, `summary()` and `loo()`). The dependence is the
  model's (`Compartmental(..., dependence=, spearman=)`). Warnings: the
  failed diagnostics of each segment's fit (with its label), failed refits,
  simulations left unrefitted and the approximate default, each raised
  once. Not `BayesOneYearFits`: that class needs a chain ladder, as above.
* **R**: `compartmental_one_year_segments(triangle, paid, premium, priors,
  outstanding = NULL, incurred = NULL, <compartmental_one_year()'s
  settings>, dependence = c("independent", "rank_correlation",
  "synchronized"), spearman = NULL)` returning a
  `compartmental_one_year_fits` with the properties of the Python class,
  `as.data.frame()`, `totals_frame()`, `summary()`, `print()` and
  `segment()`. Model 2 is `compartmental_one_year_segments()` with
  `likelihood = "lognormal"`, `scale = "loss_ratio"` and `paid_response =
  "incremental"` (there is no `compartmental_model_2_one_year_segments()`
  wrapper: its sampler defaults are `compartmental_model_2_one_year()`'s,
  which the caller passes). The extendr wrappers in
  `R/prospicio/R/extendr-wrappers.R` were written by hand (rextendr is not
  installed here) in the form the generator writes.
* **Tests.** Rust unit tests (`compartmental.rs`): errors before any
  sampling (ungated), and, gated by `slow-tests`, each segment's view equal
  to `year_streams` of its own triangle on its own streams, twins
  uncorrelated, a one-segment view equal to segment 0's streams and not
  `one_year`'s, a `max_refits` budget of each segment, the tables, and a
  Spearman 0.7 reached with each segment's results kept as a set.
  Validation (`reserving_compartmental_one_year.rs`): model 1 over two
  copies of company 337 in the pull request's smoke run (each segment's
  total result has mean zero and the sd of `one_year` on the triangle
  alone within a band, the copies are rank-uncorrelated, `segment` is its
  part of the joint result) and, under `RISK_RS_FULL_BAYES`, the default
  refit mode with a budget and model 2's default view over segments
  (reweighted, warned, at most 2,000 simulations). Python
  (`python/tests/test_reserving_compartmental_one_year_segments.py`) and R
  (`R/prospicio/tests/test-reserving-compartmental-one-year-segments.R`)
  run model 1 at 2 chains of 100 warm-up and 100 kept draws, 300
  simulations and reweighting only (about 30 s for the Python test, 25 s
  for the R one).
* **Result.** On company 337 under two names (model 1, seed 0, 500
  simulations, reweighting only) the two segments' total results have sd
  8,021 and 8,748 against 7,872 for `one_year` of the triangle alone (the
  band is 0.75 to 1.33 of it), and the copies' rank correlation is within
  `4.5/sqrt(n)` of 0. The smoke run in release is 22 s for the view (34 s
  with the single-triangle comparison); the full run, which adds the
  default refit mode over the segments under a budget and model 2's
  default view, is 225 s. One local run, not pinned by CI.
* **Not done.** A guide section (the 200-triangle calibration study is built,
  `reserving-calibration.md`, "As built (PR 4)");
  `docs/guides/reserving-compartmental.qmd` still says that fitting
  several segments is not built, which has been out of date since PR E
  (the Guides lane's file).

### 9. Bindings

Proposed, in the same PR as each model (`reserving-bayes.md` open
decision 10 (a)):

* **Python**: `Compartmental(priors, likelihood="normal", chains=4,
  tune=1000, draws=1000, n_sims=10000, seed=0, target_accept=0.99,
  max_depth=10)` with `fit(triangle, paid, premium, outstanding=None,
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

**Built (PR B, bindings).** As proposed, with these outcomes:

* **Python** (`prospicio.reserving`): `Compartmental(priors, *,
  likelihood="normal", scale="amount", paid_response="cumulative",
  fitted="paid_and_outstanding", rate_order="free", start="data",
  chains=4, tune=1000, draws=2000, n_sims=10000, seed=0,
  target_accept=0.99, max_depth=10)`: `priors` is the one positional
  argument and has no default (a `TypeError` without it), and the
  settings are keyword-only strings. The defaults are the Rust core's
  (2,000 kept draws, not the 1,000 proposed). A setting not yet built is
  accepted and named, and `fit` raises the core's "not yet built"
  `ValueError`. `CompartmentalPriors` takes every prior as a keyword
  (`(median, sd)` for a positive parameter, `("student_t", nu, scale)` or
  `("normal", scale)` for an effect sd, `("student_t", nu, location,
  scale)` or `("normal", mean, sd)` for each `log sigma`), with
  `CompartmentalPriors.gesmann_morris()` and `replace(...)` to change a
  few. `fit(triangle, paid, premium, *, outstanding=None, incurred=None)`
  returns a `CompartmentalFit` with `reserves`, `paid` and `outstanding`
  (`PredictiveDistribution`s), `diagnostics`, `rate_swaps`,
  `population_origins`, `ultimate_loss_ratio`, `parameter_names`,
  `draws`, `summary()`, `loo()` (an `Elpd`, outstanding cells first) and
  `loo_by_type()` (the elpd split), `to_frame()`, `totals_frame()`,
  `paid_frame()` and `outstanding_frame()`. `prior_predictive(triangle,
  premium, n_sims=1000)` returns a dict of the two path distributions.
* **R**: `compartmental(triangle, paid, premium, priors, outstanding =
  NULL, incurred = NULL, likelihood, scale, paid_response, fitted,
  rate_order, start, chains = 4, tune = 1000, draws = 2000, n_sims =
  10000, seed = 0, target_accept = 0.99, max_depth = 10)`, with `priors`
  required (an error naming `gesmann_morris_priors()` without it), not
  defaulted to the preset as proposed: decision 2 settled priors as a
  required argument. Priors are `compartmental_priors(...)` lists
  (`c(median = , sd = )` and `list(family = , ...)`), with
  `gesmann_morris_priors()`. `compartmental_fit` has the Python fit's
  fields as properties (`paid_table` and `outstanding_table` as data
  frames), `as.data.frame()`, `totals_frame()`, `summary()` and
  `bayes_loo()` (with `n_outstanding`, `elpd_outstanding` and
  `elpd_paid`); `development_frame()` is an error, as for Clark.
  `compartmental_prior_predictive(triangle, premium, priors, rate_order,
  n_sims = 1000, seed = 0)` returns a list of the two path
  distributions.
* Tests: both fit company 337's training triangle with the default
  chains at seed 0 with no warning, and check the total reserve within 2%
  of the Rust value, `b_kp` within 0.1 and `elpd_loo` within 2 of Stan's,
  the frames, the prior predictive check, replay by seed, a warning from
  short chains, the settings not yet built and the missing priors. A
  default fit takes about 5 seconds in a release build.

**Built (PR C, bindings).** Model 2 in both languages, in the style of
PR B's bindings:

* **Python**: `CompartmentalPriors(...)` takes `ker_effect_sd` and
  `kp_effect_sd` (a half prior's tuple, or `None` for no effect) and
  `development_effect_sd` (`None`, or four items on `k_er`, `k_p`, `rlr`
  and `rrf`, each `None` or a half prior), all `None` by default, so PR
  B's calls build model 1 unchanged; getters for the three (the last a
  list of four), `CompartmentalPriors.gesmann_morris_2()`, and in
  `replace` `None` keeps an effect, `False` removes it and a
  `development_effect_sd` replaces all four. `Compartmental.model_2(priors,
  *, rate_order, start, chains=4, tune=2000, draws=8000, n_sims=10000,
  seed=0, target_accept=0.999, max_depth=10)` is the core's
  `Compartmental::model_2`; the constructor accepts every likelihood,
  scale and paid response, and `fitted` other than paid and outstanding
  is still the core's "not yet built" `ValueError`.
  `CompartmentalFit.cell_rate_swaps` is a float, or `None` when the rates
  are shared. `prior_predictive` uses the model's settings, so a
  `model_2` check sums the incremental paths.
* **R**: `compartmental_priors()` gains `ker_effect_sd = NULL`,
  `kp_effect_sd = NULL` and `development_effect_sd = NULL`, the last a
  named list with any of `k_er`, `k_p`, `rlr` and `rrf` (one left out has
  no development effect); `gesmann_morris_2_priors()`;
  `compartmental_model_2()`, `compartmental()` with model 2's settings
  and sampler as defaults; the fit's `cell_rate_swaps` property (`NA`
  when the rates are shared); and `compartmental_prior_predictive()` gains
  `paid_response` (last, so PR B's positional calls are unchanged). The
  priors cross to Rust as 31 numbers and 10 family names, `"none"` for an
  absent effect.
* Tests: both fit company 337's training triangle with model 2's priors
  and settings on shorter chains than `model_2`'s (4 x 1,000 warm-up and
  2,000 draws, about 20 seconds; the default's minute and its
  warning-free seed 0 are checked by the Rust validation), with the
  total reserve within 3% of Stan's 152,242, `b_kp` within 0.5 of Stan's
  -5.064 (posterior sd 1.72) and `elpd_loo` within 3 of 204.29; the new
  parameter names, `cell_rate_swaps`, the presets and their errors, the
  three mixed settings on short chains, the log-normal's non-positive
  response and incremental paid's missing paid before as errors, and the
  prior predictive check with development effects, replayed by seed.

### 10. Docs and knowledge

Proposed. Each PR updates in itself: the Reserving lane's status file (`docs/lanes/reserving.md`), the
v0.7 row and the Reserving section of `docs/roadmap.md`,
`docs/architecture.md` if a roadmap row changes, `CHANGELOG.md`, and this
note's status. `knowledge/` gains: company 337's data and its agreement
with Morris' printed table, with both OS definitions (datasets, PR B); the
swap invariance of the one-OS means and the stable kernel `g` (findings,
`verified` once PR B's test runs in CI); the paper's text and code
disagreements, on model 2's σ prior (text: median 0.1; code: 0.2) and on
the origin hierarchy of both models (text: an additive effect `RLR[i] =
μ_RLR + u` with the sd prior on `u`; code: an effect on the standardised
`o` scale, `RLR_i = 0.7 e^{0.2 o_i}`, with the sd prior on `o`), and the
reconstruction notes for its brms code (references); the chunked-dual
measurement (PR A, in the existing autodiff finding). A user guide page on company 337 (paid and OS
fitted jointly, the prior predictive check) comes with PR C, once both
likelihoods exist.

Built (PR C, Rust core): model 2's code and printed posterior and its
reconstruction in the Gesmann and Morris reference, the gm2 parity
finding, the per-cell gradient in the autodiff finding and the model 2
reference's run time in the CmdStan environment fact. The guide page waits
for PR C's bindings, so that it can show both likelihoods from Python and
R.

Built (PR C, bindings): both likelihoods now reach Python and R. The
guide page belongs in `docs/guides/`, which the Guides lane owns, so it
is left to that lane (or to the user's go-ahead for this one) rather than
written in this PR, tracked as
[#253](https://github.com/EKtheSage/prospicio/issues/253).

### 11. The one-year view (PR F)

Built in Rust, Python and R on `claude/bayes-one-year-compartmental`
(2026-10-15), at the user's decisions of 2026-10-15 (`reserving-bayes.md`
open decisions 12 and 14: every Bayesian model gets the one-year view
through `YearModel`, with moment matching as its default) and of
2026-10-16 (its open decision 15: the defaults below). The engine, the
measurements and the decision are in `reserving-bayes.md` decision 20 and
open decision 15; what is particular to this model:

* `CompartmentalFit` keeps its draws on the unconstrained scale beside
  `posterior` (unchanged, bit for bit), and the view takes the model
  with a level for every origin of the premium column and every age of
  the square: 1997 (no observed cell) and, in model 2, the age-120
  development effects get standard normal `z` added to each draw, their
  posterior, where `fit`'s simulations draw them from the population
  (decision 2). The year's cells then update them.
* The year is every origin's next diagonal of paid and outstanding (1997
  at 12 months, 1988 at 120 on the papers' triangle: 20 cells), as
  amounts; the likelihood converts them to the fit's response (loss
  ratios, increments). The reserve given a draw is `fit`'s, the expected
  paid to the square's last age less the paid so far (decision 5).
* Every effect is stored centred (`u = τ z`) for moment matching, which
  on the `z` scale cannot follow the curved ridge the year's cells make;
  on model 1 that took matching from about 1 in 10 flagged simulations
  rescued to about 19 in 20. On model 2 (87 parameters, effect sds near
  0) matching rescues none on either scale, and a refit takes as long as
  a fit (about 31 s of wall time with 8 running side by side).
* The defaults, as the user decided on 2026-10-16 (`reserving-bayes.md`
  open decision 15): model 1's view moment-matches and refits what
  matching cannot rescue; with development effects in the priors (model
  2, `Compartmental::one_year_refit`) the view reweights only and its
  `approximate` says so (about 0.4 sd of the total off a refit; Python
  and R warn), unless the settings choose a refit mode. Both run at most
  `Compartmental::ONE_YEAR_SIMS` (2,000) simulations unless
  `BayesOneYear::n_sims` says otherwise; the lifetime view keeps
  `n_sims`. Pull requests run model 1's one-year validation without its
  refit comparison; the weekly and manual runs run all of it, model 2
  included.
* A fit needs a future cell, so a fully developed square has no fit and
  no view; the conformance suite's "no year" check samples such a square
  directly in the unit tests.

### 12. Paid and incurred, and one response alone (PRs G and H)

Decided by the user on 2026-10-22: "`Fitted::PaidAndIncurred`: incurred
replaces outstanding as the second response." Built in Rust, Python and R
on `claude/compartmental-paid-incurred`.

* **The model.** Two responses, paid and incurred, with the same parameters
  as paid and outstanding. An incurred cell's mean is `RLR_i w_i (OS(t) +
  RRF_i PD(t))`, with `w_i` the premium on the amount scale and 1 on the
  loss-ratio scale and `OS` and `PD` from the closed form `one_os`. `PD`
  is the cumulative paid even when the paid response is incremental (the
  incurred is a stock, not a flow). The second `sigma` (`log_sigma_os` in
  the posterior, the slot that holds the outstanding's) is the incurred
  response's. Both likelihoods apply; the log-normal needs every incurred
  cell positive, which outstanding does not guarantee (the calibration
  panel has 123 cells with an outstanding at or below zero,
  `knowledge/findings/meyers-calibration-compartmental.md`).
* **Inputs.** The second column of `fit` supplies the incurred.
  `Outstanding::FromIncurred(col)` (Python `incurred=`, R `incurred =`)
  gives incurred cells directly, wherever the column has a value, with no
  need for a paid cell. `Outstanding::Column(col)` (`outstanding=`) gives
  incurred as the outstanding plus the paid, where both are observed. Under
  `PaidAndOutstanding` the two arguments mean what they did, bit for bit.
* **Consumers of the second response.** The fit's `outstanding`
  distribution (and `outstanding_table`, `Future.outstanding`) holds the
  future cells of the second response, so future incurred under this
  setting, drawn with the incurred sigma; the reserve and the paid cells
  are unchanged in meaning. `loo()` splits as before, `n_outstanding` and
  `elpd_outstanding` counting the incurred cells. `fit_segments` fits each
  segment with the same setting. The one-year view (`YearModel`) needs no
  change: its second-response cells use `cell_mean`, so they are incurred
  cells, and the conformance suite passes on it
  (`the_one_year_view_of_paid_and_incurred_conforms`). The diagonal
  backtest's second table scores held-out incurred diagonals under this
  setting (the closed-form density uses the same mean). The prior
  predictive check keeps drawing the mean outstanding and paid paths.
  The model name in the provenance is `compartmental` (not `_gm1`).
* **Validation (decision 1: each shipped value has its own reference).**
  `scripts/stan/compartmental_gm1_incurred.stan` is the model 1 program
  with the one changed mean, run by
  `scripts/reserving_compartmental_incurred_stan.py` on company 337 (the
  same data, settings and tolerances as model 1) into
  `reference/reserving_compartmental_incurred_stan.csv`, read by
  `validation/tests/reserving_compartmental_incurred.rs`.
  Built result: all 3,040 reference-row comparisons of a 20-seed sweep pass at model 1's tolerances (not re-measured on 200 seeds for this variant), and the total reserve is 175,318 (sd 9,469) against Stan's 175,570 (9,478), 1.3% above the paid and outstanding fit's 173,341. The results are in `knowledge/findings/compartmental-paid-incurred.md`.
  Unit tests check the log density against a hand computation, the
  gradient against central differences and against the generic dual pass,
  and the incurred cells against the two ways of building them.
* **One response alone (PR H).** Decided by the user on 2026-10-22:
  `Fitted::PaidOnly` and `Fitted::IncurredOnly` require
  `RateOrder::Ordered`, and without it the fit returns an `Error::Bayes`
  that says why. Built in Rust, Python and R on
  `claude/compartmental-one-response`, stacked on PR G. The choices:
  * *Why ordered rates.* From one response the means are unchanged by the
    rate swap `(k_er, k_p, RLR, RRF) -> (k_p, k_er, RLR k_er / k_p, RRF
    k_p / k_er)`, and from paid alone only `RLR x RRF` is identified
    (decision 2), so under free rates the posterior has two modes the data
    cannot tell apart. `validate` refuses `RateOrder::Free` for both
    settings before any data is read, so every entry point that fits
    (`fit`, `fit_segments`, `one_year`, the backtest) refuses it. Under ordered rates no draw has the
    swap, so the rate-swap warning never fires (`rate_swaps` is 0).
  * *One sigma.* The parameter vector drops the sigma of the response not
    fitted: `log_sigma_pd` (named `sigma_pd`) under `PaidOnly`,
    `log_sigma_os` (`sigma_os`, the incurred's) under `IncurredOnly`. Its
    prior is the same slot of `CompartmentalPriors::log_sigma`; the other
    slot is not read. Model 1 on company 337 has 26 parameters (27 with
    both responses).
  * *The second column under `PaidOnly`.* `fit`'s Rust signature is
    unchanged, and its `outstanding` argument is not read (any value will
    do; the input hash and the provenance skip it, the provenance records
    `"not read"`). Python and R accept neither `outstanding` nor `incurred`
    under `"paid_only"` (and still accept one). The fit has no second
    response, so `CompartmentalFit::outstanding` and
    `CompartmentalFits::outstanding` are now `Option<PredictiveDistribution>`
    (a breaking change of the Rust field type), `None` under `PaidOnly`
    (Python `None`, R `NULL`), and `outstanding_table` returns an error.
  * *`IncurredOnly`.* The incurred is given as under `PaidAndIncurred`
    (`Outstanding::FromIncurred` or an outstanding column plus paid). The
    mean of an incurred cell is `RLR_i w_i (OS(t) + RRF_i PD(t))`. The paid
    column supplies only each origin's latest paid. The reserve is the
    expected paid at the square's last age less the latest paid, and each
    future paid cell is its expected value without noise: no paid cell is
    fitted, so there is no paid sigma to draw it with, and inventing one
    from the incurred sigma has no basis. The reserve's sd is then
    parameter uncertainty only. The future incurred cells (in
    `outstanding`) are drawn with the incurred sigma.
  * *The start.* The least squares of the population curve behind
    `CompartmentalStart::Data` (decision 5) is not identified from one
    response and runs off along the free direction: on company 337 paid
    alone it ends at `RLR` 5e-136 and `RRF` 1e134 (log posterior -6.0
    million), incurred alone at `RRF` 1e-13 and `k_p` 0.03 (-44,700),
    against -456 and -452 at the prior centre. NUTS with the default
    warm-up walks back from there, but a short warm-up stays stuck (chains
    with `k_er` up to 340,000). So under one response the data start is
    kept only when its log posterior is at least the prior centre's, and
    otherwise the pilot run starts at the prior centre; with two responses
    the start is unchanged, bit for bit.
  * *Consumers.* `loo()` has the one response's cells (45 on company 337):
    `n_outstanding` and `elpd_outstanding` are 0 under `PaidOnly`, and
    `elpd_paid` is 0 under `IncurredOnly`. The one-year view's year has the
    cells of the fitted response only (under `IncurredOnly` the paid in the
    year is 0 and the closing reserve runs from the latest paid), and the
    conformance suite passes on both settings
    (`the_one_year_view_of_paid_alone_conforms`,
    `the_one_year_view_of_incurred_alone_conforms`). In the diagonal
    backtest `PaidOnly` scores the paid table and refuses the second
    table, and `IncurredOnly` scores the second table and refuses the paid
    one (the held-out paid would be scored against a noiseless forecast).
    The prior predictive check still draws the mean outstanding and paid
    paths. R's one-year views and backtest candidates keep fitting paid and
    outstanding, as under PR G; Python and Rust take every setting there.
  * *Validation.* `scripts/stan/compartmental_gm1_paid_only.stan` and
    `compartmental_gm1_incurred_only.stan`, model 1 with ordered rates and
    one response, run by `scripts/reserving_compartmental_one_response_stan.py`
    on company 337 into `reference/reserving_compartmental_one_response_stan.csv`
    (276 rows), read by `validation/tests/reserving_compartmental_one_response.rs`.
    Tolerances are model 1's (decision 7) except `elpd_loo`, which takes
    2.5 instead of 1.5, measured on a sweep of 20 seeds per setting: the
    incurred-only `elpd_loo`, whose largest Pareto k is above 1, was up to
    2.1 from Stan's. With it all 5,520 comparisons of the 40 fits pass.
    At the defaults 17 of the 20 paid-only fits passed every diagnostic
    threshold (two had one divergent transition each, one an R-hat of
    1.013 on `tau_RRF`), and all 20 incurred-only fits. The results are in
    `knowledge/findings/compartmental-one-response.md`.
  * *The diffuse preset (decided by the user on 2026-10-22).* On company
    337 incurred alone gave a 26% higher reserve than paid and outstanding
    because, with no paid cell, `k_p` comes from its prior and the paper's
    is `LogNormal(median 1, sd 0.1)`. The user: "have a diffuse prior where
    we don't influence the fit much as a default and have the option for
    users to implement a narrow prior if they have info not in the data."
    Built: `CompartmentalPriors::diffuse()` (Python
    `CompartmentalPriors.diffuse()`, R `compartmental_diffuse_priors()`),
    the paper's medians (`k_er` 3, `k_p` 1, `RLR` 0.7, `RRF` 0.8) with a
    log-scale sd of 1 on each of the four (the paper's are 0.1 to 0.2),
    and model 1's origin effect sds and `log sigma` priors. The prior
    predictive check (`the_diffuse_preset`) shows paid at the last age as a
    share of premium at 0.06, 0.53 and 5.5 (5th, 50th, 95th percentiles),
    against the paper's 0.39, 0.56 and 0.81: the same median, the plausible
    range and well beyond it. Priors stay a required argument (open
    decision 2); `diffuse()` is the documented choice for a fit to one
    response alone, and a narrower prior (the paper's, or a struct update
    of this one) is the option of a user who holds information about the
    rates that is not in the data. The docs of `Fitted::PaidOnly`,
    `Fitted::IncurredOnly`, the presets, and the Python and R help say so
    (since 2026-10-23 `diffuse()` is the starting point for paid alone
    only; incurred alone refuses it, see "A payment-rate prior the user
    believes" below).
  * *What company 337 shows.* The diffuse prior works for paid alone: `k_p`
    is pinned by the paid (the posterior sd of its log is 5% of the
    prior's), the fit converges at the defaults (5 seeds: max R-hat at most
    1.009, no divergences) and the total reserve is 181,500 (sd 12,400),
    2.9% above the paper's preset on paid alone and 4.7% above paid and
    outstanding. It does not work for incurred alone: the incurred does
    not identify `k_p`, the posterior sd of its log is 60% of the prior's,
    and the total reserve is -51,000 (sd 152,000), negative on average,
    because the paid run-off is whatever the wide prior and the ridge
    between `k_p`, `RRF` and `RLR` allow. Between the two, the reserve falls
    as the prior widens: 203,000 at a log-scale sd of 0.3 on the four
    priors, 163,000 at 0.5, 88,000 at 0.7 and -51,000 at 1. So there is no
    prior on `k_p` that leaves incurred alone's reserve to the data; every
    value is the prior's choice. The numbers are in
    `knowledge/findings/compartmental-one-response.md`.
  * *The warning.* Under `Fitted::IncurredOnly` the fit adds a warning to
    its diagnostics (so Python raises its warning and R its) when the
    posterior sd of `log k_p` is more than half the prior's: "the posterior
    sd of log k_p is N% of its prior's: the incurred does not identify the
    payment rate ...". It follows from the mechanism that already exists
    (`diagnostics.warnings`, as the rate-swap warning), needs no new
    setting, and is silent when the data pin `k_p`: the share is 0.99 under
    the paper's preset, 0.6 under the diffuse one and 0.7 to 0.9 between,
    against 0.05 to 0.4 when paid is fitted (not checked there). The
    threshold of 0.5 sits below every incurred-alone fit measured. A user
    who sets a narrow prior on purpose still gets the warning, as the
    reserve is then their prior's; the warning says what the number
    depends on rather than that it is wrong. The check is on the prior's
    width, not its centre, so it cannot tell a good prior from a bad one.
  * *A payment-rate prior the user believes (decided by the user on
    2026-10-23).* Incurred alone with the diffuse default gives a reserve
    that is not usable, which the user's rule ("the data set the fit")
    cannot repair, since the data do not set `k_p`. The options were (a)
    keep the preset and the warning, (b) refuse a `k_p` prior wider than
    some sd, (c) carry the prior's role in the fit's type. The user chose
    (b): "Require a payment-rate prior. Incurred-only refuses a k_p prior
    wider than a set width, and the user must give one they believe.
    Diffuse stays the starting point for paid alone. The warning stays."
    Built: `Compartmental::validate` returns `Error::Bayes` under
    `Fitted::IncurredOnly` when the log-scale sd of the `k_p` prior exceeds
    `MAX_INCURRED_ONLY_KP_SD`, 0.5 (the bound itself is accepted: 0.5 and
    the paper's 0.1 pass, 1.0 and 0.51 do not). `PositivePrior` has one
    variant, `LogNormal { median, sd }`, whose `sd` is the log-scale sd
    (`median e^(sd o)`, `o ~ N(0, 1)`), so the rule reads `sd` directly; a
    new variant would have to define its log-scale sd for the rule. The
    check is in `validate`, which `fit`, `fit_segments`, `one_year`,
    `one_year_segments` and the diagonal backtest each call before any data
    are read, so all of them refuse; `prior_predictive` does not apply it,
    so the user can still see what a refused set implies. The message says
    that incurred alone does not identify the payment rate, so the reserve
    follows this prior, and that the user must give a `k_p` prior they
    believe (from the book's paid history or a paid-and-incurred fit of a
    similar book) or fit paid as well. Only `k_p` is checked: the other
    three priors may stay wide (the reserve moved with `k_p`'s, which is
    where the paid run-off comes from). The bound 0.5 is the widest width
    measured whose reserve is still informative (163,000, sd 51,500, `k_p`
    0.41, against 203,000 at 0.3 and 87,700 at 0.7); it is a rule, not a
    result, and the warning stays for any accepted prior. `diffuse()` stays
    the documented starting point for `Fitted::PaidOnly`, and the docs of
    `Fitted::IncurredOnly`, `Fitted::PaidOnly`, `diffuse()`, the priors'
    struct, and the Python and R help say so.
* **Unequal grain (PR I).** Built as decision 13.

A check of what the priors imply for the reserve, in particular the incurred-only `k_p` prior, is noted but not planned in `reserving-bayes.md`, "Prior predictive checks (a note, not planned)".

### 13. Unequal origin and development grain (PR I)

Decided by the user on 2026-10-22: unequal grain is in scope. The
user confirmed time in years the same day, "for now": how to fit
quarterly or monthly data well (rate priors that suit sub-annual
development, grouping development effects by year, seasonality, and
a calibration check on sub-annual books) is open, the lane's Next
item `[compartmental-subannual]`. Built in
Rust, Python and R on `claude/compartmental-grain`, stacked on PR H. The
choices:

* **Which grains.** The origin grain must be a whole multiple of the
  development grain: annual origins with semiannual, quarterly or monthly
  development, semiannual origins with quarterly development, quarterly
  origins with monthly development, or equal grains. Otherwise the fit
  returns an `Error::Bayes` that names both grains. A `Triangle` already
  requires that its development grain divide its origin grain, so the
  check guards the model's own invariant and no triangle a user can build
  reaches it.
* **Time is in years on every grain.** Each cell's development time `t`
  in the closed form is its age in months over 12, as it was before this
  PR on every grain, equal or not. On annual origins this is the age in
  origin periods. The user's brief asked for origin periods, "so
  equal-grain fits stay bit for bit unchanged"; the two differ only for
  sub-annual origins, where origin periods would change every fit on an
  equal quarterly or monthly triangle (`t` of 1, 2, ... quarters where it
  is 0.25, 0.5, ... years today) and would make the rates per quarter, so
  that the paper's preset (rates per year, fitted to annual data) no
  longer means what it says. Years keep both: every existing fit is bit
  for bit unchanged (checked on company 337 annually and relabelled as
  quarterly origins and development, models 1 and 2), the rates are per
  year whatever the grains, and the same book fitted on annual or on
  quarterly development has comparable rates. The Stan programs already
  take `t` per cell as data, so the reference needs no new program.
* **The square.** The square runs for as many origin periods as the
  premium column has origins, in development periods (decision 5's
  horizon, "future paid to the last age of the square"): `n_origins x
  origin months / development months` ages, or the latest observed age if
  later. Ten annual origins with quarterly development have 40 ages, to
  120 months, the same horizon as the annual square. Each origin's
  reserve is its paid at that age less its latest paid.
* **Incremental paid** is the increment over one development period: a
  paid cell's increment is measured from the paid of the development
  period before, with the closed form at both times (`t` and `t` less a
  quarter on quarterly development).
* **Model 2's development effects** have one level per development age
  with an observed cell, as on annual data: 40 levels on a ten-year
  quarterly square, against 10 on the annual one, each shrunk by the same
  population sd. A grouping of the ages by development year (four
  quarterly ages sharing one effect) would keep the level count of the
  annual model, but it needs a second index per cell and a choice of the
  grouping, and with one level per age the rule is the same on every
  grain; it is not built.
* **The one-year view.** The year is the cells valued in the twelve
  months of calendar time after the valuation, as the generic engine
  defines it for every model (`YearCells`, `reserving-bayes.md` decision
  16: "one per open origin with an annual grain, up to four with a
  quarterly one"), so with quarterly development it holds up to four paid
  and four outstanding cells per origin, chained as increments under
  incremental paid. `CompartmentalYear` already found the year by
  valuation date, so it needed no change; the conformance suite passes on
  a quarterly triangle for models 1 and 2
  (`the_one_year_view_on_unequal_grain_conforms`; the light refit's own
  score check failed for model 2 on a near-noiseless version of that
  triangle, with 2% noise, and passes with 8%).
* **Segments and the backtest** use the same data path, so both work:
  `fit_segments` fits each segment on its own square, and the diagonal
  backtest holds out calendar periods at the development grain (a
  quarter, `TriangleFrame`), each split's training data keeping the
  square of the whole triangle. A held-out cell at an age no training row
  has (the oldest origin's latest quarter) is excluded from scoring by
  the generic backtest's rule, as on annual data.
* **The prior predictive check** draws the same square (origins by
  development ages).
* **Bindings.** Python and R take the grains from the triangle, so no
  argument changed; the help says what the square and the time are.
* **Validation.** No public dataset has unequal grain, so
  `scripts/compartmental_grain_simulate.py` simulates one from model 1
  with known parameters (accident years 2011 to 2020, quarterly
  development, 220 outstanding and 220 paid cells,
  `data/compartmental_sim_quarterly.csv`), and
  `scripts/reserving_compartmental_grain_stan.py` runs
  `scripts/stan/compartmental_gm1.stan` unchanged on it into
  `reference/reserving_compartmental_grain_stan.csv`, read by
  `validation/tests/reserving_compartmental_grain.rs` at model 1's
  tolerances (decision 7). All 2,620 reference-row comparisons of a 20-seed sweep pass, the largest at 0.51 of its tolerance (a parameter mean), and the total reserve is 74,834 (sd 1,900) at seed 0 against Stan's 74,873 (1,895); every seed passed every diagnostic threshold (max R-hat at most 1.0064, no divergent transition). The Stan program draws a reserve for 2011, observed at the square's last age, where decision 5 gives it 0, so the generator sets Stan's to 0 too. The default fit recovers the truth:
  each of the 31 true parameters (the population's, the sigmas, and each
  origin's `RLR` and `RRF`) lies within the central 95% of its posterior,
  and the total reserve's 74,834 is 0.23 sd above the truth's expected
  74,389. A default fit takes about 32 s in release (440 cells, against
  5.5 s on company 337's 90). Unit tests check the ages and times, the
  square, the refused grains, incremental paid, the gradient against
  central differences, the prior check, a fit, segments, the backtest and
  the one-year conformance on a quarterly triangle. The results are in
  `knowledge/findings/compartmental-grain.md`.

### 14. A paid development pattern per posterior draw

Status: **proposed and built as recommended, awaiting the user's
acceptance** (2026-10-28), issue #679 (`reserving-v02.md` decision 28,
item 7), branch `claude/pattern-draws-compartmental`, in Rust, Python and
R. `CompartmentalFit` and `CompartmentalFits` carry `patterns`
(`DevelopmentPatterns`): row `i` of a segment's patterns is the paid
pattern of the posterior draw that simulation `i` took, the same
simulation as row `i` of that segment's `reserves`, `paid` and
`outstanding`. The choices decision 28 left open:

1. **What the curve is.** The cumulative paid of an origin with nothing
   observed under the draw: the closed form of decision 2 for the draw's
   population rates, `RLR RRF PD(t)` with `PD(t) = (1 - e^(-k_p t)) - k_p
   g(t)` (the ODE's solution, reused through `one_os`; no numerical solver
   is added), at each age `t` of the square in years. The premium cancels.
   It is the parameter uncertainty in the pattern only, the same for every
   origin, with no process error (decision 28 item 1). Nothing new is
   sampled: the pattern is computed from the draw the simulation already
   took, so the cells and the reserves are bit for bit what they were
   (a test replays the cells without it).
2. **The normaliser: the paid at the square's last age** (alternative:
   the curve's limit as `t` grows, the ODE's ultimate paid). The reserves
   are the paid at the square's last age less the latest paid (decision
   5's horizon), so the pattern divided by the paid at that age is what
   the reserve measures, and it reaches exactly 1 at the last age, as
   decision 28 item 2 says the Bayesian models do. The tail past the
   square is not in the model, so no tail share is invented; with `k_p`
   near 0.4 on company 337 the mean paid at ten years is 0.981 of the
   ODE's limit (`knowledge/findings/compartmental-pattern-draws.md`, from a
   short fit), and a user who wants the limit can read the curve from
   the posterior. The ages are the square's, in months: the premium
   column's origin periods in development periods (ten annual ages on
   company 337, twenty for quarterly development of five annual origins),
   not only the ages the triangle observes.
3. **The level: the population** (alternative: one origin's own rates,
   rejected because `DevelopmentPatterns` holds one pattern per segment
   per simulation). Origin effects are at zero, the expected effect of a
   new origin: `k_er`, `k_p`, `RLR` and `RRF` are the draw's population
   values (`nat[0..4]`). `RLR` and `RRF` cancel in the share unless a
   development effect on them differs by age (model 2), when they enter as
   in item 4.
4. **Development-age effects (model 2) enter as they enter the mean.** A
   cell's parameters are the origin's times `e^(s v)` with the draw's
   effect `v` at the cell's age; the pattern uses the multipliers that the
   simulation of the same row used: the draw's own effect at an age that
   has an observed cell, and at an age without (120 months on company 337,
   the square's last age) the effect the simulation drew from the
   population for that row's future cells, shared by every origin.
   `simulate` returns the multipliers it used, so the pattern takes no
   random number of its own and describes the development of that row's
   reserves (alternative: a multiplier of 1, the population's zero effect,
   at an age without a cell, rejected because the row's reserves did not
   use it). Cumulative paid at age
   `k` is `RLR RRF PD(t_k)` with that age's parameters; under incremental
   paid it is the sum of the increments `PD(t_j) - PD(t_(j-1))` with each
   age's parameters, as the simulation writes them. Under incremental
   paid (model 2's preset) the shares rise strictly whatever the
   effects: each increment is `RLR RRF (PD(t_k) - PD(t_(k-1)))` at one
   age's parameters, `RLR RRF > 0`, and `PD` rises in `t` for any
   positive rates (`PD'(t) = k_p k_er g(t) > 0` for `t > 0`). Only
   cumulative paid with development effects can make a share fall, when
   an age's `RLR RRF PD(t_k)` lies below the age before it; the monthly
   conversion refuses that by default and `Monotone::RunningMax` holds
   it (decision 28 item 5). Without development effects (model 1,
   shared rates) the share rises with age under either response.
5. **Settings that change nothing here.** The likelihood (normal or
   log-normal), the loss-ratio or amount scale and the pair fitted do not
   change the curve; under `Fitted::IncurredOnly` no paid cell is fitted,
   so the paid rate `k_p` and the pattern come from its prior, which the
   fit already warns of (decision 12). The pattern is a paid pattern
   whatever the column named, as decision 28 item 7 says of every fit.
6. **Over segments.** `fit_segments` fits each segment alone on its own
   seeds, then, as it does for the reserves and the future cells, puts a
   segment's pattern columns after its other columns in the row that
   Iman-Conover moves whole (`reorder_rows`), so a segment's pattern in
   row `i` is the pattern of the draw whose reserves are in row `i` of
   that segment after the reorder; a test checks it against the same fit
   without the correlation. `CompartmentalFits::patterns` joins the
   segments' patterns by label, and `segment(...)` keeps its own.

Bindings: Python `fit.patterns` on `CompartmentalFit` and
`CompartmentalFits` (the `DevelopmentPatterns` of decision 28 item 8); R
`pattern_draws()` and `monthly_pattern_draws()` take a `compartmental_fit`
or `compartmental_fits`. Breaking in Rust: the two fit structs gain a
public field.

## Risks

* **Prior-data conflict.** On company 337 the data already overrule the
  preset: the paper's model 1 posterior puts `k_er` at about 1.75 and
  `k_p` at about 0.43, 5.4 and 8.5 prior sds from medians 3 and 1, so the
  preset's own rationale (sec. 5.2.1: `k_p` between 0.85 and 1.2 with
  about 95% probability) does not describe the fitted result. The
  posterior is a compromise between tight rate priors and data that
  disagree with them, so it is sensitive to those priors, and on another
  book the preset may be further off while the fit still looks healthy.
  The mitigation is open decision 2 (priors are an explicit choice), the
  prior predictive check, and a docs example that compares prior and
  posterior on company 337 and shows the conflict.
* **Sampler geometry.** Funnels in the origin effects (worse in model 2,
  with development effects too), a posterior far from the prior centre,
  and the swap (excluded by about 150 nats at the posterior, decision 2).
  The data-based start (decision 5), the non-centred effects, the pilot
  whitening and `target_accept` 0.99 are the mitigation, and a test checks
  convergence from both the default start and `o = 0`; a reference run
  with divergences at `adapt_delta` 0.99 would mean tightening, and the
  note would record it.
* **The reference is a reconstruction.** A brms convention missed in
  reconstructing its generated program (those listed under References: no
  intercept centring under `nl = TRUE`, the `sd` priors' truncation, the
  `|ID|` correlation, the priors on `log σ`) shifts both sides together
  and hides from the parity test; the printed posteriors in decision 7
  are the guard. If they disagree beyond the looser tolerance, open
  decision 6 (b) is the fallback.
* **Model misfit.** The papers state the limits: a non-steady exposure is
  absorbed into `k_er`; cohorts are assumed homogeneous, so large-claim
  books fit poorly; short histories are not credible. A change in case
  reserve strength (company 337 in 1996 and 1997) makes incurred data
  misleading. The docs say so and the backtest of PR E measures it: it
  forecasts company 337's held-out outstanding to within 8% at each of
  three splits and its paid increments poorly, and does not isolate
  the shift (
  `knowledge/findings/compartmental-segments-backtest.md`).
* **Dual cost.** If PR A's measurement fails its target, PR B's gradient
  is written by hand (decision 4), which adds a finite-difference test
  but no risk to the result.

## Open decisions for the user

Decided 2026-10-13 by the user: every item as recommended ((a) for 1, 2,
3, 5, 6, 7 and 8; (c) for 4), with item 1 widened so that the likelihood,
the response scale, cumulative or incremental paid and the pair fitted
are settings from PR B on (decision 1).

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
   not choose them silently; on company 337 the data already pull the
   rates 5 to 9 prior sds from the preset (decision 2). Parity tests use
   the preset.
3. **Data and the OS definition.** (a) Company 337 workers' compensation
   from the CAS data model mart (the source decided for group 353), with
   both OS definitions and both premiums in the CSV; parity on the papers'
   OS (with bulk and IBNR) and direct premium; (b) the `raw` R package's
   `wkcomp` (not installed; a download). *Recommended: (a)*: same source,
   pinned release and checks as group 353, and Morris' printed table
   agrees with it.
4. **Guarding the rate swap.** (a) The paper's independent rate priors,
   with a warning when a draw's population rates have `k_er ≤ k_p` (in
   model 2, the population intercepts; the share of cells with `k_er ≤
   k_p` is reported as a diagnostic); (b) an ordered pair `k_er = k_p +
   e^δ` on the population rates by default; (c) (a) by default with (b)
   as a `rate_order` setting. *Recommended: (c)*: (a) keeps the parity
   with the paper exact, the warning is expected never to fire under its
   priors (the swap costs about 150 nats at the paper's posterior), and
   (b) is there for a user who widens them.
5. **Reserve horizon.** (a) Future paid to the last age of the square,
   with the model's ultimate reported as a mean; (b) also a reserve to
   ultimate with process error past the square. *Recommended: (a) now*,
   as `Meyers` and the paper's validation; (b) on request.
6. **Reference route.** (a) The paper's brms code reconstructed as the
   Stan program brms generates (the conventions listed under References)
   and run with CmdStan 2.39.0 under cmdstanpy, as Meyers' code was;
   (b) install brms and cmdstanr in R 4.5.3 to run the published code as
   is (package downloads, needs the user's approval); (c) printed numbers
   only. *Recommended: (a)*, with the printed posteriors as a looser
   check, and (b) only if the reconstruction misses them.
7. **Gradients for the closed-form models.** (a) Chunked duals
   (`reserving-bayes.md` decision 15), with a hand-written gradient only if
   PR A's measurement misses its target; (b) hand-written from the start,
   duals as the oracle. *Recommended: (a)*: one generic density per new
   model is the point of decision 15, and the closed form keeps (b) cheap if
   needed.
8. **Extensions.** (a) Multistage exposure, market cycles, AR(1) residuals
   and calendar shocks each as a later PR on request; (b) plan them now.
   *Recommended: (a)*: none has a reference we can rerun yet, and each is
   a model of its own.
