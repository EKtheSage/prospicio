# Design note: excess-layer loss development (excess LDFs, benchmark blending, a layer BF prior)

Status: **Proposed** (2026-10-21; issue #192, which the user chose on 2026-10-21; the points under "Open decisions" wait for the user) · Lane: Reserving
(`crates/prospicio-reserving/`; `prospicio-prob` severities used read-only;
`prospicio-pricing` exposure curves reached only through the bindings, see
open decision 2) · Depends on: `reserving-v02.md` (decision 2, the
expected-loss family; decision 12, selected development patterns),
`pareto.md` and `aggregate.md` (Aggregate lane, read only: Pareto and
exposure-curve layer shares)

## Goal

Excess layers develop more slowly than ground-up losses, and a layer's own
triangle is sparse: most cells are zero and the first non-zero link ratios
are enormous. Today the chain ladder, BF and Cape Cod run on a triangle of
layer losses, which is unstable on such data (#192). Three pieces are
missing:

1. **Excess LDFs from a ground-up triangle and a severity curve.** The
   ground-up triangle is dense and stable; the severity model says how
   much of the ground-up development lands in a layer.
2. **Credibility blending** of the layer's own development with benchmark
   patterns (the derived pattern, an industry pattern).
3. **A BF prior from the layer share**: the a-priori layer ultimate is
   exposure times a ground-up loss ratio times the layer's share of
   ground-up loss, the share taken from a Pareto (or any severity, or an
   exposure curve).

No claim-level reserving (v0.8, parked) is needed: everything here works on
an aggregate triangle plus a severity assumption.

## What exists

* `Development` takes a `SelectedPattern` (`reserving-v02.md` decision 12):
  a full or partial benchmark pattern, in three kinds, with a weight per age
  that blends it with the triangle's own factors, `w * selected + (1 - w) *
  estimated`. **The blend of item 2 is therefore already an engine feature.**
  What is missing is the pattern and the weights.
* `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and `CapeCod` take a
  scalar `apriori` loss ratio and an exposure column. A layer-share prior
  needs no engine change: the exposure column is premium times layer share.
* `Severity::lev` and `Severity::layer` (`prospicio-prob`), `Pareto` and
  the other families; `ExposureCurve::layer_share` (`prospicio-pricing`).
* `Triangle.link_ratios` for diagnostics; `PreparedLosses` and
  `DevelopmentPattern` in `prospicio-pricing` for per-claim IBNER on the
  experience-rating side (Aggregate lane, not touched).

## The method

### 1. Layer LDFs from ground-up development

Let the layer be `l xs a` (`b = a + l`). At maturity `j`, claims have count
`n_j` and a severity distribution `S_j` (the severity of the claims as
reported at `j`, including their case reserves). The expected layer loss is

    L_j(a, b) = n_j * ( LEV_j(b) - LEV_j(a) )

and the layer's age-to-ultimate factor is

    CDF_layer(j) = [ n_inf * (LEV_inf(b) - LEV_inf(a)) ]
                   / [ n_j  * (LEV_j(b)  - LEV_j(a))  ]

The ground-up factor is the same expression with `a = 0`, `b = infinity`.
The count ratio `n_inf / n_j` is the claim-count development (from a count
triangle if one exists, else 1 where counts are complete early). So the work
is to say how `S_j` moves to `S_inf`.

**Illustration (own calculation, to become a unit test).** A lognormal
severity with `sigma = 1.5` and mean 50,000, every claim scaled up by 1.3 on
the way to ultimate and counts fixed. The ground-up LDF is 1.3. The factor of
the layer 1m xs 0 is 1.275, of 750k xs 250k is 1.545, of 2m xs 1m is 1.761
and of 7m xs 3m is 1.982. A ground-up factor of 1.3 puts a factor near 2 on
a high layer, because development pushes claims over the attachment, and the
sparse layer triangle cannot show that until late.

How `S_j` becomes `S_inf` is the modelling choice. The literature offers
these; the note proposes to build the first two.

* **Scale shift** (the default): `X_inf = c_j X_j`. For a Pareto with
  fixed shape the threshold scales; for a lognormal `mu` moves by `log c_j`.
  One parameter per maturity, identified from the ground-up triangle:
  `c_j` solves "ground-up severity LDF equals the triangle's ground-up LDF
  over the count LDF". This is what the triangle can identify without claim
  detail, and it is what the illustration does.
* **Shape drift** (Pinto and Gogol): the Pareto shape `alpha_j` falls with
  maturity (a heavier tail at later ages) and is fitted by maturity, to
  layer data or claim data. It needs a severity at each maturity, so it
  needs claim-level or layered input. The layer LDF is then a ratio of
  Pareto layer expectations at `alpha_j` and `alpha_inf`, a closed form.
* **Inflation leverage** (Sahasrabuddhe; Bloom and Patrik): trend acts on
  the whole size distribution, so a layer's trend, and with it the
  development between calendar years, exceeds the ground-up one. Handled by
  trending `S` before the layer is cut, i.e. the scale shift with the trend
  between accident years. Needs a trend input per origin.
* **Direct claim development** (Gillam and Couret; Mahler; Boor): a
  probability distribution of ultimate given the reported value, claim by
  claim. Out of scope here; the `prospicio-pricing` IBNER
  (`DevelopmentPattern`) is the claim-level counterpart.

Bloom and Patrik's warning shapes the design: excess triangles carry
too little data and mis-state inflation's leverage, but the ground-up route
is only as good as the severity assumption, and a retention that changes by
year distorts the layer data. The method therefore reports its inputs (the
`c_j` and the implied ground-up check) and never hides the severity
assumption.

### 2. Blending with the layer's own triangle

Use decision 12's blend: `f_j = w_j * f_derived_j + (1 - w_j) * f_own_j`,
where `w_j` is the weight on the derived or benchmark pattern. Sources of
`w_j`:

* **Fixed by the user** by age (`PatternWeight::ByAge`), which exists.
* **From the data**: `w_j = 1 - n_j / (n_j + k)`, with `n_j` the number of
  non-zero link observations at age `j` (or the claim count in the layer)
  and `k` a user constant (the limited fluctuation or Bühlmann form).
* **Bühlmann–Straub** over the layer's origins, with the within and
  between variances from the link ratios. Possible later; `prospicio-pricing`
  has the credibility routines (Aggregate lane), reached from the bindings
  only (open decision 2).

Decision 12's rule that selected ages carry no parameter error, and a
blend carries a `(1 - w)` share of it, applies unchanged, so Mack, the
one-year view and the bootstraps need no new code.

### 3. A BF prior from the layer share

For origin `i`, the a-priori layer ultimate is `E_i * ELR * s`, with exposure
`E_i`, a ground-up expected loss ratio `ELR` and the layer share

    s = (LEV(b) - LEV(a)) / mean               (a severity, ultimate basis)
    s = ExposureCurve::layer_share(l, a, mpl)   (an exposure curve)

For a Pareto severity `s` has a closed form (`pareto.md`). Because
`BornhuetterFerguson` and `CapeCod` take an `apriori` scalar and an
exposure column, the helper returns the **layer exposure column**
(`E_i * s`, or `E_i * s_i` when the severity is trended by origin) and the
caller passes it. Cape Cod then estimates the layer loss ratio from the
layer's own used-up exposure, as it does today. An uncertain `s` (a Pareto
shape with a standard error) becomes a prior for the Bayesian BF only
after #410's normalised-pattern BF exists; that is a later PR.

## API

Rust, a new module `excess` in `prospicio-reserving`. Names are proposed.

```rust
pub struct Layer { pub attachment: f64, pub limit: f64 }

/// How the severity at each maturity relates to the ultimate severity.
pub enum SeverityByMaturity {
    /// X_inf = c_j * X_j; c_j solved from the ground-up triangle.
    ScaleShift { severity: Box<dyn Severity> },
    /// A Pareto whose shape changes with maturity (Pinto and Gogol).
    ParetoShape { alphas: Vec<(Lag, f64)>, threshold: f64 },
}

pub struct ExcessPattern {
    pub ages: Vec<Lag>,
    pub layer_ldf: Vec<f64>,       // age to next age
    pub layer_cdf: Vec<f64>,       // age to ultimate
    pub ground_up_ldf: Vec<f64>,   // the check
    pub shifts: Vec<f64>,          // c_j, for ScaleShift
}

impl ExcessPattern {
    /// Layer pattern from the ground-up column and an optional count column.
    pub fn from_ground_up(triangle: &Triangle, column: &str, counts: Option<&str>,
                          layer: Layer, model: &SeverityByMaturity) -> Result<Self>;
    /// As a `SelectedPattern` of kind `AgeToAge`, with the given weight.
    pub fn selected(&self, weight: PatternWeight) -> SelectedPattern;
    /// The weight on the benchmark from layer observation counts: 1 - n/(n + k).
    pub fn credibility_weights(layer_triangle: &Triangle, column: &str, k: f64)
                          -> Result<PatternWeight>;
}

/// The layer exposure column: exposure times the layer's share.
pub fn layer_exposure(exposure: &[f64], severity: &dyn Severity, layer: Layer) -> Result<Vec<f64>>;
```

Python (`prospicio.reserving`) and R (`excess_pattern()`,
`excess_credibility_weights()`, `layer_exposure()`) mirror this, with the
severity passed as an existing distribution object. The derived pattern
goes in through the existing `ldf=`, `ldf_kind=` and `ldf_weight=`
arguments, so `ChainLadder`, `Mack`, the BF family and the bootstraps get no
new parameter. `layer_exposure` also takes a plain list of shares, so an
exposure curve's `layer_share` can be passed from Python or R.

## Crates touched

* `prospicio-reserving`: the `excess` module (new). It uses `Severity`
  from `prospicio-prob`, already a dependency.
* `prospicio-prob`, `prospicio-pricing`, `prospicio-aggregate`: read only.
  Exposure curves live in `prospicio-pricing`, which depends on
  `prospicio-aggregate`; making `prospicio-reserving` depend on it would
  add a heavy edge to the reserving crate and to the WASM scope, so the
  curve route goes through the bindings (open decision 2).
* Bindings, the generated stub, `man/` and `NAMESPACE`: the usual regeneration.
* No Aggregate-lane file is edited; a wish for a change there becomes an
  issue for that lane.

## Validation targets

Published examples to reproduce, in order of confidence. None is checked
yet: PR 1 reads each source in full and records its numbers in `knowledge/`
before a test uses them.

* **Closed forms** (first, they need no source): the Pareto layer LDF at
  `alpha_j` and `alpha_inf`; the lognormal scale-shift illustration above
  (1.275, 1.545, 1.761, 1.982); the identity "a layer of 0 xs infinity
  returns the ground-up factor"; agreement with `Severity::layer` of
  `prospicio-prob`. Cross-checked in mpmath, as the severity parity tests
  are.
* **Clark, "Basics of Reinsurance Pricing"** (already in
  `knowledge/references/clark-reinsurance-pricing.md`): its 400 xs 100 layer
  with an excess development factor per year. Reproduce the layer total
  from a given ground-up triangle and severity within the note's rounding.
* **Pinto and Gogol (1987)**: the Pareto-by-maturity layer factors. The
  paper's tables are the target; reading them is part of PR 1.
* **Bloom and Patrik (2013)**: their ELF-based excess factor from ground-up
  data. Same condition.
* **Self-consistency by simulation**: simulate a ground-up claim process
  with a known severity drift, build the ground-up and layer triangles, and
  check that the derived layer pattern beats the layer's own chain ladder in
  mean squared error of the layer ultimate on sparse layers (the point of
  the feature) and matches it in expectation on dense ones.
* **chainladder-python and R `ChainLadder` have no excess LDFs.** Parity
  with them holds only for the engine side: the derived pattern run through
  the chain ladder equals chainladder-python's `DevelopmentConstant` on the
  same factors (already covered by decision 12).

## Pull requests

| PR | Branch | Holds | Base |
|---|---|---|---|
| 0 | `claude/excess-development-note` | this note, the lane bullet, the literature entry in `knowledge/` | `main` |
| 1 | `claude/excess-pattern` | `ExcessPattern::from_ground_up` with scale shift, `selected`, `layer_exposure`; Rust only; the closed-form and Clark checks; the sources read in full and recorded in `knowledge/` | PR 0 |
| 2 | `claude/excess-credibility` | `credibility_weights`, the shape-drift model, the simulation study | PR 1 |
| 3 | `claude/excess-bindings` | Python and R, the stub, `man/`, a note for the Guides lane | PR 2 |
| 4 | `claude/excess-bayes-prior` | the layer share's uncertainty as a prior, after #410 | PR 3 and #410 |

Each PR ends with its design status, the lane row, the roadmap row, a
changelog fragment and a knowledge fragment.

## Open decisions

1. **Default severity-by-maturity model.** (a) Scale shift in PR 1, shape
   drift in PR 2. (b) Both in PR 1. (c) Scale shift only, shape drift when
   a user asks. *Recommendation: (a).* Scale shift is identified by the
   ground-up triangle alone, so it works for every user; shape drift needs a
   severity at each maturity.
2. **Where the exposure-curve route lives.** (a) Bindings only: the user
   computes `layer_share` and passes shares. (b) `prospicio-reserving`
   depends on `prospicio-pricing`. (c) Move the trait to `prospicio-prob`
   (an Aggregate and Probability lane change). *Recommendation: (a).* It
   keeps the reserving crate light and in the WASM scope; (b) can follow if
   users ask.
3. **Counts.** (a) An optional count column and the count ratio in the
   formula. (b) Ground-up severity only, counts ignored. *Recommendation:
   (a).* Without counts the factor mixes frequency and severity
   development; with them it separates.
4. **Credibility weight in PR 2.** (a) `n / (n + k)` on non-zero link
   observations. (b) Bühlmann–Straub over origins. (c) The user's weights
   only. *Recommendation: (a), with (c) always available;* (b) later if
   asked.
5. **Trend.** (a) A per-origin trend input in PR 1. (b) The caller supplies
   an already trended severity and the note says so. *Recommendation: (b) in
   PR 1, (a) in PR 2,* since leveraged trend is the main reason excess
   development exceeds ground-up development.
6. **Layers that change by year.** (a) Reject a triangle whose layer
   changes. (b) Take the layer per origin. *Recommendation: (b), with a
   vector of layers,* since retentions change by year in practice and Bloom
   and Patrik treat that case.
7. **Guides.** Whether PR 3 touches `docs/guides/reserving-methods.md` or
   leaves it to the Guides lane. *Recommendation: leave it to the Guides
   lane, with the issue they already track.*

## References

Each exists, checked on 2026-10-21 against CAS abstract pages and the
publishers' records. Only abstracts were read, not full texts, so every
formula above is this note's own, not a quotation.

* Pinto, E. and Gogol, D. F. (1987). An Analysis of Excess Loss Development.
  *Proceedings of the Casualty Actuarial Society* 74, from p. 227 (discussion
  by G. M. Levine, p. 256). Pareto fits by maturity and a method for
  development factors by layer.
* Siewert, J. J. (1996). A Model for Reserving Workers' Compensation High
  Deductibles. *CAS Forum*, Summer 1996, 217–244. Consistent development
  factors across deductible levels from a loss distribution model.
* Clark, D. R. (2003). LDF Curve-Fitting and Stochastic Reserving: A Maximum
  Likelihood Approach. *CAS Forum*, Fall 2003. The growth-curve method the
  repo implements as `ClarkLdf`. The excess development factor in Clark's
  *Basics of Reinsurance Pricing* (CAS study note) is the pricing side.
* Sahasrabuddhe, R. (2010). Claims Development by Layer: The Relationship
  between Claims Development Patterns, Trend and Claim Size Models. *CAS
  E-Forum*, Fall 2010. Development, trend and increased limit factors made
  consistent with each other.
* Bloom, J. and Patrik, G. S. (2013). A Methodology for Avoiding the
  Pitfalls of Excess Loss Development. *CAS E-Forum*, Fall 2013, vol. 1.
  Excess factors from ground-up data through excess loss factors. The
  authors come from a file name (`05-Bloom-Patrik`) and a second search
  result, since the abstract page names none: confirm in PR 1.
* Korn, U. Strategies for Modeling Loss Development: Curve Fitting,
  Credibility, and Layer Adjustments. *Variance* 11(1-2). Year and pages to
  be confirmed in PR 1.
* Gillam, W. R. and Couret, J. R. (1997). Retrospective Rating: 1997 Excess
  Loss Factors. *PCAS* 84; Mahler, H. C. (1998), discussion, *PCAS* 85,
  316–344; Boor, J. (2017), Unbiased Development for Individual Claims,
  *CAS E-Forum*, Spring 2017. A distribution of ultimate per claim, hence
  unbiased for any layer. No standalone Mahler paper on excess development
  was found; only this discussion is cited.
* Verrall, R. J. (2004). A Bayesian generalized linear model for the
  Bornhuetter-Ferguson method of claims reserving. *North American
  Actuarial Journal* 8(3), 67–89 (for PR 4 and #410).
