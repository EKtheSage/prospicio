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
  estimated`. The blend of item 2 exists for a dense triangle. On a sparse
  layer triangle it does not work as built: see "Engine changes" below.
* `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and `CapeCod` take a
  scalar `apriori` loss ratio and an exposure column of the same triangle,
  cumulated like the losses, whose latest value per origin is the exposure.
  A layer-share prior needs no change to these methods, only a layer
  exposure column in the triangle (premium times layer share); PR 1 adds
  the helper that writes that column.
* `Severity::lev` and `Severity::layer` (`prospicio-prob`), `Pareto` and
  the other families; `ExposureCurve::layer_share` (`prospicio-pricing`).
* `Triangle.link_ratios` for diagnostics; `PreparedLosses` and
  `DevelopmentPattern` in `prospicio-pricing` for per-claim IBNER on the
  experience-rating side (Aggregate lane, not touched).

## Engine changes

The derived pattern cannot go through `SelectedPattern` unchanged on the
data this feature is for, for two reasons.

* **Sparse ages.** `Development::fit_segment` estimates the triangle's own
  factor at every age and applies the pattern's weights only afterwards.
  At an age where every earlier layer value is zero it returns
  `Error::Factor` with "earlier values sum to zero" (volume and simple
  weighting) or "a zero value gets an infinite weight" (`alpha` of 0),
  before a weight of 1 on the derived pattern can be used; and Mack's
  sigma at such an age has one informative origin or none. A high layer's
  early ages are exactly this case. PR 1 therefore changes the engine in
  the Reserving lane's own crate (open decision 8).
* **One factor per age.** A `SelectedPattern` is keyed by age only, and the
  chain ladder, Mack and both bootstraps take one factor per age. A layer
  that changes by origin, or a severity trended by origin, gives a different
  layer factor per origin and age, which no method takes today. Trend
  between accident years is also a calendar-year effect that an age-only
  pattern cannot represent. The first PRs are therefore restricted to one
  layer for all origins and a severity at one price level (open decisions 5
  and 6); per-origin patterns are a larger engine change, listed there, or
  are handled by fitting each group of origins with the same layer as its
  own segment.

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

**Scope and assumptions of the derived factor.**

* **Basis: incurred only** (open decision 9). The formula scales reported
  claim sizes, case reserves included. On a paid triangle large claims pay
  late, so a paid layer develops through payment timing, not through the
  sizes of reported claims; PR 1 rejects a paid basis with an error, and a
  paid layer pattern needs an explicit payment-timing assumption by size,
  which is not in this plan.
* **The supplied severity is `S_inf`**, the ultimate severity at the price
  level of the layer's origins. `S_j` is derived from it by the model
  below (for the scale shift, `X_j = X_inf / c_j`). A severity fitted to
  reported claims at one maturity is a different input; the API names the
  argument `ultimate` so the basis is explicit.
* **The tail** (open decision 10). The oldest age's factor to ultimate
  needs a ground-up factor past the triangle. PR 1 takes the ground-up tail
  from the same `Tail` the chain ladder uses (`reserving-v02.md`), solves
  one more `c` for it and returns the layer's tail factor beside the layer
  pattern; a cumulative layer pattern is then rescaled to that tail as
  decision 12's `pattern_rescale` does.
* **Late-reported claims** (in `n_inf / n_j`) are assumed to share the size
  distribution of the claims reported by `j`, after the shift. Where late
  claims are smaller (or larger) the layer factor is over (or under)
  stated; the note records this as an assumption and the simulation study
  varies it.

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
* **Inflation leverage** (as the Sahasrabuddhe and Bloom and Patrik
  abstracts describe it): trend acts on the whole size distribution, so a
  layer's trend, and with it the development between calendar years,
  exceeds the ground-up one. In the first PRs the caller trends `S` to the
  origins' price level before the layer is cut. A trend that differs by
  origin needs a per-origin pattern (see "Engine changes" and open
  decision 5).
* **Direct claim development** (Boor's abstract describes development of
  individual claims; Gillam and Couret, with Mahler's discussion, give
  excess loss factors for retrospective rating and are cited for context):
  development to ultimate claim by claim. Out of scope here; the `prospicio-pricing` IBNER
  (`DevelopmentPattern`) is the claim-level counterpart.

The warning in the Bloom and Patrik abstract shapes the design: excess
triangles carry too little data and mis-state inflation's leverage, but the ground-up route
is only as good as the severity assumption, and a retention that changes by
year distorts the layer data. The method therefore reports its inputs (the
`c_j` and the implied ground-up check) and never hides the severity
assumption.

### 2. Blending with the layer's own triangle

Use decision 12's blend: `f_j = w_j * f_derived_j + (1 - w_j) * f_own_j`,
where `w_j` is the weight on the derived or benchmark pattern. Sources of
`w_j`:

* **Fixed by the user** by age (`PatternWeight::ByAge`), which exists.
* **From the data**: `w_j = 1 - n_j / (n_j + k)` (open decisions 4 and
  11). `n_j` must measure information, not luck: counting non-zero link
  ratios conditions on the layer having been hit, and a near-empty
  column's ratios are huge. So `n_j` counts the claims in the layer at age
  `j` when a layer count column exists, and otherwise the origins observed
  at both ages whose earlier layer value is positive. `k` is the real
  credibility choice and has no default yet (open decision 11). Because
  `n / (n + k)` ignores the variance of the ratios, the weight is also
  capped so that one origin's link ratio cannot move the blended factor by
  more than a stated share; PR 2 tests that a single huge ratio does not
  dominate.
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
`BornhuetterFerguson` and `CapeCod` take an `apriori` scalar and the name
of an exposure column of the same triangle, read at its latest value per
origin and cumulated like the losses, the helper returns a **copy of the
triangle with a layer exposure column** added: every cell of the exposure
column times the origin's share, so the latest cumulative value is
`E_i * s` (or `E_i * s_i` with a share per origin; a per-origin share is
fine for the prior, which is per origin, while the pattern stays one per
age). The caller passes that column's name. Cape Cod then estimates the layer loss ratio from the
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
    /// `ultimate` is S_inf at the origins' price level (incurred basis).
    ScaleShift { ultimate: Box<dyn Severity> },
    /// A Pareto whose shape changes with maturity (Pinto and Gogol).
    ParetoShape { alphas: Vec<(Lag, f64)>, threshold: f64 },
}

pub struct ExcessPattern {
    pub ages: Vec<Lag>,
    pub layer_ldf: Vec<f64>,       // age to next age
    pub layer_cdf: Vec<f64>,       // age to ultimate
    pub ground_up_ldf: Vec<f64>,   // the check
    pub shifts: Vec<f64>,          // c_j, for ScaleShift
    pub layer_tail: f64,           // from the ground-up tail, 1 without one
}

impl ExcessPattern {
    /// Layer pattern from the ground-up incurred column, an optional count
    /// column and the ground-up tail; a paid basis is an error.
    pub fn from_ground_up(triangle: &Triangle, column: &str, counts: Option<&str>,
                          tail: &Tail, layer: Layer, model: &SeverityByMaturity)
                          -> Result<Self>;
    /// As a `SelectedPattern` of kind `AgeToAge`, with the given weight.
    pub fn selected(&self, weight: PatternWeight) -> SelectedPattern;
    /// The weight on the benchmark: 1 - n/(n + k), n as in section 2.
    pub fn credibility_weights(layer_triangle: &Triangle, column: &str,
                               counts: Option<&str>, k: f64) -> Result<PatternWeight>;
}

/// A copy of the triangle with column `name`: the exposure column times
/// each origin's layer share (one share, or one per origin).
pub fn layer_exposure(triangle: &Triangle, exposure: &str, shares: &[f64],
                      name: &str) -> Result<Triangle>;
/// The layer share of a severity, (LEV(b) - LEV(a)) / mean.
pub fn layer_share(severity: &dyn Severity, layer: Layer) -> Result<f64>;
```

Python (`prospicio.reserving`) and R (`excess_pattern()`,
`excess_credibility_weights()`, `layer_exposure()`) mirror this, with the
severity passed as an existing distribution object. The derived pattern
goes in through the existing `ldf=`, `ldf_kind=` and `ldf_weight=`
arguments, so `ChainLadder`, `Mack`, the BF family and the bootstraps get no
new parameter. `layer_exposure` takes shares, so a severity's
`layer_share` or an exposure curve's `layer_share` can be passed from
Python or R.

## Crates touched

* `prospicio-reserving`: the `excess` module (new), and the sparse-age
  change to `Development` (open decision 8). It uses `Severity` from
  `prospicio-prob`, already a dependency.
* `prospicio-prob`, `prospicio-pricing`, `prospicio-aggregate`: read only.
  Exposure curves live in `prospicio-pricing`, which depends on
  `prospicio-aggregate`; making `prospicio-reserving` depend on it would
  add a heavy edge to the reserving crate and to the WASM scope, so the
  curve route goes through the bindings (open decision 2).
* Bindings, the generated stub, `man/` and `NAMESPACE`: the usual regeneration.
* No Aggregate-lane file is edited; a wish for a change there becomes an
  issue for that lane.

## Validation targets

No external source is confirmed to print a worked example of a layer
pattern derived from a ground-up triangle and a severity, and neither R
`ChainLadder` nor chainladder-python has the feature. **Reading Pinto and
Gogol, Siewert and Bloom and Patrik in full for such an example is a gate
for PR 1**: if one is found its numbers go to `knowledge/` and become a
parity test; if none is, the PR says plainly that the derived factor has no
external parity target and rests on the closed forms and the simulation
study below.

* **Closed forms** (first, they need no source): the Pareto layer LDF at
  `alpha_j` and `alpha_inf`; the lognormal scale-shift illustration above
  (1.275, 1.545, 1.761, 1.982); the identity "a layer of 0 xs infinity
  returns the ground-up factor"; agreement with `Severity::layer` of
  `prospicio-prob`. Cross-checked in mpmath, as the severity parity tests
  are. These check the implementation against the note's formula, not the
  formula itself.
* **Clark, "Basics of Reinsurance Pricing"** (already in
  `knowledge/references/clark-reinsurance-pricing.md`) applies a *given*
  excess development factor to each year's 400 xs 100 layer total; it
  derives none. It can check only that a supplied layer pattern is applied
  as Clark applies it, not the derivation.
* **Pinto and Gogol (1987)**, **Siewert (1996)**, **Bloom and Patrik
  (2013)**: candidates for the gate above; only abstracts have been read.
* **Self-consistency by simulation** (the main test of the method): with
  fixed seeds, simulate a ground-up incurred claim process with a known
  severity drift and known late-claim sizes, build the ground-up and layer
  triangles, and compare the layer ultimate's root mean squared error over
  at least 500 simulated triangles. Proposed acceptance (open decision
  11): on a sparse layer (most cells at early ages zero) the derived
  pattern's RMSE is below 0.8 times the layer's own chain ladder's; on a
  dense layer the derived pattern's mean error is within two Monte Carlo
  standard errors of zero. A run with late claims smaller than early ones
  shows the bias the assumption above predicts.
* **Engine and input checks**: a layer triangle whose early ages are all
  zero fits with a weight of 1 on the derived pattern there, and errors as
  today with a weight below 1; Mack's sigma at those ages follows open
  decision 8; a paid-basis input is rejected; a single huge layer link
  ratio does not dominate the credibility blend.
* **chainladder-python and R `ChainLadder` have no excess LDFs.** Parity
  with them holds only for the engine side: the derived pattern run through
  the chain ladder equals chainladder-python's `DevelopmentConstant` on the
  same factors (already covered by decision 12).

## Pull requests

| PR | Branch | Holds | Base |
|---|---|---|---|
| 0 | `claude/excess-development-note` | this note, the lane bullet, the literature entry in `knowledge/` | `main` |
| 1 | `claude/excess-pattern` | the sparse-age change to `Development` (decision 8); `ExcessPattern::from_ground_up` with scale shift, incurred only, one layer, the layer tail; `selected`, `layer_share`, `layer_exposure`; Rust only; the closed-form and engine checks; the sources read in full for a worked example (the gate above) and recorded in `knowledge/` | PR 0 |
| 2 | `claude/excess-credibility` | `credibility_weights` with its cap, the shape-drift model, the simulation study with its acceptance thresholds | PR 1 |
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
4. **Credibility weight in PR 2.** (a) `n / (n + k)`, with `n` the layer
   claim count or the origins with a positive earlier value (section 2),
   capped against a single large ratio. (b) Bühlmann–Straub over origins
   with the layer's volume-weighted variance of the link ratios. (c) The
   user's weights only. *Recommendation: (a), with (c) always available;*
   (b) later if asked, since it uses the ratios' variance that (a) ignores.
5. **Trend.** (a) The caller supplies a severity at one price level, already
   trended, for all origins. (b) A per-origin trend, which needs a
   per-origin pattern: a change to every development method, not only this
   module. *Recommendation: (a) in PR 1 and PR 2; (b) as its own design
   decision later if the user wants it,* since leveraged trend between
   accident years is a calendar-year effect that one factor per age cannot
   carry.
6. **Layers that change by year.** (a) Reject origins whose layer differs
   from the one given. (b) Group the origins by layer and fit each group as
   its own segment (`fit_segments`, which exists). (c) A per-origin layer,
   which needs the per-origin pattern of decision 5 (b). *Recommendation:
   (a) in PR 1, with (b) documented as the way to handle changing
   retentions;* (c) only with 5 (b).
7. **Guides.** Whether PR 3 touches `docs/guides/reserving-methods.md` or
   leaves it to the Guides lane. *Recommendation: leave it to the Guides
   lane, with the issue they already track.*
8. **Sparse ages in `Development`.** (a) Skip estimating an age whose
   selected weight is 1 (its own factor `NaN`, its sigma `None`, Mack's
   sigma there from the interpolation rule already used for missing
   sigmas, and its parameter error 0 as decision 12 gives a selected age).
   (b) Let `credibility_weights` mark an age's own factor as missing, which
   forces weight 1 there. (c) Replace the error by a factor of 1 at an
   all-zero age. *Recommendation: (a), with (b) built on it in PR 2;* (c)
   silently invents a factor.
9. **Basis.** (a) Incurred only, paid rejected. (b) Paid too, with a
   payment-timing assumption by claim size. *Recommendation: (a);* (b)
   needs claim-level data this plan does not use.
10. **The tail of the layer pattern.** (a) From the ground-up `Tail`, one
    more shift solved for it. (b) No tail: the derived pattern ends at the
    oldest age and the user supplies a layer tail. *Recommendation: (a),
    with (b) as no tail.*
11. **The credibility constant `k`** and the simulation study's acceptance
    thresholds. (a) `k` a required argument, no default. (b) A default
    chosen from the simulation study. (c) Replace `k` by Bühlmann–Straub's
    variance ratio (decision 4 (b)). *Recommendation: (a) in PR 2, with the
    study reporting the RMSE across `k` so a default can follow;* the
    thresholds as proposed under "Validation targets".

## References

Checked on 2026-10-21 against CAS abstract pages and the publishers'
records. Confirmed from those pages: Pinto and Gogol, Siewert, Clark,
Sahasrabuddhe, Gillam and Couret with Mahler's discussion, Boor, Verrall,
and the Fall 2013 paper's title. Not confirmed: the Fall 2013 paper's
authors, and Korn's year and pages; PR 1 confirms or drops them. Only
abstracts were read, not full texts, so every formula above is this note's
own, and what a paper is said to contain is what its abstract says.

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
  *CAS E-Forum*, Spring 2017. Cited for claim-level development (Boor's
  abstract) and excess loss factors (Gillam and Couret); whether any of
  them gives a distribution of ultimate per claim that holds for any layer
  is unread. No standalone Mahler paper on excess development was found;
  only this discussion is cited.
* Verrall, R. J. (2004). A Bayesian generalized linear model for the
  Bornhuetter-Ferguson method of claims reserving. *North American
  Actuarial Journal* 8(3), 67–89 (for PR 4 and #410).
