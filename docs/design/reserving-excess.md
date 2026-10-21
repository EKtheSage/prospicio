# Design note: excess-layer loss development (excess LDFs, benchmark blending, a layer BF prior)

Status: **Accepted** (2026-10-21; issue #192, which the user chose on 2026-10-21; on 2026-10-21 the user decided all eleven open decisions as recommended). **PR 1 is built** (branch `claude/excess-pr1`, 2026-10-21): the sparse-age change to `Development`, `ExcessPattern::from_ground_up` with the scale shift, `layer_share` and `layer_exposure`, in Rust; the literature gate is passed (see "Validation targets"). PR 2 to PR 4 are not built. Lane: Reserving
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

**Illustration (own calculation, now a unit test).** A lognormal
severity with `sigma = 1.5` whose mean is 50,000 at the earlier age and
65,000 at ultimate (every claim scaled up by 1.3 on the way to ultimate) and
counts fixed. The ground-up LDF is 1.3. The factor of
the layer 1m xs 0 is 1.275, of 750k xs 250k is 1.545, of 2m xs 1m is 1.761
and of 7m xs 3m is 1.982. A ground-up factor of 1.3 puts a factor near 2 on
a high layer, because development pushes claims over the attachment, and the
sparse layer triangle cannot show that until late.

**Scope and assumptions of the derived factor.**

* **Basis: incurred only** (open decision 9). The formula scales reported
  claim sizes, case reserves included. On a paid triangle large claims pay
  late, so a paid layer develops through payment timing as well as through
  the sizes of reported claims; PR 1 rejects a paid basis with an error (the
  `Basis` argument). The literature does derive paid layer factors: Pinto
  and Gogol and Bear get them from the incurred layer factors and ratios of
  paid-to-reported ratios, which hardly vary by retention, and Sahasrabuddhe
  says his method covers paid or reported losses. A paid pattern is
  therefore a later option, built on such ratios or on an explicit
  payment-timing assumption by size, not an impossible one.
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
  The sources find this drift in data (Pinto and Gogol, Bear, Siewert's
  Weibull shape and Korn all find the shape moving with maturity), so a pure
  scale shift may understate the development of a high layer: PR 2's
  simulation study must generate shape drift, not only scale shifts.
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

Rust, a new module `excess` in `prospicio-reserving` (PR 1 built it; the
names are as built, with `credibility_weights` and `ParetoShape` left to
PR 2).

```rust
pub struct Layer { pub attachment: f64, pub limit: f64 }  // limit may be INFINITY

/// The loss basis of the ground-up triangle: only Incurred is derived.
pub enum Basis { Incurred, Paid }

/// How the severity at each maturity relates to the ultimate severity.
#[non_exhaustive]
pub enum SeverityByMaturity {
    /// X_inf = c_j * X_j; c_j solved from the ground-up triangle.
    /// `ultimate` is S_inf at the origins' price level (incurred basis);
    /// `ground_up_cap` is the per-claim limit of a capped ground-up column.
    ScaleShift { ultimate: Box<dyn Severity>, ground_up_cap: Option<f64> },
    // PR 2: ParetoShape { alphas: Vec<(Lag, f64)>, threshold: f64, ... }
}

pub struct ExcessPattern {
    pub ages: Vec<Lag>,
    pub layer_ldf: Vec<f64>,       // age to next age
    pub layer_cdf: Vec<f64>,       // age to ultimate, with the tail
    pub ground_up_ldf: Vec<f64>,   // the check
    pub ground_up_tail: f64,
    pub shifts: Vec<f64>,          // c_j, for ScaleShift, the last for the tail
    pub layer_tail: f64,           // from the ground-up tail, 1 without one
}

impl ExcessPattern {
    /// Layer pattern from the ground-up incurred column, an optional count
    /// column and the ground-up tail; a paid basis is an error.
    pub fn from_ground_up(triangle: &Triangle, column: &str, basis: Basis,
                          counts: Option<&str>, tail: &Tail, layer: Layer,
                          model: &SeverityByMaturity) -> Result<Self>;
    /// As a `SelectedPattern` of kind `AgeToAge`, with the given weight.
    pub fn selected(&self, weight: PatternWeight) -> SelectedPattern;
    /// The layer's tail as a constant `Tail`.
    pub fn tail(&self) -> Tail;
    // PR 2:
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
/// The expected loss to the layer of one claim at the maturity whose
/// claims are the ultimate ones divided by `shift`: (LEV(c b) - LEV(c a)) / c.
pub fn layer_at_shift(ultimate: &dyn Severity, layer: Layer, shift: f64) -> f64;
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

**The gate is passed** (2026-10-21; the sources are listed in
`knowledge/references/excess-development-literature.md`). Neither R
`ChainLadder` nor chainladder-python has the feature, but one source prints
an example that is exactly the scale shift, and PR 1 reproduces it:

* **Bloom and Patrik (CAS E-Forum Fall 2013), pp. 4 to 8, the end-to-end
  parity target.** Four claims (100, 375, 250 and 500 at 12 months), ground-up
  factors 3.60 and 1.25, a deductible of 350 and 5% inflation. With the
  ultimate severity as four point masses at 450, 1,687.5, 1,125 and 2,250
  and the unlimited layer xs 350, `ExcessPattern::from_ground_up` returns
  the printed excess factors of Years One, Two and Three (17.2, 1.3663 and
  23.5; 14.768, 1.3583 and 20.06; 13.0797, 1.3511 and 17.6719, the paper
  printing two decimals), the layer values 175, 3,010 and 4,112.5 and, for
  the layer limited to 350, 1.3333 and 1,400. The severity is a `Grid`
  of step 112.5, whose `lev` is exact at any limit.
  `validation/tests/reserving_excess.rs`.
* **Identities of printed inputs**: Bloom and Patrik's excess CDF from the
  limited and unlimited factors and an ELF (17.54 from the unrounded
  path; the rounded inputs give 17.59) and Pinto and Gogol's layer identity
  (2.891, 2.697, 2.437 and 2.144 to the three decimals of their inputs).
* **Sahasrabuddhe (E-Forum Fall 2010), Example 1, p. 18**: the rebasing of a
  limited factor by the ratio of limited means at two exponential scales,
  14.352 (computed 14.3519). It checks the rebasing identity through
  `layer_at_shift`, not the solve for the shifts.
* **The capped-data solve has no external example.** With an uncapped
  ground-up column the shift is closed form, `c_j` equal to the ground-up
  factor over the count factor. The numerical solve matters only for a
  capped column, and no source prints one (Korn gives the equations, no
  numbers). PR 1 tests it by round trip, a capped triangle built from a
  known scale path and the scales recovered to 1e-9, and PR 2's simulation
  study carries a capped arm.

Other sources, read in full: Pinto and Gogol (a shape change,
`(n_j/n_i)((q_i-1)/(q_j-1))c^(q_i-q_j)`: their 5.0 and 1.996 are PR 2's
targets), Bear's generalization to bounded layers (3.88), Siewert (counts
times severity with Weibull fits that change shape; its 1.1684 agrees only
within rounding) and Korn's section 6 (this note's model, no numbers).

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
| 1 | `claude/excess-pr1` | the sparse-age change to `Development` (decision 8); `ExcessPattern::from_ground_up` with scale shift, incurred only, one layer, the layer tail; `selected`, `layer_share`, `layer_exposure`; Rust only; the closed-form and engine checks; the sources read in full for a worked example (the gate above) and recorded in `knowledge/` | PR 0 |
| 2 | `claude/excess-credibility` | `credibility_weights` with its cap, the shape-drift model, the simulation study with its acceptance thresholds | PR 1 |
| 3 | `claude/excess-bindings` | Python and R, the stub, `man/`, a note for the Guides lane | PR 2 |
| 4 | `claude/excess-bayes-prior` | the layer share's uncertainty as a prior, after #410 | PR 3 and #410 |

Each PR ends with its design status, the lane row, the roadmap row, a
changelog fragment and a knowledge fragment.

## Open decisions

The user decided every point below on 2026-10-21, taking each recommendation as written.

1. **Default severity-by-maturity model.** (a) Scale shift in PR 1, shape
   drift in PR 2. (b) Both in PR 1. (c) Scale shift only, shape drift when
   a user asks. *Recommendation: (a).* Scale shift is identified by the
   ground-up triangle alone, so it works for every user; shape drift needs a
   severity at each maturity.
   *Decided 2026-10-21 (the user): the recommendation.*

2. **Where the exposure-curve route lives.** (a) Bindings only: the user
   computes `layer_share` and passes shares. (b) `prospicio-reserving`
   depends on `prospicio-pricing`. (c) Move the trait to `prospicio-prob`
   (an Aggregate and Probability lane change). *Recommendation: (a).* It
   keeps the reserving crate light and in the WASM scope; (b) can follow if
   users ask.
   *Decided 2026-10-21 (the user): the recommendation.*

3. **Counts.** (a) An optional count column and the count ratio in the
   formula. (b) Ground-up severity only, counts ignored. *Recommendation:
   (a).* Without counts the factor mixes frequency and severity
   development; with them it separates.
   *Decided 2026-10-21 (the user): the recommendation.*

4. **Credibility weight in PR 2.** (a) `n / (n + k)`, with `n` the layer
   claim count or the origins with a positive earlier value (section 2),
   capped against a single large ratio. (b) Bühlmann–Straub over origins
   with the layer's volume-weighted variance of the link ratios. (c) The
   user's weights only. *Recommendation: (a), with (c) always available;*
   (b) later if asked, since it uses the ratios' variance that (a) ignores.
   *Decided 2026-10-21 (the user): the recommendation.*

5. **Trend.** (a) The caller supplies a severity at one price level, already
   trended, for all origins. (b) A per-origin trend, which needs a
   per-origin pattern: a change to every development method, not only this
   module. *Recommendation: (a) in PR 1 and PR 2; (b) as its own design
   decision later if the user wants it,* since leveraged trend between
   accident years is a calendar-year effect that one factor per age cannot
   carry.
   *Decided 2026-10-21 (the user): the recommendation.*

6. **Layers that change by year.** (a) Reject origins whose layer differs
   from the one given. (b) Group the origins by layer and fit each group as
   its own segment (`fit_segments`, which exists). (c) A per-origin layer,
   which needs the per-origin pattern of decision 5 (b). *Recommendation:
   (a) in PR 1, with (b) documented as the way to handle changing
   retentions;* (c) only with 5 (b).
   *Decided 2026-10-21 (the user): the recommendation.*

7. **Guides.** Whether PR 3 touches `docs/guides/reserving-methods.md` or
   leaves it to the Guides lane. *Recommendation: leave it to the Guides
   lane, with the issue they already track.*
   *Decided 2026-10-21 (the user): the recommendation.*

8. **Sparse ages in `Development`.** (a) Skip estimating an age whose
   selected weight is 1 (its own factor `NaN`, its sigma `None`, Mack's
   sigma there from the interpolation rule already used for missing
   sigmas, and its parameter error 0 as decision 12 gives a selected age).
   (b) Let `credibility_weights` mark an age's own factor as missing, which
   forces weight 1 there. (c) Replace the error by a factor of 1 at an
   all-zero age. *Recommendation: (a), with (b) built on it in PR 2;* (c)
   silently invents a factor.
   *Decided 2026-10-21 (the user): the recommendation.*
   *Built in PR 1, as refined:* the rule is never an error at a fully
   selected age, and the triangle's own factor is still estimated where it
   can be. Skipping the estimate at every weight-1 age would change the
   reported `ldf`, `sigma` and `volume` of every existing fit that selects a
   factor fully (decision 12 of `reserving-v02.md` keeps the estimated
   fields the triangle's), so the new branch fires only where the old code
   returned `Error::Factor` (earlier values summing to zero, a zero value
   with an infinite weight, a geometric mean of a zero ratio) at an age
   with weight 1. There `ldf` and `volume` are NaN, `sigma` is interpolated
   as for any missing sigma and the standard error of the blended factor is
   0. `DevelopmentFit::has_missing_factor` flags it; Merz and Wüthrich's
   closed form and the bootstraps, which need every own factor, reject such
   a fit with an error. `SigmaInterpolation::Mack` cannot fill a sparse
   first age (it needs two earlier sigmas), so Mack's fit errors there;
   `LogLinear`, the default, extrapolates.

9. **Basis.** (a) Incurred only, paid rejected. (b) Paid too, with a
   payment-timing assumption by claim size. *Recommendation: (a);* (b)
   needs a payment-timing assumption by size that this plan does not have
   (the literature derives paid layer factors from incurred ones and ratios
   of paid-to-reported ratios, so paid is a later option, not impossible).
   *Decided 2026-10-21 (the user): the recommendation.*

10. **The tail of the layer pattern.** (a) From the ground-up `Tail`, one
    more shift solved for it. (b) No tail: the derived pattern ends at the
    oldest age and the user supplies a layer tail. *Recommendation: (a),
    with (b) as no tail.*
   *Decided 2026-10-21 (the user): the recommendation.*

11. **The credibility constant `k`** and the simulation study's acceptance
    thresholds. (a) `k` a required argument, no default. (b) A default
    chosen from the simulation study. (c) Replace `k` by Bühlmann–Straub's
    variance ratio (decision 4 (b)). *Recommendation: (a) in PR 2, with the
    study reporting the RMSE across `k` so a default can follow;* the
    thresholds as proposed under "Validation targets".
   *Decided 2026-10-21 (the user): the recommendation.*

Opened in PR 1, for the user (each was chosen as the option closest to the
note, and is built so):

12. **The basis argument.** The note rejected a paid basis (decision 9) but
    gave `from_ground_up` no way to say which basis a column is. PR 1 adds
    `Basis { Incurred, Paid }` as an argument, `Paid` an error. (a) As built.
    (b) A flag on the severity model. (c) Infer from the column name, which
    is fragile. *Recommendation: (a).*

13. **A capped ground-up column.** The solve for `c_j` matters only when the
    ground-up column is limited per claim, and the note's API had nowhere to
    say so. PR 1 adds `ground_up_cap: Option<f64>` to `ScaleShift`: `None`
    gives the closed form, a cap solves `c / LEV(c u) = r / LEV(u)` by
    bisection. (a) As built. (b) A separate model variant. *Recommendation:
    (a).*

14. **The count tail.** With a count column the severity factor is the
    ground-up factor over the count factor. Past the oldest age the count
    factor is taken as 1 (counts are complete there) and the ground-up
    `Tail` is all severity. (a) As built. (b) The same `Tail` fitted to the
    counts as well, so a constant tail of 1.05 on both leaves the severity
    alone. *Recommendation: (a);* revisit if a user's counts are still
    developing at the oldest age.

15. **The ground-up average.** The ground-up factors are the
    volume-weighted ones (`Development::default()`), not a parameter of
    `from_ground_up`. (a) As built. (b) Take a `Development`, for exclusions
    and averages. *Recommendation: (b) in PR 2 or PR 3*, when the bindings
    expose it; it is a field addition with a default.

16. **Shape drift's inputs and the simulation.** Pinto and Gogol and Bear
    both need a base development (the count ratio `n_i/n_j`, or the
    excess-of-threshold development) beside the shapes, which `ParetoShape {
    alphas, threshold }` lacks; PR 2 adds an input for it. The sources find
    the shape moving with maturity, so PR 2's simulation must generate shape
    drift, a capped arm and late claims of other sizes, not only scale
    shifts, and report the derived pattern's RMSE under each.

## References

Checked on 2026-10-21 against CAS abstract pages and the publishers'
records, and in PR 1 against the full texts of Pinto and Gogol with Levine's
and Bear's discussions, Siewert, Sahasrabuddhe, Bloom and Patrik and Korn
(section 6). Bloom and Patrik's authors are confirmed from the paper, and
Korn is Variance 11 (2017), 95-117. Boor and Mahler were skimmed; Gillam
and Couret were not found. What each source prints, and the examples, are
in `knowledge/references/excess-development-literature.md`.

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
  Excess factors from ground-up data through excess loss factors. Authors
  Lynne Bloom and Lela Patrik (PwC), confirmed from the paper; its pages 4
  to 8 are PR 1's parity target.
* Bear, R. A. (1992). Discussion of Pinto and Gogol, *PCAS* 79, 134-148.
  Generalizes Pinto and Gogol's shape-change factor to bounded layers
  (`d c^(q_i - q_j)` for layers with equal limit-to-attachment ratios).
* Korn, U. (2017). Strategies for Modeling Loss Development: Curve Fitting,
  Credibility, and Layer Adjustments. *Variance* 11(1), 95-117. Section 6 is
  this note's scale-shift model, with no numbers.
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
