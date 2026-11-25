# Design note: earned exposure, experience modification and retrospective rating

Status: **In progress** (#981; the note merged on the user's go-ahead of
2026-11-24 to start the work, with decisions (a) to (j) as recommended and
still open to change before the PR each one governs; PR 2 built) ·
v0.3 · Depends on: `ratemaking.md` (`Period`, `year_fraction`),
`aggregate.md` (`Grid`, `fft`, per-occurrence limits on a grid),
`predictive-distribution.md` · Lane: Aggregate

## Goal

`prospicio-pricing` rates a book in aggregate (the parallelogram method,
trend, the indication) and rates reinsurance layers (exposure and
experience rating). Three things a primary or commercial lines actuary uses
every day are missing (#981):

1. **Earned exposure and premium from policies.** Each policy's term gives
   exactly what it earns in any window, with mid-term endorsements and
   cancellations, by calendar period or policy period. The parallelogram
   method approximates this when only totals are known; with the policy
   list there is nothing to approximate.
2. **Experience modification.** The workers compensation split plan
   (NCCI's Experience Rating Plan: primary and excess losses, D-ratios,
   the weighting value and the ballast) and the general liability plan
   (ISO's Commercial General Liability Experience and Schedule Rating
   Plan: an actual experience ratio against an expected one, with a
   credibility).
3. **Retrospective rating.** The insurance charge `phi(r)` and the
   insurance savings `psi(r)` of an aggregate loss distribution (Table M),
   their versions under a per-occurrence limit (Table L and the limited
   Table M), and the plan parameters that balance a retrospective plan to
   the guaranteed-cost premium.

Out of scope for #981, kept as Next items: short-rate cancellation
penalties and audit premium adjustments, the bureau's own tables (decision
(b)), schedule rating, large deductible and dividend plans (which reuse the
charge and savings), loss-sensitive plans with several adjustments over
time, and paid-loss retros.

## What exists

- `ratemaking::Period` (a calendar or a policy window in year fractions),
  `year_fraction` (dates on Werner and Modlin's half-month grid) and
  `RateHistory::rate_level` (the parallelogram method): policies written
  uniformly and earned uniformly over one term. Nothing holds a policy,
  and no date is finer than a half month.
- `prospicio_core::period` has `Month`, `Grain` and `Period` for triangles
  (months, quarters, years). It has no day.
- `prospicio_prob::Severity` gives any distribution, a `Grid` among them,
  `lev(d)`, `stop_loss(d)` and `mean()`, with `lev(d) + stop_loss(d) =
  mean()`. `prospicio_aggregate::fft` and `panjer` give the aggregate on a
  `Grid`, and `grid_reinsurance` applies a per-occurrence limit to the
  severity before compounding. A `PredictiveDistribution` holds simulated
  totals.
- `credibility` has limited fluctuation and Bühlmann credibility;
  `experience` rates a reinsurance layer from large losses. Neither is an
  experience modification of a primary policy.
- `prospicio-aggregate`'s terms include a swing-rated premium for a
  reinsurance layer (`reinsurance/terms.rs`). It applies a formula to each
  simulated year; it does not give the charge and savings curves or
  balance a plan.

## The model

### 1. Earned exposure

A policy has an effective date, an expiration date, and an amount (an
exposure count, or a premium) that is earned evenly over its term. With a
day number `d(date)` and the term `[d_eff, d_exp)`, the part earned in a
window `[a, b)` is

```
earned(a, b) = amount * |[d_eff, d_exp) ∩ [a, b)| / (d_exp - d_eff)
```

The four quantities of Werner and Modlin, chapter 4, follow, each by
calendar window (`Period::Calendar`) or by the window the policy was
written in (`Period::Policy`):

| Quantity | A policy's contribution |
|---|---|
| written | the whole amount, in the window that holds its effective date |
| earned | `earned(a, b)` |
| unearned, at a date `t` | the amount not yet earned at `t`, if in force |
| in force, at a date `t` | the amount, if `d_eff <= t < d_exp` |

A **mid-term endorsement** changes the amount from a date on: it is a
second segment, `[d_end, d_exp)` with the change in the annual amount,
written at `d_end`. A **cancellation** at `d_c` ends every segment at
`d_c`: the amount written is reduced by the unearned part, at `d_c`, which
is how a calendar year's written premium goes negative for a policy
cancelled in the next year.

Two methods:

- **Daily**: exact days, as above.
- **Twenty-fourths**: every policy written in a month is treated as
  written at mid-month, so a twelve-month policy earns 1/24 in its first
  and thirteenth months and 2/24 in the eleven between. It needs only
  written amounts by month, the form in which older data arrives.

### 2. Experience modification

**The split plan** (workers compensation). With expected losses `E`,
expected primary losses `E_p = D * E` (`D` the D-ratio) and expected
excess `E_e = E - E_p`, actual primary `A_p` and actual excess `A_e`, the
weighting value `W` and the ballast `B`:

```
mod = (A_p + W * A_e + (1 - W) * E_e + B) / (E + B)
```

`E` is summed over classes and years as payroll in hundreds times the
expected loss rate. Each claim is split at the split point: its first
`S` is primary and the rest, up to the per-claim limit, excess; the claims
of one accident are limited together; and a medical-only claim enters at a
reduced share (30% under the plan's experience rating adjustment).

**The experience ratio plan** (general liability). With the actual
experience ratio `AER` (basic-limits losses and allocated expense, each
occurrence limited to the maximum single loss, plus the expected
unreported losses, over the subject loss cost), the expected experience
ratio `EER` and the credibility `Z`:

```
mod = 1 + Z * (AER - EER) / EER
```

Both are arithmetic on a table of inputs. What needs care is the order of
the limits and the roll-up over years and classes, so the API takes the
claims and the exposures, not the pre-summed totals, and returns the
exhibit (every intermediate sum) with the modification.

### 3. Retrospective rating

For aggregate losses `A` with mean `E` and the entry ratio `r`:

```
phi(r) = E[max(A / E - r, 0)] = stop_loss(r E) / E        the charge
psi(r) = E[max(r - A / E, 0)] = phi(r) + r - 1            the savings
```

With a per-occurrence limit, `A*` the limited aggregate with mean `E*`
and `k = (E - E*) / E` the share of losses the limit removes:

- **Table L** keeps the entry ratio on the unlimited expected loss and
  puts the limit's cost in the charge:
  `phi*(r) = E[max(A* / E - r, 0)] + k`,
  `psi*(r) = E[max(r - A* / E, 0)]`, so `phi*(r) - psi*(r) = 1 - r` still.
- **The limited Table M** uses the limited aggregate throughout:
  `phi_LM(r) = E[max(A* / E* - r, 0)]`, and the limit's cost is charged
  apart as an excess loss premium.

A retrospective premium is `R = T (b + c L)` held between a minimum `H`
and a maximum `G`, with `T` the tax multiplier, `c` the loss conversion
factor, `b` the basic premium and `L` the losses (limited, when the plan
has a loss limit). The plan balances when its expected premium is the
guaranteed-cost premium `T (e + E)`, `e` being expenses and profit. With
`r_H` and `r_G` the entry ratios at which the minimum and the maximum
bind, that is the pair of equations

```
r_G - r_H           = (G - H) / (c E T)
phi(r_H) - phi(r_G) = (e + E - H / T) / (c E)
```

and then `b = e - (c - 1) E + c E (phi(r_G) - psi(r_H))`. The left side
of the second equation falls as `r_H` rises with the first held, so one
bisection in `r_H` solves the pair.

## API

Proposed; the names are open until each PR.

```rust
// prospicio_pricing::earned
pub struct Date { /* days since 1970-01-01 */ }
impl Date {
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self>;
    pub fn parse(iso: &str) -> Result<Self>;          // "2024-03-15"
    pub fn year_fraction(&self) -> f64;               // exact, not the half-month grid
}
pub struct Policy { /* effective, expiration, amount, endorsements, cancellation */ }
impl Policy {
    pub fn new(effective: Date, expiration: Date, amount: f64) -> Result<Self>;
    pub fn endorsed(self, date: Date, annual_change: f64) -> Result<Self>;
    pub fn cancelled(self, date: Date) -> Result<Self>;
}
pub enum EarningMethod { Daily, TwentyFourths }
pub struct Windows { /* calendar or policy windows: years, quarters, months, or given dates */ }
pub fn earned_exposure(policies: &[Policy], windows: &Windows, method: EarningMethod)
    -> Result<EarnedTable>;   // written, earned, and unearned and in force at each window's end

// prospicio_pricing::retro
pub struct InsuranceCharge { /* the expected loss and the stop-loss; the limit's k if any */ }
impl InsuranceCharge {
    // Built in PR 2:
    pub fn from_severity<S: Severity + Send + Sync + 'static>(aggregate: S) -> Result<Self>; // Table M
    pub fn from_sample(losses: &[f64]) -> Result<Self>;                        // empirical
    pub fn expected(&self) -> f64;
    pub fn charge(&self, r: f64) -> f64;
    pub fn savings(&self, r: f64) -> f64;
    pub fn table(&self, entry_ratios: &[f64]) -> Vec<ChargeRow>;               // r, phi, psi
    // Built in PR 3. The limited Table M is the table of the limited
    // aggregate, from the two constructors above; Table L is made from it:
    pub fn table_l(self, unlimited_expected: f64) -> Result<Self>;
    pub fn loss_elimination(&self) -> f64;                                     // k; 0 unless a Table L
}
// Built in PR 4. Amounts in money; f64::INFINITY for no maximum.
pub struct RetroPlan { /* E, e, c, T, H, G */ }
impl RetroPlan {
    pub fn new(expected_loss: f64, expense: f64, loss_conversion: f64, tax_multiplier: f64,
               minimum: f64, maximum: f64) -> Result<Self>;
    pub fn guaranteed_cost(&self) -> f64;                                      // T (e + E)
    pub fn premium(&self, basic: f64, losses: f64) -> f64;                     // T (b + c L), clamped
    pub fn balance(&self, charge: &InsuranceCharge) -> Result<BalancedPlan>;
}
pub struct BalancedPlan {
    pub basic: f64, pub entry_ratio_minimum: f64, pub entry_ratio_maximum: f64,
    pub net_insurance_charge: f64, pub expected_premium: f64,
}

// prospicio_pricing::emod
pub struct SplitPlan { /* split point, per-claim and per-accident limits, medical-only share */ }
pub struct ClassExposure { /* year, class, payroll, expected loss rate, D-ratio */ }
pub struct Claim { /* year, accident id, amount, medical_only */ }
impl SplitPlan {
    pub fn modification(&self, exposures: &[ClassExposure], claims: &[Claim], w: f64, ballast: f64)
        -> Result<SplitExhibit>;
}
pub struct ExperienceRatioPlan { /* maximum single loss, expected experience ratio, credibility */ }
impl ExperienceRatioPlan {
    pub fn modification(&self, years: &[LiabilityYear]) -> Result<RatioExhibit>;
}
```

`retro`'s types are bound in Python and R in one PR after its three Rust
PRs, on the allow-list until then, as the reinsurer panel was; `earned`
and `emod` get theirs in the PR that adds each type (`pricing` module
files), with dates as ISO strings in R and Excel and as
`datetime.date` or ISO strings in Python.

## Decisions

Proposed for the user's acceptance; each has a recommendation.

- (a) **A day-level `Date` in `prospicio-pricing`, not in
  `prospicio-core`.** Earned exposure needs days; `prospicio-core` has
  months. A small type in `pricing::earned` (a day number from a civil
  date, no time zones, no dependency) keeps this out of the crate every
  lane shares. If reserving later wants days too, it moves to core in a PR
  of its own. *Recommended.* The alternative is a `chrono` or `time`
  dependency, which the WASM and R builds would then carry for one
  subtraction.
- (b) **The bureaus' tables are inputs, not shipped.** NCCI's expected
  loss rates, D-ratios, weighting and ballast tables and split points, and
  ISO's credibility, expected experience ratios and maximum single losses,
  are licensed and change every filing. The functions take them as
  arguments, and the docs say where an actuary reads each from.
  *Recommended.* The mechanics are the part a library can own.
- (c) **Expiration is exclusive, and days are exact.** A policy effective
  1 January and expiring the next 1 January has 365 or 366 days and earns
  nothing on its expiration date (the 12:01 a.m. convention). *Recommended.*
- (d) **A cancellation earns pro rata to its date.** The return premium
  under a short-rate table is a premium accounting matter, not an
  exposure one; it stays a Next item. *Recommended.*
- (e) **Endorsements are changes in the annual amount from a date.** A
  change of exposure and a change of premium are both this. An
  endorsement that changes the expiration date is a cancellation followed
  by a new policy. *Recommended.*
- (f) **`earned_exposure` returns a table, by window, of written, earned,
  unearned and in force**, and can keep a label per policy (state, class)
  so one call gives the table by label. *Recommended.* It does not return
  one row per policy and window, which for a large book is the whole
  input times the number of windows; a per-policy function gives that
  when wanted.
- (g) **The charge comes from any `Severity`**: `phi(r) = stop_loss(r E) /
  E` on the aggregate `Grid` from `fft` or `panjer`, on a fitted
  distribution, or on a sample. *Recommended.* A `Grid`'s `stop_loss` is
  exact on the grid, so the charge inherits the grid audit's error and
  adds none.
- (h) **Both limit conventions, named for what they are**: `table_l`
  (entry ratio on the unlimited mean, `k` in the charge) and
  `limited_table_m` (limited throughout). *Recommended.* They answer
  different filings, and mixing them is the classic error. As built (PR
  3): the limited Table M needs no function of its own, being the table
  of the limited aggregate (`from_severity` or `from_sample` on it), and
  `table_l(unlimited_expected)` is a method that turns that table into a
  Table L, so the limited aggregate is always the starting point and the
  module doc sets the two side by side. The curves are one curve on two
  scales: `phi*(r) = (1 - k) phi_LM(r E / E*) + k`, which the validation
  test checks.
- (i) **`RetroPlan::balance` solves for the basic premium given the
  minimum and maximum**, the common case. Solving for the maximum given
  the basic, or for the loss conversion factor, are small variations on
  the same bisection and can follow as asked. *Recommended.* As built
  (PR 4): a plan with a loss limit balances against a Table L on the
  unlimited expected loss, with `L` the limited losses, and its basic
  premium then carries the cost of the limit; the limited Table M, being
  on another expected loss, is refused with a message that says so. A
  minimum at or above the guaranteed-cost premium, or a maximum at or
  below it, has no balancing basic premium and is an error. An entry ratio
  of the minimum below 0 means the minimum never binds.
- (j) **Order: retrospective rating, then earned exposure, then the
  modifications.** The charge builds on what the lane already has (the
  aggregate on a grid) and has closed forms to validate against; earned
  exposure is self-contained; the modifications depend most on reference
  exhibits (Validation, below). *Recommended.*

## Validation

- **Charge and savings.** Closed forms: for exponential aggregate losses
  `phi(r) = e^{-r}`; for a gamma, the incomplete gamma function. The
  identity `phi(r) - psi(r) = 1 - r` at every entry ratio, and `phi*(r) -
  psi*(r) = 1 - r` for Table L; `phi(0) = 1`; both curves monotone and
  convex. A compound Poisson on a grid against the same from a simulated
  sample, within the sample's error.
- **The balanced plan.** The expected retrospective premium over the
  aggregate distribution, computed directly, equals `T (e + E)` to
  round-off; a plan with no minimum and an unreachable maximum has
  `b = e - (c - 1) E`.
- **Earned exposure.** Werner and Modlin, chapter 4: the written, earned,
  unearned and in-force exposures of its six policies, by calendar year
  and by policy year, and its premium examples with a mid-term change in
  chapter 5. Twenty-fourths against daily on policies written at
  mid-month, where they agree. Totals: the earned amounts over all
  windows sum to the written amount net of cancellations.
- **Modifications.** Worked exhibits from the CAS study note on
  individual risk rating (Fisher, McTaggart, Petker and Pettingell) and
  from the bureaus' public explanatory material. These texts are the
  reference; the exhibits' numbers go into `validation/reference/` with a
  generator that is a transcription, as the Werner and Modlin appendices
  did. If the cloud session cannot reach a text, the user supplies the
  exhibit, as with the AP pricing demo's inputs.

Each reference file gets its `MANIFEST.toml` entry and a dataset or
reference concept in `knowledge/`.

## PR plan

Stacked, each on the one before:

1. This note and the lane's Next bullets. Docs only.
2. `retro::InsuranceCharge` from a `Severity` and from a sample, with
   `charge`, `savings` and `table`; the closed-form tests; allow-list
   entries (reason: bound in PR 4b). Done.
3. `table_l` and `loss_elimination`; the identities, and an FFT aggregate
   with a per-occurrence limit against the definition. Done.
4. `RetroPlan::balance` and `premium`; the balance tests (the mean
   premium over a sample, an exponential's closed integral, a loss limit
   against Table L). Done.
4b. Python and R bindings for `retro` (`[retro-bindings]`), their tests in
   new files, the allow-list entries removed. Done: `InsuranceCharge`
   (`from_grid`, `from_sample`, `table_l`) and `RetroPlan` in
   `prospicio.pricing`; `insurance_charge()` and `retro_balance()` in R,
   which are functions of the aggregate (a `grid_distribution` or a vector
   of totals), not objects. R's `retro_premium()` already existed, as the
   reinsurance module's `LossSensitivePremium`: `RetroPlan::premium` is
   that premium with the tax multiplier folded into the basic premium and
   the loss conversion factor, and a test holds the two together.
5. `earned::Date`, `Policy` and `earned_exposure` with the daily method;
   bindings; the Werner and Modlin chapter 4 exhibit.
6. Endorsements, cancellations and the twenty-fourths method; bindings.
7. `emod::SplitPlan`; bindings; a reference exhibit.
8. `emod::ExperienceRatioPlan`; bindings; a reference exhibit; the Done
   bullet `#981`.

Every PR appends a Done bullet repeating its own Next id to
`docs/lanes/done/aggregate.md`, adds a knowledge fragment (or says it
taught nothing new), and a changelog fragment under "Pricing".

## Next

The Next items with ids in `docs/lanes/aggregate.md`: `[retro-charge]`,
`[retro-limited]`, `[retro-balance]`, `[retro-bindings]`, `[earned-daily]`,
`[earned-endorsements]`, `[emod-split]`, `[emod-ratio]`, and after #981:
`[earned-short-rate]`, `[retro-deductible-dividend]`.

## Sources

- Werner, G. and Modlin, C., *Basic Ratemaking*, 5th ed., CAS, 2016:
  chapter 4 (exposures: written, earned, unearned, in force; calendar and
  policy year aggregation) and chapter 5 (premium, mid-term changes).
- Fisher, G., McTaggart, L., Petker, J. and Pettingell, R., *Individual
  Risk Rating*, CAS study note: experience rating plans and their
  credibility, the split plan, retrospective rating, Table M, Table L and
  the limited Table M.
- Brosius, E., *Table M Construction*, CAS study note: the charge and
  savings from a sample of risks, and their identities.
- Skurnick, D., "The California Table L", *PCAS* LXI, 1974: the charge
  with a per-occurrence limit.
- NCCI, *Experience Rating Plan Manual* and its public explanatory
  material: the split plan's formula and the treatment of medical-only
  claims and of several claims in one accident. The manual's tables are
  licensed (decision (b)).
- ISO, *Commercial General Liability Experience and Schedule Rating
  Plan*: the experience ratio plan. Its tables are licensed (decision
  (b)).
