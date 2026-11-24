# Design note: reinsurer default on a tower

Status: **Accepted** (#980; decisions (a) to (l) accepted by the user on
2026-11-24 as recommended; PRs 2 and 3 built) · v0.3 ·
Depends on: `aggregate.md` (`Tower`, `EventSet`, the tower's simulated years
in parallel, #970), `predictive-distribution.md`, `rng.md` · Lane: Aggregate

## Goal

A tower's ceded recoveries are only worth what the reinsurers pay. Capital
models (Solvency II, rating agencies, internal models) take less than full
credit for a recovery because the reinsurer can default, and they want the
default to be likeliest in the years the recoveries are largest: a reinsurer
that pays a large share of an industry-wide catastrophe is the one most
likely to fail on it (wrong-way risk).

This note adds three things to `prospicio-aggregate`:

1. A **panel**: the reinsurers on the tower, each with a one-year
   probability of default, a recovery rate and collateral, and each one's
   signed line on each layer.
2. A **default overlay** on a tower's simulated years: each year draws which
   reinsurers default, with the probability of default rising with the
   year's loss when asked, and gives the recoveries net of the credit loss,
   the credit loss by layer and by reinsurer, and the net loss after
   default, as a `PredictiveDistribution` that feeds `capital(...)` and
   the risk measures directly.
3. **Solvency II helpers**: the standard formula's counterparty default
   module for type 1 exposures (Delegated Regulation (EU) 2015/35, Articles
   192, 199, 200 and 201), as pure functions.

Out of scope for #980, kept as Next items: default during a long payout
(cumulative probabilities of default over a recovery pattern), the set-off
of premiums owed to a reinsurer against its recoveries, a structural model
in which a reinsurer fails when its own share of an industry loss exceeds
its capital, copulas with tail dependence for the systematic factor, a
"reinsurance bad debt" row in the P&L ledger (`pnl.md`), and the panel in
the tower's JSON (and so in the JavaScript and Excel front ends).

## What exists

- `Tower::apply(&EventSet)`, `apply_aggregate`, `apply_annual` and
  `apply_term` return a `PredictiveDistribution` with dims
  `["kind", "layer"]` (plus label dims, or `"year"` for a term): the gross
  loss, each layer's ceded loss, the net loss (`net = gross - sum of ceded`
  in every year) and the annual terms. Simulation `i` of the result is
  year (or path) `i` of the input.
- The tower draws no random numbers; all of them come from the events
  (`StreamRng::new(seed, i)` per year, `rng.md`). PR #1053 (#970) fills the
  years in parallel, each row from its own inputs only, so the result has the
  same bits on any number of threads; its test is
  `validation/tests/aggregate_tower_threads.rs`.
- `Layer::share` is a single participation figure (the placed share); there
  is no reinsurer, panel or rating anywhere in the crates. Nothing in the
  repository models counterparty default or the Solvency II counterparty
  module.
- `prospicio_math::special` has `norm_cdf`, `norm_quantile` and
  `bivariate_norm_cdf`, which are all a one-factor Gaussian model needs.
- `prospicio_prob::capital` allocates capital over a
  `PredictiveDistribution`'s components (Euler, covariance, Shapley and
  others).

## The model

Write `X_{l,i}` for layer `l`'s ceded loss in year `i` (the tower's
`("ceded", l)` component, after `Layer::share`), and `s_{r,l}` for
reinsurer `r`'s signed line on layer `l`, as a fraction of the layer's
ceded loss. Reinsurer `r`'s exposure in year `i` is

    E_{r,i} = sum over l of s_{r,l} X_{l,i},

its exposure at default after collateral `C_r` is `(E_{r,i} - C_r)^+`, and
if it defaults in year `i` the cedant loses

    L_{r,i} = (1 - R_r) (E_{r,i} - C_r)^+,

where `R_r` is the recovery rate (the share of the uncollateralised amount
recovered from the estate). The recoveries after default are
`sum over r of (E_{r,i} - L_{r,i} D_{r,i})`, where `D_{r,i}` is 1 if
reinsurer `r` defaults in year `i`, and the net loss after default is the
net loss plus `sum over r of L_{r,i} D_{r,i}`.

### Who defaults: one systematic factor

Reinsurer `r` defaults in year `i` when

    sqrt(rho_r) Z_i + sqrt(1 - rho_r) eps_{r,i} < Phi^{-1}(p_r),

with `p_r` its one-year probability of default, `eps_{r,i}` independent
standard normals, and `Z_i` one systematic factor per year shared by every
reinsurer (Vasicek's one-factor model). Given `Z_i = z` the reinsurers
default independently, each with

    p_r(z) = Phi((Phi^{-1}(p_r) - sqrt(rho_r) z) / sqrt(1 - rho_r)),

and since `Z_i` is standard normal, `E[p_r(Z)] = p_r`: the correlation moves
defaults into the bad years without changing how often a reinsurer
defaults. Two reinsurers default together with probability
`Phi_2(Phi^{-1}(p_r), Phi^{-1}(p_s); sqrt(rho_r rho_s))`.

Wrong-way risk ties `Z_i` to the year's loss: `Z_i = -Phi^{-1}(U_i)`, where
`U_i` is the rank of a driver in year `i` among all years, so the worst year
has the most negative factor and the highest probability of default. The
driver is the year's gross loss unless the caller gives one per year (an
industry loss, say, aligned with the simulation index). `rho_r = 0` for
every reinsurer gives independent default, and the draws are then the same
uniforms, so an independent run and a correlated run differ only where the
correlation moves a default.

The rank is made exactly uniform even with ties (many years with no loss
tie at zero): with `b_i` years strictly below year `i`'s driver and `t_i`
years tied with it (itself included), `U_i = (b_i + v_i t_i) / n`, with
`v_i` uniform from year `i`'s own stream. Without ties this is the usual
randomised rank, and in either case `U_i` is uniform on (0, 1) over the
years (each tie group fills its own interval `(b_i / n, (b_i + t_i) / n)`),
so `Z_i` is standard normal and the mean probability of default stays
`p_r`. `v_i` comes from `StreamRng::next_open01`, so `U_i` is strictly
inside (0, 1) and `Z_i` is finite; `p_r(z)` is taken as exactly `p_r` when
`rho_r = 0`.

### Over a term

In `apply_term`'s result a row is a path of `T` years. A default is final:
a reinsurer that defaults in year `t` of a path stays in default for the
rest of the path, and each later year's exposure is lost at `1 - R_r` too.
Collateral is held once per path and used up: in each year in default the
collateral still held is set against that year's exposure, and whatever it
covered is gone for the later years. Each year's factor comes from that
year's driver ranked among all path-years (a driver given for a term is
path-major, `n_paths * T` values, year `t` of path `j` at `j * T + t`), and
the draws come from one stream per path, year after year: `v` and the
reinsurers' uniforms for year 1, then for year 2, and so on. A reinsurer
already in default still draws, so the draws do not depend on the path's
history.

## API

```rust
/// A reinsurer on a panel.
#[non_exhaustive]
pub struct Reinsurer {
    pub name: String,
    /// One-year probability of default, in [0, 1].
    pub probability: f64,
    /// Share of the uncollateralised exposure recovered on default, in [0, 1].
    pub recovery: f64,
    /// Collateral held against this reinsurer, across all its layers, >= 0.
    pub collateral: f64,
    /// Loading on the systematic factor, in [0, 1); 0 is independent default.
    pub correlation: f64,
}

impl Reinsurer {
    pub fn new(name: &str, probability: f64, recovery: f64) -> Result<Self>;
    pub fn collateral(self, amount: f64) -> Result<Self>;
    pub fn correlation(self, rho: f64) -> Result<Self>;
}

/// The reinsurers on a tower and their signed lines.
pub struct Panel { /* reinsurers, lines by (reinsurer, layer), driver */ }

impl Panel {
    pub fn new(reinsurers: Vec<Reinsurer>) -> Result<Self>;
    /// Reinsurer `reinsurer` writes `line` of layer `layer`'s ceded loss.
    pub fn line(self, reinsurer: &str, layer: &str, line: f64) -> Result<Self>;
    /// Rank the years by this driver instead of the gross loss.
    pub fn driver(self, by_year: Vec<f64>) -> Result<Self>;
    /// A one-year result (`apply`, `apply_aggregate`, `apply_annual`) after default.
    pub fn apply(&self, tower: &PredictiveDistribution, seed: u64)
        -> Result<PredictiveDistribution>;
    /// A term's paths (`apply_term`, `apply_aggregate_term`) after default.
    pub fn apply_term(&self, tower: &PredictiveDistribution, n_years: usize, seed: u64)
        -> Result<PredictiveDistribution>;
    /// Credit loss, exposure and default by reinsurer, from the same draws.
    pub fn by_reinsurer(&self, tower: &PredictiveDistribution, seed: u64)
        -> Result<PredictiveDistribution>;
}

/// Solvency II, Delegated Regulation (EU) 2015/35.
pub mod solvency2 {
    pub fn probability_by_step(step: u8) -> Result<f64>;          // Art 199(2)
    pub fn probability_by_ratio(scr_ratio: f64) -> Result<f64>;   // Art 199(3)
    /// `lgd_share` is 0.5, or 0.9 for a counterparty with 60 % or more of
    /// its assets collateralised; `factor` is F, or F' in the 90 % case.
    pub fn reinsurance_lgd(recoverables: f64, risk_mitigation: f64,
        collateral: f64, lgd_share: f64, factor: f64) -> Result<f64>; // Art 192(2)
    pub fn type1(exposures: &[(f64, f64)]) -> Result<Type1>;       // Arts 200, 201
}
```

`Panel::apply` takes the tower's one-year result and returns it, its
components copied bit for bit, with these components added (`apply_term`
does the same for a term's paths, with the year as the last key of each
new component, as `apply_term`'s own components have it). A term and a
one-year result labelled by `year` have the same dims, so the term has an
entry point of its own rather than being guessed from the shape:

| Component | Meaning |
|---|---|
| `("credit_loss", l)` | the credit loss on layer `l` (decision (f)) |
| `("recovered", l)` | layer `l`'s ceded loss less its credit loss |
| `("net_after_default", "retained")` | the net loss plus the year's credit loss |

On a labelled result the three new kinds go under the label `""` in every
label dim, as the annual terms do: collateral nets across labels, so the
credit loss has no split by label without a further rule, and
`net_after_default` is the total over labels. The three kind names join
`RESERVED` (`deal.rs`), so no deduction can take them.

`ceded` and `net` are left as they are, so `net = gross - sum of ceded`
still holds, and the new kinds sit beside them: `select("kind",
"net_after_default")` is the cedant's loss after default, and
`select("kind", "credit_loss")` feeds `capital(...)`. `by_reinsurer` gives
dims `["kind", "reinsurer"]` with `credit_loss`, `exposure` (after
collateral) and `default` (0 or 1) for each reinsurer, from the same draws
as `apply` with the same seed.

As built in PR 2 (`crates/prospicio-aggregate/src/reinsurance/panel.rs`),
where the note left a choice open: `credit_loss` and `recovered` come for
every layer of the input, in its order, not only the layers the panel
names (zero credit loss, and the ceded loss recovered in full, on a layer
the panel does not name), so `recovered` summed over layers is the total
recovered. The lines are checked against the result when the panel is
applied (a layer the result does not have, a named layer's lines not
summing to 1, a driver of the wrong length), since a panel is built
before it meets a tower. `apply` refuses a result a panel was already
applied to, and a term's paths (a `"year"` dimension with integer years)
until `apply_term` exists; this check reads the result's shape, so it does
not catch a one-year result labelled by year, whose labels are text. The
draw of `U` is kept below 1 and a probability of default of 1 defaults on
any uniform, since `StreamRng::next_open01` can return exactly 1.0 (with
chance about `2^-53`). The provenance also records the driver
(`default_driver`) and each reinsurer's terms and lines
(`reinsurer:<name>`), and `Panel::reinsurers` lists the panel.

## Decisions

Accepted by the user on 2026-11-24 as the lane recommended them:

(a) *The panel is a value of its own, applied to the tower's result, not a
field of `Tower` or `Layer`.* A field would break the public structs
(`Tower` and `Layer` are not `#[non_exhaustive]`), bump the JSON format
(`serial.rs` version 10) and put default draws inside `push_year`, which
#1053 just made parallel. Applied to the result, the panel touches none of
#1053's code, leaves every tower without a panel bit-identical, and works
the same on `apply`, `apply_aggregate`, `apply_annual` and `apply_term`.
The cost is one extra pass over the result's rows, which is small beside
the tower. Alternative: a `Tower::panel` field with the draws in
`push_year`, breaking, which would only pay if a later feature needs the
default inside the year (a reinstatement not bought after a default, say).

(b) *A line is a share of the layer's ceded loss, and a layer's lines sum
to 1.* "Signed line" here is the line written on the slip, not a sign.
`Layer::share` already holds how much of the layer is placed, so the panel
splits what is placed: lines of one layer must sum to 1 (to 1e-9), or the
panel is rejected, which catches a slip copied wrong. A layer the panel
does not name carries no credit risk (a captive, or a layer fully
collateralised outside the model). Alternative: lines as shares of the
whole layer, summing to `Layer::share`, which makes the panel depend on
the tower's shares.

(c) *Exposure is the ceded loss alone, with no set-off.* Reinstatement
premiums and commissions owed to a defaulted reinsurer may be set off
against its recoveries in practice, but whether they can depends on the
contract and the jurisdiction. Next item `[reinsurer-default-setoff]`.

(d) *Collateral is per reinsurer, netted once across all its layers*
(Article 190's single-name exposure, and how a trust or letter of credit is
usually held). A Solvency II "group" is the caller's to make one reinsurer.

(e) *One-year default at the year's level.* A reinsurer that defaults in a
year fails on all of that year's recoveries, as in the Solvency II one-year
view. Default part way through a long payout, with cumulative probabilities
over a recovery pattern, is Next item `[reinsurer-default-payout]`.

(f) *Credit loss by layer splits a reinsurer's loss over its layers in
proportion to its exposure on each:* layer `l` gets
`L_{r,i} s_{r,l} X_{l,i} / E_{r,i}` (0 when `E_{r,i} = 0`). Collateral
nets across layers, so some rule is needed, and this one makes the layers'
credit losses sum to the reinsurer's. On labelled results the new kinds go
under the label `""`, as the annual terms do.

(g) *Wrong-way risk by one Gaussian factor ranked on the gross loss* (the
model above). It needs one number per reinsurer, keeps each reinsurer's
probability of default, and `rho = 0` is independent default. Alternatives
considered: a stress multiplier (`p` times `k` above a return period) is
simpler but changes the mean probability unless rescaled and has a jump; a
structural model (a reinsurer fails when its share of the industry loss
exceeds its capital) needs market shares and capital the cedant rarely has,
and is Next item `[reinsurer-default-structural]`. The Gaussian factor has
no tail dependence; a t or Clayton factor is Next item
`[reinsurer-default-tail]`.

(h) *Ties in the rank are broken at random within the tie* (the formula
above), so the factor is standard normal and the mean probability of
default is kept. Mid-ranks would shrink the factor's variance where many
years tie at zero.

(i) *Random numbers.* `Panel::apply` takes its own `seed`, and year `i`
draws from `StreamRng::new(seed, DEFAULT_STREAM + i)` with
`DEFAULT_STREAM = 3 << 62`, in this order: `v_i` for the rank, then one
uniform per reinsurer in panel order, drawn whether or not the reinsurer
has an exposure that year. Reinsurer `r` defaults when its uniform is below
`p_r(Z_i)`. The stream block keeps the draws apart from the events' own
streams (`i` for the year, `TIME_STREAM = 1 << 63` plus `i` for event
times) even when the caller passes the events' seed. The result keeps the
tower result's provenance (`draws_from`), so `join` still sees that it
shares the events' streams, and records the panel's seed and stream block
as parameters (`default_seed`, `default_streams`). It adds no sampler id:
`SAMPLERS` lists a model's draw order only once it changes (`rng.md`), and
the order above is documented here and in the module's doc. Each row
depends only on its index and the ranks, which are computed once before the
rows, so the result is the same on any number of threads (#1053's rule).
Over a term the stream is the path's, and the years draw in order.
`rng.md` rule 4 prefers seeds derived by a label hash, which `prospicio-core`
does not have; the stream block follows `TIME_STREAM`'s precedent and needs
no change to the shared crate.

(j) *Probabilities of default are inputs.* The library ships only the
Solvency II table (`solvency2::probability_by_step` and `_by_ratio`, the
regulation's own numbers). Agency default studies change every year and
are not ours to pin.

(k) *Solvency II helpers are pure functions with every input explicit*: the
best estimate of recoverables, the risk-mitigating effect `RM_re` and the
risk-adjusted collateral are the caller's (the library has no standard
formula underwriting SCR to compute `RM_re` from), and the factor of
Article 197(7) is an argument (F is 100 % or 50 %; F' is 100 % or 18 %, a
figure EIOPA Q&A 3119 questions), as is the share, 50 % or the 90 % of a
counterparty with 60 % or more of its assets collateralised.
`probability_by_ratio` clamps at both ends (0.01 % above 196 %, 4.2 %
below 75 %). `type1` groups exposures by probability of default and
returns `V_inter` (over all ordered pairs `(j, k)`, the diagonal included,
so each pair of different groups counts twice; a term with both
probabilities 0 is 0, EIOPA Q&A 2530), `V_intra`, the standard deviation
and `SCR_def,1`.

(l) *Weighted years.* `simulate_events_weighted` returns its weights apart
from the `EventSet`, and the tower's result is unweighted, so ranks and
closed forms here are unweighted; a caller with weights applies them to the
result as for any other component.

## Validation

- **Exact identities** (unit tests): `p = 0` gives no credit loss; `p = 1`
  gives `(1 - R)(E - C)^+` in every year exactly, and over a term the
  collateral used up year by year; `R = 1` or `C >= E` gives no credit
  loss; `rho = 0` gives the same defaults as the independent draws; credit
  loss by layer sums to the credit loss by reinsurer; the input's
  components are copied bit for bit.
- **Determinism**: the result's bits are the same on 1, 2, 3 and 8 threads,
  as in `aggregate_tower_threads.rs`, for `apply`, labelled `apply` and
  `apply_term`.
- **Closed forms against simulation** (validation test, reference file):
  with independent default, `E[credit loss] = sum over r of p_r (1 - R_r)
  E[(E_r - C_r)^+]` and the variance from the law of total variance; with
  the factor, a single excess layer on an annual lognormal gross loss `G`
  ranked on itself, one reinsurer with line `s` and collateral `C`, has
  `E[credit loss] = (1 - R)` times the integral over `u` in (0, 1) of
  `(s c(F_G^{-1}(u)) - C)^+ p(Phi^{-1}(u))` with
  `p(x) = Phi((Phi^{-1}(p) + sqrt(rho) x) / sqrt(1 - rho))` and `c` the
  layer's ceded loss, by quadrature. The simulated mean must fall within
  four standard errors. The probabilities used here are large (5 % to
  30 %) so that a few hundred thousand years resolve them; realistic
  probabilities (0.05 %) are checked by the exact identities instead.
- **Joint default**: the frequency of two reinsurers defaulting together
  against `bivariate_norm_cdf`, within four standard errors.
- **Solvency II**: `type1` and `reinsurance_lgd` against an independent
  implementation of Articles 192, 199, 200 and 201 in the generator, to
  1e-12.
- **Reference generator**: `validation/scripts/aggregate_reinsurer_default.py`
  (PEP 723 metadata, NumPy and SciPy pinned, `exclude-newer` set), writing
  `validation/reference/reinsurer_default.csv` (the CSV layout of
  `reinsurance_aggregate.csv`: closed-form means and variances, the
  quadrature for the factor case, the joint default probabilities, the
  Solvency II cases) with its `MANIFEST.toml` entry; the test is
  `validation/tests/aggregate_reinsurer_default.rs`. The generator computes
  everything from the formulas, with no simulation, so it does not mirror
  the Rust.

## PR plan

Stacked, each on the one before, after #1053 merges (or on its branch, said
in the PR, if it is still open):

1. This note, the lane's Next bullets and a knowledge reference for the
   Solvency II articles. Docs only.
2. `Panel`, `Reinsurer` and `Panel::apply` and `by_reinsurer` on one-year
   results (`apply`, `apply_aggregate`, `apply_annual`, labelled or not),
   with the unit tests, the determinism test
   (`validation/tests/aggregate_reinsurer_default_threads.rs`), the new
   kinds in `RESERVED`, and allow-list entries (reason: bound in PR 5) for
   `Panel` and `Reinsurer`. Independent and factor default together, since
   the factor with `rho = 0` is the independent case.
3. The reference generator, file and validation test.
4. The `solvency2` module, with its rows in the reference file and its
   Python and R bindings (pure functions on numbers).
5. Python and R bindings for the panel (`reinsurance` module files;
   `prospicio.reinsurance.Panel` and `Reinsurer`, `reinsurer_panel` in R),
   their tests in new files, the allow-list entries removed.
6. Default over a term (`Panel::apply_term`, final default, collateral used
   up), in Rust, Python and R, and the Done bullet `#980`.

Every PR appends a Done bullet repeating its own Next id
(`[reinsurer-default-core]` and so on) to `docs/lanes/done/aggregate.md`,
adds a knowledge fragment (or says it taught nothing new), and, when it
changes the public API, a changelog fragment under "Aggregate and
reinsurance".

## Next

The Next items with ids in `docs/lanes/aggregate.md`:
`[reinsurer-default-core]`, `[reinsurer-default-validation]`,
`[reinsurer-default-solvency2]`, `[reinsurer-default-bindings]`,
`[reinsurer-default-term]`, and after #980:
`[reinsurer-default-payout]`, `[reinsurer-default-setoff]`,
`[reinsurer-default-structural]`, `[reinsurer-default-tail]`,
`[reinsurer-default-pnl]`, `[reinsurer-default-json]`.

## Sources

- Commission Delegated Regulation (EU) 2015/35, Articles 189 to 202, as
  amended by 2016/467 and 2019/981: Art 192(2) (loss given default of a
  reinsurance arrangement), Art 196 (risk-mitigating effect), Art 197
  (collateral, factors F), Art 199 (probability of default by credit
  quality step and by solvency ratio), Art 200 (`SCR_def,1`), Art 201
  (`V_inter`, `V_intra`). EIOPA Solvency II single rulebook, and EIOPA Q&A 2530 and
  3119,
  https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook_en
- Reinsurance Bad Debt Provisions for General Insurance Companies, working
  party report (Lloyd's and the Institute of Actuaries, 2000, with a 2005
  note): recoveries from failed reinsurers' estates of 20 % to 60 %, and
  defaults clustered after catastrophes.
- Krvavych, Reinsurance Credit Risk Modelling, ASTIN Colloquium 2009:
  reinsurer default in a DFA model with copulas.
- Vasicek, The Distribution of Loan Portfolio Value, Risk 15(12), 2002: the
  one-factor model and its conditional probability of default.
