# Design note: discounted reserves and an IFRS 17 risk adjustment

Status: **Accepted** (2026-11-14; issue #553, which the user took up on 2026-11-14, lifting its P4 parking). The user decided open decisions 1 to 4 on 2026-11-14 as recommended: (1a) spreading by each simulation's pattern, with `Compartmental`'s paid cells; (2a) both capital charges, end of year by default; (3a) mid-period payments by default; (4b) a stated basis. Decisions 5 to 8 are built as recommended unless the user decides otherwise. **PR 1 is built** (Rust, branch `claude/ifrs17-reserves`, 2026-11-14): `prospicio_reserving::discount`, with the decisions the build made recorded as open decisions 9 to 19, built as described there; on 2026-11-15 the user decided 17 (the running maximum kept) and 18 (the incurred basis discounts the whole unpaid, IBNR plus case reserves, built on the same branch). **PR 2 is built** (branch `claude/ifrs17-bindings`, 2026-11-16): the Python and R bindings, with the decisions the build made recorded as open decisions 20 to 25, and the section of `docs/guides/reserving-methods.md` (a Guides-lane file, edited with the user's go-ahead of 2026-11-16). Lane: Reserving (`crates/prospicio-reserving/`, the lane's binding modules) · Depends on: `time-value.md` (Aggregate lane, read only: decisions 3, 4, 7 and 37(a))

## Goal

Under IFRS 17 a reserving actuary reports the liability for incurred
claims (LIC) as the present value of the unpaid claims plus a risk
adjustment for non-financial risk (RA), and discloses the confidence level
that the RA corresponds to. Today a user takes the reserve draws out of
the library and does this by hand.

Three pieces are missing:

1. **Discounted reserves**: each simulation's unpaid claims, by origin and
   in total, discounted from the valuation date along their payment timing.
2. **A risk adjustment** on the discounted total: by a confidence level,
   by TVaR, or by a cost-of-capital method, each reporting its amount and
   the implied confidence level (the confidence level its result
   corresponds to).
3. **A doc section** on the one-year view's claims development result as
   the Solvency II reserve-risk input (its 99.5% VaR). No new engine.

## What exists

* Reserve draws: every stochastic fit's `reserves` is a
  `PredictiveDistribution` with one component per origin (per segment and
  origin for the `*Fits`), simulation-major (`prospicio-prob`
  `predictive.rs:128`). The trait `UltimateDraws` (`draws.rs:104`) is
  implemented by the ODP and Mack bootstraps, `BayesOdp`, `BayesBf`,
  `BayesBfNormalised`, `Meyers` and `Compartmental`, and their `*Fits`.
* Timing: the same fits carry `patterns: DevelopmentPatterns`
  (`pattern_draws.rs:75`), one cumulative pattern per simulation, row `i`
  belonging to simulation `i` of `reserves`. The pattern carries parameter
  uncertainty only, no process error (`pattern_draws.rs:15-23`). In the
  Bayesian fits the reserve and the pattern of a simulation come from the
  same posterior draw (`bayes/odp.rs:667-683` through
  `predictive_with_patterns`), and `reorder_with` reorders the patterns
  with the reserves across segments. With a tail the pattern gains one
  age, one development period past the oldest, where it reaches 1
  (`pattern_draws.rs:719`); the bootstraps pay the whole tail at that age.
* `CompartmentalFit.paid` (`bayes/compartmental/results.rs:82`) is the
  only output with each simulation's future paid by cell: cumulative paid
  by origin and age. Under `Fitted::IncurredOnly` it is the expected paid
  without process noise.
* The simulated future cells of the other fits exist only inside their
  simulation loops (`bootstrap.rs:1143-1151`, `mack_bootstrap.rs:712-735`,
  `bayes/odp.rs:673-678`, `bayes/bf.rs:326-331`,
  `bayes/bf_normalised.rs:447`) and are summed away. `Meyers` simulates no
  path at all: the ultimate is drawn from the oldest age alone.
* Discounting today lives in `prospicio-pricing`: `PaymentPattern`
  (`payout.rs:47`) with `unpaid_from`, `discount_factor` (one flat annual
  effective rate, discrete payments at the stated times) and
  `capital_years` (capital charged at the start of each year, year 0 in
  full, `time-value.md` decision 4). `PaymentPattern::new` rejects negative
  shares. No discount curve or term-structure type exists anywhere in the
  workspace.
* `prospicio-reserving` and `prospicio-pricing` do not depend on each other
  (`time-value.md` decision 37(a)); both are in the WASM scope.
* Quantiles: `Sampled::cdf(x) = #{d ≤ x}/n` and the default inverted-cdf
  quantile (`prospicio-prob` `sampled.rs:197`, `risk.rs:31`) round-trip:
  `F(Q(p)) ≥ p`.
* Neither `HierarchicalBf` (no `UltimateDraws`, no pattern per simulation)
  nor analytic `Mack` (no draws) has what this needs.

## The method

### 1. Each simulation's payments

For simulation `s`, origin `w` at age `a_w` (its latest position) with
unpaid draw `R_{s,w}` (the reserve on the paid basis; the reserve plus the
origin's case reserve on the incurred basis, decision 18) and cumulative
paid pattern `c_s(d)` at the pattern's ages `d`:

```
m_s(d)         = max(c_s(a_w), ..., c_s(d))      the running maximum
share_{s,w}(d) = (m_s(d) - m_s(d-1)) / (m_s(last) - c_s(a_w))   for d > a_w
payment_{s,w}(d) = R_{s,w} · share_{s,w}(d)
```

For a pattern that never falls after `a_w` and ends at 1 this is
`(c_s(d) - c_s(d-1)) / (1 - c_s(a_w))` (decision 17).

paid at time `t = (d - a_w) · Δ - m` years after the valuation date, where
`Δ` is the development period in years and `m` is 0 for payments at the
end of each period or `Δ/2` for the middle (open decision 3).

* When `1 - c_s(a_w)` is at most `1e-12` (an origin fully developed in that
  simulation, or a pattern that has passed 1 by pseudo factors below 1),
  the whole unpaid is paid at the next age.
* A falling step of the pattern pays nothing and the rises are rescaled to
  sum to 1, so every share is in `[0, 1]` and each draw's present value is
  its unpaid times a weighted average of its payment times' discount
  factors (decision 17, which replaces keeping negative increments). A
  negative reserve discounts to a negative present value. The code does
  not go through `PaymentPattern::new`.
* Under `PaymentBasis::Incurred { paid_pattern, latest_paid }` the amount
  spread is the whole unpaid: each simulation's reserve (ultimate less
  latest incurred, the IBNR) plus the origin's case reserve (the fit's
  latest incurred less `latest_paid`), along `paid_pattern` from `a_w` by
  the same share rule (decision 18). An origin fully paid by the pattern at
  `a_w` pays its case reserve at the next age; a latest paid above the
  latest incurred is a negative case reserve, kept.
* `Compartmental` uses its `paid` cells directly: the increments along
  `age` from `latest_paid`, each at its own calendar time. This is exact,
  process timing included. Under `IncurredOnly` it is an error, since the
  paid cells have no noise.
* Ages are `Lag`s in months on the triangle's grain; each origin is
  discounted from its own age, so ragged triangles and quarterly or
  monthly grains need nothing special.

This is the cost of spreading (open decision 1): the timing varies between
simulations only through the pattern, so the uncertainty in the timing is
the parameter uncertainty of the pattern, not the process error of each
cell. IFRS 17 B91 lists the uncertainty about timing among the
characteristics of the RA; the spreading understates it. The exact route
keeps each simulation's future cells inside the loops (open decision 1,
option (b)).

### 2. Discounting

`DiscountCurve` in `prospicio-reserving` (open decision 5): a flat annual
effective rate, or spot rates by maturity in years, annually compounded,
linear in the spot rate between maturities and flat beyond the last one.
The discount factor of a payment at `t` is `(1 + r(t))^-t`.

```
PV_{s,w} = Σ_d payment_{s,w}(d) · (1 + r(t_d))^-t_d
PV_s     = Σ_w PV_{s,w}
```

Every simulation is discounted on its own timing, then summed; the
quantiles of the discounted total are read from `PV_s`, never by scaling
the undiscounted quantiles origin by origin (`time-value.md` decision 7
scales each draw by one factor, which is exact for one origin only and
changes the ranks of a total across origins).

Output: `DiscountedReserves` with the undiscounted unpaid (`unpaid`: the
fit's reserves on the paid basis, the reserves plus the case reserves on
the incurred basis) and its present values (`discounted`) as
`PredictiveDistribution`s on the same dimensions and simulations as the
fit's `reserves`, the discount factor of each origin's mean
(`mean PV / mean R`), and the mean discounted run-off at each future year
(the best estimate outstanding at `t`, used by the cost-of-capital method).

### 3. The risk adjustment

All on the discounted total `X = PV_s` (the joint total of a `*Fits`, with
the fit's dependence between segments). The best estimate is
`BE = mean(X)`.

* `RiskAdjustment::ConfidenceLevel(p)`: `RA = Q_p(X) - BE`, inverted-cdf
  quantile.
* `RiskAdjustment::Tvar(p)`: `RA = TVaR_p(X) - BE`.
* `RiskAdjustment::CostOfCapital { rate, capital, charge }` (open
  decisions 2 and 6): `K_0` is the capital at the valuation date,
  `K_t = K_0 · BE(t) / BE(0)`, where `BE(t)` is the mean discounted
  outstanding at `t` (EIOPA's method 2 run-off proxy), and

  ```
  start of year (time-value.md decision 4):  RA = rate · Σ_{t≥0} K_t · v(t)
  end of year (Solvency II art. 37):          RA = rate · Σ_{t≥0} K_t · v(t+1)
  ```

  with `v(t) = (1 + r(t))^-t` from the same curve. `capital` is
  `Capital::Var(q)` (`Q_q(X) - BE`; no default, decision 6b),
  `Capital::Tvar(q)` or `Capital::Amount(k)` (a capital the user brings, such as the one-year
  view's capital, `mean CDR - Q_0.005(CDR)`, decision 24).

Each result reports `amount`, `best_estimate`, and
`implied_confidence = #{X_s ≤ BE + RA} / n` (IFRS 17 para. 119), on the
same `X`. Under `ConfidenceLevel(p)` it is at least `p` by the inverted-cdf
round trip.

Per segment: the RA of each segment stand-alone is reported beside the
total's; the total is not allocated (open decision 8).

### 4. The one-year view for Solvency II

A doc section only: the one-year view's `cdr` (`OneYearFit`, the Bayesian
`one_year`, `MackFit.cdr`) is the reserve-risk input. The CDR is the
opening ultimate less the closing one (`one_year_bootstrap.rs:27-30`), so
an adverse year is a negative CDR and the 99.5% VaR of the loss is
`-cdr.total().quantile(0.005)`. The capital above the expected loss is that
less the expected loss, `-mean CDR`, so `mean CDR - Q_0.005(CDR)` (decision
24). That amount can be brought to `Capital::Amount`.

## API

Rust (`crates/prospicio-reserving/src/discount.rs`, a new module):

```rust
pub struct DiscountCurve { /* (maturity, spot) pairs; flat is one pair */ }
impl DiscountCurve {
    pub fn flat(rate: f64) -> Result<Self>;
    pub fn spot(maturities: &[f64], rates: &[f64]) -> Result<Self>;
    pub fn rate(&self, t: f64) -> f64;
    pub fn factor(&self, t: f64) -> f64;
}

#[derive(Default)]
pub enum PaymentTiming { #[default] MidPeriod, EndOfPeriod }

pub struct Discounting { pub curve: DiscountCurve, pub timing: PaymentTiming }

pub enum PaymentBasis {                               // decision 4b; 9, 18
    Paid,
    Incurred { paid_pattern: Vec<f64>, latest_paid: Vec<f64> },
}

pub trait Discountable: sealed::Sealed {}

pub fn discounted_reserves<F: Discountable + ?Sized>(
    fit: &F,
    discounting: &Discounting,
    basis: &PaymentBasis,
) -> Result<DiscountedReserves>;

pub struct DiscountedReserves {
    pub unpaid: PredictiveDistribution,  // reserves; plus case reserves if incurred
    pub discounted: PredictiveDistribution,
    pub best_estimate: f64,
    pub runoff: Vec<f64>,              // BE(t), t = 0, 1, ..., ending at 0
    pub discount_factors: Vec<f64>,    // mean PV / mean unpaid per component
    /* the discounting, segments and their run-offs */
}

pub enum Capital { Var(f64), Tvar(f64), Amount(f64) }   // no default (6b)
#[derive(Default)]
pub enum CapitalCharge { #[default] EndOfYear, StartOfYear }

pub enum RiskAdjustment {
    ConfidenceLevel(f64),
    Tvar(f64),
    CostOfCapital { rate: f64, capital: Capital, charge: CapitalCharge },
}
impl DiscountedReserves {
    pub fn discounting(&self) -> &Discounting;
    pub fn segment_labels(&self) -> Vec<Vec<String>>;
    pub fn segment_runoff(&self, i: usize) -> Option<&[f64]>;
    pub fn risk_adjustment(&self, method: &RiskAdjustment) -> Result<RiskAdjustmentResult>;
}

pub struct RiskAdjustmentResult {
    pub amount: f64,
    pub best_estimate: f64,
    pub implied_confidence: f64,
    pub capital: Option<f64>,                     // K_0 of a cost of capital
    pub segments: Vec<SegmentRiskAdjustment>,     // stand-alone, decision 8a
}
pub struct SegmentRiskAdjustment { pub label: Vec<String>, pub result: RiskAdjustmentResult }
```

`Discountable` is implemented by the fits with reserves and patterns:
the ODP and Mack bootstraps, `BayesOdp`, `BayesBf`, `BayesBfNormalised`,
`Meyers` (spreading only) and `Compartmental` (its paid cells), and their
`*Fits`. `HierarchicalBfFit` and analytic `MackFit` implement it too and
return `Error::NotSupported` with the reason (no pattern per simulation; no
simulations) (decision 13).

Python, as built (PR 2, decisions 20 to 23), in `prospicio.reserving`:

```python
fit.discounted(*, basis, rate=None, curve=None, timing="mid",
               paid_pattern=None, latest_paid=None) -> DiscountedReserves
DiscountCurve(maturities, rates); DiscountCurve.flat(rate)   # .rate(t), .factor(t)
DiscountedReserves: unpaid, discounted, best_estimate, runoff, discount_factors,
    curve, timing, segment_labels, segment_runoff(i),
    risk_adjustment(method, p=None, *, rate=None, capital=None, charge="end")
RiskAdjustmentResult: amount, best_estimate, implied_confidence, capital, segments
SegmentRiskAdjustment: label, result
```

on `OdpBootstrapFit`, `MackBootstrapFit`, `BayesOdpFit`, `BayesOdpFits`,
`BayesBfFit`, `BayesBfNormalisedFit`, `MeyersFit`, `MeyersFits`,
`CompartmentalFit` and `CompartmentalFits`, and on `MackFit` and
`HierarchicalBfFit`, where it raises `NotSupportedError`.

R, as built:

```r
discounted_reserves(fit, basis, rate = NULL, curve = NULL, timing = c("mid", "end"),
                    paid_pattern = NULL, latest_paid = NULL)   # a discounted_reserves object
risk_adjustment(d, method = c("confidence", "tvar", "cost_of_capital"), p = NULL,
                rate = NULL, capital = NULL, charge = c("end", "start"))   # a list
segment_runoff(d, i)   # segment i's run-off, i counted from 1
discount_curve(maturities, rates); discount_curve_factor(curve, t); discount_curve_rate(curve, t)
```

## Crates touched

`prospicio-reserving` only (a new `discount.rs`, `lib.rs` one line, the
`Discountable` impls beside each fit), then the lane's binding modules.
No dependency on `prospicio-pricing` and nothing in `prospicio-core`
(`time-value.md` decision 37(a); open decision 5). WASM check:
`cargo check --target wasm32-unknown-unknown -p prospicio-reserving`.

## Validation targets

1. A zero rate reproduces the undiscounted reserves, by simulation, bit
   for bit up to the sum's order.
2. A flat rate on RAA with the ODP bootstrap's mean pattern matches a cash
   flow built by hand from the chain ladder's factors (each origin's
   future increments from the factors, discounted).
3. With every simulation given one pattern (a selected pattern, no
   parameter error), the discounted total equals the undiscounted total
   times the per-origin factors summed, and agrees with
   `PaymentPattern::from_development(...).unpaid_from(age).discount_factor(rate)`
   per origin (checked in the bindings test, where both crates are
   reachable).
4. The cost of capital at the start-of-year charge, with `Capital::Amount`
   and a flat rate, equals `capital_years`' formula on the same run-off
   (time-value.md decision 4); the end-of-year charge equals the
   Solvency II art. 37 sum built by hand.
5. `implied_confidence` of `ConfidenceLevel(p)` is at least `p`, and less
   than `p + 1/n`.
6. `Compartmental` with process noise: the discounted total from the paid
   cells and from spreading by its patterns, side by side. As built
   (decision 19) the means agree within 0.5% and the sds within 2%, not
   within simulation error: the spread uses the population's pattern.
7. The tail: an origin at the oldest age with a tail pays its whole
   reserve at the tail's age, one period on.

No published IFRS 17 example uses these models on a public triangle, so
there is no book parity target; Meyers' chapter 11 risk margin (TVaR 97%
of the ultimate, the spread `r - i` discounted at `r`) is a different
formula, not built (open decision 7).

## Pull requests

1. Rust: `discount.rs`, `DiscountCurve`, `discounted_reserves` and
   `RiskAdjustment` for the fits above, the tests above, a knowledge
   concept on the IFRS 17 RA conventions (paragraph texts marked as not
   checked against the standard where they are not).
2. Python and R bindings, with the regenerated stub, `man/`, `NAMESPACE`
   and wrappers; the guide `docs/guides/reserving-methods.md` gets the
   discounted reserve, the RA and the Solvency II section (Guides lane's
   file: with the user's go-ahead, or as an issue for that lane).

## Not in this note

ULAE and expense risk (`expense.rs` exists; the RA would cover them under
IFRS 17 B89-B90), reinsurance held (a separate RA under IFRS 17 para. 64),
an illiquidity premium (the user brings it in the curve), the Solvency II
lambda and floor from 2027 (Delegated Regulation 2026/269; not verified
against the English text), currency, the LRC.

## Open decisions

1. **Where each simulation's payment timing comes from.** (a) Spread each
   origin's reserve draw over that simulation's pattern (section 1), and
   use `Compartmental`'s paid cells: no change to any simulation loop, so
   the eleven seed pins cannot move; timing uncertainty is the pattern's
   parameter uncertainty only. (b) Keep each simulation's future cells
   inside the loops of the bootstraps and the Bayesian models and discount
   them exactly: process timing included, but every loop changes and must
   leave every pin bit for bit, memory grows with simulations times cells,
   and `Meyers` still needs (a). Recommended: (a), with (b) as a
   follow-up if the spread of the discounted total matters.
2. **The cost-of-capital charge.** (a) Both conventions as a setting, the
   default end of year on the curve (Solvency II art. 37, the convention
   most IFRS 17 cost-of-capital RAs borrow). (b) Default start of year,
   as `price_over_payout` (`time-value.md` decision 4), which the issue's
   test names. (c) One convention only. Recommended: (a).
3. **Payment timing within a period.** (a) Middle of the period by
   default, end as a setting. (b) End by default, as `PaymentPattern`.
   Recommended: (a): claims are paid through the period.
4. **The paid basis.** A fit does not know whether its column is paid;
   discounting an incurred fit by its own (reporting) pattern is wrong.
   (a) Use the fit's own patterns and say in the docs that the fit must be
   on paid claims; a `pattern` argument lets the user bring a paid pattern
   (one cumulative pattern by age, applied to every simulation) for an
   incurred fit. (b) Require the user to state the basis
   (`Basis::Paid` uses the patterns; `Basis::Incurred` requires a paid
   pattern), so an incurred fit cannot be discounted by mistake.
   Recommended: (b).
5. **Where the discounting code lives.** (a) A small `DiscountCurve` in
   `prospicio-reserving`. (b) A dependency on `prospicio-pricing`, which
   reverses `time-value.md` decision 37(a). (c) Move `PaymentPattern` to
   `prospicio-core` in a PR of its own first. Recommended: (a): the curve
   is new in any case, and `PaymentPattern` cannot carry negative shares.
6. **The default capital of the cost-of-capital method.** (a)
   `Capital::Var(0.995)` of the discounted lifetime total minus the best
   estimate. (b) No default: the user names it. Recommended: (b), since
   the lifetime 99.5% VaR is larger than Solvency II's one-year capital
   and neither is standard under IFRS 17.
7. **Meyers' risk margin** (chapter 11: TVaR 97% of the ultimate projected
   per future year, the spread `r - i` discounted at `r`). (a) Not built.
   (b) A further `RiskAdjustment` variant with the book's group 353
   example as a parity target. Recommended: (a) for now.
8. **Segments.** (a) The RA of the joint total, with each segment's
   stand-alone RA beside it and no allocation. (b) The total allocated to
   segments by a `prospicio-prob` allocation method. Recommended: (a).

9. **Names and the paid basis's pattern** (decided by the build).
   `Basis` is already `excess::Basis` at the crate root, so the stated
   basis is `PaymentBasis { Paid, Incurred { paid_pattern, latest_paid } }`;
   the cost of capital's field is `charge`. Built: `paid_pattern` is the
   cumulative paid share at each of the fit's pattern ages (the triangle's
   development ages, plus the tail's age when the fit has a tail), finite
   and ending at 1 within 1e-9, applied to every simulation and every
   segment; a pattern of another length is `Error::LengthMismatch`, so
   segments with different ages need their own fits discounted one by one.
   `latest_paid` (decision 18) is one finite value per component of the
   fit's reserves, segment-major for a fit over segments; another length
   is `Error::LengthMismatch`, a value that is not finite
   `Error::InvalidSetting`.
10. **Shares that cannot be used** (decided by the build). Built: when
    `1 - c(a_w)` is at most 1e-12 (or not a number), or any share is not
    finite, the whole unpaid is paid at the next pattern age; when the
    pattern has no later age, one development period on (the step between
    the last two ages, or the only age itself). Negative reserves are kept;
    negative shares were kept too until decision 17.
11. **The run-off's end and a zero best estimate** (decided by the build).
    Built: `runoff` runs over `t = 0, 1, ...` whole years up to the first
    whole year at or after the last payment, where it is 0, so it ends with
    that 0. In the cost of capital, `BE(0) = 0` gives a risk adjustment of
    0 when `K_0` is 0 and an error otherwise.
12. **Each origin's discount factor** (decided by the build). Built:
    `discount_factors` is mean PV over mean unpaid per component (mean
    reserve on the paid basis), and 1 where the mean unpaid is exactly 0
    (an origin with nothing to pay).
13. **The unsupported fits and the trait** (decided by the build). Built:
    `Discountable` is sealed (its cash flows are internal); analytic
    `MackFit` and `HierarchicalBfFit` implement it and return
    `Error::NotSupported` with the reason. Errors reuse the existing
    variants (`InvalidSetting`, `LengthMismatch`, `PatternDraws`,
    `NotSupported`), so no binding's match on `Error` changes.
14. **The implied confidence level against the level itself** (decided by
    the build). Built: for `ConfidenceLevel(p)` and `Tvar(p)` the draws are
    counted against `Q_p` or `TVaR_p` itself rather than `BE + RA`, since
    `BE + (Q_p - BE)` can round below `Q_p` and drop a tied draw; for the
    cost of capital against `BE + RA`.
15. **Segments under a capital amount** (decided by the build). Built: a
    `Capital::Amount` is the total's capital, so no stand-alone segment
    results are reported for it (`segments` is empty); `Var` and `Tvar`
    capitals give each segment its own `K_0` and run-off. A fit of one
    segment reports no segments.
16. **`Compartmental`'s valuation age** (decided by the build). Built: each
    origin's paid increments are timed from its latest observed paid age
    (an origin with no observed paid from age 0, the start of the origin),
    each cell's increment from the previous cell, the first from the latest
    paid. Its unit test (`the_discounted_reserves_are_paid_by_the_cells`)
    lives in `bayes/compartmental/tests.rs`, beside the triangle and quick
    sampler it uses; it also checks that `IncurredOnly` and
    `PaymentBasis::Incurred` are refused.
17. **Patterns that fall after the latest age** (decided by the build,
    after review; decided by the user on 2026-11-15: the running maximum
    kept as built). Keeping negative increments as shares over
    `1 - c_s(a_w)` amplifies a reserve when a bootstrap's pattern goes just
    past 1 and comes back: with `c(a_w) = 0.99998`, then 1.005 and 1.0, the
    shares are about +251 and -250 and the draw's present value about 8.2
    times its reserve at 3%. On RAA with 20,000 simulations at 3%
    mid-period the ODP bootstrap's unit present values ran from -33 to 68
    by origin and its discounted total from 0.05 to 1.92 times the
    undiscounted. Built: the shares are the rises of the pattern's running
    maximum after `a_w`, over their sum (falling steps pay nothing), for
    each simulation's pattern and for a stated paid pattern alike; the
    whole unpaid is paid at the next age when `1 - c_s(a_w)` or that sum
    is at most 1e-12 or the pattern is not finite. Every share is then in
    `[0, 1]` and the unit present values on RAA lie between the first and
    the last payment time's factors (0.79 to 0.99 for the Mack bootstrap,
    0.89 to 0.99 for the ODP's; a unit test checks the bound on 2,000
    simulations of each); a pattern that never falls is unchanged.
    Salvage in a stated paid pattern is no longer timed as a negative
    payment. The other rules considered (a clamp of each draw's
    increments, a fallback to the mean pattern) were not taken.
18. **The incurred basis discounts the whole unpaid** (decided by the user
    on 2026-11-15). The reserves of a fit on incurred claims are ultimate
    less latest incurred, the IBNR; the first build discounted that alone
    and left the case reserves to the caller, which understates both the
    liability for incurred claims and its risk adjustment. Built:
    `PaymentBasis::Incurred { paid_pattern, latest_paid }`; each
    simulation's unpaid for origin `w` is `reserve_{s,w} + (latest
    incurred_w - latest_paid_w)`, with the latest incurred the fit's own
    (its chain ladder's latest values, as `UltimateDraws::latest_values`
    gives them, for every spread fit including Meyers' and the `*Fits`),
    spread along `paid_pattern` from the origin's latest age with the paid
    basis's share rule (decision 17). `DiscountedReserves::reserves` is
    renamed `unpaid`: the reserves plus the case reserves on the incurred
    basis, the reserves themselves on the paid basis; the discount factors
    and the run-off follow it. The case reserves are fixed, the same in
    every simulation, so they add to the mean and not to the spread of the
    undiscounted total, but their timing does move the discounted one. An
    origin that the paid pattern has fully paid at its latest age pays its
    case reserve at the next age, and a latest paid above the latest
    incurred is a negative case reserve, kept and documented. Unit tests:
    a hand-built four-origin case against present values by hand, the
    IBNR-only result when every latest paid equals the latest incurred,
    RAA at a zero rate (the mean reserve plus the case reserves), a fit
    over segments (segment-major), and the length and finiteness errors.
19. **Validation target 6 as built** (decided by the build). The spread
    route reaches `Compartmental` through a test-only helper
    (`discount::compartmental_spread`), no public API, from each origin's
    latest observed paid age, so the test uses the training triangle
    without 1997 (which has no paid). On company 337 at 5% mid-period with
    the quick sampler (Windows): from the cells 125,357 (sd 5,791), spread
    125,450 (sd 5,780), a paired difference of -92 with a standard error of
    16, so the means differ by more than simulation error (the spread uses
    the population's pattern, the cells each origin's own development) but
    by 0.07%, and process timing barely changes the spread. The test
    (`the_discounted_cells_and_the_spread_patterns_side_by_side`) asserts
    the means within 0.5% and the sds within 2%.
20. **The bindings' arguments** (decided by the build, PR 2). Python's
    `discounted` takes every argument by keyword and `basis` has no
    default, so leaving it out is a `TypeError` (decision 4b); R's
    `discounted_reserves()` stops with a message when `basis` is missing.
    Exactly one of `rate` (a flat curve) and `curve` is given. The option
    names are `basis` `"paid"` or `"incurred"`, `timing` `"mid"` (the
    default) or `"end"`, `method` `"confidence"`, `"tvar"` or
    `"cost_of_capital"` and `charge` `"end"` (the default) or `"start"`.
    `paid_pattern` and `latest_paid` given under `"paid"`, or missing under
    `"incurred"`, are an error before the fit is read. `p` is for the first
    two methods, `rate` and `capital` for the third, and a stray one is an
    error. Python's `segment_runoff(i)` counts from 0, R's from 1.
21. **How the capital is given** (decided by the build, PR 2). Python: a
    number is `Capital::Amount`, a pair `("var", q)`, `("tvar", q)` or
    `("amount", k)` the variant it names. R: a number, or a named number
    `c(var = q)`, `c(tvar = q)` or `c(amount = k)`. One spelling covers the
    three variants without three arguments that exclude each other, and an
    amount (the one-year capital of section 4) needs no wrapper.
22. **The unsupported fits** (decided by the build, PR 2). Python's
    `MackFit` and `HierarchicalBfFit` have the method and R's
    `discounted_reserves()` takes them, so the user meets Rust's reason
    (decision 13) as `NotSupportedError` or the `prospicio_not_supported`
    condition, rather than a missing method. Python's and R's analytic Mack
    fit holds Mack over segments; the first segment's fit is passed, whose
    error is the same for every segment.
23. **Where the binding code lives and what is bound** (decided by the
    build, PR 2). Each language's module is `reserving_discount.rs`
    (`crates/prospicio-python/src`, `crates/prospicio-r/src`) and
    `R/prospicio/R/reserving_discount.R`. A pyo3 class has one
    `#[pymethods]` block, so each fit's `discounted` method in
    `reserving.rs` is a delegate to the module's generic helper. R has one
    entry point that matches the fit's external pointer against each
    supported fit, which needs those fits' `inner` fields `pub(crate)`. The
    helpers are generic over `Discountable`, which binds the trait, so all
    twelve items of the allow-list are bound and no entry remains. R's
    curve is an S7 object of plain numbers checked by `DiscountCurve::spot`
    when it is made, with `discount_curve_factor()` and
    `discount_curve_rate()` to read it; R's result is the S7 class
    `discounted_reserves` (`discounted_reserves_class`) and its risk
    adjustment a list whose `segments` is a data frame.
24. **The Solvency II capital from the one-year view** (decided by the
    build, PR 2; the guide's section). Section 4's 99.5% VaR of the loss
    is `-Q_0.005(CDR)`; the capital is the amount above the expected loss,
    `-mean CDR`, so `mean CDR - Q_0.005(CDR)`, which is what the guide
    brings to `capital=`. On the guide's triangle the Mack bootstrap's
    mean CDR is -1.2, so the capital is 1.2 below the VaR.
25. **A fit over one segment reports no segments** (decided by the build,
    PR 2; a change to PR 1's Rust). The bindings hold the fit of a
    triangle without keys as a `*Fits` of one segment (the bootstraps
    always, the Bayesian and compartmental fits over segments when the
    triangle has one), and PR 1 reported that segment, with an empty
    label, beside the total in every risk adjustment, contrary to decision
    15's "a fit of one segment reports no segments". Built: `segments`,
    `segment_labels()` and the segments of a risk adjustment are reported
    for a fit over two or more segments only; `segment_runoff(0)` still
    gives the one segment's run-off. Unit test:
    `a_fit_over_one_segment_reports_no_segments`.
