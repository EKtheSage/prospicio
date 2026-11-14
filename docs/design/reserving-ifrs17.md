# Design note: discounted reserves and an IFRS 17 risk adjustment

Status: **Accepted** (2026-11-14; issue #553, which the user took up on 2026-11-14, lifting its P4 parking). The user decided open decisions 1 to 4 on 2026-11-14 as recommended: (1a) spreading by each simulation's pattern, with `Compartmental`'s paid cells; (2a) both capital charges, end of year by default; (3a) mid-period payments by default; (4b) a stated basis. Decisions 5 to 8 are built as recommended unless the user decides otherwise. Nothing is built yet. Lane: Reserving (`crates/prospicio-reserving/`, the lane's binding modules) · Depends on: `time-value.md` (Aggregate lane, read only: decisions 3, 4, 7 and 37(a))

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
reserve draw `R_{s,w}` and cumulative pattern `c_s(d)` at the pattern's
ages `d`:

```
share_{s,w}(d) = (c_s(d) - c_s(d-1)) / (1 - c_s(a_w))   for d > a_w
payment_{s,w}(d) = R_{s,w} · share_{s,w}(d)
```

paid at time `t = (d - a_w) · Δ - m` years after the valuation date, where
`Δ` is the development period in years and `m` is 0 for payments at the
end of each period or `Δ/2` for the middle (open decision 3).

* When `1 - c_s(a_w)` is at most `1e-12` (an origin fully developed in that
  simulation, or a pattern that has passed 1 by pseudo factors below 1),
  the whole reserve is paid at the next age.
* Negative increments are kept and discounted as they are (incurred
  patterns, salvage, falling bootstrap patterns); a negative reserve
  discounts to a negative present value. The code does not go through
  `PaymentPattern::new`.
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

Output: `DiscountedReserves` with the undiscounted and discounted reserves
as `PredictiveDistribution`s on the same dimensions and simulations as the
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
* `RiskAdjustment::CostOfCapital { rate, capital, convention }` (open
  decisions 2 and 6): `K_0` is the capital at the valuation date,
  `K_t = K_0 · BE(t) / BE(0)`, where `BE(t)` is the mean discounted
  outstanding at `t` (EIOPA's method 2 run-off proxy), and

  ```
  start of year (time-value.md decision 4):  RA = rate · Σ_{t≥0} K_t · v(t)
  end of year (Solvency II art. 37):          RA = rate · Σ_{t≥0} K_t · v(t+1)
  ```

  with `v(t) = (1 + r(t))^-t` from the same curve. `capital` is
  `Capital::Var(q)` (`Q_q(X) - BE`, default `q = 0.995`), `Capital::Tvar(q)`
  or `Capital::Amount(k)` (a capital the user brings, such as the one-year
  view's 99.5% VaR of the claims development result).

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
`-cdr.total().quantile(0.005)` (less the mean CDR, if it is not zero).
That amount can be brought to `Capital::Amount`.

## API

Rust (`crates/prospicio-reserving/src/discount.rs`, a new module):

```rust
pub struct DiscountCurve { /* flat rate or (maturity, spot) pairs */ }
impl DiscountCurve {
    pub fn flat(rate: f64) -> Result<Self>;
    pub fn spot(maturities: &[f64], rates: &[f64]) -> Result<Self>;
    pub fn factor(&self, t: f64) -> f64;
}

pub enum PaymentTiming { EndOfPeriod, MidPeriod }

pub struct Discounting { pub curve: DiscountCurve, pub timing: PaymentTiming }

pub fn discounted_reserves<F: Discountable>(fit: &F, discounting: &Discounting)
    -> Result<DiscountedReserves>;

pub enum RiskAdjustment {
    ConfidenceLevel(f64),
    Tvar(f64),
    CostOfCapital { rate: f64, capital: Capital, convention: CapitalCharge },
}
impl DiscountedReserves {
    pub fn risk_adjustment(&self, method: &RiskAdjustment) -> Result<RiskAdjustmentResult>;
}
```

`Discountable` is implemented by the fits with reserves and patterns:
the ODP and Mack bootstraps, `BayesOdp`, `BayesBf`, `BayesBfNormalised`,
`Meyers` (spreading only) and `Compartmental` (its paid cells), and their
`*Fits`. `HierarchicalBf` and analytic `Mack` are not: the message says why
(no pattern per simulation; no simulations).

Python: `fit.discounted(rate=0.03 | curve=..., timing="mid")` returning
`DiscountedReserves` with `reserves`, `discounted`, `best_estimate`,
`runoff`, and `risk_adjustment(method="confidence", p=0.75)` (and
`"tvar"`, `"cost_of_capital"` with `rate`, `capital`, `charge`).
R: `discounted_reserves(fit, rate =, curve =, timing =)` and
`risk_adjustment(d, method =, ...)`.

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
   cells and from spreading by its patterns agree in the mean within
   simulation error and differ in spread; the test records both.
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
