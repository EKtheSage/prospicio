# Design note: the time value of money in a price

Status: **Implemented** · Depends on: `risk_load` prices · Lane: Aggregate ·
Issues: #195, #480 (return metrics)

## Goal

A long-tail treaty's price depends on when its losses are paid: the
reinsurer earns investment income on the premium until it pays, and holds
capital while losses are outstanding, so the cost of capital accrues over
several years rather than one. Before this note, pricing had no payment
patterns and no discounting. `Pentagon`'s discount is a different idea (the
spread between the premium and the expected loss in the natural allocation,
not a payment delay).

Clark's *Basics of Reinsurance Pricing* names the timing of premium and
loss payments as part of the final price but gives no worked example; the
validation checks closed forms instead.

## What exists

`prospicio_pricing::payout`:

- `PaymentPattern::new(times, shares)`: the share of the ultimate loss
  paid at each time, in years from the treaty's inception.
  `PaymentPattern::from_development(ages, factors)` builds one from
  factors to ultimate, such as a paid chain ladder's `cdf`: the share paid
  by each age is one over its factor. `shifted(by)` moves it later.
- `duration()`, `outstanding(t)`, `discount_factor(rate)` and
  `capital_years(rate)`.
- `price_over_payout(&price, &rule, &pattern, rate) -> PayoutPrice`: a
  one-period `Price` restated over the payout.

Bindings: Python `prospicio.pricing.PaymentPattern`, `price_over_payout`,
`PayoutPrice`; R `payment_pattern()`, `payment_pattern_from_development()`,
`pattern_values()` and `price_over_payout()`.

`prospicio_pricing::returns` (#480), return metrics on simulated cash
flows, one simulation a row:

- `npv` and `npv_rows` at an annual rate; `irr` and `irr_rows`, the
  internal rate of return with `found`, the value at the rate and the
  number of roots the search saw (`Irr`, searched as `IrrOptions` says);
  `TimeUnit` for month or day offsets.
- `moic`, `leverage` and `TieredFee`.

Bindings: Python `npv`, `irr` (an `IrrResult` of lists), `moic`,
`leverage`, `tiered_fee` in `prospicio.pricing`; R `npv()`, `irr()` (a data
frame), `moic()`, `leverage()`, `tiered_fee()`. The metrics are arithmetic
on flows and times; they do not depend on how the flows were built, and
move with #479's cash-flow engine if that lands in another crate.

## Decisions

1. **Times run from inception; ages are read as times.** A paid chain
   ladder's ages run from the start of the origin period, which for a
   losses-occurring treaty incepting with the accident year is the
   inception. For risks attaching, or an inception mid-year, shift the
   pattern.
2. **The last factor to ultimate must be 1.** A tail beyond the last age
   goes in as one more age; the remainder is not placed silently.
3. **Annual effective rates**, `(1 + rate)^-t`.
4. **Capital runs off with the outstanding share.** Capital `C = a − P`,
   with `a` the one-year risk measure, is held in year `k` in proportion to
   the share outstanding at its start, and its cost `r C outstanding(k)` is
   charged at the start of the year, discounted, as the one-period price
   charges `r C` at inception. The premium solves
   `P = E[X] d + r (a − P) A`, with `d` the discount factor and `A` the
   discounted capital years. Year 0 counts in full, so a pattern that pays
   everything within the first year has `A = 1`, and at a zero rate the
   one-period price comes back exactly; the tests check this identity.
5. **The cost of capital is a spread over the discount rate.** Investment
   income on the premium is what discounting the losses represents; `r`
   is the extra return the capital asks for.
6. **A distortion price is discounted with the losses**, `P = ρ(X) d`; it
   holds no capital over time.
7. **Simulations are discounted by scaling.** The pattern does not depend
   on the size of the loss, so every simulated year's present value is its
   loss times `d`; there is no per-draw API.

8. **Return metrics discount as the payout does** (#480): `(1 + r)^-t`
   with `t` in years. Month offsets are twelfths of a year and day
   offsets actual/365 (`TimeUnit`); the two give different rates on the
   same monthly flows.
9. **The IRR search is the reference model's, as the default.** The value
   of the flows on 400 rates from −0.95 to 1; of its sign changes and
   exact zeros, the bracket whose midpoint is nearest 0; 8 bisections of
   it, then 3 Newton steps clipped to it. The procedure is the parity
   target, so the defaults are its numbers and every one is an option.
10. **A missing root is a flag, not an error.** Flows of one sign, or with
    their root outside the range, give `found = false` and NaN; flows with
    several roots give the one nearest 0 and `sign_changes` above 1, so a
    caller can see both cases.
11. **MOIC and leverage divide by the capital funded plus the opening
    capital**, which must be positive: `1 + released / C` and
    `rate × premium / C`. The leverage rate has no default.
12. **A tiered fee is piecewise linear above a floor**:
    `Σ_i s_i (clip(m, b_i, b_{i+1}) − b_i)`, the last tier unbounded. The
    tiers are the caller's; none ships with the library.

## Validation

- Closed forms: an annuity's discount factor, the capital years of an
  even run-off, the outstanding share.
- The reduction to the one-period price for one payment within the first
  year at a zero rate, and the premium equation `P = E d + r (a − P) A`
  checked term by term at a positive rate.
- Against the Reserving lane's chain ladder on GenIns paid: the pattern from
  `cdf` has paid, by each origin's latest age, the fit's latest over its
  ultimate (`validation/tests/pricing.rs`, module `payment_pattern`).
- Return metrics (#480): an IRR in closed form (−100 now, 121 in two
  years, 10%); two roots, 10% and 20%, where the one nearest 0 is kept and
  a search from 15% finds the other; flows of one sign and a root outside
  the range flagged; an exact zero on the grid; the rows call against one
  row at a time; monthly against actual/365 in closed form; a three-tier
  fee by hand (unit tests in `returns.rs`, and the same cases in Python
  and R).
