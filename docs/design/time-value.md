# Design note: the time value of money in a price

Status: **Implemented** · Depends on: `risk_load` prices · Lane: Aggregate ·
Issue: #195

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

## Validation

- Closed forms: an annuity's discount factor, the capital years of an
  even run-off, the outstanding share.
- The reduction to the one-period price for one payment within the first
  year at a zero rate, and the premium equation `P = E d + r (a − P) A`
  checked term by term at a positive rate.
- Against the Reserving lane's chain ladder on GenIns paid: the pattern from
  `cdf` has paid, by each origin's latest age, the fit's latest over its
  ultimate (`validation/tests/pricing.rs`, module `payment_pattern`).
