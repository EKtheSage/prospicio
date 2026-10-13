# Design note: experience rating from large losses

Status: **In progress** (PR 1 of 3: trend, as-if layer loss, burning cost) ·
Depends on: `prospicio-aggregate` towers · Lane: Aggregate · Issue: #190

## Goal

A reinsurance submission brings a list of individual large losses, each
with an accident date and amounts by valuation. Experience rating puts
each loss on the prospective basis, applies the treaty to each year's
losses as if it had been in force, develops the layer totals and divides
by the on-level subject premium: the burning cost. Before this note the
user wrote that loop by hand around `trend_factor` and `Tower::ceded`.

The reference is Clark's *Basics of Reinsurance Pricing* (CAS study note,
2014), the section on per-risk experience rating, whose worked example the
validation reproduces.

## What exists (PR 1)

`prospicio_pricing::experience`:

- `LargeLoss { id, accident, amount }`, with the accident date a decimal
  year as elsewhere in `ratemaking`; `Experience::new(losses)` and
  `with_threshold(t)` for the record's reporting threshold.
- `Experience::prepare(&Trend)`: `Trend::Rate { rate, to }` trends each loss
  by `(1 + rate)^(to − accident)`; `Trend::Factors` takes a factor per loss
  (from an index, or as an exhibit prints them). The result, `Prepared`,
  holds each loss with its accident year (the floor of the date), factor
  and trended amount.
- `Prepared::complete_above()`: the threshold times the largest trend
  factor. A record reported above `T` in nominal terms is short, in its
  older years, of losses that trend above `T`; losses trended below this
  level are left out of the burning cost, so every year is complete on the
  same basis.
- `Prepared::burning_cost(&[Tower], &[YearBasis])`: each year's tower
  (the terms in force that year, or the prospective tower repeated) applied
  to the year's trended losses with `Tower::ceded`, annual terms included;
  the layer totals developed by the year's excess development factor and
  divided by its premium. `BurningCost` holds the rows (`YearCost`) and per
  layer the premium-weighted `loss_cost` (developed losses over premium,
  summed, Clark's total) and the equally weighted `mean_loss_cost`.

Bindings: Python `prospicio.pricing.LargeLossExperience`, `PreparedLosses`,
`BurningCost`; R `trend_large_losses()` and `burning_cost()`.

## Decisions

1. **Development by year in PR 1.** Clark develops each year's layer total
   with an excess factor; per-claim development (IBNER) waits for PR 2,
   where it changes which losses reach the layer.
2. **Losses below `complete_above` are dropped, not adjusted.** Dropping is
   the standard treatment and needs no assumption about the missing
   losses' sizes.
3. **The total is a ratio of sums.** Clark's 12.4% is the developed losses
   over the premium, summed; the equally weighted mean is reported beside
   it.
4. **Dates are actual days over 365.25** in the validation, which reproduces
   all fourteen of Clark's printed factors to three decimals; the library
   itself takes decimal years and leaves the day count to the caller.

## Next

- **PR 2:** per-claim development (IBNER) from factors by age or a
  large-loss triangle, applied before the layer; count IBNR for
  late-reported claims.
- **PR 3:** an indexation (stability) clause that moves the retention and
  limit with an index, with the franchise variant; FX conversion at a
  stated or historical rate.
- #191 (as-if history) builds a record from a premium and treaty history
  as a constructor on top of this.
