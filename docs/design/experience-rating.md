# Design note: experience rating from large losses

Status: **In progress** (PRs 1 and 2 of 3: trend, as-if layer loss, burning cost;
per-claim development and count IBNR) ·
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

## What exists (PRs 1 and 2)

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

- PR 2, per-claim development: `DevelopmentPattern::new(ages, factors)`
  holds factors to ultimate by age, and `DevelopmentPattern::from_history`
  estimates one from each claim's valuations (`ClaimValuation { id, age,
  amount }`). `Experience::with_valuation(date)` and
  `prepare_developed(&Trend, &Development)` develop each known loss to
  ultimate (IBNER) before trending, by `Development::Pattern` at its age at
  the valuation date or by `Development::Factors`, one per loss.
  `PreparedLoss` gains `development` and `ultimate`, and `trended` is now
  `ultimate × factor`.
- PR 2, count IBNR: `YearBasis::from_count_pattern(year, premium, &counts,
  valuation)` sets a year's development factor from a count pattern at the
  year's average age, `valuation − (year + ½)`.

Bindings: Python `prospicio.pricing.LargeLossExperience` (with
`valuation=`), `DevelopmentPattern`, `PreparedLosses` (`prepare(...,
development=)`), `BurningCost` (`burning_cost` takes a count pattern as
`development`); R `trend_large_losses()` (with `valuation` and
`development`), `burning_cost()`, `development_pattern()`,
`development_pattern_from_history()` and `pattern_factor()`.

## Decisions

1. **Development by year, or claim by claim.** Clark develops each year's
   layer total with an excess factor. PR 2 adds per-claim development
   (IBNER), which changes which losses reach the layer and how far. The
   two are alternatives: after per-claim development, a year's factor in
   `YearBasis` is the count IBNR factor only, never an excess factor that
   already holds IBNER (Clark's "smaller development factor … to include
   only 'true IBNR' claims").
2. **Losses below `complete_above` are dropped, not adjusted.** Dropping is
   the standard treatment and needs no assumption about the missing
   losses' sizes.
3. **The total is a ratio of sums.** Clark's 12.4% is the developed losses
   over the premium, summed; the equally weighted mean is reported beside
   it.
4. **Dates are actual days over 365.25** in the validation, which reproduces
   all fourteen of Clark's printed factors to three decimals; the library
   itself takes decimal years and leaves the day count to the caller.
5. **Two shapes of development, as for trend.** A pattern read at each
   loss's age, or a factor per loss (as an exhibit prints them or another
   tool gives them).
6. **A pattern holds at its ends.** Between ages it interpolates linearly;
   before the first age it takes the first factor and after the last the
   last, rather than extrapolate. A pattern should start at the record's
   youngest age and end where development stops.
7. **Only claims seen at both ages enter a link.** `from_history` pairs a
   claim into the factor from age `k` to `k + 1` only when it is valued at
   both with a positive amount at `k`; a claim first seen later is IBNR,
   not IBNER, and would bias the factor up. A link with no pairs is an
   error. With every claim valued at every age up to its latest, the
   pattern is the volume-weighted chain ladder on the summed triangle,
   which the validation checks against `prospicio-reserving`'s
   `ChainLadder`. Clark describes the step but prints no exhibit for it.
8. **Completeness stays on the reported amount.** The record is complete
   for losses reported above the threshold, whatever they develop to, so
   `complete_above` is compared with the reported amount trended. A
   truncated record also biases the IBNER factors estimated from it (small
   claims that grow are missing); the pattern is the user's to choose.
9. **A count pattern is read at the year's mid-point**, `valuation − (year
   + ½)`, the average accident date of a year written evenly.

## Next

- **PR 3:** an indexation (stability) clause that moves the retention and
  limit with an index, with the franchise variant; FX conversion at a
  stated or historical rate.
- #191 (as-if history) builds a record from a premium and treaty history
  as a constructor on top of this.
