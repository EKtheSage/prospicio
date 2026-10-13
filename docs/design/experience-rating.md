# Design note: experience rating from large losses

Status: **Implemented** (PR 1: trend, as-if layer loss, burning cost; PR 2:
per-claim development and count IBNR; PR 3: index clause and FX; #191: as-if
history) ·
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

## What exists

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

- PR 3, FX: `Experience::with_fx(Fx::Rate(r) | Fx::PerLoss(rates))`
  converts each loss to the pricing currency before development and
  trend; `PreparedLoss` gains `fx`.
- PR 3, index clause: `IndexClause::new(franchise, IndexKind::Full |
  IndexKind::Severe)` and `Prepared::burning_cost_indexed(&towers, &years,
  &clause, &ratios)`, with a ratio per loss of the index at its payment to
  the index at the treaty's base date.

- #191, as-if history: `Prepared::as_if(&historical, &years, &proposed)`
  sets each year's losses as reported under the program in force that
  year (`historical[k]`, which may differ in layers from year to year)
  beside the year's trended losses under the proposed tower (`AsIf`,
  `AsIfYear`), to reconcile with the cedant's recoveries.

Bindings: Python `prospicio.pricing.LargeLossExperience` (with
`valuation=` and `fx=`), `IndexClause` (`burning_cost(..., index=,
ratios=)`), `DevelopmentPattern`, `PreparedLosses` (`prepare(...,
development=)`), `BurningCost` (`burning_cost` takes a count pattern as
`development`); R `trend_large_losses()` (with `valuation` and
`development`), `burning_cost()`, `development_pattern()`,
`development_pattern_from_history()` and `pattern_factor()`, with `fx`
in `trend_large_losses()` and `index_clause()` with `index` and `ratios`
in `burning_cost()`; Python `PreparedLosses.as_if` (`AsIf`) and R
`as_if_history()`.

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
10. **FX first, completeness in the reported currency.** Amounts are
    converted before development and trend, so a trend `rate` is the
    pricing currency's. With a rate per loss there is no single
    `complete_above` in the pricing currency, so the threshold test stays
    in the currency each loss was reported in.
11. **The index clause covers per-occurrence terms, exactly.** With every
    term per occurrence (attachment, limit, share, inuring stages), terms
    moved by `a` cede `a × ceded(x / a)`, so each loss goes through the
    tower at its own adjustment with no approximation. Annual terms (an
    aggregate deductible or limit, a corridor, and so limited
    reinstatements) span losses with different ratios; rather than reset
    them per loss, a tower with them is an error. A surplus treaty is too.
    A treaty that indexes its annual terms as well is left for when one is
    needed.
12. **Franchise rules.** `Full`: nothing moves while `|r − 1|` is within
    the franchise; beyond it the terms move by all of `r`, up or down.
    `Severe` (a severe inflation clause): the terms move only by the rise
    beyond the franchise, `r / (1 + f)`, and never down. The caller gives
    each loss's ratio, from an index and its payment date (or a payment
    pattern's average), as for trend and development factors. Clark
    prints no index clause example; the validation checks the clause
    against a tower built with the moved terms.
13. **As-if history compares like with like on each side.** The historical
    side takes every loss of the year as reported (its own currency,
    untrended, undeveloped, threshold or not), because that is what the
    cedant recovered and reports; the proposed side is exactly the
    burning cost's layer loss before development by year. Reconciling
    the historical side with the cedant's recoveries checks the record
    before it is trended.

## Next

