# Aggregate lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-aggregate/`, `crates/prospicio-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md`, `docs/design/credibility.md`, `docs/design/ratemaking.md`, `docs/design/experience-rating.md`, `docs/design/time-value.md`, `docs/design/cat-inputs.md`, `docs/design/pnl.md`, `docs/design/casualty-exposure.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/aggregate.md](done/aggregate.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [970-other-loops] the rest of #970: the other simulation loops it lists, each to start from a benchmark as the tower did: `simulate_events` and `simulate_events_weighted` collecting a vector per year before flattening, `EventLossTable::simulate`, `Hawkes::annual_counts` and `event_set`, and `frontier_events`, `frontier_annual` and `frontier_grid` in `prospicio-pricing` (which has no rayon yet); the first three draw from `StreamRng` by year, so a parallel split keeps their seeded numbers
