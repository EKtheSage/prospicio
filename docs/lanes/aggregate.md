# Aggregate lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-aggregate/`, `crates/prospicio-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md`, `docs/design/credibility.md`, `docs/design/ratemaking.md`, `docs/design/experience-rating.md`, `docs/design/time-value.md`, `docs/design/cat-inputs.md`, `docs/design/pnl.md`, `docs/design/casualty-exposure.md`, `docs/design/reinsurer-default.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/aggregate.md](done/aggregate.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [970-other-loops] the rest of #970: the other simulation loops it lists, each to start from a benchmark as the tower did: `simulate_events` and `simulate_events_weighted` collecting a vector per year before flattening, `EventLossTable::simulate`, `Hawkes::annual_counts` and `event_set`, and `frontier_events`, `frontier_annual` and `frontier_grid` in `prospicio-pricing` (which has no rayon yet); the first three draw from `StreamRng` by year, so a parallel split keeps their seeded numbers
- [reinsurer-default-payout] default part way through a long payout: cumulative probabilities of default over a recovery pattern (`reinsurer-default.md` decision (e)); after #980
- [reinsurer-default-setoff] premiums and commissions owed to a defaulted reinsurer set off against its recoveries (`reinsurer-default.md` decision (c)); after #980
- [reinsurer-default-structural] a structural default model: a reinsurer fails when its share of an industry loss exceeds its capital (`reinsurer-default.md` decision (g)); after #980
- [reinsurer-default-tail] a systematic factor with tail dependence (t or Clayton) for wrong-way risk (`reinsurer-default.md` decision (g)); after #980
- [reinsurer-default-pnl] a reinsurance bad debt row in the P&L ledger (`pnl.md`) from the panel's expected credit loss; after #980
- [reinsurer-default-json] the panel in the tower's JSON, so the JavaScript and Excel front ends can use it (`serial.rs`); after #980
- [emod-split] #981 the workers compensation split plan: primary and excess losses, D-ratios, the weighting value and the ballast as inputs, the exhibit and the modification (`primary-rating.md`, PR 7)
- [emod-ratio] #981 the general liability experience ratio plan, and the Done bullet `#981` (`primary-rating.md`, PR 8)
- [earned-short-rate] short-rate cancellation and audit premium adjustments (`primary-rating.md`, decision (d)); after #981
- [retro-deductible-dividend] large deductible and dividend plans from the charge and savings (`primary-rating.md`); after #981
- [method-docs-next] #991 next families: the Mack bootstrap's and the other credibility models' binding docs link to their Rust reference instead of restating the method (`docs/design/bindings.md`, "Method documentation")
- [method-docs-credibility] #991 the other credibility models (Buhlmann-Straub, Hachemeister): their binding docs link to the Rust reference instead of restating the method, and the Done bullet `#991` (`docs/design/bindings.md`, "Method documentation")
