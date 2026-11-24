# Aggregate lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-aggregate/`, `crates/prospicio-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md`, `docs/design/credibility.md`, `docs/design/ratemaking.md`, `docs/design/experience-rating.md`, `docs/design/time-value.md`, `docs/design/cat-inputs.md`, `docs/design/pnl.md`, `docs/design/casualty-exposure.md`, `docs/design/reinsurer-default.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/aggregate.md](done/aggregate.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [970-other-loops] the rest of #970: the other simulation loops it lists, each to start from a benchmark as the tower did: `simulate_events` and `simulate_events_weighted` collecting a vector per year before flattening, `EventLossTable::simulate`, `Hawkes::annual_counts` and `event_set`, and `frontier_events`, `frontier_annual` and `frontier_grid` in `prospicio-pricing` (which has no rayon yet); the first three draw from `StreamRng` by year, so a parallel split keeps their seeded numbers
- [fft-consolidate] #973 the remaining FFT sites onto `prospicio_math::fft`, one small PR each: `bivariate` (the 2-D transform and its aliasing mass, which is the long marginal buffer's second half), `renewal` and `ruin` in `prospicio-aggregate`, `natural::Portfolio` in `prospicio-pricing`, `chf` in `prospicio-prob` (after #972, which edits that crate) and `prospicio-bayes`; then drop `rustfft` from those crates' manifests
- [fft-other-crates] #973 the FFT sites outside `prospicio-aggregate` onto `prospicio_math::fft`, one small PR each: `natural::Portfolio` in `prospicio-pricing`, `prospicio-bayes`, and `chf` in `prospicio-prob` (after #972, which edits that crate); each drops `rustfft` from its crate's manifest
- [reinsurer-default-core] #980 `Panel` and `Reinsurer` in `prospicio-aggregate`: each year's reinsurer defaults (one Gaussian factor ranked on the gross loss, `rho = 0` independent) applied to a tower's one-year result, giving `credit_loss` and `recovered` by layer and `net_after_default`, and `by_reinsurer`; same bits on any number of threads (`reinsurer-default.md`, PR 2; after #1053)
- [reinsurer-default-validation] #980 the reference generator `aggregate_reinsurer_default.py` (closed forms and quadrature, no simulation), `reinsurer_default.csv` and `validation/tests/aggregate_reinsurer_default.rs` (`reinsurer-default.md`, PR 3)
- [reinsurer-default-solvency2] #980 the Solvency II counterparty default module for type 1 exposures as pure functions, with their Python and R bindings: probability of default by credit quality step and by solvency ratio, the reinsurance loss given default, `V_inter`, `V_intra` and `SCR_def,1` (Arts 192, 199 to 201; `reinsurer-default.md`, PR 4)
- [reinsurer-default-bindings] #980 the panel and the Solvency II helpers in Python and R, tests in new files, the allow-list entries removed (`reinsurer-default.md`, PR 5)
- [reinsurer-default-term] #980 reinsurer default over `apply_term`'s paths (`Panel::apply_term`), a default final for the rest of the path and collateral used up, in Rust, Python and R; the Done bullet `#980` (`reinsurer-default.md`, PR 6)
- [reinsurer-default-payout] default part way through a long payout: cumulative probabilities of default over a recovery pattern (`reinsurer-default.md` decision (e)); after #980
- [reinsurer-default-setoff] premiums and commissions owed to a defaulted reinsurer set off against its recoveries (`reinsurer-default.md` decision (c)); after #980
- [reinsurer-default-structural] a structural default model: a reinsurer fails when its share of an industry loss exceeds its capital (`reinsurer-default.md` decision (g)); after #980
- [reinsurer-default-tail] a systematic factor with tail dependence (t or Clayton) for wrong-way risk (`reinsurer-default.md` decision (g)); after #980
- [reinsurer-default-pnl] a reinsurance bad debt row in the P&L ledger (`pnl.md`) from the panel's expected credit loss; after #980
- [reinsurer-default-json] the panel in the tower's JSON, so the JavaScript and Excel front ends can use it (`serial.rs`); after #980
- [retro-charge] #981 `retro::InsuranceCharge`: the insurance charge `phi(r)` and savings `psi(r)` of an aggregate `Severity` (a `Grid` from `fft` or `panjer`, a fitted law) or a sample, with Python and R bindings and the closed-form tests (`primary-rating.md`, PR 2); waits for the user's acceptance of the note's decisions (a) to (j)
- [retro-limited] #981 Table L and the limited Table M: the charge under a per-occurrence limit in both conventions (`primary-rating.md`, PR 3)
- [retro-balance] #981 `RetroPlan::balance`: the entry ratios and the basic premium that balance a retrospective plan to the guaranteed-cost premium, and `premium` (`primary-rating.md`, PR 4)
- [earned-daily] #981 `earned::Date`, `Policy` and `earned_exposure`: written, earned, unearned and in-force exposure or premium by calendar or policy window from policy dates, exact days, with Werner and Modlin's chapter 4 exhibit (`primary-rating.md`, PR 5)
- [earned-endorsements] #981 mid-term endorsements, cancellations and the twenty-fourths method (`primary-rating.md`, PR 6)
- [emod-split] #981 the workers compensation split plan: primary and excess losses, D-ratios, the weighting value and the ballast as inputs, the exhibit and the modification (`primary-rating.md`, PR 7)
- [emod-ratio] #981 the general liability experience ratio plan, and the Done bullet `#981` (`primary-rating.md`, PR 8)
- [earned-short-rate] short-rate cancellation and audit premium adjustments (`primary-rating.md`, decision (d)); after #981
- [retro-deductible-dividend] large deductible and dividend plans from the charge and savings (`primary-rating.md`); after #981
