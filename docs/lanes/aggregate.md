# Aggregate lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-aggregate/`, `crates/prospicio-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md`, `docs/design/credibility.md`, `docs/design/ratemaking.md`, `docs/design/experience-rating.md`, `docs/design/time-value.md`, `docs/design/cat-inputs.md`, `docs/design/pnl.md`, `docs/design/casualty-exposure.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/aggregate.md](done/aggregate.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next
- [fft-consolidate] #973 the remaining FFT sites onto `prospicio_math::fft`, one small PR each: `bivariate` (the 2-D transform and its aliasing mass, which is the long marginal buffer's second half), `renewal` and `ruin` in `prospicio-aggregate`, `natural::Portfolio` in `prospicio-pricing`, `chf` in `prospicio-prob` (after #972, which edits that crate) and `prospicio-bayes`; then drop `rustfft` from those crates' manifests
