# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [chf-tilting] #779's second PR: exponential tilting of the characteristic-function engine with the totally skewed stable law (the tilt that balances the damped wrap against the amplified round-off, the bucket cap, the mass outside the window into the first bucket, a left tail by reflection; aggregate 2.1.0 parity)
- [chf-laws-rest] #780 the other laws through the characteristic-function engine (normal inverse Gaussian, variance gamma, tempered stable, Tweedie with p <= 0 or p > 2) and the stable law with |beta| < 1 under the tilt, split into its two totally skewed halves; `Stable` as a `Distribution` with its algebra
