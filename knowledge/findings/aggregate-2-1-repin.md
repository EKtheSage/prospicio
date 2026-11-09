---
type: Finding
title: aggregate 2.1.0 against the 1.0.1 references, what moved and why
description: Every parity reference regenerated with aggregate 2.1.0; nearly all values are unchanged to round-off, and the three that moved are aggregate's own fixes, each toward prospicio (its quoted skewness is now its density's unless the grid drops real tail mass, PricingBounds keeps the p -> 1 vertex, Bounds' grid search reaches the exact lower envelope), so no prospicio result changes and the tolerances on layer skewness come down from 5e-7 and 1e-5 to 1e-8; the P&L books annual aad and agglimit terms on an occurrence layer as aggregate 2.1.0 does, to 1e-12.
tags: [aggregate, parity, references, pnl, bounds, skewness]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T12:00:00Z }
verified: { by: process:ci, at: 2026-11-09T12:00:00Z }
sources:
  - id: scripts
    resource: ../validation/scripts/
    title: validation/scripts/aggregate_*.py, every script pinned to aggregate==2.1.0
  - id: tests
    resource: ../validation/tests/aggregate.rs
    title: validation/tests/aggregate.rs, distributions.rs and pricing.rs, the parity tests that read the references
  - id: aggregate
    resource: https://pypi.org/project/aggregate/2.1.0/
    title: aggregate 2.1.0 (released 2026-11-07), spectral.py, bounds.py, _pnl.py, _pnl_builders.py
---

# Finding

* All 27 scripts in `validation/scripts/aggregate_*.py` were run under
  `aggregate` 2.1.0 (Python 3.13, `uv run --with aggregate==2.1.0`). The
  references of 1.0.1 (and of 1.2.0 for the mixing, sharpen and common
  shock checks) and of 2.1.0 agree to round-off (1e-16 to 7e-13 relative)
  except in three places, all of them fixes in `aggregate`.[^scripts]
* **Quoted skewness.** 1.0.1's `*_skew` of a layer or tower placed the
  density's rounding deficit (`1 - sum(p)`, 1e-15 to 1e-14) one bucket past
  the grid, so it differed from the density's own skewness by up to 8.1e-5
  (`re_pricing_aggregate.csv`) and 1.35e-6 (`reinsurance_aggregate.csv`,
  an aggregate cover of skewness 32). 2.1.0 (`xsden_to_mwrangler`) treats a
  deficit of 1e-12 or less as rounding and leaves it out, so its quoted
  skewness is the density's to about 1e-11; a real deficit (a grid built
  with `normalize=False` that drops tail mass, 5.7e-10 for the stop loss at
  2,800 of `re_pricing_aggregate.csv`) is still placed past the grid, 8.6e-7
  off there. `examples/reinsurance_pricing.ipynb` and its R twin say so now.
  prospicio already matched the density's: against 2.1.0 the largest
  difference is 3.3e-9 on an occurrence layer and 1.7e-10 on an aggregate
  cover, so `aggregate_reinsurance.py`'s tolerances are now 1e-8 on both
  (they were 5e-7 and 1e-5). The `_skew_central` rows moved by 6e-13.
* **`PricingBounds`.** Before 1.52 it dropped the vertex `p -> 1` when X's
  top atom is larger than Y's, so InsCo's X3 capped at 65 at `P = 65`
  printed 52 where 64 is attained, and the envelope below X's top atom
  (s < 0.1) was too low (0.043 against 0.148 at s = 0.02). The script held
  a brute force over the BiTVaRs for six rows; under 2.1.0 every row is
  `aggregate`'s own, equal to the brute force, and the envelope reference
  gains s = 0.02, 0.05 and 0.08, which prospicio matches to 1e-12.
* **`Bounds`' grid search.** `min_envelope` at s = 0.1, 0.15 and 0.2 moved
  down by 9.3%, 3.3% and 1.1% (0.0756 to 0.0686 at 0.1), to the exact
  lower envelope at 0.1 and 0.15. The test's check (prospicio's lower at or
  below it, within 0.05) is unchanged and passes.
* **API changes met by the scripts.** `Portfolio.price(100, d)` now refuses
  `p > 1` and wants `a=100`; the distortion frame's `gini_p` column is
  `gini`. `cll` is off the default distortion list but still built by name.
  Nothing else in the scripts needed a change.
* **Annual terms on an occurrence cover in the P&L.** `aggregate` 2.1.0
  books `occurrence net of 2 xs 2 aad 1 agglimit 3 deposit 2` as an
  `AnnualLayerTerms` on the joint of the gross loss and the layer's
  unlimited annual cession `R`, the recovery `min(max(R - aad, 0),
  agglimit)`. prospicio's ledger strips the annual terms per claim and
  applies them to `R` in the same way: six new cases (aad, agglimit and
  both, walked and consolidated; 990 rows) match to 1e-12.[^tests]
* `Layer::franchise` has its Python (`with_franchise`) and R bindings since
  #591, so the issue's item on it needed nothing.
* The guide for `aggregate` users quotes the 1.2.0 Twenty Minutes session;
  2.1.0 prints every one of those numbers again (the casualty book's
  quantiles, the mixture, the ILF table, the tower, the 99.6% VaR and the
  pentagon).

[^scripts]: validation/scripts/aggregate_*.py, run 2026-11-09.
[^tests]: validation/tests/aggregate.rs, `pnl_reinstatements_match_aggregate`.
