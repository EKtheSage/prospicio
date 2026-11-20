### Pricing

- `irr` and `irr_rows` (and the forward and since-inception IRRs of a valuation) no longer report a found root for flows that are all zero, or that sum to 0 at every time: such a row is worth 0 at every rate, so it now gives `found = false`, NaN rate and NPV, and no sign changes. Summaries that counted a run-off deal's -0.19% as a return now leave it out (#941, #996).
