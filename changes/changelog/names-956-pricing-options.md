### Distributions and probability

- The capital allocation method and the piecewise Pareto's truncation kind take the same names in Python, R and Excel, from `AllocationMethod::NAMES` and `Truncation::NAMES`: Python and R now accept Excel's `natural` (the Euler allocation), `last_piece` and `whole_distribution`. No spelling is removed.

### Aggregate and reinsurance

- The P&L face takes the same names in Python, R and Excel, from `Face::NAMES`: Python and R now accept Excel's `top-down` and `bottom-up`, in any case with `_` or a space for `-`. No spelling is removed.

### Pricing

- The cash-flow bases and options (base, adjustment base, funding rule and source, reference, interest basis, fee basis, release and funding dates), the rate-change rule (`applies`) and the period kind take the same names in Python, R and Excel, from the new `NAMES` tables in `prospicio_pricing`: Python and R now accept Excel's any-case spellings with `_`, `-` or a space as the separator (`Unearned Premium`), `inforce` and `accident`. `Period::new` and `PeriodKind` are new. No spelling is removed.
