### Reserving

- A triangle table with many segments over a wide grid (a nearly unique key
  such as a claim id), a latest-valuation table with a mistyped origin, and
  an Excel sheet with a stray cell far from the table now return an error
  instead of asking for tens of gigabytes and aborting the process (#951,
  #1019).

### Aggregate and reinsurance

- `YearLossTable::with_n_years` refuses a year count far above the rows,
  as `YearLossTable::new` does (#951, #1019).

### Distributions and probability

- `ChfDistribution::to_grid` refuses a window too far above 0 to pad, and
  `ks_bootstrap` refuses weights summing past 2^24, instead of aborting on
  the allocation (#951, #1019).
