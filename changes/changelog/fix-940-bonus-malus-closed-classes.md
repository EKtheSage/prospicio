### Pricing

- `BonusMalus::stationary_at` and everything built on it (`efficiency`,
  `mean_relativity`, `stationary`, `optimal_relativities`) now return an
  error for a table with more than one closed class of levels, which has no
  unique stationary distribution. Before, round-off could let the solve
  return one of the many solutions as if it were the answer (#940, #997).
