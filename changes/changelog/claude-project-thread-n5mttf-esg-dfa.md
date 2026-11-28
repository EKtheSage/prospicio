### Aggregate and reinsurance

- `prospicio_aggregate::dfa`: a multi-year projection of one company
  along the ESG's scenarios. `Projection` takes the opening assets and
  reserves, each year's premium and expenses, simulated losses by path and
  year scaled by the path's claims index, a payout pattern, and cash and
  bond buckets that earn the path's rates; `ProjectionPaths` gives premium,
  losses, reserve development, investment income, the result, assets,
  reserves and surplus by path and year, the probability of ruin within a
  horizon and of falling below a required capital, with standard errors.
  Rust only for now (#983).
