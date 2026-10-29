### Aggregate and reinsurance

- Contract years for dated events: `EventSet::contract_years(start, PartialYears)` reads the simulated years as one timeline and regroups the losses into contract years starting at `start`, a fraction of the year, each loss's time rebased and its sum insured and labels kept, the partial first and last years dropped by default or kept (`cat-inputs.md` decision 16). Python `EventSet.contract_years(start, partial=)`, R `contract_years()` (#586).
