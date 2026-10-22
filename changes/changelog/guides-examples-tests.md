### Examples

- The notebook tests check the numbers and claims the notebooks' prose states
  (`reinsurance_pricing`, the "Reading the result" bullets of `one_year_view`)
  and run the chart cells under matplotlib's Agg backend; two stale skewness
  figures in `reinsurance_pricing` and its R twin are corrected. A notebook
  opened outside the repository now raises a `FileNotFoundError` that says
  where to run it, not a bare `StopIteration` (#269, #531).
