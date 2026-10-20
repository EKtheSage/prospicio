### Reserving

- The one-year view of `Compartmental` over segments,
  `Compartmental::one_year_segments` (behind the `bayes` feature): each
  segment of a triangle is viewed as `Compartmental::one_year` views a
  single-segment triangle (its own premium column and outstanding cells,
  model 1 moment-matching by default and model 2 reweighting only with the
  warning that it is approximate, at most 2,000 simulations a segment,
  `max_refits` a budget of each segment) on the seeds of `fit_segments`, so
  a one-segment triangle's view is `one_year` on segment 0's streams. The
  segments' claims development results are joined by row into one joint
  result as `Compartmental::dependence` says: `Independent` as drawn,
  `RankCorrelation` by Iman-Conover on their totals (as `BayesOdp` and
  `Meyers` do); `Synchronized` is an error. The result is a
  `CompartmentalOneYearFits` (new): per-segment views, `to_long`, `totals`
  and `segment`. In Python `Compartmental.one_year_segments(triangle, paid,
  premium, *, outstanding=None, incurred=None, k_threshold=0.7, refit=None,
  max_refits=None, n_sims=None)`, returning a `CompartmentalOneYearFits`;
  in R `compartmental_one_year_segments()` with
  `compartmental_one_year()`'s arguments and `dependence` and `spearman`,
  returning a `compartmental_one_year_fits`.
