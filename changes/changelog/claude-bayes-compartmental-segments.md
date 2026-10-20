### Reserving

- Gesmann and Morris' compartmental model over segments and in the diagonal
  backtest (behind the `bayes` feature). `Compartmental::fit_segments`
  fits each segment of a keyed triangle on its own seeds (so `fit`'s draws
  are unchanged) into joint distributions of the reserves and of the
  future paid and outstanding cells, the segments independent or
  rank-correlated by a Spearman matrix; the rank correlation reorders each
  segment's reserves, paid cells and outstanding cells together, and
  `Synchronized` is an error. `Compartmental` gains a `dependence`
  setting. `CompartmentalCandidate` puts the model in the diagonal
  backtest next to the ODP GLM, `BayesOdp` and `Meyers`, and
  `outstanding_backtest` scores the held-out outstanding diagonals with the
  same metrics (Rust only). In Python `Compartmental(..., dependence=,
  spearman=)` and `Compartmental.fit_segments(triangle, paid, premium, *,
  outstanding=None, incurred=None)`, returning a `CompartmentalFits`; in R
  `compartmental_segments()`, returning a `compartmental_fits`.
