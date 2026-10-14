### Distributions and probability

- A copula's distribution function, `Copula::cdf`: the Archimedean
  copulas' in any dimension (with `ArchimedeanCopula::inverse_generator`)
  and the Gaussian copula's in two, from a new bivariate normal cdf
  (`prospicio_math::special::bivariate_norm_cdf`, Genz's BVND, about
  `1e-15`); an error for the t copula. Python `cdf` on the copula classes;
  R `copula_cdf()` (#234).
- A distribution on finitely many values of either sign and any spacing,
  `prospicio_prob::Discrete`, for the rows of a P&L: equal values merged,
  the lower and upper quantiles, TVaR and the return-period table as
  `aggregate` 1.0.1's `GridDistribution`, `pmf` and skewness. Python
  `Discrete`; R `discrete_distribution()`, `quantile_upper()`,
  `discrete_skewness()` and `return_periods()` (#241).
- A severity under a mix of policy limits, `prospicio_prob::PolicyLimits`
  (#213): `min(X, L)` with each limit's weight, exact layer moments from
  the base severity's and an atom at each finite limit, so it discretizes,
  compounds and saves as JSON like any severity. Python `PolicyLimits`; R
  `policy_limits_distribution()`.
- Standard errors of the Pareto fits from the observed information
  (#213): `Pareto::fit_with_standard_error` (truncated or not) and
  `PiecewisePareto::fit_with_standard_errors` (uncorrelated by piece).
  Python `Pareto.fit_with_standard_error`,
  `PiecewisePareto.fit_with_standard_errors`; R `pareto_standard_error()`,
  `piecewise_pareto_standard_errors()`.
- Maximum likelihood fits of the lognormal, gamma, Weibull and Burr to
  large losses with reporting thresholds (left truncation), policy limits
  (censoring) and weights (#194):
  `prospicio_prob::severity_fit::fit_severity`, returning the
  distribution, parameters, standard errors and covariance,
  log-likelihood, AIC and BIC. Python `fit_severity` and `SeverityFit`; R
  `severity_fit()`.
- Goodness of fit for large losses with reporting thresholds and
  censoring (#194): the product-limit (Kaplan–Meier) estimator with left
  truncation, `severity_fit::product_limit`, and
  `severity_fit::goodness_of_fit`, which sets any severity against it
  (P-P, Q-Q and the largest distance). Python `product_limit`,
  `goodness_of_fit`; R `product_limit()`, `goodness_of_fit()`.

### Aggregate and reinsurance

- Two lines hit by the same events: their joint annual loss by a 2-D
  compound FFT from a joint per-event severity
  (`prospicio_aggregate::bivariate::{fft_bivariate, JointGrid}`, with each
  line's marginal, the total, mixed moments and the correlation), and
  `aggregate`'s `clash` model, given by the expected events hitting line
  A only, line B only and both (`Clash`), matching `aggregate` 1.0.1.
  Python `fft_bivariate`, `JointGrid`, `Clash`; R `fft_bivariate()`,
  `clash_model()`, `clash_severity()`, `clash_aggregate()` (#233).
- Two lines whose losses in an event are coupled by a copula: the joint
  per-event severity as the copula's rectangle probabilities over the two
  marginals (`JointGrid::from_copula`, with `bivariate::triggered` for a
  line an event hits with some probability), for `fft_bivariate`,
  matching `aggregate` 1.0.1's copula `bivariate`. Python
  `JointGrid.from_copula`; R `copula_severity()` (#235).
- Renewal claim counts: the number of claims by a horizon when the waits
  between claims follow any law (`prospicio_aggregate::renewal::renewal_count`,
  a `Wait` that is a distribution or discrete waits, possibly defective,
  with waits of zero arriving in clusters), on `aggregate`'s wait grid or
  a given one, matching `aggregate` 1.0.1's `years ... wait ...`
  frequency. Python `Count.renewal`; R `renewal_count()` (#236).
- Eventual ruin for renewal claims: `ψ(u)` when the waits between claims
  follow any law and the premium comes in at `(1 + ρ)` times the expected
  loss rate, by the cepstral Wiener–Hopf factorization of the claim walk
  (`prospicio_aggregate::ruin::wiener_hopf`, returning `RenewalRuin`),
  matching `aggregate` 1.0.1's `wiener_hopf`. Python `wiener_hopf`; R
  `wiener_hopf()` (#237).
- The P&L ledger, as `aggregate` 1.0.1's `PnL`
  (`prospicio_aggregate::pnl::{Ledger, Group, Leg}`): signed legs in books
  that sell or buy cover over a source, every row (legs, side totals,
  results, running nets, spans and net-of blocks, grand rows, the covers'
  total impact) a sum at each atom with its exact distribution, the card,
  and the scenario columns that condition each row on the grand result.
  Python `prospicio.aggregate.Ledger`, `Group`, `Leg`; R `pnl_ledger()`,
  `pnl_group()`, `pnl_leg()` and `ledger_sheet()`, `ledger_card()`,
  `ledger_marginal()`, `ledger_legs()`, `ledger_result()`, `ledger_row()`
  (#242).
- Legs rebucketed on a step (`Leg::step`, with `Ledger::validation`
  comparing the exact and grid means), the ratio exhibit
  (`Ledger::ratios`: premium, loss, expense and result by block, ratios of
  means and means of ratios, shares) and the stack of several ledgers'
  results with their differences (`pnl::stack`), as `aggregate` 1.0.1's
  `bs`, `economic_ratios_df` and `stack_marginal_pnls`. Python
  `Leg(step=)`, `Ledger.validation`, `Ledger.ratios`, `stack`; R
  `pnl_leg(step =)`, `ledger_validation()`, `ledger_ratios()`,
  `ledger_stack()` (#244).
- The ledger of a seller's book and the annual covers it buys, as
  `aggregate` 1.0.1's `pnl` and `xpnl` builders
  (`prospicio_aggregate::pnl::{Book, Expense, Face, derive_premium}`):
  consolidated (the net position over the net loss), walked (one group
  per stage of the tower) or peeled (one group per layer, top down or
  bottom up), with expenses on a fixed, premium or loss basis and
  `derive premium`. `pnl::Source` now keeps probabilities that already
  sum to 1 (to `1e-12`). Python `Book`, `Expense`, `derive_premium`; R
  `pnl_book()`, `pnl_expense()`, `book_ledger()`, `derive_premium()`
  (#245).
- Occurrence covers in the P&L ledger: a joint source of the annual
  gross and occurrence-ceded loss (`pnl::Source::joint`,
  `pnl::occurrence_source` by a 2-D FFT of each claim and its cession,
  `aggregate`'s `occ_bivariate`), legs of both coordinates
  (`pnl::Leg::of2`), and `Book::occurrence_ledger`: the walk through the
  occurrence covers and then aggregate covers on their net, the peel and
  the consolidated face, matching `aggregate` 1.0.1. Python
  `occurrence_source`, `Book.occurrence_ledger`; R `occurrence_source()`,
  `book_occurrence_ledger()` (#246).
- Variable terms in the P&L builders: a swing-rated premium, a
  sliding-scale or profit commission and a loss corridor booked as legs
  at each atom from the layer's own terms, on every face; retro rating of
  the book (`Book::retro`); and `pnl::pnl_program`, the premium and each
  layer's deposit from target combined ratios, with `size_premium` and
  `round_premium`, matching `aggregate` 1.0.1. The builders no longer
  refuse loss-sensitive terms (reinstatement premiums still are). Python
  `Book(retro=)`, `pnl_program`, `size_premium`, `round_premium`; R
  `pnl_book(retro =)`, `pnl_program()`, `size_premium()`,
  `round_premium()` (#247).
- Reinstatement premiums in the P&L ledger: an occurrence cover of one
  layer with annual terms and paid reinstatements over the joint of the
  gross loss and its unlimited annual cession, its recovery `A(R)` and
  premium `D + h(R)` varying with the year on every face, and an
  aggregate cover behind it, matching `aggregate` 1.0.1. Annual covers
  now refuse paid reinstatements with a pointer to
  `Book::occurrence_ledger` (#248).
- `pnl::pnl_program` also respells each priced deposit as a rate
  (`Program::rates`, `aggregate`'s `premium_style='rate'`), and
  `pnl::pnl_program_stated` prices an engine that states its own technical
  premium; `ProgramLayer` says its tier. Python `pnl_program_stated` and a
  `rates` entry; R `pnl_program_stated()` and `rates` (#249).
- The P&L's consolidated face books a loss-basis expense at each year's
  gross loss, so its margin is the walk's net margin year by year (the
  mean is unchanged; its spread now includes the expense's).
  `Book::expense_at_expected_loss` keeps the expected-loss booking of
  `aggregate`'s guaranteed-cost builder. Python
  `Book(expense_at_expected_loss=)`; R `pnl_book(expense_at_expected_loss
  =)` (#250).
- The stitched peel of several occurrence layers, `aggregate`'s `xpnl ...
  peel` with two or more occurrence layers: `Book::occurrence_peel` builds
  every row as its own exact marginal (compounds of per-claim maps, and
  pushforwards for aggregate covers behind them) through
  `pnl::Ledger::stitched`; the mean column foots, the other statistics and
  the card are each row's own, with no scenario columns. Python
  `Book.occurrence_peel` and `Ledger.is_stitched`; R
  `book_occurrence_peel()` and `ledger_is_stitched()` (#251).
- `PredictiveDistribution::select(dim, keys)`: the components with given
  values on one dimension, joint structure kept, so a tower's ceded part
  can be joined to a book (#193). Python `PredictiveDistribution.select`, R
  `select_components()`; `Tower.apply`'s docs now warn against summing over
  `kind`.
- Tower matching diagnostics (#211): `TowerModel::diagnostics` reports
  alphas below a minimum, jumps between neighbouring alphas and an
  infinite mean, and `fit_references_bounded` keeps the alpha between
  reference points within bounds; `fit_references` no longer says
  `references = NaN` when it refuses. Python `TowerModel.diagnostics()` and
  `fit_references(..., alpha_bounds=)`; R `tower_diagnostics()` and
  `fit_references(alpha_bounds =)`.
- Exposure-rating helpers (#212): `TabulatedCurve::from_severity` (a
  severity's exposure curve, tabulated where it bends),
  `Mbbefd::swiss_re_for_mean_rate` and `swiss_re_for_layer_share` (the
  Swiss Re `c` calibrated to a mean destruction rate or to one layer's
  share), and `layer::layer_cv` (the CV of a layer's loss per loss that
  reaches it). A risk profile's band may start at 0, and
  `Band::with_spread_bounds` keeps the band's sum insured as the mean
  when it can. The Python and R risk profiles now warn when the default
  uniform spread moves a band's mean, and take `spread="auto"`. Python
  `TabulatedCurve.from_severity`, `Mbbefd.swiss_re_for_mean_rate`,
  `Mbbefd.swiss_re_for_layer_share`, `layer_cv`; R
  `tabulated_curve_from_severity()`, `swiss_re_for_mean_rate()`,
  `swiss_re_for_layer_share()`, `layer_cv()` and
  `risk_profile(spread = "auto")`.
- Small fixes from the guides (#214): `LimitedFluctuation::p` and `k`
  (Python `LimitedFluctuation.p`, `.k`); `year_fraction`'s docs warn
  against assigning accident years with it; the exponential premium
  principle keeps its loading at a very small `k` (it fell below the
  mean at `k = 1e-12`); `Portfolio::epd` no longer returns a tiny
  negative ratio.
