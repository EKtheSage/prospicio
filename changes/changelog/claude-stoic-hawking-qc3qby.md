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
  `rates` entry; R `pnl_program_stated()` and `rates`.
