# Design note: the P&L ledger

Status: **In progress** (PRs 1 to 3 done) · Depends on: `Grid`, `Tower::on_annual_grid` and
`Tower::on_grid`, `bivariate::{JointGrid, fft_bivariate}` · Lane: Aggregate

## Goal

A treaty's profit and loss as distributions, not just means: premium,
loss, expenses, ceded premium, recoveries and commissions as signed cash
flows (legs), their totals and the result, each with its exact
distribution, and the scenario view that shows what every line looks like
in an adverse year. This is the last large item parked from `aggregate`
parity (`docs/design/aggregate.md`): the Python package `aggregate`'s
`PnL` (`_pnl.py`), its DecL builders (`pnl` and `xpnl`,
`_pnl_builders.py`) and `pnl_program` (`_program.py`), about 7,300 lines
in 1.0.1.

The lane row used to say the ledger "needs signed grids". It does not need
a signed FFT grid: every row is a function of a non-negative source (the
annual gross loss, or the joint gross and ceded loss), evaluated atom by
atom, so a row's distribution is the source's probabilities grouped by
the row's signed values. What it needs is a distribution on an irregular,
signed support. A regular signed grid comes in only for legs rebucketed
on a step (PR 3).

## What exists

### In `aggregate` 1.0.1

- **The source.** A distribution whose atoms (grid points, or cells of a
  two-dimensional grid) carry probabilities: a gross aggregate, a net
  marginal, or the joint `(gross, ceded)` of an occurrence program. The
  probabilities are rescaled to sum to 1 (`_pnl.py`, about line 1100), so
  a ledger over an FFT density whose mass is `1 − 2e-5` shows the
  rescaled means.
- **Legs.** `Leg(label, func, bs=0, is2d=False, kind=None)`: a constant
  or a function of the source's value (of both axes for `is2d`), written
  as a positive magnitude. `kind` is one of `premium`, `loss`, `expense`,
  `recovery`, `commission` and feeds only the ratio exhibit.
- **Groups.** `Group(label, role, consideration, obligation)`: the role
  `sell` books `+consideration − obligation`, `buy` the contra (`−`ceded
  premium, `+`recovery, `+`commission). The book's result is the sum of
  its signed legs at each atom.
- **The ledger.** `PnL(name, source, groups, result_name, tier_spans)`:
  every row (each leg, each side's total when it has several legs, each
  group's result, the running net through each group, tier and net-of
  blocks, the grand totals and result, and the total impact of the covers
  on the gross result) is a per-atom sum of signed legs. Means add down
  the sheet and the covariance comes for free; no row is arithmetic on
  marginal distributions.
- **Exhibits.** `summary_df`, the card: each block's consideration,
  obligation and margin with `EX`, `SD`, `CV`, skew, `P01`, median and
  `P99` (marginal quantiles, which do not add down the card).
  `economic_df`, the sheet: every row with `EX`, `SD`, `CV`, skew and the
  scenario columns `κ01, κ05, κ10, κ25, κ50, κ75, κ90, κ95, κ99`: the
  conditional mean of each row given that the grand result lands at its
  lower `q`-quantile, so each column is one consistent state of the world
  and foots exactly. Where the result is not monotone in the source the
  condition is a level set (several atoms with the same result).
  `economic_marginal_df` (each row's own quantiles), `legs_df`,
  `economic_ratios_df` (premium, loss, expense, margin and the loss,
  expense and combined ratios per block, with the mean of each ratio per
  atom), `validation_df` (rebucketed legs against their exact means),
  `tail_df` (VaR, TVaR, `VaR − mean` at return periods 2 to 1,000),
  `prob_eq_0`, and `pnl_a + pnl_b` on the same source.
- **The distribution of a row** (`GridDistribution`): values ascending,
  any sign and spacing, with their probabilities. Quantiles (lower and
  upper), `tvar` (always the upper tail, even on a payoff, so on a P&L it
  sits near the mean), `cdf`, `sf`, `mean`, and `return_period`, the only
  accessor that reads the payoff sign (`T = 1 / p` on a P&L,
  `1 / (1 − p)` on a loss). No accessor flips signs: `P_q(−X) =
  −P_{1−q}(X)` holds because the rows are built on signed values.
- **Builders.** `pnl` is the consolidated net position (one group: net
  premium, net loss, commission, expenses); `xpnl` the walk from gross
  through each cover (one buy group per tier, or per layer when peeled
  top down or bottom up); expenses on a fixed, premium or loss basis, and
  `derive premium` (`P = (T + F) / (1 − r)`); variable terms (swing,
  sliding scale, profit commission, corridor, retro) and reinstatement
  premiums as legs.
- **`pnl_program(loss_ratio, expense_ratio, net_combined_ratio, …)`**: a
  premium and each layer's deposit or rate from target combined ratios,
  returned as DecL text.

### In prospicio

- `Grid` holds a non-negative regular grid; `natural::Portfolio`
  (`prospicio-pricing`) holds scenarios' totals with probabilities and
  each unit's `κ`, the probability-weighted mean given the total, which is
  the ledger's scenario column, but it refuses negative values.
- `Tower::on_annual_grid` evaluates every layer's ceded loss, premium
  (with swing rating), reinstatement premium, ceding and profit
  commission at each annual loss: the leg functions of an aggregate cover.
  `Tower::on_grid` gives the gross, each layer's annual ceded and the net
  by FFT.
- `bivariate::{JointGrid, fft_bivariate}` compounds a joint per-claim
  severity: with the comonotone per-claim pair `(X, ceded(X))` it is
  `aggregate`'s `occ_bivariate`, the source of the occurrence walk.
- `quote::{Loads, Quote}` (expenses and brokerage on a technical premium)
  and the contract terms on `Layer`.

## Decisions

1. **A new signed discrete distribution in `prospicio-prob`.**
   `Discrete::new(values, probs)` groups equal values, sorts them and
   rescales the probabilities to 1; any sign, any spacing. It implements
   `Distribution` (mean, variance, `cdf`, `survival`, `quantile`) with
   `quantile_upper`, `tvar` (upper tail, as `aggregate`), skewness,
   `pmf`, `return_period(p, payoff)` and `points()`. `Portfolio` keeps its
   own non-negative check; it can later be built from `Discrete` rows.
   The ledger lives in `prospicio_aggregate::pnl`, next to the towers its
   builders use; `prospicio-prob` gains no dependency.
2. **Atoms are snapped before they are grouped.** Summation order changes
   the last bit of a row's value (`aggregate`'s walk and consolidated
   faces of one program produced `3.1` and `3.0999999999999996`, and
   `0.75` against `0.7499999999999999` for a cumulative probability, so
   their `κ75` differed). Values within `1e-12` relative (and `1e-12`
   absolute near 0) of each other are one atom, both when a row's
   distribution is built and when the `κ` level set is taken. A source
   point's value is its grid point; leg values are not rounded to a step.
   Parity tests compare at a tolerance and keep their ladder points off
   cumulative-probability jumps.
3. **Signs come from the role, once.** A leg is a magnitude; `sell`
   books `+consideration, −obligation`, `buy` the contra. Rows are always
   in the payoff convention (profit positive), so the adverse tail is the
   low one and `κ01` is the bad state. The source probabilities are
   rescaled to 1, as `aggregate`.
4. **`tvar` stays the upper tail on every row**, as `aggregate`, so the
   two agree number for number; the adverse view of a payoff is its low
   quantiles and the `κ` columns, which the exhibits already show. A
   lower-tail measure can be added under its own name.
5. **Legs are closures over the source's value**, `Fn(f64) -> f64` (and
   `Fn(f64, f64) -> f64` on a joint source), or constants. The builders
   pass the towers' own functions (`Tower::year`, `premium_for`,
   `ceding_commission_for`, `profit_commission_for`), so a leg cannot
   disagree with the tower's statistics.
6. **Exhibits are typed rows, not data frames.** The ledger returns its
   rows in plan order, each with label, step, side, kind, its `Discrete`
   and the `κ` column values; the card, sheet, ratios and tail table are
   views over them. The bindings turn them into a pandas data frame and an
   R data frame with `aggregate`'s labels.
7. **`pnl_program` returns numbers** (premium, each layer's deposit or
   rate, the targets used), not DecL text.
8. **Left out:** charts, the DecL writer and parser, help text, the
   massive one-sweep route and the stitched peel with its Palm scenario
   ladder (several occurrence layers peeled one by one), `waterfall_df`
   and `pentagon_df`, `evaluate` (Cherny–Madan break-even), and the
   engine pass-throughs (`stats_df`, `reins_*`).

## PR plan

Each PR is one type or method, with its parity tests, bindings and docs.

1. (Done.) **`Discrete`** in `prospicio-prob`: the signed irregular
   distribution of decision 1, with `aggregate`'s `make_var_tvar` (built on
   the atoms with mass; equal masses read as `k / n` exactly; the lower
   quantile by `searchsorted(F, p, 'left')`, `tvar` at `p ≥ F[n−2]` the
   top atom, and for a lone atom that atom, where `aggregate` raises an
   `IndexError`) and the return-period table. Parity:
   `validation/scripts/aggregate_discrete.py`, 544 values on the results
   of examples 1 and 2, equal masses, a signed sample and a lone atom, at
   `1e-12` (`1e-10` on TVaR and skewness). Python `Discrete`, R
   `discrete_distribution()`. The display snap of values within `1e-12`
   of 0 moves to PR 2, where the exhibits are.
2. (Done.) **The ledger kernel**, `prospicio_aggregate::pnl::{Leg,
   Group, Ledger, Source, Layout}`, over a one-dimensional source: the row
   plan (legs, side totals, group results, running nets, spans with
   subtotal and net-of blocks, grand rows, total impact), the moments, the
   card (`Ledger::card`), the scenario columns (`Ledger::scenarios`), each
   row's own quantiles, the legs, `prob_eq_0`, and `Ledger::extend` (`+`).
   Legs are constants, functions of the source's value, or values at each
   atom (the bindings call a callable once on all the atoms). Parity:
   `validation/scripts/aggregate_pnl_ledger.py`, 1,704 values of
   `aggregate`'s `PnL` built by hand on examples 1, 2, 2c, 3a and a
   three-group ledger with a span and a net-of block, all at `1e-12`
   (`1e-10` on skewness). `Discrete` now keeps probabilities that already
   sum to 1 (to `1e-12`) instead of rescaling them again: the second
   rescaling moved a cumulative 0.1 to 0.09999999999999999 and the 10%
   quantile across its jump. Python `prospicio.aggregate.{Ledger, Group,
   Leg}`, R `pnl_ledger()`, `pnl_group()`, `pnl_leg()` and the
   `ledger_*()` exhibits.
3. (Done.) **Rebucketed legs and ratios**: `Leg::step` puts a leg's own
   distribution on a signed regular grid by `aggregate`'s linear,
   mean-keeping scatter (from `floor(lo / h) h`; a constant gets a span of
   `(|lo| + 1) 1e-9`), which changes only that row's marginal quantiles:
   its mean, the sums down the sheet and the scenario columns stay on the
   exact values. `Ledger::validation` compares the exact and grid means,
   `Ledger::ratios` is `economic_ratios_df` (premium, loss net of
   recoveries, expense net of commission and result by block, the ratios
   of means and, where the premium never vanishes, the means of ratios,
   the shares of the first block; a leg without a kind is premium on the
   consideration side and loss on the other), and `pnl::stack` is
   `stack_marginal_pnls` (each ledger's result statistics, and impacts as
   differences statistic by statistic). Parity:
   `validation/scripts/aggregate_pnl_ratios.py`, 324 values: the ratios of
   examples 1, 2, 3a and the three-group ledger, example 2b's sheet,
   marginals, card, validation and grid, and the stack of 2b, 2c and their
   difference, at `1e-12` (`1e-10` on skewness, `1e-15` on the grid).
   Python `Leg(step=)`, `Ledger.validation`, `Ledger.ratios`,
   `prospicio.aggregate.stack`; R `pnl_leg(step =)`, `ledger_validation()`,
   `ledger_ratios()`, `ledger_stack()`.
4. **Builders for annual covers**: expenses (fixed, premium, loss basis;
   `derive premium`), the plain and consolidated ledgers, the walk and
   the per-atom peel over aggregate covers, on `Tower::on_annual_grid`.
   Parity: 3a, 3b, 5.
5. **Occurrence covers on a joint source**: the comonotone `(gross,
   ceded)` per claim, compounded by `fft_bivariate`, and the walk through
   an occurrence tier then an aggregate one. Parity: 4.
6. **`pnl_program`** (premium and layer deposits or rates from target
   combined ratios), then the variable terms and reinstatement premiums
   as legs (swing, sliding scale, profit commission, corridor, retro;
   `D + h(R)` on the `(L, R)` joint).

## Validation

Targets come from `aggregate` 1.0.1 (`uv run --python 3.12 --with
aggregate==1.0.1`), written by a script per PR into
`validation/reference/`, at `1e-12` on exact discrete cases and at the
FFT's tolerance on continuous ones, with the source `(x, p)` taken from
`aggregate` when the engine's grid matters (example 5). The cases, with
the numbers a first run gave (2026-10-14):

- **Example 1**, plain: `build('pnl Ex1 10 premium less agg Ex1e 2
  claims dsev [1 2 4] [.5 .3 .2] fixed less 10% premium expense', log2=6,
  bs=1)`. Margin `EX` 5.2, `SD` 1.60623784, skew −0.7181304929, `P01`,
  median, `P99` 1, 6, 7. `κ01…κ99` of the loss −8, −6, −6, −5, −3, −2,
  −2, −2, −2 and of the margin 1, 3, 3, 4, 6, 7, 7, 7, 7. The result's
  atoms (9 less the loss) 1, 3, 4, 5, 6, 7 with probabilities 0.04, 0.12,
  0.2, 0.09, 0.3, 0.25; `q` at 0.01, 0.05, 0.5, 0.99 is 1, 3, 6, 7; `tvar` at 0.01, 0.5,
  0.9 is 5.242424242, 6.5, 7.
- **Example 2**, a hand-built ledger on `x = [0, 10, 20, 50]`, `p = [.4,
  .3, .2, .1]`: `Gross` sells a premium `clip(5 + 1.1x, 12, 40)` against
  the loss `x` and LAE `0.05x`; `XL` buys for a premium of 4 a recovery
  `clip(x − 15, 0, 25)` with a commission 0.4. Gross margin 6.4 (`SD`
  6.967065379), XL −0.1 (7.433034374), net 6.3 (2.913760457); net `κ`
  1.9, 1.9, 1.9, 1.9, 7.4, 8.4, 8.4, 8.9, 8.9; gross premium `κ` 16, 16,
  16, 16, 27, 12, 12, 40, 40 (not monotone); the result's atoms 1.9, 7.4,
  8.4, 8.9 with 0.3, 0.2, 0.4, 0.1, `tvar(0.5)` 8.5. Ratios over all: P
  15, L 8.5, E 0.2, M 6.3, the mean loss ratio 0.4498792271.
- **Example 2b**: one seller with consideration 20 and the loss `1.03x`
  on a step of 4: loss `EX` −12.36, rebucketed on −52 to 0 by 4 with
  0.0875 at −52, 0.0125 at −48 and 0.4 at 0; the validation error
  1.8e-15.
- **Example 2c**, a level set: premium 30 at `x = 20`, else 10, against
  the loss `x`. Margin `EX` 2, `SD` 14.69693846; `κ50` premium 16.6667,
  loss −6.6667, margin 10 (atoms 0 and 20 share the result 10).
- **Example 3a**, the walk through an aggregate cover: `'xpnl Ex3 10
  premium less agg Ex3e 2 claims dsev [1 2 4] [.5 .3 .2] fixed aggregate
  net of 3 xs 5 deposit 1 cede 10% less 10% premium expense'` (log2=6,
  bs=1). The cover's recovery `EX` 0.24, its margin −0.66 (`SD`
  0.6499230724), the net 4.54 (`SD` 1.21095004), the impact −0.66; net
  `κ` 3.1, 3.1, 3.1, 3.1, 5.1, 6.1, 6.1, 6.1, 6.1.
- **Example 3b**, the same program as `pnl`: net premium 9, net loss
  −3.56, commission 0.1, expense −1, margin 4.54; its `κ75` is 5.1, not
  the walk's 6.1, because of the last-bit tie of decision 2.
- **Example 4**, an occurrence cover: `'xpnl Ex4 20 premium less agg Ex4e
  2 claims dsev [1 2 4] [.5 .3 .2] occurrence net of 2 xs 2 deposit 2
  cede 25% poisson less 10% premium expense'` (log2=6, bs=1; the joint is
  64 × 32). Gross margin 14.2 (`SD` 3.130495168), recovery `EX` 0.8, the
  cover's margin −0.7, net 13.5 (`SD` 2.236067977); loss `κ01`
  −11.80204592, `κ50` −3.685714286; net `κ` 7.5, 9.5, 10.5, 12.5, 13.5,
  15.5, 16.5, 16.5, 16.5.
- **Example 5**, continuous: `'pnl Ex5 1000 premium less agg Ex5e 10
  claims sev lognorm 50 cv 1.5 poisson less 5% loss expense as LAE 25%
  premium expense as Acq'` (log2=12, bs=1). Loss `EX` −499.8136106 (`SD`
  283.4317877; the engine's mean 499.885 over a mass of 0.99998,
  rescaled), LAE −24.99068053, Acq −250, margin 225.1957088 (`SD`
  297.6033771, skew −1.663950107); margin `κ01`, `κ50`, `κ99` −756.75,
  281.7, 659.7; `q(0.005)` −968.85, `cdf(0)` 0.1802988921.
- **`pnl_program`**: `agg … 10 claims sev lognorm 50 cv 1.5 poisson`
  gives a premium of 714 (499.885 / 0.7) and expenses of 25% of premium;
  with `occurrence net of 200 xs 100` and `aggregate net of 500 xs 1000`
  (log2=13) at `net_combined_ratio=.9, occ_combined_ratio=.6,
  agg_combined_ratio=.5`, a premium of 809 and deposits of 146 and 6.09.
