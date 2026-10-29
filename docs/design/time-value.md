# Design note: the time value of money in a price

Status: **Implemented** · Depends on: `risk_load` prices · Lane: Aggregate ·
Issues: #195, #480 (return metrics), #479 (the cash-flow engine), #490
(a book of deals), #603 (a deal valued after binding)

## Goal

A long-tail treaty's price depends on when its losses are paid: the
reinsurer earns investment income on the premium until it pays, and holds
capital while losses are outstanding, so the cost of capital accrues over
several years rather than one. Before this note, pricing had no payment
patterns and no discounting. `Pentagon`'s discount is a different idea (the
spread between the premium and the expected loss in the natural allocation,
not a payment delay).

Clark's *Basics of Reinsurance Pricing* names the timing of premium and
loss payments as part of the final price but gives no worked example; the
validation checks closed forms instead.

## What exists

`prospicio_pricing::payout`:

- `PaymentPattern::new(times, shares)`: the share of the ultimate loss
  paid at each time, in years from the treaty's inception.
  `PaymentPattern::from_development(ages, factors)` builds one from
  factors to ultimate, such as a paid chain ladder's `cdf`: the share paid
  by each age is one over its factor. `shifted(by)` moves it later.
- `duration()`, `outstanding(t)`, `discount_factor(rate)` and
  `capital_years(rate)`.
- `unpaid_from(age)`: the pattern of what is unpaid at an age, its times
  from the age and its shares over `outstanding(age)`, to discount reserves
  held at a valuation (#214).
- `price_over_payout(&price, &rule, &pattern, rate) -> PayoutPrice`: a
  one-period `Price` restated over the payout.

Bindings: Python `prospicio.pricing.PaymentPattern`, `price_over_payout`,
`PayoutPrice`; R `payment_pattern()`, `payment_pattern_from_development()`,
`pattern_values()` and `price_over_payout()`.

`prospicio_pricing::returns` (#480), return metrics on simulated cash
flows, one simulation a row:

- `npv` and `npv_rows` at an annual rate; `irr` and `irr_rows`, the
  internal rate of return with `found`, the value at the rate and the
  number of roots the search saw (`Irr`, searched as `IrrOptions` says);
  `TimeUnit` for month or day offsets.
- `moic`, `leverage` and `TieredFee`.

Bindings: Python `npv`, `irr` (an `IrrResult` of lists), `moic`,
`leverage`, `tiered_fee` in `prospicio.pricing`; R `npv()`, `irr()` (a data
frame), `moic()`, `leverage()`, `tiered_fee()`. The metrics are arithmetic
on flows and times; they do not depend on how the flows were built.

`prospicio_pricing::cashflow` (#479), a collateralised deal's monthly
accounts per simulation:

- `CashflowTerms`: the premium, `MonthlyPatterns` (cumulative written,
  earned and paid shares by month), the commutation month `T`, and one rule
  per account, each a public field: `requirement` and
  `commutation_requirement` (`Requirement`), `premium_flow`
  (`PremiumFlow` with `Adjustment`s), `funding` (`Funding`: `Dates`,
  `FundingRule`, `FundingSource`), `collateral_release` (`Timing`),
  `interest` (`Interest`) and `operating` (`Operating`: opening capital,
  an expense rate on the requirement, a `Buffer`, a release `Timing`, a
  `PerformanceFee`).
- `run(u, commission)` gives `Cashflows`: the investor's monthly net flows
  (`n × T`, row-major, months `0..T`), funded, opening, released before
  commutation, distributed at it, the fee and the commutation payment,
  with `npv(rate)` and `irr(options)` through `npv_rows` and `irr_rows`.
  `run_with_requirement` takes each simulation's requirement path in
  place of the terms, and `run_with_patterns` each simulation's paid
  pattern (`PatternDraws`, decision 33).
- `trace(u, commission)` gives every account line of one simulation
  month by month (`Trace`), with `conservation_error()`;
  `trace_with_pattern` the same under a paid pattern of its own.
- `valuation(ActualState)` values the deal after binding from its actual
  accounts at a month: `Valuation::run` gives the forward flows with the
  returns from the valuation month and since inception (`Forward`), each
  simulation conditioned on the paid to date (`condition_on_paid`), and
  `deal_ave` the actual against expected paid by month (decision 34).

In Python (`prospicio.pricing`, `[cashflow-bindings]`): `CashflowTerms`
takes the premium, the commutation month, the three patterns and a
`CollateralRequirement` (loads by base name, a floor, a `(trigger, cap)`
limit), with every rule a keyword (names for the enums; dates as
`"monthly"`, `"quarterly"`, an interval, a list of months or
`{"every": n, "first": m}`; `"at-commutation"` for a release;
`PremiumAdjustment`s; a fee as `(breakpoints, slopes)`); `run` gives
`Cashflows` (`flows` a list of rows, `npv`, `irr` as `IrrResult`),
`run_with_requirement` takes the rows, `run(..., paid_patterns=,
pattern_index=)` each simulation's paid pattern, and `trace` (with
`paid_pattern=`) gives a dict of the account columns with
`conservation_error`. In R: `cashflow_requirement()`,
`cashflow_adjustment()`, `cashflow_terms()` with the same arguments,
`cashflow_run()` (a `flows` matrix and `months` for `npv()` and `irr()`
with `unit = "months"`, `months_requirement` for a given path,
`paid_patterns` and `pattern_index` for a paid pattern by simulation) and
`cashflow_trace()` (an `accounts` data frame, `paid_pattern` too). Both check the terms when
they are built.

`prospicio_pricing::book` (#490), several deals on one shared simulation:

- `DealRun`: one deal's premium, loss and deductions in each simulation
  (`new` and `deduction`, `from_tower` or `from_columns` over
  `Tower::deal_columns`, the amounts behind `deal_ratios`, with the
  year's reinstatement premium in the premium by default, decision 31),
  its inception
  in months from 1 January of the book's first year, its
  `MonthlyPatterns`, and its investor's flows (`cashflows` from a
  `CashflowTerms` and its `Cashflows`, or `flows` from any model).
- `DealBook`: `select` and `shifted` by name, and four tables:
  `amounts` (every item by deal and in total, `["item", "deal"]`) with
  `summary` (means and ratios to the mean premium);
  `distribution(item, grid, tail)` (the book's statistics, each deal's
  contribution at the book's ranks and its marginal effect);
  `cashflows(rate, options)` (the summed flows on the book's calendar and
  the book's IRR, NPV and MOIC on them, with each deal's mean flows and
  own IRR); `calendar(start_year)` (written and earned premium, paid loss
  and investor flows by calendar year); and `marginal_price(name, rule, assets)`
  with `marginal_prices`, each deal's price written last (decision 32).

In Python (`prospicio.pricing`): `DealRun(name, premium, loss,
deductions=, inception=)` (a deduction NMBF leaves out given as
`(amounts, True)`), `DealRun.from_tower(..., reinstatement_premium=True)`,
the `premium` and `loss` it holds, and `with_inception`,
`with_patterns`, `with_cashflows(terms, cashflows)` and `with_flows`, each
returning a new run; `DealBook` with `select`, `shifted`, `amounts()` and
`calendar(start_year)` as `PredictiveDistribution`s, `summary()` as
columns, `distribution(item, percentiles, ..., tail=)` as a dict of the
total and each deal's contribution and marginal cells, and `cashflows(rate)`
as a `BookCashflows` (the book's `irr` an `IrrResult`, `irr_by_deal` and
`mean_by_deal` by name), and `marginal_price(assets, cost_of_capital=,
distortion=, deals=)` as columns. In R: `deal_run()` (deductions a named list,
`outside_nmbf` their names NMBF leaves out, flows from a `cashflow_run()`
with its `terms` or a matrix) and `deal_run_from_tower(...,
reinstatement_premium = TRUE)`, `deal_book()`,
`book_select()`, `book_shift()`, `book_amounts()`, `book_summary()` (a
data frame), `book_distribution()` (matrices with a column a deal),
`book_cashflows()`, `book_calendar()` and `book_marginal_price()` (a data
frame).

## Decisions

1. **Times run from inception; ages are read as times.** A paid chain
   ladder's ages run from the start of the origin period, which for a
   losses-occurring treaty incepting with the accident year is the
   inception. For risks attaching, or an inception mid-year, shift the
   pattern.
2. **The last factor to ultimate must be 1.** A tail beyond the last age
   goes in as one more age; the remainder is not placed silently.
3. **Annual effective rates**, `(1 + rate)^-t`.
4. **Capital runs off with the outstanding share.** Capital `C = a − P`,
   with `a` the one-year risk measure, is held in year `k` in proportion to
   the share outstanding at its start, and its cost `r C outstanding(k)` is
   charged at the start of the year, discounted, as the one-period price
   charges `r C` at inception. The premium solves
   `P = E[X] d + r (a − P) A`, with `d` the discount factor and `A` the
   discounted capital years. Year 0 counts in full, so a pattern that pays
   everything within the first year has `A = 1`, and at a zero rate the
   one-period price comes back exactly; the tests check this identity.
5. **The cost of capital is a spread over the discount rate.** Investment
   income on the premium is what discounting the losses represents; `r`
   is the extra return the capital asks for.
6. **A distortion price is discounted with the losses**, `P = ρ(X) d`; it
   holds no capital over time.
7. **Simulations are discounted by scaling.** The pattern does not depend
   on the size of the loss, so every simulated year's present value is its
   loss times `d`; there is no per-draw API.

8. **Return metrics discount as the payout does** (#480): `(1 + r)^-t`
   with `t` in years. Month offsets are twelfths of a year and day
   offsets actual/365 (`TimeUnit`); the two give different rates on the
   same monthly flows.
9. **The IRR search is the reference model's, as the default.** The value
   of the flows on 400 rates from −0.95 to 1; of its sign changes and
   exact zeros, the bracket whose midpoint is nearest 0; 8 bisections of
   it, then 3 Newton steps clipped to it. The procedure is the parity
   target, so the defaults are its numbers and every one is an option.
10. **A missing root is a flag, not an error.** Flows of one sign, or with
    their root outside the range, give `found = false` and NaN; flows with
    several roots give the one nearest 0 and `sign_changes` above 1, so a
    caller can see both cases. Since decision 35, `root_above` and
    `root_below` tell a root outside the range from no root at all.
11. **MOIC and leverage divide by the capital funded plus the opening
    capital**, which must be positive: `1 + released / C` and
    `rate × premium / C`. The leverage rate has no default.
12. **A tiered fee is piecewise linear above a floor**:
    `Σ_i s_i (clip(m, b_i, b_{i+1}) − b_i)`, the last tier unbounded. The
    tiers are the caller's; none ships with the library.

13. **The cash-flow engine is a module of `prospicio-pricing`** (the user's
    decision of 2026-10-22, over a crate of its own): it sits beside the
    payout and the return metrics, and moves to a crate only if cash-flow
    work grows into a domain of its own.
14. **Every rule is a parameter; none is the reference model's by
    default.** The constructor's defaults are the simplest deal (premium
    collected when written, the shortfall called every month from the
    investor, excess released monthly, no interest, expenses, buffer or
    fee); #479's chain is one setting of the fields, written out in the
    module's tests, not a preset.
15. **A requirement is a sum of loads on named bases** (unearned premium
    `W − E`, the reserve on earned `u E − C`, the outstanding `u P − C`,
    written, earned, a fixed amount) **with a floor and an optional switch
    to the limit**: when `trigger × u ≥ cap` it is `cap × W − C`, #479's
    rule. The commutation requirement is the same type, so the issue's
    `UEP_load × UEP + commutation_load × u P (1 − cPD)` is two terms; no
    load on the outstanding is assumed. A per-simulation requirement path
    can be given instead (`run_with_requirement`).
16. **Funding has dates, a rule and a source.** The rule is the rise in
    the obligation `S_t − premium_offset × W_{t−lag}` (#479's) measured
    from the previous funding date's obligation or from the highest so far,
    both offered since the issue does not say which, or the trust's
    shortfall. The source is the investor or the operating account first.
17. **The trust's month**: opening balance, plus the call, the premium
    collected (`collected_rate × ΔW_{t−lag}`, less `c` under decision 28) and any adjustment
    (`written_rate × W − c × E`, read at the patterns' last month or at
    the adjustment's month), less the claims paid; interest at
    `yield / 12` on the average, opening or closing balance; then the
    excess over the requirement is released when `collateral_release`
    says. The balance may go below zero (an overdraft): under the
    obligation rule nothing calls the claims paid, and the trace shows it.
18. **Commutation is month `T − 1`**, where the requirement is the
    commutation requirement, paid to the cedant; the trust's balance above
    it, or below it, goes to the operating account, which pays the fee and
    distributes the rest, so both accounts end at zero.
19. **The operating account** opens with `opening_fixed + opening_share ×
    P`, paid by the investor at month 0, pays `expense_rate / 12` of each
    month's requirement, and releases what it holds above its buffer (the
    next `months` months' claims, expenses or both, on the regular
    requirement, so the patterns must reach `T + months`).
20. **The fee reads the profit**, the investor's inflows before the fee
    less its outflows, as an amount or over the premium or the capital
    (funded plus opening); the tiered amount is multiplied back by the
    divisor.
21. **A book's deals read one shared simulation** (the user's decision of
    2026-10-22): simulation `j` of every deal is the same year, from one
    year loss table or event catalog, so the book adds the deals without
    reordering, and a book whose deals have different numbers of
    simulations is an error. A book of deals simulated separately would
    need a copula first, which the book does not apply.
22. **The inception moves only the cash flows** (the user's decision of
    2026-10-22): a deal's flows and patterns start at its inception month
    on the book's calendar, and its losses stay those of its own contract
    year. Book month 0 is 1 January of the first calendar year, and a
    shift is a whole number of months. The cost is accepted: an event in
    March of simulation `j` falls in a 1 January deal's first calendar
    year and a 1 July deal's second, so the calendar-year view of one
    event can split across years. Rebuilding a shifted deal's year from
    two catalog years would need every deal's event times, and changes
    the deal's own result.
23. **Each deal keeps its own collateral trust** (the user's decision of
    2026-10-22): the book adds the investors' flows and does not pool
    collateral, so the cash-flow engine runs once a deal.
24. **All four tables are built** (the user's decision of 2026-10-22):
    amounts with their summary, the distribution, the investors' flows
    and the calendar years. A book's ratio is its total over its total
    premium, a sum over sums. A deduction a deal does not have is 0 for
    it, and two deductions of one name must agree on whether NMBF leaves
    them out.
25. **The book's IRR is on the summed flows** (the user's decision of
    2026-10-22), each simulation's flows of every deal added month by
    month, never an average of the deals' IRRs; its NPV is to book month
    0. Its MOIC is `1 + Σ flows / C` with `C` the capital funded plus the
    opening capital over every deal, `moic` with the investor's net gain
    (decision 11), so a book of one deal gives the deal's own.
26. **A deal's contribution is read at the book's ranks**
    (`StatisticsGrid::co_statistics` with the book's total as the
    driver): at each percentile, the deal's value in the simulation the
    book's percentile selects; at each TVaR, its mean over the book's
    tail. The mean is the deal's own mean, not its value at the book's
    mean, so every contribution adds up to the book's. The marginal
    effect is the book's statistic less the book's without the deal, and
    the marginal effects need not add up.
27. **The tail is a setting**: `Tail::Upper` reads high values as the
    risk (a loss), `Tail::Lower` low ones (a margin) by reading the
    item's negative and turning the values back, so `"TVaR 99"` of a
    margin is the mean of its lowest 1%.
28. **A simulation's commission can come off the premium collected**
    (#609): under `PremiumFlow::deduct_commission`, off by default so
    every result stays as it was, the premium paid in is
    `(collected_rate − c) × ΔW_{t−lag}` with `c` the simulation's
    commission rate, so a sliding scale runs in one call; it equals a run
    at `collected_rate − c` simulation by simulation. Without it `c` is
    read only by adjustments with `less_commission` and by the fee.
29. **A reserve rule that pays the investor first is flagged, not
    refused** (#608): the reserve on earned premium is small early on, so
    a requirement on it alone releases premium collected ahead of earning
    and the investor's flows start positive, with several or no internal
    rates of return. It is the base as documented, so it stays;
    `Cashflows::paid_first` marks such runs, `irr`'s `sign_changes` was
    already in Python and R, and the docs point a collateralised fund to
    the outstanding loss.
30. **Monthly patterns come from the patterns actuaries hold** (#601):
    `PaymentPattern::to_monthly` reads a pattern by age at each month end,
    `(m + 1) / 12` years, between its times as a step (paid at the
    stated times, as the pattern discounts), linearly (evenly over each
    period, the default) or log-linearly in the unpaid share; at month
    `12k − 1` every rule gives the pattern's share at `k` years, and past
    the last time all is paid (the shares, any tail included, sum to 1).
    `MonthlyPatterns::from_annual` builds the engine's three patterns from
    a written one: premium written in month `s` is written at its start
    and earned evenly over the policy term from it, losses occur with
    earning in the middle of their month and are paid by the pattern from
    the accident date. A single inception and policies written through a
    year are the same call with a different written pattern, so there is
    no separate earning type.
31. **A deal's premium includes its reinstatement premium** (#595, the
    user's decision of 2026-10-24): `DealRun::from_tower` adds each year's
    reinstatement premium, as the tower computes it (pro rata as to amount
    and, where the layer says so, as to time), to the deposit or
    swing-rated premium, so a book's premium, margin and returns count
    what a year that reaches a layer pays back. `from_tower_with(...,
    false)` keeps the deposit alone, as `deal_ratios` reads it.
    `Tower::deal_columns` gives the two apart (`premium` and
    `reinstatement_premium`), and `from_columns` adds them. It is added to
    the premium rather than kept as a leg of its own, so the book's items
    and ratios do not change shape.
32. **A deal's marginal price is the book priced with it less without
    it** (#588): `DealBook::marginal_price(name, rule, assets)` prices the
    book's loss with the deal and without it, by the `PremiumRule` and the
    capital standard `assets` that `price_portfolio` takes, and the deal's
    expected loss, assets `dA` and premium are the differences. Under a
    cost of capital `r` that is `P = (EL + r dA) / (1 + r)`, the formula
    the guide built by hand; under a pricing distortion it is
    `rho_g(book) - rho_g(book without)`. Only the loss is priced, as in
    `price_portfolio`; the premium, deductions and flows do not enter.
    Under a subadditive standard the marginal premium is at most the
    Euler (co-measure) premium, so pricing every deal on its marginal
    leaves the book short; `marginal_prices` gives every deal in turn.

33. **A paid pattern for each simulation** (#600, engine side): the
    cash-flow engine takes, in place of the terms' one paid pattern,
    `PatternDraws`: `K` monthly cumulative patterns and the pattern each
    simulation pays by (`new` for one a simulation, `indexed` for a few
    shared by many, as a reserving model's posterior draws or a handful of
    scenarios would give them). Only the paid pattern varies; written and
    earned stay the terms'. Each pattern is finite, non-decreasing, of the
    terms' length and ends at the terms' paid share, so a simulation's
    ultimate loss ratio means the same whichever pattern pays it; a
    pattern that ends elsewhere is refused rather than rescaled.
    Identical patterns equal to the terms' reproduce `run` bit for bit,
    and a mixture is the runs of each pattern recombined. Patterns and a
    requirement given month by month are not combined (the given
    requirement already fixes what the pattern would move). Where the
    patterns come from, a Reserving lane fit's posterior or bootstrap
    draws turned into monthly patterns, is that lane's second part of
    #600 and not built here.

34. **A bound deal valued from its actual accounts** (#603; the user's
    decisions of 2026-10-28 on M1 to M6 of the proposal, each as
    recommended). After binding, an investor asks whether the deal is
    emerging as priced and what the rest of it is now worth.
    `CashflowTerms::valuation(ActualState)` gives a `Valuation` (named
    `from_valuation` in the proposal; clippy reserves `from_*` for
    constructors), whose `run(u, commission, how)` simulates months `v` to
    `T − 1` from the actual accounts, and `deal_ave` compares the paid to
    date with what the priced simulations expected.
    - **M1, conditioning on the paid to date** (`condition_on_paid`,
      `Conditioning`): with `A` the actual paid and `S` the paid pattern
      reached by the end of month `v − 1`, `BornhuetterFerguson` (the
      default) keeps every simulation at ultimate `A + u_j P (1 − S)`, its
      unpaid following the rest of the pattern (`A + u_j P (S(t) − S)` paid
      by month `t`); `Truncation` keeps the simulations with `u_j P ≥ A`,
      unchanged, their `u_j P − A` following the rest of the pattern in
      proportion. Reweighting by a likelihood of `A` is left for later.
      Under the first the forward distribution narrows with the unpaid
      share: on the test deal the sd of the since-inception NPV is
      0.88, 0.75, 0.62, 0.49 and 0.38 of inception's at months 6, 12, 18,
      24 and 29 of a 48-month pattern (`1 − S` is 0.875, 0.75, 0.625, 0.5,
      0.396), while truncation keeps 100% to 80% of the simulations
      (`knowledge/findings/deal-valuation-conditioning.md`).
    - **M2, the actual state** (`ActualState`): the premium collected, the
      losses paid, the trust's and the operating account's balances, the
      capital funded and the equity released, all to the end of month
      `v − 1`, with the investor's actual flows by month (optional) and,
      under `FundingRule::ObligationIncrease`, the obligation its next call
      is measured from (`funding_reference`, which the engine carries from
      month to month and `Trace` now reports). Premium the schedule had
      collected by then and the trust has not received (or received beyond
      it) is collected (or paid back) in month `v`. A valuation after an
      adjustment month is refused; the fee crystallises only at
      commutation, so a valuation is always before it. At month 0 the state
      must be empty and the run is `run`'s, bit for bit.
      `ActualState::from_trace` takes one simulated path's state.
    - **M3, both returns** (`Forward`): the forward return takes the net
      asset value (the trust's and the operating account's balances) as the
      investment at month `v`, paid in at the first forward month as `run`
      puts the opening capital in month 0 (`forward_rows`, `forward_npv`,
      `forward_irr`, `forward_moic` as `1 + net / (NAV + funded after v)`);
      the return since inception puts the actual flows before each
      simulation's forward flows (`since_inception_rows` and its NPV, IRR
      and MOIC). The fee reads the totals since inception.
    - **M4, the actual's percentile**: the share of the priced simulations'
      expected paid at or below the actual, ties counted half; `deal_ave`
      uses the priced simulations, not the conditioned ones.
    - **M5, where `deal_ave` lives**: in `prospicio_pricing::cashflow`,
      with no dependency on `prospicio-models`, its columns named as
      `monitor`'s (`actual`, `expected`, `ratio`) with `percentile` and
      quantiles at given levels (numpy's linear). With a `PatternDraws`
      each simulation's own pattern gives its expected paid (decision 33);
      a reported loss runs the same way with the reported pattern.
    - **M6, the step**: a valuation is at a month end, the engine's step.

    In Python `CashflowTerms.valuation(month, **state)`,
    `CashflowTerms.actual_state(u, month)`, `Valuation.run(u, commission,
    conditioning=)` giving `Forward`, `condition_on_paid` and `deal_ave`;
    in R `cashflow_valuation()`, `cashflow_actual_state()`,
    `cashflow_forward()`, `cashflow_valuation_trace()`,
    `condition_on_paid()` and `deal_ave()`. A valuation with a paid
    pattern by simulation is not built.

35. **A root outside the IRR's range is flagged, and found on request**
    (#670, proposed by the Aggregate lane and accepted by the user on
    2026-10-29 as built). Under
    the reference model's range, −95% to 100% a year (decision 9), a
    collateralised deal's best years return more than 100% and were
    reported as having no root, like flows with none, so leaving them out
    biased the IRR statistics down (`knowledge/findings/deal-report-tables.md`).
    `Irr::root_above` and `root_below` say whether a root lies outside the
    range: the value at the range's end and its limit beyond it differ in
    sign, the limit as the rate grows being the sign of the earliest
    nonzero flow (summed by time) and as it falls to −100% that of the
    latest. `IrrOptions::extend` (off by default, so the reference model's
    results are unchanged) searches for that root when the grid sees none:
    above `upper` at `(1 + upper) 2^k − 1` and below `lower` at
    `−1 + (1 + lower) / 2^k` for `k` up to 60, the first bracket on each
    side and of the two the one nearer 0, refined as in the range;
    `sign_changes` then counts it. Breaking for struct literals of `Irr`
    and `IrrOptions`. In Python `IrrResult.root_above` and `root_below` and
    `extend=` on `irr`, `Cashflows.irr`, `Forward.forward_irr` and
    `since_inception_irr` and `DealBook.cashflows`; in R `root_above` and
    `root_below` columns and `extend =` on `irr()` and `book_cashflows()`.
36. **A deal's and a book's price adequacy in one table** (#589, items
    (a) to (e) proposed by the Aggregate lane for the user's acceptance).
    `DealBook::adequacy(rule, assets)` gives a row a deal and the book's
    `"total"`: the premium charged and kept, the expected loss and margin,
    the allocated and marginal assets and capital, the return on each, the
    technical and marginal premiums, the target premium and the adequacy.
    (a) *The premium kept is the premium less every deduction*
    (commissions, brokerage and the deal's own deductions, inside NMBF or
    not), the money that pays losses and earns the return; the premium
    charged includes the reinstatement premium a run from a tower carries
    (decision 31). (b) *Allocated assets are each deal's Euler share of the
    book's* under the capital standard, as `price_portfolio` allocates, and
    *marginal assets the book's with the deal less without it* (decision
    32); each basis's capital is its assets less the premium kept, so a
    deal that keeps more needs less capital, and its return the margin over
    that capital. (c) *The target return is the book's return at its
    technical price*: the cost of capital under `PremiumRule::CostOfCapital`
    and the distortion's implied return under a pricing distortion. (d)
    *The target premium is the premium kept that returns the target on the
    allocated capital*, `(EL + t A) / (1 + t)`, which under a cost of capital
    is the allocated technical premium, and the adequacy is the premium
    kept over it. (e) *One deal's adequacy is a one-deal book*, where the
    allocated and marginal bases agree, so there is no separate quote
    method. In Python `DealBook.adequacy(assets, cost_of_capital=,
    distortion=)` and in R `book_adequacy()`, each a table of columns.

## Conventions left as settings (cash-flow engine)

The user accepted these on 2026-10-22 as built: each stays a setting with
the default it has, and no parity run against the reference model is
planned.

- Which obligation the issue's `ob_prev` is: the previous funding date's
  (`ObligationReference::PreviousDate`) or the highest so far
  (`RunningMax`).
- Whether the obligation should net the claims paid: as the issue writes
  it, the trust runs below zero by the claims paid between rises.
- What the performance fee's margin is (`FeeBasis`) and whether
  `adj_rate × cWP_end − cc × cEP_end` reads the ultimate or the
  adjustment month's premium (`AdjustmentBase`).
- Whether the commutation reserve carries a load: the issue lists a load
  on it among the negative controls.

## Validation

- Closed forms: an annuity's discount factor, the capital years of an
  even run-off, the outstanding share.
- The reduction to the one-period price for one payment within the first
  year at a zero rate, and the premium equation `P = E d + r (a − P) A`
  checked term by term at a positive rate.
- Against the Reserving lane's chain ladder on GenIns paid: the pattern from
  `cdf` has paid, by each origin's latest age, the fit's latest over its
  ultimate (`validation/tests/pricing.rs`, module `payment_pattern`).
- Price adequacy (#589): on a one-deal book the allocated and marginal
  assets and premiums agree with the book's own price; under a cost of
  capital the target return is the rate and the target premium the
  allocated technical premium; a deal whose premium kept (charged less a
  brokerage deduction) is its technical premium returns the target on its
  allocated capital to `1e-9` and has adequacy 1; the allocated assets,
  technical premiums and premiums kept add up to the book's; under a
  distortion the target is the book's return at its price (unit test in
  `book.rs`; Python and R tests against `marginal_price` and
  `price_portfolio`).
- Return metrics (#480): an IRR in closed form (−100 now, 121 in two
  years, 10%); two roots, 10% and 20%, where the one nearest 0 is kept and
  a search from 15% finds the other; flows of one sign and a root outside
  the range flagged, the second told from the first by `root_above` or
  `root_below` and found under `extend` (300% above, −98.5% below, about
  6,000% a year on monthly flows, an in-range root unchanged); an exact zero on the grid; the rows call against one
  row at a time; monthly against actual/365 in closed form; a three-tier
  fee by hand (unit tests in `returns.rs`, and the same cases in Python
  and R).
- The cash-flow engine (#479; unit tests in `cashflow.rs`): money is
  conserved (premium, adjustments and interest in equal the investor's net
  flows, the fee, claims, expenses and the commutation payment out, to
  1e-9) and both accounts end at zero, under five settings that between
  them use every rule, at loss ratios from 0 to 2; a three-month deal by
  hand; the limit switch's requirement by hand; calls only on funding
  dates; the running maximum calling no more than the previous date; each
  of the issue's negative controls (monthly funding, no buffer, interest on
  the opening balance, a load on the commutation reserve, no unearned
  premium at commutation) moving the flows; the flows' NPV at 0 equal to
  released plus distributed less funded and opening. After binding
  (decision 34): a valuation at month 0 is `run` bit for bit; a path
  valued at months 1 to 18 under either conditioning, at its own loss
  ratio, gives the rest of its flows to 1e-9 under all five settings, and
  money is conserved from the actual balances; premium short of the
  schedule is collected at the valuation month; over 300 paths drawn from
  the priced distribution the actual's A vs E percentile and the realised
  NPV's percentile among the conditioned forward NPVs are uniform
  (Kolmogorov-Smirnov at 1%). No parity with the
  reference model: its numbers are in the user's local reproduction, so
  the findings are not marked verified.
- A book of deals (#490): a book of one layer gives its `deal_ratios`
  times its premium for every item; the book's amounts are the deals'
  sums and its ratios sums over sums; contributions add up to the book's
  at every cell; a book of one deal gives the deal's own flows, IRR, NPV
  and MOIC, and a shift of `k` months leaves the IRR and multiplies the
  NPV by `(1 + r)^(−k/12)`; the book's flows are the deals' shifted and
  summed; the calendar places a 1 July deal's written premium half in
  each year and each deal's paid loss adds up to its loss
  (`prospicio_pricing::book` tests).
