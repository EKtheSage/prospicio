# Design note: ratemaking

Status: **Implemented** (premium at the current rate level, trend, the indication) ·
v0.6 · Depends on: nothing · Lane: Aggregate

## Goal

The overall rate indication of Werner and Modlin's *Basic Ratemaking* (5th
ed., CAS, 2016), chapters 5 to 8: bring the experience period's premium to
the current rate level, trend premium and losses to the period the new
rates will be in effect, and turn the projected loss ratio or pure premium,
with expenses and profit, into an indicated rate change weighted by
credibility against a complement. It came in two parts: premium at the
current rate level and trend (#179), then the indication itself.

## What exists

`prospicio_pricing::ratemaking`:

- `year_fraction(year, month, day)`: dates as year fractions on the book's
  half-month grid (30 June ≡ 1 July, 31 December ≡ 1 January).
- `Period::calendar(from, to)` and `Period::policy(from, to)`: premium
  earned (losses occurring) in a window, or policies written in it. Their
  `average_written_date(term)` and `average_earned_date(term)` set the
  trend lengths; the forecast period is the policy period of the policies
  written while the new rates are in effect.
- `RateHistory::new(&[RateChange])` or `RateHistory::from_levels(...)`, and
  `rate_level(&period, term)`: the parallelogram method, each cell's share
  of the period's premium at one rate level, the average and current
  levels, and `on_level_factor()`.
- `TrendFit::exponential(times, values)` and `TrendFit::linear(...)`: least
  squares on times in years, with `annual_change()`, `fitted(t)` and
  `r_squared`; `trend_factor(rate, years)`.
- `LossRatioYear` and `LossRatioExperience`: each experience year's
  premium on level and trended, and its losses developed, trended and
  loaded for LAE; the years' projected loss ratio as the ratio of totals.
  `PurePremiumYear` and `PurePremiumExperience` likewise over exposure.
- `Provisions::new(variable, profit)`: the permissible loss ratios
  `1 − V − Q` and `1 − F − V − Q`, the loss ratio method's
  `loss_ratio_change(LR, F)` and the pure premium method's
  `pure_premium_rate(PP, fixed per exposure)`.
- `premium_based_expenses(&[ExpenseCategory])`: expense ratios split into
  fixed and variable by their fixed shares (all variable when every share
  is 0).
- `credibility_weighted(estimate, Z, complement)` and
  `trended_present_rates(prior indicated, prior implemented, trend)`, with
  `credibility::LimitedFluctuation` for `Z`.
- Python, in `prospicio.pricing`: `year_fraction`, `Period`,
  `RateHistory`, `RateLevel`, `TrendFit`, `trend_factor`,
  `LossRatioExperience`, `PurePremiumExperience`, `Provisions`,
  `premium_based_expenses`, `credibility_weighted`,
  `trended_present_rates`. R: `year_fraction()`, `average_written_date()`,
  `average_earned_date()`, `rate_level()`, `trend_fit()`, `trend_factor()`,
  `loss_ratio_experience()`, `pure_premium_experience()`,
  `indicated_change()`, `indicated_rate()`, `permissible_loss_ratio()`,
  `premium_based_expenses()`, `credibility_weighted()`,
  `trended_present_rates()`.

## Decisions

1. **Time is a year fraction, on the book's half-month grid.** No date type
   and no calendar crate (the WASM scope stays dependency-free). The book
   never counts days: it rounds dates to the nearest half-month, so
   `year_fraction` maps the first of a month to its start, a mid-month day
   to half way and the last day to the next month's start. Every trend
   length in chapters 5 and 6 and Appendices A and C then comes out exactly.
2. **Trend lengths come from periods.** Premium is trended between average
   written dates and losses between average earned (accident) dates. A
   calendar period's premium was written on average half a term before its
   midpoint; a policy period's losses occur on average half a term after its
   midpoint. The forecast is a policy period, so the book's rule "effective
   date + half the rate-effective period (+ half the term for losses)"
   needs no separate case. Lengths are plain numbers, so an exhibit's
   rounded length (Appendix C's 6.83 for 6.8333) is the caller's input.
3. **Areas are exact.** The parallelogram method's shares are areas in the
   plane of written date `w` and earned date `t`: the period's region
   (`w ≤ t ≤ w + T`, with `t` in a calendar window or `w` in a policy
   window) clipped to each cell of constant level, by Sutherland–Hodgman,
   with the shoelace formula on coordinates measured from the period's
   start. Changes to written policies cut `w`; law changes (`Applies::InForce`)
   cut `t`, so both kinds and any term are one computation.
4. **The current level is the level after every change**, including
   changes after the experience period (Werner and Modlin's step 5).
5. **Nothing rounds.** The book's exhibits round each step to the printed
   precision. The library is exact and exposes each step (shares, levels,
   average level, factors) so a test rounds where the book does.
   `RateHistory::from_levels` takes an exhibit's printed cumulative levels,
   which is how Appendix A's 1.2176 (exact 1.2175) is reproduced.
6. **Trend fits are on times in years.** Werner and Modlin regress on an
   index and annualise with `exp(4b) − 1` for quarterly points; times a
   quarter apart give the same fit with `exp(b) − 1`. A linear fit's annual
   change is its slope, an amount a year.
7. **The indication is formulas over projected years, not a pipeline.**
   Each year takes its factors (on-level, premium trend, development, loss
   trend, LAE) as numbers, so they can come from `RateHistory`, `TrendFit`,
   `prospicio-reserving` or an exhibit's rounded values alike; a factor that
   does not apply is 1, and a net trend (Appendix C) is the loss trend with
   no premium trend. Years combine as the ratio of totals, which is the
   premium- or exposure-weighted average of the yearly ratios.
8. **Fixed expenses enter each method in its own unit.** The loss ratio
   method takes a fixed ratio to premium, the pure premium method an amount
   per exposure; other costs per exposure (Appendix B's net reinsurance and
   modelled catastrophe load) add to the pure premium or the fixed amount.
   The exposure-based projection is fixed dollars over exposures, trended
   with `trend_factor`, so it needs no function of its own.
9. **The profit provision may be negative** (Appendix C's −5% for
   investment income); `Provisions` only requires `V + Q < 1`.

## Out of scope

Exposure aggregation from individual policies (the 24ths rule), the
extension of exposures (it needs a rating algorithm), premium development,
and loss development (the indication takes developed losses; development is
in `prospicio-reserving`). Workers compensation's benefit-change chains and
payroll trend (Appendix D) are arithmetic a caller does with
`trend_factor`.

## Validation

`validation/tests/pricing.rs` (module `ratemaking`) against Werner and
Modlin: the parallelogram method for chapter 5's calendar-year annual and
six-month cases, policy year, and the law change (Tables 5.19 to 5.23,
exact to `1e-7` and the printed values at 4 dp), Appendix A's six-month
history over five calendar years and Appendix C's annual history over six;
fifteen trend lengths from chapters 5 and 6 and the appendices; one- and
two-step premium trend (Table 5.29, Appendix A Premium Trend-3); the
exponential fits of Table 6.14, Appendix A's premium trend and Appendix C's
net trend (within 0.05 points of each printed tenth of a percent), the
linear fit (−$0.75), and Appendix A and C's loss trend factors.

Module `indication`: chapter 7's fundamental equation, Table 7.3 and the
premium-based split of Table 7.4; chapter 8's $500 and +10%; chapter 12's
square root rule and trended present rates ($229); the two methods agreeing
on consistent synthetic data; and the three worked indications end to end:
Appendix A (every year's projected premium and losses to the dollar, the
61.9% loss ratio, 11.3% fixed and 17.0% variable expenses, −6.2%, the 6.2%
complement), Appendix B (every year's losses, $469.88, Z 79.5%, $493.63,
reinsurance $15.68, fixed expense $77.83, $850.76) and Appendix C (net
trend, 108.5%, 34.7% all-variable expenses, a −5% profit, +54.3%, Z 64.4%,
+41.6%). The book's rounding is in
`knowledge/references/werner-modlin-ratemaking.md`.
