# Design note: ratemaking

Status: **Part implemented** (premium at the current rate level, trend) ·
v0.6 · Depends on: nothing · Lane: Aggregate

## Goal

The overall rate indication of Werner and Modlin's *Basic Ratemaking* (5th
ed., CAS, 2016), chapters 5 to 8: bring the experience period's premium to
the current rate level, trend premium and losses to the period the new
rates will be in effect, and turn the projected loss ratio or pure premium,
with expenses and profit, into an indicated rate change weighted by
credibility against a complement. It comes in two parts:

1. **Premium at the current rate level and trend** (done): this note's
   "What exists".
2. **The indication** (next): expense provisions (all-variable,
   premium-based and exposure-based projections), the permissible loss
   ratio, the loss ratio and pure premium methods, and credibility against
   a complement, reusing `credibility::LimitedFluctuation`. Checked end to
   end against Appendices A to C, and the equivalence of the two methods
   on synthetic data.

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
- Python, in `prospicio.pricing`: `year_fraction`, `Period`,
  `RateHistory`, `RateLevel`, `TrendFit`, `trend_factor`. R:
  `year_fraction()`, `average_written_date()`, `average_earned_date()`,
  `rate_level()`, `trend_fit()`, `trend_factor()`.

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
linear fit (−$0.75), and Appendix A and C's loss trend factors. The book's
rounding is in `knowledge/references/werner-modlin-ratemaking.md`.
