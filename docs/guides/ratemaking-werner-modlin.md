# Basic ratemaking: Werner and Modlin reproduced

Geoff Werner and Claudine Modlin's *Basic Ratemaking* (5th edition, CAS, 2016)
is the CAS study note on the overall rate level indication for personal and
small commercial lines. Its chapters 5 to 8 and 12 build the indication step by
step (premium at current rates, loss development and trend, expenses and
profit, credibility), and Appendices A to C run it end to end on an auto, a
homeowners and a medical malpractice book. This guide follows those exhibits
in prospicio's Python API (`prospicio.pricing`, with `prospicio.credibility`
and `prospicio.reserving` for two steps) and prints the book's numbers next to
prospicio's, so you can see what each step computes and where the book's
rounding moves a digit. This guide covers the steps; its companion,
[`ratemaking-werner-modlin-indications.md`](ratemaking-werner-modlin-indications.md),
runs Appendices A to C end to end, reproduces chapter 11's increased limits
and deductible figures, and lists what prospicio does not cover.

The book's text and exhibits are the authors'; what follows explains them in
new words with the numbers recomputed. Read the book for the reasoning and
this guide for the mechanics. Table and exhibit names below are the book's,
money is in dollars as in the book, and the full parity suite behind the
library is `validation/tests/pricing.rs` (modules `ratemaking` and
`indication`). Rating a reinsurance layer from experience or exposure is a
different job, covered in
[`experience-and-exposure-rating.md`](experience-and-exposure-rating.md).

Every block below runs in well under a second in total. The book behaves like a
spreadsheet that rounds each cell to its printed precision, half up, and
carries the rounded value on. prospicio never rounds, so where the book's
rounding changes a printed digit the code below shows both: the exact value,
and the value with the book's rounding applied by hand. The rounding rules are
collected in `knowledge/references/werner-modlin-ratemaking.md`.

## 1. The process, its data and its dates

An indication compares the premium the current rates will bring in with the
losses, expenses and profit they must pay for, during the period the new rates
will be in effect. The ingredients (chapters 3 and 4) are exposures (car-years,
house-years), premium (written, earned, in force) and losses with loss
adjustment expense (LAE), each aggregated by calendar year, accident year or
policy year. The indication then needs three adjustments to the historical
data: premium to the current rate level, losses to their ultimate value, and
both to the cost level of the future period (trend).

Trend needs the length of time between two average dates, and the book never
counts days: it puts every date on a half-month grid, so June 30 and July 1
are the same point and November 15 is ten and a half months into the year.
`year_fraction` does the same, and a `Period` (a calendar window of premium
earned or losses occurring, or a policy window of policies written) knows its
average written and average accident ("earned") dates for a policy term.
`half_up` below is the book's rounding, used wherever an exhibit rounds a
cell:

```python
from decimal import Decimal, ROUND_HALF_UP

from prospicio.pricing import Period, RateHistory, TrendFit, trend_factor, year_fraction


def half_up(x, dp):
    """Round half up to dp places, as the book's exhibits do (after clearing binary noise at 10 places)."""
    return float(Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


print(year_fraction(2013, 6, 30), year_fraction(2013, 7, 1), year_fraction(2011, 11, 15))

cy2011 = Period.calendar(2011.0, 2012.0)
rates_2013 = Period.policy(2013.0, 2014.0)      # policies written while new rates apply
rates_2015 = Period.policy(2015.0, 2016.0)
print("premium, CY2011 to rates of 1/1/2013, annual   :",
      rates_2013.average_written_date(1.0) - cy2011.average_written_date(1.0), "(book 2.5)")
print("premium, CY2011 to rates of 1/1/2013, 6-month  :",
      rates_2013.average_written_date(0.5) - cy2011.average_written_date(0.5), "(book 2.25)")
print("losses,  AY2011 to rates of 1/1/2015, annual   :",
      rates_2015.average_earned_date(1.0) - cy2011.average_earned_date(1.0), "(book 4.5)")
print("losses,  AY2011 to rates of 1/1/2015, 6-month  :",
      rates_2015.average_earned_date(0.5) - cy2011.average_earned_date(0.5), "(book 4.25)")
py2011 = Period.policy(2011.0, 2012.0)
print("losses,  PY2011 to rates of 1/1/2015, annual   :",
      rates_2015.average_earned_date(1.0) - py2011.average_earned_date(1.0), "(book 4.0)")
```

```text
2013.5 2013.5 2011.875
premium, CY2011 to rates of 1/1/2013, annual   : 2.5 (book 2.5)
premium, CY2011 to rates of 1/1/2013, 6-month  : 2.25 (book 2.25)
losses,  AY2011 to rates of 1/1/2015, annual   : 4.5 (book 4.5)
losses,  AY2011 to rates of 1/1/2015, 6-month  : 4.25 (book 4.25)
losses,  PY2011 to rates of 1/1/2015, annual   : 4.0 (book 4.0)
```

These are the trend lengths of Tables 5.26 to 5.28 and 6.15 to 6.18. Premium
is trended between average written dates and losses between average accident
dates; a calendar year's premium was written on average half a term before its
midpoint, and a policy year's losses occur on average half a term after its
midpoint. The future period is a policy period (the policies written while
the new rates are in effect), so the book's rule "effective date plus half the
rate period, plus half the term for losses" needs no special case.

prospicio does not aggregate exposures or premium from individual policies
(chapter 4's written, earned, unearned and in-force exposures, and chapter 5's
Tables 5.1 to 5.14 are arithmetic on policy dates); it starts from the
aggregated figures an exhibit would show.

## 2. Premium at the current rate level: the parallelogram method

Historical premium was written at the rates of its day. To ask what it would
be at today's rates, the book's parallelogram method (Tables 5.15 to 5.23)
assumes policies are written evenly over time, draws each rate change as a
diagonal line in the square of written date against earned date, and measures
the share of a year's earned premium in each region of constant rate level.
The on-level factor is the current cumulative level divided by the year's
average level.

`RateHistory` takes the effective dates and changes, and `rate_level` returns
the shares, the average level and the factor for a period and a policy term.
Chapter 5's example has +5% on 7/1/2010, +10% on 1/1/2011 and -1% on
4/1/2012:

```python
ch5 = RateHistory([2010.5, 2011.0, 2012.25], [0.05, 0.10, -0.01])
print(f"current level {ch5.current_level:.5f}  (book 1.1435, Table 5.19)")
cases = [
    ("CY2011, annual policies  (Tables 5.17-5.19)", Period.calendar(2011.0, 2012.0), 1.0, "1.0963", "1.0431"),
    ("CY2011, 6-month policies (Table 5.20)      ", Period.calendar(2011.0, 2012.0), 0.5, "1.1288", "1.0130"),
    ("PY2012, annual policies  (Table 5.21)      ", Period.policy(2012.0, 2013.0), 1.0, "1.1464", "0.9975"),
]
for name, period, term, book_avg, book_factor in cases:
    lv = ch5.rate_level(period, term)
    print(name, "shares", [round(s, 4) for s in lv.shares])
    print(f"    average {lv.average_level:.5f} (book {book_avg})"
          f"  factor {lv.on_level_factor:.4f} (book {book_factor})")
```

```text
current level 1.14345  (book 1.1435, Table 5.19)
CY2011, annual policies  (Tables 5.17-5.19) shares [0.125, 0.375, 0.5]
    average 1.09625 (book 1.0963)  factor 1.0431 (book 1.0431)
CY2011, 6-month policies (Table 5.20)       shares [0.25, 0.75]
    average 1.12875 (book 1.1288)  factor 1.0130 (book 1.0130)
PY2012, annual policies  (Table 5.21)       shares [0.25, 0.75]
    average 1.14634 (book 1.1464)  factor 0.9975 (book 0.9975)
```

Each share, average and factor agrees with the book. Two averages sit exactly
on a rounding boundary (1.09625 and 1.12875), which is why they are printed to
five places here; the book rounds them up. The book also rounds the
cumulative index to 1.1435 (exact 1.14345) before dividing; the policy-year
average of 1.1464 uses that rounded index and is 1.14634 from the exact one.

A law change (Tables 5.22 and 5.23) applies to every policy in force on its
date, not only to new ones, so its line in the square is vertical rather than
diagonal. Add the book's -5% benefit change on 7/1/2011 as an `in_force`
change:

```python
law = RateHistory(
    [2010.5, 2011.0, 2011.5, 2012.25],
    [0.05, 0.10, -0.05, -0.01],
    applies=["written", "written", "in_force", "written"],
)
lv = law.rate_level(Period.calendar(2011.0, 2012.0), 1.0)
print("shares", lv.shares)
print("levels", [round(x, 4) for x in lv.levels], " current", round(lv.current_level, 4))
print(f"factor {lv.on_level_factor:.6f}  (book 1.0171)")
```

```text
shares [0.125, 0.25, 0.125, 0.125, 0.375]
levels [1.0, 1.05, 0.9975, 1.155, 1.0973]  current 1.0863
factor 1.017084  (book 1.0171)
```

The five cells of 2011 and their levels are Table 5.23's (1.0000, 1.0500,
0.9975, 1.1550, 1.0973), the current level is its 1.0863, and the factor
rounds to the printed 1.0171.

### Appendix A: six-month auto policies


Appendix A's Current Rate Level exhibits apply six changes (-5% on 4/1/2011,
+10% on 7/1/2012, +5% on 10/1/2013, -2% on 7/1/2014, +5% on 10/1/2015 and +5%
on 1/1/2016) to six-month policies, by calendar year. The book rounds the
cumulative index to four places at every step (it ends at 1.1857; the exact
chain is 1.18552), then the average level to four places, then divides.
`RateHistory.from_levels` takes the printed cumulative indices, so the book's
chain is reproduced exactly, and `RateHistory` with the changes gives the
exact chain for comparison:

```python
eff = [2011.25, 2012.5, 2013.75, 2014.5, 2015.75, 2016.0]
changes = [-0.05, 0.10, 0.05, -0.02, 0.05, 0.05]
book_index = [0.9500, 1.0450, 1.0973, 1.0754, 1.1292, 1.1857]
book_avg = [0.9750, 0.9738, 1.0483, 1.0886, 1.0788]
book_factor = [1.2161, 1.2176, 1.1311, 1.0892, 1.0991]

printed_chain = RateHistory.from_levels(eff, book_index)
exact_chain = RateHistory(eff, changes)
a_on_level = []  # kept for the indication in section 8
print(f"exact current level {exact_chain.current_level:.6f}  (book 1.1857)")
print("year  average  book    factor  book    exact factor")
for k, year in enumerate(range(2011, 2016)):
    cy = Period.calendar(float(year), float(year + 1))
    lv = printed_chain.rate_level(cy, 0.5)
    factor = 1.1857 / half_up(lv.average_level, 4)  # the book divides by its rounded average
    exact = exact_chain.rate_level(cy, 0.5).on_level_factor
    a_on_level.append(half_up(factor, 4))
    print(f"{year}  {lv.average_level:.5f}  {book_avg[k]:.4f}  {factor:.4f}  {book_factor[k]:.4f}  {exact:.4f}")
```

```text
exact current level 1.185524  (book 1.1857)
year  average  book    factor  book    exact factor
2011  0.97500  0.9750  1.2161  1.2161  1.2159
2012  0.97375  0.9738  1.2176  1.2176  1.2175
2013  1.04827  1.0483  1.1311  1.1311  1.1309
2014  1.08856  1.0886  1.0892  1.0892  1.0891
2015  1.07876  1.0788  1.0991  1.0991  1.0991
```

Every average and factor matches. The 2012 average is 0.97375 exactly, a
half-up boundary the book rounds to 0.9738; in binary floating point it may
come out a hair below, which is why `half_up` first rounds to ten places.
The last column is the exact answer: rounding the
index at each step moves the factors by up to 0.0002 (2012's 1.2176 is 1.2175
exactly).

### Appendix C: annual medical malpractice policies

Appendix C's changes are -7.6% on 10/1/2011, +14.6% on 3/1/2014 and +13.6% on
7/1/2015 on annual policies. Its cumulative indices 0.9240, 1.0589 and 1.2029
are the exact chain rounded, and its factors divide rounded figures, so the
code rounds the current and average levels to four places before dividing:

```python
mm = RateHistory([2011.75, 2014 + 2 / 12, 2015.5], [-0.076, 0.146, 0.136])
c_on_level = []
book_factor = {2010: 1.2029, 2011: 1.2058, 2012: 1.2724, 2013: 1.3018, 2014: 1.2391, 2015: 1.1190}
for year, book in book_factor.items():
    lv = mm.rate_level(Period.calendar(float(year), float(year + 1)), 1.0)
    factor = half_up(lv.current_level, 4) / half_up(lv.average_level, 4)
    c_on_level.append(half_up(factor, 4))
    shares = " ".join(f"{100 * s:.3f}%" for s in lv.shares)
    print(f"{year}  {factor:.4f}  (book {book:.4f})  exact {lv.on_level_factor:.4f}  shares {shares}")
```

```text
2010  1.2029  (book 1.2029)  exact 1.2029  shares 100.000%
2011  1.2058  (book 1.2058)  exact 1.2058  shares 96.875% 3.125%
2012  1.2724  (book 1.2724)  exact 1.2724  shares 28.125% 71.875%
2013  1.3018  (book 1.3018)  exact 1.3019  shares 100.000%
2014  1.2391  (book 1.2391)  exact 1.2390  shares 65.278% 34.722%
2015  1.1190  (book 1.1190)  exact 1.1190  shares 1.389% 86.111% 12.500%
```

The shares are the exhibit's (65.278% and 34.722% in 2014, where the 3/1
change does not fall on a quarter), and every factor agrees to the printed
four places. Without the rounding of the two levels, 2013 and 2014 would move
by 0.0001.

## 3. Premium trend

Beyond rate changes, the average premium drifts as the mix of business shifts
(newer cars, higher amounts of insurance). The book measures that drift on
average written premium at current rate level and projects it with an
exponential trend, `(1 + trend) ** years`; `trend_factor` is that formula.

Two-step trending (Table 5.29) first moves the historical average to the
latest observed average written premium (the current step), then projects
from there. The book rounds the average earned premium to the cent and each
factor to four places before multiplying:

```python
print(f"one step, 2% for 2.5 years: {trend_factor(0.02, 2.5):.4f} (book 1.0508)")
avg_earned = half_up(1_440_788 / 1_947, 2)
step1 = 753.00 / avg_earned
step2 = trend_factor(-0.01, 1.625)
total = half_up(half_up(step1, 4) * half_up(step2, 4), 4)
print(f"average earned {avg_earned:.2f} (book 740.00)  current {step1:.4f} (book 1.0176)"
      f"  projected {step2:.4f} (book 0.9838)")
print(f"total {total:.4f} (book 1.0011)  projected premium {1_440_788 * total:,.0f} (book 1,442,373)")
print(f"unrounded total {753.00 / (1_440_788 / 1_947) * step2:.6f}")
```

```text
one step, 2% for 2.5 years: 1.0508 (book 1.0508)
average earned 740.00 (book 740.00)  current 1.0176 (book 1.0176)  projected 0.9838 (book 0.9838)
total 1.0011 (book 1.0011)  projected premium 1,442,373 (book 1,442,373)
unrounded total 1.001078
```

Without any rounding the total is 1.001078, which also rounds to 1.0011.

Appendix A selects its premium trend from exponential fits to the average
written premium at current rate level over the 12 months ending each quarter
(Premium Trend-1). `TrendFit.exponential` takes times in years, so quarterly
points are a quarter apart and the annual change comes out directly (the book
regresses on a quarter index and annualizes with the fourth power, which is
the same fit):

```python
awp = [
    103.05, 103.58, 104.15, 104.72, 105.29, 105.84, 106.39, 106.92, 107.42, 107.96, 108.55, 109.16,
    109.75, 110.32, 110.87, 111.40, 111.95, 112.54, 113.10, 113.69, 114.25, 114.79, 115.35,
]
book = {20: 2.1, 16: 2.1, 12: 2.0, 8: 2.0, 6: 2.0, 4: 1.9}
for n, pct in book.items():
    fit = TrendFit.exponential([i / 4 for i in range(n)], awp[-n:])
    print(f"{n:2d} points  {100 * fit.annual_change:.3f}%  (book {pct}%)")
```

```text
20 points  2.062%  (book 2.1%)
16 points  2.058%  (book 2.1%)
12 points  2.023%  (book 2.0%)
 8 points  2.018%  (book 2.0%)
 6 points  1.994%  (book 2.0%)
 4 points  1.947%  (book 1.9%)
```

Each fit rounds to the printed tenth. The book selects 2.0% and, in Premium
Trend-3, trends each calendar year's average earned premium at current rate
level to the latest 115.35, then 2.0% for two years: from 6/30/2015, the
average written date of the latest 12 months, to 6/30/2017. (The exhibit's
footnote reads "06/30/2017 to 06/30/2017", a typo; the 2.0 years printed
beside it and the text agree with 6/30/2015.)

```python
ep_crl = [1_364_916.59, 1_405_728.94, 1_448_424.45, 1_492_177.86, 1_536_267.03]
exposures = [12_900, 13_020, 13_130, 13_258, 13_380]
book = [1.1342, 1.1116, 1.0879, 1.0663, 1.0452]
years = (Period.policy(2017.0, 2018.0).average_written_date(0.5)
         - Period.policy(2015.0, 2016.0).average_written_date(0.5))
projected = half_up(trend_factor(0.02, years), 4)
a_premium_trend = []
print("projected period", years, "years, factor", projected)
for year, ep, x, b in zip(range(2011, 2016), ep_crl, exposures, book):
    current = 115.35 / half_up(ep / x, 2)
    a_premium_trend.append(half_up(half_up(current, 4) * projected, 4))
    print(year, f"{a_premium_trend[-1]:.4f}  (book {b:.4f})")
```

```text
projected period 2.0 years, factor 1.0404
2011 1.1342  (book 1.1342)
2012 1.1116  (book 1.1116)
2013 1.0879  (book 1.0879)
2014 1.0663  (book 1.0663)
2015 1.0452  (book 1.0452)
```

## 4. Loss development

Losses of recent accident years are not yet at their ultimate value. The book's
chapter 6 develops them by the chain ladder: link ratios between successive
ages, an average per age, a selected factor, and the product of the selected
factors from each age to ultimate. Development is outside
`prospicio.pricing` (the indication takes developed losses as an input), but
`prospicio.reserving` runs the chain ladder. Table 6.11's triangle is in
thousands, by accident year 2002 to 2007 at 15 to 75 months:

```python
from prospicio.reserving import ChainLadder, Triangle


def triangle(first_origin, ages, columns):
    origin, age, value = [], [], []
    for a, col in zip(ages, columns):
        for i, x in enumerate(col):
            origin.append(first_origin + i)
            age.append(a)
            value.append(float(x))
    return Triangle.from_long(origin, age, {"reported": value})


t611 = triangle(2002, [15, 27, 39, 51, 63, 75], [
    [1000, 1030, 1061, 1093, 1126, 1159],
    [1500, 1584, 1560, 1651, 1662],
    [1925, 2020, 2070, 2125],
    [2145, 2209, 2276],
    [2190, 2240],
    [2188],
])
rows = t611.values[0][0]  # one row per accident year, NaN past the latest diagonal
links = [[r[j + 1] / r[j] for r in rows if r[j + 1] == r[j + 1]] for j in range(5)]
rounded = [sum(half_up(x, 2) for x in col) / len(col) for col in links]
simple = ChainLadder(average="simple").fit(t611, "reported")
volume = ChainLadder().fit(t611, "reported")
print("arithmetic, exact         ", [f"{f:.4f}" for f in simple.ldf])
print("arithmetic, rounded links ", [f"{half_up(f, 2):.2f}" for f in rounded])
print("book arithmetic           ", ["1.50", "1.30", "1.10", "1.02", "1.00"])
print("volume weighted           ", [f"{half_up(f, 2):.2f}" for f in volume.ldf])
print("book ratio of totals      ", ["1.50", "1.29", "1.10", "1.02", "1.00"])
```

```text
arithmetic, exact          ['1.4989', '1.2932', '1.1025', '1.0175', '0.9991']
arithmetic, rounded links  ['1.50', '1.30', '1.10', '1.02', '1.00']
book arithmetic            ['1.50', '1.30', '1.10', '1.02', '1.00']
volume weighted            ['1.50', '1.29', '1.10', '1.02', '1.00']
book ratio of totals       ['1.50', '1.29', '1.10', '1.02', '1.00']
```

The book averages link ratios already rounded to two places. At 27 to 39
months that gives 1.295, rounded up to 1.30, where the average of the exact
link ratios is 1.2932, which rounds to 1.29. The book's "ratio of total
losses" is the volume-weighted average over the same pairs, which is what the
chain ladder estimates by default. With the book's selections (the rounded
arithmetic averages) as a selected pattern, `ChainLadder` gives the factors to
ultimate and the ultimate losses of Table 6.13; the book rounds the factors
to two places before multiplying:

```python
selected = ChainLadder(ldf={15: 1.50, 27: 1.30, 39: 1.10, 51: 1.02, 63: 1.00}, ldf_kind="ldf")
fit = selected.fit(t611, "reported")
book_cdf = ["1.00", "1.00", "1.02", "1.12", "1.46", "2.19"]
book_ult = [2188, 2240, 2322, 2380, 2427, 2538]
cdf = [half_up(c, 2) for c in fit.cdf][::-1]  # oldest accident year first
ult = [half_up(lat * c, 0) for lat, c in zip(fit.latest, cdf)]
print("factor to ultimate", [f"{c:.2f}" for c in cdf], " book", book_cdf)
print("ultimate          ", [f"{u:,.0f}" for u in ult], f"total {sum(ult):,.0f}")
print("book              ", [f"{u:,}" for u in book_ult], f"total {sum(book_ult):,}")
print(f"exact chain ladder total {fit.total_ultimate:,.1f}")
```

```text
factor to ultimate ['1.00', '1.00', '1.02', '1.12', '1.46', '2.19']  book ['1.00', '1.00', '1.02', '1.12', '1.46', '2.19']
ultimate           ['2,188', '2,240', '2,322', '2,380', '2,427', '2,538'] total 14,095
book               ['2,188', '2,240', '2,322', '2,380', '2,427', '2,538'] total 14,095
exact chain ladder total 14,093.7
```

Table 6.13 reproduces exactly once the factors are rounded the book's way;
the unrounded factors (2.1879 at 15 months rather than 2.19) give a total
2.5 thousand lower.

Appendix A's Loss Development exhibit does the same on reported losses and
paid ALAE for accident years 2009 to 2015, and selects the straight average
excluding the highest and lowest link ratio. That average is built in:
`ChainLadder(average="simple", drop_high=1, drop_low=1)`, whose `n_links`
shows how many link ratios each factor kept. The book, though, averages link
ratios rounded to four places and chains its selections rounded to four
places, so its factors to ultimate come from that rounded selection, passed as
a selected pattern. The block prints both:

```python
ta = triangle(2009, [15, 27, 39, 51, 63], [
    [705_088, 712_475, 714_196, 764_101, 774_384, 785_068, 797_866],
    [725_592, 753_295, 763_913, 861_114, 846_167, 821_509],
    [738_686, 782_248, 855_150, 884_498, 835_120],
    [753_027, 800_258, 874_106, 867_184],
    [732_239, 813_949, 856_495],
])
rows = ta.values[0][0]
links = [[half_up(r[j + 1] / r[j], 4) for r in rows if r[j + 1] == r[j + 1]] for j in range(4)]
all_year = ChainLadder(average="simple").fit(ta, "reported").ldf
print("all-year average, exact ", [f"{f:.4f}" for f in all_year])
print("all-year, rounded links ", [f"{half_up(sum(c) / len(c), 4):.4f}" for c in links])
print("book all-year average   ", ["1.0704", "1.0380", "1.0113", "0.9898"])
exact = ChainLadder(average="simple", drop_high=1, drop_low=1).fit(ta, "reported")
print("high/low out, exact     ", [f"{f:.4f}" for f in exact.ldf], " n_links", exact.n_links)
hi_lo = [half_up((sum(c) - max(c) - min(c)) / (len(c) - 2), 4) for c in links]
print("high/low out, rounded   ", [f"{f:.4f}" for f in hi_lo], " book", ["1.0665", "1.0279", "1.0208", "0.9799"])
fit = ChainLadder(ldf=dict(zip([15, 27, 39, 51], hi_lo)), ldf_kind="ldf").fit(ta, "reported")
print("to ultimate, exact      ", [f"{c:.4f}" for c in exact.cdf])
print("to ultimate, rounded    ", [f"{c:.4f}" for c in fit.cdf], " book", ["1.0966", "1.0282", "1.0003", "0.9799", "1.0000"])
a_development = [half_up(c, 4) for c in fit.cdf][::-1]  # accident years 2011 to 2015
```

```text
all-year average, exact  ['1.0703', '1.0380', '1.0113', '0.9898']
all-year, rounded links  ['1.0704', '1.0380', '1.0113', '0.9898']
book all-year average    ['1.0704', '1.0380', '1.0113', '0.9898']
high/low out, exact      ['1.0665', '1.0279', '1.0208', '0.9799']  n_links [4, 3, 2, 1]
high/low out, rounded    ['1.0665', '1.0279', '1.0208', '0.9799']  book ['1.0665', '1.0279', '1.0208', '0.9799']
to ultimate, exact       ['1.0965', '1.0281', '1.0002', '0.9799', '1.0000']
to ultimate, rounded     ['1.0966', '1.0282', '1.0003', '0.9799', '1.0000']  book ['1.0966', '1.0282', '1.0003', '0.9799', '1.0000']
```

Dropping the highest and lowest leaves 4, 3, 2 and 1 of the 6, 5, 4 and 3 link
ratios (`n_links`). The library's exact average agrees with the book's
selection to four places at every age, but chained unrounded it gives factors
to ultimate 1 lower in the fourth place at 15, 27 and 39 months (1.0965 against
the book's 1.0966); chaining the rounded selection, as the book does,
reproduces the exhibit. The selected factors and the factors to ultimate then
agree with the exhibit; the
last five accident years' factors (1.0000, 0.9799, 1.0003, 1.0282, 1.0966)
are the development column of the indication in section 8. The 15-to-27
straight average is 1.07035 on the rounded link ratios and prints as 1.0704;
on the exact ratios it is 1.0703. Factors below 1 at 51 months are real here:
property damage claims develop downward as salvage and subrogation come in.

## 5. Loss trend

Claim frequency and severity change over time with inflation, safety, law and
behaviour. The book fits exponential trends to quarterly frequency, severity
and pure premium (Table 6.14, 12 months ending each quarter) over the latest
20, 16, 12, 8, 6 and 4 points:

```python
freq = [0.0587, 0.0587, 0.0593, 0.0587, 0.0582, 0.0578, 0.0565, 0.0555, 0.0552, 0.0553,
        0.0556, 0.0560, 0.0559, 0.0554, 0.0551, 0.0549, 0.0550, 0.0549, 0.0547, 0.0544]
sev = [1061.45, 1076.56, 1085.56, 1098.02, 1102.46, 1107.52, 1117.71, 1123.70, 1127.63, 1124.01,
       1116.42, 1109.48, 1105.38, 1102.61, 1096.67, 1093.77, 1100.67, 1108.79, 1122.13, 1126.46]
pp = [62.32, 63.16, 64.33, 64.44, 64.17, 64.04, 63.20, 62.41, 62.24, 62.11,
      62.09, 62.19, 61.82, 61.13, 60.48, 60.03, 60.49, 60.86, 61.37, 61.28]
book = {
    "frequency": (freq, [-1.7, -1.3, -0.7, -1.2, -0.9, -1.5]),
    "severity": (sev, [0.5, -0.1, -0.2, 1.2, 2.5, 3.3]),
    "pure premium": (pp, [-1.2, -1.4, -0.9, -0.1, 1.6, 1.9]),
}
print("points:          " + "".join(f"{n:>8d}" for n in (20, 16, 12, 8, 6, 4)))
for name, (values, printed) in book.items():
    got = [100 * TrendFit.exponential([i / 4 for i in range(n)], values[-n:]).annual_change
           for n in (20, 16, 12, 8, 6, 4)]
    print(f"{name:13s} prosp" + "".join(f"{g:8.2f}" for g in got))
    print(f"{'':13s} book " + "".join(f"{p:8.1f}" for p in printed))
line = TrendFit.linear([i / 4 for i in range(20)], pp)
print(f"linear fit to 20 pure premiums: {line.annual_change:+.2f} a year (book -0.75)")
```

```text
points:                20      16      12       8       6       4
frequency     prosp   -1.72   -1.28   -0.69   -1.24   -0.87   -1.45
              book     -1.7    -1.3    -0.7    -1.2    -0.9    -1.5
severity      prosp    0.54   -0.09   -0.21    1.24    2.52    3.31
              book      0.5    -0.1    -0.2     1.2     2.5     3.3
pure premium  prosp   -1.19   -1.39   -0.91   -0.07    1.59    1.91
              book     -1.2    -1.4    -0.9    -0.1     1.6     1.9
linear fit to 20 pure premiums: -0.75 a year (book -0.75)
```

Every fit is within half a unit of the printed tenth. The frequencies must be
the printed four-place values: the book fits what it prints, and the
unrounded counts over exposures move the 8-point frequency fit from -1.24% to
-1.30%.

The loss trend runs from the average accident date of each experience period
to the average accident date of the future policies (Tables 6.15 to 6.18,
checked in section 1). Appendix A trends in two steps: -0.5% a year from each
accident year's midpoint to 6/30/2015, then +0.5% for the 2.25 years from
there to 9/30/2017 (six-month policies written in 2017):

```python
years = (Period.policy(2017.0, 2018.0).average_earned_date(0.5)
         - Period.calendar(2015.0, 2016.0).average_earned_date(0.5))
projected = half_up(trend_factor(0.005, years), 4)
book = [0.9912, 0.9962, 1.0012, 1.0062, 1.0113]
a_loss_trend = []
print("projected period", years, "years, factor", projected)
for k, year in enumerate(range(2011, 2016)):
    current = half_up(trend_factor(-0.005, 2015 - year), 4)
    a_loss_trend.append(half_up(current * projected, 4))
    print(year, f"{a_loss_trend[-1]:.4f}  (book {book[k]:.4f})")
```

```text
projected period 2.25 years, factor 1.0113
2011 0.9912  (book 0.9912)
2012 0.9962  (book 0.9962)
2013 1.0012  (book 1.0012)
2014 1.0062  (book 1.0062)
2015 1.0113  (book 1.0113)
```

Appendix C, a long-tailed line with few claims, fits severity and an adjusted
frequency on annual points by accident year (Net Trend-1 and -2) and selects
one net trend of 8.9% for losses relative to premium; it rounds each trend
length to two places (6.83 years from accident year 2010 to 5/1/2017, the
average accident date of annual policies written from 5/1/2016):

```python
severity = [91_032, 56_271, 49_372, 100_052, 69_625, 166_764, 169_710, 78_630, 217_319, 299_634]
frequency = [2.92, 2.22, 4.31, 3.91, 5.87, 3.90, 3.78, 4.35, 4.35, 3.45]
fits = [("severity, 2005-2011", severity[:7], 17.0), ("frequency, 2005-2011", frequency[:7], 8.2),
        ("severity, 2005-2014", severity, 16.3), ("severity, 2010-2014", severity[5:], 15.2),
        ("frequency, 2005-2014", frequency, 3.5), ("frequency, 2010-2014", frequency[5:], -1.0)]
for name, values, pct in fits:
    fit = TrendFit.exponential(list(range(len(values))), values)
    print(f"{name:22s} {100 * fit.annual_change:6.2f}%  (book {pct}%)")
length = (Period.policy(year_fraction(2016, 5, 1), year_fraction(2017, 5, 1)).average_earned_date(1.0)
          - Period.calendar(2010.0, 2011.0).average_earned_date(1.0))
print(f"AY2010 trend length {length:.4f}, book 6.83: factor {trend_factor(0.089, 6.83):.4f}"
      f" (book 1.7902), unrounded {trend_factor(0.089, length):.4f}")
for years, b in [(5.83, 1.6439), (4.83, 1.5095), (3.83, 1.3862), (2.83, 1.2729)]:
    print(f"  {years} years: {trend_factor(0.089, years):.4f} (book {b})")
```

```text
severity, 2005-2011     16.95%  (book 17.0%)
frequency, 2005-2011     8.21%  (book 8.2%)
severity, 2005-2014     16.34%  (book 16.3%)
severity, 2010-2014     15.25%  (book 15.2%)
frequency, 2005-2014     3.54%  (book 3.5%)
frequency, 2010-2014    -1.04%  (book -1.0%)
AY2010 trend length 6.8333, book 6.83: factor 1.7902 (book 1.7902), unrounded 1.7907
  5.83 years: 1.6439 (book 1.6439)
  4.83 years: 1.5095 (book 1.5095)
  3.83 years: 1.3862 (book 1.3862)
  2.83 years: 1.2729 (book 1.2729)
```

The fits round to the printed tenths. Rounding the trend length to 6.83 is
what makes the book's 1.7902; the exact 6.8333 years give 1.7907.

Chapter 6 also explains why a constant trend in ground-up severity is
leveraged in an excess layer and damped under a basic limit (Tables 6.20 and
6.21); that is a property of the severity distribution, and
[`distributions-for-actuaries-claim-size.md`](distributions-for-actuaries-claim-size.md)
computes it with prospicio's severities.

## 6. Expenses, profit and the permissible loss ratio

Chapter 7 splits the cost of a policy into losses and LAE, fixed expenses
(the same per policy), variable expenses (a share of premium) and the profit
provision (a share of premium). The fundamental insurance equation then
gives the rate per exposure as the pure premium plus fixed expense, divided
by one minus the variable expense and profit ratios. `Provisions(variable,
profit)` holds `V` and `Q`, and `pure_premium_rate` applies the equation:

```python
from prospicio.pricing import Provisions, premium_based_expenses

p = Provisions(0.15, 0.05)
print(f"chapter 7: {p.pure_premium_rate(180, 20):.2f}  (book 250.00)")
all_variable = Provisions(0.15 + 0.08, 0.05)  # the $20 fixed treated as 8% of the average premium
for loss, right, wrong in [(135, "193.75", "187.50"), (180, "250.00", "250.00"), (225, "306.25", "312.50")]:
    r, w = p.pure_premium_rate(loss, 20), all_variable.pure_premium_rate(loss, 0)
    print(f"Table 7.3 loss {loss}: {r:.2f} (book {right})  all variable {w:.2f} (book {wrong})"
          f"  {round(100 * (w / r - 1), 1) + 0.0:+.1f}%")
fixed, variable = premium_based_expenses([0.059], [0.75])
print(f"Table 7.4: 5.9% general expense, 75% fixed: fixed {100 * fixed:.3f}% (book 4.4%),"
      f" variable {100 * variable:.3f}% (book 1.5%)")
```

```text
chapter 7: 250.00  (book 250.00)
Table 7.3 loss 135: 193.75 (book 193.75)  all variable 187.50 (book 187.50)  -3.2%
Table 7.3 loss 180: 250.00 (book 250.00)  all variable 250.00 (book 250.00)  +0.0%
Table 7.3 loss 225: 306.25 (book 306.25)  all variable 312.50 (book 312.50)  +2.0%
Table 7.4: 5.9% general expense, 75% fixed: fixed 4.425% (book 4.4%), variable 1.475% (book 1.5%)
```

Both parts are 4.425% and 1.475% before the book rounds them to a tenth.
Table 7.3 is the book's warning about the all variable expense method:
treating a fixed $20 as 8% of premium undercharges small risks and
overcharges large ones. The premium-based projection method (Table 7.4)
splits each historical expense ratio into a fixed and a variable part by a
fixed share, which is what `premium_based_expenses` does with several
categories at once. (Table 7.5, the exposure-based projection, divides fixed
dollars by exposures and trends them, plain arithmetic with `trend_factor`.
Its printed 1.4% uses 2015 earned premium of 545,250,000 where Table 7.4 shows
530,000,000.)

Chapter 8 turns this into the two methods of the indication. The pure premium
method computes the indicated rate per exposure from the projected pure
premium and fixed expense per exposure. The loss ratio method computes the
change to the current rates: the projected loss and LAE ratio plus the fixed
expense ratio, over the variable permissible loss ratio `1 - V - Q`, minus 1.
On consistent data the two agree:

```python
p = Provisions(0.25, 0.10)
print(f"pure premium method: {p.pure_premium_rate(300, 25):.2f}  (book 500.00)")
print(f"loss ratio method:   {100 * p.loss_ratio_change(0.65, 0.065):+.1f}%  (book +10%)")
print(f"permissible loss ratios: variable {p.variable_permissible_loss_ratio:.2f},"
      f" with 6.5% fixed {p.permissible_loss_ratio(0.065):.3f}")

exposure, current_rate, losses, fixed_dollars = 1_000, 480.0, 310_000.0, 26_000.0
q = Provisions(0.22, 0.04)
premium = exposure * current_rate
by_lr = current_rate * (1 + q.loss_ratio_change(losses / premium, fixed_dollars / premium))
by_pp = q.pure_premium_rate(losses / exposure, fixed_dollars / exposure)
print(f"same book both ways: {by_lr:.4f} and {by_pp:.4f}")
```

```text
pure premium method: 500.00  (book 500.00)
loss ratio method:   +10.0%  (book +10%)
permissible loss ratios: variable 0.65, with 6.5% fixed 0.585
same book both ways: 454.0541 and 454.0541
```

The last two lines use made-up figures to show the identity the book proves
in chapter 8: the loss ratio method's indicated factor times the current rate
is the pure premium method's indicated rate, as long as premium at current
rates is the exposure times the current average rate.

## 7. Credibility

Chapter 12 weights the indication with a complement when the experience is
too thin to stand alone. Its classical (limited fluctuation) method asks for
the number of claims at which the observed frequency is within `k` of its
mean with probability `p`, under a Poisson count: `(z / k) ** 2` with
`z` the normal quantile at `(1 + p) / 2`. Below the standard the credibility
is the square root rule. Table 12.1:

```python
from prospicio.credibility import LimitedFluctuation
from prospicio.pricing import credibility_weighted, trended_present_rates

book = [(0.05, 0.90, 1.645, 1082), (0.10, 0.90, 1.645, 271), (0.05, 0.95, 1.960, 1537),
        (0.10, 0.95, 1.960, 384), (0.05, 0.99, 2.575, 2652), (0.10, 0.99, 2.575, 663)]
print("  k    p   standard  book   with z = book's z")
for k, p, z, claims in book:
    n = LimitedFluctuation(p, k).frequency_standard()
    print(f"{k:4.0%} {p:4.0%}  {n:9.2f}  {claims:5,d}  {(z / k) ** 2:9.2f}")
lf = LimitedFluctuation(0.90, 0.05)
z = lf.credibility(100, round(lf.frequency_standard()))
print(f"100 claims: Z {z:.4f} (book 0.30); {credibility_weighted(200, round(z, 2), 300):.0f} (book 270)")
```

```text
  k    p   standard  book   with z = book's z
  5%  90%    1082.22  1,082    1082.41
 10%  90%     270.55    271     270.60
  5%  95%    1536.58  1,537    1536.64
 10%  95%     384.15    384     384.16
  5%  99%    2653.96  2,652    2652.25
 10%  99%     663.49    663     663.06
100 claims: Z 0.3040 (book 0.30); 270 (book 270)
```

Five rows round to the printed standards. The 99% rows use the book's
three-figure z of 2.575; the exact quantile is 2.5758, which gives 2,654
claims for k = 5% where the book prints 2,652 (and still 663 for k = 10%).
The exposure standards in the table's last column are these claim counts over
a 5% frequency. `LimitedFluctuation` also has severity and pure premium
standards (the book's formula with the severity's coefficient of variation),
not used in the appendices.

The book's second method is Bühlmann (least squares) credibility,
`Z = N / (N + K)` with `K` the expected process variance over the variance of
the hypothetical means. The book gives both variances and leaves their
estimation out of scope: with 21 observations, EVPV 2.00, VHM 0.50 and a
prior mean of $225, an observed $200 becomes $204. `credibility_premiums`
computes the Bühlmann-Straub premium at given variances; a tight prior on the
collective mean fixes it at the book's $225 (the function needs at least two
groups, so a second, arbitrary group is added):

```python
from prospicio.credibility import buhlmann_straub, credibility_premiums

cp = credibility_premiums([[200.0] * 21, [250.0] * 21], [[1.0] * 21] * 2,
                          within_variance=2.00, between_variances=[0.50],
                          collective_mean=(225.0, 1e-6))
print(f"K {2.00 / 0.50:.2f}  Z {cp['credibility'][0]:.2f} (book 0.84)"
      f"  estimate {cp['premiums'][0]:.2f} (book 204)")
```

```text
K 4.00  Z 0.84 (book 0.84)  estimate 204.00 (book 204)
```

Where the book stops, `buhlmann_straub` goes on: it estimates both variances
from a panel of groups observed over several periods with weights (exposures),
as R's `actuar::cm`, so `K` comes from the data. On made-up pure premiums for
three territories over four years, each territory's credibility is its total
exposure over that exposure plus `K`:

```python
pure_premiums = [[310.0, 240.0, 365.0, 290.0], [262.0, 330.0, 228.0, 301.0], [402.0, 296.0, 348.0, 441.0]]
exposures = [[120.0, 125.0, 130.0, 134.0], [80.0, 76.0, 82.0, 79.0], [30.0, 32.0, 28.0, 31.0]]
bs = buhlmann_straub(pure_premiums, exposures)
print(f"EVPV {bs.within_variance:,.0f}  VHM {bs.between_variance:,.1f}  K {bs.k:,.1f}")
for name, w, zz, mean, prem in zip("ABC", bs.group_weights, bs.credibility, bs.group_means, bs.premiums):
    print(f"{name}: exposure {w:,.0f}  Z {zz:.4f} = {w / (w + bs.k):.4f}  mean {mean:.2f}  premium {prem:.2f}")
print(f"collective mean {bs.collective_mean:.2f}")
```

```text
EVPV 207,908  VHM 600.7  K 346.1
A: exposure 509  Z 0.5952 = 0.5952  mean 301.59  premium 303.84
B: exposure 317  Z 0.4780 = 0.4780  mean 279.23  premium 293.80
C: exposure 121  Z 0.2590 = 0.2590  mean 371.46  premium 323.81
collective mean 307.15
```

So the two approaches differ in what they need, as the book says: classical
credibility needs only a claim count and a full standard and reaches 1, while
Bühlmann credibility needs the two variances, never reaches 1, and gives
`Z = 0` if the risks are assumed identical (VHM of 0). prospicio has both,
and Bayesian hierarchical credibility besides (`hierarchical_credibility`).

The complement matters as much as `Z`. For an overall indication the book
favours trended present rates: the change indicated last time but not taken,
trended forward. Chapter 12's example has a prior indication of +10%, +6%
taken and a 5% loss trend over two years:

```python
c = trended_present_rates(0.10, 0.06, trend_factor(0.05, 2.0))
print(f"complement {100 * c:+.2f}%, on a $200 rate: {200 * (1 + c):.0f}  (book 229)")
```

```text
complement +14.41%, on a $200 rate: 229  (book 229)
```

## Next

With every factor in hand, the companion guide
[`ratemaking-werner-modlin-indications.md`](ratemaking-werner-modlin-indications.md)
puts them together into the book's three overall indications (Appendices A to
C), and ends with the parts of the book prospicio does not cover and a table
of every place where the book's rounding moves a printed digit.
