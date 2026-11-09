# Basic ratemaking: Werner and Modlin reproduced

<!-- site: execute -->

Geoff Werner and Claudine Modlin wrote *Basic Ratemaking* (5th edition, CAS, 2016), the CAS study note on the overall rate level indication for personal and small commercial lines. Chapters 5 to 8 and 12 build it step by step (premium at current rates, loss development and trend, expenses and profit, credibility). Appendices A to C apply it to an auto, a homeowners and a medical malpractice book.

This guide follows those exhibits in prospicio's Python API
(`prospicio.pricing`, with `prospicio.credibility` and `prospicio.reserving`
for two steps) and shows the book's numbers beside prospicio's. The companion
guide,
[`ratemaking-werner-modlin-indications.md`](ratemaking-werner-modlin-indications.md),
runs Appendices A to C end to end and reproduces chapter 11's increased limits
and deductible figures.

The book's text and exhibits are the authors'. This guide recomputes the
numbers and explains them in new words. Table names are the book's. Money is in
dollars. The parity suite is `validation/tests/pricing.rs` (modules
`ratemaking` and `indication`). To rate a reinsurance layer, see
[`experience-and-exposure-rating.md`](experience-and-exposure-rating.md).

The book rounds each cell half up to its printed precision and uses the rounded
value in the next step. prospicio does not round. Where this changes a printed
digit, the code shows the exact value and applies the book's rounding by hand.
`knowledge/references/werner-modlin-ratemaking.md` lists the rounding rules.

## 1. The process, its data and its dates

An indication compares, for the period when the new rates are in effect, the
premium from the current rates with the losses, expenses and profit that this
premium must pay. The data (chapters 3 and 4) are exposures, premium (written,
earned, in force) and losses including loss adjustment expense (LAE), by
calendar, accident or policy year. Three adjustments follow: premium to the
current rate level, losses to ultimate, and both trended to the cost level of
the future period.

Trend needs the time between two average dates. The book puts each date on a
half-month grid, so June 30 and July 1 are the same point and November 15 is
ten and a half months into the year. `year_fraction` uses the same grid. A
`Period` is a calendar window (premium earned or losses that occur in it) or a
policy window (policies written in it). For a policy term it gives the average
written date and the average accident ("earned") date. `half_up` is the book's
rounding:

```python
from decimal import Decimal, ROUND_HALF_UP

from prospicio.pricing import Period, RateHistory, TrendFit, trend_factor, year_fraction


def half_up(x, dp):
    """Round half up to dp places, as the book's exhibits do (after clearing binary noise at 10 places)."""
    return float(Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


print(year_fraction(2013, 6, 30), year_fraction(2013, 7, 1), year_fraction(2011, 11, 15))
```

```text
2013.5 2013.5 2011.875
```

```python
import pandas as pd
from great_tables import GT

cy2011 = Period.calendar(2011.0, 2012.0)
py2011 = Period.policy(2011.0, 2012.0)
rates_2013 = Period.policy(2013.0, 2014.0)      # policies written while new rates apply
rates_2015 = Period.policy(2015.0, 2016.0)
lengths = pd.DataFrame({
    "length": ["premium, CY2011 to 1/1/2013, annual", "premium, CY2011 to 1/1/2013, 6-month",
               "losses, AY2011 to 1/1/2015, annual", "losses, AY2011 to 1/1/2015, 6-month",
               "losses, PY2011 to 1/1/2015, annual"],
    "prospicio": [
        rates_2013.average_written_date(1.0) - cy2011.average_written_date(1.0),
        rates_2013.average_written_date(0.5) - cy2011.average_written_date(0.5),
        rates_2015.average_earned_date(1.0) - cy2011.average_earned_date(1.0),
        rates_2015.average_earned_date(0.5) - cy2011.average_earned_date(0.5),
        rates_2015.average_earned_date(1.0) - py2011.average_earned_date(1.0),
    ],
    "book": [2.5, 2.25, 4.5, 4.25, 4.0],
})
(GT(lengths, rowname_col="length")
 .tab_header("Trend lengths, Tables 5.26 to 5.28 and 6.15 to 6.18", "Years between average dates")
 .fmt_number(["prospicio", "book"], decimals=2))
```

```text
                                      prospicio  book
premium, CY2011 to 1/1/2013, annual        2.50  2.50
premium, CY2011 to 1/1/2013, 6-month       2.25  2.25
losses, AY2011 to 1/1/2015, annual         4.50  4.50
losses, AY2011 to 1/1/2015, 6-month        4.25  4.25
losses, PY2011 to 1/1/2015, annual         4.00  4.00
```

The book trends premium between average written dates and losses between
average accident dates. On average, a calendar year's premium was written half
a term before the year's midpoint, and a policy year's losses occur half a term
after it. So the book's rule "effective date plus half the rate period, plus
half the term for losses" needs no special case.

## 2. Premium at the current rate level: the parallelogram method

The parallelogram method (Tables 5.15 to 5.23) restates historical premium at
the current rates. It assumes that policies are written evenly over time and
draws each rate change as a diagonal in the square of written date against
earned date. The share of a year's earned premium in each region of constant
rate level gives the year's average level. The on-level factor is the current
cumulative level divided by that average.

`RateHistory` takes the effective dates and the changes. `rate_level` takes a
period and a policy term and returns the shares, average level and factor.
Chapter 5's example has +5% on 7/1/2010, +10% on 1/1/2011 and -1% on 4/1/2012:

```python
ch5 = RateHistory([2010.5, 2011.0, 2012.25], [0.05, 0.10, -0.01])
print(f"current level {ch5.current_level:.5f}  (book 1.1435, Table 5.19)")
```

```text
current level 1.14345  (book 1.1435, Table 5.19)
```

```python
cases = [
    ("CY2011, annual policies (Tables 5.17-5.19)", Period.calendar(2011.0, 2012.0), 1.0, 1.0963, 1.0431),
    ("CY2011, 6-month policies (Table 5.20)", Period.calendar(2011.0, 2012.0), 0.5, 1.1288, 1.0130),
    ("PY2012, annual policies (Table 5.21)", Period.policy(2012.0, 2013.0), 1.0, 1.1464, 0.9975),
]
level_rows = []
for name, period, term, book_avg, book_factor in cases:
    lv = ch5.rate_level(period, term)
    shares = ", ".join(f"{round(s, 4):g}" for s in lv.shares)
    level_rows.append((name, shares, lv.average_level, book_avg, lv.on_level_factor, book_factor))
(GT(pd.DataFrame(level_rows, columns=["case", "shares", "average", "book_average", "factor", "book_factor"]),
    rowname_col="case")
 .tab_header("Chapter 5's rate changes: shares, average level and on-level factor")
 .tab_spanner("average level", ["average", "book_average"])
 .tab_spanner("on-level factor", ["factor", "book_factor"])
 .cols_label(average="prospicio", book_average="book", factor="prospicio", book_factor="book")
 .fmt_number("average", decimals=5)
 .fmt_number(["book_average", "factor", "book_factor"], decimals=4))
```

```text
                                                       shares  prospicio    book  prospicio    book
CY2011, annual policies (Tables 5.17-5.19)  0.125, 0.375, 0.5    1.09625  1.0963     1.0431  1.0431
CY2011, 6-month policies (Table 5.20)              0.25, 0.75    1.12875  1.1288     1.0130  1.0130
PY2012, annual policies (Table 5.21)               0.25, 0.75    1.14634  1.1464     0.9975  0.9975
```

Two averages (1.09625 and 1.12875) lie exactly on a rounding boundary, so the
table shows five places. The book rounds them up. It also rounds the cumulative
index to 1.1435 (exact 1.14345) before it divides, so its policy-year average
is 1.1464, not 1.14634.

A law change (Tables 5.22 and 5.23) applies to all policies in force on its
date, so its line in the square is vertical. Add the book's -5% benefit change
on 7/1/2011 as an `in_force` change:

```python
law = RateHistory(
    [2010.5, 2011.0, 2011.5, 2012.25],
    [0.05, 0.10, -0.05, -0.01],
    applies=["written", "written", "in_force", "written"],
)
lv = law.rate_level(Period.calendar(2011.0, 2012.0), 1.0)
print(f"current level {lv.current_level:.4f}  factor {lv.on_level_factor:.6f}  (book 1.0171)")
```

```text
current level 1.0863  factor 1.017084  (book 1.0171)
```

```python
cells = pd.DataFrame({
    "cell": [str(k) for k in range(1, 6)],
    "share": lv.shares,
    "level": lv.levels,
    "book_level": [1.0, 1.05, 0.9975, 1.155, 1.0973],
})
(GT(cells, rowname_col="cell")
 .tab_header("Table 5.23: the five cells of 2011", "Share of the year's earned premium and its rate level")
 .fmt_number("share", decimals=3, drop_trailing_zeros=True)
 .fmt_number(["level", "book_level"], decimals=4)
 .tab_spanner("level", ["level", "book_level"])
 .cols_label(level="prospicio", book_level="book"))
```

```text
   share  prospicio    book
1  0.125     1.0000  1.0000
2   0.25     1.0500  1.0500
3  0.125     0.9975  0.9975
4  0.125     1.1550  1.1550
5  0.375     1.0973  1.0973

```

### Appendix A: six-month auto policies

Appendix A's Current Rate Level exhibits apply six changes to six-month
policies, by calendar year: -5% on 4/1/2011, +10% on 7/1/2012, +5% on
10/1/2013, -2% on 7/1/2014, +5% on 10/1/2015 and +5% on 1/1/2016. The book
rounds the cumulative index to four places at each change (it ends at 1.1857,
where the exact chain gives 1.18552), rounds the average level to four places,
and divides the current level by the average level.

`RateHistory.from_levels` takes the printed cumulative indices and so gives the
book's chain exactly. `RateHistory` with the changes gives the exact chain:

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
```

```text
exact current level 1.185524  (book 1.1857)
```

```python
level_rows = []
for k, year in enumerate(range(2011, 2016)):
    cy = Period.calendar(float(year), float(year + 1))
    lv = printed_chain.rate_level(cy, 0.5)
    factor = 1.1857 / half_up(lv.average_level, 4)  # the book divides by its rounded average
    a_on_level.append(half_up(factor, 4))
    exact = exact_chain.rate_level(cy, 0.5).on_level_factor
    level_rows.append((str(year), lv.average_level, book_avg[k], a_on_level[-1], book_factor[k], exact))
(GT(pd.DataFrame(level_rows, columns=["year", "average", "book_average", "factor", "book_factor", "exact"]),
    rowname_col="year")
 .tab_header("Appendix A: on-level factors of six-month policies, by calendar year",
             "Exact is the chain without the book's rounding")
 .tab_spanner("average level", ["average", "book_average"])
 .tab_spanner("on-level factor", ["factor", "book_factor", "exact"])
 .cols_label(average="prospicio", book_average="book", factor="prospicio", book_factor="book")
 .fmt_number("average", decimals=5)
 .fmt_number(["book_average", "factor", "book_factor", "exact"], decimals=4))
```

```text
      prospicio    book  prospicio    book   exact
2011    0.97500  0.9750     1.2161  1.2161  1.2159
2012    0.97375  0.9738     1.2176  1.2176  1.2175
2013    1.04827  1.0483     1.1311  1.1311  1.1309
2014    1.08856  1.0886     1.0892  1.0892  1.0891
2015    1.07876  1.0788     1.0991  1.0991  1.0991
```

The 2012 average (0.97375) is a half-up boundary that binary floating point can
put just below, so `half_up` first rounds to ten places. Rounding the index at
each change moves the factors by up to 0.0002.

### Appendix C: annual medical malpractice policies

Appendix C has annual policies and three changes: -7.6% on 10/1/2011, +14.6% on
3/1/2014 and +13.6% on 7/1/2015. Its cumulative indices 0.9240, 1.0589 and
1.2029 are the exact chain, rounded. Its factors divide rounded figures, so the
code rounds the current and average levels to four places first:

```python
mm = RateHistory([2011.75, 2014 + 2 / 12, 2015.5], [-0.076, 0.146, 0.136])
c_on_level = []
book_factor = {2010: 1.2029, 2011: 1.2058, 2012: 1.2724, 2013: 1.3018, 2014: 1.2391, 2015: 1.1190}
level_rows = []
for year, book in book_factor.items():
    lv = mm.rate_level(Period.calendar(float(year), float(year + 1)), 1.0)
    factor = half_up(lv.current_level, 4) / half_up(lv.average_level, 4)
    c_on_level.append(half_up(factor, 4))
    shares = " ".join(f"{100 * s:.3f}%" for s in lv.shares)
    level_rows.append((str(year), c_on_level[-1], book, lv.on_level_factor, shares))
(GT(pd.DataFrame(level_rows, columns=["year", "factor", "book", "exact", "shares"]), rowname_col="year")
 .tab_header("Appendix C: on-level factors of annual policies, by calendar year",
             "Exact is the factor without rounding the levels")
 .cols_label(factor="prospicio")
 .fmt_number(["factor", "book", "exact"], decimals=4))
```

```text
      prospicio    book   exact                  shares
2010     1.2029  1.2029  1.2029                100.000%
2011     1.2058  1.2058  1.2058          96.875% 3.125%
2012     1.2724  1.2724  1.2724         28.125% 71.875%
2013     1.3018  1.3018  1.3019                100.000%
2014     1.2391  1.2391  1.2390         65.278% 34.722%
2015     1.1190  1.1190  1.1190  1.389% 86.111% 12.500%

```

Without rounding the levels, the 2013 and 2014 factors move by 0.0001.

## 3. Premium trend

Average premium also changes with the mix of business (newer cars, higher
amounts of insurance). The book projects average written premium at current
rate level with `(1 + trend) ** years`. `trend_factor` is that formula.

Two-step trending (Table 5.29) first moves the historical average to the latest
observed average written premium, then projects from there. The book rounds the
average earned premium to the cent and each factor to four places:

```python
avg_earned = half_up(1_440_788 / 1_947, 2)
step1 = 753.00 / avg_earned
step2 = trend_factor(-0.01, 1.625)
total = half_up(half_up(step1, 4) * half_up(step2, 4), 4)
steps = pd.DataFrame({
    "step": ["one step, 2% for 2.5 years", "average earned premium", "current step", "projected step",
             "total factor", "projected premium", "total factor, unrounded"],
    "prospicio": [trend_factor(0.02, 2.5), avg_earned, step1, step2, total, 1_440_788 * total,
                  753.00 / (1_440_788 / 1_947) * step2],
    "book": [1.0508, 740.00, 1.0176, 0.9838, 1.0011, 1_442_373, float("nan")],
})
(GT(steps, rowname_col="step")
 .tab_header("Premium trend: one step and two steps (Table 5.29)")
 .fmt_number(["prospicio", "book"], rows=[0, 2, 3, 4], decimals=4)
 .fmt_number(["prospicio", "book"], rows=[1], decimals=2)
 .fmt_number(["prospicio", "book"], rows=[5], decimals=0)
 .fmt_number("prospicio", rows=[6], decimals=6)
 .sub_missing(missing_text=""))
```

```text
                            prospicio       book
one step, 2% for 2.5 years     1.0508     1.0508
average earned premium         740.00     740.00
current step                   1.0176     1.0176
projected step                 0.9838     0.9838
total factor                   1.0011     1.0011
projected premium           1,442,373  1,442,373
total factor, unrounded      1.001078
```

Appendix A selects its premium trend from exponential fits (Premium Trend-1) to
average written premium at current rate level over the 12 months ending at each
quarter. With times in years and quarterly points a quarter apart,
`TrendFit.exponential` gives the annual change directly. The book regresses on
a quarter index and annualizes with the fourth power, the same fit:

```python
awp = [
    103.05, 103.58, 104.15, 104.72, 105.29, 105.84, 106.39, 106.92, 107.42, 107.96, 108.55, 109.16,
    109.75, 110.32, 110.87, 111.40, 111.95, 112.54, 113.10, 113.69, 114.25, 114.79, 115.35,
]
book = {20: 2.1, 16: 2.1, 12: 2.0, 8: 2.0, 6: 2.0, 4: 1.9}
trend_fits = pd.DataFrame({
    "points": [f"{n} points" for n in book],
    "prospicio": [100 * TrendFit.exponential([i / 4 for i in range(n)], awp[-n:]).annual_change for n in book],
    "book": list(book.values()),
})
(GT(trend_fits, rowname_col="points")
 .tab_header("Premium Trend-1: annual change by number of latest points", "Percent a year")
 .fmt_number("prospicio", decimals=3, pattern="{x}%")
 .fmt_number("book", decimals=1, pattern="{x}%"))
```

```text
           prospicio  book
20 points     2.062%  2.1%
16 points     2.058%  2.1%
12 points     2.023%  2.0%
8 points      2.018%  2.0%
6 points      1.994%  2.0%
4 points      1.947%  1.9%
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([i / 4 for i in range(len(awp))], awp, "o", color="0.75", label="average written premium")
for n in (20, 4):
    fit = TrendFit.exponential([i / 4 for i in range(n)], awp[-n:])
    start = (len(awp) - n) / 4
    ax.plot([start + i / 4 for i in range(n)], [fit.fitted(i / 4) for i in range(n)],
            label=f"latest {n} points: {100 * fit.annual_change:.1f}% a year")
ax.set_xlabel("years since the first quarter")
ax.set_ylabel("average written premium, current rates ($)")
ax.set_title("Premium Trend-1: two exponential fits to 23 quarters")
ax.legend()
```

The book selects 2.0%. In Premium Trend-3 it trends each calendar year's
average earned premium at current rate level to the latest 115.35, then at 2.0%
a year for two years, from 6/30/2015 (the average written date of the latest 12
months) to 6/30/2017. The exhibit's footnote prints "06/30/2017 to 06/30/2017",
a typo: the 2.0 years printed next to it and the text agree with 6/30/2015.

```python
ep_crl = [1_364_916.59, 1_405_728.94, 1_448_424.45, 1_492_177.86, 1_536_267.03]
exposures = [12_900, 13_020, 13_130, 13_258, 13_380]
book = [1.1342, 1.1116, 1.0879, 1.0663, 1.0452]
years = (Period.policy(2017.0, 2018.0).average_written_date(0.5)
         - Period.policy(2015.0, 2016.0).average_written_date(0.5))
projected = half_up(trend_factor(0.02, years), 4)
a_premium_trend = [half_up(half_up(115.35 / half_up(ep / x, 2), 4) * projected, 4)
                   for ep, x in zip(ep_crl, exposures)]
print("projected period", years, "years, factor", projected)
```

```text
projected period 2.0 years, factor 1.0404
```

```python
(GT(pd.DataFrame({"year": [str(y) for y in range(2011, 2016)], "prospicio": a_premium_trend, "book": book}),
    rowname_col="year")
 .tab_header("Premium Trend-3: trend factors to the future period",
             "Average earned premium at current rate level, by calendar year")
 .fmt_number(["prospicio", "book"], decimals=4))
```

```text
      prospicio    book
2011     1.1342  1.1342
2012     1.1116  1.1116
2013     1.0879  1.0879
2014     1.0663  1.0663
2015     1.0452  1.0452
```

## 4. Loss development

Chapter 6 develops recent accident years to ultimate by the chain ladder: link
ratios, an average and a selected factor for each age, and the product of the
selected factors. `prospicio.reserving` runs it. Table 6.11's triangle is in
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
averages = pd.DataFrame({
    "ages": ["15-27", "27-39", "39-51", "51-63", "63-75"],
    "exact": simple.ldf,
    "rounded": [half_up(f, 2) for f in rounded],
    "book": [1.50, 1.30, 1.10, 1.02, 1.00],
    "volume": [half_up(f, 2) for f in volume.ldf],
    "book_volume": [1.50, 1.29, 1.10, 1.02, 1.00],
})
(GT(averages, rowname_col="ages")
 .tab_header("Table 6.11: averages of the link ratios by age pair", "Months")
 .tab_spanner("arithmetic average", ["exact", "rounded", "book"])
 .tab_spanner("ratio of totals", ["volume", "book_volume"])
 .cols_label(exact="exact", rounded="rounded links", book="book", volume="prospicio", book_volume="book")
 .fmt_number("exact", decimals=4)
 .fmt_number(["rounded", "book", "volume", "book_volume"], decimals=2))
```

```text
        exact  rounded links  book  prospicio  book
15-27  1.4989           1.50  1.50       1.50  1.50
27-39  1.2932           1.30  1.30       1.29  1.29
39-51  1.1025           1.10  1.10       1.10  1.10
51-63  1.0175           1.02  1.02       1.02  1.02
63-75  0.9991           1.00  1.00       1.00  1.00
```

```python
import matplotlib.pyplot as plt

steps = ["15-27", "27-39", "39-51", "51-63", "63-75"]
fig, ax = plt.subplots()
for j in range(5):
    ratios = [r[j + 1] / r[j] for r in rows if r[j + 1] == r[j + 1]]
    ax.plot([j] * len(ratios), ratios, "o", color="0.75", label="link ratios" if j == 0 else None)
ax.plot([j - 0.1 for j in range(5)], simple.ldf, "o", label="arithmetic average")
ax.plot([j + 0.1 for j in range(5)], volume.ldf, "s", label="volume-weighted average")
ax.set_xticks(range(5), steps)
ax.set_xlabel("development age (months)")
ax.set_ylabel("link ratio (next age / this age)")
ax.set_title("Table 6.11: link ratios and their averages")
ax.legend()
```

The book averages link ratios that it first rounds to two places. At 27 to 39
months this gives 1.295, which rounds up to 1.30, where the exact average is
1.2932. The book's "ratio of total losses" is the volume-weighted average, the
chain ladder's default.

With the book's selections (the rounded arithmetic averages) as a selected
pattern, `ChainLadder` gives the factors to ultimate and the ultimate losses of
Table 6.13. The book rounds the factors to two places before it multiplies:

```python
selected = ChainLadder(ldf={15: 1.50, 27: 1.30, 39: 1.10, 51: 1.02, 63: 1.00}, ldf_kind="ldf")
fit = selected.fit(t611, "reported")
book_cdf = [1.00, 1.00, 1.02, 1.12, 1.46, 2.19]
book_ult = [2188, 2240, 2322, 2380, 2427, 2538]
cdf = [half_up(c, 2) for c in fit.cdf][::-1]  # oldest accident year first
ult = [half_up(lat * c, 0) for lat, c in zip(fit.latest, cdf)]
print(f"exact chain ladder total {fit.total_ultimate:,.1f}")
```

```text
exact chain ladder total 14,093.7
```

```python
nan = float("nan")
(GT(pd.DataFrame({
    "year": [str(y) for y in range(2002, 2008)] + ["total"],
    "cdf": cdf + [nan], "book_cdf": book_cdf + [nan],
    "ult": ult + [sum(ult)], "book_ult": book_ult + [sum(book_ult)],
}), rowname_col="year")
 .tab_header("Table 6.13: factors to ultimate and ultimate losses", "Thousands, by accident year")
 .tab_spanner("factor to ultimate", ["cdf", "book_cdf"])
 .tab_spanner("ultimate loss", ["ult", "book_ult"])
 .cols_label(cdf="prospicio", book_cdf="book", ult="prospicio", book_ult="book")
 .fmt_number(["cdf", "book_cdf"], decimals=2)
 .fmt_integer(["ult", "book_ult"])
 .sub_missing(missing_text=""))
```

```text
       prospicio  book  prospicio    book
2002        1.00  1.00      2,188   2,188
2003        1.00  1.00      2,240   2,240
2004        1.02  1.02      2,322   2,322
2005        1.12  1.12      2,380   2,380
2006        1.46  1.46      2,427   2,427
2007        2.19  2.19      2,538   2,538
total                      14,095  14,095
```

Unrounded, the 15-month factor is 2.1879, not 2.19, which gives a total of 14,093.7 thousand.

Appendix A's Loss Development exhibit applies the same method to reported
losses and paid ALAE for accident years 2009 to 2015, with the straight average
without the highest and lowest link ratio: `ChainLadder(average="simple",
drop_high=1, drop_low=1)`, where `n_links` counts the link ratios kept. The
book averages link ratios rounded to four places and chains its selections
rounded to four places, so the code passes that rounded selection as a selected
pattern:

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
exact = ChainLadder(average="simple", drop_high=1, drop_low=1).fit(ta, "reported")
hi_lo = [half_up((sum(c) - max(c) - min(c)) / (len(c) - 2), 4) for c in links]
fit = ChainLadder(ldf=dict(zip([15, 27, 39, 51], hi_lo)), ldf_kind="ldf").fit(ta, "reported")
a_development = [half_up(c, 4) for c in fit.cdf][::-1]  # accident years 2011 to 2015
(GT(pd.DataFrame({
    "ages": ["15-27", "27-39", "39-51", "51-63"],
    "all_exact": all_year,
    "all_rounded": [half_up(sum(c) / len(c), 4) for c in links],
    "all_book": [1.0704, 1.0380, 1.0113, 0.9898],
    "hl_exact": exact.ldf,
    "hl_rounded": hi_lo,
    "hl_book": [1.0665, 1.0279, 1.0208, 0.9799],
    "n_links": exact.n_links,
}), rowname_col="ages")
 .tab_header("Appendix A: link ratio averages by age pair", "Months; the average is of link ratios rounded to four places")
 .tab_spanner("all-year average", ["all_exact", "all_rounded", "all_book"])
 .tab_spanner("without highest and lowest", ["hl_exact", "hl_rounded", "hl_book", "n_links"])
 .cols_label(all_exact="exact", all_rounded="rounded links", all_book="book",
             hl_exact="exact", hl_rounded="rounded links", hl_book="book", n_links="links kept")
 .fmt_number(["all_exact", "all_rounded", "all_book", "hl_exact", "hl_rounded", "hl_book"], decimals=4)
 .fmt_integer("n_links"))
```

```text
        exact  rounded links    book   exact  rounded links    book  links kept
15-27  1.0703         1.0704  1.0704  1.0665         1.0665  1.0665           4
27-39  1.0380         1.0380  1.0380  1.0279         1.0279  1.0279           3
39-51  1.0113         1.0113  1.0113  1.0208         1.0208  1.0208           2
51-63  0.9898         0.9898  0.9898  0.9799         0.9799  0.9799           1
```

```python
(GT(pd.DataFrame({
    "age": ["15", "27", "39", "51", "63"],
    "exact": exact.cdf, "rounded": fit.cdf, "book": [1.0966, 1.0282, 1.0003, 0.9799, 1.0000],
}), rowname_col="age")
 .tab_header("Appendix A: factors to ultimate", "From each age in months; rounded is the chain of the rounded selection")
 .fmt_number(["exact", "rounded", "book"], decimals=4))
```

```text
     exact  rounded    book
15  1.0965   1.0966  1.0966
27  1.0281   1.0282  1.0282
39  1.0002   1.0003  1.0003
51  0.9799   0.9799  0.9799
63  1.0000   1.0000  1.0000
```

The rounded chain is the development column of
the companion guide's Appendix A indication. The 15-to-27 average is 1.07035 on
the rounded link ratios (printed 1.0704) and 1.0703 on the exact ones. Factors
below 1 at 51 months are valid: property damage claims develop downward when
the insurer recovers salvage and subrogation.

## 5. Loss trend

The book fits exponential trends to quarterly frequency, severity
and pure premium (Table 6.14, 12 months ending at each quarter) over the
latest 20, 16, 12, 8, 6 and 4 points:

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
points = (20, 16, 12, 8, 6, 4)
trend_rows = []
for name, (values, printed) in book.items():
    got = [100 * TrendFit.exponential([i / 4 for i in range(n)], values[-n:]).annual_change for n in points]
    trend_rows += [(name, *got), (name + ", book", *printed)]
trend_table = pd.DataFrame(trend_rows, columns=["series", *[str(n) for n in points]])
line = TrendFit.linear([i / 4 for i in range(20)], pp)
print(f"linear fit to 20 pure premiums: {line.annual_change:+.2f} a year (book -0.75)")
```

```text
linear fit to 20 pure premiums: -0.75 a year (book -0.75)
```

```python
point_columns = [str(n) for n in points]
(GT(trend_table, rowname_col="series")
 .tab_header("Table 6.14: annual loss trend by number of latest points", "Percent a year, exponential fit")
 .tab_spanner("latest points", point_columns)
 .fmt_number(point_columns, rows=[0, 2, 4], decimals=2)
 .fmt_number(point_columns, rows=[1, 3, 5], decimals=1))
```

```text
                       20     16     12      8      6      4
frequency           -1.72  -1.28  -0.69  -1.24  -0.87  -1.45
frequency, book      -1.7   -1.3   -0.7   -1.2   -0.9   -1.5
severity             0.54  -0.09  -0.21   1.24   2.52   3.31
severity, book        0.5   -0.1   -0.2    1.2    2.5    3.3
pure premium        -1.19  -1.39  -0.91  -0.07   1.59   1.91
pure premium, book   -1.2   -1.4   -0.9   -0.1    1.6    1.9
```

```python
import matplotlib.pyplot as plt

panels = [("frequency (claims per exposure)", freq), ("severity ($ per claim)", sev),
          ("pure premium ($ per exposure)", pp)]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (title, values) in zip(axes, panels):
    ax.plot([i / 4 for i in range(20)], values, "o", color="0.75", label="quarterly value")
    for n in (20, 8):
        fit = TrendFit.exponential([i / 4 for i in range(n)], values[-n:])
        start = (20 - n) / 4
        ax.plot([start + i / 4 for i in range(n)], [fit.fitted(i / 4) for i in range(n)],
                label=f"latest {n} points")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("years since the first quarter")
axes[0].legend(fontsize=8)
fig.tight_layout()
```

The frequencies must be the printed four-place values, because the book fits
what it prints. Unrounded counts over exposures move the 8-point frequency fit
from -1.24% to -1.30%.

The loss trend runs from the average accident date of each experience period to
that of the future policies (section 1, Tables 6.15 to 6.18). Appendix A trends
in two steps: -0.5% a year from the midpoint of each accident year to
6/30/2015, then +0.5% for the 2.25 years to 9/30/2017 (six-month policies
written in 2017):

```python
years = (Period.policy(2017.0, 2018.0).average_earned_date(0.5)
         - Period.calendar(2015.0, 2016.0).average_earned_date(0.5))
projected = half_up(trend_factor(0.005, years), 4)
book = [0.9912, 0.9962, 1.0012, 1.0062, 1.0113]
a_loss_trend = [half_up(half_up(trend_factor(-0.005, 2015 - year), 4) * projected, 4)
                for year in range(2011, 2016)]
print("projected period", years, "years, factor", projected)
```

```text
projected period 2.25 years, factor 1.0113
```

```python
(GT(pd.DataFrame({"year": [str(y) for y in range(2011, 2016)], "prospicio": a_loss_trend, "book": book}),
    rowname_col="year")
 .tab_header("Appendix A: loss trend factors to the future period", "By accident year")
 .fmt_number(["prospicio", "book"], decimals=4))
```

```text
      prospicio    book
2011     0.9912  0.9912
2012     0.9962  0.9962
2013     1.0012  1.0012
2014     1.0062  1.0062
2015     1.0113  1.0113
```

Appendix C is a long-tailed line with few claims. It fits severity and an
adjusted frequency on annual points by accident year (Net Trend-1 and -2) and
selects one net trend of 8.9% for losses relative to premium, with each trend
length rounded to two places. For accident year 2010 the length to 5/1/2017
(the average accident date of annual policies written from 5/1/2016) is 6.83
years:

```python
severity = [91_032, 56_271, 49_372, 100_052, 69_625, 166_764, 169_710, 78_630, 217_319, 299_634]
frequency = [2.92, 2.22, 4.31, 3.91, 5.87, 3.90, 3.78, 4.35, 4.35, 3.45]
fits = [("severity, 2005-2011", severity[:7], 17.0), ("frequency, 2005-2011", frequency[:7], 8.2),
        ("severity, 2005-2014", severity, 16.3), ("severity, 2010-2014", severity[5:], 15.2),
        ("frequency, 2005-2014", frequency, 3.5), ("frequency, 2010-2014", frequency[5:], -1.0)]
(GT(pd.DataFrame({
    "fit": [name for name, _, _ in fits],
    "prospicio": [100 * TrendFit.exponential(list(range(len(v))), v).annual_change for _, v, _ in fits],
    "book": [pct for _, _, pct in fits],
}), rowname_col="fit")
 .tab_header("Net Trend-1 and -2: annual trend of severity and frequency", "Percent a year")
 .fmt_number("prospicio", decimals=2, pattern="{x}%")
 .fmt_number("book", decimals=1, pattern="{x}%"))
```

```text
                      prospicio   book
severity, 2005-2011      16.95%  17.0%
frequency, 2005-2011      8.21%   8.2%
severity, 2005-2014      16.34%  16.3%
severity, 2010-2014      15.25%  15.2%
frequency, 2005-2014      3.54%   3.5%
frequency, 2010-2014     -1.04%  -1.0%
```

```python
length = (Period.policy(year_fraction(2016, 5, 1), year_fraction(2017, 5, 1)).average_earned_date(1.0)
          - Period.calendar(2010.0, 2011.0).average_earned_date(1.0))
print(f"AY2010 trend length {length:.4f} years, factor {trend_factor(0.089, length):.4f} unrounded")
```

```text
AY2010 trend length 6.8333 years, factor 1.7907 unrounded
```

```python
book_years = [(6.83, 1.7902), (5.83, 1.6439), (4.83, 1.5095), (3.83, 1.3862), (2.83, 1.2729)]
(GT(pd.DataFrame({
    "length": [f"{y} years" for y, _ in book_years],
    "prospicio": [trend_factor(0.089, y) for y, _ in book_years],
    "book": [b for _, b in book_years],
}), rowname_col="length")
 .tab_header("Trend factors at 8.9% a year", "Accident years 2010 to 2014 to 5/1/2017")
 .fmt_number(["prospicio", "book"], decimals=4))
```

```text
            prospicio    book
6.83 years     1.7902  1.7902
5.83 years     1.6439  1.6439
4.83 years     1.5095  1.5095
3.83 years     1.3862  1.3862
2.83 years     1.2729  1.2729
```

The book's 1.7902 uses the length rounded to 6.83, not the exact 6.8333 years.

A constant trend is leveraged in an excess layer and damped under a basic limit
(Tables 6.20 and 6.21).
[`distributions-for-actuaries-claim-size.md`](distributions-for-actuaries-claim-size.md)
computes it.

## 6. Expenses, profit and the permissible loss ratio

Chapter 7 divides a policy's cost into losses and LAE, fixed expenses, variable
expenses and the profit provision. Fixed expenses are the same for each policy.
Variable expenses and profit are shares of premium. The fundamental insurance
equation gives the rate per exposure: the pure premium plus the fixed expense,
divided by one minus the variable expense and profit ratios.
`Provisions(variable, profit)` holds `V` and `Q`, and `pure_premium_rate`
applies the equation:

```python
from prospicio.pricing import Provisions, premium_based_expenses

p = Provisions(0.15, 0.05)
print(f"chapter 7: {p.pure_premium_rate(180, 20):.2f}  (book 250.00)")
fixed, variable = premium_based_expenses([0.059], [0.75])
print(f"Table 7.4: 5.9% general expense, 75% fixed: fixed {100 * fixed:.3f}% (book 4.4%),"
      f" variable {100 * variable:.3f}% (book 1.5%)")
```

```text
chapter 7: 250.00  (book 250.00)
Table 7.4: 5.9% general expense, 75% fixed: fixed 4.425% (book 4.4%), variable 1.475% (book 1.5%)
```

```python
all_variable = Provisions(0.15 + 0.08, 0.05)  # the $20 fixed treated as 8% of the average premium
book_rows = [(135, 193.75, 187.50), (180, 250.00, 250.00), (225, 306.25, 312.50)]
rate_rows = []
for loss, right, wrong in book_rows:
    r, w = p.pure_premium_rate(loss, 20), all_variable.pure_premium_rate(loss, 0)
    rate_rows.append((str(loss), r, right, w, wrong, w / r - 1))
(GT(pd.DataFrame(rate_rows, columns=["loss", "rate", "book_rate", "flat", "book_flat", "change"]),
    rowname_col="loss")
 .tab_header("Table 7.3: the rate per exposure by expected loss", "Fixed $20 as such, or as 8% of the average premium")
 .tab_spanner("fixed expense as such", ["rate", "book_rate"])
 .tab_spanner("all variable", ["flat", "book_flat"])
 .cols_label(rate="prospicio", book_rate="book", flat="prospicio", book_flat="book", change="all variable vs as such")
 .fmt_number(["rate", "book_rate", "flat", "book_flat"], decimals=2)
 .fmt_percent("change", decimals=1, force_sign=True))
```

```text
     prospicio    book  prospicio    book  all variable vs as such
135     193.75  193.75     187.50  187.50                    -3.2%
180     250.00  250.00     250.00  250.00                     0.0%
225     306.25  306.25     312.50  312.50                    +2.0%

```

Table 7.3 shows the problem of the all variable expense method: if it treats a
fixed $20 as 8% of premium, it undercharges small risks and overcharges large
risks. The premium-based projection method (Table 7.4) divides each historical
expense ratio into a fixed part and a variable part by a fixed share.
`premium_based_expenses` does this for several categories at once.

Table 7.5, the exposure-based projection, divides fixed dollars by exposures
and trends them, with `trend_factor`. Its printed 1.4% uses 2015 earned premium
of 545,250,000, but Table 7.4 shows 530,000,000.

Chapter 8 gives two methods. The pure premium method gives the indicated rate
per exposure from the projected pure premium and fixed expense per exposure.
The loss ratio method gives the change to the current rates: the projected loss
and LAE ratio plus the fixed expense ratio, divided by the variable permissible
loss ratio `1 - V - Q`, minus 1. On consistent data they agree:

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

The last line uses made-up figures to show chapter 8's identity: the loss ratio
method's factor times the current rate is the pure premium method's rate, when
premium at current rates is the exposure times the current average rate.

## 7. Credibility

Chapter 12 weights the indication with a complement when the experience is too
thin to use alone. The classical (limited fluctuation) method assumes a Poisson
count. The standard is the number of claims at which the observed frequency is
within `k` of its mean with probability `p`: `(z / k) ** 2`, with `z` the
normal quantile at `(1 + p) / 2`. Below the standard, credibility follows the
square root rule. Table 12.1:

```python
from prospicio.credibility import LimitedFluctuation
from prospicio.pricing import credibility_weighted, trended_present_rates

book = [(0.05, 0.90, 1.645, 1082), (0.10, 0.90, 1.645, 271), (0.05, 0.95, 1.960, 1537),
        (0.10, 0.95, 1.960, 384), (0.05, 0.99, 2.575, 2652), (0.10, 0.99, 2.575, 663)]
(GT(pd.DataFrame({
    "k": [k for k, _, _, _ in book],
    "p": [p for _, p, _, _ in book],
    "standard": [LimitedFluctuation(p, k).frequency_standard() for k, p, _, _ in book],
    "book": [claims for _, _, _, claims in book],
    "book_z": [(z / k) ** 2 for k, _, z, _ in book],
}))
 .tab_header("Table 12.1: full-credibility standards for claim frequency", "Claims")
 .tab_spanner("standard", ["standard", "book", "book_z"])
 .cols_label(standard="prospicio", book="book", book_z="prospicio with the book's z")
 .fmt_percent(["k", "p"], decimals=0)
 .fmt_number(["standard", "book_z"], decimals=2)
 .fmt_integer("book"))
```

```text
  k    p  prospicio   book  prospicio with the book's z
 5%  90%   1,082.22  1,082                     1,082.41
10%  90%     270.55    271                       270.60
 5%  95%   1,536.58  1,537                     1,536.64
10%  95%     384.15    384                       384.16
 5%  99%   2,653.96  2,652                     2,652.25
10%  99%     663.49    663                       663.06
```

```python
lf = LimitedFluctuation(0.90, 0.05)
z = lf.credibility(100, round(lf.frequency_standard()))
print(f"100 claims: Z {z:.4f} (book 0.30); {credibility_weighted(200, round(z, 2), 300):.0f} (book 270)")
```

```text
100 claims: Z 0.3040 (book 0.30); 270 (book 270)
```

Five rows round to the printed standards. The 99% rows use the book's
three-figure z of 2.575. The exact quantile, 2.5758, gives 2,654 claims for k =
5% and still 663 for k = 10%. The book's exposure standards are these counts
divided by a 5% frequency. `LimitedFluctuation` also has severity and pure
premium standards, which the appendices do not use.

The book's second method is Bühlmann (least squares) credibility, `Z = N / (N +
K)`. `K` is the expected process variance (EVPV) divided by the variance of the
hypothetical means (VHM). The book gives both and does not estimate them: with
21 observations, EVPV 2.00, VHM 0.50 and a prior mean of $225, an observed $200
becomes $204. `credibility_premiums` calculates the Bühlmann-Straub premium at
given variances. A tight prior on the collective mean sets it at the book's
$225. The function needs at least two groups, so the code adds a second,
arbitrary group:

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

`buhlmann_straub` estimates both variances from a panel of groups, as R's
`actuar::cm` does, with exposures as weights. The example uses made-up pure
premiums for three territories over four years. A territory's credibility is
its exposure divided by exposure plus `K`:

```python
pure_premiums = [[310.0, 240.0, 365.0, 290.0], [262.0, 330.0, 228.0, 301.0], [402.0, 296.0, 348.0, 441.0]]
exposures = [[120.0, 125.0, 130.0, 134.0], [80.0, 76.0, 82.0, 79.0], [30.0, 32.0, 28.0, 31.0]]
bs = buhlmann_straub(pure_premiums, exposures)
print(f"EVPV {bs.within_variance:,.0f}  VHM {bs.between_variance:,.1f}  K {bs.k:,.1f}")
print(f"collective mean {bs.collective_mean:.2f}")
```

```text
EVPV 207,908  VHM 600.7  K 346.1
collective mean 307.15
```

```python
(GT(pd.DataFrame({
    "territory": list("ABC"),
    "exposure": bs.group_weights,
    "z": bs.credibility,
    "z_check": [w / (w + bs.k) for w in bs.group_weights],
    "mean": bs.group_means,
    "premium": bs.premiums,
}), rowname_col="territory")
 .tab_header("Bühlmann-Straub credibility by territory")
 .cols_label(z="Z", z_check="exposure / (exposure + K)", mean="mean", premium="premium")
 .fmt_integer("exposure")
 .fmt_number(["z", "z_check"], decimals=4)
 .fmt_number(["mean", "premium"], decimals=2))
```

```text
   exposure       Z  exposure / (exposure + K)    mean  premium
A       509  0.5952                     0.5952  301.59   303.84
B       317  0.4780                     0.4780  279.23   293.80
C       121  0.2590                     0.2590  371.46   323.81
```

Classical credibility needs only a claim count and a full standard, and it
reaches 1. Bühlmann credibility needs the two variances, never reaches 1, and
gives `Z = 0` for identical risks (VHM of 0). prospicio also has Bayesian
hierarchical credibility (`hierarchical_credibility`).

The complement matters as much as `Z`. For an overall indication the book
prefers trended present rates: the change indicated last time but not taken,
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

The companion guide
[`ratemaking-werner-modlin-indications.md`](ratemaking-werner-modlin-indications.md)
combines these factors into the book's three overall indications (Appendices A
to C) and ends with a table of each place where the book's rounding moves a
printed digit.
