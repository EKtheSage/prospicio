# Pricing Insurance Risk, Part I: risk measures in prospicio

<!-- site: execute -->

Part I of Stephen Mildenhall and John Major's *Pricing Insurance Risk: Theory
and Practice* (Wiley, 2022) is chapters 3 to 7. It measures risk before it
prices risk: how to represent a risk, what VaR and TVaR are, which properties
a risk measure needs, and how spectral (distortion) risk measures join these
ideas. This guide runs those chapters in prospicio and shows the book's value
beside each number it prints.

The book's text and tables belong to the authors, and this guide explains them
in new words. Table, example and exercise numbers are the book's. Related
guides:

- Part II, portfolio pricing: [Pricing Insurance Risk, Part II](pir-portfolio-pricing.md).
- The book's case studies, priced by five distortions and allocated to units:
  [the case studies guide](pricing-insurance-risk-case-studies.md).
- The same method on a ten-year example:
  [CAS Monograph 15 reproduced](capital-modeling-monograph-15.md).

## 1. Representing a risk: the Simple Discrete Example

Chapter 3 represents a risk as a random variable. Its expected value is a sum
of outcomes times probabilities, and also the area under the survival function
(the Lee diagram's horizontal slices).

The running example has two independent policies, losing 0, 8 or 10 and 0, 1
or 90, each with probabilities 1/2, 1/4 and 1/4. The total has nine values
with probabilities in sixteenths, so sixteen equally likely draws represent it
exactly:

```python
import pandas as pd
from great_tables import GT
from prospicio.distributions import Discrete, Sampled

OUTCOMES = [0, 1, 8, 9, 10, 11, 90, 98, 100]
SIXTEENTHS = [4, 2, 2, 1, 2, 1, 2, 1, 1]
discrete = Sampled([float(x) for x, k in zip(OUTCOMES, SIXTEENTHS) for _ in range(k)])


def two_forms(xs, probs):
    """E[X] as outcome times probability, and as the sum of S(x_j) times the step to x_(j+1)."""
    by_outcome = sum(x * p for x, p in zip(xs, probs))
    survival, by_slice = 1.0, 0.0
    for j in range(len(xs) - 1):
        survival -= probs[j]
        by_slice += survival * (xs[j + 1] - xs[j])
    return by_outcome, by_slice


probs = [k / 16 for k in SIXTEENTHS]
capped = Sampled([min(x, 80.0) for x in discrete.draws])
shifted = Sampled([x + 100.0 for x in discrete.draws])
two_forms_table = pd.DataFrame(
    [("E[X], Table 3.2", *two_forms(OUTCOMES, probs), discrete.mean(), 27.25),
     ("E[X ^ 80], Table 3.4", *two_forms([min(x, 80) for x in OUTCOMES], probs), capped.mean(), 23.625),
     ("E[X + 100], Table 3.3", *two_forms([0] + [x + 100 for x in OUTCOMES], [0.0] + probs), shifted.mean(), 127.25)],
    columns=["case", "by outcome", "by slice", "prospicio", "book"],
)
GT(two_forms_table, rowname_col="case").tab_header(
    "The expected value, two ways", "By outcome and by survival slice, then prospicio's mean and the book's value"
).fmt_number(columns=["by outcome", "by slice", "prospicio", "book"], decimals=3, drop_trailing_zeros=True)
```

```text
                       by outcome  by slice  prospicio    book
E[X], Table 3.2             27.25     27.25      27.25   27.25
E[X ^ 80], Table 3.4       23.625    23.625     23.625  23.625
E[X + 100], Table 3.3      127.25    127.25     127.25  127.25
```

A cap at assets of 80 changes only the outcomes. To add 100 to every outcome,
first add a zero outcome (the book's Step (1)): the survival form integrates
from zero and must include the slice from 0 to 100.

In prospicio, a `Sampled` holds simulations or equally likely scenarios, a
`Discrete` values and probabilities, and a `Grid` an evenly spaced grid, such
as the output of an FFT. All three have `mean`, `quantile` and `tvar`.
`Distortion.measure` prices a `Sampled` or a `Grid`.

## 2. Quantiles and VaR

Chapter 4 defines the p quantile as any x with P(X < x) <= p <= P(X <= x).
Where the distribution function jumps, there is one quantile. Where it is flat
at p, there is a range of quantiles, and VaR is the lower end. For a fair die
(Exercise 38), the 0.1 quantile is 1, and at p = 1/6 every value in [1, 2] is a
quantile:

```python
die = Discrete([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1 / 6] * 6)
print("die: 0.1 quantile", die.quantile(0.1), " 1/6 quantiles", die.quantile(1 / 6),
      "to", die.quantile_upper(1 / 6), "  (book 1; any value in [1, 2])")
```

```text
die: 0.1 quantile 1.0  1/6 quantiles 1.0 to 2.0   (book 1; any value in [1, 2])
```

Exercise 49 has ten equally likely outcomes 0, 1, 1, 1, 2, 3, 4, 8, 12 and 25.
Its 0.4 quantiles are [1, 2], so its VaR is 1:

```python
ex49 = Discrete([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 25.0], [0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BOOK_VAR = {0.4: 1.0, 0.41: 2.0, 0.85: None, 0.95: 25.0}   # the book: 0.4 quantiles [1, 2]; 25 for any p > 0.9
var_table = pd.DataFrame(
    [(f"{p:.2f}", ex49.quantile(p), ex49.quantile_upper(p), book) for p, book in BOOK_VAR.items()],
    columns=["p", "VaR", "upper quantile", "book VaR"],
)
GT(var_table, rowname_col="p").tab_header("Exercise 49: VaR and the upper quantile").fmt_number(
    columns=["VaR", "upper quantile", "book VaR"], decimals=1
).sub_missing(missing_text="")
```

```text
       VaR  upper quantile  book VaR
0.40   1.0             2.0       1.0
0.41   2.0             2.0       2.0
0.85  12.0            12.0          
0.95  25.0            25.0      25.0
```

The book gives 25 for any p > 0.9.

`quantile` gives the lower quantile, which is VaR in the book's sense, and
`quantile_upper` the upper end of the range. They are equal where the
distribution function jumps across p. `Sampled.quantile` and `Grid.quantile`
also give the lower quantile. In this guide, VaR is always the lower quantile.

## 3. Return periods and the PML of a hurricane model

VaR at p is also the 1 / (1 - p) year loss: the 99% VaR is the 100-year loss,
and the 99.6% VaR is the 250-year loss (Table 4.1). A probable maximum loss
(PML) is either the annual aggregate loss at that return period (an aggregate
VaR) or the loss from the largest single event (an occurrence PML).

Example 54 compares the two for US hurricanes: Poisson counts at 1.74
landfalls a year, and a lognormal severity with mean USD 5 billion and sigma
2.58. Table 4.3 gives, by return period, the occurrence PML, the severity
quantile, the aggregate VaR by FFT and the book's quick estimate of the
aggregate VaR from the severity alone (Eq. (4.5): the severity quantile at
1 - (1 - p) / 1.74). Each year's second row is the book's value, in USD
millions:

```python
import math

from prospicio.aggregate import fft, occurrence_pml
from prospicio.distributions import Grid, Lognormal, Poisson

SIGMA, RATE = 2.58, 1.74
hurricane = Lognormal(math.log(5000.0) - SIGMA**2 / 2, SIGMA)
annual, report = fft(Poisson(RATE), Grid.rounding(hurricane, 20.0, 2**21)[0], 2**21)

BOOK_43 = {  # Table 4.3: occurrence PML, severity VaR, aggregate VaR, estimated VaR
    2: (348, 179, 420, 763), 5: (3350, 1572, 3940, 3971), 10: (9783, 4892, 11080, 10467),
    20: (23420, 12490, 25620, 24107), 50: (62630, 35870, 66180, 63256),
    100: (121162, 72473, 125760, 121714), 250: (268049, 167935, 273900, 268492),
    1000: (787463, 520108, 794820, 787752),
}
rows = []
for years, book in BOOK_43.items():
    p = 1 - 1 / years
    ours = (occurrence_pml(Poisson(RATE), hurricane, 1 / years).loss, hurricane.quantile(p),
            annual.quantile(p), hurricane.quantile(1 - (1 - p) / RATE))
    rows += [(f"{years} years", *ours), (f"{years} years, book", *book)]
pml_table = pd.DataFrame(rows, columns=["return period", "occurrence PML", "severity VaR", "aggregate VaR", "estimate"])
GT(pml_table, rowname_col="return period").tab_header(
    "Table 4.3: return periods of a hurricane model", "USD millions"
).fmt_number(columns=["occurrence PML", "severity VaR", "aggregate VaR", "estimate"], decimals=0)
```

```text
                  occurrence PML  severity VaR  aggregate VaR  estimate
2 years                      348           179            420       763
2 years, book                348           179            420       763
5 years                    3,350         1,572          3,940     3,971
5 years, book              3,350         1,572          3,940     3,971
10 years                   9,783         4,892         11,080    10,467
10 years, book             9,783         4,892         11,080    10,467
20 years                  23,420        12,490         25,620    24,107
20 years, book            23,420        12,490         25,620    24,107
50 years                  62,630        35,870         66,180    63,256
50 years, book            62,630        35,870         66,180    63,256
100 years                121,162        72,473        125,760   121,714
100 years, book          121,162        72,473        125,760   121,714
250 years                268,049       167,935        273,900   268,492
250 years, book          268,049       167,935        273,900   268,492
1000 years               787,463       520,108        794,820   787,752
1000 years, book         787,463       520,108        794,820   787,752
```

Every value agrees to the printed unit. The grid has a step of USD 20 million
and 2^21 points; all the book's aggregate VaRs are multiples of 20.

One printed parameter is wrong:

```python
print(f"meanlog {hurricane.meanlog:.3f} (in millions; {hurricane.meanlog + math.log(1e6):.3f} in dollars)")
```

```text
meanlog 5.189 (in millions; 19.005 in dollars)
```

The book prints mu as 21.315, but a mean of USD 5 billion with sigma 2.58
needs 19.005 in dollars, and only that value reproduces the table's severity
column (a median of 179 million). Build the severity from its mean and sigma.

The figure plots the table:

```python
import matplotlib.pyplot as plt

years = list(BOOK_43)
probs = [1 - 1 / y for y in years]
fig, ax = plt.subplots()
ax.loglog(years, [hurricane.quantile(p) for p in probs], marker="o", color="0.75", label="severity VaR")
ax.loglog(years, [occurrence_pml(Poisson(RATE), hurricane, 1 / y).loss for y in years], marker="o",
          label="occurrence PML")
ax.loglog(years, [annual.quantile(p) for p in probs], marker="o", label="aggregate VaR")
ax.set_xlabel("return period (years)")
ax.set_ylabel("loss (USD millions)")
ax.set_title("Example 54: hurricane losses by return period")
ax.legend()
```

The occurrence PML is always below the aggregate VaR, and close to it at long
return periods, because a bad hurricane year is one bad hurricane. For the same
reason, the quick estimate is good in the tail (2% off at 1,000 years) and poor
for frequent events (82% off at 2 years).

## 4. VaR is not subadditive

A risk measure is subadditive if the risk of a sum never exceeds the sum of the
risks. The book shows three ways VaR fails this:

1. Dependence. Risks that are large in different scenarios can give a sum
   whose VaR exceeds the sum of their VaRs.
2. Skewness, with no dependence. The sum of two independent exponentials with
   mean 1 is a gamma with shape 2. Below p = 0.7, its VaR is more than twice
   the VaR of one exponential (Table 4.4).
3. A very thick tail. For a Pareto with tail index below 1 (no mean), VaR fails
   at every high enough p, because one large loss dominates the sum.

Table 4.4 checks the second way:

```python
from prospicio.distributions import Gamma

one, two = Gamma(1.0, 1.0), Gamma(2.0, 1.0)
BOOK_44 = {0.5: -0.2921, 0.6: -0.1897, 0.7: -0.03127, 0.8: 0.2246, 0.9: 0.7155, 0.95: 1.248, 0.99: 2.572,
           0.999: 4.582}
sum_table = pd.DataFrame(
    [(str(p), 2 * one.quantile(p), two.quantile(p), 2 * one.quantile(p) - two.quantile(p), book)
     for p, book in BOOK_44.items()],
    columns=["p", "2 VaR(X1)", "VaR(X1 + X2)", "difference", "book"],
)
GT(sum_table, rowname_col="p").tab_header(
    "Table 4.4: VaR of a sum against the sum of VaRs", "Exponential X1 with mean 1; X1 + X2 is a gamma with shape 2"
).fmt_number(columns=["2 VaR(X1)", "VaR(X1 + X2)"], decimals=3).fmt_number(
    columns="difference", decimals=4
).fmt_number(columns="book", decimals=5, drop_trailing_zeros=True)
```

```text
       2 VaR(X1)  VaR(X1 + X2)  difference      book
0.5        1.386         1.678     -0.2921   -0.2921
0.6        1.833         2.022     -0.1897   -0.1897
0.7        2.408         2.439     -0.0313  -0.03127
0.8        3.219         2.994      0.2246    0.2246
0.9        4.605         3.890      0.7155    0.7155
0.95       5.991         4.744      1.2476     1.248
0.99       9.210         6.638      2.5720     2.572
0.999     13.816         9.233      4.5821     4.582
```

The differences agree with Table 4.4 to its four significant figures. (In the
printed table, the difference for the 0.8 row is one line too low, under
0.999.) In the figure, twice the exponential VaR crosses the gamma VaR near
p = 0.7:

```python
import numpy as np
import matplotlib.pyplot as plt

grid = np.linspace(0.5, 0.999, 60)
fig, ax = plt.subplots()
ax.plot(grid, [2 * one.quantile(p) for p in grid], label="2 VaR(X1)")
ax.plot(grid, [two.quantile(p) for p in grid], label="VaR(X1 + X2)")
ax.set_xlabel("probability level p")
ax.set_ylabel("VaR")
ax.set_title("Table 4.4: VaR of a sum against the sum of VaRs")
ax.legend()
```

### The worst VaR of a sum: the Rearrangement Algorithm

With known marginals and unknown dependence, the worst possible VaR of the sum
is a useful bound. The book's Rearrangement Algorithm (RA, Puccetti and
Rüschendorf) finds it numerically:

1. Take the top 1 - p of each marginal.
2. Arrange each column in the opposite order to the sum of the other columns.
3. Repeat step 2 until the smallest row sum stops increasing.

Exercise 59 asks for the worst 99% VaR of three lognormals with mean 10 and CVs
1, 2 and 3, using 40 points of each at and above its 99th percentile:

```python
import numpy as np

STRATA = 40
columns = [[Lognormal.from_mean_cv(10.0, cv).quantile(0.99 + 0.01 * i / STRATA) for i in range(STRATA)]
           for cv in (1.0, 2.0, 3.0)]
top = np.array(columns).T


def rearrange(x, shift, eps=1e-6):
    """Puccetti and Rueschendorf's rearrangement: each column counter-monotonic to the rest,
    from a start with column j rotated by shift * j places."""
    x = np.column_stack([np.roll(c, shift * j) for j, c in enumerate(x.T)])
    best = -np.inf
    while True:
        for j in range(x.shape[1]):
            rest = x.sum(axis=1) - x[:, j]
            x[np.argsort(rest, kind="stable"), j] = np.sort(x[:, j])[::-1]
        low = x.sum(axis=1).min()
        if low - best < eps:
            return low
        best = low


worst = pd.DataFrame(
    [(f"99% VaR, CV {cv}", v, None) for cv, v in zip((1, 2, 3), top[0])]
    + [("comonotonic sum", top[0].sum(), 242.5)]
    + [(f"worst VaR, start {s}", float(rearrange(top, s)), 352.8) for s in (7, 13, 19)],
    columns=["quantity", "prospicio", "book"],
)
GT(worst, rowname_col="quantity").tab_header("Exercise 59: the worst 99% VaR of a sum").fmt_number(
    columns=["prospicio", "book"], decimals=1
).sub_missing(missing_text="")
```

```text
                     prospicio   book
99% VaR, CV 1             49.0       
99% VaR, CV 2             85.6       
99% VaR, CV 3            107.9       
comonotonic sum          242.5  242.5
worst VaR, start 7       353.8  352.8
worst VaR, start 13      353.6  352.8
worst VaR, start 19      352.9  352.8
```

The comonotonic sum agrees with the book. The RA is a heuristic and the book
does not give its start, so the worst VaR depends on the start: 352.9 to 353.8
here (column j rotated down by 7, 13 or 19 times j places, so the R version
prints the same numbers), against 352.8 in the book. The conclusion holds: the
worst arrangement raises the 99% VaR of the sum by 45% over perfect positive
dependence, so comonotonic VaR is not a worst case.

## 5. TVaR

TVaR at p is the average of the worst 1 - p of outcomes: the integral of VaR
from p to 1, divided by 1 - p. For a standard normal, the book puts TVaR above
VaR by 14.6% of the VaR at p = 0.99, 9% at 0.999 and 6.4% at 0.9999. Python's
`statistics.NormalDist` checks these values:

```python
from statistics import NormalDist

z = NormalDist()
BOOK_EXCESS = {0.99: 0.146, 0.999: 0.09, 0.9999: 0.064}
excess_table = pd.DataFrame(
    [(str(p), z.pdf(z.inv_cdf(p)) / (1 - p) / z.inv_cdf(p) - 1, book) for p, book in BOOK_EXCESS.items()],
    columns=["p", "TVaR / VaR - 1", "book"],
)
GT(excess_table, rowname_col="p").tab_header("TVaR above VaR for a standard normal").fmt_number(
    columns=["TVaR / VaR - 1", "book"], decimals=4, drop_trailing_zeros=True
)
```

```text
        TVaR / VaR - 1   book
0.99            0.1457  0.146
0.999           0.0896   0.09
0.9999          0.0644  0.064
```

Exercise 63 shows why the book prefers TVaR. A restaurant bill is lognormal
with mean 2,500 and CV 0.2. On a regular night you pay it. On a Pot Luck night
one of 25 diners, chosen at random, pays it, so you pay with probability 0.04.
The 95% VaR of the Pot Luck bill is zero, so VaR cash is no cash; the 95% TVaR
is 0.04 / 0.05 of the mean:

```python
bill = Lognormal.from_mean_cv(2500.0, 0.2)
var95 = bill.quantile(0.95)
tvar95 = var95 + (bill.mean() - bill.lev(var95)) / 0.05   # TVaR = VaR + E[(X - VaR)+] / (1 - p)
pot_luck = 0.04 / 0.05 * bill.mean()
shortfall = (bill.mean() - bill.lev(pot_luck)) / (1 - bill.cdf(pot_luck))
bill_table = pd.DataFrame(
    [("mu", bill.meanlog, 7.8044), ("sigma", bill.sdlog, 0.1980),
     ("regular night: VaR", var95, 3395), ("regular night: TVaR", tvar95, 3699),
     ("Pot Luck: TVaR", pot_luck, 2000), ("Pot Luck: P(enough)", bill.cdf(pot_luck), 0.152),
     ("Pot Luck: shortfall when not", shortfall, 624)],
    columns=["item", "prospicio", "book"],
)
table = GT(bill_table, rowname_col="item").tab_header("Exercise 63: the restaurant bill")
for rows, places in (([0, 1], 4), ([2, 3, 4, 6], 0), ([5], 3)):
    table = table.fmt_number(columns=["prospicio", "book"], rows=rows, decimals=places)
table
```

```text
                              prospicio    book
mu                               7.8044  7.8044
sigma                            0.1980  0.1980
regular night: VaR                3,395   3,395
regular night: TVaR               3,699   3,699
Pot Luck: TVaR                    2,000   2,000
Pot Luck: P(enough)               0.152   0.152
Pot Luck: shortfall when not        624     624
```

All seven values agree. TVaR = VaR + E[(X - VaR)+] / (1 - p), from the
quantile and the limited expected value (`lev`). You can also put the severity
on a grid and call `Grid.tvar`.

For a discrete distribution, TVaR takes a fraction of the observation at the
quantile (the book's algorithm in Section 4.3.6). The conditional tail
expectation (CTE), E[X | X >= VaR], takes the whole atom, so the two differ
(Exercise 49 continued):

```python
ten = Sampled([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 25.0])
BOOK_CTE = {0.75: 15.0, 0.85: 18.5, 0.9: 18.5}   # Exercise 49 continued
BOOK_TVAR = {0.75: None, 0.85: (12 * (0.9 - 0.85) + 2.5) / (1 - 0.85), 0.9: None}   # the book's TVaR on 0.8 <= p < 0.9
rows = []
for p, book in BOOK_CTE.items():
    var = ten.quantile(p)
    tail = [x for x in ten.draws if x >= var]
    rows.append((str(p), var, ten.tvar(p), BOOK_TVAR[p], sum(tail) / len(tail), book))
cte_table = pd.DataFrame(rows, columns=["p", "VaR", "TVaR", "book TVaR", "CTE", "book CTE"])
GT(cte_table, rowname_col="p").tab_header("Exercise 49 continued: TVaR and CTE").fmt_number(
    columns="VaR", decimals=1
).fmt_number(columns=["TVaR", "book TVaR", "CTE"], decimals=3).fmt_number(
    columns="book CTE", decimals=1
).sub_missing(missing_text="")
```

```text
       VaR    TVaR  book TVaR     CTE  book CTE
0.75   8.0  16.400             15.000      15.0
0.85  12.0  20.667     20.667  18.500      18.5
0.9   12.0  25.000             18.500      18.5
```

TVaR is continuous in p; CTE jumps at each observation, where the two are
equal. Use TVaR: it is subadditive.

## 6. VaR, TVaR and EPD in the case studies

Section 4.3.7 tabulates VaR, TVaR and the EPD capital for the case studies,
gross and net of reinsurance. The EPD capital is the assets at which the
expected policyholder deficit E[(X - a)+] is a given share of the expected
loss.

The Tame case has two gamma units with mean 50 and CVs 0.10 and 0.15. The net
book covers unit B with an aggregate 12.91 excess of 56.17 (its 80% to 99%
quantiles, Table 7.3). The grid is `aggregate`'s, with a step of 1/64 and 2^16
points, and the severity is renormalized as in the
[case studies guide](pricing-insurance-risk-case-studies.md). The EPD rows use
the ratio 1 - p:

```python
from prospicio.pricing import Portfolio

STEP, POINTS = 1 / 64, 2**16
LEVELS = (0.9, 0.95, 0.975, 0.99, 0.996, 0.999)
EPDS = (0.10, 0.05, 0.025, 0.01, 0.004, 0.001)


def discretize(severity):
    """A severity on aggregate's grid: rounded, the mass above the top spread by renormalizing."""
    probs = Grid.rounding(severity, STEP, 2 * POINTS)[0].probs[:POINTS]
    total = sum(probs)
    return Grid(STEP, [q / total for q in probs])


def total_grid(book):
    """The total of a portfolio as a grid."""
    probs = [0.0] * (round(max(book.totals) / STEP) + 1)
    for x, q in zip(book.totals, book.probs):
        probs[round(x / STEP)] += q
    return Grid(STEP, probs)


def epd_assets(grid, ratio):
    """The smallest assets with E[(X - a)+] at most ratio times E[X]."""
    one = Portfolio(["x"], [[i * STEP] for i, q in enumerate(grid.probs) if q > 0],
                    [q for q in grid.probs if q > 0])
    return one.assets_for_epd(ratio)


def capital_table(rows):
    """VaR, TVaR and EPD assets by level, for each named grid."""
    data = []
    for name, grid in rows.items():
        data.append((f"{name}, VaR", *[grid.quantile(p) for p in LEVELS]))
        data.append((f"{name}, TVaR", *[grid.tvar(p) for p in LEVELS]))
        data.append((f"{name}, EPD", *[epd_assets(grid, e) for e in EPDS]))
    return pd.DataFrame(data, columns=["measure", *map(str, LEVELS)])


a_unit = discretize(Gamma.from_mean_cv(50.0, 0.10))
b_unit = discretize(Gamma.from_mean_cv(50.0, 0.15))
b_net, _ = b_unit.map(lambda x: x - min(max(x - 56.17, 0.0), 12.91))
tame = Portfolio.from_independent(["A", "B"], [a_unit, b_unit])
tame_net = Portfolio.from_independent(["A", "B"], [a_unit, b_net])
tame_rows = {"A": a_unit, "B": b_unit, "Total": total_grid(tame), "B net": b_net, "Net": total_grid(tame_net)}
GT(capital_table(tame_rows), rowname_col="measure").tab_header(
    "Tame case: VaR, TVaR and EPD capital", "Assets by level p; the EPD ratio is 1 - p"
).fmt_number(columns=[str(p) for p in LEVELS], decimals=1)
```

```text
               0.9   0.95  0.975   0.99  0.996  0.999
A, VaR        56.5   58.5   60.3   62.4   64.3   66.9
A, TVaR       59.1   60.9   62.4   64.4   66.1   68.6
A, EPD        45.5   49.0   51.8   54.7   57.2   60.5
B, VaR        59.8   62.9   65.7   69.1   72.1   76.4
B, TVaR       64.0   66.7   69.2   72.3   75.2   79.2
B, EPD        46.5   51.1   54.8   59.0   62.7   67.6
Total, VaR   111.7  115.3  118.5  122.3  125.8  130.6
Total, TVaR  116.5  119.6  122.5  126.0  129.2  133.7
Total, EPD    90.6   97.4  102.5  108.0  112.6  118.5
B net, VaR    56.2   56.2   56.2   56.2   59.2   63.5
B net, TVaR   56.5   56.8   57.5   59.4   62.3   66.3
B net, EPD    45.3   49.2   52.0   54.2   55.4   56.1
Net, VaR     108.9  111.4  113.5  115.9  118.2  121.2
Net, TVaR    112.1  114.2  116.0  118.3  120.4  123.4
Net, EPD      89.7   96.0  100.5  104.8  108.2  112.3
```

Table 4.6 prints these values to the unit:

| Row | Book, gross A | B | Total | Book, net B | Net total |
|---|---|---|---|---|---|
| VaR | 56 58 60 62 64 67 | 60 63 66 69 72 76 | 112 115 119 122 126 131 | 56 56 56 56 59 64 | 109 111 113 116 118 121 |
| TVaR | 59 61 62 64 66 69 | 64 67 69 72 75 79 | 117 120 123 126 129 134 | 58 59 59 59 62 66 | 112 114 116 118 120 123 |
| EPD | 45 49 52 55 57 60 | 46 51 55 59 63 68 | 91 97 103 108 113 119 | 45 49 52 54 55 56 | 90 96 101 105 108 112 |

After rounding, every entry agrees (some of ours are on .5, such as the total
EPD at 0.1%, 118.5 against 119) except **the TVaR of net unit B at 90%, 95% and
97.5%**, where the book prints 58, 59 and 59 and prospicio gives 56.5, 56.8
and 57.5.

prospicio's values are correct, and the book's own VaR row shows why. Net B is
flat at 56.17 from its 80% to its 99% quantile, because the cover absorbs all
loss between them. TVaR at 90% is therefore 56.17 plus the expected excess over
56.17, divided by 0.1, and that excess is only the part above the exhaustion
point of the cover:

```python
excess = b_net.mean() - b_net.lev(56.171875)
print(f"E[(B net - 56.17)+] = {excess:.4f}; TVaR 90% = {56.171875 + excess / 0.1:.2f} (book 58)")
```

```text
E[(B net - 56.17)+] = 0.0323; TVaR 90% = 56.50 (book 58)
```

A printed 58 needs an expected excess of about 0.18. The net totals agree, so
the error is only in the net unit's column. Table 4.7 has the same error for
net Cat (below). We read it as a defect in the book's TVaR code for a unit with
a large probability mass under a stop loss, at the levels where VaR is on that
mass, not as a different convention.

The Cat/Non-Cat case has a gamma Non-Cat unit (mean 80, CV 0.15) and a
lognormal Cat unit (mean 20, CV 1). An aggregate cover of 79.64 excess of 41.11
applies to Cat (its 90% to 99.5% quantiles). Example 53 gives a VaR heuristic
for a thin-tailed unit added to a thick-tailed unit: the VaR of the total is
approximately the mean of the thin unit plus the VaR of the thick unit
(Table 4.2):

```python
non_cat = discretize(Gamma.from_mean_cv(80.0, 0.15))
cat = discretize(Lognormal.from_mean_cv(20.0, 1.0))
cat_net, _ = cat.map(lambda x: x - min(max(x - 41.11, 0.0), 79.64))
cnc = total_grid(Portfolio.from_independent(["Cat", "NonCat"], [cat, non_cat]))
cnc_net = total_grid(Portfolio.from_independent(["Cat", "NonCat"], [cat_net, non_cat]))

BOOK_42 = {0.5: 96.1, 0.9: 126.0, 0.99: 181.1, 0.999: 267.2, 0.9999: 394.0}
heuristic = pd.DataFrame(
    [(str(p), 80 + cat.quantile(p), cnc.quantile(p), book, (80 + cat.quantile(p)) / cnc.quantile(p) - 1)
     for p, book in BOOK_42.items()],
    columns=["p", "80 + VaR(Cat)", "VaR(total)", "book", "error"],
)
GT(heuristic, rowname_col="p").tab_header("Table 4.2: the VaR heuristic for a thin plus a thick tail").fmt_number(
    columns=["80 + VaR(Cat)", "VaR(total)", "book"], decimals=1
).fmt_number(columns="error", decimals=3, force_sign=True)
```

```text
        80 + VaR(Cat)  VaR(total)   book   error
0.5              94.1        96.1   96.1  -0.021
0.9             121.1       126.0  126.0  -0.039
0.99            178.1       181.1  181.1  -0.017
0.999           265.3       267.2  267.2  -0.007
0.9999          392.7       394.0  394.0  -0.003
```

From the 99% level up, the heuristic is within 2%. The gross rows of Table 4.7
agree. For net Cat, the VaR and the totals agree, but three TVaRs do not, for
the reason above:

```python
BOOK_47 = {  # Table 4.7, at LEVELS
    "net Cat VaR": (cat_net, "quantile", [41, 41, 41, 41, 49, 106]),
    "net Cat TVaR": (cat_net, "tvar", [43, 64, 75, 81, 92, 160]),
    "net TVaR": (cnc_net, "tvar", [131, 139, 146, 160, 181, 242]),
    "gross TVaR": (cnc, "tvar", [149, 167, 187, 218, 254, 321]),
}
rows = []
for name, (grid, method, book) in BOOK_47.items():
    rows += [(name, *[round(getattr(grid, method)(p)) for p in LEVELS]), (f"{name}, book", *book)]
GT(pd.DataFrame(rows, columns=["measure", *map(str, LEVELS)]), rowname_col="measure").tab_header(
    "Table 4.7: net and gross VaR and TVaR", "Cat/Non-Cat case, by level p"
).fmt_number(columns=[str(p) for p in LEVELS], decimals=0)
```

```text
                    0.9  0.95  0.975  0.99  0.996  0.999
net Cat VaR          41    41     41    41     49    106
net Cat VaR, book    41    41     41    41     49    106
net Cat TVaR         43    45     49    62     92    160
net Cat TVaR, book   43    64     75    81     92    160
net TVaR            131   139    146   160    181    242
net TVaR, book      131   139    146   160    181    242
gross TVaR          149   167    187   218    254    321
gross TVaR, book    149   167    187   218    254    321
```

Net Cat's TVaR at 95% cannot be 64 when its VaR is 41.1 from 90% to 99.5%: the
integral of VaR from 0.95 to 1 is at most 0.045 times 41.1, plus the tail above
the cover. The net book total uses the same net Cat and agrees at every level.

## 7. Properties of risk measures

Chapter 5 lists the properties a risk measure can have: monotone, translation
invariant, positive homogeneous, subadditive (these four make a measure
coherent), convex, comonotonic additive and law invariant. Four examples from
the book follow.

**Bid and ask: a sublinear price has a spread (Example 83).** Ins Co. sells a
Hot August policy that pays 1 if August is hot (probability 0.1), and a Cold
August policy that pays in the other 90% of years. It prices a Bernoulli risk
with probability s at the square root of s, a proportional hazard distortion
with rho 0.5. Both asks exceed the expected loss. The bid, the price at which
Ins Co. takes the Hot risk from another party, is minus the price of minus the
risk, and it is below the expected loss. `g_dual` is the dual
distortion 1 - g(1 - s), the bid side of the same price:

```python
from prospicio.risk import Distortion

ph = Distortion.proportional_hazard(0.5)
hot = Sampled([1.0] + [0.0] * 9)
cold = Sampled([0.0] + [1.0] * 9)
bid_ask = pd.DataFrame(
    [("ask, Hot", ph.measure(hot), 0.316), ("ask, Cold", ph.measure(cold), 0.95),
     ("price of minus Hot", ph.measure(Sampled([-x for x in hot.draws])), -0.05),
     ("bid, Hot", ph.g_dual(0.1), None)],
    columns=["price", "prospicio", "book"],
)
GT(bid_ask, rowname_col="price").tab_header("Example 83: bid and ask", "Expected loss of Hot: 0.1").fmt_number(
    columns="prospicio", decimals=3
).fmt_number(columns="book", decimals=3, drop_trailing_zeros=True).sub_missing(missing_text="")
```

```text
                    prospicio   book
ask, Hot                0.316  0.316
ask, Cold               0.949   0.95
price of minus Hot     -0.051  -0.05
bid, Hot                0.051
```

The book rounds -0.0513 to -0.05.

**First-order dominance is not state by state (Example 102, Bäuerle and
Müller).** Two states have probabilities 1/3 and 2/3. X is 3 then 1, and Y is 2
then 4. Each quantile of X is at most the same quantile of Y, so every
monotone, law-invariant risk measure ranks X below Y, but X is larger than Y in
the first state:

```python
x_bm = Discrete([3.0, 1.0], [1 / 3, 2 / 3])
y_bm = Discrete([2.0, 4.0], [1 / 3, 2 / 3])
grid_p = [0.1, 0.3, 0.5, 0.7, 0.9]
dominance = pd.DataFrame(
    [("VaR X", *[x_bm.quantile(p) for p in grid_p]), ("VaR Y", *[y_bm.quantile(p) for p in grid_p]),
     ("TVaR X", *[x_bm.tvar(p) for p in grid_p]), ("TVaR Y", *[y_bm.tvar(p) for p in grid_p])],
    columns=["measure", *map(str, grid_p)],
)
GT(dominance, rowname_col="measure").tab_header("Example 102: X ranks below Y at every level p").fmt_number(
    columns=[str(p) for p in grid_p], rows=[0, 1], decimals=1
).fmt_number(columns=[str(p) for p in grid_p], rows=[2, 3], decimals=3)
```

```text
          0.1    0.3    0.5    0.7    0.9
VaR X     1.0    1.0    1.0    3.0    3.0
VaR Y     2.0    2.0    4.0    4.0    4.0
TVaR X  1.741  1.952  2.333  3.000  3.000
TVaR Y  3.481  3.905  4.000  4.000  4.000
```

A law-invariant measure sees only these distributions, so the book compares
risks by their quantile functions, not by their states.

**Variance and the tail disagree (Exercise 103).** Risk can increase for every
risk-averse agent (higher stop loss premiums at every retention, or
equivalently higher TVaR at every level) while the variance does not. A uniform
on [0, 2] and a Pareto with tail index 4, rescaled to mean 1, have the same
mean. The uniform has the larger variance, the Pareto the larger tail:

```python
from prospicio.distributions import Pareto

uniform = Sampled([(i + 0.5) / 50_000 * 2 for i in range(50_000)])
pareto = Pareto(0.75, 4.0)   # minimum 0.75, tail index 4: mean 1, variance 1/8
spread = pd.DataFrame(
    [("mean", uniform.mean(), pareto.mean()), ("variance", 4 / 12, pareto.variance()),
     ("TVaR 0.99", uniform.tvar(0.99), pareto.quantile(0.99) * 4 / 3),
     ("TVaR 0.999", uniform.tvar(0.999), pareto.quantile(0.999) * 4 / 3)],
    columns=["statistic", "uniform", "Pareto"],
)
table = GT(spread, rowname_col="statistic").tab_header("Exercise 103: variance against the tail")
for rows, places in (([0, 2, 3], 3), ([1], 4)):
    table = table.fmt_number(columns=["uniform", "Pareto"], rows=rows, decimals=places)
table
```

```text
            uniform  Pareto
mean          1.000   1.000
variance     0.3333  0.1250
TVaR 0.99     1.990   3.162
TVaR 0.999    1.999   5.623
```

Variance ranks the uniform as riskier, and TVaR at 99% and above ranks the
Pareto as riskier (at tail index 4, the Pareto TVaR is 4/3 of its VaR). No
single number ranks all risks.

**A convex measure that is not coherent (Example 97).** The exponential
(entropic) risk measure is log E[exp(theta X)] / theta. It is convex and
translation invariant, but it is not positive homogeneous: when you double the
risk, the measure more than doubles.

```python
from prospicio.risk import entropic

entropy = pd.DataFrame(
    [(f"theta {theta}", entropic(discrete, theta), entropic(Sampled([2 * x for x in discrete.draws]), theta),
      entropic(Sampled([x + 10 for x in discrete.draws]), theta)) for theta in (0.05, 0.2)],
    columns=["theta", "rho(X)", "rho(2X)", "rho(X + 10)"],
)
GT(entropy, rowname_col="theta").tab_header("Example 97: the entropic measure").fmt_number(
    columns=["rho(X)", "rho(2X)", "rho(X + 10)"], decimals=2
)
```

```text
            rho(X)  rho(2X)  rho(X + 10)
theta 0.05   67.96   163.31        77.96
theta 0.2    89.45   188.12        99.45
```

It is the classical exponential premium principle (Part II), and it is
unbounded for any loss with a thick tail.

## 8. Spectral risk measures and prospicio's `Distortion`

The book's main object is the spectral risk measure (SRM), introduced in
Section 3.6 and developed in Chapters 10 and 11. An SRM is the expected loss
under a survival function distorted by a concave g with g(0) = 0 and g(1) = 1:
rho(X) = integral of g(S(x)) dx. The SRMs are exactly the coherent,
comonotonic additive, law invariant risk measures (Theorem 3). They have six
equivalent representations: a distortion of the survival function, a weighted
average of VaRs with increasing weights, a weighted average of TVaRs, a
maximum over probability scenarios, and two integral forms.

A prospicio `Distortion` is a g, and its methods give these representations:

| Representation | prospicio |
|---|---|
| g applied to the survival function | `d.g(s)`, `d.measure(x)` |
| dual (bid) distortion 1 - g(1 - s) | `d.g_dual(s)` |
| VaR weights: the weight on each sorted outcome | `d.weights(n)` |
| a mass at zero (charge for the maximum) | `d.mass` |
| a mixture of TVaRs | `Distortion.tvar`, `bitvar`, `weighted_tvar`, `mixture` |
| a probability scenario for a given risk | `weights` on the sorted sample, or the natural allocation in `Portfolio` |

The proportional hazard distortion with rho 0.5 prices the Simple Discrete
Example at 51.156 (Table 11.1). Its spectral weights are the risk-adjusted
probabilities of the sixteen sorted draws, and their weighted average of the
sorted draws gives the same price:

```python
w = ph.weights(16)
ordered = sorted(discrete.draws)
price_table = pd.DataFrame(
    [("PH(0.5) price", ph.measure(discrete), 51.156),
     ("weighted average of sorted outcomes", sum(a * b for a, b in zip(w, ordered)), None),
     ("TVaR(0.48732) price", Distortion.tvar(0.48732).measure(discrete), 51.156)],
    columns=["price", "prospicio", "book"],
)
GT(price_table, rowname_col="price").tab_header("The Simple Discrete Example under two SRMs").fmt_number(
    columns=["prospicio", "book"], decimals=3
).sub_missing(missing_text="")
```

```text
                                     prospicio    book
PH(0.5) price                           51.156  51.156
weighted average of sorted outcomes     51.156        
TVaR(0.48732) price                     51.156  51.156
```

The two SRMs (Exercise 183) give one price for this risk and different prices
for all other risks. The weights on the four worst draws, and the mass at zero:

```python
print([round(v, 4) for v in w[-4:]], ph.mass)
```

```text
[0.067, 0.0795, 0.1036, 0.25] 0.0
```

Example 176 takes the minimum of two distortions, the PH with rho 0.5 and a
constant cost of capital (CCoC) of 50%, g(s) = (2 s + 1) / 3. They cross at
s = 1/4: the minimum is the PH below it and the CCoC above it. The book finds
the TVaR representation: a mass of 2/3 on the mean (TVaR at 0), a mass of 1/12
at TVaR(0.75) and a density over the levels from 0.75 to 1.

```python
mixed = Distortion.minimum([ph, Distortion.ccoc(0.5)])
print("g at 0.01, 1/4, 1/2, 1:", [round(mixed.g(s), 4) for s in (0.01, 0.25, 0.5, 1.0)])
h = 1e-6
slope_top = (mixed.g(1.0) - mixed.g(1.0 - h)) / h
jump = (mixed.g(0.25) - mixed.g(0.25 - h)) / h - (mixed.g(0.25 + h) - mixed.g(0.25)) / h
print(f"slope at s = 1, the weight on the mean: {slope_top:.4f}  (book 2/3)")
print(f"drop in slope at s = 1/4, times 1/4: {0.25 * jump:.4f}  (book 1/12, the weight on TVaR 0.75)")
```

```text
g at 0.01, 1/4, 1/2, 1: [0.1, 0.5, 0.6667, 1.0]
slope at s = 1, the weight on the mean: 0.6667  (book 2/3)
drop in slope at s = 1/4, times 1/4: 0.0833  (book 1/12, the weight on TVaR 0.75)
```

The figure shows the three functions.

```python
import numpy as np
import matplotlib.pyplot as plt

ccoc = Distortion.ccoc(0.5)
sgrid = np.linspace(0.0, 1.0, 201)
fig, ax = plt.subplots()
ax.plot(sgrid, sgrid, color="0.75", label="no distortion, g(s) = s")
ax.plot(sgrid, [ph.g(v) for v in sgrid], label="PH, rho 0.5")
ax.plot(sgrid, [ccoc.g(v) for v in sgrid], label="CCoC, 50%")
ax.plot(sgrid, [mixed.g(v) for v in sgrid], linestyle="--", label="minimum of the two")
ax.set_xlabel("survival probability s")
ax.set_ylabel("distorted probability g(s)")
ax.set_title("Example 176: the minimum of two distortions")
ax.legend()
```

`Distortion.ccoc(r)` takes the cost of capital r, so `ccoc(0.5)` is the book's
(2 s + 1) / 3. `aggregate` builds the same function from the discount
d = r / (1 + r).

## 9. Choosing a risk measure

Chapter 6 separates two uses of a risk measure:

A **capital** risk measure sets the assets that a company holds: VaR, TVaR or
EPD, chosen by a regulator or rating agency and applied to the total. A
**pricing** risk measure sets the premium: an SRM or a classical principle.

Table 6.1 lists the measures and families. Their prospicio names:

| Measure | prospicio |
|---|---|
| mean, VaR, upper quantile | `.mean()`, `.quantile(p)`, `Discrete.quantile_upper` |
| TVaR | `.tvar(p)` on `Sampled`, `Grid` and `Discrete`, and `Distortion.tvar(p)` |
| EPD and EPD capital | `Portfolio.epd(a)`, `Portfolio.assets_for_epd(ratio)` |
| SRMs: CCoC, PH, Wang, dual, TVaR, bi-TVaR, beta, LEP and others | `Distortion` constructors |
| exponential (entropic), Esscher | `risk.entropic`, `risk.esscher` |
| classical premium principles | `pricing.classical_premium`, `calibrate_classical` |
| CoVaR, marginal expected shortfall | `risk.covar`, `risk.marginal_expected_shortfall` |

For practice, the book advises a simple, robust capital measure. VaR is both,
and it is the regulator's choice. A pricing measure must be coherent and
comonotonic additive, so that the layers and parts of one risk add up. These
conditions point to an SRM. Part II continues from here.
