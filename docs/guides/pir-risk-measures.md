# Pricing Insurance Risk, Part I: risk measures in prospicio

Part I of Stephen Mildenhall and John Major's *Pricing Insurance Risk: Theory
and Practice* (Wiley, 2022) is chapters 3 to 7. It measures risk before it
prices risk, and it covers four topics:

- how to represent a risk
- what quantiles, value at risk (VaR) and tail value at risk (TVaR) are, and
  how they behave
- which properties a risk measure needs
- how spectral (distortion) risk measures join these ideas

This guide runs those chapters in prospicio. Where the book prints a number,
the code computes it and prints the book's value beside it.

The book's text and tables belong to the authors, and this guide explains them
in new words. Table, example and exercise numbers are the book's. Related
guides:

- Part II, portfolio pricing: [Pricing Insurance Risk, Part II](pir-portfolio-pricing.md).
- The book's case studies, priced by five distortions and allocated to units:
  [the case studies guide](pricing-insurance-risk-case-studies.md).
- The same method on a ten-year example:
  [CAS Monograph 15 reproduced](capital-modeling-monograph-15.md).

## 1. Representing a risk: the Simple Discrete Example

Chapter 3 represents a risk as a random variable on a sample space. It computes
the expected value in two ways: as a sum of outcomes times their probabilities,
and as the area under the survival function. The second form is a sum of
horizontal slices (the Lee diagram).

The book's running example, the Simple Discrete Example, has two independent
policies. The first policy loses 0, 8 or 10, and the second loses 0, 1 or 90.
Each has probabilities 1/2, 1/4 and 1/4. The total has nine values, and each
probability is a multiple of 1/16. Thus sixteen equally likely draws represent
the total exactly:

```python
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
print("E[X]       ", two_forms(OUTCOMES, probs), " book 27.25 both ways")
print("E[X ^ 80]  ", two_forms([min(x, 80) for x in OUTCOMES], probs), " book 23.625 (Table 3.4)")
print("E[X + 100] ", two_forms([0] + [x + 100 for x in OUTCOMES], [0.0] + probs), " book 127.25 (Table 3.3)")
print("prospicio mean", discrete.mean(), "capped", Sampled([min(x, 80.0) for x in discrete.draws]).mean())
```

```text
E[X]        (27.25, 27.25)  book 27.25 both ways
E[X ^ 80]   (23.625, 23.625)  book 23.625 (Table 3.4)
E[X + 100]  (127.25, 127.25)  book 127.25 (Table 3.3)
prospicio mean 27.25 capped 23.625
```

Both sums give the book's 27.25 (Table 3.2). A cap at assets of 80 changes only
the outcome column. The survival probabilities do not change, and the limited
expected value is 23.625 (Table 3.4). To add 100 to every outcome, first add a
zero outcome (the book's Step (1)). The survival form integrates from zero, so
it must include the slice from 0 to 100 (Table 3.3).

In prospicio, a `Sampled` holds a risk given by simulations or equally likely
scenarios. A `Discrete` holds a risk given by values and probabilities. A
`Grid` holds a risk on an evenly spaced grid, such as the output of an FFT. All
three have `mean`, `quantile` and `tvar`. A distortion prices a `Sampled` or a
`Grid` with `Distortion.measure`.

## 2. Quantiles and VaR

Chapter 4 defines the p quantile as any x with P(X < x) <= p <= P(X <= x).
Where the distribution function jumps, there is one quantile. Where it is flat
at p, there is a range of quantiles, and VaR is the lower end (the lower
quantile). For a fair die (Exercise 38), the 0.1 quantile is 1, and at p = 1/6
every value in [1, 2] is a quantile. Exercise 49 has ten equally likely
outcomes 0, 1, 1, 1, 2, 3, 4, 8, 12 and 25. Its 0.4 quantiles are [1, 2], and
its VaR is 1:

```python
die = Discrete([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1 / 6] * 6)
print("die: 0.1 quantile", die.quantile(0.1), " 1/6 quantiles", die.quantile(1 / 6),
      "to", die.quantile_upper(1 / 6), "  (book 1; any value in [1, 2])")

ex49 = Discrete([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 25.0], [0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
for p in (0.4, 0.41, 0.85, 0.95):
    upper = ex49.quantile_upper(p)
    print(f"p {p:4.2f}: VaR {ex49.quantile(p):4.1f}  upper quantile {upper:4.1f}")
print("book: 0.4 quantiles [1, 2] with VaR 1; 0.41 VaR 2; 25 for any p > 0.9")
```

```text
die: 0.1 quantile 1.0  1/6 quantiles 1.0 to 2.0   (book 1; any value in [1, 2])
p 0.40: VaR  1.0  upper quantile  2.0
p 0.41: VaR  2.0  upper quantile  2.0
p 0.85: VaR 12.0  upper quantile 12.0
p 0.95: VaR 25.0  upper quantile 25.0
book: 0.4 quantiles [1, 2] with VaR 1; 0.41 VaR 2; 25 for any p > 0.9
```

`quantile` gives the lower quantile, which is VaR in the book's sense.
`quantile_upper` gives the upper end of the range. The two are equal where the
distribution function jumps across p. `Sampled.quantile` and `Grid.quantile`
give the same lower quantile. In this guide, VaR always means this lower
quantile.

## 3. Return periods and the PML of a hurricane model

VaR at p is also called the 1 / (1 - p) year loss. Thus the 99% VaR is the
100-year loss, and the 99.6% VaR is the 250-year loss (Table 4.1). A probable
maximum loss (PML) has two meanings:

- the annual aggregate loss at that return period (an aggregate VaR)
- the loss from the largest single event (an occurrence PML)

Example 54 compares the two for US hurricanes. The count is Poisson with 1.74
landfalls a year. The severity is lognormal with mean USD 5 billion and sigma
2.58. For each return period, Table 4.3 prints four values:

- the occurrence PML
- the severity quantile
- the aggregate VaR by FFT
- the book's quick estimate of the aggregate VaR from the severity alone
  (Eq. (4.5): the severity quantile at 1 - (1 - p) / 1.74)

Amounts are in USD millions:

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
print(f"meanlog {hurricane.meanlog:.3f} (in millions; {hurricane.meanlog + math.log(1e6):.3f} in dollars)")
print(f"{'years':>5s} {'occ PML':>17s} {'sev VaR':>17s} {'agg VaR':>17s} {'estimate':>17s}")
for years, book in BOOK_43.items():
    p = 1 - 1 / years
    ours = (occurrence_pml(Poisson(RATE), hurricane, 1 / years).loss, hurricane.quantile(p),
            annual.quantile(p), hurricane.quantile(1 - (1 - p) / RATE))
    print(f"{years:5d} " + " ".join(f"{o:8,.0f} ({b:6,})" for o, b in zip(ours, book)))
```

```text
meanlog 5.189 (in millions; 19.005 in dollars)
years           occ PML           sev VaR           agg VaR          estimate
    2      348 (   348)      179 (   179)      420 (   420)      763 (   763)
    5    3,350 ( 3,350)    1,572 ( 1,572)    3,940 ( 3,940)    3,971 ( 3,971)
   10    9,783 ( 9,783)    4,892 ( 4,892)   11,080 (11,080)   10,467 (10,467)
   20   23,420 (23,420)   12,490 (12,490)   25,620 (25,620)   24,107 (24,107)
   50   62,630 (62,630)   35,870 (35,870)   66,180 (66,180)   63,256 (63,256)
  100  121,162 (121,162)   72,473 (72,473)  125,760 (125,760)  121,714 (121,714)
  250  268,049 (268,049)  167,935 (167,935)  273,900 (273,900)  268,492 (268,492)
 1000  787,463 (787,463)  520,108 (520,108)  794,820 (794,820)  787,752 (787,752)
```

Every occurrence PML, severity VaR and estimate agrees with Table 4.3 to the
printed unit. Every aggregate VaR agrees too. The code computes them on a grid
with a step of 20 (USD 20 million) and 2^21 points. All the book's aggregate
VaRs are multiples of 20, which suggests the same step.

One printed parameter is wrong. The book gives the lognormal mu as 21.315. A
mean of USD 5 billion with sigma 2.58 needs mu = 19.005 in dollars. Only
mu = 19.005 reproduces the table's severity column (a median of 179 million).
Thus the table is correct and the printed mu is not. Build the severity from
its mean and sigma.

The occurrence PML is always below the aggregate VaR. At long return periods
the two are close, because a bad hurricane year is one bad hurricane. For the
same reason, the quick estimate from the severity alone is good in the tail and
poor for frequent events. Its error is 2% at 1,000 years, and it is 82% too
high at 2 years.

## 4. VaR is not subadditive

A risk measure is subadditive if the risk of a sum is never more than the sum
of the risks. Then pooling never seems to add risk. The book shows three ways
in which VaR is not subadditive:

1. Dependence. Two risks that are large in different scenarios can give a sum
   whose VaR is more than the sum of their VaRs.
2. Skewness, with no dependence. The sum of two independent exponentials with
   mean 1 is a gamma with shape 2. Below p = 0.7, its VaR is more than twice
   the VaR of one exponential (Table 4.4).
3. A very thick tail. For a Pareto with tail index below 1 (no mean), VaR is
   not subadditive at any p that is high enough. The reason is that one large
   loss dominates the sum.

The code checks the second way against Table 4.4:

```python
from prospicio.distributions import Gamma

one, two = Gamma(1.0, 1.0), Gamma(2.0, 1.0)
BOOK_44 = {0.5: -0.2921, 0.6: -0.1897, 0.7: -0.03127, 0.8: 0.2246, 0.9: 0.7155, 0.95: 1.248, 0.99: 2.572,
           0.999: 4.582}
for p, book in BOOK_44.items():
    gap = 2 * one.quantile(p) - two.quantile(p)
    print(f"p {p:<5}: 2 VaR(X1) {2 * one.quantile(p):6.3f}  VaR(X1 + X2) {two.quantile(p):6.3f}  "
          f"difference {gap:8.4f}  (book {book})")
```

```text
p 0.5  : 2 VaR(X1)  1.386  VaR(X1 + X2)  1.678  difference  -0.2921  (book -0.2921)
p 0.6  : 2 VaR(X1)  1.833  VaR(X1 + X2)  2.022  difference  -0.1897  (book -0.1897)
p 0.7  : 2 VaR(X1)  2.408  VaR(X1 + X2)  2.439  difference  -0.0313  (book -0.03127)
p 0.8  : 2 VaR(X1)  3.219  VaR(X1 + X2)  2.994  difference   0.2246  (book 0.2246)
p 0.9  : 2 VaR(X1)  4.605  VaR(X1 + X2)  3.890  difference   0.7155  (book 0.7155)
p 0.95 : 2 VaR(X1)  5.991  VaR(X1 + X2)  4.744  difference   1.2476  (book 1.248)
p 0.99 : 2 VaR(X1)  9.210  VaR(X1 + X2)  6.638  difference   2.5720  (book 2.572)
p 0.999: 2 VaR(X1) 13.816  VaR(X1 + X2)  9.233  difference   4.5821  (book 4.582)
```

The differences agree with Table 4.4 to the four significant figures that it
prints. In the printed table, the difference for the 0.8 row is one line too
low, under 0.999. The columns to its left show which value belongs to which
row.

### The worst VaR of a sum: the Rearrangement Algorithm

If you know the marginal distributions but not the dependence, the worst
possible VaR of the sum is a useful bound. The book's Rearrangement Algorithm
(RA, Puccetti and Rüschendorf) finds this bound numerically:

1. Take the top 1 - p of each marginal.
2. Arrange each column in the opposite order to the sum of the other columns.
3. Do step 2 again until the smallest row sum stops increasing.

Exercise 59 asks for the worst 99% VaR of three lognormals with mean 10 and CVs
1, 2 and 3. It uses 40 points of each lognormal at and above its 99th
percentile. The code runs the RA in numpy on prospicio's lognormal quantiles:

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


print(f"99% VaRs {top[0].round(1)}, comonotonic sum {top[0].sum():.1f}  (book 242.5)")
print("worst VaR by RA, three starts:", [round(float(rearrange(top, s)), 1) for s in (7, 13, 19)], " (book 352.8)")
```

```text
99% VaRs [ 49.   85.6 107.9], comonotonic sum 242.5  (book 242.5)
worst VaR by RA, three starts: [353.8, 353.6, 352.9]  (book 352.8)
```

The VaRs of the three lognormals and their comonotonic sum, 242.5, agree with
the book. The worst VaR found is 352.9 to 353.8, and it changes with the start.
The book gives 352.8. The algorithm is a heuristic, its result depends on the
start arrangement, and the book does not give its start.

The starts here are fixed rotations: column j moves down by 7, 13 or 19 times j
places. Fixed starts, not random shuffles, make the R version of this guide
print the same numbers. No start gives exactly 352.8, but the best is within
0.1. The conclusion agrees with the book. The worst arrangement increases the
99% VaR of the sum by 45% over perfect positive dependence, so comonotonic VaR
is not a worst case.

## 5. TVaR

TVaR at p is the average of the worst 1 - p of outcomes. It equals the integral
of VaR from p to 1, divided by 1 - p. For a normal distribution, TVaR is above
VaR by a fixed number of standard deviations. The book gives this excess as
14.6% of the standard normal VaR at p = 0.99, 9% at 0.999 and 6.4% at 0.9999.
Python's `statistics.NormalDist` checks these values:

```python
from statistics import NormalDist

z = NormalDist()
for p, book in ((0.99, 0.146), (0.999, 0.09), (0.9999, 0.064)):
    var = z.inv_cdf(p)
    print(f"p {p}: TVaR / VaR - 1 = {z.pdf(var) / (1 - p) / var - 1:.4f}  (book {book})")
```

```text
p 0.99: TVaR / VaR - 1 = 0.1457  (book 0.146)
p 0.999: TVaR / VaR - 1 = 0.0896  (book 0.09)
p 0.9999: TVaR / VaR - 1 = 0.0644  (book 0.064)
```

Exercise 63 shows why the book prefers TVaR to VaR. A restaurant bill is
lognormal with mean 2,500 and CV 0.2. On a regular night you pay the bill. On a
Pot Luck night, one of 25 diners pays it, chosen at random, so you pay with
probability 0.04. The 95% VaR of the Pot Luck bill is zero, so VaR cash is no
cash. TVaR cash is the 95% TVaR, which is 0.04 / 0.05 of the mean:

```python
bill = Lognormal.from_mean_cv(2500.0, 0.2)
var95 = bill.quantile(0.95)
tvar95 = var95 + (bill.mean() - bill.lev(var95)) / 0.05   # TVaR = VaR + E[(X - VaR)+] / (1 - p)
pot_luck = 0.04 / 0.05 * bill.mean()
shortfall = (bill.mean() - bill.lev(pot_luck)) / (1 - bill.cdf(pot_luck))
print(f"mu {bill.meanlog:.4f} sigma {bill.sdlog:.4f}   (book 7.8044, 0.1980)")
print(f"regular night: VaR {var95:,.0f} TVaR {tvar95:,.0f}   (book 3,395 and 3,699)")
print(f"Pot Luck: TVaR {pot_luck:,.0f}, enough with probability {bill.cdf(pot_luck):.3f}, "
      f"shortfall when not {shortfall:,.0f}   (book 2,000, 0.152, 624)")
```

```text
mu 7.8044 sigma 0.1980   (book 7.8044, 0.1980)
regular night: VaR 3,395 TVaR 3,699   (book 3,395 and 3,699)
Pot Luck: TVaR 2,000, enough with probability 0.152, shortfall when not 624   (book 2,000, 0.152, 624)
```

All six numbers agree. The code computes TVaR from the quantile and the limited
expected value (`lev`): TVaR = VaR + E[(X - VaR)+] / (1 - p). You can also put
the severity on a grid and call `Grid.tvar`.

For a discrete distribution, TVaR takes a fraction of the observation at the
quantile (the book's algorithm in Section 4.3.6). The conditional tail
expectation (CTE), E[X | X >= VaR], takes the whole atom, so the two are
different. Exercise 49 continued gives TVaR and CTE for the ten outcomes:

```python
ten = Sampled([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 25.0])
BOOK_CTE = {0.75: 15.0, 0.85: 18.5, 0.9: 18.5}   # Exercise 49 continued
for p, book in BOOK_CTE.items():
    var = ten.quantile(p)
    tail = [x for x in ten.draws if x >= var]
    print(f"p {p:<4}: VaR {var:4.1f}  TVaR {ten.tvar(p):6.3f}  CTE {sum(tail) / len(tail):6.3f} (book {book})")
print(f"book's TVaR for 0.8 <= p < 0.9 at p = 0.85: {(12 * (0.9 - 0.85) + 2.5) / (1 - 0.85):.3f}")
```

```text
p 0.75: VaR  8.0  TVaR 16.400  CTE 15.000 (book 15.0)
p 0.85: VaR 12.0  TVaR 20.667  CTE 18.500 (book 18.5)
p 0.9 : VaR 12.0  TVaR 25.000  CTE 18.500 (book 18.5)
book's TVaR for 0.8 <= p < 0.9 at p = 0.85: 20.667
```

TVaR is continuous in p, and CTE jumps at each observation. The two are equal
at the observations. Use TVaR, because TVaR is subadditive.

## 6. VaR, TVaR and EPD in the case studies

Section 4.3.7 tabulates three capital amounts for the case studies: VaR, TVaR
and the EPD capital. The EPD capital is the assets at which the expected
policyholder deficit E[(X - a)+] is a given share of the expected loss. The
tables show each unit, the sum of the units and the total, gross and net of
reinsurance.

The Tame case has two gamma units with mean 50 and CVs 0.10 and 0.15. The net
book has an aggregate cover of 12.91 excess of 56.17 on unit B (its 80% to 99%
quantiles, Table 7.3). The code uses the grid of `aggregate`, with a step of
1/64 and 2^16 points. It renormalizes the severity as the
[case studies guide](pricing-insurance-risk-case-studies.md) does:

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


def table(rows):
    print(f"{'':6s}" + "".join(f"{lvl:>7}" for lvl in LEVELS))
    for name, grid in rows.items():
        print(f"{name:6s} VaR " + " ".join(f"{grid.quantile(p):6.1f}" for p in LEVELS))
        print(f"{'':6s}TVaR " + " ".join(f"{grid.tvar(p):6.1f}" for p in LEVELS))
        print(f"{'':6s} EPD " + " ".join(f"{epd_assets(grid, e):6.1f}" for e in EPDS))


a_unit = discretize(Gamma.from_mean_cv(50.0, 0.10))
b_unit = discretize(Gamma.from_mean_cv(50.0, 0.15))
b_net, _ = b_unit.map(lambda x: x - min(max(x - 56.17, 0.0), 12.91))
tame = Portfolio.from_independent(["A", "B"], [a_unit, b_unit])
tame_net = Portfolio.from_independent(["A", "B"], [a_unit, b_net])
table({"A": a_unit, "B": b_unit, "Total": total_grid(tame), "B net": b_net, "Net": total_grid(tame_net)})
```

```text
          0.9   0.95  0.975   0.99  0.996  0.999
A      VaR   56.5   58.5   60.3   62.4   64.3   66.9
      TVaR   59.1   60.9   62.4   64.4   66.1   68.6
       EPD   45.5   49.0   51.8   54.7   57.2   60.5
B      VaR   59.8   62.9   65.7   69.1   72.1   76.4
      TVaR   64.0   66.7   69.2   72.3   75.2   79.2
       EPD   46.5   51.1   54.8   59.0   62.7   67.6
Total  VaR  111.7  115.3  118.5  122.3  125.8  130.6
      TVaR  116.5  119.6  122.5  126.0  129.2  133.7
       EPD   90.6   97.4  102.5  108.0  112.6  118.5
B net  VaR   56.2   56.2   56.2   56.2   59.2   63.5
      TVaR   56.5   56.8   57.5   59.4   62.3   66.3
       EPD   45.3   49.2   52.0   54.2   55.4   56.1
Net    VaR  108.9  111.4  113.5  115.9  118.2  121.2
      TVaR  112.1  114.2  116.0  118.3  120.4  123.4
       EPD   89.7   96.0  100.5  104.8  108.2  112.3
```

Table 4.6 prints these values to the unit. Its EPD rows go from an EPD ratio of
10% down to 0.1%:

| Row | Book, gross A | B | Total | Book, net B | Net total |
|---|---|---|---|---|---|
| VaR | 56 58 60 62 64 67 | 60 63 66 69 72 76 | 112 115 119 122 126 131 | 56 56 56 56 59 64 | 109 111 113 116 118 121 |
| TVaR | 59 61 62 64 66 69 | 64 67 69 72 75 79 | 117 120 123 126 129 134 | 58 59 59 59 62 66 | 112 114 116 118 120 123 |
| EPD | 45 49 52 55 57 60 | 46 51 55 59 63 68 | 91 97 103 108 113 119 | 45 49 52 54 55 56 | 90 96 101 105 108 112 |

After rounding, every entry agrees except one group. Some of our values are on
.5, for example the total EPD at 0.1%, 118.5 against 119. The exception is
**the TVaR of net unit B at 90%, 95% and 97.5%**. There the book prints 58, 59
and 59, and prospicio gives 56.5, 56.8 and 57.5.

prospicio's values are correct, and the book's own VaR row shows why. Net B is
flat at 56.17 from its 80% quantile to its 99% quantile, because the cover
absorbs all loss between them. Thus TVaR at 90% is 56.17 plus the expected
excess over 56.17, divided by 0.1. That excess is only the part above the
exhaustion point of the cover, 0.032:

```python
excess = b_net.mean() - b_net.lev(56.171875)
print(f"E[(B net - 56.17)+] = {excess:.4f}; TVaR 90% = {56.171875 + excess / 0.1:.2f} (book 58)")
```

```text
E[(B net - 56.17)+] = 0.0323; TVaR 90% = 56.50 (book 58)
```

For a printed 58, the expected excess must be approximately 0.18. The net
totals agree, so the error is only in the column for the net unit. Table 4.7
has the same error for net Cat in the Cat/Non-Cat case (below). The book's TVaR
is too high for a unit with a large probability mass under a stop loss. The
error occurs at the levels where VaR is on that mass. We read this as a defect in the TVaR that the
book's code computed for such distributions, not as a different convention.

The Cat/Non-Cat case has a gamma Non-Cat unit (mean 80, CV 0.15) and a
lognormal Cat unit (mean 20, CV 1). An aggregate cover of 79.64 excess of 41.11
applies to Cat (its 90% to 99.5% quantiles). Example 53 gives a VaR heuristic
for a thin-tailed unit added to a thick-tailed unit. The VaR of the total is
approximately the mean of the thin unit plus the VaR of the thick unit
(Table 4.2):

```python
non_cat = discretize(Gamma.from_mean_cv(80.0, 0.15))
cat = discretize(Lognormal.from_mean_cv(20.0, 1.0))
cat_net, _ = cat.map(lambda x: x - min(max(x - 41.11, 0.0), 79.64))
cnc = total_grid(Portfolio.from_independent(["Cat", "NonCat"], [cat, non_cat]))
cnc_net = total_grid(Portfolio.from_independent(["Cat", "NonCat"], [cat_net, non_cat]))

BOOK_42 = {0.5: 96.1, 0.9: 126.0, 0.99: 181.1, 0.999: 267.2, 0.9999: 394.0}
for p, book in BOOK_42.items():
    estimate = 80 + cat.quantile(p)
    print(f"p {p:6}: 80 + VaR(Cat) {estimate:6.1f}  VaR(total) {cnc.quantile(p):6.1f} (book {book})  "
          f"error {estimate / cnc.quantile(p) - 1:+.3f}")
```

```text
p    0.5: 80 + VaR(Cat)   94.1  VaR(total)   96.1 (book 96.1)  error -0.021
p    0.9: 80 + VaR(Cat)  121.1  VaR(total)  126.0 (book 126.0)  error -0.039
p   0.99: 80 + VaR(Cat)  178.1  VaR(total)  181.1 (book 181.1)  error -0.017
p  0.999: 80 + VaR(Cat)  265.3  VaR(total)  267.2 (book 267.2)  error -0.007
p 0.9999: 80 + VaR(Cat)  392.7  VaR(total)  394.0 (book 394.0)  error -0.003
```

The VaRs agree with Table 4.2. From the 99% level up, the heuristic is within
2%. The gross rows of Table 4.7 also agree. For net Cat, the VaR and the totals
agree, but three TVaRs do not, for the reason above:

```python
print("net Cat VaR ", [round(cat_net.quantile(p)) for p in LEVELS], " book [41, 41, 41, 41, 49, 106]")
print("net Cat TVaR", [round(cat_net.tvar(p)) for p in LEVELS], " book [43, 64, 75, 81, 92, 160]")
print("net TVaR    ", [round(cnc_net.tvar(p)) for p in LEVELS], " book [131, 139, 146, 160, 181, 242]")
print("gross TVaR  ", [round(cnc.tvar(p)) for p in LEVELS], " book [149, 167, 187, 218, 254, 321]")
```

```text
net Cat VaR  [41, 41, 41, 41, 49, 106]  book [41, 41, 41, 41, 49, 106]
net Cat TVaR [43, 45, 49, 62, 92, 160]  book [43, 64, 75, 81, 92, 160]
net TVaR     [131, 139, 146, 160, 181, 242]  book [131, 139, 146, 160, 181, 242]
gross TVaR   [149, 167, 187, 218, 254, 321]  book [149, 167, 187, 218, 254, 321]
```

Net Cat's TVaR at 95% cannot be 64 when its VaR is 41.1 from 90% to 99.5%. The
integral of VaR from 0.95 to 1 is at most 0.045 times 41.1, plus the tail above
the cover. The net book total uses the same net Cat, and it agrees with the
book at every level.

## 7. Properties of risk measures

Chapter 5 lists the properties that a risk measure can have:

- monotone
- translation invariant
- positive homogeneous
- subadditive (with the three properties above, this makes a measure coherent)
- convex
- comonotonic additive
- law invariant

The code runs four examples from the book.

**Bid and ask: a sublinear price has a spread (Example 83).** Ins Co. sells a
Hot August policy that pays 1 if August is hot (probability 0.1). It prices a
Bernoulli risk with probability s at the square root of s, which is a
proportional hazard distortion with rho 0.5. The Cold August policy pays in the
other 90% of years. Both prices are above the expected loss. The bid is the
price at which Ins Co. takes the Hot August risk from another party. The bid is
minus the price of minus the risk, and it is below the expected loss:

```python
from prospicio.risk import Distortion

ph = Distortion.proportional_hazard(0.5)
hot = Sampled([1.0] + [0.0] * 9)
cold = Sampled([0.0] + [1.0] * 9)
print(f"ask Hot {ph.measure(hot):.3f}, ask Cold {ph.measure(cold):.3f}   (book 0.316, 0.95)")
print(f"price of minus Hot {ph.measure(Sampled([-x for x in hot.draws])):.3f}  (book -0.05); "
      f"bid {ph.g_dual(0.1):.3f} < 0.1 < ask")
```

```text
ask Hot 0.316, ask Cold 0.949   (book 0.316, 0.95)
price of minus Hot -0.051  (book -0.05); bid 0.051 < 0.1 < ask
```

`g_dual` is the dual distortion 1 - g(1 - s), the bid side of the same price.
The book rounds -0.0513 to -0.05.

**First-order dominance is not state by state (Example 102, Bäuerle and
Müller).** Two states have probabilities 1/3 and 2/3. X is 3 then 1, and Y is 2
then 4. Each quantile of X is at most the same quantile of Y. Thus every
monotone, law-invariant risk measure ranks X below Y, but X is larger than Y in
the first state:

```python
x_bm = Discrete([3.0, 1.0], [1 / 3, 2 / 3])
y_bm = Discrete([2.0, 4.0], [1 / 3, 2 / 3])
grid_p = [0.1, 0.3, 0.5, 0.7, 0.9]
print("VaR X", [x_bm.quantile(p) for p in grid_p], " VaR Y", [y_bm.quantile(p) for p in grid_p])
print("TVaR X", [round(x_bm.tvar(p), 3) for p in grid_p], " TVaR Y", [round(y_bm.tvar(p), 3) for p in grid_p])
```

```text
VaR X [1.0, 1.0, 1.0, 3.0, 3.0]  VaR Y [2.0, 2.0, 4.0, 4.0, 4.0]
TVaR X [1.741, 1.952, 2.333, 3.0, 3.0]  TVaR Y [3.481, 3.905, 4.0, 4.0, 4.0]
```

A law-invariant measure sees only these distributions. For this reason, the
book compares risks by their quantile functions, not by their states.

**Variance and the tail disagree (Exercise 103).** Risk can increase for every
risk-averse agent while the variance does not increase. Here, an increase in
risk means higher stop loss premiums at every retention, or equivalently higher
TVaR at every level. A uniform on [0, 2] and a Pareto with tail index 4,
rescaled to mean 1, have the same mean. The uniform has the larger variance,
and the Pareto has the larger tail:

```python
from prospicio.distributions import Pareto

uniform = Sampled([(i + 0.5) / 50_000 * 2 for i in range(50_000)])
pareto = Pareto(0.75, 4.0)   # minimum 0.75, tail index 4: mean 1, variance 1/8
print(f"means {uniform.mean():.3f} {pareto.mean():.3f}; variances {4 / 12:.4f} {pareto.variance():.4f}")
for p in (0.99, 0.999):
    print(f"TVaR {p}: uniform {uniform.tvar(p):.3f}, Pareto {pareto.quantile(p) * 4 / 3:.3f}")
```

```text
means 1.000 1.000; variances 0.3333 0.1250
TVaR 0.99: uniform 1.990, Pareto 3.162
TVaR 0.999: uniform 1.999, Pareto 5.623
```

Variance ranks the uniform as riskier. TVaR at 99% and above ranks the Pareto
as riskier. (At tail index 4, the Pareto TVaR is 4/3 of its VaR.) The book's
point is that no single number ranks all risks.

**A convex measure that is not coherent (Example 97).** The exponential
(entropic) risk measure is log E[exp(theta X)] / theta. It is convex and
translation invariant, but it is not positive homogeneous. When you double the
risk, the measure more than doubles.

```python
from prospicio.risk import entropic

for theta in (0.05, 0.2):
    print(f"theta {theta}: rho(X) {entropic(discrete, theta):6.2f}, rho(2X) "
          f"{entropic(Sampled([2 * x for x in discrete.draws]), theta):6.2f}, rho(X + 10) "
          f"{entropic(Sampled([x + 10 for x in discrete.draws]), theta):6.2f}")
```

```text
theta 0.05: rho(X)  67.96, rho(2X) 163.31, rho(X + 10)  77.96
theta 0.2: rho(X)  89.45, rho(2X) 188.12, rho(X + 10)  99.45
```

It is the classical exponential premium principle (Part II). It is also
unbounded for any loss with a thick tail.

## 8. Spectral risk measures and prospicio's `Distortion`

The book's main object is the spectral risk measure (SRM). Section 3.6
introduces it, and Chapters 10 and 11 develop it. An SRM distorts the survival
function by a concave function g with g(0) = 0 and g(1) = 1. The SRM is the
expected loss under the distorted survival function: rho(X) = integral of
g(S(x)) dx. The SRMs are exactly the risk measures that are coherent,
comonotonic additive and law invariant (Theorem 3). They have six equivalent
representations:

- a distortion of the survival function
- a weighted average of VaRs with increasing weights
- a weighted average of TVaRs
- a maximum over probability scenarios
- two integral forms

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
probabilities of the sixteen sorted draws. The weighted average of the sorted
draws gives the same price:

```python
w = ph.weights(16)
ordered = sorted(discrete.draws)
print(f"PH(0.5) price {ph.measure(discrete):.3f} (book 51.156); "
      f"weighted average of sorted outcomes {sum(a * b for a, b in zip(w, ordered)):.3f}")
print("weights on the four worst draws", [round(v, 4) for v in w[-4:]], "mass at zero", ph.mass)
print(f"TVaR(0.48732) {Distortion.tvar(0.48732).measure(discrete):.3f} (book: the same 51.156, Exercise 183)")
```

```text
PH(0.5) price 51.156 (book 51.156); weighted average of sorted outcomes 51.156
weights on the four worst draws [0.067, 0.0795, 0.1036, 0.25] mass at zero 0.0
TVaR(0.48732) 51.156 (book: the same 51.156, Exercise 183)
```

TVaR at 0.48732 charges the same premium, as Exercise 183 states. The two SRMs
give one price for this risk and different prices for all other risks.

A distortion built from other distortions is also a distortion. Example 176
takes the minimum of two distortions. The first is the PH with rho 0.5. The
second is a constant cost of capital (CCoC) of 50%, g(s) = (2 s + 1) / 3. The two cross at s = 1/4. Below
s = 1/4 the minimum is the PH, and above it the minimum is the CCoC. The book
calculates the TVaR representation:

- a mass of 2/3 on the mean (TVaR at 0)
- a mass of 1/12 at TVaR(0.75)
- a density over the levels from 0.75 to 1

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

`Distortion.ccoc(r)` takes the cost of capital r, so `ccoc(0.5)` is the book's
(2 s + 1) / 3. `aggregate` builds the same function from the discount
d = r / (1 + r).

## 9. Choosing a risk measure

Chapter 6 separates two uses of a risk measure:

- A **capital** risk measure sets the assets that a company holds. It is VaR,
  TVaR or EPD, chosen by a regulator or rating agency and applied to the total.
- A **pricing** risk measure sets the premium. It is an SRM or a classical
  principle.

Table 6.1, the book's compendium, lists the measures and families. This table
gives the prospicio name of each:

| Measure | prospicio |
|---|---|
| mean, VaR, upper quantile | `.mean()`, `.quantile(p)`, `Discrete.quantile_upper` |
| TVaR | `.tvar(p)` on `Sampled`, `Grid` and `Discrete`, and `Distortion.tvar(p)` |
| EPD and EPD capital | `Portfolio.epd(a)`, `Portfolio.assets_for_epd(ratio)` |
| SRMs: CCoC, PH, Wang, dual, TVaR, bi-TVaR, beta, LEP and others | `Distortion` constructors |
| exponential (entropic), Esscher | `risk.entropic`, `risk.esscher` |
| classical premium principles | `pricing.classical_premium`, `calibrate_classical` |
| CoVaR, marginal expected shortfall | `risk.covar`, `risk.marginal_expected_shortfall` |

The book gives this advice for practice. A capital measure should be simple and
robust. VaR is both, and it is the regulator's choice. A pricing measure must be
coherent and comonotonic additive, so that the layers and parts of one risk add
up. These conditions point to an SRM. Part II continues from here.
