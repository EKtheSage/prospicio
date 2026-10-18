# Pricing Insurance Risk, Part I: risk measures in prospicio

Part I of Stephen Mildenhall and John Major's *Pricing Insurance Risk: Theory
and Practice* (Wiley, 2022), chapters 3 to 7, is about measuring risk before
pricing it: how a risk is represented, what quantiles, value at risk (VaR) and
tail value at risk (TVaR) are and how they behave, which properties a risk
measure should have, and how spectral (distortion) risk measures tie these
together. This guide works through those chapters in prospicio. Wherever the
book prints a number, the code recomputes it and prints the book's value as a
literal beside it, so you can see what agrees, what does not, and why.

The book's text and tables are the authors'; what follows explains them in new
words. Read the book for the theory and this guide for the mechanics. Table,
example and exercise numbers are the book's. Part II, pricing a portfolio, is in
[Pricing Insurance Risk, Part II](pir-portfolio-pricing.md); the book's case
studies priced by five distortions and allocated to units are in
[the case studies guide](pricing-insurance-risk-case-studies.md), and
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md) covers the same
method on a ten-year example.

## 1. Representing a risk: the Simple Discrete Example

Chapter 3 represents a risk as a random variable on a sample space and computes
its expected value in two ways: as a sum over outcomes weighted by their
probabilities, and as the area under the survival function, a sum of horizontal
slices (the Lee diagram). The book's running example, the Simple Discrete
Example, is two independent policies. The first loses 0, 8 or 10, the second 0,
1 or 90, each with probabilities 1/2, 1/4 and 1/4. The total takes nine values,
and every probability is a multiple of 1/16, so sixteen equally likely draws
represent it exactly:

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

Both sums give the book's 27.25 (Table 3.2). Capping the loss at assets of 80
changes only the outcome column; the survival probabilities stay the same, and
the limited expected value is 23.625 (Table 3.4). Adding 100 to every outcome
needs a zero outcome added first, which is the book's Step (1): the survival
form integrates from zero, so it must see the slice from 0 to 100 (Table 3.3).

In prospicio a risk given by simulations or equally likely scenarios is a
`Sampled`; one given by values and probabilities is a `Discrete`; one on an
evenly spaced grid (the output of an FFT) is a `Grid`. All three have `mean`,
`quantile` and `tvar`, and a distortion prices a `Sampled` or a `Grid`
(`Distortion.measure`).

## 2. Quantiles and VaR

Chapter 4 defines the p quantile as any x with P(X < x) <= p <= P(X <= x). When
the distribution function jumps there is one answer; when it is flat at p there
is a range, and VaR is the lower end (the lower quantile). For a fair die,
Exercise 38: the 0.1 quantile is 1, and at p = 1/6 every value in [1, 2] is a
quantile. Exercise 49 has ten equally likely outcomes 0, 1, 1, 1, 2, 3, 4, 8, 12
and 25, where the 0.4 quantiles are [1, 2] and VaR is 1:

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

`quantile` is the lower quantile, VaR in the book's sense, and `quantile_upper`
the upper end of the range (it equals `quantile` where the distribution
function crosses p in a jump). `Sampled.quantile` and `Grid.quantile` use the same
lower convention, which is what VaR means throughout this guide.

The book also reads VaR from grouped data with an ogive (Exercise 50).
prospicio has no ogive constructor; interpolate the tabulated distribution
function yourself, or put the losses on a grid.

## 3. Return periods and the PML of a hurricane model

VaR at p is quoted as the 1 / (1 - p) year loss: the 99% VaR is the 100-year
loss and the 99.6% VaR the 250-year loss (Table 4.1). A probable maximum loss
(PML) can mean two things: the annual aggregate loss at that return period (an
aggregate VaR) or the largest single event's loss (an occurrence PML). Example
54 compares them for US hurricanes: a Poisson count of 1.74 landfalls a year and
a lognormal severity with mean USD 5 billion and sigma 2.58. Table 4.3 prints,
for each return period, the occurrence PML, the severity's own quantile, the
aggregate VaR by FFT and the book's quick estimate of the aggregate VaR from
the severity alone (Eq. (4.5): the severity quantile at 1 - (1 - p) / 1.74).
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
unit printed, and so does every aggregate VaR, computed here on a grid of 20
(USD 20 million) with 2^21 points. The book's aggregate VaRs are all multiples
of 20, which suggests the same bucket. One printed parameter does not fit: the
book gives the lognormal mu as 21.315, while a mean of USD 5 billion with sigma
2.58 needs mu = 19.005 in dollars, and the table's severity column (a median of
179 million) is reproduced only by 19.005. The table is right and the printed mu is
not; build the severity from its mean and sigma.

The occurrence PML is always below the aggregate VaR, and close to it at long
return periods: a bad hurricane year is one bad hurricane. That is also why the
quick estimate, which looks only at the severity, is good in the tail (an error
of 2% at 1,000 years) and poor for frequent events (82% too high at 2 years).

## 4. VaR is not subadditive

A risk measure is subadditive if the risk of a sum is never more than the sum of
the risks, so that pooling never appears to add risk. The book gives three ways
VaR fails it. The first is dependence: two risks arranged to be large in
different scenarios can make the VaR of their sum exceed the sum of their VaRs.
The second needs no dependence at all, only skewness: the sum of two independent
exponentials with mean 1 is a gamma with shape 2, and below p = 0.7 its VaR is
more than twice an exponential's (Table 4.4):

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

The differences match Table 4.4 to the four significant figures it prints (in
the table as printed the 0.8 row's difference is set one line low, under 0.999;
the columns to its left fix which value belongs where).
The third way is a very thick tail: for a Pareto with tail index below 1 (no
mean), VaR fails subadditivity at every high enough p, because one large loss
dominates a sum.

### The worst VaR of a sum: the Rearrangement Algorithm

When the marginal distributions are known and the dependence is not, the worst
possible VaR of their sum is a useful bound. The book's Rearrangement Algorithm
(RA, Puccetti and Rüschendorf) finds it numerically: take the top 1 - p of each
marginal, and repeatedly reorder each column against the sum of the others
until the smallest row sum stops rising. Exercise 59 asks for the worst 99% VaR
of three lognormals with mean 10 and CVs 1, 2 and 3, from 40 points of each at and
above its 99th percentile. prospicio has no RA, so a dozen lines of numpy do it,
on prospicio's lognormal quantiles:

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

The VaRs of the three lognormals and their comonotonic sum, 242.5, are the
book's. The worst-case VaR found is 352.9 to 353.8 depending on the start,
against the book's 352.8: the algorithm is a heuristic, its answer depends on
the starting arrangement, and the book does not give its start. The starts here
are fixed rotations (column j shifted down by 7, 13 or 19 times j places) rather
than random shuffles, so that the R version of this guide prints the same
numbers. None reaches the book's 352.8 exactly, though the best is within 0.1. Either way the conclusion is the
book's: the worst arrangement raises the 99% VaR of the sum by 45% over
perfect positive dependence, so comonotonic VaR is not a worst case.

## 5. TVaR

TVaR at p is the average of the worst 1 - p of outcomes, the integral of VaR
from p to 1 divided by 1 - p. For a normal it exceeds VaR by a fixed number of
standard deviations, which the book quotes as 14.6% of the standard normal's VaR at
p = 0.99, 9% at 0.999 and 6.4% at 0.9999. prospicio has no normal distribution
for losses; Python's own `statistics.NormalDist` gives the check:

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

Exercise 63 is the book's case for TVaR over VaR. A restaurant bill is lognormal
with mean 2,500 and CV 0.2. On a regular night you pay it; on a Pot Luck night
one of 25 diners pays it, chosen at random, so you pay with probability 0.04.
The 95% VaR of the Pot Luck bill is zero, and VaR cash is no cash. TVaR cash is
the 95% TVaR, 0.04 / 0.05 of the mean:

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

All six numbers agree. A lognormal has no `tvar` method in prospicio; TVaR
follows from its quantile and limited expected value (`lev`) as above, or put
it on a grid and call `Grid.tvar`.

For a discrete distribution TVaR takes a fraction of the observation at the
quantile (the book's algorithm in Section 4.3.6), and it differs from the
conditional tail expectation (CTE), E[X | X >= VaR], which takes the whole atom.
Exercise 49 continued gives TVaR and CTE for the ten outcomes:

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

TVaR is continuous in p and CTE jumps at each observation; they agree at the
observations. Use TVaR: it is the one that is subadditive.

## 6. VaR, TVaR and EPD in the case studies

Section 4.3.7 tabulates VaR, TVaR and the EPD capital (the assets at which the
expected policyholder deficit E[(X - a)+] is a given share of the expected loss)
for the case studies, by unit, summed, and in total, gross and net of
reinsurance. The Tame case is two gamma units with mean 50 and CVs 0.10 and
0.15; the net book has an aggregate cover of 12.91 excess of 56.17 on unit B (its
80% to 99% quantiles, Table 7.3). The grid is `aggregate`'s, a step of 1/64 and
2^16 points, with the severity renormalized as in the
[case studies guide](pricing-insurance-risk-case-studies.md):

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

Table 4.6 prints these to the unit (EPD rows by EPD ratio 10% down to 0.1%):

| Row | Book, gross A | B | Total | Book, net B | Net total |
|---|---|---|---|---|---|
| VaR | 56 58 60 62 64 67 | 60 63 66 69 72 76 | 112 115 119 122 126 131 | 56 56 56 56 59 64 | 109 111 113 116 118 121 |
| TVaR | 59 61 62 64 66 69 | 64 67 69 72 75 79 | 117 120 123 126 129 134 | 58 59 59 59 62 66 | 112 114 116 118 120 123 |
| EPD | 45 49 52 55 57 60 | 46 51 55 59 63 68 | 91 97 103 108 113 119 | 45 49 52 54 55 56 | 90 96 101 105 108 112 |

Every entry agrees after rounding (a few of ours sit on .5, like the total's EPD
at 0.1% of 118.5 against 119), with one exception: **the TVaR of net unit B at
90%, 95% and 97.5%**, where the book prints 58, 59 and 59 and prospicio gives
56.5, 56.8 and 57.5. prospicio's values are the right ones, and the book's own
VaR row shows why. Net B is flat at 56.17 from its 80% to its 99% quantile (the
cover absorbs everything in between), so TVaR at 90% is 56.17 plus the expected
excess over 56.17 divided by 0.1, and that excess is only the part above the
cover's exhaustion, 0.032:

```python
excess = b_net.mean() - b_net.lev(56.171875)
print(f"E[(B net - 56.17)+] = {excess:.4f}; TVaR 90% = {56.171875 + excess / 0.1:.2f} (book 58)")
```

```text
E[(B net - 56.17)+] = 0.0323; TVaR 90% = 56.50 (book 58)
```

To print 58 the expected excess would have to be about 0.18. The net totals
agree, so only the column for the net unit is affected. The same thing happens
in Table 4.7 for net Cat in the Cat/Non-Cat case (below): the book's TVaR of a
unit with a large probability mass under a stop loss is too high at the levels
where VaR sits on that mass. We read it as a defect in the TVaR the book's code
computed for such distributions, not a convention.

The Cat/Non-Cat case is a gamma Non-Cat unit (mean 80, CV 0.15) and a lognormal
Cat unit (mean 20, CV 1), with an aggregate cover of 79.64 excess of 41.11 on
Cat (its 90% to 99.5% quantiles). Example 53's VaR heuristic says that when a
thin-tailed unit is added to a thick-tailed one, the VaR of the total is about
the thin unit's mean plus the thick unit's VaR (Table 4.2):

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

The VaRs agree with Table 4.2, and the heuristic is within 2% from the 99% level up.
Table 4.7's gross rows agree as well; for net Cat the VaR and the totals agree
and three TVaRs do not, for the reason above:

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

Net Cat's TVaR at 95% cannot be 64 when its VaR is 41.1 all the way from 90% to
99.5%: the integral of VaR from 0.95 to 1 is at most 0.045 times 41.1 plus the
tail above the cover. The net book's total, which uses the same net Cat, agrees
with the book at every level.

## 7. Properties of risk measures

Chapter 5 lists the properties a risk measure can have: monotone, translation
invariant, positive homogeneous, subadditive (together: coherent), convex,
comonotonic additive and law invariant. Three of the book's examples are
worth running.

**Bid and ask: a sublinear price has a spread (Example 83).** Ins Co. sells a
Hot August policy, paying 1 if August is hot (probability 0.1), and prices a
Bernoulli risk with probability s at the square root of s, a proportional hazard
distortion with rho 0.5. The Cold August policy pays in the other 90% of years.
Both prices exceed the expected loss, and the price at which Ins Co. would take
the Hot August risk off someone's hands (the bid, minus the price of minus the
risk) is below it:

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
Müller).** On two states with probabilities 1/3 and 2/3, X is 3 then 1 and Y is
2 then 4. Every quantile of X is at most Y's, so every monotone, law-invariant
risk measure ranks X below Y, yet X is larger than Y in the first state:

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

A law-invariant measure sees only these distributions. That is why the book
works with quantile functions rather than states when it compares risks.

**Variance and the tail disagree (Exercise 103).** Risk can increase in the
sense that every risk-averse agent dislikes it more (higher stop loss premiums at
every retention, equivalently higher TVaR at every level) without the variance
rising. A uniform on [0, 2] and a Pareto with tail index 4 rescaled to mean 1
have the same mean; the uniform has the larger variance and the Pareto the
larger tail:

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

Variance ranks the uniform as riskier; TVaR at 99% and above ranks the Pareto
riskier. (Pareto TVaR is 4/3 of its VaR at tail index 4.) The book's point is
that no single number is a universal ranking of risk.

**A convex measure that is not coherent (Example 97).** The exponential
(entropic) risk measure, log E[exp(theta X)] / theta, is convex and translation
invariant but not positive homogeneous: doubling the risk more than doubles
it.

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

It is the classical exponential premium principle (Part II), and it is also
unbounded for any loss with a thick tail.

## 8. Spectral risk measures and prospicio's `Distortion`

The book's main object, introduced in Section 3.6 and developed in Chapters 10
and 11, is the spectral risk measure (SRM): the expected loss after the survival
function is distorted by a concave function g with g(0) = 0 and g(1) = 1,
rho(X) = integral of g(S(x)) dx. SRMs are exactly the risk measures that are
coherent, comonotonic additive and law invariant (Theorem 3), and they have six
equivalent representations: as a distortion of the survival function, a
weighted average of VaRs with increasing weights, a weighted average of TVaRs, a
maximum over probability scenarios, and two integral forms. A prospicio
`Distortion` is a g, and its methods are those representations:

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
probabilities of the sixteen sorted draws, and they reproduce the price as a
weighted average:

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

TVaR at 0.48732 charges the same premium, as Exercise 183 states: two different
SRMs, one price for this risk, and different prices for everything else.

A distortion built from others is still a distortion. Example 176 takes the
minimum of the PH with rho 0.5 and a constant cost of capital (CCoC) of 50%, g(s) =
(2 s + 1) / 3. They cross at s = 1/4: the PH below, the CCoC above. The book
works out the TVaR representation: a mass of 2/3 on the mean (TVaR at 0), a mass
of 1/12 at TVaR(0.75) and a density over the levels from 0.75 to 1:

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
(2 s + 1) / 3; `aggregate` builds the same function from the discount d = r /
(1 + r).

## 9. Choosing a risk measure

Chapter 6 separates two uses: a **capital** risk measure sets the assets a
company holds (VaR, TVaR or EPD, chosen by a regulator or rating agency and
applied to the total), and a **pricing** risk measure sets the premium (an SRM,
or a classical principle). Its compendium (Table 6.1) lists the measures and
families. Where each one is in prospicio:

| Measure | prospicio |
|---|---|
| mean, VaR, upper quantile | `.mean()`, `.quantile(p)`, `Discrete.quantile_upper` |
| TVaR | `.tvar(p)` on `Sampled`, `Grid`, `Discrete`; `Distortion.tvar(p)` |
| EPD and EPD capital | `Portfolio.epd(a)`, `Portfolio.assets_for_epd(ratio)` |
| SRMs: CCoC, PH, Wang, dual, TVaR, bi-TVaR, beta, LEP and others | `Distortion` constructors |
| exponential (entropic), Esscher | `risk.entropic`, `risk.esscher` |
| classical premium principles | `pricing.classical_premium`, `calibrate_classical` |
| CoVaR, marginal expected shortfall | `risk.covar`, `risk.marginal_expected_shortfall` |
| worst VaR (Rearrangement Algorithm) | none: section 4's numpy code |
| ogive VaR from grouped data | none: interpolate |

The book's advice in practice: a capital measure should be simple and robust
(VaR is both, and the regulator's choice); a pricing measure must be coherent
and comonotonic additive so that layers and pieces of one risk add up, which
points to an SRM. Part II takes it from there.

## 10. What this guide leaves out

- **Chapter 3's taxonomies of risk and Chapter 6's characterisation method**
  (Wang and Zitikis' axioms): concepts with no computation.
- **Exercises without printed answers** (56, 58, the PELVE estimates of Exercise
  68): the tools are above.
- **The representation theorem and Delbaen's differentiation theorem**
  (Sections 5.4 and 5.5): the derivative of an SRM is the natural allocation,
  which is Part III's subject and is in
  [the case studies guide](pricing-insurance-risk-case-studies.md).
- **Table 4.8 (Hu/SCS)**: its gross and net VaR and TVaR rows need the
  2^19-point grid of the case studies guide, and its EPD rows as extracted do
  not read as a table we could compare.
- **Gaps in prospicio**: the Rearrangement Algorithm, an ogive from grouped
  data, a normal distribution and a `tvar` on parametric severities such as
  the lognormal (issue #366, for the Probability lane).
