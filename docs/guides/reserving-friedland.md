# Friedland's basic reserving techniques, reproduced

Jacqueline Friedland's *Estimating Unpaid Claims Using Basic Techniques*
(Casualty Actuarial Society, version 3, 2010) is the CAS syllabus text on
reserving. It applies each technique to the same few data sets, so the results
of one chapter feed the next: the development factors of chapter 7 become the
reporting pattern of chapters 9 and 10, and the expected claims of chapter 8 the
prior of chapter 9. This guide follows that path in prospicio with the book's
own data, and prints the book's printed figures next to prospicio's.

The text and exhibits are Friedland's; what follows explains them in new words
and recomputes the numbers. Exhibit numbers are the book's, and they restart in
every chapter, so "Exhibit II" always comes with its chapter. Money is in
thousands of dollars, as in the exhibits. For the choice between methods on your
own data, read [`reserving-methods.md`](reserving-methods.md); this guide does
not repeat it.

## 1. The two data sets

The book's chapters 7 to 10 work through two examples.

- **U.S. Industry Auto**: reported and paid claims of the whole U.S. private
  passenger auto industry, accident years 1998 to 2007, valued at December 31,
  2007, with earned premium. Large and stable: every method should agree.
- **XYZ Insurer, Auto BI**: a small insurer's bodily injury book, accident
  years 1998 to 2008, valued at December 31, 2008. Its case reserving, its
  settlement speed and a tort reform all changed, so the methods disagree, which
  is the point of the example. Only calendar years 2000 to 2008 exist, so the
  two oldest rows start at 36 and 24 months.

The triangles below are typed from chapter 7, Exhibits I and II (sheets 1 and
2), and the premiums from chapter 8, Exhibits II and III:

```python
from prospicio.reserving import (
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ExpectedLoss, TailCurve, Triangle,
)

US_YEARS = list(range(1998, 2008))
US_REPORTED = [
    [37017487, 43169009, 45568919, 46784558, 47337318, 47533264, 47634419, 47689655, 47724678, 47742304],
    [38954484, 46045718, 48882924, 50219672, 50729292, 50926779, 51069285, 51163540, 51185767],
    [41155776, 49371478, 52358476, 53780322, 54303086, 54582950, 54742188, 54837929],
    [42394069, 50584112, 53704296, 55150118, 55895583, 56156727, 56299562],
    [44755243, 52971643, 56102312, 57703851, 58363564, 58592712],
    [45163102, 52497731, 55468551, 57015411, 57565344],
    [45417309, 52640322, 55553673, 56976657],
    [46360869, 53790061, 56786410],
    [46582684, 54641339],
    [48853563],
]
US_PAID = [
    [18539254, 33231039, 40062008, 43892039, 45896535, 46765422, 47221322, 47446877, 47555456, 47644187],
    [20410193, 36090684, 43259402, 47159241, 49208532, 50162043, 50625757, 50878808, 51000534],
    [22120843, 38976014, 46389282, 50562385, 52735280, 53740101, 54284334, 54533225],
    [22992259, 40096198, 47767835, 52093916, 54363436, 55378801, 55878421],
    [24092782, 41795313, 49903803, 54352884, 56754376, 57807215],
    [24084451, 41399612, 49070332, 53584201, 55930654],
    [24369770, 41489863, 49236678, 53774672],
    [25100697, 42702229, 50644994],
    [25608776, 43606497],
    [27229969],
]
US_PREMIUM = [68574209, 68544981, 68907977, 72544955, 79228887,
              86643542, 91763523, 94115312, 95272279, 95176240]

XYZ_YEARS = list(range(1998, 2009))
XYZ_FIRST_AGE = [36, 24] + [12] * 9  # months at the first observed cell of each row
XYZ_REPORTED = [
    [11171, 12380, 13216, 14067, 14688, 16366, 16163, 15835, 15822],
    [13255, 16405, 19639, 22473, 23764, 25094, 24795, 25071, 25107],
    [15676, 18749, 21900, 27144, 29488, 34458, 36949, 37505, 37246],
    [11827, 16004, 21022, 26578, 34205, 37136, 38541, 38798],
    [12811, 20370, 26656, 37667, 44414, 48701, 48169],
    [9651, 16995, 30354, 40594, 44231, 44373],
    [16995, 40180, 58866, 71707, 70288],
    [28674, 47432, 70340, 70655],
    [27066, 46783, 48804],
    [19477, 31732],
    [18632],
]
XYZ_PAID = [
    [6309, 8521, 10082, 11620, 13242, 14419, 15311, 15764, 15822],
    [4666, 9861, 13971, 18127, 22032, 23511, 24146, 24592, 24817],
    [1302, 6513, 12139, 17828, 24030, 28853, 33222, 35902, 36782],
    [1539, 5952, 12319, 18609, 24387, 31090, 37070, 38519],
    [2318, 7932, 13822, 22095, 31945, 40629, 44437],
    [1743, 6240, 12683, 22892, 34505, 39320],
    [2221, 9898, 25950, 43439, 52811],
    [3043, 12219, 27073, 40026],
    [3531, 11778, 22819],
    [3529, 11865],
    [3409],
]
XYZ_PREMIUM = [20000, 31500, 45000, 50000, 61183, 69175, 99322, 138151, 107578, 62438, 47797]


def triangle(years, columns, exposures=None, first_age=None):
    """A Triangle from rows of cumulative values; exposures are repeated on every cell."""
    exposures = exposures or {}
    first_age = first_age or [12] * len(years)
    origin, age = [], []
    values = {name: [] for name in [*columns, *exposures]}
    rows = next(iter(columns.values()))
    for i, year in enumerate(years):
        for k in range(len(rows[i])):
            months = first_age[i] + 12 * k
            origin.append(year)
            age.append(months)
            for name, table in columns.items():
                values[name].append(float(table[i][k]))
            for name, column in exposures.items():
                values[name].append(float(column[i]))
    return Triangle.from_long(origin, age, values)


us = triangle(US_YEARS, {"reported": US_REPORTED, "paid": US_PAID}, {"premium": US_PREMIUM})
xyz = triangle(XYZ_YEARS, {"reported": XYZ_REPORTED, "paid": XYZ_PAID},
               {"premium": XYZ_PREMIUM}, first_age=XYZ_FIRST_AGE)
print(us)
print(xyz.view("reported").iloc[:3, :5])
```

```text
Triangle: 1 segment x 3 columns (cumulative, valuation 2007-12)
column    n_origins  first_origin  last_origin  valuation       latest
reported         10          1998         2007    2007-12  543,481,587
paid             10          1998         2007    2007-12  498,050,368
premium          10          1998         2007    2007-12  820,771,905
development       12       24       36       48       60
origin
1998             NaN      NaN  11171.0  12380.0  13216.0
1999             NaN  13255.0  16405.0  19639.0  22473.0
2000         15676.0  18749.0  21900.0  27144.0  29488.0
```

The XYZ rows for 1998 and 1999 begin with unobserved cells (`NaN`). prospicio
keeps them as missing, not zero, so they take no part in any factor.

## 2. The development technique (chapter 7)

### Age-to-age factors and their averages

The development technique starts from the link ratios: each cell divided by
the one before it in the same row. Friedland then lays several averages of each
column side by side and selects one factor per age. Her averages run over the
**latest** few accident years: simple averages of the latest 5 and 3, a medial
average of the latest 5 without the highest and lowest, volume-weighted
averages of the latest 5 and 3, and a geometric average of the latest 4.

All of them are options of `ChainLadder`: `average=` picks the mean
(`"volume"`, `"simple"`, `"regression"` or `"geometric"`), `n_periods=n` keeps
only the link ratios that start in the latest n calendar diagonals (the latest
n accident years at each age), and `drop_high=` and `drop_low=` leave out the
highest and lowest of those, so a medial average is a simple one with one of
each dropped. A fit's `n_links` says how many link ratios each factor averaged:

```python
AVERAGES = {
    "simple, latest 5": ChainLadder(average="simple", n_periods=5),
    "simple, latest 3": ChainLadder(average="simple", n_periods=3),
    "medial, latest 5x1": ChainLadder(average="simple", n_periods=5, drop_high=1, drop_low=1),
    "volume, latest 5": ChainLadder(n_periods=5),
    "volume, latest 3": ChainLadder(n_periods=3),
    "geometric, latest 4": ChainLadder(average="geometric", n_periods=4),
}
fits = {name: method.fit(us, "reported") for name, method in AVERAGES.items()}
for name, fit in fits.items():
    print(f"{name:20s}", " ".join(f"{f:.3f}" for f in fit.ldf))
print(f"{'n_links, medial 5x1':20s}", " ".join(f"{n:5d}" for n in fits["medial, latest 5x1"].n_links))
```

```text
simple, latest 5     1.168 1.058 1.027 1.011 1.004 1.003 1.002 1.001 1.000
simple, latest 3     1.164 1.056 1.027 1.012 1.005 1.003 1.002 1.001 1.000
medial, latest 5x1   1.165 1.057 1.027 1.010 1.004 1.003 1.002 1.001 1.000
volume, latest 5     1.168 1.058 1.027 1.011 1.004 1.003 1.002 1.001 1.000
volume, latest 3     1.164 1.056 1.027 1.012 1.005 1.003 1.002 1.001 1.000
geometric, latest 4  1.164 1.057 1.027 1.011 1.004 1.003 1.002 1.001 1.000
n_links, medial 5x1      3     3     3     3     3     2     1     2     1
```

Each factor row equals the corresponding row of chapter 7, Exhibit I, sheet 1,
part 3, to the three decimals printed there (12-24 at the left, 108-120 at the
right); the paid averages on sheet 2 agree in the same way. Two conventions in
the book matter here, and prospicio follows both. When only a few years exist,
"latest 5" quietly uses all of them. And its medial average of two values is
their simple average and of one value the value itself: prospicio skips a drop
that would leave no link ratio. The last line shows it: the five link ratios at
12-24 keep 3, the four at 72-84 keep 2 and the three at 84-96 keep 1, while the
two at 96-108 and the one at 108-120 keep them all.

The all-year averages, by contrast, are not what the book uses:

```python
print("volume, latest 3 ", " ".join(f"{f:.3f}" for f in ChainLadder(n_periods=3).fit(us, "reported").ldf))
print("volume, all years", " ".join(f"{f:.3f}" for f in ChainLadder().fit(us, "reported").ldf))
print("simple, all years", " ".join(f"{f:.3f}" for f in ChainLadder(average="simple").fit(us, "reported").ldf))
```

```text
volume, latest 3  1.164 1.056 1.027 1.012 1.005 1.003 1.002 1.001 1.000
volume, all years 1.175 1.058 1.027 1.011 1.004 1.003 1.002 1.001 1.000
simple, all years 1.175 1.058 1.027 1.011 1.004 1.003 1.002 1.001 1.000
```

The first line is the book's volume-weighted latest 3 again; the all-year
factors at 12-24 (1.175) are well above the book's selection of
1.164, because the early accident years developed more in their first year.

### Selected factors, factors to ultimate, and rounding

For U.S. Industry Auto, Friedland selects the latest-3 averages, rounded to
three decimals, with no tail for reported claims and a tail of 1.002 for paid.
Multiplying the selected factors gives the factor to ultimate (CDF), which she
also rounds to three decimals before multiplying it into the latest diagonal.
In prospicio the selected factors go in as a pattern with `ldf=` and
`ldf_kind="ldf"`, and a constant tail as `tail=`:

```python
AGES = us.development
SEL_REPORTED = [1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000]
SEL_PAID = [1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002]
dev_r = ChainLadder(ldf=dict(zip(AGES, SEL_REPORTED)), ldf_kind="ldf").fit(us, "reported")
dev_p = ChainLadder(ldf=dict(zip(AGES, SEL_PAID)), ldf_kind="ldf", tail=1.002).fit(us, "paid")
print("reported CDF", " ".join(f"{c:.5f}" for c in dev_r.cdf))
print("paid CDF    ", " ".join(f"{c:.5f}" for c in dev_p.cdf))
print(f"total ultimate: reported {sum(dev_r.ultimate):,.0f} (book 569,172,456)")
print(f"                paid     {sum(dev_p.ultimate):,.0f} (book 572,041,099)")
```

```text
reported CDF 1.29163 1.10964 1.05080 1.02317 1.01104 1.00601 1.00300 1.00100 1.00000 1.00000
paid CDF     2.39003 1.40425 1.18402 1.08526 1.03952 1.02014 1.01104 1.00601 1.00400 1.00200
total ultimate: reported 569,136,323 (book 569,172,456)
                paid     572,052,982 (book 572,041,099)
```

The totals miss the book by 36,133 and 11,883 (thousands). The cause is the
rounding of the CDF: the product of the reported factors from 12 months is
1.29163, which the book prints and uses as 1.292. To reproduce the book, give
its rounded CDFs as the pattern instead, with `ldf_kind="cdf"`. A pattern given
to the oldest age with no `tail=` keeps its last value as the tail, so the
paid CDF of 1.002 at 120 months is the tail:

```python
CDF_REPORTED = [1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000]
CDF_PAID = [2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002]
US_CDF = {
    "reported": dict(zip(AGES, CDF_REPORTED)),
    "paid": dict(zip(AGES, CDF_PAID)),
}
dev_r = ChainLadder(ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported")
dev_p = ChainLadder(ldf=US_CDF["paid"], ldf_kind="cdf").fit(us, "paid")
print("paid tail", dev_p.pattern_tail)

BOOK_DEV = {  # chapter 7, Exhibit I, sheet 3, columns (7) and (8)
    "reported": [47742304, 51185767, 54892767, 56468461, 58944268,
                 58198563, 58287120, 59682517, 60651886, 63118803],
    "paid": [47739475, 51204536, 54860424, 56493084, 58963359,
             58167880, 58345519, 59963673, 61223522, 65079626],
}
print(f"{'year':>4s} {'reported':>12s} {'book':>12s} {'paid':>12s} {'book':>12s}")
for y, a, b, c, d in zip(US_YEARS, dev_r.ultimate, BOOK_DEV["reported"], dev_p.ultimate, BOOK_DEV["paid"]):
    print(f"{y:4d} {a:12,.0f} {b:12,} {c:12,.0f} {d:12,}")
print(f"{'all':>4s} {sum(dev_r.ultimate):12,.0f} {569172456:12,} {sum(dev_p.ultimate):12,.0f} {572041099:12,}")
```

```text
paid tail 1.002
year     reported         book         paid         book
1998   47,742,304   47,742,304   47,739,475   47,739,475
1999   51,185,767   51,185,767   51,204,536   51,204,536
2000   54,892,767   54,892,767   54,860,424   54,860,424
2001   56,468,461   56,468,461   56,493,084   56,493,084
2002   58,944,268   58,944,268   58,963,359   58,963,359
2003   58,198,563   58,198,563   58,167,880   58,167,880
2004   58,287,120   58,287,120   58,345,519   58,345,519
2005   59,682,517   59,682,517   59,963,673   59,963,673
2006   60,651,886   60,651,886   61,223,522   61,223,522
2007   63,118,803   63,118,803   65,079,626   65,079,626
 all  569,172,456  569,172,456  572,041,099  572,041,099
```

Every ultimate now equals sheet 3 to the thousand dollars. The unpaid claim
estimate follows the book's sheet 4: IBNR is ultimate less reported, and the
total unpaid is ultimate less paid, which is IBNR plus case outstanding.
prospicio's `reserve` is ultimate less the column's latest value, so on the
reported triangle it is IBNR, and on the paid triangle it is the total unpaid:

```python
def unpaid(ultimate, tri=us):
    reported = sum(tri.latest_diagonal()[0][tri.columns.index("reported")])
    paid = sum(tri.latest_diagonal()[0][tri.columns.index("paid")])
    return sum(ultimate) - reported, sum(ultimate) - paid


for name, fit, book in [("reported", dev_r, (25690869, 71122088)), ("paid", dev_p, (28559512, 73990731))]:
    ibnr, total = unpaid(fit.ultimate)
    print(f"{name:8s} IBNR {ibnr:11,.0f} (book {book[0]:11,})  total unpaid {total:11,.0f} (book {book[1]:11,})")
print(f"paid reserve = total unpaid: {dev_p.total_reserve:,.0f}")
```

```text
reported IBNR  25,690,869 (book  25,690,869)  total unpaid  71,122,088 (book  71,122,088)
paid     IBNR  28,559,512 (book  28,559,512)  total unpaid  73,990,731 (book  73,990,731)
paid reserve = total unpaid: 73,990,731
```

### The tail factor

Friedland's tail of 1.002 for paid claims is a judgment that the paid
development after 120 months matches the reported claims still open then (the
reported pattern has reached 1.000). prospicio can also extrapolate a tail from
the factors; on this triangle a fitted exponential curve gives a factor close to
hers:

```python
curve = ChainLadder(tail=TailCurve()).fit(us, "paid")
print(f"fitted tail {curve.tail:.4f}, book's selection 1.002")
```

```text
fitted tail 1.0011, book's selection 1.002
```

A curve fitted to all ages is pulled by the early factors, so look at it as a
second opinion beside a benchmark such as the reported pattern, as the book
does, rather than as the answer.

### XYZ Insurer: unstable factors and a pattern below 1

For XYZ, Friedland selects the volume-weighted average of the latest 2 years at
every age, with no reported tail and a paid tail of 1.010. `n_periods=2`
reproduces her selected factors (chapter 7, Exhibit II, sheets 1 and 2, part 4):

```python
for column in ("reported", "paid"):
    print(f"{column:8s}", " ".join(f"{f:.3f}" for f in ChainLadder(n_periods=2).fit(xyz, column).ldf))
```

```text
reported 1.687 1.265 1.102 1.020 1.050 1.010 1.011 1.000 0.993 0.999
paid     3.349 2.079 1.574 1.316 1.203 1.136 1.059 1.022 1.017 1.004
```

The book prints 1.687 1.265 1.102 1.020 1.050 1.010 1.011 1.000 0.993 0.999
for reported and 3.349 2.079 1.574 1.316 1.203 1.136 1.059 1.022 1.017 1.004
for paid: all equal. Reported claims fall after 96 months (case reserves set too
high and then released), so the reported CDFs at 96 and 108 months are 0.992,
below 1. A selected pattern with `ldf_kind="cdf"` needs only positive
factors to ultimate, in any order, so the book's rounded CDFs go in directly,
the reported ones as well as the paid:

```python
XYZ_AGES = xyz.development
CDF_XYZ_REPORTED = [2.551, 1.512, 1.196, 1.085, 1.064, 1.013, 1.003, 0.992, 0.992, 0.999, 1.000]
CDF_XYZ_PAID = [21.999, 6.569, 3.160, 2.007, 1.525, 1.268, 1.116, 1.054, 1.031, 1.014, 1.010]
xdev_r = ChainLadder(ldf=dict(zip(XYZ_AGES, CDF_XYZ_REPORTED)), ldf_kind="cdf").fit(xyz, "reported")
xdev_p = ChainLadder(ldf=dict(zip(XYZ_AGES, CDF_XYZ_PAID)), ldf_kind="cdf").fit(xyz, "paid")

BOOK_XDEV = {  # chapter 7, Exhibit II, sheet 3, columns (7) and (8)
    "reported": [15822, 25082, 36948, 38487, 48313, 44950, 74787, 76661, 58370, 47979, 47530],
    "paid": [15980, 25164, 37922, 40600, 49592, 49858, 80537, 80333, 72108, 77941, 74995],
}


def compare(years, columns, width=8):
    """Print prospicio's ultimates beside the book's; columns: name -> (ultimates, book, book total)."""
    print("year " + " ".join(f"{n:>{width}s} {'book':>{width}s}" for n in columns))
    for i, y in enumerate(years):
        print(f"{y:4d} " + " ".join(f"{u[i]:{width},.0f} {b[i]:{width},}" for u, b, _ in columns.values()))
    print(" all " + " ".join(f"{sum(u):{width},.0f} {t:{width},}" for u, _, t in columns.values()))


compare(XYZ_YEARS, {"reported": (xdev_r.ultimate, BOOK_XDEV["reported"], 514929),
                    "paid": (xdev_p.ultimate, BOOK_XDEV["paid"], 605030)})
```

```text
year reported     book     paid     book
1998   15,822   15,822   15,980   15,980
1999   25,082   25,082   25,164   25,164
2000   36,948   36,948   37,922   37,922
2001   38,488   38,487   40,599   40,600
2002   48,314   48,313   49,592   49,592
2003   44,950   44,950   49,858   49,858
2004   74,786   74,787   80,537   80,537
2005   76,661   76,661   80,332   80,333
2006   58,370   58,370   72,108   72,108
2007   47,979   47,979   77,941   77,941
2008   47,530   47,530   74,995   74,995
 all  514,929  514,929  605,028  605,030
```

The reported total equals the book's 514,929 and the paid total is 2 below
605,030. Single years differ by 1 (reported 2001, 2002 and 2004, paid 2001 and
2005), in both directions. That is the rounding of the data, not of the method:
the book's spreadsheet holds claims with decimals and prints them in whole
thousands. For example, 38,798 times 0.992 is 38,487.6, which rounds to 38,488,
but a latest value of 38,797.5 (printed as 38,798) gives the book's 38,487.
Section 3 shows the same effect in a printed column total.

The two projections tell different stories for the latest years (paid 74,995
against reported 47,530 for 2008), because the speed-up in settlement inflates
the paid factors and the stronger case reserves inflate the reported ones. The
book returns to this in chapter 13; see section 8 below.

## 3. The expected claims technique (chapter 8)

The expected claims technique ignores the claims to date: ultimate is an
expected claim ratio times earned premium. For U.S. Industry Auto, Friedland
selects 75% for 1998 to 2002 and 65% for 2003 to 2007 (chapter 8, Exhibit II,
sheet 1). When the ratio differs by year, put the expected claims themselves in
a column and fit with `apriori=1.0`, as [`reserving-methods.md`](reserving-methods.md)
section 4 explains:

```python
US_RATIO = [0.75] * 5 + [0.65] * 5
us_expected = [r * p for r, p in zip(US_RATIO, US_PREMIUM)]
us = triangle(US_YEARS, {"reported": US_REPORTED, "paid": US_PAID},
              {"premium": US_PREMIUM, "expected": us_expected})
ec = ExpectedLoss(apriori=1.0).fit(us, "reported", "expected")
ibnr, total = unpaid(ec.ultimate)
print(f"expected claims {sum(ec.ultimate):,.0f} (book 569,281,839)")
print(f"IBNR {ibnr:,.0f} (book 25,800,252), total unpaid {total:,.0f} (book 71,231,471)")
print("IBNR 2000:", f"{ec.reserve[2]:,.0f} (book -3,156,946)")
```

```text
expected claims 569,281,839 (book 569,281,839)
IBNR 25,800,252 (book 25,800,252), total unpaid 71,231,471 (book 71,231,471)
IBNR 2000: -3,156,946 (book -3,156,946)
```

All three totals equal the book's sheet 2. The expected claims method can give
negative IBNR, as in 2000, where the 75% ratio is below what is already
reported; the book keeps such values to show the method's weakness on mature
years.

For XYZ the selected ratios come from a long adjustment for trend, rate changes
and tort reform (chapter 8, Exhibit III, sheet 2); the guide takes them as
printed in sheet 1, column (11):

```python
XYZ_RATIO = [0.783] * 6 + [0.871, 0.783, 0.658, 0.638, 0.825]
xyz_expected = [r * p for r, p in zip(XYZ_RATIO, XYZ_PREMIUM)]
xyz = triangle(XYZ_YEARS, {"reported": XYZ_REPORTED, "paid": XYZ_PAID},
               {"premium": XYZ_PREMIUM, "expected": xyz_expected}, first_age=XYZ_FIRST_AGE)
xec = ExpectedLoss(apriori=1.0).fit(xyz, "reported", "expected")
ibnr, total = unpaid(xec.ultimate, xyz)
print(f"expected claims {sum(xec.ultimate):,.0f} (book 561,516)")
print(f"IBNR {ibnr:,.0f} (book 111,890), total unpaid {total:,.0f} (book 230,887)")
```

```text
expected claims 561,516 (book 561,516)
IBNR 111,890 (book 111,890), total unpaid 230,889 (book 230,887)
```

The expected claims and the IBNR agree. The total unpaid is 2 above the book's
because it subtracts paid claims: the book's paid total at December 31, 2008 is
330,629, while its eleven printed paid values add to 330,627. This is the
clearest evidence that the exhibits hold claims with decimals and print them
rounded.

## 4. The Bornhuetter-Ferguson technique (chapter 9)

Bornhuetter-Ferguson keeps the claims to date and adds expected claims times
the share still unreported (or unpaid), `1 - 1 / CDF`. With the expected claims
of chapter 8 and the rounded CDFs of chapter 7, prospicio reproduces chapter 9,
Exhibit I:

```python
BOOK_BF_US = {  # chapter 9, Exhibit I, sheet 1, columns (11) and (12)
    "reported": [47742304, 51185767, 54889558, 56462300, 58947116,
                 58178105, 58317678, 59754938, 60778247, 62835336],
    "paid": [47746843, 51205350, 54841461, 56470405, 58972346,
             58096743, 58447423, 60151912, 61425942, 63209774],
}
bf = {
    col: BornhuetterFerguson(apriori=1.0, ldf=US_CDF[col], ldf_kind="cdf").fit(us, col, "expected")
    for col in ("reported", "paid")
}
BOOK_BF_US_TOTAL = {"reported": 569091348, "paid": 570568198}
compare(US_YEARS, {col: (bf[col].ultimate, BOOK_BF_US[col], BOOK_BF_US_TOTAL[col]) for col in bf}, width=11)
for col, book in [("reported", (25609761, 71040980)), ("paid", (27086611, 72517830))]:
    ibnr, total = unpaid(bf[col].ultimate)
    print(f"{col:8s} IBNR {ibnr:11,.0f} (book {book[0]:11,})  total unpaid {total:11,.0f} (book {book[1]:11,})")
```

```text
year    reported        book        paid        book
1998  47,742,304  47,742,304  47,746,843  47,746,843
1999  51,185,767  51,185,767  51,205,350  51,205,350
2000  54,889,558  54,889,558  54,841,461  54,841,461
2001  56,462,300  56,462,300  56,470,405  56,470,405
2002  58,947,116  58,947,116  58,972,346  58,972,346
2003  58,178,105  58,178,105  58,096,743  58,096,743
2004  58,317,678  58,317,678  58,447,423  58,447,423
2005  59,754,938  59,754,938  60,151,912  60,151,912
2006  60,778,247  60,778,247  61,425,942  61,425,942
2007  62,835,336  62,835,336  63,209,774  63,209,774
 all 569,091,348 569,091,348 570,568,198 570,568,198
reported IBNR  25,609,761 (book  25,609,761)  total unpaid  71,040,980 (book  71,040,980)
paid     IBNR  27,086,611 (book  27,086,611)  total unpaid  72,517,830 (book  72,517,830)
```

Every figure agrees. For XYZ the book adds one convention: in chapter 9,
Exhibit II the CDFs are **capped at a minimum of 1.00**, so that a year whose
reported claims are expected to fall gets no negative unreported claims. The
capped pattern is a valid `ldf_kind="cdf"`:

```python
capped = [max(c, 1.0) for c in CDF_XYZ_REPORTED]
XYZ_CDF = {"reported": dict(zip(XYZ_AGES, capped)), "paid": dict(zip(XYZ_AGES, CDF_XYZ_PAID))}
BOOK_BF_XYZ = {  # chapter 9, Exhibit II, sheet 1, columns (11) and (12)
    "reported": [15822, 25107, 37246, 38798, 48312, 45068, 75492, 79129, 60404, 45221, 42607],
    "paid": [15977, 25158, 37841, 40525, 49417, 50768, 82593, 94301, 71205, 45636, 41049],
}
xbf = {
    col: BornhuetterFerguson(apriori=1.0, ldf=XYZ_CDF[col], ldf_kind="cdf").fit(xyz, col, "expected")
    for col in ("reported", "paid")
}
BOOK_BF_XYZ_TOTAL = {"reported": 513207, "paid": 554471}
compare(XYZ_YEARS, {col: (xbf[col].ultimate, BOOK_BF_XYZ[col], BOOK_BF_XYZ_TOTAL[col]) for col in xbf})
```

```text
year reported     book     paid     book
1998   15,822   15,822   15,977   15,977
1999   25,107   25,107   25,158   25,158
2000   37,246   37,246   37,841   37,841
2001   38,798   38,798   40,525   40,525
2002   48,312   48,312   49,417   49,417
2003   45,068   45,068   50,768   50,768
2004   75,492   75,492   82,593   82,593
2005   79,129   79,129   94,301   94,301
2006   60,404   60,404   71,205   71,205
2007   45,221   45,221   45,636   45,636
2008   42,607   42,607   41,049   41,049
 all  513,207  513,207  554,469  554,471
```

Every year agrees for both. The paid total is 2 below the book's 554,471, which
is also 2 above the sum of the book's own eleven paid rows: the spreadsheet adds
unrounded values, as in section 2.

## 5. The Benktander technique (chapter 9)

Benktander's method runs Bornhuetter-Ferguson twice: the first BF ultimate
becomes the expected claims of the second. The book shows it only on its
synthetic "changing conditions" scenarios (chapter 9, Exhibits V and VI), whose
triangles this guide does not transcribe. Its formula, Exhibit V's column (11),
is reported claims plus the BF ultimate times the percentage unreported. Fed
with the book's printed U.S. Industry BF ultimates, that formula is a check on
prospicio's `Benktander(n_iters=2)`:

```python
gb = Benktander(apriori=1.0, n_iters=2, ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "expected")
latest = us.latest_diagonal()[0][us.columns.index("reported")]
cdf_by_year = CDF_REPORTED[::-1]  # the oldest year has the oldest age
by_hand = [r + (1 - 1 / c) * u for r, c, u in zip(latest, cdf_by_year, BOOK_BF_US["reported"])]
print(f"{'year':>4s} {'prospicio':>11s} {'formula':>11s} {'dev':>11s} {'BF':>11s}")
for y, a, b, d, f in zip(US_YEARS, gb.ultimate, by_hand, BOOK_DEV["reported"], BOOK_BF_US["reported"]):
    print(f"{y:4d} {a:11,.0f} {b:11,.0f} {d:11,} {f:11,}")
print("largest difference under 0.5:", max(abs(a - b) for a, b in zip(gb.ultimate, by_hand)) < 0.5)
```

```text
year   prospicio     formula         dev          BF
1998  47,742,304  47,742,304  47,742,304  47,742,304
1999  51,185,767  51,185,767  51,185,767  51,185,767
2000  54,892,764  54,892,764  54,892,767  54,889,558
2001  56,468,442  56,468,442  56,468,461  56,462,300
2002  58,944,285  58,944,285  58,944,268  58,947,116
2003  58,198,340  58,198,340  58,198,563  58,178,105
2004  58,287,807  58,287,807  58,287,120  58,317,678
2005  59,686,031  59,686,031  59,682,517  59,754,938
2006  60,664,409  60,664,409  60,651,886  60,778,247
2007  63,054,738  63,054,738  63,118,803  62,835,336
largest difference under 0.5: True
```

The two columns agree in every year, within the rounding of the printed BF
ultimates. Each Benktander ultimate lies between the BF and the
development ultimates, closer to development where the year is mature, which is
the book's description of the method as a credibility weighting that always
gives more weight to development than BF does.

## 6. The Cape Cod technique (chapter 10)

Cape Cod is Bornhuetter-Ferguson with an expected claim ratio estimated from
the data: reported claims over "used-up" premium, the premium times the
percentage reported. For U.S. Industry Auto (chapter 10, Exhibit I):

```python
cc = CapeCod(ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "premium")
print(f"expected claim ratio {cc.apriori[0]:.5f} (book prints 69.5%)")
print(f"ultimate {sum(cc.ultimate):,.0f} (book 570,800,677)")
ibnr, total = unpaid(cc.ultimate)
print(f"IBNR {ibnr:,.0f} (book 27,319,090), total unpaid {total:,.0f} (book 72,750,309)")
flat = BornhuetterFerguson(apriori=0.695, ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "premium")
print(f"BF at exactly 69.5%: ultimate {sum(flat.ultimate):,.0f}")
```

```text
expected claim ratio 0.69544 (book prints 69.5%)
ultimate 570,800,677 (book 570,800,677)
IBNR 27,319,090 (book 27,319,090), total unpaid 72,750,309 (book 72,750,309)
BF at exactly 69.5%: ultimate 570,783,246
```

The totals agree to the unit. The ratio the book prints as 69.5% is 69.544%
in its spreadsheet, which the last line shows: at exactly 69.5% the ultimate
would be 17,431 lower.

For XYZ (chapter 10, Exhibit II) Friedland first puts every year on the 2008
basis: premium at current rate levels (column (4), typed below), claims trended
at 3.425% a year to 2008 and adjusted for the tort reform (column (8)). The
ratio of adjusted reported claims to used-up on-level premium is 70.8%, and
each year's expected claim ratio is that ratio taken back to the year's own
cost level, tort environment and rate level. prospicio's `CapeCod(trend=...)`
applies a trend of this kind, with the ratio of the latest origin as the
reference; the tort factor and the rate level are put in by hand. The ratio is
estimated on the adjusted claims, and the expected claims are then used in a
BF fit on the actual claims:

```python
ON_LEVEL_PREMIUM = [19783, 30548, 42784, 46606, 55911, 60204, 80411, 97258, 68850, 49950, 47797]
TORT = [0.670] * 8 + [0.750, 1.000, 1.000]
adjusted = [[v * t for v in row] for row, t in zip(XYZ_REPORTED, TORT)]
xyz_cc = triangle(XYZ_YEARS, {"adjusted": adjusted, "reported": XYZ_REPORTED},
                  {"on_level": ON_LEVEL_PREMIUM}, first_age=XYZ_FIRST_AGE)
ratio = CapeCod(trend=0.03425, ldf=XYZ_CDF["reported"], ldf_kind="cdf").fit(xyz_cc, "adjusted", "on_level")
print(f"adjusted expected claim ratio {ratio.apriori[-1]:.4f} (book 70.8%)")
cc_expected = [a * p / t for a, p, t in zip(ratio.apriori, ON_LEVEL_PREMIUM, TORT)]
xyz_cc = triangle(XYZ_YEARS, {"reported": XYZ_REPORTED}, {"cc_expected": cc_expected}, first_age=XYZ_FIRST_AGE)
xcc = BornhuetterFerguson(apriori=1.0, ldf=XYZ_CDF["reported"], ldf_kind="cdf").fit(xyz_cc, "reported", "cc_expected")
print("unadjusted ratios", " ".join(f"{e / p:.3f}" for e, p in zip(cc_expected, XYZ_PREMIUM)))
BOOK_CC_XYZ = [15822, 25107, 37246, 38798, 48313, 45062, 74754, 77931, 58759, 43307, 39201]
compare(XYZ_YEARS, {"Cape Cod": (xcc.ultimate, BOOK_CC_XYZ, 504300)})
```

```text
adjusted expected claim ratio 0.7078 (book 70.8%)
unadjusted ratios 0.746 0.757 0.767 0.778 0.789 0.777 0.747 0.672 0.565 0.547 0.708
year Cape Cod     book
1998   15,822   15,822
1999   25,107   25,107
2000   37,246   37,246
2001   38,798   38,798
2002   48,313   48,313
2003   45,063   45,062
2004   74,754   74,754
2005   77,931   77,931
2006   58,759   58,759
2007   43,307   43,307
2008   39,201   39,201
 all  504,300  504,300
```

The adjusted ratio, the unadjusted ratios by year (the book's column (15):
74.6%, 75.7%, ... 70.8%) and the total ultimate of 504,300 all agree; 2003 is 1
above, the rounding of the printed data again. The tort factor has to multiply
the claims, not divide the premium: the ratio is a ratio of sums, and the two
are not the same once the years have different factors.

## 7. Reasonableness checks (chapters 8 and 15)

The book's last step is to read the results outside the frame that produced
them: claim ratios, severities, and how far the methods agree. For U.S. Industry
Auto, chapter 15 summarizes the unpaid claims of each method in billions of
dollars. prospicio's fits from the sections above give the same table:

```python
book_billions = {  # chapter 15, IBNR and total unpaid, $ billions
    "Development, reported": (26, 71), "Development, paid": (29, 74),
    "Expected claims": (26, 71), "B-F, reported": (26, 71),
    "B-F, paid": (27, 73), "Cape Cod": (27, 73),
}
fits = [dev_r, dev_p, ec, bf["reported"], bf["paid"], cc]
print(f"{'method':22s} {'IBNR':>5s} {'book':>5s} {'total':>6s} {'book':>5s}")
for (name, book), fit in zip(book_billions.items(), fits):
    ibnr, total = unpaid(fit.ultimate)
    print(f"{name:22s} {ibnr / 1e6:5.0f} {book[0]:5d} {total / 1e6:6.0f} {book[1]:5d}")
```

```text
method                  IBNR  book  total  book
Development, reported     26    26     71    71
Development, paid         29    29     74    74
Expected claims           26    26     71    71
B-F, reported             26    26     71    71
B-F, paid                 27    27     73    73
Cape Cod                  27    27     73    73
```

All twelve figures agree (the book's row for the case outstanding development
technique of chapter 12 is not reproduced; see section 8). The spread across
methods is small, about 4% of the unpaid estimate, as one expects of a whole
industry.

For XYZ the methods disagree, and the claim ratios show why. Chapter 8,
Exhibit III, sheet 1, column (10) prints the claim ratio of the average of the
reported and paid development ultimates; beside it, the ratio implied by each
method:

```python
BOOK_AVG_RATIO = [79.5, 79.8, 83.2, 79.1, 80.0, 68.5, 78.2, 56.8, 60.6, 100.8, 128.2]
methods = {
    "dev rep": xdev_r, "dev paid": xdev_p, "EC": xec,
    "BF rep": xbf["reported"], "BF paid": xbf["paid"], "CC": xcc,
}
print("year  avg dev   book " + " ".join(f"{n:>8s}" for n in methods))
for i, y in enumerate(XYZ_YEARS):
    avg = (xdev_r.ultimate[i] + xdev_p.ultimate[i]) / 2 / XYZ_PREMIUM[i]
    print(f"{y:4d} {100 * avg:7.1f}% {BOOK_AVG_RATIO[i]:5.1f}% "
          + " ".join(f"{100 * f.ultimate[i] / XYZ_PREMIUM[i]:7.1f}%" for f in methods.values()))
```

```text
year  avg dev   book  dev rep dev paid       EC   BF rep  BF paid       CC
1998    79.5%  79.5%    79.1%    79.9%    78.3%    79.1%    79.9%    79.1%
1999    79.8%  79.8%    79.6%    79.9%    78.3%    79.7%    79.9%    79.7%
2000    83.2%  83.2%    82.1%    84.3%    78.3%    82.8%    84.1%    82.8%
2001    79.1%  79.1%    77.0%    81.2%    78.3%    77.6%    81.0%    77.6%
2002    80.0%  80.0%    79.0%    81.1%    78.3%    79.0%    80.8%    79.0%
2003    68.5%  68.5%    65.0%    72.1%    78.3%    65.2%    73.4%    65.1%
2004    78.2%  78.2%    75.3%    81.1%    87.1%    76.0%    83.2%    75.3%
2005    56.8%  56.8%    55.5%    58.1%    78.3%    57.3%    68.3%    56.4%
2006    60.6%  60.6%    54.3%    67.0%    65.8%    56.1%    66.2%    54.6%
2007   100.8% 100.8%    76.8%   124.8%    63.8%    72.4%    73.1%    69.4%
2008   128.2% 128.2%    99.4%   156.9%    82.5%    89.1%    85.9%    82.0%
```

The average-of-development column equals the book's to the tenth of a percent.
The development ratios for 2007 and 2008 (77% and 99% reported, 125% and 157%
paid) are out of line with every earlier year, while the expected claims, BF
and Cape Cod methods keep them near the history. Chapter 15 also divides
ultimates by ultimate claim counts (severity) and divides IBNR plus case by the
open and unreported counts; with a counts triangle those are the same one-line
divisions of `fit.ultimate` and `fit.reserve`. The book's conclusion for XYZ
rests on the Berquist-Sherman adjusted projections, which prospicio does not
make.

## 8. What prospicio does not cover

The book's techniques that this guide could not run, and why:

- **Frequency-severity techniques (chapter 11).** Claim counts and severities
  projected separately, with trend, disposal rates and the claims closure
  model. prospicio can develop a counts triangle with `ChainLadder` like any
  other column, but it has no severity trend fit, disposal-rate projection or
  closure model.
- **Case outstanding development technique (chapter 12).** Projecting unpaid
  claims from case outstanding with report-year or self-insurer benchmarks is
  not a prospicio method.
- **Berquist-Sherman adjustments (chapter 13).** Restating case outstanding
  for a change in case adequacy (by trending average case reserves) and paid
  claims for a change in settlement speed (by disposal rates) is not built in;
  an adjusted triangle prepared by hand can then be developed with any method
  here.
- **Recoveries (chapter 14) and claim adjustment expenses (chapters 16 and
  17).** Salvage, subrogation and reinsurance are separate triangles developed
  the same way, which works; the ratio approaches to ALAE and the paid-to-paid
  and Kittel methods for ULAE are not methods of the library.
- **Adjustments for trend, rate level and tort reform** in the expected claims
  method are inputs prepared outside prospicio (section 3). `CapeCod(trend=)`
  and `decay=` cover the Cape Cod trend only.
- **Monitoring between reviews** (chapter 15's actual against expected for DC
  Insurer) is close to prospicio's diagonal backtest, which is described in
  [`reserving-methods.md`](reserving-methods.md) section 9.

## See also

- [`reserving-methods.md`](reserving-methods.md): which reserving method to use
  when, with Mack, the bootstraps, Clark, the Bayesian models and the one-year
  view, none of which are in Friedland's basic text.
- `docs/design/triangle.md` and `docs/design/reserving-v02.md`: the triangle
  and the expected-loss methods as designed, including selected patterns.
