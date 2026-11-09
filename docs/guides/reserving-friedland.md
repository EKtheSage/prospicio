# Friedland's basic reserving techniques, reproduced

<!-- site: execute -->

Jacqueline Friedland's *Estimating Unpaid Claims Using Basic Techniques*
(Casualty Actuarial Society, version 3, 2010) applies each technique to the
same data sets, so each chapter feeds the next: the chapter 7 factors become
the reporting pattern of chapters 9 and 10, and the chapter 8 expected claims
become the prior of chapter 9. The tables put the book's figures next to
prospicio's.

Exhibit numbers restart in every chapter. Money is in thousands of dollars. To
choose a method for your own data, read [`reserving-methods.md`](reserving-methods.md).

## 1. The two data sets

- **U.S. Industry Auto**: reported and paid claims and earned premium of the
  whole U.S. private passenger auto industry, accident years 1998 to 2007,
  valued at December 31, 2007. The data is large and stable, so the methods
  should agree.
- **XYZ Insurer, Auto BI**: the bodily injury book of a small insurer, accident
  years 1998 to 2008, valued at December 31, 2008. Its case reserving,
  settlement speed and a tort reform all changed, so the methods disagree. The
  data has only calendar years 2000 to 2008, so the two oldest rows start at 36
  and 24 months.

The triangles are from chapter 7, Exhibits I and II (sheets 1 and 2), the
premiums from chapter 8, Exhibits II and III:

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

The XYZ rows for 1998 and 1999 start with unobserved cells (`NaN`). They stay
missing, not zero, so no factor uses them.

## 2. The development technique (chapter 7)

### Age-to-age factors and their averages

A link ratio is a cell divided by the cell before it in the same row.
Friedland averages the **latest** few accident years of each column and selects
one factor for each age: simple averages of the latest 5 and 3, a medial
average of the latest 5 (without the highest and lowest), volume-weighted
averages of the latest 5 and 3, and a geometric average of the latest 4.

`ChainLadder` takes each as an option:

- `average=` selects the mean (`"volume"`, `"simple"`, `"regression"` or
  `"geometric"`).
- `n_periods=n` keeps only the link ratios that start in the latest n calendar
  diagonals (the latest n accident years at each age).
- `drop_high=` and `drop_low=` remove the highest and lowest of those. A medial
  average is a simple average with one of each removed.

`n_links` counts the link ratios in each factor's average:

```python
import pandas as pd
from great_tables import GT

AVERAGES = {
    "simple, latest 5": ChainLadder(average="simple", n_periods=5),
    "simple, latest 3": ChainLadder(average="simple", n_periods=3),
    "medial, latest 5x1": ChainLadder(average="simple", n_periods=5, drop_high=1, drop_low=1),
    "volume, latest 5": ChainLadder(n_periods=5),
    "volume, latest 3": ChainLadder(n_periods=3),
    "geometric, latest 4": ChainLadder(average="geometric", n_periods=4),
}
fits = {name: method.fit(us, "reported") for name, method in AVERAGES.items()}
LINKS = [f"{a}-{a + 12}" for a in us.development[:-1]]
rows = [[name, *fit.ldf] for name, fit in fits.items()]
rows.append(["links in the medial average", *fits["medial, latest 5x1"].n_links])
factors = pd.DataFrame(rows, columns=["average", *LINKS])
(GT(factors, rowname_col="average")
 .tab_header("U.S. Industry Auto, reported: age-to-age factors", "Months; the last row counts the link ratios in the medial average")
 .fmt_number(LINKS, rows=list(range(6)), decimals=3)
 .fmt_integer(LINKS, rows=[6]))
```

```text
                             12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
simple, latest 5             1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001    1.000
simple, latest 3             1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001    1.000
medial, latest 5x1           1.165  1.057  1.027  1.010  1.004  1.003  1.002   1.001    1.000
volume, latest 5             1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001    1.000
volume, latest 3             1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001    1.000
geometric, latest 4          1.164  1.057  1.027  1.011  1.004  1.003  1.002   1.001    1.000
links in the medial average      3      3      3      3      3      2      1       2        1
```

Each factor row equals the book's chapter 7, Exhibit I, sheet 1, part 3, to
three decimals. The paid averages on sheet 2 agree.

```python
import matplotlib.pyplot as plt

reported = us.view("reported")
link = reported.iloc[:, 1:].to_numpy() / reported.iloc[:, :-1].to_numpy()
ages = [int(a) for a in reported.columns[:-1]]
fig, ax = plt.subplots()
for row in link:
    ax.plot(ages, 100 * (row - 1), color="0.75", linewidth=1)
ax.plot([], [], color="0.75", label="one accident year")
ax.plot(ages, [100 * (f - 1) for f in fits["volume, latest 3"].ldf], marker="o", label="volume, latest 3")
ax.set_yscale("log")
ax.set_xticks(ages)
ax.set_xlabel("age at the start of the link, months")
ax.set_ylabel("link ratio minus 1, % (log scale)")
ax.set_title("U.S. Industry Auto, reported: link ratios and the latest-3 average")
ax.legend()
```

prospicio follows the book's two conventions:

- When fewer years exist, "latest 5" uses all of them.
- The medial average of two values is their simple average, and of one value
  the value itself. prospicio skips a drop that leaves no link ratio.

In the last row of the first table, 3 of the five link ratios at 12-24 remain,
2 of the four at 72-84 and 1 of the three at 84-96. The two at 96-108 and the
one at 108-120 all remain.

All-year averages, which the book does not use:

```python
RECENT = {
    "volume, latest 3": ChainLadder(n_periods=3),
    "volume, all years": ChainLadder(),
    "simple, all years": ChainLadder(average="simple"),
}
recent = pd.DataFrame([[name, *m.fit(us, "reported").ldf] for name, m in RECENT.items()],
                      columns=["average", *LINKS])
(GT(recent, rowname_col="average")
 .tab_header("U.S. Industry Auto, reported: latest 3 against all years", "Age-to-age factors, months")
 .fmt_number(LINKS, decimals=3))
```

```text
                   12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
volume, latest 3   1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001    1.000
volume, all years  1.175  1.058  1.027  1.011  1.004  1.003  1.002   1.001    1.000
simple, all years  1.175  1.058  1.027  1.011  1.004  1.003  1.002   1.001    1.000
```

The all-year factor at 12-24 (1.175) is higher than the book's 1.164 because
the early accident years developed more in their first year.

### Selected factors, factors to ultimate, and rounding

Friedland selects the latest-3 averages rounded to three decimals, with no
reported tail and a paid tail of 1.002. Their product is the factor to ultimate
(CDF), which she rounds to three decimals before she applies it to the latest
diagonal. Pass the factors as `ldf=` with `ldf_kind="ldf"`, and the tail as
`tail=`:

```python
AGES = us.development
SEL_REPORTED = [1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000]
SEL_PAID = [1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002]
dev_r = ChainLadder(ldf=dict(zip(AGES, SEL_REPORTED)), ldf_kind="ldf").fit(us, "reported")
dev_p = ChainLadder(ldf=dict(zip(AGES, SEL_PAID)), ldf_kind="ldf", tail=1.002).fit(us, "paid")
cdfs = pd.DataFrame([["reported", *dev_r.cdf], ["paid", *dev_p.cdf]], columns=["claims", *map(str, AGES)])
(GT(cdfs, rowname_col="claims")
 .tab_header("U.S. Industry Auto: factors to ultimate from the selected factors", "CDF by age in months")
 .fmt_number([str(a) for a in AGES], decimals=5))
```

```text
               12       24       36       48       60       72       84       96      108      120
reported  1.29163  1.10964  1.05080  1.02317  1.01104  1.00601  1.00300  1.00100  1.00000  1.00000
paid      2.39003  1.40425  1.18402  1.08526  1.03952  1.02014  1.01104  1.00601  1.00400  1.00200
```

The totals differ from the book's:

```python
def vs_book(items, title, subtitle=None):
    """prospicio against the book; items are (measure, prospicio, book or None, decimals)."""
    frame = pd.DataFrame({
        "measure": [i[0] for i in items],
        "prospicio": [i[1] for i in items],
        "book": [i[2] for i in items],
        "difference": [None if i[2] is None else i[1] - i[2] for i in items],
    })
    table = GT(frame, rowname_col="measure").tab_header(title, subtitle)
    for k, item in enumerate(items):
        table = table.fmt_number(["prospicio", "book", "difference"], rows=[k], decimals=item[3])
    return table.sub_missing(missing_text="")


vs_book([("reported", sum(dev_r.ultimate), 569172456, 0), ("paid", sum(dev_p.ultimate), 572041099, 0)],
        "U.S. Industry Auto: total ultimate from the unrounded CDFs", "Thousands of dollars")
```

```text
            prospicio         book  difference
reported  569,136,323  569,172,456     -36,133
paid      572,052,982  572,041,099      11,883
```

The cause is the rounding of the CDF: the reported factors multiply to 1.29163
from 12 months, but the book uses 1.292. Pass the book's rounded CDFs with
`ldf_kind="cdf"`. A pattern that reaches the oldest age with no `tail=` keeps
its last value as the tail, so the paid CDF of 1.002 at 120 months is the tail:

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


def compare(years, columns, title, subtitle=None, ratio=None):
    """prospicio's ultimates beside the book's; columns: name -> (ultimates, book, book total)."""
    frame = pd.DataFrame({"year": [*map(str, years), "all"]})
    if ratio is not None:
        frame["ratio"] = [*ratio, None]
    pairs = {}
    for name, (ultimate, book, total) in columns.items():
        frame[name] = [*ultimate, sum(ultimate)]
        frame[f"{name} (book)"] = [*book, total]
        pairs[name] = [name, f"{name} (book)"]
    values = [c for p in pairs.values() for c in p]
    table = GT(frame, rowname_col="year").tab_header(title, subtitle).fmt_integer(values)
    if ratio is not None:
        table = table.fmt_number("ratio", decimals=3).cols_label(ratio="expected claim ratio")
    for name, cols in pairs.items():
        table = table.tab_spanner(name, cols)
    labels = {c: ("book" if c.endswith("(book)") else "prospicio") for c in values}
    return table.cols_label(**labels).sub_missing(missing_text="")


compare(US_YEARS, {"reported": (dev_r.ultimate, BOOK_DEV["reported"], 569172456),
                   "paid": (dev_p.ultimate, BOOK_DEV["paid"], 572041099)},
        "U.S. Industry Auto: ultimate by accident year", "Development technique, thousands of dollars")
```

```text
paid tail 1.002
        prospicio         book    prospicio         book
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
all   569,172,456  569,172,456  572,041,099  572,041,099
```

Each ultimate now equals sheet 3. On sheet 4, IBNR is ultimate less reported
claims and total unpaid is ultimate less paid claims. prospicio's `reserve` is
ultimate less the latest value of the fitted column: IBNR on the reported
triangle, total unpaid on the paid one:

```python
def unpaid(ultimate, tri=us):
    reported = sum(tri.latest_diagonal()[0][tri.columns.index("reported")])
    paid = sum(tri.latest_diagonal()[0][tri.columns.index("paid")])
    return sum(ultimate) - reported, sum(ultimate) - paid


items = []
for name, fit, book in [("reported", dev_r, (25690869, 71122088)), ("paid", dev_p, (28559512, 73990731))]:
    ibnr, total = unpaid(fit.ultimate)
    items += [(f"{name}: IBNR", ibnr, book[0], 0), (f"{name}: total unpaid", total, book[1], 0)]
items.append(("paid: reserve", dev_p.total_reserve, 73990731, 0))
vs_book(items, "U.S. Industry Auto: unpaid claims, development technique", "Thousands of dollars")
```

```text
                         prospicio        book  difference
reported: IBNR          25,690,869  25,690,869           0
reported: total unpaid  71,122,088  71,122,088           0
paid: IBNR              28,559,512  28,559,512           0
paid: total unpaid      73,990,731  73,990,731           0
paid: reserve           73,990,731  73,990,731           0
```

### The tail factor

Friedland's paid tail of 1.002 is a judgment: it sets the paid development
after 120 months equal to the claims still open in the reported pattern (at
1.000). A fitted exponential curve gives a factor near hers:

```python
curve = ChainLadder(tail=TailCurve()).fit(us, "paid")
print(f"fitted tail {curve.tail:.4f}, book's selection 1.002")
```

```text
fitted tail 1.0011, book's selection 1.002
```

The early factors pull a curve fitted to all ages. Use it as a second opinion
beside a benchmark, as the book does, not as the answer.

### XYZ Insurer: unstable factors and a pattern below 1

For XYZ, Friedland selects the volume-weighted average of the latest 2 years at
every age, with no reported tail and a paid tail of 1.010. `n_periods=2` gives
her selected factors (chapter 7, Exhibit II, sheets 1 and 2, part 4):

```python
XYZ_LINKS = [f"{a}-{a + 12}" for a in xyz.development[:-1]]
xfactors = pd.DataFrame([[column, *ChainLadder(n_periods=2).fit(xyz, column).ldf] for column in ("reported", "paid")],
                        columns=["claims", *XYZ_LINKS])
(GT(xfactors, rowname_col="claims")
 .tab_header("XYZ Insurer: volume-weighted factors, latest 2 years", "Age-to-age factors, months")
 .fmt_number(XYZ_LINKS, decimals=3))
```

```text
          12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
reported  1.687  1.265  1.102  1.020  1.050  1.010  1.011   1.000    0.993    0.999
paid      3.349  2.079  1.574  1.316  1.203  1.136  1.059   1.022    1.017    1.004
```

The book prints the same factors. Reported claims fall after 96 months
because the insurer set case reserves too high and then released them, so the
reported CDFs at 96 and 108 months are 0.992, below 1. A pattern with
`ldf_kind="cdf"` accepts any positive CDFs, so the book's rounded CDFs go in
directly:

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
compare(XYZ_YEARS, {"reported": (xdev_r.ultimate, BOOK_XDEV["reported"], 514929),
                    "paid": (xdev_p.ultimate, BOOK_XDEV["paid"], 605030)},
        "XYZ Insurer: ultimate by accident year", "Development technique, thousands of dollars")
```

```text
      prospicio     book  prospicio     book
1998     15,822   15,822     15,980   15,980
1999     25,082   25,082     25,164   25,164
2000     36,948   36,948     37,922   37,922
2001     38,488   38,487     40,599   40,600
2002     48,314   48,313     49,592   49,592
2003     44,950   44,950     49,858   49,858
2004     74,786   74,787     80,537   80,537
2005     76,661   76,661     80,332   80,333
2006     58,370   58,370     72,108   72,108
2007     47,979   47,979     77,941   77,941
2008     47,530   47,530     74,995   74,995
all     514,929  514,929    605,028  605,030
```

The reported total equals the book's. The paid total is 2 less, and some single
years differ by 1 either way (reported 2001, 2002 and 2004, paid 2001 and
2005). The cause is the book's data, not the method: its spreadsheet holds
claims with decimals and prints whole thousands. For example, 38,798 times
0.992 is 38,487.6, which rounds to 38,488, but 38,797.5 (printed as 38,798)
gives the book's 38,487. Section 3 shows the same effect in a column total.

For the latest years the projections differ: faster settlement raises the paid
factors, and stronger case reserves raise the reported ones. The book returns
to this in chapter 13.

## 3. The expected claims technique (chapter 8)

The ultimate is an expected claim ratio times earned premium; the claims to
date are ignored. For U.S. Industry Auto, Friedland selects 75% for 1998 to
2002 and 65% for 2003 to 2007 (chapter 8, Exhibit II, sheet 1). A ratio that
changes by year is `apriori` as a dict keyed by year; the premium stays in the
exposure column (see section 4 of [`reserving-methods.md`](reserving-methods.md)):

```python
US_RATIO = [0.75] * 5 + [0.65] * 5
us = triangle(US_YEARS, {"reported": US_REPORTED, "paid": US_PAID}, {"premium": US_PREMIUM})
US_APRIORI = dict(zip(US_YEARS, US_RATIO))
ec = ExpectedLoss(apriori=US_APRIORI).fit(us, "reported", "premium")
ibnr, total = unpaid(ec.ultimate)
vs_book([("expected claims", sum(ec.ultimate), 569281839, 0), ("IBNR", ibnr, 25800252, 0),
         ("total unpaid", total, 71231471, 0), ("IBNR 2000", ec.reserve[2], -3156946, 0)],
        "U.S. Industry Auto: expected claims technique", "Thousands of dollars")
```

```text
                   prospicio         book  difference
expected claims  569,281,839  569,281,839           0
IBNR              25,800,252   25,800,252           0
total unpaid      71,231,471   71,231,471           0
IBNR 2000         -3,156,946   -3,156,946           0
```

All figures equal sheet 2. IBNR is negative for 2000, where the 75% ratio is
below the reported claims. The book keeps such values to show the method's
weakness on mature years.

For XYZ, the ratios come from a long adjustment for trend, rate changes and
tort reform (chapter 8, Exhibit III, sheet 2). This guide takes them from sheet
1, column (11):

```python
XYZ_RATIO = [0.783] * 6 + [0.871, 0.783, 0.658, 0.638, 0.825]
xyz = triangle(XYZ_YEARS, {"reported": XYZ_REPORTED, "paid": XYZ_PAID},
               {"premium": XYZ_PREMIUM}, first_age=XYZ_FIRST_AGE)
XYZ_APRIORI = dict(zip(XYZ_YEARS, XYZ_RATIO))
xec = ExpectedLoss(apriori=XYZ_APRIORI).fit(xyz, "reported", "premium")
ibnr, total = unpaid(xec.ultimate, xyz)
vs_book([("expected claims", sum(xec.ultimate), 561516, 0), ("IBNR", ibnr, 111890, 0),
         ("total unpaid", total, 230887, 0)],
        "XYZ Insurer: expected claims technique", "Thousands of dollars")
```

```text
                 prospicio     book  difference
expected claims    561,516  561,516           0
IBNR               111,890  111,890           0
total unpaid       230,889  230,887           2
```

The total unpaid is 2 more than the book's because the book's paid total at
December 31, 2008 is 330,629, but its eleven printed paid values add to
330,627 (the exhibits hold claims with decimals). The other rows agree.

## 4. The Bornhuetter-Ferguson technique (chapter 9)

Bornhuetter-Ferguson (BF) keeps the claims to date and adds expected claims
times the unreported (or unpaid) share, `1 - 1 / CDF`. With chapter 8's
expected claims and chapter 7's rounded CDFs, prospicio reproduces chapter 9,
Exhibit I:

```python
BOOK_BF_US = {  # chapter 9, Exhibit I, sheet 1, columns (11) and (12)
    "reported": [47742304, 51185767, 54889558, 56462300, 58947116,
                 58178105, 58317678, 59754938, 60778247, 62835336],
    "paid": [47746843, 51205350, 54841461, 56470405, 58972346,
             58096743, 58447423, 60151912, 61425942, 63209774],
}
bf = {
    col: BornhuetterFerguson(apriori=US_APRIORI, ldf=US_CDF[col], ldf_kind="cdf").fit(us, col, "premium")
    for col in ("reported", "paid")
}
BOOK_BF_US_TOTAL = {"reported": 569091348, "paid": 570568198}
compare(US_YEARS, {col: (bf[col].ultimate, BOOK_BF_US[col], BOOK_BF_US_TOTAL[col]) for col in bf},
        "U.S. Industry Auto: Bornhuetter-Ferguson ultimate by accident year", "Thousands of dollars")
```

```text
        prospicio         book    prospicio         book
1998   47,742,304   47,742,304   47,746,843   47,746,843
1999   51,185,767   51,185,767   51,205,350   51,205,350
2000   54,889,558   54,889,558   54,841,461   54,841,461
2001   56,462,300   56,462,300   56,470,405   56,470,405
2002   58,947,116   58,947,116   58,972,346   58,972,346
2003   58,178,105   58,178,105   58,096,743   58,096,743
2004   58,317,678   58,317,678   58,447,423   58,447,423
2005   59,754,938   59,754,938   60,151,912   60,151,912
2006   60,778,247   60,778,247   61,425,942   61,425,942
2007   62,835,336   62,835,336   63,209,774   63,209,774
all   569,091,348  569,091,348  570,568,198  570,568,198
```

```python
items = []
for col, book in [("reported", (25609761, 71040980)), ("paid", (27086611, 72517830))]:
    ibnr, total = unpaid(bf[col].ultimate)
    items += [(f"{col}: IBNR", ibnr, book[0], 0), (f"{col}: total unpaid", total, book[1], 0)]
vs_book(items, "U.S. Industry Auto: unpaid claims, Bornhuetter-Ferguson", "Thousands of dollars")
```

```text
                         prospicio        book  difference
reported: IBNR          25,609,761  25,609,761           0
reported: total unpaid  71,040,980  71,040,980           0
paid: IBNR              27,086,611  27,086,611           0
paid: total unpaid      72,517,830  72,517,830           0
```

All figures agree. For XYZ, the book adds one convention: chapter 9, Exhibit II
**caps the CDFs at a minimum of 1.00**, so a year with a CDF below 1 gets no
negative unreported claims:

```python
capped = [max(c, 1.0) for c in CDF_XYZ_REPORTED]
XYZ_CDF = {"reported": dict(zip(XYZ_AGES, capped)), "paid": dict(zip(XYZ_AGES, CDF_XYZ_PAID))}
BOOK_BF_XYZ = {  # chapter 9, Exhibit II, sheet 1, columns (11) and (12)
    "reported": [15822, 25107, 37246, 38798, 48312, 45068, 75492, 79129, 60404, 45221, 42607],
    "paid": [15977, 25158, 37841, 40525, 49417, 50768, 82593, 94301, 71205, 45636, 41049],
}
xbf = {
    col: BornhuetterFerguson(apriori=XYZ_APRIORI, ldf=XYZ_CDF[col], ldf_kind="cdf").fit(xyz, col, "premium")
    for col in ("reported", "paid")
}
BOOK_BF_XYZ_TOTAL = {"reported": 513207, "paid": 554471}
compare(XYZ_YEARS, {col: (xbf[col].ultimate, BOOK_BF_XYZ[col], BOOK_BF_XYZ_TOTAL[col]) for col in xbf},
        "XYZ Insurer: Bornhuetter-Ferguson ultimate by accident year", "Thousands of dollars")
```

```text
      prospicio     book  prospicio     book
1998     15,822   15,822     15,977   15,977
1999     25,107   25,107     25,158   25,158
2000     37,246   37,246     37,841   37,841
2001     38,798   38,798     40,525   40,525
2002     48,312   48,312     49,417   49,417
2003     45,068   45,068     50,768   50,768
2004     75,492   75,492     82,593   82,593
2005     79,129   79,129     94,301   94,301
2006     60,404   60,404     71,205   71,205
2007     45,221   45,221     45,636   45,636
2008     42,607   42,607     41,049   41,049
all     513,207  513,207    554,469  554,471
```

Every year agrees. The paid total is 2 less than the book's 554,471, which is
itself 2 more than the sum of the book's eleven paid rows (unrounded values, as
in section 2).

## 5. The Benktander technique (chapter 9)

Benktander's method applies BF twice: the first BF ultimate becomes the
expected claims of the second. The book shows it only on synthetic "changing
conditions" scenarios (chapter 9, Exhibits V and VI), so this guide checks the
formula of Exhibit V, column (11), reported claims plus the BF ultimate times
the percentage unreported, on the book's U.S. Industry BF ultimates.
`Benktander(n_iters=2)` matches it:

```python
gb = Benktander(apriori=US_APRIORI, n_iters=2, ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "premium")
latest = us.latest_diagonal()[0][us.columns.index("reported")]
cdf_by_year = CDF_REPORTED[::-1]  # the oldest year has the oldest age
by_hand = [r + (1 - 1 / c) * u for r, c, u in zip(latest, cdf_by_year, BOOK_BF_US["reported"])]
print("largest difference under 0.5:", max(abs(a - b) for a, b in zip(gb.ultimate, by_hand)) < 0.5)
```

```text
largest difference under 0.5: True
```

```python
benktander = pd.DataFrame({
    "year": [str(y) for y in US_YEARS], "prospicio": gb.ultimate, "formula": by_hand,
    "dev": BOOK_DEV["reported"], "BF": BOOK_BF_US["reported"],
})
(GT(benktander, rowname_col="year")
 .tab_header("U.S. Industry Auto: Benktander ultimate, reported", "Thousands of dollars; dev and BF are the book's ultimates")
 .fmt_integer(["prospicio", "formula", "dev", "BF"]))
```

```text
       prospicio     formula         dev          BF
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
```

The columns agree in every year, within the rounding of the printed BF
ultimates. Each Benktander ultimate lies between the BF and the development
ultimate, nearer development for mature years: the book calls the method a
credibility weighting that always weights development more than BF does.

## 6. The Cape Cod technique (chapter 10)

Cape Cod is BF with an expected claim ratio from the data: reported claims
divided by "used-up" premium (premium times the percentage reported). For U.S.
Industry Auto (chapter 10, Exhibit I):

```python
cc = CapeCod(ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "premium")
ibnr, total = unpaid(cc.ultimate)
flat = BornhuetterFerguson(apriori=0.695, ldf=US_CDF["reported"], ldf_kind="cdf").fit(us, "reported", "premium")
vs_book([("expected claim ratio", cc.apriori[0], 0.695, 5), ("ultimate", sum(cc.ultimate), 570800677, 0),
         ("IBNR", ibnr, 27319090, 0), ("total unpaid", total, 72750309, 0),
         ("ultimate, BF at exactly 69.5%", sum(flat.ultimate), None, 0)],
        "U.S. Industry Auto: Cape Cod", "Thousands of dollars; the book prints the ratio as 69.5%")
```

```text
                                 prospicio         book  difference
expected claim ratio               0.69544      0.69500     0.00044
ultimate                       570,800,677  570,800,677           0
IBNR                            27,319,090   27,319,090           0
total unpaid                    72,750,309   72,750,309           0
ultimate, BF at exactly 69.5%  570,783,246                         
```

The totals agree to the unit. The book's spreadsheet holds a ratio of 69.544%,
not the printed 69.5%; at exactly 69.5% the ultimate is 17,431 lower.

For XYZ (chapter 10, Exhibit II), Friedland first puts every year on the 2008
basis: premium at current rate levels (column (4), typed below), and claims
trended at 3.425% a year to 2008 and adjusted for the tort reform (column (8)).

The ratio of adjusted reported claims to used-up on-level premium is 70.8%.
Each year's ratio is that ratio taken back to the year's cost level, tort
environment and rate level. `CapeCod(trend=...)` applies such a trend, with the
latest origin as reference. The code applies the tort factor and rate level by
hand, estimates the ratio on the adjusted claims, then gives each year's ratio
on the actual premium to a BF fit on the actual claims:

```python
ON_LEVEL_PREMIUM = [19783, 30548, 42784, 46606, 55911, 60204, 80411, 97258, 68850, 49950, 47797]
TORT = [0.670] * 8 + [0.750, 1.000, 1.000]
adjusted = [[v * t for v in row] for row, t in zip(XYZ_REPORTED, TORT)]
xyz_cc = triangle(XYZ_YEARS, {"adjusted": adjusted, "reported": XYZ_REPORTED},
                  {"on_level": ON_LEVEL_PREMIUM}, first_age=XYZ_FIRST_AGE)
ratio = CapeCod(trend=0.03425, ldf=XYZ_CDF["reported"], ldf_kind="cdf").fit(xyz_cc, "adjusted", "on_level")
print(f"adjusted expected claim ratio {ratio.apriori[-1]:.4f} (book 70.8%)")
cc_expected = [a * p / t for a, p, t in zip(ratio.apriori, ON_LEVEL_PREMIUM, TORT)]
cc_ratio = {y: e / p for y, e, p in zip(XYZ_YEARS, cc_expected, XYZ_PREMIUM)}
xcc = BornhuetterFerguson(apriori=cc_ratio, ldf=XYZ_CDF["reported"], ldf_kind="cdf").fit(xyz, "reported", "premium")
BOOK_CC_XYZ = [15822, 25107, 37246, 38798, 48313, 45062, 74754, 77931, 58759, 43307, 39201]
compare(XYZ_YEARS, {"Cape Cod": (xcc.ultimate, BOOK_CC_XYZ, 504300)},
        "XYZ Insurer: Cape Cod ultimate by accident year",
        "Thousands of dollars; the ratio is on the actual premium", ratio=list(cc_ratio.values()))
```

```text
adjusted expected claim ratio 0.7078 (book 70.8%)
      expected claim ratio  prospicio     book
1998                 0.746     15,822   15,822
1999                 0.757     25,107   25,107
2000                 0.767     37,246   37,246
2001                 0.778     38,798   38,798
2002                 0.789     48,313   48,313
2003                 0.777     45,063   45,062
2004                 0.747     74,754   74,754
2005                 0.672     77,931   77,931
2006                 0.565     58,759   58,759
2007                 0.547     43,307   43,307
2008                 0.708     39,201   39,201
all                           504,300  504,300
```

The adjusted ratio, the ratios by year (column (15): 74.6%, 75.7%, ... 70.8%)
and the total of 504,300 all agree. The 2003 ultimate is 1 more than the
book's, from the rounding of the printed data. The tort factor must multiply
the claims, not divide the premium: the ratio is a ratio of sums, so the two
differ when the factors differ by year.

## 7. Reasonableness checks (chapters 8 and 15)

The book's last step checks the results against claim ratios, severities and
agreement between methods. Chapter 15 gives the unpaid claims of each method
for U.S. Industry Auto in billions of dollars. The fits above reproduce them:

```python
book_billions = {  # chapter 15, IBNR and total unpaid, $ billions
    "Development, reported": (26, 71), "Development, paid": (29, 74),
    "Expected claims": (26, 71), "B-F, reported": (26, 71),
    "B-F, paid": (27, 73), "Cape Cod": (27, 73),
}
fits = [dev_r, dev_p, ec, bf["reported"], bf["paid"], cc]
summary = pd.DataFrame({
    "method": list(book_billions),
    "IBNR, book": [b[0] for b in book_billions.values()],
    "IBNR, prospicio": [unpaid(f.ultimate)[0] / 1e6 for f in fits],
    "total, book": [b[1] for b in book_billions.values()],
    "total, prospicio": [unpaid(f.ultimate)[1] / 1e6 for f in fits],
})
(GT(summary, rowname_col="method")
 .tab_header("U.S. Industry Auto: unpaid claims by method", "$ billions")
 .tab_spanner("IBNR", ["IBNR, book", "IBNR, prospicio"])
 .tab_spanner("total unpaid", ["total, book", "total, prospicio"])
 .cols_label(**{"IBNR, book": "book", "IBNR, prospicio": "prospicio",
                "total, book": "book", "total, prospicio": "prospicio"})
 .fmt_integer(["IBNR, book", "total, book"])
 .fmt_number(["IBNR, prospicio", "total, prospicio"], decimals=1))
```

```text
                       book  prospicio  book  prospicio
Development, reported    26       25.7    71       71.1
Development, paid        29       28.6    74       74.0
Expected claims          26       25.8    71       71.2
B-F, reported            26       25.6    71       71.0
B-F, paid                27       27.1    73       72.5
Cape Cod                 27       27.3    73       72.8
```

All twelve figures agree. The spread across methods is about 4% of the unpaid
estimate.

For XYZ the methods disagree, and the claim ratios show why. Chapter 8,
Exhibit III, sheet 1, column (10) prints the claim ratio of the average of the
reported and paid development ultimates, shown here beside each method's:

```python
BOOK_AVG_RATIO = [79.5, 79.8, 83.2, 79.1, 80.0, 68.5, 78.2, 56.8, 60.6, 100.8, 128.2]
methods = {
    "dev rep": xdev_r, "dev paid": xdev_p, "EC": xec,
    "BF rep": xbf["reported"], "BF paid": xbf["paid"], "CC": xcc,
}
ratios = pd.DataFrame({
    "year": [str(y) for y in XYZ_YEARS],
    "avg dev": [(xdev_r.ultimate[i] + xdev_p.ultimate[i]) / 2 / p for i, p in enumerate(XYZ_PREMIUM)],
    "book": [r / 100 for r in BOOK_AVG_RATIO],
    **{name: [u / p for u, p in zip(f.ultimate, XYZ_PREMIUM)] for name, f in methods.items()},
})
(GT(ratios, rowname_col="year")
 .tab_header("XYZ Insurer: ultimate claim ratio by method", "Percent of premium")
 .tab_spanner("average of the two development ultimates", ["avg dev", "book"])
 .tab_spanner("each method", list(methods))
 .cols_label(**{"avg dev": "prospicio"})
 .fmt_percent(list(ratios.columns[1:]), decimals=1))
```

```text
      prospicio    book  dev rep  dev paid     EC  BF rep  BF paid     CC
1998      79.5%   79.5%    79.1%     79.9%  78.3%   79.1%    79.9%  79.1%
1999      79.8%   79.8%    79.6%     79.9%  78.3%   79.7%    79.9%  79.7%
2000      83.2%   83.2%    82.1%     84.3%  78.3%   82.8%    84.1%  82.8%
2001      79.1%   79.1%    77.0%     81.2%  78.3%   77.6%    81.0%  77.6%
2002      80.0%   80.0%    79.0%     81.1%  78.3%   79.0%    80.8%  79.0%
2003      68.5%   68.5%    65.0%     72.1%  78.3%   65.2%    73.4%  65.1%
2004      78.2%   78.2%    75.3%     81.1%  87.1%   76.0%    83.2%  75.3%
2005      56.8%   56.8%    55.5%     58.1%  78.3%   57.3%    68.3%  56.4%
2006      60.6%   60.6%    54.3%     67.0%  65.8%   56.1%    66.2%  54.6%
2007     100.8%  100.8%    76.8%    124.8%  63.8%   72.4%    73.1%  69.4%
2008     128.2%  128.2%    99.4%    156.9%  82.5%   89.1%    85.9%  82.0%
```

```python
fig, ax = plt.subplots()
for name in ["EC", "BF rep", "BF paid"]:
    ax.plot(XYZ_YEARS, [100 * u / p for u, p in zip(methods[name].ultimate, XYZ_PREMIUM)],
            color="0.75", linewidth=1)
ax.plot([], [], color="0.75", label="EC, BF rep, BF paid")
for name in ["dev rep", "dev paid", "CC"]:
    ax.plot(XYZ_YEARS, [100 * u / p for u, p in zip(methods[name].ultimate, XYZ_PREMIUM)],
            marker="o", label=name)
ax.set_xlabel("accident year")
ax.set_ylabel("ultimate claim ratio, % of premium")
ax.set_title("XYZ Insurer: claim ratio by accident year and method")
ax.legend()
```

The development average equals the book's to the tenth of a percent. For 2007
and 2008 the development ratios are far above earlier years; expected claims,
BF and Cape Cod keep them near the history.

Chapter 15 also divides ultimates by claim counts (severity), and IBNR plus
case by open and unreported counts: with a counts triangle these are one-line
divisions of `fit.ultimate` and `fit.reserve`. The book's conclusion for XYZ
uses chapter 13's Berquist-Sherman projections.

## See also

- [`reserving-methods.md`](reserving-methods.md): which reserving method to use
  when, with Mack, the bootstraps, Clark, the Bayesian models and the one-year
  view.
- `docs/design/triangle.md` and `docs/design/reserving-v02.md`: the design of
  the triangle and the expected-loss methods, with selected patterns.
