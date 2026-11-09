# Excess-layer loss development: layer factors from the ground-up triangle

<!-- site: execute -->

The losses above an attachment (a reinsurer's excess layer, or a primary
insurer's high deductible) develop much more slowly than the ground-up losses
they come from. The layer's own triangle is sparse: most early cells are zero,
and the first non-zero link ratios are very large, so the chain ladder is
unstable or fails.

The usual method derives the layer's development from the dense ground-up
triangle and a claim size distribution, then blends it with the layer's own
data. `prospicio.reserving` has the parts (#192, design note
`docs/design/reserving-excess.md`):

* `ExcessPattern.from_ground_up` derives a layer pattern under a scale shift
  or under Pinto and Gogol's Pareto shape drift.
* `credibility_weights` blends the pattern with the layer's own factors.
* `layer_share`, `layer_exposure` and `LayerSharePrior` give the
  Bornhuetter-Ferguson prior of a layer.

Sections 2 and 3 run each part on the printed examples of the literature:
Bloom and Patrik, *A Methodology for Avoiding the Pitfalls of Excess Loss
Development* (CAS E-Forum, Fall 2013), pages 4 to 8; Pinto and Gogol, *An
Analysis of Excess Loss Development* (PCAS 74, 1987); and Bear's discussion of
it (PCAS 79, 1992).
`knowledge/references/excess-development-literature.md` summarises what each
source prints. Sections 4 to 6 use a simulated book with a sparse layer.

**How the numbers here are checked.** CI (`python/tests/test_guides.py`)
runs each block that has a text block after it and compares its output (a
table prints as its formatted cells) with that text.

## 1. Why a layer develops slowly

Suppose every claim grows by the same factor between two ages, as case
reserves do. Claims below the attachment give the layer nothing at the earlier
age, and some rise above it later, so the layer grows from the claims already
in it and from the claims that enter it. The table is the design note's own
calculation (not from a source): lognormal severity, log-sd 1.5, mean 65,000 at
ultimate, every claim 1.3 times its earlier value, fixed counts.

```python
import math

import pandas as pd
from great_tables import GT

from prospicio.distributions import Lognormal
from prospicio.reserving import ExcessLayer, ExcessPattern, ScaleShift, Triangle

ultimate = Lognormal(math.log(65_000.0) - 1.5**2 / 2, 1.5)  # mean 65,000
two_ages = Triangle.from_long([2020, 2020], [12, 24], {"incurred": [1.0, 1.3]})
note = {"1m xs 0": 1.275, "750k xs 250k": 1.545, "2m xs 1m": 1.761, "7m xs 3m": 1.982}
factors = []
for name, layer in zip(note, [ExcessLayer(0.0, 1e6), ExcessLayer(250e3, 750e3),
                              ExcessLayer(1e6, 2e6), ExcessLayer(3e6, 7e6)]):
    p = ExcessPattern.from_ground_up(two_ages, "incurred", layer, ScaleShift(ultimate))
    factors.append({"layer": name, "prospicio": p.layer_ldf[0], "design note": note[name]})
(GT(pd.DataFrame(factors), rowname_col="layer")
 .tab_header("Layer factor from 12 to 24 months", "The ground-up factor is 1.3")
 .fmt_number("prospicio", decimals=4)
 .fmt_number("design note", decimals=3))
```

```text
              prospicio  design note
1m xs 0          1.2754        1.275
750k xs 250k     1.5449        1.545
2m xs 1m         1.7612        1.761
7m xs 3m         1.9819        1.982
```

The higher the layer, the larger its factor (near 2 on the highest). The
layer's own triangle shows this only late, so the rest of this guide takes the
factors from the dense ground-up triangle.

## 2. Bloom and Patrik's example: the pitfall and the cure

Year One has four claims, reported at 100, 375, 250 and 500 at 12 months.
Ground-up losses develop by 3.60 from 12 to 24 months and by 1.25 from 24 to
36 months (the end of development), so every claim reaches 4.5 times its 12-month value.
Years Two and Three have the same claims with 5% inflation a year. The layer
is all loss above a deductible of 350.

Every claim has the same factor, so this is exactly a **scale shift**: the
claims at age `j` are the ultimate claims divided by a shift `c_j`, the
ground-up factor to ultimate (4.5, 1.25 and 1 at 12, 24 and 36 months). The
block builds the paper's triangle at the end of Year Three and sets its excess
column beside the paper's:

```python
from prospicio.reserving import ChainLadder

CLAIMS = [100.0, 375.0, 250.0, 500.0]   # Year One at 12 months
TO_ULTIMATE = {12: 4.5, 24: 1.25, 36: 1.0}
PRICE = {1: 1.0, 2: 1.05, 3: 1.05**2}  # 5% inflation a year
DEDUCTIBLE = 350.0

rows = []
for year, price in PRICE.items():
    for age, c in TO_ULTIMATE.items():
        if year + age // 12 <= 4:  # the cells known at the end of Year Three
            sizes = [x * price * 4.5 / c for x in CLAIMS]
            rows.append((year, age, sum(sizes), sum(max(s - DEDUCTIBLE, 0.0) for s in sizes),
                         sum(min(s, DEDUCTIBLE) for s in sizes)))
bp = Triangle.from_long([r[0] for r in rows], [r[1] for r in rows],
                        {"ground_up": [r[2] for r in rows], "excess": [r[3] for r in rows],
                         "limited": [r[4] for r in rows]})
printed = {(1, 12): 175, (1, 24): 3010, (1, 36): 4113, (2, 12): 219, (2, 24): 3231, (3, 12): 265}
excess_cells = [{"cell": f"Year {year}, {age} months", "prospicio": excess,
                 "paper": printed[(year, age)]} for year, age, _, excess, _ in rows]
(GT(pd.DataFrame(excess_cells), rowname_col="cell")
 .tab_header("Excess of 350 at the end of Year Three", "Currency units")
 .fmt_number("prospicio", decimals=2, use_seps=False)
 .fmt_integer("paper"))
```

```text
                   prospicio  paper
Year 1, 12 months     175.00    175
Year 1, 24 months    3010.00  3,010
Year 1, 36 months    4112.50  4,113
Year 2, 12 months     218.75    219
Year 2, 24 months    3230.50  3,231
Year 3, 12 months     264.69    265
```

The trap is to develop this excess triangle alone. Inflation moved more of
Year Two's claims above the deductible early, so the 12 to 24 month factor,
from Years One and Two, overstates the development of Year Three:

```python
own = ChainLadder().fit(bp, "excess")
truth = sum(max(x * PRICE[3] * 4.5 - DEDUCTIBLE, 0.0) for x in CLAIMS)
print(f"own factors {own.ldf[0]:.4f} {own.ldf[1]:.4f}")
```

```text
own factors 15.8489 1.3663
```

```python
overstated = pd.DataFrame({
    "measure": ["Year Three ultimate", "True ultimate", "Overstatement (%)"],
    "prospicio": [own.ultimate[2], truth, 100 * (own.ultimate[2] / truth - 1)],
    "paper": [5732, 4678, 22.5]})
(GT(overstated, rowname_col="measure")
 .tab_header("Developing the excess triangle alone", "Year Three, currency units")
 .fmt_number("prospicio", rows=[0, 1], decimals=1)
 .fmt_number("paper", rows=[0, 1], decimals=0)
 .fmt_number(["prospicio", "paper"], rows=[2], decimals=1))
```

```text
                     prospicio  paper
Year Three ultimate    5,731.5  5,732
True ultimate          4,677.5  4,678
Overstatement (%)         22.5   22.5
```

The cure is to trend the severity, not the pattern: cut the layer from each
year's claim sizes at that year's price level. The severity is each year's
ultimate claims (450, 1,687.5, 1,125 and 2,250 at Year One's prices), a `Grid`
of four equal point masses. A step of 112.5 hits all four exactly, so the layer
expectations are exact. `ExcessPattern.from_ground_up` solves the shifts from
the ground-up triangle and returns the layer's factors:

```python
from prospicio.distributions import Grid

def ultimate_claims(price):
    probs = [0.0] * 21
    for k in (4, 15, 10, 20):  # 450, 1,687.5, 1,125 and 2,250 at Year One's prices
        probs[k] = 0.25
    return Grid(112.5 * price, probs)

paper = {1: (17.20, 1.37, 23.50), 2: (14.77, 1.36, 20.06), 3: (13.08, 1.35, 17.67)}
patterns, by_year = {}, []
for year, price in PRICE.items():
    one_year = Triangle.from_long(
        [year] * 3, [12, 24, 36],
        {"ground_up": [sum(CLAIMS) * price * 4.5 / c for c in TO_ULTIMATE.values()]})
    p = ExcessPattern.from_ground_up(one_year, "ground_up", ExcessLayer(DEDUCTIBLE),
                                     ScaleShift(ultimate_claims(price)))
    patterns[year] = p
    by_year.append({"year": f"Year {year}", "12 to 24": p.layer_ldf[0],
                    "24 to 36": p.layer_ldf[1], "12 to ultimate": p.layer_cdf[0]})
    by_year.append({"year": f"Year {year}, paper", "12 to 24": paper[year][0],
                    "24 to 36": paper[year][1], "12 to ultimate": paper[year][2]})
columns = ["12 to 24", "24 to 36", "12 to ultimate"]
(GT(pd.DataFrame(by_year), rowname_col="year")
 .tab_header("Layer factors of each year", "Excess of 350")
 .fmt_number(columns, rows=[0, 2, 4], decimals=4)
 .fmt_number(columns, rows=[1, 3, 5], decimals=2))
```

```text
               12 to 24  24 to 36  12 to ultimate
Year 1          17.2000    1.3663         23.5000
Year 1, paper     17.20      1.37           23.50
Year 2          14.7680    1.3583         20.0600
Year 2, paper     14.77      1.36           20.06
Year 3          13.0797    1.3511         17.6719
Year 3, paper     13.08      1.35           17.67
```

The factors agree to the paper's two decimals, and the unrounded values are
exact (17.2 = 3,010 / 175). `shifts` holds the `c_j`, the ground-up factors to
ultimate, with the tail last (1 here). With a `counts=` column the count
development is divided out first; without one the counts are taken as
complete.

```python
print("shifts", " ".join(f"{c:g}" for c in patterns[1].shifts), "  ground-up factors",
      " ".join(f"{f:g}" for f in patterns[1].ground_up_ldf))
```

```text
shifts 4.5 1.25 1   ground-up factors 3.6 1.25
```

The pattern goes into any development method through `ldf`, `ldf_kind` and
`ldf_weight`: `p.selected()` gives them as a dictionary to unpack with `**`,
and `p.tail` is the layer's tail. One pattern serves every origin at the
severity's price level, so trend the severity to the origins' level first. Year
Three's pattern on the full triangle is exact for Year Three and a little low
for Year Two, whose claims are at the previous year's prices:

```python
p3 = patterns[3]
fit = ChainLadder(tail=p3.tail, **p3.selected()).fit(bp, "excess")
refit = []
for year, ult in zip(PRICE, fit.ultimate):
    true = sum(max(x * PRICE[year] * 4.5 - DEDUCTIBLE, 0.0) for x in CLAIMS)
    refit.append({"year": f"Year {year}", "ultimate": ult, "true": true,
                  "error (%)": 100 * (ult / true - 1)})
(GT(pd.DataFrame(refit), rowname_col="year")
 .tab_header("Year Three's pattern on the full triangle", "Excess of 350")
 .fmt_number(["ultimate", "true"], decimals=1, use_seps=False)
 .fmt_number("error (%)", decimals=2, force_sign=True))
```

```text
        ultimate    true  error (%)
Year 1    4112.5  4112.5       0.00
Year 2    4364.7  4388.1      -0.53
Year 3    4677.5  4677.5       0.00
```

Fit origins at different price levels or retentions in separate groups, each
with its own pattern.

**The excess loss factor route.** Bloom and Patrik's section 4 reaches the
same answer without a severity curve, from the dense triangle limited to 350.
The excess loss factor (ELF) is the share of ultimate ground-up loss above
350, and `excess CDF = ELF / (1 / CDF_unlimited - (1 - ELF) / CDF_limited)`
(Pinto and Gogol's layer identity). For Year Three, with the limited factor
from the chain ladder:

```python
limited = ChainLadder().fit(bp, "limited")
cdf_limited = limited.ldf[0] * limited.ldf[1]
limited_ult = rows[-1][4] * cdf_limited   # Year Three at 12 months
ground_up_ult = rows[-1][2] * 4.5
elf = 1.0 - limited_ult / ground_up_ult
route = pd.DataFrame({
    "measure": ["Limited factor, 12 to 24", "ELF", "Excess factor to ultimate",
                "From the paper's rounded inputs", "Scale shift, exact"],
    "prospicio": [limited.ldf[0], elf, elf / (1 / 4.5 - (1 - elf) / cdf_limited),
                  0.764 / (1 / 4.5 - (1 - 0.764) / 1.32), p3.layer_cdf[0]],
    "paper": [1.32, 0.764, 17.54, None, None]})
(GT(route, rowname_col="measure")
 .tab_header("Year Three by the excess loss factor route", "Excess of 350")
 .fmt_number("prospicio", decimals=4)
 .fmt_number("paper", rows=[0, 2], decimals=2)
 .fmt_number("paper", rows=[1], decimals=3)
 .sub_missing(missing_text=""))
```

```text
                                 prospicio  paper
Limited factor, 12 to 24            1.3223   1.32
ELF                                 0.7637  0.764
Excess factor to ultimate          17.5364  17.54
From the paper's rounded inputs    17.5898       
Scale shift, exact                 17.6719       
```

The paper's 17.54 (from unrounded inputs) is 0.77% below the exact 17.67,
because the limited factor, like the excess factor above, is estimated on older
years.

## 3. Shape drift: Pinto and Gogol, and Bear

Pinto and Gogol found in ISO data that the tail of the size distribution also
becomes heavier as claims mature: a Pareto fitted above a threshold `k` has a
shape `q` that falls with maturity. Their factor for the losses excess of `c k`
is the development of the losses excess of `k` times `c^(q_i - q_j)`; with the
counts above `k` as the base it also has a ratio of `(q - 1)` terms.
`ParetoShape` takes the threshold, a shape at every age and at ultimate, and
the base that the column holds (`base="counts"`, or an `ExcessLayer`). The
column then holds the base development, not the ground-up losses.

```python
from prospicio.reserving import ParetoShape

def shape_factor(base_development, threshold, shapes, layer, base):
    tri = Triangle.from_long([2020, 2020], [12, 24], {"base": [1.0, base_development]})
    model = ParetoShape(threshold, {12: shapes[0], 24: shapes[1]}, shapes[1], base=base)
    return ExcessPattern.from_ground_up(tri, "base", layer, model).layer_ldf[0]

# Pinto and Gogol p. 247: losses excess of 25,000 develop by 2.5, q from 1.75 to 1.25.
pg = shape_factor(2.5, 25_000.0, (1.75, 1.25), ExcessLayer(100_000.0), ExcessLayer(25_000.0))
# The same from the counts above 25,000: they develop by 2.5 (0.25 / 0.75).
pgc = shape_factor(2.5 * 0.25 / 0.75, 25_000.0, (1.75, 1.25), ExcessLayer(100_000.0), "counts")
# Bear p. 138: the layer 100,000 to 300,000 develops by 3.5, q from 1.25 to 1.10.
bear = shape_factor(3.5, 100_000.0, (1.25, 1.10), ExcessLayer(200_000.0, 400_000.0),
                    ExcessLayer(100_000.0, 200_000.0))
examples = pd.DataFrame({
    "example": ["Pinto and Gogol p. 247, excess of 100,000",
                "The same from the counts above 25,000",
                "Bear p. 138, layer 200,000 to 600,000"],
    "prospicio": [pg, pgc, bear], "paper": [5.0, None, 3.88]})  # 2.5 (4)^0.5 = 5.0
(GT(examples, rowname_col="example")
 .tab_header("Layer factors under shape drift", "Pinto and Gogol print 2.5 x 4^0.5 = 5.0")
 .fmt_number("prospicio", decimals=4)
 .fmt_number("paper", decimals=2)
 .sub_missing(missing_text=""))
```

```text
                                           prospicio  paper
Pinto and Gogol p. 247, excess of 100,000     5.0000   5.00
The same from the counts above 25,000         5.0000       
Bear p. 138, layer 200,000 to 600,000         3.8835   3.88
```

Pinto and Gogol's Appendix B works the shape drift from 27 months to ultimate
above 10,000 (`q` from 1.6 to 1.3). It prints the losses of several
retentions and layers as multiples of the losses excess of 10,000, to three
decimals. The ratio of the two ages is the layer's factor relative to the base
development `a`:

```python
appendix_b = {  # (from, to) in thousands: (at 27 months, at ultimate), as printed
    (100, None): (0.251, 0.501), (300, None): (0.130, 0.360), (500, None): (0.096, 0.309),
    (1000, None): (0.063, 0.251), (100, 300): (0.121, 0.141), (300, 500): (0.034, 0.051),
    (500, 1000): (0.033, 0.058),
}
appendix = []
for (lo, hi), (at27, ult) in appendix_b.items():
    layer = ExcessLayer(lo * 1e3) if hi is None else ExcessLayer(lo * 1e3, (hi - lo) * 1e3)
    f = shape_factor(1.0, 10_000.0, (1.6, 1.3), layer, ExcessLayer(10_000.0))
    name = f"excess of {lo:,}k" if hi is None else f"{lo:,}k to {hi:,}k"
    appendix.append({"layer": name, "prospicio": f, "ultimate": ult, "27 months": at27,
                     "paper ratio": ult / at27})
(GT(pd.DataFrame(appendix), rowname_col="layer")
 .tab_header("Pinto and Gogol's Appendix B", "Layer factor as a multiple of a")
 .fmt_number("prospicio", decimals=4)
 .fmt_number(["ultimate", "27 months", "paper ratio"], decimals=3))
```

```text
                  prospicio  ultimate  27 months  paper ratio
excess of 100k       1.9953     0.501      0.251        1.996
excess of 300k       2.7742     0.360      0.130        2.769
excess of 500k       3.2336     0.309      0.096        3.219
excess of 1,000k     3.9811     0.251      0.063        3.984
100k to 300k         1.1606     0.141      0.121        1.165
300k to 500k         1.4932     0.051      0.034        1.500
500k to 1,000k       1.7843     0.058      0.033        1.758
```

Excess of 100,000 is exactly `10^0.3 = 1.99526` times `a`; the paper's 1.996 is
the ratio of rounded amounts. The other rows agree within that rounding (0.033
can be anything from 0.0325 to 0.0335).

Shape drift needs a shape at every age, so claim-level data or layered losses
by maturity; a scale shift needs only the ground-up triangle. The two differ on
high layers when the tail becomes heavier. In a simulation study
(`knowledge/findings/excess-simulation-study.md`, 500 books per arm) with
drifting shapes, the scale shift understates the ultimate of the sparse layer by
about 26% and of the thin layer by 14%; `ParetoShape` with the true shapes is
unbiased.

## 4. A sparse layer on a simulated book

We simulate a book with a known truth: ten accident years, 2015 to 2024,
of 120 claims each, valued at the end of 2024. Ultimate claim sizes are
lognormal (mean about 30,000, log-sd 1.8). The reported value at each age is
the ultimate divided by a shift that falls from 3.0 at 12 months to 1.01 at 120
months (the last 1% is the tail). The book follows the scale shift by
construction, so it shows the mechanics and does not test the model; the
simulation study above is the test.

A small linear congruential generator, written out in the block, draws the
uniforms (the R twin draws the same claims). The block tracks a high layer
(750,000 xs 250,000) and a working layer (400,000 xs 100,000), and counts the
claims in each:

```python
SEV = Lognormal(8.7, 1.8)
YEARS = list(range(2015, 2025))
AGES = [12 * k for k in range(1, 11)]
SHIFT = [3.0, 1.8, 1.4, 1.2, 1.1, 1.05, 1.03, 1.02, 1.015, 1.01]  # ultimate / reported
N, PREMIUM, TAIL = 120, 5_500_000.0, 1.01
HIGH, WORKING = ExcessLayer(250_000.0, 750_000.0), ExcessLayer(100_000.0, 400_000.0)

def in_layer(x, layer):
    return min(max(x - layer.attachment, 0.0), layer.limit)

def simulate(seed):
    """One book: its triangle and each layer's true ultimate by origin."""
    state, claims = seed, {}
    for y in YEARS:
        u = []
        for _ in range(N):  # Park and Miller's minimal standard generator
            state = state * 16807 % 2147483647
            u.append(state / 2147483647)
        claims[y] = [SEV.quantile(v) for v in u]
    cells = {"origin": [], "age": [], "incurred": [], "premium": [], "high": [], "high_n": [],
             "working": [], "working_n": []}
    for y in YEARS:
        for j, age in enumerate(AGES):
            if (y - 2015) + j <= 9:  # the cells known at the end of 2024
                reported = [x / SHIFT[j] for x in claims[y]]
                cells["origin"].append(y)
                cells["age"].append(age)
                cells["incurred"].append(sum(reported))
                cells["premium"].append(PREMIUM)
                for name, layer in (("high", HIGH), ("working", WORKING)):
                    cells[name].append(sum(in_layer(v, layer) for v in reported))
                    cells[name + "_n"].append(float(sum(v > layer.attachment for v in reported)))
    book = Triangle.from_long(cells.pop("origin"), cells.pop("age"), cells)
    truth = {name: [sum(in_layer(x, layer) for x in claims[y]) for y in YEARS]
             for name, layer in (("high", HIGH), ("working", WORKING))}
    return book, truth

book, truth = simulate(20241231)
latest = dict(zip(book.columns, book.latest_diagonal()[0]))
latest_high = pd.DataFrame({
    "origin": [str(y) for y in YEARS], "ground-up": latest["incurred"],
    "high layer": latest["high"], "claims": latest["high_n"], "true high ultimate": truth["high"]})
latest_high.loc[len(latest_high)] = ["total", *latest_high.iloc[:, 1:].sum()]
(GT(latest_high, rowname_col="origin")
 .tab_header("The simulated book at the end of 2024", "Currency units; claims in the high layer")
 .fmt_integer(["ground-up", "high layer", "claims", "true high ultimate"]))
```

```text
        ground-up  high layer  claims  true high ultimate
2015    2,297,915           0       0                   0
2016    3,338,009     534,042       3             553,303
2017    2,509,633     255,071       2             270,172
2018    2,473,626     210,437       1             224,250
2019    3,066,471     565,998       2             619,298
2020    2,652,024     173,574       1             215,931
2021    2,499,806     101,581       1             206,309
2022    3,187,793     614,715       4           1,260,601
2023    2,103,713      17,007       1             494,749
2024      965,657           0       0             161,796
total  25,094,646   2,472,425      15           4,006,410
```

The high layer's own chain ladder fails: 2015, the only origin at the oldest
link, is zero, so that factor has no estimate. The working layer's runs, on
factors from a few claims:

```python
try:
    ChainLadder(tail=TAIL).fit(book, "high")
except ValueError as e:
    print("high layer, own chain ladder:", e)
own_working = ChainLadder(tail=TAIL).fit(book, "working")
print("working layer, own factors", " ".join(f"{f:.3f}" for f in own_working.ldf))
```

```text
high layer, own chain ladder: factor from development index 8: earlier values sum to zero
working layer, own factors 3.055 1.521 1.321 1.159 1.079 1.036 1.018 1.009 1.016
```

The derived patterns need only the ground-up column and the ultimate severity
(here the true one; in practice fit it to trended ultimate claims, for example
with `prospicio.distributions`). They are incurred patterns from
volume-weighted ground-up factors, and the ground-up tail of 1.01 gives each
layer its own, larger tail. The table's shift is the one at the start of each
link; the last row is the tail:

```python
derived = {}
for name, layer in (("high", HIGH), ("working", WORKING)):
    derived[name] = ExcessPattern.from_ground_up(book, "incurred", layer, ScaleShift(SEV),
                                                 tail=TAIL)
p = derived["high"]
dev = pd.DataFrame({
    "link": [f"{a} to {a + 12}" for a in AGES[:-1]] + ["tail"], "shift": p.shifts,
    "ground-up": [*p.ground_up_ldf, p.ground_up_tail],
    "high layer": [*derived["high"].layer_ldf, derived["high"].tail],
    "working layer": [*derived["working"].layer_ldf, derived["working"].tail]})
factor_columns = ["ground-up", "high layer", "working layer"]
(GT(dev, rowname_col="link")
 .tab_header("Age-to-age factors, in months", "Derived from the ground-up triangle")
 .fmt_number("shift", decimals=3)
 .fmt_number(factor_columns, rows=list(range(9)), decimals=3)
 .fmt_number(factor_columns, rows=[9], decimals=4))
```

```text
            shift  ground-up  high layer  working layer
12 to 24    3.000      1.667       2.446          2.180
24 to 36    1.800      1.286       1.513          1.431
36 to 48    1.400      1.167       1.278          1.236
48 to 60    1.200      1.091       1.145          1.124
60 to 72    1.100      1.048       1.074          1.064
72 to 84    1.050      1.019       1.030          1.026
84 to 96    1.030      1.010       1.015          1.013
96 to 108   1.020      1.005       1.008          1.006
108 to 120  1.015      1.005       1.008          1.006
tail        1.010     1.0100      1.0153         1.0131
```

```python
print("to ultimate at 12 months: ground-up", f"{SHIFT[0]:.2f},",
      ", ".join(f"{name} {q.layer_cdf[0]:.2f}" for name, q in derived.items()))
```

```text
to ultimate at 12 months: ground-up 3.00, high 6.27, working 4.91
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
link_ages = AGES[:-1]
ax.plot(link_ages, p.ground_up_ldf, marker="o", label="ground-up")
ax.plot(link_ages, derived["working"].layer_ldf, marker="o", label="working layer")
ax.plot(link_ages, derived["high"].layer_ldf, marker="o", label="high layer")
ax.set_yscale("log")
ax.set_yticks([1.0, 1.5, 2.0, 2.5], ["1.0", "1.5", "2.0", "2.5"])
ax.set_xticks(link_ages)
ax.set_xlabel("age at the start of the link (months)")
ax.set_ylabel("age-to-age factor (log scale)")
ax.set_title("The higher the layer, the slower it develops")
ax.legend()
```

The next blocks fit the chain ladder, Mack and the Bornhuetter-Ferguson (BF) on
the high layer with `**p.selected()` and `tail=p.tail`. The BF prior is the
ground-up expected loss ratio times the premium times the layer share (section
6):

```python
from prospicio.reserving import BornhuetterFerguson, Mack, layer_exposure, layer_share

elr = N * SEV.mean() / PREMIUM
share = layer_share(SEV, HIGH)
layered = layer_exposure(book, "premium", share, "high_premium")
cl = ChainLadder(tail=p.tail, **p.selected()).fit(book, "high")
mack = Mack(tail=p.tail, **p.selected()).fit(book, "high")
bf = BornhuetterFerguson(apriori=elr, tail=p.tail, **p.selected()).fit(
    layered, "high", "high_premium")
print(f"ground-up ELR {elr:.4f}, layer share {share:.4f}, "
      f"expected layer ultimate a year {elr * share * PREMIUM:,.0f}")
```

```text
ground-up ELR 0.6618, layer share 0.1603, expected layer ultimate a year 583,548
```

```python
ultimates = pd.DataFrame({
    "origin": [str(y) for y in YEARS], "chain ladder": cl.ultimate, "BF": bf.ultimate,
    "true": truth["high"]})
ultimates.loc[len(ultimates)] = ["total", *ultimates.iloc[:, 1:].sum()]
(GT(ultimates, rowname_col="origin")
 .tab_header("Ultimate loss in the high layer", "Currency units")
 .fmt_integer(["chain ladder", "BF", "true"]))
```

```text
       chain ladder         BF       true
2015              0      8,785          0
2016        546,307    547,143    553,303
2017        262,895    272,438    270,172
2018        220,153    236,191    224,250
2019        609,855    607,963    619,298
2020        200,936    253,038    215,931
2021        134,694    245,040    206,309
2022      1,041,936    853,985  1,260,601
2023         43,614    373,000    494,749
2024              0    490,530    161,796
total     3,060,390  3,888,113  4,006,410
```

```python
print(f"Mack reserve {mack.total_reserve:,.0f}, standard error {mack.total_standard_error:,.0f}")
```

```text
Mack reserve 587,966, standard error 210,007
```

```python
fig, ax = plt.subplots()
ax.bar(YEARS, [t / 1e3 for t in truth["high"]], color="0.75", label="true ultimate")
ax.plot(YEARS, [u / 1e3 for u in cl.ultimate], marker="o", label="chain ladder")
ax.plot(YEARS, [u / 1e3 for u in bf.ultimate], marker="o", label="Bornhuetter-Ferguson")
ax.set_xticks(YEARS)
ax.set_xlabel("accident year")
ax.set_ylabel("ultimate loss in the high layer (thousands)")
ax.set_title("The BF fills the immature years that the chain ladder leaves empty")
ax.legend(loc="upper left")
```

The chain ladder does not see unreported losses: 2024's layer is empty at 12
months, so its ultimate is zero, and the single small claim of 2023 develops to
only 43,614. The BF puts the expected layer loss on the unreported part, so it
is the usual method for an immature layer, and its total is nearer the truth.
Where the layer losses are nearly all reported (2016 to 2019), the methods
agree. This is one book; section 5 compares them over a hundred.

Mack's standard error is large against its reserve: the selected pattern has
no parameter error, and the process error of a few claims is large.

## 5. Blending with the layer's own development

When a layer has data of its own, blend the derived pattern with its own
factors: `f = w f_derived + (1 - w) f_own`, with `w = 1 - n / (n + k)` at each
age. `n` is the number of layer claims at the earlier age (with a `counts=`
column), or else the number of origins with a positive earlier value. `k` is a
required credibility constant; the simulation study's candidate, `k = 20`, is
used here. A cap limits the credibility further: removing any one origin moves
the blended factor by at most `max_influence` of it (0.1 here), so one very
large link ratio cannot dominate. Where the layer's own factor cannot be
estimated, the weight is exactly 1:

```python
weights = {}
for name in ("high", "working"):
    w = derived[name].credibility_weights(book, name, 20.0, 0.1, counts=name + "_n")
    weights[name + " layer"] = [w[a] for a in AGES[:-1]]
weights = pd.DataFrame({"link": [f"{a} to {a + 12}" for a in AGES[:-1]], **weights})
(GT(weights, rowname_col="link")
 .tab_header("Weight on the derived pattern", "Months; k = 20, max_influence = 0.1")
 .fmt_number(["high layer", "working layer"], decimals=3))
```

```text
            high layer  working layer
12 to 24         0.909          0.556
24 to 36         0.741          0.455
36 to 48         0.714          0.426
48 to 60         0.714          0.426
60 to 72         0.714          0.444
72 to 84         0.769          0.476
84 to 96         0.800          0.513
96 to 108        0.870          0.606
108 to 120       1.000          0.741
```

The high layer's few claims leave most of the weight on the derived pattern,
and exactly 1 at the oldest age, where the zero of 2015 gives no own factor.
Pass the weights to `selected`:
`ChainLadder(tail=p.tail, **p.selected(w))`.

One book cannot rank the methods, because a few large claims decide the result.
So we simulate a hundred books (seeds 1 to 100, the same model) and compare the
error of each layer's total ultimate under four methods: the layer's own chain
ladder, the derived pattern, the blend with `k = 20`, and the BF with the derived
pattern. Mean error and root mean square error (RMSE) are relative to the mean
true ultimate:

```python
errors = {}
for seed in range(1, 101):
    b, t = simulate(seed)
    for name, layer in (("high", HIGH), ("working", WORKING)):
        q = ExcessPattern.from_ground_up(b, "incurred", layer, ScaleShift(SEV), tail=TAIL)
        w = q.credibility_weights(b, name, 20.0, 0.1, counts=name + "_n")
        lb = layer_exposure(b, "premium", layer_share(SEV, layer), "layer_premium")
        methods = {
            "own chain ladder": lambda: ChainLadder(tail=TAIL).fit(b, name),
            "derived pattern": lambda: ChainLadder(tail=q.tail, **q.selected()).fit(b, name),
            "blend, k = 20": lambda: ChainLadder(tail=q.tail, **q.selected(w)).fit(b, name),
            "BF, derived": lambda: BornhuetterFerguson(apriori=elr, tail=q.tail, **q.selected())
            .fit(lb, name, "layer_premium"),
        }
        for label, fit in methods.items():
            try:
                error = sum(fit().ultimate) - sum(t[name])
            except ValueError:  # the own factors cannot be estimated
                error = None
            errors.setdefault((name, label), []).append((error, sum(t[name])))

study = []
for (name, label), v in errors.items():
    ok = [e for e, _ in v if e is not None]
    mean_truth = sum(t for _, t in v) / len(v)
    study.append({"method": f"{name}, {label}", "runs": len(ok),
                  "mean error (%)": 100 * sum(ok) / len(ok) / mean_truth,
                  "RMSE (%)": 100 * math.sqrt(sum(e * e for e in ok) / len(ok)) / mean_truth})
(GT(pd.DataFrame(study), rowname_col="method")
 .tab_header("Error of the total ultimate over 100 simulated books")
 .fmt_number("mean error (%)", decimals=1, force_sign=True)
 .fmt_number("RMSE (%)", decimals=1)
 .fmt_integer("runs"))
```

```text
                           runs  mean error (%)  RMSE (%)
high, own chain ladder       93           +19.2      95.1
high, derived pattern       100            -3.9      15.5
high, blend, k = 20         100            -1.9      19.4
high, BF, derived           100            +0.5       9.5
working, own chain ladder   100            -0.3      13.1
working, derived pattern    100            -2.0       8.1
working, blend, k = 20      100            -1.2       9.9
working, BF, derived        100            -0.6       4.4
```

The own chain ladder fails on seven high-layer books. Where it runs, it
overstates the high layer by a fifth, with an RMSE near the full ultimate,
because the early link ratios of a few claims are very large. The derived
pattern cuts the RMSE to a fraction of that on both layers; it is a little low because an origin with no loss yet in the
layer stays at zero. The BF is best, because it does not multiply the chance
results of the immature years, but the test favours it: it uses the true loss
ratio and severity, and a real prior only approximates them.

Here the model holds exactly, so the derived pattern needs no help. The reason
to blend is that the severity assumption may be wrong: in the simulation study,
under shape drift, a blend near `k = 20` beat both the derived pattern and the
own factors on the thin layer, and without the `max_influence` cap a small `k`
gave very large errors on the sparse layer.

## 6. A Bornhuetter-Ferguson prior from the layer share

The a-priori layer ultimate of origin `i` is its exposure times a ground-up
expected loss ratio times the layer share `s`, the expected share of ground-up
loss in the layer. For a severity, `s = (LEV(b) - LEV(a)) / mean` on the
ultimate basis (`layer_share`); for an exposure curve, use the curve's own
`layer_share(limit, attachment, mpl)` (`Mbbefd`, `TabulatedCurve`).
`layer_exposure` copies the triangle with the exposure column times the share
(one share, or one per origin), and each expected-loss method accepts the copy
as its exposure. Cape Cod, unlike the BF of section 4, estimates the layer's
loss ratio from the layer's own used-up exposure, through the same derived
pattern:

```python
from prospicio.reserving import CapeCod

cc = CapeCod(tail=p.tail, **p.selected()).fit(layered, "high", "high_premium")
true_ratio = sum(truth["high"]) / (len(YEARS) * PREMIUM * share)
methods = pd.DataFrame({
    "method": ["BF (the ground-up ELR)", "Cape Cod", "True"],
    "loss ratio": [elr, cc.apriori[0], true_ratio],
    "total ultimate": [sum(bf.ultimate), sum(cc.ultimate), sum(truth["high"])]})
(GT(methods, rowname_col="method")
 .tab_header("Loss ratio and total ultimate of the high layer",
             "Loss ratio to the layer premium; currency units")
 .fmt_number("loss ratio", decimals=4)
 .fmt_integer("total ultimate"))
```

```text
                        loss ratio  total ultimate
BF (the ground-up ELR)      0.6618       3,888,113
Cape Cod                    0.3702       3,264,359
True                        0.4544       4,006,410
```

This book is light: its layer losses are about two thirds of what the prior
expects. Cape Cod follows the experience down, below the true ratio; the BF
keeps the prior's weight on the immature years.

The share is an estimate, and its error belongs in the distribution of the
reserve. `LayerSharePrior` puts it there through the normalised Bayesian BF
(`BayesBfNormalised`): the layer's expected ultimate gets a gamma prior with
mean `ELR x share x exposure` and, for a product of independent variables,
`cv^2 = (1 + cv_elr^2)(1 + cv_share^2) - 1`. The model estimates its pattern
from the layer's own triangle under a vague prior, so the derived excess
pattern does not enter and an own factor is needed at every age. The high
layer has none at its oldest age, so this part uses the working layer:

```python
from prospicio.reserving import BayesBfNormalised, LayerSharePrior

working_share = layer_share(SEV, WORKING)
prior = LayerSharePrior(elr, working_share, apriori_cv=0.1, share_cv=0.3)
print(f"working layer share {working_share:.4f}; prior cv {prior.cv:.4f} "
      f"= sqrt(1.01 x 1.09 - 1) = {math.sqrt(1.01 * 1.09 - 1):.4f}")
fit = prior.fit(book, "working", "premium", model=BayesBfNormalised(seed=1, n_sims=2000))
print(f"prior ultimate a year {fit.prior_ultimate[0]:,.0f} "
      f"= {elr:.4f} x {working_share:.4f} x {PREMIUM:,.0f}")
```

```text
working layer share 0.2590; prior cv 0.3176 = sqrt(1.01 x 1.09 - 1) = 0.3176
prior ultimate a year 942,605 = 0.6618 x 0.2590 x 5,500,000
```

The posterior is a Monte Carlo result whose last digits differ between
platforms (see [`reserving-meyers.md`](reserving-meyers.md)), so the guide
prints only the exact prior. `fit.reserves` is the predictive distribution of
the reserve.

The layer has one share, so its error is common to every origin.
`LayerSharePrior` models it so by default (`share_dependence="common"`, #518):
one gamma multiplier with mean 1 and cv `share_cv` applies to every origin's
prior and is drawn with the other parameters, so all origins update the share
together. `"per_origin"` draws an independent error at each origin. Each
origin's prior is the same under both, the prior of the total is not;
`prior_total` gives it in closed form:

```python
exposure = [PREMIUM] * len(YEARS)
for dependence in ("common", "per_origin"):
    p = LayerSharePrior(elr, working_share, apriori_cv=0.1, share_cv=0.3,
                        share_dependence=dependence)
    mean, sd = p.prior_total(exposure)
    print(f"{dependence:10}  prior total mean {mean:,.0f}  sd {sd:,.0f}")
```

```text
common      prior total mean 9,426,049  sd 2,844,887
per_origin  prior total mean 9,426,049  sd 946,837
```

Independent errors understate the prior sd of the total by a factor of three.
The posterior moves the other way: when the ten origins identify the share,
each origin stays near its prior with the loss ratio's own cv, so the common
share gives the narrower fitted interval.

On a validation layer (a quarter of GenIns's paid triangle, in
`knowledge/findings/excess-bayes-prior.md`), the mean total-reserve sd over seeds
1 to 20 (common share) or 10 (independent errors) was about 497,000 at a share
cv of 0.3 under the common share, and rose only about 2% as the cv went to 1.5.
Under independent errors it rose from 514,000 to 730,000 (about 42%). One fit's
sd moves by about 1.5% between seeds, so compare runs over several seeds. Where
the data cannot identify the share (few origins, or all origins immature), the
common model's total stays wider.

## See also

- [`reserving-methods.md`](reserving-methods.md): which reserving method to
  use when.
- [`choosing-a-method.md`](choosing-a-method.md): the library's methods by
  question.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md): claim-level development (IBNER)
  with `DevelopmentPattern`.
- `docs/design/reserving-excess.md`: the method, the API and the validation
  targets as designed, with the simulation study's results.
- `python/tests/test_reserving_excess.py` and
  `python/tests/test_reserving_bayes_excess_prior.py`: the bindings' tests.
