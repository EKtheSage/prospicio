# Excess-layer loss development: layer factors from the ground-up triangle

A reinsurer that writes an excess layer must develop the losses above an
attachment. A primary insurer that reserves a high deductible or an excess
policy must do the same. These losses develop much more slowly than the
ground-up losses that they come from. The layer's own triangle is sparse: most
early cells are zero, and the first non-zero link ratios are very large. On
such a triangle, the chain ladder is unstable or fails.

The usual method in the literature derives the layer's development from the
dense ground-up triangle and a claim size distribution. It then blends the
result with the layer's own data. `prospicio.reserving` has the parts of this
method (#192, design note `docs/design/reserving-excess.md`):

* `ExcessPattern.from_ground_up` derives a layer pattern under a scale shift
  or under Pinto and Gogol's Pareto shape drift.
* `credibility_weights` blends the pattern with the layer's own factors.
* `layer_share`, `layer_exposure` and `LayerSharePrior` give the
  Bornhuetter-Ferguson prior of a layer.

This guide shows each part on the printed examples of the literature (Bloom
and Patrik, Pinto and Gogol, Bear). It then uses a simulated book with a
sparse layer. Where a source prints a value, the guide prints it next to
prospicio's value.

The papers are their authors'. This guide explains them and calculates their
numbers again. Section 2 uses Bloom and Patrik, *A Methodology for Avoiding
the Pitfalls of Excess Loss Development* (CAS E-Forum, Fall 2013), pages 4 to
8. Section 3 uses Pinto and Gogol, *An Analysis of Excess
Loss Development* (PCAS 74, 1987), and Bear's discussion of it (PCAS 79,
1992). `knowledge/references/excess-development-literature.md` summarises
what each source prints. For a general choice of reserving method, read
[`reserving-methods.md`](reserving-methods.md).

**How the numbers here are checked.** CI (`python/tests/test_guides.py`)
runs each Python block that has a text block after it, and the block must
print exactly that text. The full guide runs in about a second.

## 1. Why a layer develops slowly

Suppose that every claim grows by the same factor between two ages, as the
case reserves increase. Claims below the attachment give nothing to the layer
at the earlier age, and some of them go above it later. Thus the layer grows
from two sources: the growth of the claims already in it, and the claims that
enter it. The design note shows this with its own calculation (not from a
source). It uses a lognormal severity with log-sd 1.5 and mean 65,000 at
ultimate, every claim 1.3 times its earlier value, and fixed counts. The
ground-up factor is 1.3, and the higher the layer, the larger the layer factor.

```python
import math

from prospicio.distributions import Lognormal
from prospicio.reserving import ExcessLayer, ExcessPattern, ScaleShift, Triangle

ultimate = Lognormal(math.log(65_000.0) - 1.5**2 / 2, 1.5)  # mean 65,000
two_ages = Triangle.from_long([2020, 2020], [12, 24], {"incurred": [1.0, 1.3]})
note = {"1m xs 0": 1.275, "750k xs 250k": 1.545, "2m xs 1m": 1.761, "7m xs 3m": 1.982}
for name, layer in zip(note, [ExcessLayer(0.0, 1e6), ExcessLayer(250e3, 750e3),
                              ExcessLayer(1e6, 2e6), ExcessLayer(3e6, 7e6)]):
    p = ExcessPattern.from_ground_up(two_ages, "incurred", layer, ScaleShift(ultimate))
    print(f"{name:13s} layer factor {p.layer_ldf[0]:.4f}   design note {note[name]:.3f}")
```

```text
1m xs 0       layer factor 1.2754   design note 1.275
750k xs 250k  layer factor 1.5449   design note 1.545
2m xs 1m      layer factor 1.7612   design note 1.761
7m xs 3m      layer factor 1.9819   design note 1.982
```

A ground-up factor of 1.3 gives a factor near 2 on a high layer. The layer's
own triangle shows this only late, because at the early ages it holds almost
nothing to develop. The rest of this guide gets the layer's factors from more
stable data.

## 2. Bloom and Patrik's example: the pitfall and the cure

Bloom and Patrik's example is small enough to calculate by hand. Year One has
four claims, reported at 100, 375, 250 and 500 at 12 months. Ground-up losses
develop by 3.60 from 12 to 24 months and by 1.25 from 24 to 36 months, the
end of development. Claim counts are fixed, so every claim reaches 4.5 times
its 12-month value. Years Two and Three have the same claims with 5% inflation
a year. The layer is all the loss above a deductible of 350.

Every claim has the same factor, so this is exactly a **scale shift**. The
claims at age `j` are the ultimate claims divided by a shift `c_j`. Here
`c_j` is the ground-up factor to ultimate (4.5 at 12 months, 1.25 at 24 and 1
at 36). The block below builds the triangle of the paper's three years at the
end of Year Three, with the ground-up, excess and limited columns:

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
for year, age, _, excess, _ in rows:
    paper = printed[(year, age)]
    print(f"Year {year} at {age} months: excess of 350 {excess:8.2f}   paper {paper:,}")
```

```text
Year 1 at 12 months: excess of 350   175.00   paper 175
Year 1 at 24 months: excess of 350  3010.00   paper 3,010
Year 1 at 36 months: excess of 350  4112.50   paper 4,113
Year 2 at 12 months: excess of 350   218.75   paper 219
Year 2 at 24 months: excess of 350  3230.50   paper 3,231
Year 3 at 12 months: excess of 350   264.69   paper 265
```

This is the paper's excess triangle, to the dollar. The trap is to develop it
alone. The 12 to 24 month factor comes from Years One and Two. Inflation moved
more of the claims of Year Two above the deductible early. Thus a factor
estimated on older years overstates the development of a newer year:

```python
own = ChainLadder().fit(bp, "excess")
truth = sum(max(x * PRICE[3] * 4.5 - DEDUCTIBLE, 0.0) for x in CLAIMS)
print(f"own factors {own.ldf[0]:.4f} {own.ldf[1]:.4f}")
print(f"Year Three ultimate {own.ultimate[2]:,.1f}   paper 5,732")
print(f"true ultimate       {truth:,.1f}   paper 4,678")
print(f"overstatement       {own.ultimate[2] / truth - 1:.1%}     paper 22.5%")
```

```text
own factors 15.8489 1.3663
Year Three ultimate 5,731.5   paper 5,732
true ultimate       4,677.5   paper 4,678
overstatement       22.5%     paper 22.5%
```

The cure is to trend the severity, not the pattern. Cut the layer from the
claim sizes of each year, at the price level of that year. The severity is the
ultimate claims of each year: 450, 1,687.5, 1,125 and 2,250 for Year One,
times the price level of the year. With this severity, `ExcessPattern.from_ground_up` solves the
shifts from the ground-up triangle and returns the layer's factors. Here the
ultimate severity is a `Grid` with four equal point masses. A step of 112.5
hits all four claims exactly, so the layer expectations are exact:

```python
from prospicio.distributions import Grid

def ultimate_claims(price):
    probs = [0.0] * 21
    for k in (4, 15, 10, 20):  # 450, 1,687.5, 1,125 and 2,250 at Year One's prices
        probs[k] = 0.25
    return Grid(112.5 * price, probs)

paper = {1: (17.20, 1.37, 23.50), 2: (14.77, 1.36, 20.06), 3: (13.08, 1.35, 17.67)}
patterns = {}
for year, price in PRICE.items():
    one_year = Triangle.from_long(
        [year] * 3, [12, 24, 36],
        {"ground_up": [sum(CLAIMS) * price * 4.5 / c for c in TO_ULTIMATE.values()]})
    p = ExcessPattern.from_ground_up(one_year, "ground_up", ExcessLayer(DEDUCTIBLE),
                                     ScaleShift(ultimate_claims(price)))
    patterns[year] = p
    print(f"Year {year}: 12-24 {p.layer_ldf[0]:.4f}  24-36 {p.layer_ldf[1]:.4f}  "
          f"12-ult {p.layer_cdf[0]:.4f}   paper {paper[year][0]:.2f} {paper[year][1]:.2f} "
          f"{paper[year][2]:.2f}")
print("shifts", " ".join(f"{c:g}" for c in patterns[1].shifts), "  ground-up factors",
      " ".join(f"{f:g}" for f in patterns[1].ground_up_ldf))
```

```text
Year 1: 12-24 17.2000  24-36 1.3663  12-ult 23.5000   paper 17.20 1.37 23.50
Year 2: 12-24 14.7680  24-36 1.3583  12-ult 20.0600   paper 14.77 1.36 20.06
Year 3: 12-24 13.0797  24-36 1.3511  12-ult 17.6719   paper 13.08 1.35 17.67
shifts 4.5 1.25 1   ground-up factors 3.6 1.25
```

Every printed factor agrees to the paper's two decimals. The unrounded values
are exact (the 17.2 of Year One is 3,010 / 175). `shifts` holds the `c_j`
from which the pattern is derived: one for each age, and the last for the tail
(1 here, as there is no tail). They are the ground-up factors to ultimate.
With no claim count column, the counts are taken as complete. With a
`counts=` column, the count development is divided out first.

The pattern goes into any development method through the `ldf`, `ldf_kind`
and `ldf_weight` arguments. `p.selected()` gives them as a dictionary to unpack with `**`, and
`p.tail` is the layer's tail. One pattern serves every origin, but it is
accurate only for the origins at the price level of the severity. Thus trend the
severity to the price level of the origins before you cut the layer.

With the pattern of Year Three on the full triangle, the ultimate of Year
Three is exact. The ultimate of Year Two is a little low, because its claims
are at the prices of the previous year:

```python
p3 = patterns[3]
fit = ChainLadder(tail=p3.tail, **p3.selected()).fit(bp, "excess")
for year, ult in zip(PRICE, fit.ultimate):
    true = sum(max(x * PRICE[year] * 4.5 - DEDUCTIBLE, 0.0) for x in CLAIMS)
    print(f"Year {year}: ultimate {ult:8.1f}   true {true:8.1f}   error {ult / true - 1:+.2%}")
```

```text
Year 1: ultimate   4112.5   true   4112.5   error +0.00%
Year 2: ultimate   4364.7   true   4388.1   error -0.53%
Year 3: ultimate   4677.5   true   4677.5   error +0.00%
```

Put origins at different price levels, or under different retentions, into
groups that share a layer and a price level. Fit each group with its own
pattern.

**The excess loss factor route.** Section 4 of Bloom and Patrik gets the same
answer without a severity curve. The triangle limited to 350 is dense and
develops reliably. An excess loss factor (ELF) is the share of ultimate
ground-up loss above 350. It converts limited and unlimited factors into an
excess factor, `excess CDF = ELF / (1 / CDF_unlimited - (1 - ELF) / CDF_limited)`.
This formula is Pinto and Gogol's layer identity in a different form. The
block below applies it to Year Three, with the limited factor from the chain
ladder of the triangle:

```python
limited = ChainLadder().fit(bp, "limited")
cdf_limited = limited.ldf[0] * limited.ldf[1]
limited_ult = rows[-1][4] * cdf_limited   # Year Three at 12 months
ground_up_ult = rows[-1][2] * 4.5
elf = 1.0 - limited_ult / ground_up_ult
print(f"limited 12-24 {limited.ldf[0]:.4f} (paper 1.32)  ELF {elf:.4f} (paper .764)")
print(f"excess CDF {elf / (1 / 4.5 - (1 - elf) / cdf_limited):.4f}   paper 17.54")
print(f"from the paper's rounded inputs {0.764 / (1 / 4.5 - (1 - 0.764) / 1.32):.4f}")
print(f"the scale shift's exact factor  {p3.layer_cdf[0]:.4f}")
```

```text
limited 12-24 1.3223 (paper 1.32)  ELF 0.7637 (paper .764)
excess CDF 17.5364   paper 17.54
from the paper's rounded inputs 17.5898
the scale shift's exact factor  17.6719
```

The paper's 17.54 comes from its unrounded inputs. Its rounded inputs give
17.59. The paper's 17.54 is 0.77% below the exact 17.67, which is the error
that the paper prints. The cause is that the limited factor, like the excess
factor above, is estimated on older years.

## 3. Shape drift: Pinto and Gogol, and Bear

A scale shift moves every claim by the same factor. Pinto and Gogol found in
ISO data that the tail of the size distribution also becomes heavier as
claims mature. A single-parameter Pareto fitted above a threshold `k` has a
shape `q` that decreases with maturity. Their factor for the losses excess of
`c k` is the development of the losses excess of `k` times `c^(q_i - q_j)`.
With the counts above `k` as the base, it also has a ratio of `(q - 1)` terms.

`ParetoShape` takes the threshold, a shape at every age and at ultimate,
and the base that the column holds. The base is `base="counts"` or an
`ExcessLayer` whose losses the column holds. The column then holds the base
development, not the ground-up losses.

```python
from prospicio.reserving import ParetoShape

def shape_factor(base_development, threshold, shapes, layer, base):
    tri = Triangle.from_long([2020, 2020], [12, 24], {"base": [1.0, base_development]})
    model = ParetoShape(threshold, {12: shapes[0], 24: shapes[1]}, shapes[1], base=base)
    return ExcessPattern.from_ground_up(tri, "base", layer, model).layer_ldf[0]

# Pinto and Gogol p. 247: losses excess of 25,000 develop by 2.5, q from 1.75 to 1.25.
pg = shape_factor(2.5, 25_000.0, (1.75, 1.25), ExcessLayer(100_000.0), ExcessLayer(25_000.0))
print(f"Pinto and Gogol p. 247, excess of 100,000: {pg:.4f}   paper 2.5 (4)^0.5 = 5.0")
# The same from the counts above 25,000: they develop by 2.5 (0.25 / 0.75).
pgc = shape_factor(2.5 * 0.25 / 0.75, 25_000.0, (1.75, 1.25), ExcessLayer(100_000.0), "counts")
print(f"the same from the counts above 25,000:     {pgc:.4f}")
# Bear p. 138: the layer 100,000 to 300,000 develops by 3.5, q from 1.25 to 1.10.
bear = shape_factor(3.5, 100_000.0, (1.25, 1.10), ExcessLayer(200_000.0, 400_000.0),
                    ExcessLayer(100_000.0, 200_000.0))
print(f"Bear p. 138, layer 200,000 to 600,000:     {bear:.4f}   paper 3.88")
```

```text
Pinto and Gogol p. 247, excess of 100,000: 5.0000   paper 2.5 (4)^0.5 = 5.0
the same from the counts above 25,000:     5.0000
Bear p. 138, layer 200,000 to 600,000:     3.8835   paper 3.88
```

Pinto and Gogol's Appendix B calculates the shape drift from 27 months to
ultimate above 10,000 (`q` from 1.6 to 1.3). For several retentions and
layers, it prints the losses as a multiple of the losses excess of 10,000 at
the two ages, to three decimals. Their ratio is the layer's factor relative to
the base development `a`. Here the second age is ultimate:

```python
appendix_b = {  # (from, to) in thousands: (at 27 months, at ultimate), as printed
    (100, None): (0.251, 0.501), (300, None): (0.130, 0.360), (500, None): (0.096, 0.309),
    (1000, None): (0.063, 0.251), (100, 300): (0.121, 0.141), (300, 500): (0.034, 0.051),
    (500, 1000): (0.033, 0.058),
}
for (lo, hi), (at27, ult) in appendix_b.items():
    layer = ExcessLayer(lo * 1e3) if hi is None else ExcessLayer(lo * 1e3, (hi - lo) * 1e3)
    f = shape_factor(1.0, 10_000.0, (1.6, 1.3), layer, ExcessLayer(10_000.0))
    name = f"excess of {lo:,}k" if hi is None else f"{lo:,}k to {hi:,}k"
    print(f"{name:16s} {f:.4f} a   paper {ult:.3f} / {at27:.3f} = {ult / at27:.3f} a")
```

```text
excess of 100k   1.9953 a   paper 0.501 / 0.251 = 1.996 a
excess of 300k   2.7742 a   paper 0.360 / 0.130 = 2.769 a
excess of 500k   3.2336 a   paper 0.309 / 0.096 = 3.219 a
excess of 1,000k 3.9811 a   paper 0.251 / 0.063 = 3.984 a
100k to 300k     1.1606 a   paper 0.141 / 0.121 = 1.165 a
300k to 500k     1.4932 a   paper 0.051 / 0.034 = 1.500 a
500k to 1,000k   1.7843 a   paper 0.058 / 0.033 = 1.758 a
```

The factor for the losses excess of 100,000 is exactly `10^0.3 = 1.99526`
times `a`. The paper's 1.996 is the ratio of its rounded amounts. Every other
row agrees within the rounding of the printed amounts. This rounding is wide
for the small amounts: 0.033 can be any value from 0.0325 to 0.0335.

Shape drift needs a shape at every age, and thus claim-level data or layered
losses by maturity. A scale shift needs only the ground-up triangle. When the
tail becomes heavier, the two methods give different results on high layers.

A simulation study (`knowledge/findings/excess-simulation-study.md`, 500 books
per arm) measured this. When the shapes of claims drift, the scale shift
understates the ultimate of the sparse layer by about 26%. It understates the
thin layer by 14%. `ParetoShape` with the true shapes is unbiased.

## 4. A sparse layer on a simulated book

Each example above has one claim history. A real layer triangle has many
origins, and most of them have no claim in the layer for years. To compare the
methods, we simulate a book with a known truth.

The book has ten accident years, 2015 to 2024, of 120 claims each, valued at
the end of 2024 at 12 to 120 months. The ultimate size of each claim is
lognormal (mean about 30,000, log-sd 1.8). At each age, the reported value of
a claim is its ultimate divided by a shift. The shift decreases from 3.0 at 12
months to 1.01 at 120 months, and the last 1% is the tail.

The book follows the scale shift by construction. Thus it shows the mechanics,
but it does not test the model. The simulation study above is the test.

The uniform draws come from a small linear congruential generator that the
block writes out. Thus the R twin of this guide draws the same claims. A seed selects the
book. The block tracks two layers: a high layer, 750,000 xs 250,000, and a
working layer, 400,000 xs 100,000. It also counts the claims in each layer.

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
print("origin  ground-up  high layer  claims  true high ult")
for k, y in enumerate(YEARS):
    print(f"{y}  {latest['incurred'][k]:10,.0f} {latest['high'][k]:10,.0f}"
          f"  {latest['high_n'][k]:6.0f}  {truth['high'][k]:12,.0f}")
print(f"total   {sum(latest['incurred']):10,.0f} {sum(latest['high']):10,.0f}"
      f"  {sum(latest['high_n']):6.0f}  {sum(truth['high']):12,.0f}")
```

```text
origin  ground-up  high layer  claims  true high ult
2015   2,297,915          0       0             0
2016   3,338,009    534,042       3       553,303
2017   2,509,633    255,071       2       270,172
2018   2,473,626    210,437       1       224,250
2019   3,066,471    565,998       2       619,298
2020   2,652,024    173,574       1       215,931
2021   2,499,806    101,581       1       206,309
2022   3,187,793    614,715       4     1,260,601
2023   2,103,713     17,007       1       494,749
2024     965,657          0       0       161,796
total   25,094,646  2,472,425      15     4,006,410
```

The high layer has zero to four claims a year, and 2015 never had a claim in
it. The own chain ladder of the high layer fails. At the oldest link, the only
origin, 2015, is zero, so there is no factor to estimate. The own chain ladder
of the working layer runs, but its factors come from a few claims:

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

The derived patterns need only the ground-up column and the ultimate severity.
They are incurred (reported) patterns, and the ground-up factors are
volume-weighted averages. Here the severity is the true one. In practice, fit
the severity to trended ultimate claims, for example with the fitting functions of `prospicio.distributions`. The
ground-up tail of 1.01 gives each layer its own, larger tail:

```python
derived = {}
for name, layer in (("high", HIGH), ("working", WORKING)):
    derived[name] = ExcessPattern.from_ground_up(book, "incurred", layer, ScaleShift(SEV),
                                                 tail=TAIL)
p = derived["high"]
print("shifts           ", " ".join(f"{c:.3f}" for c in p.shifts))
print("ground-up factors", " ".join(f"{f:.3f}" for f in p.ground_up_ldf),
      f"tail {p.ground_up_tail:.2f}")
for name, q in derived.items():
    print(f"{name:7s} factors  ", " ".join(f"{f:.3f}" for f in q.layer_ldf), f"tail {q.tail:.4f}")
print("to ultimate at 12 months: ground-up", f"{SHIFT[0]:.2f},",
      ", ".join(f"{name} {q.layer_cdf[0]:.2f}" for name, q in derived.items()))
```

```text
shifts            3.000 1.800 1.400 1.200 1.100 1.050 1.030 1.020 1.015 1.010
ground-up factors 1.667 1.286 1.167 1.091 1.048 1.019 1.010 1.005 1.005 tail 1.01
high    factors   2.446 1.513 1.278 1.145 1.074 1.030 1.015 1.008 1.008 tail 1.0153
working factors   2.180 1.431 1.236 1.124 1.064 1.026 1.013 1.006 1.006 tail 1.0131
to ultimate at 12 months: ground-up 3.00, high 6.27, working 4.91
```

The shifts are the book's own shifts (the ground-up factors to ultimate, tail
included), recovered from the triangle. At 12 months, the high layer has more
than twice as far to develop as the ground-up losses.

With the derived pattern, every method runs on the high layer. The block below
fits the chain ladder, Mack, and the Bornhuetter-Ferguson with a layer
exposure. Each method takes the pattern through `**p.selected()` and `tail=p.tail`. The
BF prior is the ground-up expected loss ratio applied to the premium times the
layer share. The layer share, `layer_share(SEV, HIGH)`, is the share of the mean
severity in the layer (section 6 gives more detail):

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
print("origin  chain ladder        BF      true")
for k, y in enumerate(YEARS):
    print(f"{y}  {cl.ultimate[k]:12,.0f} {bf.ultimate[k]:9,.0f} {truth['high'][k]:9,.0f}")
print(f"total {sum(cl.ultimate):14,.0f} {sum(bf.ultimate):9,.0f} {sum(truth['high']):9,.0f}")
print(f"Mack reserve {mack.total_reserve:,.0f}, standard error {mack.total_standard_error:,.0f}")
```

```text
ground-up ELR 0.6618, layer share 0.1603, expected layer ultimate a year 583,548
origin  chain ladder        BF      true
2015             0     8,785         0
2016       546,307   547,143   553,303
2017       262,895   272,438   270,172
2018       220,153   236,191   224,250
2019       609,855   607,963   619,298
2020       200,936   253,038   215,931
2021       134,694   245,040   206,309
2022     1,041,936   853,985 1,260,601
2023        43,614   373,000   494,749
2024             0   490,530   161,796
total      3,060,390 3,888,113 4,006,410
Mack reserve 587,966, standard error 210,007
```

Even with the correct pattern, the chain ladder does not see unreported
losses. The layer of 2024 is still empty at 12 months, so its chain ladder
ultimate is zero. The chain ladder develops the single small claim of 2023 to
43,614, against a true 494,749.

The BF puts the expected layer loss on the unreported part. For this reason,
it is the usual method for an immature layer. Here the BF total is nearer to
the truth. The years 2016 to 2019, whose layer losses are nearly all reported,
have almost the same results under both methods. This is one simulated book.
Section 5 compares the methods over a hundred books.

Mack's standard error is large relative to its reserve, as expected. The
derived pattern has no parameter error (it is selected), and the process error
of a few claims is large.

## 5. Blending with the layer's own development

When a layer has some data of its own, you can blend the derived pattern with
the layer's own factors. The blend is `f = w f_derived + (1 - w) f_own`, where
`w` is the weight on the derived pattern at each age. `credibility_weights` sets
`w = 1 - n / (n + k)`. Here `n` is the number of layer claims at the earlier
age (with a `counts=` column), or else the number of origins with a positive
earlier value. `k` is a credibility constant, and it is a required argument.

The simulation study's candidate is `k = 20`, which this guide uses. A cap also
limits the credibility. If you remove any one origin, the blended factor moves
by at most `max_influence` of it (0.1 here). Thus one very large link ratio
cannot dominate. Where the layer's own factor cannot be estimated, the weight
is exactly 1:

```python
for name in ("high", "working"):
    w = derived[name].credibility_weights(book, name, 20.0, 0.1, counts=name + "_n")
    print(f"{name:7s} weight on derived", " ".join(f"{w[a]:.3f}" for a in AGES[:-1]))
```

```text
high    weight on derived 0.909 0.741 0.714 0.714 0.714 0.769 0.800 0.870 1.000
working weight on derived 0.556 0.455 0.426 0.426 0.444 0.476 0.513 0.606 0.741
```

The few claims of the high layer leave most of the weight on the derived
pattern. The oldest age gets exactly 1, because the zero of 2015 gives no own
factor there. The working layer has more claims and thus more credibility. The
weights go directly into `selected`: `ChainLadder(tail=p.tail, **p.selected(w))`.

One book cannot rank the methods, because its few large claims decide the
result. Thus we simulate a hundred books (seeds 1 to 100, the same model). For
each layer, we compare the error of the total ultimate under four methods:

* the layer's own chain ladder
* the derived pattern
* the blend with `k = 20`
* the BF with the derived pattern.

The measures are the mean error and the root mean square error (RMSE), both
relative to the mean true ultimate:

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

print("layer    method             runs  mean error   RMSE")
for (name, label), v in errors.items():
    ok = [e for e, _ in v if e is not None]
    mean_truth = sum(t for _, t in v) / len(v)
    rmse = math.sqrt(sum(e * e for e in ok) / len(ok))
    print(f"{name:8s} {label:17s} {len(ok):5d} {sum(ok) / len(ok) / mean_truth:+10.1%}"
          f" {rmse / mean_truth:7.1%}")
```

```text
layer    method             runs  mean error   RMSE
high     own chain ladder     93     +19.2%   95.1%
high     derived pattern     100      -3.9%   15.5%
high     blend, k = 20       100      -1.9%   19.4%
high     BF, derived         100      +0.5%    9.5%
working  own chain ladder    100      -0.3%   13.1%
working  derived pattern     100      -2.0%    8.1%
working  blend, k = 20       100      -1.2%    9.9%
working  BF, derived         100      -0.6%    4.4%
```

The own chain ladder fails on seven of the hundred high-layer books. Where it
runs, it overstates by a fifth, with an RMSE near the full ultimate. The
cause is that the early link ratios of a few claims are very large. The derived pattern
decreases the RMSE to a fraction of that on both layers. It is a little low on
average, because an origin with no loss yet in the layer stays at zero.

The BF with the layer-share prior is the best on both layers, because it does
not multiply the chance results of the immature years. But this test favours
it: it uses the true loss ratio and the true severity, and a real prior only
approximates them. The blend is between the derived pattern and the own
factors.

Here the model holds exactly, so the derived pattern needs no help from the
data. The reason to blend is that the severity assumption may be wrong. In the
simulation study, under shape drift, a blend near `k = 20` was better than
both the derived pattern and the own factors on the thin layer. Without the
`max_influence` cap, a small `k` gave very large errors on the sparse layer.

## 6. A Bornhuetter-Ferguson prior from the layer share

The a-priori layer ultimate of origin `i` is its exposure times a ground-up
expected loss ratio times the layer share `s`. The layer share is the expected
share of ground-up loss in the layer. For a severity,
`s = (LEV(b) - LEV(a)) / mean` on the ultimate basis, which is `layer_share`. For an exposure curve, use
the curve's own `layer_share(limit, attachment, mpl)` (`Mbbefd` and
`TabulatedCurve`). `layer_exposure` writes a copy of the triangle with the exposure column
times the share (one share, or one share for each origin). Each expected-loss
method accepts this copy as its exposure.

Section 4 used this copy with the BF. Cape Cod instead estimates the layer's
loss ratio from the layer's own used-up exposure, through the same derived
pattern:

```python
from prospicio.reserving import CapeCod

cc = CapeCod(tail=p.tail, **p.selected()).fit(layered, "high", "high_premium")
print(f"loss ratio to the layer premium: a priori {elr:.4f} (the ground-up ELR), "
      f"Cape Cod {cc.apriori[0]:.4f}")
print(f"total ultimate: BF {sum(bf.ultimate):,.0f}, Cape Cod {sum(cc.ultimate):,.0f}, "
      f"true {sum(truth['high']):,.0f}")
true_ratio = sum(truth["high"]) / (len(YEARS) * PREMIUM * share)
print(f"this book's true loss ratio to the layer premium {true_ratio:.4f}")
```

```text
loss ratio to the layer premium: a priori 0.6618 (the ground-up ELR), Cape Cod 0.3702
total ultimate: BF 3,888,113, Cape Cod 3,264,359, true 4,006,410
this book's true loss ratio to the layer premium 0.4544
```

This book is light: its layer losses are about two thirds of what the prior
expects. The Cape Cod follows the book's own experience down, and here it goes
below the true ratio. The BF keeps the weight of the prior on the immature
years.

The share is an estimate, and its error is worth including in the distribution
of the reserve. `LayerSharePrior` does this through the normalised Bayesian BF
(`BayesBfNormalised`). The layer's expected ultimate gets a gamma prior with mean
`ELR x share x exposure`. Its coefficient of variation combines the two
uncertainties as for a product of independent variables:
`cv^2 = (1 + cv_elr^2)(1 + cv_share^2) - 1`.

The model estimates its development pattern from the layer's own triangle,
under its vague prior. Thus the derived excess pattern does not enter, and the
model needs an own factor at every age. The high layer's triangle has no own
factor at its oldest age, so this part of the guide uses the working layer:

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

The posterior is a Monte Carlo result, and the last digits of a NUTS run
differ between platforms (see [`reserving-meyers.md`](reserving-meyers.md)). Thus this guide prints
only the prior, which is exact. In your own run, `fit.reserves` is the predictive
distribution of the reserve.

The effect of the share's cv on that distribution depends on the model of the
share's error. The layer has one share, so the error is the same for every
origin. By default, `LayerSharePrior` models it in this way
(`share_dependence="common"`, #518).
One gamma multiplier, with mean 1 and cv `share_cv`, applies to the prior of
every origin. The sampler draws it with the other parameters, so all the
origins together update the share.

The other form, `share_dependence="per_origin"`, draws an independent share error at each origin.
The prior of each origin is the same under both forms, but the prior of the
total is not. `prior_total` gives the prior of the layer's total expected ultimate in
closed form:

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

Here, independent errors understate the prior sd of the total by a factor of
three. The posterior moves in the other direction. When the ten origins
identify the share, each origin stays near its prior with the loss ratio's
own cv. Thus the common share gives the narrower fitted interval.

A validation layer (a quarter of GenIns's paid triangle, in
`knowledge/findings/excess-bayes-prior.md`) shows the size of the effect. Each
figure is the mean of the total reserve sd over seeds 1 to 20 for the common
share. For independent errors, it is the mean over 10 seeds. Under the common share, the
posterior sd of the total reserve was about 497,000 at a share cv of 0.3. It
increased only about 2% as the cv went to 1.5. Under independent errors, it
increased from 514,000 to 730,000 (about 42%).

The sd of one fit moves by about 1.5% from seed to seed, so compare such runs
over several seeds. Where the data cannot identify the share (few origins, or
all origins immature), the total of the common model stays wider.

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
