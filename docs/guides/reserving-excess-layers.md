# Excess-layer loss development: layer factors from the ground-up triangle

A reinsurer writing an excess layer, or a primary insurer reserving a high
deductible or an excess policy, has to develop losses that sit above an
attachment. Those losses develop much more slowly than the ground-up losses
they are cut from, and the layer's own triangle is sparse: most early cells are
zero, and the first non-zero link ratios are enormous. The chain ladder on such
a triangle is unstable or fails outright. The usual fix in the literature is to
derive the layer's development from the dense ground-up triangle and a claim
size distribution, then blend it with whatever the layer's own data say.

prospicio builds that in `prospicio.reserving` (issue #192, design note
`docs/design/reserving-excess.md`): `ExcessPattern.from_ground_up` derives a
layer pattern under a scale shift or Pinto and Gogol's Pareto shape drift,
`credibility_weights` blends it with the layer's own factors, and
`layer_share`, `layer_exposure` and `LayerSharePrior` give the
Bornhuetter-Ferguson prior of a layer. This guide shows each piece on the
literature's printed examples (Bloom and Patrik, Pinto and Gogol, Bear), then
on a simulated book with a sparse layer, the source's value printed next to
prospicio's wherever a source prints one.

The papers are their authors'; what follows explains them in new words and
recomputes their numbers. Section 2's example is from Bloom and Patrik, *A
Methodology for Avoiding the Pitfalls of Excess Loss Development* (CAS
E-Forum, Fall 2013), pages 4 to 8; section 3's are from Pinto and Gogol, *An
Analysis of Excess Loss Development* (PCAS 74, 1987) and Bear's discussion of
it (PCAS 79, 1992). What each source prints is summarised in
`knowledge/references/excess-development-literature.md`. For the choice
between reserving methods in general, read
[`reserving-methods.md`](reserving-methods.md); this guide does not repeat it.

**How the numbers here are checked.** Every Python block below that is
followed by a text block runs in CI (`python/tests/test_guides.py`) and must
print exactly that text. The whole guide runs in about a second.

## 1. Why a layer develops slowly

Suppose every claim grows by the same factor between two ages, as case
reserves are raised. Claims below the attachment contribute nothing to the
layer at the earlier age, and some of them cross it later, so the layer gains
both from the growth of claims already in it and from claims entering it. The
design note's illustration (its own calculation, not a source's) makes this
concrete: a lognormal severity with log-sd 1.5 whose mean is 65,000 at ultimate,
every claim 1.3 times its earlier value, counts fixed. The ground-up factor is
1.3; the layer factors are larger the higher the layer.

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

A ground-up factor of 1.3 puts a factor near 2 on a high layer. The layer's
own triangle cannot show that until late, because at the early ages it holds
almost nothing to develop. The rest of this guide is about getting the layer's
factors from somewhere sturdier.

## 2. Bloom and Patrik's example: the pitfall and the cure

Bloom and Patrik's example is small enough to follow by hand. Year One has
four claims reported at 100, 375, 250 and 500 at 12 months. Ground-up losses
develop by 3.60 from 12 to 24 months and by 1.25 from 24 to 36, the end of
development, with claim counts fixed, so every claim reaches 4.5 times its
12-month value. Years Two and Three have the same claims with 5% inflation a
year. The layer is everything excess of a deductible of 350.

Because every claim is scaled by the same factor, this is exactly a **scale
shift**: the claims at age `j` are the ultimate claims divided by a shift
`c_j`, with `c_j` the ground-up factor to ultimate (4.5 at 12 months, 1.25 at
24, 1 at 36). We build the triangle of the paper's three years, valued at the
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

The paper's excess triangle, to the dollar. The trap is to develop it on its
own. The 12 to 24 month factor comes from Years One and Two, and inflation has
pushed more of Year Two's claims over the deductible early, so a factor
estimated on older years overstates a newer year's development:

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

The cure is to leave the pattern alone and trend the severity: cut the layer
from each year's own claim sizes, at that year's price level. With the
ultimate claims of each year (450, 1,687.5, 1,125 and 2,250 for Year One,
times the year's price level) as the severity, `ExcessPattern.from_ground_up`
solves the shifts from the ground-up triangle and returns the layer's factors.
The ultimate severity here is a `Grid` with four equal point masses (a step of
112.5 hits all four claims exactly), so the layer expectations are exact:

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

Every printed factor, to the paper's two decimals; the unrounded values are
exact (Year One's 17.2 is 3,010 / 175). `shifts` holds the `c_j` the
pattern was derived with, one per age, the last for the tail (none here, so
1). They are the ground-up factors to ultimate: with no claim count column,
the counts are taken as complete, and with a `counts=` column the count
development is divided out first.

The pattern enters any development method through the existing `ldf`,
`ldf_kind` and `ldf_weight` arguments: `p.selected()` gives them as a
dictionary to splat, and `p.tail` is the layer's tail. One pattern serves
every origin, so it is right for the origins at the severity's price level
(decision 5 of the note: trend the severity to the origins' level before the
layer is cut). With Year Three's pattern on the whole triangle, Year Three's
ultimate is exact and Year Two's is a little low, since its claims are at the
previous year's prices:

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

Origins written at different price levels, or under different retentions,
are best split into groups that share a layer and a price level, each fitted
with its own pattern.

**The excess loss factor route.** Bloom and Patrik's section 4 gets the same
answer without a severity curve: the limited-to-350 triangle is dense and
develops reliably, and an excess loss factor (ELF, the share of ultimate
ground-up loss above 350) converts limited and unlimited factors into an
excess factor, `excess CDF = ELF / (1 / CDF_unlimited - (1 - ELF) / CDF_limited)`
(Pinto and Gogol's layer identity, restated). On Year Three, with the limited
factor from the triangle's own chain ladder:

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

The paper's 17.54 comes from its unrounded inputs; its rounded ones give 17.59.
Its 17.54 is 0.77% below the exact 17.67, the error the paper prints, because
the limited factor, like the excess one above, is estimated on older years.

## 3. Shape drift: Pinto and Gogol, and Bear

A scale shift moves every claim by the same factor. Pinto and Gogol found in
ISO data that the tail of the size distribution also gets heavier as claims
mature: a single-parameter Pareto fitted above a threshold `k` has a shape
`q` that falls with maturity. Their factor for the losses excess of `c k` is
the development of the losses excess of `k` times `c^(q_i - q_j)` (with the
counts above `k` as the base, times a ratio of `(q - 1)` terms as well).
`ParetoShape` takes the threshold, a shape at every age and at ultimate, and
the base the column holds: `base="counts"` or an `ExcessLayer` whose losses
the column is. The column then holds the base development, not the ground-up
losses.

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

Pinto and Gogol's Appendix B works the shape drift from 27 months to ultimate
above 10,000 (`q` from 1.6 to 1.3) and prints, for several retentions and
layers, the losses as a multiple of the losses excess of 10,000 at both ages,
to three decimals. Their ratio is the layer's factor relative to the base
development `a`. Here the second age stands for ultimate:

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

The excess of 100,000 is exactly `10^0.3 = 1.99526` times `a`; the paper's
1.996 is the ratio of its rounded amounts. Every other row agrees within the
rounding of the printed amounts, which is wide for the small ones (0.033 could
be anything from 0.0325 to 0.0335).

Shape drift needs a shape at every age, which needs claim-level data or
layered losses by maturity; a scale shift needs only the ground-up triangle.
The two disagree on high layers whenever the tail does thicken: the Reserving
lane's simulation study (`knowledge/findings/excess-simulation-study.md`, 500
books per arm) found that when claims' shapes drift, the scale shift
understates the sparse layer's ultimate by about 26% and the thin layer's by
14%, while `ParetoShape` with the true shapes is unbiased.

## 4. A sparse layer on a simulated book

The examples above have one claim history each. A real layer triangle has
many origins, most of them with no claim in the layer for years. To see the
methods side by side we simulate a book whose truth is known: ten accident
years, 2015 to 2024, of 120 claims each, valued at the end of 2024 at 12 to
120 months. Each claim's ultimate size is lognormal (mean about 30,000, log-sd
1.8), and at each age it is reported at its ultimate divided by a shift that
falls from 3.0 at 12 months to 1.01 at 120 months, the last 1% being the tail.
The book follows the scale shift by construction, so it shows the mechanics,
not a test of the model; the simulation study above is the test.

The uniform draws come from a tiny linear congruential generator written out
here, so that the R twin of this guide draws the same claims, and a seed
picks the book. Two layers are
tracked: a high one, 750,000 xs 250,000, and a working one, 400,000 xs
100,000, each with the count of claims in it.

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

The high layer has between none and four claims a year, and 2015 never had
one. Its own chain ladder fails: at the oldest link the only origin, 2015, is
zero, so there is no factor to estimate. The working layer's own chain ladder
runs, but its factors rest on a handful of claims:

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

The derived patterns need only the ground-up column and the ultimate
severity, here the true one (in practice a severity fitted to trended
ultimate claims; section 7). The ground-up tail of 1.01 gives each layer
its own, larger tail:

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

The shifts are the book's own (the ground-up factors to ultimate, tail
included), recovered from the triangle. At 12 months the high layer has more
than twice as far to go as the ground-up losses.

On the high layer the derived pattern makes every method run. Here are the
chain ladder, Mack, and the Bornhuetter-Ferguson with a layer exposure, each
taking the pattern through `**p.selected()` and `tail=p.tail`. The BF prior
is the ground-up expected loss ratio applied to the premium times the layer
share, `layer_share(SEV, HIGH)`, the share of the mean severity that falls in
the layer (section 6 says more):

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

Even with the right pattern, the chain ladder cannot see what has not been
reported: 2024's layer is still empty at 12 months, so its chain ladder
ultimate is zero, and 2023's single small claim is multiplied up to 43,614
against a true 494,749. The BF
puts the expected layer loss on the unreported part instead, which is why it
is the usual method for an immature layer. Here it lands closer to the truth
in total, while 2016 to 2019, whose layer losses are nearly all reported,
come out much the same under both. This is one simulated book; section 5
compares the methods over a hundred. Mack's standard error
is large next to its reserve, as it should be: the derived pattern carries no
parameter error (it is selected), and the process error of a few claims is
big.

## 5. Blending with the layer's own development

When a layer has some data of its own, the derived pattern can be blended
with the layer's own factors, `f = w f_derived + (1 - w) f_own`, with `w` the
weight on the derived pattern at each age. `credibility_weights` sets
`w = 1 - n / (n + k)`, with `n` the layer's claims at the earlier age (with a
`counts=` column) or else its origins with a positive earlier value, and `k`
a credibility constant. `k` has no default (decision 19 of the design note);
the simulation study's candidate is `k = 20`, which is what we use. The
credibility is also capped so that removing any one origin moves the blended
factor by at most `max_influence` of it (0.1 here), so a single huge link
ratio cannot dominate, and where the layer's own factor cannot be estimated
the weight is exactly 1:

```python
for name in ("high", "working"):
    w = derived[name].credibility_weights(book, name, 20.0, 0.1, counts=name + "_n")
    print(f"{name:7s} weight on derived", " ".join(f"{w[a]:.3f}" for a in AGES[:-1]))
```

```text
high    weight on derived 0.909 0.741 0.714 0.714 0.714 0.769 0.800 0.870 1.000
working weight on derived 0.556 0.455 0.426 0.426 0.444 0.476 0.513 0.606 0.741
```

The high layer's few claims leave most of the weight on the derived pattern,
and the oldest age gets exactly 1, since 2015's zero leaves no own factor
there. The working layer has more claims and so more credibility. The
weights go straight into `selected`, `ChainLadder(tail=p.tail, **p.selected(w))`.

One book cannot rank methods: its few large claims decide the outcome. So we
simulate a hundred books (seeds 1 to 100, the same model) and compare, for
each layer, the error of the total ultimate under the layer's own chain
ladder, the derived pattern, the blend with `k = 20` and the BF with the
derived pattern, as the mean error and the root mean square error, both
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

The own chain ladder fails on seven of the hundred high-layer books, and where
it runs it overstates by a fifth with an RMSE near the whole ultimate: the
early link ratios of a few claims are huge. The derived pattern cuts the RMSE
to a fraction of that on both layers; it runs a little low on average because
an origin with nothing yet in the layer stays at zero. The BF with the
layer-share prior is best on both, since it does not multiply the immature
years' luck; but it is flattered here, being given the true loss ratio and the
true severity, which a real prior only approximates. The blend sits between
the derived pattern and the own factors. Here the model holds exactly, so the
derived pattern needs no help from the data; the reason to blend anyway is
that the severity assumption may be wrong. In the Reserving lane's study,
under shape drift a blend near `k = 20` beat both the derived pattern and the
own factors on its thin layer, and without the `max_influence` cap a small `k`
was ruinous on its sparse layer.

## 6. A Bornhuetter-Ferguson prior from the layer share

The a-priori layer ultimate of origin `i` is its exposure times a ground-up
expected loss ratio times the layer share `s`, the expected share of ground-up
loss that lands in the layer. For a severity, `s = (LEV(b) - LEV(a)) / mean`
on the ultimate basis, which is `layer_share`; for an exposure curve, the
curve's own `layer_share(limit, attachment, mpl)` (`Mbbefd` and
`TabulatedCurve`). `layer_exposure` writes a copy of the triangle with the
exposure column times the share (one share, or one per origin), which any of
the expected-loss methods takes as its exposure. Section 4 used it with the
BF; Cape Cod estimates the layer's loss ratio from the layer's own used-up
exposure instead, through the same derived pattern:

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

This book is light (its layer losses are about two thirds of what the prior
expects), and the Cape Cod follows its own experience down, here past the
true ratio, while the BF keeps the prior's weight on the immature years.

The share is an estimate, and its error is worth carrying into the reserve's
distribution. `LayerSharePrior` does that through the normalised Bayesian BF
(`BayesBfNormalised`): the layer's expected ultimate gets a gamma prior with
mean `ELR x share x exposure` and a coefficient of variation combining the
two uncertainties as for a product of independent variables,
`cv^2 = (1 + cv_elr^2)(1 + cv_share^2) - 1`. Its development pattern is
estimated from the layer's own triangle, under the model's vague prior, so
the derived excess pattern does not enter (decision 22 of the design note),
and the model needs every age's own factor to exist. The high layer's
triangle has none at its oldest age, so this part of the guide uses the
working layer:

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
differ between platforms (see [`reserving-meyers.md`](reserving-meyers.md)), so
this guide prints only the prior, which is exact, and leaves the posterior to
your own run: `fit.reserves` is the predictive distribution of the reserve.

What the share's cv does to that distribution depends on how the share's error
is modelled. The error is one and the same for every origin, since the layer
has one share, and `LayerSharePrior` models it so by default
(`share_dependence="common"`, issue #518): one gamma multiplier on every
origin's prior, with mean 1 and cv `share_cv`, sampled with the other
parameters, so that all the origins together update the share. The first form,
`share_dependence="per_origin"`, draws an independent share error at each
origin, which understates the prior uncertainty of the total. On the Reserving
lane's validation layer (a quarter of GenIns's paid triangle; means over 20
seeds in `knowledge/findings/excess-bayes-prior.md`) the total reserve's sd
rose only about 2% as the share's cv went from 0.3 to 1.5 under the common
share, since ten origins pin the share down, against about 42% (514,000 to
730,000) under independent errors.

## 7. What prospicio does not cover

- **A paid basis.** `from_ground_up` derives incurred (reported) patterns only;
  `basis="paid"` is an error. Pinto and Gogol and Bear get paid layer factors
  from incurred ones and ratios of paid-to-reported ratios; prospicio has no
  helper for that.
- **One pattern for all origins.** A severity trended differently by origin, or
  a retention that changes by year, needs a pattern per origin, which no
  development method takes. Group the origins that share a layer and a price
  level and fit each group with its own pattern (section 2 shows the error of
  one pattern across price levels).
- **Fitting the severity.** The scale shift takes the ultimate severity at the
  origins' price level as given; fitting it (from trended ultimate claims,
  say, with `prospicio.distributions`' fitting routines) is the caller's step,
  and so is the shape at each maturity for `ParetoShape`.
- **The ground-up average.** The ground-up factors are the volume-weighted
  ones, with no exclusions or other averages (decision 15 of the design note).
- **The derived pattern in the Bayesian BF.** `LayerSharePrior` estimates the
  layer's pattern from its own triangle, so it does not use the derived
  pattern and cannot fit a layer whose own factor is missing at some age
  (section 6). The share's cv is an input; no helper derives it from a
  severity's parameter uncertainty (decision 24).
- **Bühlmann-Straub credibility** of the layer's link ratios, and a default `k`.
  The weights are `1 - n / (n + k)` with a required `k`.
- **Claim-level development** (Boor, Gillam and Couret's excess loss factors):
  prospicio's per-claim IBNER is on the pricing side (`DevelopmentPattern`, see
  [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md)).
- **Sahasrabuddhe's and Siewert's examples** are not reproduced here: the first
  needs its example's trend indices, the second's Weibull parameters are
  rounded and both its scale and shape move.

## See also

- [`reserving-methods.md`](reserving-methods.md): which reserving method to
  use when.
- [`choosing-a-method.md`](choosing-a-method.md): the library's methods by
  question.
- `docs/design/reserving-excess.md`: the method, the API and the validation
  targets as designed, with the simulation study's results.
- `python/tests/test_reserving_excess.py` and
  `python/tests/test_reserving_bayes_excess_prior.py`: the bindings' tests.
