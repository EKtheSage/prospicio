# Pricing Insurance Risk: the case studies in prospicio

Stephen Mildenhall and John Major's book *Pricing Insurance Risk* (Wiley, 2022)
runs one method through a handful of small portfolios, its case studies: build
the distribution of a book's total from its units, choose a capital standard,
price the total with a spectral risk measure calibrated to a target return, and
split the price over the units by the natural allocation. The book's exhibits
were produced with Mildenhall's Python package `aggregate`, and version 0.30.1
still ships the code (`aggregate.extensions.case_studies`) and the case studies
as DecL programs in its test suite.

This guide runs the four case studies in prospicio and prints, next to each
number, what `aggregate` 0.30.1 returns for the same program. It does not
reproduce the book's text or tables: the comparisons are with `aggregate` run on
the settings of `case_studies.py` (a 10% return on capital, the capital standard
and grid of each case), and what each case study is for is read here from its
programs and results. Read the book for the theory; read
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md) first if the
words distortion, natural allocation or lifted capital are new.

## 1. The method in one function

Each case study has two independent units. Their distributions go on one grid,
`Portfolio.from_independent` combines them and keeps each unit's expected loss
given the total, and the assets are the total's quantile at the capital
standard `p`. The constant cost of capital (CCoC) at 10% sets the premium of the
total: `P = v E[min(X, a)] + d a` with `v = 1 / 1.1` and `d = 1 - v`, which earns
exactly 10% on the capital `a - P`. The four other distortions, proportional
hazard (PH), Wang, dual power and TVaR, are then calibrated so that they charge
the same premium for the total. They differ only in how they split it.

```python
import math

from prospicio.aggregate import fft
from prospicio.distributions import Count, Gamma, Grid, Lognormal, Poisson
from prospicio.pricing import Portfolio
from prospicio.reinsurance import Layer, Tower

FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
RETURN = 0.10


def discretize(severity, step, points):
    """A severity on a grid, its mass above the top spread over the grid as aggregate does."""
    probs = Grid.rounding(severity, step, 2 * points)[0].probs[:points]
    total = sum(probs)
    return Grid(step, [q / total for q in probs])


def calibrate(book, p):
    """The five distortions that price the book's total at a 10% return at standard p."""
    a = book.assets(p)
    capped = Portfolio(["total"], [[min(x, a)] for x in book.totals], book.probs)
    return {f: capped.calibrate(f, assets=a, return_on_capital=RETURN) for f in FAMILIES}


def show(book, p, distortions, aggregate_030):
    """Premium by unit under each distortion, aggregate 0.30.1's in brackets."""
    priced = {f: book.price(d, p=p) for f, d in distortions.items()}
    first = priced["ccoc"]
    losses = ", ".join(f"{u} {x.loss:.3f}" for u, x in zip(book.units, first.allocated))
    print(f"assets {book.assets(p):.3f}; loss within assets: {losses}, total {first.total.loss:.3f}")
    with_parameter = len(aggregate_030["ccoc"]) == len(book.units) + 2
    header = (f"{'parameter':>17s}" if with_parameter else "")
    print(f"{'':5s}{header}" + "".join(f"{u:>19s}" for u in [*book.units, "total"]))
    largest = 0.0
    for f, pr in priced.items():
        ref = list(aggregate_030[f])
        line = f"{f:5s}"
        if with_parameter:
            ((_, value),) = distortions[f].parameters.items()
            line += f"{value:8.4f} ({ref.pop(0):6.4f})"
        for got, want in zip([x.premium for x in [*pr.allocated, pr.total]], ref):
            line += f"  {got:8.3f} ({want:7.3f})"
            largest = max(largest, abs(got - want))
        print(line)
    print(f"largest difference in a premium: {largest:.4f}")
    return priced
```

`calibrate` uses one shortcut. The premium of the total depends only on the
total capped at the assets, so the distortions are calibrated on that capped
total, a one-unit portfolio of a few thousand points, rather than on the
portfolio's full grid; the parameters are the same, and on the half million
points of section 6 it is much faster. `discretize` matches the way `aggregate`
puts a severity on a grid: the mass of each cell goes to its midpoint, and the
probability above the grid's top is spread over the grid by renormalizing
(`aggregate`'s default `normalize=True`). prospicio's own `Grid.rounding` keeps
that probability on the last point instead; section 6 shows why that matters
for a thick tail.

## 2. Discrete: a book you can price by hand

The first case study is two units with three outcomes each:

```text
port Discrete
    agg X1 1 claim dsev [0 8 10] [1/2 1/4 1/4] fixed
    agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed
```

There are nine joint outcomes, each with a total of its own, so every number
below can be checked with a pencil. Unit X2's 90 makes the tail; X1 never loses more
than 10. `case_studies.py` prices it at the standard `p = 1`, so the assets are
the largest total, 100, and nothing defaults.

```python
x1 = Grid(1.0, [0.5] + [0.0] * 7 + [0.25, 0.0, 0.25])
x2 = Grid(1.0, [0.5, 0.25] + [0.0] * 88 + [0.25])
discrete = Portfolio.from_independent(["X1", "X2"], [x1, x2])
print("totals", discrete.totals)
print("probs ", discrete.probs)

DISCRETE_030 = {  # aggregate 0.30.1: (parameter, P X1, P X2, P total)
    "ccoc": (0.1000, 5.0000, 28.8636, 33.8636),
    "ph": (0.8236, 4.8989, 28.9648, 33.8636),
    "wang": (0.2014, 4.9954, 28.8682, 33.8636),
    "dual": (1.3202, 5.1691, 28.6945, 33.8636),
    "tvar": (0.1953, 5.5922, 28.2715, 33.8636),
}
discrete_d = calibrate(discrete, 1.0)
discrete_p = show(discrete, 1.0, discrete_d, DISCRETE_030)
```

```text
totals [0.0, 1.0, 8.0, 9.0, 10.0, 11.0, 90.0, 98.0, 100.0]
probs  [0.25, 0.125, 0.125, 0.0625, 0.125, 0.0625, 0.125, 0.0625, 0.0625]
assets 100.000; loss within assets: X1 4.500, X2 22.750, total 27.250
             parameter                 X1                 X2              total
ccoc   0.1000 (0.1000)     5.000 (  5.000)    28.864 ( 28.864)    33.864 ( 33.864)
ph     0.8236 (0.8236)     4.899 (  4.899)    28.965 ( 28.965)    33.864 ( 33.864)
wang   0.2014 (0.2014)     4.995 (  4.995)    28.868 ( 28.868)    33.864 ( 33.864)
dual   1.3202 (1.3202)     5.169 (  5.169)    28.695 ( 28.695)    33.864 ( 33.864)
tvar   0.1953 (0.1953)     5.592 (  5.592)    28.271 ( 28.271)    33.864 ( 33.864)
largest difference in a premium: 0.0000
```

The parameters and all fifteen premiums agree. The book's total premium is
`27.25 / 1.1 + 100 / 11 = 33.864`, and the distortions disagree about X1: the
CCoC charges it 5.000 and TVaR 5.592. The CCoC reading is the simplest to
follow. It discounts every unit's expected loss by `v` and adds `d` times the
unit's share of the assets in the worst outcome. In the one outcome where the
total is 100, X1 loses 10 and X2 90, so X1 pays
`4.5 / 1.1 + 10 / 11 = 5.000`. TVaR at 0.195 instead weights every outcome above
the 19.5% quantile equally, and X1 contributes to many of them.

## 3. Tame: when the choice hardly matters

The Tame case study is two gamma units with the same mean of 50 and small
coefficients of variation, 0.10 and 0.15, at a 99.9% standard:

```text
port Tame
    agg A 1 claim sev gamma 50 cv 0.10 fixed
    agg B 1 claim sev gamma 50 cv 0.15 fixed
```

(`aggregate`'s `tame.py` sets the standard at 99.99%; this guide uses 99.9%, as
the other continuous cases do, and runs `aggregate` at 99.9% too.) The grid is
`aggregate`'s, a step of 1/64 and 2^16 points:

```python
STEP, POINTS = 1 / 64, 2**16
a_unit = discretize(Gamma.from_mean_cv(50.0, 0.10), STEP, POINTS)
b_unit = discretize(Gamma.from_mean_cv(50.0, 0.15), STEP, POINTS)
tame = Portfolio.from_independent(["A", "B"], [a_unit, b_unit])

TAME_030 = {  # aggregate 0.30.1, linear allocation
    "ccoc": (0.1000, 50.6820, 52.0950, 102.7770),
    "ph": (0.7289, 50.8134, 51.9636, 102.7770),
    "wang": (0.3062, 50.8437, 51.9333, 102.7770),
    "dual": (1.4415, 50.8693, 51.9077, 102.7770),
    "tvar": (0.1793, 50.8958, 51.8812, 102.7770),
}
tame_d = calibrate(tame, 0.999)
tame_p = show(tame, 0.999, tame_d, TAME_030)
```

```text
assets 130.578; loss within assets: A 49.999, B 49.998, total 99.997
             parameter                  A                  B              total
ccoc   0.1000 (0.1000)    50.682 ( 50.682)    52.095 ( 52.095)   102.777 (102.777)
ph     0.7289 (0.7289)    50.813 ( 50.813)    51.964 ( 51.964)   102.777 (102.777)
wang   0.3062 (0.3062)    50.844 ( 50.844)    51.933 ( 51.933)   102.777 (102.777)
dual   1.4415 (1.4415)    50.869 ( 50.869)    51.908 ( 51.908)   102.777 (102.777)
tvar   0.1793 (0.1793)    50.896 ( 50.896)    51.881 ( 51.881)   102.777 (102.777)
largest difference in a premium: 0.0003
```

Everything agrees to the third decimal. B, the more volatile unit, pays the
larger margin under every distortion, and the five distortions move A's premium
over a range of 0.21 and B's by the same amount the other way. With two thin,
similar tails there is little for the distortions to disagree about: this is the
baseline the next two case studies depart from.

## 4. Cat and non-cat: who pays depends on the distortion

The Cat/Non-Cat case study puts a large, steady unit next to a small, volatile
one. NonCat is a gamma with mean 80 and CV 0.15, Cat a lognormal with mean 20
and CV 1; the total's mean is 100, as in Tame, but with a real tail:

```text
port CNC.Gross
    agg CNC.NonCat 1 claim sev gamma   80 cv 0.15 fixed
    agg CNC.Cat    1 claim sev lognorm 20 cv 1.00 fixed
```

```python
noncat = discretize(Gamma.from_mean_cv(80.0, 0.15), STEP, POINTS)
cat = discretize(Lognormal.from_mean_cv(20.0, 1.0), STEP, POINTS)
cnc = Portfolio.from_independent(["NonCat", "Cat"], [noncat, cat])

CNC_030 = {  # aggregate 0.30.1, linear allocation
    "ccoc": (0.1000, 79.2018, 35.9470, 115.1488),
    "ph": (0.5965, 82.3610, 32.7897, 115.1507),
    "wang": (0.6109, 83.8990, 31.2519, 115.1509),
    "dual": (2.4634, 85.4756, 29.6755, 115.1510),
    "tvar": (0.4816, 86.4562, 28.6948, 115.1511),
}
cnc_d = calibrate(cnc, 0.999)
cnc_p = show(cnc, 0.999, cnc_d, CNC_030)
```

```text
assets 267.203; loss within assets: NonCat 79.988, Cat 19.958, total 99.946
             parameter             NonCat                Cat              total
ccoc   0.1000 (0.1000)    79.202 ( 79.202)    35.950 ( 35.947)   115.151 (115.149)
ph     0.5965 (0.5965)    82.361 ( 82.361)    32.790 ( 32.790)   115.151 (115.151)
wang   0.6109 (0.6109)    83.899 ( 83.899)    31.252 ( 31.252)   115.151 (115.151)
dual   2.4634 (2.4634)    85.476 ( 85.476)    29.676 ( 29.675)   115.151 (115.151)
tvar   0.4816 (0.4816)    86.456 ( 86.456)    28.695 ( 28.695)   115.151 (115.151)
largest difference in a premium: 0.0025
```

The parameters and the expected losses agree, and the premiums to within
0.003. The largest gap is in the CCoC row, and it is `aggregate`'s: 0.30.1's
linear allocation adds up to 115.149 there, 0.002 short of the premium it
calibrated (115.151, which its lifted allocation does reach), while
prospicio's units add up to the calibrated premium.

Now the split matters. The CCoC charges Cat 35.95 for an expected loss of 19.96,
and charges NonCat less than its expected loss: its margin is negative. TVaR
charges Cat 28.69 and NonCat 86.46. The margins, and the assets each unit holds
in the linear allocation, show why:

```python
print(f"{'':5s} {'NonCat M':>9s} {'Cat M':>7s}   NonCat's assets")
for f, pr in cnc_p.items():
    nc, ct = pr.allocated
    print(f"{f:5s} {nc.premium - nc.loss:9.2f} {ct.premium - ct.loss:7.2f}   {nc.assets:7.2f}")
```

```text
       NonCat M   Cat M   NonCat's assets
ccoc      -0.79   15.99     71.34
ph         2.37   12.83    108.30
wang       3.91   11.29    124.54
dual       5.49    9.72    143.91
tvar       6.47    8.74    149.64
```

Under the CCoC a unit pays `v` times its expected loss plus `d` times its share
of the assets in the outcomes at or above them. NonCat's share there, 71.34, is
below its mean of 80: when the book is at its 99.9% quantile it is because the
cat was large. So the CCoC hands nearly all of the margin to Cat. The other
distortions put weight on the whole upper part of the distribution, where
NonCat's own variation counts, and TVaR at 0.48, which weights everything above
the median equally, gives NonCat a margin of 6.47 and Cat 8.74. The case study
is built so that one premium for the book can be split in very different ways;
which distortion is right is a statement about what the capital providers are
paid for, the question of [Risk loads and distortions](risk-loads-and-distortions.md).

## 5. Cat and non-cat, net of an aggregate cover

The net version buys an aggregate stop loss on the cat unit, 79.64 excess of
41.11:

```text
agg CNC.Net.Cat 1 claim sev lognorm 20 cv 1.00 fixed aggregate net of 79.64 xs 41.11
```

The numbers are read off the cat distribution: the cover attaches at its 90%
quantile and exhausts at its 99.5% quantile, 120.75. The net book is priced with
the distortions calibrated on the gross book, at the net book's own 99.9%
assets, which is how `case_studies.py` compares gross and net.

```python
print(f"cat quantiles: 90% {cat.quantile(0.9):.3f}, 99.5% {cat.quantile(0.995):.3f}")
cover = Tower([Layer.stop_loss("79.64 xs 41.11", 79.64, 41.11)]).on_grid(Count.fixed(1), cat, POINTS)
print(f"cat mean {cat.mean():.4f}, ceded {cover.ceded[0].mean():.4f}, net {cover.net.mean():.4f}"
      "  (aggregate 0.30.1: net 17.7870)")
cnc_net = Portfolio.from_independent(["NonCat", "Cat"], [noncat, cover.net])

CNC_NET_030 = {  # aggregate 0.30.1, linear allocation, gross distortions: (P NonCat, P Cat, P total)
    "ccoc": (78.8836, 27.0171, 105.9008),
    "ph": (83.6084, 23.9472, 107.5556),
    "wang": (84.9090, 23.8837, 108.7927),
    "dual": (85.9201, 24.2165, 110.1366),
    "tvar": (86.4892, 24.3973, 110.8865),
}
cnc_net_p = show(cnc_net, 0.999, cnc_d, CNC_NET_030)
```

```text
cat quantiles: 90% 41.109, 99.5% 120.750
cat mean 19.9998, ceded 2.2142, net 17.7856  (aggregate 0.30.1: net 17.7870)
assets 187.578; loss within assets: NonCat 79.985, Cat 17.747, total 97.732
                  NonCat                Cat              total
ccoc     78.883 ( 78.884)    27.016 ( 27.017)   105.900 (105.901)
ph       83.609 ( 83.608)    23.944 ( 23.947)   107.553 (107.556)
wang     84.910 ( 84.909)    23.880 ( 23.884)   108.790 (108.793)
dual     85.921 ( 85.920)    24.213 ( 24.216)   110.134 (110.137)
tvar     86.489 ( 86.489)    24.394 ( 24.397)   110.884 (110.886)
largest difference in a premium: 0.0035
```

The net cat's mean is 17.7856 in prospicio and 17.7870 in `aggregate`, 0.0014
apart. The attachment 41.11 falls between two grid points (41.109375 and
41.125): prospicio's `Tower` splits each recovery between neighbouring points so
that the mean is exact on the grid, and `aggregate` places the cover on the grid
in its own way. That moves the net premiums by up to 0.004; the assets,
187.578, agree.

The cover takes 2.21 of expected loss out of the book and 79.6 out of its
assets. With the distortions held at their gross calibration, the premium the
book needs falls by different amounts:

```python
print(f"{'':5s} {'gross P':>8s} {'net P':>8s} {'saving':>7s}  (ceded expected loss {cover.ceded[0].mean():.2f})")
for f in FAMILIES:
    g, n = cnc_p[f].total.premium, cnc_net_p[f].total.premium
    print(f"{f:5s} {g:8.2f} {n:8.2f} {g - n:7.2f}")
```

```text
       gross P    net P  saving  (ceded expected loss 2.21)
ccoc    115.15   105.90    9.25
ph      115.15   107.55    7.60
wang    115.15   108.79    6.36
dual    115.15   110.13    5.02
tvar    115.15   110.88    4.27
```

Under the CCoC the book's premium falls by 9.25, four times the expected
recovery, because the CCoC prices the tail that the cover removes; under TVaR it
falls by 4.27. Read that as the most the cedent would pay for the cover under
each yardstick (the cover's expected loss plus the margin it releases): the
distortion that matters to the cedent is a business choice, not a technicality.

## 6. Hurricane and severe convective storm

The last case study has two catastrophe perils with a frequency and a severity
each. SCS (severe convective storm) has 70 events a year, lognormal severity
with mean 1 and shape 1.9; Hu (hurricane) has 2 a year, lognormal with mean 15
and shape 2.5, a much thicker tail. Both counts are Poisson, and the grid is a
step of 1/4 and 2^19 points:

```text
port HuSCS.Gross
    agg SCS 70 claims sev 0.16447 * lognorm 1.9 poisson
    agg Hu   2 claims sev 0.65905 * lognorm 2.5 poisson
```

The scale of each lognormal is `m exp(-sigma^2 / 2)` for its mean `m`.
`aggregate`'s test suite writes it `exp(-1.9**2 / 2)`, and 0.30.1's DecL parser
reads `-1.9**2` as `(-1.9)**2`, so that program gives SCS events a mean of 36.97,
not 1; `hs.py`, which builds the program in Python, gets 1. The comparison here
uses the numbers.

```python
HU_STEP, HU_POINTS = 0.25, 2**19
scs_sev = discretize(Lognormal(-(1.9**2) / 2, 1.9), HU_STEP, HU_POINTS)
hu_sev = discretize(Lognormal(math.log(15.0) - 2.5**2 / 2, 2.5), HU_STEP, HU_POINTS)
scs, _ = fft(Poisson(70.0), scs_sev, HU_POINTS)
hu, _ = fft(Poisson(2.0), hu_sev, HU_POINTS)
print(f"annual means: SCS {scs.mean():.4f}, Hu {hu.mean():.4f}  (aggregate 0.30.1: 69.1332, 29.7272)")
hs = Portfolio.from_independent(["SCS", "Hu"], [scs, hu])

HS_030 = {  # aggregate 0.30.1, linear allocation
    "ccoc": (0.1000, 69.9780, 251.3356, 321.3136),
    "ph": (0.4492, 92.8223, 228.4915, 321.3138),
    "wang": (1.1897, 107.3655, 213.9484, 321.3138),
    "dual": (12.0290, 128.7668, 192.5471, 321.3139),
    "tvar": (0.8990, 135.7558, 185.5581, 321.3139),
}
hs_d = calibrate(hs, 0.999)
hs_p = show(hs, 0.999, hs_d, HS_030)
```

```text
annual means: SCS 69.1332, Hu 29.7273  (aggregate 0.30.1: 69.1332, 29.7272)
assets 2583.000; loss within assets: SCS 69.087, Hu 26.058, total 95.145
             parameter                SCS                 Hu              total
ccoc   0.1000 (0.1000)    69.977 ( 69.978)   251.337 (251.336)   321.314 (321.314)
ph     0.4492 (0.4492)    92.822 ( 92.822)   228.492 (228.492)   321.314 (321.314)
wang   1.1897 (1.1897)   107.365 (107.365)   213.949 (213.948)   321.314 (321.314)
dual  12.0290 (12.0290)   128.767 (128.767)   192.547 (192.547)   321.314 (321.314)
tvar   0.8990 (0.8990)   135.756 (135.756)   185.558 (185.558)   321.314 (321.314)
largest difference in a premium: 0.0012
```

The parameters agree, and the premiums to within 0.0012, which is again
`aggregate` 0.30.1's CCoC total falling short of its calibrated premium
(321.3136 against 321.3140). The annual means are below 70 and 30 because
the grid stops at 131,072 and, as in `aggregate`, the severity is renormalized
there; Hu loses 0.27 of its mean of 30 that way. With 2 hurricanes a year the
99.9% assets are 2,583, twenty-six times the expected loss, and the premium of
321.31 is a loss ratio of 30%. CCoC puts 251.34 of it on Hu, and TVaR, at the 90%
level, 185.56. The tail is thick enough that the top of the grid shows in the
answer: with `Grid.rounding`'s convention, which keeps the probability above the
grid on its last point, the 99.9% assets move:

```python
lumped = Grid.rounding(Lognormal(math.log(15.0) - 2.5**2 / 2, 2.5), HU_STEP, HU_POINTS)[0]
hu_lumped, _ = fft(Poisson(2.0), lumped, HU_POINTS)
hs_lumped = Portfolio.from_independent(["SCS", "Hu"], [scs, hu_lumped])
print(f"Hu mean {hu_lumped.mean():.4f}, assets at 99.9% {hs_lumped.assets(0.999):.2f}")
```

```text
Hu mean 29.8662, assets at 99.9% 2584.75
```

Neither convention is the true distribution: the first loses the hurricanes
above 131,072, the second piles them at 131,072. For a severity this thick, check
how the answer moves with the grid before quoting it.

The net version buys a per-occurrence cover on hurricanes, 372.4 excess of
40.25, roughly the band between the hurricane severity's 95% and 99.5%
quantiles:

```text
agg Net.Hu 2 claims sev 0.65905 * lognorm 2.5 occurrence net of 372.4 xs 40.25 poisson
```

```python
print(f"Hu severity quantiles: 95% {hu_sev.quantile(0.95):.2f}, 99.5% {hu_sev.quantile(0.995):.2f}")
hu_cover = Tower([Layer("372.4 xs 40.25", 372.4, 40.25)]).on_grid(Poisson(2.0), hu_sev, HU_POINTS)
print(f"net Hu mean {hu_cover.net.mean():.4f}  (aggregate 0.30.1: 19.6066)")
hs_net = Portfolio.from_independent(["SCS", "Hu"], [scs, hu_cover.net])

HS_NET_030 = {  # aggregate 0.30.1, linear allocation, gross distortions: (P SCS, P Hu, P total)
    "ccoc": (70.8172, 207.5519, 278.3692),
    "ph": (103.5237, 157.0879, 260.6115),
    "wang": (121.5553, 131.7081, 253.2635),
    "dual": (148.2679, 92.9822, 241.2502),
    "tvar": (155.4211, 83.3008, 238.7219),
}
hs_net_p = show(hs_net, 0.999, hs_d, HS_NET_030)
```

```text
Hu severity quantiles: 95% 40.25, 99.5% 412.50
net Hu mean 19.6066  (aggregate 0.30.1: 19.6066)
assets 2211.750; loss within assets: SCS 69.079, Hu 15.952, total 85.031
                     SCS                 Hu              total
ccoc     70.816 ( 70.817)   207.553 (207.552)   278.369 (278.369)
ph      103.523 (103.524)   157.088 (157.088)   260.612 (260.611)
wang    121.555 (121.555)   131.708 (131.708)   253.263 (253.263)
dual    148.268 (148.268)    92.982 ( 92.982)   241.250 (241.250)
tvar    155.421 (155.421)    83.301 ( 83.301)   238.722 (238.722)
largest difference in a premium: 0.0012
```

The net book agrees to within 0.0012 too. The cover takes a third of the hurricane loss (10.1 of
29.7) but only 371 of the 2,583 of assets: the 99.9% outcome is driven by
hurricanes larger than the cover's top of 412.65, which the cedent keeps. Under
the CCoC the book's premium falls from 321.31 to 278.37; under the dual power
and TVaR, which load the SCS unit more, Hu's premium falls by more than half.

## 7. Lifted capital: the version difference again

The tables above use the linear allocation, prospicio's default. With
`allocation="lifted"` the capital, and in one case the premium, of each unit
differ between `aggregate` 0.30.1 and prospicio in the way
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md#5-capital-by-unit-if-you-must)
describes for InsCo, where the cause was traced to a change in `aggregate`
between 0.30.1 and 1.0.1, the version prospicio's natural allocation is tested
against (1.0.1 is not run in this guide). On the gross Cat/Non-Cat book:

```python
LIFTED_030 = {  # aggregate 0.30.1, lifted: (P NonCat, Q NonCat, P Cat, Q Cat)
    "ccoc": (74.6401, -39.5909, 40.5110, 191.6431),
    "ph": (82.2573, 48.5354, 32.8938, 103.5170),
    "wang": (83.8865, 59.8745, 31.2646, 92.1783),
    "dual": (85.4756, 63.0400, 29.6755, 89.0144),
    "tvar": (86.4562, 63.1853, 28.6949, 88.8705),
}
print(f"{'':5s} {'NonCat P':>18s} {'NonCat Q':>18s} {'Cat P':>18s} {'Cat Q':>18s}")
for f, d in cnc_d.items():
    nc, ct = cnc.price(d, p=0.999, allocation="lifted").allocated
    got = (nc.premium, nc.capital, ct.premium, ct.capital)
    print(f"{f:5s} " + " ".join(f"{g:8.3f} ({w:7.3f})" for g, w in zip(got, LIFTED_030[f])))
```

```text
                NonCat P           NonCat Q              Cat P              Cat Q
ccoc    75.430 ( 74.640)  -45.576 (-39.591)   39.721 ( 40.511)  197.627 (191.643)
ph      82.257 ( 82.257)   48.804 ( 48.535)   32.894 ( 32.894)  103.248 (103.517)
wang    83.886 ( 83.886)   59.875 ( 59.874)   31.265 ( 31.265)   92.177 ( 92.178)
dual    85.476 ( 85.476)   63.040 ( 63.040)   29.676 ( 29.675)   89.012 ( 89.014)
tvar    86.456 ( 86.456)   63.185 ( 63.185)   28.695 ( 28.695)   88.867 ( 88.871)
```

Wang, the dual power and TVaR agree to about 0.004. Under PH the premiums agree
and the capital does not (NonCat 48.80 against 48.54). Under the CCoC even the
lifted premiums differ (NonCat 75.43 against 74.64), and both programs give
NonCat a negative capital. The totals agree in every row. As in the monograph
guide, use the premiums and margins of the linear allocation, which agree, and
treat capital by unit under the CCoC and PH as version dependent.

## 8. What this guide leaves out

- **The classical premium principles** and the "blend" distortions that
  `case_studies.py` calibrates to three spreads (`calibrate_blends`); prospicio
  has the classical principles (see
  [Risk loads and distortions](risk-loads-and-distortions.md)) and the bitvar
  and weighted TVaR families, but this guide does not reproduce those exhibits.
- **The Tame case's net** (an aggregate cover between B's 80% and 99%
  quantiles in `tame.py`) and the equal-points variant of Discrete (X1's 8
  becomes 9), which run the same way.
- **Bounds on the premiums** (`Portfolio.premium_bounds`), the Bodoff
  allocation (`Portfolio.bodoff`) and the stand-alone prices
  (`Portfolio.price_stand_alone`), which exist in prospicio and are compared
  with `aggregate` in CI on the Discrete case.
- **The book's figures** and its reasoning, for which read the book.

Where to go next: [prospicio for aggregate users](prospicio-for-aggregate-users.md)
for the DecL clauses used above, and
[What a reinsurance price is](reinsurance-price.md) for pricing the covers of
sections 5 and 6 from the reinsurer's side.
