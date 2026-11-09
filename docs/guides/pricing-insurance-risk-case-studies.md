# Pricing Insurance Risk: the case studies in prospicio

<!-- site: execute -->

Stephen Mildenhall and John Major's book *Pricing Insurance Risk* (Wiley, 2022)
applies one method to four case studies:

1. Build the distribution of the total of a book from its units.
2. Select a capital standard.
3. Price the total with a spectral risk measure calibrated to a target return.
4. Split the price over the units by the natural allocation.

Mildenhall's package `aggregate` made the book's exhibits. Version 0.30.1 still
holds the code (`aggregate.extensions.case_studies`) and the case studies as
DecL programs in its tests. This guide runs the case studies in prospicio, with
the value that `aggregate` 0.30.1 returns beside each number. The settings are
those of `case_studies.py`: a 10% return on capital, and each case's capital
standard and grid. If distortion, natural allocation or lifted capital are new
words, read [CAS Monograph 15 reproduced](capital-modeling-monograph-15.md)
first.

## 1. The method in one function

Each case study has two independent units on one grid.
`Portfolio.from_independent` combines them and keeps each unit's expected loss
given the total. The assets are the quantile of the total at the capital
standard `p`.

The constant cost of capital (CCoC) at 10% sets the premium of the total,
`P = v E[min(X, a)] + d a`, with `v = 1 / 1.1` and `d = 1 - v`; it earns exactly
10% on the capital `a - P`. The code calibrates the four other distortions
(proportional hazard (PH), Wang, dual power and TVaR) to the same total premium.
They differ only in how they split it.

```python
import math

import pandas as pd
from great_tables import GT

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
    """Price the book under each distortion. Print the assets and the largest gap to aggregate 0.30.1.

    Returns the prices and a frame whose rows alternate prospicio and aggregate 0.30.1.
    """
    priced = {f: book.price(d, p=p) for f, d in distortions.items()}
    first = priced["ccoc"]
    losses = ", ".join(f"{u} {x.loss:.3f}" for u, x in zip(book.units, first.allocated))
    print(f"assets {book.assets(p):.3f}; loss within assets: {losses}, total {first.total.loss:.3f}")
    with_parameter = len(aggregate_030["ccoc"]) == len(book.units) + 2
    rows, largest = [], 0.0
    for f, pr in priced.items():
        ref = list(aggregate_030[f])
        mine, theirs = [], []
        if with_parameter:
            ((_, value),) = distortions[f].parameters.items()
            mine, theirs = [value], [ref.pop(0)]
        got = [x.premium for x in [*pr.allocated, pr.total]]
        largest = max(largest, *(abs(g - w) for g, w in zip(got, ref)))
        rows += [(f, *mine, *got), (f"{f}, aggregate", *theirs, *ref)]
    print(f"largest difference in a premium: {largest:.4f}")
    columns = ["distortion", *(["parameter"] if with_parameter else []), *book.units, "total"]
    return priced, pd.DataFrame(rows, columns=columns)


def comparison(frame, title, subtitle):
    """The frame of `show` as a table: premiums at 3 decimals, the distortion's parameter at 4."""
    table = GT(frame, rowname_col="distortion").tab_header(title, subtitle)
    premiums = [c for c in frame.columns[1:] if c != "parameter"]
    table = table.fmt_number(premiums, decimals=3, use_seps=False)
    return table.fmt_number("parameter", decimals=4) if "parameter" in frame else table
```

The premium of the total depends only on the total capped at the assets, so
`calibrate` fits the distortions on that capped total, a one-unit portfolio. The
parameters are the same, and on the half million points of section 6 this is much
faster. `discretize` renormalizes as `aggregate` does (`normalize=True`): it
spreads the probability above the top of the grid over the grid.
`Grid.rounding` keeps it on the last point. Section 6 shows why this matters for
a thick tail.

## 2. Discrete: a book you can price by hand

The first case study is two units with three outcomes each:

```text
port Discrete
    agg X1 1 claim dsev [0 8 10] [1/2 1/4 1/4] fixed
    agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed
```

The nine joint outcomes have different totals, so you can check every number
with a pencil. The 90 of X2 makes the tail. `case_studies.py` prices the book at
`p = 1`, so the assets are the largest total, 100, and no outcome defaults.

```python
x1 = Grid(1.0, [0.5] + [0.0] * 7 + [0.25, 0.0, 0.25])
x2 = Grid(1.0, [0.5, 0.25] + [0.0] * 88 + [0.25])
discrete = Portfolio.from_independent(["X1", "X2"], [x1, x2])
outcomes = pd.DataFrame({"total": discrete.totals, "probability": discrete.probs})
(GT(outcomes)
 .tab_header("Discrete: the nine joint outcomes", "Total loss of X1 and X2 and its probability")
 .fmt_number("total", decimals=0)
 .fmt_number("probability", decimals=4, drop_trailing_zeros=True))
```

```text
total  probability
    0         0.25
    1        0.125
    8        0.125
    9       0.0625
   10        0.125
   11       0.0625
   90        0.125
   98       0.0625
  100       0.0625
```

```python
DISCRETE_030 = {  # aggregate 0.30.1: (parameter, P X1, P X2, P total)
    "ccoc": (0.1000, 5.0000, 28.8636, 33.8636),
    "ph": (0.8236, 4.8989, 28.9648, 33.8636),
    "wang": (0.2014, 4.9954, 28.8682, 33.8636),
    "dual": (1.3202, 5.1691, 28.6945, 33.8636),
    "tvar": (0.1953, 5.5922, 28.2715, 33.8636),
}
discrete_d = calibrate(discrete, 1.0)
discrete_p, discrete_t = show(discrete, 1.0, discrete_d, DISCRETE_030)
```

```text
assets 100.000; loss within assets: X1 4.500, X2 22.750, total 27.250
largest difference in a premium: 0.0000
```

```python
comparison(discrete_t, "Discrete: premium by unit", "Standard p = 1, 10% return; each distortion against aggregate 0.30.1")
```

```text
                 parameter     X1      X2   total
ccoc                0.1000  5.000  28.864  33.864
ccoc, aggregate     0.1000  5.000  28.864  33.864
ph                  0.8236  4.899  28.965  33.864
ph, aggregate       0.8236  4.899  28.965  33.864
wang                0.2014  4.995  28.868  33.864
wang, aggregate     0.2014  4.995  28.868  33.864
dual                1.3202  5.169  28.695  33.864
dual, aggregate     1.3202  5.169  28.695  33.864
tvar                0.1953  5.592  28.271  33.864
tvar, aggregate     0.1953  5.592  28.271  33.864
```

The total premium is `27.25 / 1.1 + 100 / 11 = 33.864`. The distortions
disagree about X1. The CCoC discounts each unit's expected loss by `v` and adds
`d` times its share of the assets in the worst outcome. There X1 loses 10 and X2
loses 90, so X1 pays `4.5 / 1.1 + 10 / 11 = 5.000`. TVaR at 0.195 weights every
outcome above the 19.5% quantile equally, and X1 contributes to many of them, so
it pays 5.592.

## 3. Tame: when the choice hardly matters

Tame has two gamma units with mean 50 and small CVs, 0.10 and 0.15:

```text
port Tame
    agg A 1 claim sev gamma 50 cv 0.10 fixed
    agg B 1 claim sev gamma 50 cv 0.15 fixed
```

`aggregate`'s `tame.py` sets the standard at 99.99%. This guide uses 99.9%, as in
the other continuous cases, and ran `aggregate` at 99.9% too. The grid has a step
of 1/64 and 2^16 points:

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
tame_p, tame_t = show(tame, 0.999, tame_d, TAME_030)
```

```text
assets 130.578; loss within assets: A 49.999, B 49.998, total 99.997
largest difference in a premium: 0.0003
```

```python
comparison(tame_t, "Tame: premium by unit", "Standard p = 99.9%, 10% return; each distortion against aggregate 0.30.1")
```

```text
                 parameter       A       B    total
ccoc                0.1000  50.682  52.095  102.777
ccoc, aggregate     0.1000  50.682  52.095  102.777
ph                  0.7289  50.813  51.964  102.777
ph, aggregate       0.7289  50.813  51.964  102.777
wang                0.3062  50.844  51.933  102.777
wang, aggregate     0.3062  50.844  51.933  102.777
dual                1.4415  50.869  51.908  102.777
dual, aggregate     1.4415  50.869  51.908  102.777
tvar                0.1793  50.896  51.881  102.777
tvar, aggregate     0.1793  50.896  51.881  102.777
```

B, the more volatile unit, pays the larger margin under each distortion. The
distortions move A's premium over a range of 0.21 and B's by the same amount the
other way. Thin, similar tails leave little to disagree about.

## 4. Cat and non-cat: who pays depends on the distortion

Cat/Non-Cat puts a large, steady unit next to a small, volatile one. NonCat is a
gamma with mean 80 and CV 0.15, and Cat is a lognormal with mean 20 and CV 1.
The total has mean 100, as in Tame, but a real tail:

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
cnc_p, cnc_t = show(cnc, 0.999, cnc_d, CNC_030)
```

```text
assets 267.203; loss within assets: NonCat 79.988, Cat 19.958, total 99.946
largest difference in a premium: 0.0025
```

```python
comparison(cnc_t, "Cat/Non-Cat: premium by unit", "Standard p = 99.9%, 10% return; each distortion against aggregate 0.30.1")
```

```text
                 parameter  NonCat     Cat    total
ccoc                0.1000  79.202  35.950  115.151
ccoc, aggregate     0.1000  79.202  35.947  115.149
ph                  0.5965  82.361  32.790  115.151
ph, aggregate       0.5965  82.361  32.790  115.151
wang                0.6109  83.899  31.252  115.151
wang, aggregate     0.6109  83.899  31.252  115.151
dual                2.4634  85.476  29.676  115.151
dual, aggregate     2.4634  85.476  29.675  115.151
tvar                0.4816  86.456  28.695  115.151
tvar, aggregate     0.4816  86.456  28.695  115.151
```

The largest difference is in the CCoC row, and `aggregate` causes it: its linear
allocation adds up to 115.149, 0.002 less than the premium it calibrated
(115.151, which its lifted allocation does reach). The units of prospicio add up
to the calibrated premium.

Now the split matters: the CCoC charges NonCat less than its expected loss. The
margins and NonCat's assets in the linear allocation show why:

```python
margins = pd.DataFrame({
    "distortion": FAMILIES,
    "NonCat margin": [cnc_p[f].allocated[0].premium - cnc_p[f].allocated[0].loss for f in FAMILIES],
    "Cat margin": [cnc_p[f].allocated[1].premium - cnc_p[f].allocated[1].loss for f in FAMILIES],
    "NonCat assets": [cnc_p[f].allocated[0].assets for f in FAMILIES],
})
(GT(margins, rowname_col="distortion")
 .tab_header("Cat/Non-Cat: margin and assets by unit", "Margin is premium minus expected loss; linear allocation")
 .fmt_number(["NonCat margin", "Cat margin", "NonCat assets"], decimals=2))
```

```text
      NonCat margin  Cat margin  NonCat assets
ccoc          -0.79       15.99          71.34
ph             2.37       12.83         108.30
wang           3.91       11.29         124.54
dual           5.49        9.72         143.91
tvar           6.47        8.74         149.64
```

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(FAMILIES))
fig, ax = plt.subplots()
for k, (unit, shift) in enumerate([("NonCat", -0.2), ("Cat", 0.2)]):
    unit_margin = [cnc_p[f].allocated[k].premium - cnc_p[f].allocated[k].loss for f in FAMILIES]
    ax.bar(x + shift, unit_margin, width=0.4, label=unit)
ax.axhline(0, color="0.75")
ax.set_xticks(x, FAMILIES)
ax.set_xlabel("distortion")
ax.set_ylabel("margin (premium minus expected loss)")
ax.set_title("Cat/Non-Cat: the margin of each unit, by distortion")
ax.legend()
```

Under the CCoC a unit pays `v` times its expected loss, plus `d` times its share
of the assets in the outcomes at or above the assets. NonCat's share there, 71.34,
is below its mean of 80, because a large cat loss causes the book's 99.9%
outcome. So the CCoC gives almost all of the margin to Cat. The other
distortions weight the whole upper tail, where NonCat's own variation counts.
The right distortion depends on what the capital providers get paid for, the
question of
[Risk loads and distortions](risk-loads-and-distortions.md).

## 5. Cat and non-cat, net of an aggregate cover

The net version buys an aggregate stop loss on the cat unit, 79.64 excess of
41.11:

```text
agg CNC.Net.Cat 1 claim sev lognorm 20 cv 1.00 fixed aggregate net of 79.64 xs 41.11
```

The cover attaches at the 90% quantile of the cat distribution and exhausts at
its 99.5% quantile, 120.75. As in `case_studies.py`, the net book is priced with
the distortions calibrated on the gross book, at the net book's 99.9% assets.

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
cnc_net_p, cnc_net_t = show(cnc_net, 0.999, cnc_d, CNC_NET_030)
```

```text
cat quantiles: 90% 41.109, 99.5% 120.750
cat mean 19.9998, ceded 2.2142, net 17.7856  (aggregate 0.30.1: net 17.7870)
assets 187.578; loss within assets: NonCat 79.985, Cat 17.747, total 97.732
largest difference in a premium: 0.0035
```

```python
comparison(cnc_net_t, "Cat/Non-Cat net of 79.64 xs 41.11: premium by unit",
           "Standard p = 99.9%, gross distortions; each against aggregate 0.30.1")
```

```text
                 NonCat     Cat    total
ccoc             78.883  27.016  105.900
ccoc, aggregate  78.884  27.017  105.901
ph               83.609  23.944  107.553
ph, aggregate    83.608  23.947  107.556
wang             84.910  23.880  108.790
wang, aggregate  84.909  23.884  108.793
dual             85.921  24.213  110.134
dual, aggregate  85.920  24.216  110.137
tvar             86.489  24.394  110.884
tvar, aggregate  86.489  24.397  110.886
```

The attachment 41.11 lies between two grid points (41.109375 and 41.125).
`Tower` splits each recovery between adjacent points, so the mean is exact on the
grid. `aggregate` places the cover differently, which moves the net cat mean by
0.0014 and the net premiums by up to 0.004. The assets, 187.578, agree.

The cover removes 2.21 of expected loss and 79.6 of assets. With the
distortions at their gross calibration, the premium falls by different amounts:

```python
saving = pd.DataFrame({
    "distortion": FAMILIES,
    "gross P": [cnc_p[f].total.premium for f in FAMILIES],
    "net P": [cnc_net_p[f].total.premium for f in FAMILIES],
})
saving["saving"] = saving["gross P"] - saving["net P"]
(GT(saving, rowname_col="distortion")
 .tab_header("Cat/Non-Cat: premium of the book gross and net of the cover",
             "Each distortion at its gross calibration, 10% return, p = 99.9%")
 .fmt_number(["gross P", "net P", "saving"], decimals=2))
```

```text
      gross P   net P  saving
ccoc   115.15  105.90    9.25
ph     115.15  107.55    7.60
wang   115.15  108.79    6.36
dual   115.15  110.13    5.02
tvar   115.15  110.88    4.27
```

Under the CCoC the saving is four times the expected recovery, because the CCoC
prices the tail that the cover removes. Read each saving as the most that the
cedent can pay for the cover under that measure: its expected loss plus the
margin it releases. Which distortion the cedent uses is a business decision.

## 6. Hurricane and severe convective storm

Two catastrophe perils, each with a frequency and a severity. SCS (severe
convective storm) has 70 Poisson events a year with a lognormal severity of mean
1 and shape 1.9. Hu (hurricane) has 2 Poisson events a year with a lognormal
severity of mean 15 and shape 2.5, a much thicker tail. The grid has a step of
1/4 and 2^19 points:

```text
port HuSCS.Gross
    agg SCS 70 claims sev 0.16447 * lognorm 1.9 poisson
    agg Hu   2 claims sev 0.65905 * lognorm 2.5 poisson
```

The scale of each lognormal is `m exp(-sigma^2 / 2)` for its mean `m`. The DecL
parser of 0.30.1 reads the test suite's `exp(-1.9**2 / 2)` with `-1.9**2` as
`(-1.9)**2`, so that program gives SCS events a mean of 36.97, not 1. `hs.py`
builds the program in Python and gets 1. The comparison here uses the numbers.

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
hs_p, hs_t = show(hs, 0.999, hs_d, HS_030)
```

```text
annual means: SCS 69.1332, Hu 29.7273  (aggregate 0.30.1: 69.1332, 29.7272)
assets 2583.000; loss within assets: SCS 69.087, Hu 26.058, total 95.145
largest difference in a premium: 0.0012
```

```python
comparison(hs_t, "Hurricane/SCS: premium by unit", "Standard p = 99.9%, 10% return; each distortion against aggregate 0.30.1")
```

```text
                 parameter      SCS       Hu    total
ccoc                0.1000   69.977  251.337  321.314
ccoc, aggregate     0.1000   69.978  251.336  321.314
ph                  0.4492   92.822  228.492  321.314
ph, aggregate       0.4492   92.822  228.492  321.314
wang                1.1897  107.365  213.949  321.314
wang, aggregate     1.1897  107.365  213.948  321.314
dual               12.0290  128.767  192.547  321.314
dual, aggregate    12.0290  128.767  192.547  321.314
tvar                0.8990  135.756  185.558  321.314
tvar, aggregate     0.8990  135.756  185.558  321.314
```

The remaining difference is again `aggregate`'s CCoC total, below its calibrated
premium (321.3136 against 321.3140). The annual means are below 70 and 30
because the grid stops at 131,072. Renormalizing the severity, as `aggregate`
does, makes Hu lose 0.27 of its mean of 30.

The 99.9% assets, 2,583, are twenty-six times the expected loss, and the premium
of 321.31 is a loss ratio of 30%. The tail is so thick that the top of the grid
changes the answer. With `Grid.rounding`, which keeps the probability above the
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

Neither convention gives the true distribution. The first loses the hurricanes
above 131,072, and the second puts them all at 131,072. For a severity this
thick, check the answer against the grid before you quote it.

The net version buys a per-occurrence cover on hurricanes, 372.4 excess of
40.25, about the band between the 95% and 99.5% severity quantiles:

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
hs_net_p, hs_net_t = show(hs_net, 0.999, hs_d, HS_NET_030)
```

```text
Hu severity quantiles: 95% 40.25, 99.5% 412.50
net Hu mean 19.6066  (aggregate 0.30.1: 19.6066)
assets 2211.750; loss within assets: SCS 69.079, Hu 15.952, total 85.031
largest difference in a premium: 0.0012
```

```python
comparison(hs_net_t, "Hurricane/SCS net of 372.4 xs 40.25: premium by unit",
           "Standard p = 99.9%, gross distortions; each against aggregate 0.30.1")
```

```text
                     SCS       Hu    total
ccoc              70.816  207.553  278.369
ccoc, aggregate   70.817  207.552  278.369
ph               103.523  157.088  260.612
ph, aggregate    103.524  157.088  260.611
wang             121.555  131.708  253.263
wang, aggregate  121.555  131.708  253.263
dual             148.268   92.982  241.250
dual, aggregate  148.268   92.982  241.250
tvar             155.421   83.301  238.722
tvar, aggregate  155.421   83.301  238.722
```

```python
fig, ax = plt.subplots()
x = np.arange(len(FAMILIES))
ax.bar(x - 0.2, [hs_p[f].allocated[1].premium for f in FAMILIES], width=0.4, label="gross")
ax.bar(x + 0.2, [hs_net_p[f].allocated[1].premium for f in FAMILIES], width=0.4, label="net of the cover")
ax.set_xticks(x, FAMILIES)
ax.set_xlabel("distortion")
ax.set_ylabel("premium of Hu")
ax.set_title("Hurricane/SCS: the premium of Hu, gross and net of 372.4 xs 40.25")
ax.legend()
```

The cover removes a third of the hurricane loss (10.1 of 29.7) but only 371 of
the 2,583 of assets: hurricanes above the top of the cover, 412.65, cause the
99.9% outcome, and the cedent keeps them. Under the dual power and TVaR, which
load SCS more, the Hu premium falls by more than half.

## 7. Lifted capital: the version difference again

The tables above use the linear allocation, the default. With
`allocation="lifted"`, the capital of each unit, and in one case the premium,
differ between `aggregate` 0.30.1 and prospicio.
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md#5-capital-by-unit-if-you-must)
shows the same difference for InsCo and traces it to a change in `aggregate`
between 0.30.1 and 1.0.1; prospicio's tests of the natural allocation use 1.0.1.
Gross Cat/Non-Cat, with P the premium and Q the capital:

```python
LIFTED_030 = {  # aggregate 0.30.1, lifted: (P NonCat, Q NonCat, P Cat, Q Cat)
    "ccoc": (74.6401, -39.5909, 40.5110, 191.6431),
    "ph": (82.2573, 48.5354, 32.8938, 103.5170),
    "wang": (83.8865, 59.8745, 31.2646, 92.1783),
    "dual": (85.4756, 63.0400, 29.6755, 89.0144),
    "tvar": (86.4562, 63.1853, 28.6949, 88.8705),
}
rows = []
for f, d in cnc_d.items():
    nc, ct = cnc.price(d, p=0.999, allocation="lifted").allocated
    rows += [(f, nc.premium, nc.capital, ct.premium, ct.capital), (f"{f}, aggregate", *LIFTED_030[f])]
lifted = pd.DataFrame(rows, columns=["distortion", "NonCat P", "NonCat Q", "Cat P", "Cat Q"])
(GT(lifted, rowname_col="distortion")
 .tab_header("Cat/Non-Cat: premium and capital by unit, lifted allocation",
             "Standard p = 99.9%, 10% return; each distortion against aggregate 0.30.1")
 .fmt_number(["NonCat P", "NonCat Q", "Cat P", "Cat Q"], decimals=2, use_seps=False))
```

```text
                 NonCat P  NonCat Q  Cat P   Cat Q
ccoc                75.43    -45.58  39.72  197.63
ccoc, aggregate     74.64    -39.59  40.51  191.64
ph                  82.26     48.80  32.89  103.25
ph, aggregate       82.26     48.54  32.89  103.52
wang                83.89     59.88  31.26   92.18
wang, aggregate     83.89     59.87  31.26   92.18
dual                85.48     63.04  29.68   89.01
dual, aggregate     85.48     63.04  29.68   89.01
tvar                86.46     63.18  28.69   88.87
tvar, aggregate     86.46     63.19  28.69   88.87
```

The table shows two decimals because the CCoC row is sensitive to round-off: it
moves in the fourth decimal when the CCoC's `r` changes in its last bit. Wang,
dual power and TVaR agree to about 0.004. Under PH the premiums agree and the
capital does not (NonCat 48.80 against 48.54). Under the CCoC the premiums also
differ (NonCat 75.43 against 74.64), and both programs give NonCat a negative
capital. The totals agree in each row. As in the monograph guide, use the
premiums and margins of the linear allocation, which agree, and treat capital by
unit under the CCoC and PH as dependent on the version.

Where to go next:
[prospicio for aggregate users](prospicio-for-aggregate-users.md) for the DecL
clauses used above. [What a reinsurance price is](reinsurance-price.md) for how
a reinsurer prices the covers of sections 5 and 6.
