# Capital modeling and portfolio pricing: CAS Monograph 15 reproduced

<!-- site: execute -->

CAS Monograph 15 is *Introduction to Capital Modeling and Portfolio
Management* (John Major and Stephen Mildenhall, 2026). It shows how an insurer
prices its risk and splits the price over its business units, from a
distribution of outcomes alone. The monograph's examples run in Mildenhall's
Python package `aggregate`. This guide runs them in prospicio and shows both
sets of numbers.

Read the monograph for the reasoning (chapters 3 to 6 and 8) and this guide for
the mechanics (chapter 9). Table numbers are the monograph's. Money is in the
units of its examples.

## 1. The example company

The running example, InsCo, has three units, A, B and C, and ten equally
likely simulated years. Each row is a year, and each cell is the unit's loss in
that year. The first table is the last two rows of Table 9.1.

```python
import pandas as pd
from great_tables import GT

from prospicio.distributions import Sampled
from prospicio.pricing import Portfolio
from prospicio.risk import Distortion

ROWS = [
    [5, 20, 11], [7, 33, 0], [15, 13, 0], [15, 7, 0], [13, 20, 7],
    [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64],
]
insco = Portfolio(["A", "B", "C"], ROWS)
totals = [sum(row) for row in ROWS]

stats = []
for name, losses in zip(["A", "B", "C", "Total"], [*zip(*ROWS), totals]):
    s = Sampled([float(x) for x in losses])
    sd = (sum((x - s.mean()) ** 2 for x in losses) / len(losses)) ** 0.5
    stats.append([name, s.mean(), sd / s.mean()])
(GT(pd.DataFrame(stats, columns=["unit", "mean", "cv"]), rowname_col="unit")
 .tab_header("InsCo: mean and CV of each unit's loss", "Table 9.1, last two rows")
 .fmt_number("mean", decimals=2)
 .fmt_number("cv", decimals=3))
```

```text
        mean     cv
A      13.40  0.453
B      18.30  0.412
C      14.90  1.324
Total  46.60  0.455
```

Unit C is the volatile one: a small mean and a loss of 64 in the worst year.

`Portfolio` takes the rows directly. There is no frequency-severity model, no
grid and no approximation.

Pricing works on the distinct totals, not on the years. Four years each total
40, so the portfolio keeps one scenario at 40 with probability 0.4. For each
unit, it keeps the average loss at that total, the conditional expectation
`kappa`. Each row of the table adds up to its total, so a split of the price by
`kappa` never creates or loses money. The row at 40 is Table 9.4.

```python
kappa = pd.DataFrame(
    {"total": [f"{t:.0f}" for t in insco.totals], "probability": insco.probs,
     **{unit: insco.kappa(unit) for unit in ("A", "B", "C")}}
)
(GT(kappa, rowname_col="total")
 .tab_header("InsCo: expected loss of each unit at each total", "kappa, by distinct total")
 .fmt_number("probability", decimals=1)
 .fmt_number(["A", "B", "C"], decimals=2))
```

```text
     probability      A      B      C
22           0.1  15.00   7.00   0.00
28           0.1  15.00  13.00   0.00
36           0.1   5.00  20.00  11.00
40           0.4  10.00  24.00   6.00
55           0.1  26.00  19.00  10.00
65           0.1  17.00   8.00  40.00
100          0.1  16.00  20.00  64.00
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for unit in ("A", "B", "C"):
    ax.plot(insco.totals, insco.kappa(unit), marker="o", label=f"unit {unit}")
ax.set_xlabel("total loss of the book")
ax.set_ylabel("expected loss of the unit at that total")
ax.set_title("InsCo: each unit's conditional expectation, kappa, by total")
ax.legend()
```

## 2. The capital standard

A price needs a capital standard: the assets that stand behind the book.
Chapter 3 of the monograph defines value at risk (VaR, a quantile) and tail
value at risk (TVaR, the average above that quantile). `aggregate` and
prospicio read a standard of "p" as assets equal to the p quantile of the
total. The table is Table 9.5:

```python
total = Sampled([float(t) for t in totals])
standards = pd.DataFrame(
    [[f"{p:.2f}", total.quantile(p), total.tvar(p), insco.assets(p)] for p in (0.8, 0.85, 0.9, 1.0)],
    columns=["p", "VaR", "TVaR", "assets at p"],
)
(GT(standards, rowname_col="p")
 .tab_header("InsCo: capital standards", "Table 9.5")
 .fmt_number(["VaR", "assets at p"], decimals=0)
 .fmt_number("TVaR", decimals=3))
```

```text
      VaR     TVaR  assets at p
0.80   55   82.500           55
0.85   65   88.333           65
0.90   65  100.000           65
1.00  100  100.000          100
```

With ten years, the 100% standard is the worst year. The monograph's main
example uses it: assets of 100. A 99.5% standard gives the same assets.

## 3. Calibrating five distortions

The monograph prices with a **distortion** (a spectral risk measure): a
function `g` that increases the probability of bad outcomes. The price is the
expected loss under the increased probabilities.

Here the whole book must earn 15% on capital, with assets of 100 and an
expected loss of 46.6. The premium `P` and the capital `Q = 100 - P` satisfy
`P - 46.6 = 0.15 Q`, so `P = 53.565`.

Five families each have one parameter: constant cost of capital (CCoC),
proportional hazard (PH), Wang, dual power and TVaR. Calibration finds the
parameter that gives that premium. The table compares it with Table 9.3:

```python
FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
distortions = {
    family: insco.calibrate(family, assets=100, return_on_capital=0.15)
    for family in FAMILIES
}
BOOK_PARAMS_9_3 = {"ccoc": 0.150, "ph": 0.720, "wang": 0.343, "dual": 1.595, "tvar": 0.271}
rows = []
for family, d in distortions.items():
    ((name, value),) = d.parameters.items()
    rows.append([family, name, value, BOOK_PARAMS_9_3[family]])
(GT(pd.DataFrame(rows, columns=["distortion", "parameter", "prospicio", "book"]), rowname_col="distortion")
 .tab_header("InsCo: calibrated distortion parameters", "Table 9.3")
 .fmt_number(["prospicio", "book"], decimals=3))
```

```text
      parameter  prospicio   book
ccoc          r      0.150  0.150
ph          rho      0.720  0.720
wang     lambda      0.343  0.343
dual       beta      1.595  1.595
tvar          p      0.271  0.271
```

The CCoC parameter differs in meaning: `aggregate` uses a risk discount
`d = 0.15 / 1.15`, prospicio the cost of capital `r = 0.15`.
`Distortion.ccoc(0.15)` is the same function.

Each distortion is a curve `g(s)`. The figure draws the five curves that the
calibration found. The grey line is `g(s) = s`, which gives no risk load. The
price rises with the area between a curve and this line:

```python
s_grid = [i / 100 for i in range(101)]
panels = [("ccoc", "ph", "wang"), ("dual", "tvar")]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
for ax, group in zip(axes, panels):
    ax.plot(s_grid, s_grid, color="0.75")
    for family in group:
        ax.plot(s_grid, [distortions[family].g(x) for x in s_grid], label=family)
    ax.set_xlabel("probability of exceedance, s")
    ax.legend()
axes[0].set_ylabel("distorted probability, g(s)")
axes[0].set_title("Distortions fitted to InsCo (1 of 2)")
axes[1].set_title("Distortions fitted to InsCo (2 of 2)")
```

## 4. The price and its allocation to units

After calibration, `price` returns the premium of the loss capped at the
assets. It also returns the **natural allocation** of that premium: each unit
pays the distorted average of its own loss in the bad years, so a unit that
loses most when the total is large pays more. With `allocation="linear"`,
`price` also splits the assets by expected shares above the asset level. This
is Table 9.6, the "industry standard" CCoC pricing:

```python
priced = insco.price(distortions["ccoc"], assets=100, allocation="linear")
rows = []
for name, u in zip(priced.units + ["Total"], [*priced.allocated, priced.total]):
    rows.append([name, u.loss, u.assets, u.capital, u.premium, u.premium - u.loss, (u.premium - u.loss) / u.capital])
(GT(pd.DataFrame(rows, columns=["unit", "L", "a", "Q", "P", "M", "COC"]), rowname_col="unit")
 .tab_header("InsCo: CCoC pricing at 15%", "Table 9.6; L loss, a assets, Q capital, P premium, M margin, COC margin over capital")
 .fmt_number(["L", "a", "Q", "P", "M", "COC"], decimals=3))
```

```text
            L        a       Q       P      M    COC
A      13.400   16.000   2.261  13.739  0.339  0.150
B      18.300   20.000   1.478  18.522  0.222  0.150
C      14.900   64.000  42.696  21.304  6.404  0.150
Total  46.600  100.000  46.435  53.565  6.965  0.150
```

Every unit earns 15% on its allocated capital, because this allocation defines
capital that way. Unit C takes 64 of the 100 of assets and most of the margin
(6.4 of 7.0).

The five distortions give the same total premium, because calibration sets it.
They differ in which unit pays. Rows alternate prospicio and Table 9.8:

```python
def with_book(got, book, columns):
    """Rows alternate prospicio (`X`) and the monograph (`X, book`)."""
    rows = []
    for family, values in got.items():
        rows += [[family, *values], [f"{family}, book", *book[family]]]
    return pd.DataFrame(rows, columns=["distortion", *columns])


BOOK_P = {  # Table 9.8, premium by unit (A, B, C)
    "ccoc": (13.739, 18.522, 21.304),
    "ph": (14.060, 18.349, 21.156),
    "wang": (14.109, 18.637, 20.819),
    "dual": (14.127, 19.117, 20.322),
    "tvar": (13.783, 20.412, 19.371),
}
got = {
    family: [u.premium for u in insco.price(d, assets=100).allocated]
    for family, d in distortions.items()
}
(GT(with_book(got, BOOK_P, ["A", "B", "C"]), rowname_col="distortion")
 .tab_header("InsCo: premium by unit under five distortions", "Table 9.8")
 .fmt_number(["A", "B", "C"], decimals=3))
```

```text
                 A       B       C
ccoc        13.739  18.522  21.304
ccoc, book  13.739  18.522  21.304
ph          14.060  18.349  21.156
ph, book    14.060  18.349  21.156
wang        14.109  18.637  20.819
wang, book  14.109  18.637  20.819
dual        14.127  19.117  20.322
dual, book  14.127  19.117  20.322
tvar        13.783  20.412  19.371
tvar, book  13.783  20.412  19.371
```

All fifteen premiums match. CCoC charges for the capital each unit uses, and C
uses 64 of the 100, so C pays most. TVaR charges only for the worst years,
where B's 20 is a larger share of the total than its average share, so B pays
more. Neither distortion is right: it states what you are paid to bear (see
[Risk loads and distortions](risk-loads-and-distortions.md)).

## 5. Capital by unit, if you must

Section 8.3 of the monograph advises against capital allocation, because
layers of capital cost different amounts. It allocates margin instead. To state
the capital that each unit uses, it allocates each layer of the asset range by
the units' shares of that layer's margin, and adds the layers. In prospicio
this is `allocation="lifted"`:

```python
BOOK_Q = {  # Table 9.8, capital by unit (A, B, C)
    "ccoc": (4.874, 5.288, 36.273),
    "ph": (7.074, 6.565, 32.796),
    "wang": (8.411, 8.691, 29.333),
    "dual": (8.873, 9.143, 28.419),
    "tvar": (9.034, 9.247, 28.154),
}
got = {
    family: [u.capital for u in insco.price(d, assets=100, allocation="lifted").allocated]
    for family, d in distortions.items()
}
(GT(with_book(got, BOOK_Q, ["A", "B", "C"]), rowname_col="distortion")
 .tab_header("InsCo: capital by unit, lifted allocation", "Table 9.8; the assets are 100")
 .fmt_number(["A", "B", "C"], decimals=3))
```

```text
                A      B       C
ccoc        2.261  1.478  42.696
ccoc, book  4.874  5.288  36.273
ph          6.481  5.597  34.356
ph, book    7.074  6.565  32.796
wang        8.411  8.691  29.333
wang, book  8.411  8.691  29.333
dual        8.873  9.143  28.419
dual, book  8.873  9.143  28.419
tvar        9.034  9.247  28.154
tvar, book  9.034  9.247  28.154
```

Wang, dual power and TVaR agree with the monograph to three decimals. **CCoC
and PH do not.** For CCoC, prospicio gives the same split as the linear
allocation. The premiums and margins are the same as the monograph's. Only the split
of capital between units differs.

The cause is the `aggregate` version. The monograph's code ran on 0.24.0, and
0.30.1 prints its values. 1.0.1 gives prospicio's, and CI checks prospicio
against 1.0.1 on InsCo at 1e-10, for both allocations and all five
distortions. The monograph's own argument also gives prospicio's CCoC value:
every layer of capital earns 15%, so a unit's capital is its margin over 0.15
(unit A: 0.339 / 0.15 = 2.261). Use the premiums and margins, which agree in
all versions.

## 6. The reinsurance decision

Chapter 5 asks how a reinsurance cover changes the price of the book. The
example is an aggregate stop loss of 35 excess of 65: the reinsurer pays the
amount by which the year's total exceeds 65, up to 35. On these ten years it
pays in the year of 100 only, so the recovery is 35 with probability 0.1, a
mean of 3.5.

Make the net loss and the recovery two units of a new portfolio. It prices the
cover and shows what the retained book needs:

```python
def stop_loss(total, attach=65, limit=35):
    return min(max(total - attach, 0), limit)

split = [[t - stop_loss(t), stop_loss(t)] for t in totals]
reinsured = Portfolio(["Net", "Ceded"], split)
print("means", reinsured.expected(), " (book: net 43.1, ceded 3.5)")
```

```text
means [43.1, 3.5]  (book: net 43.1, ceded 3.5)
```

Rows alternate prospicio and Table 9.10:

```python
BOOK_RE = {  # Table 9.10, premium and margin of ceded and net, (ceded, net)
    "ccoc": ((7.609, 4.109), (45.957, 2.857)),
    "ph": ((6.662, 3.162), (46.903, 3.803)),
    "wang": ((6.087, 2.587), (47.478, 4.378)),
    "dual": ((5.415, 1.915), (48.151, 5.051)),
    "tvar": ((4.803, 1.303), (48.762, 5.662)),
}
book = {family: [*ceded, *net] for family, (ceded, net) in BOOK_RE.items()}
got = {}
for family, d in distortions.items():
    ceded, net = reinsured.price(d, assets=100).allocated[::-1]
    got[family] = [ceded.premium, ceded.premium - ceded.loss, net.premium, net.premium - net.loss]
columns = ["ceded premium", "ceded margin", "net premium", "net margin"]
(GT(with_book(got, book, columns), rowname_col="distortion")
 .tab_header("Stop loss 35 xs 65: ceded and net price", "Table 9.10")
 .fmt_number(columns, decimals=3))
```

```text
            ceded premium  ceded margin  net premium  net margin
ccoc                7.609         4.109       45.957       2.857
ccoc, book          7.609         4.109       45.957       2.857
ph                  6.662         3.162       46.903       3.803
ph, book            6.662         3.162       46.903       3.803
wang                6.087         2.587       47.478       4.378
wang, book          6.087         2.587       47.478       4.378
dual                5.415         1.915       48.151       5.051
dual, book          5.415         1.915       48.151       5.051
tvar                4.803         1.303       48.762       5.662
tvar, book          4.803         1.303       48.762       5.662
```

The cover protects only against the worst year, so its risk load is all tail
and its price depends strongly on the distortion. The two margins add up to the
book's 6.965 under every distortion, so a dearer cover leaves less margin to
the retained book. Chapter 6 asks which distortion is right for the reinsurer's
price or the buyer's willingness to pay. The table gives the answer for each.

## 7. A more realistic example

The last section of chapter 9 builds a book from frequency and severity. It
has three units, Auto, GL and WC. Each has a lognormal severity with a
per-claim limit and a gamma-mixed Poisson count, compounded by FFT.
Iman-Conover rank correlation then ties the units together. The monograph
prices the book at a 99.5% standard and a 15% return. The units are:

```python
from prospicio.aggregate import fft
from prospicio.distributions import Count, Grid, Lognormal


def compound_unit(loss, limit, mean, cv, mixing_cv, step, points):
    """A unit: expected loss, a per-claim limit, a lognormal, a gamma mixed count."""
    severity, _ = Grid.local_moment(Lognormal.from_mean_cv(mean, cv), step, points)
    claim, _ = severity.map(lambda x: min(x, limit))
    claims = loss / claim.mean()
    annual, report = fft(Count.mixed_poisson(claims, mixing_cv), claim, points)
    return claims, claim.mean(), annual, report


UNITS = {
    "Auto": (5000.0, 5000.0, 50.0, 2.0, 0.2, 0.25, 2**16),
    "GL": (3000.0, 2500.0, 75.0, 3.0, 0.3, 0.25, 2**16),
    "WC": (7000.0, 25000.0, 5.0, 10.0, 0.25, 0.25, 2**17),
}
annual = {}
rows = []
for name, args in UNITS.items():
    claims, claim_mean, annual[name], report = compound_unit(*args)
    a = annual[name]
    rows.append([name, claims, claim_mean, a.mean(), a.variance() ** 0.5 / a.mean(), f"{report.tail_mass:.1e}"])
columns = ["unit", "claims", "claim mean", "agg mean", "agg cv", "tail mass"]
(GT(pd.DataFrame(rows, columns=columns), rowname_col="unit")
 .tab_header("The three compound units", "Table 9.11")
 .fmt_number("claims", decimals=1, use_seps=False)
 .fmt_number("claim mean", decimals=3)
 .fmt_number("agg mean", decimals=1, use_seps=False)
 .fmt_number("agg cv", decimals=3))
```

```text
      claims  claim mean  agg mean  agg cv  tail mass
Auto   100.0      49.982    5000.0   0.298    2.3e-06
GL      41.0      73.156    3000.0   0.502    4.4e-07
WC    1401.1       4.996    6999.7   0.349    1.6e-04
```

Table 9.11 prints the same values (41.008 claims for GL, and 4.945 for WC's
claim mean on its grid). Unit WC drives the tail, with a thick-tailed severity and a high limit.

Iman-Conover pairs the units' draws so that their rank correlations match a
target, and leaves each unit's distribution unchanged. Three steps:

1. Draw each unit at evenly spaced quantiles.
2. Tie the draws with the monograph's matrix (Table 9.12).
3. Build a portfolio from the result.

```python
from prospicio.distributions import PredictiveDistribution
from prospicio.risk import iman_conover

TARGET = [[1.0, 0.5, 0.4], [0.5, 1.0, 0.1], [0.4, 0.1, 1.0]]


def correlated_portfolio(n_sims, seed):
    draws = {k: [g.quantile((i + 0.5) / n_sims) for i in range(n_sims)] for k, g in annual.items()}
    rows = [[draws[k][i] for k in UNITS] for i in range(n_sims)]
    keys = [(k,) for k in UNITS]
    joint = iman_conover(PredictiveDistribution(["unit"], keys, rows), TARGET, seed)
    return Portfolio.from_predictive(joint)


book = correlated_portfolio(50_000, 4817)
print("units", book.units, " means", [round(m, 1) for m in book.expected()])
```

```text
units ['Auto', 'GL', 'WC']  means [5000.0, 3000.0, 6999.7]
```

Now calibrate the five distortions to a 15% return at the 99.5% standard and
price each unit:

```python
BOOK_PARAMS = {"ccoc": 0.150, "ph": 0.654, "wang": 0.437, "dual": 1.756, "tvar": 0.299}
priced_book, params = {}, {}
for family in FAMILIES:
    d = book.calibrate(family, p=0.995, return_on_capital=0.15)
    priced_book[family] = book.price(d, p=0.995)
    ((_, params[family]),) = d.parameters.items()
print(f"assets at 99.5%: {book.assets(0.995):,.0f}   (book 28,898)")
```

```text
assets at 99.5%: 28,448   (book 28,898)
```

The table is Tables 9.16 (parameters) and 9.17 (premiums):

```python
rows = [
    [family, params[family], BOOK_PARAMS[family],
     *[u.premium for u in priced_book[family].allocated], priced_book[family].total.premium]
    for family in FAMILIES
]
columns = ["distortion", "parameter", "book parameter", "Auto", "GL", "WC", "total"]
(GT(pd.DataFrame(rows, columns=columns), rowname_col="distortion")
 .tab_header("The correlated book at 99.5%, 15% return", "Premium by unit")
 .fmt_number(["parameter", "book parameter"], decimals=3)
 .fmt_number(["Auto", "GL", "WC"], decimals=1, use_seps=False)
 .fmt_number("total", decimals=1))
```

```text
      parameter  book parameter    Auto      GL      WC     total
ccoc      0.150           0.150  5351.0  3230.1  8154.6  16,735.8
ph        0.654           0.654  5483.4  3377.8  7874.6  16,735.8
wang      0.434           0.437  5511.0  3405.1  7819.6  16,735.8
dual      1.740           1.756  5529.0  3421.1  7785.6  16,735.8
tvar      0.291           0.299  5536.0  3423.9  7776.0  16,735.8
```

The parameters are within about 0.01 of the monograph's, and the total premium
of about 16,740 is close to its 16,800.3. Exact agreement is not expected: the
monograph's 1,000 simulations put its 99.5% quantile on five points, and the
draws here are 50,000 and evenly spaced. The table repeats the CCoC calculation
on other sample sizes:

```python
rows = []
for n_sims, seed in ((1_000, 11), (1_000, 12), (10_000, 11), (50_000, 11)):
    pf = correlated_portfolio(n_sims, seed)
    d = pf.calibrate("ccoc", p=0.995, return_on_capital=0.15)
    p = pf.price(d, p=0.995)
    rows.append([f"{n_sims}, seed {seed}", p.total.assets, p.total.premium])
(GT(pd.DataFrame(rows, columns=["simulations", "assets", "premium"]), rowname_col="simulations")
 .tab_header("CCoC price of the correlated book by sample size", "99.5% standard, 15% return")
 .fmt_number("assets", decimals=0)
 .fmt_number("premium", decimals=1))
```

```text
                assets   premium
1000, seed 11   27,850  16,657.6
1000, seed 12   29,368  16,855.1
10000, seed 11  28,692  16,771.1
50000, seed 11  28,520  16,744.9
```

The premium moves by one or two percent between two samples of 1,000. Do not
quote a small sample to four digits; use more simulations and a capital
standard that you can defend.

## 8. Related guides

- [Chapters 1, 2 and 7](capital-modeling-monograph-15-practice.md): the
  charter of a capital model, business operations, reinsurance, capital
  structure, pro formas and the evaluation of models.
- [Chapters 3 and 4](capital-modeling-monograph-15-theory.md) and
  [chapters 5 and 6](capital-modeling-monograph-15-applications.md), table by
  table: capital adequacy, pricing and allocation, applications and the choice
  of a distortion.
- [Chapter 8](capital-modeling-monograph-15-iman-conover.md): uncertainty,
  Iman-Conover, capital by layer and Bodoff's allocation (Table 8.12 from
  `Portfolio.bodoff`).
- [Risk loads and distortions](risk-loads-and-distortions.md): which
  distortion to use, and how to calibrate it to a quote.
- [Aggregate distributions](aggregate-distributions.md): the compound
  distributions of section 7.
- [prospicio for aggregate users](prospicio-for-aggregate-users.md): the
  package behind the monograph's code.
