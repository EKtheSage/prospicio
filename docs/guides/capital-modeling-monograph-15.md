# Capital modeling and portfolio pricing: CAS Monograph 15 reproduced

CAS Monograph 15, *Introduction to Capital Modeling and Portfolio Management*
(John Major and Stephen Mildenhall, 2026), shows how an insurer can price its
risk and split the price over its business units, from a distribution of
outcomes alone. Its worked examples run in Mildenhall's Python package
`aggregate`. This guide runs them in prospicio and prints both sets of numbers
next to each other, so you can see what the method does, what each step costs in
code, and where the two programs agree or differ.

The text of the monograph is the authors'; what follows is an explanation in new
words with the numbers recomputed. Read the monograph for the reasoning (its
chapters 3 to 6 and 8) and this guide for the mechanics (its chapter 9). Table
numbers below are the monograph's. Money is in the units of its examples.

## 1. The example company

The monograph's running example, InsCo, has three units, A, B and C, and ten
equally likely simulated years. Each row is a year and each cell the unit's loss
in it:

```python
from prospicio.distributions import Sampled
from prospicio.pricing import Portfolio
from prospicio.risk import Distortion

ROWS = [
    [5, 20, 11], [7, 33, 0], [15, 13, 0], [15, 7, 0], [13, 20, 7],
    [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64],
]
insco = Portfolio(["A", "B", "C"], ROWS)
totals = [sum(row) for row in ROWS]

for name, losses in zip(["A", "B", "C", "Total"], [*zip(*ROWS), totals]):
    s = Sampled([float(x) for x in losses])
    sd = (sum((x - s.mean()) ** 2 for x in losses) / len(losses)) ** 0.5
    print(f"{name:6s} mean {s.mean():6.2f}  cv {sd / s.mean():.3f}")
```

```text
A      mean  13.40  cv 0.453
B      mean  18.30  cv 0.412
C      mean  14.90  cv 1.324
Total  mean  46.60  cv 0.455
```

This is Table 9.1's last two rows: means of 13.4, 18.3, 14.9 and 46.6, and CVs of
0.453, 0.412, 1.324 and 0.455. Unit C is the volatile one: a small mean, and a
64 in the worst year. A `Portfolio` is built from the rows directly, so there is
no frequency-severity model or grid here, and no approximation: every number
below is computed on the ten years as they are.

Pricing works on the distinct totals, not on the years. Four of the years
(the second, fifth, sixth and seventh) all total 40, so the portfolio keeps one scenario at 40 with probability 0.4 and
each unit's average loss given that total, which the monograph calls its
conditional expectation, `kappa`:

```python
print("totals ", insco.totals)
print("probs  ", [round(p, 1) for p in insco.probs])
for unit in ("A", "B", "C"):
    print(f"kappa {unit}", [round(k, 2) for k in insco.kappa(unit)])
```

```text
totals  [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0]
probs   [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1]
kappa A [15.0, 15.0, 5.0, 10.0, 26.0, 17.0, 16.0]
kappa B [7.0, 13.0, 20.0, 24.0, 19.0, 8.0, 20.0]
kappa C [0.0, 0.0, 11.0, 6.0, 10.0, 40.0, 64.0]
```

At a total of 40, A is expected to lose 10, B 24 and C 6, the row of Table 9.4.
The conditional expectations always add up to the total, so splitting the
portfolio's price by them can never create or lose money.

## 2. The capital standard

A price needs a capital standard: how many assets stand behind the book. The
monograph's chapter 3 introduces value at risk (VaR, a quantile) and tail value
at risk (TVaR, the average above it). The `aggregate` package, and prospicio,
read a standard of "p" as assets equal to the p quantile of the total:

```python
total = Sampled([float(t) for t in totals])
print(f"{'p':>5s} {'VaR':>6s} {'TVaR':>8s}  assets at p")
for p in (0.8, 0.85, 0.9, 1.0):
    print(f"{p:5.2f} {total.quantile(p):6.0f} {total.tvar(p):8.3f}  {insco.assets(p):.0f}")
```

```text
    p    VaR     TVaR  assets at p
 0.80     55   82.500  55
 0.85     65   88.333  65
 0.90     65  100.000  65
 1.00    100  100.000  100
```

These are Table 9.5. With ten years the standard of 100% is the worst year, and
TVaR reaches 100 from 90% up, which is why the monograph's main example speaks
of full capitalization: assets of 100, the largest loss. With only ten
scenarios a 99.5% standard would also be the worst year, so the example uses
the cleanest one.

## 3. Calibrating five distortions

The monograph prices by a **distortion** (a spectral risk measure): a function
`g` that raises the probability of bad outcomes before taking the expected
loss. A distortion is chosen from a family and then calibrated, here to one
requirement: the whole book earns a 15% return on capital, with assets of 100.
The expected loss is 46.6, so the premium `P` and capital `Q = 100 - P` must
satisfy `P - 46.6 = 0.15 Q`, which gives `P = 53.565`. Each of five families, a
constant cost of capital (CCoC), proportional hazard (PH), Wang, dual power and
TVaR, has one parameter, and calibration finds the value that hits that premium:

```python
FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
distortions = {
    family: insco.calibrate(family, assets=100, return_on_capital=0.15)
    for family in FAMILIES
}
for family, d in distortions.items():
    ((name, value),) = d.parameters.items()
    print(f"{family:5s} {name:7s} {value:.3f}")
```

```text
ccoc  r       0.150
ph    rho     0.720
wang  lambda  0.343
dual  beta    1.595
tvar  p       0.271
```

Table 9.3 prints 0.150, 0.720, 0.343, 1.595 and 0.271 for these parameters, and
the values agree. Mind the names: in `aggregate` the CCoC distortion is built
from a risk discount `d = 0.15 / 1.15`, in prospicio from the cost of capital
`r = 0.15`, and `Distortion.ccoc(0.15)` is the same function.

## 4. The price and its allocation to units

With a distortion calibrated, `price` returns the premium of the loss capped at
the assets, and the **natural allocation** of it to the units: each unit pays
the distorted average of its own loss in the bad years, so a unit that loses
most when the total is large pays more. With `allocation="linear"` the assets
are also split by expected shares above the asset level.

```python
BOOK_LINEAR = {  # Table 9.6 for CCoC: premium by unit
    "A": 13.739, "B": 18.522, "C": 21.304,
}
priced = insco.price(distortions["ccoc"], assets=100, allocation="linear")
print(f"{'':6s} {'L':>7s} {'a':>7s} {'Q':>7s} {'P':>7s} {'M':>7s} {'COC':>6s}")
for name, u in zip(priced.units + ["Total"], [*priced.allocated, priced.total]):
    margin = u.premium - u.loss
    print(f"{name:6s} {u.loss:7.3f} {u.assets:7.3f} {u.capital:7.3f} {u.premium:7.3f} "
          f"{margin:7.3f} {margin / u.capital:6.3f}")
```

```text
             L       a       Q       P       M    COC
A       13.400  16.000   2.261  13.739   0.339  0.150
B       18.300  20.000   1.478  18.522   0.222  0.150
C       14.900  64.000  42.696  21.304   6.404  0.150
Total   46.600 100.000  46.435  53.565   6.965  0.150
```

This is Table 9.6, the "industry standard" CCoC pricing: every unit earns 15% on
the capital allocated to it, because that is what the capital was allocated to
mean. Unit C takes 64 of the 100 of assets and pays 21.3 of the premium for an
expected loss of 14.9, so most of the margin (6.4 of 7.0) is C's. The units'
premiums add up to the 53.565 of the book.

The same premium holds for every distortion in the five, because each was
calibrated to it; the distortions differ in who pays. The monograph's Table 9.8
shows the premiums by unit, and they agree here:

```python
BOOK_P = {  # Table 9.8, premium by unit (A, B, C)
    "ccoc": (13.739, 18.522, 21.304),
    "ph": (14.060, 18.349, 21.156),
    "wang": (14.109, 18.637, 20.819),
    "dual": (14.127, 19.117, 20.322),
    "tvar": (13.783, 20.412, 19.371),
}
print(f"{'':5s} {'A':>20s} {'B':>20s} {'C':>20s}")
for family, d in distortions.items():
    got = [u.premium for u in insco.price(d, assets=100).allocated]
    cells = [f"{g:7.3f} (book {b:6.3f})" for g, b in zip(got, BOOK_P[family])]
    print(f"{family:5s} " + " ".join(cells))
```

```text
                         A                    B                    C
ccoc   13.739 (book 13.739)  18.522 (book 18.522)  21.304 (book 21.304)
ph     14.060 (book 14.060)  18.349 (book 18.349)  21.156 (book 21.156)
wang   14.109 (book 14.109)  18.637 (book 18.637)  20.819 (book 20.819)
dual   14.127 (book 14.127)  19.117 (book 19.117)  20.322 (book 20.322)
tvar   13.783 (book 13.783)  20.412 (book 20.412)  19.371 (book 19.371)
```

All fifteen premiums match to the three decimals printed. The pattern across the
rows is the substance: the CCoC prices unit C at 21.3 and unit B at 18.5, the
TVaR prices B at 20.4 and C at 19.4. TVaR charges for the units' losses in the
worst years only, where B's 20 is a larger share of the total than its average
share; CCoC charges for the volume of capital each unit consumes, and C uses 64
of the 100. Neither is the right
answer; the distortion is a statement about what you are paid to bear (see
[Risk loads and distortions](risk-loads-and-distortions.md)).

## 5. Capital by unit, if you must

The monograph's section 8.3 advises against allocating capital, because
layers of capital cost different amounts, and allocates margin instead. If you
must say how much capital each unit uses, it describes a method that allocates
each layer of the asset range by the units' shares of that layer's margin, and
sums over layers. In prospicio this is `allocation="lifted"`:

```python
BOOK_Q = {  # Table 9.8, capital by unit (A, B, C)
    "ccoc": (4.874, 5.288, 36.273),
    "ph": (7.074, 6.565, 32.796),
    "wang": (8.411, 8.691, 29.333),
    "dual": (8.873, 9.143, 28.419),
    "tvar": (9.034, 9.247, 28.154),
}
print(f"{'':5s} {'A':>20s} {'B':>20s} {'C':>20s}")
for family, d in distortions.items():
    got = [u.capital for u in insco.price(d, assets=100, allocation="lifted").allocated]
    cells = [f"{g:7.3f} (book {b:6.3f})" for g, b in zip(got, BOOK_Q[family])]
    print(f"{family:5s} " + " ".join(cells))
```

```text
                         A                    B                    C
ccoc    2.261 (book  4.874)   1.478 (book  5.288)  42.696 (book 36.273)
ph      6.481 (book  7.074)   5.597 (book  6.565)  34.356 (book 32.796)
wang    8.411 (book  8.411)   8.691 (book  8.691)  29.333 (book 29.333)
dual    8.873 (book  8.873)   9.143 (book  9.143)  28.419 (book 28.419)
tvar    9.034 (book  9.034)   9.247 (book  9.247)  28.154 (book 28.154)
```

Wang, dual power and TVaR agree with the monograph to three decimals. **CCoC and
PH do not.** The monograph prints a CCoC capital of 4.874, 5.288 and 36.273, and
`aggregate` 0.30.1 returns exactly those; prospicio returns 2.261, 1.478 and
42.696, the same as the linear allocation, and a PH capital that is 6.481, 5.597
and 34.356 against the printed 7.074, 6.565 and 32.796. Premiums and margins are
identical in both programs; only the split of capital between units differs.

The difference is between versions of `aggregate`, not between prospicio and
`aggregate`. The monograph's code ran on `aggregate` 0.24.0, and 0.30.1 (run
here) prints the monograph's numbers. prospicio's natural allocation is tested
in CI against reference values computed with `aggregate` 1.0.1 on this very
company, at 1e-10, for both allocations and all five distortions, and those
reference values for CCoC and PH are prospicio's: 2.261, 1.478 and 42.696 for
CCoC. So the lifted capital for these two families changed between 0.30.1 and
1.0.1. The CCoC value is also what the monograph's own argument implies: under a
CCoC every layer of capital earns the same 15%, so a unit's capital is its
margin divided by 0.15, and unit A's margin of 0.339 gives 2.261. Use the
premiums and margins, which agree everywhere, and treat capital by unit under
CCoC and PH as version dependent; the monograph advises against allocating
capital for much the same reason.

## 6. The reinsurance decision

Chapter 5 asks what a reinsurance cover does to the price of the book. The
monograph's example is an aggregate stop loss of 35 excess of 65: the reinsurer
pays whatever the year's total exceeds 65, up to 35. On these ten years it pays
in one year only, the 100, which is 35 with probability 0.1, a mean of 3.5.
Treating the cover's recovery and the net loss as two units of a new portfolio
prices the cover, and shows what the retained book then needs:

```python
def stop_loss(total, attach=65, limit=35):
    return min(max(total - attach, 0), limit)

split = [[t - stop_loss(t), stop_loss(t)] for t in totals]
reinsured = Portfolio(["Net", "Ceded"], split)
print("means", reinsured.expected(), " (book: net 43.1, ceded 3.5)")

BOOK_RE = {  # Table 9.10, premium and margin of ceded and net, (ceded, net)
    "ccoc": ((7.609, 4.109), (45.957, 2.857)),
    "ph": ((6.662, 3.162), (46.903, 3.803)),
    "wang": ((6.087, 2.587), (47.478, 4.378)),
    "dual": ((5.415, 1.915), (48.151, 5.051)),
    "tvar": ((4.803, 1.303), (48.762, 5.662)),
}
print(f"{'':5s} {'ceded P':>8s} {'M':>7s} {'net P':>8s} {'M':>7s}")
for family, d in distortions.items():
    r = reinsured.price(d, assets=100).allocated
    (cp, cm), (np_, nm) = BOOK_RE[family]
    ceded, net = r[1], r[0]
    print(f"{family:5s} {ceded.premium:8.3f} {ceded.premium - ceded.loss:7.3f} "
          f"{net.premium:8.3f} {net.premium - net.loss:7.3f}   book {cp} {cm} {np_} {nm}")
```

```text
means [43.1, 3.5]  (book: net 43.1, ceded 3.5)
       ceded P       M    net P       M
ccoc     7.609   4.109   45.957   2.857   book 7.609 4.109 45.957 2.857
ph       6.662   3.162   46.903   3.803   book 6.662 3.162 46.903 3.803
wang     6.087   2.587   47.478   4.378   book 6.087 2.587 47.478 4.378
dual     5.415   1.915   48.151   5.051   book 5.415 1.915 48.151 5.051
tvar     4.803   1.303   48.762   5.662   book 4.803 1.303 48.762 5.662
```

The reinsurer's price depends strongly on the distortion. The cover protects
only against the one worst year, so the risk loads are in the tail: a CCoC
prices it at 7.609 (a margin of 4.109, more than the expected loss of 3.5), and a dual power at 5.415. Which one is the right yardstick for the
reinsurer's price, or for the buyer's willingness to pay, is the question of the
monograph's chapter 6. The method's contribution is that the answer is on one
page for every yardstick. The two margins always add up to the book's 6.965, so a
dearer cover (CCoC) leaves the retained book less margin (2.857) than a cheaper
one (the dual power leaves 5.051).

## 7. A more realistic example

The last section of the monograph's chapter 9 builds a book from frequency and
severity: three units, Auto, GL and WC, with lognormal severities limited per
claim and a mixed-Poisson (gamma-mixed) count each, compounded by FFT, then tied
together by Iman-Conover rank correlation and priced at a 99.5% standard and a
15% return. The units are:

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
print(f"{'':5s} {'claims':>8s} {'claim mean':>11s} {'agg mean':>9s} {'agg cv':>7s} {'tail mass':>10s}")
for name, args in UNITS.items():
    claims, claim_mean, annual[name], report = compound_unit(*args)
    a = annual[name]
    print(f"{name:5s} {claims:8.1f} {claim_mean:11.3f} {a.mean():9.1f} "
          f"{a.variance() ** 0.5 / a.mean():7.3f} {report.tail_mass:10.1e}")
```

```text
        claims  claim mean  agg mean  agg cv  tail mass
Auto     100.0      49.982    5000.0   0.298    2.3e-06
GL        41.0      73.156    3000.0   0.502    4.4e-07
WC      1401.1       4.996    6999.7   0.349    1.6e-04
```

The monograph prints (Table 9.11) 100.0, 41.008 and 1,401.1 claims, mean claims
of 49.982, 73.156 and 4.996 (4.945 as achieved on its grid), and aggregate CVs of
0.298, 0.502 and 0.349. The numbers agree. Unit WC
drives the tail, with a thick severity and a high limit.

Iman-Conover pairs the units' draws so that their rank correlations match a
target without changing any unit's distribution. Draw each unit at evenly spaced
quantiles, tie them with the monograph's matrix (Table 9.12), and build a
portfolio from the result:

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

Now calibrate the five distortions to a 15% return at the 99.5% standard
(Table 9.16) and price each unit (Table 9.17):

```python
BOOK_PARAMS = {"ccoc": 0.150, "ph": 0.654, "wang": 0.437, "dual": 1.756, "tvar": 0.299}
priced_book = {}
print(f"assets at 99.5%: {book.assets(0.995):,.0f}   (book 28,898)")
for family in FAMILIES:
    d = book.calibrate(family, p=0.995, return_on_capital=0.15)
    priced_book[family] = book.price(d, p=0.995)
    ((_, value),) = d.parameters.items()
    cells = " ".join(f"{u.premium:8.1f}" for u in priced_book[family].allocated)
    print(f"{family:5s} parameter {value:.3f} (book {BOOK_PARAMS[family]:.3f})   "
          f"premium Auto, GL, WC: {cells}   total {priced_book[family].total.premium:,.1f}")
```

```text
assets at 99.5%: 28,448   (book 28,898)
ccoc  parameter 0.150 (book 0.150)   premium Auto, GL, WC:   5351.0   3230.1   8154.6   total 16,735.8
ph    parameter 0.654 (book 0.654)   premium Auto, GL, WC:   5483.4   3377.8   7874.6   total 16,735.8
wang  parameter 0.434 (book 0.437)   premium Auto, GL, WC:   5511.0   3405.1   7819.6   total 16,735.8
dual  parameter 1.740 (book 1.756)   premium Auto, GL, WC:   5529.0   3421.1   7785.6   total 16,735.8
tvar  parameter 0.291 (book 0.299)   premium Auto, GL, WC:   5536.0   3423.9   7776.0   total 16,735.8
```

The parameters are within about 0.01 of Table 9.16 (the PH parameter is equal;
the TVaR level is 0.291 against 0.299), and the total premium of about 16,740
is close to the monograph's 16,800.3. They are not expected to agree exactly. The monograph's sample has 1,000 simulations, so its
99.5% quantile rests on five points, while the draws here are 50,000 and evenly
spaced. How much a sample of 1,000 moves the answer is easy to see by running
the same calculation on samples of different sizes:

```python
for n_sims, seed in ((1_000, 11), (1_000, 12), (10_000, 11), (50_000, 11)):
    pf = correlated_portfolio(n_sims, seed)
    d = pf.calibrate("ccoc", p=0.995, return_on_capital=0.15)
    p = pf.price(d, p=0.995)
    print(f"{n_sims:6d} sims, seed {seed}: assets {p.total.assets:9,.0f}  premium {p.total.premium:9,.1f}")
```

```text
  1000 sims, seed 11: assets    27,850  premium  16,657.6
  1000 sims, seed 12: assets    29,368  premium  16,855.1
 10000 sims, seed 11: assets    28,692  premium  16,771.1
 50000 sims, seed 11: assets    28,520  premium  16,744.9
```

The premium moves by a percent or two between two samples of 1,000 and settles
as the sample grows, so numbers from a small sample are not worth quoting to
four digits. For more units, more events and a real model, the monograph's
advice is the same: more simulations, and a capital standard you can defend.

## 8. What this guide leaves out

- **Chapters 1, 2 and 7** (the charter of a capital model, business
  operations, reinsurance, capital structure, pro formas, evaluating models):
  see [chapters 1, 2 and 7](capital-modeling-monograph-15-practice.md).
- **Chapters 3 to 6** (capital adequacy, pricing and allocation, the
  applications and the choice of a distortion): reproduced table by table in
  [chapters 3 and 4](capital-modeling-monograph-15-theory.md) and
  [chapters 5 and 6](capital-modeling-monograph-15-applications.md).
- **Chapter 8** (uncertainty, Iman-Conover, capital by layer and Bodoff's
  allocation, Table 8.12 reproduced by `Portfolio.bodoff`): see
  [chapter 8](capital-modeling-monograph-15-iman-conover.md).
- **Capital by unit under CCoC and PH** (section 5): version dependent, see the
  note there.

Where to go next: [Risk loads and distortions](risk-loads-and-distortions.md) for
which distortion to use and how to calibrate it to a quote,
[Aggregate distributions](aggregate-distributions.md) for the compound
distributions of section 7, and
[prospicio for aggregate users](prospicio-for-aggregate-users.md) for the
package behind the monograph's code.
