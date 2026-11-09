# Price allocation in practice: the case-study tables and the reinsurance decision

<!-- site: execute -->

This guide continues
[Price allocation in *Pricing Insurance Risk*](pir-price-allocation.md) on a
realistic book, the Cat/Non-Cat case study of Mildenhall and Major's *Pricing
Insurance Risk* (Wiley, 2022). Section 2 allocates it by the classical recipes
of chapter 13 (Table 13.3). Section 3 uses the natural allocation and Bodoff's
percentile layer of capital of chapter 15 (Tables 15.36 and 15.39). Section 4
is the chapter 19 test of whether a reinsurance cover is worth buying. Each
number that the book prints is shown next to the number from prospicio.
[Pricing Insurance Risk: the case studies](pricing-insurance-risk-case-studies.md)
compares the same case studies with `aggregate` 0.30.1.

## 1. The book and its capital standard

Cat/Non-Cat has two independent units with a total mean of 100: Non-Cat is a
gamma (mean 80, CV 0.15) and Cat is a lognormal (mean 20, CV 1). As in the case
studies guide, the code puts each unit on the grid of `aggregate` (step 1/64,
2^16 points), combines the units by FFT, sets the assets to the 99.9% quantile
of the total and calibrates five distortions to a 10% return on capital:

```python
from prospicio.distributions import Gamma, Grid, Lognormal
from prospicio.pricing import Portfolio
from prospicio.risk import Distortion

FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
STEP, POINTS = 1 / 64, 2**16


def discretize(severity, step, points):
    """A severity on a grid, its mass above the top spread over the grid as aggregate does."""
    probs = Grid.rounding(severity, step, 2 * points)[0].probs[:points]
    total = sum(probs)
    return Grid(step, [q / total for q in probs])


cat = discretize(Lognormal.from_mean_cv(20.0, 1.0), STEP, POINTS)
noncat = discretize(Gamma.from_mean_cv(80.0, 0.15), STEP, POINTS)
book = Portfolio.from_independent(["Cat", "NonCat"], [cat, noncat])
a = book.assets(0.999)
capped = Portfolio(["total"], [[min(x, a)] for x in book.totals], book.probs)
dists = {f: capped.calibrate(f, assets=a, return_on_capital=0.10) for f in FAMILIES}
loss = [u.loss for u in book.price(dists["ccoc"], p=0.999).allocated]
print(f"assets {a:.2f}  (book 267.2)")
print(f"expected loss within the assets: Cat {loss[0]:.2f}, Non-Cat {loss[1]:.2f}  (book 19.96, 79.99)")
```

```text
assets 267.20  (book 267.2)
expected loss within the assets: Cat 19.96, Non-Cat 79.99  (book 19.96, 79.99)
```

Calibrating on the capped total gives the same parameters as on the full
portfolio.

## 2. Allocated CCoC pricing (Table 13.3)

Chapter 13 allocates the 267.2 of assets to the units and charges a 10% return
on the capital of each unit. The margin of a unit is `0.1 / 1.1` times its
assets minus its expected loss, so the result depends on the asset allocation.
The chapter uses nine methods:

- **expected loss**: assets in proportion to expected loss.
- **scaled VaR, TVaR and EPD**: the stand-alone VaR, TVaR or EPD-based assets of
  each unit, scaled to add up to 267.2. For TVaR and EPD, the level is the one
  at which the measure of the **total** gives 267.2.
- **equal risk VaR, TVaR and EPD**: the one level at which the stand-alone
  measures of the units add up to 267.2.
- **coTVaR**: the average loss of each unit in the years beyond the level at
  which the TVaR of the total is 267.2.
- **covariance**: `k` such that `E[X] + k Var(X)` is 267.2. Unit `i` gets
  `E[X_i] + k Cov(X_i, X)`.

A recipe that must find a level uses a bisection:

```python
def solve(f, lo, hi, steps=100):
    """The root of an increasing function f between lo and hi, by bisection."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return (lo + hi) / 2


probs = [0.0] * (round(book.totals[-1] / STEP) + 1)
for x, p in zip(book.totals, book.probs):
    probs[round(x / STEP)] += p
total = Grid(STEP, probs)
units = (cat, noncat)


def epd_assets(g, epd):
    """The assets at which a unit's expected shortfall is epd times its mean."""
    return solve(lambda x: epd - g.stop_loss(x) / g.mean(), 0.0, POINTS * STEP)


p_tvar = solve(lambda p: total.tvar(p) - a, 0.5, 0.999)
epd_total = total.stop_loss(a) / total.mean()
p_var = solve(lambda p: cat.quantile(p) + noncat.quantile(p) - a, 0.9, 0.99999)
p_eq_tvar = solve(lambda p: cat.tvar(p) + noncat.tvar(p) - a, 0.5, 0.99999)
log_epd = solve(lambda e: a - sum(epd_assets(g, 10**e) for g in units), -14.0, 0.0, 60)
k = (a - total.mean()) / total.variance()


def scaled(values):
    return [a * v / sum(values) for v in values]


ASSETS = {
    "expected loss": scaled(loss),
    "scaled EPD": scaled([epd_assets(g, epd_total) for g in units]),
    "scaled TVaR": scaled([g.tvar(p_tvar) for g in units]),
    "scaled VaR": scaled([g.quantile(0.999) for g in units]),
    "equal risk EPD": [epd_assets(g, 10**log_epd) for g in units],
    "equal risk TVaR": [g.tvar(p_eq_tvar) for g in units],
    "equal risk VaR": [g.quantile(p_var) for g in units],
    "coTVaR": [u.premium for u in book.price(Distortion.tvar(p_tvar)).allocated],
    "covariance": [g.mean() + k * g.variance() for g in units],
}
BOOK_133 = {  # Table 13.3, gross margin: (Cat, Non-Cat)
    "expected loss": (3.04, 12.17), "scaled EPD": (15.68, -0.471), "scaled TVaR": (12.85, 2.36),
    "scaled VaR": (12.82, 2.38), "equal risk EPD": (13.33, 1.87), "equal risk TVaR": (11.78, 3.42),
    "equal risk VaR": (11.74, 3.47), "coTVaR": (14.84, 0.37), "covariance": (11.18, 4.03),
}
print(f"levels: TVaR {p_tvar:.4f}, equal risk VaR {p_var:.4f}, equal risk TVaR {p_eq_tvar:.4f}")
```

```text
levels: TVaR 0.9970, equal risk VaR 0.9977, equal risk TVaR 0.9931
```

```python
import pandas as pd
from great_tables import GT

d = 0.1 / 1.1
rows = []
for method, assets in ASSETS.items():
    margin = [d * (x - l) for x, l in zip(assets, loss)]
    rows.append({"recipe": method, "assets": assets[0], "cat": margin[0], "cat_book": BOOK_133[method][0],
                 "noncat": margin[1], "noncat_book": BOOK_133[method][1]})
table_133 = pd.DataFrame(rows)
(GT(table_133, rowname_col="recipe")
 .tab_header("Table 13.3: allocated CCoC margins", "Margin of each unit, prospicio and the book; currency units")
 .tab_spanner("Cat margin", ["cat", "cat_book"])
 .tab_spanner("Non-Cat margin", ["noncat", "noncat_book"])
 .cols_label(assets="Cat assets", cat="prospicio", cat_book="book", noncat="prospicio", noncat_book="book")
 .fmt_number(["assets"], decimals=2)
 .fmt_number(["cat", "noncat"], decimals=3)
 .fmt_number(["cat_book", "noncat_book"], decimals=3, drop_trailing_zeros=True))
```

```text
                 Cat assets  prospicio   book  prospicio    book
expected loss         53.36      3.036   3.04     12.169   12.17
scaled EPD           192.38     15.675  15.68     -0.470  -0.471
scaled TVaR          161.25     12.845  12.85      2.361    2.36
scaled VaR           160.98     12.821  12.82      2.385    2.38
equal risk EPD       166.64     13.335  13.33      1.870    1.87
equal risk TVaR      149.59     11.785  11.78      3.420    3.42
equal risk VaR       149.06     11.737  11.74      3.470    3.47
coTVaR               183.15     14.835  14.84      0.370    0.37
covariance           142.93     11.179  11.18      4.026    4.03
```

Sixteen of the eighteen margins match the book to the digits that it prints.
The other two, Cat's scaled TVaR and scaled EPD margins, are within 0.005 of the
printed value, on the rounding boundary. The two bars of a recipe add up to
15.2:

```python
import matplotlib.pyplot as plt

methods = list(ASSETS)
margins = {m: [d * (x - l) for x, l in zip(ASSETS[m], loss)] for m in methods}
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ys = range(len(methods))
ax.barh([y - 0.2 for y in ys], [margins[m][0] for m in methods], height=0.4, label="Cat")
ax.barh([y + 0.2 for y in ys], [margins[m][1] for m in methods], height=0.4, label="Non-Cat")
ax.axvline(0, color="0.5", linewidth=0.8)
ax.set_yticks(list(ys), methods)
ax.invert_yaxis()
ax.set_xlabel("margin of the unit (premium less expected loss)")
ax.set_title("Table 13.3: the same total margin, shared by nine recipes")
ax.legend()
```

The same premium of 115.2 gives Cat a margin from 3.04 to 15.68, and only the
asset allocation causes the difference. The recipes that look at the tail
(scaled EPD, coTVaR) leave Non-Cat almost no margin, because Cat causes the
99.9% outcomes. The expected loss recipe ignores risk. Each recipe charges every
unit the same 10% return; the natural allocation does not.

## 3. The natural allocation of the case study (Tables 15.36 and 15.39)

Chapter 15 prices the same book with the five distortions, each calibrated to
the same total premium, and uses the lifted natural allocation. Table 15.36
prints the premiums and capital by unit, and Table 15.39 the assets:

```python
BOOK_1536 = {  # lifted: Cat premium, Cat capital, Cat assets, Non-Cat assets (Tables 15.36 and 15.39)
    "ccoc": (47.04, 270.8, 317.8, -50.62),
    "ph": (32.89, 103.2, 136.1, 131.1),
    "wang": (31.26, 92.20, 123.5, 143.7),
    "dual": (29.68, 89.01, 118.7, 148.5),
    "tvar": (28.69, 88.87, 117.6, 149.6),
}
rows = []
for family, dist in dists.items():
    cat_u, noncat_u = book.price(dist, p=0.999, allocation="lifted").allocated
    got = (cat_u.premium, cat_u.capital, cat_u.assets, noncat_u.assets)
    rows.append({"distortion": family, **dict(zip(("p", "q", "a", "n"), got)),
                 **{k + "_book": v for k, v in zip(("p", "q", "a", "n"), BOOK_1536[family])}})
lifted = pd.DataFrame(rows)
mine, theirs = ["p", "q", "a", "n"], ["p_book", "q_book", "a_book", "n_book"]
(GT(lifted, rowname_col="distortion")
 .tab_header("Tables 15.36 and 15.39: the lifted natural allocation",
             "Cat premium, Cat capital, Cat assets and Non-Cat assets, prospicio and the book")
 .tab_spanner("Cat premium", ["p", "p_book"])
 .tab_spanner("Cat capital", ["q", "q_book"])
 .tab_spanner("Cat assets", ["a", "a_book"])
 .tab_spanner("Non-Cat assets", ["n", "n_book"])
 .cols_label(**{c: ("book" if c.endswith("_book") else "prospicio") for c in lifted.columns[1:]})
 .fmt_number(mine, decimals=2)
 .fmt_number(theirs, decimals=2, drop_trailing_zeros=True))
```

```text
      prospicio   book  prospicio   book  prospicio   book  prospicio    book
ccoc      39.72  47.04     197.63  270.8     237.35  317.8      29.85  -50.62
ph        32.89  32.89     103.25  103.2     136.14  136.1     131.06   131.1
wang      31.26  31.26      92.18   92.2     123.44  123.5     143.76   143.7
dual      29.68  29.68      89.01  89.01     118.69  118.7     148.52   148.5
tvar      28.69  28.69      88.87  88.87     117.56  117.6     149.64   149.6
```

Bodoff's percentile layer of capital (PLC) allocates assets too:

```python
plc = book.bodoff(p=0.999)
print(f"PLC   assets Cat {plc[0]:.2f} (113.2), Non-Cat {plc[1]:.2f} (154.0)")
```

```text
PLC   assets Cat 113.17 (113.2), Non-Cat 154.03 (154.0)
```

```python
fig, ax = plt.subplots()
xs = range(len(lifted) + 1)
labels = list(lifted["distortion"]) + ["PLC"]
ax.bar([x - 0.2 for x in xs], list(lifted["a"]) + [plc[0]], width=0.4, label="Cat")
ax.bar([x + 0.2 for x in xs], list(lifted["n"]) + [plc[1]], width=0.4, label="Non-Cat")
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set_xticks(list(xs), labels)
ax.set_xlabel("distortion, and Bodoff's percentile layer of capital (PLC)")
ax.set_ylabel("assets allocated to the unit")
ax.set_title("Table 15.39: Cat assets fall from CCoC to PLC")
ax.legend()
```

For PH, Wang, the dual power and TVaR, prospicio agrees with the book within
0.06, and so do Bodoff's PLC assets. The 0.02 gap in the Wang capital (92.18
against 92.20) is small next to the spread between distortions, and its cause is
unknown. The PH row agrees with the book and **not** with `aggregate` 0.30.1
(Cat capital 103.52; case studies guide, section 7).

**The CCoC row does not agree.** `aggregate` 0.30.1 returns 40.51 for the Cat
premium, against 39.72 from prospicio and 47.04 in the book. The row of the book
cannot be a natural allocation of the CCoC distortion. Under the CCoC each
outcome gets a weight of at least `1 / 1.1` of its probability (the worst
outcome gets the remainder). So each unit pays at least `1 / 1.1` of its
expected loss within the assets:

```python
floor = loss[1] / 1.1
noncat_ccoc = book.price(dists["ccoc"], p=0.999, allocation="lifted").allocated[1]
print(f"lowest possible Non-Cat premium under the CCoC: {floor:.2f}")
print(f"prospicio {noncat_ccoc.premium:.2f}; book {115.15 - 47.04:.2f}")
```

```text
lowest possible Non-Cat premium under the CCoC: 72.72
prospicio 75.43; book 68.11
```

The Non-Cat premium of the book (115.15 minus 47.04, a loss ratio of 117%) is
below the floor. So the CCoC row of Table 15.36 came from a lifted calculation
that does not keep all weights non-negative, most probably in how it shared the
mass at the worst outcome.

The lifted capital of the CCoC is the quantity that
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md#5-capital-by-unit-if-you-must)
found to change between versions of `aggregate`. The book is a third version. As
in that guide, use the premiums of the linear allocation for the CCoC (Cat 35.95
and Non-Cat 79.20, which `aggregate` 0.30.1 matches).

The main result stays: the Cat premium falls from the CCoC to TVaR, as the
distortion moves its weight from the worst outcome to the upper half of the
distribution. PLC weights layers by objective probability, so it is the least
sensitive to the tail of all methods (section 15.4.4).

How much can the choice of distortion move the Cat premium?
`Portfolio.premium_bounds` gives the lowest and highest premium of each unit
under the linear allocation, over **every** distortion that prices the capped
total at the same premium (section 7 of the companion guide). All five
distortions here have the total 115.15, so one range applies to all. The linear
premium of each lies in it:

```python
total = book.price(dists["ph"], p=0.999).total.premium
bounds = {r["unit"]: r for r in book.premium_bounds(total, assets=a)}
linear = {f: [u.premium for u in book.price(d, p=0.999, allocation="linear").allocated] for f, d in dists.items()}
rows = [("lowest possible", bounds["Cat"]["lower"], bounds["NonCat"]["lower"])]
rows += [(f, *p) for f, p in linear.items()]
rows.append(("highest possible", bounds["Cat"]["upper"], bounds["NonCat"]["upper"]))
(GT(pd.DataFrame(rows, columns=["distortion", "cat", "noncat"]), rowname_col="distortion")
 .tab_header("Linear premiums and their bounds", "By unit; the bounds hold for every distortion with total premium 115.15")
 .cols_label(cat="Cat", noncat="Non-Cat")
 .fmt_number(["cat", "noncat"], decimals=2))
```

```text
                    Cat  Non-Cat
lowest possible   28.69    79.20
ccoc              35.95    79.20
ph                32.79    82.36
wang              31.25    83.90
dual              29.68    85.48
tvar              28.69    86.46
highest possible  35.95    86.46
```

The CCoC and TVaR sit at the two ends, as in the small book of the companion
guide. The CCoC puts its spare weight on the worst outcome, which loads Cat, the
unit that causes it. TVaR spreads the weight over the upper tail.

## 4. Is a reinsurance cover worth buying? (chapter 19)

An insurer prices with a spectral risk measure. Should it buy a cover with cash
flows `C` at a premium `P`? The direct test (the book's "A/B" method) prices the
book gross and net and compares the saving with `P`. Two shortcuts price `C` as
a unit of the gross book or of the net book. They are first-order
approximations, so their results differ slightly from the direct test.

Example 302 puts an aggregate cover of 2.5 excess of 7 on `X1` of the Simple
Discrete Example, where `X1` takes 0, 8 or 10 and `X2` takes 0, 1 or 90. It
prices with the proportional hazard `g(s) = s^0.5`. The gross allocation uses a
three-unit portfolio: `X1` net of the cover, the cover and `X2`, which add up to
the gross loss:

```python
ph = Distortion.proportional_hazard(0.5)
OUTCOMES, PROBS = [], []
for x1, p1 in ((0.0, 0.5), (8.0, 0.25), (10.0, 0.25)):
    for x2, p2 in ((0.0, 0.5), (1.0, 0.25), (90.0, 0.25)):
        ceded = min(max(x1 - 7.0, 0.0), 2.5)
        OUTCOMES.append((x1, x2, ceded))
        PROBS.append(p1 * p2)
gross = Portfolio(["X1 net", "ceded", "X2"], [[x1 - c, c, x2] for x1, x2, c in OUTCOMES], PROBS)
g = gross.price(ph)
net = Portfolio(["net"], [[x1 + x2 - c] for x1, x2, c in OUTCOMES], PROBS).price(ph).total.premium
example_302 = pd.DataFrame({
    "measure": ["gross premium", "ceded, allocated gross", "net premium", "A/B saving"],
    "prospicio": [g.total.premium, g.allocated[1].premium, net, g.total.premium - net],
    "book": [51.1560, 1.2613, 49.8986, 1.2574],
})
(GT(example_302, rowname_col="measure")
 .tab_header("Example 302: the cover 2.5 xs 7 under the proportional hazard")
 .fmt_number(["prospicio", "book"], decimals=4))
```

```text
                        prospicio     book
gross premium             51.1560  51.1560
ceded, allocated gross     1.2613   1.2613
net premium               49.8986  49.8986
A/B saving                 1.2574   1.2574
```

The net allocation prices `C` with the weights of the **net** book, which depend
only on the order of the net outcomes. The cover changes that order: 10 + 0 (net
7.5) now ranks below 8 + 1 (net 8). `Portfolio` sorts
outcomes by its own total. So the code adds the cover as a unit scaled down by a
very small factor, which keeps the net order and the weights, and scales the
allocation of the cover back up:

```python
tiny = 1e-6
net_view = Portfolio(["X1 net", "X2", "ceded, scaled"],
                     [[x1 - c, x2, tiny * c] for x1, x2, c in OUTCOMES], PROBS)
ceded_net = net_view.price(ph).allocated[2].premium / tiny
print(f"ceded, allocated net {ceded_net:.4f} (book 1.2497)")
```

```text
ceded, allocated net 1.2497 (book 1.2497)
```

Below about 1.25 the cover is worth buying on all tests, and above about 1.26 on
none. Between these prices the answer depends on the method. The book then lets
the insurer buy a share `t` of the cover for `t P`. At an intermediate premium
the best share is neither 0 nor 1 but 2/3, where the two net outcomes that
change places tie:

```python
rows = []
for premium in (1.2497, 1.2574, 1.2613):
    best = min((Portfolio(["net"], [[x1 + x2 - t * c] for x1, x2, c in OUTCOMES], PROBS)
                .price(ph).total.premium + t * premium, t) for t in [i / 30 for i in range(31)])
    rows.append({"premium": str(premium), "share": best[1], "cost": best[0]})
(GT(pd.DataFrame(rows), rowname_col="premium")
 .tab_header("Example 302: the cheapest share of the cover", "By cover premium; cost = net premium + share x premium")
 .cols_label(share="cheapest share", cost="total cost")
 .fmt_number("share", decimals=3)
 .fmt_number("cost", decimals=4))
```

```text
        cheapest share  total cost
1.2497           1.000     51.1483
1.2574           0.667     51.1534
1.2613           0.000     51.1560
```

```python
shares = [i / 30 for i in range(31)]
costs = [Portfolio(["net"], [[x1 + x2 - t * c] for x1, x2, c in OUTCOMES], PROBS)
         .price(ph).total.premium for t in shares]
fig, ax = plt.subplots()
for premium in (1.2497, 1.2574, 1.2613):
    extra = [cost + t * premium - costs[0] for cost, t in zip(costs, shares)]
    best = extra.index(min(extra))
    (line,) = ax.plot(shares, extra, label=f"cover premium {premium}")
    ax.plot(shares[best], extra[best], "o", color=line.get_color())
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set_xlabel("share t of the cover bought")
ax.set_ylabel("total cost, less the cost with no cover")
ax.set_title("Example 302: the cheapest share depends on the cover premium")
ax.legend()
```

This is Figure 19.1 of the book. With many covers the direct test needs every
combination; the book suggests a search that uses the allocations as
gradients.
