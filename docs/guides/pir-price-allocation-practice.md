# Price allocation in practice: the case-study tables and the reinsurance decision

This guide continues
[Price allocation in *Pricing Insurance Risk*](pir-price-allocation.md). That
guide runs the small examples of chapters 12 to 15 of Mildenhall and Major's
*Pricing Insurance Risk* (Wiley, 2022). This guide applies the methods to a
realistic book, the Cat/Non-Cat case study. Section 2 allocates it by the
classical recipes of chapter 13 (Table 13.3). Section 3 allocates it by the
natural allocation and Bodoff's percentile layer of capital of chapter 15
(Tables 15.36 and 15.39). Section 4 is the chapter 19 test of whether a
reinsurance cover is worth buying, which is also an allocation question.

The guide puts each number that the book prints next to the number from
prospicio, and explains the book in new words.
[Pricing Insurance Risk: the case studies](pricing-insurance-risk-case-studies.md)
compares the case studies with `aggregate` 0.30.1, the package that made the
exhibits of the book. This guide compares the gross Cat/Non-Cat book with the
printed tables of the book.

## 1. The book and its capital standard

Cat/Non-Cat has two independent units with a total mean of 100. Non-Cat is a
gamma with mean 80 and CV 0.15, and Cat is a lognormal with mean 20 and CV 1. As
in the case studies guide, the code does these steps:

1. It puts each unit on the grid of `aggregate` (a step of 1/64, 2^16 points).
2. It combines the units by FFT.
3. It sets the assets to the 99.9% quantile of the total.
4. It calibrates five distortions to a 10% return on capital:

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

The calibration on the capped total is the shortcut that the case studies guide
explains. It gives the same parameters as a calibration on the full portfolio.

## 2. Allocated CCoC pricing (Table 13.3)

Chapter 13 allocates the 267.2 of assets to the units and charges a 10% return
on the capital of each unit. The margin of a unit is `0.1 / 1.1` times its
assets minus its expected loss. The result depends on the asset allocation, and
the chapter uses nine methods:

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

The grids have `quantile`, `tvar` and `stop_loss`, so each recipe is a few
lines. Where a recipe must find a level, it uses a bisection:

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
d = 0.1 / 1.1
print(f"{'':16s} {'Cat assets':>10s} {'Cat M':>8s} {'NonCat M':>9s}   book margins")
for method, assets in ASSETS.items():
    margin = [d * (x - l) for x, l in zip(assets, loss)]
    print(f"{method:16s} {assets[0]:10.2f} {margin[0]:8.3f} {margin[1]:9.3f}   {BOOK_133[method]}")
print(f"levels: TVaR {p_tvar:.4f}, equal risk VaR {p_var:.4f}, equal risk TVaR {p_eq_tvar:.4f}")
```

```text
                 Cat assets    Cat M  NonCat M   book margins
expected loss         53.36    3.036    12.169   (3.04, 12.17)
scaled EPD           192.38   15.675    -0.470   (15.68, -0.471)
scaled TVaR          161.25   12.845     2.361   (12.85, 2.36)
scaled VaR           160.98   12.821     2.385   (12.82, 2.38)
equal risk EPD       166.64   13.335     1.870   (13.33, 1.87)
equal risk TVaR      149.59   11.785     3.420   (11.78, 3.42)
equal risk VaR       149.06   11.737     3.470   (11.74, 3.47)
coTVaR               183.15   14.835     0.370   (14.84, 0.37)
covariance           142.93   11.179     4.026   (11.18, 4.03)
levels: TVaR 0.9970, equal risk VaR 0.9977, equal risk TVaR 0.9931
```

Sixteen of the eighteen margins match the book to the digits that it prints. The
other two are the scaled TVaR and scaled EPD margins of Cat: 12.845 and 15.675
against the printed 12.85 and 15.68. Both are within 0.005 of the printed value,
on the rounding boundary. The Non-Cat margins (2.36 and -0.47) and the totals
match, so the allocation is the same. A small difference in the level that the
code finds for the TVaR or EPD of the total can move them by that amount.

The table shows the critique of chapter 13 in numbers. The same premium of 115.2
gives Cat a margin from 3.04 to 15.68 and Non-Cat a margin from 12.17 to -0.47.
Only the recipe that allocates the assets causes this difference. The recipes
that look at the tail (scaled EPD, coTVaR) give Non-Cat almost no margin,
because Cat causes the 99.9% outcomes. The expected loss recipe ignores risk.
Each recipe charges every unit the same 10% return, and the natural allocation
does not.

## 3. The natural allocation of the case study (Tables 15.36 and 15.39)

Chapter 15 prices the same book with the five distortions, each calibrated to
the same total premium, and uses the lifted natural allocation. Table 15.36
prints the premiums and capital by unit. Table 15.39 prints the assets, with
Bodoff's percentile layer of capital (PLC) next to them:

```python
BOOK_1536 = {  # lifted: Cat premium, Cat capital, Cat assets, Non-Cat assets (Tables 15.36 and 15.39)
    "ccoc": (47.04, 270.8, 317.8, -50.62),
    "ph": (32.89, 103.2, 136.1, 131.1),
    "wang": (31.26, 92.20, 123.5, 143.7),
    "dual": (29.68, 89.01, 118.7, 148.5),
    "tvar": (28.69, 88.87, 117.6, 149.6),
}
print(f"{'':5s} {'Cat P':>15s} {'Cat Q':>17s} {'Cat assets':>17s} {'NonCat assets':>17s}")
for family, dist in dists.items():
    cat_u, noncat_u = book.price(dist, p=0.999, allocation="lifted").allocated
    got = (cat_u.premium, cat_u.capital, cat_u.assets, noncat_u.assets)
    print(f"{family:5s} " + " ".join(f"{g:7.2f} ({b:6.2f})" for g, b in zip(got, BOOK_1536[family])))
plc = book.bodoff(p=0.999)
print(f"PLC   assets Cat {plc[0]:.2f} (113.2), Non-Cat {plc[1]:.2f} (154.0)")
```

```text
                Cat P             Cat Q        Cat assets     NonCat assets
ccoc    39.72 ( 47.04)  197.63 (270.80)  237.35 (317.80)   29.85 (-50.62)
ph      32.89 ( 32.89)  103.25 (103.20)  136.14 (136.10)  131.06 (131.10)
wang    31.26 ( 31.26)   92.18 ( 92.20)  123.44 (123.50)  143.76 (143.70)
dual    29.68 ( 29.68)   89.01 ( 89.01)  118.69 (118.70)  148.52 (148.50)
tvar    28.69 ( 28.69)   88.87 ( 88.87)  117.56 (117.60)  149.64 (149.60)
PLC   assets Cat 113.17 (113.2), Non-Cat 154.03 (154.0)
```

For PH, Wang, the dual power and TVaR, the premiums, capital and assets agree
with the book within 0.06, and Bodoff's PLC assets also agree. The Wang capital
is 92.18 against 92.20, with equal premiums. This guide could not find the cause
of this difference of 0.02, which is small next to the spread between
distortions. The PH row agrees with the book and **not** with `aggregate`
0.30.1, which gives Cat a capital of 103.52 (see the case studies guide, section
7). The PH capital of the book, 103.2, is the 103.25 of prospicio.

**The CCoC row does not agree.** The book prints a Cat premium of 47.04 and
Non-Cat assets of -50.62. prospicio returns 39.72 and 29.85, and `aggregate`
0.30.1 returns 40.51 for the Cat premium. The row of the book cannot be a
natural allocation of the CCoC distortion. Under the CCoC each outcome gets a
weight of at least `1 / 1.1` of its probability (the worst outcome gets the
remainder). So each natural allocation charges each unit at least `1 / 1.1` of
its expected loss within the assets:

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

The Non-Cat premium of the book, 115.15 minus 47.04, is 68.11, a loss ratio of
117% that the text of the book discusses. This premium is below the floor of
72.72. So a version of the lifted calculation that does not keep all weights
non-negative produced the CCoC row of Table 15.36. The most probable cause is
how that version shared the mass of the distortion at the worst outcome on the
grid.

The lifted capital of the CCoC is the quantity that
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md#5-capital-by-unit-if-you-must)
found to change between versions of `aggregate`. The book is a third version. As
in that guide, use the premiums of the linear allocation for the CCoC (Cat 35.95
and Non-Cat 79.20, which `aggregate` 0.30.1 matches).

The main result of the table stays the same. The Cat premium decreases from the
CCoC to TVaR. Along that order, the distortion moves its weight from the worst
outcome to the whole upper half of the distribution. The Cat assets decrease
from 136 under PH to 118 under TVaR, and PLC gives Cat less again (113.2). PLC
weights layers by objective probability, so it is the least sensitive to the
tail of all methods, as section 15.4.4 observes.

The natural allocation depends on the distortion. How much can that choice move
the Cat premium? `Portfolio.premium_bounds` gives the lowest and highest premium
of each unit under the linear allocation. The range covers **every** distortion
that prices the capped total at the same premium (section 7 of the companion
guide explains the method). All five distortions here have the same total of
115.15, so one range applies to all of them. The linear Cat premium of each
distortion is in that range:

```python
total = book.price(dists["ph"], p=0.999).total.premium
for row in book.premium_bounds(total, assets=a):
    print(f"{row['unit']}: {row['lower']:.2f} to {row['upper']:.2f}")
for family, dist in dists.items():
    linear = book.price(dist, p=0.999, allocation="linear").allocated[0].premium
    print(f"{family:5s} linear Cat premium {linear:.2f}")
```

```text
Cat: 28.69 to 35.95
NonCat: 79.20 to 86.46
ccoc  linear Cat premium 35.95
ph    linear Cat premium 32.79
wang  linear Cat premium 31.25
dual  linear Cat premium 29.68
tvar  linear Cat premium 28.69
```

The CCoC and TVaR are at the two ends of the range, as in the small book of the
companion guide. The CCoC puts all its spare weight on the worst outcome, which
loads Cat, the unit that causes the worst outcomes. TVaR spreads the weight over
the upper tail. The range for Cat, 28.69 to 35.95, is the freedom that the
choice of distortion gives.

## 4. Is a reinsurance cover worth buying? (chapter 19)

Chapter 19 asks a question. An insurer prices with a spectral risk measure:
should it buy a cover with cash flows `C` at a premium `P`? The direct test (the
book's "A/B" method) prices the book gross and net and compares the saving with
`P`. Two shortcuts allocate instead. They price `C` as a unit of the gross book,
or as a unit of the net book. They are first-order approximations of the direct
test, and their results are slightly different from it.

Example 302 puts an aggregate cover of 2.5 excess of 7 on `X1` of the Simple
Discrete Example. In that example, `X1` takes 0, 8 or 10 and `X2` takes 0, 1 or 90.
Example 302 prices with the proportional hazard `g(s) = s^0.5`. The gross
allocation uses a three-unit portfolio. Its units are `X1` net of the cover, the
cover and `X2`, and they add up to the gross loss:

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
print(f"gross premium {g.total.premium:.4f} (book 51.1560); ceded, allocated gross {g.allocated[1].premium:.4f}"
      " (book 1.2613)")
net = Portfolio(["net"], [[x1 + x2 - c] for x1, x2, c in OUTCOMES], PROBS).price(ph).total.premium
print(f"net premium {net:.4f} (book 49.8986); A/B saving {g.total.premium - net:.4f} (book 1.2574)")
```

```text
gross premium 51.1560 (book 51.1560); ceded, allocated gross 1.2613 (book 1.2613)
net premium 49.8986 (book 49.8986); A/B saving 1.2574 (book 1.2574)
```

The net allocation prices `C` with the weights of the **net** book. Those
weights depend only on the order of the net outcomes. The cover changes that
order: 10 + 0 (net 7.5) now ranks below 8 + 1 (net 8). `Portfolio` sorts
outcomes by its own total. So the code adds the cover as a unit scaled down by a
very small factor, which keeps the net order and the weights. Then it scales the
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

The three reservation prices are 1.2613, 1.2574 and 1.2497. Below about 1.25 the
cover is worth buying on all tests, and above about 1.26 it is worth buying on
no test. Between these prices the answer depends on the method. The book then
lets the insurer buy a share `t` of the cover for `t P`. At an intermediate
premium the best share is neither 0 nor 1 but 2/3, where the two net outcomes
that change places tie:

```python
for premium in (1.2497, 1.2574, 1.2613):
    best = min((Portfolio(["net"], [[x1 + x2 - t * c] for x1, x2, c in OUTCOMES], PROBS)
                .price(ph).total.premium + t * premium, t) for t in [i / 30 for i in range(31)])
    print(f"cover premium {premium}: cheapest share {best[1]:.3f}, total cost {best[0]:.4f}")
```

```text
cover premium 1.2497: cheapest share 1.000, total cost 51.1483
cover premium 1.2574: cheapest share 0.667, total cost 51.1534
cover premium 1.2613: cheapest share 0.000, total cost 51.1560
```

At the lowest premium the full cover is best, and at the highest premium no
cover is best. At the A/B premium a share of 2/3 has the lowest cost, as Figure
19.1 of the book shows. With many covers the direct test needs every
combination. The book suggests a search that uses the allocations as gradients.
