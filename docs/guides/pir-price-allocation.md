# Price allocation in *Pricing Insurance Risk*: classical methods, the natural allocation and Bodoff

<!-- site: execute -->

Part III of Stephen Mildenhall and John Major's *Pricing Insurance Risk: Theory
and Practice* (Wiley, 2022), chapters 12 to 15, asks how to split a portfolio's
price over its units. Chapter 12 gives the classical answers, which allocate
capital first and then price it. Chapter 14 builds the book's own answer, the
**natural allocation** of a spectral risk measure. Chapter 15 works it on small
discrete examples and compares it with Bodoff's percentile layer of capital.

This guide runs the book's small examples in prospicio, with the book's values
beside prospicio's. Exercise, example and table numbers are the book's.

New to distortions, CCoC or the pentagon? Read
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md) first. The case
studies of chapters 13 and 15 and the reinsurance evaluation of chapter 19 are in
[Price allocation in practice](pir-price-allocation-practice.md).

## 1. The Simple Discrete Example

Most of chapter 12 uses the Simple Discrete Example (section 2.4.1): two
independent units, `X1` on 0, 8 and 10 and `X2` on 0, 1 and 90, each with
probabilities 1/2, 1/4 and 1/4. The nine joint outcomes have probabilities in
multiples of 1/16, so sixteen equally likely rows give the portfolio exactly.
The allocation functions need such equally likely simulations, a
`PredictiveDistribution`:

```python
from prospicio.distributions import PredictiveDistribution, Sampled
from prospicio.pricing import Portfolio
from prospicio.risk import Distortion, capital

X1 = {0: 2, 8: 1, 10: 1}  # value: weight out of 4
X2 = {0: 2, 1: 1, 90: 1}
ROWS = [[float(a), float(b)] for a, n in X1.items() for b, m in X2.items() for _ in range(n * m)]
sims = PredictiveDistribution(["unit"], [("X1",), ("X2",)], ROWS)
discrete = Portfolio(["X1", "X2"], ROWS)
x1, x2 = Sampled([r[0] for r in ROWS]), Sampled([r[1] for r in ROWS])
print(len(ROWS), "rows; means", x1.mean(), x2.mean(), "total", x1.mean() + x2.mean())
print("totals", discrete.totals)
```

```text
16 rows; means 4.5 22.75 total 27.25
totals [0.0, 1.0, 8.0, 9.0, 10.0, 11.0, 90.0, 98.0, 100.0]
```

`X2` is the dangerous unit: one year in four it loses 90.

## 2. The classical recipes (chapter 12)

Chapter 12 starts from allocated constant cost of capital (CCoC) pricing: each
unit gets capital and pays the same return on it. In Exercise 203 three units
have an expected loss of 100 and capital of 40, 80 and 200. At 12% their
premiums are 104.8, 109.6 and 124.0. The premium depends on the capital
allocation, so the chapter examines the common recipes.

The chapter allocates two amounts: the TVaR of the total at 75% (94.5) and a
premium of 80. `capital` takes a distortion (here TVaR) and a method:

- `"euler"`: the co-measure. For TVaR it is the coTVaR, each unit's average
  loss in the worst quarter of years.
- `"marginal"`: Merton and Perold's saving when the unit is removed.
- `"shapley"`: the Shapley value of the game whose cost is the TVaR of each
  sub-portfolio.
- `"proportional"`: the stand-alone measures, scaled to the total.

```python
tvar75 = Distortion.tvar(0.75)
print(f"TVaR 75% of the total: {capital(sims, tvar75).total}  (book 94.5)")
```

```text
TVaR 75% of the total: 94.5  (book 94.5)
```

Tables 12.5 and 12.8, with the book's values under prospicio's:

```python
import pandas as pd
from great_tables import GT


def pair_table(frame, decimals):
    """A table whose rows are prospicio's, each followed by the book's row, named 'x, book'."""
    names = list(frame.iloc[:, 0])
    book = [i for i, name in enumerate(names) if name.endswith(", book")]
    mine = [i for i in range(len(names)) if i not in book]
    table = GT(frame, rowname_col=frame.columns[0])
    for column, places in decimals.items():
        table = table.fmt_number(column, rows=mine, decimals=places, use_seps=False)
        table = table.fmt_number(column, rows=book, decimals=places, drop_trailing_zeros=True, use_seps=False)
    return table.sub_missing(missing_text="")


NAN = float("nan")
BOOK_12 = {"euler": (4.5, 90), "marginal": (4.5, 84.5), "shapley": (7.25, 87.25)}
rows = []
for method in ("euler", "marginal", "shapley", "proportional"):
    a = capital(sims, tvar75, method)
    rows.append((method, *a.allocated))
    if method in BOOK_12:
        rows.append((f"{method}, book", *BOOK_12[method]))
rows += [("stand-alone", *a.standalone), ("stand-alone, book", 10, 90)]
(pair_table(pd.DataFrame(rows, columns=["method", "X1", "X2"]), {"X1": 2, "X2": 2})
 .tab_header("Tables 12.5 and 12.8: the TVaR 75% of the total (94.5) by unit", "Currency units"))
```

```text
                      X1     X2
euler               4.50  90.00
euler, book          4.5     90
marginal            4.50  84.50
marginal, book       4.5   84.5
shapley             7.25  87.25
shapley, book       7.25  87.25
proportional        9.45  85.05
stand-alone        10.00  90.00
stand-alone, book     10     90
```

The marginal amounts add up to 89, not 94.5: no single unit causes the last
5.5, so the method gives it to no unit. The coTVaR gives `X1` only its expected
loss of 4.5, so `X1` gets no margin: in the worst quarter of years `X2` loses 90
and `X1` loses no more than usual. Shapley averages the stand-alone and marginal
views.

Exercise 211 also allocates a premium of 80. `Portfolio.calibrate` finds the
TVaR level that prices the total at 80:

```python
d80 = discrete.calibrate("tvar", premium=80.0)
print(f"TVaR level pricing the total at 80: {d80.parameters['p']:.4f}  (book 0.6975)")
BOOK_80 = {"stand-alone": (9.65, 74.54), "marginal": (5.46, 70.35), "shapley": (7.55, 72.45)}
MEANS = (4.5, 22.75)
split80 = []  # label, X1, X2, loss ratio of X1, loss ratio of X2


def add_split(label, got, book, ratios):
    split80.append((label, *got, *([m / g for m, g in zip(MEANS, got)] if ratios else [NAN, NAN])))
    split80.append((f"{label}, book", *book, *ratios_book.get(label, (NAN, NAN))))


ratios_book = {"shapley": (0.596, 0.314), "haircut": (0.563, 0.316)}
a = capital(sims, d80, "shapley")
add_split("stand-alone", a.standalone, BOOK_80["stand-alone"], False)
add_split("marginal", capital(sims, d80, "marginal").allocated, BOOK_80["marginal"], True)
add_split("shapley", a.allocated, BOOK_80["shapley"], True)
```

```text
TVaR level pricing the total at 80: 0.6975  (book 0.6975)
```

Exercises 205 and 206 use each unit's stand-alone TVaR. The haircut splits the
80 in proportion to the TVaRs at 75% (10 and 90). The equal risk allocation
finds the level `p` at which the stand-alone TVaRs add up to 80, which prices
each unit as if the two were comonotonic. A bisection finds `p`:

```python
t75 = [x1.tvar(0.75), x2.tvar(0.75)]
add_split("haircut", [80 * t / sum(t75) for t in t75], (8, 72), True)
lo, hi = 0.5, 0.75
for _ in range(60):  # bisection on the level p
    p = (lo + hi) / 2
    lo, hi = (lo, p) if x1.tvar(p) + x2.tvar(p) > 80 else (p, hi)
add_split("equal risk", [x1.tvar(p), x2.tvar(p)], (9.56, 70.44), True)
print(f"equal risk level {p:.4f} (book 0.6796)")
```

```text
equal risk level 0.6796 (book 0.6796)
```

```python
(pair_table(pd.DataFrame(split80, columns=["method", "X1", "X2", "LR X1", "LR X2"]),
            {"X1": 2, "X2": 2, "LR X1": 4, "LR X2": 4})
 .tab_header("Exercises 205, 206 and 211: a premium of 80 split five ways",
             "Currency units; LR is the expected loss over the allocated premium"))
```

```text
                     X1     X2   LR X1   LR X2
stand-alone        9.65  74.54                
stand-alone, book  9.65  74.54                
marginal           5.46  70.35  0.8249  0.3234
marginal, book     5.46  70.35                
shapley            7.55  72.45  0.5957  0.3140
shapley, book      7.55  72.45   0.596   0.314
haircut            8.00  72.00  0.5625  0.3160
haircut, book         8     72   0.563   0.316
equal risk         9.56  70.44  0.4707  0.3230
equal risk, book   9.56  70.44
```

Every row matches. The recipes split the same 80 as 8 and 72, 9.56 and 70.44, or
7.55 and 72.45, and nothing says which split is correct.

### The covariance allocation: two definitions

Exercise 217 uses Kalkbrener's method with the standard deviation as auxiliary
measure: each unit gets its expected loss plus a share of the margin
`80 - 27.25` in proportion to its covariance with the total. prospicio's
`"covariance"` method spreads the whole measure, expected loss included, by
`rho(S) Cov(X_j, S) / Var(S)`. The covariance shares are the same, so the
book's version follows from them:

```python
cov = capital(sims, d80, "covariance")
shares = [x / cov.total for x in cov.allocated]  # Cov(X_j, S) / Var(S)
kalkbrener = [m + (80 - 27.25) * s for m, s in zip((4.5, 22.75), shares)]
print(f"prospicio 'covariance'  X1 {cov.allocated[0]:7.3f}  X2 {cov.allocated[1]:7.3f}")
print(f"mean plus covariance    X1 {kalkbrener[0]:7.3f}  X2 {kalkbrener[1]:7.3f}   book 5.216, 74.784")
print(f"loss ratios {4.5 / kalkbrener[0]:.3f}, {22.75 / kalkbrener[1]:.3f}  (book 0.863, 0.304)")
```

```text
prospicio 'covariance'  X1   1.086  X2  78.914
mean plus covariance    X1   5.216  X2  74.784   book 5.216, 74.784
loss ratios 0.863, 0.304  (book 0.863, 0.304)
```

`X1` is almost uncorrelated with the total (covariance 20.75, variance 1,528),
so spreading all of 80 by covariance gives it 1.09, a quarter of its expected
loss. No insurer charges that. The book's form keeps the expected loss and
allocates only the margin.

### Equal priority in default

Section 12.3: when an insurer cannot pay all claims, equal priority pays each
the same fraction of what is owed. Exercise 219 has assets of 1,200 against
liabilities of 800, 600 and 400. A one-scenario `Portfolio` reports each line's
recovery as its `loss`, and `Portfolio.priority` adds the senior and junior
rules:

```python
owed = Portfolio(["A", "B", "C"], [[800.0, 600.0, 400.0]])
paid = owed.price(Distortion.ccoc(0.1), assets=1200).allocated
print("equal priority:", [round(u.loss, 1) for u in paid], " (book 533, 400, 267)")
ladder = owed.priority(assets=1200)
print("A senior, equal, junior:", ladder["senior"][0], round(ladder["equal"][0], 1), ladder["junior"][0])
```

```text
equal priority: [533.3, 400.0, 266.7]  (book 533, 400, 267)
A senior, equal, junior: 800.0 533.3 200.0
```

Each line gets two thirds. Ranked behind B and C, A would get only 200.

## 3. The natural allocation, layer by layer (chapter 14)

Chapter 14 prices the portfolio with a spectral risk measure (a distortion `g`
of the survival function `S` of the total) and then allocates the premium:

1. Sort the outcomes by the **total** loss. Each total `x_k` gets the weight
   `g(S(x_k-)) - g(S(x_k))`: more than its probability if the total is bad, less
   if it is good. The weights add up to 1, and the premium is the expected total
   under them.
2. A unit's premium is its expected loss under the **same** weights. They depend
   only on the total, so a unit needs only its average loss at each total,
   `kappa_i(x) = E[X_i | X = x]`. The `kappa` values add up to the total, so the
   unit premiums add up to the portfolio premium.
3. Read the result as layers. A layer of assets between two consecutive totals
   costs `g(S)` times its width, against an expected loss of `S` times its
   width. Each unit pays in proportion to its expected contribution to the losses
   that reach the layer. A unit's return depends on the layers it uses.
   Allocated CCoC pricing does the opposite: one return, allocated capital.

Example 258 runs the algorithm on an adjusted Simple Discrete Example: `X1`
takes 9 instead of 8, so the total 10 has two causes (9 + 1 and 10 + 0). The
distortion is the proportional hazard `g(s) = s^0.5`. At a total of 10 the
averages are 9.6667 for `X1` and 0.3333 for `X2`:

```python
ADJ_ROWS = [[x1, x2] for x1 in (0.0, 9.0, 10.0) for x2 in (0.0, 1.0, 90.0)]
ADJ_PROBS = [p1 * p2 for p1 in (0.5, 0.25, 0.25) for p2 in (0.5, 0.25, 0.25)]
adjusted = Portfolio(["X1", "X2"], ADJ_ROWS, ADJ_PROBS)
ph = Distortion.proportional_hazard(0.5)
kappas = pd.DataFrame({
    "total": [f"{x:.0f}" for x in adjusted.totals],
    "prob": adjusted.probs,
    "kappa1": adjusted.kappa("X1"),
    "kappa2": adjusted.kappa("X2"),
})
(GT(kappas, rowname_col="total")
 .tab_header("Table 15.3: the units' average losses given the total", "Example 258, by total loss of the portfolio")
 .fmt_number("prob", decimals=4)
 .fmt_number(["kappa1", "kappa2"], decimals=4, use_seps=False))
```

```text
       prob   kappa1   kappa2
0    0.2500   0.0000   0.0000
1    0.1250   0.0000   1.0000
9    0.1250   9.0000   0.0000
10   0.1875   9.6667   0.3333
11   0.0625  10.0000   1.0000
90   0.1250   0.0000  90.0000
99   0.0625   9.0000  90.0000
100  0.0625  10.0000  90.0000
```

The premiums are those of Table 15.4 to four decimals:

```python
priced = adjusted.price(ph, assets=100)
print("premium", [round(u.premium, 4) for u in priced.allocated], round(priced.total.premium, 4),
      " (book 6.2048, 45.1838, 51.3887)")
```

```text
premium [6.2048, 45.1838] 51.3887  (book 6.2048, 45.1838, 51.3887)
```

```python
import matplotlib.pyplot as plt

labels = [f"{x:.0f}" for x in adjusted.totals]
k1, k2 = adjusted.kappa("X1"), adjusted.kappa("X2")
fig, ax = plt.subplots()
ax.bar(labels, k1, label="X1")
ax.bar(labels, k2, bottom=k1, label="X2")
ax.set_xlabel("total loss of the portfolio")
ax.set_ylabel("average loss of the unit given the total")
ax.set_title("Example 258: the units' average losses add up to each total")
ax.legend()
```

### When totals tie: the natural allocation set

When outcomes tie on the total, the distortion cannot rank them, and each order
gives a valid allocation. These form the **natural allocation set**, and the
`kappa` average (the **linear natural allocation**) is in it. Example 259 gives the extreme orders of the tie at 10:
`X1` gets 6.2085 or 6.2009. Exercise 260 does the same for TVaR at 65%: `X1` gets
6.1429 or 6.2500 and `X2` 64.5714 or 64.4643. The linear allocation lies
between:

```python
linear65 = [u.premium for u in adjusted.price(Distortion.tvar(0.65), assets=100).allocated]
print(f"TVaR 65%: X1 {linear65[0]:.4f} (book range 6.1429 to 6.2500),"
      f" X2 {linear65[1]:.4f} (book range 64.4643 to 64.5714)")
```

```text
TVaR 65%: X1 6.2143 (book range 6.1429 to 6.2500), X2 64.5000 (book range 64.4643 to 64.5714)
```

## 4. Limited assets: the linear and lifted allocations

With assets `a` below the largest loss, the insurer pays `X ∧ a`, and under
equal priority unit `i` gets `X_i a / X` when the total exceeds `a`. All default
outcomes share the capped total `a`: a tie, so the natural allocation is not
unique. Chapter 14 selects two members:

- **Linear**: all default outcomes count alike. The default event's weight is
  split by each unit's expected share of the payments in default.
- **Lifted**: default outcomes are ranked by the uncapped total `X`, so the worst
  defaults get the most weight. It measures the size of default.

`Portfolio.price` takes `allocation="linear"` or `"lifted"`. Example 235 has
ten equally likely outcomes, assets of 10 and TVaR at 80%. Four totals exceed
10:

```python
E235_X1 = [1, 2, 3, 4, 5, 6, 7, 8, 12, 10]
E235_X2 = [1, 2, 3, 4, 5, 9, 8, 12, 13, 20]
e235 = Portfolio(["X1", "X2"], [[float(a), float(b)] for a, b in zip(E235_X1, E235_X2)])
tvar80 = Distortion.tvar(0.8)
rows = []
for allocation, book in (("lifted", (4.067, 5.933)), ("linear", (4.3, 5.7))):
    for assets, label in ((10, "10"), (10 - 1e-9, "10 - 1e-9")):
        got = [u.premium for u in e235.price(tvar80, assets=assets, allocation=allocation).allocated]
        rows.append((f"{allocation}, assets {label}", *got, *book))
(GT(pd.DataFrame(rows, columns=["case", "X1", "X2", "book X1", "book X2"]), rowname_col="case")
 .tab_header("Example 235: premium by unit, TVaR 80%", "Ten equally likely outcomes; currency units")
 .fmt_number(["X1", "X2"], decimals=4, use_seps=False)
 .fmt_number(["book X1", "book X2"], decimals=3, drop_trailing_zeros=True, use_seps=False))
```

```text
                              X1      X2  book X1  book X2
lifted, assets 10         4.0667  5.9333    4.067    5.933
lifted, assets 10 - 1e-9  4.0667  5.9333    4.067    5.933
linear, assets 10         4.1600  5.8400      4.3      5.7
linear, assets 10 - 1e-9  4.3000  5.7000      4.3      5.7
```

The lifted allocation agrees with the book at assets of 10 and gives more to
`X2`, which dominates the two worst outcomes. The linear allocation agrees only
slightly below 10, because one outcome totals exactly 10, the asset level.

The book counts an outcome at exactly the asset level as a default. prospicio
follows `aggregate` 1.0.1: it pays that outcome in full and splits the default
weight over outcomes above `a`. The total price is the same. Assets of
`a - 1e-9` give the book's linear numbers.

Example 263 prices the adjusted example with assets of 80, so three outcomes
default:

```python
got = adjusted.price(ph, assets=80, allocation="lifted").allocated
print(f"assets 80, lifted: loss {got[0].loss:.4f}, {got[1].loss:.4f}"
      f"  premium {got[0].premium:.4f}, {got[1].premium:.4f}")
print("            book: loss 4.5170, 19.2955  premium 5.5260, 37.4307")
```

```text
assets 80, lifted: loss 4.5170, 19.2955  premium 5.5260, 37.4307
            book: loss 4.5170, 19.2955  premium 5.5260, 37.4307
```

```python
TWO_ROWS = [[a, b] for a in (0.0, 10.0, 20.0) for b in (0.0, 10.0, 40.0)]
TWO_PROBS = [p * q for p in (0.5, 0.25, 0.25) for q in (0.5, 0.25, 0.25)]
two = Portfolio(["X1", "X2"], TWO_ROWS, TWO_PROBS)
ph87 = Distortion.proportional_hazard(0.87)
BOOK_1520 = {  # assets: lifted X1, X2, linear X1, X2, total
    10: (3.5622, 4.2236, 3.6983, 4.0875, 7.7858),
    20: (5.9672, 7.2901, 6.2183, 7.0390, 13.2573),
    30: (6.8419, 10.0505, 6.8350, 10.0574, 16.8924),
    40: (7.2890, 12.5971, 7.0972, 12.7889, 19.8861),
    50: (7.7361, 13.7880, 7.6846, 13.8395, 21.5241),
    60: (8.0348, 14.3855, 8.0348, 14.3855, 22.4203),
}
largest, rows = 0.0, []
for a, book in BOOK_1520.items():
    lifted = two.price(ph87, assets=a, allocation="lifted")
    linear = two.price(ph87, assets=a - 1e-9, allocation="linear")
    got = [u.premium for u in lifted.allocated] + [u.premium for u in linear.allocated]
    got.append(lifted.total.premium)
    largest = max(largest, *(abs(g - b) for g, b in zip(got, book)))
    rows.append((str(a), *got))
columns = ["assets", "lifted X1", "lifted X2", "linear X1", "linear X2", "total"]
(GT(pd.DataFrame(rows, columns=columns), rowname_col="assets")
 .tab_header("Table 15.20: premium by unit and assets", "Example 272, proportional hazard 0.87")
 .fmt_number(columns[1:], decimals=4, use_seps=False))
```

```text
    lifted X1  lifted X2  linear X1  linear X2    total
10     3.5622     4.2236     3.6983     4.0875   7.7858
20     5.9672     7.2901     6.2183     7.0390  13.2573
30     6.8419    10.0505     6.8350    10.0574  16.8924
40     7.2890    12.5971     7.0972    12.7889  19.8861
50     7.7361    13.7880     7.6846    13.8395  21.5241
60     8.0348    14.3855     8.0348    14.3855  22.4203
```

```python
print(f"largest gap to the book's thirty numbers: {largest:.1e}")
```

```text
largest gap to the book's thirty numbers: 4.6e-05
```

Example 272 prices two independent units at several asset levels (Table
15.20). All thirty numbers match to four decimals. The lifted allocation puts
more on `X2` at assets of 20 and 30, where `X2`'s loss of 40 makes the worst
defaults. At 60 nothing defaults, so the two agree.

## 5. Margin and capital by unit (section 15.2)

The layer view gives each unit a margin and a capital in each layer of assets.
Margin is premium density minus expected loss density, and in each layer capital
and margin keep the ratio the distortion sets, `(1 - g(S)) / (g(S) - S)`. A
unit's capital is the sum over layers, and its return is margin over capital.
`capital` on the lifted allocation gives this sum (section 15.2.5, Tables 15.14
and 15.15):

```python
FUNDING_BOOK = {  # margins, capitals and returns of X1, X2 as the book prints them
    100: (1.4548, 22.4338, 3.5274, 45.0839, 0.4124, 0.4976),
    80: (1.0089, 18.1353, NAN, NAN, 0.4151, 0.5239),
    11: (-0.5423, 2.4365, NAN, NAN, NAN, NAN),
}
rows = []
for a, book in FUNDING_BOOK.items():
    u1, u2 = adjusted.price(ph, assets=a, allocation="lifted").allocated
    m1, m2 = u1.premium - u1.loss, u2.premium - u2.loss
    rows.append((f"assets {a}", m1, m2, u1.capital, u2.capital, m1 / u1.capital, m2 / u2.capital))
    rows.append((f"assets {a}, book", *book))
columns = ["assets", "margin X1", "margin X2", "capital X1", "capital X2", "return X1", "return X2"]
(pair_table(pd.DataFrame(rows, columns=columns), {c: 4 for c in columns[1:]})
 .tab_header("Tables 15.14 and 15.15: margin, capital and return by unit", "Lifted allocation; return is margin over capital"))
```

```text
                  margin X1  margin X2  capital X1  capital X2  return X1  return X2
assets 100           1.4548    22.4338      3.5274     45.0839     0.4124     0.4976
assets 100, book     1.4548    22.4338      3.5274     45.0839     0.4124     0.4976
assets 80            1.0089    18.1353      2.4305     34.6128     0.4151     0.5239
assets 80, book      1.0089    18.1353                             0.4151     0.5239
assets 11           -0.5423     2.4365     -0.6720      3.2153     0.8070     0.7578
assets 11, book     -0.5423     2.4365
```

Every value the book prints matches. At assets of 11, `X1` has a negative
margin: default is so likely that `X2` takes payments from `X1` most of the time,
and `X1` is paid for sharing its recoveries. Returns differ by unit (41% against
50% at full assets) because `X2` uses the expensive high layers.

## 6. Bodoff's percentile layer of capital (section 15.3)

Bodoff (2007) also allocates capital layer by layer, but with **objective**
probabilities: the units share each layer in proportion to their expected share
of the losses that reach it. His Thought Experiment 1 (section 15.3): a
hurricane that costs 99 with probability 0.2 and an independent earthquake that
costs 100 with probability 0.05. Assets are 100, equal priority, 10% cost of
capital.

```python
bodoff_book = Portfolio(["Hu", "EQ"], [[0.0, 0.0], [99.0, 0.0], [0.0, 100.0], [99.0, 100.0]],
                        [0.76, 0.19, 0.04, 0.01])
ccoc = Distortion.ccoc(0.1)
lifted = bodoff_book.price(ccoc, assets=100, allocation="lifted")
el = [u.loss for u in lifted.allocated]
total_epd, unit_epd = bodoff_book.epd(100)
plc_assets = bodoff_book.bodoff(assets=100)
shared = [a - e for a, e in zip(plc_assets, el)]
rows = [
    ("limited loss", *el, lifted.total.loss), ("limited loss, book", 19.3075, 4.5025, 23.81),
    ("EPD ratio", *unit_epd, total_epd), ("EPD ratio, book", 0.0249, 0.0995, 0.0399),
    ("PLC assets", *plc_assets, NAN),
    ("shared liability", *shared, NAN), ("shared liability, book", 61.2191, 14.9709, NAN),
]
(pair_table(pd.DataFrame(rows, columns=["measure", "Hu", "EQ", "total"]), {"Hu": 4, "EQ": 4, "total": 4})
 .tab_header("Bodoff's Thought Experiment 1", "Assets 100; PLC assets are Bodoff's shares of the assets"))
```

```text
                             Hu       EQ    total
limited loss            19.3075   4.5025  23.8100
limited loss, book      19.3075   4.5025    23.81
EPD ratio                0.0249   0.0995   0.0399
EPD ratio, book          0.0249   0.0995   0.0399
PLC assets              80.5266  19.4734         
shared liability        61.2191  14.9709         
shared liability, book  61.2191  14.9709
```

`Portfolio.bodoff` returns each unit's share of the assets; the share minus the
expected loss is the shared liability (Table 15.22). PLC gives no price. The book
splits each shared liability into margin and capital at the 10% return, so the
margin is `0.1 / 1.1` of it (Table 15.24). The natural allocation instead prices
the total with the CCoC distortion and allocates the premium (Tables 15.27, 15.29
and 15.30). The linear allocation uses assets of `a - 1e-9` (section 4: the
earthquake alone totals exactly 100):

```python
d = 0.1 / 1.1
plc_premium = [e + d * s for e, s in zip(el, shared)]
linear = bodoff_book.price(ccoc, assets=100 - 1e-9, allocation="linear")
l1, l2 = lifted.allocated
n1, n2 = linear.allocated
rows = [
    ("PLC", *plc_premium, sum(plc_premium), NAN, NAN),
    ("PLC, book", 24.8729, 5.8635, NAN, NAN, NAN),
    ("lifted", l1.premium, l2.premium, lifted.total.premium, l1.capital, l2.capital),
    ("lifted, book", 22.0749, 8.6615, 30.7364, 27.6739, 41.5898),
    ("linear", n1.premium, n2.premium, linear.total.premium, n1.capital, n2.capital),
    ("linear, book", 18.4568, 12.2796, NAN, -8.5070, 77.7707),
]
columns = ["method", "P Hu", "P EQ", "P total", "Q Hu", "Q EQ"]
(pair_table(pd.DataFrame(rows, columns=columns), {c: 4 for c in columns[1:]})
 .tab_header("Tables 15.27, 15.29 and 15.30: premium and capital by unit", "P is premium, Q is capital"))
```

```text
                 P Hu     P EQ  P total     Q Hu     Q EQ
PLC           24.8729   5.8635  30.7364                  
PLC, book     24.8729   5.8635                           
lifted        22.0749   8.6615  30.7364  27.6739  41.5898
lifted, book  22.0749   8.6615  30.7364  27.6739  41.5898
linear        18.4568  12.2796  30.7364  -8.5070  77.7707
linear, book  18.4568  12.2796            -8.507  77.7707
```

PLC charges the earthquake 5.86, close to its share of the expected loss. The
lifted allocation charges 8.66: the CCoC distortion puts weight 0.1 on the worst
outcome, and the earthquake is half of it. The linear allocation charges 12.28
and gives the hurricane a **negative** margin: it merges the earthquake-only
year (total 100) and the year with both events into one default event of weight
0.1364, which the earthquake dominates. The lifted allocation avoids this by
seeing through the cap, as PLC does.

Exercises 276 to 278 repeat this with a proportional hazard and a Wang transform
calibrated to the same total. The book prints the parameters 0.82156 and
0.209615; calibrated to the CCoC premium here they are 0.821554 and 0.209615. The
allocations use the printed ones:

```python
BOOK_276 = {  # printed parameter, lifted premium and capital by unit, linear premium by unit
    "ph": (Distortion.proportional_hazard(0.82156), 0.82156, (23.3342, 7.4020, 40.0416, 29.2223), (23.0517, 7.6844)),
    "wang": (Distortion.wang(0.209615), 0.209615, (24.0282, 6.7082, 46.8427, 22.4209), (23.9277, 6.8087)),
}
rows = []
for family, (dist, parameter, book_lifted, book_linear) in BOOK_276.items():
    found = bodoff_book.calibrate(family, assets=100, premium=lifted.total.premium)
    li = bodoff_book.price(dist, assets=100, allocation="lifted").allocated
    ln = bodoff_book.price(dist, assets=100 - 1e-9, allocation="linear").allocated
    rows.append((family, list(found.parameters.values())[0], li[0].premium, li[1].premium, li[0].capital,
                 li[1].capital, ln[0].premium, ln[1].premium))
    rows.append((f"{family}, book", parameter, *book_lifted, *book_linear))
columns = ["family", "parameter", "P Hu", "P EQ", "Q Hu", "Q EQ", "linear P Hu", "linear P EQ"]
(pair_table(pd.DataFrame(rows, columns=columns), {"parameter": 6, **{c: 4 for c in columns[2:]}})
 .tab_header("Exercises 276 to 278: proportional hazard and Wang, calibrated to the CCoC premium",
             "Parameter calibrated here; P is premium, Q is capital; lifted unless marked linear"))
```

```text
            parameter     P Hu    P EQ     Q Hu     Q EQ  linear P Hu  linear P EQ
ph           0.821554  23.3342  7.4020  40.0416  29.2223      23.0517       7.6844
ph, book      0.82156  23.3342   7.402  40.0416  29.2223      23.0517       7.6844
wang         0.209615  24.0282  6.7082  46.8427  22.4209      23.9277       6.8087
wang, book   0.209615  24.0282  6.7082  46.8427  22.4209      23.9277       6.8087
```

All values match to the fourth decimal, except the PH parameter in its sixth
digit. At the printed 0.82156 the total premium is 30.7361, the PH total the book
prints, so the book calibrated slightly off 30.7364 or rounded 0.821554 up.

### The summary table

Table 15.31 compares five methods. Two are classical: the expected value premium
(proportional to the limited expected loss) and the variance-covariance premium
(expected loss plus the margin split by each unit's covariance with the total).
The book uses the covariances of the **unlimited** losses, which
`capital(..., "covariance")` on the uncapped outcomes gives:

```python
premium, margin = 30.7364, 30.7364 - 23.81
expected_value = [premium * e / sum(el) for e in el]
uncapped = PredictiveDistribution(["unit"], [("Hu",), ("EQ",)],
                                  [[0.0, 0.0]] * 76 + [[99.0, 0.0]] * 19 + [[0.0, 100.0]] * 4 + [[99.0, 100.0]])
cov = capital(uncapped, ccoc, "covariance")
var_cov = [e + margin * c / cov.total for e, c in zip(el, cov.allocated)]
TABLE_1531 = [
    ("expected value", expected_value, (24.9241, 5.8123)),
    ("variance/covariance", var_cov, (24.6236, 6.1128)),
    ("Bodoff PLC", plc_premium, (24.8729, 5.8635)),
    ("lifted natural", [u.premium for u in lifted.allocated], (22.0749, 8.6615)),
    ("linear natural", [u.premium for u in linear.allocated], (18.4568, 12.2796)),
]
rows = []
for name, got, book in TABLE_1531:
    rows.append((name, *got, el[0] / got[0], el[1] / got[1]))
    rows.append((f"{name}, book", *book, NAN, NAN))
(pair_table(pd.DataFrame(rows, columns=["method", "Hu", "EQ", "LR Hu", "LR EQ"]),
            {"Hu": 4, "EQ": 4, "LR Hu": 3, "LR EQ": 3})
 .tab_header("Table 15.31: premium and loss ratio by unit and method",
             "Premium in currency units; loss ratio on the limited loss"))
```

```text
                                Hu       EQ  LR Hu  LR EQ
expected value             24.9241   5.8123  0.775  0.775
expected value, book       24.9241   5.8123              
variance/covariance        24.6236   6.1128  0.784  0.737
variance/covariance, book  24.6236   6.1128              
Bodoff PLC                 24.8729   5.8635  0.776  0.768
Bodoff PLC, book           24.8729   5.8635              
lifted natural             22.0749   8.6615  0.875  0.520
lifted natural, book       22.0749   8.6615              
linear natural             18.4568  12.2796  1.046  0.367
linear natural, book       18.4568  12.2796
```

The expected value, covariance and PLC premiums give the two perils almost the
same loss ratio. The lifted allocation charges the rarer, more volatile
earthquake a loss ratio of 0.52, and the cap misleads the linear allocation. The
book prefers the lifted allocation but says no single answer is correct.

```python
import numpy as np

methods = [name for name, _, _ in TABLE_1531]
x = np.arange(len(methods))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
for ax, values, ylabel, title in (
    (ax1, [got for _, got, _ in TABLE_1531], "premium", "Premium by unit"),
    (ax2, [(el[0] / got[0], el[1] / got[1]) for _, got, _ in TABLE_1531], "loss ratio on the limited loss",
     "Loss ratio by unit"),
):
    ax.bar(x - 0.2, [v[0] for v in values], width=0.4, label="Hu")
    ax.bar(x + 0.2, [v[1] for v in values], width=0.4, label="EQ")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
ax1.legend()
fig.tight_layout()
```

## 7. Pricing bounds, and stand-alone against diversified

The natural allocation depends on the distortion. `Portfolio.premium_bounds`
gives each unit's lowest and highest premium under the linear allocation over
**every** distortion that prices the capped total at the same premium. The
extremes are BiTVaRs, found exactly (section 11.2.3 and Remark 262):

```python
for row in bodoff_book.premium_bounds(30.7364, assets=100):
    print(f"{row['unit']}: {row['lower']:.4f} to {row['upper']:.4f}")
```

```text
Hu: 18.4568 to 24.9241
EQ: 5.8123 to 12.2796
```

The earthquake premium can be anything from 5.81 (the expected value premium) to
12.28 (the linear CCoC premium, itself an extreme because it puts all its extra
weight on the worst outcome). Every method in the summary table is in the range.
A split outside it cannot come from a spectral risk measure that prices the book
at 30.7364.

The diversification benefit: each unit priced alone at its own capital standard,
against the book as one portfolio.
`Portfolio.price_stand_alone` uses a VaR standard:

```python
alone = bodoff_book.price_stand_alone(ccoc, 0.99)
rows = [(f"{name} alone", u.assets, u.premium) for name, u in zip(alone.units, alone.stand_alone)]
rows += [("sum of the parts", alone.sum.assets, alone.sum.premium), ("together", alone.total.assets, alone.total.premium)]
(GT(pd.DataFrame(rows, columns=["case", "assets", "premium"]), rowname_col="case")
 .tab_header("Hurricane and earthquake priced alone and together", "CCoC 10%, VaR 99% standard; currency units")
 .fmt_number("assets", decimals=0, use_seps=False)
 .fmt_number("premium", decimals=4, use_seps=False))
```

```text
                  assets  premium
Hu alone              99  27.0000
EQ alone             100  13.6364
sum of the parts     199  40.6364
together             100  30.7364
```

Alone, each peril needs almost a full limit of assets. Together, assets of 100
support both and the premium falls from 40.64 to 30.74. The natural allocation
shares that saving between the units.

Next: [Price allocation in practice](pir-price-allocation-practice.md) for the case
studies and the reinsurance evaluation, and
[Risk loads and distortions](risk-loads-and-distortions.md) for how to select a
distortion.
