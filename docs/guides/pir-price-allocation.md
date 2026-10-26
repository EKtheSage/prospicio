# Price allocation in *Pricing Insurance Risk*: classical methods, the natural allocation and Bodoff

Part III of Stephen Mildenhall and John Major's *Pricing Insurance Risk: Theory
and Practice* (Wiley, 2022), chapters 12 to 15, asks one question. An insurer
prices its whole portfolio: how do you split that price over the units of the
portfolio? Chapter 12 gives the classical answers, which allocate capital first
and then price the capital. Chapter 13 applies them. Chapter 14 builds the
book's own answer, the **natural allocation** of a spectral risk measure.
Chapter 15 calculates it on small discrete examples, compares it with Bodoff's
percentile layer of capital, and applies it to the case studies.

This guide runs the small numerical examples of the book in prospicio. It prints
the value from the book next to the value from prospicio. The text and exhibits
are the authors'. This guide explains the methods in new words and calculates
the numbers again. Exercise, example and table numbers are those of the book.

The companion guide
[Price allocation in practice](pir-price-allocation-practice.md) runs the
case-study tables of chapters 13 and 15 and the reinsurance evaluation of
chapter 19.
[Pricing Insurance Risk: the case studies](pricing-insurance-risk-case-studies.md)
compares the case studies of the book with `aggregate` 0.30.1. If the words
distortion, CCoC or pentagon are new, read
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md) first.

## 1. The Simple Discrete Example

Most of chapter 12 uses the Simple Discrete Example of the book (its section
2.4.1). It has two independent units. `X1` takes 0, 8 or 10, and `X2` takes 0, 1
or 90, each with probabilities 1/2, 1/4 and 1/4. The probabilities of the nine
joint outcomes are multiples of 1/16, so sixteen equally likely rows give the
portfolio exactly. This is important because the allocation functions of
`prospicio.risk` work on equally likely simulations, a `PredictiveDistribution`:

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

The expected losses are 4.5, 22.75 and 27.25. `X2` is the dangerous unit: one
year in four it loses 90.

## 2. The classical recipes (chapter 12)

Chapter 12 starts from allocated constant cost of capital (CCoC) pricing. Each
unit gets some capital, each unit pays the same return on its capital, and the
premium follows. Exercise 203 shows the arithmetic. Three units each have an
expected loss of 100 and capital of 40, 80 and 200. At a 12% cost of capital,
their premiums are 104.8, 109.6 and 124.0. The result depends on the capital
allocation, and the chapter examines the recipes that actuaries used in the
1990s and 2000s.

On the Simple Discrete Example, the chapter allocates two amounts: the TVaR of
the total at 75%, which is 94.5, and a premium of 80.

`capital` takes a distortion (here TVaR) and a method:

- `"euler"` is the co-measure. For TVaR this is the coTVaR: the average loss of
  each unit in the years where the total is in its worst quarter.
- `"marginal"` is Merton and Perold's saving when you remove the unit.
- `"shapley"` is the Shapley value of the game whose cost is the TVaR of each
  sub-portfolio.
- `"proportional"` is the stand-alone measures, scaled to the total.

```python
BOOK_211 = {"stand-alone": (10, 90), "marginal": (4.5, 84.5), "shapley": (7.25, 87.25)}
BOOK_215 = {"euler": (4.5, 90)}  # Exercise 215, the coTVaR
tvar75 = Distortion.tvar(0.75)
print(f"TVaR 75% of the total: {capital(sims, tvar75).total}  (book 94.5)")
for method in ("euler", "marginal", "shapley", "proportional"):
    a = capital(sims, tvar75, method)
    book = {**BOOK_211, **BOOK_215}.get(method, ("-", "-"))
    print(f"{method:12s} X1 {a.allocated[0]:6.2f}  X2 {a.allocated[1]:6.2f}   book {book[0]}, {book[1]}")
print(f"stand-alone  X1 {a.standalone[0]:6.2f}  X2 {a.standalone[1]:6.2f}   book 10, 90")
```

```text
TVaR 75% of the total: 94.5  (book 94.5)
euler        X1   4.50  X2  90.00   book 4.5, 90
marginal     X1   4.50  X2  84.50   book 4.5, 84.5
shapley      X1   7.25  X2  87.25   book 7.25, 87.25
proportional X1   9.45  X2  85.05   book -, -
stand-alone  X1  10.00  X2  90.00   book 10, 90
```

These are Tables 12.5 and 12.8. The marginal method gives 4.5 and 84.5, which
add up to 89, not 94.5. No single unit causes the remaining 5.5, so the method
gives it to no unit. The coTVaR gives `X1` exactly its expected loss of 4.5, so
a price on that basis gives `X1` no margin. In the worst quarter of years `X2`
loses 90 each time, and the loss of `X1` there is not larger than usual. Shapley
averages the stand-alone and marginal views and gives 7.25 to `X1`.

The second half of Exercise 211 allocates a premium of 80. The book finds the
TVaR level that prices the total at 80, 0.6975. `Portfolio.calibrate` finds the
same level, and the remaining values follow:

```python
d80 = discrete.calibrate("tvar", premium=80.0)
print(f"TVaR level pricing the total at 80: {d80.parameters['p']:.4f}  (book 0.6975)")
BOOK_80 = {"stand-alone": (9.65, 74.54), "marginal": (5.46, 70.35), "shapley": (7.55, 72.45)}
for method in ("marginal", "shapley"):
    a = capital(sims, d80, method)
    print(f"{method:12s} X1 {a.allocated[0]:6.2f}  X2 {a.allocated[1]:6.2f}   book {BOOK_80[method]}")
print(f"stand-alone  X1 {a.standalone[0]:6.2f}  X2 {a.standalone[1]:6.2f}   book {BOOK_80['stand-alone']}")
sh = capital(sims, d80, "shapley").allocated
print(f"Shapley loss ratios: X1 {4.5 / sh[0]:.1%}, X2 {22.75 / sh[1]:.1%}  (book 59.6%, 31.4%)")
```

```text
TVaR level pricing the total at 80: 0.6975  (book 0.6975)
marginal     X1   5.46  X2  70.35   book (5.46, 70.35)
shapley      X1   7.55  X2  72.45   book (7.55, 72.45)
stand-alone  X1   9.65  X2  74.54   book (9.65, 74.54)
Shapley loss ratios: X1 59.6%, X2 31.4%  (book 59.6%, 31.4%)
```

The haircut and equal risk allocations (Exercises 205 and 206) use TVaR as an
auxiliary measure of each unit alone. The haircut splits the 80 in proportion to
the stand-alone TVaRs at 75%, 10 and 90. The equal risk allocation finds one
level `p` at which the two stand-alone TVaRs add up to 80. This prices each unit
as if the two units were comonotonic. The code calculates both from
`Sampled.tvar`, with a bisection for the level `p`:

```python
haircut = [80 * t / (x1.tvar(0.75) + x2.tvar(0.75)) for t in (x1.tvar(0.75), x2.tvar(0.75))]
print(f"haircut     X1 {haircut[0]:6.2f}  X2 {haircut[1]:6.2f}   book 8, 72;"
      f" loss ratios {4.5 / haircut[0]:.4f}, {22.75 / haircut[1]:.4f} (book 56.3%, 31.6%)")
lo, hi = 0.5, 0.75
for _ in range(60):  # bisection on the level p
    p = (lo + hi) / 2
    lo, hi = (lo, p) if x1.tvar(p) + x2.tvar(p) > 80 else (p, hi)
print(f"equal risk  X1 {x1.tvar(p):6.2f}  X2 {x2.tvar(p):6.2f}   book 9.56, 70.44;"
      f" level {p:.4f} (book 0.6796)")
```

```text
haircut     X1   8.00  X2  72.00   book 8, 72; loss ratios 0.5625, 0.3160 (book 56.3%, 31.6%)
equal risk  X1   9.56  X2  70.44   book 9.56, 70.44; level 0.6796 (book 0.6796)
```

Both match. The loss ratios show the difficulty of the classical recipes. The
recipes split the same premium of 80 between `X1` and `X2` as 8 and 72, 9.56 and
70.44, or 7.55 and 72.45. They do not tell which split is correct.

### The covariance allocation: two definitions

Exercise 217 allocates the premium of 80 by Kalkbrener's method, with the
standard deviation as the auxiliary measure. Each unit gets its expected loss
plus a share of the margin `80 - 27.25` in proportion to its covariance with the
total. The `"covariance"` method of prospicio uses a different formula,
`rho(S) Cov(X_j, S) / Var(S)`. It spreads the whole measure, expected loss
included, by covariance. The covariance shares are the same, so the code
calculates the book's version from them:

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

The two definitions give very different results here, because `X1` has almost no
correlation with the total (a covariance of 20.75 against a variance of 1,528).
When the method spreads all of 80 by covariance, `X1` gets 1.09, a quarter of
its expected loss. No insurer charges that price. The book's form, which
Exercise 217 uses, keeps the expected loss and allocates only the margin.

### Equal priority in default

Section 12.3 shows how an insurer pays when it cannot pay all claims. Under
equal priority, each claim gets the same fraction of the amount owed. Exercise
219 has assets of 1,200 against liabilities of 800, 600 and 400. A one-scenario
`Portfolio` priced with those assets reports the recovery of each line (its
`loss` is the payment under equal priority). `Portfolio.priority` puts the equal
rule next to the senior and junior rules:

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

Each line gets two thirds. The junior column shows the rule that equal priority
replaces. If A ranked behind B and C, A would get the 200 that B and C leave.

## 3. The natural allocation, layer by layer (chapter 14)

Chapter 14 changes the order of the argument. It does not allocate capital
first. It prices the portfolio with a spectral risk measure (a distortion `g`
applied to the survival function of the total) and allocates the premium
directly. The steps are:

1. Sort the outcomes by the **total** loss. The distortion changes the objective
   probability of each total into a risk-adjusted probability. The weight of
   total `x_k` is `g(S(x_k-)) - g(S(x_k))`, where `S` is the survival function.
   Bad totals get more weight than their probability, and good totals get less.
   The weights still add up to 1, and the premium is the expected total under
   these weights.
2. Calculate the premium of a unit as its expected loss under the **same**
   weights. The weights depend only on the total, so a unit needs only its
   average loss at each total, `kappa_i(x) = E[X_i | X = x]`. The `kappa` values
   add up to the total, so the unit premiums add up to the portfolio premium.
3. Read the result as layers. The insurer uses the layer of assets between two
   consecutive totals when the total reaches it. The premium for the layer is
   `g(S)` times its width, against an expected loss of `S` times its width. Each
   unit pays for the layer in proportion to its expected contribution to the
   losses that reach the layer. Each layer has its own return on capital, and
   the average return of a unit depends on the layers that it uses. Allocated
   CCoC pricing does the opposite: it fixes one return and allocates capital.

Example 258 runs the algorithm on an adjusted Simple Discrete Example. In it,
`X1` takes 9 instead of 8, so the total 10 has two causes (9 + 1 and 10 + 0).
The distortion is the proportional hazard `g(s) = s^0.5`:

```python
ADJ_ROWS = [[x1, x2] for x1 in (0.0, 9.0, 10.0) for x2 in (0.0, 1.0, 90.0)]
ADJ_PROBS = [p1 * p2 for p1 in (0.5, 0.25, 0.25) for p2 in (0.5, 0.25, 0.25)]
adjusted = Portfolio(["X1", "X2"], ADJ_ROWS, ADJ_PROBS)
ph = Distortion.proportional_hazard(0.5)
print("total  prob    kappa1  kappa2")
for x, p, k1, k2 in zip(adjusted.totals, adjusted.probs, adjusted.kappa("X1"), adjusted.kappa("X2")):
    print(f"{x:5.0f}  {p:.4f}  {k1:6.4f}  {k2:6.4f}")
priced = adjusted.price(ph, assets=100)
print("premium", [round(u.premium, 4) for u in priced.allocated], round(priced.total.premium, 4),
      " (book 6.2048, 45.1838, 51.3887)")
```

```text
total  prob    kappa1  kappa2
    0  0.2500  0.0000  0.0000
    1  0.1250  0.0000  1.0000
    9  0.1250  9.0000  0.0000
   10  0.1875  9.6667  0.3333
   11  0.0625  10.0000  1.0000
   90  0.1250  0.0000  90.0000
   99  0.0625  9.0000  90.0000
  100  0.0625  10.0000  90.0000
premium [6.2048, 45.1838] 51.3887  (book 6.2048, 45.1838, 51.3887)
```

The grouped table is Table 15.3 of the book (at a total of 10, the average of
`X1` is 9.6667 and the average of `X2` is 0.3333). The premiums are those of
Table 15.4 to four decimals.

### When totals tie: the natural allocation set

The `kappa` average is a choice. When two outcomes have the same total, the
distortion has no reason to rank one above the other. Each order of the tied
outcomes gives a valid allocation. The set of these allocations is the **natural
allocation set**, and the `kappa` average (the **linear natural allocation**) is
in it.

Example 259 gives the two extreme orders of the tie at 10: `X1` gets 6.2085 or
6.2009. Exercise 260 does the same for TVaR at 65%: `X1` gets 6.1429 or 6.2500,
and `X2` gets 64.5714 or 64.4643. The code calculates the linear allocation,
which is between the extremes:

```python
linear65 = [u.premium for u in adjusted.price(Distortion.tvar(0.65), assets=100).allocated]
print(f"TVaR 65%: X1 {linear65[0]:.4f} (book range 6.1429 to 6.2500),"
      f" X2 {linear65[1]:.4f} (book range 64.4643 to 64.5714)")
```

```text
TVaR 65%: X1 6.2143 (book range 6.1429 to 6.2500), X2 64.5000 (book range 64.4643 to 64.5714)
```

## 4. Limited assets: the linear and lifted allocations

With assets `a` below the largest loss, the insurer pays `X ∧ a`. Under equal
priority, unit `i` gets `X_i a / X` when the total is more than `a`. All default
outcomes then have the same capped total `a`. This is a tie of the type above,
so the natural allocation of the capped total is not unique. Chapter 14 selects
two members of the set:

- The **linear** allocation treats all default outcomes the same. It splits the
  weight of the default event by the expected share of each unit in the payments
  in default.
- The **lifted** allocation ranks the default outcomes by the uncapped total
  `X`, as if the distortion could see how bad each default is. The worst
  defaults get the most weight. This allocation measures the size of default,
  which is important to policyholders and regulators.

`Portfolio.price` takes `allocation="linear"` or `"lifted"`. Example 235 has ten
equally likely outcomes, assets of 10 and TVaR at 80%. Four of the ten totals
are more than 10:

```python
E235_X1 = [1, 2, 3, 4, 5, 6, 7, 8, 12, 10]
E235_X2 = [1, 2, 3, 4, 5, 9, 8, 12, 13, 20]
e235 = Portfolio(["X1", "X2"], [[float(a), float(b)] for a, b in zip(E235_X1, E235_X2)])
tvar80 = Distortion.tvar(0.8)
for allocation, book in (("lifted", "4.067, 5.933"), ("linear", "4.3, 5.7")):
    for assets in (10, 10 - 1e-9):
        got = [u.premium for u in e235.price(tvar80, assets=assets, allocation=allocation).allocated]
        print(f"{allocation:6s} assets {assets:12.9f}: X1 {got[0]:.4f}  X2 {got[1]:.4f}   book {book}")
```

```text
lifted assets 10.000000000: X1 4.0667  X2 5.9333   book 4.067, 5.933
lifted assets  9.999999999: X1 4.0667  X2 5.9333   book 4.067, 5.933
linear assets 10.000000000: X1 4.1600  X2 5.8400   book 4.3, 5.7
linear assets  9.999999999: X1 4.3000  X2 5.7000   book 4.3, 5.7
```

The lifted allocation agrees with the book at assets of 10. It gives more to
`X2` (5.933), because `X2` dominates the two worst outcomes. The linear
allocation agrees with the book only when the assets are slightly below 10. The
cause is one outcome whose total is exactly 10, the asset level.

The book's linear allocation puts this outcome in the default outcomes, because
its capped total is equal to `a`. prospicio follows `aggregate` 1.0.1: it pays
an outcome at exactly the asset level in full and splits the default weight only
over outcomes above `a`. Both allocations are members of the natural allocation
set and give the same total price. They split the total differently only when
the asset level is a possible total with positive probability. On a fine grid
the atom at `a` is very small, and so is the difference.

On the small examples of the book, assets of `a - 1e-9` give the book's linear
numbers.

Example 263 prices the adjusted Simple Discrete Example with assets of 80, so
three outcomes default. Example 272 has two independent units priced at several
asset levels (its Table 15.20):

```python
got = adjusted.price(ph, assets=80, allocation="lifted").allocated
print(f"assets 80, lifted: loss {got[0].loss:.4f}, {got[1].loss:.4f}"
      f"  premium {got[0].premium:.4f}, {got[1].premium:.4f}")
print("            book: loss 4.5170, 19.2955  premium 5.5260, 37.4307")

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
print("  a  lifted X1        X2  linear X1        X2     total")
largest = 0.0
for a, book in BOOK_1520.items():
    lifted = two.price(ph87, assets=a, allocation="lifted")
    linear = two.price(ph87, assets=a - 1e-9, allocation="linear")
    got = [u.premium for u in lifted.allocated] + [u.premium for u in linear.allocated]
    got.append(lifted.total.premium)
    largest = max(largest, *(abs(g - b) for g, b in zip(got, book)))
    print(f"{a:3d}  " + "  ".join(f"{g:8.4f}" for g in got))
print(f"largest gap to the book's thirty numbers: {largest:.1e}")
```

```text
assets 80, lifted: loss 4.5170, 19.2955  premium 5.5260, 37.4307
            book: loss 4.5170, 19.2955  premium 5.5260, 37.4307
  a  lifted X1        X2  linear X1        X2     total
 10    3.5622    4.2236    3.6983    4.0875    7.7858
 20    5.9672    7.2901    6.2183    7.0390   13.2573
 30    6.8419   10.0505    6.8350   10.0574   16.8924
 40    7.2890   12.5971    7.0972   12.7889   19.8861
 50    7.7361   13.7880    7.6846   13.8395   21.5241
 60    8.0348   14.3855    8.0348   14.3855   22.4203
largest gap to the book's thirty numbers: 4.6e-05
```

All thirty numbers of Table 15.20 match to four decimals. Where the two
allocations differ, the lifted allocation puts more on `X2` at assets of 20 and 30.
There, the loss of 40 of `X2` makes the worst defaults. At assets of 50 only
the outcome of 60 defaults, and at 60 no outcome defaults, so there the two
allocations agree.

## 5. Margin and capital by unit (section 15.2)

The layer view gives each unit a margin and a capital in each layer of assets.
The margin is the premium density minus the expected loss density. In each
layer, capital and margin have the ratio that the distortion sets for that
layer, `(1 - g(S)) / (g(S) - S)`. The capital of a unit is the sum over layers,
and its return is its margin divided by its capital. `capital` on the lifted
allocation is this sum (section 15.2.5, Tables 15.14 and 15.15):

```python
print("assets  margin X1  margin X2  capital X1  capital X2  return X1  return X2")
for a in (100, 80, 11):
    u1, u2 = adjusted.price(ph, assets=a, allocation="lifted").allocated
    m1, m2 = u1.premium - u1.loss, u2.premium - u2.loss
    print(f"{a:6d}  {m1:9.4f}  {m2:9.4f}  {u1.capital:10.4f}  {u2.capital:10.4f}"
          f"  {m1 / u1.capital:9.4f}  {m2 / u2.capital:9.4f}")
print("book: a = 100 margins 1.4548, 22.4338, capital 3.5274, 45.0839, returns 41.24%, 49.76%;")
print("      a = 80 margins 1.0089, 18.1353, returns 41.51%, 52.39%; a = 11 margins -0.5423, 2.4365")
```

```text
assets  margin X1  margin X2  capital X1  capital X2  return X1  return X2
   100     1.4548    22.4338      3.5274     45.0839     0.4124     0.4976
    80     1.0089    18.1353      2.4305     34.6128     0.4151     0.5239
    11    -0.5423     2.4365     -0.6720      3.2153     0.8070     0.7578
book: a = 100 margins 1.4548, 22.4338, capital 3.5274, 45.0839, returns 41.24%, 49.76%;
      a = 80 margins 1.0089, 18.1353, returns 41.51%, 52.39%; a = 11 margins -0.5423, 2.4365
```

All values that the book prints match. With assets of 11, `X1` has a negative
margin. Default is so probable that `X2` takes payments from `X1` most of the
time, and `X1` must get compensation because it shares its recoveries. The
returns are different for each unit (41% against 50% at full assets), because
`X2` uses the expensive high layers that `X1` uses very little.

## 6. Bodoff's percentile layer of capital (section 15.3)

Bodoff (2007) also allocates capital layer by layer, but by **objective**
probabilities. The units share each layer of assets in proportion to their
expected share of the losses that reach it. Section 15.3 runs his Thought
Experiment 1: a hurricane that costs 99 with probability 0.2 and an independent
earthquake that costs 100 with probability 0.05. The assets are 100, with equal
priority and a 10% cost of capital.

```python
bodoff_book = Portfolio(["Hu", "EQ"], [[0.0, 0.0], [99.0, 0.0], [0.0, 100.0], [99.0, 100.0]],
                        [0.76, 0.19, 0.04, 0.01])
ccoc = Distortion.ccoc(0.1)
lifted = bodoff_book.price(ccoc, assets=100, allocation="lifted")
el = [u.loss for u in lifted.allocated]
total_epd, unit_epd = bodoff_book.epd(100)
print(f"limited loss {el[0]:.4f}, {el[1]:.4f}, total {lifted.total.loss:.4f}  (book 19.3075, 4.5025, 23.8100)")
print(f"EPD ratio {unit_epd[0]:.4f}, {unit_epd[1]:.4f}, total {total_epd:.4f}  (book 0.0249, 0.0995, 0.0399)")
plc_assets = bodoff_book.bodoff(assets=100)
shared = [a - e for a, e in zip(plc_assets, el)]
print(f"PLC assets {plc_assets[0]:.4f}, {plc_assets[1]:.4f}; shared liability {shared[0]:.4f}, {shared[1]:.4f}"
      "  (book 61.2191, 14.9709)")
```

```text
limited loss 19.3075, 4.5025, total 23.8100  (book 19.3075, 4.5025, 23.8100)
EPD ratio 0.0249, 0.0995, total 0.0399  (book 0.0249, 0.0995, 0.0399)
PLC assets 80.5266, 19.4734; shared liability 61.2191, 14.9709  (book 61.2191, 14.9709)
```

`Portfolio.bodoff` returns the share of the assets for each unit. The share
minus the expected loss is what the book calls the shared liability (Table
15.22). PLC gives no price. To get a price, the book splits the shared liability
of each unit into margin and capital at the 10% return. The margin is then
`0.1 / 1.1` of it (Table 15.24). The natural allocation prices with the CCoC
distortion and allocates the premium (Tables 15.27, 15.29 and 15.30):

```python
d = 0.1 / 1.1
plc_premium = [e + d * s for e, s in zip(el, shared)]
print(f"PLC premium     {plc_premium[0]:8.4f} {plc_premium[1]:8.4f}   book 24.8729, 5.8635")
linear = bodoff_book.price(ccoc, assets=100 - 1e-9, allocation="linear")
for name, pr, book in (("lifted", lifted, "22.0749, 8.6615; capital 27.6739, 41.5898"),
                       ("linear", linear, "18.4568, 12.2796; capital -8.5070, 77.7707")):
    u1, u2 = pr.allocated
    print(f"{name:6s} premium {u1.premium:8.4f} {u2.premium:8.4f}  capital {u1.capital:8.4f} {u2.capital:8.4f}"
          f"   book {book}")
print(f"total premium {lifted.total.premium:.4f}  (book 30.7364)")
```

```text
PLC premium      24.8729   5.8635   book 24.8729, 5.8635
lifted premium  22.0749   8.6615  capital  27.6739  41.5898   book 22.0749, 8.6615; capital 27.6739, 41.5898
linear premium  18.4568  12.2796  capital  -8.5070  77.7707   book 18.4568, 12.2796; capital -8.5070, 77.7707
total premium 30.7364  (book 30.7364)
```

The linear allocation uses assets of `a - 1e-9`, for the reason in section 4
(the earthquake alone totals exactly 100). The three answers are very different.
PLC charges the earthquake 5.86, close to its share of the expected loss. The
lifted allocation charges 8.66: the CCoC distortion puts a weight of 0.1 on the
worst outcome, and the earthquake is half of that outcome.

The linear allocation charges 12.28 and gives the hurricane a **negative**
margin. It puts the earthquake-only year (total 100, all earthquake) and the
year with both events into one default event of weight 0.1364. The earthquake
dominates that event. Bodoff designed PLC to prevent this problem, and the
lifted allocation prevents it because it sees through the cap.

The book repeats the calculation with a proportional hazard and a Wang transform
calibrated to the same total (Exercises 276 to 278). The book prints the
parameters 0.82156 and 0.209615. Calibrated here to the CCoC premium, they are
0.821554 and 0.209615. The allocations below use the printed parameters:

```python
for family, book_parameter in (("ph", 0.82156), ("wang", 0.209615)):
    found = bodoff_book.calibrate(family, assets=100, premium=lifted.total.premium)
    print(f"{family}: calibrated {list(found.parameters.values())[0]:.6f} (book {book_parameter})")
BOOK_276 = {  # lifted premium and capital by unit, linear premium by unit
    "ph": (Distortion.proportional_hazard(0.82156), (23.3342, 7.4020, 40.0416, 29.2223), (23.0517, 7.6844)),
    "wang": (Distortion.wang(0.209615), (24.0282, 6.7082, 46.8427, 22.4209), (23.9277, 6.8087)),
}
for family, (dist, book_lifted, book_linear) in BOOK_276.items():
    li = bodoff_book.price(dist, assets=100, allocation="lifted").allocated
    ln = bodoff_book.price(dist, assets=100 - 1e-9, allocation="linear").allocated
    print(f"{family:4s} lifted P {li[0].premium:.4f} {li[1].premium:.4f} Q {li[0].capital:.4f} {li[1].capital:.4f}"
          f"  linear P {ln[0].premium:.4f} {ln[1].premium:.4f}")
    print(f"     book   P {book_lifted[0]:.4f} {book_lifted[1]:.4f} Q {book_lifted[2]:.4f} {book_lifted[3]:.4f}"
          f"  linear P {book_linear[0]:.4f} {book_linear[1]:.4f}")
```

```text
ph: calibrated 0.821554 (book 0.82156)
wang: calibrated 0.209615 (book 0.209615)
ph   lifted P 23.3342 7.4020 Q 40.0416 29.2223  linear P 23.0517 7.6844
     book   P 23.3342 7.4020 Q 40.0416 29.2223  linear P 23.0517 7.6844
wang lifted P 24.0282 6.7082 Q 46.8427 22.4209  linear P 23.9277 6.8087
     book   P 24.0282 6.7082 Q 46.8427 22.4209  linear P 23.9277 6.8087
```

All printed values match to the fourth decimal. The only difference is the PH
parameter, in its sixth digit. At the printed 0.82156 the total premium is
30.7361, the total that the book prints for PH. So the book calibrated its
parameter slightly off the CCoC premium of 30.7364, or rounded 0.821554 up.

### The summary table

Table 15.31 shows five methods side by side. Two more methods are classical. The
expected value premium is in proportion to the limited expected loss. The
variance-covariance premium is the expected loss plus the margin, split by the
covariance of each unit with the total. For this premium the book uses the
covariances of the **unlimited** losses, and `capital(..., "covariance")` on the
uncapped outcomes gives those shares:

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
print(f"{'':20s} {'Hu':>8s} {'EQ':>8s}  {'LR Hu':>6s} {'LR EQ':>6s}   book")
for name, got, book in TABLE_1531:
    print(f"{name:20s} {got[0]:8.4f} {got[1]:8.4f}  {el[0] / got[0]:6.3f} {el[1] / got[1]:6.3f}   {book[0]}, {book[1]}")
```

```text
                           Hu       EQ   LR Hu  LR EQ   book
expected value        24.9241   5.8123   0.775  0.775   24.9241, 5.8123
variance/covariance   24.6236   6.1128   0.784  0.737   24.6236, 6.1128
Bodoff PLC            24.8729   5.8635   0.776  0.768   24.8729, 5.8635
lifted natural        22.0749   8.6615   0.875  0.520   22.0749, 8.6615
linear natural        18.4568  12.2796   1.046  0.367   18.4568, 12.2796
```

The loss ratios are on the limited loss, as in Table 15.31 of the book. The
expected value, covariance and PLC premiums give almost the same loss ratio to
the two perils. The lifted allocation charges the rarer, more volatile
earthquake a loss ratio of 0.52. The cap misleads the linear allocation. The
book concludes that no single answer is correct, and it prefers the lifted
allocation.

## 7. Pricing bounds, and stand-alone against diversified

The natural allocation depends on the distortion. `Portfolio.premium_bounds`
shows how much that choice can move a unit. It gives the lowest and highest
premium of each unit under the linear allocation. The range covers **every**
distortion that prices the capped total at the same premium. The extremes are
BiTVaRs, found exactly (section 11.2.3 and Remark 262 of the book refer to
this):

```python
for row in bodoff_book.premium_bounds(30.7364, assets=100):
    print(f"{row['unit']}: {row['lower']:.4f} to {row['upper']:.4f}")
```

```text
Hu: 18.4568 to 24.9241
EQ: 5.8123 to 12.2796
```

The earthquake premium can be any value from 5.81 (the expected value premium of
Table 15.31) to 12.28 (the linear CCoC premium). On this book the CCoC
distortion, which puts all its extra weight on the worst outcome, is itself an
extreme. All methods in the summary table are in the range. A split outside the
bounds cannot come from a spectral risk measure that prices the whole book at
30.7364.

The book also compares the diversification benefit: each unit priced alone, at
its own capital standard, against the book priced as one portfolio.
`Portfolio.price_stand_alone` does this at a VaR standard:

```python
alone = bodoff_book.price_stand_alone(ccoc, 0.99)
for name, u in zip(alone.units, alone.stand_alone):
    print(f"{name} alone: assets {u.assets:.0f}, premium {u.premium:.4f}")
print(f"sum of the parts {alone.sum.premium:.4f}; together {alone.total.premium:.4f}")
```

```text
Hu alone: assets 99, premium 27.0000
EQ alone: assets 100, premium 13.6364
sum of the parts 40.6364; together 30.7364
```

Alone, each peril needs almost a full limit of assets, 99 and 100, and charges
for it. Together, assets of 100 support both perils, and the premium decreases
from 40.64 to 30.74. The natural allocation shares that saving between the
units.

Where to go next:
[Price allocation in practice](pir-price-allocation-practice.md) for the
case-study tables of chapters 13 and 15 and the ceded reinsurance evaluation of
chapter 19. [Risk loads and distortions](risk-loads-and-distortions.md) for how
to select a distortion.
