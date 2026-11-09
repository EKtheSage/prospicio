# Capital adequacy and the natural allocation: CAS Monograph 15, chapters 3 and 4

<!-- site: execute -->

CAS Monograph 15, *Introduction to Capital Modeling and Portfolio Management*
(John Major and Stephen Mildenhall, 2026), builds its theory on one small
company, InsCo. This guide recomputes the tables of chapter 3 (capital
adequacy) and chapter 4 (pricing and allocation), the monograph's value beside
prospicio's. Table numbers are the monograph's. Money is in its units.

For the theory, read Mildenhall and Major's *Pricing Insurance Risk*. The
[case studies guide](pricing-insurance-risk-case-studies.md) runs its examples.

## 1. InsCo and the four adjustments

InsCo writes units A, B and C. Its capital model is ten equally likely years
(Table 2.1), each cell a unit's loss. Plan premiums are 13.9, 18.7 and 19.6:

```python
import pandas as pd
from great_tables import GT

from prospicio.distributions import Sampled
from prospicio.pricing import Pentagon, Portfolio
from prospicio.risk import Distortion

UNITS = ["A", "B", "C"]
ROWS = [
    [5, 20, 11], [7, 33, 0], [15, 13, 0], [15, 7, 0], [13, 20, 7],
    [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64],
]
PLAN = [13.9, 18.7, 19.6]
insco = Portfolio(UNITS, ROWS)
totals = [float(sum(row)) for row in ROWS]
total = Sampled(totals)

kappa = pd.DataFrame({
    "total": [f"{x:.0f}" for x in insco.totals] + ["E[X]"],
    "p": list(insco.probs) + [None],
    **{u: [insco.kappa(u)[j] for j in range(len(insco.totals))] + [m]
       for u, m in zip(UNITS, insco.expected())},
})
(GT(kappa, rowname_col="total")
 .tab_header("Table 4.2 without its zero row: InsCo's seven totals",
             "p is the probability of the total; A, B, C are each unit's expected loss given it")
 .fmt_number(["p", "A", "B", "C"], decimals=1)
 .sub_missing(missing_text=""))
```

```text
        p     A     B     C
22    0.1  15.0   7.0   0.0
28    0.1  15.0  13.0   0.0
36    0.1   5.0  20.0  11.0
40    0.4  10.0  24.0   6.0
55    0.1  26.0  19.0  10.0
65    0.1  17.0   8.0  40.0
100   0.1  16.0  20.0  64.0
E[X]       13.4  18.3  14.9
```

Section 4.3 changes the raw simulation output four ways before pricing:

1. Sort the years by total.
2. Start from a total of zero.
3. Cap the losses at the assets.
4. Make the totals distinct.

`Portfolio` makes all four changes when you build it. It needs no zero row,
which only starts the layer sums. Nothing is capped, because the assets of 100
cover the worst year. The four years that total 40 become one scenario of
probability 0.4, each unit at its average over those years (the conditional
expectation `kappa`), so the allocation does not depend on the order of years
with an equal total.

Algorithm 3.4.1 gives the expected loss two ways: the probability-weighted sum
of the totals, and the sum of the layers between totals, each weighted by the
probability that the total reaches it (the survival function).

```python
xs = [0.0] + insco.totals                       # pad with a zero outcome
ps = [0.0] + insco.probs
survival = [1.0 - sum(ps[: j + 1]) for j in range(len(xs))]
by_outcome = sum(x * p for x, p in zip(xs, ps))
by_layer = sum(s * (xs[j + 1] - xs[j]) for j, s in enumerate(survival[:-1]))
print(f"E[X] by outcomes {by_outcome:.1f}, by layers {by_layer:.1f}")
```

```text
E[X] by outcomes 46.6, by layers 46.6
```

## 2. Capital adequacy: what a loss level means

Section 3.1 reads the balance sheet as a ladder of thresholds. The plan
premium is 52.2 and the assets are 100: 27.8 of equity and 20 of debt with 0.6
of coupon due (Table 3.1):

```python
LEVELS = [  # event, loss at which it happens, the book's probability
    ("zero profit", 52.2, 0.30),
    ("shareholders wiped out", 52.2 + 27.8, 0.10),
    ("coupon not paid", 52.2 + 27.8 + 0.6, 0.10),
    ("bonds not redeemed", 100.0, 0.10),
]
levels = pd.DataFrame({
    "event": [event for event, _, _ in LEVELS],
    "loss": [level for _, level, _ in LEVELS],
    "prospicio": [sum(1 for t in totals if t >= level) / len(totals) for _, level, _ in LEVELS],
    "book": [book for _, _, book in LEVELS],
})
(GT(levels, rowname_col="event")
 .tab_header("Table 3.1: the thresholds of InsCo's balance sheet",
             "Loss at which each happens; probability that a year reaches it")
 .cols_label(loss="loss level")
 .fmt_number("loss", decimals=1)
 .fmt_percent(["prospicio", "book"], decimals=0))
```

```text
                        loss level  prospicio  book
zero profit                   52.2        30%   30%
shareholders wiped out        80.0        10%   10%
coupon not paid               80.6        10%   10%
bonds not redeemed           100.0        10%   10%
```

```python
print(f"TVaR at 99%: {total.tvar(0.99):.1f} (book 100.0)")
```

```text
TVaR at 99%: 100.0 (book 100.0)
```

Each threshold is a possible VaR standard. Every standard above 90% gives the
worst year, so the monograph calls the assets of 100 a TVaR 99% standard, or
full collateral. No default occurs.

## 3. VaR and TVaR

VaR at level p is the smallest loss not exceeded with probability p. TVaR is
the average of the worst 1 - p of outcomes, with a fraction of the boundary
outcome when 1 - p of the years is not a whole number. Algorithms 3.4.2 and
3.4.3 compute them by sorting and counting. They match `Sampled.quantile` and
`Sampled.tvar` at all but two levels:

```python
import math


def var_sorted(xs, p):
    """Algorithm 3.4.2: the floor(n p)-th smallest outcome, counting from 0."""
    xs = sorted(xs)
    return xs[min(math.floor(len(xs) * p), len(xs) - 1)]


def tvar_sorted(xs, p):
    """Algorithm 3.4.3: the average of the largest n (1 - p) outcomes."""
    xs, n = sorted(xs), len(xs)
    k = math.floor(n * p)
    if k + 1 >= n:
        return xs[-1]
    whole = sum(xs[k + 1 :])
    part = xs[k] * ((k + 1) - n * p)
    return (whole + part) / (n * (1 - p))


levels_p = (0.0, 0.25, 0.5, 0.7, 0.75, 0.85, 0.9, 0.95)
var_tvar = pd.DataFrame({
    "p": [f"{p:.2f}" for p in levels_p],
    "var_alg": [var_sorted(totals, p) for p in levels_p],
    "var_sampled": [total.quantile(p) for p in levels_p],
    "tvar_alg": [tvar_sorted(totals, p) for p in levels_p],
    "tvar_sampled": [total.tvar(p) for p in levels_p],
})
(GT(var_tvar, rowname_col="p")
 .tab_header("VaR and TVaR of InsCo's total: the algorithms against Sampled")
 .cols_label(var_alg="VaR, algorithm", var_sampled="VaR, Sampled",
             tvar_alg="TVaR, algorithm", tvar_sampled="TVaR, Sampled")
 .fmt_number(["var_alg", "var_sampled"], decimals=1)
 .fmt_number(["tvar_alg", "tvar_sampled"], decimals=3))
```

```text
      VaR, algorithm  VaR, Sampled  TVaR, algorithm  TVaR, Sampled
0.00            22.0          22.0           46.600         46.600
0.25            36.0          36.0           53.067         53.067
0.50            40.0          40.0           60.000         60.000
0.70            55.0          40.0           73.333         73.333
0.75            55.0          55.0           77.000         77.000
0.85            65.0          65.0           88.333         88.333
0.90           100.0          65.0          100.000        100.000
0.95           100.0         100.0          100.000        100.000
```

They differ at p = 0.7 and 0.9, where `n p` is a whole number and falls on a
jump of the distribution (the total is at most 40 in exactly 70% of the
years). The monograph's text defines VaR as the smallest p quantile (40 at 0.7,
65 at 0.9), which `Sampled` returns. The algorithm, read literally, takes the
next outcome up. TVaR is the same under both.

TVaR at 0 is the mean, and TVaR at 1 is the largest outcome. The monograph
prefers TVaR because it is subadditive (the TVaR of a sum is at most the sum of
the TVaRs) and VaR is not. Its examples: a discrete one with outcome
probabilities 0.9, 0.05 and 0.05 (here twenty equally likely years) and two
independent standard exponentials at p = 0.7.

```python
unit1 = [0.0] * 18 + [0.0, 10.0]
unit2 = [0.0] * 18 + [10.0, 0.0]
both = [a + b for a, b in zip(unit1, unit2)]
discrete = pd.DataFrame({
    "case": ["unit 1", "unit 2", "sum"],
    "var90": [Sampled(v).quantile(0.9) for v in (unit1, unit2, both)],
    "alg90": [var_sorted(v, 0.9) for v in (unit1, unit2, both)],
    "var91": [Sampled(v).quantile(0.91) for v in (unit1, unit2, both)],
    "tvar90": [Sampled(v).tvar(0.9) for v in (unit1, unit2, both)],
})
(GT(discrete, rowname_col="case")
 .tab_header("The discrete example: VaR is not subadditive, TVaR is")
 .cols_label(var90="VaR(0.9)", alg90="VaR(0.9), algorithm 3.4.2", var91="VaR(0.91)",
             tvar90="TVaR(0.9)")
 .fmt_number(["var90", "alg90", "var91", "tvar90"], decimals=1))
```

```text
        VaR(0.9)  VaR(0.9), algorithm 3.4.2  VaR(0.91)  TVaR(0.9)
unit 1       0.0                        0.0        0.0        5.0
unit 2       0.0                        0.0        0.0        5.0
sum          0.0                       10.0       10.0       10.0
```

```python
from prospicio.distributions import Gamma

one, two = Gamma(1.0, 1.0).quantile(0.7), Gamma(2.0, 1.0).quantile(0.7)
exponentials = pd.DataFrame({
    "case": ["one exponential", "twice that", "sum of two"],
    "prospicio": [one, 2 * one, two],
    "book": [1.204, 2.408, 2.439],
})
(GT(exponentials, rowname_col="case")
 .tab_header("VaR(0.7) of independent standard exponentials")
 .fmt_number(["prospicio", "book"], decimals=3))
```

```text
                 prospicio   book
one exponential      1.204  1.204
twice that           2.408  2.408
sum of two           2.439  2.439
```

In both examples, the VaR of the sum exceeds the sum of the VaRs. In the
discrete example the total is zero in 90% of the years, so the smallest 90%
quantile of the sum is 0. The monograph's failure case (0 for each unit, 10 in
total) is the algorithm's reading at 0.9, and every definition gives it just
above 0.9. TVaR adds up exactly in the first example and is less than the sum
in the second.

## 4. Risk analysis: marginal assets

Section 3.3 asks how each unit drives the assets. For a TVaR standard, the
coTVaR (the derivative of the assets with respect to a unit's volume) is the
unit's average loss in the tail that defines the TVaR. At 99% that tail is the
worst year, so the coTVaRs are the unit losses in that year, and the linear
natural allocation gives the same split. Capital on plan is the coTVaR less the
plan premium (Table 3.2). The VaR(0.85) rows are Table 3.3, the marginal split
of the 85% VaR of 65:

```python
priced = insco.price(Distortion.ccoc(0.15), assets=100)
co_tvar = [u.assets for u in priced.allocated]
BOOK_CAPITAL, BOOK_VAR85 = [2.1, 1.3, 44.4, 47.8], [17, 8, 40, 65]
co_tvar += [sum(co_tvar)]
plans = PLAN + [sum(PLAN)]
var85 = [insco.kappa(u)[5] for u in UNITS] + [65.0]
rows = []
for i, unit in enumerate(UNITS + ["total"]):
    rows.append({"unit": unit, "co_tvar": co_tvar[i], "plan": plans[i],
                 "capital": co_tvar[i] - plans[i], "var85": var85[i]})
    rows.append({"unit": f"{unit}, book", "capital": BOOK_CAPITAL[i], "var85": BOOK_VAR85[i]})
(GT(pd.DataFrame(rows), rowname_col="unit")
 .tab_header("Tables 3.2 and 3.3: capital on plan and the 85% VaR split",
             "Monograph units")
 .cols_label(co_tvar="coTVaR", plan="plan premium", capital="capital on plan",
             var85="VaR(0.85)")
 .fmt_number(["co_tvar", "plan", "capital"], decimals=1)
 .fmt_number("var85", rows=[0, 2, 4, 6], decimals=1)
 .fmt_number("var85", rows=[1, 3, 5, 7], decimals=0)
 .sub_missing(missing_text=""))
```

```text
             coTVaR  plan premium  capital on plan  VaR(0.85)
A              16.0          13.9              2.1       17.0
A, book                                        2.1         17
B              20.0          18.7              1.3        8.0
B, book                                        1.3          8
C              64.0          19.6             44.4       40.0
C, book                                       44.4         40
total         100.0          52.2             47.8       65.0
total, book                                   47.8         65
```

The 85% VaR split is one simulated year and is fragile: unit B loses 8 in that
year but 19 and 20 in the years on each side. The monograph's remedies are the
conditional expectations of section 1 and a range VaR (an average between two
levels). Chapter 4 takes another way out: risk-adjusted probabilities.

## 5. Pricing the whole

Chapter 4 prices from the top down. Investors want 15% on their capital. With
assets of 100 and an expected loss of 46.6, that sets the premium. The
monograph's pentagon of loss, margin, premium, capital and assets (Table 4.1)
is `Pentagon`; `Pentagon.solve` finds it from any three quantities:

```python
whole = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
d = 0.15 / 1.15
pentagon = pd.DataFrame({
    "quantity": ["premium", "margin", "capital", "risk discount d", "(1 - d) L + d a",
                 "loss ratio", "premium to capital", "plan premium less premium"],
    "prospicio": [whole.premium, whole.margin, whole.capital, whole.discount,
                  (1 - d) * 46.6 + d * 100, whole.loss_ratio, whole.premium_to_capital,
                  52.2 - whole.premium],
    "book": [53.565, None, None, 0.13043, None, None, None, -1.4],
})
(GT(pentagon, rowname_col="quantity")
 .tab_header("Table 4.1: InsCo's pentagon at a 15% return",
             "Loss 46.6, assets 100; monograph units, ratios as shown")
 .fmt_number("prospicio", rows=[0, 1, 2, 4, 5, 6, 7], decimals=3)
 .fmt_number("prospicio", rows=[3], decimals=5)
 .fmt_number("book", decimals=5, drop_trailing_zeros=True)
 .sub_missing(missing_text=""))
```

```text
                           prospicio     book
premium                       53.565   53.565
margin                         6.965
capital                       46.435
risk discount d              0.13043  0.13043
(1 - d) L + d a               53.565
loss ratio                     0.870
premium to capital             1.154
plan premium less premium     -1.365     -1.4
```

The premium is a weighted average of the expected loss and the assets, with the
risk discount `d = r / (1 + r)` as the weight on the assets. The plan is 1.365
short of what investors need; the companion guide finds the units that cause it.

## 6. Pricing layers in total

Section 4.4 splits the 100 of assets into layers between consecutive totals.
A layer pays in full when the total exceeds its bottom, so it is a bet with one
parameter, the exceedance probability `S`. With a rate on line `g(S)` for a
distortion `g`, the premium is the sum of `g(S)` times the layer width, the
distorted form of the layer sum in section 1. For the constant cost of capital
(CCoC) at 15%, `g(s) = d + (1 - d) s` for any `s > 0` (Table 4.4):

```python
ccoc = Distortion.ccoc(0.15)
widths = [xs[j + 1] - xs[j] for j in range(len(xs) - 1)]
layer_loss = sum(s * w for s, w in zip(survival, widths))
layer_premium = sum(ccoc.g(s) * w for s, w in zip(survival, widths))
layers = pd.DataFrame({
    "from": [f"{x:.0f}" for x in xs[:-1]] + ["total", "total, book"],
    "width": widths + [None, None],
    "S": survival[: len(widths)] + [None, None],
    "gS": [ccoc.g(s) for s in survival[: len(widths)]] + [None, None],
    "loss": [s * w for s, w in zip(survival, widths)] + [layer_loss, 46.6],
    "premium": [ccoc.g(s) * w for s, w in zip(survival, widths)] + [layer_premium, 53.565],
})
(GT(layers, rowname_col="from")
 .tab_header("Table 4.4: InsCo's layers priced by CCoC at 15%",
             "A layer starts at its bottom total; monograph units")
 .cols_label(gS="g(S)")
 .fmt_number("width", decimals=0)
 .fmt_number("S", decimals=1)
 .fmt_number(["gS", "loss", "premium"], decimals=3)
 .sub_missing(missing_text=""))
```

```text
             width    S   g(S)    loss  premium
0               22  1.0  1.000  22.000   22.000
22               6  0.9  0.913   5.400    5.478
28               8  0.8  0.826   6.400    6.609
36               4  0.7  0.739   2.800    2.957
40              15  0.3  0.391   4.500    5.870
55              10  0.2  0.304   2.000    3.043
65              35  0.1  0.217   3.500    7.609
total                           46.600   53.565
total, book                     46.600   53.565
```

Grey bars are each layer's expected loss, the others its CCoC premium:

```python
import matplotlib.pyplot as plt

labels = [f"{xs[j]:.0f} to {xs[j + 1]:.0f}" for j in range(len(widths))]
pos = range(len(widths))
fig, ax = plt.subplots()
ax.bar([i - 0.2 for i in pos], [s * w for s, w in zip(survival, widths)], width=0.4,
       color="0.75", label="expected loss")
ax.bar([i + 0.2 for i in pos], [ccoc.g(s) * w for s, w in zip(survival, widths)], width=0.4,
       label="premium under CCoC 15%")
ax.set_xticks(list(pos), labels)
ax.set_xlabel("layer of the total loss (monograph units)")
ax.set_ylabel("expected loss or premium (monograph units)")
ax.set_title("Table 4.4: the premium exceeds the loss most in the top layer")
ax.legend()
```

Under CCoC, every layer earns the same 15% on its capital, so the layer sum
gives the premium of the pentagon.

By outcome, the differences of the distorted survival are the risk-adjusted
probabilities `q`, and the premium is the `q`-weighted average of the totals.
The ratio `Z = q / p` is the likelihood ratio (Table 4.10):

```python
def distorted_probs(g, probs):
    """Algorithm 4.10.1, steps 4 to 6: decumulate, distort, difference."""
    s, gs = 1.0, [1.0]
    for p in probs:
        s -= p
        gs.append(g.g(s) if s > 1e-12 else 0.0)  # nothing exceeds the largest total
    return [gs[j] - gs[j + 1] for j in range(len(probs))]


q = distorted_probs(ccoc, insco.probs)
likelihood = pd.DataFrame({
    "total": [f"{t:.0f}" for t in insco.totals],
    "p": insco.probs,
    "q": q,
    "Z": [x / p for x, p in zip(q, insco.probs)],
})
(GT(likelihood, rowname_col="total")
 .tab_header("Table 4.10: risk-adjusted probabilities under CCoC at 15%")
 .fmt_number("p", decimals=1)
 .fmt_number(["q", "Z"], decimals=3))
```

```text
       p      q      Z
22   0.1  0.087  0.870
28   0.1  0.087  0.870
36   0.1  0.087  0.870
40   0.4  0.348  0.870
55   0.1  0.087  0.870
65   0.1  0.087  0.870
100  0.1  0.217  2.174
```

```python
print(f"sum of q x = {sum(x * t for x, t in zip(q, insco.totals)):.3f}  (book 53.565)")
```

```text
sum of q x = 53.565  (book 53.565)
```

CCoC multiplies the probability of every year by `1 - d = 0.870` and gives the
remaining weight to the worst year alone.

## 7. Pricing units via layers

Section 4.5 prices each unit with the same risk-adjusted probabilities:
`P_i = sum of q_j kappa_i(x_j)`. The conditional expectations add up to the
total, so the unit premiums add up to the portfolio's premium. This is
algorithm 4.10.2; `Portfolio.price` gives the same values:

```python
def natural_allocation(rows, g):
    """Algorithm 4.10.2: group by total, average units, distort, sum-product."""
    groups = {}
    for row in rows:
        groups.setdefault(float(sum(row)), []).append(row)
    xs = sorted(groups)
    probs = [len(groups[x]) / len(rows) for x in xs]
    kappa = [[sum(col) / len(groups[x]) for col in zip(*groups[x])] for x in xs]
    q = distorted_probs(g, probs)
    return [sum(qj * k[i] for qj, k in zip(q, kappa)) for i in range(len(rows[0]))]


by_hand = pd.DataFrame({
    "unit": UNITS,
    "by_hand": natural_allocation(ROWS, ccoc),
    "portfolio": [u.premium for u in priced.allocated],
})
(GT(by_hand, rowname_col="unit")
 .tab_header("Unit premiums under CCoC at 15%: algorithm 4.10.2 against Portfolio")
 .cols_label(by_hand="by hand", portfolio="Portfolio")
 .fmt_number(["by_hand", "portfolio"], decimals=3))
```

```text
   by hand  Portfolio
A   13.739     13.739
B   18.522     18.522
C   21.304     21.304
```

Two properties make this allocation the natural one:

1. With distinct totals, it is the derivative of the portfolio premium with
   respect to each unit's volume (Delbaen's result), hence the four
   adjustments.
2. It is the expected loss plus a covariance, `P_i = E[X_i] + cov(X_i, Z)`: a
   unit pays margin in proportion to how its losses align with the years that
   the distortion weights up.

```python
zs = [x / p for x, p in zip(q, insco.probs)]
parts = {"mean": [], "cov": []}
for u in UNITS:
    k = insco.kappa(u)
    mean = sum(p * x for p, x in zip(insco.probs, k))
    parts["mean"].append(mean)
    parts["cov"].append(sum(p * x * z for p, x, z in zip(insco.probs, k, zs)) - mean)
split = pd.DataFrame({"unit": UNITS, **parts})
split["premium"] = split["mean"] + split["cov"]
(GT(split, rowname_col="unit")
 .tab_header("The natural allocation as expected loss plus covariance")
 .cols_label(mean="E[X]", cov="cov(X, Z)", premium="premium")
 .fmt_number("mean", decimals=2)
 .fmt_number(["cov", "premium"], decimals=3))
```

```text
    E[X]  cov(X, Z)  premium
A  13.40      0.339   13.739
B  18.30      0.222   18.522
C  14.90      6.404   21.304
```

## 8. Five representative distortions

Section 4.6 introduces the five one-parameter families used throughout: CCoC,
proportional hazard (PH), Wang, dual power and TVaR. Calibrated to a 15% return
on the whole book at assets of 100, they give Table 4.5:

```python
FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
BOOK_PARAMETER = {"ccoc": 0.1500, "ph": 0.7205, "wang": 0.3427, "dual": 1.5951, "tvar": 0.2713}
distortions = {f: insco.calibrate(f, assets=100, return_on_capital=0.15) for f in FAMILIES}
rows = []
for f, g in distortions.items():
    ((name, value),) = g.parameters.items()
    rows.append({"family": f, "parameter": name, "value": value, "book": BOOK_PARAMETER[f],
                 "premium": insco.price(g, assets=100).total.premium})
(GT(pd.DataFrame(rows), rowname_col="family")
 .tab_header("Table 4.5: the five distortions calibrated to a 15% return",
             "Assets of 100; every family reaches the same premium")
 .cols_label(value="value", book="value, book")
 .fmt_number(["value", "book"], decimals=4)
 .fmt_number("premium", decimals=3))
```

```text
      parameter   value  value, book  premium
ccoc          r  0.1500       0.1500   53.565
ph          rho  0.7205       0.7205   53.565
wang     lambda  0.3427       0.3427   53.565
dual       beta  1.5952       1.5951   53.565
tvar          p  0.2713       0.2713   53.565
```

Each family is a function `g(s)` that lifts the survival probability `s`; the
grey line `g(s) = s` has no margin. The curves cross because all five give the
premium 53.565:

```python
grid = [i / 100 for i in range(101)]
panels = [("ccoc", "ph", "wang"), ("dual", "tvar")]
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.0, 3.6))
for ax, group in zip(axes, panels):
    ax.plot(grid, grid, color="0.75", label="no margin")
    for f in group:
        ax.plot(grid, [distortions[f].g(s) for s in grid], label=f)
    ax.set_xlabel("survival probability s")
    ax.legend()
axes[0].set_ylabel("distorted probability g(s)")
fig.suptitle("The five calibrated distortions (Table 4.5)", x=0.01, ha="left")
```

The families differ in where they put the probability weight. This table
recomputes the last two of Tables 4.6 to 4.8 (q and Z) for the median year
(total 40) and the worst year (total 100):

```python
BOOK_Z = {  # Table 4.8, likelihood ratio Z at the totals 40 and 100
    "ccoc": (0.8696, 2.1739), "ph": (0.8834, 1.9033), "wang": (0.9479, 1.7391),
    "dual": (1.0490, 1.5470), "tvar": (1.3723, 1.3723),
}
qs = {f: distorted_probs(g, insco.probs) for f, g in distortions.items()}
rows = []
for f in FAMILIES:
    rows.append({"family": f, "q40": qs[f][3], "q100": qs[f][6],
                 "z40": qs[f][3] / 0.4, "z100": qs[f][6] / 0.1})
    rows.append({"family": f"{f}, book", "z40": BOOK_Z[f][0], "z100": BOOK_Z[f][1]})
(GT(pd.DataFrame(rows), rowname_col="family")
 .tab_header("Tables 4.7 and 4.8: risk-adjusted probability q and likelihood ratio Z",
             "At the median year (total 40) and the worst year (total 100)")
 .cols_label(q40="q(40)", q100="q(100)", z40="Z(40)", z100="Z(100)")
 .fmt_number(["q40", "q100", "z40", "z100"], decimals=4)
 .sub_missing(missing_text=""))
```

```text
             q(40)  q(100)   Z(40)  Z(100)
ccoc        0.3478  0.2174  0.8696  2.1739
ccoc, book                  0.8696  2.1739
ph          0.3534  0.1903  0.8834  1.9034
ph, book                    0.8834  1.9033
wang        0.3791  0.1739  0.9479  1.7391
wang, book                  0.9479  1.7391
dual        0.4196  0.1547  1.0490  1.5470
dual, book                  1.0490  1.5470
tvar        0.5489  0.1372  1.3723  1.3723
tvar, book                  1.3723  1.3723
```

CCoC puts the most weight on the worst year and the least on the median year.
TVaR does the reverse, and the other three lie between, in the order listed.
This ranking from tail-centric to body-centric decides, in chapter 6, which
unit pays. The other entries of Tables 4.6 to 4.8 agree to the four decimals
printed, except the PH ratio for the worst year (1.9034 here, 1.9033 in the
book) and the TVaR ratio for the total 36 (0.3940 here, 0.3941).

These digits, like the dual power parameter (1.5952 here, 1.5951 in Table 4.5),
show where each program's calibration stopped, not a different method.
prospicio solves for the premium of 53.565217 to about 1e-8, and its TVaR
level, 0.27128713, is exact. A parameter a few millionths away prints the same
and gives the book's digits:

```python
p_tvar = distortions["tvar"].parameters["p"]
rho = distortions["ph"].parameters["rho"]
cases = (
    ("tvar, calibrated", Distortion.tvar(p_tvar), lambda g: (1 - g.g(0.7)) / 0.1),
    ("tvar, rounded", Distortion.tvar(0.271283), lambda g: (1 - g.g(0.7)) / 0.1),
    ("ph, calibrated", Distortion.proportional_hazard(rho), lambda g: g.g(0.1) / 0.1),
    ("ph, rounded", Distortion.proportional_hazard(0.7205), lambda g: g.g(0.1) / 0.1),
)
digits = pd.DataFrame({
    "case": [name for name, _, _ in cases],
    "parameter": [list(g.parameters.values())[0] for _, g, _ in cases],
    "ratio": [z(g) for _, g, z in cases],
    "premium": [insco.price(g, assets=100).total.premium for _, g, _ in cases],
})
(GT(digits, rowname_col="case")
 .tab_header("The last digits of Tables 4.5 and 4.8",
             "TVaR ratio at the total 36 and PH ratio at the total 100")
 .fmt_number("parameter", decimals=8)
 .fmt_number("ratio", decimals=5)
 .fmt_number("premium", decimals=4))
```

```text
                   parameter    ratio  premium
tvar, calibrated  0.27128713  0.39402  53.5652
tvar, rounded     0.27128300  0.39408  53.5651
ph, calibrated    0.72047928  1.90336  53.5652
ph, rounded       0.72050000  1.90327  53.5646
```

Table 4.9 allocates the premium to units under the Wang distortion, the
middle of the five:

```python
wang = insco.price(distortions["wang"], assets=100)
BOOK_WANG = [(14.109, 0.709, 0.102), (18.637, 0.337, 0.048), (20.818, 5.918, 0.850)]
units = wang.allocated
rows = []
for u, unit, book in zip(UNITS, units, BOOK_WANG):
    rows.append({"unit": u, "premium": unit.premium, "margin": unit.margin,
                 "share": unit.margin / wang.total.margin})
    rows.append({"unit": f"{u}, book", **dict(zip(["premium", "margin", "share"], book))})
rows.append({"unit": "total", "premium": wang.total.premium, "margin": wang.total.margin})
rows.append({"unit": "total, book", "premium": 53.565, "margin": 6.965})
(GT(pd.DataFrame(rows), rowname_col="unit")
 .tab_header("Table 4.9: the natural allocation under the Wang distortion")
 .cols_label(share="margin share")
 .fmt_number(["premium", "margin", "share"], decimals=3)
 .sub_missing(missing_text=""))
```

```text
             premium  margin  margin share
A             14.109   0.709         0.102
A, book       14.109   0.709         0.102
B             18.637   0.337         0.048
B, book       18.637   0.337         0.048
C             20.819   5.919         0.850
C, book       20.818   5.918         0.850
total         53.565   6.965
total, book   53.565   6.965
```

Unit C's premium is 20.81858, so Table 4.9's 20.818 and Tables 5.1 and 5.2's
20.819 are the same number, rounded two ways.

## 9. The industry standard approach

Section 4.7 shows that the 1990s practice of a constant cost of capital across
units is the CCoC distortion in the layer view. Each unit's margin is 15% of its
capital, and the marginal TVaR allocates the capital. The premium is the
expected loss plus `d` times the excess of the maximum over it (Table 4.11, the
book's row under each unit's):

```python
BOOK_ISA = [(16, 2.261, 13.739, 0.339, 0.049), (20, 1.478, 18.522, 0.222, 0.032),
            (64, 42.696, 21.304, 6.404, 0.919)]
rows = []
for u, unit, book in zip(UNITS, priced.allocated, BOOK_ISA):
    rows.append({"unit": u, "a": unit.assets, "Q": unit.capital, "P": unit.premium,
                 "M": unit.margin, "share": unit.margin / priced.total.margin,
                 "ROC": unit.return_on_capital})
    rows.append({"unit": f"{u}, book", **dict(zip(["a", "Q", "P", "M", "share"], book))})
(GT(pd.DataFrame(rows), rowname_col="unit")
 .tab_header("Table 4.11: the industry standard approach",
             "Assets a, capital Q, premium P, margin M, margin share and return on capital")
 .fmt_number("a", decimals=0)
 .fmt_number(["Q", "P", "M", "share", "ROC"], decimals=3)
 .sub_missing(missing_text=""))
```

```text
          a       Q       P      M  share    ROC
A        16   2.261  13.739  0.339  0.049  0.150
A, book  16   2.261  13.739  0.339  0.049       
B        20   1.478  18.522  0.222  0.032  0.150
B, book  20   1.478  18.522  0.222  0.032       
C        64  42.696  21.304  6.404  0.919  0.150
C, book  64  42.696  21.304  6.404  0.919       
```

The choice of distortion moves the premium between units:

```python
compare = pd.DataFrame({
    "unit": UNITS,
    "expected loss": insco.expected(),
    "CCoC": [u.premium for u in priced.allocated],
    "Wang": [u.premium for u in wang.allocated],
})
(GT(compare, rowname_col="unit")
 .tab_header("Premium by unit at a 15% return (monograph units)",
             "Natural allocation under CCoC and under the Wang distortion")
 .tab_spanner("premium", ["CCoC", "Wang"])
 .fmt_number(["expected loss", "CCoC", "Wang"], decimals=2))
```

```text
   expected loss   CCoC   Wang
A          13.40  13.74  14.11
B          18.30  18.52  18.64
C          14.90  21.30  20.82
```

Unit C takes 91.9% of the margin, against 85.0% under Wang. The monograph
makes three objections to the industry standard:

1. Capital does not cost the same in every layer (bond spreads fall with
   seniority).
2. CCoC weights only the worst outcome.
3. CCoC gives no credit for diversification. For independent risks, the price
   is additive, because the mean and the maximum are additive.

Two independent coin flips, one that pays 10 and one that pays 6:

```python
flip1, flip2 = [0.0, 0.0, 10.0, 10.0], [0.0, 6.0, 0.0, 6.0]
pooled = [a + b for a, b in zip(flip1, flip2)]
for name, g in (("ccoc", ccoc), ("wang", distortions["wang"])):
    alone = g.measure(Sampled(flip1)) + g.measure(Sampled(flip2))
    print(f"{name}: priced apart {alone:.3f}, pooled {g.measure(Sampled(pooled)):.3f}")
```

```text
ccoc: priced apart 9.043, pooled 9.043
wang: priced apart 10.146, pooled 9.829
```

## 10. Pricing events

Section 4.8 warns against reading `q` as the prices of the years themselves
(prices of events are not unique; see *Pricing Insurance Risk*). But `q` prices
consistently anything that moves with the total: a layer gets the same price
from `q` as from the distortion applied to the layer alone. A unit's allocation
is at most its stand-alone price. For the Wang distortion:

```python
gw, qw = distortions["wang"], qs["wang"]
layer = [min(max(t - 55.0, 0.0), 10.0) for t in insco.totals]   # the 55 to 65 layer
alone = gw.measure(Sampled([min(max(t - 55.0, 0.0), 10.0) for t in totals]))
print(f"layer 55 to 65: by q {sum(a * b for a, b in zip(qw, layer)):.3f}, alone {alone:.3f}")
```

```text
layer 55 to 65: by q 3.089, alone 3.089
```

```python
stand_alone = pd.DataFrame({
    "unit": UNITS,
    "allocated": [u.premium for u in wang.allocated],
    "stand_alone": [gw.measure(Sampled([float(row[i]) for row in ROWS])) for i in range(3)],
})
(GT(stand_alone, rowname_col="unit")
 .tab_header("Wang premium by unit: allocated against stand-alone")
 .cols_label(stand_alone="stand-alone")
 .fmt_number(["allocated", "stand_alone"], decimals=3))
```

```text
   allocated  stand-alone
A     14.109       15.357
B     18.637       20.792
C     20.819       21.104
```

The stand-alone prices add up to more than 53.565. The difference is the
diversification that pooling buys. The natural allocation shares it out and
depends on the whole book, so a unit's price moves when the book does (chapter
5).

## 11. Related guides

- [Chapters 5 and 6](capital-modeling-monograph-15-applications.md): the
  applications and the choice of a distortion.
- Section 5 of [the chapter 9 guide](capital-modeling-monograph-15.md):
  capital by unit (section 8.3 of the monograph).
- [Risk loads and distortions](risk-loads-and-distortions.md): which
  distortion to use, and how to calibrate it to quotes.
- `prospicio.pricing.PaymentPattern`: payment patterns and discounting. The
  examples of the monograph ignore the time value of money.
