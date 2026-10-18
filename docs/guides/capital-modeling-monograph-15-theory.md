# Capital adequacy and the natural allocation: CAS Monograph 15, chapters 3 and 4

CAS Monograph 15, *Introduction to Capital Modeling and Portfolio Management*
(John Major and Stephen Mildenhall, 2026), builds its theory on one small
company, InsCo, and prints every step as a table. This guide follows its
chapters 3 (capital adequacy) and 4 (pricing and allocation) in order,
recomputes each table with prospicio and prints the monograph's value beside
prospicio's. Its companion,
[Applications and the choice of a distortion](capital-modeling-monograph-15-applications.md),
does the same for chapters 5 and 6, and
[Capital modeling and portfolio pricing](capital-modeling-monograph-15.md)
reproduces chapter 9, the monograph's code.

The reasoning is the authors'; what follows explains it in new words and
recomputes the numbers. For the theory behind spectral risk measures and the
natural allocation at full length, read Mildenhall and Major's *Pricing
Insurance Risk*; the
[case studies guide](pricing-insurance-risk-case-studies.md) runs its
examples. Table numbers below are the monograph's, and money is in its units.

## 1. InsCo and the four adjustments

InsCo writes three units, A, B and C, and its capital model is ten equally
likely years. Each row below is a year (the monograph's Table 2.1 order) and
each cell the unit's loss in it. The plan premiums by unit are 13.9, 18.7 and
19.6:

```python
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

print(f"{'x':>5s} {'p':>4s}  " + " ".join(f"{u:>5s}" for u in UNITS))
for j, (x, p) in enumerate(zip(insco.totals, insco.probs)):
    kappa = [insco.kappa(u)[j] for u in UNITS]
    print(f"{x:5.0f} {p:4.1f}  " + " ".join(f"{k:5.1f}" for k in kappa))
print(f"{'E[X]':>10s}  " + " ".join(f"{m:5.1f}" for m in insco.expected()))
```

```text
    x    p      A     B     C
   22  0.1   15.0   7.0   0.0
   28  0.1   15.0  13.0   0.0
   36  0.1    5.0  20.0  11.0
   40  0.4   10.0  24.0   6.0
   55  0.1   26.0  19.0  10.0
   65  0.1   17.0   8.0  40.0
  100  0.1   16.0  20.0  64.0
      E[X]   13.4  18.3  14.9
```

This is Table 4.2 without its zero row. Section 4.3 of the monograph asks for
four changes to raw simulation output before any pricing: sort the years by
total, start from a total of zero, cap losses at the assets, and make the
totals distinct. A `Portfolio` makes all of them when it is built. It sorts;
it does not need the zero row, which only makes the layer sums start at zero
and carries no probability; there is nothing to cap because the assets of 100
cover the worst year; and the four years that total 40 become one scenario of
probability 0.4 in which each unit's loss is its average over those years, the
conditional expectation `kappa`. With unequal losses at an equal total, the
order of those years would otherwise be arbitrary, and the allocation would
depend on it.

The expected loss can be computed two ways on these seven totals (the
monograph's algorithm 3.4.1): as the probability-weighted sum of the
outcomes, or as the sum of the layers between consecutive totals, each
weighted by its probability of being reached, the survival function.

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

Section 3.1 reads InsCo's balance sheet as a ladder of thresholds. With the
plan premium of 52.2 and 100 of assets, made of 27.8 of equity and 20 of debt
on which 0.6 of coupon is due, a year's loss crosses one threshold after
another (Table 3.1):

```python
LEVELS = [  # event, loss at which it happens, the book's probability
    ("zero profit", 52.2, 0.30),
    ("shareholders wiped out", 52.2 + 27.8, 0.10),
    ("coupon not paid", 52.2 + 27.8 + 0.6, 0.10),
    ("bonds not redeemed", 100.0, 0.10),
]
for event, level, book in LEVELS:
    prob = sum(1 for t in totals if t >= level) / len(totals)
    print(f"{event:24s} {level:6.1f}  probability {prob:.0%} (book {book:.0%})")
print(f"TVaR at 99%: {total.tvar(0.99):.1f} (book 100.0)")
```

```text
zero profit                52.2  probability 30% (book 30%)
shareholders wiped out     80.0  probability 10% (book 10%)
coupon not paid            80.6  probability 10% (book 10%)
bonds not redeemed        100.0  probability 10% (book 10%)
TVaR at 99%: 100.0 (book 100.0)
```

Each threshold is a candidate VaR standard; regulators and rating agencies
pick their own. All of them past 90% land on the worst year here, which is
why the monograph can describe InsCo's 100 of assets as a TVaR 99% standard
or as full collateral: there is no default in this example.

## 3. VaR and TVaR

VaR at level p is the smallest loss not exceeded with probability p; TVaR is
the average of the worst 1 - p of outcomes, with a fraction of the boundary
outcome when 1 - p of the years is not a whole number. The monograph's
algorithms 3.4.2 and 3.4.3 do this on n equally likely outcomes by sorting
and counting. Written out, they agree with `Sampled.quantile` and
`Sampled.tvar`:

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


print(f"{'p':>5s} {'VaR':>6s} {'Sampled':>8s} {'TVaR':>8s} {'Sampled':>8s}")
for p in (0.0, 0.25, 0.5, 0.7, 0.75, 0.85, 0.9, 0.95):
    print(f"{p:5.2f} {var_sorted(totals, p):6.1f} {total.quantile(p):8.1f} "
          f"{tvar_sorted(totals, p):8.3f} {total.tvar(p):8.3f}")
```

```text
    p    VaR  Sampled     TVaR  Sampled
 0.00   22.0     22.0   46.600   46.600
 0.25   36.0     36.0   53.067   53.067
 0.50   40.0     40.0   60.000   60.000
 0.70   55.0     40.0   73.333   73.333
 0.75   55.0     55.0   77.000   77.000
 0.85   65.0     65.0   88.333   88.333
 0.90  100.0     65.0  100.000  100.000
 0.95  100.0    100.0  100.000  100.000
```

The two agree except at p = 0.7 and 0.9, where `n p` is a whole number and
falls exactly on a jump of the distribution: the total is at most 40 in
exactly 70% of the years. The monograph's text defines VaR as the smallest
p quantile, which is 40 at 0.7 and 65 at 0.9, and that is what `Sampled`
returns; its algorithm, read literally, takes the next outcome up. Only at
such levels do the two readings differ, and TVaR is the same under both.

TVaR at 0 is the mean and TVaR at 1 the largest outcome. The monograph's
reason to prefer TVaR is subadditivity: the TVaR of a sum never exceeds the
sum of the TVaRs, while VaR can. Its two examples, a discrete one with
outcome probabilities 0.9, 0.05 and 0.05 (here as twenty equally likely
years) and two independent standard exponentials at p = 0.7:

```python
from prospicio.distributions import Gamma

unit1 = [0.0] * 18 + [0.0, 10.0]
unit2 = [0.0] * 18 + [10.0, 0.0]
both = [a + b for a, b in zip(unit1, unit2)]
for name, vals in (("unit 1", unit1), ("unit 2", unit2), ("sum", both)):
    s = Sampled(vals)
    print(f"{name:6s} VaR(0.9) {s.quantile(0.9):4.1f}, by algorithm 3.4.2 "
          f"{var_sorted(vals, 0.9):4.1f}, at 0.91 {s.quantile(0.91):4.1f}   TVaR(0.9) {s.tvar(0.9):4.1f}")
one, two = Gamma(1.0, 1.0).quantile(0.7), Gamma(2.0, 1.0).quantile(0.7)
print(f"exponential VaR(0.7) {one:.3f} (book 1.204), twice {2 * one:.3f} (book 2.408), "
      f"of the sum {two:.3f} (book 2.439)")
```

```text
unit 1 VaR(0.9)  0.0, by algorithm 3.4.2  0.0, at 0.91  0.0   TVaR(0.9)  5.0
unit 2 VaR(0.9)  0.0, by algorithm 3.4.2  0.0, at 0.91  0.0   TVaR(0.9)  5.0
sum    VaR(0.9)  0.0, by algorithm 3.4.2 10.0, at 0.91 10.0   TVaR(0.9) 10.0
exponential VaR(0.7) 1.204 (book 1.204), twice 2.408 (book 2.408), of the sum 2.439 (book 2.439)
```

In both cases the VaR of the sum is larger than the sum of the VaRs. The
discrete example sits on a jump again: 90% of the years have a total of
zero, so the smallest 90% quantile of the sum is 0, and the failure the
monograph describes (0 for each unit, 10 in total) is the algorithm's reading
at 0.9 and everyone's at any level just above it. TVaR
adds up exactly in the first case and stays below in the second. Which one to
use is a matter of taste on which the two authors disagree; what matters for
the rest of the monograph is that the capital standard fixes the assets.

## 4. Risk analysis: marginal assets

Section 3.3 asks how each unit drives the assets. For a TVaR standard the
derivative of the assets with respect to a unit's volume, the coTVaR, is the
unit's average loss in the tail that defines the TVaR. At 99% that tail is the
worst year, so the coTVaRs are the unit losses in it, 16, 20 and 64. The
linear natural allocation of the assets gives the same split, and
subtracting the plan premium gives the capital each unit uses on plan (Table
3.2):

```python
priced = insco.price(Distortion.ccoc(0.15), assets=100)
co_tvar = [u.assets for u in priced.allocated]
BOOK_CAPITAL = [2.1, 1.3, 44.4]
for u, a, plan, book in zip(UNITS, co_tvar, PLAN, BOOK_CAPITAL):
    print(f"{u}: coTVaR {a:5.1f}  plan premium {plan:5.1f}  capital {a - plan:5.1f} (book {book})")
print(f"total capital on plan {100 - sum(PLAN):.1f} (book 47.8)")
print("VaR(0.85) by unit:", [insco.kappa(u)[5] for u in UNITS], " (book 17, 8, 40)")
```

```text
A: coTVaR  16.0  plan premium  13.9  capital   2.1 (book 2.1)
B: coTVaR  20.0  plan premium  18.7  capital   1.3 (book 1.3)
C: coTVaR  64.0  plan premium  19.6  capital  44.4 (book 44.4)
total capital on plan 47.8 (book 47.8)
VaR(0.85) by unit: [17.0, 8.0, 40.0]  (book 17, 8, 40)
```

The last line is Table 3.3: the marginal split of the 85% VaR of 65 is the
year that produced it. The monograph's point is that this is fragile. Unit B
loses 8 in that year but 19 and 20 in the years either side, so a VaR
standard allocated this way rests on one simulated year and moves with the
next run. Its remedies are the conditional expectations of section 1, which
average over years with the same total, or a range VaR, an average between two
levels. The other road out of the marginal analysis, risk-adjusted
probabilities, is where chapter 4 starts.

## 5. Pricing the whole

Chapter 4 prices from the top down. Investors want 15% on their capital, the
assets are 100 and the expected loss 46.6, which fixes the premium. The
monograph's pentagon of loss, margin, premium, capital and assets (Table 4.1)
is `Pentagon` in prospicio, solved from any three of its quantities:

```python
whole = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
d = 0.15 / 1.15
print(f"premium {whole.premium:.3f} (book 53.565)  margin {whole.margin:.3f}  "
      f"capital {whole.capital:.3f}")
print(f"risk discount d = {whole.discount:.5f} (book 0.13043); v L + d a = "
      f"{(1 - d) * 46.6 + d * 100:.3f}")
print(f"loss ratio {whole.loss_ratio:.3f}, premium to capital {whole.premium_to_capital:.3f}")
print(f"plan premium 52.2 against the required: EVA {52.2 - whole.premium:.3f} (book -1.4)")
```

```text
premium 53.565 (book 53.565)  margin 6.965  capital 46.435
risk discount d = 0.13043 (book 0.13043); v L + d a = 53.565
loss ratio 0.870, premium to capital 1.154
plan premium 52.2 against the required: EVA -1.365 (book -1.4)
```

The premium is a weighted average of the expected loss and the assets, with
the risk discount `d = r / (1 + r)` as the weight on the assets. The plan is
1.365 short of what investors need; the companion guide asks which units
cause the shortfall.

## 6. Pricing layers in total

Section 4.4 splits the 100 of assets into layers between consecutive totals:
0 to 22, 22 to 28, and so on to 65 to 100. A layer pays out in full whenever
the total exceeds its bottom, so it is a bet with one parameter, the
exceedance probability `S`. If each layer's rate on line is `g(S)` for a
distortion `g`, the premium is the sum of `g(S)` times the layer width, the
distorted analogue of the layer sum for the expected loss in section 1. For
the constant cost of capital (CCoC) at 15%, `g(s) = d + (1 - d) s` for any
`s > 0` (Table 4.4):

```python
ccoc = Distortion.ccoc(0.15)
widths = [xs[j + 1] - xs[j] for j in range(len(xs) - 1)]
print(f"{'from':>5s} {'width':>6s} {'S':>5s} {'g(S)':>6s} {'loss':>6s} {'premium':>8s}")
for j, w in enumerate(widths):
    s = survival[j]
    print(f"{xs[j]:5.0f} {w:6.0f} {s:5.1f} {ccoc.g(s):6.3f} {s * w:6.3f} {ccoc.g(s) * w:8.3f}")
layer_loss = sum(s * w for s, w in zip(survival, widths))
layer_premium = sum(ccoc.g(s) * w for s, w in zip(survival, widths))
print(f"{'':12s} total {layer_loss:13.3f} {layer_premium:8.3f}   (book 46.6, 53.565)")
```

```text
 from  width     S   g(S)   loss  premium
    0     22   1.0  1.000 22.000   22.000
   22      6   0.9  0.913  5.400    5.478
   28      8   0.8  0.826  6.400    6.609
   36      4   0.7  0.739  2.800    2.957
   40     15   0.3  0.391  4.500    5.870
   55     10   0.2  0.304  2.000    3.043
   65     35   0.1  0.217  3.500    7.609
             total        46.600   53.565   (book 46.6, 53.565)
```

Every layer earns the same 15% on its capital under the CCoC, which is why
this layer sum reproduces the pentagon's premium. The top layer, 35 wide and
reached one year in ten, carries 7.609 of premium for 3.5 of expected loss.

The same sum can be rearranged by outcome: differencing the distorted
survival gives risk-adjusted probabilities `q`, and the premium is the
`q`-weighted average of the totals. Their ratio to the true probabilities,
`Z = q / p`, is the monograph's likelihood ratio (Table 4.10):

```python
def distorted_probs(g, probs):
    """Algorithm 4.10.1, steps 4 to 6: decumulate, distort, difference."""
    s, gs = 1.0, [1.0]
    for p in probs:
        s -= p
        gs.append(g.g(s) if s > 1e-12 else 0.0)  # nothing exceeds the largest total
    return [gs[j] - gs[j + 1] for j in range(len(probs))]


q = distorted_probs(ccoc, insco.probs)
print("q", [round(x, 3) for x in q])
print("Z", [round(x / p, 3) for x, p in zip(q, insco.probs)])
print(f"sum of q x = {sum(x * t for x, t in zip(q, insco.totals)):.3f}  (book 53.565)")
```

```text
q [0.087, 0.087, 0.087, 0.348, 0.087, 0.087, 0.217]
Z [0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 2.174]
sum of q x = 53.565  (book 53.565)
```

The CCoC scales every year down by `1 - d = 0.870` and gives the worst year
alone the remaining weight: 0.1 of true probability becomes 0.217.

## 7. Pricing units via layers

Section 4.5 takes the step from the whole to the units: price each unit by
the same risk-adjusted probabilities, `P_i = sum of q_j kappa_i(x_j)`. Because
the conditional expectations add up to the total, the unit premiums add up to
the portfolio's premium. Written out, this is the monograph's algorithm
4.10.2, and it is what `Portfolio.price` computes:

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


by_hand = natural_allocation(ROWS, ccoc)
by_prospicio = [u.premium for u in priced.allocated]
print("by hand     ", [round(x, 3) for x in by_hand])
print("Portfolio   ", [round(x, 3) for x in by_prospicio])
```

```text
by hand      [13.739, 18.522, 21.304]
Portfolio    [13.739, 18.522, 21.304]
```

Two properties make this allocation the natural one. It is the derivative of
the portfolio premium with respect to each unit's volume whenever the totals
are distinct (Delbaen's result, which is why the four adjustments insist on
distinct totals), and it can be read as the expected loss plus a covariance,
`P_i = E[X_i] + cov(X_i, Z)`: a unit pays a margin in proportion to how its
losses line up with the years the distortion weights up.

```python
zs = [x / p for x, p in zip(q, insco.probs)]
for u in UNITS:
    k = insco.kappa(u)
    mean = sum(p * x for p, x in zip(insco.probs, k))
    cov = sum(p * x * z for p, x, z in zip(insco.probs, k, zs)) - mean
    print(f"{u}: E[X] {mean:5.2f} + cov(X, Z) {cov:6.3f} = {mean + cov:6.3f}")
```

```text
A: E[X] 13.40 + cov(X, Z)  0.339 = 13.739
B: E[X] 18.30 + cov(X, Z)  0.222 = 18.522
C: E[X] 14.90 + cov(X, Z)  6.404 = 21.304
```

## 8. Five representative distortions

Section 4.6 introduces the five families the monograph uses throughout: CCoC,
proportional hazard (PH), Wang, dual power and TVaR, each with one parameter,
calibrated so that the whole book earns 15% at assets of 100 (Table 4.5):

```python
FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
BOOK_PARAMETER = {"ccoc": 0.1500, "ph": 0.7205, "wang": 0.3427, "dual": 1.5951, "tvar": 0.2713}
distortions = {f: insco.calibrate(f, assets=100, return_on_capital=0.15) for f in FAMILIES}
for f, g in distortions.items():
    ((name, value),) = g.parameters.items()
    premium = insco.price(g, assets=100).total.premium
    print(f"{f:5s} {name:7s} {value:.4f} (book {BOOK_PARAMETER[f]:.4f})  premium {premium:.3f}")
```

```text
ccoc  r       0.1500 (book 0.1500)  premium 53.565
ph    rho     0.7205 (book 0.7205)  premium 53.565
wang  lambda  0.3427 (book 0.3427)  premium 53.565
dual  beta    1.5952 (book 1.5951)  premium 53.565
tvar  p       0.2713 (book 0.2713)  premium 53.565
```

All five charge the same 53.565 for the book; they differ in where the
probability weight goes. Tables 4.6 to 4.8 show the distorted survival, the
distorted probabilities and the likelihood ratios by year. The last two,
recomputed, for the median year (total 40) and the worst (total 100):

```python
BOOK_Z = {  # Table 4.8, likelihood ratio Z at the totals 40 and 100
    "ccoc": (0.8696, 2.1739), "ph": (0.8834, 1.9033), "wang": (0.9479, 1.7391),
    "dual": (1.0490, 1.5470), "tvar": (1.3723, 1.3723),
}
qs = {f: distorted_probs(g, insco.probs) for f, g in distortions.items()}
print(f"{'':5s} {'q(40)':>7s} {'q(100)':>7s} {'Z(40)':>16s} {'Z(100)':>16s}")
for f, qf in qs.items():
    z40, z100 = qf[3] / 0.4, qf[6] / 0.1
    b40, b100 = BOOK_Z[f]
    print(f"{f:5s} {qf[3]:7.4f} {qf[6]:7.4f} {z40:7.4f} ({b40:.4f}) {z100:7.4f} ({b100:.4f})")
```

```text
        q(40)  q(100)            Z(40)           Z(100)
ccoc   0.3478  0.2174  0.8696 (0.8696)  2.1739 (2.1739)
ph     0.3534  0.1903  0.8834 (0.8834)  1.9034 (1.9033)
wang   0.3791  0.1739  0.9479 (0.9479)  1.7391 (1.7391)
dual   0.4196  0.1547  1.0490 (1.0490)  1.5470 (1.5470)
tvar   0.5489  0.1372  1.3723 (1.3723)  1.3723 (1.3723)
```

The orderings are opposite: the CCoC puts the most weight on the worst year
and the least on the median one, TVaR the reverse, and the other three sit
between in the order listed. That is the monograph's ranking from tail-centric
to body-centric, and it decides who pays in chapter 6. The other entries of
Tables 4.6 to 4.8 agree to the four decimals printed as well, except two: the
PH ratio for the worst year, 1.9034 here and 1.9033 in the book, and the TVaR
ratio for the year totalling 36, 0.3940 here and 0.3941 in the book. Like the
dual power parameter above (1.5952 here, 1.5951 in Table 4.5), these last
digits are where each program's calibration stopped, not a different method.
prospicio solves for the premium of 53.565217 to about 1e-8 (its TVaR level,
0.27128713, is the exact solution); a parameter a few millionths away, which
prints the same to four decimals, leaves the premium at 53.565 to the three
decimals printed and reproduces the book's digits:

```python
p_tvar = distortions["tvar"].parameters["p"]
rho = distortions["ph"].parameters["rho"]
for name, g, z in (
    ("tvar", Distortion.tvar(p_tvar), lambda g: (1 - g.g(0.7)) / 0.1),
    ("tvar", Distortion.tvar(0.271283), lambda g: (1 - g.g(0.7)) / 0.1),
    ("ph", Distortion.proportional_hazard(rho), lambda g: g.g(0.1) / 0.1),
    ("ph", Distortion.proportional_hazard(0.7205), lambda g: g.g(0.1) / 0.1),
):
    ((_, value),) = g.parameters.items()
    print(f"{name:4s} {value:.8f}: ratio {z(g):.5f}  premium {insco.price(g, assets=100).total.premium:.4f}")
```

```text
tvar 0.27128713: ratio 0.39402  premium 53.5652
tvar 0.27128300: ratio 0.39408  premium 53.5651
ph   0.72047928: ratio 1.90336  premium 53.5652
ph   0.72050000: ratio 1.90327  premium 53.5646
```

With the Wang distortion, the middle of the five, the allocation to units is
Table 4.9:

```python
wang = insco.price(distortions["wang"], assets=100)
BOOK_WANG = [(14.109, 0.709, 0.102), (18.637, 0.337, 0.048), (20.818, 5.918, 0.850)]
for u, unit, (bp, bm, bs) in zip(UNITS, wang.allocated, BOOK_WANG):
    share = unit.margin / wang.total.margin
    print(f"{u}: premium {unit.premium:.3f} ({bp:.3f})  margin {unit.margin:.3f} ({bm:.3f})  "
          f"share {share:.3f} ({bs:.3f})")
print(f"total premium {wang.total.premium:.3f}, margin {wang.total.margin:.3f}  (book 53.565, 6.965)")
```

```text
A: premium 14.109 (14.109)  margin 0.709 (0.709)  share 0.102 (0.102)
B: premium 18.637 (18.637)  margin 0.337 (0.337)  share 0.048 (0.048)
C: premium 20.819 (20.818)  margin 5.919 (5.918)  share 0.850 (0.850)
total premium 53.565, margin 6.965  (book 53.565, 6.965)
```

Unit C's premium is 20.81858, so it rounds to 20.819; Table 4.9 prints 20.818
and Tables 5.1 and 5.2 print 20.819, the same number rounded two ways.

## 9. The industry standard approach

Section 4.7 explains how a constant cost of capital across units, the
practice of the 1990s, turns out to be the CCoC distortion in the layer
view: if every unit's margin is 15% of its capital and capital is allocated
by marginal TVaR, the book's premium is the expected loss plus `d` times
the excess of the maximum over it, which is a CCoC layer sum. The allocation
it gives (Table 4.11):

```python
BOOK_ISA = [(16, 2.261, 13.739, 0.339, 0.049), (20, 1.478, 18.522, 0.222, 0.032),
            (64, 42.696, 21.304, 6.404, 0.919)]
print(f"{'':2s} {'a':>4s} {'Q':>7s} {'P':>7s} {'M':>6s} {'share':>6s} {'ROC':>6s}")
for u, unit, book in zip(UNITS, priced.allocated, BOOK_ISA):
    share = unit.margin / priced.total.margin
    print(f"{u:2s} {unit.assets:4.0f} {unit.capital:7.3f} {unit.premium:7.3f} {unit.margin:6.3f} "
          f"{share:6.3f} {unit.return_on_capital:6.3f}   book {book}")
```

```text
      a       Q       P      M  share    ROC
A    16   2.261  13.739  0.339  0.049  0.150   book (16, 2.261, 13.739, 0.339, 0.049)
B    20   1.478  18.522  0.222  0.032  0.150   book (20, 1.478, 18.522, 0.222, 0.032)
C    64  42.696  21.304  6.404  0.919  0.150   book (64, 42.696, 21.304, 6.404, 0.919)
```

Unit C takes 91.9% of the margin, against 85.0% under Wang: a thick-tailed
unit gets nearly all the capital and margin under the industry standard, and
the thin-tailed units almost none. The monograph's objections are that
capital does not cost the same in every layer (bond spreads fall with
seniority), that the CCoC weights only the worst outcome, and that it gives no
credit for diversification: for independent risks the CCoC price is additive,
because both the mean and the maximum are. A check on two independent coin
flips, one paying 10 and one 6:

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

Section 4.8 warns against reading the risk-adjusted probabilities `q` as the
prices of the years themselves; the prices of events are not unique, for
reasons the monograph leaves to *Pricing Insurance Risk*. What does hold is
that `q` prices consistently anything that moves with the total: a layer of
the total gets the same price from `q` as from the distortion applied to it
alone, and any unit's allocation is at most its stand-alone price. For the
Wang distortion:

```python
gw, qw = distortions["wang"], qs["wang"]
layer = [min(max(t - 55.0, 0.0), 10.0) for t in insco.totals]   # the 55 to 65 layer
alone = gw.measure(Sampled([min(max(t - 55.0, 0.0), 10.0) for t in totals]))
print(f"layer 55 to 65: by q {sum(a * b for a, b in zip(qw, layer)):.3f}, alone {alone:.3f}")
for u, i, unit in zip(UNITS, range(3), wang.allocated):
    stand_alone = gw.measure(Sampled([float(row[i]) for row in ROWS]))
    print(f"{u}: allocated {unit.premium:.3f}, stand-alone {stand_alone:.3f}")
```

```text
layer 55 to 65: by q 3.089, alone 3.089
A: allocated 14.109, stand-alone 15.357
B: allocated 18.637, stand-alone 20.792
C: allocated 20.819, stand-alone 21.104
```

The stand-alone prices add up to more than the book's 53.565; the gap is the
diversification that pooling the units buys; the natural allocation
shares it out and depends on the whole book, so a unit's price changes when
the rest of the book does. That is the subject of chapter 5.

## 11. What this guide leaves out

- **Chapter 4's discounting**: the monograph's examples ignore the time value
  of money; for payment patterns see `prospicio.pricing.PaymentPattern`.
- **Capital by unit** (section 8.3): see section 5 of
  [the chapter 9 guide](capital-modeling-monograph-15.md), where the lifted
  capital under CCoC and PH depends on the `aggregate` version. Chapters 3 and
  4 print no lifted capital, so nothing here depends on it.
- **The figures** (the pentagon, the Lee diagrams, the distortion plots): the
  numbers behind them are the tables above.

Next: [chapters 5 and 6](capital-modeling-monograph-15-applications.md), and
for which distortion to use and how to calibrate it to quotes,
[Risk loads and distortions](risk-loads-and-distortions.md).
