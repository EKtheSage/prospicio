# A first course: from premium and losses to a priced book

<!-- site: execute -->

Read this page first. It is for a student or a new analyst who knows what a
premium and a claim are. It goes from a table of premium and losses to a price,
and from prices to decisions. Each idea has a small example that you can run.

1. **The ideas** (sections 1 to 7): loss distributions, the aggregate loss,
   price against expected loss, distortions, diversification and allocation.
2. **Two decisions** (sections 8 and 9): write a new unit, and choose a treaty.
3. **The checklist** (section 10): ten steps from data to a quote, with a
   template for your own data.

Money is in millions of a synthetic currency. All examples are synthetic. The
appendix defines the terms.

## Part 1. The ideas

### 1. A loss is a distribution

Start with two small units. In a year, unit A loses 0, 4 or 10 with probability
1/2, 1/4 and 1/4. Unit B loses 0, 2 or 40 with the same probabilities. The units
are independent, so the chances in 16 of a pair are the product of the chances
in 4: the pair (0, 0) has 2 times 2 = 4. Each pair, repeated as many times as
its chances, gives sixteen equally likely years.

`Sampled` holds equally likely outcomes. Read `TVAR75.measure(x)` as the 75%
TVaR of x (section 5 shows that TVaR is a distortion). This block has the
imports for the page.

```python
import math

import pandas as pd
from great_tables import GT
from prospicio.aggregate import fft, simulate_events
from prospicio.distributions import Grid, Lognormal, NegativeBinomial, Poisson, PredictiveDistribution, Sampled
from prospicio.pricing import Pentagon, Portfolio, Quote, price
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion, GaussianCopula, capital, simulate

A = {0.0: 2, 4.0: 1, 10.0: 1}   # outcome: chances in 4
B = {0.0: 2, 2.0: 1, 40.0: 1}
rows = []
for a, n in A.items():
    for b, m in B.items():
        rows += [[a, b]] * (n * m)   # the pair (a, b) has n times m chances in 16
unit_a = Sampled([a for a, _ in rows])
unit_b = Sampled([b for _, b in rows])
total = Sampled([a + b for a, b in rows])
TVAR75 = Distortion.tvar(0.75)

tails = pd.DataFrame(
    [(name, x.mean(), x.quantile(0.75), TVAR75.measure(x)) for name, x in [("A", unit_a), ("B", unit_b), ("A + B", total)]],
    columns=["unit", "mean", "VaR 75%", "TVaR 75%"],
)
(GT(tails, rowname_col="unit")
 .tab_header("Mean and 75% tail measures of two independent units", "Millions")
 .fmt_number(["mean", "VaR 75%", "TVaR 75%"], decimals=2))
```

```text
        mean  VaR 75%  TVaR 75%
A       3.50     4.00     10.00
B      10.50     2.00     40.00
A + B  14.00    12.00     43.50
```

The means add: 3.5 + 10.5 = 14. The tails do not. B loses 40 in one year in
four, above the 75% point, so its VaR is 2 and its TVaR is 40. A VaR shows where
the bad years start, a TVaR how bad they are. The TVaR of the book is less than
50, the sum for A and B, because their worst years do not always occur together.

### 2. Adding independent risks

The loss of the book is the loss of A plus the loss of B. For independent
units, the probability of a total is a sum over the pairs that give it: a total
of 0 needs both units at 0, 1/2 times 1/2 = 4/16. `Portfolio` does this for any
table of outcomes.

```python
port = Portfolio(["A", "B"], rows)
totals = pd.DataFrame({"total": [f"{t:.0f}" for t in port.totals], "chances": [round(16 * p) for p in port.probs]})
(GT(totals, rowname_col="total")
 .tab_header("The nine totals of the book", "Chances in 16")
 .fmt_integer("chances"))
```

```text
    chances
0         4
2         2
4         2
6         1
10        2
12        1
40        2
44        1
50        1
```

This calculation is a **convolution**. Thousands of claims need a tool, the
next section.

### 3. The aggregate loss: count times severity, by FFT

A real line has a random number of claims (the **frequency**) and a random size
for each claim (the **severity**). Their sum for the year is the **aggregate
loss**. The library puts the severity on a grid of equally spaced points and
adds the claims with the fast Fourier transform (FFT). Here the mean claim is 1
with a CV of 1.5, and the mean count is 10 claims a year.

```python
severity, _ = Grid.rounding(Lognormal.from_mean_cv(1.0, 1.5), 0.05, 801)   # claim sizes on a 0.05 step, up to 40
year, report = fft(Poisson(10.0), severity, 2**12)                          # 4,096 points: totals up to 204.8
print(f"mean claim {severity.mean():.3f}, mean year {year.mean():.2f}, sd {math.sqrt(year.variance()):.2f}")
print(f"99% VaR {year.quantile(0.99):.2f}, 99% TVaR {year.tvar(0.99):.2f}")
print(f"aliasing error below 1e-12: {report.aliasing_error < 1e-12}, tail mass below 1e-12: {report.tail_mass < 1e-12}")
```

```text
mean claim 0.999, mean year 9.99, sd 5.65
99% VaR 28.70, 99% TVaR 34.91
aliasing error below 1e-12: True, tail mass below 1e-12: True
```

The right tail of the year is long: the 99% VaR is almost three times the mean.

```python
import matplotlib.pyplot as plt

x = [year.step * i for i in range(len(year.probs))]
fig, ax = plt.subplots()
ax.plot(x, [q / year.step for q in year.probs], label="aggregate loss of the year")
ax.axvline(year.mean(), color="0.5", linestyle=":", label="mean")
ax.axvline(year.quantile(0.99), color="0.3", linestyle="--", label="99% VaR")
ax.axvline(year.tvar(0.99), color="0.3", linestyle="-.", label="99% TVaR")
ax.set_xlim(0, 50)
ax.set_xlabel("loss in the year (millions)")
ax.set_ylabel("probability density")
ax.set_title("The aggregate loss: 10 claims a year on average, CV 1.5 each")
ax.legend(loc="center right")
```

Two checks are important. The FFT wraps any probability past the end of the
grid back to the start (**aliasing**). The **tail mass** is the probability that
stays past the grid. Both must be very small, or the tail that you price is
wrong. If they are not, use more points or a larger step; `audit_grid` selects
them for you (section 10).

### 4. Why a price must exceed the mean

Go back to the book of section 1. An insurer that charges only the expected
loss, 14, makes no profit on average, and it still needs capital for the bad
years. The owners supply that capital only for a return. Let the assets be the
75% TVaR, 43.5, and let the owners want 10%. Then the premium P must give a
margin of 10% of the capital: P - 14 = 10% of (43.5 - P).

The **pricing pentagon** solves this equation. It connects loss, premium,
margin, capital and assets; any three set the other two.

```python
book = Pentagon.solve(loss=total.mean(), assets=TVAR75.measure(total), return_on_capital=0.10)
print(f"loss {book.loss:.2f}  premium {book.premium:.2f}  margin {book.margin:.2f}  "
      f"capital {book.capital:.2f}  assets {book.assets:.2f}  loss ratio {book.loss_ratio:.1%}")
```

```text
loss 14.00  premium 16.68  margin 2.68  capital 26.82  assets 43.50  loss ratio 83.9%
```

The price is above the mean because capital has a cost: a riskier cover needs
more capital, so more margin. The
[pentagon section of the price guide](reinsurance-price.md#the-pricing-pentagon)
has the formula.

### 5. What a distortion is

A distortion prices the loss directly, with no capital standard. It replaces
each probability s of a loss above a value with a larger probability, g(s), so
bad outcomes count for more than their probability. The proportional hazard
distortion is g(s) = s to the power 0.5. For four equally likely outcomes it
gives one weight to each outcome, smallest first. The price is the weighted sum.

```python
ph = Distortion.proportional_hazard(0.5)
print("g(s) at s = 0.25, 0.5, 0.75: " + ", ".join(f"{ph.g(s):.3f}" for s in (0.25, 0.5, 0.75)))
w = ph.weights(4)
print("weights of 4 equally likely outcomes, smallest first: " + ", ".join(f"{x:.3f}" for x in w))
print(f"A as 0, 0, 4, 10 by hand: {w[2] * 4 + w[3] * 10:.2f}")
```

```text
g(s) at s = 0.25, 0.5, 0.75: 0.500, 0.707, 0.866
weights of 4 equally likely outcomes, smallest first: 0.134, 0.159, 0.207, 0.500
A as 0, 0, 4, 10 by hand: 5.83
```

The distortion prices a chance of one in four as one in two, so the worst of
four outcomes gets weight 0.5, not 0.25. The prices:

```python
ph_prices = pd.DataFrame(
    [(name, x.mean(), ph.measure(x)) for name, x in [("A", unit_a), ("B", unit_b), ("A + B", total)]],
    columns=["unit", "expected loss", "price"],
)
ph_prices.loc[len(ph_prices)] = ["A's plus B's", ph_prices["expected loss"][:2].sum(), ph_prices["price"][:2].sum()]
(GT(ph_prices, rowname_col="unit")
 .tab_header("Proportional hazard price, g(s) = s^0.5", "Millions")
 .fmt_number(["expected loss", "price"], decimals=2))
```

```text
              expected loss  price
A                      3.50   5.83
B                     10.50  20.41
A + B                 14.00  25.41
A's plus B's          14.00  26.24
```

The book costs less than A and B priced apart: pricing the two units together
already gives a discount.

In the figure, the grey line is no distortion. Proportional hazard lies above it
everywhere. The TVaR curve rises straight to 1 at s = 0.25, so all the weight is
on the worst quarter.

```python
ss = [i / 200 for i in range(201)]
fig, ax = plt.subplots()
ax.plot(ss, ss, color="0.75", label="no distortion, g(s) = s")
ax.plot(ss, [ph.g(v) for v in ss], label="proportional hazard 0.5")
ax.plot(ss, [TVAR75.g(v) for v in ss], label="TVaR 75%")
ax.set_xlabel("s, the probability of a loss above a value")
ax.set_ylabel("g(s), the distorted probability")
ax.set_title("A distortion raises the probability of bad outcomes")
ax.legend()
```

### 6. Why a book's price is not the sum of its parts

A book is also cheaper than its parts because it needs less capital.
`price_stand_alone` prices each unit as a separate book, with assets at its own
90% VaR, and compares them with the full book, with assets at the 90% VaR of the
book. This section uses a VaR standard, not the 75% TVaR of section 4. A VaR
standard leaves the book a small chance of default, and the section shows the
effect of that chance.

```python
sa = port.price_stand_alone(ph, 0.9)
parts = [(f"{name} alone", alone.assets, alone.premium) for name, alone in zip(sa.units, sa.stand_alone)]
parts += [("sum of the parts", sa.sum.assets, sa.sum.premium), ("the book", sa.total.assets, sa.total.premium)]
parts += [("diversification benefit", sa.sum.assets - sa.total.assets, sa.sum.premium - sa.total.premium)]
(GT(pd.DataFrame(parts, columns=["price", "assets", "premium"]), rowname_col="price")
 .tab_header("Stand-alone and book prices at the 90% VaR standard", "Proportional hazard 0.5, millions")
 .fmt_number(["assets", "premium"], decimals=2))
```

```text
                         assets  premium
A alone                   10.00     5.83
B alone                   40.00    20.41
sum of the parts          50.00    26.24
the book                  44.00    23.91
diversification benefit    6.00     2.34
```

Alone, A holds 10 and B holds 40, a sum of 50. The book total reaches 50 in one
year in sixteen only, so at the 90% standard the book holds 44.

The premium decreases in two steps:

1. The distortion alone decreases the 26.24 of the parts to 25.41 (section 5).
   That 0.83 is pure pooling.
2. With assets of 44, the book can pay at most 44 (limited liability), so its
   premium prices that smaller loss: 23.91. That 1.50 is not pooling: the book
   defaults in the one year in sixteen with a total of 50, and its policyholders
   bear the 6 that it cannot pay.

The full difference of 2.34 is the **diversification benefit**. The next
section asks which unit gets it.

### 7. The main allocation methods

To judge each unit, an analyst splits the premium and the assets of the book.
Each method answers a different question. `capital` splits the 75% TVaR of the
book, 43.5, by four methods. `Portfolio.price` splits the proportional hazard
premium by the **natural allocation**: each unit has the book's distortion
weights, outcome by outcome, so the unit premiums add up to the book's.

```python
pdist = PredictiveDistribution(["unit"], [("A",), ("B",)], rows)
rows_t = []
for method in ["euler", "covariance", "shapley", "marginal"]:
    al = capital(pdist, TVAR75, method)
    rows_t.append((method, al.allocated[0], al.allocated[1], sum(al.allocated)))
rows_t.append(("stand-alone", al.standalone[0], al.standalone[1], sum(al.standalone)))
natural = port.price(ph).to_dict()
rows_t.append(("natural PH (premium)", natural["premium"][0], natural["premium"][1], natural["premium"][2]))
alloc = pd.DataFrame(rows_t, columns=["method", "A", "B", "sum"])
(GT(alloc, rowname_col="method")
 .tab_header("Allocation of the book's 75% TVaR of assets, 43.50, to units A and B",
             "Millions; the last row splits the premium, not the assets")
 .fmt_number(["A", "B", "sum"], decimals=2))
```

```text
                          A      B    sum
euler                  3.50  40.00  43.50
covariance             2.37  41.13  43.50
shapley                6.75  36.75  43.50
marginal               3.50  33.50  37.00
stand-alone           10.00  40.00  50.00
natural PH (premium)   5.05  20.36  25.41
```

| Method | The question it answers |
|---|---|
| Euler (coTVaR) | What does each unit lose, on average, in the worst years of the book? This is the coTVaR of the unit. In those years the mean loss of A is 3.5, its own mean |
| Covariance | How much does each unit move with the total? It can give a unit less than its mean |
| Shapley | What does each unit add, on average over all orders in which you can build the book? |
| Marginal | How much does the TVaR of the book decrease without the unit? The parts do not add up to the total (37 of 43.5) |
| Natural allocation | The share of each unit in the distorted price, outcome by outcome. The shares add up, and A pays 5.05 for an expected loss of 3.5 |

No method is correct for all questions. Select the method for the question,
and say which you used. Never compare the return of one unit under one method
with that of another unit under a different method.
[Section 8 of choosing a
method](choosing-a-method.md#8-capital-risk-measures-and-allocation) runs these
methods on a simulated book; [the price allocation guide](pir-price-allocation.md)
has more.

## Part 2. Two decisions

### 8. Should we write a new unit?

An insurer writes motor and home. A broker offers a block of storm-exposed
commercial property for a premium of 7.3. Should the insurer write it? We ask
the same question for a second insurer that writes motor and liability.

The FFT gives the annual loss of each line from a count and a claim size capped
at the policy limit:

```python
STEP, POINTS = 0.1, 2**12


def line(frequency, severity, cap):
    """A line's annual loss on the grid: claims capped at the policy limit, then the FFT."""
    claims, _ = Grid.rounding(severity, STEP, round(cap / STEP) + 1)
    return fft(frequency, claims, POINTS)[0]


motor = line(NegativeBinomial.from_mean_variance(100, 150), Lognormal.from_mean_cv(0.4, 1.0), 10)
home = line(Poisson(8), Lognormal.from_mean_cv(1.5, 2.5), 40)
liability = line(Poisson(3), Lognormal.from_mean_cv(3.0, 2.0), 40)
storm = line(Poisson(4), Lognormal.from_mean_cv(1.5, 2.5), 40)   # the new unit: storm-exposed commercial property
lines_t = pd.DataFrame(
    [(n, g.mean(), g.quantile(0.99)) for n, g in [("motor", motor), ("home", home), ("liability", liability), ("new unit", storm)]],
    columns=["line", "mean", "99% VaR"],
)
(GT(lines_t, rowname_col="line")
 .tab_header("Annual loss of each line", "Millions")
 .fmt_number("mean", decimals=2)
 .fmt_number("99% VaR", decimals=1))
```

```text
            mean  99% VaR
motor      40.03     56.1
home       11.76     48.2
liability   8.73     45.9
new unit    5.88     35.8
```

The lines are not independent. A storm hits home and commercial property in the
same year, so the new unit has a copula correlation of 0.7 with home. Liability
has no relation to the weather: 0. Motor has 0.1 with all lines. `simulate`
draws 10,000 years from the lines, joined by a Gaussian copula, a standard
method to put dependence on given distributions. Its correlation is that of the
normal scores behind the draws. The correlation of the losses is smaller.

First, the book of today. The insurer holds assets at the 99% TVaR of its total
and wants a 10% return on capital. `Portfolio.calibrate` finds the Wang
distortion that gives exactly that premium (Wang has one parameter, lambda: a
higher lambda makes risk more expensive). We keep it as the **price of risk** of
the company, the charge of its investors for risk of any type.

```python
N, SEED, RETURN = 10_000, 2026, 0.10
TAIL = Distortion.tvar(0.99)   # the capital standard: assets = 99% TVaR of the book's total


def book_of(lines, correlation, keys):
    """Simulate the lines joined by a Gaussian copula: the draws, the Portfolio and its assets."""
    pd = simulate(GaussianCopula(correlation), lines, N, SEED, keys=[(k,) for k in keys], dims=["line"])
    return pd, Portfolio.from_predictive(pd), TAIL.measure(pd)


pd2, p2, a2 = book_of([motor, home], [[1.0, 0.1], [0.1, 1.0]], ["motor", "home"])
house = p2.calibrate("wang", assets=a2, return_on_capital=RETURN)
today = p2.price(house, assets=a2).to_dict()
print(f"motor + home: assets {a2:.1f}, Wang lambda {house.parameters['lambda']:.3f}")
```

```text
motor + home: assets 100.5, Wang lambda 0.365
```

```python
book_t = pd.DataFrame({"line": today["unit"], "loss": today["loss"], "premium": today["premium"],
                       "capital": today["capital"], "return": today["return_on_capital"]})
(GT(book_t, rowname_col="line")
 .tab_header("The book of today, calibrated to a 10% return", "Millions")
 .fmt_number(["loss", "premium", "capital"], decimals=2)
 .fmt_percent("return", decimals=1))
```

```text
        loss  premium  capital  return
motor  40.10    41.56    16.97    8.6%
home   11.73    14.69    27.24   10.9%
total  51.83    56.25    44.21   10.0%
```

Motor is large but steady, and gets little capital. Home has a heavy tail, and
gets most of the capital.

Now add the new unit to each book, and price the larger book with the same
distortion. The **marginal premium** is the increase in the Wang premium of the
book. A lower premium pays the owners less than the company's price of risk. The
table also shows the allocated premium, the premium that keeps the larger book
at a flat 10% on its 99% TVaR assets, and two splits of the assets.

```python
OFFERED = 7.3
results = {}
for name, lines, r_new in [("motor + home", [motor, home], (0.1, 0.7)),
                           ("motor + liability", [motor, liability], (0.1, 0.0))]:
    keys = name.split(" + ")
    _, p2, a2 = book_of(lines, [[1.0, 0.1], [0.1, 1.0]], keys)
    house = p2.calibrate("wang", assets=a2, return_on_capital=RETURN)   # each insurer calibrates its own book
    pd3, p3, a3 = book_of(lines + [storm], [[1.0, 0.1, r_new[0]], [0.1, 1.0, r_new[1]], [r_new[0], r_new[1], 1.0]],
                          keys + ["new"])
    before, after = p2.price(house, assets=a2).total, p3.price(house, assets=a3)
    results[name] = {
        "expected loss": after.allocated[2].loss,
        "allocated premium": after.allocated[2].premium,
        "marginal premium": after.total.premium - before.premium,
        "premium for 10%": Pentagon.solve(loss=after.total.loss, assets=a3, return_on_capital=RETURN).premium
                           - before.premium,
        "allocated assets": after.allocated[2].assets,
        "Shapley assets": capital(pd3, TAIL, "shapley").allocated[2],
        "marginal assets": a3 - a2,
        "return at 7.3": Pentagon.solve(loss=after.total.loss, premium=before.premium + OFFERED,
                                        assets=a3).return_on_capital,
    }

cols = list(results)
new_unit = pd.DataFrame(results).rename_axis("measure").reset_index()
(GT(new_unit, rowname_col="measure")
 .tab_header("The new unit in each book", f"Millions; offered premium {OFFERED}")
 .fmt_number(cols, rows=list(range(7)), decimals=2)
 .fmt_percent(cols, rows=[7], decimals=1))
```

```text
                   motor + home  motor + liability
expected loss              5.79               5.82
allocated premium          7.72               7.19
marginal premium           7.47               6.61
premium for 10%            8.11               6.32
allocated assets          27.87              18.96
Shapley assets            34.92              25.35
marginal assets           31.42              11.26
return at 7.3              8.7%              12.2%
```

```python
for name, r in results.items():
    print(f"offered {OFFERED:.2f} in {name}: {'write' if OFFERED >= r['marginal premium'] else 'decline'}")
```

```text
offered 7.30 in motor + home: decline
offered 7.30 in motor + liability: write
```

**The decision.** The same unit at the same price gets opposite answers. In the
motor and home book it adds 31.42 of assets, because its storms occur with the
storms of home, and the Wang premium rises by 7.47: 7.3 is too little, so
decline or ask for at least 7.5. In the motor and liability book it adds only
11.26 of assets and the Wang premium rises by 6.61: 7.3 is enough, so write it.

The two criteria in the table do not agree, so select the one that controls the
decision. A flat 10% on the assets of the larger book (the constant cost of
capital distortion) prices each unit of extra capital the same. Wang charges
less for the far tail and more for the moderate years. The storm adds much tail
to the motor and home book, so the flat hurdle asks more than Wang. It adds
little tail to the motor and liability book, so the hurdle asks less.

This page uses the calibrated Wang price, because it prices all risks on the
same terms. Say so when you report the decision. An insurer that uses the flat
hurdle should ask for 8.11 for motor and home.

There are two lessons:

1. The price of a unit depends on the book that it joins. There is no single
   correct premium for it.
2. The allocated premium is not the marginal premium. An allocation shares the
   diversification of the full book. The marginal premium measures only the
   change that the unit causes, so use it to add or remove a unit.

### 9. Choosing a treaty for one cedant

A cedant writes 100 of premium. It expects 40 claims a year (negative binomial,
variance 80), with mean 1.3 and CV 3, capped at a policy limit of 25. It must
select a reinsurance structure.

This cedant measures capital as a solvency regulator does: the 99.5% VaR of its
yearly cost minus the mean. Its owners want 16% on that capital. The reinsurer
prices with a 10% return on its own 99% TVaR, then adds expenses and brokerage.
Reinsurance pays when it costs less than 16% of the capital that it releases.

```python
PREMIUM, LIMIT = 100.0, 25.0
frequency = NegativeBinomial.from_mean_variance(40, 80)
claims, _ = Grid.rounding(Lognormal.from_mean_cv(1.3, 3.0), STEP, round(LIMIT / STEP) + 1)   # capped at 25
gross, report = fft(frequency, claims, POINTS)
HURDLE = 0.16   # the cedant's cost of capital


def cedant_capital(g):
    """The cedant's capital: the 99.5% VaR of its cost less the mean."""
    return g.quantile(0.995) - g.mean()


GROSS_CAPITAL = cedant_capital(gross)
print(f"gross: expected loss {gross.mean():.2f}, 99.5% VaR {gross.quantile(0.995):.1f}, capital {GROSS_CAPITAL:.2f}")
print(f"aliasing error below 1e-12: {report.aliasing_error < 1e-12}")
```

```text
gross: expected loss 49.28, 99.5% VaR 114.8, capital 65.52
aliasing error below 1e-12: True
```

#### Excess of loss: which attachment and limit?

`Tower.on_grid` gives the annual loss of a layer (`ceded`) and the net loss of
the cedant (`net`) on the same grid. `price` gives the risk premium of the
reinsurer. `Quote.technical` adds 5% expenses and 10% brokerage to it. In the
tables, **ceded** is the expected loss that the layer pays, **premium** is what
the cedant pays for it, **cost** is the premium minus the expected recoveries,
and **saved** is the capital that the cedant saves. **Cost %** is the cost per
unit of capital saved. **Value** is 16% of the capital saved minus the cost; a
positive value means that the cover costs less than the capital.

```python
ROWS = {}


def record(name, ceded_loss, premium, cost, saved):
    ROWS[name] = {"ceded": ceded_loss, "premium": premium, "cost": cost, "saved": saved}


def structure_table(names, title, drop=()):
    """The recorded structures as a table, with the cost per unit of capital saved and the value."""
    df = pd.DataFrame([{"structure": n, **ROWS[n], "cost %": ROWS[n]["cost"] / ROWS[n]["saved"],
                        "value": HURDLE * ROWS[n]["saved"] - ROWS[n]["cost"]} for n in names]).drop(columns=list(drop))
    numbers = [c for c in df.columns if c not in ("structure", "cost %")]
    return (GT(df, rowname_col="structure")
            .tab_header(title, "Millions; value = 16% of the capital saved, minus the cost")
            .fmt_number(numbers, decimals=2)
            .fmt_percent("cost %", decimals=1))


xl_quotes = {}
for limit, attachment in [(23, 2), (20, 5), (10, 5), (15, 10)]:
    xl = Layer(f"XL {limit}x{attachment}", float(limit), float(attachment))
    t = Tower([xl]).on_grid(frequency, claims, POINTS)
    q = Quote.technical(price(t.ceded[0], TAIL, cost_of_capital=RETURN), expenses=0.05, brokerage=0.10,
                        limit=float(limit))
    xl_quotes[xl.name] = (q, t)
    record(xl.name, t.ceded[0].mean(), q.premium, q.premium - t.ceded[0].mean(),
           GROSS_CAPITAL - cedant_capital(t.net))
structure_table(list(ROWS), "Excess of loss: which attachment and limit?")
```

```text
          ceded  premium  cost  saved  cost %  value
XL 23x2   20.91    30.52  9.61  42.89   22.4%  -2.75
XL 20x5   10.67    17.26  6.59  29.73   22.2%  -1.84
XL 10x5    8.46    13.07  4.61  22.64   20.3%  -0.98
XL 15x10   4.72     8.76  4.03  17.88   22.6%  -1.17
```

A lower attachment saves more capital, but costs more. Each layer costs 20% to
23% of the capital that it releases, above the 16% of the cedant. The 10x5 layer
has the best value, still negative.

#### Quota share with a sliding-scale commission

A quota share cedes a fixed share of each claim and of the premium, here 25%.
The reinsurer pays a **ceding commission** for the expenses of the cedant. The
technical commission is the ceded premium left after the risk premium of the
reinsurer, 2% expenses and 1.5% brokerage
([section 4 of the price guide](reinsurance-price.md#4-quota-share)).

A **sliding scale** pays more commission when the ceded loss ratio is low, and
less when it is high. Here the scale has the same expected commission as the
flat commission.

```python
CESSION = 0.25
ceded_premium = CESSION * PREMIUM
qs = Layer.quota_share("QS 25%", CESSION).with_premium_rate(1.0, PREMIUM)
ceded = Tower([qs]).on_grid(frequency, claims, POINTS).ceded[0]
risk_premium = price(ceded, TAIL, cost_of_capital=RETURN).premium
technical = 1 - risk_premium / ceded_premium - 0.02 - 0.015    # less the reinsurer's expenses and brokerage
flat = qs.with_ceding_commission(technical)

shape = [(0.40, 0.45), (0.25, 0.65)]   # (commission, loss ratio): 40% at 45% or less, 25% at 65% or more
expected = Tower([qs.with_sliding_scale(shape)]).on_grid(frequency, claims, POINTS).expected_ceding_commission[0]
shift = technical - expected / ceded_premium
slide = qs.with_sliding_scale([(c + shift, lr) for c, lr in shape])
print(f"technical commission {technical:.1%}; the scale runs from {shape[0][0] + shift:.1%} to {shape[1][0] + shift:.1%}")
for layer in (flat, slide):
    result = [(ceded_premium - layer.ceding_commission_for(CESSION * x) - CESSION * x, p)
              for x, p in ((i * gross.step, p) for i, p in enumerate(gross.probs))]
    mean = sum(r * p for r, p in result)
    sd = math.sqrt(sum((r - mean) ** 2 * p for r, p in result))
    print(f"reinsurer, {'flat' if layer is flat else 'sliding':<7}: expected result {mean:.2f}, sd {sd:.2f}")
```

```text
technical commission 41.0%; the scale runs from 46.2% to 31.2%
reinsurer, flat   : expected result 2.44, sd 4.98
reinsurer, sliding: expected result 2.44, sd 3.66
```

```python
ratios = [0.4, 0.5, 0.6, 0.7, 0.8]
scale_t = pd.DataFrame({"loss ratio": [f"{lr:.0%}" for lr in ratios],
                        "flat": [flat.ceding_commission_for(lr * ceded_premium) / ceded_premium for lr in ratios],
                        "sliding": [slide.ceding_commission_for(lr * ceded_premium) / ceded_premium for lr in ratios]})
(GT(scale_t, rowname_col="loss ratio")
 .tab_header("Ceding commission by ceded loss ratio", "Share of the ceded premium")
 .fmt_percent(["flat", "sliding"], decimals=1))
```

```text
      flat  sliding
40%  41.0%    46.2%
50%  41.0%    42.5%
60%  41.0%    35.0%
70%  41.0%    31.2%
80%  41.0%    31.2%
```

The sliding commission falls from 46.2% to 31.2% as the loss ratio rises from
45% to 65%. The expected result of the reinsurer is 2.44 for both commissions,
but its standard deviation falls from 4.98 to 3.66: in a bad year it pays less
commission, so the cedant takes back a part of the bad years.
[The sliding scales of the price guide](reinsurance-price.md#sliding-scales-and-profit-commissions)
show how much more commission a reinsurer can give for that. `gross.map` turns
each gross outcome into the cost of the cedant in that outcome:

```python
for label, layer in [("QS 25% flat", flat), ("QS 25% slide", slide)]:
    net_cost, _ = gross.map(lambda x: (1 - CESSION) * x + ceded_premium - layer.ceding_commission_for(CESSION * x))
    record(label, ceded.mean(), ceded_premium * (1 - technical), net_cost.mean() - gross.mean(),
           GROSS_CAPITAL - cedant_capital(net_cost))
structure_table(["QS 25% flat", "QS 25% slide"], "Quota share, flat and sliding commission")
```

```text
              ceded  premium  cost  saved  cost %  value
QS 25% flat   12.32    14.76  2.44  16.34   14.9%   0.18
QS 25% slide  12.32    14.76  2.44  13.94   17.5%  -0.21
```

Here "premium" is the ceded premium minus the expected commission, and the
cost is the expected result of the reinsurer for both commissions. The flat
commission releases 16.34 of capital, the scale only 13.94: under the scale, the
cedant still pays for its bad years through a lower commission.

#### Reinstatements

A real excess of loss layer limits how often it pays in a year. With one
**reinstatement**, the 10x5 layer pays a maximum of 20 a year. A **paid**
reinstatement charges extra premium when a loss uses the limit, so a part of
the price comes after the losses. Annual terms need the claims year by year, so
this block simulates 10,000 years.

Both versions have the expected premium of the 10x5 technical premium above.
When the reinstatement is free, all of it is deposit. When it is paid at 100%,
the deposit is smaller and the expected reinstatement premium supplies the rest.

```python
events = simulate_events(frequency, claims, N, 7)
technical_xl = xl_quotes["XL 10x5"][0].premium


def cedant_cost(deposit, rates):
    """Gross and the cedant's cost (net loss, deposit and reinstatement premium) in each simulated year."""
    years = Tower([Layer("10x5", 10.0, 5.0, premium=deposit, reinstatement_rates=rates)]).apply(events).draw_matrix()
    # the columns: gross, ceded, net, reinstatement premium
    gross_years = Sampled([g for g, _, _, _ in years])
    cost_years = Sampled([net + deposit + rp for _, _, net, rp in years])
    return gross_years, cost_years, sum(rp for *_, rp in years) / len(years)


_, _, rp = cedant_cost(technical_xl, [1.0])
paid_deposit = technical_xl / (1 + rp / technical_xl)   # the same expected premium, part paid after losses
reinstated = []
for label, deposit, rates in [("free", technical_xl, [0.0]), ("paid at 100%", paid_deposit, [1.0])]:
    gross_years, cost_years, rp = cedant_cost(deposit, rates)
    saved = cedant_capital(gross_years) - cedant_capital(cost_years)
    reinstated.append((label, deposit, rp, saved))
(GT(pd.DataFrame(reinstated, columns=["reinstatement", "deposit", "expected reinstatement premium", "capital saved"]),
    rowname_col="reinstatement")
 .tab_header("The 10x5 layer with one reinstatement", "Millions; the same expected premium in both rows")
 .fmt_number(["deposit", "expected reinstatement premium", "capital saved"], decimals=2))
```

```text
              deposit  expected reinstatement premium  capital saved
free            13.07                            0.00          12.06
paid at 100%     8.19                            4.88           8.74
```

Without an annual limit, the layer saves about 22 of capital (22.64 on the
grid, 22.26 on these simulated years). The annual limit cuts this to 12.06,
because the losses of the layer exceed 20 in the worst years. A paid
reinstatement cuts it to 8.74: the cedant pays the reinstatement premium in the
same years that it wants protection.

The two rows compare only the two versions. A layer with an annual limit pays
less than the unlimited 10x5, so its price is lower, and the decision table
below omits them.
[The reinstatements section of the price guide](reinsurance-price.md#reinstatements)
prices them.

#### The decision

The structures in order of value, best first:

```python
order = sorted(ROWS, key=lambda k: HURDLE * ROWS[k]["saved"] - ROWS[k]["cost"], reverse=True)
structure_table(order, "The six structures in order of value", drop=("ceded", "premium"))
```

```text
              cost  saved  cost %  value
QS 25% flat   2.44  16.34   14.9%   0.18
QS 25% slide  2.44  13.94   17.5%  -0.21
XL 10x5       4.61  22.64   20.3%  -0.98
XL 15x10      4.03  17.88   22.6%  -1.17
XL 20x5       6.59  29.73   22.2%  -1.84
XL 23x2       9.61  42.89   22.4%  -2.75
```

The last line reprices 10x5 with the loads of the quota share (2% expenses and
1.5% brokerage, not 15%) to show how much of the difference they explain.

```python
print(f"buy: {order[0]}")
q, t = xl_quotes["XL 10x5"]
cheap = Quote.technical(price(t.ceded[0], TAIL, cost_of_capital=RETURN), expenses=0.02, brokerage=0.015, limit=10.0)
cost, saved = cheap.premium - t.ceded[0].mean(), ROWS["XL 10x5"]["saved"]
print(f"XL 10x5 at the quota share's loads: premium {cheap.premium:.2f}, cost {cost:.2f}, "
      f"value {HURDLE * saved - cost:.2f}")
```

```text
buy: QS 25% flat
XL 10x5 at the quota share's loads: premium 11.51, cost 3.05, value 0.57
```

With these inputs, the cedant should buy the 25% quota share at a flat
commission. It is the only structure that releases capital for less than the
16% that the owners charge. The sliding scale gives back a part of that relief.

Most of the difference from the excess of loss layers is the frictional load:
the layers carry 15% for expenses and brokerage, the quota share 3.5%. At the
loads of the quota share, 10x5 has a value of 0.57, the best of all. The
decision depends on the real charge of the excess of loss market. Get a firm
quote for 10x5 before you sign the quota share.

A higher cost of capital for the cedant, or a TVaR standard that weights the
tail more, favours the excess of loss. Repeat the comparison with your own
numbers. Do not reuse this result for another cedant.
[What the reinsurer decides](reinsurance-price.md#8-what-the-reinsurer-decides)
shows the other side.

## Part 3. The checklist

### 10. Ten steps from data to a quote

| Step | Call | Check | Usual mistake |
|---|---|---|---|
| 1. Premium at today's rate level, losses trended | `RateHistory`, `Period`, `TrendFit`, `trend_factor` | On-level factors decrease toward recent years | Trending premium and losses with the same factor, or forgetting the rate changes |
| 2. Develop losses to ultimate | `Triangle`, `ChainLadder`. For large losses: `DevelopmentPattern.from_history`, `LargeLossExperience` | Compare with `BornhuetterFerguson` on the youngest years | Developing a layer with ground-up factors (use `ExcessPattern`) |
| 3. Severity and frequency | `fit_severity`, `goodness_of_fit`, `Poisson`, `NegativeBinomial`, `Truncated` | Fit above the reporting threshold. Censor claims at their policy limit | A ground-up severity with a count of claims above the threshold, or a Poisson count when the variance is larger than the mean |
| 4. Aggregate distribution | `fft`, `audit_grid` | `report.aliasing_error`, `report.tail_mass` | A grid too short for the tail |
| 5. Contract terms | `Layer`, `Tower`, then `on_grid` or `apply` | `on_points` | Annual terms (reinstatements, aggregate limits) on the grid. Then `net` is `None`, so simulate and use `apply` |
| 6. Expected loss two ways, blended | Experience: `PreparedLosses.burning_cost`. Exposure: `RiskProfile`, `Mbbefd`. Blend: `LimitedFluctuation`, `credibility_weighted` | The two estimates are in a plausible range of each other | Full credibility for a layer with a small number of claims |
| 7. Capital standard | `Distortion.tvar`, `Portfolio.assets`, `Portfolio.assets_for_epd` | The same standard for all options that you compare | Comparing one option on VaR with another on TVaR |
| 8. Distortion calibrated | `Portfolio.calibrate`, `calibrate` | The calibrated book earns its target return | Calibrating to a premium on uncapped losses, then pricing with capped assets |
| 9. Allocation | `Portfolio.price`, `capital` | Allocations add up to the total of the book | Comparing returns by line under different methods |
| 10. Quote and sensitivity | `Quote.technical`, `Quote.at` | Rate on line and return at the market price | Quoting the technical premium without a comparison with the market rate on line for similar layers |

[The submission evaluation notebook](../../examples/submission_evaluation.ipynb)
runs these steps on a synthetic submission.

The template below runs the ten steps for one excess of loss layer. It needs two
files of your own:

- `my_premium.csv`, one row for each accident year: `year`, `earned_premium`.
- `my_losses.csv`, one row for each claim and development age: `claim_id`,
  `accident_year`, `development` (in months: 12, 24, ...) and `incurred`
  (cumulative). List each claim at each age up to the latest. When a value did
  not change, repeat it. If you do not, the summed triangle is wrong.

Change each value marked "yours" and check the output of each step. The tests
do not run the template.

```python no-run
import csv
from statistics import mean, variance

from prospicio.aggregate import audit_grid, fft
from prospicio.credibility import LimitedFluctuation
from prospicio.distributions import Grid, NegativeBinomial, Truncated, fit_severity, goodness_of_fit
from prospicio.pricing import DevelopmentPattern, LargeLossExperience, Period, Quote, RateHistory, credibility_weighted, price
from prospicio.reinsurance import Layer, Tower
from prospicio.reserving import ChainLadder, Triangle
from prospicio.risk import Distortion

VALUATION, FUTURE, LOSS_TREND = 2026.0, 2027.5, 0.05      # yours: valuation date, mid-point of the period priced, trend
FUTURE_PREMIUM = 120.0                                    # yours: subject premium of the period priced
THRESHOLD, POLICY_LIMIT = 1.0, 25.0                       # yours: reporting threshold of large claims, policy limit
ATTACHMENT, LIMIT = 5.0, 10.0                             # the layer priced
COUNT_IBNR = {}                                           # yours: large-claim count development factor by year (default 1)
premium = {int(r["year"]): float(r["earned_premium"]) for r in csv.DictReader(open("my_premium.csv"))}
rows = list(csv.DictReader(open("my_losses.csv")))
years = sorted(premium)

# 1. Premium at today's rate level (the losses are trended in step 2)
history = RateHistory([2023.0, 2025.0], [0.05, 0.03])    # yours: effective dates and rate changes
on_level = [premium[y] * history.rate_level(Period.calendar(y, y + 1), 1.0).on_level_factor for y in years]

# 2. Develop to ultimate: the ground-up triangle, and each large claim's IBNER from the claims' own histories
tri = Triangle.from_long([int(r["accident_year"]) for r in rows], [int(r["development"]) for r in rows],
                         {"incurred": [float(r["incurred"]) for r in rows]})
cl = ChainLadder().fit(tri, "incurred")
print("ultimate loss ratios:", [round(u / p, 3) for u, p in zip(cl.ultimate, on_level)])
age = [int(r["development"]) / 12 - 0.5 for r in rows]                         # years since a mid-year accident
ibner = DevelopmentPattern.from_history([r["claim_id"] for r in rows], age,
                                        [float(r["incurred"]) for r in rows], ages=sorted(set(age)))
latest = {r["claim_id"]: r for r in sorted(rows, key=lambda r: int(r["development"]))}
ids = list(latest)
record = LargeLossExperience(ids, [int(latest[i]["accident_year"]) + 0.5 for i in ids],
                             [float(latest[i]["incurred"]) for i in ids], threshold=THRESHOLD, valuation=VALUATION)
prepared = record.prepare(rate=LOSS_TREND, to=FUTURE, development=ibner)   # developed (IBNER) and trended
floor = prepared.complete_above                           # every year is complete above this trended size

# 3. Severity of the large claims above the floor, and their yearly count at next year's premium
large = [(int(latest[i]["accident_year"]), x, float(latest[i]["incurred"]) >= POLICY_LIMIT)
         for i, x in zip(ids, prepared.trended) if x > floor]
sizes = [x for _, x, _ in large]
capped = [c for _, _, c in large]                         # censored: a policy limit capped the loss
fit = fit_severity("lognormal", sizes, reporting_thresholds=[floor] * len(sizes), censored=capped)
gof = goodness_of_fit(fit.distribution, sizes, reporting_thresholds=[floor] * len(sizes), censored=capped)
print(fit.parameters, "largest gap to the data:", round(gof["max_distance"], 3))
counts = [FUTURE_PREMIUM * COUNT_IBNR.get(y, 1.0) * sum(v == y for v, _, _ in large) / p for y, p in zip(years, on_level)]
frequency = NegativeBinomial.from_mean_variance(mean(counts), max(variance(counts), 1.01 * mean(counts)))
severity = Truncated(fit.distribution, lower=floor)       # the fit is ground up; the count is of claims above the floor

# 4. Aggregate distribution, with the grid chosen and checked
audit = audit_grid(frequency, severity)
claims, _ = Grid.rounding(severity, audit.step, round(POLICY_LIMIT / audit.step) + 1)   # capped at the policy limit
_, report = fft(frequency, claims, 2**audit.log2)
assert report.aliasing_error < 1e-6 and report.tail_mass < 1e-6, "use more points or a larger step"

# 5. Contract terms
tower = Tower([Layer("layer", LIMIT, ATTACHMENT)])
layer_loss = tower.on_grid(frequency, claims, 2**audit.log2).ceded[0]

# 6. Expected loss two ways, blended by credibility
burn = prepared.burning_cost(tower, years, premium=on_level, development=[COUNT_IBNR.get(y, 1.0) for y in years])
experience = burn.loss_cost[0] * FUTURE_PREMIUM          # the layer's loss cost on on-level premium, at next year's premium
exposure = layer_loss.mean()                              # yours: a stand-in; use a RiskProfile or an exposure curve
lf = LimitedFluctuation(0.90, 0.05)
z = lf.credibility(sum(x > ATTACHMENT for x in sizes), lf.frequency_standard())
blended = credibility_weighted(experience, z, exposure)
print(f"experience {experience:.3f}, exposure {exposure:.3f}, credibility {z:.2f}, blended {blended:.3f}")

# 7 to 9. Capital standard and price of risk (for a book: Portfolio.calibrate and Portfolio.price, section 8)
TAIL = Distortion.tvar(0.99)
risk = price(layer_loss, TAIL, cost_of_capital=0.10)
scale = blended / layer_loss.mean()                       # moves the modelled layer loss to the blended estimate

# 10. Quote and sensitivity. Scaling loss, premium and assets together leaves every return unchanged.
quote = Quote.technical(risk, expenses=0.05, brokerage=0.10, limit=LIMIT)
print(f"technical premium {quote.premium * scale:.3f}, rate on line {quote.premium * scale / LIMIT:.1%}")
for market in (0.8, 1.0, 1.2):
    print(f"at {market:.0%} of technical: return on capital {quote.at(market * quote.premium).return_on_capital:.1%}")
```

Replace these three simplifications in the template:

1. The exposure estimate is the same model that the experience fits, so the
   credibility blend has nothing independent to blend. Use an exposure rating
   instead: a `RiskProfile` with an exposure curve, as in
   [that guide](experience-and-exposure-rating.md).
2. The count development factors are 1 by default, correct only for mature
   years.
3. The layer is developed with the IBNER of each claim, not with ground-up
   factors.

## Where to go next

- Steps 1 to 9: [the ratemaking guide](ratemaking-werner-modlin.md),
  [reserving methods](reserving-methods.md),
  [severity and frequency](severity-and-frequency.md),
  [aggregate distributions](aggregate-distributions.md) (when to use the FFT,
  Panjer's recursion, simulation or moments),
  [experience and exposure rating](experience-and-exposure-rating.md) and
  [risk loads and distortions](risk-loads-and-distortions.md) (also the
  families of distortions of section 5).
- [What a reinsurance price is](reinsurance-price.md): the full worked example
  behind sections 4 and 9, and the terms and the quote of steps 5 and 10.
- [Choosing a method](choosing-a-method.md): a decision table for the usual
  situations of an actuary.
- *Pricing Insurance Risk* reproduced: [risk measures](pir-risk-measures.md),
  [portfolio pricing](pir-portfolio-pricing.md) and
  [price allocation](pir-price-allocation.md).
- [Monograph 15's applications](capital-modeling-monograph-15-applications.md):
  new business, reinsurance and portfolio decisions on a full capital model.
- Notebooks: [the reading path](../../examples/reading_path.ipynb) and
  [a submission evaluation](../../examples/submission_evaluation.ipynb).

## Appendix: terms

- **Loss distribution**: what a cover can pay in a year, with the probability of
  each amount. Its **mean** is the expected loss. Its **coefficient of
  variation** (CV) is the standard deviation divided by the mean.
- **VaR** (value at risk) at 99%: the loss that only 1% of years exceed.
- **TVaR** (tail value at risk) at 99%: the average loss in the worst 1% of
  years. It is never smaller than the VaR.
- **Assets**: the money available to pay claims. A **capital standard** sets the
  amount, for example the 99% TVaR of the loss. In a year with a larger loss,
  the company defaults and the policyholders bear the rest (**limited
  liability**).
- **Premium** and **margin**: the premium pays part of the assets. The margin is
  the premium minus the expected loss, the reward of the owners.
- **Capital**: assets minus premium, supplied by the owners. **Cost of
  capital**: the return that the owners want on it, here 10% a year.
- **Distortion**: a rule that prices a loss. It gives bad outcomes more weight
  than their probability (section 5).
- **Allocation**: a split of the premium or capital of a book between its units.
- **Cedant**: an insurer that buys reinsurance.
- **Layer**: the part of each loss that an excess of loss contract pays, written
  limit x attachment (also called the retention). The layer 10x5 pays the part
  of each loss between 5 and 15.
- **Risk premium**: the expected loss plus the margin for capital. **Technical
  premium**: the risk premium plus expenses and brokerage. **Rate on line**: the
  premium of a layer divided by its limit.
- **On-level premium**: past premium at the rates of today. **IBNR**: claims
  incurred but not yet reported. A **censored** claim is capped by a policy
  limit: you know only that its true size is at least the cap.

[The glossary of the price guide](reinsurance-price.md#10-glossary-and-further-reading) has more reinsurance words.
