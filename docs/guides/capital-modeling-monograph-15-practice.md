# Building and judging a capital model: CAS Monograph 15, chapters 1, 2 and 7

<!-- site: execute -->

CAS Monograph 15 is *Introduction to Capital Modeling and Portfolio
Management* (John Major and Stephen Mildenhall, 2026). Its chapters 1, 2 and 7
describe the system around the mathematics: the structure of a capital model,
the output of its business operations module, and how to judge a model or a
platform.

This guide covers the numeric parts (the InsCo simulation table, balance
sheets, pro forma statements) and maps the monograph's structure and
checklists onto prospicio. Table numbers are the monograph's.

Companion guides: [chapters 3 and 4](capital-modeling-monograph-15-theory.md),
[chapters 5 and 6](capital-modeling-monograph-15-applications.md),
[chapter 8](capital-modeling-monograph-15-iman-conover.md) and
[chapter 9](capital-modeling-monograph-15.md).

## 1. The parts of a capital model

The monograph's system (sections 1.4.4 and 1.5) has three modules:

1. Business Operations simulates the firm's results.
2. Capital Adequacy turns the results into the assets needed for a risk
   tolerance.
3. Pricing and Allocation splits those assets into premium and capital, and
   allocates the margin to units.

Business Operations holds the units, an optional economic scenario generator
(ESG) for assets and inflation, reinsurance (gross to net) and accounting
(pro formas). In prospicio:

| Monograph | prospicio |
|---|---|
| Units: losses by unit, frequency and severity | `Severity` families, `Count`, `fft` (compound distributions), `simulate_events`, event loss tables (`EventSet`), reserving models (`MackBootstrap`, `OdpBootstrap`, `BayesOdp`) |
| Correlation between units | `iman_conover`, `reorder_groups` (block Iman-Conover), copulas with `simulate` ([chapter 8 guide](capital-modeling-monograph-15-iman-conover.md)) |
| A sample of joint outcomes (the "probabilistic database") | `PredictiveDistribution`: one row per simulated year, one column per unit |
| Reinsurance, gross to net | towers and treaties in `prospicio.reinsurance`, or a ceded unit with negative losses (section 4) |
| Economic scenario generator | the scenarios as columns of a `PredictiveDistribution` |
| Accounting and pro formas | the P&L ledger (`Ledger` in `prospicio.aggregate`) for premium, loss and expense legs, and balance sheets in code (section 5) |
| Capital Adequacy | `quantile`, `tvar`, `Portfolio.assets`, `Portfolio.epd`, `Portfolio.priority` |
| Pricing and Allocation | `Distortion`, `Portfolio.calibrate`, `Portfolio.price` (the natural allocation), `Portfolio.bodoff` |

Structural parameters (volatility, correlation) change rarely, business
parameters (premium, loss ratio) in every planning run. Build the unit shapes
once and pass volumes and target return as arguments.

## 2. InsCo: the simulation table and the funding equation

InsCo (sections 1.6 and 2.2) writes three units for one period, with ten
equally likely outcomes. Table 2.1 sorts them by total and adds each unit's
expected loss (EX), CV and plan premium:

```python
import pandas as pd
from great_tables import GT
from prospicio.distributions import Discrete
from prospicio.pricing import Portfolio

ROWS = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
        [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]
PLAN = [13.9, 18.7, 19.6]
insco = Portfolio(["A", "B", "C"], ROWS)
totals = [sum(r) for r in ROWS]
columns = [*zip(*ROWS), totals]
probs = [0.1] * 10

dists = [Discrete([float(v) for v in col], probs) for col in columns]
stats = [[d.mean() for d in dists],
         [d.std_dev() / d.mean() for d in dists],
         [d.skewness() for d in dists],
         PLAN + [sum(PLAN)]]
sim = pd.DataFrame([[float(v) for v in r] + [float(t)] for r, t in zip(ROWS, totals)] + stats,
                   columns=["A", "B", "C", "Total"])
sim.insert(0, "row", [str(k) for k in range(1, 11)] + ["EX", "CV", "skewness", "plan"])
(GT(sim, rowname_col="row")
 .tab_header("InsCo, Table 2.1: ten equally likely outcomes", "Loss by unit and plan premium, money units")
 .fmt_number(["A", "B", "C", "Total"], rows=list(range(10)) + [10, 13], decimals=1)
 .fmt_number(["A", "B", "C", "Total"], rows=[11, 12], decimals=3))
```

```text
              A      B      C  Total
1          15.0    7.0    0.0   22.0
2          15.0   13.0    0.0   28.0
3           5.0   20.0   11.0   36.0
4           7.0   33.0    0.0   40.0
5          13.0   20.0    7.0   40.0
6           5.0   27.0    8.0   40.0
7          15.0   16.0    9.0   40.0
8          26.0   19.0   10.0   55.0
9          17.0    8.0   40.0   65.0
10         16.0   20.0   64.0  100.0
EX         13.4   18.3   14.9   46.6
CV        0.453  0.412  1.324  0.455
skewness  0.281  0.269  1.603  1.420
plan       13.9   18.7   19.6   52.2
```

The means and CVs are the values of Table 2.1, which does not print the
skewness. The monograph divides by 10, not 9, in the standard deviation, and
`Discrete` does the same.

The funding equation of section 1.6 is assets = premium plus capital,
`a = P + Q`. InsCo has assets of 100 and a plan premium of 52.2, so the
capital is 47.8. The expected margin is the premium minus the expected loss:

```python
assets, plan = 100.0, sum(PLAN)
loss = insco.expected()
capital = assets - plan
margin = plan - sum(loss)
print(f"premium {plan:.1f}  capital {capital:.1f}  expected loss {sum(loss):.1f}  margin {margin:.1f}")
print(f"loss ratio {sum(loss) / plan:.1%}  premium to capital {plan / capital:.2f}  "
      f"expected return {margin / capital:.1%}")
```

```text
premium 52.2  capital 47.8  expected loss 46.6  margin 5.6
loss ratio 89.3%  premium to capital 1.09  expected return 11.7%
```

The margin of 5.6 is the value in section 2.2. A 15% return needs a premium of
53.565 (see the [chapter 9 guide](capital-modeling-monograph-15.md)), so at
plan the investors expect 11.7%.

## 3. Margins, leverage and returns by unit

Section 1.4.2 (Table 1.1) compares attritional business (high loss ratio,
high leverage, capital that loses money about half the time) with catastrophe
business (low loss ratio, low leverage, capital that rarely loses money).
InsCo's unit A is steady, and unit C is like catastrophe business. The code
uses the Wang distortion of chapter 4 and the capital allocation of section
8.3:

```python
wang = insco.calibrate("wang", assets=100, return_on_capital=0.15)
priced = insco.price(wang, assets=100, allocation="lifted")
tab = pd.DataFrame({
    "unit": list(insco.units),
    "premium": [u.premium for u in priced.allocated],
    "loss_ratio": [u.loss_ratio for u in priced.allocated],
    "pq": [u.premium_to_capital for u in priced.allocated],
    "ret": [u.return_on_capital for u in priced.allocated],
    "p_loss": [sum(1 for x in col if x > u.premium) / 10 for u, col in zip(priced.allocated, columns)],
})
(GT(tab, rowname_col="unit")
 .tab_header("InsCo under the Wang distortion at a 15% return", "Lifted capital allocation")
 .fmt_number("premium", decimals=3)
 .fmt_percent(["loss_ratio", "ret"], decimals=1)
 .fmt_number("pq", decimals=2)
 .fmt_number("p_loss", decimals=1)
 .cols_label(premium="premium", loss_ratio="loss ratio", pq="P:Q", ret="return",
             p_loss="P(loss > premium)"))
```

```text
   premium  loss ratio   P:Q  return  P(loss > premium)
A   14.109       95.0%  1.68    8.4%                0.6
B   18.637       98.2%  2.14    3.9%                0.6
C   20.819       71.6%  0.71   20.2%                0.2
```

The loss ratio and leverage columns follow Table 1.1. A's loss exceeds its
premium in six years of ten, and C's in two.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
units = list(insco.units)
axes[0].bar(units, [u.premium_to_capital for u in priced.allocated])
axes[0].set_ylabel("premium to capital (ratio)")
axes[0].set_xlabel("unit")
axes[0].set_title("Leverage by unit")
axes[1].bar(units, [100 * u.return_on_capital for u in priced.allocated])
axes[1].set_ylabel("return on capital (%)")
axes[1].set_xlabel("unit")
axes[1].set_title("Return by unit")
fig.suptitle("InsCo under the Wang distortion at a 15% return")
```

The return column does not follow Table 1.1: the volatile unit C earns the
most. Ten years are too few to separate the units' layers, and part of C's
margin comes from the bottom layer of the assets, which holds no capital
(section 9 of the
[chapter 8 guide](capital-modeling-monograph-15-iman-conover.md)).

Section 1.4.3 lists the types of premium (filed, required, plan, indicated,
priced, quoted, sold, market) and of margin. Only the actual margin, the
margin after the fact, is not an expected value. For InsCo it is a
distribution at the plan premium:

```python
actual = Discrete([plan - t for t in totals], probs)
print(f"actual margin: mean {actual.mean():.1f}, worst {min(plan - t for t in totals):.1f}, "
      f"P(margin < 0) {actual.cdf(-1e-9):.1f}")
```

```text
actual margin: mean 5.6, worst -47.8, P(margin < 0) 0.3
```

The expected margin of 5.6 hides a loss in three years of ten.

## 4. Reinsurance as a negative unit

Section 2.3 gives two ways to model ceded reinsurance: as a unit of its own
with negative losses, beside the gross units, or downstream of the units,
applied to their results. Both give the same net result. The example is
InsCo's 35 xs 65 aggregate stop loss (section 2.3, priced in chapter 5):

```python
from prospicio.distributions import PredictiveDistribution, Sampled


def stop_loss(total, attach=65.0, limit=35.0):
    return min(max(total - attach, 0.0), limit)


with_ceded = PredictiveDistribution(
    ["unit"], [("A",), ("B",), ("C",), ("Ceded",)],
    [[float(a), float(b), float(c), -stop_loss(a + b + c)] for a, b, c in ROWS],
)
net = Sampled([t - stop_loss(t) for t in totals])
print(f"negative unit: mean {with_ceded.mean():.1f}  VaR 90% {with_ceded.var(0.9):.0f}  "
      f"largest {with_ceded.quantile(1.0):.0f}")
print(f"downstream:    mean {net.mean():.1f}  VaR 90% {net.var(0.9):.0f}  largest {net.quantile(1.0):.0f}")
```

```text
negative unit: mean 43.1  VaR 90% 65  largest 65
downstream:    mean 43.1  VaR 90% 65  largest 65
```

The monograph recommends the downstream form when one cover protects several
units, as a corporate catastrophe cover does. `Portfolio` needs non-negative
units, so for pricing the cover becomes a unit of recoveries beside the net
book, as in the [applications guide](capital-modeling-monograph-15-applications.md).

## 5. Capital structure and pro forma statements

Sections 2.5 to 2.7 give InsCo a capital structure and turn each outcome into
financial statements. The assets are 100 in cash (Table 2.2), funded by the
unearned premium of 52.2, a bond of 20 with a 3% coupon, and equity of 27.8.
There are no expenses or investment income. InsCo pays the losses first, then
the bond and its coupon. Equity takes the remainder:

```python
CASH, UPR, BOND, COUPON = 100.0, sum(PLAN), 20.0, 0.03
EQUITY = CASH - UPR - BOND
print(f"opening balance sheet: cash {CASH:.1f} = premium reserve {UPR:.1f} + bond {BOND:.1f} "
      f"+ equity {EQUITY:.1f}")
```

```text
opening balance sheet: cash 100.0 = premium reserve 52.2 + bond 20.0 + equity 27.8
```

```python
bond_paid, equity_end, income = [], [], []
for loss in totals:
    after_claims = max(CASH - loss, 0.0)
    to_bond = min(after_claims, BOND * (1 + COUPON))
    bond_paid.append(to_bond)
    equity_end.append(after_claims - to_bond)
    income.append(UPR - loss - BOND * COUPON)
pro = pd.DataFrame({"event": [str(k) for k in range(1, 11)], "loss": [float(x) for x in totals],
                    "coupon": BOND * COUPON, "income": income, "equity": equity_end, "bond": bond_paid})
(GT(pro, rowname_col="event")
 .tab_header("InsCo's ten outcomes: statements for each claimant", "Money units")
 .fmt_number(["loss", "coupon", "income", "equity", "bond"], decimals=1)
 .cols_label(income="net income", equity="end equity", bond="bond paid"))
```

```text
     loss  coupon  net income  end equity  bond paid
1    22.0     0.6        29.6        57.4       20.6
2    28.0     0.6        23.6        51.4       20.6
3    36.0     0.6        15.6        43.4       20.6
4    40.0     0.6        11.6        39.4       20.6
5    40.0     0.6        11.6        39.4       20.6
6    40.0     0.6        11.6        39.4       20.6
7    40.0     0.6        11.6        39.4       20.6
8    55.0     0.6        -3.4        24.4       20.6
9    65.0     0.6       -13.4        14.4       20.6
10  100.0     0.6       -48.4         0.0        0.0
```

Outcome 4 matches Tables 2.3 and 2.4. In outcome 10, the loss of 100 takes
all the assets. The policyholders are paid in full, the shareholders' 27.8
cannot absorb the loss of 48.4, and the bondholders lose their 20.6.

```python
fig, ax = plt.subplots()
events = list(range(1, 11))
ax.plot(events, equity_end, marker="o", label="equity, ending value")
ax.plot(events, bond_paid, marker="s", label="bond, amount paid")
ax.set_xticks(events)
ax.set_xlabel("outcome, sorted by total loss")
ax.set_ylabel("money units")
ax.set_title("What each claimant receives in the ten outcomes")
ax.legend()
```

Over the ten outcomes, each claimant's result is a distribution:

```python
bond = Discrete(bond_paid, probs)
equity = Discrete(equity_end, probs)
print(f"bond:   expected payment {bond.mean():.2f} of {BOND * (1 + COUPON):.1f} promised, "
      f"P(default) {bond.cdf(BOND * (1 + COUPON) - 1e-9):.1f}")
print(f"equity: expected ending value {equity.mean():.2f} on {EQUITY:.1f} invested, "
      f"P(wiped out) {equity.cdf(1e-9):.1f}, expected return {equity.mean() / EQUITY - 1:.1%}")
```

```text
bond:   expected payment 18.54 of 20.6 promised, P(default) 0.1
equity: expected ending value 34.86 on 27.8 invested, P(wiped out) 0.1, expected return 25.4%
```

The bond promises 3%, but it has a one-in-ten chance of losing everything.
Its expected payment is 18.54 on 20, a return of -7.3%, so the bondholders are
underpaid for their risk. The shareholders are last in line and expect 25.4%.

The monograph's main text uses total capital `Q = 47.8` and leaves the split
between debt and equity to reporting. The
[chapter 3 and 4 guide](capital-modeling-monograph-15-theory.md) shows how
the same layers of claimants define the critical loss levels.

## 6. Evaluating a model

Chapter 7 asks three questions about your own model: does it do what the
charter specified, does it calculate correctly, and is it realistic? The third
is validation, in four steps: best estimates that agree with plan, variability
that matches experience, back testing, and repetition.

Back testing uses the probability integral transform (PIT). For each past
year, the PIT is the model's probability of a result no worse than the actual
result. Over many years, a good model gives uniform PIT values. Values at the
ends show bias or too little spread. Values in the middle show too much
spread.

The example uses InsCo as the model and six past totals (hypothetical, for
illustration):

```python
from prospicio.models import pit_from_draws, pit_histogram

history = [38.0, 61.0, 45.0, 52.0, 70.0, 33.0]
pits = [pit_from_draws([float(t) for t in totals], y) for y in history]
pit = pd.DataFrame({"year": [str(k) for k in range(1, 7)], "total": history, "pit": pits})
(GT(pit, rowname_col="year")
 .tab_header("Back test: the PIT of six past totals", "InsCo as the model; totals are hypothetical")
 .fmt_number("total", decimals=1)
 .fmt_number("pit", decimals=2)
 .cols_label(total="past total", pit="PIT"))
```

```text
   past total   PIT
1        38.0  0.30
2        61.0  0.80
3        45.0  0.70
4        52.0  0.70
5        70.0  0.90
6        33.0  0.20
```

```python
print("in quintiles", pit_histogram(pits, 5))
```

```text
in quintiles [0, 2, 0, 2, 2]
```

With six values, only a gross failure shows, as the monograph warns. Keep the
record and repeat the check every year. For fitted models, `prospicio.models`
has `ppc_check` and `loo_pit`. The Rust crate also has a diagonal backtest of
the reserving models against each origin's next diagonal.

The checklist of section 7.3 for a modeling platform, answered for prospicio:

| Item | prospicio |
|---|---|
| Accounting bases | the P&L ledger for signed premium, loss and expense legs, and the balance sheet in code (section 5) |
| Loss payout patterns | `PaymentPattern`, discounting and `price_over_payout`, and reserving models for run-off |
| Assets | an ESG's output, joined as columns |
| Dependency | Iman-Conover (and by blocks), Gaussian, t, Archimedean, nested Archimedean and vine copulas, common shocks through event sets |
| Required capital | VaR, TVaR, EPD and assets at a standard (`Portfolio.assets`, `assets_for_epd`) |
| Required margins and their allocation | distortions, calibration, the natural allocation, the range of unit premiums (`premium_bounds`), Bodoff |
| Parameter uncertainty | Bayesian models and predictive distributions, blending of models |
| Input and output | Python, R and an Excel add-in, with distributions and towers saved as JSON |
| Scripting and automation | every step is a library call, so runs over many assumptions are loops |
| Version control | plain code and JSON, so any version control works |
| Track record | results checked against published references, as in these guides |
