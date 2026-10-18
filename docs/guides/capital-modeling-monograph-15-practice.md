# Building and judging a capital model: CAS Monograph 15, chapters 1, 2 and 7

Chapters 1, 2 and 7 of CAS Monograph 15 (John Major and Stephen Mildenhall,
*Introduction to Capital Modeling and Portfolio Management*, 2026) are about
the system around the mathematics: what a capital model is for, how it is
organised, what its business operations module must produce, and how to
judge a model or a platform. They are mostly prose and checklists; read the
monograph for them. This guide keeps to the parts with numbers, the InsCo
example's simulation table, balance sheets and pro forma statements, and maps
the monograph's structure and checklists onto prospicio, so you can see which
piece of a capital model each object plays. The explanations are in new
words, and table numbers are the monograph's.

Companion guides: [chapters 3 and 4](capital-modeling-monograph-15-theory.md),
[chapters 5 and 6](capital-modeling-monograph-15-applications.md),
[chapter 8](capital-modeling-monograph-15-iman-conover.md) and
[chapter 9](capital-modeling-monograph-15.md).

## 1. The parts of a capital model

The monograph's system (sections 1.4.4 and 1.5) has three modules. Business
Operations simulates the firm's results; Capital Adequacy turns them into
the assets needed for a risk tolerance; Pricing and Allocation splits those
assets into premium and capital and allocates the margin to units. Inside
Business Operations, units produce gross losses, an optional economic
scenario generator drives assets and inflation, reinsurance turns gross into
net, and accounting turns results into pro forma statements. In prospicio:

| Monograph | prospicio |
|---|---|
| Units: losses by unit, frequency and severity | `Severity` families, `Count`, `fft` (compound distributions), `simulate_events`, event loss tables (`EventSet`), reserving models (`MackBootstrap`, `OdpBootstrap`, `BayesOdp`) |
| Correlation between units | `iman_conover`, `reorder_groups` (block Iman-Conover), copulas with `simulate` ([chapter 8 guide](capital-modeling-monograph-15-iman-conover.md)) |
| A sample of joint outcomes (the "probabilistic database") | `PredictiveDistribution`: one row per simulated year, one column per unit |
| Reinsurance, gross to net | towers and treaties in `prospicio.reinsurance`, or a ceded unit with negative losses (section 4) |
| Economic scenario generator | not provided; bring the scenarios in as columns of a `PredictiveDistribution` |
| Accounting and pro formas | the P&L ledger (`Ledger` in `prospicio.aggregate`) for premium, loss and expense legs; balance sheets are left to the user (section 5) |
| Capital Adequacy | `quantile`, `tvar`, `Portfolio.assets`, `Portfolio.epd`, `Portfolio.priority` |
| Pricing and Allocation | `Distortion`, `Portfolio.calibrate`, `Portfolio.price` (the natural allocation), `Portfolio.bodoff` |

The monograph separates structural parameters, changed rarely (volatility,
correlation), from business parameters, changed every planning run (premium,
loss ratio). In code the line is the same: build the units' shapes once, and
let volumes and the target return be arguments.

## 2. InsCo: the simulation table and the funding equation

InsCo (sections 1.6 and 2.2) writes three units for one period and has ten
equally likely outcomes. Table 2.1 lists them sorted by total, with each
unit's expected loss, CV and plan premium:

```python
from prospicio.distributions import Discrete
from prospicio.pricing import Portfolio

ROWS = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
        [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]
PLAN = [13.9, 18.7, 19.6]
insco = Portfolio(["A", "B", "C"], ROWS)
totals = [sum(r) for r in ROWS]
columns = [*zip(*ROWS), totals]
probs = [0.1] * 10

print(f"{'':10s} {'A':>7s} {'B':>7s} {'C':>7s} {'Total':>7s}")
dists = [Discrete([float(v) for v in col], probs) for col in columns]
print(f"{'EX':10s} " + " ".join(f"{d.mean():7.1f}" for d in dists))
print(f"{'CV':10s} " + " ".join(f"{d.std_dev() / d.mean():7.3f}" for d in dists))
print(f"{'skewness':10s} " + " ".join(f"{d.skewness():7.3f}" for d in dists))
print(f"{'plan':10s} " + " ".join(f"{p:7.1f}" for p in PLAN + [sum(PLAN)]))
```

```text
                 A       B       C   Total
EX            13.4    18.3    14.9    46.6
CV           0.453   0.412   1.324   0.455
skewness     0.281   0.269   1.603   1.420
plan          13.9    18.7    19.6    52.2
```

The means and CVs are Table 2.1's (13.4, 18.3, 14.9 and 46.6; 0.453, 0.412,
1.324 and 0.455); the skewness, which the table does not print, shows how
much of unit C's risk sits in one or two years. The monograph uses the
population standard deviation (dividing by 10, not 9), as `Discrete` does.

Section 1.6's funding equation says the assets are premium plus capital,
`a = P + Q`. With InsCo's assets of 100 and the plan premium of 52.2, capital
is 47.8, and the plan's expected margin is the premium less the expected
loss:

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

The expected margin of 5.6 is the one section 2.2 states. The required
premium at a 15% return is higher, 53.565 (see the
[chapter 9 guide](capital-modeling-monograph-15.md)): at plan, InsCo's
investors expect 11.7%, not 15%.

## 3. Margins, leverage and returns by unit

Section 1.4.2 (Table 1.1) contrasts attritional business, written at a high
loss ratio and high leverage, with catastrophe business, written at a low
loss ratio and low leverage, and notes that capital behind attritional
business loses money about half the time, capital behind catastrophe
business rarely. InsCo's unit A is the steady one and unit C the
catastrophe-like one. With the Wang distortion of chapter 4 and the capital
allocation of section 8.3:

```python
wang = insco.calibrate("wang", assets=100, return_on_capital=0.15)
priced = insco.price(wang, assets=100, allocation="lifted")
print(f"{'':2s} {'premium':>8s} {'loss ratio':>11s} {'P:Q':>6s} {'return':>7s} {'P(loss > premium)':>18s}")
for name, u, col in zip(insco.units, priced.allocated, columns):
    p_loss = sum(1 for x in col if x > u.premium) / 10
    print(f"{name:2s} {u.premium:8.3f} {u.loss_ratio:11.1%} {u.premium_to_capital:6.2f} "
          f"{u.return_on_capital:7.1%} {p_loss:18.1f}")
```

```text
    premium  loss ratio    P:Q  return  P(loss > premium)
A    14.109       95.0%   1.68    8.4%                0.6
B    18.637       98.2%   2.14    3.9%                0.6
C    20.819       71.6%   0.71   20.2%                0.2
```

The loss ratio and leverage columns follow Table 1.1: unit A is written at a
95% loss ratio with premium 1.7 times its capital, unit C at 72% with premium
0.7 times its capital, and A's premium is exceeded by its loss in six years
of ten, C's in two. The return column does not: on InsCo the volatile unit
earns the most, 20.2%. Table 1.1 describes market economics for very
different business, while InsCo's ten years are too few to separate the
units' layers cleanly; part of C's margin is earned in the bottom layer of
the assets, which holds no capital at all (section 9 of the
[chapter 8 guide](capital-modeling-monograph-15-iman-conover.md)).

Section 1.4.3 also lists the flavours of a premium (filed, required, plan,
indicated, priced, quoted, sold, market) and of a margin, where actual, the
margin after the fact, is the only one that is not expected. For InsCo the
actual margin at the plan premium is a distribution:

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
with negative losses beside the gross units, or downstream of the units,
applied to their results. Both give the same net result. With InsCo's 35 xs
65 aggregate stop loss (section 2.3; priced in chapter 5):

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

The monograph recommends the downstream form when a cover protects several
units at once, as a corporate catastrophe cover does, since the cover then
applies to their sum. A `Portfolio`, which prices units by their share of the
total, needs non-negative units, so for pricing the cover becomes a unit of
recoveries beside the net book, as in the
[applications guide](capital-modeling-monograph-15-applications.md).

## 5. Capital structure and pro forma statements

Sections 2.5 to 2.7 give InsCo a capital structure and turn each outcome into
financial statements. The assets of 100 in cash are funded by the unearned
premium of 52.2, a bond of 20 with a 3% coupon, and shareholder equity of 27.8
(Table 2.2). There are no expenses or investment income. At the end of the
period, losses are paid first, then the bond and its coupon, and equity takes
what is left. For outcome 4 (a total loss of 40), Tables 2.3 and 2.4 show an
income of 11.6 to shareholders and ending equity of 39.4. Every outcome gives
such a pair of statements:

```python
CASH, UPR, BOND, COUPON = 100.0, sum(PLAN), 20.0, 0.03
EQUITY = CASH - UPR - BOND
print(f"opening balance sheet: cash {CASH:.1f} = premium reserve {UPR:.1f} + bond {BOND:.1f} "
      f"+ equity {EQUITY:.1f}")
print(f"{'event':>5s} {'loss':>6s} {'coupon':>7s} {'net income':>11s} {'end equity':>11s} {'bond paid':>10s}")
bond_paid, equity_end = [], []
for event, loss in enumerate(totals, start=1):
    after_claims = max(CASH - loss, 0.0)
    to_bond = min(after_claims, BOND * (1 + COUPON))
    to_equity = after_claims - to_bond
    bond_paid.append(to_bond)
    equity_end.append(to_equity)
    income = UPR - loss - BOND * COUPON
    print(f"{event:5d} {loss:6.1f} {BOND * COUPON:7.1f} {income:11.1f} {to_equity:11.1f} {to_bond:10.1f}")
```

```text
opening balance sheet: cash 100.0 = premium reserve 52.2 + bond 20.0 + equity 27.8
event   loss  coupon  net income  end equity  bond paid
    1   22.0     0.6        29.6        57.4       20.6
    2   28.0     0.6        23.6        51.4       20.6
    3   36.0     0.6        15.6        43.4       20.6
    4   40.0     0.6        11.6        39.4       20.6
    5   40.0     0.6        11.6        39.4       20.6
    6   40.0     0.6        11.6        39.4       20.6
    7   40.0     0.6        11.6        39.4       20.6
    8   55.0     0.6        -3.4        24.4       20.6
    9   65.0     0.6       -13.4        14.4       20.6
   10  100.0     0.6       -48.4         0.0        0.0
```

Outcome 4's net income of 11.6 and ending equity of 39.4 are Tables 2.3 and
2.4. In outcome 10 the loss of 100 takes all the assets: policyholders are
paid in full, the income statement shows a loss of 48.4, and the
shareholders' 27.8 is not enough to absorb it, so the bondholders lose their
20.6 of principal and coupon as well. Read across the ten
statements, the model answers the questions a capital model must answer for
each claimant:

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

The bond promises 3%, but with a one-in-ten chance of losing everything it
is expected to pay 18.54 on 20, a return of -7.3%: at this coupon the
bondholders are underpaid for their risk, and the shareholders, who stand
last, expect 25.4%. The monograph's main text works
with total capital `Q = 47.8` and leaves the split between debt and equity to
reporting; the [chapter 3 and 4 guide](capital-modeling-monograph-15-theory.md)
shows how the same layering of claimants defines the critical loss levels.

## 6. Evaluating a model

Chapter 7 asks three questions of your own model: does it do what the
charter specified, does it calculate correctly, and is it realistic. The
last is validation, in four steps: best estimates that reconcile to plan,
variability that matches experience, back testing, and repeating it. The
third, back testing, needs the probability integral transform (PIT): for
each past year, the model's probability of a result no worse than what
happened. Over many years the PIT values should look uniform; values bunched
at the ends mean bias or too little spread, bunched in the middle too much
spread. With InsCo as the model and six past totals (hypothetical, for
illustration):

```python
from prospicio.models import pit_from_draws, pit_histogram

history = [38.0, 61.0, 45.0, 52.0, 70.0, 33.0]
pits = [pit_from_draws([float(t) for t in totals], y) for y in history]
print("PIT values  ", [round(p, 2) for p in pits])
print("in quintiles", pit_histogram(pits, 5))
```

```text
PIT values   [0.3, 0.8, 0.7, 0.7, 0.9, 0.2]
in quintiles [0, 2, 0, 2, 2]
```

With six values only a gross failure would show, as the monograph warns; the
point is to keep the record and repeat it every year. For fitted models
prospicio has the same checks built in: `ppc_check` and `loo_pit` in
`prospicio.models`, and, in the Rust crate, a diagonal backtest of the
reserving models against each origin's next diagonal.

Section 7.3's checklist for a modeling platform, answered for prospicio:

| Item | prospicio |
|---|---|
| Accounting bases | none built in; the P&L ledger gives signed premium, loss and expense legs, and the balance sheet is the user's (section 5) |
| Loss payout patterns | `PaymentPattern`, discounting and `price_over_payout`; reserving models for run-off |
| Assets | not modelled; an ESG's output can be joined as columns |
| Dependency | Iman-Conover (and by blocks), Gaussian, t, Archimedean, nested Archimedean and vine copulas, common shocks through event sets |
| Required capital | VaR, TVaR, EPD and assets at a standard (`Portfolio.assets`, `assets_for_epd`) |
| Required margins and their allocation | distortions, calibration, the natural allocation, the range of unit premiums (`premium_bounds`), Bodoff |
| Parameter uncertainty | Bayesian models and predictive distributions, blending of models |
| Input and output | Python, R and an Excel add-in; distributions and towers save as JSON |
| Scripting and automation | everything is a library call, so runs over many assumptions are loops |
| Version control | plain code and JSON, so any version control works |
| Track record | none in regulatory internal models; results are checked against published references, as in these guides |

## 7. What this guide leaves out

- **The charter and organisation** (sections 1.1 to 1.3): who uses the
  model, for which decisions, and with what staff. Nothing to compute.
- **Asset risk and the economic scenario generator** (section 2.4): not part
  of prospicio, and InsCo has none.
- **Third-party models** (section 7.2): regulatory and rating agency
  formulas, outside the library.
