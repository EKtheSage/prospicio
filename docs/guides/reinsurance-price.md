# What a reinsurance price is

<!-- site: execute -->

This guide is for an actuary who prices primary insurance and has a loss ratio
for a reinsurance submission. It answers four questions, then works through one
example.

## Answers in brief

- **What does reinsurance pricing produce?** The **terms** (layers or share,
  attachments, limits, reinstatements, commissions), the **price** for those
  terms, and the **expected economics**: expected loss, margin, capital held and
  the return on that capital.
- **What should we charge?** For an excess of loss layer, the technical premium
  is (expected loss + return asked × capital held) / (1 - expenses -
  brokerage). For a quota share, the price is the ceding commission: 1 -
  expected loss ratio - expenses - brokerage - margin. Both are floors for a
  reinsurer with no other business. A reinsurer whose book diversifies the
  contract can accept less.
- **How do we price without a rating plan?** From a loss model of the cedant's
  book: how many large losses, how big, how heavy the tail. The cedant's own
  large-loss history gives one estimate (experience rating). Exposure curves on
  its profile of limits and sums insured give another (exposure rating).
- **What does a hard market mean?** Higher prices and less cover. A price rise
  has three parts: a higher expected loss (trend, development, a new view of
  the tail), a higher **price of risk** (the return the market demands on
  capital) and tighter terms. A loss model separates the first part from the
  other two.

For the 5x5 layer of the running example (all money in millions):

| Output | Value | Where |
|---|---|---|
| Expected loss | 0.586 | section 1 |
| Capital held (99% TVaR standard) | 6.54 | section 1 |
| Risk premium: expected loss plus a 10% return on capital | 1.240 | section 1 |
| Technical premium: the risk premium after 3% expenses and 10% brokerage | 1.425 | section 2 |
| Rate on line, multiple, loss ratio of the technical premium | 28.5%, 2.43, 41.1% | section 2 |
| With a paid reinstatement: deposit plus expected reinstatement premium | 1.282 + 0.143 | section 2 |
| Technical premium inside a book: unrelated, and at correlation 0.5 | 0.768, 1.089 | section 8 |
| A soft and a hard market quote | 1.10, 1.70 | section 7 |

## 1. What is priced

A primary insurer prices a policy: one risk, one year, a rating plan. A
reinsurer prices a **contract** on a **portfolio** for a **period**, from a
model of the cedant's losses.

The running example is a synthetic commercial property book with per-risk
excess of loss cover. Money is in millions. The cedant writes 60 of subject
premium (the premium the reinsurance applies to). Losses below 1 are
**attritional**: frequent, stable, expected total 28. Losses above 1 are the
**large losses** that excess of loss covers. Their count is Poisson with mean 4
a year. Their size has a Pareto tail with alpha 1.8, capped at the policy limit
(the largest sum insured) of 20; below alpha 2 the variance is infinite without
a cap. The numbers show the mechanics and do not match a market.

```python
import math

from prospicio.aggregate import fft
from prospicio.distributions import Grid, Lognormal, NegativeBinomial, Pareto, Poisson, PredictiveDistribution
from prospicio.pricing import PaymentPattern, Pentagon, Quote, price, price_over_payout, price_portfolio
from prospicio.reinsurance import Layer, Tower, occurrence_layer_statistics
from prospicio.risk import Distortion, GaussianCopula, calibrate

SUBJECT_PREMIUM = 60.0   # premium the reinsurance applies to
ATTRITIONAL = 28.0       # expected losses below 1
POLICY_LIMIT = 20.0      # the largest loss a single risk can produce
STEP = 0.1
POINTS = 2**14           # grid points for the annual distribution


def severity_grid(alpha):
    """Losses above 1 with a Pareto tail, censored at the policy limit, on the grid."""
    grid, _ = Grid.rounding(Pareto(1.0, alpha), STEP, round(POLICY_LIMIT / STEP) + 1)
    return grid


frequency = Poisson(4.0)
severity = severity_grid(1.8)
gross, report = fft(frequency, severity, POINTS)
print(f"mean large loss {severity.mean():.2f}, expected large losses a year {gross.mean():.2f}")
print(f"1-in-10 year {gross.quantile(0.90):.1f}, 1-in-100 year {gross.quantile(0.99):.1f}")
print(f"expected loss ratio {(ATTRITIONAL + gross.mean()) / SUBJECT_PREMIUM:.1%}")
```

```text
mean large loss 2.14, expected large losses a year 8.54
1-in-10 year 16.2, 1-in-100 year 29.5
expected loss ratio 60.9%
```

The library works on a **grid** that rounds each size to a step of 0.1. `fft`
combines frequency and severity into the annual total, with no simulation
error. The reinsurer needs the whole distribution, not the 60.9% mean: a layer
above a high attachment depends on the tail.

### The pricing pentagon

Five quantities describe a priced contract: the expected **loss** (L), the
**premium** (P), the **margin** (M = P - L), the **capital** (Q) and the
**assets** (a = P + Q). Any three independent quantities set the other two. The
**return on capital**, M / Q, can replace one of the three. `Pentagon.solve`
takes the loss, the assets and the return.

Take the 5x5 layer with one free reinstatement (the limit is restored once at
no charge, so the layer pays at most 10 a year). Set assets at the 99% **tail
value at risk** (TVaR) of the annual loss, the average of the worst 1% of
years, and ask for a 10% return on capital. The result is the **risk
premium**: expected loss plus margin, before expenses and brokerage.

```python
TAIL = Distortion.tvar(0.99)   # the capital standard: 99% TVaR of the annual loss
RETURN = 0.10                  # the return on capital the reinsurer asks for

mid = Layer("5x5", limit=5.0, attachment=5.0, reinstatements=1)
mid_grids = Tower([mid]).on_grid(frequency, severity, POINTS)
annual = mid_grids.ceded[0]    # the layer's annual loss, as a distribution
expected_loss = annual.mean()
assets = TAIL.measure(annual)

pentagon = Pentagon.solve(loss=expected_loss, assets=assets, return_on_capital=RETURN)
print(f"loss {pentagon.loss:.3f}  margin {pentagon.margin:.3f}  risk premium {pentagon.premium:.3f}")
print(f"capital {pentagon.capital:.3f}  assets {pentagon.assets:.3f}")
print(f"loss ratio on the risk premium {pentagon.loss_ratio:.1%}")
```

```text
loss 0.586  margin 0.654  risk premium 1.240
capital 6.538  assets 7.778
loss ratio on the risk premium 47.3%
```

The layer pays about 0.59 a year, but the reinsurer holds 7.8 of assets for the
bad years. The risk premium supplies 1.2; the reinsurer supplies the other 6.5
as capital and wants 10% on it, the margin of 0.65.

## 2. An excess of loss layer

An **excess of loss** (XoL) layer pays the part of each loss between its
attachment and attachment plus limit. First its shape: how often a loss reaches
the layer, how often one exhausts it, and the cost per unit of cover.

```python
stats = occurrence_layer_statistics(frequency, severity, share=1.0, limit=5.0, attachment=5.0)
print(f"of the large losses, {stats.pr_attach:.2%} reach the layer and {stats.pr_detach:.2%} exhaust it")
print(f"expected losses reaching the layer a year {stats.count.mean:.3f}")
print(f"chance the year has any loss to the layer {stats.pr_loss:.1%}")
print(f"expected annual loss {stats.aggregate.mean:.4f}, loss on line {stats.loss_on_line:.2%}")
print(f"with one reinstatement (annual limit 10) {annual.mean():.4f}")
```

```text
of the large losses, 5.42% reach the layer and 1.60% exhaust it
expected losses reaching the layer a year 0.217
chance the year has any loss to the layer 19.5%
expected annual loss 0.5873, loss on line 11.75%
with one reinstatement (annual limit 10) 0.5863
```

The annual limit of 10 lowers the expected loss slightly (the loss on line has
none): a year with three exhausting losses pays 10, not 15.

### The technical premium

The **technical premium** is the price that the reinsurer's own model
supports: the **expected loss**, the **cost of capital** (the return asked
times the capital the layer needs, the pentagon's margin), **internal
expenses** (here 3% of the premium) and **brokerage** (here 10%, paid to the
intermediary). Expenses and brokerage are fractions of the premium, so we
divide by them: premium = (expected loss + cost of capital) / (1 - 3% - 10%).
Three simplifications apply: real expenses are partly a fixed cost per
contract, so a flat percentage is too low for small contracts; there is no tax
line, so the 10% is a return after tax; brokerage applies to reinstatement
premium and to the deposit.

The table builds the premium by hand and with the library (`price` at the 99%
TVaR standard, then `Quote.technical` for the loads).

```python
import pandas as pd
from great_tables import GT

EXPENSES, BROKERAGE = 0.03, 0.10
KEEP = 1 - EXPENSES - BROKERAGE    # the share of each unit of premium the reinsurer keeps


def gross_up(risk_premium):
    """The price the cedant pays for a risk premium: expenses and brokerage on top."""
    return risk_premium / KEEP


capital = pentagon.capital
cost_of_capital = RETURN * capital
risk_premium = expected_loss + cost_of_capital
technical = gross_up(risk_premium)

tvar_price = price(annual, TAIL, cost_of_capital=RETURN)
technical_quote = Quote.technical(tvar_price, expenses=EXPENSES, brokerage=BROKERAGE, limit=mid.limit)

build_up = pd.DataFrame({
    "item": ["expected loss", f"cost of capital ({RETURN:.0%} of {capital:.3f})", "risk premium",
             "technical premium", "rate on line", "payback (years)", "multiple", "loss ratio"],
    "by hand": [expected_loss, cost_of_capital, risk_premium, technical, technical / mid.limit,
                mid.limit / technical, technical / expected_loss, expected_loss / technical],
    "library": [tvar_price.expected_loss, tvar_price.margin, tvar_price.premium, technical_quote.premium,
                technical_quote.rate_on_line, technical_quote.payback, technical_quote.multiple,
                technical_quote.loss_ratio],
})
(GT(build_up, rowname_col="item")
 .tab_header("Technical premium of the 5x5 layer", "Millions; expenses and brokerage take 13% of the price")
 .fmt_number(rows=[0, 1, 2, 3], decimals=3)
 .fmt_percent(rows=[4, 7], decimals=1)
 .fmt_number(rows=[5], decimals=1)
 .fmt_number(rows=[6], decimals=2))
```

```text
                                by hand  library
expected loss                     0.586    0.586
cost of capital (10% of 6.538)    0.654    0.654
risk premium                      1.240    1.240
technical premium                 1.425    1.425
rate on line                      28.5%    28.5%
payback (years)                     3.5      3.5
multiple                           2.43     2.43
loss ratio                        41.1%    41.1%
```

Brokers quote three ratios for a layer: the **rate on line** (ROL), premium
divided by limit, whose inverse is the **payback** (the years without loss that
the premium needs to pay one full limit); the **multiple**, premium divided by
expected loss; and the **loss ratio**, its inverse. This loss ratio is not the
cedant's book loss ratio of 60.9%. Later sections use `gross_up`, the same step
by hand.

A distortion with a constant cost of capital (`Distortion.ccoc`) prices a loss
as (expected loss + r × assets) / (1 + r). `TowerGrids.price` applies it to
each layer with the most the layer can pay in a year as its assets, here the
annual limit of 10: a different capital standard from the 99% TVaR. The
full-limit formula agrees with `TowerGrids.price`, and `price` agrees with the
pentagon of section 1:

```python
by_hand = (expected_loss + RETURN * mid.aggregate_limit) / (1 + RETURN)
by_call = mid_grids.price(Distortion.ccoc(RETURN)).ceded[0]
print(f"full-limit capital: by hand {by_hand:.4f}, TowerGrids.price {by_call.premium:.4f}")
print(f"99% TVaR capital:   Pentagon {pentagon.premium:.4f}, price {tvar_price.premium:.4f}")
```

```text
full-limit capital: by hand 1.4421, TowerGrids.price 1.4421
99% TVaR capital:   Pentagon 1.2401, price 1.2401
```

The capital standard is a pricing decision: a full-limit standard charges as if
the reinsurer will lose the whole limit, a tail standard for the loss it
expects in bad years. The rest of this guide uses the 99% TVaR standard.

### Reinstatements

Property and catastrophe covers usually have paid reinstatements; casualty
working layers often have free ones or only an annual limit. Two more annual
terms change what a layer pays: an **annual aggregate deductible** (the cedant
keeps the first part of the year's layer losses) and an **annual aggregate
limit** (the most the layer pays in a year).

With a paid reinstatement, the reinsurer charges a **deposit premium** at
inception and, each time a loss uses limit that is restored, a **reinstatement
premium**: here 100% of the deposit, pro rata to the amount reinstated. The
first column sets the deposit so that deposit plus expected reinstatement
premium equals the technical premium. The second holds capital against the loss
net of the reinstatement premium that the reinsurer keeps after expenses and
brokerage, and solves for the deposit again.

```python
RATE = 1.0   # reinstatement premium as a fraction of the deposit, pro rata to the amount reinstated
used_limit = annual.layer(5.0, 0.0) / 5.0    # expected share of the first limit used up in a year
deposit = technical / (1 + RATE * used_limit)

paid = Layer("5x5", limit=5.0, attachment=5.0, premium=deposit, reinstatement_rates=[RATE])
paid_grids = Tower([paid]).on_grid(frequency, severity, POINTS)
expected_reinstatement = paid_grids.expected_reinstatement_premium[0]
total = paid_grids.expected_premium[0] + expected_reinstatement

net_deposit = deposit
for _ in range(20):   # the reinstatement premium depends on the deposit, so iterate
    net_loss, _ = annual.map(lambda x: x - KEEP * RATE * net_deposit * min(x, 5.0) / 5.0)
    net = price(net_loss, TAIL, cost_of_capital=RETURN)
    net_deposit = net.premium / KEEP
net_total = net_deposit * (1 + RATE * used_limit)

reinstated = pd.DataFrame({
    "item": ["capital held", "deposit", "deposit rate on line", "expected reinstatement premium",
             "expected total", "change on the technical premium"],
    "priced on the gross loss": [capital, deposit, deposit / mid.limit, expected_reinstatement, total,
                                 total / technical - 1],
    "priced net of reinstatement premium": [net.capital, net_deposit, net_deposit / mid.limit,
                                            net_total - net_deposit, net_total, net_total / technical - 1],
})
(GT(reinstated, rowname_col="item")
 .tab_header("5x5 layer with one paid reinstatement at 100%", "Millions; the technical premium is 1.425")
 .fmt_number(rows=[0, 1, 3, 4], decimals=3)
 .fmt_percent(rows=[2], decimals=1)
 .fmt_percent(rows=[5], decimals=1, force_sign=True))
```

```text
                                 priced on the gross loss  priced net of reinstatement premium
capital held                                        6.538                                5.698
deposit                                             1.282                                1.196
deposit rate on line                                25.6%                                23.9%
expected reinstatement premium                      0.143                                0.133
expected total                                      1.425                                1.329
change on the technical premium                      0.0%                                -6.8%
```

The market quotes rates on line on the deposit, so the comparable figure is
25.6%, not 28.5%. A paid reinstatement moves premium from good years to bad
years, and it lowers the capital the reinsurer needs, so it is a real discount.
The rest of the guide uses the free reinstatement: each quote is the whole
premium.

## 3. Where the layer sits

Price three layers of the same program: 2x3, a working layer that attaches at 3
and that losses reach often; 5x5; and 10x10, a top layer that attaches at 10,
that losses reach rarely, and that pays a large amount when they do. Together
they cover losses from 3 to 20, the policy limit. Use **one** pricing rule for
all three.

The rule is a **distortion**: it prices a loss as a weighted average that gives
bad outcomes more weight than their probability. The **proportional hazard**
distortion raises the probability of exceeding each loss size to a power below
1. A parameter of 1 gives no loading; a lower parameter gives a larger one.
`calibrate` finds the parameter that gives a specified premium on one layer,
here the 5x5 risk premium, and that parameter prices the other two layers.

```python
LAYERS = [Layer("2x3", limit=2.0, attachment=3.0, reinstatements=1), mid,
          Layer("10x10", limit=10.0, attachment=10.0, reinstatements=1)]
tower_grids = Tower(LAYERS).on_grid(frequency, severity, POINTS)
price_of_risk = calibrate("ph", tower_grids.ceded[1], pentagon.premium)
print(f"proportional hazard parameter {price_of_risk.parameters['rho']:.3f}: a 1% chance is weighted as "
      f"{price_of_risk.g(0.01):.1%}, a 10% chance as {price_of_risk.g(0.10):.1%}")
```

```text
proportional hazard parameter 0.686: a 1% chance is weighted as 4.3%, a 10% chance as 20.6%
```

```python
layer_rows = []
for layer, grid in zip(LAYERS, tower_grids.ceded):
    own = price(grid, TAIL, distortion=price_of_risk)   # premium by the distortion, assets at 99% TVaR
    quoted = Quote.technical(own, expenses=EXPENSES, brokerage=BROKERAGE, limit=layer.limit)
    at_target = Quote.technical(price(grid, TAIL, cost_of_capital=RETURN),   # exactly 10% on that capital
                                expenses=EXPENSES, brokerage=BROKERAGE)
    layer_rows.append({"layer": layer.name, "exp loss": own.expected_loss,
                       "assets / EL": own.assets / own.expected_loss, "price": quoted.premium,
                       "multiple": quoted.multiple, "ROL": quoted.rate_on_line,
                       "return": own.return_on_capital, "multiple at 10%": at_target.multiple})
(GT(pd.DataFrame(layer_rows), rowname_col="layer")
 .tab_header("Three layers under one proportional hazard rule", "Millions; price includes 3% expenses and 10% brokerage")
 .fmt_number(["exp loss", "price"], decimals=3)
 .fmt_number("assets / EL", decimals=1)
 .fmt_number(["multiple", "multiple at 10%"], decimals=2)
 .fmt_percent(["ROL", "return"], decimals=1))
```

```text
       exp loss  assets / EL  price  multiple    ROL  return  multiple at 10%
2x3       0.686          5.8  1.271      1.85  63.6%   14.5%             1.65
5x5       0.586         13.3  1.425      2.43  28.5%   10.0%             2.43
10x10     0.337         31.1  1.160      3.44  11.6%    7.1%             4.30
```

**assets / EL** is the 99% TVaR of the layer's annual loss per unit of expected
loss. **price** is the distortion's risk premium grossed up for expenses and
brokerage (the 5x5 row agrees with section 2). **return** is the margin over
the 99% TVaR capital at that price. **multiple at 10%** comes from a rule that
charges exactly 10% on each layer's capital (`price` with `cost_of_capital`).

The figure compares the two rules layer by layer:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
names = [layer.name for layer in LAYERS]
by_distortion, by_return = [], []
for grid in tower_grids.ceded:
    by_distortion.append(Quote.technical(price(grid, TAIL, distortion=price_of_risk),
                                         expenses=EXPENSES, brokerage=BROKERAGE).multiple)
    by_return.append(Quote.technical(price(grid, TAIL, cost_of_capital=RETURN),
                                     expenses=EXPENSES, brokerage=BROKERAGE).multiple)
xs = range(len(names))
ax.bar([x - 0.2 for x in xs], by_distortion, width=0.4, label="proportional hazard rule")
ax.bar([x + 0.2 for x in xs], by_return, width=0.4, label="10% on the 99% TVaR capital")
ax.set_xticks(list(xs), names)
ax.set_xlabel("layer (limit x attachment)")
ax.set_ylabel("price as a multiple of the expected loss")
ax.set_title("Same program, two pricing rules")
ax.legend(loc="upper left")
```

The multiple increases from the working layer to the top layer under the same
rule, because of the capital behind each unit of expected loss. The working
layer pays often and predictably (about 6 of assets per unit). The top layer's
expected loss is a small probability of a very large payment (about 31).

The proportional hazard rule does not give a constant return on capital: a
constant-return rule makes the working layer cheaper and the top layer dearer,
because a one-parameter distortion charges the remote tail less. Calibrate to
market quotes on more than one layer. The rate on line moves the other way,
because the top layer pays much less often: it is the cost per unit of limit,
the multiple the loading for risk. A loss ratio alone cannot price a layer. Two
layers with the same expected loss have different prices when one is a
1-in-100 event.

## 4. Quota share

A **quota share** takes a fixed share of the premium and losses of each policy.
The reinsurer pays the cedant a **ceding commission** for its acquisition
costs, so the price of a quota share is the commission. The **technical
commission** is what remains after the reinsurer's other needs:

technical commission = 1 - expected loss ratio - reinsurer expenses - brokerage - margin

A quota share carries all of the cedant's loss-ratio risk, so the attritional
losses cannot sit at their mean. Here they get a lognormal spread with a
coefficient of variation of 10%, for pricing, inflation and reserve
uncertainty. Take a 30% quota share of the whole book. The margin comes from
the same pentagon: assets at the 99% TVaR of the ceded loss, and a 10% return
on the capital.

<details>
<summary>Helpers: add two independent losses, expectations and tail means over the book</summary>

```python
def add_independent(a, b):
    """The total of two independent losses on grids with the same step, as a grid."""
    total = [0.0] * (len(a.probs) + len(b.probs))
    bs = [(j, q) for j, q in enumerate(b.probs) if q > 1e-16]
    for i, p in enumerate(a.probs):
        if p > 1e-16:
            for j, q in bs:
                total[i + j] += p * q
    return Grid(a.step, total)


def expect(f):
    """The expected value of f over the book's annual loss."""
    return sum(p * f(x) for x, p in book)


def tail_mean(values, level=0.99):
    """Mean of the worst (1 - level) of outcomes; values is a list of (value, probability)."""
    left, total = 1 - level, 0.0
    for v, p in sorted(values, reverse=True):
        take = min(p, left)
        total, left = total + v * take, left - take
        if left <= 0:
            break
    return total / (1 - level)
```

</details>

```python
CESSION = 0.30
ATTRITIONAL_CV = 0.10
ceded_premium = CESSION * SUBJECT_PREMIUM

attritional, _ = Grid.local_moment(Lognormal.from_mean_cv(ATTRITIONAL, ATTRITIONAL_CV), STEP, 600)
book_grid = add_independent(attritional, gross)    # the whole book's annual loss
book = [(k * STEP, p) for k, p in enumerate(book_grid.probs) if p > 1e-16]   # as (value, probability)
loss_ratio = book_grid.mean() / SUBJECT_PREMIUM

qs = Layer.quota_share("QS", CESSION).with_premium_rate(1.0, SUBJECT_PREMIUM)
qs_ceded = Tower([qs]).on_annual_grid(book_grid).ceded[0]   # the ceded share of each year's total
qs_price = price(qs_ceded, TAIL, cost_of_capital=RETURN)
margin = qs_price.margin / ceded_premium
QS_EXPENSES, QS_BROKERAGE = 0.02, 0.015
technical_commission = 1 - loss_ratio - QS_EXPENSES - QS_BROKERAGE - margin
print(f"ceded premium {ceded_premium:.1f}, expected loss ratio {loss_ratio:.1%}")
print(f"margin {margin:.1%} of premium (capital {qs_price.capital:.2f}, {RETURN:.0%} return)")
print(f"technical commission = 1 - {loss_ratio:.1%} - {QS_EXPENSES:.1%} - {QS_BROKERAGE:.1%} "
      f"- {margin:.1%} = {technical_commission:.1%}")
```

```text
ceded premium 18.0, expected loss ratio 60.9%
margin 4.0% of premium (capital 7.25, 10% return)
technical commission = 1 - 60.9% - 2.0% - 1.5% - 4.0% = 31.6%
```

A commission above that value costs the reinsurer margin; one below it gives
margin to the cedant. For a long-tail line, the commission also depends on the
investment income before claims are paid. This one-year example has none
(section 9 restates a price over a payout).

### Sliding scales and profit commissions

With these two variants the commission depends on the result of the year, so
part of the loss-ratio volatility (not the mean) goes back to the cedant.

- A **sliding scale** pays a higher commission when the loss ratio is low and a
  lower one when it is high, linear between two points.
- A **profit commission** is a flat commission plus a share of the reinsurer's
  profit: premium less commission, losses and an allowance for management
  expenses, usually with a deficit carried forward. Here the allowance also
  includes the reinsurer's margin and there is no carry-forward. Both choices
  change the expected profit commission, so model the actual wording.

The code sets the scale and the profit share so that the **expected**
commission equals the technical commission. `Tower.on_annual_grid` gives the
exact expected commissions, and `ceding_commission_for` and
`profit_commission_for` give the commission in each outcome.

```python
costs = (QS_EXPENSES + QS_BROKERAGE) * ceded_premium
result_rows = []


def expected_commissions(layer):
    """Expected ceding and profit commission of a quota share over the book's annual loss."""
    terms = Tower([layer]).on_annual_grid(book_grid)
    return terms.expected_ceding_commission[0], terms.expected_profit_commission[0]


def underwriting_result(layer, with_profit_commission=False):
    """The reinsurer's result in each outcome: premium less commissions, losses and costs."""
    def result(x):
        ceded = CESSION * x
        pc = layer.profit_commission_for(ceded) if with_profit_commission else 0.0
        return ceded_premium - layer.ceding_commission_for(ceded) - pc - ceded - costs
    return result


def describe(name, result):
    mean = expect(result)
    sd = math.sqrt(expect(lambda x: (result(x) - mean) ** 2))
    worst = -tail_mean([(-result(x), p) for x, p in book])   # average result in the worst 1% of years
    needed = RETURN * (mean - worst) / (1 + RETURN)           # margin that earns 10% on the capital then needed
    result_rows.append({"terms": name, "mean": mean, "sd": sd, "worst 1%": worst,
                        "commission it can add": (mean - needed) / ceded_premium})
    return sd


flat = qs.with_ceding_commission(technical_commission)
sd_flat = describe("flat", underwriting_result(flat))

shape = [(0.37, 0.55), (0.26, 0.75)]   # (commission, loss ratio) points of the scale, before the shift
shift = technical_commission - expected_commissions(qs.with_sliding_scale(shape))[0] / ceded_premium
anchors = [(c + shift, lr) for c, lr in shape]    # moved so the expected commission is the technical one
slide = qs.with_sliding_scale(anchors)
sd_slide = describe("sliding scale", underwriting_result(slide))
print(f"scale: {anchors[0][0]:.1%} commission at a loss ratio of {anchors[0][1]:.0%} or less, "
      f"{anchors[1][0]:.1%} at {anchors[1][1]:.0%} or more")

# Profit commission: 25% of the profit left after an allowance (commission, costs and margin, as a
# share of premium). The flat part is the technical commission less the expected profit commission,
# which itself depends on the flat part, so solve by repeating.
commission = technical_commission
for _ in range(30):
    allowance = commission + QS_EXPENSES + QS_BROKERAGE + margin
    pc_layer = qs.with_ceding_commission(commission).with_profit_commission(0.25, allowance)
    expected_pc = expected_commissions(pc_layer)[1] / ceded_premium
    commission = technical_commission - expected_pc
sd_pc = describe("profit commission", underwriting_result(pc_layer, with_profit_commission=True))
print(f"flat commission {commission:.1%} plus 25% of profit after an allowance of {allowance:.1%} "
      f"(expected {expected_pc:.1%} of premium)")
print(f"reinsurer's sd falls {1 - sd_slide / sd_flat:.0%} with the scale and {1 - sd_pc / sd_flat:.0%} "
      f"with the profit commission")
```

```text
scale: 35.1% commission at a loss ratio of 55% or less, 24.1% at 75% or more
flat commission 30.3% plus 25% of profit after an allowance of 37.9% (expected 1.2% of premium)
reinsurer's sd falls 31% with the scale and 10% with the profit commission
```

```python
(GT(pd.DataFrame(result_rows), rowname_col="terms")
 .tab_header("The reinsurer's result under three commission forms",
             "Millions; the last column as a share of ceded premium")
 .fmt_number(["mean", "sd"], decimals=3)
 .fmt_number("worst 1%", decimals=2)
 .fmt_percent("commission it can add", decimals=1))
```

```text
                    mean     sd  worst 1%  commission it can add
flat               0.725  2.005     -7.25                   0.0%
sliding scale      0.725  1.392     -5.91                   0.7%
profit commission  0.725  1.814     -7.03                   0.1%
```

The mean is the same in all three rows: the margin the reinsurer priced. Under
flat terms, the worst 1% of years equals minus the pentagon's capital. The
scale and the profit commission lower the spread, and the cedant takes the
difference: its commission income now moves with its own loss ratio. Less
spread needs less capital. The last column is the extra expected commission the
reinsurer can pay and still earn 10% on the capital it then needs: the economic
reason for variable commissions. Its size depends on the loss-ratio
distribution.

## 5. How price follows the risk profile

### Where the loss model comes from

This guide sets the model by assumption. A reinsurer builds it in two ways and
blends them:

- **Experience rating**: trend each of the cedant's large losses to the treaty
  year and develop it to ultimate, then fit a frequency above a threshold and a
  severity tail. `LargeLossExperience` converts, develops and trends each loss
  and applies the layers to it. See
  [experience and exposure rating](experience-and-exposure-rating.md).
- **Exposure rating**: use the cedant's **risk profile**, its policies grouped
  by sum insured or limit, with the premium in each band. An **exposure curve**
  gives the share of each band's expected loss above each attachment, so a
  layer's expected loss follows directly. Use `RiskProfile` and curves such as
  `Mbbefd`.
- **Blending**: weight the two by credibility: more on experience when the
  cedant has many large losses and a stable book.

### The levers

These parts of the profile change the price of a layer:

- **frequency**: how many losses reach the program.
- **inflation and trend**: limits and attachments stay fixed, so the loss to a
  layer grows by more than the inflation rate. This leverage is the most
  important effect for excess of loss. Index clauses move the attachment with
  inflation to share it.
- **variability of the count**: a Poisson count understates the year-to-year
  change in the number of large losses when the mean count is uncertain. The
  usual correction is a negative binomial count.
- **severity tail**: how heavy it is, and where policy limits cut it.
  Uncertainty in alpha is often the largest driver of a quote.
- **limits and sums insured**: the most one risk can cost, which exposure
  rating reads from the profile.
- **catastrophe and clash**: one event can hit many risks or policies. The tail
  of the annual loss gets heavier and the typical loss does not change. The
  definition of an occurrence sets how losses add.
- **development**: reported losses grow to ultimate, large and slow for excess
  layers. This example has none.
- **loss adjustment expenses**: inside the limit, shared pro rata, or paid in
  addition, they change what the layer pays.
- **cedant quality**: underwriting and claims handling, put in the model's
  parameters and loads.

The code prices the same program with the rule of section 3 for the base book
and four variants: 25% more large losses; 5% inflation on each loss (with a
Pareto tail, this multiplies the number of losses above 1 by 1.05 to the power
1.8 and keeps their shape); a count with twice the Poisson variance; and a
heavier tail (alpha 1.5, not 1.8). The base row is in money, the others are
changes from it.

```python
profiles = {
    "base": (Poisson(4.0), 1.8),
    "25% more losses": (Poisson(5.0), 1.8),
    "5% inflation": (Poisson(4.0 * 1.05**1.8), 1.8),
    "variable count": (NegativeBinomial.from_mean_variance(4.0, 8.0), 1.8),
    "heavier tail": (Poisson(4.0), 1.5),
}
levers = {}
for name, (count, alpha) in profiles.items():
    grids = Tower(LAYERS).on_grid(count, severity_grid(alpha), POINTS)
    row = {}
    for layer, p in zip(LAYERS, grids.price(price_of_risk).ceded):
        row[f"{layer.name} loss"], row[f"{layer.name} price"] = p.loss, gross_up(p.premium)
    levers[name] = row
base = levers["base"]
for name, row in levers.items():
    if name != "base":   # changes from the base; + 0.0 turns -0 into 0
        levers[name] = {k: round(v / base[k] - 1, 2) + 0.0 for k, v in row.items()}
print(f"1.05 raised to alpha 1.8: {1.05 ** 1.8:.3f}; 1.25 raised to the hazard parameter: "
      f"{1.25 ** price_of_risk.parameters['rho']:.3f}")
```

```text
1.05 raised to alpha 1.8: 1.092; 1.25 raised to the hazard parameter: 1.165
```

```python
lever_table = pd.DataFrame.from_dict(levers, orient="index").rename_axis("book").reset_index()
(GT(lever_table, rowname_col="book")
 .tab_header("Loss and price by layer", "Base in millions; other rows as the change from base")
 .tab_spanner("2x3", ["2x3 loss", "2x3 price"])
 .tab_spanner("5x5", ["5x5 loss", "5x5 price"])
 .tab_spanner("10x10", ["10x10 loss", "10x10 price"])
 .cols_label(**{f"{n} {k}": k for n in ("2x3", "5x5", "10x10") for k in ("loss", "price")})
 .fmt_number(rows=[0], decimals=3)
 .fmt_percent(rows=[1, 2, 3, 4], decimals=0, force_sign=True))
```

```text
                  loss  price   loss  price   loss  price
base             0.686  1.271  0.586  1.425  0.337  1.160
25% more losses   +24%   +17%   +25%   +18%   +25%   +17%
5% inflation       +9%    +7%    +9%    +7%    +9%    +6%
variable count     -1%     0%     0%    +1%     0%    +1%
heavier tail      +47%   +33%   +78%   +54%  +120%   +77%
```

A quarter more losses raises each layer's expected loss by a quarter but the
price by less: with the distortion fixed, a proportional hazard price grows
roughly as frequency raised to the parameter (the second number above). So the
multiple falls for a worse book. A reinsurer does not accept that, and usually
raises the price of risk too.

Inflation of 5% raises each layer's expected loss by about 9%, near 1.05 raised
to the power 1.8 (the first number above). With a Pareto tail and fixed limits
this leverage is the same at each layer; with a lighter tail it grows with the
attachment.

A more variable count has almost no effect here: few losses reach a layer, and
most of the extra variance goes away in that thinned count. It matters more for
covers of the annual total, such as a quota share or a stop loss.

A heavier tail moves the layers apart: a book with the same frequency and a
different tail is a different risk.

## 6. Stand-alone and as part of a book

A reinsurer holds a portfolio, and the capital a contract needs depends on the
other contracts. If the contract and the book rarely have bad years together,
the contract adds little to the book's tail. If they coincide, it needs more.

Add the 5x5 layer to a book with expected loss 30 and a coefficient of
variation of 25%, once independent and once at a correlation of 0.5 between
their annual losses. A **Gaussian copula** simulates the pairs, and a
`PredictiveDistribution` holds them, one column per part. `price_portfolio`
prices the whole and **allocates** the price to its parts with the **natural
allocation**: each part gets the share of the total capital that it contributes
in the bad years, so the parts add to the whole exactly. The **marginal**
capital is the extra capital the book needs because of the contract. The
tables show both.

```python
other_book = Lognormal.from_mean_cv(30.0, 0.25)
DRAWS = 40_000


def in_a_book(correlation):
    u = GaussianCopula([[1.0, correlation], [correlation, 1.0]]).sample(DRAWS, 1)
    rows = [[annual.quantile(a), other_book.quantile(b)] for a, b in u]
    sims = PredictiveDistribution(["part"], [("contract",), ("book",)], rows)
    return price_portfolio(sims, TAIL, cost_of_capital=RETURN)


in_book, assets_rows, price_rows = {}, [], []
for correlation in (0.0, 0.5):
    p = in_a_book(correlation)
    alone, joined, book_alone = p.standalone[0], p.allocated[0], p.standalone[1]
    in_book[correlation] = joined.premium
    assets_rows.append({"correlation": f"{correlation:.1f}", "book alone": book_alone.assets,
                        "with the contract": p.total.assets, "added": p.total.assets - book_alone.assets})
    for label, own in (("contract alone", alone), ("in the book", joined)):
        price_rows.append({"case": f"{correlation:.1f}, {label}", "risk premium": own.premium,
                           "capital": own.capital, "loading": own.premium - own.expected_loss})
(GT(pd.DataFrame(assets_rows), rowname_col="correlation")
 .tab_header("The book's assets (marginal view)", "Millions, at the 99% TVaR standard")
 .fmt_number(["book alone", "with the contract", "added"], decimals=2))
```

```text
     book alone  with the contract  added
0.0       56.43              57.45   1.02
0.5       56.48              60.33   3.84
```

```python
(GT(pd.DataFrame(price_rows), rowname_col="case")
 .tab_header("The contract's price alone and in the book (natural allocation)",
             "Correlation, then view; millions; loading is risk premium less expected loss")
 .fmt_number(["risk premium", "loading"], decimals=3)
 .fmt_number("capital", decimals=2))
```

```text
                     risk premium  capital  loading
0.0, contract alone         1.230     6.43    0.643
0.0, in the book            0.668     0.81    0.081
0.5, contract alone         1.230     6.43    0.643
0.5, in the book            0.947     3.60    0.360
```

The draws use a fixed seed, so the stand-alone risk premium differs slightly
from the grid's 1.240. Diversification is a discount on capital: in an
unrelated book, the risk premium is about half its stand-alone value and the
**loading** falls by much more. At a correlation of 0.5, much of the discount
goes away.

The marginal assets and the allocated capital are near but not equal. The
natural allocation divides the total so that the parts add up; marginal figures
for several contracts do not. A reinsurer that already holds much of the same
type of loss quotes higher than one that does not, and both prices are correct
for their portfolios. Two cautions apply:

- The discount depends on the size and shape of the book. Here a book with a
  mean of 30 is much larger than a contract that pays nothing in four years out
  of five.
- A Gaussian copula has no tail dependence. Even at a correlation of 0.5, it
  understates how often both parts have their worst years together.

## 7. Hard and soft markets

In a **hard market**, reinsurers charge more for the same cover and offer less
of it: less limit, higher attachments, fewer free reinstatements, stricter
wording. A **soft market** is the opposite. A price rise for the same cover has
three parts: a higher **expected loss** (loss trend, adverse development, a new
view of the tail), a higher **price of risk** (the return the market demands on
its capital, here the parameter of the pricing distortion) and **tighter
terms**.

A calibration to a quote separates the first two parts only if you trust the
loss model. Take two illustrative quotes for the 5x5 layer with a free
reinstatement, as rates on line: 22% in a soft market and 34% in a hard market.
Assume the expected loss has not changed, so the difference is all price of
risk. The risk premium is the quote less expenses and brokerage. `Quote.at`
restates the technical quote of section 2 at a quoted premium, keeping the
expected loss and assets. The code calibrates the proportional hazard parameter
for each quote and prices the other layers at that parameter. The technical
premium is the reference row.

```python
QUOTES = {"soft": 0.22, "hard": 0.34}   # rate on line for the 5x5 layer


def layer_multiples(distortion):
    return [gross_up(p.premium) / p.loss for p in tower_grids.price(distortion).ceded]


market, offers = {}, {}
for name, rol in QUOTES.items():
    offers[name] = technical_quote.at(rol * mid.limit)   # same loss and assets, at the quoted premium
    market[name] = calibrate("ph", annual, offers[name].risk_premium)

market_rows = []
for label, offered, distortion in [("soft quote", offers["soft"], market["soft"]),
                                   ("technical premium", technical_quote, price_of_risk),
                                   ("hard quote", offers["hard"], market["hard"])]:
    m2x3, m5x5, m10x10 = layer_multiples(distortion)
    market_rows.append({"case": label, "premium": offered.premium, "ROL": offered.rate_on_line,
                        "return on capital": offered.return_on_capital,
                        "hazard parameter": distortion.parameters["rho"],
                        "2x3": m2x3, "5x5": m5x5, "10x10": m10x10})
(GT(pd.DataFrame(market_rows), rowname_col="case")
 .tab_header("The 5x5 layer in three markets", "Premium in millions; the layer columns are multiples of expected loss")
 .tab_spanner("multiple by layer", ["2x3", "5x5", "10x10"])
 .fmt_number("premium", decimals=2)
 .fmt_percent(["ROL", "return on capital"], decimals=1)
 .fmt_number("hazard parameter", decimals=3)
 .fmt_number(["2x3", "5x5", "10x10"], decimals=2))
```

```text
                   premium    ROL  return on capital  hazard parameter   2x3   5x5  10x10
soft quote            1.10  22.0%               5.4%             0.791  1.57  1.88   2.37
technical premium     1.43  28.5%              10.0%             0.686  1.85  2.43   3.44
hard quote            1.70  34.0%              14.2%             0.616  2.07  2.90   4.42
```

The soft quote gives about half the 10% target return. A lower hazard parameter
means a higher price of risk. The figure plots the multiples by layer:

```python
fig, ax = plt.subplots()
layer_names = [layer.name for layer in LAYERS]
for label, distortion in [("soft quote", market["soft"]), ("technical premium", price_of_risk),
                          ("hard quote", market["hard"])]:
    ax.plot(layer_names, [gross_up(p.premium) / p.loss for p in tower_grids.price(distortion).ceded],
            marker="o", label=label)
ax.set_xlabel("layer (limit x attachment)")
ax.set_ylabel("price as a multiple of the expected loss")
ax.set_title("Price multiple by layer in three markets")
ax.legend(loc="upper left")
```

The hard market costs more on each layer, and most on the top layer. This
pattern comes from the single parameter of the proportional hazard family; a
cost-of-capital rule or another distortion spreads the increase differently. In
practice, a hard casualty market often hits working layers most, and a hard
catastrophe market the upper layers.

### Tighter terms

The cedant can change the terms to get a lower price. The code prices the 5x5
layer under the hard market distortion with a higher attachment, and with no
reinstatement:

```python
variants = [
    Layer("base", limit=5.0, attachment=5.0, reinstatements=1),
    Layer("attach 7", limit=5.0, attachment=7.0, reinstatements=1),
    Layer("no reinstatement", limit=5.0, attachment=5.0, aggregate_limit=5.0),
]
variant_grids = Tower(variants).on_grid(frequency, severity, POINTS)
terms_rows = []
for layer, p in zip(variants, variant_grids.price(market["hard"]).ceded):
    quote = gross_up(p.premium)
    terms_rows.append({"terms": layer.name, "exp loss": p.loss, "quote": quote,
                       "multiple": quote / p.loss, "ROL": quote / layer.limit})
(GT(pd.DataFrame(terms_rows), rowname_col="terms")
 .tab_header("The 5x5 layer under the hard market distortion", "Millions")
 .fmt_number(["exp loss", "quote"], decimals=3)
 .fmt_number("multiple", decimals=2)
 .fmt_percent("ROL", decimals=1))
```

```text
                  exp loss  quote  multiple    ROL
base                 0.586  1.700      2.90  34.0%
attach 7             0.369  1.257      3.41  25.1%
no reinstatement     0.558  1.469      2.63  29.4%
```

A higher attachment cuts the expected loss by more than the quote, so each unit
of expected loss costs more: the remaining loss is more remote. With no
reinstatement, the expected loss falls by only 5% but the quote by 14%, because
the most the layer can pay in a year is half, and so is the capital. The quote
follows the capital.

These factors change the price of risk:

- **capital supply**: after large losses, reinsurers have less capital, and
  investors want a higher return to supply it.
- **capital requirements**: rating agencies and regulators set how much capital
  a reinsurer must hold for a risk.
- **interest rates**: higher rates raise the return investors require on
  capital, which raises prices. They also raise what the reinsurer earns on
  premium and reserves until it pays claims, which permits a lower underwriting
  margin. The larger effect depends on the payout and the return asked over the
  rate. The one-year model here has no investment income; section 9 restates
  the price over a payout.
- **retrocession cost**: when the reinsurers' own reinsurance costs more, they
  raise their prices.

Loss trend, the cedant's own experience, reserve development and new views of
the tail change the expected loss, and so the technical price in all markets.

## 8. What the reinsurer decides

The decision needs two technical prices: the **stand-alone price** (section 2)
and the **price in a book** (section 6), the cost of the contract in the
reinsurer's portfolio, grossed up for expenses and brokerage. Each quote row
shows its distance from each:

```python
floors = {
    "alone": technical,
    "book, unrelated": gross_up(in_book[0.0]),
    "book, corr 0.5": gross_up(in_book[0.5]),
}
floor_rows = [{"row": "technical", **floors}]
for name, rol in QUOTES.items():
    quote = rol * mid.limit
    floor_rows.append({"row": f"{name} {quote:.2f}", **{k: quote / v - 1 for k, v in floors.items()}})
(GT(pd.DataFrame(floor_rows), rowname_col="row")
 .tab_header("Technical premium and market quotes", "Technical premium in millions; quotes as the distance from it")
 .fmt_number(rows=[0], decimals=3)
 .fmt_percent(rows=[1, 2], decimals=0, force_sign=True))
```

```text
           alone  book, unrelated  book, corr 0.5
technical  1.425            0.768           1.089
soft 1.10   -23%             +43%             +1%
hard 1.70   +19%            +121%            +56%
```

For each submission, a reinsurer selects one action:

- **write** the share offered, if the quote is at or above the technical price
  for its own portfolio.
- **write less** at the same terms, but only if capital is not linear in the
  share. Examples are a concentration or accumulation limit, or a share large
  enough to change the portfolio's tail. Otherwise a smaller share of an
  underpriced contract is only a smaller underpriced position.
- **change the structure** (a higher attachment, a lower limit, no free
  reinstatement) if different terms meet the price.
- **decline** in all other cases.

## 9. What the model leaves out

Each item below changes a real price:

- **Development**: reported excess losses grow slowly to ultimate. Ignoring
  development makes the expected loss too low, more so on higher layers.
  `LargeLossExperience.prepare(development=...)` develops each known large loss
  to ultimate, and a count development pattern (`DevelopmentPattern`) given to
  `burning_cost(development=...)` adds the late-reported claims. See the
  [experience and exposure rating guide](experience-and-exposure-rating.md).
- **Loss adjustment expenses**: claim expenses added to the loss raise what the
  layer pays.
- **Payout, discounting and investment income**: the guide prices one year of
  capital without discounting. For a long-tail line such as liability, the
  reinsurer holds capital and reserves for the whole payout. This raises the
  cost of capital, while the investment income on them lowers the price. The
  example below is not a statement about liability multiples: market prices
  include more (interest rates and spreads, capital released as claims settle).
- **Parameter uncertainty**: alpha and the frequency are estimates. Including
  their uncertainty makes the tail heavier and the price higher.
- **Data and credibility loads**: a small or unreliable submission gets a load.
- **Clash and catastrophe aggregation**: when one event hits several of the
  cedant's risks, the occurrence definition adds their losses. Ignoring it
  makes the tail too light. For catastrophe covers, the library takes the event
  loss table of a cat model (`EventLossTable`), and `EventSet.grouped` groups
  losses into occurrences by an hours clause.
- **Credit risk**: funds withheld, collateral and the reinsurer's own credit
  standing change what each side accepts.
- **Expenses and tax**: fixed costs per contract and tax raise the price of
  small contracts above a percentage loading. `Quote.technical` takes a `fixed`
  cost per contract with the percentages.

### Pricing over the payout

`price_over_payout` restates a one-year price over a **payment pattern** (the
share of the ultimate loss paid in each year). It discounts the losses at the
given rate, which is how investment income enters, and holds the capital while
losses are outstanding, so the cost of capital is a return over the discount
rate. Give it the rule of the one-year price: here the 99% TVaR price at a 10%
cost of capital. Assume the 5x5 layer pays 40% of its losses in the first year,
then 30%, 20% and 10%, and that money earns 3%. The table also shows the same
payout with no discounting, and with 10% as the total return (7% over 3%).

```python
payout = PaymentPattern([1.0, 2.0, 3.0, 4.0], [0.4, 0.3, 0.2, 0.1])   # years from inception, shares
over_payout = price_over_payout(tvar_price, payout, rate=0.03, cost_of_capital=RETURN)
undiscounted = price_over_payout(tvar_price, payout, rate=0.0, cost_of_capital=RETURN)
total_10 = price_over_payout(tvar_price, payout, rate=0.03, cost_of_capital=RETURN - 0.03)

payout_rows = [{"basis": "one year", "discount factor": 1.0, "capital years": 1.0,
                "discounted loss": tvar_price.expected_loss, "margin": tvar_price.margin,
                "capital": tvar_price.capital, "risk premium": tvar_price.premium}]
for label, p in [("payout, zero rate", undiscounted), ("payout, 3%, 10% over the rate", over_payout),
                 ("payout, 3%, 7% over the rate", total_10)]:
    payout_rows.append({"basis": label, "discount factor": p.discount_factor,
                        "capital years": p.capital_years, "discounted loss": p.discounted_loss,
                        "margin": p.margin, "capital": p.capital, "risk premium": p.premium})
(GT(pd.DataFrame(payout_rows), rowname_col="basis")
 .tab_header("The 5x5 risk premium over the payout", "Millions; payment shares 40%, 30%, 20% and 10% in years 1 to 4")
 .fmt_number(["discount factor", "capital years", "discounted loss", "margin", "capital", "risk premium"],
             decimals=4))
```

```text
                               discount factor  capital years  discounted loss  margin  capital  risk premium
one year                                1.0000         1.0000           0.5863  0.6538   6.5375        1.2401
payout, zero rate                       1.0000         2.0000           0.5863  1.1985   5.9927        1.7849
payout, 3%, 10% over the rate           0.9430         1.9568           0.5529  1.1824   6.0423        1.7353
payout, 3%, 7% over the rate            0.9430         1.9568           0.5529  0.8704   6.3543        1.4233
```

Discounting lowers the expected loss: that is the investment income. But the
capital runs off with the outstanding losses, which equals 1.9568 discounted
years of it, and earns 10% a year over those years, so the margin rises.

Three things changed together: the reinsurer holds capital for longer, the
losses are discounted, and 10% over a 3% rate gives investors 13% in total,
against 10% at the zero rate of section 2. Taken one at a time: at a zero rate
the price over the payout is already 1.7849, so discounting alone lowers it
only to 1.7353. If 10% is the total return, the return over the rate is 7% and
the price is 1.4233, still above the one-year 1.2401.

With these assumptions, the cost of holding capital for longer is larger than
the investment income. For a real layer, the payout pattern, the discount rate
and the return over it decide which effect is larger.
`PaymentPattern.from_development` gives a payout pattern, for example from a
paid chain ladder.

## 10. Glossary and further reading

**Annual aggregate deductible**: the part of the year's layer losses that the
cedant keeps before the layer pays. **Annual aggregate limit** (**annual
limit**): the most a layer pays in a year. **Assets**: premium plus capital.
**Attachment**: the loss size above which an excess of loss layer pays.
**Attritional losses**: the frequent small losses below the large-loss
threshold. **Capital**: the reinsurer's own money that it holds in addition to
the premium, so it can pay in a bad year. **Capital years**: how long the
capital is held over a payout, in discounted years, with the capital running
off as losses are paid. **Cedant**: the insurer that buys reinsurance.
**Ceding commission**: what the reinsurer pays the cedant on a quota share.
**Cost of capital**: the return the reinsurer asks on its capital, here 10% a
year (over a payout, a return over the discount rate). **Deposit premium**: the
premium paid at inception, before any reinstatement premium. **Discount
factor**: the present value of one unit of ultimate loss paid on the payment
pattern. **Distortion**: a reweighting of probabilities that prices a loss by
its tail. **Exhaustion**: a loss that uses a layer's whole limit (for 5x5, any
loss of 10 or more). **Hard market**: prices up and less cover for the same
risk. **Layer**: the slice of each loss that one excess of loss contract pays,
written limit x attachment. **5x5** pays the part of a loss between 5 and 10,
and **2x3** pays the part between 3 and 5. **Limit**: the most the layer pays
for one loss. **Loss on line**: expected loss divided by limit, the pure cost of
each unit of cover. **Multiple**: premium divided by expected loss. **Natural
allocation**: a split of a portfolio's capital and price among its parts that
adds up exactly. **Payback**: limit divided by premium, the inverse of the rate
on line. **Payment pattern**: the share of the ultimate loss paid in each year.
**Pentagon**: loss, premium, margin, capital and assets, linked by two
identities. **Price of risk**: the loading the market puts on risk, as a return
on capital or a distortion parameter. **Profit commission**: a share of the
reinsurer's profit returned to the cedant. **Program**: all the layers a cedant
buys together. **Quota share**: reinsurance of a fixed share of each policy.
**Rate on line (ROL)**: premium divided by limit, the price of each unit of
cover. **Reinstatement**: a layer's limit restored after a loss uses it, so the
layer can pay again in the same year, free or for a **reinstatement premium**.
**Retrocession**: reinsurance bought by a reinsurer. **Risk premium**: expected
loss plus margin, before expenses and brokerage. **Sliding scale**: a ceding
commission that decreases as the loss ratio increases. **Soft market**: prices
down and much cover available. **Stand-alone price**: a contract's price with
no diversification benefit. **Subject premium**: the cedant's premium that the
reinsurance applies to. **Technical commission**: the ceding commission that
the reinsurer's model supports. **Technical premium**: the price that the
reinsurer's model supports, after expenses and brokerage. **Treaty**: a
reinsurance contract that covers a whole book, not one policy. **TVaR**: the
average of the worst outcomes beyond a probability level.

Further reading, by question:

- Clark, "Basics of Reinsurance Pricing", CAS Study Note: experience and
  exposure rating of layers and increased limits. Start here.
- Werner and Modlin, *Basic Ratemaking* (CAS): the trend and development work
  that experience rating of a layer uses.
- Mildenhall and Major, *Pricing Insurance Risk: Theory and Practice* (Wiley,
  2022): the pentagon, distortions and the natural allocation of sections 1, 3
  and 6.
- The notebook [`examples/reinsurance_pricing.ipynb`](../../examples/reinsurance_pricing.ipynb)
  prices six contracts and checks them against Mildenhall's `aggregate`.

### About these numbers

CI runs each Python block in this guide (`python/tests/test_guides.py`) and
compares each printed result, and each table's formatted cells, with the text
under the block.
