# What a reinsurance price is

This guide is for an actuary who prices primary insurance and has a loss ratio
for a reinsurance submission. It answers four questions in brief, then works
through one example.

## Answers in brief

- **What does reinsurance pricing produce?** Three things. The **terms** are
  the layers or the share, attachments, limits, reinstatements and commissions.
  The **price** is the premium for those terms. The **expected economics** are
  the expected loss, the margin, the capital the reinsurer holds and the return
  on that capital.
- **What should we charge?** For an excess of loss layer, the technical premium
  is (expected loss + return asked × capital held) / (1 - expenses -
  brokerage). For a quota share, the price is the ceding commission: 1 -
  expected loss ratio - expenses - brokerage - margin. Both are floors for a
  reinsurer with no other business. A reinsurer whose book diversifies the
  contract can accept less.
- **How do we price without a rating plan?** The price follows a loss model of
  the cedant's book: how many large losses, how big, and how heavy the tail.
  The cedant's own large-loss history gives one estimate (experience rating).
  Exposure curves on its profile of limits and sums insured give another
  (exposure rating).
- **What does a hard market mean?** Reinsurers charge more for the same cover
  and offer less of it. A price rise has three parts. The first is a higher
  expected loss (trend, development, a new view of the tail). The second is a
  higher **price of risk** (the return the market demands on capital). The
  third is tighter terms. A loss model
  separates the first part from the other two.

For the 5x5 layer of the running example (all money in millions):

| Output | Value | Where |
|---|---|---|
| Expected loss | 0.586 | section 2 |
| Capital held (99% TVaR standard) | 6.54 | section 2 |
| Risk premium: expected loss plus a 10% return on capital | 1.240 | section 2 |
| Technical premium: the risk premium after 3% expenses and 10% brokerage | 1.425 | section 3 |
| Rate on line, multiple, loss ratio of the technical premium | 28.5%, 2.43, 41.1% | section 3 |
| With a paid reinstatement: deposit plus expected reinstatement premium | 1.282 + 0.143 | section 3 |
| Technical premium inside a book: unrelated, and at correlation 0.5 | 0.768, 1.089 | section 9 |
| A soft and a hard market quote | 1.10, 1.70 | section 8 |

## 1. Words you need first

- **Cedant**: the insurer that buys reinsurance. **Treaty**: a reinsurance
  contract that covers a whole book, not one policy. **Program**: all the
  layers a cedant buys together.
- **Layer**: the slice of each loss that one excess of loss contract pays. We
  write it as limit x attachment. **5x5** pays the part of a loss between 5 and
  10, and **2x3** pays the part between 3 and 5.
- **Attachment**: the loss size where the layer starts to pay. **Limit**: the
  most the layer pays for one loss. **Exhaustion**: a loss that uses the whole
  limit (for 5x5, any loss of 10 or more).
- **Reinstatement**: the limit restored after a loss uses it, so the layer can
  pay again in the same year. The **annual limit** is the most the layer pays
  in a year.
- **Capital**: the reinsurer's own money that it holds in addition to the
  premium, so it can pay in a bad year. **Assets**: premium plus capital.
- **Cost of capital**: the return the reinsurer wants on that capital, here 10%
  a year.
- **Loss on line**: expected loss divided by limit, the pure cost of each unit
  of cover. **Rate on line**: premium divided by limit, the price of each unit
  of cover.
- **Retrocession**: reinsurance that a reinsurer buys for itself.

Section 11 has the full glossary.

## 2. What is priced

A primary insurer prices a policy: one risk, one year, and a rating plan that
turns rating variables into a premium. A reinsurer prices a **contract** on a
**portfolio** for a **period**. The cedant gives a book, not a policy, and the
price comes from a model of the cedant's losses.

The running example is a synthetic commercial property book with excess of
loss cover per risk. Money is in millions. The cedant writes 60 of subject
premium (the premium the reinsurance applies to). Losses below 1 are
**attritional**: frequent, fairly stable, with an expected total of 28. Losses
above 1 are the **large losses**, which excess of loss reinsurance covers.

Their count is Poisson with a mean of 4 a year. Their size has a Pareto tail
with alpha 1.8. With an alpha below 2, the mean loss is finite but the variance
is infinite without a cap. Here the cap is the policy limit (the largest sum
insured) of 20. The numbers show the mechanics and do not match a market.

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

In the code, `frequency` is the number of large losses in a year and
`severity` is the size of each. The library works on a **grid**: it rounds each
size to a step of 0.1. `fft` combines frequency and severity into the
distribution of the annual total by fast Fourier transform, with no simulation
error. Later, `Tower(...).on_grid` applies layers to each loss and gives the
distribution of what each layer pays in a year.

A primary actuary brings the 60.9% loss ratio to a submission. The reinsurer
needs the whole distribution, because a contract that pays only above a high
attachment depends on the tail, not on the mean.

### The pricing pentagon

Five quantities describe a priced contract:

- **loss** (L): the expected loss the reinsurer pays.
- **premium** (P).
- **margin** (M = P - L): what the reinsurer expects to keep for the risk it
  carries.
- **capital** (Q) and **assets** (a = P + Q).

Two identities link them, M = P - L and a = P + Q. Thus any three independent
quantities set the other two. The **return on capital**, M / Q, comes from them
and can replace one of the three. Below, `Pentagon.solve` gets the loss, the
assets and the return.

Take the 5x5 layer with one free reinstatement. The limit is restored once at
no charge, so the layer pays at most 10 in a year. Set the assets at the 99%
**tail value at risk** (TVaR) of the annual loss, which is the average of the
worst 1% of years. Ask for a 10% return on capital. The premium that results is
the **risk premium**: expected loss plus margin, before expenses and brokerage.

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

The layer is expected to pay about 0.59 a year, but the reinsurer must hold 7.8
of assets for the bad years. The risk premium supplies 1.2 of that. The
reinsurer supplies the other 6.5 as capital and wants 10% on it, which is the
margin of 0.65. Each method in this guide gets these five numbers for a
contract.

## 3. An excess of loss layer

An **excess of loss** (XoL) layer pays the part of each loss between its
attachment and attachment plus limit. Start with its shape: how often a loss
reaches the layer, how often a loss exhausts it, and the cost per unit of
cover.

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

About one large loss in eighteen reaches the layer, and about one in sixty
exhausts it. Only one year in five has a loss to the layer, so in most years
the layer pays nothing. The loss on line above uses the occurrence limit with
no annual limit. The annual limit of 10 decreases the expected loss a little,
because a year with three exhausting losses pays 10, not 15.

### The technical premium

The **technical premium** is the price that the reinsurer's own model
supports. It has four parts:

1. the **expected loss**.
2. the **cost of capital**: the return asked times the capital the layer needs
   (the margin from the pentagon).
3. **internal expenses**: here 3% of the premium.
4. **brokerage**: here 10% of the premium, paid to the intermediary.

Expenses and brokerage are fractions of the premium, so we divide by them:
premium = (expected loss + cost of capital) / (1 - 3% - 10%). Three
simplifications apply:

- Real internal expenses are partly a fixed cost per contract, so a flat
  percentage is too low for small contracts.
- There is no tax line, so the 10% is a return after tax.
- Brokerage applies to reinstatement premium and to the deposit.

```python
EXPENSES, BROKERAGE = 0.03, 0.10
KEEP = 1 - EXPENSES - BROKERAGE    # the share of each unit of premium the reinsurer keeps


def gross_up(risk_premium):
    """The price the cedant pays for a risk premium: expenses and brokerage on top."""
    return risk_premium / KEEP


capital = pentagon.capital
cost_of_capital = RETURN * capital
risk_premium = expected_loss + cost_of_capital
technical = gross_up(risk_premium)

print(f"expected loss     {expected_loss:7.3f}")
print(f"cost of capital   {cost_of_capital:7.3f}   ({RETURN:.0%} of capital {capital:.3f})")
print(f"risk premium      {risk_premium:7.3f}")
print(f"technical premium {technical:7.3f}   (expenses and brokerage take {EXPENSES + BROKERAGE:.0%})")
print(f"rate on line {technical / mid.limit:.1%} (payback {mid.limit / technical:.1f} years), "
      f"multiple {technical / expected_loss:.2f}, loss ratio {expected_loss / technical:.1%}")

# The same build-up from the library: the 99% TVaR price, then the quote with its loads.
tvar_price = price(annual, TAIL, cost_of_capital=RETURN)
technical_quote = Quote.technical(tvar_price, expenses=EXPENSES, brokerage=BROKERAGE, limit=mid.limit)
print(f"Quote.technical {technical_quote.premium:.3f}, rate on line {technical_quote.rate_on_line:.1%}, "
      f"payback {technical_quote.payback:.1f}, multiple {technical_quote.multiple:.2f}, "
      f"loss ratio {technical_quote.loss_ratio:.1%}")
```

```text
expected loss       0.586
cost of capital     0.654   (10% of capital 6.538)
risk premium        1.240
technical premium   1.425   (expenses and brokerage take 13%)
rate on line 28.5% (payback 3.5 years), multiple 2.43, loss ratio 41.1%
Quote.technical 1.425, rate on line 28.5%, payback 3.5, multiple 2.43, loss ratio 41.1%
```

Brokers quote three ratios for the price of a layer:

- the **rate on line** (ROL): premium divided by limit. Its inverse is the
  **payback**, the number of years without loss that the premium needs to pay
  one full limit.
- the **multiple**: premium divided by expected loss, the number of times the
  pure cost that the buyer pays.
- the **loss ratio**: expected loss divided by the premium for this layer, the
  inverse of the multiple. It is not the cedant's book loss ratio of 60.9%.

The last line does the same calculation in the library. `price(annual, TAIL,
cost_of_capital=RETURN)` holds the 99% TVaR of the annual loss as assets and
charges a 10% return on the capital. The result is the risk premium of section
2. `Quote.technical` grosses up that risk premium for expenses and brokerage,
as above, and gives the market ratios. Later sections use `gross_up`, the same
step by hand, where one division is clearer than a quote.

The call states the capital standard. A distortion with a constant cost of
capital (`Distortion.ccoc`) prices a loss as (expected loss + r × assets) / (1
+ r). `TowerGrids.price` applies it to each layer of a tower, with the most the
layer can pay in a year as its assets. For this layer that is the annual limit
of 10, a different capital standard from the 99% TVaR above. `price` takes the
99% TVaR standard directly. The full-limit formula agrees with
`TowerGrids.price`, and `price` agrees with the pentagon of section 2:

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

The capital standard is a pricing decision. A full-limit standard charges each
layer as if the reinsurer will lose its limit. A tail standard charges for the
loss that the reinsurer expects in bad years. The rest of this guide uses the
99% TVaR standard.

### Reinstatements

Property and catastrophe covers usually have paid reinstatements. Casualty
working layers often have free reinstatements or only an annual limit. Two
more annual terms change what a layer pays in a year. With an **annual
aggregate deductible**, the cedant keeps the first part of the year's layer
losses. The **annual aggregate limit** is the most the layer pays in a year.

With a paid reinstatement, the reinsurer charges a **deposit premium** at
inception. Each time a loss uses limit that is restored, it charges a
**reinstatement premium**: here 100% of the deposit, pro rata to the amount
reinstated. First, set the deposit so that the deposit plus the expected
reinstatement premium equals the technical premium:

```python
RATE = 1.0   # reinstatement premium as a fraction of the deposit, pro rata to the amount reinstated
used_limit = annual.layer(5.0, 0.0) / 5.0    # expected share of the first limit used up in a year
deposit = technical / (1 + RATE * used_limit)

paid = Layer("5x5", limit=5.0, attachment=5.0, premium=deposit, reinstatement_rates=[RATE])
paid_grids = Tower([paid]).on_grid(frequency, severity, POINTS)
expected_reinstatement = paid_grids.expected_reinstatement_premium[0]
print(f"deposit {deposit:.3f} (rate on line {deposit / mid.limit:.1%}), "
      f"expected reinstatement premium {expected_reinstatement:.3f}")
print(f"expected total {paid_grids.expected_premium[0] + expected_reinstatement:.3f} "
      f"(technical premium {technical:.3f})")
print(f"expected loss unchanged: {paid_grids.ceded[0].mean():.3f}")
```

```text
deposit 1.282 (rate on line 25.6%), expected reinstatement premium 0.143
expected total 1.425 (technical premium 1.425)
expected loss unchanged: 0.586
```

The buyer sees a lower rate on line at inception and pays the remainder when
losses occur. The market quotes rates on line for such a layer on the deposit.
Thus the comparable figure for this layer is the deposit rate on line, not
28.5%.

The reinsurer collects reinstatement premium in the same years that it pays
large losses. Its net loss in a bad year is smaller, so it needs less capital.
Hold capital against the loss net of the reinstatement premium that the
reinsurer keeps after expenses and brokerage. Then solve for the deposit again:

```python
net_deposit = deposit
for _ in range(20):   # the reinstatement premium depends on the deposit, so iterate
    net_loss, _ = annual.map(lambda x: x - KEEP * RATE * net_deposit * min(x, 5.0) / 5.0)
    net = price(net_loss, TAIL, cost_of_capital=RETURN)
    net_deposit = net.premium / KEEP
net_total = net_deposit * (1 + RATE * used_limit)
print(f"capital against the net loss {net.capital:.3f} (against the gross loss {capital:.3f})")
print(f"deposit {net_deposit:.3f}, expected total {net_total:.3f} "
      f"({net_total / technical - 1:+.1%} on the technical premium)")
```

```text
capital against the net loss 5.698 (against the gross loss 6.538)
deposit 1.196, expected total 1.329 (-6.8% on the technical premium)
```

A paid reinstatement moves premium from good years to bad years. It is also a
real discount, because it decreases the capital that the reinsurer needs. The
rest of the guide uses the free reinstatement, so each quote is the whole
premium for the layer.

## 4. Where the layer sits

Price three layers of the same program:

- 2x3, a working layer that attaches at 3 and that losses reach often.
- 5x5.
- 10x10, a top layer that attaches at 10, that losses reach rarely, and that
  pays a large amount when they do.

Together they cover losses from 3 to 20, the policy limit. Use **one** pricing
rule for all three, so that you can compare them.

Here the rule is a **distortion**. A distortion prices a loss as a weighted
average that gives bad outcomes more weight than their probability. The
**proportional hazard** distortion raises the probability of exceeding each
loss size to a power below 1. A parameter of 1 gives no loading, and a lower
parameter gives a larger loading. `calibrate` finds the parameter that gives a
specified premium on one layer, here the 5x5 risk premium from above. That one
parameter then prices the other two layers.

```python
LAYERS = [Layer("2x3", limit=2.0, attachment=3.0, reinstatements=1), mid,
          Layer("10x10", limit=10.0, attachment=10.0, reinstatements=1)]
tower_grids = Tower(LAYERS).on_grid(frequency, severity, POINTS)
price_of_risk = calibrate("ph", tower_grids.ceded[1], pentagon.premium)
print(f"proportional hazard parameter {price_of_risk.parameters['rho']:.3f}: a 1% chance is weighted as "
      f"{price_of_risk.g(0.01):.1%}, a 10% chance as {price_of_risk.g(0.10):.1%}")


def layer_table(layers, grids, distortion):
    print(f"{'layer':>6} {'exp loss':>9} {'assets/EL':>10} {'price':>6} {'multiple':>9} {'ROL':>6} "
          f"{'return':>7} {'at 10%':>7}")
    for layer, grid in zip(layers, grids.ceded):
        own = price(grid, TAIL, distortion=distortion)   # premium by the distortion, assets at 99% TVaR
        quoted = Quote.technical(own, expenses=EXPENSES, brokerage=BROKERAGE, limit=layer.limit)
        at_target = Quote.technical(price(grid, TAIL, cost_of_capital=RETURN),   # exactly 10% on that capital
                                    expenses=EXPENSES, brokerage=BROKERAGE)
        print(f"{layer.name:>6} {own.expected_loss:9.3f} {own.assets / own.expected_loss:10.1f} "
              f"{quoted.premium:6.3f} {quoted.multiple:9.2f} {quoted.rate_on_line:6.1%} "
              f"{own.return_on_capital:7.1%} {at_target.multiple:7.2f}")


layer_table(LAYERS, tower_grids, price_of_risk)
```

```text
proportional hazard parameter 0.686: a 1% chance is weighted as 4.3%, a 10% chance as 20.6%
 layer  exp loss  assets/EL  price  multiple    ROL  return  at 10%
   2x3     0.686        5.8  1.271      1.85  63.6%   14.5%    1.65
   5x5     0.586       13.3  1.425      2.43  28.5%   10.0%    2.43
 10x10     0.337       31.1  1.160      3.44  11.6%    7.1%    4.30
```

The columns of the table:

- **assets/EL** is the 99% TVaR of the layer's annual loss per unit of expected
  loss: the money behind each unit of pure cost.
- **price** is the technical premium, the distortion's risk premium grossed up
  for expenses and brokerage. The multiple and the rate on line use this price,
  so the 5x5 row agrees with section 3.
- **return** is the margin over the 99% TVaR capital at that price.
- **at 10%** is the multiple from a rule that charges exactly 10% on the
  capital of each layer.

In the code, `price` with a `distortion` prices the layer by the distortion and
holds its 99% TVaR as assets. Its `return_on_capital` is the return column.
`price` with a `cost_of_capital` is the 10% rule.

The multiple increases from the working layer to the top layer, with the same
pricing rule. The cause is the capital behind each unit of expected loss. The
working layer needs about 6 of assets per unit, because it pays often and its
result is predictable. The top layer needs about 31, because its expected loss
is a small probability of a very large payment.

The proportional hazard rule does not give a constant return on capital. Its
return decreases from 14.5% on the working layer to 7.1% on the top layer. Take
a rule that charges exactly 10% on the 99% TVaR capital of each layer. It makes
the working layer cheaper (a multiple of 1.65) and the top layer more expensive
(4.30). A distortion with one parameter charges the remote tail less than a
constant-return rule. This is a modelling choice. To make it, calibrate to
market quotes on more than one layer.

The rate on line moves in the opposite direction, from 64% to 12%, because the
top layer pays much less often. Rate on line gives the cost per unit of limit.
The multiple gives the loading for risk. Thus a loss ratio alone cannot price a
layer: two layers with the same expected loss have different prices when one
of them is a 1-in-100 event.

## 5. Quota share

A **quota share** takes a fixed share of the premium and losses of each policy.
The reinsurer pays the cedant a **ceding commission** for its acquisition
costs. The commission, not a rate, sets the reinsurer's economics, so the price
of a quota share is the commission.

The **technical commission** is the premium that remains after all the other
needs of the reinsurer:

technical commission = 1 - expected loss ratio - reinsurer expenses - brokerage - margin

A quota share carries all of the cedant's loss-ratio risk, not only the large
losses. Thus the model cannot hold the attritional losses at their mean. Here
they get a lognormal spread with a coefficient of variation of 10%, for
pricing, inflation and reserve uncertainty. Take a 30% quota share of the whole
book. The margin comes from the same pentagon: assets at the 99% TVaR of the
ceded loss, and a 10% return on the capital.

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

A commission above that value costs the reinsurer margin. A commission below it
gives margin to the cedant. The cedant can compare this number with its own
acquisition cost to decide if the reinsurance is a good deal. For a long-tail
line, the ceding commission also depends strongly on the investment income that
the reinsurer earns before it pays claims. This one-year example does not
include that income (section 10 restates a price over a payout).

### Sliding scales and profit commissions

With these two variants, the commission depends on the result of the year. Part
of the loss-ratio volatility (not the mean) goes back to the cedant.

- A **sliding scale** pays a higher commission when the loss ratio is low and a
  lower commission when it is high. The commission is linear between two
  points.
  Below, the commission is 35.1% at a loss ratio of 55% or less, and decreases
  to 24.1% at 75% or more.
- A **profit commission** is a flat commission plus a share of the reinsurer's
  profit, if there is a profit. The standard form is a share of premium less
  commission, losses and an allowance for the reinsurer's management expenses.
  It usually carries a deficit forward to later years. In this example the
  allowance also includes the reinsurer's margin, and there is no
  carry-forward. Both choices change the expected profit commission, so model
  the actual wording.

First, set the scale and the profit share so that the **expected** commission
over the loss-ratio distribution equals the technical commission.
`Tower.on_annual_grid` applies the contract to the annual total on the grid and
gives the exact expected commissions. The reinsurer's result in each outcome
comes from the contract's commission for that outcome
(`ceding_commission_for`, `profit_commission_for`). This result gives the
spread and the worst years:

```python
costs = (QS_EXPENSES + QS_BROKERAGE) * ceded_premium


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
    print(f"{name:<14} mean {mean:6.3f}  sd {sd:5.3f}  worst 1% {worst:6.2f}  "
          f"commission it can add {(mean - needed) / ceded_premium:.1%}")
    return sd


flat = qs.with_ceding_commission(technical_commission)
sd_flat = describe("flat", underwriting_result(flat))

shape = [(0.37, 0.55), (0.26, 0.75)]   # (commission, loss ratio) points of the scale, before the shift
shift = technical_commission - expected_commissions(qs.with_sliding_scale(shape))[0] / ceded_premium
anchors = [(c + shift, lr) for c, lr in shape]    # moved so the expected commission is the technical one
slide = qs.with_sliding_scale(anchors)
sd_slide = describe("sliding scale", underwriting_result(slide))
print(f"  scale: {anchors[0][0]:.1%} commission at a loss ratio of {anchors[0][1]:.0%} or less, "
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
sd_pc = describe("profit comm.", underwriting_result(pc_layer, with_profit_commission=True))
print(f"  flat commission {commission:.1%} plus 25% of profit after an allowance of {allowance:.1%} "
      f"(expected {expected_pc:.1%} of premium)")
print(f"reinsurer's sd falls {1 - sd_slide / sd_flat:.0%} with the scale and {1 - sd_pc / sd_flat:.0%} "
      f"with the profit commission")
```

```text
flat           mean  0.725  sd 2.005  worst 1%  -7.25  commission it can add 0.0%
sliding scale  mean  0.725  sd 1.392  worst 1%  -5.91  commission it can add 0.7%
  scale: 35.1% commission at a loss ratio of 55% or less, 24.1% at 75% or more
profit comm.   mean  0.725  sd 1.814  worst 1%  -7.03  commission it can add 0.1%
  flat commission 30.3% plus 25% of profit after an allowance of 37.9% (expected 1.2% of premium)
reinsurer's sd falls 31% with the scale and 10% with the profit commission
```

The mean result is the same in all three rows, and it is the margin that the
reinsurer priced. Under flat terms, the worst 1% of years equals minus the
capital from the pentagon. The scale and the profit commission decrease the
spread, and the cedant takes the difference: its commission income now moves
with its own loss ratio.

Less spread needs less capital, and in this framework less capital gives a
smaller margin. The last column is the additional expected commission that the
reinsurer can pay under each form and still earn 10% on the capital it then
needs. This saving is the economic reason for variable commissions: the
reinsurer can give part of it back to the cedant. The size of the saving
depends on the loss-ratio distribution. A model with more parameter and reserve
risk than this one gives a different spread.

## 6. How price follows the risk profile

A primary rating plan puts a risk profile into rating variables. A reinsurer
has no rating plan at policy level. The profile goes into the **loss model**,
and the price follows from that model.

### Where the loss model comes from

This guide sets the model by assumption (4 large losses a year, alpha 1.8). A
reinsurer builds it in two ways and blends them:

- **Experience rating**: trend each of the cedant's large losses to the treaty
  year and develop it to ultimate. Then fit a frequency above a threshold and a
  severity tail. This step uses the cedant's loss triangles and the trend work
  that a primary actuary knows. `LargeLossExperience` converts, develops and
  trends each loss and applies the layers to it. The guide
  [experience and exposure rating](experience-and-exposure-rating.md) shows the
  steps.
- **Exposure rating**: use the cedant's **risk profile**, its policies grouped
  by sum insured or limit, with the premium in each band. An **exposure curve**
  gives the share of the expected loss of each band above each attachment. The
  expected loss of a layer follows directly from the profile. Use `RiskProfile`
  and exposure curves such as `Mbbefd`.
- **Blending**: weight the two by credibility. Put more weight on experience
  when the cedant has many large losses and a stable book.

### The levers

These parts of the profile change the price of a layer. The table below shows
the first four.

- **frequency**: how many losses reach the program.
- **inflation and trend**: inflation increases each loss, but policy limits and
  attachments stay the same. Thus the loss to a layer increases by more than
  the inflation rate. This leverage is the most important effect for excess of
  loss. Index clauses move the attachment with inflation to share it.
- **variability of the count**: when the mean count is uncertain, a Poisson
  count understates how much the number of large losses changes from year to
  year. The usual correction is a negative binomial count.
- **severity tail**: how heavy the tail is, and where policy limits cut it.
  Uncertainty in alpha is often the largest driver of a quote.
- **limits and sums insured**: the most that one risk can cost, which exposure
  rating reads from the profile.
- **catastrophe and clash**: one event can hit many risks, or several of the
  cedant's policies together. This makes the tail of the annual loss heavier
  and does not change the typical loss. The definition of an occurrence sets how
  losses add.
- **development**: how much reported losses develop to ultimate (late reports
  and growth on known claims). It is large and slow for excess layers. This
  example does not include it.
- **loss adjustment expenses**: the layer pays a different amount when claim
  expenses are included in the limit, shared pro rata, or paid in addition.
- **cedant quality**: underwriting and claims handling. A reinsurer judges
  these outside the model and puts them in its parameters and loads.

The code prices the same program with the pricing rule of section 4. It
compares the base book with four variants:

1. 25% more large losses.
2. 5% inflation on each loss. With a Pareto tail, this multiplies the number of
   losses above 1 by 1.05 to the power 1.8 and keeps their shape.
3. a count with twice the Poisson variance.
4. a heavier tail (alpha 1.5, not 1.8).

```python
profiles = {
    "base": (Poisson(4.0), 1.8),
    "25% more losses": (Poisson(5.0), 1.8),
    "5% inflation": (Poisson(4.0 * 1.05**1.8), 1.8),
    "variable count": (NegativeBinomial.from_mean_variance(4.0, 8.0), 1.8),
    "heavier tail": (Poisson(4.0), 1.5),
}
print(f"{'':>16}" + "".join(f"{layer.name:>16}" for layer in LAYERS))
print(f"{'':>16}" + f"{'loss':>8}{'price':>8}" * len(LAYERS))


def change(new, old):
    return f"{round(new / old - 1, 2) + 0.0:+8.0%}"    # + 0.0 prints -0% as +0%


base = None
for name, (count, alpha) in profiles.items():
    grids = Tower(LAYERS).on_grid(count, severity_grid(alpha), POINTS)
    row = [(p.loss, gross_up(p.premium)) for p in grids.price(price_of_risk).ceded]
    if base is None:
        base = row
        cells = [f"{loss:8.3f}{price:8.3f}" for loss, price in row]
    else:
        cells = [f"{change(loss, l0)}{change(price, p0)}" for (loss, price), (l0, p0) in zip(row, base)]
    print(f"{name:>16}" + "".join(cells))
print(f"1.05 raised to alpha 1.8: {1.05 ** 1.8:.3f}; 1.25 raised to the hazard parameter: "
      f"{1.25 ** price_of_risk.parameters['rho']:.3f}")
```

```text
                             2x3             5x5           10x10
                    loss   price    loss   price    loss   price
            base   0.686   1.271   0.586   1.425   0.337   1.160
 25% more losses    +24%    +17%    +25%    +18%    +25%    +17%
    5% inflation     +9%     +7%     +9%     +7%     +9%     +6%
  variable count     -1%     +0%     +0%     +1%     +0%     +1%
    heavier tail    +47%    +33%    +78%    +54%   +120%    +77%
1.05 raised to alpha 1.8: 1.092; 1.25 raised to the hazard parameter: 1.165
```

The base row is in money. The other rows are changes from it.

A quarter more losses increases the expected loss of each layer by about a
quarter, but the price by less (17% to 18%). The pricing rule causes this, not
the book. The distortion stays fixed, and a proportional hazard distortion with
a parameter below 1 prices a layer approximately in proportion to its frequency
raised to that parameter. The second number on the last line is 1.25 raised to
0.686. Thus the multiple decreases for a worse book. A reinsurer does not
accept that, so for a worse book it usually changes the price of risk too.

Inflation of 5% increases the expected loss of each layer by about 9%, near
1.05 raised to the power 1.8 (the first number on the last line). With a Pareto
tail and fixed limits, this leverage is the same at each layer. With a lighter
tail it increases with the attachment.

A more variable count has almost no effect on any layer here. Few losses reach
a layer, and for that thinned count most of the extra variance goes away. The
count variance has a larger effect on covers of the annual total, such as a
quota share or a stop loss.

A heavier tail moves the layers apart. The expected loss increases by about
half in the working layer and more than doubles in the top layer. Thus the
cedant's loss ratio is only the first question: a book with the same frequency
and a different tail is a different risk.

## 7. Stand-alone and as part of a book

The sections above price each contract alone. A reinsurer holds a portfolio,
and the capital a contract needs depends on the other contracts. When the
contract and the book rarely have bad years together, the contract adds little
to the tail of the book and needs little extra capital. If the bad years are the
same, it needs more.

Add the 5x5 layer to a book with expected loss 30 and a coefficient of
variation of 25%. Do this once with the two independent, and once with a
correlation of 0.5 between their annual losses. A **Gaussian copula**
simulates the pairs of annual losses with the chosen correlation. A
`PredictiveDistribution` holds them, as a table of simulated outcomes with one
column per part.

`price_portfolio` prices the whole and **allocates** the price to its parts
with the **natural allocation**. Each part gets the share of the total capital
that it contributes in the bad years, so the parts add to the whole exactly.
The **marginal** capital is different: it is the extra capital the book needs
because of the contract. The code shows both.

```python
other_book = Lognormal.from_mean_cv(30.0, 0.25)
DRAWS = 40_000


def in_a_book(correlation):
    u = GaussianCopula([[1.0, correlation], [correlation, 1.0]]).sample(DRAWS, 1)
    rows = [[annual.quantile(a), other_book.quantile(b)] for a, b in u]
    pd = PredictiveDistribution(["part"], [("contract",), ("book",)], rows)
    return price_portfolio(pd, TAIL, cost_of_capital=RETURN)


in_book = {}
for correlation in (0.0, 0.5):
    p = in_a_book(correlation)
    alone, joined, book_alone = p.standalone[0], p.allocated[0], p.standalone[1]
    in_book[correlation] = joined.premium
    print(f"correlation {correlation:.1f}: the book's assets {book_alone.assets:.2f} alone, "
          f"{p.total.assets:.2f} with the contract ({p.total.assets - book_alone.assets:.2f} more)")
    for label, price in (("contract alone", alone), ("in the book", joined)):
        print(f"  {label:<15} risk premium {price.premium:.3f}  capital {price.capital:.2f}  "
              f"loading {price.premium - price.expected_loss:.3f}")
```

```text
correlation 0.0: the book's assets 56.43 alone, 57.45 with the contract (1.02 more)
  contract alone  risk premium 1.230  capital 6.43  loading 0.643
  in the book     risk premium 0.668  capital 0.81  loading 0.081
correlation 0.5: the book's assets 56.48 alone, 60.33 with the contract (3.84 more)
  contract alone  risk premium 1.230  capital 6.43  loading 0.643
  in the book     risk premium 0.947  capital 3.60  loading 0.360
```

The draws are simulated with a fixed seed, so the stand-alone risk premium is a
little different from the grid's 1.240.

Diversification is a discount on capital. In an unrelated book, the risk
premium of the same contract is about half the stand-alone value. Its
**loading** (risk premium less expected loss) decreases by much more than half.
At a correlation of 0.5, much of the discount goes away.

The marginal assets (the increase in the book's assets) and the allocated
capital are near but not equal. The natural allocation divides the total so
that the parts add up. Marginal figures for several contracts do not add up. A
reinsurer that already has much of the same type of loss quotes higher than
one that does not. Both prices are correct for their portfolios.

Two cautions apply:

- The discount depends on the size and shape of the book. Here a book with a
  mean of 30 is much larger than a contract that pays nothing in four years out
  of five.
- A Gaussian copula has no tail dependence. Even at a correlation of 0.5, it
  understates how often both parts have their worst years together.

## 8. Hard and soft markets

In a **hard market**, reinsurers charge more for the same cover and offer less
of it: less limit, higher attachments, fewer free reinstatements, and stricter
wording. A **soft market** is the opposite. A price rise for the same cover and
exposure has three parts:

- a higher **expected loss**: loss trend, adverse development on the reserves
  of the cedant and of the market, or a new view of the tail.
- a higher **price of risk**: the return the market demands on the capital it
  holds, here the parameter of the pricing distortion.
- **tighter terms**, which give less cover for the money.

A calibration to a quote separates the first part from the second only if you
trust the loss model. Take two illustrative quotes for the 5x5 layer with a
free reinstatement, as rates on line: 22% in a soft market and 34% in a hard
market. Assume that the reinsurer's expected loss has not changed, so the
difference is all price of risk.

The cedant pays the quote, so the reinsurer's risk premium is the quote less
expenses and brokerage. `Quote.at` restates the technical quote of section 3
at a quoted premium. It keeps the expected loss and assets, and the risk
premium is the remainder after the loads. For each quote, the code gives two
results:

- the return on capital that the reinsurer gets.
- the proportional hazard parameter that gives the quote on this layer, and
  the prices of the other layers at that parameter.

```python
QUOTES = {"soft": 0.22, "hard": 0.34}   # rate on line for the 5x5 layer

market = {}
for name, rol in QUOTES.items():
    offered = technical_quote.at(rol * mid.limit)   # same loss and assets, at the quoted premium
    distortion = calibrate("ph", annual, offered.risk_premium)
    market[name] = distortion
    multiples = [gross_up(p.premium) / p.loss for p in tower_grids.price(distortion).ceded]
    print(f"{name}: quote {offered.premium:.2f} (multiple {offered.multiple:.2f}), "
          f"return on capital {offered.return_on_capital:.1%}, "
          f"hazard parameter {distortion.parameters['rho']:.3f}")
    print(f"      multiples by layer {', '.join(f'{m:.2f}' for m in multiples)}")
print(f"technical premium {technical_quote.premium:.2f} (rate on line {technical_quote.rate_on_line:.1%}) "
      f"at a {RETURN:.0%} return")
```

```text
soft: quote 1.10 (multiple 1.88), return on capital 5.4%, hazard parameter 0.791
      multiples by layer 1.57, 1.88, 2.37
hard: quote 1.70 (multiple 2.90), return on capital 14.2%, hazard parameter 0.616
      multiples by layer 2.07, 2.90, 4.42
technical premium 1.43 (rate on line 28.5%) at a 10% return
```

The expected loss and the layer are the same. The soft quote gives the
reinsurer about half its 10% target return, and the hard quote gives much more.
The technical premium is between them. A lower hazard parameter gives a higher
price of risk.

In this model the hard market costs more on each layer, and most on the top
layer. This pattern comes from the single parameter of the proportional hazard
family. A cost-of-capital rule or another distortion spreads the increase
differently. In practice, a hard casualty market often affects working layers
most (pricing on losses, higher attachments). A hard catastrophe market often
affects the upper layers most. When quotes on more than one layer are
available, calibrate to all of them.

### Tighter terms

A harder market also shows as terms, and the cedant can change the terms to
get a lower price. The code prices the same layer under the hard market
distortion with a higher attachment, and with no reinstatement:

```python
variants = [
    Layer("base", limit=5.0, attachment=5.0, reinstatements=1),
    Layer("attach 7", limit=5.0, attachment=7.0, reinstatements=1),
    Layer("no reinstatement", limit=5.0, attachment=5.0, aggregate_limit=5.0),
]
variant_grids = Tower(variants).on_grid(frequency, severity, POINTS)
print(f"{'terms':>17} {'exp loss':>9} {'quote':>7} {'multiple':>9} {'ROL':>6}")
for layer, grid, p in zip(variants, variant_grids.ceded, variant_grids.price(market["hard"]).ceded):
    quote = gross_up(p.premium)
    print(f"{layer.name:>17} {p.loss:9.3f} {quote:7.3f} {quote / p.loss:9.2f} {quote / layer.limit:6.1%}")
```

```text
            terms  exp loss   quote  multiple    ROL
             base     0.586   1.700      2.90  34.0%
         attach 7     0.369   1.257      3.41  25.1%
 no reinstatement     0.558   1.469      2.63  29.4%
```

A higher attachment decreases the expected loss by 37% and the quote by 26%.
The layer is cheaper, but each unit of expected loss costs more, because the
remaining loss is more remote. With no reinstatement, the expected loss
decreases by only 5% and the quote by 14%. The most the layer can pay in a year
is half, and so is the capital. A buyer who changes terms for price gives up
very different amounts of expected loss and of capital, and the quote follows
the capital.

These factors change the price of risk:

- **capital supply**: after large losses (catastrophes in property, reserve
  increases in casualty), reinsurers have less capital, and investors want a
  higher return to supply it.
- **capital requirements**: rating agencies and regulators set how much capital
  a reinsurer must hold for a risk.
- **interest rates**: higher rates increase the return that investors require
  on capital, which increases prices. Higher rates also increase what the
  reinsurer earns on premium and reserves until it pays claims, which permits a
  lower underwriting margin. The larger effect depends on the payout and on the
  return asked over the rate. This guide's one-year model has no investment
  income. Section 10 restates the price over a payout, with investment income
  as the discount on losses and capital held for longer.
- **retrocession cost**: when the reinsurers' own reinsurance costs more, they
  increase their prices.

These factors change the expected loss: loss trend, the cedant's own loss
experience and reserve development, and new views of the tail. They change the
technical price in all markets.

## 9. What the reinsurer decides

The decision needs two technical prices:

- the **stand-alone technical price** (section 3): the reference price for a
  reinsurer with no other business, or for a contract with no diversification
  benefit.
- the **technical price in a book** (section 7): the cost of the contract in
  the reinsurer's current portfolio. It is grossed up for expenses and
  brokerage like the stand-alone price.

```python
floors = {
    "alone": technical,
    "book, unrelated": gross_up(in_book[0.0]),
    "book, corr 0.5": gross_up(in_book[0.5]),
}
print(f"{'':>11}" + "".join(f"{name:>17}" for name in floors))
print(f"{'technical':>11}" + "".join(f"{price:17.3f}" for price in floors.values()))
for name, rol in QUOTES.items():
    quote = rol * mid.limit
    print(f"{name + ' ' + format(quote, '.2f'):>11}" + "".join(f"{quote / price - 1:+17.0%}" for price in floors.values()))
```

```text
                       alone  book, unrelated   book, corr 0.5
  technical            1.425            0.768            1.089
  soft 1.10             -23%             +43%              +1%
  hard 1.70             +19%            +121%             +56%
```

Each row gives the distance of the quote from each technical price. The soft
quote is below the stand-alone price, well above the price in an unrelated
book, and only just above the price in a correlated book. The hard quote is
above all three. For each submission, a reinsurer selects one action:

- **write** the share offered, if the quote is at or above the technical price
  for its own portfolio.
- **write less** at the same terms, but only if capital is not linear in the
  share. Examples are a concentration or accumulation limit, or a share large
  enough to change the tail of the portfolio. Otherwise a smaller share of an
  underpriced contract is only a smaller underpriced position.
- **change the structure** (a higher attachment, a lower limit, no free
  reinstatement) if different terms meet the price.
- **decline** in all other cases.

## 10. What the model leaves out

The example is small on purpose. Each item below changes a real price in the
direction given:

- **Development**: reported excess losses grow slowly to ultimate. If you
  ignore development, the expected loss is too low, more so on higher layers.
  This example's loss model has no development. For experience rating,
  `LargeLossExperience.prepare(development=...)` develops each known large loss
  to ultimate. A count development pattern (`DevelopmentPattern`) given to
  `burning_cost(development=...)` adds the late-reported claims. See the
  [experience and exposure rating guide](experience-and-exposure-rating.md).
- **Loss adjustment expenses**: claim expenses added to the loss increase what
  the layer pays.
- **Payout, discounting and investment income**: the guide does not discount
  and prices one year of capital. For a long-tail line such as liability, the
  reinsurer holds capital and reserves for the whole payout, often many years.
  This increases the cost of capital. The reinsurer also earns investment
  income on them, which decreases the price. The example below restates a
  one-year price over a payout. With a constant return on capital held through
  the payout, the price over the payout is higher. Market prices for liability
  layers include more than this (interest rates and spreads, capital released
  as claims settle). Thus the example is not a statement about liability
  multiples.
- **Parameter uncertainty**: alpha and the frequency are estimates. If you
  include their uncertainty, the tail is heavier and the price is higher.
- **Data and credibility loads**: a small or unreliable submission gets a load.
- **Clash and catastrophe aggregation**: when one event hits several of the
  cedant's risks, the occurrence definition adds their losses. If you ignore
  it, the tail is too light. For catastrophe covers, the library takes the
  event loss table of a cat model (`EventLossTable`). `EventSet.grouped` groups
  losses into occurrences by an hours clause.
- **Credit risk**: funds withheld by the cedant, collateral and the reinsurer's
  own credit standing change what each side accepts.
- **Expenses and tax**: fixed costs per contract and tax both increase the
  price of small contracts above a percentage loading. `Quote.technical` takes
  a `fixed` cost per contract with the percentages.

### Pricing over the payout

`price_over_payout` restates a one-year price over a **payment pattern** (the
share of the ultimate loss paid in each year). It discounts the losses at the
given rate, which is how investment income enters. It holds the capital while
losses are outstanding, so it charges capital for more than one year. The cost
of capital is then a return over the discount rate. Give it the rule of the
one-year price: here the 99% TVaR price of section 3 at a 10% cost of capital.
Assume that the 5x5 layer pays 40% of its losses in the first year, then 30%,
20% and 10%, and that money earns 3%:

```python
payout = PaymentPattern([1.0, 2.0, 3.0, 4.0], [0.4, 0.3, 0.2, 0.1])   # years from inception, shares
over_payout = price_over_payout(tvar_price, payout, rate=0.03, cost_of_capital=RETURN)
print(f"discount factor {over_payout.discount_factor:.4f}, capital years {over_payout.capital_years:.4f}")
print(f"discounted loss {over_payout.discounted_loss:.4f}, margin {over_payout.margin:.4f}, "
      f"capital {over_payout.capital:.4f}")
print(f"risk premium {over_payout.premium:.4f} over the payout, {tvar_price.premium:.4f} for one year")

# The same payout with other rates: no discounting, and 10% as the total return (7% over 3%).
undiscounted = price_over_payout(tvar_price, payout, rate=0.0, cost_of_capital=RETURN)
total_10 = price_over_payout(tvar_price, payout, rate=0.03, cost_of_capital=RETURN - 0.03)
print(f"at a zero rate {undiscounted.premium:.4f}; at 7% over 3% {total_10.premium:.4f}")
```

```text
discount factor 0.9430, capital years 1.9568
discounted loss 0.5529, margin 1.1824, capital 6.0423
risk premium 1.7353 over the payout, 1.2401 for one year
at a zero rate 1.7849; at 7% over 3% 1.4233
```

Discounting decreases the expected loss from 0.586 to 0.5529: that is the
investment income. But the reinsurer now holds capital while losses are
outstanding. The initial capital of 6.0423 runs off with the outstanding
losses, which is equal to 1.9568 discounted years of it. The margin (10% a year
on that capital over those years) is 1.1824, against 0.654 for one year. The
risk premium increases from 1.2401 to 1.7353.

Three things changed together:

- the reinsurer holds capital for longer.
- the losses are discounted.
- 10% over a 3% rate gives investors 13% in total, against 10% at the zero
  rate of section 3.

Take them one at a time:

- At a zero rate, the price over the payout is 1.7849. Discounting alone
  decreases it only to 1.7353.
- If 10% is the total return, the return over the rate is 7%, and the price is
  1.4233. This is still above 1.2401.

With these assumptions, the cost of holding capital for longer is larger than
the investment income. The payout here is an assumption. For a real layer, the
payout pattern, the discount rate and the return over it decide which effect is
larger. `PaymentPattern.from_development` gives a payout pattern, for example
from a paid chain ladder.

## 11. Glossary and further reading

**Annual aggregate deductible**: the part of the year's layer losses that the
cedant keeps before the layer pays. **Annual aggregate limit**: the most a
layer pays in a year. **Assets**: premium plus capital. **Attachment**: the
loss size above which an excess of loss layer pays. **Attritional losses**: the
frequent small losses below the large-loss threshold. **Capital**: the
reinsurer's own money held for a contract. **Capital years**: how long the
capital is held over a payout, in discounted years, with the capital running
off as losses are paid. **Cedant**: the insurer that buys reinsurance.
**Ceding commission**: what the reinsurer pays the cedant on a quota share.
**Cost of capital**: the return the reinsurer asks on its capital (over a
payout, a return over the discount rate). **Deposit premium**: the premium paid
at inception, before any reinstatement premium. **Discount factor**: the
present value of one unit of ultimate loss paid on the payment pattern.
**Distortion**: a reweighting of probabilities that prices a loss by its tail.
**Exhaustion**: a loss that uses a layer's whole limit. **Hard market**: prices
up and less cover for the same risk. **Layer**: the slice of each loss that one
contract pays, written limit x attachment. **Loss on line**: expected loss
divided by limit. **Multiple**: premium divided by expected loss. **Natural
allocation**: a split of a portfolio's capital and price among its parts that
adds up exactly. **Payback**: limit divided by premium, the inverse of the rate
on line. **Payment pattern**: the share of the ultimate loss paid in each year.
**Pentagon**: loss, premium, margin, capital and assets, linked by two
identities. **Price of risk**: the loading the market puts on risk, as a return
on capital or a distortion parameter. **Profit commission**: a share of the
reinsurer's profit returned to the cedant. **Program**: all the layers a cedant
buys together. **Quota share**: reinsurance of a fixed share of each policy.
**Rate on line (ROL)**: premium divided by limit. **Reinstatement**: a layer's
limit restored after a loss, free or for a **reinstatement premium**.
**Retrocession**: reinsurance bought by a reinsurer. **Risk premium**: expected
loss plus margin, before expenses and brokerage. **Sliding scale**: a ceding
commission that decreases as the loss ratio increases. **Soft market**: prices
down and much cover available. **Stand-alone price**: a contract's price with
no diversification benefit. **Subject premium**: the cedant's premium that the
reinsurance applies to. **Technical commission**: the ceding commission that
the reinsurer's model supports. **Technical premium**: the price that the
reinsurer's model supports, after expenses and brokerage. **Treaty**: a
reinsurance contract that covers a whole book. **TVaR**: the average of the
worst outcomes beyond a probability level.

Further reading, by question:

- Clark, "Basics of Reinsurance Pricing", CAS Study Note: experience and
  exposure rating of layers and increased limits. Start here.
- Werner and Modlin, *Basic Ratemaking* (CAS): the trend and development work
  that experience rating of a layer uses.
- Mildenhall and Major, *Pricing Insurance Risk: Theory and Practice* (Wiley,
  2022): the pentagon, distortions and the natural allocation of sections 2, 4
  and 7.
- The notebook [`examples/reinsurance_pricing.ipynb`](../../examples/reinsurance_pricing.ipynb)
  prices six contracts and checks them against Mildenhall's `aggregate`.

### About these numbers

CI runs each Python block in this guide (`python/tests/test_guides.py`) and
compares each printed result with the text under the block.
