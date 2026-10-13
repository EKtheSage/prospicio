# What a reinsurance price is

This guide is for an actuary who prices primary insurance, has computed a loss
ratio for a reinsurance submission, and wants the whole picture. It answers four
questions, first in brief and then with a worked example.

## Answers in brief

- **What does reinsurance pricing produce?** Three things: the **terms** (the
  layers or the share, attachments, limits, reinstatements, commissions), a
  **price** for those terms, and the reinsurer's **expected economics** of the
  contract (expected loss, margin, the capital it must hold and the return on
  that capital).
- **What should we charge?** For an excess of loss layer, the technical
  premium is (expected loss + return asked × capital held) / (1 - expenses -
  brokerage). For a quota share the price is the ceding commission:
  1 - expected loss ratio - expenses - brokerage - margin. Both are floors for a
  reinsurer with no other business; a reinsurer whose book diversifies the
  contract can accept less.
- **How do we price without a rating plan?** There is no policy-level rating
  plan. The price follows a loss model of the cedant's book (how many large
  losses, how big, how heavy the tail), built from the cedant's own large-loss
  history (experience rating) and from exposure curves applied to its profile
  of limits and sums insured (exposure rating).
- **What does a hard market mean?** Reinsurers charge more for the same cover
  and offer less of it. Part of a price rise can be a higher expected loss
  (trend, development, a new view of the tail), part a higher **price of risk**
  (the return the market demands on capital), and part tighter terms. A model
  of the loss separates the first from the other two.

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
- **Layer**: the slice of each loss that one excess of loss contract pays.
  It is written limit x attachment: **5x5** pays the part of a loss between 5
  and 10, and **2x3** the part between 3 and 5.
- **Attachment**: the loss size where the layer starts paying. **Limit**: the
  most it pays for one loss. **Exhaustion**: a loss large enough to use up the
  whole limit (for 5x5, any loss of 10 or more).
- **Reinstatement**: restoring the limit after a loss has used it, so the layer
  can pay again in the same year. The **annual limit** is the most the layer
  pays in a year.
- **Capital**: the reinsurer's own money that it must hold, over and above the
  premium, so it can still pay in a bad year. **Assets**: premium plus capital,
  everything that stands behind the contract.
- **Cost of capital**: the return the reinsurer wants on that capital, here 10%
  a year.
- **Loss on line** and **rate on line**: expected loss divided by limit, and
  premium divided by limit. Loss on line is the pure cost of each unit of
  cover; rate on line is its price.
- **Retrocession**: reinsurance that a reinsurer buys for itself.

The glossary in section 11 has the rest.

## 2. What is priced

A primary insurer prices a policy: one risk, one year, a rating plan that turns
rating variables into a premium. A reinsurer prices a **contract** on a
**portfolio** for a **period**. The cedant hands over a book, not a policy, and
the price comes from a model of the cedant's losses.

We use one running example, a synthetic commercial property book with excess
of loss cover per risk. Money is in millions. The cedant writes 60 of subject
premium (the premium the reinsurance applies to). Losses below 1 are
**attritional**: frequent, fairly stable, expected to cost 28 in total. Losses
above 1 are the **large losses**, and they are what excess of loss reinsurance
is about. They arrive as a Poisson count with a mean of 4 a year, and their size
has a Pareto tail with alpha 1.8. An alpha below 2 means the average loss
exists but the variance would be infinite without a cap; here the cap is the
policy limit (the largest sum insured) of 20. The numbers are illustrative,
chosen to show the mechanics, not to match any market.

```python
import math
import re

from prospicio.aggregate import fft
from prospicio.distributions import Grid, Lognormal, NegativeBinomial, Pareto, Poisson, PredictiveDistribution
from prospicio.pricing import Pentagon, price_portfolio
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

How to read the code: `frequency` is the number of large losses in a year and
`severity` the size of each. The library works on a **grid**: each size is
rounded to a step of 0.1, and `fft` combines frequency and severity into the
distribution of the year's total (by fast Fourier transform, with no
simulation error). Later, `Tower(...).on_grid` applies layers to each loss and
gives the distribution of what each layer pays in a year.

The 60.9% is the loss ratio a primary actuary would bring to a submission. The
reinsurer needs the whole distribution behind it, because a contract that pays
only above a high attachment cares about the tail, not the mean.

### The pricing pentagon

Five quantities describe a priced contract:

- **loss** (L): the expected loss the reinsurer pays;
- **premium** (P);
- **margin** (M = P - L): what the reinsurer expects to keep for bearing risk;
- **capital** (Q) and **assets** (a = P + Q).

Two identities link them, M = P - L and a = P + Q, so any three that are
independent fix the other two. The **return on capital**, M / Q, is a ratio
derived from them, and it can stand in for one of the three: the library's
`Pentagon.solve` below is given the loss, the assets and the return.

Take the 5x5 layer with one free reinstatement (the limit is restored once at
no charge, so the layer pays at most 10 in a year). Set the assets at the 99%
**tail value at risk** (TVaR) of its annual loss, the average of the worst 1% of
years, and ask for a 10% return on capital. The premium this gives is the
**risk premium**: expected loss plus margin, before the reinsurer's expenses
and brokerage.

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

Read it from the reinsurer's side. The layer is expected to pay about 0.59 a
year, but the reinsurer must have 7.8 of assets behind it for the bad years.
The risk premium supplies 1.2 of that, so the reinsurer puts up the other 6.5 as
capital, and wants 10% on it. That return is the margin of 0.65. Everything in
this guide is a way of getting these five numbers for some contract.

## 3. An excess of loss layer

An **excess of loss** (XoL) layer pays the part of each loss that falls between
its attachment and attachment plus limit. The first questions are about shape.
How often is the layer touched, how often is it exhausted, and what does it cost
per unit of cover?

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

Of the large losses (those above 1), about one in eighteen reaches the layer
and about one in sixty exhausts it, so in most years the layer pays nothing:
only one year in five has a loss to it. The loss on line above is on the
occurrence limit with no annual limit. The annual limit of 10 shaves the
expected loss slightly, because a year with three exhausting losses pays 10,
not 15.

### The technical premium

The **technical premium** is the price the reinsurer's own model supports,
built from four parts:

1. the **expected loss**;
2. the **cost of capital**: the return asked for times the capital the layer
   needs (the margin from the pentagon);
3. **internal expenses**: here 3% of the premium;
4. **brokerage**: here 10% of the premium, paid to the intermediary.

Expenses and brokerage are fractions of the premium itself, so they divide
rather than add: premium = (expected loss + cost of capital) / (1 - 3% - 10%).
Three simplifications are worth knowing. Real internal expenses are partly a
fixed cost per contract, so a flat percentage understates them on small
contracts. There is no tax line, so the 10% is a return after tax. And
brokerage is charged on reinstatement premium as well as on the deposit.

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
```

```text
expected loss       0.586
cost of capital     0.654   (10% of capital 6.538)
risk premium        1.240
technical premium   1.425   (expenses and brokerage take 13%)
rate on line 28.5% (payback 3.5 years), multiple 2.43, loss ratio 41.1%
```

Three ratios describe a layer's price, and brokers quote all of them:

- the **rate on line** (ROL): premium divided by limit. Its inverse is the
  **payback**, the number of loss-free years the premium takes to pay one full
  limit;
- the **multiple**: premium divided by expected loss, how many times the pure
  cost the buyer pays;
- the **loss ratio**: expected loss divided by the premium the reinsurer
  receives for this layer, the inverse of the multiple. It is not the cedant's
  book loss ratio of 60.9%.

The same build-up is in the library. A distortion with a constant cost of
capital (`Distortion.ccoc`) prices a loss as (expected loss + r × assets)
divided by (1 + r), and `TowerGrids.price` applies it to every layer of a tower.
That call takes the most the layer can pay in a year as its assets, which for
this layer is the annual limit of 10, a different capital standard from the 99%
TVaR above. The arithmetic and the call agree:

```python
by_hand = (expected_loss + RETURN * mid.aggregate_limit) / (1 + RETURN)
by_call = mid_grids.price(Distortion.ccoc(RETURN)).ceded[0]
print(f"full-limit capital: by hand {by_hand:.4f}, TowerGrids.price {by_call.premium:.4f}")
print(f"99% TVaR capital:   by hand {risk_premium:.4f}, Pentagon {pentagon.premium:.4f}")
```

```text
full-limit capital: by hand 1.4421, TowerGrids.price 1.4421
99% TVaR capital:   by hand 1.2401, Pentagon 1.2401
```

The choice of capital standard is a pricing decision, not a detail. A full-limit
standard charges every layer as if its limit would be lost; a tail standard
charges for the loss the reinsurer actually fears. The rest of this guide uses
the 99% TVaR standard.

### Reinstatements

Paid reinstatements are usual in property and catastrophe covers; casualty
working layers often have free reinstatements or just an annual limit. Two more
annual terms change what a layer pays in a year: an **annual aggregate
deductible** (the cedant keeps the first part of the year's layer losses) and
the **annual aggregate limit**.

A paid reinstatement works like this. The reinsurer charges a **deposit
premium** at inception, and then a **reinstatement premium** each time a loss
uses up limit that is restored: here 100% of the deposit, pro rata to the amount
reinstated. Start by setting the deposit so that the deposit plus the expected
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

The buyer sees a lower rate on line at inception and pays the rest when losses
come. Market rates on line for such a layer are quoted on the deposit, so this
layer's comparable figure is the deposit rate on line, not 28.5%.

That is not the whole story, because the reinsurer collects reinstatement
premium in exactly the years it pays large losses. Its net loss in a bad year
is smaller, so it needs less capital. Hold capital against the loss net of the
reinstatement premium it keeps (after expenses and brokerage), and solve for
the deposit again:

```python
net_deposit = deposit
for _ in range(20):   # the reinstatement premium depends on the deposit, so iterate
    net_loss, _ = annual.map(lambda x: x - KEEP * RATE * net_deposit * min(x, 5.0) / 5.0)
    net = Pentagon.solve(loss=net_loss.mean(), assets=TAIL.measure(net_loss), return_on_capital=RETURN)
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

So a paid reinstatement is partly a transfer of premium from good years to bad
years and partly a real discount, because it cuts the capital the reinsurer
needs. The rest of the guide keeps the free reinstatement, so its quotes are
the whole premium for the layer.

## 4. Where the layer sits

Price three layers of the same program: 2x3 (a working layer, attaching at 3
and touched often), 5x5, and 10x10 (a top layer, attaching at 10, touched
rarely, but with a large loss when it is). Together they cover losses from 3 to
20, the policy limit. To compare them, use **one** pricing rule for all three.

Here that rule is a **distortion**. A distortion prices a loss as a weighted
average in which bad outcomes get more weight than their probability. The
**proportional hazard** distortion raises the chance of exceeding every loss
size to a power below 1: a parameter of 1 means no loading, and a lower
parameter means a bigger loading. `calibrate` finds the parameter that
reproduces a given premium on one layer; here, the 5x5 risk premium found
above. That one parameter then prices the other two layers with no further
choice.

<details>
<summary>Helper: read the parameter of a distortion</summary>

```python
def parameter(distortion):
    """The one number that names a distortion, such as 0.69 in proportional_hazard(0.69).

    The library has no accessor for it yet, so this relies on the repr's format.
    """
    return float(re.search(r"\(([-0-9.e]+)\)", repr(distortion)).group(1))
```

</details>

```python
LAYERS = [Layer("2x3", limit=2.0, attachment=3.0, reinstatements=1), mid,
          Layer("10x10", limit=10.0, attachment=10.0, reinstatements=1)]
tower_grids = Tower(LAYERS).on_grid(frequency, severity, POINTS)
price_of_risk = calibrate("ph", tower_grids.ceded[1], pentagon.premium)
print(f"proportional hazard parameter {parameter(price_of_risk):.3f}: a 1% chance is weighted as "
      f"{price_of_risk.g(0.01):.1%}, a 10% chance as {price_of_risk.g(0.10):.1%}")


def layer_table(layers, grids, distortion):
    print(f"{'layer':>6} {'exp loss':>9} {'assets/EL':>10} {'price':>6} {'multiple':>9} {'ROL':>6} "
          f"{'return':>7} {'at 10%':>7}")
    for layer, grid, p in zip(layers, grids.ceded, grids.price(distortion).ceded):
        a = TAIL.measure(grid)
        on_capital = (p.premium - p.loss) / (a - p.premium)        # return on 99% TVaR capital
        at_target = gross_up((p.loss + RETURN * a) / (1 + RETURN))  # price at exactly 10% on that capital
        print(f"{layer.name:>6} {p.loss:9.3f} {a / p.loss:10.1f} {gross_up(p.premium):6.3f} "
              f"{gross_up(p.premium) / p.loss:9.2f} {gross_up(p.premium) / layer.limit:6.1%} "
              f"{on_capital:7.1%} {at_target / p.loss:7.2f}")


layer_table(LAYERS, tower_grids, price_of_risk)
```

```text
proportional hazard parameter 0.686: a 1% chance is weighted as 4.3%, a 10% chance as 20.6%
 layer  exp loss  assets/EL  price  multiple    ROL  return  at 10%
   2x3     0.686        5.8  1.271      1.85  63.6%   14.5%    1.65
   5x5     0.586       13.3  1.425      2.43  28.5%   10.0%    2.43
 10x10     0.337       31.1  1.160      3.44  11.6%    7.1%    4.30
```

The columns: **assets/EL** is the 99% TVaR of the layer's annual loss per unit
of expected loss, how much money stands behind each unit of pure cost. The
**price** is the technical premium (the distortion's risk premium grossed up for
expenses and brokerage), and the multiple and rate on line are on that price,
so the 5x5 row matches section 3. **Return** is the margin over the 99% TVaR
capital the price implies, and **at 10%** is the multiple a rule charging
exactly 10% on each layer's own capital would give.

The multiple rises from the working layer to the top layer, although the
pricing rule is the same. The reason is the capital behind each unit of
expected loss. The working layer needs about 6 of assets per unit, because it
pays often and its result is predictable. The top layer needs about 31, because
its expected loss is a small chance of a very large payment.

The proportional hazard rule is not a constant return on capital, though. Its
implied return falls from 14.5% on the working layer to 7.1% on the top layer.
A rule that charged exactly 10% on every layer's 99% TVaR capital would make the
working layer cheaper (a multiple of 1.65) and the top layer dearer (4.30). A
one-parameter distortion charges the remote tail less than a constant-return
rule does. Which is right is a modelling choice; calibrating to market quotes on
more than one layer is how to choose.

The rate on line moves the other way, from 64% to 12%, because the top layer
pays far less often. Rate on line says how much cover costs per unit of limit;
the multiple says how much risk the reinsurer is loading for. This is why a
loss ratio alone cannot price a layer: two layers with the same expected loss
have different prices when one of them is a 1-in-100 event.

## 5. Quota share

A **quota share** takes a fixed share of every policy's premium and losses. The
reinsurer pays the cedant a **ceding commission** towards its acquisition costs,
so the reinsurer's economics are decided by the commission, not by a rate. The
price of a quota share is the commission.

The **technical commission** is what is left of the premium after everything
else the reinsurer needs:

technical commission = 1 - expected loss ratio - reinsurer expenses - brokerage - margin

A quota share carries all of the cedant's loss-ratio risk, not just the large
losses, so the attritional losses cannot be held at their mean. Here they get a
lognormal spread with a coefficient of variation of 10%, standing in for
pricing, inflation and reserve uncertainty. Take a 30% quota share of the whole
book. The margin comes from the same pentagon: assets at the 99% TVaR of the
ceded loss, and a 10% return on the capital.

<details>
<summary>Helpers: add two independent losses, expectations and tail means over the book</summary>

```python
def add_independent(a, b):
    """The total of two independent losses on grids with the same step, as (value, probability) pairs."""
    total = [0.0] * (len(a.probs) + len(b.probs))
    bs = [(j, q) for j, q in enumerate(b.probs) if q > 1e-16]
    for i, p in enumerate(a.probs):
        if p > 1e-16:
            for j, q in bs:
                total[i + j] += p * q
    return [(k * a.step, p) for k, p in enumerate(total) if p > 1e-16]


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
book = add_independent(attritional, gross)    # the whole book's annual loss
loss_ratio = expect(lambda x: x) / SUBJECT_PREMIUM
ceded_loss = CESSION * expect(lambda x: x)
ceded_assets = tail_mean([(CESSION * x, p) for x, p in book])
qs_pentagon = Pentagon.solve(loss=ceded_loss, assets=ceded_assets, return_on_capital=RETURN)
margin = qs_pentagon.margin / ceded_premium
QS_EXPENSES, QS_BROKERAGE = 0.02, 0.015
technical_commission = 1 - loss_ratio - QS_EXPENSES - QS_BROKERAGE - margin
print(f"ceded premium {ceded_premium:.1f}, expected loss ratio {loss_ratio:.1%}")
print(f"margin {margin:.1%} of premium (capital {qs_pentagon.capital:.2f}, {RETURN:.0%} return)")
print(f"technical commission = 1 - {loss_ratio:.1%} - {QS_EXPENSES:.1%} - {QS_BROKERAGE:.1%} "
      f"- {margin:.1%} = {technical_commission:.1%}")
```

```text
ceded premium 18.0, expected loss ratio 60.9%
margin 4.0% of premium (capital 7.25, 10% return)
technical commission = 1 - 60.9% - 2.0% - 1.5% - 4.0% = 31.6%
```

Any commission above that costs the reinsurer margin; any commission below it
hands margin to the cedant. It is also the number a cedant can test against its
own acquisition cost to see whether the reinsurance is a good deal. For a
long-tail line, the ceding commission also depends heavily on the investment
income the reinsurer earns before claims are paid, which this example leaves
out (section 10).

### Sliding scales and profit commissions

Both variants make the commission depend on how the year turns out, so part of
the loss-ratio volatility (not the mean) moves back to the cedant.

- A **sliding scale** pays a higher commission when the loss ratio is low and a
  lower one when it is high, in a straight line between two points. Below, the
  commission is 35.1% at a loss ratio of 55% or less, falling to 24.1% at 75% or
  more.
- A **profit commission** is a flat commission plus a share of the reinsurer's
  profit, if there is one. The standard form is a share of premium less
  commission, losses and an allowance for the reinsurer's management expenses,
  usually with a deficit carried forward to later years. This example's
  allowance also includes the reinsurer's margin and has no carry-forward; both
  choices change the expected profit commission, so model the actual wording.

Start by setting the scale and the profit share so that the **expected**
commission over the loss-ratio distribution equals the technical commission.
The library gives a contract's commission for any outcome, so the expectation
is a sum over the distribution:

```python
qs = Layer.quota_share("QS", CESSION).with_premium_rate(1.0, SUBJECT_PREMIUM)
costs = (QS_EXPENSES + QS_BROKERAGE) * ceded_premium


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
shift = technical_commission - expect(
    lambda x: qs.with_sliding_scale(shape).ceding_commission_for(CESSION * x)) / ceded_premium
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
    expected_pc = expect(lambda x: pc_layer.profit_commission_for(CESSION * x)) / ceded_premium
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

The mean result is the same in all three, and it is the margin the reinsurer
priced; the worst 1% of years under flat terms is minus the capital from the
pentagon, as it should be. The spread is smaller with the scale or the profit
commission, and the cedant has taken the difference: its commission income now
moves with its own loss ratio.

Less spread means less capital, and in this framework less capital means a
smaller margin. The last column is how much more expected commission the
reinsurer could pay under each form and still earn 10% on the capital it then
needs. That saving is the economic reason for variable commissions: the
reinsurer can pass part of it back to the cedant. The size of the saving
depends on the loss-ratio distribution, and a model with more parameter and
reserve risk than this one would show a different spread.

## 6. How price follows the risk profile

A primary rating plan encodes a risk profile in rating variables. A reinsurer
has no policy-level rating plan. The profile enters through the **loss model**,
and the price follows from it.

### Where the loss model comes from

This guide fixes the model by assumption (4 large losses a year, alpha 1.8). In
practice a reinsurer builds it in two ways and blends them:

- **Experience rating**: take the cedant's large-loss history, trend each loss
  to the treaty year, develop it to ultimate, and fit a frequency above a
  threshold and a severity tail. This is the step that uses the cedant's loss
  triangles and the trend work a primary actuary already knows.
- **Exposure rating**: take the cedant's **risk profile** (its policies grouped
  by sum insured or limit, with the premium in each band). An **exposure curve**
  says what share of each band's expected loss falls above each attachment, so
  a layer's expected loss follows from the profile directly. The library has
  `RiskProfile` and exposure curves such as `Mbbefd` for this.
- **Blending**: weight the two by credibility, more on experience when the
  cedant has many large losses and a stable book.

### The levers

These are the parts of the profile that move a layer's price. The first four
are shown in the table below; the rest are described only.

- **frequency**: how many losses reach the program (shown);
- **inflation and trend**: inflation lifts every loss, but policy limits and
  attachments stay where they are, so a layer's loss rises by more than the
  inflation rate. This leverage is the most important effect for excess of
  loss (shown), and index clauses that move the attachment with inflation exist
  to share it;
- **variability of the count**: a Poisson count understates how much the
  number of large losses varies from year to year when the mean itself is
  uncertain; a negative binomial count is the usual fix (shown);
- **severity tail**: how heavy the tail is, and where policy limits cut it
  (shown). Uncertainty in alpha itself is often the largest driver of a quote;
- **limits and sums-insured mix**: what a single risk can cost, which is what
  exposure rating reads from the profile;
- **catastrophe and clash**: one event that hits many risks, or several of the
  cedant's policies at once, raises the tail of the annual loss without raising
  the typical loss. The definition of an occurrence decides how losses add up;
- **development**: how much reported losses develop to ultimate (late reports
  and growth on known claims). It is large and slow for excess layers, and this
  example ignores it;
- **loss adjustment expenses**: whether claim expenses are added to the loss
  inside the limit, shared pro rata, or paid on top changes what the layer pays;
- **cedant quality**: underwriting and claims handling, which a reinsurer
  judges outside the model and reflects in its parameters and loads.

Here is the same program priced with the same pricing rule from section 4 for
the base book and four variants: 25% more large losses, 5% inflation on every
loss (which, with a Pareto tail, lifts the number of losses above 1 by a factor
of 1.05 to the power 1.8 and leaves their shape alone), a count with twice the
Poisson variance, and a heavier tail (alpha 1.5 instead of 1.8):

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
      f"{1.25 ** parameter(price_of_risk):.3f}")
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

The base row is in money; the others are changes from it. A quarter more
losses raises every layer's expected loss by about a quarter, but the price by
less (17% to 18%). That comes from the pricing rule, not from the book: the
distortion is held fixed, and a proportional hazard distortion with a parameter
below 1 prices a layer roughly in proportion to its frequency raised to that
parameter (1.25 raised to 0.686 is the second number on the last line). The multiple therefore
falls for a worse book, which a reinsurer would not accept; re-pricing a worse
book, it would normally move the price of risk too.

Inflation of 5% raises every layer's expected loss by about 9%, close to 1.05
raised to the power 1.8 (the first number on the last line). With a Pareto tail
and fixed limits that leverage is the same at every layer; with a lighter tail
it grows with the attachment. A more variable count barely moves any layer
here: thinned to the few losses that reach a layer, the extra variance mostly
disappears. It matters more for covers on the year's total, such as a quota
share or a stop loss. A heavier tail moves the
layers apart: expected loss rises by about half in the working layer and more
than doubles in the top layer. That is why "what is the cedant's loss ratio" is
only the first question: a book with the same frequency and a different tail is
a different risk.

## 7. Stand-alone and as part of a book

So far each layer was priced alone. A reinsurer holds a portfolio, and the
capital a contract needs depends on what else is in it. If the contract's bad
years rarely coincide with the rest of the book's, adding it barely raises the
book's tail, and it needs little extra capital. If they coincide, it needs
more.

We add the 5x5 layer to an existing book with expected loss 30 and a
coefficient of variation of 25%, once with the two independent and once with a
correlation of 0.5 between their annual losses. The pairs of annual losses are
simulated with a **Gaussian copula**, a way to make two annual losses move
together with a chosen correlation, and held in a `PredictiveDistribution` (a
table of simulated outcomes, one column per part). `price_portfolio` prices the
whole and **allocates** the price to its parts with the **natural allocation**:
each part gets the share of the total capital that it contributes in the bad
years, so the parts add up exactly to the whole. That differs from the
**marginal** capital (how much more the book needs because of the contract),
which is shown too.

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

The draws are simulated (a fixed seed), so the stand-alone risk premium differs
a little from the grid's 1.240. Diversification is a discount on capital. The
contract is the same, but inside an unrelated book its risk premium is about
half the stand-alone one, and its **loading** (risk premium less expected loss)
falls by far more than half. At a correlation of 0.5 much of the discount is
gone. The marginal assets (how much the book's assets grow) and the allocated
capital are close but not equal: the natural allocation splits the total so
that the parts add up, and marginal figures for several contracts do not. A reinsurer already heavy in the same kind of loss quotes higher than one
that is not. Neither is wrong: the prices reflect different portfolios.

Two cautions. The size of the discount depends on the size and shape of the
existing book: here a book with a mean of 30 swamps a contract that pays
nothing in four years out of five. And a Gaussian copula has no tail
dependence, so even at a correlation of 0.5 it understates how often both
have their worst years together.

## 8. Hard and soft markets

A **hard market** is one where reinsurers charge more for the same cover and
also offer less of it: less limit, higher attachments, fewer free
reinstatements, tighter wording. A **soft market** is the reverse. A price rise
for the same cover and exposure has three parts:

- a higher **expected loss**: loss trend, adverse development on the cedant's
  and the market's reserves, a new view of the tail;
- a higher **price of risk**: the return the market demands on the capital it
  holds, written here as the parameter of the pricing distortion;
- **tighter terms**, which show up as less cover for the money.

Calibrating a quote separates the first part from the second only when the loss
model is trusted. Take the 5x5 layer and two illustrative quotes for it, as
rates on line: a soft market quote of 22% and a hard market quote of 34%, for
the layer with a free reinstatement as priced above. Assume the reinsurer's
model of the expected loss has not moved, so the difference is all price of
risk. The quote is what the cedant pays, so the reinsurer's risk premium is the
quote less expenses and brokerage. We read off what each quote implies: the
return on capital the reinsurer is getting, and the proportional hazard
parameter that reproduces the quote on this layer and prices the other layers
consistently with it.

```python
QUOTES = {"soft": 0.22, "hard": 0.34}   # rate on line for the 5x5 layer

market = {}
for name, rol in QUOTES.items():
    risk = rol * mid.limit * KEEP
    implied = Pentagon.solve(loss=expected_loss, assets=assets, premium=risk)
    distortion = calibrate("ph", annual, risk)
    market[name] = distortion
    multiples = [gross_up(p.premium) / p.loss for p in tower_grids.price(distortion).ceded]
    print(f"{name}: quote {rol * mid.limit:.2f} (multiple {rol * mid.limit / expected_loss:.2f}), "
          f"return on capital {implied.return_on_capital:.1%}, hazard parameter {parameter(distortion):.3f}")
    print(f"      multiples by layer {', '.join(f'{m:.2f}' for m in multiples)}")
print(f"technical premium {technical:.2f} (rate on line {technical / mid.limit:.1%}) at a {RETURN:.0%} return")
```

```text
soft: quote 1.10 (multiple 1.88), return on capital 5.4%, hazard parameter 0.791
      multiples by layer 1.57, 1.88, 2.37
hard: quote 1.70 (multiple 2.90), return on capital 14.2%, hazard parameter 0.616
      multiples by layer 2.07, 2.90, 4.42
technical premium 1.43 (rate on line 28.5%) at a 10% return
```

Same expected loss, same layer: the soft quote gives the reinsurer a return
about half its 10% target, the hard quote a return well over it, and the
technical premium sits between them. A lower hazard parameter means a higher
price of risk. In this model the hard market is dearer on every layer, and
dearer by more on the top layer. That pattern comes from moving the single
parameter of the proportional hazard family; a cost-of-capital rule or another
distortion would spread the increase differently. In practice, casualty
hardening often hits working layers hardest (loss-driven pricing, higher
attachments), and catastrophe hardening the upper layers. Calibrate to quotes on
more than one layer when they exist.

### Tighter terms

"Harder to get" shows up as terms. The cedant can also answer a price with
different terms. Under the hard market distortion, here is the same layer with
a higher attachment, and with no reinstatement:

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

A higher attachment cuts expected loss by 37% and the quote by 26%: the layer
is cheaper, but each unit of expected loss is dearer, because what is left is
more remote. Dropping the reinstatement cuts expected loss by only 5% and the
quote by 14%, because it halves the most the layer can pay in a year, and so the
capital. A buyer who trades terms for price is giving up very different amounts
of expected loss and of capital, and the quote follows the capital.

What moves the price of risk?

- **capital supply**: after large losses (catastrophes in property, reserve
  strengthening in casualty) reinsurers' capital is lower and investors want
  more for supplying it;
- **capital requirements**: rating agencies and regulators set how much capital
  a reinsurer must hold for a given risk;
- **interest rates**: higher rates raise the return investors require on
  capital, which pushes prices up, but also raise what the reinsurer earns on
  premium and reserves until claims are paid, which allows a lower underwriting
  margin. For long-tail lines the second effect is usually the larger. This
  guide's model has no investment income, so it cannot show either;
- **retrocession cost**: when the reinsurers' own reinsurance costs more, they
  pass it on.

And what moves the expected loss: loss trend, the cedant's own loss experience
and reserve development, and new views of the tail. These show up in the
technical price whatever the market does.

## 9. What the reinsurer decides

There are two technical prices, and the decision needs both:

- the **stand-alone technical price** (section 3): the reference price for a
  reinsurer with no other business, or for a contract with no diversification
  benefit;
- the **technical price in a book** (section 7): what the contract costs given
  the portfolio the reinsurer already holds, grossed up for expenses and
  brokerage like the stand-alone one.

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

Each row is the quote's distance from each technical price. The soft quote is
below the stand-alone price, comfortably above the price in an unrelated book,
and only just above the price in a correlated one. The hard quote clears all
three. A reinsurer decides, for each submission:

- **write** the share it is offered, if the quote is at or above the technical
  price for its own portfolio;
- **write less** at the same terms only if capital is not linear in the share:
  a concentration or accumulation limit, or a share large enough to move the
  portfolio's tail. Otherwise a smaller share of an underpriced contract is just
  a smaller underpriced position;
- **change the structure** (a higher attachment, a lower limit, no free
  reinstatement) if a different set of terms meets the price;
- **decline** otherwise.

## 10. What the model leaves out

The example is deliberately small. Each of these would move a real price, and
the direction is given:

- **Development**: reported excess losses grow to ultimate slowly; ignoring it
  understates expected loss, more so on higher layers.
- **Loss adjustment expenses**: adding claim expenses to the loss raises what
  the layer pays.
- **Payout, discounting and investment income**: the guide is undiscounted and
  prices one year of capital. For a long-tail line such as liability, the
  reinsurer holds capital and reserves for the whole payout (often many years),
  which raises the cost of capital, and earns investment income on them, which
  lowers the margin needed. Liability layers usually carry lower multiples than
  this property example shows.
- **Parameter uncertainty**: alpha and the frequency are estimated, not known;
  allowing for that fattens the tail and raises the price.
- **Data and credibility loads**: a thin or unreliable submission gets a load.
- **Clash and catastrophe aggregation**: several of the cedant's risks hit by
  one event add up under the occurrence definition; ignoring it understates the
  tail.
- **Credit risk**: funds withheld by the cedant, collateral and the reinsurer's
  own credit standing change what each side will accept.
- **Expenses and tax**: fixed costs per contract and tax both raise the price
  of small contracts above a percentage loading.

## 11. Glossary and further reading

**Annual aggregate deductible**: the part of the year's layer losses the cedant
keeps before the layer pays. **Annual aggregate limit**: the most a layer pays
in a year. **Assets**: premium plus capital. **Attachment**: the loss size above
which an excess of loss layer pays. **Attritional losses**: the frequent small
losses below the large-loss threshold. **Capital**: the reinsurer's own money
held behind a contract. **Cedant**: the insurer that buys reinsurance. **Ceding
commission**: what the reinsurer pays the cedant on a quota share. **Cost of
capital**: the return the reinsurer asks on its capital. **Deposit premium**:
the premium paid at inception, before any reinstatement premium.
**Distortion**: a reweighting of probabilities that prices a loss by its tail.
**Exhaustion**: a loss that uses up a layer's whole limit. **Hard market**:
prices up and cover scarcer for the same risk. **Layer**: the slice of each
loss one contract pays, written limit x attachment. **Loss on line**: expected
loss divided by limit. **Multiple**: premium divided by expected loss.
**Natural allocation**: a split of a portfolio's capital and price among its
parts that adds up exactly. **Payback**: limit divided by premium, the inverse
of the rate on line. **Pentagon**: loss, premium, margin, capital and assets,
linked by two identities. **Price of risk**: the loading the market puts on
risk, as a return on capital or a distortion parameter. **Profit commission**:
a share of the reinsurer's profit returned to the cedant. **Program**: all the
layers a cedant buys together. **Quota share**: reinsurance of a fixed share of
every policy. **Rate on line (ROL)**: premium divided by limit.
**Reinstatement**: restoring a layer's limit after a loss, free or for a
**reinstatement premium**. **Retrocession**: reinsurance bought by a reinsurer.
**Risk premium**: expected loss plus margin, before expenses and brokerage.
**Sliding scale**: a ceding commission that falls as the loss ratio rises.
**Soft market**: prices down and cover plentiful. **Stand-alone price**: a
contract's price with no diversification benefit. **Subject premium**: the
cedant's premium that the reinsurance applies to. **Technical commission**: the
ceding commission the reinsurer's model supports. **Technical premium**: the
price the reinsurer's model supports, after expenses and brokerage. **Treaty**:
a reinsurance contract covering a whole book. **TVaR**: the average of the
worst outcomes beyond a probability level.

Further reading, by question:

- Clark, "Basics of Reinsurance Pricing", CAS Study Note: start here for
  experience and exposure rating of layers and increased limits.
- Werner and Modlin, *Basic Ratemaking* (CAS): the trend and development work
  you already know, which experience rating of a layer rests on.
- Mildenhall and Major, *Pricing Insurance Risk: Theory and Practice* (Wiley,
  2022): the pentagon, distortions and the natural allocation used in sections
  2, 4 and 7.
- The notebook [`examples/reinsurance_pricing.ipynb`](../../examples/reinsurance_pricing.ipynb)
  prices six contracts and checks them against Mildenhall's `aggregate`.

### About these numbers

Every Python block in this guide runs in CI (`python/tests/test_guides.py`),
and every printed result under a block is checked against the library, so the
numbers cannot go stale.
