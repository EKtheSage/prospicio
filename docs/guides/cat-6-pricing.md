# Property catastrophe 6: a price for Alder's first layer

<!-- site: execute -->

This page prices the ceded losses of one contract. The story, the files and
the notation are on [the overview](cat-0-overview.md). The contract is
`alder_cat_1`, the 30% line of Larkspur on the first cat layer of Alder
Mutual: 40 xs 30 with one reinstatement at 100%. Its terms come from
`examples/data/cat_book_contracts.csv`.

A price has three parts: the expected loss, a risk load
([risk loads and distortions](risk-loads-and-distortions.md)) for the capital
behind the bad years, and expenses with brokerage.

## 1. The contract's ceded losses

The helper `cedent_events` is the same on every page of the track. It reads the
YELT for one cedent and keeps the time in the year and the peril of each loss.
The upfront
premium of Larkspur is the rate on line times the limit times its share.

```python
import csv

from prospicio.aggregate import EventSet
from prospicio.distributions import Sampled
from prospicio.pricing import Quote, price, price_with_reinstatements
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion

DATA = "../../examples/data/"


def read(name):
    with open(DATA + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


YELT = read("cat_book_yelt.csv")
CONTRACTS = {r["contract"]: r for r in read("cat_book_contracts.csv")}
N_YEARS = 2000


def cedent_events(cedent, perils="all"):
    """One cedent's years of losses, with times (fraction of the year) and peril labels."""
    keep = None if perils == "all" else set(perils.split("+"))
    years = [[] for _ in range(N_YEARS)]
    times = [[] for _ in range(N_YEARS)]
    peril = []
    for r in YELT:
        if r["cedent"] != cedent or (keep and r["peril"] not in keep):
            continue
        y = int(r["year"]) - 1
        years[y].append(float(r["loss"]))
        times[y].append(((int(r["day"]) - 1) * 24 + int(r["hour"])) / 8760)
        peril.append(r["peril"])
    return EventSet.from_years(years, times=times).with_labels({"peril": peril})


def xl_layer(c, pro_rata_time=False):
    """An occurrence excess of loss layer from a row of the contracts file."""
    limit, attachment, share = float(c["limit"]), float(c["attachment"]), float(c["share"])
    return Layer(c["contract"], limit, attachment, share=share,
                 premium=float(c["rate_on_line"]) * limit * share,
                 reinstatement_rates=[float(c["reinstatement_rate"])] * int(c["reinstatements"]),
                 pro_rata_time=pro_rata_time)


def ceded(layer, events):
    """The layer's annual ceded loss and reinstatement premium, one draw a year."""
    res = Tower([layer]).apply(events).aggregate(["kind", "layer"])
    return res.marginal(["ceded", layer.name]), res.marginal(["reinstatement_premium", layer.name])


c = CONTRACTS["alder_cat_1"]
raw = cedent_events("alder")
grouped = raw.grouped(float(c["hours"]))
layer = xl_layer(c)
loss, rip = ceded(layer, grouped)
X = Sampled(loss.draws)
print(f"limit at Larkspur's share {layer.limit * layer.share:.1f}, upfront premium {layer.premium:.4f}")
print(f"expected ceded loss {X.mean():.4f}, loss ratio {X.mean() / layer.premium:.3f}")
print(f"years with a ceded loss {sum(x > 0 for x in loss.draws)} of {N_YEARS}")
```

```text
limit at Larkspur's share 12.0, upfront premium 1.8000
expected ceded loss 0.8230, loss ratio 0.457
years with a ceded loss 230 of 2000
```

The loss ratio matches page 3, section 6. Most years cost the layer nothing,
and a few cost it the whole line. Thus the risk load matters more than the
expected loss.

## 2. Cost of capital

Larkspur holds assets `A` against the bad years. The premium `P` is part of
the assets, and the capital is `K = A - P`
([a first course](a-first-course.md),
[what a reinsurance price is](reinsurance-price.md)). With a cost of capital
`r`, the margin `P - EL` must equal `r K`. Solve for `P`:

```
P = (EL + r A) / (1 + r),    K = A - P
```

The assets are the TVaR at 99% of the ceded loss: the mean of the worst 1% of
years. A common shortcut is `EL + r (TVaR - EL)`. It charges for capital as if
the premium does not help pay the tail. `price` counts the premium as part of
the assets, so the capital and the price are both a little lower.

```python
import pandas as pd
from great_tables import GT

p = price(X, Distortion.tvar(0.99), cost_of_capital=0.10)
cost = pd.DataFrame({
    "item": ["expected loss", "assets (TVaR 99)", "premium", "capital", "margin",
             "return on capital", "premium by the formula",
             "shortcut EL + r (TVaR - EL)"],
    "value": [p.expected_loss, p.assets, p.premium, p.capital, p.margin,
              p.return_on_capital, (p.expected_loss + 0.10 * p.assets) / 1.10,
              p.expected_loss + 0.10 * (p.assets - p.expected_loss)],
})
(GT(cost, rowname_col="item")
 .tab_header("Alder 40 xs 30: the cost of capital price at r = 10%",
             "At Larkspur's share")
 .fmt_number("value", decimals=4))
```

```text
                               value
expected loss                 0.8230
assets (TVaR 99)             13.8428
premium                       2.0066
capital                      11.8362
margin                        1.1836
return on capital             0.1000
premium by the formula        2.0066
shortcut EL + r (TVaR - EL)   2.1250
```

## 3. A distortion price

A pricing distortion gives more weight to the years in the tail. The premium is
the expected loss under the new weights. [Risk loads and
distortions](risk-loads-and-distortions.md) compares the families, and
[portfolio pricing](pir-portfolio-pricing.md) gives the theory. Here the
distortion is the Wang transform with a parameter of 0.3.

```python
pw = price(X, Distortion.tvar(0.99), distortion=Distortion.wang(0.3))
rules = pd.DataFrame({
    "rule": ["cost of capital, TVaR 99", "Wang 0.3"],
    "premium": [p.premium, pw.premium],
    "capital": [p.capital, pw.capital],
    "return on capital": [p.return_on_capital, pw.return_on_capital],
})
(GT(rules, rowname_col="rule")
 .tab_header("Alder 40 xs 30: two pricing rules", "At Larkspur's share")
 .fmt_number(["premium", "capital", "return on capital"], decimals=4))
```

```text
                          premium  capital  return on capital
cost of capital, TVaR 99   2.0066  11.8362             0.1000
Wang 0.3                   1.4225  12.4204             0.0483
```

The Wang price is lower and implies under 5% on the TVaR capital. The two
rules answer different questions: use the one that matches your capital model,
and the other as a check.

The figure shows the years with a ceded loss, and where each price falls.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist([x for x in loss.draws if x > 0], bins=24, color="0.75", label="years with a ceded loss")
ax.axvline(p.expected_loss, color="C0", label="expected loss")
ax.axvline(pw.premium, color="C1", label="Wang 0.3 premium")
ax.axvline(p.premium, color="C2", label="cost of capital premium")
ax.set_xlabel("annual ceded loss at Larkspur's share")
ax.set_ylabel("years of 2,000")
ax.set_title("Alder 40 xs 30: the 230 years with a loss, and three prices")
ax.legend()
```

## 4. From risk premium to a quote

The premium above is the risk premium. With expenses `e` and brokerage `b` as
fractions of the quoted premium `Q`, `Q = P / (1 - e - b)`
([what a reinsurance price is](reinsurance-price.md), "The technical
premium"). `Quote.technical` does this and reports the rate on line at the
share of Larkspur.

```python
q = Quote.technical(p, expenses=0.05, brokerage=0.10, limit=layer.limit * layer.share)
print(f"technical premium {q.premium:.4f}, rate on line {q.rate_on_line:.4f}")
```

```text
technical premium 2.3607, rate on line 0.1967
```

The technical rate on line is above the 15% in the contract. Section 7 compares
it with the market price.

## 5. The reinstatement premium inside the price

When the layer pays a loss, the cedent buys the limit back: with one
reinstatement at 100%, it pays the upfront premium again in proportion to the
limit used. Thus the reinsurer collects more exactly when it pays claims.

**Expected value.** The reinstatement premium of a year is the upfront premium
times a factor that depends only on the losses (the share of the limit used,
times the rate). Its expected value is `f P` for a fixed `f`. If the risk
premium `R` must come from the upfront premium and the expected reinstatement
premium together, then `P (1 + f) = R`, so `P = R / (1 + f)`.

```python
f = rip.mean() / layer.premium
print(f"expected reinstatement premium {rip.mean():.4f}, f = {f:.5f}")
print(f"upfront premium for a risk premium of {p.premium:.4f}: {p.premium / (1 + f):.4f}")
```

```text
expected reinstatement premium 0.1207, f = 0.06705
upfront premium for a risk premium of 2.0066: 1.8805
```

**In the tail.** Under TVaR, the expected value is not sufficient. The
reinstatement premium of year `t` is `P g_t`, where `g_t` depends only on that
year's losses. It comes in the years with losses, so it lowers the tail of the
net outcome `X_t - P g_t`. The assets then depend on `P`, and the upfront
premium solves `P = price(X - P g)`, one equation in one unknown (Sundt 1991,
Mata 2000).

`price_with_reinstatements` finds the root (#587). The price of the net outcome
falls as `P` increases, so `price(X - P g) - P` changes sign once between 0 and
the price without reinstatement premium. The call finds that root by bisection.

```python
g = [b / layer.premium for b in rip.draws]  # reinstatement premium per unit of upfront premium
ri = price_with_reinstatements(loss.draws, g, Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"upfront premium {ri.upfront:.4f}, assets of the net outcome {ri.net.assets:.4f}")
print(f"quoted with 5% expenses and 10% brokerage {ri.upfront / (1 - 0.05 - 0.10):.4f}")
```

```text
upfront premium 1.7421, assets of the net outcome 12.1008
quoted with 5% expenses and 10% brokerage 2.0495
```

Counting the reinstatement premium in the tail lowers the upfront premium more
than the expected value alone does: it saves capital in the worst years.

## 6. Sensitivity

This section varies three inputs: the cost of capital, the hours clause and
the timing of the season.

**Cost of capital.** The assets do not change with `r`, but the premium is
concave in `r`: `P = (EL + r A) / (1 + r)` increases by less with each step.

```python
rates = [0.05, 0.10, 0.15, 0.20]
by_rate = pd.DataFrame({
    "r": [f"{r:.2f}" for r in rates],
    "premium": [price(X, Distortion.tvar(0.99), cost_of_capital=r).premium for r in rates],
})
(GT(by_rate, rowname_col="r")
 .tab_header("Alder 40 xs 30: the premium by cost of capital", "At Larkspur's share")
 .fmt_number("premium", decimals=4))
```

```text
      premium
0.05   1.4430
0.10   2.0066
0.15   2.5212
0.20   2.9930
```

```python
rs = [i / 100 for i in range(2, 26)]
fig, ax = plt.subplots()
ax.plot(rs, [(p.expected_loss + r * p.assets) / (1 + r) for r in rs], label="price (premium helps pay the tail)")
ax.plot(rs, [p.expected_loss + r * (p.assets - p.expected_loss) for r in rs], label="shortcut EL + r (TVaR - EL)")
ax.axhline(layer.premium, color="0.75", linestyle=":", label="contract premium")
ax.set_xlabel("cost of capital r")
ax.set_ylabel("premium at Larkspur's share")
ax.set_title("The premium rises with r, and less than the shortcut says")
ax.legend()
```

**Hours clause.** Without the 72 hour clause, the pieces of a split windstorm
or convective storm are separate occurrences, each with its own retention.

```python
loss_raw, _ = ceded(layer, raw)
prices = {name: price(Sampled(ls.draws), Distortion.tvar(0.99), cost_of_capital=0.10)
          for name, ls in (("72 hours", loss), ("no clause", loss_raw))}
clause = pd.DataFrame({
    "clause": list(prices),
    "expected loss": [pr.expected_loss for pr in prices.values()],
    "assets": [pr.assets for pr in prices.values()],
    "premium": [pr.premium for pr in prices.values()],
})
(GT(clause, rowname_col="clause")
 .tab_header("Alder 40 xs 30: with and without the hours clause", "At Larkspur's share")
 .fmt_number(["expected loss", "assets", "premium"], decimals=4))
```

```text
           expected loss   assets  premium
72 hours          0.8230  13.8428   2.0066
no clause         0.7778  15.4227   2.1092
```

The clause raises the expected loss, because the pieces of a storm add up
before the retention applies. It lowers the tail: without it, a storm in two
large pieces can take the limit twice in one year (limit and reinstatement),
and grouped it takes one limit. On this book the tail effect is larger, so the
price without the clause is higher.

**Season.** A reinstatement pro rata as to time charges only for the part of
the year left. Most windstorms of Alder (nine in ten in the generator) arrive
between day 213 and day 305, so the cedent pays much less to reinstate and `f`
falls.

```python
_, rip_time = ceded(xl_layer(c, pro_rata_time=True), grouped)
f_time = rip_time.mean() / layer.premium
season = pd.DataFrame({
    "reinstatement": ["pro rata as to amount", "pro rata as to time"],
    "f": [f, f_time],
    "upfront premium": [p.premium / (1 + f), p.premium / (1 + f_time)],
})
(GT(season, rowname_col="reinstatement")
 .tab_header("Alder 40 xs 30: the reinstatement terms",
             "Upfront premium by expected value, at Larkspur's share")
 .fmt_number("f", decimals=5)
 .fmt_number("upfront premium", decimals=4))
```

```text
                             f  upfront premium
pro rata as to amount  0.06705           1.8805
pro rata as to time    0.02484           1.9580
```

## 7. Is the market price adequate?

The rate on line of the contract is 15%, a premium of 1.8 at the share of
Larkspur. `Quote.at` reads a quoted premium backwards: it takes off expenses
and brokerage, and reports the margin and the return on the same capital.

```python
m = q.at(layer.premium)
build_up = pd.DataFrame({
    "item": ["risk premium", "expenses", "brokerage", "quoted premium", "rate on line"],
    "technical": [q.risk_premium, q.expenses, q.brokerage, q.premium, q.rate_on_line],
    "market": [m.risk_premium, m.expenses, m.brokerage, m.premium, m.rate_on_line],
})
(GT(build_up, rowname_col="item")
 .tab_header("Alder 40 xs 30: the technical quote and the market quote",
             "Premium at Larkspur's share; the rate on line is premium over limit")
 .tab_spanner("quote", ["technical", "market"])
 .fmt_number(["technical", "market"], decimals=4))
```

```text
                technical  market
risk premium       2.0066  1.5300
expenses           0.1180  0.0900
brokerage          0.2361  0.1800
quoted premium     2.3607  1.8000
rate on line       0.1967  0.1500
```

```python
print(f"loss ratio {m.loss_ratio:.3f}, margin {m.margin:.4f}")
print(f"return on capital {m.return_on_capital:.4f} against a target of 0.10")
```

```text
loss ratio 0.457, margin 0.7070
return on capital 0.0574 against a target of 0.10
```

At the market price, `alder_cat_1` earns much less than the 10% target on its
stand-alone capital. The reinstatement premium does not close the gap: for the
same risk premium, the upfront premium of section 5 is 1.7421, and 2.0495 with
expenses and brokerage, still above the 1.8 charged. What remains is the
diversification in the book of Larkspur, which page 7 measures.

## In R

The R twin, [cat-6-pricing.qmd](cat-6-pricing.qmd), prices the layer with
`risk_loaded_price()`, `distortion()`, `technical_quote()` and
`implied_quote()`.

## Further reading

- Sundt, B. (1991), "On excess of loss reinsurance with reinstatements",
  *Bulletin of the Swiss Association of Actuaries*, 51-65.
- Mata, A. J. (2000), "Pricing excess of loss reinsurance with
  reinstatements", *ASTIN Bulletin* 30(2), 349-368.
- [Risk loads and distortions](risk-loads-and-distortions.md) and
  [What a reinsurance price is](reinsurance-price.md) explain the pricing
  rules in general.
- [reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is another
  introduction to property catastrophe reinsurance.
