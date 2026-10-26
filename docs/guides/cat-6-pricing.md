# Property catastrophe 6: a price for Alder's first layer

The earlier pages of this track made annual ceded losses from the year event
loss table of Larkspur Re. This page makes a price from the ceded losses of one
contract. The story, the files and the notation are on
[the overview](cat-0-overview.md). The contract is `alder_cat_1`, the 30%
line of Larkspur on the first cat layer of Alder Mutual: 40 xs 30 with one
reinstatement at 100%. Its terms come from
`examples/data/cat_book_contracts.csv`, as on every page.

A price has three parts. The expected loss pays the claims on average. A risk
load, the margin above the expected loss
([risk loads and distortions](risk-loads-and-distortions.md)), pays for the
capital behind the bad years. The price adds expenses and brokerage to these
two parts. The sections below build each part. Then they show what the
reinstatement premium does to the price, how sensitive the price is, and
whether the market rate on line is sufficient.

## 1. The contract's ceded losses

The helper `cedent_events` is the same on every page of the track. It reads
the YELT for one cedent and keeps the time in the year and the peril of each
loss. The code builds the layer from the row of the contract. The upfront
premium of Larkspur is the rate on line times the limit times its share.

```python
import csv

from prospicio.aggregate import EventSet
from prospicio.distributions import Sampled
from prospicio.pricing import Quote, price
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

The loss ratio matches the table in page 3, section 6. Most years cost the
layer nothing, and a few cost it the whole line. Because of this shape, the
risk load matters more than the expected loss for a cat layer.

## 2. Cost of capital

Larkspur holds assets `A` against the bad years of the layer. The premium `P`
is part of the assets, and the capital `K = A - P` is the rest. [A first course](a-first-course.md)
and [what a reinsurance price is](reinsurance-price.md) explain the idea. With a cost of
capital `r`, the margin `P - EL` must equal `r K`. Solve for `P`:

```
P = (EL + r A) / (1 + r),    K = A - P
```

Here the assets are the TVaR at 99% of the ceded loss: the mean of the worst 1%
of years. Note the convention. A common shortcut is `EL + r (TVaR - EL)`, which
charges for capital as if the premium does not help pay the tail. The `price`
function of prospicio counts the premium as part of the assets. Thus the
capital and the price are both a little lower.

```python
p = price(X, Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"expected loss {p.expected_loss:.4f}")
print(f"assets (TVaR 99) {p.assets:.4f}")
print(f"premium {p.premium:.4f}, capital {p.capital:.4f}")
print(f"margin {p.margin:.4f}, return on capital {p.return_on_capital:.4f}")
print(f"by the formula {(p.expected_loss + 0.10 * p.assets) / 1.10:.4f}")
print(f"shortcut EL + r (TVaR - EL) {p.expected_loss + 0.10 * (p.assets - p.expected_loss):.4f}")
```

```text
expected loss 0.8230
assets (TVaR 99) 13.8428
premium 2.0066, capital 11.8362
margin 1.1836, return on capital 0.1000
by the formula 2.0066
shortcut EL + r (TVaR - EL) 2.1250
```

## 3. A distortion price

A pricing distortion gives more weight to the years in the tail. The premium is
the expected loss under the new weights, and the return on the same capital
follows from the price.
[Risk loads and distortions](risk-loads-and-distortions.md) compares the
families, and [portfolio pricing](pir-portfolio-pricing.md) gives the theory. This section
uses the Wang transform with a parameter of 0.3.

```python
pw = price(X, Distortion.tvar(0.99), distortion=Distortion.wang(0.3))
print(f"Wang 0.3 premium {pw.premium:.4f}, capital {pw.capital:.4f}")
print(f"return on capital {pw.return_on_capital:.4f}")
```

```text
Wang 0.3 premium 1.4225, capital 12.4204
return on capital 0.0483
```

The Wang price is lower, and the return that it implies on the TVaR capital is
under 5%. The two rules answer different questions. Use the rule that matches
the calibration of your capital model, and use the other as a check.

## 4. From risk premium to a quote

The premium above is the risk premium. With expenses `e` and brokerage `b` as
fractions of the quoted premium `Q`, `Q = P / (1 - e - b)`
([what a reinsurance price is](reinsurance-price.md), "The technical
premium"). `Quote.technical` does this calculation. It also reports the rate on
line against the limit at the share of Larkspur.

```python
q = Quote.technical(p, expenses=0.05, brokerage=0.10, limit=layer.limit * layer.share)
print(f"technical premium {q.premium:.4f}, rate on line {q.rate_on_line:.4f}")
print(f"expenses {q.expenses:.4f}, brokerage {q.brokerage:.4f}, risk premium {q.risk_premium:.4f}")
print(f"contract rate on line {float(c['rate_on_line']):.2f}, premium {layer.premium:.4f}")
```

```text
technical premium 2.3607, rate on line 0.1967
expenses 0.1180, brokerage 0.2361, risk premium 2.0066
contract rate on line 0.15, premium 1.8000
```

The technical rate on line is above the 15% in the contract. Section 7 shows
what the price of the contract earns.

## 5. The reinstatement premium inside the price

When the layer pays a loss, the cedent buys the limit back. With one
reinstatement at 100%, the cedent pays the upfront premium again in proportion
to the limit used. Thus the reinsurer collects more than the upfront premium in
exactly the years when it pays claims.

**Expected value.** The reinstatement premium of each year is the upfront
premium times a factor that depends only on the losses. The factor is the share
of the limit used, times the rate. Thus its expected value is `f P` for a fixed
`f`. Suppose that the risk premium `R` must come from the upfront premium and
the expected reinstatement premium together. Then `P (1 + f) = R`, and
`P = R / (1 + f)` in closed form.

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
reinstatement premium of year `t` is `P g_t`, where `g_t` depends only on the
losses of that year. It comes in the years with losses, so it also lowers the
tail of the net outcome `X_t - P g_t`. The assets then depend on `P`, and the
upfront premium solves `P = price(X - P g)`, one equation in one unknown.
Sundt (1991) and Mata (2000) study premiums with reinstatements in this way.

The block below finds the root by bisection (#587). The price of the net
outcome falls as `P` increases. Thus `price(X - P g) - P` changes sign once
between 0 and the price without reinstatement premium.

```python
g = [b / layer.premium for b in rip.draws]  # reinstatement premium per unit of upfront premium


def gap(P):
    net = Sampled([x - P * gi for x, gi in zip(loss.draws, g)])
    return price(net, Distortion.tvar(0.99), cost_of_capital=0.10).premium - P


lo, hi = 0.0, p.premium
while hi - lo > 1e-10:
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
P = (lo + hi) / 2
p_net = price(Sampled([x - P * gi for x, gi in zip(loss.draws, g)]), Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"upfront premium {P:.4f}, assets of the net outcome {p_net.assets:.4f}")
print(f"quoted with 5% expenses and 10% brokerage {P / (1 - 0.05 - 0.10):.4f}")
```

```text
upfront premium 1.7421, assets of the net outcome 12.1008
quoted with 5% expenses and 10% brokerage 2.0495
```

Counting the reinstatement premium in the tail lowers the upfront premium more
than the expected value alone does. The difference is the capital that the
reinstatement premium saves in the worst years.

## 6. Sensitivity

A price is only as firm as its inputs. This section loops over three of them:
the cost of capital, the hours clause and the timing of the season.

**Cost of capital.** The assets do not change with `r`, but the premium is
concave in `r`. `P = (EL + r A) / (1 + r)` increases by less with each step.

```python
for r in (0.05, 0.10, 0.15, 0.20):
    print(f"r = {r:.2f}: premium {price(X, Distortion.tvar(0.99), cost_of_capital=r).premium:.4f}")
```

```text
r = 0.05: premium 1.4430
r = 0.10: premium 2.0066
r = 0.15: premium 2.5212
r = 0.20: premium 2.9930
```

**Hours clause.** Without the 72 hour clause, the pieces of a split windstorm
or convective storm are separate occurrences. Each piece has its own retention.

```python
loss_raw, _ = ceded(layer, raw)
for name, ls in (("72 hours", loss), ("no clause", loss_raw)):
    pr = price(Sampled(ls.draws), Distortion.tvar(0.99), cost_of_capital=0.10)
    print(f"{name:>9}: expected loss {pr.expected_loss:.4f}, assets {pr.assets:.4f}, premium {pr.premium:.4f}")
```

```text
 72 hours: expected loss 0.8230, assets 13.8428, premium 2.0066
no clause: expected loss 0.7778, assets 15.4227, premium 2.1092
```

The clause increases the expected loss, because the pieces of a storm add up
before the retention applies. It decreases the tail. Without the clause, a
storm in two large pieces can take the full limit two times in one year. It
takes it once from the limit and once from the reinstatement. Grouped, the same
storm is one occurrence and takes one limit. On this book the tail effect is
larger, and the price without the clause is higher.

**Season.** A reinstatement pro rata as to time charges only for the part of
the year left. Most windstorms of Alder (nine in ten in the generator) arrive
between day 213 and day 305. Thus the cedent pays much less to reinstate, and
`f` falls.

```python
_, rip_time = ceded(xl_layer(c, pro_rata_time=True), grouped)
f_time = rip_time.mean() / layer.premium
print(f"f pro rata as to amount {f:.5f}, as to time {f_time:.5f}")
print(f"upfront premium (expected value) {p.premium / (1 + f):.4f} against {p.premium / (1 + f_time):.4f}")
```

```text
f pro rata as to amount 0.06705, as to time 0.02484
upfront premium (expected value) 1.8805 against 1.9580
```

## 7. Is the market price adequate?

The rate on line of the contract is 15%, a premium of 1.8 at the share of
Larkspur. `Quote.at` reads a quoted premium backwards. It takes off expenses
and brokerage, and it reports the margin and the return on the same capital.

```python
m = q.at(layer.premium)
print(f"quoted premium {m.premium:.4f}, risk premium {m.risk_premium:.4f}")
print(f"loss ratio {m.loss_ratio:.3f}, margin {m.margin:.4f}")
print(f"return on capital {m.return_on_capital:.4f} against a target of 0.10")
```

```text
quoted premium 1.8000, risk premium 1.5300
loss ratio 0.457, margin 0.7070
return on capital 0.0574 against a target of 0.10
```

At the market price, `alder_cat_1` earns much less than the 10% target on its
stand-alone capital. The reinstatement premium does not close the gap. With the
reinstatement premium in the tail, the upfront premium of section 5, 1.7421, is
a risk premium. With expenses and brokerage it is 2.0495, still above the 1.8
charged. What remains is the diversification in the book of Larkspur, which
page 7 measures.

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
