# Property catastrophe 5: the money around the loss

A treaty moves more money than its claims. The cedent pays a premium, may pay
again to restore a used limit, and gets some of the premium back as
commission. A broker takes a fee. Collateral sits in a trust and earns
interest. Each of these flows changes what the reinsurer keeps, and some of
them depend on the losses in a way that changes the shape of the result, not
only its mean.

This page follows Larkspur Re's book from the earlier pages of the track and
prices each flow on the same 2,000 simulated years. It covers:

1. Reinstatement premium, as to amount and as to time.
2. Ceding commission, profit commission, sliding scale and brokerage on
   Corran's quota share, read as deal ratios.
3. A no-claims bonus, built from a sliding scale.
4. The reinsurer's P&L ledger of Bracken's aggregate cover.

Money is in millions of Varland crowns. The story, the files and the notation
are on [the overview](cat-0-overview.md).

## 1. Words

[What a reinsurance price is](reinsurance-price.md) explains reinstatements,
commissions and sliding scales for any layer. The short list below fixes the
words; the rest of the page keeps to what is particular to cat business: the
season, and the shape of a cat loss distribution.

- **Reinstatement premium**: what the cedent pays to restore limit that a loss
  has used, so the layer can pay again in the same year. It is set as a rate
  of the upfront premium for each full limit restored.
- **Pro rata as to amount**: the cedent pays for the share of the limit
  restored. **Pro rata as to time** also scales each restoration by the share
  of the year still to run when the loss happens.
- **Ceding commission**: a share of the premium the reinsurer pays back to the
  cedent for its acquisition and handling costs. A **sliding scale** makes
  that share depend on the year's loss ratio.
- **Profit commission**: a share of the reinsurer's profit on the treaty,
  after an allowance for its own costs and margin, paid back to the cedent.
- **Brokerage**: the broker's fee, a share of the premium, deducted from
  what the reinsurer receives.
- **Deal ratios**: each amount over the premium, year by year: the loss ratio,
  each deduction, the combined ratio and the margin left.

## 2. The book

The helper below rebuilds one cedent's years from the YELT with times and
peril labels, turns a row of the contracts file into a `Layer`, and applies
the hours clause where the contract has one. It is the same helper as on the
other pages, with one addition: `contract_layer` passes extra settings (here
`pro_rata_time`) through to the layer.

```python
import csv
import math

from prospicio.aggregate import EventSet
from prospicio.reinsurance import Layer, Tower

DATA = "../../examples/data/"


def read(name):
    with open(DATA + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


YELT = read("cat_book_yelt.csv")
CONTRACTS = {r["contract"]: r for r in read("cat_book_contracts.csv")}
CEDENTS = {r["cedent"]: r for r in read("cat_book_cedents.csv")}
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


def num(x):
    return float(x) if x != "" else None


def contract_layer(c, **extra):
    sh = num(c["share"])
    if c["kind"] == "occurrence_xl":
        lim = num(c["limit"])
        return Layer(c["contract"], lim, num(c["attachment"]), share=sh,
                     premium=num(c["rate_on_line"]) * lim * sh,
                     reinstatement_rates=[num(c["reinstatement_rate"])] * int(c["reinstatements"]), **extra)
    if c["kind"] == "aggregate_xl":
        al = num(c["aggregate_limit"])
        return Layer(c["contract"], math.inf, 0.0, share=sh,
                     aggregate_deductible=num(c["aggregate_deductible"]), aggregate_limit=al,
                     premium=num(c["rate_on_line"]) * al * sh, **extra)
    return (Layer(c["contract"], num(c["limit"]), 0.0, share=sh, aggregate_limit=num(c["aggregate_limit"]),
                  premium=sh * float(CEDENTS[c["cedent"]]["cat_premium"]), **extra)
            .with_ceding_commission(num(c["ceding_commission"])))


def subject(c):
    e = cedent_events(c["cedent"], c["perils"])
    return e.grouped(float(c["hours"])) if c["hours"] else e


def mean_of(res, kind, name):
    return res.aggregate(["kind", "layer"]).marginal([kind, name]).mean()


for name in ["alder_cat_1", "corran_qs", "bracken_agg"]:
    print(name, CONTRACTS[name]["kind"], "premium", round(contract_layer(CONTRACTS[name]).premium, 4))
```

```text
alder_cat_1 occurrence_xl premium 1.8
corran_qs quota_share premium 3.0
bracken_agg aggregate_xl premium 0.45
```

Each premium is Larkspur's, for its own share: rate on line times limit times
share for the excess covers, and the ceded share of Corran's catastrophe
premium for the quota share.

## 3. Reinstatement premium

**Intuition.** `alder_cat_1` is 40 xs 30 with one reinstatement at 100%. After
a loss uses part of the limit, the cedent pays to put it back. It pays the
same share of the upfront premium as the share of the limit it restores. Only
one limit can be restored, so the layer pays at most 80 a year at 100%.

**Formula.** With `P` the upfront premium, `l` the limit, `r_j` the rate of
reinstatement `j` and `Y` the year's layer loss at 100%:

```
RIP = P * sum over j of r_j * min(max(Y - (j - 1) * l, 0), l) / l
```

for the reinstatements the contract has. As to time, each restored piece is
also multiplied by `1 - t`, where `t` is the share of the year gone when its
loss happens.

**By hand.** Two losses of 45 and 65 put 15 and then 35 into the layer. The
first is restored in full. Of the second, only 25 more can be restored, since
one limit of 40 is all there is. As to amount that is 40/40 of the premium,
1.8. If the losses fall a quarter and three quarters of the way through the
year, as to time gives (15 x 0.75 + 25 x 0.25) / 40 x 1.8 = 0.7875.

```python
c = CONTRACTS["alder_cat_1"]
by_amount = contract_layer(c)
by_time = contract_layer(c, pro_rata_time=True)
print("as to amount:", round(by_amount.reinstatement_premium([45.0, 65.0]), 4))
print("as to time:  ", round(by_time.reinstatement_premium([45.0, 65.0], times=[0.25, 0.75]), 4))
```

```text
as to amount: 1.8
as to time:   0.7875
```

**On the book.** The tower's result carries a `reinstatement_premium`
component next to the ceded loss. Over the 2,000 years:

```python
alder = subject(c)
amount_res = Tower([by_amount]).apply(alder)
time_res = Tower([by_time]).apply(alder)
rp_amount = mean_of(amount_res, "reinstatement_premium", "alder_cat_1")
rp_time = mean_of(time_res, "reinstatement_premium", "alder_cat_1")
print("expected ceded loss:", round(mean_of(amount_res, "ceded", "alder_cat_1"), 4))
print("expected reinstatement premium, as to amount:", round(rp_amount, 4))
print("expected reinstatement premium, as to time:  ", round(rp_time, 4))
print("as to time is lower by", round(1 - rp_time / rp_amount, 3))
print("as to amount, over the upfront premium:", round(rp_amount / by_amount.premium, 5))
```

```text
expected ceded loss: 0.823
expected reinstatement premium, as to amount: 0.1207
expected reinstatement premium, as to time:   0.0447
as to time is lower by 0.629
as to amount, over the upfront premium: 0.06705
```

As to time collects about 63% less. That is not a fault in the model. It is
the season: the table below splits the occurrences that reach the layer by
the quarter of the year they fall in.

```python
hits = [t for i in range(N_YEARS) for x, t in zip(alder.events(i), alder.times(i)) if x > 30.0]
print("occurrences that reach the layer:", len(hits))
print("quarter  occurrences  mean share of the year left")
for k in range(4):
    left = [1 - t for t in hits if k / 4 <= t < (k + 1) / 4]
    print(f"Q{k + 1}       {len(left):11d}  {sum(left) / len(left):27.3f}")
```

```text
occurrences that reach the layer: 241
quarter  occurrences  mean share of the year left
Q1                22                        0.843
Q2                37                        0.622
Q3               114                        0.340
Q4                68                        0.190
```

182 of the 241 occurrences come in the second half of the year, because
Alder's large losses are mostly windstorms and the generator puts nine in ten
windstorms between day 213 and day 305. A loss then restores limit for only
the few months left, and as to time the cedent pays for only those. A peril
spread evenly over the year would show a smaller gap. Choose the basis the
treaty wording uses; for a seasonal peril the difference is material.

Each year's reinstatement premium is proportional to the upfront premium, so
the expected ratio, 0.06705 here as to amount, does not depend on the premium
charged; page 6 uses that to price a layer that counts on it. A ceding
commission is not paid on reinstatement premiums.

## 4. Commissions and brokerage on the quota share

`corran_qs` cedes 10% of each Corran occurrence up to an event limit of 150
and an annual limit of 300, for 10% of Corran's catastrophe premium. The
contract pays a flat ceding commission of 25%.

**Formulas**, all as shares of the premium `P` with ceded loss `C`:

- ceding commission `CC = c`
- profit commission `PC = s * max(P * (1 - E) - C, 0) / P`, where `E` is the
  allowance: the reinsurer's ceding commission, expenses and margin
- brokerage `BK = b`
- combined ratio `C/P + CC + PC + BK`, margin `1 - combined ratio`

Here the profit commission returns 20% of the profit after an allowance of
35%: the 25% ceding commission plus 10% for Larkspur's costs. Brokerage is
2%, a typical level for a proportional treaty, where the premium is large
against the risk; the excess cover in section 6 pays 10%, nearer what excess
of loss business pays.
`Tower.deal_ratios` gives every ratio in every year; the means are below.

```python
q = CONTRACTS["corran_qs"]
corran = subject(q)


def ratio_means(layer):
    tower = Tower([layer])
    ratios = tower.deal_ratios(tower.apply(corran), layer.name)
    return {k[0]: round(ratios.marginal(list(k)).mean(), 4) for k in ratios.components()}


qs_terms = contract_layer(q).with_profit_commission(0.2, 0.35).with_brokerage(0.02)
for k, v in ratio_means(qs_terms).items():
    print(f"{k:18s}{v:8.4f}")
```

```text
loss_ratio          0.4008
ceding_commission   0.2500
profit_commission   0.0838
brokerage           0.0200
total_deductions    0.3538
combined_ratio      0.7547
nmbf                0.2453
margin              0.2453
```

The expected loss ratio is 40%, but the profit commission is worth 8.4% of
premium, not 20% of (65% - 40%) = 5%. The difference is the shape: in a quiet
year the loss ratio is far below 40% and the commission is large, while in a
bad year the commission is zero and cannot go negative. The reinsurer gives
away part of the good years and keeps all of the bad ones. A margin computed
from the expected loss ratio alone would miss this.

`nmbf` (net margin before fees) equals the margin here because no deduction
is kept outside it.

**A sliding scale instead.** A sliding scale moves the commission against the
loss ratio: here 30% at a loss ratio of 30% or less, 20% at 50% or more, and a
straight line between. `with_sliding_scale` replaces the flat commission, so
it is added to the contract's layer as it is.

```python
qs_slide = contract_layer(q).with_sliding_scale([(0.30, 0.30), (0.20, 0.50)]).with_brokerage(0.02)
for k, v in ratio_means(qs_slide).items():
    print(f"{k:18s}{v:8.4f}")
```

```text
loss_ratio          0.4008
ceding_commission   0.2748
brokerage           0.0200
total_deductions    0.2948
combined_ratio      0.6957
nmbf                0.3043
margin              0.3043
```

At the expected loss ratio of 40% the scale reads 25%, yet the mean
commission is 27.5%. The same shape is at work: most years sit below the
mean and earn the top of the scale.

The profit commission and the sliding scale cannot be put on a layer with
paid reinstatements, so on this book they belong on the quota share and the
aggregate cover, not on the occurrence excess layers.

## 5. A no-claims bonus

A no-claims bonus returns a share of the premium when the cover pays nothing
in the year. The library has no term for it. A sliding scale comes close: a
commission of 10% at a loss ratio of 0 and of 0 at any loss ratio above a tiny
threshold. On `bracken_agg`, the aggregate cover on Bracken's convective storm
and flood losses:

```python
b = CONTRACTS["bracken_agg"]
bracken_cs_fl = cedent_events("bracken", "CS+FL")
ncb = contract_layer(b).with_sliding_scale([(0.10, 0.0), (0.0, 1e-9)])
tower = Tower([ncb])
res = tower.apply(bracken_cs_fl)
ratios = tower.deal_ratios(res, "bracken_agg")
ceded = res.aggregate(["kind", "layer"]).marginal(["ceded", "bracken_agg"]).draws
p_clean = sum(x == 0.0 for x in ceded) / N_YEARS
print("chance of a year with no claim:", round(p_clean, 4))
print("10% times that chance:         ", round(0.10 * p_clean, 4))
print("mean bonus from the scale:     ", round(ratios.marginal(["ceding_commission"]).mean(), 4))
```

```text
chance of a year with no claim: 0.9675
10% times that chance:          0.0968
mean bonus from the scale:      0.0967
```

The scale gives the bonus to rounding. It is an approximation: a real bonus is
often paid only after the year is closed, may depend on several years, and
cannot sit beside a sliding scale on the same layer (#592).

## 6. The reinsurer's ledger of the aggregate cover

A P&L ledger lays each flow out as a row, signed from the reinsurer's side,
and gives each row's distribution. `Tower.deal_ledger` builds it for one layer
over a distribution of annual gross losses. Here the source is Bracken's
2,000 annual totals of convective storm and flood, each equally likely, and
the cover pays 10% brokerage.

```python
totals = bracken_cs_fl.totals().total().draws
agg_cover = contract_layer(b).with_brokerage(0.10)
ledger = Tower([agg_cover]).deal_ledger((list(totals), [1.0] * len(totals)), "bracken_agg")
for row in ledger.rows:
    print(f"{row:18s}{ledger.row(row).mean():8.4f}")
```

```text
Premium             0.4500
Loss               -0.1621
Brokerage          -0.0450
total obligation   -0.2071
result              0.2429
```

The premium is fixed, the loss and the result are not. `ledger.sheet()`
shows each row in the scenario columns, from the bad year to the good one,
and `ledger.card()` gives each row's standard deviation and percentiles.

The ledger applies the tower to an annual loss, so it suits annual covers:
quota shares and aggregate excess layers. It refuses a layer with paid
reinstatements, which rules out the occurrence excess layers on this book.
Their flows are read from the tower's result instead, as in section 3.

## Collateral

A collateralised reinsurer holds the limit in a trust, which changes the
return on a deal but not its expected loss. The library's monthly cash-flow
engine (`CashflowTerms`, `Cashflows`, `DealRun.with_cashflows`) models it;
this track does not run it.

## What is missing

- A true no-claims bonus (#592), and franchise deductibles, nth-event
  covers, top-and-drop covers and parametric triggers (#591).
- `deal_ledger` and the profit commission and sliding scale do not combine
  with paid reinstatements, so an occurrence layer's full deal (reinstatement
  premium, brokerage and a bonus) is assembled by hand from the tower's
  result.

## In R

The R twin, [cat-5-beyond-loss.qmd](cat-5-beyond-loss.qmd), runs the same
steps with `xol_layer(pro_rata_time = )`, `reinstatement_premium()`,
`with_profit_commission()`, `with_sliding_scale()`, `with_brokerage()`,
`deal_ratios()` and `deal_ledger()`.

## Further reading

Within this track, [page 3](cat-3-contract-terms.md) sets the terms these
flows sit on and [page 6](cat-6-pricing.md) prices them.
[What a reinsurance price is](reinsurance-price.md) covers commissions and the
quote for a single layer. [reinsurance.dev](https://reinsurance.dev) by Oliver
Baltzer is another introduction to property catastrophe reinsurance.
