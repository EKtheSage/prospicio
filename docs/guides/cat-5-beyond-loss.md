# Property catastrophe 5: the money around the loss

A treaty moves more money than its claims. The cedent pays a premium. It can
pay again to restore a used limit, and it gets some of the premium back as
commission. A broker takes a fee, and collateral in a trust earns interest.
Each of these flows changes what the reinsurer keeps. Some of them depend on
the losses, so they change the shape of the result, not only its mean.

This page follows the book of Larkspur Re from the earlier pages of the track.
It prices each flow on the same 2,000 simulated years. It covers:

1. Reinstatement premium, as to amount and as to time.
2. Ceding commission, profit commission, sliding scale and brokerage on
   Corran's quota share, read as deal ratios.
3. A no-claims bonus, over one year and over several.
4. The reinsurer's P&L ledger of Bracken's aggregate cover.

Money is in millions of Varland crowns. The story, the files and the notation
are on [the overview](cat-0-overview.md).

## 1. Words

[What a reinsurance price is](reinsurance-price.md) explains reinstatements,
commissions and sliding scales for any layer. The list below sets the words.
The rest of the page shows what is particular to cat business: the season, and
the shape of a cat loss distribution.

- **Reinstatement premium**: what the cedent pays to restore the limit that a
  loss used. The layer can then pay again in the same year. The contract
  sets it as a rate of the upfront premium for each full limit restored.
- **Pro rata as to amount**: the cedent pays for the share of the limit
  restored. **Pro rata as to time** also scales each restoration by the share
  of the year still to run at the time of the loss.
- **Ceding commission**: a share of the premium that the reinsurer pays back
  to the cedent for its acquisition and handling costs. A **sliding scale**
  makes that share depend on the loss ratio of the year.
- **Profit commission**: a share of the profit on the treaty that the
  reinsurer pays back to the cedent. The profit is after an allowance for the
  costs and margin of the reinsurer.
- **Brokerage**: the fee of the broker, a share of the premium. It comes off
  what the reinsurer receives.
- **Deal ratios**: each amount over the premium, year by year: the loss ratio,
  each deduction, the combined ratio and the margin left.

## 2. The book

The helper below rebuilds the years of one cedent from the YELT, with times
and peril labels. It makes a `Layer` from a row of the contracts file, and it
applies the hours clause where the contract has one. It is the same helper as
on the other pages, with one addition: `contract_layer` passes extra settings
(here `pro_rata_time`) to the layer.

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

Each premium is the premium of Larkspur for its own share. For the excess
covers it is rate on line times limit times share. For the quota share it is
the ceded share of the catastrophe premium of Corran.

## 3. Reinstatement premium

**Intuition.** `alder_cat_1` is 40 xs 30 with one reinstatement at 100%. After
a loss uses part of the limit, the cedent pays to restore it. It pays the same
share of the upfront premium as the share of the limit that it restores. The
contract restores only one limit, so the layer pays at most 80 a year at 100%.

**Formula.** With `P` the upfront premium, `l` the limit, `r_j` the rate of
reinstatement `j` and `Y` the layer loss of the year at 100%:

```
RIP = P * sum over j of r_j * min(max(Y - (j - 1) * l, 0), l) / l
```

for the reinstatements that the contract has. As to time, the formula also
multiplies each restored piece by `1 - t`. Here `t` is the share of the year
gone at the time of its loss.

**By hand.** Two losses of 45 and 65 put 15 and then 35 into the layer. The
contract restores all of the first, but only 25 more of the second, because
there is only one limit of 40. As to amount that is 40/40 of the premium, 1.8.
Suppose that the losses occur a quarter and three quarters of the way through
the year. As to time then gives (15 x 0.75 + 25 x 0.25) / 40 x 1.8 = 0.7875.

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

As to time collects about 63% less because of the season. The table below
splits the occurrences that reach the layer by the quarter of the year in which
they occur.

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

182 of the 241 occurrences come in the second half of the year. The large
losses of Alder are mostly windstorms, and the generator puts nine in ten
windstorms between day 213 and day 305. A loss then restores limit for only the
few months left, and as to time the cedent pays only for those months. A peril
spread evenly over the year would show a smaller gap. Use the basis of the
treaty wording. For a seasonal peril the difference is material.

The reinstatement premium of each year is proportional to the upfront premium.
Thus the expected ratio, 0.06705 here as to amount, does not depend on the
premium charged. Page 6 uses this fact to price a layer that counts on it.
Reinstatement premiums carry no ceding commission.

## 4. Commissions and brokerage on the quota share

`corran_qs` cedes 10% of each Corran occurrence up to an event limit of 150
and an annual limit of 300. Its premium is 10% of the catastrophe premium of
Corran. The contract pays a flat ceding commission of 25%.

**Formulas**, all as shares of the premium `P` with ceded loss `C`:

- ceding commission `CC = c`
- profit commission `PC = s * max(P * (1 - E) - C, 0) / P`, where `E` is the
  allowance: the reinsurer's ceding commission, expenses and margin
- brokerage `BK = b`
- combined ratio `C/P + CC + PC + BK`, margin `1 - combined ratio`

Here the profit commission returns 20% of the profit after an allowance of
35%: the 25% ceding commission plus 10% for the costs of Larkspur. Brokerage is
2%, a typical level for a proportional treaty, where the premium is large
against the risk. The excess cover in section 6 pays 10%, nearer to what
excess of loss business pays. `Tower.deal_ratios` gives every ratio in every year. The
means are below.

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
premium, not 20% of (65% - 40%) = 5%. The difference comes from the shape. In
a quiet year the loss ratio is far below 40% and the commission is large. In a
bad year the commission is zero and cannot go negative. The reinsurer gives
away part of the good years and keeps all of the bad ones. A margin from the
expected loss ratio alone does not show this.

`nmbf` (net margin before fees) equals the margin here, because no deduction
is outside it.

**A sliding scale instead.** A sliding scale moves the commission against the
loss ratio. Here it is 30% at a loss ratio of 30% or less, 20% at 50% or more,
and a straight line between. `with_sliding_scale` replaces the flat
commission, so you can add it to the layer of the contract as it is.

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

At the expected loss ratio of 40% the scale reads 25%, but the mean commission
is 27.5%. The same shape causes this: most years are below the mean and earn
the top of the scale.

A layer with paid reinstatements cannot take a profit commission or a sliding
scale. On this book they go on the quota share and the aggregate cover.

## 5. A no-claims bonus

A no-claims bonus returns a share of the premium when the cover pays nothing in
the year. `Layer.with_no_claims_bonus(share, years, threshold)` sets it: the
share of the premium returned, the contract years it looks back over, and the
most ceded loss that still earns it (0, a year with no claim, by default). Here
the bonus is 10% of the premium on `bracken_agg`, the aggregate cover on the
convective storm and flood losses of Bracken:

```python
b = CONTRACTS["bracken_agg"]
bracken_cs_fl = cedent_events("bracken", "CS+FL")
ncb = contract_layer(b).with_no_claims_bonus(0.10)
tower = Tower([ncb])
res = tower.apply(bracken_cs_fl)
ratios = tower.deal_ratios(res, "bracken_agg")
ceded = res.aggregate(["kind", "layer"]).marginal(["ceded", "bracken_agg"]).draws
p_clean = sum(x == 0.0 for x in ceded) / N_YEARS
print("chance of a year with no claim:", round(p_clean, 4))
print("10% times that chance:         ", round(0.10 * p_clean, 4))
print("mean no-claims bonus:          ", round(ratios.marginal(["no_claims_bonus"]).mean(), 4))
```

```text
chance of a year with no claim: 0.9675
10% times that chance:          0.0968
mean no-claims bonus:           0.0967
```

The mean bonus is 10% times the chance of a year with no claim, 0.09675; the
two lines differ only in how that rounds. The bonus reads the ceded loss on
its own, so a layer can carry it beside a sliding scale or a profit
commission. Deal ratios count it as a deduction inside NMBF.

A bonus can also look back over several contract years: it is paid only in a
year whose last `years` years together ceded at most the threshold. That needs
the paths of a term, so `Tower.apply_term` applies the tower year by year, and
`Tower.apply` refuses it. Over a term of two years, each path made of two of
the 2,000 simulated years, a bonus that looks back two years:

```python
full = ncb.no_claims_bonus_for(0.0)
two_years = Tower([contract_layer(b).with_no_claims_bonus(0.10, years=2)])
term = two_years.apply_term(bracken_cs_fl.term_years(2))
for year in (1, 2):
    paid = term.select("year", [year]).select("kind", ["no_claims_bonus"]).mean() / full
    print(f"year {year}: bonus paid on {paid:.4f} of the paths")
```

```text
year 1: bonus paid on 0.9710 of the paths
year 2: bonus paid on 0.9360 of the paths
```

The first year looks back to the start of the term only, so it pays as often
as a one-year bonus on the same 1,000 simulated years (0.9710 here, against
0.9675 on all 2,000). The second year pays only when both years were free of
claims, about the square of that chance, since the years are independent.

## 6. The reinsurer's ledger of the aggregate cover

A P&L ledger shows each flow as a row, signed from the side of the reinsurer,
and gives the distribution of each row. `Tower.deal_ledger` builds it for one
layer over a distribution of annual gross losses. Here the source is the 2,000
annual totals of convective storm and flood of Bracken, each equally likely.
The cover pays 10% brokerage.

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

The premium is fixed, but the loss and the result are not. `ledger.sheet()`
shows each row in the scenario columns, from the bad year to the good one.
`ledger.card()` gives the standard deviation and percentiles of each row.

The ledger applies the tower to an annual loss, so it suits annual covers:
quota shares and aggregate excess layers. It refuses a layer with paid
reinstatements. For the occurrence excess layers, read the flows from the
result of the tower, as in section 3.

## Collateral

A collateralised reinsurer holds the limit in a trust. This changes the return
on a deal but not its expected loss. The monthly cash-flow engine of the
library (`CashflowTerms`, `Cashflows`,
`DealRun.with_cashflows`) models it.

## In R

The R twin, [cat-5-beyond-loss.qmd](cat-5-beyond-loss.qmd), runs the same
steps with `xol_layer(pro_rata_time = )`, `reinstatement_premium()`,
`with_profit_commission()`, `with_sliding_scale()`, `with_no_claims_bonus()`,
`apply_tower_term()`, `with_brokerage()`, `deal_ratios()` and `deal_ledger()`.

## Further reading

Within this track, [page 3](cat-3-contract-terms.md) sets the terms of these
flows, and [page 6](cat-6-pricing.md) prices them.
[What a reinsurance price is](reinsurance-price.md) covers commissions and the
quote for a single layer. [reinsurance.dev](https://reinsurance.dev) by Oliver
Baltzer is another introduction to property catastrophe reinsurance.
