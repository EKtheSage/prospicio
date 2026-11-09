# Property catastrophe 5: the money around the loss

<!-- site: execute -->

A treaty moves more money than its claims: premium, reinstatement premium,
commission, brokerage and interest on collateral. Some of these flows depend on
the losses, so they change the shape of the result, not only its mean.

This page prices each flow on the same 2,000 simulated years of the Larkspur Re
book.

Money is in millions of Varland crowns. The story, the files and the notation
are on [the overview](cat-0-overview.md).

## 1. Words

[What a reinsurance price is](reinsurance-price.md) explains these terms for any
layer. This page adds the cat-specific part: the season and the loss shape.

- **Reinstatement premium**: what the cedent pays to restore a used limit, so
  the layer can pay again in the same year. The contract sets it as a rate of
  the upfront premium for each full limit restored.
- **Pro rata as to amount**: the cedent pays for the share of the limit
  restored. **Pro rata as to time** also scales each restoration by the share
  of the year still to run at the time of the loss.
- **Ceding commission**: a share of the premium paid back to the cedent for its
  costs. A **sliding scale** makes that share depend on the loss ratio.
- **Profit commission**: a share of the treaty's profit paid back to the
  cedent, after an allowance for the reinsurer's costs and margin.
- **Brokerage**: the broker's fee, a share of the premium.
- **Deal ratios**: each amount over the premium, year by year.

## 2. The book

The helper rebuilds one cedent's years from the YELT, with times and peril
labels, and applies the hours clause where the contract has one. It is the
helper of the other pages, except that `contract_layer` passes extra settings
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

Premium is Larkspur's share: rate on line times limit times share for the
excess covers, the ceded share of Corran's catastrophe premium for the quota
share.

## 3. Reinstatement premium

**Intuition.** `alder_cat_1` is 40 xs 30 with one reinstatement at 100%. The
cedent pays the same share of the upfront premium as the share of the limit
restored. The layer pays at most 80 a year at 100%.

**Formula.** With `P` the upfront premium, `l` the limit, `r_j` the rate of
reinstatement `j` and `Y` the layer loss of the year at 100%:

```
RIP = P * sum over j of r_j * min(max(Y - (j - 1) * l, 0), l) / l
```

over the contract's reinstatements. As to time, each restored piece is also
multiplied by `1 - t`, where `t` is the share of the year gone at its loss.

**By hand.** Two losses of 45 and 65 put 15 and then 35 into the layer. The
contract restores all of the first but only 25 of the second (one limit of 40):
40/40 of the premium, 1.8, as to amount. If the losses fall a quarter and three
quarters of the way through the year, as to time gives
(15 x 0.75 + 25 x 0.25) / 40 x 1.8 = 0.7875.

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
component next to the ceded loss.

```python
import pandas as pd
from great_tables import GT

alder = subject(c)
basis = {"as to amount": by_amount, "as to time": by_time}
res = {name: Tower([layer]).apply(alder) for name, layer in basis.items()}
rip = {name: mean_of(r, "reinstatement_premium", "alder_cat_1") for name, r in res.items()}
book = pd.DataFrame({
    "basis": list(basis),
    "ceded": [mean_of(r, "ceded", "alder_cat_1") for r in res.values()],
    "rip": list(rip.values()),
    "ratio": [rip[n] / by_amount.premium for n in basis],
    "change": [None, rip["as to time"] / rip["as to amount"] - 1],
})
(GT(book, rowname_col="basis")
 .tab_header("alder_cat_1: expected flows per year over 2,000 years",
             "Millions of crowns; the ratio is over the upfront premium")
 .cols_label(ceded="ceded loss", rip="reinstatement premium", ratio="premium ratio",
             change="change vs amount")
 .fmt_number(["ceded", "rip"], decimals=4)
 .fmt_number("ratio", decimals=5)
 .fmt_number("change", decimals=3)
 .sub_missing(missing_text=""))
```

```text
              ceded loss  reinstatement premium  premium ratio  change vs amount
as to amount      0.8230                 0.1207        0.06705
as to time        0.8230                 0.0447        0.02484            -0.629
```

As to time collects 63% less because of the season. By quarter:

```python
hits = [t for i in range(N_YEARS) for x, t in zip(alder.events(i), alder.times(i)) if x > 30.0]
quarters = []
for k in range(4):
    left = [1 - t for t in hits if k / 4 <= t < (k + 1) / 4]
    quarters.append((f"Q{k + 1}", len(left), sum(left) / len(left)))
quarters.append(("All", len(hits), sum(1 - t for t in hits) / len(hits)))
(GT(pd.DataFrame(quarters, columns=["quarter", "n", "left"]), rowname_col="quarter")
 .tab_header(f"{len(hits)} occurrences reach alder_cat_1, by quarter of the year")
 .cols_label(n="occurrences", left="mean share of the year left")
 .fmt_integer("n")
 .fmt_number("left", decimals=3))
```

```text
     occurrences  mean share of the year left
Q1            22                        0.843
Q2            37                        0.622
Q3           114                        0.340
Q4            68                        0.190
All          241                        0.387
```

182 of the 241 occurrences come in the second half of the year. Alder's large
losses are mostly windstorms, and nine in ten fall between day 213 and day 305.
A late loss restores limit for only a few months, so as to time the cedent pays
only for those months. A peril spread evenly over the year would show a
smaller gap. Use the basis of the treaty wording: for a seasonal peril the
difference is material.

The reinstatement premium is proportional to the upfront premium, so the
expected ratio (0.06705 as to amount) does not depend on the premium charged.
Page 6 uses this to price a layer. Reinstatement premiums carry no ceding
commission.

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

Both sets of terms pay 2% brokerage (the excess cover in section 6 pays 10%).

- **Flat commission plus profit commission.** The profit commission returns
  20% of the profit after an allowance of 35%: the 25% ceding commission plus
  10% for the costs of Larkspur.
- **Sliding scale.** The commission is 30% at a loss ratio of 30% or less, 20%
  at 50% or more, and a straight line between. `with_sliding_scale` replaces the
  flat commission.

`Tower.deal_ratios` gives every ratio in every year. The table shows the means.

```python
q = CONTRACTS["corran_qs"]
corran = subject(q)


def ratio_means(layer):
    tower = Tower([layer])
    ratios = tower.deal_ratios(tower.apply(corran), layer.name)
    return {k[0]: ratios.marginal(list(k)).mean() for k in ratios.components()}


qs_terms = contract_layer(q).with_profit_commission(0.2, 0.35).with_brokerage(0.02)
qs_slide = contract_layer(q).with_sliding_scale([(0.30, 0.30), (0.20, 0.50)]).with_brokerage(0.02)
qs_means = pd.DataFrame({"flat + profit commission": ratio_means(qs_terms),
                         "sliding scale": ratio_means(qs_slide)})
qs_means.index.name = "ratio"
(GT(qs_means.reset_index(), rowname_col="ratio")
 .tab_header("Corran quota share: mean ratios under two commission terms",
             "Shares of premium; brokerage 2%")
 .fmt_number(["flat + profit commission", "sliding scale"], decimals=4)
 .sub_missing(missing_text=""))
```

```text
                   flat + profit commission  sliding scale
loss_ratio                           0.4008         0.4008
ceding_commission                    0.2500         0.2748
profit_commission                    0.0838
brokerage                            0.0200         0.0200
total_deductions                     0.3538         0.2948
combined_ratio                       0.7547         0.6957
nmbf                                 0.2453         0.3043
margin                               0.2453         0.3043
```

The expected loss ratio is 40%, but the profit commission is worth 8.4% of
premium, not 20% of (65% - 40%) = 5%: the commission is large in a quiet year
and zero, never negative, in a bad one. The reinsurer gives away part of the
good years and keeps all of the bad ones. Likewise the scale reads 25% at a
loss ratio of 40%, yet its mean commission is 27.5%, because most years are
below the mean and earn the top of the scale. A margin from the expected loss
ratio alone misses this. `nmbf` (net margin before fees) equals the margin
here: no deduction is outside it.

Commission by loss ratio, and the loss ratio of Corran over the 2,000 years (up
to 150%; 6% of the years are above):

```python
import matplotlib.pyplot as plt
import numpy as np

lr = np.linspace(0, 1, 101)
flat = np.full_like(lr, 0.25)
with_pc = 0.25 + 0.2 * np.maximum(0.65 - lr, 0)
slide = np.interp(lr, [0.30, 0.50], [0.30, 0.20])
tower_qs = Tower([qs_terms])
qs_lr = tower_qs.deal_ratios(tower_qs.apply(corran), "corran_qs").marginal(["loss_ratio"]).draws

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
ax.plot(100 * lr, 100 * flat, label="flat 25%")
ax.plot(100 * lr, 100 * with_pc, label="flat 25% + profit commission")
ax.plot(100 * lr, 100 * slide, label="sliding scale")
ax.set_xlabel("loss ratio (% of premium)")
ax.set_ylabel("commission (% of premium)")
ax.set_title("Commission by loss ratio")
ax.legend(fontsize=7)
ax2.hist(100 * np.asarray(qs_lr), bins=30, range=(0, 150), color="0.75")
ax2.set_xlabel("loss ratio (% of premium)")
ax2.set_ylabel("simulated years")
ax2.set_title("Loss ratio of Corran (up to 150%)")
```

A layer with paid reinstatements cannot take a profit commission or a sliding
scale, so here they go on the quota share and the aggregate cover.

## 5. A no-claims bonus

A no-claims bonus returns a share of the premium when the cover pays nothing in
the year. `Layer.with_no_claims_bonus(share, years, threshold)` sets the share
returned, the years it looks back over, and the most ceded loss that still earns
it (default 0). Here it is 10% on `bracken_agg`, the aggregate cover on
Bracken's convective storm and flood losses:

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

The mean bonus is 10% times that chance, 0.09675 (the figures differ only in
rounding). The bonus reads the ceded loss alone, so a layer can carry it beside
a sliding scale or a profit commission. Deal ratios count it as a deduction
inside NMBF.

A bonus can look back over several contract years: it is paid only if the last
`years` years together ceded at most the threshold. That needs
the paths of a term, so `Tower.apply_term` applies the tower year by year
(`Tower.apply` refuses it). Here a term is two of the 2,000 simulated years,
and the bonus looks back two years:

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

Year 1 looks back only to the start of the term, so it pays about as often as
a one-year bonus (0.9675 above). Year 2 needs two clean years: about the square
of that chance, since the years are independent.

## 6. The reinsurer's ledger of the aggregate cover

A P&L ledger shows each flow as a row, signed from the reinsurer's side, with
the distribution of each row. `Tower.deal_ledger` builds it for one layer over a
distribution of annual gross losses: here Bracken's 2,000 annual totals of
convective storm and flood, each equally likely. The cover pays 10% brokerage.

```python
totals = bracken_cs_fl.totals().total().draws
agg_cover = contract_layer(b).with_brokerage(0.10)
ledger = Tower([agg_cover]).deal_ledger((list(totals), [1.0] * len(totals)), "bracken_agg")
rows = list(ledger.rows)
(GT(pd.DataFrame({"row": rows, "mean": [ledger.row(r).mean() for r in rows]}), rowname_col="row")
 .tab_header("Bracken aggregate cover: mean of each ledger row",
             "Millions of crowns per year, signed from the reinsurer's side")
 .fmt_number("mean", decimals=4))
```

```text
                     mean
Premium            0.4500
Loss              -0.1621
Brokerage         -0.0450
total obligation  -0.2071
result             0.2429
```

Only the premium is fixed. `ledger.sheet()` shows each row in the scenario
columns, from the bad year to the good one; `ledger.card()` gives the standard
deviation and percentiles of each row.

The ledger applies the tower to an annual loss, so it suits quota shares and
aggregate excess layers. It refuses a layer with paid reinstatements: for
occurrence layers, read the flows from the tower's result, as in section 3.

## Collateral

A collateralised reinsurer holds the limit in a trust. This changes the return
on a deal, not its expected loss. The monthly cash-flow engine (`CashflowTerms`,
`Cashflows`, `DealRun.with_cashflows`) models it.

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
