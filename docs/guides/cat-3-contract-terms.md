# Property catastrophe 3: writing Larkspur's contracts in code

<!-- site: execute -->

A catastrophe treaty is a function: it takes a year of the cedent's event
losses and returns what the reinsurer pays. The contract terms set its shape:
how the pieces of one storm add up, where each loss attaches, how much limit a
year has and which losses the treaty sees.

This page turns the five contracts of Larkspur Re
(`examples/data/cat_book_contracts.csv`) into code, checks the terms on small
years you can follow by hand, then runs them on the 2,000 simulated years. The
story and notation are on
[the overview](cat-0-overview.md). [Page 1](cat-1-loss-tables.md) introduces
the loss tables.

## 1. The book and its contracts

The year event loss table (YELT) has one row per cedent per event piece, with a
day and an hour. `cedent_events` builds each cedent's event set by hand (#594):
losses per year, loss times as fractions of the year, perils as labels.

```python
import csv
import math

import pandas as pd
from great_tables import GT
from prospicio.aggregate import EventSet, Exposure, Timeline
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


terms = pd.DataFrame([
    {"contract": name, "kind": c["kind"], "perils": c["perils"], "share": float(c["share"]),
     "hours": c["hours"] or "-"}
    for name, c in CONTRACTS.items()
])
(GT(terms, rowname_col="contract")
 .tab_header("Larkspur's five contracts", "Terms from cat_book_contracts.csv; share is Larkspur's signed line")
 .fmt_number("share", decimals=2))
```

```text
                      kind  perils  share  hours
alder_cat_1  occurrence_xl     all   0.30     72
alder_cat_2  occurrence_xl     all   0.20     72
bracken_cat  occurrence_xl     all   0.25     72
bracken_agg   aggregate_xl   CS+FL   0.25      -
corran_qs      quota_share     all   0.10     72
```

Each row of the file becomes one `Layer`. Amounts are at 100%; `share` is
Larkspur's signed line and scales every recovery. Premium is rate on line times
limit times share. Only this function reads the terms.

```python
def num(x):
    return float(x) if x != "" else None

def contract_layer(c):
    name, share = c["contract"], num(c["share"])
    if c["kind"] == "occurrence_xl":
        limit = num(c["limit"])
        return Layer(name, limit, num(c["attachment"]), share=share,
                     premium=num(c["rate_on_line"]) * limit * share,
                     reinstatement_rates=[num(c["reinstatement_rate"])] * int(c["reinstatements"]))
    if c["kind"] == "aggregate_xl":
        agg_limit = num(c["aggregate_limit"])
        return Layer(name, math.inf, 0.0, share=share,
                     aggregate_deductible=num(c["aggregate_deductible"]), aggregate_limit=agg_limit,
                     premium=num(c["rate_on_line"]) * agg_limit * share)
    premium = share * float(CEDENTS[c["cedent"]]["cat_premium"])
    return (Layer(name, num(c["limit"]), 0.0, share=share, aggregate_limit=num(c["aggregate_limit"]),
                  premium=premium)
            .with_ceding_commission(num(c["ceding_commission"])))

LAYERS = {name: contract_layer(c) for name, c in CONTRACTS.items()}
layers = pd.DataFrame([
    {"contract": name, "limit": layer.limit, "attachment": layer.attachment,
     "annual limit": layer.aggregate_limit, "premium": layer.premium}
    for name, layer in LAYERS.items()
])
(GT(layers, rowname_col="contract")
 .tab_header("The five contracts as layers", "Limits and attachments at 100%, in millions; premium on Larkspur's line")
 .fmt_number(["limit", "attachment", "annual limit"], decimals=0)
 .fmt_number("premium", decimals=3))
```

```text
             limit  attachment  annual limit  premium
alder_cat_1     40          30            80    1.800
alder_cat_2    100          70           200    1.000
bracken_cat     50          25           150    1.750
bracken_agg    inf           0            30    0.450
corran_qs      150           0           300    3.000
```

One paid reinstatement doubles a layer's annual limit. The aggregate cover has
no occurrence limit.

## 2. What counts as one loss: the hours clause

A storm can enter the loss table in pieces (a windstorm over two days, hail
cells). The hours clause sets how many hours of losses make one **occurrence**.
Under 40 xs 30, two pieces of 25 recover nothing separately and 20 together.

**The rule on this track.** `EventSet.grouped(hours)` starts a window at a
year's first piece and joins every piece within that many hours of it, then
starts the next window at the next piece. It groups all perils together, with
the one `hours` value of the contracts file. Both choices are simplifications.

A real clause usually lets the cedent choose where each window starts, so a
fixed window can understate recoveries. Many clauses also set a window per
peril, longer for flood (often 168 hours) than for windstorm. The blocks below
test the per-peril rule on this book.

```python
alder = cedent_events("alder")
alder_72 = alder.grouped(72.0)
print("labels before", alder.label_dims, " after", alder_72.label_dims)

def ceded_mean(layer, events):
    res = Tower([layer]).apply(events)
    return res.aggregate(["kind", "layer"]).marginal(["ceded", layer.name]).mean()

cat_1 = LAYERS["alder_cat_1"]
```

```text
labels before ['peril']  after []
```

Grouping removes the labels, because an occurrence of pieces of two perils has
no single peril. To group by peril, group each peril separately, label it again
and merge. The merged set keeps the labels and orders each year's occurrences in
time. Page 4 reuses this helper.

```python
def grouped_by_peril(cedent, hours):
    out = None
    for p in ["WS", "EQ", "FL", "CS"]:
        g = cedent_events(cedent, p).grouped(hours)
        g = g.with_labels({"peril": [p] * sum(g.counts())})
        out = g if out is None else out.merge(g)
    return out

alder_72p = grouped_by_peril("alder", 72.0)
rules = {"no clause": alder, "72 hours, all perils": alder_72, "72 hours, by peril": alder_72p}
grouping = pd.DataFrame({
    "rule": list(rules),
    "occurrences": [sum(e.counts()) for e in rules.values()],
    "expected loss": [ceded_mean(cat_1, e) for e in rules.values()],
})
(GT(grouping, rowname_col="rule")
 .tab_header("alder_cat_1 under three grouping rules", "Expected loss in millions on Larkspur's line")
 .fmt_number("occurrences", decimals=0, use_seps=False)
 .fmt_number("expected loss", decimals=4))
```

```text
                      occurrences  expected loss
no clause                    5391         0.7778
72 hours, all perils         4238         0.8230
72 hours, by peril           4275         0.8229
```

Grouping raises the expected loss by about 6%. Grouping by peril adds a few
occurrences and barely moves it, so the track prices with the single all-perils
clause and groups by peril only where it needs the labels.

## 3. Alder's layers: occurrence excess of loss with reinstatements

An occurrence `x` recovers `min(max(x - a, 0), l)` from a layer `l xs a`.
Recoveries add up over the year until the annual limit, `l` times one plus the
reinstatements, is used. Page 5 prices the reinstatements.

`ceded_by_event` shows the order. Take `alder_cat_1` (40 xs 30, 30%, one
reinstatement, so 80 of annual limit) and four occurrences: one below the
attachment, one through the top, two in the middle.

```python
year = [28.0, 110.0, 64.0, 95.0]
line = float(CONTRACTS["alder_cat_1"]["share"])
left = cat_1.aggregate_limit
steps = []
for x, rec in zip(year, cat_1.ceded_by_event(year)):
    left -= rec / line
    steps.append({"loss": f"{x:.0f}", "at 100%": rec / line, "limit left": max(left, 0.0), "Larkspur's line": rec})
print(f"year: {cat_1.ceded(year):.2f}")
(GT(pd.DataFrame(steps), rowname_col="loss")
 .tab_header("alder_cat_1 on a year of four occurrences", "Amounts in millions, in time order")
 .fmt_number(["at 100%", "limit left"], decimals=1)
 .fmt_number("Larkspur's line", decimals=2))
```

```text
year: 24.00
     at 100%  limit left  Larkspur's line
28       0.0        80.0             0.00
110     40.0        40.0            12.00
64      34.0         6.0            10.20
95       6.0         0.0             1.80
```

Alone, the 95 recovers 40, but only 6 of the annual limit remains: late
occurrences get what is left, so order matters. The book's years are in time
order (the file lists pieces by `slot`). In the figure, the 28 pays nothing and
the 95 gets only the 6 that remain.

```python
import matplotlib.pyplot as plt

paid = [r / line for r in cat_1.ceded_by_event(year)]
remaining, left = [], cat_1.aggregate_limit
for p in paid:
    left -= p
    remaining.append(max(left, 0.0))
x = range(len(year))
fig, ax = plt.subplots()
ax.bar([i - 0.2 for i in x], paid, width=0.4, label="paid by the layer (100%)")
ax.bar([i + 0.2 for i in x], remaining, width=0.4, label="annual limit left")
ax.set_xticks(list(x), [f"loss {v:.0f}" for v in year])
ax.set_xlabel("occurrence, in time order")
ax.set_ylabel("amount at 100% (millions)")
ax.set_title("alder_cat_1, 40 xs 30: the annual limit runs out")
ax.legend()
```

## 4. Corran's quota share: an event limit and an annual limit

A plain quota share cedes a fixed share of every loss. A cat quota share
adds an event limit and an annual limit. `corran_qs` cedes 10% with an event
limit of 150 and an annual limit of 300 at 100%: a layer 150 xs 0 at a 10% share
with an annual limit of 300. In a year of 200 and 120, the event limit caps the
first event at 150, so the cover pays 10% of 270. `Layer.quota_share` has no
limits and pays 10% of 320.

```python
qs = LAYERS["corran_qs"]
print(qs.ceded([200.0, 120.0]), [round(x, 4) for x in qs.ceded_by_event([200.0, 120.0])])
print(Layer.quota_share("uncapped", 0.1).ceded([200.0, 120.0]))
```

```text
27.0 [15.0, 12.0]
32.0
```

## 5. Bracken's aggregate cover, and the losses it sees

The **subject losses** of a contract are the cedent's losses of its perils,
grouped by its hours clause, before its terms. `bracken_agg` covers only
convective storm and flood, so remove the other perils first:
`cedent_events("bracken", "CS+FL")` (#584).

A split by label cannot do this, because it comes after the cession. Take an
earthquake of 25, then a windstorm of 30, under 20 xs 10 with an annual limit of
20. The earthquake recovers 15 and leaves 5 for the windstorm. A windstorm-only
treaty pays 20 for it.

```python
year = EventSet.from_years([[25.0, 30.0]], times=[[0.1, 0.2]]).with_labels({"peril": ["EQ", "WS"]})
layer = Layer("20 xs 10", 20.0, 10.0, aggregate_limit=20.0)
split = Tower([layer]).apply(year)
for peril in ["EQ", "WS"]:
    print(peril, "share of the all-perils recovery", split.marginal(["ceded", "20 xs 10", peril]).mean())
ws_only = Tower([layer]).apply(EventSet.from_years([[30.0]], times=[[0.2]]))
print("WS under a windstorm-only treaty", ws_only.aggregate(["kind", "layer"]).marginal(["ceded", "20 xs 10"]).mean())
```

```text
EQ share of the all-perils recovery 15.0
WS share of the all-perils recovery 5.0
WS under a windstorm-only treaty 20.0
```

A label split gives the windstorm's part of this treaty's recovery, not what a
windstorm treaty pays.

An aggregate cover pays `share * min(max(S - AD, 0), AL)` on the annual total
`S`, with aggregate deductible `AD` and limit `AL`. In prospicio it is a layer
with an infinite occurrence limit, no attachment and the annual terms.
`bracken_agg` is 30 xs 40 in the aggregate at 25%. A year of 10, 20 and 25
totals 55, so the cover pays a quarter of 15.

```python
agg = LAYERS["bracken_agg"]
print(agg.ceded([10.0, 20.0, 25.0]))
agg_pro_rata = Layer("bracken_agg", math.inf, 0.0, share=0.25, aggregate_deductible=40.0,
                     aggregate_limit=30.0, allocation="pro_rata")
split = pd.DataFrame({
    "loss": ["10", "20", "25"],
    "event order": agg.ceded_by_event([10.0, 20.0, 25.0]),
    "pro rata": agg_pro_rata.ceded_by_event([10.0, 20.0, 25.0]),
})
(GT(split, rowname_col="loss")
 .tab_header("bracken_agg on a year of 10, 20 and 25", "Recovery by event, in millions on Larkspur's line")
 .fmt_number(["event order", "pro rata"], decimals=4))
```

```text
3.75
    event order  pro rata
10       0.0000    0.6818
20       0.0000    1.3636
25       3.7500    1.7045
```

The annual value is the same; the split by event differs. In event order the
deductible takes the first losses and the third event gets all the recovery.
`allocation="pro_rata"` scales every event by the same factor. The split matters
only when a later step reads events, such as an inuring stage (page 4).

On the book, the cover sees only Bracken's convective storm and flood losses.

```python
bracken_csfl = cedent_events("bracken", "CS+FL")
annual = bracken_csfl.totals().total().draws
largest = bracken_csfl.maxima().total().draws
past = [i for i in range(N_YEARS) if annual[i] > 40]
print(f"mean CS+FL annual loss {sum(annual) / N_YEARS:.4f}")
print(f"years past the deductible {len(past) / N_YEARS:.4f},"
      f" years that use the whole limit {sum(s >= 70 for s in annual) / N_YEARS:.4f}")
print(f"of the {len(past)} years past the deductible, {sum(largest[i] > 40 for i in past)} have one loss above 40")
print(f"bracken_agg expected loss {ceded_mean(agg, bracken_csfl):.4f}, premium {agg.premium:.4f}")
```

```text
mean CS+FL annual loss 7.1642
years past the deductible 0.0325, years that use the whole limit 0.0130
of the 65 years past the deductible, 52 have one loss above 40
bracken_agg expected loss 0.1621, premium 0.4500
```

The cover attaches in about one year in thirty. It does not cap one event's
contribution, so a single large flood or storm outbreak is the usual trigger: in
52 of the 65 years past the deductible, one loss alone is above it. For
frequency protection, add an each-and-every-loss cap on each event's
contribution.

## 6. All five contracts on the book

Each contract runs on its own perils and hours clause. The loss ratio is
expected ceded loss over upfront premium, before reinstatement premium and
commissions.

```python
def subject(c):
    e = cedent_events(c["cedent"], c["perils"])
    return e.grouped(float(c["hours"])) if c["hours"] else e

rows = []
for name, c in CONTRACTS.items():
    layer = LAYERS[name]
    rows.append({"contract": name, "expected loss": ceded_mean(layer, subject(c)), "premium": layer.premium})
contracts = pd.DataFrame(rows)
contracts["loss ratio"] = contracts["expected loss"] / contracts["premium"]
(GT(contracts, rowname_col="contract")
 .tab_header("Larkspur's five contracts on the 2,000 simulated years",
             "Amounts in millions, on Larkspur's line, before reinstatement premium and commissions")
 .fmt_number("expected loss", decimals=4)
 .fmt_number(["premium", "loss ratio"], decimals=3))
```

```text
             expected loss  premium  loss ratio
alder_cat_1         0.8230    1.800       0.457
alder_cat_2         0.4163    1.000       0.416
bracken_cat         0.6847    1.750       0.391
bracken_agg         0.1621    0.450       0.360
corran_qs           1.2025    3.000       0.401
```

```python
fig, ax = plt.subplots()
y = range(len(contracts))
ax.barh([i + 0.2 for i in y], contracts["premium"], height=0.4, color="0.75", label="premium")
ax.barh([i - 0.2 for i in y], contracts["expected loss"], height=0.4, label="expected loss")
ax.set_yticks(list(y), contracts["contract"])
ax.invert_yaxis()
ax.set_xlabel("millions, on Larkspur's line")
ax.set_title("Expected loss against premium, by contract")
ax.legend()
```

## 7. The contract year

A treaty covers losses that occur in a period (losses occurring) or losses on
policies written in a period (risks attaching). prospicio models this with a
`Timeline` of accident months, an `Exposure` of how much of each month each
underwriting year carries, and a tower with a cover basis.

Under losses occurring, contract years start at month 0 of the timeline. Trap:
an 18-month timeline with one underwriting year keeps an event in month 3 and
silently drops an event in month 16.

```python
events = EventSet.from_years([[100.0, 80.0]], times=[[3.0 / 18, 16.2 / 18]])
lo = Tower([Layer("XL", 50.0, 20.0)]).with_cover_basis("losses_occurring")
res = lo.apply_timeline(Timeline(events, 18), Exposure([[1.0] * 18]))
print(res.components())
print("gross seen", res.marginal(["gross", "ground_up", 1]).mean())
```

```text
[('gross', 'ground_up', 1), ('ceded', 'XL', 1), ('net', 'retained', 1)]
gross seen 100.0
```

The 80 in month 16 is in no contract year, and prospicio gives no warning
(#586). Here the simulated years are already contract years.

## 8. Surplus and per-risk covers

A surplus treaty cedes each risk in proportion to the amount by which its sum
insured exceeds a retention. A per-risk excess of loss attaches on each risk,
not each event. Both need losses by risk, which a catastrophe event table does
not have. [`reinsurance-price.md`](reinsurance-price.md) shows `Layer.surplus`
and risk profiles.

## In R

The R twin, [cat-3-contract-terms.qmd](cat-3-contract-terms.qmd), builds the
same contracts with `xol_layer()`, `quota_share()`,
`with_ceding_commission()`, `hours_clause()`, `merge_event_sets()`,
`ceded_by_event()`, `apply_tower()`, `timeline()` and
`apply_tower_timeline()`.

## Further reading

Page 4, [`cat-4-programs.md`](cat-4-programs.md), stacks these contracts into
programs. [reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is
another introduction to property catastrophe reinsurance.
