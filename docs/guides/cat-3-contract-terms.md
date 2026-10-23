# Property catastrophe 3: writing Larkspur's contracts in code

A catastrophe treaty is a function. It takes a year of the cedent's event
losses and returns what the reinsurer pays. The terms of the contract decide
the shape of that function: how pieces of one storm are added up, where each
loss attaches, how much limit a year has, which losses it sees and how the
recovery is shared. This page turns each of Larkspur Re's five contracts into
code, cedent by cedent, from `examples/data/cat_book_contracts.csv`, checks the
terms on small years you can follow by hand, and then runs them on the 2,000
simulated years of the book. The story, the files and the notation are on
[the overview](cat-0-overview.md); the loss tables are introduced on
[page 1](cat-1-loss-tables.md).

## 1. The book and its contracts

The year event loss table (YELT) holds one row per cedent per event piece,
with a day and an hour. `read_ylt` does not read a time column yet
(#594), so this page builds each cedent's event set by hand: one list of
losses per year, the time of each loss as a fraction of the year, and the
peril as a label.

```python
import csv
import math

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


for name, c in CONTRACTS.items():
    print(f"{name:12s} {c['kind']:14s} perils {c['perils']:6s} share {c['share']:5s} hours {c['hours'] or '-'}")
```

```text
alder_cat_1  occurrence_xl  perils all    share 0.3   hours 72
alder_cat_2  occurrence_xl  perils all    share 0.2   hours 72
bracken_cat  occurrence_xl  perils all    share 0.25  hours 72
bracken_agg  aggregate_xl   perils CS+FL  share 0.25  hours -
corran_qs    quota_share    perils all    share 0.1   hours 72
```

Each row of the file becomes one `Layer`. The amounts in the file are on a
100% basis and `share` is Larkspur's signed line, so the layer's `share`
scales every recovery and the premium is the rate on line times the limit
times the share. The function below is the only place the terms are read;
nothing later on the page types a number from the table.

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
for name, layer in LAYERS.items():
    print(f"{name:12s} {layer.limit:6.0f} xs {layer.attachment:4.0f}  annual limit {layer.aggregate_limit:6.0f}"
          f"  premium {layer.premium:.3f}")
```

```text
alder_cat_1      40 xs   30  annual limit     80  premium 1.800
alder_cat_2     100 xs   70  annual limit    200  premium 1.000
bracken_cat      50 xs   25  annual limit    150  premium 1.750
bracken_agg     inf xs    0  annual limit     30  premium 0.450
corran_qs       150 xs    0  annual limit    300  premium 3.000
```

An occurrence layer with one paid reinstatement has an annual limit of twice
its occurrence limit; the aggregate cover has no occurrence limit at all.

## 2. What counts as one loss: the hours clause

A storm can reach the loss table in pieces: a windstorm that crosses the coast
over two days, or a run of hail cells. The hours clause says how many hours of
losses make one **occurrence**. For a layer this matters a lot: two pieces of
25 under 40 xs 30 recover nothing apart and 20 together.

**The rule on this track.** `EventSet.grouped(hours)` starts a window at the
first piece of a year, adds every piece that falls within that many hours of
it into one occurrence, and starts the next window at the next piece. All
perils are grouped together, with the single `hours` value the contracts file
gives each contract. Both are simplifications. A real clause usually lets the
cedent choose where each window starts, to its advantage, so a window fixed
at the first piece can understate recoveries. Many clauses also define an
occurrence per peril, with a longer window for flood (often 168 hours) than
for windstorm. The block below checks how much the per-peril reading changes
on this book.

```python
alder = cedent_events("alder")
alder_72 = alder.grouped(72.0)
print("pieces", sum(alder.counts()), " occurrences", sum(alder_72.counts()))
print("labels before", alder.label_dims, " after", alder_72.label_dims)

def ceded_mean(layer, events):
    res = Tower([layer]).apply(events)
    return res.aggregate(["kind", "layer"]).marginal(["ceded", layer.name]).mean()

cat_1 = LAYERS["alder_cat_1"]
print(f"alder_cat_1 expected loss, pieces {ceded_mean(cat_1, alder):.4f}, 72 hours {ceded_mean(cat_1, alder_72):.4f}")
```

```text
pieces 5391  occurrences 4238
labels before ['peril']  after []
alder_cat_1 expected loss, pieces 0.7778, 72 hours 0.8230
```

Grouping lifts the expected loss by about 6%. It also drops the labels: an
occurrence made of pieces of two perils has no single peril. To group peril
by peril, group each peril on its own, label it again, and merge the perils
back into one set, which keeps the labels and puts each year's occurrences
in time order. Page 4 reuses this helper to split a program by peril.

```python
def grouped_by_peril(cedent, hours):
    out = None
    for p in ["WS", "EQ", "FL", "CS"]:
        g = cedent_events(cedent, p).grouped(hours)
        g = g.with_labels({"peril": [p] * sum(g.counts())})
        out = g if out is None else out.merge(g)
    return out

alder_72p = grouped_by_peril("alder", 72.0)
print("occurrences, grouped peril by peril", sum(alder_72p.counts()), " labels", alder_72p.label_dims)
print(f"alder_cat_1 expected loss {ceded_mean(cat_1, alder_72p):.4f}")
```

```text
occurrences, grouped peril by peril 4275  labels ['peril']
alder_cat_1 expected loss 0.8229
```

A few more occurrences survive, because pieces of different perils that fall
within 72 hours no longer join, and the expected loss barely moves on this
book. So the track keeps the single all-perils clause for pricing, and uses
the per-peril grouping only where it needs the peril labels.

## 3. Alder's layers: occurrence excess of loss with reinstatements

Each occurrence of size `x` recovers `min(max(x - a, 0), l)` from a layer
`l xs a`. Over a year the recoveries add up until the annual limit is used,
which is `l` times one plus the number of reinstatements. The reinsurer's
part is its share of that. Paid reinstatements cost the cedent extra premium;
page 5 prices them.

`ceded_by_event` shows the order. Take `alder_cat_1` (40 xs 30, 30%, one
reinstatement, so 80 of annual limit) and a year of four occurrences: one
below the attachment, one through the top of the layer, and two in the
middle.

```python
year = [28.0, 110.0, 64.0, 95.0]
line = float(CONTRACTS["alder_cat_1"]["share"])
left = cat_1.aggregate_limit
print(" loss  at 100%  limit left  Larkspur's line")
for x, rec in zip(year, cat_1.ceded_by_event(year)):
    left -= rec / line
    print(f"{x:5.0f} {rec / line:8.1f} {max(left, 0.0):11.1f} {rec:16.2f}")
print(f"year: {cat_1.ceded(year):.2f}")
```

```text
 loss  at 100%  limit left  Larkspur's line
   28      0.0        80.0             0.00
  110     40.0        40.0            12.00
   64     34.0         6.0            10.20
   95      6.0         0.0             1.80
year: 24.00
```

The 95 would recover 40 on its own, but only 6 of the year's limit is left.
The losses must be in time order, since the limit runs out on the last
occurrences. The book's years already are: the file lists each year's pieces
by `slot`, which is chronological.

## 4. Corran's quota share: an event limit and an annual limit

A plain quota share cedes a fixed share of every loss. Cat quota shares
usually cap it: an event limit on each occurrence and an annual limit on the
year. `corran_qs` cedes 10% with an event limit of 150 and an annual limit of
300, both on a 100% basis. As a layer, that is 150 xs 0 at a 10% share with an
annual limit of 300. In a year of 200 and 120 the first event is capped at 150,
so the cover pays 10% of 270. `Layer.quota_share` has no limits and pays 10%
of 320.

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

The **subject losses** of a contract are the cedent's losses it applies to:
its perils, grouped by its hours clause, before the contract's own terms.
`bracken_agg` covers only convective storm and flood. The cleanest way to
scope a contract to some perils is to leave the other perils out of the
subject losses before the contract sees them. Here that happens in the data,
with `cedent_events("bracken", "CS+FL")`.

The library also splits results by label, but the split comes after ceding.
Take a year with an earthquake of 25 and then a windstorm of 30 under 20 xs 10
with an annual limit of 20. The earthquake recovers 15 and leaves 5 of limit
for the windstorm. A windstorm-only treaty would have paid the windstorm 20.

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

So a label split answers "how much of this treaty's recovery came from
windstorm", not "what would a windstorm treaty pay". There is no call that
filters an `EventSet` by label or scopes a layer to some labels
(#584).

An aggregate cover looks at the year's total: `share * min(max(S - AD, 0), AL)`
with `S` the sum of the covered losses, `AD` the aggregate deductible and `AL`
the aggregate limit. In prospicio it is a layer with an infinite occurrence
limit, no attachment, and the annual terms. `bracken_agg` is 30 xs 40 in the
aggregate at 25%. A year of 10, 20 and 25 totals 55, so the cover pays a
quarter of 15.

```python
agg = LAYERS["bracken_agg"]
print(agg.ceded([10.0, 20.0, 25.0]))
print("event order", [round(x, 4) for x in agg.ceded_by_event([10.0, 20.0, 25.0])])
agg_pro_rata = Layer("bracken_agg", math.inf, 0.0, share=0.25, aggregate_deductible=40.0,
                     aggregate_limit=30.0, allocation="pro_rata")
print("pro rata   ", [round(x, 4) for x in agg_pro_rata.ceded_by_event([10.0, 20.0, 25.0])])
```

```text
3.75
event order [0.0, 0.0, 3.75]
pro rata    [0.6818, 1.3636, 1.7045]
```

The annual figure is the same either way. The split by event differs: in
event order the deductible eats the first losses and the third event carries
all of the recovery, while `allocation="pro_rata"` scales every event by the
same factor. The split matters only when something downstream reads events,
such as a later inuring stage (page 4).

On the book, the cover sees Bracken's convective storm and flood losses only.

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

The cover attaches in about one year in thirty. An aggregate cover is often
sold to protect a cedent's earnings against a run of medium-sized events, but
this one has no cap on what one event can contribute, so on this book it is
mostly triggered by a single large flood or storm outbreak: in 52 of the 65
years past the deductible, one loss alone is above it. A cedent that wants
frequency protection adds an each-and-every-loss cap, which limits each
event's contribution to the aggregate.

## 6. All five contracts on the book

With the subject losses chosen by perils and grouped by each contract's
hours clause, every contract runs the same way. The loss ratio here is the
expected ceded loss over the upfront premium, before reinstatement premium
and commissions.

```python
def subject(c):
    e = cedent_events(c["cedent"], c["perils"])
    return e.grouped(float(c["hours"])) if c["hours"] else e

for name, c in CONTRACTS.items():
    layer = LAYERS[name]
    el = ceded_mean(layer, subject(c))
    print(f"{name:12s} expected loss {el:7.4f}  premium {layer.premium:6.3f}  loss ratio {el / layer.premium:.3f}")
```

```text
alder_cat_1  expected loss  0.8230  premium  1.800  loss ratio 0.457
alder_cat_2  expected loss  0.4163  premium  1.000  loss ratio 0.416
bracken_cat  expected loss  0.6847  premium  1.750  loss ratio 0.391
bracken_agg  expected loss  0.1621  premium  0.450  loss ratio 0.360
corran_qs    expected loss  1.2025  premium  3.000  loss ratio 0.401
```

## 7. The contract year

A treaty covers losses occurring in a period, or losses on policies written in
a period (risks attaching). prospicio models that with a `Timeline` of
accident months, an `Exposure` that says how much of each month each
underwriting year carries, and a tower with a cover basis. Under losses
occurring, contract years start at month 0 of the timeline. A small example
shows a trap: an 18-month timeline with one underwriting year keeps an event
in month 3 and drops one in month 16 without a word.

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

The second event, 80 in month 16, is in no contract year and so in no
result. There is no way to say that a contract runs from, say, 1 July to
30 June of dated events, or to be warned when an event falls outside every
contract year (#586). Here the simulated years are already contract
years, so the page needs none of this.

## 8. Surplus and per-risk covers

A surplus treaty cedes each risk in proportion to how far its sum insured
exceeds a retention, and a per-risk excess of loss attaches on each risk
rather than each event. Both need losses by risk, which a catastrophe event
table does not have. `Layer.surplus` and risk profiles are covered in
[`reinsurance-price.md`](reinsurance-price.md). What the library lacks for a
cat program is a per-risk layer with an event limit (a cap on the sum of
per-risk recoveries from one event) and a per-risk layer inuring to a cat
layer, so the cat layer sees each event net of the per-risk recoveries
(#585).

## What is missing

Several common cat terms have no direct form in the library:

- a franchise deductible, which pays the whole loss once it passes the
  deductible;
- an nth-event cover, which pays only the second (or third) occurrence of a
  year;
- a top-and-drop layer, which sits on top of a program and drops down when
  the lower layers are exhausted;
- index or parametric triggers and industry loss warranties, which pay on an
  industry figure or a physical measure rather than the cedent's loss.

Each can be built by hand from `ceded_by_event` or from the event losses, but
none is a `Layer` setting (#591; parametric and index triggers for
cat bonds are also part of #570). A no-claims bonus, which page 5
approximates with a sliding scale, is #592. The earlier sections name
the other gaps: a time column for `read_ylt` (#594), filtering by label
(#584), the contract year (#586) and per-risk covers in a
cat program (#585).

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
