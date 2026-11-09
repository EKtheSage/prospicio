# Property catastrophe 4: stacked covers, and who keeps what

<!-- site: execute -->

A cedent seldom buys one layer. It buys a program: several layers on the same
losses, sometimes behind a quota share, sometimes with a peril excluded. A
reinsurer signs a line, a share of each layer, next to other markets, and can
buy its own cover, a retro. Each party sees the same years of losses from a
different position.

This page builds these views on the Larkspur Re book: the two layers of Alder
Mutual together, the quota share of Corran General ahead of a cat cover, the
losses of Bracken split by what its aggregate cover can reach, and a small
retro for Larkspur.

The contract terms come from `examples/data/cat_book_contracts.csv`. Larkspur
does not write two covers on this page, the cat cover of Corran and its own
retro. They are hypothetical. The story, the files and the notation are on
[the overview](cat-0-overview.md). Read [page 3](cat-3-contract-terms.md)
first for a single layer, the hours clause and reinstatements.

## 1. The book and the helpers

The year event loss table (YELT) has one row per cedent per event piece.
`cedent_events` makes an `EventSet` from the rows of one cedent, with event
times and a peril label. `contract_layer` makes a `Layer` from a row of the
contracts file. `subject` applies the peril scope and the hours clause of the
contract.

```python
import csv
import math

from prospicio.aggregate import EventSet
from prospicio.distributions import Sampled
from prospicio.pricing import DealBook, DealRun
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


def num(x, default=None):
    return float(x) if x != "" else default


def contract_layer(c):
    """A row of the contracts file as a Layer: Larkspur's line, premium and terms."""
    share = num(c["share"])
    if c["kind"] == "occurrence_xl":
        limit = num(c["limit"])
        return Layer(c["contract"], limit, num(c["attachment"]), share=share,
                     premium=num(c["rate_on_line"]) * limit * share,
                     reinstatement_rates=[num(c["reinstatement_rate"])] * int(c["reinstatements"]))
    if c["kind"] == "aggregate_xl":
        agg_limit = num(c["aggregate_limit"])
        return Layer(c["contract"], math.inf, 0.0, share=share,
                     aggregate_deductible=num(c["aggregate_deductible"]), aggregate_limit=agg_limit,
                     premium=num(c["rate_on_line"]) * agg_limit * share)
    premium = share * float(CEDENTS[c["cedent"]]["cat_premium"])
    return (Layer(c["contract"], num(c["limit"]), 0.0, share=share,
                  aggregate_limit=num(c["aggregate_limit"]), premium=premium)
            .with_ceding_commission(num(c["ceding_commission"])))


def subject(c):
    """The losses a contract sees: its perils, grouped by its hours clause."""
    e = cedent_events(c["cedent"], c["perils"])
    return e.grouped(float(c["hours"])) if c["hours"] else e


def ceded(res, name):
    """One layer's ceded loss from a tower result."""
    return res.aggregate(["kind", "layer"]).marginal(["ceded", name])


for name, c in CONTRACTS.items():
    print(f"{name:12s} {c['kind']:14s} perils {c['perils']:5s} share {c['share']}")
```

```text
alder_cat_1  occurrence_xl  perils all   share 0.3
alder_cat_2  occurrence_xl  perils all   share 0.2
bracken_cat  occurrence_xl  perils all   share 0.25
bracken_agg  aggregate_xl   perils CS+FL share 0.25
corran_qs    quota_share    perils all   share 0.1
```

## 2. A program: two layers on one subject

Alder buys 40 xs 30 and 100 xs 70. The layers touch: the first layer exhausts
where the second attaches. A `Tower` applies both layers to the same
occurrences in one pass and uses up the reinstatements of each layer in event
order.

Both contracts of Alder have a 72 hour clause and cover all perils, so they
share one subject. `aggregate(["kind"])` adds the layers and leaves four
components.

```python
import pandas as pd
from great_tables import GT

alder = subject(CONTRACTS["alder_cat_1"])
cat_1 = contract_layer(CONTRACTS["alder_cat_1"])
cat_2 = contract_layer(CONTRACTS["alder_cat_2"])
program = Tower([cat_1, cat_2])
res = program.apply(alder)

by_kind = res.aggregate(["kind"])
rows = [(kind, by_kind.marginal([kind]).mean()) for (kind,) in by_kind.components()]
rows += [(f"ceded {name}", ceded(res, name).mean()) for name in program.layer_names]
(GT(pd.DataFrame(rows, columns=["item", "mean"]), rowname_col="item")
 .tab_header("Alder's program, Larkspur's lines (30% and 20%)", "Mean a year, millions")
 .fmt_number("mean", decimals=4))
```

```text
                          mean
gross                  16.7784
ceded                   1.2392
net                    15.5392
reinstatement_premium   0.1410
ceded alder_cat_1       0.8230
ceded alder_cat_2       0.4163
```

Gross is ceded plus net in every year, and thus also in the mean. The ceded
loss here is the line of Larkspur on each layer, not the whole layer (section
4). The net is net of recoveries only. Alder also pays the reinstatement
premium of 0.1410 (page 5).

## 3. What adds up and what does not

Expected losses add across layers. Tail measures do not: the TVaR at 99% of a
sum is at most the sum of the TVaRs, and smaller when the worst years of the
parts differ. For two layers on one subject the worst years overlap a lot,
because a large occurrence fills both, but not fully: two moderate events can
fill the first layer and not touch the second.

```python
c1 = Sampled(ceded(res, "alder_cat_1").draws)
c2 = Sampled(ceded(res, "alder_cat_2").draws)
both = Sampled([a + b for a, b in zip(c1.draws, c2.draws)])
additivity = pd.DataFrame({
    "measure": ["alder_cat_1", "alder_cat_2", "sum of the two", "the program"],
    "mean": [c1.mean(), c2.mean(), c1.mean() + c2.mean(), both.mean()],
    "tvar_99": [c1.tvar(0.99), c2.tvar(0.99), c1.tvar(0.99) + c2.tvar(0.99), both.tvar(0.99)],
})
(GT(additivity, rowname_col="measure")
 .tab_header("Larkspur's ceded loss: the mean adds, the TVaR does not",
             "Millions a year, 2,000 simulated years")
 .cols_label(mean="mean", tvar_99="TVaR 99%")
 .fmt_number("mean", decimals=4)
 .fmt_number("tvar_99", decimals=3))
```

```text
                  mean  TVaR 99%
alder_cat_1     0.8230    13.843
alder_cat_2     0.4163    20.998
sum of the two  1.2392    34.841
the program     1.2392    33.983
```

The gap, 0.858, is a small diversification credit inside the program of one
cedent. Across cedents and perils it is much larger. Page 7
([`cat-7-portfolio.md`](cat-7-portfolio.md)) measures it on the whole book.

## 4. Whose line? Participation

The `share=` of a `Layer` is the line that the modelled party signs. With
`share=0.3`, the tower reports 30% of the layer loss as ceded and all the rest
as net. This "net" is the loss of the cedent less the part of Larkspur only,
not the true retention of the cedent. Other reinsurers take the other 70% of
the layer, and they are not in this model.

To see the program as Alder sees it, model each layer at 100%. A share scales
the layer loss in every year, so the 100% layer is the line divided by the
share.

```python
def at_100(c):
    """The whole layer: the contract's terms without Larkspur's share."""
    return Layer(c["contract"], num(c["limit"]), num(c["attachment"]),
                 reinstatement_rates=[num(c["reinstatement_rate"])] * int(c["reinstatements"]))


full = Tower([at_100(CONTRACTS["alder_cat_1"]), at_100(CONTRACTS["alder_cat_2"])])
alder_view = full.apply(alder).aggregate(["kind"])
views = pd.DataFrame({
    "kind": ["gross", "ceded", "net"],
    "alder": [alder_view.marginal([k]).mean() for k in ["gross", "ceded", "net"]],
    "larkspur": [by_kind.marginal([k]).mean() for k in ["gross", "ceded", "net"]],
})
(GT(views, rowname_col="kind")
 .tab_header("The program at 100% and at Larkspur's lines", "Mean a year, millions")
 .cols_label(alder="Alder, 100%", larkspur="Larkspur's lines")
 .fmt_number(["alder", "larkspur"], decimals=4))
```

```text
       Alder, 100%  Larkspur's lines
gross      16.7784           16.7784
ceded       4.8246            1.2392
net        11.9538           15.5392
```

```python
line = sum(ceded(res, n).mean() / num(CONTRACTS[n]["share"]) for n in program.layer_names)
print(f"Larkspur's lines grossed up to 100%: {line:.4f}")
```

```text
Larkspur's lines grossed up to 100%: 4.8246
```

When a later page adds the deals of Larkspur, the lines of Larkspur add up,
never the 100% layers.

## 5. Inuring: a quota share ahead of a cat cover

Corran General cedes 10% of each occurrence to Larkspur under the quota share
`corran_qs`, with an event limit of 150 and an annual limit of 300. Suppose
that Corran also buys a cat cover of 60 xs 20 for its own retention. The quota
share inures to the benefit of the cat cover: the cat layer sees each
occurrence after the quota share takes its part.

For one occurrence of size `X` and the quota share cession `Q(X)`, the cat
layer cedes

```
C = min(max(X - Q(X) - 20, 0), 60)
```

`Q(X)` is 10% of `X`, capped at the event limit, and zero after the quota
share uses up its annual limit. Thus `Q` also depends on the earlier
occurrences of the year.

`Tower.inuring` takes the stages in order. Each stage sees the losses net of
all earlier stages, event by event, and the stages use up annual limits in
event order.

```python
corran = subject(CONTRACTS["corran_qs"])
qs = contract_layer(CONTRACTS["corran_qs"])
inuring = Tower.inuring([[qs], [Layer("corran_cat", 60.0, 20.0)]]).apply(corran)
stages = pd.DataFrame(
    [(" ".join(comp), inuring.marginal(list(comp)).mean()) for comp in inuring.components()],
    columns=["component", "mean"])
(GT(stages, rowname_col="component")
 .tab_header("Corran: quota share, then cat cover", "Mean a year, millions")
 .fmt_number("mean", decimals=4))
```

```text
                                mean
gross ground_up              13.0490
ceded corran_qs               1.2025
ceded corran_cat              2.6164
net retained                  9.2301
ceding_commission corran_qs   0.7500
```

```python
alone = Tower([Layer("corran_cat", 60.0, 20.0)]).apply(corran)
print(f"cat cover on the gross, no quota share ahead: {ceded(alone, 'corran_cat').mean():.4f}")
```

```text
cat cover on the gross, no quota share ahead: 2.9890
```

The quota share takes slightly less than 10% of the gross (1.2025 against
1.3049) because of its event and annual limits. With the quota share ahead,
the expected loss of the cat cover falls from 2.989 to 2.616, so the cat
reinsurer should charge less. The ceding commission is 25% of the quota share
premium of 3.0, a fixed 0.75 in every year.

## 6. What Bracken's aggregate cover can reach

For Bracken, the aggregate cover `bracken_agg` splits the gross loss into
these amounts:

- **Gross**: all the losses of Bracken.
- **Subject**: the losses that the contract applies to (page 3, section 5),
  here convective storm and flood only.
- **Ceded**: what the whole contract pays, at 100%.
- **Retained**: subject less ceded, what Bracken keeps of the losses that the
  contract can pay.

The gross less the subject is outside the scope of the contract. Windstorm and
earthquake are excluded perils for this cover, and no change to its deductible
or limit can reach them. As in section 4, the page models the contract at
100%, so "retained" is what Bracken keeps, not the subject less the line of
Larkspur.

```python
bc = CONTRACTS["bracken_agg"]
gross = cedent_events("bracken").totals().total()
subj = subject(bc)
whole = Layer(bc["contract"], math.inf, 0.0, aggregate_deductible=num(bc["aggregate_deductible"]),
              aggregate_limit=num(bc["aggregate_limit"]))
agg = Tower([whole]).apply(subj)
subject_mean = subj.totals().total().mean()
ceded_mean = ceded(agg, "bracken_agg").mean()
split = pd.DataFrame({
    "amount": ["gross", "subject", "ceded, 100%", "retained", "outside the scope",
               "Larkspur's 25% of ceded"],
    "mean": [gross.mean(), subject_mean, ceded_mean, subject_mean - ceded_mean,
             gross.mean() - subject_mean, ceded_mean * num(bc["share"])],
})
(GT(split, rowname_col="amount")
 .tab_header("Bracken's aggregate cover: what it can reach", "Mean a year, millions")
 .fmt_number("mean", decimals=4))
```

```text
                            mean
gross                    15.3765
subject                   7.1642
ceded, 100%               0.6484
retained                  6.5158
outside the scope         8.2122
Larkspur's 25% of ceded   0.1621
```

More than half of the loss of Bracken is windstorm and earthquake, outside the
scope of this contract. The occurrence layer `bracken_cat` covers those
losses.

## 7. Splitting a program by peril

With peril labels on the events, the result of a tower has a third dimension,
`peril`, next to `kind` and `layer`. It answers "how much of the ceded loss is
windstorm?". `grouped` removes the labels, so this section uses the
`grouped_by_peril` helper of page 3, section 2, repeated here so that the page
runs on its own.

```python
def grouped_by_peril(cedent, hours):
    out = None
    for p in ["WS", "EQ", "FL", "CS"]:
        g = cedent_events(cedent, p).grouped(hours)
        g = g.with_labels({"peril": [p] * sum(g.counts())})
        out = g if out is None else out.merge(g)
    return out


by_peril = program.apply(grouped_by_peril("alder", 72.0)).aggregate(["kind", "peril"])
perils = ["WS", "EQ", "FL", "CS"]
split_peril = pd.DataFrame({
    "peril": perils,
    "gross": [by_peril.marginal(["gross", p]).mean() for p in perils],
    "ceded": [by_peril.marginal(["ceded", p]).mean() for p in perils],
})
(GT(split_peril, rowname_col="peril")
 .tab_header("Alder's program by peril", "Mean a year, millions; ceded is Larkspur's lines")
 .fmt_number(["gross", "ceded"], decimals=4))
```

```text
     gross   ceded
WS  9.9146  0.8825
EQ  2.4733  0.2800
FL  3.0416  0.0759
CS  1.3489  0.0000
```

```python
print(f"ceded, all perils: {by_peril.aggregate(['kind']).marginal(['ceded']).mean():.4f}")
```

```text
ceded, all perils: 1.2385
```

The figure shows the same split:

```python
import matplotlib.pyplot as plt

xs = range(len(perils))
fig, ax = plt.subplots()
ax.bar([x - 0.2 for x in xs], [by_peril.marginal(["gross", p]).mean() for p in perils],
       width=0.4, color="0.75", label="gross")
ax.bar([x + 0.2 for x in xs], [by_peril.marginal(["ceded", p]).mean() for p in perils],
       width=0.4, label="ceded to Larkspur")
ax.set_xticks(list(xs), perils)
ax.set_xlabel("peril (WS windstorm, EQ earthquake, FL flood, CS convective storm)")
ax.set_ylabel("mean loss a year (millions)")
ax.set_title("Alder's program by peril: gross and ceded loss")
ax.legend()
```

Windstorm drives the program. No convective storm occurrence of Alder reaches
the attachment of 30, even after grouping.

The ceded total, 1.2385, is not quite the 1.2392 of section 2: here a
windstorm and a flood in the same three days stay two occurrences. Page 3,
section 2 explains why the track prices with the all-perils clause and uses
the per-peril grouping only for splits like this one.

## 8. Retro as an aggregate cover

The simplest retro is an aggregate stop loss on the annual total of what
Larkspur wrote. Here Larkspur buys 10 xs 8 on the annual total of two deals:
its lines on `alder_cat_1` and `bracken_cat`.

A **deal** is one contract as Larkspur wrote it: the premium, losses and other
flows of its line, year by year. `DealRun.from_tower` records one layer of a
tower result as a deal, and `DealBook` aligns the deals by year.
`book.amounts()` also holds a `"total"` deal, so select the deals by name.
`Tower.apply_aggregate` then treats the total of the selection in each year as
one aggregate loss.

```python
bc = CONTRACTS["bracken_cat"]
bracken_tower = Tower([contract_layer(bc)])
alder_tower = Tower([cat_1])
book = DealBook([
    DealRun.from_tower("alder_cat_1", alder_tower, alder_tower.apply(alder), "alder_cat_1"),
    DealRun.from_tower("bracken_cat", bracken_tower, bracken_tower.apply(subject(bc)), "bracken_cat"),
])
losses = book.amounts().select("item", ["loss"]).select("deal", book.names)
annual = losses.total()
print(f"book loss mean {annual.mean():.4f}, P(loss > 8) {sum(x > 8 for x in annual.draws) / N_YEARS:.4f}")
retro = Tower([Layer.stop_loss("retro", 10.0, 8.0)]).apply_aggregate(losses)
print(f"retro 10 xs 8 ceded mean {ceded(retro, 'retro').mean():.4f}")
```

```text
book loss mean 1.5077, P(loss > 8) 0.0835
retro 10 xs 8 ceded mean 0.4565
```

In about one year in twelve the two deals lose more than 8, where the retro
starts to pay. Its expected recovery is close to a third of the mean loss of
the deals: the cover sits where these thin, volatile lines put their losses.

For a per-occurrence retro, `Layer.ceded_by_event` gives the ceded amount of each event
for one list of losses (#590).

## In R

The R twin, [cat-4-programs.qmd](cat-4-programs.qmd), runs the same steps with
`reinsurance_tower()`, `inuring_tower()`, `apply_tower()`,
`merge_event_sets()`, `deal_run_from_tower()`, `deal_book()`,
`book_amounts()` and `aggregate_stop_loss()`.

## Further reading

[Page 5](cat-5-beyond-loss.md) adds the flows other than loss: reinstatement
premium, commissions and brokerage. [reinsurance.dev](https://reinsurance.dev)
by Oliver Baltzer is another introduction to property catastrophe
reinsurance.
