# Property catastrophe 4: stacked covers, and who keeps what

A cedent rarely buys one layer. It buys a program: several layers stacked on
the same losses, sometimes behind a quota share, sometimes with a peril left
out. A reinsurer rarely writes the whole of a layer. It signs a line, a share
of each layer, next to other markets. And the reinsurer may buy its own cover,
a retro, on what it has written. Each party looks at the same years of losses
from a different seat.

This page builds those views on the Larkspur Re book. It applies Alder
Mutual's two layers together, puts Corran General's quota share ahead of a cat
cover, splits Bracken's losses by what its aggregate cover can and cannot
reach, and buys Larkspur a small retro. Larkspur's contract terms come from
`examples/data/cat_book_contracts.csv`; the page does not type them in. The
covers Larkspur does not write (Corran's cat cover and Larkspur's retro) are
hypothetical and stated where they appear. The story, the files and the
notation are on [the overview](cat-0-overview.md); for how a single layer,
the hours clause and reinstatements work, read
[page 3](cat-3-contract-terms.md) first.

## 1. The book and the helpers

The year event loss table (YELT) has one row per cedent per event piece. The
helper below turns one cedent's rows into an `EventSet` with event times and a
peril label. `contract_layer` turns a row of the contracts file into a `Layer`,
and `subject` applies the contract's peril scope and hours clause. The same
helpers appear on every page of the track that needs times.

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

Alder buys 40 xs 30 and 100 xs 70. The layers touch: the first exhausts where
the second attaches. A `Tower` applies both to the same occurrences in one
pass, so each year's events are cut into the pieces every layer takes, and the
reinstatements of each layer are used up in event order.

Both of Alder's contracts carry a 72 hour clause and cover all perils, so they
share one subject. `aggregate(["kind"])` adds the layers up and leaves four
components: the gross loss, the ceded loss, the net loss and the reinstatement
premium.

```python
alder = subject(CONTRACTS["alder_cat_1"])
cat_1 = contract_layer(CONTRACTS["alder_cat_1"])
cat_2 = contract_layer(CONTRACTS["alder_cat_2"])
program = Tower([cat_1, cat_2])
res = program.apply(alder)

by_kind = res.aggregate(["kind"])
for (kind,) in by_kind.components():
    print(f"{kind:22s} mean {by_kind.marginal([kind]).mean():8.4f}")
for name in program.layer_names:
    print(f"ceded {name:16s} mean {ceded(res, name).mean():8.4f}")
```

```text
gross                  mean  16.7784
ceded                  mean   1.2392
net                    mean  15.5392
reinstatement_premium  mean   0.1410
ceded alder_cat_1      mean   0.8230
ceded alder_cat_2      mean   0.4163
```

Gross is ceded plus net, in every year and so in the mean. The ceded line here
is Larkspur's line on each layer (30% and 20%), not the whole layer. Section 4
comes back to that. The net is net of recoveries only: the reinstatement
premium of 0.1410 is a cost Alder pays on top (page 5).

## 3. What adds up and what does not

Expected losses add across layers: 0.8230 and 0.4163 make the 1.2392 above
(the last digit is rounding). Tail measures do not. The TVaR at 99% of a sum is at most the sum of the TVaRs, and it is
smaller whenever the worst years of the parts are not all the same years.

For two layers stacked on one subject the worst years overlap a lot, because
a large occurrence fills both. They do not overlap fully. A year can fill the
first layer with two moderate events and leave the second untouched.

```python
c1 = Sampled(ceded(res, "alder_cat_1").draws)
c2 = Sampled(ceded(res, "alder_cat_2").draws)
both = Sampled([a + b for a, b in zip(c1.draws, c2.draws)])
print(f"TVaR 99 alder_cat_1      {c1.tvar(0.99):7.3f}")
print(f"TVaR 99 alder_cat_2      {c2.tvar(0.99):7.3f}")
print(f"sum of the two           {c1.tvar(0.99) + c2.tvar(0.99):7.3f}")
print(f"TVaR 99 of the program   {both.tvar(0.99):7.3f}")
```

```text
TVaR 99 alder_cat_1       13.843
TVaR 99 alder_cat_2       20.998
sum of the two            34.841
TVaR 99 of the program    33.983
```

The gap, 0.858, is a small diversification credit inside one cedent's
program. Across cedents and perils it is much larger; page 7
([`cat-7-portfolio.md`](cat-7-portfolio.md)) measures it on the whole book.

## 4. Whose line? Participation

A `share=` on a `Layer` is the line the modelled party signs. With
`share=0.3` the tower reports 30% of the layer's loss as ceded and everything
else as net. That "net" is then the cedent's loss less Larkspur's part only,
not the cedent's true retention: the other 70% of the layer is placed with
other reinsurers, who do not appear in this model.

To see the program from Alder's seat, model each layer at 100%. Because a
share scales a layer's loss in every year, the 100% layer is the line divided
by the share, and the cedent's retention follows.

```python
def at_100(c):
    """The whole layer: the contract's terms without Larkspur's share."""
    return Layer(c["contract"], num(c["limit"]), num(c["attachment"]),
                 reinstatement_rates=[num(c["reinstatement_rate"])] * int(c["reinstatements"]))


full = Tower([at_100(CONTRACTS["alder_cat_1"]), at_100(CONTRACTS["alder_cat_2"])])
alder_view = full.apply(alder).aggregate(["kind"])
for kind in ["gross", "ceded", "net"]:
    print(f"Alder's {kind:6s} mean {alder_view.marginal([kind]).mean():8.4f}")
line = sum(ceded(res, n).mean() / num(CONTRACTS[n]["share"]) for n in program.layer_names)
print(f"Larkspur's lines grossed up to 100%: {line:.4f}")
```

```text
Alder's gross  mean  16.7784
Alder's ceded  mean   4.8246
Alder's net    mean  11.9538
Larkspur's lines grossed up to 100%: 4.8246
```

So Alder keeps about 12.0 of its 16.8 a year, and Larkspur carries 1.24 of
the 4.82 that Alder places. When a page later rolls up Larkspur's deals, it is
always Larkspur's lines that add up, never the 100% layers.

## 5. Inuring: a quota share ahead of a cat cover

Corran General cedes 10% of every occurrence to Larkspur under the quota share
`corran_qs`, with an event limit of 150 and an annual limit of 300. Suppose
Corran also buys a cat cover of 60 xs 20 for its own retention. The quota
share inures to the benefit of the cat cover: the cat layer sees each
occurrence after the quota share has taken its part.

In symbols, for one occurrence of size `X`, with the quota share's cession
`Q(X)`, the cat layer cedes

```
C = min(max(X - Q(X) - 20, 0), 60)
```

Here `Q(X)` is what the quota share actually ceded on that occurrence: 10% of
`X` capped at the event limit, and nothing once the quota share's annual limit
is used up, so `Q` depends on the earlier occurrences of the year too.

`Tower.inuring` takes the stages in order. Each stage sees the losses net of
all earlier stages, event by event, and annual limits are used up in event
order.

```python
corran = subject(CONTRACTS["corran_qs"])
qs = contract_layer(CONTRACTS["corran_qs"])
inuring = Tower.inuring([[qs], [Layer("corran_cat", 60.0, 20.0)]]).apply(corran)
for comp in inuring.components():
    print(f"{comp[0]:18s} {comp[1]:10s} mean {inuring.marginal(list(comp)).mean():8.4f}")
alone = Tower([Layer("corran_cat", 60.0, 20.0)]).apply(corran)
print(f"cat cover on the gross, no quota share ahead: {ceded(alone, 'corran_cat').mean():.4f}")
```

```text
gross              ground_up  mean  13.0490
ceded              corran_qs  mean   1.2025
ceded              corran_cat mean   2.6164
net                retained   mean   9.2301
ceding_commission  corran_qs  mean   0.7500
cat cover on the gross, no quota share ahead: 2.9890
```

The quota share takes slightly less than 10% of the gross (1.2025 against
1.3049) because of its event and annual limits. With the quota share ahead,
the cat cover's expected loss falls from 2.989 to 2.616, so the cat reinsurer
should charge less. The ceding commission is 25% of the quota share premium of
3.0, a fixed 0.75 in every year.

## 6. What Bracken's aggregate cover can reach

From Bracken's seat, its aggregate cover `bracken_agg` splits the gross loss
into four amounts:

- **Gross**: everything Bracken loses.
- **Subject**: the losses the contract applies to (page 3, section 5): here
  convective storm and flood only.
- **Ceded**: what the whole contract pays, at 100%.
- **Retained**: subject less ceded, what Bracken keeps of the losses the
  contract could have paid.

The gross less the subject is outside the contract's scope: windstorm and
earthquake are excluded perils for this cover, and no change to its
deductible or limit would reach them. As in section 4, the contract is
modelled at 100% so that "retained" is what Bracken keeps, not the subject
less Larkspur's line.

```python
bc = CONTRACTS["bracken_agg"]
gross = cedent_events("bracken").totals().total()
subj = subject(bc)
whole = Layer(bc["contract"], math.inf, 0.0, aggregate_deductible=num(bc["aggregate_deductible"]),
              aggregate_limit=num(bc["aggregate_limit"]))
agg = Tower([whole]).apply(subj)
subject_mean = subj.totals().total().mean()
ceded_mean = ceded(agg, "bracken_agg").mean()
print(f"gross                    {gross.mean():7.4f}")
print(f"subject                  {subject_mean:7.4f}")
print(f"ceded, 100%              {ceded_mean:7.4f}")
print(f"retained                 {subject_mean - ceded_mean:7.4f}")
print(f"outside the scope        {gross.mean() - subject_mean:7.4f}")
print(f"Larkspur's 25% of ceded  {ceded_mean * num(bc['share']):7.4f}")
```

```text
gross                    15.3765
subject                   7.1642
ceded, 100%               0.6484
retained                  6.5158
outside the scope         8.2122
Larkspur's 25% of ceded   0.1621
```

More than half of Bracken's loss is windstorm and earthquake, outside this
contract's scope. Those losses are for `bracken_cat`, the occurrence layer,
to cover.

## 7. Splitting a program by peril

With peril labels on the events, a tower's result carries a third dimension,
`peril`, next to `kind` and `layer`. That is how to answer "how much of the
ceded loss is windstorm?". `grouped` drops the labels, so this section uses
the `grouped_by_peril` helper of page 3, section 2, repeated here so that the
page runs on its own.

```python
def grouped_by_peril(cedent, hours):
    out = None
    for p in ["WS", "EQ", "FL", "CS"]:
        g = cedent_events(cedent, p).grouped(hours)
        g = g.with_labels({"peril": [p] * sum(g.counts())})
        out = g if out is None else out.merge(g)
    return out


by_peril = program.apply(grouped_by_peril("alder", 72.0)).aggregate(["kind", "peril"])
for p in ["WS", "EQ", "FL", "CS"]:
    print(f"{p}  gross {by_peril.marginal(['gross', p]).mean():8.4f}"
          f"   ceded {by_peril.marginal(['ceded', p]).mean():7.4f}")
print(f"ceded, all perils: {by_peril.aggregate(['kind']).marginal(['ceded']).mean():.4f}")
```

```text
WS  gross   9.9146   ceded  0.8825
EQ  gross   2.4733   ceded  0.2800
FL  gross   3.0416   ceded  0.0759
CS  gross   1.3489   ceded  0.0000
ceded, all perils: 1.2385
```

Windstorm drives the program. No convective storm occurrence for Alder
reaches the attachment of 30, even after grouping.

The ceded total, 1.2385, is not quite section 2's 1.2392, because here a
windstorm and a flood in the same three days stay two occurrences. Page 3,
section 2 explains why the track prices with the all-perils clause and uses
the per-peril grouping only for splits like this one.

## 8. Retro as an aggregate cover

Larkspur can protect its own book. The simplest retro is an aggregate stop
loss on the annual total of what Larkspur has written. Here Larkspur buys
10 xs 8 on the annual total of two deals, its lines on `alder_cat_1` and
`bracken_cat`.

A **deal** is one contract as Larkspur has written it: its line's premium,
losses and other flows, year by year. `DealRun.from_tower` records one layer
of a tower result as a deal, and
`DealBook` lines the deals up year by year. `book.amounts()` holds every item
of every deal and also a `"total"` deal; select the deals by name so the total
is not counted twice. `Tower.apply_aggregate` then treats each year's total of
the selection as one aggregate loss.

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

The two deals lose 1.51 a year on average, and in about one year in twelve
they lose more than 8, where the retro starts to pay. Its expected recovery is
0.4565 a year, close to a third of the deals' mean loss, because the cover sits
exactly where these thin, volatile lines put their losses.

An occurrence retro, a per-event cover on Larkspur's own ceded losses, needs
the ceded amount of every event, across deals, in time order. A tower result
gives each layer's annual totals, not its losses by event, so prospicio cannot
yet feed one book's ceded events into another tower (#590). `Layer.ceded_by_event` gives the
split for one list of losses by hand.

## 9. Summary

- A `Tower` applies a program in one pass; `aggregate(["kind"])` gives gross,
  ceded and net, and they add up year by year.
- Means add across layers and deals. TVaR does not.
- `share=` is the modelled party's line. Model at 100% for the cedent's view.
- `Tower.inuring` puts a quota share ahead of a cat cover; the order changes
  the cat layer's loss.
- Gross less subject is outside the contract's scope, and no deductible or
  limit reaches it.
- To split a grouped subject by peril, group each peril, label it and merge.
- A retro on the book's annual total is `apply_aggregate` on the deals' loss.

## What is missing

- the ceded loss of each event as an event set, to feed an occurrence retro
  (#590)

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
