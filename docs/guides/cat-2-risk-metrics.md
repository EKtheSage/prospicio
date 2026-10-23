# Property catastrophe 2: how bad can Alder's year get?

Page 1 turned Larkspur Re's files into simulated years. Before Larkspur quotes
Alder Mutual's layers, its underwriter wants a feel for Alder's bad years: how
bad the worst 1% of years are, whether a bad year is one huge storm or a pile
of smaller ones, whether the simulation agrees with the vendor's event table,
and how often Alder's first layer would be touched. This page answers those
questions in that order. The story, the files and the notation are on
[the overview](cat-0-overview.md). Money is in millions of Varland crowns.

## 1. Alder's simulated years

Each page of the track runs on its own, so this one rebuilds Alder Mutual's
simulated years from the year event loss table (YELT) with the same helper as
page 1. It keeps each loss's time in the year, which the hours clause in
section 5 needs.

```python
import csv
import math

import numpy as np

from prospicio.aggregate import EventLossTable, EventSet
from prospicio.distributions import Grid, Poisson, Sampled
from prospicio.reinsurance import occurrence_layer_statistics
from prospicio.risk import statistics

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


alder = cedent_events("alder")
rows = read("cat_book_elt_alder.csv")
col = lambda k: [float(r[k]) for r in rows]
elt = EventLossTable(col("rate"), col("mean"), col("sd_independent"), col("sd_correlated"), col("exposure"))
print(f"years {alder.n_sims}, losses {sum(alder.counts())}, mean annual loss {alder.totals().mean():.3f}")
print(f"ELT: {len(elt)} events, {elt.rate:.4f} a year, AAL {elt.expected_annual_loss:.3f}")
```

```text
years 2000, losses 5391, mean annual loss 16.778
ELT: 299 events, 2.3365 a year, AAL 16.056
```

## 2. The worst years: VaR and TVaR

The **expected loss** (`EL`, here the AAL) is the mean of the annual totals.
**VaR** at level `p` is the `p`-quantile of the annual total. **TVaR** at `p`
is the mean of the total over the worst `1 - p` of years.

On 2,000 simulated years a quantile is an order statistic, and which one is a
matter of definition. Hyndman and Fan (1996) list nine sample quantile
definitions; prospicio offers two of them through the `quantile=` option of
`statistics` and the `method` of `Sampled.var`:

- `"inverted_cdf"` (the default, their type 1): the smallest draw whose
  empirical cdf reaches `p`. At `N = 2000` and `p = 0.99` that is the 1,980th
  smallest draw, so 20 years lie strictly above it.
- `"linear"` (their type 7, numpy's and R's default): interpolation between
  neighbouring draws.

Some reports use a third rule: with `k = floor((1 - p) N)`, the `k`-th
largest year, so that only `k - 1` years lie above it. prospicio has no
option for it yet, so the block below reads it off the sorted draws.

```python
total = alder.totals().total()
d = sorted(total.draws)
N = len(d)
k = math.floor((1 - 0.99) * N)
print(f"k = {k}")
print(f"k-th largest year       {d[N - k]:.3f}")
print(f"Sampled.var (inv. cdf)  {Sampled(d).var(0.99):.3f}")
print(f"np.quantile (linear)    {float(np.quantile(d, 0.99)):.3f}")
print(f"statistics, linear      {statistics(total, [99.0], quantile='linear')['Percentile 99']:.3f}")
```

```text
k = 20
k-th largest year       177.979
Sampled.var (inv. cdf)  176.824
np.quantile (linear)    176.836
statistics, linear      176.836
```

TVaR has the same choice. The exact `Sampled.tvar` (and `statistics`'
default `tvar="exact"`) gives each year mass `1 / N` and splits one year
when `(1 - p) N` is not a whole number. At `N = 2000` every round level (99%,
99.5%, 99.6%, 99.8%) leaves a whole number of years, so the exact TVaR is the
mean of the worst `k` whole years. They part only at a level such as 99.33%.
`tvar="linear"` integrates the interpolated quantile instead, and runs lower.

```python
print(f"Sampled.tvar (exact)      {Sampled(d).tvar(0.99):.3f}")
print(f"mean of the worst {k} years {sum(d[N - k:]) / k:.3f}")
print(f"statistics, tvar linear   {statistics(total, [99.0], tvar_levels=[99.0], tvar='linear')['TVaR 99']:.3f}")
table = statistics(total, [90.0, 99.0], tvar_levels=[99.0])
for name in ("Mean", "SD", "Percentile 90", "Percentile 99", "TVaR 99"):
    print(f"{name:14s} {table[name]:8.3f}")
```

```text
Sampled.tvar (exact)      255.969
mean of the worst 20 years 255.969
statistics, tvar linear   246.942
Mean             16.778
SD               35.158
Percentile 90    42.244
Percentile 99   176.824
TVaR 99         255.969
```

Whatever the rule, write it next to every VaR you report: on 2,000 years the
rules here differ by about 1.2, or 0.7%.

## 3. A band of bad years

Some capital and retro work asks for the mean loss between two levels, say
between the 50-year and 100-year points, rather than beyond one. This
**window TVaR** follows from two ordinary TVaRs. The worst `1 - p0` of years
is the window plus the worst `1 - p1`, so

```
window(p0, p1) = ((1 - p0) TVaR(p0) - (1 - p1) TVaR(p1)) / (p1 - p0)
```

With whole numbers of years at both levels it is exactly the mean of the
years ranked between them.

```python
p0, p1 = 0.98, 0.99
window = ((1 - p0) * Sampled(d).tvar(p0) - (1 - p1) * Sampled(d).tvar(p1)) / (p1 - p0)
print(f"window TVaR 98% to 99%: {window:.3f}")
print(f"mean of years ranked 21 to 40 from the top: {sum(d[N - 40:N - 20]) / 20:.3f}")
```

```text
window TVaR 98% to 99%: 151.818
mean of years ranked 21 to 40 from the top: 151.818
```

There is no window TVaR call yet (#583).

## 4. One large event or many: two exceedance curves

A year can hurt a cedent in two ways. One very large event can hit it, or
many middling ones can pile up. Cat pricing keeps the two apart.

- The **aggregate exceedance probability** (AEP) of an amount `x` is the
  chance that the year's **total** loss is above `x`.
- The **occurrence exceedance probability** (OEP) of `x` is the chance that
  the year's **largest single** loss is above `x`.

On simulated years both are plain counts: the share of years whose total (or
whose maximum) is above `x`. The **return period** of `x` is `1 / p`, so a
loss with a 1% annual exceedance chance is "the 100-year loss". The AEP is
always at least the OEP, since a year's total is at least its largest loss.

An occurrence layer, such as a 40 xs 30 cat XoL, pays per event, so its
attachment is read off the OEP. An aggregate cover, such as a stop loss,
reads the AEP. `EventSet.totals()` gives each year's total and
`EventSet.maxima()` each year's largest loss.

```python
largest = alder.maxima().total()
for x in (30.0, 70.0, 110.0):
    aep = sum(v > x for v in total.draws) / N_YEARS
    oep = sum(v > x for v in largest.draws) / N_YEARS
    print(f"x {x:5.0f}: AEP {aep:.4f}  OEP {oep:.4f}  OEP return period {1 / oep:5.1f} years")
```

```text
x    30: AEP 0.1465  OEP 0.1070  OEP return period   9.3 years
x    70: AEP 0.0535  OEP 0.0345  OEP return period  29.0 years
x   110: AEP 0.0235  OEP 0.0190  OEP return period  52.6 years
```

## 5. Does the simulation agree with the event table?

Alder's event loss table (ELT) gives each catalog event a rate and a loss
distribution. Under a Poisson count of events, the OEP has a closed form:

```
OEP(x) = 1 - exp(-sum_i rate_i * P(X_i > x))
```

`EventLossTable.oep` evaluates it exactly, and `occurrence_pml` inverts it.
The simulated years should agree with it up to sampling error: with 2,000
years and an exceedance chance `p`, the standard error of the simulated
share is `sqrt(p (1 - p) / 2000)`.

There is a catch. The YELT delivers some windstorms and convective storms in
pieces a few hours apart, while the ELT counts each event once. An
**occurrence** is what a contract counts as one loss: on this track, all the
pieces that fall within 72 hours of the first one (page 3 explains the hours
clause). Raw maxima see single pieces, so they understate the OEP. Grouping
by 72 hours puts the pieces back together, but it can also join two
different catalog events that happen to fall within the window, which pushes
the grouped OEP up.

```python
grouped = alder.grouped(72.0)
largest_grouped = grouped.maxima().total()
print("    x    ELT    raw  grouped  s.e.")
for x in (30.0, 70.0, 110.0):
    raw = sum(v > x for v in largest.draws) / N_YEARS
    grp = sum(v > x for v in largest_grouped.draws) / N_YEARS
    se = math.sqrt(elt.oep(x) * (1 - elt.oep(x)) / N_YEARS)
    print(f"{x:5.0f} {elt.oep(x):.4f} {raw:.4f}  {grp:.4f}  {se:.4f}")
print(f"ELT 100-year occurrence loss: {elt.occurrence_pml(0.01):.2f}")

by_year = [[] for _ in range(N_YEARS)]
for r in YELT:
    if r["cedent"] == "alder":
        hour = (int(r["day"]) - 1) * 24 + int(r["hour"])
        by_year[int(r["year"]) - 1].append((hour, r["event_id"], float(r["loss"])))
windows = joined = joined_above_30 = 0
for pieces in by_year:
    pieces.sort()
    i = 0
    while i < len(pieces):
        j = i
        while j < len(pieces) and pieces[j][0] - pieces[i][0] < 72:
            j += 1
        windows += 1
        if len({p[1] for p in pieces[i:j]}) > 1:
            joined += 1
            joined_above_30 += sum(p[2] for p in pieces[i:j]) > 30
        i = j
print(f"occurrences {windows}, joining two or more catalog events {joined}, of which above 30: {joined_above_30}")
```

```text
    x    ELT    raw  grouped  s.e.
   30 0.1136 0.1070  0.1150  0.0071
   70 0.0374 0.0345  0.0375  0.0042
  110 0.0180 0.0190  0.0215  0.0030
ELT 100-year occurrence loss: 147.94
occurrences 4238, joining two or more catalog events 87, of which above 30: 9
```

At 30 and 70 the grouped shares are well within one standard error of the
ELT. At 110 the gap, 0.0035, is a little more than one standard error, and
the nine joined occurrences above 30 push the same way. Both effects are
real: the hours clause pays on the joined occurrence, so the grouped curve is
the right one for pricing a contract, even though it is not the ELT's curve.

## 6. Losses at a return period

Read the other way, the curves give the loss at a return period: the
aggregate probable maximum loss (PML, from the totals) and the occurrence PML
(from the maxima). Underwriters call both "PML"; always say which one. Each
is a VaR of section 2, here with the default inverted-cdf rule.

```python
print(" RP   AEP loss   OEP loss  ELT OEP loss")
for rp in (10, 50, 100, 250):
    p = 1 - 1 / rp
    print(f"{rp:3d} {Sampled(total.draws).var(p):10.3f} {Sampled(largest_grouped.draws).var(p):10.3f} {elt.occurrence_pml(1 / rp):13.3f}")
```

```text
 RP   AEP loss   OEP loss  ELT OEP loss
 10     42.244     33.971        33.648
 50    125.246    120.464       103.432
100    176.824    163.418       147.938
250    232.741    223.450       199.470
```

The simulated OEP losses sit above the ELT's in the far tail, for the two
reasons of section 5. At the 250-year point only eight of the 2,000 years lie
beyond the estimate, so sampling error is large, and occurrences joined by
the hours clause add an upward push that more years would not remove.

## 7. How often is Alder's first layer touched, and used up?

For a layer `l xs a` an underwriter asks two quick questions:

- **Attachment probability**: how likely is a loss to reach the layer?
  Per occurrence that is `P(X > a)`, strictly above the attachment.
- **Exhaustion probability**: how likely is one loss to use the whole
  limit? Per occurrence that is `P(X >= a + l)`.

The strict and inclusive comparisons matter on simulated years, where a loss
can sit exactly on a boundary. Each probability has a per-occurrence and an
annual version. Under a Poisson count with `lambda` occurrences a year, the
annual chance of at least one loss above `a` is `1 - exp(-lambda P(X > a))`,
which is the OEP at `a` again.

On a severity grid, `occurrence_layer_statistics` gives all of these exactly
(see [aggregate-distributions.md](aggregate-distributions.md) for building
grids). Here the grid is the empirical distribution of Alder's grouped
occurrences, each rounded up to the next half crown, with a Poisson count at
their simulated rate. The terms come from `cat_book_contracts.csv`.

```python
c = CONTRACTS["alder_cat_1"]
a, l, share = float(c["attachment"]), float(c["limit"]), float(c["share"])
occ = [x for i in range(N_YEARS) for x in grouped.events(i)]
lam = len(occ) / N_YEARS
step = 0.5
probs = [0.0] * (math.ceil(max(occ) / step) + 2)
for x in occ:
    probs[math.ceil(x / step)] += 1 / len(occ)
stats = occurrence_layer_statistics(Poisson(lam), Grid(step, probs), share, l, a)
print(f"alder_cat_1: {l:.0f} xs {a:.0f}, {lam:.3f} occurrences a year")
print(f"grid:   P(attach) {stats.pr_attach:.4f}  P(exhaust) {stats.pr_detach:.4f}  P(layer hit in a year) {stats.pr_loss:.4f}")
```

```text
alder_cat_1: 40 xs 30, 2.119 occurrences a year
grid:   P(attach) 0.0569  P(exhaust) 0.0184  P(layer hit in a year) 0.1135
```

On the simulated years themselves there is no dedicated call, but the counts
are short: per occurrence, count the grouped losses; per year, read the
grouped maxima. The ELT gives the per-occurrence values too, by inverting its
OEP formula: `P(X > x) = -ln(1 - OEP(x)) / rate`.

```python
elt_attach = -math.log(1 - elt.oep(a)) / elt.rate
elt_exhaust = -math.log(1 - elt.oep(a + l)) / elt.rate
print(f"trials, per occurrence: P(attach) {sum(x > a for x in occ) / len(occ):.4f}  P(exhaust) {sum(x >= a + l for x in occ) / len(occ):.4f}")
print(f"trials, per year:       P(attach) {sum(x > a for x in largest_grouped.draws) / N_YEARS:.4f}  P(exhaust) {sum(x >= a + l for x in largest_grouped.draws) / N_YEARS:.4f}")
print(f"ELT, per occurrence:    P(attach) {elt_attach:.4f}  P(exhaust) {elt_exhaust:.4f}")
print(f"ELT, per {lam:.3f} occ.:    P(attach) {elt_attach * elt.rate / lam:.4f}  P(exhaust) {elt_exhaust * elt.rate / lam:.4f}")
print(f"ELT, per year:          P(attach) {elt.oep(a):.4f}  P(exhaust) {elt.oep(a + l):.4f}")
```

```text
trials, per occurrence: P(attach) 0.0569  P(exhaust) 0.0179
trials, per year:       P(attach) 0.1150  P(exhaust) 0.0375
ELT, per occurrence:    P(attach) 0.0516  P(exhaust) 0.0163
ELT, per 2.119 occ.:    P(attach) 0.0569  P(exhaust) 0.0180
ELT, per year:          P(attach) 0.1136  P(exhaust) 0.0374
```

The per-occurrence chances look different but are not: the ELT counts 2.3365
events a year, the grouped trials 2.119 occurrences (pieces joined, and
pieces below the YELT's 0.05 floor dropped). Rescaled to the same count, the
ELT's chances agree with the trials'. The annual chances need no rescaling,
which is one reason to compare annual figures.

The grid's exhaustion chance is a little higher than the trials' because
rounding up to the next half crown lifts a few losses just below 70 onto it.
"Exhausted" here means one occurrence used the full limit. Using up the
annual limit (the limit plus its one reinstatement, 80 at 100%) takes layer
losses that add up to 80 in one year, from any number of occurrences, which
is rarer. Page 3 runs the layer itself. A call that returns these
probabilities from simulated years is missing (#581).

## 8. From one cedent to the book

Larkspur writes all three cedents, so its own tail is the tail of their sum,
and each cedent's share of that tail is less than its stand-alone TVaR.
[Page 7](cat-7-portfolio.md) measures this on Larkspur's own contracts and
allocates capital from it.

## What is missing

- the k-th largest VaR and a TVaR of whole years as options (#582)
- a window TVaR (#583)
- attachment and exhaustion probabilities on simulated years (#581)

## In R

The R twin, [cat-2-risk-metrics.qmd](cat-2-risk-metrics.qmd), makes the same
calls: `annual_maxima()`, `hours_clause()`, `elt_oep()`,
`elt_occurrence_pml()`, `VaR()`, `TVaR()`, `statistics()` and
`occurrence_layer_statistics()`.

## Further reading

- Hyndman, R. J. and Fan, Y. (1996), "Sample quantiles in statistical
  packages", *The American Statistician* 50(4), 361-365.
- [Risk measures](pir-risk-measures.md) for VaR, TVaR and their properties.
- [reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer, another
  introduction to property catastrophe reinsurance.
