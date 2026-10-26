# Property catastrophe 2: how bad can Alder's year get?

Page 1 changed the files of Larkspur Re into simulated years. Before Larkspur
quotes the layers of Alder Mutual, its underwriter wants to know about the bad
years of Alder:

- How bad are the worst 1% of years?
- Is a bad year one very large storm or many smaller events?
- Does the simulation agree with the event table of the vendor?
- How often does a loss reach the first layer of Alder?

This page answers these questions in that order. The story, the files and the
notation are on [the overview](cat-0-overview.md). Money is in millions of
Varland crowns.

## 1. Alder's simulated years

Each page of the track runs on its own. So this page builds the simulated years
of Alder Mutual again from the year event loss table (YELT), with the same
helper as page 1. The helper keeps the time in the year of each loss, which the
hours clause in section 5 needs.

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

On 2,000 simulated years, a quantile is an order statistic, and the definition
sets which one. Hyndman and Fan (1996) list nine definitions of a sample
quantile. prospicio gives two of them through the `quantile=` option of
`statistics` and the `method` of `Sampled.var`:

- `"inverted_cdf"` (the default, their type 1): the smallest draw whose
  empirical cdf reaches `p`. At `N = 2000` and `p = 0.99` that is the 1,980th
  smallest draw, so 20 years lie strictly above it.
- `"linear"` (their type 7, numpy's and R's default): interpolation between
  neighbouring draws.

Some reports use a third rule: the `k`-th largest year, with
`k = floor((1 - p) N)`. Then only `k - 1` years are above it. The block below
reads this year from the sorted draws (#582).

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

TVaR has the same choice of rule. The exact `Sampled.tvar` gives each year a
mass of `1 / N` (as does the default `tvar="exact"` of `statistics`). It splits
one year when `(1 - p) N` is not a whole number. At `N = 2000`, every round
level (99%, 99.5%, 99.6%, 99.8%) gives a whole number of years. So the exact
TVaR is the mean of the worst `k` whole years, and the two differ only at a
level such as 99.33%. `tvar="linear"` integrates the interpolated quantile and
gives a lower value.

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

Always write the rule next to each VaR that you report. On 2,000 years, the
rules here differ by about 1.2, or 0.7%.

## 3. A band of bad years

Some capital and retro work needs the mean loss between two levels, for example
between the 50-year and 100-year points. This **window TVaR** comes from two
ordinary TVaRs (#583). The worst `1 - p0` of years are the window plus the
worst `1 - p1`, so:

```
window(p0, p1) = ((1 - p0) TVaR(p0) - (1 - p1) TVaR(p1)) / (p1 - p0)
```

When the two levels give whole numbers of years, the window TVaR is exactly the
mean of the years with ranks between them.

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

## 4. One large event or many: two exceedance curves

A year can be bad for a cedent in two ways: one very large event, or many
medium events. Cat pricing keeps the two cases separate.

- The **aggregate exceedance probability** (AEP) of an amount `x` is the
  chance that the year's **total** loss is above `x`.
- The **occurrence exceedance probability** (OEP) of `x` is the chance that
  the year's **largest single** loss is above `x`.

On simulated years, both are counts: the share of years with a total (or a
maximum) above `x`. The **return period** of `x` is `1 / p`. So a loss with a
1% annual probability of exceedance is "the 100-year loss". The AEP is always
at least the OEP, because the total of a year is at least its largest loss.

An occurrence layer, for example a 40 xs 30 cat XoL, pays per event. So you
read its attachment from the OEP. An aggregate cover, for example a stop loss,
uses the AEP. `EventSet.totals()` gives the total of each year, and
`EventSet.maxima()` gives the largest loss of each year.

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

The event loss table (ELT) of Alder gives each catalog event a rate and a loss
distribution. With a Poisson count of events, the OEP has a closed form:

```
OEP(x) = 1 - exp(-sum_i rate_i * P(X_i > x))
```

`EventLossTable.oep` calculates it exactly, and `occurrence_pml` inverts it.
The simulated years should agree with it up to sampling error. With 2,000 years
and an exceedance probability `p`, the standard error of the simulated share is
`sqrt(p (1 - p) / 2000)`.

The YELT gives some windstorms and convective storms in pieces a few hours
apart, but the ELT counts each event once. An **occurrence** is what a contract
counts as one loss. On this track, it is all the pieces in the 72 hours that
start at the first piece (page 3 explains the hours clause). Raw maxima see
single pieces, so they understate the OEP. Grouping by 72 hours joins the
pieces again. But it can also join two different catalog events in the same
window, and this increases the grouped OEP.

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

At 30 and 70, the grouped shares are well within one standard error of the ELT.
At 110, the difference of 0.0035 is a little more than one standard error. The
nine joined occurrences above 30 move the share in the same direction. Both
effects are real. The hours clause pays on the joined occurrence. So use the
grouped curve to price a contract, although it is not the curve of the ELT.

## 6. Losses at a return period

In the opposite direction, the curves give the loss at a return period. The
aggregate probable maximum loss (PML) comes from the totals, and the occurrence
PML comes from the maxima. Underwriters call both "PML", so always say which
one. Each is a VaR of section 2, here with the default inverted-cdf rule.

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

In the far tail, the simulated OEP losses are above the OEP losses of the ELT,
for the two reasons of section 5. At the 250-year point, only eight of the
2,000 years are beyond the estimate, so the sampling error is large. Also, the
hours clause joins occurrences, and this increases the losses. More years do
not remove this increase.

## 7. How often is Alder's first layer touched, and used up?

For a layer `l xs a`, an underwriter asks two questions:

- **Attachment probability**: what is the probability that a loss reaches
  the layer? Per occurrence, it is `P(X > a)`, strictly above the attachment.
- **Exhaustion probability**: what is the probability that one loss uses all
  of the limit? Per occurrence, it is `P(X >= a + l)`.

The strict and inclusive comparisons are important on simulated years, where a
loss can be exactly on a boundary. Each probability has a per-occurrence
version and an annual version. With a Poisson count of `lambda` occurrences a
year, the annual probability of at least one loss above `a` is
`1 - exp(-lambda P(X > a))`. This is the OEP at `a` again.

On a severity grid, `occurrence_layer_statistics` gives all of these values
exactly. [aggregate-distributions.md](aggregate-distributions.md) shows how to
make grids. Here the grid is the empirical distribution of the grouped
occurrences of Alder, each rounded up to the next half crown. The count is
Poisson at their simulated rate. The terms come from `cat_book_contracts.csv`.

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

On the simulated years, the counts are short (#581). Per occurrence, count the
grouped losses. Per year, read the grouped maxima. The ELT also gives the
per-occurrence values, from its OEP formula inverted:
`P(X > x) = -ln(1 - OEP(x)) / rate`.

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

The per-occurrence probabilities look different, but they agree. The ELT counts
2.3365 events a year. The grouped trials count 2.119 occurrences (the hours
clause joins pieces, and the YELT has no pieces below its 0.05 floor). At the
same count, the probabilities of the ELT agree with the probabilities of the
trials. The annual probabilities need no rescale, which is one reason to
compare annual values.

The exhaustion probability of the grid is a little higher than that of the
trials. The cause is that rounding up to the next half crown moves a few losses
just below 70 to 70. Here, "exhausted" means that one occurrence used all of
the limit. The annual limit is the limit plus its one reinstatement, 80 at
100%. To use all of it, the layer losses of one year must add up to 80, from
any number of occurrences, which is rarer. Page 3 runs the layer itself.

## 8. From one cedent to the book

Larkspur writes all three cedents. So its tail is the tail of their sum, and
the share of each cedent in that tail is less than its stand-alone TVaR.
[Page 7](cat-7-portfolio.md) measures this on the contracts of Larkspur and
allocates capital from it.

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
