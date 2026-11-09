# Property catastrophe 2: how bad can Alder's year get?

<!-- site: execute -->

Page 1 changed the files of Larkspur Re into simulated years. Before Larkspur
quotes the layers of Alder Mutual, its underwriter wants to know about the bad
years of Alder:

- How bad are the worst 1% of years?
- Is a bad year one very large storm or many smaller events?
- Does the simulation agree with the event table of the vendor?
- How often does a loss reach the first layer of Alder?

The story, the files and the notation are on [the overview](cat-0-overview.md).
Money is in millions of Varland crowns.

## 1. Alder's simulated years

This page builds the simulated years of Alder Mutual again from the year event
loss table (YELT), with the helper of page 1. The helper keeps the time of each
loss, which the hours clause in section 5 needs.

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
sets which one. Hyndman and Fan (1996) list nine definitions. prospicio gives two, through the `quantile=`
option of `statistics` and the `method` of `Sampled.var`:

- `"inverted_cdf"` (the default, their type 1): the smallest draw whose
  empirical cdf reaches `p`. At `N = 2000` and `p = 0.99` that is the 1,980th
  smallest draw, so 20 years lie strictly above it.
- `"linear"` (their type 7, numpy's and R's default): interpolation between
  neighbouring draws.

Some reports use a third rule: the `k`-th largest year, with
`k = floor((1 - p) N)`, so only `k - 1` years are above it (#582).

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

TVaR has the same choice. The exact `Sampled.tvar` gives each year a
mass of `1 / N` (as does the default `tvar="exact"` of `statistics`). It splits
one year when `(1 - p) N` is not a whole number. At `N = 2000`, every round
level (99%, 99.5%, 99.6%, 99.8%) gives a whole number of years, so the exact
TVaR is the mean of the worst `k` years (the two differ only at a level such
as 99.33%). `tvar="linear"` integrates the interpolated quantile and gives a
lower value.

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

Name the rule with each VaR that you report: here the rules differ by about 1.2,
or 0.7%.

## 3. A band of bad years

Capital and retro work sometimes needs the mean loss between two levels, for
example the 50-year and 100-year points. This **window TVaR** comes from two
ordinary TVaRs (#583). The worst `1 - p0` of years are the window plus the
worst `1 - p1`, so:

```
window(p0, p1) = ((1 - p0) TVaR(p0) - (1 - p1) TVaR(p1)) / (p1 - p0)
```

For whole numbers of years, this is the mean of the years ranked between the
two levels.

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

A year can be bad in two ways: one very large event, or many medium events.

- The **aggregate exceedance probability** (AEP) of an amount `x` is the
  chance that the year's **total** loss is above `x`.
- The **occurrence exceedance probability** (OEP) of `x` is the chance that
  the year's **largest single** loss is above `x`.

On simulated years, both are shares of years. The **return period** of `x` is
`1 / p`, so a loss with a 1% annual probability is "the 100-year loss". The AEP
is at least the OEP, because a year's total is at least its largest loss.

An occurrence layer, for example a 40 xs 30 cat XoL, pays per event, so read it
from the OEP. An aggregate cover, for example a stop loss, uses the AEP.
`EventSet.totals()` gives the total of each year, and `EventSet.maxima()` gives the largest loss of each year.

```python
import pandas as pd
from great_tables import GT

largest = alder.maxima().total()
levels = [30.0, 70.0, 110.0]
aep = [sum(v > x for v in total.draws) / N_YEARS for x in levels]
oep = [sum(v > x for v in largest.draws) / N_YEARS for x in levels]
curve = pd.DataFrame({
    "x": [int(x) for x in levels],
    "AEP": aep,
    "OEP": oep,
    "OEP return period": [1 / p for p in oep],
})
(GT(curve, rowname_col="x")
 .tab_header("Alder: exceedance probabilities", "x in millions of crowns; return period in years")
 .fmt_number(["AEP", "OEP"], decimals=4)
 .fmt_number("OEP return period", decimals=1))
```

```text
        AEP     OEP  OEP return period
30   0.1465  0.1070                9.3
70   0.0535  0.0345               29.0
110  0.0235  0.0190               52.6
```

The figure draws both curves for all amounts. The gap is the part of the
tail that many events make and one event does not.

```python
import matplotlib.pyplot as plt

xs = np.linspace(5, 150, 60)
totals_arr, maxima_arr = np.array(total.draws), np.array(largest.draws)
fig, ax = plt.subplots()
ax.plot(xs, [(totals_arr > x).mean() for x in xs], label="AEP (year total)")
ax.plot(xs, [(maxima_arr > x).mean() for x in xs], label="OEP (largest event)")
ax.set_yscale("log")
ax.set_xlabel("loss x (millions of crowns)")
ax.set_ylabel("probability of a loss above x in a year")
ax.set_title("Alder: aggregate and occurrence exceedance, 2,000 years")
ax.legend()
```

## 5. Does the simulation agree with the event table?

The event loss table (ELT) of Alder gives each catalog event a rate and a loss
distribution. With a Poisson count of events, the OEP has a closed form:

```
OEP(x) = 1 - exp(-sum_i rate_i * P(X_i > x))
```

`EventLossTable.oep` calculates it exactly, and `occurrence_pml` inverts it.
The simulated shares should agree with it up to a standard error of
`sqrt(p (1 - p) / 2000)`.

The YELT gives some windstorms and convective storms in pieces a few hours
apart, but the ELT counts each event once. An **occurrence** is what a contract
counts as one loss: here, all pieces in the 72 hours from the first piece (page
3 explains the hours clause). Raw maxima see single pieces and understate the
OEP. Grouping by 72 hours joins the pieces again, but it can also join two
different catalog events in the same window, which raises the grouped OEP.

```python
grouped = alder.grouped(72.0)
largest_grouped = grouped.maxima().total()
check = pd.DataFrame({"x": [int(x) for x in levels]})
check["ELT"] = [elt.oep(x) for x in levels]
check["raw"] = oep
check["grouped"] = [sum(v > x for v in largest_grouped.draws) / N_YEARS for x in levels]
check["s.e."] = [math.sqrt(p * (1 - p) / N_YEARS) for p in check["ELT"]]
(GT(check, rowname_col="x")
 .tab_header("Alder: occurrence exceedance, simulated and from the ELT", "x in millions of crowns")
 .fmt_number(["ELT", "raw", "grouped", "s.e."], decimals=4))
```

```text
        ELT     raw  grouped    s.e.
30   0.1136  0.1070   0.1150  0.0071
70   0.0374  0.0345   0.0375  0.0042
110  0.0180  0.0190   0.0215  0.0030
```

```python
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
ELT 100-year occurrence loss: 147.94
occurrences 4238, joining two or more catalog events 87, of which above 30: 9
```

The figure shows the same check for all amounts. Up to about 90, the raw maxima
(grey) are below the ELT line and the grouped line follows it. Above 90, both
simulated lines are above it, and few years lie behind each point (about 40 at
110): joined occurrences and sampling error.

```python
xs = np.linspace(20, 130, 45)
grouped_arr = np.array(largest_grouped.draws)
fig, ax = plt.subplots()
ax.plot(xs, [(maxima_arr > x).mean() for x in xs], color="0.75", label="simulated, raw maxima")
ax.plot(xs, [(grouped_arr > x).mean() for x in xs], label="simulated, grouped by 72 hours")
ax.plot(xs, [elt.oep(float(x)) for x in xs], label="ELT, closed form")
ax.set_yscale("log")
ax.set_xlabel("loss x (millions of crowns)")
ax.set_ylabel("OEP(x)")
ax.set_title("Alder: occurrence exceedance from the years and from the ELT")
ax.legend()
```

At 30 and 70, the grouped shares are well within one standard error of the ELT.
At 110, the difference of 0.0035 is a little more than one standard error. The nine joined
occurrences above 30 push the share the same way. The hours clause pays on the
joined occurrence, so price a contract on the grouped curve, although it is not
the curve of the ELT.

## 6. Losses at a return period

The curves also give the loss at a return period. The aggregate probable
maximum loss (PML) comes from the totals, and the occurrence PML from the
maxima. Underwriters call both "PML", so say which one. Each is a VaR of
section 2, here with the inverted-cdf rule.

```python
rps = [10, 50, 100, 250]
pml = pd.DataFrame({
    "return period": rps,
    "AEP loss": [Sampled(total.draws).var(1 - 1 / rp) for rp in rps],
    "OEP loss": [Sampled(largest_grouped.draws).var(1 - 1 / rp) for rp in rps],
    "ELT OEP loss": [elt.occurrence_pml(1 / rp) for rp in rps],
})
(GT(pml, rowname_col="return period")
 .tab_header("Alder: loss at a return period", "Millions of crowns; simulated losses use the inverted-cdf rule")
 .fmt_number(["AEP loss", "OEP loss", "ELT OEP loss"], decimals=3))
```

```text
     AEP loss  OEP loss  ELT OEP loss
10     42.244    33.971        33.648
50    125.246   120.464       103.432
100   176.824   163.418       147.938
250   232.741   223.450       199.470
```

In the far tail, the simulated OEP losses are above those of the ELT, for the
reasons of section 5. At 250 years, only eight of the 2,000 years are beyond
the estimate, so the sampling error is large. The hours clause also joins
occurrences, and more years do not remove that increase.

## 7. How often is Alder's first layer touched, and used up?

For a layer `l xs a`, an underwriter asks two questions:

- **Attachment probability**: what is the probability that a loss reaches
  the layer? Per occurrence, it is `P(X > a)`, strictly above the attachment.
- **Exhaustion probability**: what is the probability that one loss uses all
  of the limit? Per occurrence, it is `P(X >= a + l)`.

Each has a per-occurrence and an annual version. The strict and inclusive
comparisons matter because a simulated loss can sit exactly on a boundary. With
a Poisson count of `lambda` occurrences a year, the annual probability of a loss
above `a` is `1 - exp(-lambda P(X > a))`, the OEP at `a`.

On a severity grid, `occurrence_layer_statistics` gives all of these values
exactly ([aggregate-distributions.md](aggregate-distributions.md) shows how to
make grids). Here the grid is the empirical distribution of Alder's grouped
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

On the simulated years (#581), count the grouped losses per occurrence and read
the grouped maxima per year. The ELT gives per-occurrence values from its
inverted OEP formula:
`P(X > x) = -ln(1 - OEP(x)) / rate`.

```python
elt_attach = -math.log(1 - elt.oep(a)) / elt.rate
elt_exhaust = -math.log(1 - elt.oep(a + l)) / elt.rate
rescale = elt.rate / lam
odds = pd.DataFrame({
    "basis": ["trials, per occurrence", "trials, per year", "ELT, per occurrence",
              f"ELT, per {lam:.3f} occurrences", "ELT, per year"],
    "P(attach)": [sum(x > a for x in occ) / len(occ), sum(x > a for x in largest_grouped.draws) / N_YEARS,
                  elt_attach, elt_attach * rescale, elt.oep(a)],
    "P(exhaust)": [sum(x >= a + l for x in occ) / len(occ), sum(x >= a + l for x in largest_grouped.draws) / N_YEARS,
                   elt_exhaust, elt_exhaust * rescale, elt.oep(a + l)],
})
(GT(odds, rowname_col="basis")
 .tab_header("Alder cat 1: attachment and exhaustion probabilities")
 .fmt_number(["P(attach)", "P(exhaust)"], decimals=4))
```

```text
                            P(attach)  P(exhaust)
trials, per occurrence         0.0569      0.0179
trials, per year               0.1150      0.0375
ELT, per occurrence            0.0516      0.0163
ELT, per 2.119 occurrences     0.0569      0.0180
ELT, per year                  0.1136      0.0374
```

The per-occurrence rows differ, but they agree at the same count. The ELT counts
2.3365 events a year. The grouped trials count 2.119 occurrences (the hours
clause joins pieces, and the YELT has no pieces below its 0.05 floor). The
annual rows need no rescale, which is one reason to compare annual values.

The exhaustion probability of the grid is a little higher than that of the
trials, because rounding up to the next half crown moves a few losses just
below 70 to 70. "Exhausted" here means that one occurrence used all of the
limit. The annual limit is the limit plus its one reinstatement, 80 at 100%.
Using all of it needs layer losses of 80 in one year, from any number of
occurrences, which is rarer. Page 3 runs the layer itself.

## 8. From one cedent to the book

Larkspur writes all three cedents, so its tail is the tail of their sum, and
each cedent's share of that tail is less than its stand-alone TVaR.
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
