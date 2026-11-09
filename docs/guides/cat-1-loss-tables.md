# Property catastrophe 1: what the cat model hands over

<!-- site: execute -->

The renewal starts when the cedents' cat model results arrive. A cat model
gives tables, not a distribution: a list of possible events, or a long run of
simulated years. This page reads the Larkspur Re book into prospicio and
checks that two views of one cedent agree. The story, the files and the
notation are on [the overview](cat-0-overview.md).

## 1. The words

A **trial** is one simulated year; a cat model runs many (here 2,000). Other
names are year and simulation. From page 2, the track says **year**.

An **event** is one instance of a peril in the model's catalog: one windstorm,
one earthquake. A model can give one event as several **pieces**: rows a few
hours apart whose losses add up to the event's loss. An **occurrence** is what
a contract counts as one loss: the pieces, and sometimes several events, that
its hours clause joins (page 3). Every page keeps these three terms separate.

A **YELT** (year event loss table) has one row per event loss: the trial, the
event, its time and its cost. It is the most detailed table, and most pages use
it.

A **YLT** (year loss table) keeps one number per trial, the annual total. It is
enough for aggregate covers and annual risk measures, but not for a layer that
applies to each event, because it has no events. `read_ylt` reads both tables.
With a slot column, it keeps each event.

An **ELT** (event loss table) does not simulate. It lists each possible event
once, with an annual rate and a loss distribution. The rates add up to the
expected number of events in a year.

**Secondary uncertainty** is the spread of an event's loss around its mean,
when it occurs: building damage varies, even for a storm of given track and
strength. An ELT gives it as standard deviations. A YELT holds it implicitly:
each row is one draw.

The **AAL** (average annual loss) is the expected loss in a year. On an ELT it
is a sum over events of rate times mean loss:

```
AAL = sum over events i of  rate_i * mean_i
```

On a YELT or YLT it is the average of the annual totals over the `N` trials:

```
AAL = (1 / N) * sum over trials t of  total_t
```

The two agree up to simulation error (section 4).

## 2. Reading a YELT

The Larkspur YELT is `examples/data/cat_book_yelt.csv`, written by
`examples/data/make_cat_book.py`. Each row is the loss of one cedent from one
event (or piece) in one trial, with columns `year`, `slot`, `event_id`, `day`,
`hour`, `peril`, `region`, `cedent` and `loss`. The `slot` is the order of the
event in its year, the same for every cedent that the event hit. Losses are in
millions of Varland crowns. Some trials have no events (the first, for
example).

`read_ylt` reads such a file into an `EventSet`, the prospicio container for
years of losses. It keeps the named columns as labels, so you can split later
results by peril, region or cedent.

```python
from prospicio.aggregate import read_ylt

YELT_PATH = "../../examples/data/cat_book_yelt.csv"
book = read_ylt(YELT_PATH, year="year", loss="loss", slot="slot", labels=["peril", "region", "cedent"])
print("trials:", book.n_sims)
print("losses:", sum(book.counts()))
print("labels:", book.label_dims)
print("hash starts with:", book.input_hash[:7])
print(f"mean annual loss, all cedents: {book.totals().mean():.3f}")
```

```text
trials: 2000
losses: 16432
labels: ['peril', 'region', 'cedent']
hash starts with: blake3:
mean annual loss, all cedents: 45.204
```

`read_ylt` checks the file when it reads it. It refuses the file if:

- a loss is negative, infinite or missing. Losses are positive on every page
  of this track. If a file writes losses as negative numbers (money out), flip
  their sign first (#593). `Layer.ceded` reads a negative loss as a loss below
  the attachment, with no warning.
- a year is less than 1. Years count from 1. The largest year sets the number
  of trials, unless `n_years=` says otherwise.
- two rows have the same year, slot and labels, usually a duplicated row. One
  event that hits all three cedents has three rows with the same slot, so the
  labels must include `cedent`. Or `unique_by=` names the labels that make a
  row unique.
- a label value is not in `expected=`, if you give it.

The last check stops a wrong peril code from becoming a silent fifth peril:

```python
try:
    read_ylt(YELT_PATH, year="year", loss="loss", slot="slot", labels=["peril", "region", "cedent"],
             expected={"peril": ["WS", "EQ", "FL"]})
except ValueError as e:
    print(str(e).splitlines()[0])
```

```text
invalid data: row 18: peril "CS" is not one of ["WS", "EQ", "FL"]
```

The hash fingerprints the file's bytes, so a result traces to its table. The
page prints only the first characters, because a checkout that changes line
endings changes the full hash.

**Tiling and resampling.** A model can give 2,000 years when a capital model
runs 10,000 trials. `n_sims=` extends the table. With `expand="tile"` (the
default), the 2,000 years repeat in order, so the mean is unchanged. With
`expand="resample"`, each trial draws a year at random (seeded), so the
mean moves a little:

```python
columns = dict(year="year", loss="loss", slot="slot", labels=["peril", "region", "cedent"])
tiled = read_ylt(YELT_PATH, **columns, n_sims=10_000)
drawn = read_ylt(YELT_PATH, **columns, n_sims=10_000, expand="resample", seed=1)
print(f"tiled:     {tiled.n_sims} trials, mean {tiled.totals().mean():.3f}")
print(f"resampled: {drawn.n_sims} trials, mean {drawn.totals().mean():.3f}")
```

```text
tiled:     10000 trials, mean 45.204
resampled: 10000 trials, mean 44.184
```

Neither adds information. Tile when the trials must align with another
model's trials; resample when a whole multiple is not possible. Both have two
limits. The "5,000-year loss" of the 10,000-trial table is the worst of the
original 2,000 years. And the extra trials are copies, so a standard error that
treats them as independent is too small by about `sqrt(5)`.

## 3. Times and one cedent

Most pages need one cedent's losses with their times (the hours clause and
reinstatements pro rata as to time use them). A short helper builds these event
sets (#594): it keeps one cedent's rows, turns `day` and `hour` into a fraction
of the year and labels each loss with its peril. Each page that needs it holds
the same code.

```python
import csv

from prospicio.aggregate import EventSet

DATA = "../../examples/data/"


def read(name):
    with open(DATA + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


YELT = read("cat_book_yelt.csv")
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
alder_totals = alder.totals().total()
print("trials:", alder.n_sims, " losses:", sum(alder.counts()), " has times:", alder.has_times)
print(f"Alder's AAL from the YELT: {alder_totals.mean():.3f}")
```

```text
trials: 2000  losses: 5391  has times: True
Alder's AAL from the YELT: 16.778
```

An event set built this way has no file hash, because `from_years` did not read
the file.

```python
import matplotlib.pyplot as plt

MONTH_DAYS = 365 / 12
by_month = {}
for r in YELT:
    if r["cedent"] == "alder":
        month = min(11, int((int(r["day"]) - 1) / MONTH_DAYS))
        by_month.setdefault(r["peril"], [0] * 12)[month] += 1
fig, ax = plt.subplots()
ax.plot(range(1, 13), by_month["EQ"], color="0.75", label="EQ")
for peril in ["CS", "WS", "FL"]:
    ax.plot(range(1, 13), by_month[peril], marker="o", label=peril)
ax.set_xticks(range(1, 13))
ax.set_xlabel("month of the year")
ax.set_ylabel("losses of Alder in 2,000 trials")
ax.set_title("Alder: losses by month and peril in the YELT")
ax.legend()
```

## 4. An ELT, and reconciling it with the YELT

Alder's cat model also gives an ELT,
`examples/data/cat_book_elt_alder.csv`. It has one row per event: rate,
mean, two standard deviations and exposure. The **exposure** is the event's
maximum cost to Alder: the insured value in its footprint. The two standard
deviations give the secondary uncertainty. The **independent part** varies
separately at each location or cedent that the event hits. The **correlated
part** moves together across them.

An `EventLossTable` takes these five columns. When an event occurs, its loss
has a beta distribution on 0 to the exposure, with the row's mean and a
standard deviation equal to the sum of the two (they add, not the variances).
The table's `rate` is the expected number of events in a year, and
`expected_annual_loss` is the AAL formula of section 1.

```python
from prospicio.aggregate import EventLossTable

rows = read("cat_book_elt_alder.csv")


def column(name):
    return [float(r[name]) for r in rows]


elt = EventLossTable(column("rate"), column("mean"), column("sd_independent"),
                     column("sd_correlated"), column("exposure"))
print("events:", len(elt))
print(f"events a year: {elt.rate:.4f}")
print(f"AAL: {elt.expected_annual_loss:.3f}")
```

```text
events: 299
events a year: 2.3365
AAL: 16.056
```

`simulate` turns the ELT into trials: each year has a Poisson number of
events, each chosen in proportion to its rate and given a beta loss. It also
returns the row of each loss (rows count from 0).

```python
simulated, event_rows = elt.simulate(2000, seed=7)
print(f"simulated mean over 2,000 years: {simulated.totals().mean():.3f}")
first = next(i for i, ids in enumerate(event_rows) if ids)
print(f"year {first + 1}: losses {[round(x, 2) for x in simulated.events(first)]} from rows {event_rows[first]}")
```

```text
simulated mean over 2,000 years: 16.851
year 2: losses [0.02, 5.24, 4.89] from rows [288, 279, 237]
```

Alder's AAL has three estimates: the exact sum of the ELT, the YELT average
and the average of a new ELT simulation. Each average differs from the true
value by about one standard error:

```
SE = sd(total_t) / sqrt(N)
```

```python
import math

se = math.sqrt(alder_totals.variance()) / math.sqrt(N_YEARS)
print(f"standard error, 2,000 years: {se:.2f}")
```

```text
standard error, 2,000 years: 0.79
```

The table gives each estimate and its gap to the exact AAL. The figure shows
the two averages with bars one standard error wide on each side:

```python
import pandas as pd
from great_tables import GT

estimates = {
    "ELT exact": elt.expected_annual_loss,
    "YELT average": alder_totals.mean(),
    "ELT simulated": simulated.totals().mean(),
}
exact = estimates["ELT exact"]
reconcile = pd.DataFrame({
    "estimate": list(estimates),
    "AAL": list(estimates.values()),
    "gap to the ELT exact": [v - exact for v in estimates.values()],
    "gap in standard errors": [(v - exact) / se for v in estimates.values()],
})
(GT(reconcile, rowname_col="estimate")
 .tab_header("AAL of Alder: three estimates", "Millions of Varland crowns")
 .fmt_number(["AAL", "gap to the ELT exact"], decimals=3)
 .fmt_number("gap in standard errors", decimals=2))
```

```text
                  AAL  gap to the ELT exact  gap in standard errors
ELT exact      16.056                 0.000                    0.00
YELT average   16.778                 0.723                    0.92
ELT simulated  16.851                 0.795                    1.01
```

```python
fig, ax = plt.subplots()
ax.axvline(elt.expected_annual_loss, color="0.75", label="ELT exact AAL")
for i, name in enumerate(["YELT average", "ELT simulated"]):
    ax.errorbar(estimates[name], i, xerr=se, marker="o", capsize=4, label=name)
ax.set_yticks([0, 1], ["YELT average", "ELT simulated"])
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("average annual loss of Alder (millions of Varland crowns)")
ax.set_title("Three estimates of the AAL, with one standard error")
ax.legend()
```

The estimates agree within about one standard error, and cannot agree better:
cat losses are very skewed, so a few large years move the average. Check that
differences are this size, not zero.

One difference is built in. The script that wrote the YELT removes each loss
below 0.05 (a cedent that an event almost missed), as many vendor extracts do.
The ELT keeps every event. So the YELT's AAL is slightly lower than the ELT's,
by much less than one standard error here.

## In R

The R twin (`cat-1-loss-tables.qmd`) runs the same steps with `read_ylt()`,
`event_loss_table()`, `simulate_elt()`, `events_from_years(times = )` and
`with_labels()`.

## Further reading

[reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is another
introduction to property catastrophe reinsurance.
