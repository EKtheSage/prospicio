# Property catastrophe 1: what the cat model hands over

The renewal starts when the cedents' catastrophe model results arrive. A cat
model does not hand over a distribution. It hands over tables: a list of
possible events, or a long run of simulated years. This page explains the
tables, reads Larkspur Re's book into prospicio, and checks that two views of
the same cedent tell the same story. The story, the files and the notation are
on [the overview](cat-0-overview.md).

## 1. The words

A **trial** is one simulated year. A cat model runs many trials (here 2,000),
and every result on later pages is a statement about those trials: an average,
a share, a high percentile. Trials are also called years or simulations; from page 2 on
the track says **year**.

An **event** is one happening of a peril in the model's catalog: one windstorm,
one earthquake. In a trial, events happen at given times and cause given
losses. A model may deliver one event as several **pieces**, rows a few hours
apart whose losses add up to the event's loss. An **occurrence** is what a
contract counts as one loss: the pieces, and sometimes several events, that
its hours clause joins (page 3). The three words are kept apart on every page.

A **YELT** (year event loss table) lists every event loss in every trial, one
row per loss: the trial, the event, when it happened and what it cost. It is
the most detailed table and the one most pages use.

A **YLT** (year loss table) keeps only one number per trial, the annual total.
It is enough for aggregate covers and annual risk measures, but not for a layer
that applies to each event, since it has forgotten the events. (prospicio's
reader for both is called `read_ylt`; given a slot column it keeps each event.)

An **ELT** (event loss table) does not simulate at all. It lists each possible
event once, with an annual rate and a loss distribution. Rates add up to the
expected number of events a year.

**Secondary uncertainty** is the spread of an event's loss around its mean,
given that the event happens. A storm of a given track and strength still
causes a range of losses, because damage to buildings varies. An ELT carries it
as standard deviations; a YELT carries it implicitly, since each row is one
draw.

The **AAL** (average annual loss) is the expected loss in a year. On an ELT it
is a sum over events, each rate times its mean loss:

```
AAL = sum over events i of  rate_i * mean_i
```

On a YELT or YLT it is the average of the annual totals over the `N` trials:

```
AAL = (1 / N) * sum over trials t of  total_t
```

The two agree only up to simulation error, which section 4 measures.

## 2. Reading a YELT

Larkspur's YELT is `examples/data/cat_book_yelt.csv`, written by
`examples/data/make_cat_book.py`. Each row is one cedent's loss from one event
(or one piece of an event) in one trial. Its columns are `year`, `slot`
(the order of the event within its year, shared by every cedent it hit),
`event_id`, `day`, `hour`, `peril`, `region`, `cedent` and `loss`, in millions
of Varland crowns. Some trials have no events at all, the first one included.

`read_ylt` reads such a file into an `EventSet`, prospicio's container for
years of losses. It keeps the named columns as labels, so that later results
can be split by peril, region or cedent.

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

`read_ylt` checks the file as it reads it, and refuses it if:

- a loss is negative, infinite or missing. Losses are positive on every page
  of this track; a file that writes losses as negative numbers (money going
  out) must have its sign flipped before it is read (#593);
- a year is below 1 (years count from 1; the largest year sets the number of
  trials unless `n_years=` says otherwise);
- two rows share a year, a slot and the same labels, which would usually mean a
  row was duplicated. One event that hits all three cedents has three rows with
  the same slot, so the labels must include `cedent` for those rows to differ
  (or `unique_by=` names the labels that make a row unique);
- a label takes a value outside `expected=`, when that is given.

The last check catches a typo in a peril code before it silently becomes a
fifth peril:

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

The hash is a fingerprint of the file's bytes. Results computed from the event
set record it, so a number can be traced back to the exact table it came from.
The page prints only its first characters, because the full hash changes if a
checkout rewrites line endings.

**Tiling and resampling.** A model might give 2,000 years while a capital model
runs 10,000 trials. `n_sims=` stretches the table. With `expand="tile"` (the
default) the 2,000 years repeat in order, so the mean does not move at all.
With `expand="resample"` each trial picks a year at random, with a seed, so the
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

Neither adds information. Tiling is the honest choice when trials must line up
with another model's trials one for one; resampling is useful when a whole
multiple is not possible. Two warnings go with both. Neither lets you read a
return period beyond the original 2,000 years: the 10,000-trial table still
has only the original years in it, so its "5,000-year loss" is the worst of
the 2,000. And the stretched trials are copies, not independent years, so a
standard error computed on 10,000 of them as if they were independent is too
small by a factor of about `sqrt(5)`: tail estimates look more precise than
they are.

## 3. Times and one cedent

Most pages need one cedent's losses with the time of each loss, since the hours
clause and reinstatements pro rata as to time depend on it. `read_ylt` has no
time column (#594), so the track builds such event sets with a short helper. It reads the
YELT once, keeps one cedent's rows, turns `day` and `hour` into a fraction of
the year, and labels each loss with its peril. The same helper appears on each
page that needs it.

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

Rebuilding the set this way loses the file hash, since `from_years` never saw
the file.

## 4. An ELT, and reconciling it with the YELT

Alder's cat model also delivered an ELT, `examples/data/cat_book_elt_alder.csv`,
one row per event with its rate, mean, two standard deviations and exposure.
The **exposure** is the most the event could cost Alder: the insured value in
the event's footprint. Secondary uncertainty comes as two standard deviations:
an **independent part**, meant to vary separately at each location or cedent
the event hits, and a **correlated part**, meant to move together across them.
In prospicio an `EventLossTable` takes those five columns. Given that an event
happens, its loss is a beta distribution between 0 and the exposure, a
bounded shape that can be skewed either way, with the row's mean and a
standard deviation equal to the sum of the two standard deviations (the
standard deviations add, not the variances). The table's `rate` is the expected number of events a year,
and `expected_annual_loss` is the AAL formula of section 1.

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

`simulate` turns the ELT into trials: in each year a Poisson number of events,
each picked in proportion to its rate, each with a beta loss. It also returns
which row each loss came from (rows count from 0).

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

So there are three estimates of one number, Alder's AAL: the ELT's exact sum,
the YELT's average and the average of a fresh simulation from the ELT. The two
averages come from 2,000 trials each, so each is off the true value by about
one standard error, the standard deviation of a year's loss over the square
root of the number of trials:

```
SE = sd(total_t) / sqrt(N)
```

```python
import math

se = math.sqrt(alder_totals.variance()) / math.sqrt(N_YEARS)
print(f"ELT exact AAL        {elt.expected_annual_loss:.3f}")
print(f"YELT average         {alder_totals.mean():.3f}")
print(f"ELT simulated mean   {simulated.totals().mean():.3f}")
print(f"standard error, 2,000 years  {se:.2f}")
```

```text
ELT exact AAL        16.056
YELT average         16.778
ELT simulated mean   16.851
standard error, 2,000 years  0.79
```

The three agree to within about one standard error. That is all one can ask:
catastrophe losses are so skewed that a few large years move the average, and
2,000 years do not pin the AAL down to the first decimal. Reconciling tables
means checking that differences are of this size, not that they vanish.

One difference is built into this book. The script that wrote the YELT drops
any loss below 0.05 (a cedent barely touched by an event), as many vendor
extracts do, while the ELT keeps every event. That lowers the YELT's AAL
slightly against the ELT's, by much less than one standard error here.

## What is missing

- `read_ylt` reads no time column, so pages build timed event sets by hand, and
  the hours clause (`grouped`) drops the labels (#594).
- There is no option for files that store losses as negative numbers, and
  `Layer.ceded` treats a negative loss as below the attachment without a
  warning (#593).

## In R

The R twin (`cat-1-loss-tables.qmd`) runs the same steps with `read_ylt()`,
`event_loss_table()`, `simulate_elt()`, `events_from_years(times = )` and
`with_labels()`.

The next page turns these trials into exceedance curves and risk measures.

## Further reading

[reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer, another
introduction to property catastrophe reinsurance.
