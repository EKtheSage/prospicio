# Property catastrophe 1: what the cat model hands over

The renewal starts when the catastrophe model results of the cedents arrive. A
cat model does not give a distribution. It gives tables: a list of possible
events, or a long run of simulated years. This page explains the tables and
reads the book of Larkspur Re into prospicio. Then it checks that two views of
the same cedent agree. The story, the files and the notation are on
[the overview](cat-0-overview.md).

## 1. The words

A **trial** is one simulated year. A cat model runs many trials (here 2,000).
Every result on later pages is a statement about these trials: an average, a
share, a high percentile. Other names for a trial are year and simulation. From
page 2, the track uses **year**.

An **event** is one instance of a peril in the catalog of the model: one
windstorm, one earthquake. In a trial, events come at given times and cause
given losses. A model can give one event as several **pieces**: rows a few
hours apart whose losses add up to the loss of the event. An **occurrence** is
what a contract counts as one loss: the pieces, and sometimes several events,
that its hours clause joins (page 3). Every page keeps these three terms
separate.

A **YELT** (year event loss table) lists every event loss in every trial. It
has one row per loss: the trial, the event, its time and its cost. It is the
most detailed table, and most pages use it.

A **YLT** (year loss table) keeps only one number per trial, the annual total.
It is sufficient for aggregate covers and annual risk measures. It is not
sufficient for a layer that applies to each event, because it has no events.
The prospicio reader for the two tables is `read_ylt`. With a slot column, it
keeps each event.

An **ELT** (event loss table) does not simulate. It lists each possible event
once, with an annual rate and a loss distribution. The sum of the rates is the
expected number of events in a year.

**Secondary uncertainty** is the spread of the loss of an event around its
mean, when the event occurs. A storm with a given track and strength still
causes a range of losses, because the damage to buildings varies. An ELT gives
it as standard deviations. A YELT contains it implicitly, because each row is
one draw.

The **AAL** (average annual loss) is the expected loss in a year. On an ELT it
is a sum over events of each rate times its mean loss:

```
AAL = sum over events i of  rate_i * mean_i
```

On a YELT or YLT it is the average of the annual totals over the `N` trials:

```
AAL = (1 / N) * sum over trials t of  total_t
```

The two values agree only up to simulation error. Section 4 measures this
error.

## 2. Reading a YELT

The YELT of Larkspur is `examples/data/cat_book_yelt.csv`, and
`examples/data/make_cat_book.py` writes it. Each row is the loss of one cedent
from one event (or one piece of an event) in one trial. Its columns are `year`,
`slot`, `event_id`, `day`, `hour`, `peril`, `region`, `cedent` and `loss`. The
`slot` is the order of the event in its year, and it is the same for every
cedent that the event hit. Losses are in millions of Varland crowns, and some
trials have no events (for example the first trial).

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
  of this track. If a file writes losses as negative numbers (money that goes
  out), flip their sign before you read it (#593). `Layer.ceded` reads a
  negative loss as a loss below the attachment and gives no warning.
- a year is less than 1. Years count from 1. The largest year sets the number
  of trials, unless `n_years=` gives a different number.
- two rows have the same year, slot and labels. This usually means a
  duplicated row. One event that hits all three cedents has three rows with
  the same slot, so the labels must include `cedent` to make these rows
  different. Alternatively, `unique_by=` names the labels that make a row
  unique.
- a label has a value that is not in `expected=`, if you give `expected=`.

The last check finds an incorrect peril code before the code becomes a fifth
peril without a warning:

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

The hash is a fingerprint of the bytes of the file. Results from the event set
record it, so you can trace a number to the exact table that it came from. The
page prints only the first characters, because the full hash changes if a
checkout changes the line endings.

**Tiling and resampling.** A model can give 2,000 years when a capital model
runs 10,000 trials. `n_sims=` extends the table. With `expand="tile"` (the
default), the 2,000 years repeat in order, so the mean does not change. With
`expand="resample"`, each trial selects a year at random, with a seed, so the
mean changes a little:

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

Neither method adds information. Tiling is the choice when the trials must
align one for one with the trials of another model. Resampling is useful when a
whole multiple is not possible. Two warnings apply to both methods.

First, neither method lets you read a return period beyond the original 2,000
years. The 10,000-trial table contains only the original years, so its
"5,000-year loss" is the worst of the 2,000. Second, the extended trials are
copies, not independent years. A standard error from the 10,000 trials as
independent years is too small by a factor of about `sqrt(5)`. So tail
estimates look more precise than they are.

## 3. Times and one cedent

Most pages need the losses of one cedent with the time of each loss. The hours
clause and reinstatements pro rata as to time use this time. The track builds
these event sets with a short helper (#594). The helper reads the YELT once and
keeps the rows of one cedent. It changes `day` and `hour` into a fraction of
the year and labels each loss with its peril. Each page that needs the helper
contains the same code.

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

## 4. An ELT, and reconciling it with the YELT

The cat model of Alder also gives an ELT,
`examples/data/cat_book_elt_alder.csv`. It has one row per event with its rate,
mean, two standard deviations and exposure. The **exposure** is the maximum
cost of the event to Alder: the insured value in the footprint of the event.
The two standard deviations give the secondary uncertainty. The **independent
part** is meant to vary separately at each location or cedent that the event
hits. The **correlated part** is meant to move together across them.

In prospicio an `EventLossTable` takes these five columns. When an event
occurs, its loss has a beta distribution between 0 and the exposure. This
bounded shape can have skew in either direction. It has the mean of the row.
Its standard deviation is the sum of the two standard deviations (the standard
deviations add, not the variances). The `rate` of the table is the expected
number of events in a year, and `expected_annual_loss` is the AAL formula of
section 1.

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

`simulate` changes the ELT into trials. In each year, it draws a Poisson number
of events. It selects each event in proportion to its rate and gives each event
a beta loss. It also returns the row of each loss (rows count from 0).

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

So there are three estimates of the AAL of Alder. They are the exact sum of the
ELT, the average of the YELT and the average of a new simulation from the ELT.
The two averages each come from 2,000 trials. So each average differs from the
true value by about one standard error. The standard error is the standard
deviation of the annual loss divided by the square root of the number of
trials:

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

The three estimates agree to within about one standard error. You cannot expect
a better agreement. Catastrophe losses are very skewed, so a few large years
move the average, and a sample of 2,000 years does not fix the AAL to the first
decimal. To reconcile tables, check that the differences are of this size, not
that they are zero.

This book has one built-in difference. The script that wrote the YELT removes
each loss below 0.05 (a cedent that an event almost missed), as many vendor
extracts do. The ELT keeps every event. This removal makes the AAL of the YELT
slightly lower than the AAL of the ELT, by much less than one standard error
here.

## In R

The R twin (`cat-1-loss-tables.qmd`) runs the same steps with `read_ylt()`,
`event_loss_table()`, `simulate_elt()`, `events_from_years(times = )` and
`with_labels()`.

The next page changes these trials into exceedance curves and risk measures.

## Further reading

[reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is another
introduction to property catastrophe reinsurance.
