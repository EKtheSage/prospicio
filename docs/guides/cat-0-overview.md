# Property catastrophe: a renewal at Larkspur Re

This track follows one reinsurer through one renewal of property catastrophe
business. It starts from the loss tables of a vendor model and ends with a
priced portfolio and its capital. Each page takes one step. It explains the
idea, gives the formula if there is one, and runs it in prospicio on the same
small book.

## Who it is for

The track is for an analyst who knows what a layer and a reinstatement are. The
analyst now works with simulated catastrophe losses: a year event loss table
from a cat model, some treaties and a request for a price. If the terms layer,
rate on line or cost of capital are new to you, read
[a first course](a-first-course.md) first. Then read
[what a reinsurance price is](reinsurance-price.md).

The pages work on simulated years (trials). For an exact method on a grid, a
page links to [aggregate distributions](aggregate-distributions.md) and does
not repeat the method. That guide also covers event sets in general.

## The pages

Each page takes the next step of the renewal at Larkspur and uses the page
before it. Each page also runs on its own.

1. [What the cat model hands over](cat-1-loss-tables.md): the three loss
   tables (YELT, YLT and ELT), secondary uncertainty, how to read a YELT,
   event times, and a check of a simulation against the ELT.
2. [How bad can Alder's year get?](cat-2-risk-metrics.md): VaR and TVaR on
   simulated years, exceedance curves, return periods, and how often a loss
   reaches a layer or exhausts it.
3. [Writing Larkspur's contracts in code](cat-3-contract-terms.md): the
   hours clause, occurrence layers with reinstatements, the capped quota
   share, the aggregate cover and its perils, and the contract year.
4. [Stacked covers, and who keeps what](cat-4-programs.md): several layers
   together, a signed line against the whole layer, a quota share that
   inures to a cat cover, losses out of reach of a cover, and a retro.
5. [The money around the loss](cat-5-beyond-loss.md): reinstatement
   premium and the season, commissions, sliding scales and brokerage, and the
   deal ledger.
6. [A price for Alder's first layer](cat-6-pricing.md): cost of capital and
   distortion prices, the technical quote, reinstatement premium in the
   price, sensitivities and the market check.
7. [Larkspur's whole book](cat-7-portfolio.md): the five deals together,
   the tail of the book, capital allocation, diversification and the
   marginal price of a new deal.

Each page uses the terms of page 1 (year, event, piece, occurrence) and the
hours clause of page 3, section 2. Pages 6 and 7 also use the reinstatement
premium of page 5.

## Notation

Every page uses the same symbols.

| Symbol | Meaning |
|---|---|
| `l xs a` | a layer of limit `l` above an attachment `a` |
| `AD`, `AL` | aggregate deductible and aggregate limit |
| `j` | the index of a reinstatement |
| `k` | a rank among the simulated years, as in "the `k`-th largest" |
| `N` | the number of simulated years, here 2,000 |
| `p` | a probability level, as in VaR at `p` |
| `EL` | expected loss |
| `P` | premium (upfront premium on an excess layer) |
| `RIP` | reinstatement premium |
| `A`, `K` | assets and capital, `K = A - P` |
| `r` | cost of capital |

## The story

Larkspur Re is a fictional reinsurer. It writes property catastrophe cover for
three fictional insurers in the fictional country of Varland. Varland has three
regions: the coast, the plains and the range.

- **Alder Mutual** (`alder`) writes mostly on the coast.
- **Bracken Insurance** (`bracken`) writes mostly on the plains.
- **Corran General** (`corran`) writes mostly in the range.

Four perils hit Varland:

- **windstorm** (`WS`), late in the year, from about day 213 to day 305.
- **earthquake** (`EQ`), at any time.
- **flood** (`FL`), in spring, from about day 60 to day 180.
- **convective storm** (`CS`), hail and tornado, from about day 100 to day 220.

Money is in millions of Varland crowns, a synthetic currency. All names and
numbers in the book are fictional.

Some market terms occur on many pages. The **subject premium** of a cedent is
the premium of the business that the reinsurance protects. Its **cat premium**
is the part of the subject premium for catastrophe risk. The **signed line** of
a reinsurer is the share of a layer that it takes. Other reinsurers take the
remainder.

The **rate on line** (RoL) is the premium of an excess layer divided by its
limit. A **deal** is one contract as Larkspur wrote it: the premium, losses and
other flows of its line, year by year.

A windstorm on the coast hits all three cedents at the same time, so their
losses move together. A convective storm outbreak often comes into the loss
table as two or three pieces, one or two days apart. The hours clause (page 3)
decides if these pieces count as one loss or as several. This decision changes
what each treaty pays.

## The files

The book is in four CSV files in `examples/data/`. The script
`examples/data/make_cat_book.py` writes them from a fixed seed, so each run
gives the same bytes. A test checks that the committed files match the output
of the script.

| File | One row per |
|---|---|
| `cat_book_yelt.csv` | loss to one cedent from one event in one simulated year |
| `cat_book_elt_alder.csv` | event of the catalog that can hit Alder |
| `cat_book_cedents.csv` | cedent: home region, subject premium, catastrophe premium |
| `cat_book_contracts.csv` | contract Larkspur Re writes this year, with all its terms |

The year event loss table (YELT) holds 2,000 simulated years. Each row has
these fields: year, `slot` (the place of the event in the year), event, day,
hour, peril, region, cedent and loss. A year with no loss to any cedent has no
rows. The event loss table (ELT) gives the annual rate of each event and the
mean loss to Alder with its spread. A cat model gives its results in this form
before it simulates years. Page 1 explains the two tables and checks that they
agree.

The pages read every contract term from `cat_book_contracts.csv`. The code
types no term. To try a different case, change the script and run it again.

### The cedents and the contracts

This block prints the two small tables with only the `csv` module of Python.
The guides run from `docs/guides/`, so the data is two folders up.

```python
import csv

DATA = "../../examples/data/"


def read(name):
    with open(DATA + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


cedents = read("cat_book_cedents.csv")
contracts = read("cat_book_contracts.csv")

print(f"{'cedent':<8} {'name':<18} {'home':<7} {'subject':>8} {'cat':>6}")
for c in cedents:
    print(
        f"{c['cedent']:<8} {c['name']:<18} {c['home_region']:<7} "
        f"{float(c['subject_premium']):>8.1f} {float(c['cat_premium']):>6.1f}"
    )
print()
print(f"{'contract':<12} {'kind':<14} {'perils':<6} {'terms':<26} {'share':>5}  extra")
for k in contracts:
    num = lambda key: float(k[key]) if k[key] else 0.0
    if k["kind"] == "occurrence_xl":
        terms = f"{num('limit'):g} xs {num('attachment'):g}"
        extra = (
            f"{k['reinstatements']} reinst. at {num('reinstatement_rate'):.0%}, "
            f"RoL {num('rate_on_line'):.2f}, {k['hours']} h"
        )
    elif k["kind"] == "aggregate_xl":
        terms = f"agg {num('aggregate_limit'):g} xs {num('aggregate_deductible'):g}"
        extra = f"RoL {num('rate_on_line'):.2f} on the aggregate limit"
    else:
        terms = f"event {num('limit'):g}, annual {num('aggregate_limit'):g}"
        extra = f"ceding commission {num('ceding_commission'):.0%}, {k['hours']} h"
    print(f"{k['contract']:<12} {k['kind']:<14} {k['perils']:<6} {terms:<26} {num('share'):>5.0%}  {extra}")
```

```text
cedent   name               home     subject    cat
alder    Alder Mutual       coast      420.0   40.0
bracken  Bracken Insurance  plains     310.0   36.0
corran   Corran General     range      260.0   30.0

contract     kind           perils terms                      share  extra
alder_cat_1  occurrence_xl  all    40 xs 30                     30%  1 reinst. at 100%, RoL 0.15, 72 h
alder_cat_2  occurrence_xl  all    100 xs 70                    20%  1 reinst. at 100%, RoL 0.05, 72 h
bracken_cat  occurrence_xl  all    50 xs 25                     25%  2 reinst. at 100%, RoL 0.14, 72 h
bracken_agg  aggregate_xl   CS+FL  agg 30 xs 40                 25%  RoL 0.06 on the aggregate limit
corran_qs    quota_share    all    event 150, annual 300        10%  ceding commission 25%, 72 h
```

The amounts are on a 100% basis. On an excess of loss contract, `share` is the
signed line of Larkspur Re. On the quota share, `share` is the cession. So
Larkspur Re takes 30% of the first layer of Alder, which pays the part of each
occurrence between 30 and 70. The aggregate cover of Bracken applies only to
convective storm and flood. It pays the part of the annual total loss from
these perils between 40 and 70.

The quota share cedes 10% of the losses of Corran, with a maximum of 150 for
one event and 300 for the year. It pays Corran a 25% ceding commission on the
ceded premium.

### The year event loss table at a glance

The next block also uses only `csv`. It counts the rows of the YELT and the
simulated years with a loss. Then it splits the mean annual loss by peril and
cedent.

```python
yelt = read("cat_book_yelt.csv")
years = {int(r["year"]) for r in yelt}
print(f"rows {len(yelt):,}, years with a loss {len(years):,} of 2,000")
print(f"first year with a loss {min(years)}, last {max(years)}")

perils = ["WS", "EQ", "FL", "CS"]
names = [c["cedent"] for c in cedents]
total = {(p, n): 0.0 for p in perils for n in names}
for r in yelt:
    total[(r["peril"], r["cedent"])] += float(r["loss"])
print()
print("mean annual loss by peril and cedent")
print(f"{'peril':<6}" + "".join(f"{n:>9}" for n in names) + f"{'all':>9}")
for p in perils:
    row = [total[(p, n)] / 2000 for n in names]
    print(f"{p:<6}" + "".join(f"{v:>9.3f}" for v in row) + f"{sum(row):>9.3f}")
col = [sum(total[(p, n)] for p in perils) / 2000 for n in names]
print(f"{'all':<6}" + "".join(f"{v:>9.3f}" for v in col) + f"{sum(col):>9.3f}")
```

```text
rows 16,432, years with a loss 1,820 of 2,000
first year with a loss 2, last 2000

mean annual loss by peril and cedent
peril     alder  bracken   corran      all
WS        9.915    6.163    5.575   21.652
EQ        2.473    2.050    4.371    8.894
FL        3.042    3.999    1.997    9.038
CS        1.349    3.165    1.106    5.620
all      16.778   15.376   13.049   45.204
```

The mean annual loss over all years is the average annual loss (AAL, defined on
[page 1](cat-1-loss-tables.md)). Windstorm is the largest peril for every
cedent, also for Corran in the range. The cause is that a coastal storm is the
largest event in the catalog. The home region shows in the second peril: flood
and convective storm for Bracken on the plains, and earthquake for Corran in
the range.

In 180 of the 2,000 years, no event hits a cedent. Year 1 is one of these
years, so the file starts at year 2. A year with no row is still a year of the
simulation. So every average on these pages divides by the 2,000 simulated
years.

Each page has an R twin (the same file name with `.qmd`) with the same sections
and numbers.

## Further reading

[reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is another
introduction to property catastrophe reinsurance.
