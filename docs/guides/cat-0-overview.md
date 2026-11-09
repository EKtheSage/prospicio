# Property catastrophe: a renewal at Larkspur Re

<!-- site: execute -->

This track follows one reinsurer through one renewal of property catastrophe
business, from the loss tables of a vendor model to a priced portfolio and its
capital. Each page takes one step and runs it in prospicio on the same small
book.

## Who it is for

The track is for an analyst who knows what a layer and a reinstatement are and
now works with simulated catastrophe losses: a year event loss table, some
treaties and a request for a price. If layer, rate on line or cost of capital
are new to you, read [a first course](a-first-course.md) first, then
[what a reinsurance price is](reinsurance-price.md).

The pages work on simulated years (trials). For an exact method on a grid, see
[aggregate distributions](aggregate-distributions.md), which also covers event
sets in general.

## The pages

Each page uses the one before it and also runs on its own.

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

Larkspur Re, a fictional reinsurer, writes property catastrophe cover for three
fictional insurers in Varland. Varland has three regions: the coast, the plains
and the range.

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

Market terms used on many pages:

- **Subject premium**: the premium of the business that the reinsurance
  protects. The **cat premium** is its part for catastrophe risk.
- **Signed line**: the share of a layer that a reinsurer takes. Other
  reinsurers take the remainder.
- **Rate on line** (RoL): the premium of an excess layer divided by its limit.
- **Deal**: one contract as Larkspur wrote it, with the premium, losses and
  other flows of its line, year by year.

A windstorm on the coast hits all three cedents at the same time, so their
losses move together. A convective storm outbreak often comes into the loss
table as two or three pieces, one or two days apart. The hours clause (page 3)
decides if these pieces count as one loss or as several, which changes what
each treaty pays.

## The files

The book is in four CSV files in `examples/data/`. The script
`examples/data/make_cat_book.py` writes them from a fixed seed, and a test
checks that the committed files match its output.

| File | One row per |
|---|---|
| `cat_book_yelt.csv` | loss to one cedent from one event in one simulated year |
| `cat_book_elt_alder.csv` | event of the catalog that can hit Alder |
| `cat_book_cedents.csv` | cedent: home region, subject premium, catastrophe premium |
| `cat_book_contracts.csv` | contract Larkspur Re writes this year, with all its terms |

The year event loss table (YELT) holds 2,000 simulated years. Its fields are
year, `slot` (the place of the event in the year), event, day, hour, peril,
region, cedent and loss. A year with no loss has no rows. The event loss table
(ELT) gives the annual rate of each event and the mean loss to Alder with its
spread. A cat model gives its results in this form before it simulates years.
Page 1 explains the two tables and checks that they agree.

The pages read every contract term from `cat_book_contracts.csv`. To try a
different case, change the script and run it again.

### The cedents and the contracts

The block reads the two small files with only the `csv` module. The guides run
from `docs/guides/`, so the data is two folders up.

```python
import csv

import pandas as pd
from great_tables import GT

DATA = "../../examples/data/"


def read(name):
    with open(DATA + name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


cedents = read("cat_book_cedents.csv")
contracts = read("cat_book_contracts.csv")

cedent_table = pd.DataFrame({
    "cedent": [c["cedent"] for c in cedents],
    "name": [c["name"] for c in cedents],
    "home": [c["home_region"] for c in cedents],
    "subject premium": [float(c["subject_premium"]) for c in cedents],
    "cat premium": [float(c["cat_premium"]) for c in cedents],
})
(GT(cedent_table, rowname_col="cedent")
 .tab_header("The three cedents", "Premiums in millions of Varland crowns")
 .fmt_number(["subject premium", "cat premium"], decimals=1))
```

```text
                      name    home  subject premium  cat premium
alder         Alder Mutual   coast            420.0         40.0
bracken  Bracken Insurance  plains            310.0         36.0
corran      Corran General   range            260.0         30.0
```

```python
def num(key, k):
    return float(k[key]) if k[key] else float("nan")


def term(k, occurrence, aggregate):
    return num(aggregate if k["kind"] == "aggregate_xl" else occurrence, k)


terms_table = pd.DataFrame({
    "contract": [k["contract"] for k in contracts],
    "kind": [k["kind"] for k in contracts],
    "perils": [k["perils"] for k in contracts],
    "limit": [term(k, "limit", "aggregate_limit") for k in contracts],
    "attachment": [term(k, "attachment", "aggregate_deductible") for k in contracts],
    "annual limit": [num("aggregate_limit", k) if k["kind"] == "quota_share" else float("nan")
                     for k in contracts],
    "share": [num("share", k) for k in contracts],
    "reinstatements": [num("reinstatements", k) for k in contracts],
    "reinst. rate": [num("reinstatement_rate", k) for k in contracts],
    "rate on line": [num("rate_on_line", k) for k in contracts],
    "ceding commission": [num("ceding_commission", k) for k in contracts],
    "hours": [num("hours", k) for k in contracts],
})
(GT(terms_table, rowname_col="contract")
 .tab_header("Larkspur Re's five contracts",
             "Limits and attachments in millions of Varland crowns, 100% basis")
 .fmt_number(["limit", "attachment", "annual limit", "reinstatements", "hours"], decimals=0)
 .fmt_percent(["share", "reinst. rate", "ceding commission"], decimals=0)
 .fmt_number("rate on line", decimals=2)
 .sub_missing(missing_text=""))
```

```text
                      kind  perils  limit  attachment  annual limit  share  reinstatements  reinst. rate  rate on line  ceding commission  hours
alder_cat_1  occurrence_xl     all     40          30                  30%               1          100%          0.15                        72
alder_cat_2  occurrence_xl     all    100          70                  20%               1          100%          0.05                        72
bracken_cat  occurrence_xl     all     50          25                  25%               2          100%          0.14                        72
bracken_agg   aggregate_xl   CS+FL     30          40                  25%                                        0.06
corran_qs      quota_share     all    150           0           300    10%                                                            25%     72
```

The amounts are on a 100% basis. On an excess of loss contract, `share` is the
signed line of Larkspur Re. On the quota share, `share` is the cession, and
`limit` is the maximum for one event. The aggregate cover of Bracken applies
only to convective storm and flood. Its `limit` and `attachment` are the
aggregate limit and deductible. The quota share also pays Corran a 25% ceding
commission on the ceded premium.

### The year event loss table at a glance

```python
yelt = read("cat_book_yelt.csv")
years = {int(r["year"]) for r in yelt}
print(f"rows {len(yelt):,}, years with a loss {len(years):,} of 2,000")
print(f"first year with a loss {min(years)}, last {max(years)}")
```

```text
rows 16,432, years with a loss 1,820 of 2,000
first year with a loss 2, last 2000
```

The mean annual loss splits by peril and cedent.

```python
perils = ["WS", "EQ", "FL", "CS"]
names = [c["cedent"] for c in cedents]
total = {(p, n): 0.0 for p in perils for n in names}
for r in yelt:
    total[(r["peril"], r["cedent"])] += float(r["loss"])
mean_loss = pd.DataFrame({n: [total[(p, n)] / 2000 for p in perils] for n in names},
                         index=perils)
mean_loss["all"] = mean_loss.sum(axis=1)
mean_loss.loc["all"] = mean_loss.sum()
(GT(mean_loss.rename_axis("peril").reset_index(), rowname_col="peril")
 .tab_header("Mean annual loss by peril and cedent", "Millions of Varland crowns")
 .fmt_number(decimals=3))
```

```text
      alder  bracken  corran     all
WS    9.915    6.163   5.575  21.652
EQ    2.473    2.050   4.371   8.894
FL    3.042    3.999   1.997   9.038
CS    1.349    3.165   1.106   5.620
all  16.778   15.376  13.049  45.204
```

Two figures: the cedents by peril, then the annual loss of all cedents
together for the years with a loss.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
width = 0.27
for i, n in enumerate(names):
    ax.bar(np.arange(4) + (i - 1) * width, [total[(p, n)] / 2000 for p in perils],
           width, label=n)
ax.set_xticks(np.arange(4), perils)
ax.set_xlabel("peril")
ax.set_ylabel("mean annual loss (millions of crowns)")
ax.set_title("Windstorm is the largest peril for every cedent")
ax.legend(title="cedent")
```

```python
annual = {}
for r in yelt:
    annual[int(r["year"])] = annual.get(int(r["year"]), 0.0) + float(r["loss"])
values = np.array(list(annual.values()))
fig, ax = plt.subplots()
ax.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 40),
        color="0.75")
ax.axvline(values.sum() / 2000, linestyle="--", label="mean over all 2,000 years")
ax.set_xscale("log")
ax.set_xlabel("annual loss of the three cedents (millions of crowns)")
ax.set_ylabel("years")
ax.set_title("1,820 years with a loss: the annual loss spans four orders of magnitude")
ax.legend()
```

The mean annual loss over all years is the average annual loss (AAL, defined on
[page 1](cat-1-loss-tables.md)). Windstorm is the largest peril for every
cedent, also for Corran in the range, because a coastal storm is the largest
event in the catalog. The home region shows in the second peril: flood and
convective storm for Bracken, earthquake for Corran.

In 180 years no event hits a cedent. Year 1 is one of them, so the file starts
at year 2. A year with no row is still a year, so every average on these pages
divides by 2,000.

Each page has an R twin (same name, `.qmd`) with the same numbers.

## Further reading

[reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer is another
introduction to property catastrophe reinsurance.
