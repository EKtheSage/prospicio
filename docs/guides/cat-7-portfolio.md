# Property catastrophe 7: Larkspur's whole book

The earlier pages of this track price one contract at a time. A reinsurer
holds more than one contract. Larkspur Re holds five, on three cedents, and the
same windstorm can hit all of them in the same year. This page adds the five
contracts into one book and measures the tail of the book. It splits the
capital of the book back to the contracts, and it shows which contracts pay for
their capital.

The story, the files and the notation are on
[the overview](cat-0-overview.md). The page types no number by hand: every
attachment, limit and share comes from `cat_book_contracts.csv`.

## 1. The idea

Each row of the year event loss table belongs to one simulated year, and every
contract sees the same 2,000 years. Thus the loss of the book in a year is the
sum of the losses of the five contracts in that year. The dependence that the
events carry needs no copula and no reordering: one event in the catalog lands
on several cedents at once.

That is the only dependence in this book. The script that wrote the book draws
the loss of each cedent from an event independently. Thus the correlated part
of secondary uncertainty (page 1) is not correlated across cedents here. A real
model that shares it would give the book a heavier tail than this page shows.

Three questions follow from that sum.

- **How bad is the bad year of the book?** A tail measure of the summed loss,
  here TVaR at 99%: the mean of the worst 1% of years.
- **Which contracts cause it?** A split of the TVaR of the book back to the
  contracts. The Euler split gives each contract its mean loss in the years
  that make the tail of the book. [Price allocation](pir-price-allocation.md) compares it
  with the other ways to split.
- **Does each contract pay for the capital it uses?** A price or a return on
  that allocated capital, against the premium that the contract actually
  charges.

## 2. The book

The block below repeats the reader helpers of the earlier pages, because each
guide runs on its own. It builds the tower of each contract on its own subject
loss and wraps each result in a `DealRun`. A `DealBook` aligns the runs by
year.

```python
import csv
import math

from prospicio.aggregate import EventSet, read_ylt
from prospicio.pricing import DealBook, DealRun, price_portfolio
from prospicio.reinsurance import Layer, Tower
from prospicio.risk import Distortion, capital

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
    """The contract's terms from cat_book_contracts.csv as a Layer."""
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
    """The contract's subject loss: its cedent and perils, grouped by its hours clause."""
    e = cedent_events(c["cedent"], c["perils"])
    return e.grouped(float(c["hours"])) if c["hours"] else e


towers, results, runs = {}, {}, []
for name, c in CONTRACTS.items():
    towers[name] = Tower([contract_layer(c)])
    results[name] = towers[name].apply(subject(c))
    runs.append(DealRun.from_tower(name, towers[name], results[name], name))
book = DealBook(runs)
print(f"{len(book)} deals, {book.n_sims} years")
print("items:", ", ".join(book.items))
```

```text
5 deals, 2000 years
items: premium, loss, ceding_commission, total_deductions, nmbf, margin
```

`book.summary()` gives the mean of every item by deal and for the book, with its
ratio to the mean premium. The premium is the deposit of each year plus the
reinstatement premium that the losses of that year pay (section 3). The margin
is the premium less the loss less the deductions. Only `corran_qs` has a
deduction, its ceding commission. The data has no expenses of Larkspur, so the
margin is before them.

```python
s = book.summary()
rows = {(d, i): (m, r) for d, i, m, r in zip(s["deal"], s["item"], s["mean"], s["ratio"])}
print(f"{'deal':<12}{'premium':>9}{'loss':>8}{'deduct':>8}{'margin':>8}{'LR':>7}")
for d in [*book.names, "total"]:
    print(f"{d:<12}{rows[d, 'premium'][0]:>9.3f}{rows[d, 'loss'][0]:>8.3f}"
          f"{rows[d, 'total_deductions'][0] + 0.0:>8.3f}{rows[d, 'margin'][0]:>8.3f}{rows[d, 'loss'][1]:>7.3f}")
```

```text
deal          premium    loss  deduct  margin     LR
alder_cat_1     1.921   0.823   0.000   1.098  0.428
alder_cat_2     1.020   0.416   0.000   0.604  0.408
bracken_cat     1.846   0.685   0.000   1.161  0.371
bracken_agg     0.450   0.162   0.000   0.288  0.360
corran_qs       3.000   1.202   0.750   1.048  0.401
total           8.237   3.289   0.750   4.198  0.399
```

## 3. Reinstatement premium in the book

`DealRun.from_tower` adds the reinstatement premium of each year to the deposit
premium of the layer. The tower computes that premium for the two Alder layers
and `bracken_cat`, pro rata as to amount. Thus the premium of `alder_cat_1` is
1.8 in a quiet year and more in a year that reaches the layer.
`reinstatement_premium=False` keeps the deposit alone, for comparison. The losses do not
change, so the capital below does not change. Only the premium and the margin
change.

```python
amounts = book.amounts()
premium = amounts.marginal(["premium", "alder_cat_1"]).draws
print(f"alder_cat_1 book premium: min {min(premium):.3f}, max {max(premium):.3f}")
deposit = DealBook([DealRun.from_tower(n, towers[n], results[n], n, reinstatement_premium=False)
                    for n in book.names])
s_dep = deposit.summary()
charged = {d: m for d, i, m in zip(s_dep["deal"], s_dep["item"], s_dep["mean"]) if i == "premium"}
for d in book.names:
    print(f"{d:<12} reinstatement premium {rows[d, 'premium'][0] - charged[d]:.4f}")
```

```text
alder_cat_1 book premium: min 1.800, max 3.600
alder_cat_1  reinstatement premium 0.1207
alder_cat_2  reinstatement premium 0.0203
bracken_cat  reinstatement premium 0.0959
bracken_agg  reinstatement premium 0.0000
corran_qs    reinstatement premium 0.0000
```

## 4. The book's tail

`book.distribution("loss", ...)` gives three views of each deal at once.

- `"total"`: the statistics of the book.
- `"contribution"`: the mean loss of the deal in the years that make the
  statistic of the book. For TVaR this is the Euler split, and the
  contributions add up to the TVaR of the book.
- `"marginal"`: the statistic of the book less the same statistic of the book
  without that deal. It answers "what would we save by not writing this?".

`capital` below gives the stand-alone TVaR of each deal. The table shows it
in the first column, for comparison.

```python
dist = book.distribution("loss", [99.0], tvar_levels=[99.0])
loss = amounts.select("item", ["loss"]).select("deal", book.names)
alloc = capital(loss, Distortion.tvar(0.99))
print(f"book: mean {dist['total']['Mean']:.3f}, VaR 99 {dist['total']['Percentile 99']:.3f}, "
      f"TVaR 99 {dist['total']['TVaR 99']:.3f}")
print(f"{'deal':<12}{'alone':>8}{'contrib':>9}{'marginal':>10}")
for d, alone in zip(book.names, alloc.standalone):
    print(f"{d:<12}{alone:>8.3f}{dist['contribution'][d]['TVaR 99']:>9.3f}"
          f"{dist['marginal'][d]['TVaR 99']:>10.3f}")
contrib = sum(dist["contribution"][d]["TVaR 99"] for d in book.names)
print(f"sum of contributions {contrib:.3f}")
```

```text
book: mean 3.289, VaR 99 46.505, TVaR 99 55.096
deal           alone  contrib  marginal
alder_cat_1   13.843   13.195    12.728
alder_cat_2   20.998   19.026    14.560
bracken_cat   14.136    9.397     8.840
bracken_agg    7.500    0.262     0.193
corran_qs     16.363   13.216    12.609
sum of contributions 55.096
```

Read the table one row at a time.

- `alder_cat_2` is the high Alder layer. Alone, its TVaR is the largest of the
  five. It keeps most of that TVaR in the book, because the years that exhaust
  it are the big coastal windstorms that drive the whole book.
- `bracken_agg` looks risky alone: in its worst 1% of years it pays its whole
  share of the aggregate limit. These are mostly years of one large flood or
  storm outbreak on the plains (page 3, section 5). They are not the worst
  years of the book, so the cover adds little to the tail of the book.
- Contributions add up to the TVaR of the book. Marginals do not. When you
  remove one deal, the worst years change, so the marginals add up to less.

## 5. A trap: the total row

`book.amounts()` holds every item for each deal *and* a `"total"` deal. A
selection on the item alone keeps that total. `capital` then adds it to the
deals as a sixth contract, and the TVaR of the book doubles.

```python
wrong = capital(amounts.select("item", ["loss"]), Distortion.tvar(0.99))
right = capital(amounts.select("item", ["loss"]).select("deal", book.names), Distortion.tvar(0.99))
print(f"item only: total {wrong.total:.3f}, components {len(wrong.allocated)}")
print(f"item and deals: total {right.total:.3f}, components {len(right.allocated)}")
```

```text
item only: total 110.192, components 6
item and deals: total 55.096, components 5
```

Select the deals by name, `.select("deal", book.names)`, every time.

## 6. Capital and diversification

`capital(loss, Distortion.tvar(0.99))` allocates the TVaR of the book by
Euler, so the allocated amounts are the contributions above. Diversification is
the sum of the stand-alone TVaRs less the TVaR of the book:

```
diversification = sum over deals i of TVaR(X_i) - TVaR(X_1 + ... + X_5)
```

TVaR is subadditive, so this is never negative.

```python
print(f"{'deal':<12}{'alone':>8}{'allocated':>11}{'saved':>8}")
for d, a, b, c in zip(book.names, alloc.standalone, alloc.allocated, alloc.diversification()):
    print(f"{d:<12}{a:>8.3f}{b:>11.3f}{c:>8.3f}")
print(f"sum alone {sum(alloc.standalone):.3f}, book {alloc.total:.3f}, "
      f"benefit {alloc.diversification_benefit():.3f}")
```

```text
deal           alone  allocated   saved
alder_cat_1   13.843     13.195   0.648
alder_cat_2   20.998     19.026   1.973
bracken_cat   14.136      9.397   4.738
bracken_agg    7.500      0.262   7.238
corran_qs     16.363     13.216   3.147
sum alone 72.840, book 55.096, benefit 17.744
```

## 7. The book priced on its capital

`price_portfolio` prices the book as page 6, section 2 prices one contract:
`P = (EL + r A) / (1 + r)`, with the assets `A` now the TVaR of the book. It
then splits that premium to the deals by the same Euler weights
([portfolio pricing](pir-portfolio-pricing.md) has the theory). It also prices each deal
alone. Compare these prices with what Larkspur keeps. That is the premium
charged, plus the expected reinstatement premium of section 3, less the
deductions (the ceding commission of Corran).

```python
pp = price_portfolio(loss, Distortion.tvar(0.99), cost_of_capital=0.10)
print(f"book: EL {pp.total.expected_loss:.3f}, premium {pp.total.premium:.3f}, "
      f"capital {pp.total.capital:.3f}")
kept = {d: rows[d, "premium"][0] - rows[d, "total_deductions"][0] for d in [*book.names, "total"]}
print(f"{'deal':<12}{'charged':>9}{'kept':>8}{'in book':>9}{'alone':>8}")
for d, ab, al in zip(book.names, pp.allocated, pp.standalone):
    print(f"{d:<12}{charged[d]:>9.3f}{kept[d]:>8.3f}{ab.premium:>9.3f}{al.premium:>8.3f}")
print(f"{'book':<12}{charged['total']:>9.3f}{kept['total']:>8.3f}{pp.total.premium:>9.3f}")
print(f"premium saved by diversification {pp.diversification():.3f}")
```

```text
book: EL 3.289, premium 7.998, capital 47.097
deal          charged    kept  in book   alone
alder_cat_1     1.800   1.921    1.948   2.007
alder_cat_2     1.000   1.020    2.108   2.287
bracken_cat     1.750   1.846    1.477   1.908
bracken_agg     0.450   0.450    0.171   0.829
corran_qs       3.000   2.250    2.295   2.581
book            8.000   7.487    7.998
premium saved by diversification 1.613
```

Alone, `alder_cat_1` needs the 2.007 of page 6. In the book it needs less. The
book charges 8.000, near the 7.998 that a 10% return on TVaR capital needs. But
it keeps less: 7.487, after the reinstatement premium comes in and the ceding
commission of Corran goes out. Thus the whole book is below the target, and
section 9 shows a 9.0% return.

Both Alder layers keep less than their share, and `alder_cat_2` keeps about
half of it. The two Bracken contracts keep more than their share. The quota
share of Corran charges 3.000 but keeps 2.250, a little under its 2.295.

## 8. The marginal price of a new contract

The allocated price shares the price of the book among the deals already
written. A new contract asks a different question: what must it charge so that
the return of the book does not fall? Take the extra assets `dA` that it needs
and its expected loss `EL`. `dA` is its marginal TVaR, the book with it less
the book without it. With the same convention as above:

    P = EL + r * (dA - P),   so   P = (EL + r * dA) / (1 + r)

Below, each deal in turn is the last one written. The block computes the price
from `"marginal"` (#588).

```python
r = 0.10
print(f"{'deal':<12}{'EL':>7}{'dA':>8}{'marginal P':>12}{'charged':>9}")
for d in book.names:
    el, da = dist["marginal"][d]["Mean"], dist["marginal"][d]["TVaR 99"]
    p = (el + r * da) / (1 + r)
    print(f"{d:<12}{el:>7.3f}{da:>8.3f}{p:>12.3f}{charged[d]:>9.3f}")
```

```text
deal             EL      dA  marginal P  charged
alder_cat_1   0.823  12.728       1.905    1.800
alder_cat_2   0.416  14.560       1.702    1.000
bracken_cat   0.685   8.840       1.426    1.750
bracken_agg   0.162   0.193       0.165    0.450
corran_qs     1.202  12.609       2.239    3.000
```

The marginal price is lower than the allocated price for every deal, because
the marginal TVaR of each deal is at most its Euler contribution. TVaR is
subadditive, so a new deal raises the TVaR of the book by no more than the
deal's share of the new book's tail. Price the next contract on its marginal,
and the TVaR return of the book holds. Price every contract on its marginal,
and the book falls short.

## 9. A return on allocated capital

A risk-adjusted return on capital (RAROC) sets the margin of each deal against
the capital allocated to it. Here the capital is the allocated assets less the
premium of the deal, as above. The margin includes the reinstatement premium of
section 3. The data has no expense line, so this is a return before the costs
of Larkspur.

```python
print(f"{'deal':<12}{'margin':>8}{'capital':>9}{'return':>8}")
for d, a in zip(book.names, alloc.allocated):
    margin, premium = rows[d, "margin"][0], rows[d, "premium"][0]
    ret = f"{margin / (a - premium):.1%}" if a > premium else "none"
    print(f"{d:<12}{margin:>8.3f}{a - premium:>9.3f}{ret:>8}")
margin, premium = rows["total", "margin"][0], rows["total", "premium"][0]
print(f"{'book':<12}{margin:>8.3f}{alloc.total - premium:>9.3f}{margin / (alloc.total - premium):>8.1%}")
```

```text
deal          margin  capital  return
alder_cat_1    1.098   11.274    9.7%
alder_cat_2    0.604   18.005    3.4%
bracken_cat    1.161    7.552   15.4%
bracken_agg    0.288   -0.188    none
corran_qs      1.048   10.216   10.3%
book           4.198   46.859    9.0%
```

Against a 10% target, `alder_cat_2` is the weak contract. It uses more than a
third of the capital of the book and returns a few percent on it. `bracken_agg`
has no return to show. Its premium is larger than the assets allocated to it,
so in this book it needs no shareholder capital. A ratio over a negative
capital has no meaning.

## 10. Reproducibility

The VaR of the book above is the default inverted-cdf quantile. Page 2,
section 2 compares it with the other conventions.

Every number on this page follows from four committed files and one seed.

- `examples/data/make_cat_book.py` writes the files with a fixed seed. A test
  checks that the committed files are what the script writes.
- No step on this page draws a random number: the book adds the same 2,000
  years for every contract.
- `read_ylt` stamps the event set with a hash of the file that it read, so you
  can trace a result to its input. The rebuild by `cedent_events` drops that hash,
  so record it from `read_ylt` once.

```python
ylt = read_ylt(DATA + "cat_book_yelt.csv", year="year", loss="loss", slot="slot",
               labels=["peril", "region", "cedent"])
print(ylt.input_hash.split(":")[0], ylt.n_sims)
```

```text
blake3 2000
```

The page does not print the hash itself. The hash depends on the bytes of the
file, and a checkout that changes line endings to CRLF gives a different hash.

## In R

The R twin, [cat-7-portfolio.qmd](cat-7-portfolio.qmd), builds the book with
`deal_run_from_tower()`, `deal_run()` and `deal_book()`, and reads it with
`book_summary()`, `book_distribution()`, `capital_allocation()` and
`price_portfolio()`.

## Further reading

[Price allocation](pir-price-allocation.md) and
[portfolio pricing](pir-portfolio-pricing.md) cover allocation and pricing on
a book in depth. [reinsurance.dev](https://reinsurance.dev) by Oliver Baltzer
is another introduction to property catastrophe reinsurance.
