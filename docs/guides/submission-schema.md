# A submission's files and the deal report

A reinsurance submission arrives as a handful of files from the cedant or its
broker: the premium history, the rate changes, a loss triangle, a large-loss
listing, a risk profile and the terms of the program. This guide lists the
files the library's examples read, column by column, with the checks a file
must pass before an analysis uses it. It then turns the results of a deal
analysis into a few tidy tables and renders them with the report template in
[`examples/templates/`](../../examples/templates/README.md), which an
investor or a committee can read without the code.

The sample submission is in
[`examples/data/submission/`](../../examples/data/submission/). It is
synthetic: [`examples/data/make_submission.py`](../../examples/data/make_submission.py)
writes it from a seeded model of a commercial property book, and a test checks
the committed files against that script. Two notebooks read it:
[`submission_evaluation.ipynb`](../../examples/submission_evaluation.ipynb)
prices the per-risk excess of loss program, and
[`ils_deal_analysis.ipynb`](../../examples/ils_deal_analysis.ipynb) analyses a
quota share of the same book for an ILS investor.

## The files

Every file is a CSV with one header row, comma separated, UTF-8, with no
empty cells. Dates are ISO dates (`YYYY-MM-DD`), amounts are in the
program's currency (`currency` in `treaty_program.csv`) at the prices of the
date they refer to, and rates and shares are fractions (0.03 is 3%). The
column names are the contract: the notebooks read the columns by name, so a
file may carry more columns but not fewer, and a renamed column is a missing
one.

```python
import csv
from pathlib import Path

SUBMISSION = Path("../../examples/data/submission")
FILES = ["treaty_program.csv", "treaty_terms.csv", "subject_premium.csv", "rate_changes.csv",
         "cost_index.csv", "triangle.csv", "large_losses.csv", "risk_profile.csv"]


def read(name, folder=SUBMISSION):
    with open(folder / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


for name in FILES:
    with open(SUBMISSION / name, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    print(f"{name:20} {len(read(name)):4} rows  {','.join(header)}")
```

```text
treaty_program.csv     10 rows  item,value
treaty_terms.csv        3 rows  layer,limit,attachment,reinstatements,reinstatement_rate,quoted_rate_on_line
subject_premium.csv    10 rows  calendar_year,subject_premium
rate_changes.csv        9 rows  effective_date,average_rate_change,applies_to
cost_index.csv         40 rows  quarter_start,index
triangle.csv           55 rows  accident_year,development_months,paid,incurred
large_losses.csv      842 rows  claim_id,accident_date,report_date,policy_limit,valuation_date,paid,incurred
risk_profile.csv        8 rows  band,lower_sum_insured,upper_sum_insured,risks,total_sum_insured,premium
```

Which files an analysis needs depends on the analysis:

| Analysis | Files |
|---|---|
| Chain ladder on the cedant's book | `triangle.csv` |
| Loss ratios, Cape Cod, the Bayesian Bornhuetter-Ferguson | `triangle.csv`, `subject_premium.csv`, `rate_changes.csv` |
| A quota share for an investor (`ils_deal_analysis.ipynb`, the tables below) | `treaty_program.csv`, `subject_premium.csv`, `rate_changes.csv`, `triangle.csv` |
| Experience rating of a per-risk layer | the above, `treaty_terms.csv`, `large_losses.csv`, `cost_index.csv` |
| Exposure rating of a per-risk layer | `treaty_program.csv`, `treaty_terms.csv`, `risk_profile.csv` |

`submission_evaluation.ipynb` reads all eight. A file an analysis does not
need may be left out of the folder.

The tables below give each file's columns. Type is `integer`, `number` (a
decimal), `date` or `text`. The test of this guide reads the first column of
each table and requires it to equal the file's header, in order.

### `treaty_program.csv`

The program in one place, one row per item. The values are text; a reader
converts the numbers.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `item` | text | | the name of the item, one of the items below |
| `value` | text | | its value |

| Item | Type | Meaning | Read by |
|---|---|---|---|
| `cedant` | text | the ceding insurer | the reports |
| `line_of_business` | text | the class and the treaty type | the reports |
| `currency` | text | the currency of every amount in the submission | the reports |
| `period_start`, `period_end` | date | the treaty period | the reports |
| `basis` | text | losses occurring or risks attaching | not read by code |
| `estimated_subject_premium_income` | number | the cedant's estimate of the treaty year's subject premium (GNPI), at the treaty year's rates | the ceded premium, the loss in money |
| `brokerage` | number | brokerage as a fraction of the reinsurance premium | `Quote.technical(..., brokerage=)` |
| `listing_threshold` | number | the incurred amount above which a claim is in `large_losses.csv` | the listing check |
| `valuation_date` | date | the date of the triangle's latest diagonal and of the listing | the reports |

### `treaty_terms.csv`

One row per layer of the excess of loss program, as the broker quotes it.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `layer` | text | | the layer's name, as `5m xs 5m` |
| `limit` | number | money | the layer's limit per risk |
| `attachment` | number | money | its attachment (the cedant's retention) |
| `reinstatements` | integer | count | the number of reinstatements |
| `reinstatement_rate` | number | fraction | each reinstatement's premium as a fraction of the deposit, pro rata as to amount |
| `quoted_rate_on_line` | number | fraction | the quoted deposit premium over the limit |

Read by `Layer(name, limit, attachment, premium=, reinstatement_rates=)` and
`Tower([...])`; the quoted rate on line is compared with the technical one.

### `subject_premium.csv`

One row per calendar year: the earned premium the treaty applies to.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `calendar_year` | integer | year | the year the premium was earned |
| `subject_premium` | number | money | earned subject premium at the rates in force that year (not on level) |

Brought to the current rate level with `RateHistory(...).rate_level(...)`
and then the exposure of the loss ratio methods (the `onlevel` column of the
triangle in the notebooks).

### `rate_changes.csv`

One row per change in the cedant's rates.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `effective_date` | date | | the date the change takes effect |
| `average_rate_change` | number | fraction | the average change, 0.03 for +3% |
| `applies_to` | text | | which policies the change applies to; the examples treat every change as applying to policies written on or after the date |

Read by `RateHistory(dates, changes)` (Python) and `rate_level()` (R).

### `cost_index.csv`

One row per quarter: an index of claim costs, the source of the severity trend.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `quarter_start` | date | | the first day of the quarter (the sample's dates may be a day off) |
| `index` | number | index points | the cost index at that quarter |

Read by `TrendFit.exponential(times, values)`, with each time moved to the
middle of its quarter.

### `triangle.csv`

The cedant's whole book, one row per accident year and age.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `accident_year` | integer | year | the accident year |
| `development_months` | integer | months | the age, from the start of the accident year (12, 24, ...) |
| `paid` | number | money | cumulative paid losses |
| `incurred` | number | money | cumulative incurred losses (paid plus case reserves) |

Read by `Triangle.from_long(origins, ages, columns)` (Python) and
`triangle(rows, origin, development, columns)` (R).

### `large_losses.csv`

The large-loss listing: one row per claim and valuation date, from the
claim's report to the latest valuation.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `claim_id` | text | | the claim's identifier |
| `accident_date` | date | | the date of loss |
| `report_date` | date | | the date the cedant was told of it |
| `policy_limit` | number | money | the limit of the policy the claim falls on |
| `valuation_date` | date | | the date of this row's amounts |
| `paid` | number | money | cumulative paid on the claim at the valuation date |
| `incurred` | number | money | cumulative incurred at the valuation date |

Read by `LargeLossExperience(...)`, `DevelopmentPattern.from_history(...)`,
`Pareto.fit_with_standard_error(...)` and `PolicyLimits(...)`.

### `risk_profile.csv`

The portfolio of the treaty year in bands of sum insured, one row per band.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `band` | integer | | the band's number, from 1 |
| `lower_sum_insured` | number | money | the band's lower bound |
| `upper_sum_insured` | number | money | its upper bound |
| `risks` | integer | count | the number of risks in the band |
| `total_sum_insured` | number | money | the sum of their sums insured |
| `premium` | number | money | the band's premium |

Read by `RiskProfile(mean_sum_insured, risks, curve, premiums=, lower=, upper=)`
and then `expected_layer_loss(limit, attachment)`.

## Checking a submission

A file that reads is not yet a file an analysis can use. The function below
holds the checks the examples rely on and returns a list of problems, empty
when there are none. Each check holds on the sample. Two things that look
like checks are not among them, because real data breaks them and the sample
does too: development need not be monotone (incurred falls in three cells of
the triangle and on 255 rows of the listing, as case reserves are released),
and a quarter's date in the cost index need not be the first of its month.

```python
from datetime import date


def check_submission(folder=SUBMISSION):
    problems = []
    present = {name: read(name, folder) for name in FILES if (folder / name).exists()}

    def number(row, column):
        return float(row[column])

    for name, rows in present.items():
        if not rows:
            problems.append(f"{name}: no rows")
        for k, row in enumerate(rows):
            if any(v is None or v.strip() == "" for v in row.values()):
                problems.append(f"{name}, row {k + 1}: an empty cell")

    program = {r["item"]: r["value"] for r in present.get("treaty_program.csv", [])}
    for item in ["currency", "period_start", "period_end", "estimated_subject_premium_income"]:
        if program and item not in program:
            problems.append(f"treaty_program.csv: no item {item}")

    if "treaty_terms.csv" in present:
        terms = sorted(present["treaty_terms.csv"], key=lambda r: number(r, "attachment"))
        for r in terms:
            if number(r, "limit") <= 0 or number(r, "attachment") < 0:
                problems.append(f"treaty_terms.csv, {r['layer']}: limit and attachment must be positive")
            if not 0 < number(r, "quoted_rate_on_line") < 1:
                problems.append(f"treaty_terms.csv, {r['layer']}: a rate on line is between 0 and 1")
        for a, b in zip(terms, terms[1:]):
            if number(a, "attachment") + number(a, "limit") > number(b, "attachment"):
                problems.append(f"treaty_terms.csv: {a['layer']} overlaps {b['layer']}")

    if "subject_premium.csv" in present:
        years = [int(r["calendar_year"]) for r in present["subject_premium.csv"]]
        if years != list(range(years[0], years[0] + len(years))):
            problems.append("subject_premium.csv: the years are not consecutive and ascending")
        if any(number(r, "subject_premium") <= 0 for r in present["subject_premium.csv"]):
            problems.append("subject_premium.csv: a premium is not positive")

    for name, column in [("rate_changes.csv", "effective_date"), ("cost_index.csv", "quarter_start")]:
        if name in present:
            dates = [date.fromisoformat(r[column]) for r in present[name]]
            if any(a >= b for a, b in zip(dates, dates[1:])):
                problems.append(f"{name}: the dates are not strictly ascending")

    if "triangle.csv" in present:
        rows = present["triangle.csv"]
        cells = [(int(r["accident_year"]), int(r["development_months"])) for r in rows]
        if len(set(cells)) != len(cells):
            problems.append("triangle.csv: an accident year and age appear twice")
        ages = {}
        for year, age in cells:
            ages.setdefault(year, []).append(age)
        step = min(min(a) for a in ages.values())
        for year, a in ages.items():
            if sorted(a) != list(range(step, step * (len(a) + 1), step)):
                problems.append(f"triangle.csv, {year}: the ages are not {step}, {2 * step}, ... without a gap")
        for r in rows:
            if not 0 <= number(r, "paid") <= number(r, "incurred"):
                problems.append(f"triangle.csv, {r['accident_year']} at {r['development_months']}: paid is not between 0 and incurred")

    if "large_losses.csv" in present:
        rows = present["large_losses.csv"]
        keys = [(r["claim_id"], r["valuation_date"]) for r in rows]
        if len(set(keys)) != len(keys):
            problems.append("large_losses.csv: a claim and valuation date appear twice")
        fixed = {}
        for r in rows:
            claim = (r["accident_date"], r["report_date"], r["policy_limit"])
            if fixed.setdefault(r["claim_id"], claim) != claim:
                problems.append(f"large_losses.csv, {r['claim_id']}: its dates or limit change between rows")
            if not r["accident_date"] <= r["report_date"] <= r["valuation_date"]:
                problems.append(f"large_losses.csv, {r['claim_id']}: dates out of order")
            if not 0 <= number(r, "paid") <= number(r, "incurred") <= number(r, "policy_limit"):
                problems.append(f"large_losses.csv, {r['claim_id']} at {r['valuation_date']}: not 0 <= paid <= incurred <= limit")

    if "risk_profile.csv" in present:
        bands = sorted(present["risk_profile.csv"], key=lambda r: int(r["band"]))
        for a, b in zip(bands, bands[1:]):
            if number(a, "upper_sum_insured") != number(b, "lower_sum_insured"):
                problems.append(f"risk_profile.csv: band {a['band']} does not end where band {b['band']} starts")
        for r in bands:
            mean = number(r, "total_sum_insured") / number(r, "risks")
            if not number(r, "lower_sum_insured") <= mean <= number(r, "upper_sum_insured"):
                problems.append(f"risk_profile.csv, band {r['band']}: the mean sum insured is outside the band")
    return problems


print(check_submission() or "no problems")
```

```text
no problems
```

A copy of the folder with one claim's paid above its incurred shows how a
problem is reported:

```python
import shutil
import tempfile

broken = Path(tempfile.mkdtemp()) / "submission"
shutil.copytree(SUBMISSION, broken)
rows = read("large_losses.csv", broken)
rows[0]["paid"] = str(float(rows[0]["incurred"]) + 1)
with open(broken / "large_losses.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print("\n".join(check_submission(broken)))
shutil.rmtree(broken.parent)
```

```text
large_losses.csv, C2016-001 at 2016-12-31: not 0 <= paid <= incurred <= limit
```

## The report's tables

The report template reads its numbers from a folder of tidy tables, one row
per number, so the analysis that makes them and the page that shows them do
not depend on each other: any analysis, in Python, R or a spreadsheet, can
fill the report by writing these files. Every column is required; a file may
carry more.

| File | Columns | One row per |
|---|---|---|
| `deal.csv` | `item`, `value` | item of the deal: `cedant`, `line_of_business`, `currency`, `period_start`, `period_end`, `valuation_date`, `simulations` |
| `structures.csv` | `structure`, `item`, `value` | term of a structure (a share, a premium, a commission), as a number |
| `metrics.csv` | `structure`, `section`, `item`, `statistic`, `value` | statistic of an item of a section of the report, in a structure |
| `actual_vs_expected.csv` (optional) | `period`, `actual`, `expected`, `ratio`, `std_dev`, `z` | period of the monitoring, and a `total` row |

`section` in `metrics.csv` is one of the report's sections:

| `section` | `item` | `statistic` | Unit |
|---|---|---|---|
| `loss_ratio` | a ratio of `Tower.deal_ratios`: `loss_ratio`, `ceding_commission`, `brokerage`, `combined_ratio`, `margin` | `Mean`, `SD`, `Percentile 5`, `Percentile 50`, `Percentile 95` | fraction of the ceded premium |
| `layers` | `gross`, `ceded`, `net` (the kinds of `Tower.apply_annual`) | the same and `TVaR 99` | money |
| `returns` | `irr`, `moic` | `Mean`, `Percentile 5`, `Percentile 50`, `Percentile 95`, `Lower TVaR 5`, `Probability of loss`, and for the IRR `Simulations without a root` | fraction, multiple, count |
| `comparison` | an item of `DealBook.summary()`: `premium`, `loss`, each deduction, `nmbf`, `margin` | `Mean`, `Ratio to premium` | money, fraction |

The statistic names are those of `statistics()`, so a table from it melts
into `metrics.csv` without renaming.

The example below makes the tables for a 20% quota share of the sample
cedant's 2026 year, in two structures: a flat ceding commission and a
sliding scale. It is a short version of the analysis in
[`ils_deal_analysis.ipynb`](../../examples/ils_deal_analysis.ipynb), with
the ODP bootstrap in place of the notebook's Bayesian model, so that it runs
in a few seconds. First the cedant's book: the premium on level, the paid
triangle (the 120-month cell, only 2016 has one, is dropped, as in the
notebook) and its bootstrap.

```python
import math
import os
import tempfile
from statistics import NormalDist

from prospicio.models import actual_vs_expected
from prospicio.pricing import (CashflowTerms, CollateralRequirement, DealBook, DealRun, PaymentPattern, Period,
                               RateHistory, monthly_patterns, year_fraction)
from prospicio.reinsurance import Layer, Tower
from prospicio.reserving import OdpBootstrap, OdpGlm, Triangle
from prospicio.risk import statistics

N_SIMS, SEED = 2000, 20261028
program = {r["item"]: r["value"] for r in read("treaty_program.csv")}
premium = {int(r["calendar_year"]): float(r["subject_premium"]) for r in read("subject_premium.csv")}
YEARS = sorted(premium)
GNPI = float(program["estimated_subject_premium_income"])


def when(text):
    y, m, d = map(int, text.split("-"))
    return year_fraction(y, m, d)


rate_rows = read("rate_changes.csv")
history = RateHistory([when(r["effective_date"]) for r in rate_rows],
                      [float(r["average_rate_change"]) for r in rate_rows])
onlevel = {y: premium[y] * history.rate_level(Period.calendar(y, y + 1), 1.0).on_level_factor for y in YEARS}
paid_rows = [r for r in read("triangle.csv") if int(r["development_months"]) <= 108]


def paid_triangle(rows):
    return Triangle.from_long([int(r["accident_year"]) for r in rows], [int(r["development_months"]) for r in rows],
                              {"paid": [float(r["paid"]) for r in rows]})


boot = OdpBootstrap(n_sims=N_SIMS, seed=SEED).fit(paid_triangle(paid_rows), "paid")
RECENT = YEARS[-5:]
recent_premium = sum(onlevel[y] for y in RECENT)
pooled = [sum(u) / recent_premium for u in zip(*(boot.ultimate_draws(y) for y in RECENT))]
chain = boot.chain_ladder
by_year = [chain.ultimate[YEARS.index(y)] / onlevel[y] for y in RECENT]
mean_lr = sum(by_year) / len(by_year)
cv_year = math.sqrt(sum((x - mean_lr) ** 2 for x in by_year) / len(by_year)) / mean_lr
print(f"{RECENT[0]} to {RECENT[-1]} pooled ultimate loss ratio: mean {sum(pooled) / N_SIMS:.3f}; "
      f"between years: cv {cv_year:.3f}")
```

```text
2021 to 2025 pooled ultimate loss ratio: mean 0.496; between years: cv 0.082
```

The 2026 loss ratio in each simulation is the pooled loss ratio of the last
five years, which carries the reserve's uncertainty, times a lognormal factor
with mean 1. The factor's cv joins the cv between those years, for a single
year's own swing, and a cv of 15% for where the 2026 level stands, which the
bootstrap cannot see: it has no trend and no view of the market. That 15% is
an assumption of the example, the role the industry prior plays in the
notebook. The factor's normal scores are a fixed low-discrepancy sequence, as
in the notebook, so the draws do not depend on a random generator. Each
structure is a layer on the cedant's loss in money (a quota share on its
premium) and goes through the tower once.

```python
CV_LEVEL = 0.15
sigma = math.sqrt(math.log(1 + cv_year ** 2 + CV_LEVEL ** 2))
loss_ratio = [x * math.exp(sigma * NormalDist().inv_cdf((0.5 + i * 0.6180339887498949) % 1.0) - sigma ** 2 / 2)
              for i, x in enumerate(pooled)]
SHARE, BROKERAGE = 0.20, 0.025
SCALE = [(0.36, 0.45), (0.32, 0.52), (0.24, 0.70)]  # (commission, loss ratio) anchors
layers = {
    "flat": Layer.quota_share("flat", SHARE).with_premium_rate(1.0, GNPI).with_ceding_commission(0.32)
    .with_brokerage(BROKERAGE),
    "sliding": Layer.quota_share("sliding", SHARE).with_premium_rate(1.0, GNPI).with_sliding_scale(SCALE)
    .with_brokerage(BROKERAGE),
}
deals = {}
for name, layer in layers.items():
    tower = Tower([layer])
    result = tower.apply_annual([x * GNPI for x in loss_ratio])
    deals[name] = dict(tower=tower, result=result, ratios=tower.deal_ratios(result, name))
for name, d in deals.items():
    means = {k: sum(d["ratios"].marginal([k]).draws) / N_SIMS for k in ["loss_ratio", "ceding_commission", "margin"]}
    print(name, ", ".join(f"{k} {v:.3f}" for k, v in means.items()))
```

```text
flat loss_ratio 0.496, ceding_commission 0.320, margin 0.159
sliding loss_ratio 0.496, ceding_commission 0.329, margin 0.150
```

The investor's cash flows come from the cash-flow engine: the premium paid
into a trust, collateral of 30% of the unearned premium and 125% of the
outstanding loss, claims paid on the chain ladder's pattern spread over the
months, commutation at five years. `deduct_commission=True` takes each
simulation's commission (here the ceding commission and brokerage) off the
premium paid in, so one run covers a sliding scale. A book of one deal gives
its IRR and multiple (MOIC) in each simulation, and a book of both gives the
side-by-side means. The IRR is searched between -95% and 100% by default;
the best years of this deal return more than that, and outside the range
they would show as having no IRR and drop out of its statistics, so the
search here runs to 500%.

```python
T = 60
ages = [12 * (i + 1) for i in range(len(chain.cdf))]
pattern = PaymentPattern.from_development([a / 12 for a in ages], [*[max(1.0, c) for c in chain.cdf[:-1]], 1.0])
written, earned, paid = monthly_patterns([min(1.0, (m + 1) / 12) for m in range(T)], pattern)
terms = CashflowTerms(SHARE * GNPI, T, written, earned, paid,
                      CollateralRequirement({"unearned-premium": 0.30, "outstanding": 1.25}),
                      commutation_requirement=CollateralRequirement({"outstanding": 1.0}),
                      deduct_commission=True, annual_yield=0.03, expense_rate=0.005)
for name, d in deals.items():
    column = lambda k: d["ratios"].marginal([k]).draws
    flows = terms.run(column("loss_ratio"), [c + b for c, b in zip(column("ceding_commission"), column("brokerage"))])
    d["run"] = (DealRun.from_tower(name, d["tower"], d["result"], name)
                .with_flows(flows.flows, flows.funded, flows.opening).with_patterns(written, earned, paid))
    book = DealBook([d["run"]]).cashflows(upper=5.0)
    d["irr"], d["moic"] = list(book.irr.rate), list(book.moic)
summary = DealBook([d["run"] for d in deals.values()]).summary()
for name, d in deals.items():
    irr = sorted(x for x in d["irr"] if not math.isnan(x))
    print(f"{name}: median IRR {irr[len(irr) // 2]:+.3f}, mean MOIC {sum(d['moic']) / N_SIMS:.3f}")
```

```text
flat: median IRR +0.405, mean MOIC 1.378
sliding: median IRR +0.361, mean MOIC 1.351
```

Monitoring compares what happened with what was expected. Until the deal
has a valuation after binding there is nothing of its own to compare, so the
example fills the page with the cedant's own book: the paid in 2025 by
accident year, against the ODP GLM fitted to the triangle as it stood a year
earlier, with Pearson's dispersion for the z-scores.

```python
opening = [r for r in paid_rows if int(r["accident_year"]) + int(r["development_months"]) // 12 - 1 < YEARS[-1]]
glm = OdpGlm().fit(paid_triangle(opening), "paid")
cumulative = {(int(r["accident_year"]), int(r["development_months"])): float(r["paid"]) for r in paid_rows}
cells = [(int(o), d, mu) for (o, d), mu in zip(glm.future, glm.future_means) if int(o) + d // 12 - 1 == YEARS[-1]]
actual = [cumulative[o, d] - cumulative.get((o, d - 12), 0.0) for o, d, _ in cells]
ave = actual_vs_expected([o for o, _, _ in cells], "poisson", actual, [mu for _, _, mu in cells],
                         dispersion=glm.dispersion)
total = ave["total"]
print(f"paid in {YEARS[-1]}: actual {total['actual'] / 1e6:.1f}m, expected {total['expected'] / 1e6:.1f}m, z {total['z']:.2f}")
```

```text
paid in 2025: actual 50.7m, expected 51.6m, z -0.61
```

Last, the results melt into the tidy tables. `statistics()` returns columns
(a list per key and per statistic), so each of its rows becomes one row of
`metrics.csv` per statistic. The folder is a temporary one here; set
`PROSPICIO_DEAL_TABLES` to write it elsewhere. The committed sample in
[`examples/data/deal_report/`](../../examples/data/deal_report/) is this
example's output, and a test checks the two agree.

```python
OUT = Path(os.environ.get("PROSPICIO_DEAL_TABLES") or tempfile.mkdtemp())
OUT.mkdir(parents=True, exist_ok=True)
STATS = ["Mean", "SD", "Percentile 5", "Percentile 50", "Percentile 95"]
metrics = []


def melt(structure, section, table, key, keep, stats):
    for k, item in enumerate(table[key]):
        if item in keep:
            metrics.extend((structure, section, item, s, table[s][k]) for s in stats)


def lower_tvar(xs, p=0.05):
    n = max(1, int(p * len(xs)))
    return sum(sorted(xs)[:n]) / n


for name, d in deals.items():
    melt(name, "loss_ratio", statistics(d["ratios"], [5, 50, 95]), "ratio",
         ["loss_ratio", "ceding_commission", "brokerage", "combined_ratio", "margin"], STATS)
    kinds = d["result"].select("kind", ["gross", "ceded", "net"])
    melt(name, "layers", statistics(kinds, [5, 50, 95], tvar_levels=[99]), "kind", ["gross", "ceded", "net"],
         [*STATS, "TVaR 99"])
    for item, xs, loss in [("irr", d["irr"], 0.0), ("moic", d["moic"], 1.0)]:
        finite = [x for x in xs if not math.isnan(x)]
        st = statistics(finite, [5, 50, 95])
        metrics.extend((name, "returns", item, s, st[s]) for s in ["Mean", "Percentile 5", "Percentile 50", "Percentile 95"])
        metrics.append((name, "returns", item, "Lower TVaR 5", lower_tvar(finite)))
        metrics.append((name, "returns", item, "Probability of loss", sum(x < loss for x in finite) / len(finite)))
        if item == "irr":
            metrics.append((name, "returns", item, "Simulations without a root", len(xs) - len(finite)))
for k, deal_name in enumerate(summary["deal"]):
    if deal_name != "total":
        metrics.append((deal_name, "comparison", summary["item"][k], "Mean", summary["mean"][k]))
        metrics.append((deal_name, "comparison", summary["item"][k], "Ratio to premium", summary["ratio"][k]))

structures = []
for name in layers:
    structures += [(name, "share", SHARE), (name, "ceded_premium", SHARE * GNPI), (name, "brokerage", BROKERAGE)]
    if name == "flat":
        structures.append((name, "ceding_commission", 0.32))
    else:  # the scale's ends: the highest commission and the loss ratio it holds to, the lowest and where it starts
        (low, at_low), (high, at_high) = min(SCALE), max(SCALE)
        structures += [(name, "commission_max", high), (name, "loss_ratio_at_commission_max", at_high),
                       (name, "commission_min", low), (name, "loss_ratio_at_commission_min", at_low)]
deal = [("line_of_business", program["line_of_business"].split(",")[0] + ", quota share")]
deal += [(k, program[k]) for k in ["cedant", "currency", "period_start", "period_end", "valuation_date"]]
deal.append(("simulations", N_SIMS))
monitoring = [(r["period"], r["actual"], r["expected"], r["ratio"], r["std_dev"], r["z"]) for r in ave["periods"]]
monitoring.append(("total", total["actual"], total["expected"], total["ratio"], total["std_dev"], total["z"]))


def number(x):
    return f"{x:.10g}" if isinstance(x, float) else x


for name, header, rows in [("deal.csv", ["item", "value"], deal),
                           ("structures.csv", ["structure", "item", "value"], structures),
                           ("metrics.csv", ["structure", "section", "item", "statistic", "value"], metrics),
                           ("actual_vs_expected.csv", ["period", "actual", "expected", "ratio", "std_dev", "z"],
                            monitoring)]:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([number(x) for x in row] for row in rows)
    print(f"{name:24} {len(rows):3} rows")
```

```text
deal.csv                   7 rows
structures.csv            11 rows
metrics.csv              140 rows
actual_vs_expected.csv     9 rows
```

## Rendering the report

The template is
[`examples/templates/deal_report.qmd`](../../examples/templates/deal_report.qmd),
with R chunks, and its twin
[`deal_report_py.qmd`](../../examples/templates/deal_report_py.qmd), with
Python chunks; both show the same tables. Neither holds data: the folder of
tables is a parameter, and each stops with the name of a missing file or
column. From `examples/templates/`:

```sh
quarto render deal_report.qmd -P tables:../data/deal_report
quarto render deal_report_py.qmd -P tables:../data/deal_report
```

The R version needs `knitr` and `rmarkdown`; the Python version needs
Jupyter. Neither needs prospicio, since the analysis is done by then. The
pages show the deal's terms, the loss ratio and the ratios that make up the
result, the gross, ceded and net loss, the investor's IRR and MOIC with their
probability of loss, the two structures side by side, and the monitoring
table when `actual_vs_expected.csv` is in the folder.
