# An industry prior for a thin program, from Schedule P

A program with three or four years of data cannot set its own development
pattern or its own loss ratio. The Bayesian Bornhuetter-Ferguson
(`BayesBfNormalised`) can carry that outside view, but it needs numbers: an
expected loss ratio, how far a company's level strays from it, how far an
origin strays from its company, and a view of the pattern.
`prospicio.reserving.IndustryPrior` estimates these numbers from other
companies' triangles of the same line (#604, decision 29 of
`docs/design/reserving-bayes.md`).

No data ship with the library. This guide fetches Schedule P triangles from
the `cas-schedule-p` package at run time, builds a prior, fits a thin program
with it, and then tests the prior on every company of the panel. For the choice
of a reserving method in general, read
[`reserving-methods.md`](reserving-methods.md).

**How the numbers here are checked.** CI (`python/tests/test_guides.py`) runs
each `python` block on the two companies of
`validation/data/schedule_p_comauto_sample.csv`, and the block must print
exactly the text under it. The blocks that need the fetched panel are marked
`no-run`, because CI does not have the panel. Their output was recorded on
Windows with `cas-schedule-p` 2026.6.13. To run them again, install the package
and set `PROSPICIO_GUIDE_FETCH=1`: the test then runs them too and compares
their output. The sampler's digits can differ on another platform.

## 1. Fetch one panel of triangles

The Schedule P data are public research data from the CAS and NAIC, and they
are not in this repository. Install the package, with the extra that holds
Meyers' screen of companies:

```bash
pip install "cas-schedule-p[screens]==2026.6.13"
```

`screens.select_companies` chooses 50 companies of a line by the screen of CAS
Monograph 8's appendix. The block below reads their accident years 1988 to 1997
from the package's table and writes one CSV for four lines. Each company has
the full 10 x 10 square. The later diagonals are what happened next, so the
guide keeps them apart until it scores a fit. The columns have the names of the CAS
loss reserve database files.

```python no-run
import csv

import duckdb
from cas_schedule_p import path, screens

LINES = {"wc": "workers_compensation", "ca": "commercial_auto",
         "pa": "private_passenger_auto", "ol": "other_liability"}
mart = path("mart_reserving_model_training").as_posix()
database = duckdb.connect()
with open("schedule_p_panel.csv", "w", newline="") as f:
    out = csv.writer(f, lineterminator="\n")
    out.writerow(["line", "GRCODE", "AccidentYear", "DevelopmentLag", "CumPaidLoss_C", "EarnedPremNet_C"])
    for key, name in LINES.items():
        for code in screens.select_companies(line=name, per_line=50)["company_code"]:
            cells = database.execute(
                f"""select accident_year, development_age, cum_paid_loss, earned_prem_net
                    from read_parquet('{mart}')
                    where line_of_business = ? and company_code = ? and accident_year between 1988 and 1997
                    order by accident_year, development_age""", [name, str(code)]).fetchall()
            out.writerows([key, int(code), *cell] for cell in cells)
```

## 2. From rows to one triangle and one premium vector per company

`IndustryPrior.from_triangles` takes one triangle per company and one premium
per origin. The functions below build them from rows of that layout. `known`
cuts a company's square to what was known at the end of 1997, the upper
triangle, and its `first_origin` and `last_age` cut it further, for a thin
program. The block runs on the two commercial auto companies in the repository,
which have the same columns.

```python
import csv
from collections import defaultdict

from prospicio.reserving import BayesBfNormalised, ChainLadder, IndustryPrior, Triangle

VALUATION = 1997


def read_companies(path, line=None):
    """Rows by company code, of one line when the file holds several."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))
    companies = defaultdict(list)
    for r in rows:
        if line is None or r["line"] == line:
            companies[int(r["GRCODE"])].append(
                (int(r["AccidentYear"]), 12 * int(r["DevelopmentLag"]), float(r["CumPaidLoss_C"]),
                 float(r["EarnedPremNet_C"])))
    return companies


def known(rows, first_origin=0, last_age=120):
    """The cells known at the valuation date, from `first_origin` on and up to `last_age`."""
    return [r for r in rows if r[0] + r[1] // 12 - 1 <= VALUATION and r[0] >= first_origin and r[1] <= last_age]


def triangle(rows):
    return Triangle.from_long([r[0] for r in rows], [r[1] for r in rows],
                              {"paid": [r[2] for r in rows], "premium": [r[3] for r in rows]})


def premiums(rows):
    by_origin = {r[0]: r[3] for r in rows}
    return [by_origin[o] for o in sorted(by_origin)]


def industry_prior(companies, codes, **settings):
    rows = [known(companies[c]) for c in codes]
    return IndustryPrior.from_triangles([triangle(r) for r in rows], "paid", [premiums(r) for r in rows], **settings)


sample = read_companies("../../validation/data/schedule_p_comauto_sample.csv")
two = industry_prior(sample, sorted(sample))
print(two)
print(f"companies used {two.report['n_companies_used']}, origins used {two.report['n_origins_used']}")
print(two.report["warnings"][0])
```

```text
IndustryPrior(apriori=0.6043, between_cv=0.2586, cv=0.2179, ages=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120], companies=2)
companies used 2, origins used 16
only 2 companies: the between-company cv rests on 1 degrees of freedom (its relative standard error is about 71%); a prior from at least 10 is more reliable
```

A prior from two companies is only a check that the rows are read correctly,
and the report says so. `min_companies` (10 by default) sets when that warning
appears. It leaves the numbers alone.

## 3. The prior for workers' compensation

The same function on the 50 companies of the fetched panel:

```python no-run
companies = read_companies("schedule_p_panel.csv", "wc")
codes = sorted(companies)
prior = industry_prior(companies, codes)
print(f"apriori {prior.apriori:.4f} (se {prior.apriori_se:.4f})")
print(f"between cv {prior.between_cv:.4f}, within cv {prior.within_cv:.4f}, process cv {prior.process_cv:.4f}")
print(f"companies used {prior.report['n_companies_used']}, origins used {prior.report['n_origins_used']}, "
      f"warnings {len(prior.report['warnings'])}")
print("age      ", " ".join(f"{a:5d}" for a in prior.pattern.ages))
print("developed", " ".join(f"{p:5.3f}" for p in prior.percent_developed))
print("sd of the log share ratios from 24 months:", " ".join(f"{s:.2f}" for s in prior.pattern.sd))
```

```text
apriori 0.6385 (se 0.0193)
between cv 0.1890, within cv 0.2927, process cv 0.1697
companies used 50, origins used 450, warnings 0
age          12    24    36    48    60    72    84    96   108   120
developed 0.282 0.603 0.769 0.864 0.915 0.947 0.967 0.983 0.993 1.000
sd of the log share ratios from 24 months: 0.31 0.47 0.65 0.74 0.74 0.74 1.02 1.20 1.07
```

Each number has a place in the model:

* `apriori` is the mean over the companies of their mean ultimate loss ratio,
  each from the company's own chain ladder with no tail. The triangles' last
  age counts as ultimate.
* `between_cv` is the spread of a company's level about `apriori`. It becomes
  the model's `common_cv`, one error shared by all of a program's origins.
* `within_cv` is the spread of an origin about its company's level. It becomes
  the model's `cv`.
* `pattern` is a normal prior on each age's log share ratio, with the mean and
  sd over companies. The sd is large at the old ages, where each company's own
  estimate rests on few origins, so the prior leans on the industry little
  there.

## 4. What the settings change

```python no-run
print("setting                 apriori  between  cv      origins")
for name, settings in [("default", {}), ("min_developed=0", {"min_developed": 0.0}),
                       ("min_developed=0.9", {"min_developed": 0.9}),
                       ("subtract_process=True", {"subtract_process": True})]:
    p = industry_prior(companies, codes, **settings)
    print(f"{name:23} {p.apriori:.4f}   {p.between_cv:.4f}   {p.cv:.4f}  {p.report['n_origins_used']}")
```

```text
setting                 apriori  between  cv      origins
default                 0.6385   0.1890   0.2927  450
min_developed=0         0.6382   0.2022   0.2936  500
min_developed=0.9       0.6895   0.1887   0.2604  300
subtract_process=True   0.6385   0.1890   0.2385  450
```

* `min_developed` is the share of the ultimate an origin must have reached to
  count in the loss ratios. A young origin's ultimate is mostly the chain
  ladder's projection, and its error would pass for spread between origins. At
  0.9 only the old origins count, and the level moves to those years' loss
  ratios.
* `subtract_process` takes the chain ladder's ODP process error out of `cv`,
  since the model's likelihood adds it back. It is off by default because the
  Pearson scale of a chain ladder on Schedule P also carries misfit. On every
  incurred line of the panel it explains more than the whole within spread, and
  the subtraction would pin each origin to its company's level
  (`knowledge/findings/industry-prior-calibration-panel.md`). On workers'
  compensation paid it leaves a usable `cv`.
* `min_companies` only sets the warning.

## 5. A thin program

Take the company of median premium out of the panel and keep only its four
latest accident years: 1994 to 1997 at the end of 1997, ten cells, ages to 48
months. The prior comes from the other 49 companies.

The program is younger than the panel, and the pattern prior's ages must be the
triangle's. `at_age(48)` moves the prior to development to 48 months: `apriori`
becomes the loss ratio developed to 48 months, the pattern keeps the ages to 48
and `tail_factor` is the factor from 48 months to ultimate. **The model does not
apply the tail**: multiply its results by the factor yourself. The block below
shows the cut on the two companies.

```python
young = two.at_age(48)
print(f"apriori {two.apriori:.4f} to ultimate, {young.apriori:.4f} to 48 months")
print(f"ages {young.pattern.ages}, tail factor {young.tail_factor:.4f}")
print(young.report["warnings"][-1])
```

```text
apriori 0.6043 to ultimate, 0.5171 to 48 months
ages [12, 24, 36, 48], tail factor 1.1686
a prior for development to 48 months: 14.4% of the ultimate develops after it, outside the model; multiply the fit's ultimates by the tail factor 1.1686
```

The fit, beside a vague `BayesBfNormalised` (a flat prior on each origin,
`cv=inf`, which is the Bayesian chain ladder) and the chain ladder. All three
use the same tail factor.

```python no-run
SAMPLER = dict(chains=4, tune=500, draws=500, n_sims=2000)
FIRST_ORIGIN, THIN_AGE = 1994, 48


def thin_program(companies, code):
    """The thin triangle, its latest paid, and its paid at 48 and at 120 months: the outcomes."""
    rows = [r for r in companies[code] if r[0] >= FIRST_ORIGIN]
    latest = sum(r[2] for r in rows if r[0] + r[1] // 12 - 1 == VALUATION)
    paid_at = lambda age: sum(r[2] for r in rows if r[1] == age)
    return triangle(known(rows, FIRST_ORIGIN, THIN_AGE)), latest, paid_at(THIN_AGE), paid_at(120)


def ultimates(companies, codes, code):
    """The chain ladder's ultimate of the thin program and each model's draws, to 48 months, then the tail and outcomes."""
    prior = industry_prior(companies, [c for c in codes if c != code]).at_age(THIN_AGE)
    thin, latest, paid_48, paid_120 = thin_program(companies, code)
    fits = {"prior": prior.fit(thin, "paid", "premium", model=BayesBfNormalised(seed=code, **SAMPLER)),
            "vague": BayesBfNormalised(cv=float("inf"), seed=code, **SAMPLER).fit(thin, "paid", "premium")}
    draws = {k: sorted(latest + x for x in f.reserves.total().draws) for k, f in fits.items()}
    return latest + sum(ChainLadder().fit(thin, "paid").reserve), draws, prior.tail_factor, paid_48, paid_120


by_premium = sorted(codes, key=lambda c: sum(premiums(known(companies[c]))))
held_out = by_premium[len(by_premium) // 2]
chain, draws, tail, paid_48, paid_120 = ultimates(companies, codes, held_out)
print(f"company {held_out}, accident years 1994 to 1997, ultimate paid in thousands, tail factor {tail:.4f}")
print(f"chain ladder       {tail * chain:9,.0f}")
for name, d in draws.items():
    n = len(d)
    print(f"{name:18} {tail * sum(d) / n:9,.0f}   90% interval {tail * d[n // 20]:,.0f} to {tail * d[19 * n // 20]:,.0f}")
print(f"paid at 120 months {paid_120:9,.0f}")
```

```text
company 1066, accident years 1994 to 1997, ultimate paid in thousands, tail factor 1.1572
chain ladder          35,518
prior                 35,258   90% interval 33,746 to 37,030
vague                 35,547   90% interval 33,878 to 37,516
paid at 120 months    32,968
```

One company shows how to run the method and proves nothing about it. The next
section tests it on all of them.

## 6. Does it help? Every company in turn

The panel holds the outcomes, so the prior can be tested. The block below
repeats section 5 for each company of each line, always with the prior from the
other 49, and scores the three ultimates against the paid at 120 months. The
error is relative to that outcome. A company is skipped when the model refuses
its thin triangle: a column of one or two cells whose increments sum to zero or
less has no share to estimate, and the error says so.

```python no-run
def study(line):
    companies = read_companies("schedule_p_panel.csv", line)
    codes, rows, skipped = sorted(companies), [], 0
    for code in codes:
        try:
            rows.append(ultimates(companies, codes, code))
        except ValueError:
            skipped += 1
    return rows, skipped


def rmse(errors):
    return (sum(e * e for e in errors) / len(errors)) ** 0.5


def inside(d, x):
    return d[len(d) // 20] <= x <= d[19 * len(d) // 20]


print("                  rmse of the ultimate    90% cover at 120    90% cover at 48")
print("line scored skip  chain  vague  prior     vague  prior        vague  prior")
factors = {}
for line in ("wc", "ca", "pa", "ol"):
    rows, skipped = study(line)
    factors[line] = (rows[0][2], sorted(o / p for _, _, _, p, o in rows))
    rmses = [rmse([t * c / o - 1 for c, _, t, _, o in rows])]
    at_120, at_48 = [], []
    for k in ("vague", "prior"):
        rmses.append(rmse([t * sum(d[k]) / len(d[k]) / o - 1 for _, d, t, _, o in rows]))
        at_120.append(sum(inside([t * x for x in d[k]], o) for _, d, t, _, o in rows) / len(rows))
        at_48.append(sum(inside(d[k], p) for _, d, _, p, _ in rows) / len(rows))
    print(f"{line:4} {len(rows):6d} {skipped:4d}  {rmses[0]:.3f}  {rmses[1]:.3f}  {rmses[2]:.3f}     "
          f"{at_120[0]:.2f}   {at_120[1]:.2f}         {at_48[0]:.2f}   {at_48[1]:.2f}")
print("paid at 120 months over paid at 48, by company: the tail of the first company's prior, then the 10th and 90th percentiles")
for line, (tail, f) in factors.items():
    print(f"{line:4} {tail:.3f}   {f[len(f) // 10]:.3f}  {f[9 * len(f) // 10]:.3f}")
```

```text
                  rmse of the ultimate    90% cover at 120    90% cover at 48
line scored skip  chain  vague  prior     vague  prior        vague  prior
wc       48    2  0.141  0.158  0.135     0.42   0.38         0.71   0.73
ca       44    6  0.220  0.266  0.197     0.61   0.59         0.64   0.64
pa       48    2  0.085  0.085  0.088     0.48   0.42         0.58   0.50
ol       43    7  0.276  1.842  0.287     0.51   0.40         0.79   0.74
paid at 120 months over paid at 48, by company: the tail of the first company's prior, then the 10th and 90th percentiles
wc   1.155   1.068  1.308
ca   1.118   1.051  1.290
pa   1.083   1.028  1.170
ol   1.335   1.074  1.813
```

The table says three things, and the last matters most.

* **Against the vague model, the prior helps where a thin triangle is weak.**
  The error falls on workers' compensation and commercial auto, and it falls a
  long way on other liability, where a few thin triangles send the vague model's
  ultimates far off. Those few posteriors run away, so that one number moves with
  the smallest change: the R twin of this table prints 2.4 where this one prints
  1.8. On personal auto, the stable line, the prior is a little worse.
* **Against the plain chain ladder, the gain is small.** The prior is a little
  better on workers' compensation and commercial auto and a little worse on
  personal auto and other liability. All three methods share the tail factor,
  and the second table shows why that limits them: one factor stands for a
  spread of companies from well below it to well above it.
* **The intervals are too narrow. Do not read them as 90%.** Multiplying by a
  fixed tail removes all uncertainty about the development after 48 months, and
  the cover at 120 months is 0.38 to 0.61. Even to 48 months, with no tail in
  the way, it is 0.50 to 0.79, and the prior does not improve on the vague model.
  The model has no calendar-year effect, a likely cause. Widen the tail by hand
  (section 7), and read `knowledge/findings/industry-prior-thin-programs.md` for
  the details.

So the prior's value on a four-origin program is a stable answer, not a better
interval. It keeps a weak triangle from producing a wild ultimate. It does not
replace judgment about the tail. The sampler here is short, so a few fits warn
of an R-hat just above 1.01. A real analysis would use the default sampler.

## 7. Using it

* Build the prior on the same basis as the program: the same line, paid with
  paid and incurred with incurred, and premium on the same footing.
* A program as old as the panel needs no cut: `prior.fit(triangle, "paid",
  "premium")`. `prior.one_year(...)` gives the one-year view under the same
  prior.
* `at_age` keeps the cvs and does not add the spread of the percent developed.
  `percent_developed_sd` holds that spread: use it to widen the tail by hand,
  as section 6 shows you must.
* A model that pools the programs themselves, in place of fixing a prior from
  outside, is #599.
* `examples/ils_deal_analysis.ipynb` uses this prior on a deal, with synthetic
  peers so that it runs without the fetch.
