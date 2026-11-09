# Stochastic loss reserving with Bayesian MCMC: CAS Monograph 8 reproduced

<!-- site: execute -->

Glenn Meyers' *Stochastic Loss Reserving Using Bayesian MCMC Models* (CAS
Monograph 8, 2nd edition, 2019) asks one question of a reserving model. If an
actuary used the model on many real triangles, would the outcomes that came in
later look like draws from its predicted distributions? Mack's model and the
ODP bootstrap fail this test. Meyers then builds a sequence of Bayesian
models, fitted by Markov chain Monte Carlo (MCMC). Each model corrects a
failure of the one before.

This guide recomputes the monograph's printed results on its illustrative
insurer, with the book's value next to prospicio's. Read the monograph for the
argument and the 200-triangle results. Table numbers are the monograph's. Money
is in thousands of US dollars. For the choice between reserving methods in
general, read [`reserving-methods.md`](reserving-methods.md).

**How the numbers here are checked.** CI (`python/tests/test_guides.py`) runs
every Python block below that a text block follows. Each such block must print
exactly that text. The Bayesian fits use prospicio's default chains: 4 chains
of 1,000 warm-up and 2,000 kept draws. A fit takes less than a second in a
release build, so CI runs them all. Two facts matter:

* A Bayesian result is a Monte Carlo estimate. A different seed gives a
  slightly different number. Refits at seeds 0 to 19 measured the change (a
  recorded study that CI does not run; section 8 shows it).
* The digits of a NUTS run depend on the platform. The last bit of `exp` and
  `ln` differs on Windows and Linux, so the sampler paths diverge and the draws
  differ (prospicio's own seed-pinned test shows this). Thus each checked table
  shows the book's value, a tolerance of a few Monte Carlo errors, and whether
  prospicio's value is inside it. prospicio's own values from a real run are in
  section 8, in a block marked `python no-run`. **CI does not run that block or
  the full one-year view, also `no-run`.**

## 1. The illustrative insurer

Meyers' example is group 353 in the CAS Loss Reserve Database (his Tables 2.1
to 2.3): Commercial Auto, accident years 1988 to 1997 at 12 to 120 months, with
net earned premium. The database holds the full 10 by 10 square, so the lower
triangle, unknown when the insurer set its reserves, is known now. That makes
his test possible: fit the model on the upper triangle, then find where the
actual 120-month total falls in the predicted distribution.

```python
import csv
import math
import warnings
from pathlib import Path

import numpy as np

from prospicio.distributions import Lognormal
from prospicio.reserving import Mack, Meyers, OdpBootstrap, Triangle

DATA = Path("../../validation/data/clrd_comauto_353.csv")
with open(DATA, newline="") as f:
    square = list(csv.DictReader(line for line in f if not line.startswith("#")))

# The upper triangle: what was known at the end of 1997.
upper = [r for r in square if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]
tri = Triangle.from_long(
    origin=[int(r["origin"]) for r in upper],
    development=[int(r["development"]) for r in upper],
    values={k: [float(r[k]) for r in upper] for k in ("paid", "incurred", "premium")},
)
# The outcome: the 120-month values of the full square, summed over origins.
outcome = {k: sum(float(r[k]) for r in square if r["development"] == "120") for k in ("paid", "incurred")}
premium = [float(r["premium"]) for r in square if r["development"] == "12"]
latest = {k: sum(float(r[k]) for r in upper if int(r["origin"]) - 1988 + int(r["development"]) // 12 == 10)
          for k in ("paid", "incurred")}

print(f"cells in the upper triangle {len(upper)}, premium {sum(premium):,.0f}")
for k in ("paid", "incurred"):
    print(f"{k:8s} latest diagonal {latest[k]:,.0f}   outcome at 120 months {outcome[k]:,.0f}")
```

```text
cells in the upper triangle 55, premium 52,429
paid     latest diagonal 32,601   outcome at 120 months 40,000
incurred latest diagonal 35,789   outcome at 120 months 40,061
```

The premium and the two outcomes are the figures in his tables. Incurred is
case incurred (paid plus case reserves, no bulk reserves), as in the database.

## 2. The starting point: Mack's model and the outcome's percentile

Meyers starts from Mack's distribution-free chain ladder. Mack gives a mean and
a standard error of the ultimate, not a distribution. Meyers fits a log-normal
with that mean and sd to the total and reads the percentile of the outcome
(Table 3.1 for incurred, Table 4.1 for paid, which adds the ODP bootstrap). One
percentile tells little: 86 is not alarming. Section 3 shows what matters.

```python
import pandas as pd
from great_tables import GT


def lognormal_percentile(mean, sd, x):
    """Percentile of x under the log-normal with this mean and sd."""
    s2 = math.log(1 + (sd / mean) ** 2)
    z = (math.log(x / mean) + s2 / 2) / math.sqrt(s2)
    return 50 * (1 + math.erf(z / math.sqrt(2)))


# Meyers' printed totals: (estimate, standard error, percentile).
BOOK_MACK = {"incurred": (38_914, 1_057, 86.03), "paid": (39_177, 1_442, 72.02)}
rows = []
for col, (est, se, pct) in BOOK_MACK.items():
    mack = Mack().fit(tri, col)
    ours = lognormal_percentile(mack.total_ultimate, mack.total_standard_error, outcome[col])
    rows.append((col, est, mack.total_ultimate, se, mack.total_standard_error, pct, ours))
mack_table = pd.DataFrame(rows, columns=["triangle", "est_book", "est_ours", "se_book", "se_ours",
                                         "pct_book", "pct_ours"])
(GT(mack_table, rowname_col="triangle")
 .tab_header("Mack's total ultimate and the outcome's percentile",
             "Tables 3.1 and 4.1; thousands of dollars")
 .tab_spanner("estimate", ["est_book", "est_ours"])
 .tab_spanner("std error", ["se_book", "se_ours"])
 .tab_spanner("percentile", ["pct_book", "pct_ours"])
 .cols_label(est_book="book", est_ours="prospicio", se_book="book", se_ours="prospicio",
             pct_book="book", pct_ours="prospicio")
 .fmt_integer(["est_book", "est_ours", "se_book", "se_ours"])
 .fmt_number(["pct_book", "pct_ours"], decimals=2))
```

```text
            book  prospicio   book  prospicio   book  prospicio
incurred  38,914     38,914  1,057      1,057  86.03      86.06
paid      39,177     39,177  1,442      1,443  72.02      72.00
```

Mack's model is deterministic, so the table has no Monte Carlo error. The
paid standard error is 1,442.5 before rounding, so prospicio shows 1,443 and
the book 1,442. The percentiles differ by 0.03 and 0.02 points, from rounding
of the inputs or of the normal cdf.

The ODP bootstrap (England and Verrall) resamples residuals of an
overdispersed Poisson GLM on incremental paid values. Meyers ran R
ChainLadder with 10,000 simulations. This block runs prospicio's version and
counts the percentile from the simulations, as Meyers does for every
simulated model:

```python
odp = OdpBootstrap(n_sims=10_000, seed=0).fit(tri, "paid")
odp_total = np.array(odp.reserves.draw_matrix()).sum(axis=1) + latest["paid"]
odp_pct = 100 * np.mean(odp_total <= outcome["paid"])
odp_table = pd.DataFrame({
    "measure": ["estimate", "std error", "percentile"],
    "book": [39_193, 1_389, 73.91],
    "prospicio": [odp_total.mean(), odp_total.std(), odp_pct],
})
(GT(odp_table, rowname_col="measure")
 .tab_header("ODP bootstrap, paid total", "10,000 simulations; thousands of dollars")
 .fmt_integer(["book", "prospicio"], rows=[0, 1])
 .fmt_number(["book", "prospicio"], rows=[2], decimals=2))
```

```text
              book  prospicio
estimate    39,193     39,231
std error    1,389      1,410
percentile   73.91      72.73
```

The figure shows both predictive distributions of the paid total and the
outcome in the lower triangle. The outcome falls above the middle of both.

```python
import matplotlib.pyplot as plt

s2 = math.log(1 + (mack.total_standard_error / mack.total_ultimate) ** 2)
x = np.linspace(odp_total.min(), odp_total.max(), 300)
density = np.exp(-(np.log(x) - math.log(mack.total_ultimate) + s2 / 2) ** 2 / (2 * s2)) / (x * math.sqrt(2 * math.pi * s2))
fig, ax = plt.subplots()
ax.hist(odp_total, bins=60, density=True, color="0.75", label="ODP bootstrap, 10,000 draws")
ax.plot(x, density, label="Mack's log-normal")
ax.axvline(outcome["paid"], color="black", linestyle="--", label="outcome at 120 months")
ax.set_xlabel("total paid at 120 months")
ax.set_ylabel("density")
ax.set_title("Predicted distribution of the paid total, and the outcome")
ax.legend()
```

The bootstrap has no NUTS sampler, so it shows the same digits on every
platform. Over seeds 0 to 19 (section 8), seed 0 is low on all three measures,
by about two sds. The 20-seed means are within 4, 4 and 0.13 of the book, so
the two bootstraps agree within error. Both add process error from a gamma
with the mean and variance of the ODP (R's `process.distr = "gamma"`,
prospicio's default `process="gamma"`).

## 3. The validation method: percentiles, p-p plots and the KS test

The monograph tests a model's reputation, not one triangle. If a model's
predictive distributions are correct, the percentile of each outcome is
uniform on 0 to 100 across many triangles. Meyers fits each model to 200
triangles (50 each of commercial auto, personal auto, workers' compensation
and other liability) and tests whether the 200 percentiles are uniform.

* The **p-p plot** puts the sorted percentiles against their expected values
  under uniformity, `100 i / (n + 1)`. A good model lies on the 45 degree line.
* The **shape** of a failure is informative. Too many extreme percentiles (an
  S tilted along the diagonal) mean the predicted distribution is too light in
  the tails. Too many middle percentiles (a backward S) mean it is too heavy.
  Too many low percentiles mean the model predicts too high.
* The **Kolmogorov-Smirnov** (KS) test makes the plot a test. With sorted
  percentiles `p_i` and `f_i = 100 i / n`, the statistic is
  `D = max |p_i - f_i|`. The test rejects uniformity at the 5% level when `D`
  exceeds `136 / sqrt(n)`, a band around the diagonal. A model is "validated"
  in the monograph's sense when it stays in the band.
* The Anderson-Darling test weights the tails more. Almost every model failed
  it, so Meyers kept KS.

A short simulation shows what the test sees. One set of percentiles comes from
a correct model. The other comes from a model that is too narrow (the outcomes
spread 1.5 times wider than predicted):

```python
def ks_d(percentiles):
    p = np.sort(np.asarray(percentiles))
    f = 100 * np.arange(1, len(p) + 1) / len(p)
    return np.max(np.abs(p - f))


for n in (50, 200):
    print(f"n = {n:3d}: critical value 136 / sqrt(n) = {136 / math.sqrt(n):.1f}")

# 200 standard normal draws: the logs of a standard log-normal sample.
z = np.log(Lognormal(0.0, 1.0).sample(200, seed=353))
normal_cdf = np.vectorize(lambda x: 50 * (1 + math.erf(x / math.sqrt(2))))
print(f"right model:            D = {ks_d(normal_cdf(z)):4.1f}")
print(f"too light in the tails: D = {ks_d(normal_cdf(1.5 * z)):4.1f}")
```

```text
n =  50: critical value 136 / sqrt(n) = 19.2
n = 200: critical value 136 / sqrt(n) = 9.6
right model:            D =  5.6
too light in the tails: D = 15.0
```

In the p-p plot of the same samples, the grey band is the 5% limit for 200
triangles, 9.6 points either side of the diagonal. The model that is too light
in the tails leaves the band at both ends.

```python
fig, ax = plt.subplots()
expected = 100 * np.arange(1, 201) / 201
band = 136 / math.sqrt(200)
ax.fill_between([0, 100], [-band, 100 - band], [band, 100 + band], color="0.85", label="5% KS band")
ax.plot(expected, np.sort(normal_cdf(z)), label="right model")
ax.plot(expected, np.sort(normal_cdf(1.5 * z)), label="too light in the tails")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("expected percentile under uniformity")
ax.set_ylabel("sorted percentile of the outcome")
ax.set_title("p-p plot of 200 simulated outcomes")
ax.legend()
```

On his panel, two failures set the agenda for the rest of the book:

* Mack on **incurred** triangles is too light in the tails (D = 15.4 over the
  200 triangles, against 9.6).
* Mack and the ODP bootstrap on **paid** triangles predict too high (D of 23.1
  and 24.1, with the percentiles at the low end).

## 4. The cross classified model (CRC)

The first Bayesian model is a log-normal chain ladder on cumulative values,
with premium as the base:

* `log C[w, d]` is normal with mean `log P[w] + logelr + alpha[w] + beta[d]`
  and standard deviation `sigma[d]`.
* `logelr` is the log of the expected loss ratio. `alpha[w]` is an accident
  year level (`alpha[1] = 0`). `beta[d]` is a development effect
  (`beta[10] = 0`, so the 120-month loss ratio of the first year is about
  `exp(logelr)`).
* `sigma[d]` decreases with age. Its square is a sum of uniform steps over the
  later ages, because older cells hold more settled claims.
* For each posterior draw, the predictive distribution of the 120-month value
  of each origin is a log-normal draw. It includes parameter and process
  uncertainty.

Against Mack, CRC gives a full distribution instead of two moments, puts a
prior on a quantity actuaries know (the loss ratio), and has a variance that
falls with age. It keeps the chain ladder's structure. The priors are wide on
purpose: Meyers treats them as part of the model, not as beliefs.

prospicio's `Meyers` uses the priors of his published Stan code. They differ in
detail from the text: the prior sd is 3.162 (`sqrt(10)`), where the text says
10, and each variance step has a floor of about `1e-5`. The knowledge note
[`meyers-monograph-8.md`](../../knowledge/references/meyers-monograph-8.md)
lists the differences. The finding
[`meyers-crc-csr-parity.md`](../../knowledge/findings/meyers-crc-csr-parity.md)
gives the parity with his code at 200 seeds.

```python
def fit(model, column, seed=0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Meyers(model, seed=seed).fit(tri, column, "premium")
    return result, len(caught)


def outcome_summary(result, column):
    """Total 120-month estimate, its sd, and the outcome's percentile, counted as Meyers does."""
    total = np.array(result.ultimates.draw_matrix()).sum(axis=1)
    return total.mean(), total.std(), 100 * np.mean(total <= outcome[column])


OURS = {}  # prospicio's values, shown in section 8
ROWS = []  # the checks of the block being run


def check(label, book, ours, tol, places=0):
    """Record the book's value and whether prospicio's is within tol of it."""
    OURS[label] = (book, ours)
    ROWS.append((label, book, tol, "yes" if abs(ours - book) <= tol else "no", places))


def places_table(frame, columns):
    """A table whose rows each have their own decimals (the `places` column)."""
    table = GT(frame.drop(columns="places"), rowname_col=frame.columns[0])
    for places in sorted(set(frame["places"])):
        table = table.fmt_number(columns, rows=list(frame.index[frame["places"] == places]), decimals=places)
    return table


def check_table(title, subtitle, strip=""):
    """Show the checks recorded since the last table, then clear them."""
    frame = pd.DataFrame(ROWS, columns=["check", "book", "tolerance", "agrees", "places"])
    frame["check"] = frame["check"].str.removeprefix(strip)
    ROWS.clear()
    return places_table(frame, ["book", "tolerance"]).tab_header(title, subtitle)


def check_parameter(result, label, name, book_mean, book_sd):
    """Means within 0.15 posterior sd, sds within 15%: the tolerances of prospicio's parity tests."""
    s = result.summary().set_index("parameter")
    check(f"{label} {name} mean", book_mean, s.loc[name, "mean"], 0.15 * book_sd, 4)
    check(f"{label} {name} sd", book_sd, s.loc[name, "sd"], 0.15 * book_sd, 4)


def check_outcome(result, label, column, book_est, book_sd, book_pct):
    """Totals within 1% and percentiles within 5 points (several Monte Carlo errors);
    the total's sd within 50%, as in the parity tests, since it is heavy-tailed."""
    est, sd, pct = outcome_summary(result, column)
    check(f"{label} estimate", book_est, est, 0.01 * book_est)
    check(f"{label} sd of the total", book_sd, sd, 0.5 * book_sd)
    check(f"{label} percentile", book_pct, pct, 5.0, 2)


crc_paid, n_warn = fit("crc", "paid")
crc_inc, n_warn_inc = fit("crc", "incurred")
d = crc_paid.diagnostics
print(f"warnings {n_warn}, divergences {d['divergences']}, max R-hat below 1.01: {d['max_rhat'] < 1.01},"
      f" posterior draws {len(crc_paid.draws):,}, simulations {crc_paid.ultimates.n_sims:,}")
print("warnings on the incurred fit:", n_warn_inc)
```

```text
warnings 0, divergences 0, max R-hat below 1.01: True, posterior draws 8,000, simulations 10,000
warnings on the incurred fit: 0
```

prospicio does not stop a fit on bad diagnostics. It raises one `UserWarning`
and keeps the diagnostics in `fit.diagnostics`. The table gives the parameters
of Table 5.1 (the values come from the rendered page, since the PDF's text
layer scrambles that table). The tolerances are those of prospicio's parity
tests, which hold at 200 seeds against Meyers' Stan code: 0.15 posterior sd on
a mean and 15% on an sd. A rerun of that code agrees with his printed tables
within its own error.

```python
for name, (mean, sd) in {"logelr": (-0.3965, 0.0233), "alpha[10]": (0.3435, 0.3316),
                         "beta[1]": (-1.1999, 0.1156), "sigma[1]": (0.2965, 0.1034),
                         "sigma[10]": (0.0202, 0.0128)}.items():
    check_parameter(crc_paid, "CRC paid", name, mean, sd)
check_table("CRC paid: the posterior against Table 5.1", "Tolerance: 0.15 book sd on a mean and on an sd",
            strip="CRC paid ")
```

```text
                   book  tolerance  agrees
logelr mean     -0.3965     0.0035     yes
logelr sd        0.0233     0.0035     yes
alpha[10] mean   0.3435     0.0497     yes
alpha[10] sd     0.3316     0.0497     yes
beta[1] mean    -1.1999     0.0173     yes
beta[1] sd       0.1156     0.0173     yes
sigma[1] mean    0.2965     0.0155     yes
sigma[1] sd      0.1034     0.0155     yes
sigma[10] mean   0.0202     0.0019     yes
sigma[10] sd     0.0128     0.0019     yes
```

The level `alpha[10]` of the youngest year rests on one cell, so its sd is
large. Next come the predictive distribution and the percentile of the outcome
(Tables 5.2 and 5.3):

```python
check_outcome(crc_paid, "CRC paid", "paid", 40_121, 2_487, 51.88)
check_outcome(crc_inc, "CRC incurred", "incurred", 39_147, 1_642, 74.75)
check_table("CRC: the total at 120 months against Tables 5.2 and 5.3",
            "Thousands of dollars; tolerance 1% on the estimate, 50% on the sd, 5 points on the percentile")
```

```text
                                book  tolerance  agrees
CRC paid estimate             40,121        401     yes
CRC paid sd of the total       2,487      1,244     yes
CRC paid percentile            51.88       5.00     yes
CRC incurred estimate         39,147        391     yes
CRC incurred sd of the total   1,642        821     yes
CRC incurred percentile        74.75       5.00     yes
```

"Estimate" is the mean total at 120 months, as in Meyers' tables. It includes
the oldest year's observed 3,912 (paid) or 3,917 (incurred). The sd of the
total depends on the one-cell youngest year through a log-normal, so it is
heavy-tailed. On Windows, seed 0 is high on the incurred percentile (76.90, the
largest of the 20 seeds) and low on the incurred sd (1,541). Meyers' own run is
also a Monte Carlo estimate, with errors of about the same size. The CmdStan
rerun of his code gives totals of 40,136 (2,497) and 39,132 (1,616). Within
these errors, the book and prospicio agree.

On the 200 triangles, CRC is no better than Mack and the ODP: it predicts too
high on paid data and its tails are too light on incurred data. Each of the
next two models corrects one of these failures.

## 5. Checking a model on the data in hand

The p-p plot measures the reputation of a model. An actuary who sets reserves
today needs checks on the upper triangle alone. Meyers uses two.

**Standardized residuals.** For a sample of posterior draws, calculate
`(log C[w, d] - mu[w, d]) / sigma[d]` for every observed cell, then make box
plots by accident year and by development year. All his models look similar by
development year. The accident year plots show the differences. The residuals
come directly from prospicio's draws:

```python
cells = [(int(r["origin"]) - 1988, int(r["development"]) // 12 - 1, float(r["paid"])) for r in upper]


def standardized_residuals(result, gamma=None):
    """Residuals of every observed cell under every posterior draw (draws x cells)."""
    draws = np.array(result.draws)
    col = {name: k for k, name in enumerate(result.parameter_names)}
    out = []
    for w, d, value in cells:
        mu = math.log(premium[w]) + draws[:, col["logelr"]]
        if w > 0:
            mu = mu + draws[:, col[f"alpha[{w + 1}]"]]
        if d < 9:
            beta = draws[:, col[f"beta[{d + 1}]"]]
            if gamma is not None:
                beta = beta * (1 - draws[:, col[gamma]]) ** w
            mu = mu + beta
        out.append((math.log(value) - mu) / draws[:, col[f"sigma[{d + 1}]"]])
    return np.array(out).T


years = np.array([w for w, _, _ in cells])


def year_medians(r):
    """The median standardized residual of each accident year."""
    return np.array([np.median(r[:, years == w]) for w in range(10)])


crc_medians = year_medians(standardized_residuals(crc_paid))
print(f"CRC paid: largest |median residual| of an accident year {np.abs(crc_medians).max():.1f},"
      f" in {1988 + np.abs(crc_medians).argmax()}")
```

```text
CRC paid: largest |median residual| of an accident year 0.4, in 1993
```

No accident year is far from zero. On Windows at seed 0 the medians are -0.16, 0.25, -0.17, 0.10, -0.11, 0.36, -0.03, 0.16,
0.01 and 0.00 for 1988 to 1997, inside the interquartile range of a standard
normal, as in Meyers' box plots for CRC paid. The largest, 1993, is the year
CSR changes most (section 6).

**Leave-one-out cross-validation.** `elpd_loo` estimates how well the model
predicts each observed cell when the fit leaves that cell out. It uses Pareto
smoothed importance sampling of the posterior instead of a refit (Vehtari,
Gelman and Gabry, the R `loo` package that Meyers uses). Higher is better. It
compares only models fitted to the same column (`fit.loo()` returns it). His
Table 6.4 compares CRC with the stochastic Cape Cod (SCC), which is CRC without
accident year levels. SCC is much worse on every measure, and Meyers does not
use it again.

```python
# elpd_loo within 2, the parity tests' tolerance.
check("CRC paid elpd_loo", 47.80, crc_paid.loo().elpd, 2.0, 2)
check("CRC incurred elpd_loo", 70.97, crc_inc.loo().elpd, 2.0, 2)
check_table("CRC: elpd_loo against Table 6.4", "Tolerance 2")
```

```text
                        book  tolerance  agrees
CRC paid elpd_loo      47.80       2.00     yes
CRC incurred elpd_loo  70.97       2.00     yes
```

prospicio gives 47.90 and 71.42 at seed 0 on Windows, and elpd_loo moves by
about 0.3 to 0.4 between seeds. The CmdStan rerun of Meyers' code gives 48.04
and 71.22, so a difference of up to half a point is run error.

## 6. The changing settlement rate model (CSR)

CSR keeps CRC, but the development effects shrink toward zero in the later
accident years. The mean becomes
`log P[w] + logelr + alpha[w] + beta[d] (1 - gamma)^(w - 1)`, with a
`normal(0, 0.05)` prior on `gamma`. A positive `gamma` means claims settle
faster in the later years, so the paid values of a young year are further along
than the old pattern shows, and the reserve is smaller. With `gamma = 0` the
model is CRC.

CSR corrects the bias of CRC on paid data. Over the 200 triangles, its paid p-p
plots are the first to stay in the KS band.

```python
csr_paid, n_warn_csr = fit("csr", "paid")
print("warnings", n_warn_csr)
csr_medians = year_medians(standardized_residuals(csr_paid, gamma="gamma"))
print(f"CSR paid: largest |median residual| of an accident year {np.abs(csr_medians).max():.1f}"
      f" (CRC {np.abs(crc_medians).max():.1f})")
```

```text
warnings 0
CSR paid: largest |median residual| of an accident year 0.3 (CRC 0.4)
```

```python
check_parameter(csr_paid, "CSR paid", "gamma", 0.0446, 0.0282)
check_parameter(csr_paid, "CSR paid", "logelr", -0.3956, 0.0246)
check_outcome(csr_paid, "CSR paid", "paid", 37_597, 2_401, 86.26)
check("CSR paid elpd_loo", 49.76, csr_paid.loo().elpd, 2.0, 2)
check_table("CSR paid: the posterior and the total against Tables 7.1 to 7.3",
            "Thousands of dollars; the tolerances of CRC", strip="CSR paid ")
```

```text
                    book  tolerance  agrees
gamma mean        0.0446     0.0042     yes
gamma sd          0.0282     0.0042     yes
logelr mean      -0.3956     0.0037     yes
logelr sd         0.0246     0.0037     yes
estimate          37,597        376     yes
sd of the total    2,401      1,200     yes
percentile         86.26       5.00     yes
elpd_loo           49.76       2.00     yes
```

A faster settlement of about 4.5% a year lowers the estimate by about 2,500
against CRC and moves the outcome from the 52nd to the 86th percentile. For
this insurer, CSR now predicts less than the outcome. elpd_loo prefers CSR
here by about 2, and the accident year residuals are a little flatter (his
Figure 7.3 against 5.3). Over the 200 triangles, elpd_loo prefers CSR for 105.
Over the panel, CRC's paid forecasts are too high and CSR's are in the KS band.

## 7. The correlated accident year model (CAY)

CAY starts again from CRC and corrects the other failure: light tails on
incurred data. The mean of each accident year adds `rho` times the residual of
the previous year at the same age, `log C[w - 1, d] - mu[w - 1, d]`. The prior
keeps `rho` in `(-1, 1)`, centred on 0.

John Major shows in the monograph (eq. 8.3) that the log values are then a
moving average down each column. Adjacent years correlate by
`rho / (1 + rho^2)`, and years two apart do not correlate. Correlated years add
to a wider total, so the predicted distribution has heavier tails. The
prediction goes year by year: each 120-month draw goes into the mean of the
next year.

```python
cay_inc, n_warn_cay = fit("cay", "incurred")
print("warnings", n_warn_cay)
```

```text
warnings 0
```

```python
check_parameter(cay_inc, "CAY incurred", "rho", 0.1709, 0.2071)
check_parameter(cay_inc, "CAY incurred", "logelr", -0.3945, 0.0150)
check_outcome(cay_inc, "CAY incurred", "incurred", 39_193, 1_859, 73.24)
check("CAY incurred elpd_loo", 68.65, cay_inc.loo().elpd, 2.0, 2)
check_table("CAY incurred: the posterior and the total against Tables 8.1 to 8.3",
            "Thousands of dollars; the tolerances of CRC", strip="CAY incurred ")
```

```text
                    book  tolerance  agrees
rho mean          0.1709     0.0311     yes
rho sd            0.2071     0.0311     yes
logelr mean      -0.3945     0.0022     yes
logelr sd         0.0150     0.0022     yes
estimate          39,193        392     yes
sd of the total    1,859        930     yes
percentile         73.24       5.00     yes
elpd_loo           68.65       2.00     yes
```

The posterior of `rho` is wide (sd 0.21), and the sd of the total is about 15%
more than CRC's. elpd_loo prefers CRC on this triangle, as Meyers reports.
Over the 200 triangles, elpd_loo prefers CAY for only 26. But the score on the lower
triangle (`elpd_test`) prefers CAY for 121, and its incurred p-p plots are
better. Thus, in his words, the choice of CAY depends on its reputation.

## 8. The results side by side, and the one-year view

Every check above agreed. This block shows the values behind the checks, book
beside prospicio, from a real run of this guide (Windows x86_64, release build,
seed 0). **CI does not run this block**, because the NUTS draws, and thus the
last digits, differ on Linux:

```python no-run
side_by_side = pd.DataFrame(
    [(label, book, ours, 0 if abs(book) >= 100 else 2 if abs(book) >= 10 else 4)
     for label, (book, ours) in OURS.items()],
    columns=["check", "book", "prospicio", "places"])
(places_table(side_by_side, ["book", "prospicio"])
 .tab_header("The checks of sections 4 to 7: book against prospicio", "Seed 0; thousands of dollars"))
```

```text
                                 book  prospicio
CRC paid logelr mean          -0.3965    -0.3958
CRC paid logelr sd             0.0233     0.0237
CRC paid alpha[10] mean        0.3435     0.3363
CRC paid alpha[10] sd          0.3316     0.3326
CRC paid beta[1] mean         -1.1999    -1.2018
CRC paid beta[1] sd            0.1156     0.1115
CRC paid sigma[1] mean         0.2965     0.2938
CRC paid sigma[1] sd           0.1034     0.1012
CRC paid sigma[10] mean        0.0202     0.0203
CRC paid sigma[10] sd          0.0128     0.0129
CRC paid estimate              40,121     40,119
CRC paid sd of the total        2,487      2,538
CRC paid percentile             51.88      51.98
CRC incurred estimate          39,147     39,080
CRC incurred sd of the total    1,642      1,541
CRC incurred percentile         74.75      76.90
CRC paid elpd_loo               47.80      47.90
CRC incurred elpd_loo           70.97      71.42
CSR paid gamma mean            0.0446     0.0447
CSR paid gamma sd              0.0282     0.0286
CSR paid logelr mean          -0.3956    -0.3954
CSR paid logelr sd             0.0246     0.0231
CSR paid estimate              37,597     37,612
CSR paid sd of the total        2,401      2,487
CSR paid percentile             86.26      85.78
CSR paid elpd_loo               49.76      49.82
CAY incurred rho mean          0.1709     0.1686
CAY incurred rho sd            0.2071     0.2075
CAY incurred logelr mean      -0.3945    -0.3947
CAY incurred logelr sd         0.0150     0.0144
CAY incurred estimate          39,193     39,178
CAY incurred sd of the total    1,859      1,834
CAY incurred percentile         73.24      74.46
CAY incurred elpd_loo           68.65      69.05
```

The next table gives the mean and (sd) over refits at seeds 0 to 19 on the
same machine. The study is recorded, and CI does not run it.

| Model | Estimate | sd of the total | Percentile | elpd_loo | Parameter |
|---|---|---|---|---|---|
| ODP paid (book 39,193; 1,389; 73.91) | 39,197 (16) | 1,393 (12) | 73.78 (0.57) | | |
| CRC paid | 40,149 (37) | 2,520 (180) | 51.39 (0.81) | 48.23 (0.36) | `logelr` -0.3960 (0.0004) |
| CRC incurred | 39,124 (27) | 1,599 (44) | 75.74 (0.66) | 71.39 (0.31) | `logelr` -0.3947 (0.0003) |
| CSR paid | 37,557 (37) | 2,386 (55) | 86.58 (0.68) | 49.86 (0.37) | `gamma` 0.0448 (0.0003) |
| CAY incurred | 39,168 (24) | 1,857 (65) | 73.55 (0.68) | 68.79 (0.44) | `rho` 0.1697 (0.0026) |

Every book number is within the combined Monte Carlo error of Meyers' run and
prospicio's. Over 20 seeds, the means of the totals are within 28 (CRC paid),
23, 40 and 25 of the book (about one seed-to-seed sd), the percentiles within
1.0 point, elpd_loo within 0.5 and the sds of the totals within 3%. The sds are
the noisiest numbers: the sd of CRC paid changes by 180 from seed to seed,
because it is heavy-tailed through the one-cell youngest year.

**The one-year view.** Meyers' chapter 11 sets a cost of capital risk margin.
It compares ultimate and one-year horizons with his own simulation of future
calendar years. `Meyers.one_year` gives the claims development result (CDR)
over the next year for CRC, CSR and CAY. Each simulated next diagonal
reweights the posterior by Pareto smoothed importance sampling. The change in
the expected ultimate is the CDR.

On group 353, nearly every simulation needs more than a reweighting (the
finding [`meyers-one-year.md`](../../knowledge/findings/meyers-one-year.md)
explains why). The full method moment-matches these simulations and refits the
rest with NUTS, which takes about a minute. The quick variant only reweights.
It runs in about a second and is approximate: its CDR draws are about 0.2 sd of
the total from a refit.

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quick = Meyers("crc", seed=0).one_year(tri, "paid", "premium", refit="reweight_only", n_sims=1000)
cdr = np.array(quick.cdr.draw_matrix()).sum(axis=1)
lifetime = np.array(quick.fit.reserves.draw_matrix()).sum(axis=1)
iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))
ratio = iqr(cdr) / iqr(lifetime)
print(f"warnings {len(caught)}, one-year simulations {len(cdr):,}")
print(f"share flagged (Pareto k above 0.7) above 95%: {quick.flagged_share > 0.95}")
print(f"CDR IQR / lifetime reserve IQR between 0.6 and 0.95: {0.6 < ratio < 0.95}")
```

```text
warnings 0, one-year simulations 1,000
share flagged (Pareto k above 0.7) above 95%: True
CDR IQR / lifetime reserve IQR between 0.6 and 0.95: True
```

The CDR is a change in the expected ultimate, so it centres near 0. The
reserve centres on the expected unpaid amount, and the CDR is the narrower. The plot leaves out the extreme
0.5% of each side.

```python
fig, ax = plt.subplots()
both = np.concatenate([cdr, lifetime])
bins = np.linspace(*np.percentile(both, [0.5, 99.5]), 50)
ax.hist(lifetime, bins=bins, density=True, color="0.75", label="lifetime reserve")
ax.hist(cdr, bins=bins, density=True, histtype="step", linewidth=2, label="one-year CDR (quick view)")
ax.set_xlabel("total over the origins")
ax.set_ylabel("density")
ax.set_title("Lifetime reserve and one-year claims development result, CRC paid")
ax.legend()
```

On Windows at seed 0, 99% are flagged, and the CDR sd is 1,970 against a
lifetime sd of 2,538 (ratio 0.78). The block uses 1,000 one-year simulations;
the default is 2,000. The finding measures the sd ratio at 0.77, with a
seed-to-seed sd of 0.08.

The check compares interquartile ranges, not sds. With nearly every simulation
flagged, a few reweighted simulations can carry most of the weight and inflate
the sd of the CDR. Over seeds 0 to 9, the sd ratio ran from 0.71 to 1.13; the
IQR ratio stayed between 0.73 and 0.84.

The full view, with moment matching and refits, ran on a release build
(Windows, 16 threads, 62 s). **CI does not run this block**:

```python no-run
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    full = Meyers("crc", seed=0).one_year(tri, "paid", "premium")
cdr = np.array(full.cdr.draw_matrix()).sum(axis=1)
lifetime = np.array(full.fit.reserves.draw_matrix()).sum(axis=1)
print(f"flagged {full.flagged_share:.0%}, moment-matched {len(full.moment_matched):,}, refitted {len(full.refitted):,}")
print(f"CDR sd {cdr.std():,.0f}, lifetime reserve sd {lifetime.std():,.0f}, ratio {cdr.std() / lifetime.std():.2f}")
print("warnings", len(caught))
```

```text
flagged 99%, moment-matched 1,778, refitted 202
CDR sd 1,947, lifetime reserve sd 2,538, ratio 0.77
warnings 1
```

About three quarters of the lifetime uncertainty emerges in the first year.
This is near the share of the ODP bootstrap on the same triangle (about 0.80).
The one warning says that some refits failed their own MCMC diagnostics. Read
`full.refit_warnings` before you use the tail of the CDR.
