# Stochastic loss reserving with Bayesian MCMC: CAS Monograph 8 reproduced

Glenn Meyers' *Stochastic Loss Reserving Using Bayesian MCMC Models* (CAS
Monograph 8, 2nd edition, 2019) asks a plain question of a reserving model: if
it had been used on many real triangles, would the outcomes that later came in
look like draws from the distributions it predicted? He answers it for Mack's
model and the ODP bootstrap (they fail), and then builds a sequence of
Bayesian models, fitted by Markov chain Monte Carlo (MCMC), each meant to fix
a failure of the one before. This guide explains that sequence and the
validation method in new words, and recomputes the monograph's printed results
on its illustrative insurer with prospicio, the book's value printed next to
prospicio's.

The text of the monograph is Meyers'; read it for the argument and the
200-triangle results, which this guide cannot rerun (section 9). Table numbers
below are the monograph's. Money is in thousands of US dollars, as in his
tables. For the choice between reserving methods in general, read
[`reserving-methods.md`](reserving-methods.md); this guide does not repeat it.

**How the numbers here are checked.** Every Python block below that is
followed by a text block runs in CI (`python/tests/test_guides.py`) and must
print exactly that text. The Bayesian fits use prospicio's default chains (4
chains of 1,000 warm-up and 2,000 kept draws) and take under a second each in
a release build, so they all run in CI, in about 10 seconds for the whole
guide. Two things about them need saying plainly:

* A Bayesian result is a Monte Carlo estimate: another seed gives a slightly
  different number. Each comparison states how much a number moves from seed
  to seed, measured by refitting at seeds 0 to 19 (a recorded study, not
  re-run by CI).
* The exact digits of a NUTS run also depend on the platform: the last bit
  of `exp` and `ln` differs between Windows and Linux, the sampler's paths
  then diverge, and the draws differ (as noted in prospicio's own
  seed-pinned test). So the checked blocks print the book's value, a
  tolerance a few Monte Carlo errors wide, and whether prospicio's value
  falls inside it. prospicio's own values, from a real run, are in the table
  of section 8, in a block marked `python no-run`; **that block and the full
  one-year view, also `no-run`, are not re-run by CI.**

## 1. The illustrative insurer

Meyers' running example is group 353 in the CAS Loss Reserve Database,
Commercial Auto, accident years 1988 to 1997 at 12 to 120 months, with net
earned premium (his Tables 2.1 to 2.3). The database holds the full 10 by 10
square, so the lower triangle, unknown when the reserves were set, is known
now. That is what makes his test possible: fit on the upper triangle, then
see where the actual 120-month total falls in the predicted distribution.

```python
import csv
import math
import warnings
from pathlib import Path

import numpy as np

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

The premium total of 52,429 and the two outcomes, 40,000 paid and 40,061
incurred, are the figures in his tables. Incurred here is case incurred
(paid plus case reserves, without bulk reserves), as in the database.

## 2. The starting point: Mack's model and the outcome's percentile

Meyers starts from Mack's distribution-free chain ladder. Mack gives a mean
and a standard error of each origin's ultimate and of the total, but no
distribution. Following Mack's own suggestion, Meyers fits a log-normal with
that mean and standard deviation to the total and reads the percentile at
which the outcome falls (his Table 3.1, incurred, and Table 4.1, paid, where
he adds the ODP bootstrap). One percentile says little: 86 is not
alarming. The point is what follows in section 3.

```python
def lognormal_percentile(mean, sd, x):
    """Percentile of x under the log-normal with this mean and sd."""
    s2 = math.log(1 + (sd / mean) ** 2)
    z = (math.log(x / mean) + s2 / 2) / math.sqrt(s2)
    return 50 * (1 + math.erf(z / math.sqrt(2)))


# Meyers' printed totals: (estimate, standard error, percentile).
BOOK_MACK = {"incurred": (38_914, 1_057, 86.03), "paid": (39_177, 1_442, 72.02)}
print(f"{'Mack':9s} {'estimate':>17s} {'std error':>13s} {'percentile':>15s}")
print(f"{'':9s} {'book':>8s} {'ours':>8s} {'book':>6s} {'ours':>6s} {'book':>7s} {'ours':>7s}")
for col, (est, se, pct) in BOOK_MACK.items():
    mack = Mack().fit(tri, col)
    ours = lognormal_percentile(mack.total_ultimate, mack.total_standard_error, outcome[col])
    print(f"{col:9s} {est:8,d} {mack.total_ultimate:8,.0f} {se:6,d} {mack.total_standard_error:6,.0f}"
          f" {pct:7.2f} {ours:7.2f}")
```

```text
Mack               estimate     std error      percentile
              book     ours   book   ours    book    ours
incurred    38,914   38,914  1,057  1,057   86.03   86.06
paid        39,177   39,177  1,442  1,443   72.02   72.00
```

Mack's model is deterministic, so there is no Monte Carlo error here. The
estimates agree to the unit; the paid standard error is 1,442.5 before
rounding, so prospicio prints 1,443 against the book's 1,442, a rounding of the
last digit. The percentiles differ by 0.03 and 0.02 points: both sides use
the same log-normal by moments, and a difference that small comes from the
rounding of the inputs or of the normal cdf, not from the model.

The ODP bootstrap (England and Verrall) resamples residuals of an
overdispersed Poisson GLM on the incremental paid values. Meyers ran R
ChainLadder's version with 10,000 simulations; here is prospicio's, with its
percentile counted directly from the simulations, as Meyers does for every
simulated model:

```python
odp = OdpBootstrap(n_sims=10_000, seed=0).fit(tri, "paid")
odp_total = np.array(odp.reserves.draw_matrix()).sum(axis=1) + latest["paid"]
odp_pct = 100 * np.mean(odp_total <= outcome["paid"])
print(f"ODP paid  estimate book 39,193 ours {odp_total.mean():,.0f} | std error book 1,389 ours {odp_total.std():,.0f}"
      f" | percentile book 73.91 ours {odp_pct:.2f}")
```

```text
ODP paid  estimate book 39,193 ours 39,231 | std error book 1,389 ours 1,410 | percentile book 73.91 ours 72.73
```

This one is simulated, but without a NUTS sampler, so a platform's last-bit
rounding stays in the last bit and the printed digits are the same
everywhere. Over seeds 0 to 19 the estimate moves with a standard
deviation of 16 around a mean of 39,197, the standard error by 12 around
1,393 and the percentile by 0.57 points around 73.78. Seed 0 falls low on
all three (about two of those sds), and the 20-seed means are within 4, 4
and 0.13 of the book, so prospicio's bootstrap and Meyers' printed run agree
within their error. The two programs are not identical (R ChainLadder draws the
process from a gamma distribution by default, prospicio from the ODP), which
the comparison cannot distinguish here.

## 3. The validation method: percentiles, p-p plots and the KS test

The monograph's main idea is a test of reputation, not of one triangle. If a
model's predictive distributions are right, the percentile of each outcome is
uniform on 0 to 100 across many triangles. Meyers fits each model to 200
triangles (50 each of commercial auto, personal auto, workers' compensation
and other liability) and asks whether the 200 percentiles look uniform.

* The **p-p plot** puts the sorted percentiles on the vertical axis against
  their expected values under uniformity, `100 i / (n + 1)`, on the
  horizontal. A good model lies on the 45 degree line.
* The **shape** of a failure is informative. Too many extreme percentiles (an
  S tilted along the diagonal) means the predicted distribution is too light
  in the tails; too many middle percentiles (a backward S) means too heavy;
  too many low percentiles means the model predicts too high.
* The **Kolmogorov-Smirnov** test makes the plot a test: with the sorted
  percentiles `p_i` and `f_i = 100 i / n`, the statistic is
  `D = max |p_i - f_i|`, and uniformity is rejected at the 5% level when `D`
  exceeds `136 / sqrt(n)`. On the plot this is a band around the diagonal. A
  model is "validated" in the monograph's sense when it stays in the band.
  He tried the Anderson-Darling test too, which weighs the tails more, and
  almost every model failed it, so he kept KS as the working bar.

The critical values in his figures follow from the formula, and a short
simulation shows what the test sees. One set of percentiles comes from a
model that is right, the other from a model whose predictive distribution is
too narrow (outcomes spread 1.5 times wider than predicted):

```python
def ks_d(percentiles):
    p = np.sort(np.asarray(percentiles))
    f = 100 * np.arange(1, len(p) + 1) / len(p)
    return np.max(np.abs(p - f))


for n in (50, 200):
    print(f"n = {n:3d}: critical value 136 / sqrt(n) = {136 / math.sqrt(n):.1f}")

rng = np.random.default_rng(353)
z = rng.standard_normal(200)
normal_cdf = np.vectorize(lambda x: 50 * (1 + math.erf(x / math.sqrt(2))))
print(f"right model:          D = {ks_d(normal_cdf(z)):4.1f}")
print(f"too light in the tails: D = {ks_d(normal_cdf(1.5 * z)):4.1f}")
```

```text
n =  50: critical value 136 / sqrt(n) = 19.2
n = 200: critical value 136 / sqrt(n) = 9.6
right model:          D =  4.5
too light in the tails: D = 14.1
```

The 19.2 for 50 triangles and 9.6 for 200 are the critical values printed
on every p-p plot in the monograph. With his panel he finds two failures that
set the agenda for the rest of the book:

* Mack on **incurred** triangles is too light in the tails (D = 15.4 over the
  200 triangles, against 9.6).
* Mack and the ODP bootstrap on **paid** triangles predict too high (D of 23.1
  and 24.1, with the percentiles piled at the low end).

The 200-triangle panel is not in prospicio's validation data, so this guide
reproduces the method on one triangle and the test's mechanics, not the
panel's D statistics (section 9).

## 4. The cross classified model (CRC)

The first Bayesian model is a log-normal chain ladder on cumulative values
with premium as the base:

* `log C[w, d]` is normal with mean `log P[w] + logelr + alpha[w] + beta[d]`
  and standard deviation `sigma[d]`.
* `logelr` is the log of the expected loss ratio, `alpha[w]` an accident year
  level (`alpha[1] = 0`) and `beta[d]` a development effect (`beta[10] = 0`,
  so the 120-month loss ratio of the first year is about `exp(logelr)`).
* `sigma[d]` falls with age: its square is a sum of uniform steps over the
  later ages, because older cells hold more settled claims.
* The predictive distribution of each origin's 120-month value is a
  log-normal draw per posterior draw, so parameter and process uncertainty
  are both in it.

What it changes from Mack: a full distribution instead of two moments, a
prior on a quantity an actuary knows (the loss ratio), and variance that
falls with age by construction. It keeps the chain ladder's structure of one
level per accident year and one effect per age. The priors are wide on
purpose: Meyers treats them as part of the model, not as beliefs.

prospicio's `Meyers` uses the priors of his published Stan code, which
differ in detail from the text: a prior sd of 3.162 (`sqrt(10)`) where the
text says 10, and a floor of about `1e-5` on each variance step. The
knowledge note
[`meyers-monograph-8.md`](../../knowledge/references/meyers-monograph-8.md)
lists these quirks; the finding
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


def check(label, book, ours, tol, fmt=",.0f"):
    """Print the book's value and whether prospicio's is within tol of it."""
    OURS[label] = (book, ours)
    print(f"{label:30s} book {book:>9{fmt}}  tolerance {tol:>7{fmt}}  agrees: {abs(ours - book) <= tol}")


def check_parameter(result, label, name, book_mean, book_sd):
    """Means within 0.15 posterior sd, sds within 15%: the tolerances of prospicio's parity tests."""
    s = result.summary().set_index("parameter")
    check(f"{label} {name} mean", book_mean, s.loc[name, "mean"], 0.15 * book_sd, ".4f")
    check(f"{label} {name} sd", book_sd, s.loc[name, "sd"], 0.15 * book_sd, ".4f")


def check_outcome(result, label, column, book_est, book_sd, book_pct):
    """Totals within 1% and percentiles within 5 points (several Monte Carlo errors);
    the total's sd within 50%, as in the parity tests, since it is heavy-tailed."""
    est, sd, pct = outcome_summary(result, column)
    check(f"{label} estimate", book_est, est, 0.01 * book_est)
    check(f"{label} sd of the total", book_sd, sd, 0.5 * book_sd)
    check(f"{label} percentile", book_pct, pct, 5.0, ".2f")


crc_paid, n_warn = fit("crc", "paid")
d = crc_paid.diagnostics
print(f"warnings {n_warn}, divergences {d['divergences']}, max R-hat below 1.01: {d['max_rhat'] < 1.01},"
      f" posterior draws {len(crc_paid.draws):,}, simulations {crc_paid.ultimates.n_sims:,}")
for name, (mean, sd) in {"logelr": (-0.3965, 0.0233), "alpha[10]": (0.3435, 0.3316),
                         "beta[1]": (-1.1999, 0.1156), "sigma[1]": (0.2965, 0.1034),
                         "sigma[10]": (0.0202, 0.0128)}.items():
    check_parameter(crc_paid, "CRC paid", name, mean, sd)
```

```text
warnings 0, divergences 0, max R-hat below 1.01: True, posterior draws 8,000, simulations 10,000
CRC paid logelr mean           book   -0.3965  tolerance  0.0035  agrees: True
CRC paid logelr sd             book    0.0233  tolerance  0.0035  agrees: True
CRC paid alpha[10] mean        book    0.3435  tolerance  0.0497  agrees: True
CRC paid alpha[10] sd          book    0.3316  tolerance  0.0497  agrees: True
CRC paid beta[1] mean          book   -1.1999  tolerance  0.0173  agrees: True
CRC paid beta[1] sd            book    0.1156  tolerance  0.0173  agrees: True
CRC paid sigma[1] mean         book    0.2965  tolerance  0.0155  agrees: True
CRC paid sigma[1] sd           book    0.1034  tolerance  0.0155  agrees: True
CRC paid sigma[10] mean        book    0.0202  tolerance  0.0019  agrees: True
CRC paid sigma[10] sd          book    0.0128  tolerance  0.0019  agrees: True
```

The diagnostics come first: prospicio never fails a fit on bad diagnostics,
it raises one `UserWarning` and keeps them in `fit.diagnostics`, and none
fired. The parameters are Table 5.1's (its columns are scrambled in the PDF's
text layer; the values here were read from the rendered page). The
tolerances are those of prospicio's parity tests: 0.15 posterior sd on a
mean and 15% on an sd, which hold at 200 seeds against Meyers' Stan code,
whose rerun agrees with his printed tables within its own error. prospicio's seed-0 values (section 8) are
`logelr` -0.3958 (sd 0.0237) against -0.3965 (0.0233); the `logelr` mean
moves by 0.0004 from seed to seed, so the gap is the two runs' error. The
youngest year's level `alpha[10]` rests on one cell, which is why its sd is
0.33, and the first age's `sigma[1]` is the largest. Now the predictive
distribution and the outcome's percentile (Tables 5.2 and 5.3):

```python
crc_inc, n_warn_inc = fit("crc", "incurred")
check_outcome(crc_paid, "CRC paid", "paid", 40_121, 2_487, 51.88)
check_outcome(crc_inc, "CRC incurred", "incurred", 39_147, 1_642, 74.75)
print("warnings on the incurred fit:", n_warn_inc)
```

```text
CRC paid estimate              book    40,121  tolerance     401  agrees: True
CRC paid sd of the total       book     2,487  tolerance   1,244  agrees: True
CRC paid percentile            book     51.88  tolerance    5.00  agrees: True
CRC incurred estimate          book    39,147  tolerance     391  agrees: True
CRC incurred sd of the total   book     1,642  tolerance     821  agrees: True
CRC incurred percentile        book     74.75  tolerance    5.00  agrees: True
warnings on the incurred fit: 0
```

"Estimate" is the mean total at 120 months, including the oldest year's
observed 3,912 (paid) or 3,917 (incurred), as in Meyers' tables. Over seeds 0
to 19, prospicio's paid estimate has a mean of 40,149 and a seed-to-seed sd
of 37, its sd 2,520 (sd 180: the total's sd rides on the one-cell youngest
year through a log-normal and is heavy-tailed) and the percentile 51.39 (sd
0.81). The incurred estimate has mean 39,124 (sd 27), sd 1,599 (sd 44) and
percentile 75.74 (sd 0.66). Seed 0 on Windows sits high on the incurred
percentile (76.90, the largest of the 20 seeds) and low on the incurred sd
(1,541); the 20-seed means are one point and 43 from the book's 74.75 and
1,642. Meyers' own run is also a Monte Carlo estimate of about the
same size, and the CmdStan rerun of his code gives totals of 40,136 (2,497)
and 39,132 (1,616). Within these errors the book and prospicio agree.

On the 200 triangles CRC does no better than Mack and the ODP: on paid data
it predicts too high, on incurred data its tails are too light. The next two
models each address one of these.

## 5. Checking a model on the data in hand

The p-p plot judges a model's reputation; an actuary setting reserves today
needs checks on the upper triangle alone. Meyers uses two.

**Standardized residuals.** For a sample of posterior draws, compute
`(log C[w, d] - mu[w, d]) / sigma[d]` for every observed cell and box-plot
them by accident year and by development year. All of his models look
similar by development year; the accident year plots are what separate
them. The residuals come straight from prospicio's draws:

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

No accident year is far off: the medians run from about -0.2 to 0.4 (on
Windows at seed 0: -0.16, 0.25, -0.17, 0.10, -0.11, 0.36, -0.03, 0.16, 0.01
and 0.00 for 1988 to 1997), inside the interquartile range of a standard
normal, which is what Meyers' box plots show for CRC paid. The largest,
1993's, is the one CSR moves most (section 6).

**Leave-one-out cross-validation.** `elpd_loo` estimates how well the model
predicts each observed cell when that cell is left out, by Pareto smoothed
importance sampling of the posterior instead of refitting (Vehtari, Gelman and
Gabry, the R `loo` package Meyers uses). Higher is better, and it compares
models fitted to the same column only (`fit.loo()` returns it). His Table 6.4
compares CRC with the stochastic Cape Cod (SCC), CRC without accident year
levels: SCC is far worse on every measure, and he drops it. prospicio does
not implement SCC.

```python
# elpd_loo within 2, the parity tests' tolerance.
check("CRC paid elpd_loo", 47.80, crc_paid.loo().elpd, 2.0, ".2f")
check("CRC incurred elpd_loo", 70.97, crc_inc.loo().elpd, 2.0, ".2f")
```

```text
CRC paid elpd_loo              book     47.80  tolerance    2.00  agrees: True
CRC incurred elpd_loo          book     70.97  tolerance    2.00  agrees: True
```

prospicio gives 47.90 and 71.42 at seed 0 on Windows. elpd_loo moves by
about 0.3 to 0.4 from seed to seed (paid mean 48.23, incurred 71.39 over
seeds 0 to 19), and the CmdStan rerun of Meyers' code gives 48.04 and 71.22,
so a gap of up to about half a point is the runs' error.

## 6. The changing settlement rate model (CSR)

CSR keeps CRC and lets the development effects shrink toward zero as the
accident years get later: the mean becomes
`log P[w] + logelr + alpha[w] + beta[d] (1 - gamma)^(w - 1)`, with a
`normal(0, 0.05)` prior on `gamma`. A positive `gamma` means claims settle
faster in the later years, so the paid values of a young year are further
along than the old pattern would say, and the reserve is smaller. With
`gamma = 0` it is CRC. It targets CRC's bias on paid data, and over the 200
triangles its paid p-p plots are the first to stay in the KS band.

```python
csr_paid, n_warn_csr = fit("csr", "paid")
print("warnings", n_warn_csr)
check_parameter(csr_paid, "CSR paid", "gamma", 0.0446, 0.0282)
check_parameter(csr_paid, "CSR paid", "logelr", -0.3956, 0.0246)
check_outcome(csr_paid, "CSR paid", "paid", 37_597, 2_401, 86.26)
check("CSR paid elpd_loo", 49.76, csr_paid.loo().elpd, 2.0, ".2f")
csr_medians = year_medians(standardized_residuals(csr_paid, gamma="gamma"))
print(f"CSR paid: largest |median residual| of an accident year {np.abs(csr_medians).max():.1f}"
      f" (CRC {np.abs(crc_medians).max():.1f})")
```

```text
warnings 0
CSR paid gamma mean            book    0.0446  tolerance  0.0042  agrees: True
CSR paid gamma sd              book    0.0282  tolerance  0.0042  agrees: True
CSR paid logelr mean           book   -0.3956  tolerance  0.0037  agrees: True
CSR paid logelr sd             book    0.0246  tolerance  0.0037  agrees: True
CSR paid estimate              book    37,597  tolerance     376  agrees: True
CSR paid sd of the total       book     2,401  tolerance   1,200  agrees: True
CSR paid percentile            book     86.26  tolerance    5.00  agrees: True
CSR paid elpd_loo              book     49.76  tolerance    2.00  agrees: True
CSR paid: largest |median residual| of an accident year 0.3 (CRC 0.4)
```

These are Tables 7.1 to 7.3. On Windows at seed 0 prospicio gives `gamma`
0.0447 (sd 0.0286), an estimate of 37,612 (sd 2,487), the 85.78th
percentile and elpd_loo 49.82. Over seeds 0 to 19 the estimate's mean is
37,557 (sd 37), the sd's 2,386 (sd 55), the percentile's 86.58 (sd 0.68),
`gamma`'s 0.0448 (sd 0.0003) and elpd_loo's 49.86 (sd 0.37), all within
error of the book. A speed-up of about 4.5% a year lowers the estimate by
about 2,500 against CRC and moves the outcome from the 52nd to the 86th
percentile: for this insurer CSR predicts lower than the outcome, while over
the panel CRC's paid forecasts run too high and CSR's sit in the KS band.
elpd_loo prefers CSR here by about 2, and the accident year residuals are a
little flatter, as in his Figure 7.3 against 5.3. Over the 200 triangles
elpd_loo prefers CSR for 105.

## 7. The correlated accident year model (CAY)

CAY goes back to CRC and attacks the other failure, light tails on incurred
data. Each accident year's mean takes `rho` times the previous year's
residual at the same age, `log C[w - 1, d] - mu[w - 1, d]`, with a prior
that keeps `rho` in `(-1, 1)` and centred on 0. John Major's result in the
monograph (eq. 8.3) is that this makes the log values a moving average down
each column: neighbouring years correlate by `rho / (1 + rho^2)`, years two
apart not at all. Correlated years add up to a wider total, so the predicted
distribution gets heavier tails. The prediction runs year by year, each
120-month draw feeding the next year's mean.

```python
cay_inc, n_warn_cay = fit("cay", "incurred")
print("warnings", n_warn_cay)
check_parameter(cay_inc, "CAY incurred", "rho", 0.1709, 0.2071)
check_parameter(cay_inc, "CAY incurred", "logelr", -0.3945, 0.0150)
check_outcome(cay_inc, "CAY incurred", "incurred", 39_193, 1_859, 73.24)
check("CAY incurred elpd_loo", 68.65, cay_inc.loo().elpd, 2.0, ".2f")
```

```text
warnings 0
CAY incurred rho mean          book    0.1709  tolerance  0.0311  agrees: True
CAY incurred rho sd            book    0.2071  tolerance  0.0311  agrees: True
CAY incurred logelr mean       book   -0.3945  tolerance  0.0022  agrees: True
CAY incurred logelr sd         book    0.0150  tolerance  0.0022  agrees: True
CAY incurred estimate          book    39,193  tolerance     392  agrees: True
CAY incurred sd of the total   book     1,859  tolerance     930  agrees: True
CAY incurred percentile        book     73.24  tolerance    5.00  agrees: True
CAY incurred elpd_loo          book     68.65  tolerance    2.00  agrees: True
```

These are Tables 8.1 to 8.3. On Windows at seed 0 prospicio gives `rho`
0.1686 (sd 0.2075), an estimate of 39,178 (sd 1,834), the 74.46th
percentile and elpd_loo 69.05. Over seeds 0 to 19: estimate 39,168 (sd 24),
sd 1,857 (sd 65), percentile 73.55 (sd 0.68), `rho` 0.1697 (sd 0.0026) and
elpd_loo 68.79 (sd 0.44), all within error of the book. The posterior of
`rho` is wide (sd 0.21), its sd of the total is about 15% above CRC's, and
elpd_loo prefers CRC on this triangle, as Meyers reports. Over the 200
triangles elpd_loo prefers CAY for only 26, but the score on the lower
triangle (`elpd_test`) prefers it for 121 and its incurred p-p plots are
better, so in his words the choice of CAY rests on its reputation.

## 8. The results side by side, and the one-year view

Every check above printed `agrees: True`. The values behind them, book
beside prospicio, come from this block, recorded from a real run of this
guide (Windows x86_64, release build, seed 0). **It is not re-run by CI**,
because on Linux the NUTS draws, and so the last digits, differ:

```python no-run
print(f"{'':30s} {'book':>9s} {'prospicio':>10s}")
for label, (book, ours) in OURS.items():
    fmt = ",.0f" if abs(book) >= 100 else (".2f" if abs(book) >= 10 else ".4f")
    print(f"{label:30s} {book:>9{fmt}} {ours:>10{fmt}}")
```

```text
                                    book  prospicio
CRC paid logelr mean             -0.3965    -0.3958
CRC paid logelr sd                0.0233     0.0237
CRC paid alpha[10] mean           0.3435     0.3363
CRC paid alpha[10] sd             0.3316     0.3326
CRC paid beta[1] mean            -1.1999    -1.2018
CRC paid beta[1] sd               0.1156     0.1115
CRC paid sigma[1] mean            0.2965     0.2938
CRC paid sigma[1] sd              0.1034     0.1012
CRC paid sigma[10] mean           0.0202     0.0203
CRC paid sigma[10] sd             0.0128     0.0129
CRC paid estimate                 40,121     40,119
CRC paid sd of the total           2,487      2,538
CRC paid percentile                51.88      51.98
CRC incurred estimate             39,147     39,080
CRC incurred sd of the total       1,642      1,541
CRC incurred percentile            74.75      76.90
CRC paid elpd_loo                  47.80      47.90
CRC incurred elpd_loo              70.97      71.42
CSR paid gamma mean               0.0446     0.0447
CSR paid gamma sd                 0.0282     0.0286
CSR paid logelr mean             -0.3956    -0.3954
CSR paid logelr sd                0.0246     0.0231
CSR paid estimate                 37,597     37,612
CSR paid sd of the total           2,401      2,487
CSR paid percentile                86.26      85.78
CSR paid elpd_loo                  49.76      49.82
CAY incurred rho mean             0.1709     0.1686
CAY incurred rho sd               0.2071     0.2075
CAY incurred logelr mean         -0.3945    -0.3947
CAY incurred logelr sd            0.0150     0.0144
CAY incurred estimate             39,193     39,178
CAY incurred sd of the total       1,859      1,834
CAY incurred percentile            73.24      74.46
CAY incurred elpd_loo              68.65      69.05
```

How far a number moves from seed to seed, from refits at seeds 0 to 19 on
the same machine (mean and sd over the 20 seeds; also recorded, not re-run
by CI):

| Model | Estimate | sd of the total | Percentile | elpd_loo | Parameter |
|---|---|---|---|---|---|
| ODP paid (book 39,193; 1,389; 73.91) | 39,197 (16) | 1,393 (12) | 73.78 (0.57) | | |
| CRC paid | 40,149 (37) | 2,520 (180) | 51.39 (0.81) | 48.23 (0.36) | `logelr` -0.3960 (0.0004) |
| CRC incurred | 39,124 (27) | 1,599 (44) | 75.74 (0.66) | 71.39 (0.31) | `logelr` -0.3947 (0.0003) |
| CSR paid | 37,557 (37) | 2,386 (55) | 86.58 (0.68) | 49.86 (0.37) | `gamma` 0.0448 (0.0003) |
| CAY incurred | 39,168 (24) | 1,857 (65) | 73.55 (0.68) | 68.79 (0.44) | `rho` 0.1697 (0.0026) |

Every printed number of the book is within the combined Monte Carlo error of
Meyers' run and prospicio's. Over 20 seeds the means of the totals are
within 28 (CRC paid), 23, 40 and 25 of the book, about one seed-to-seed sd;
the percentiles within 1.0 point; elpd_loo within 0.5. The sds of the totals
are the noisiest numbers (CRC paid's moves by 180 from seed to seed and has a
heavy tail, through the youngest year's one-cell level and a log-normal), and
they agree within 3%.

**The one-year view.** Meyers' chapter 11 sets a cost of capital risk margin
and compares ultimate and one-year horizons with his own simulation of future
calendar years; prospicio does not reproduce those tables. What prospicio
offers is the claims development result over the coming year for CRC, CSR
and CAY (`Meyers.one_year`): each simulated next diagonal reweights the
posterior by Pareto smoothed importance sampling, and the change in the
expected ultimate is the CDR. On group 353 nearly every simulation needs
more than reweighting (the finding
[`meyers-one-year.md`](../../knowledge/findings/meyers-one-year.md)
explains why), so the full method moment-matches them and refits the rest
with NUTS, which takes about a minute. The quick variant reweights only. It
runs in about a second and is approximate (its CDR draws sit about 0.2 sd of
the total from a refit):

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quick = Meyers("crc", seed=0).one_year(tri, "paid", "premium", refit="reweight_only", n_sims=1000)
cdr = np.array(quick.cdr.draw_matrix()).sum(axis=1)
lifetime = np.array(quick.fit.reserves.draw_matrix()).sum(axis=1)
ratio = cdr.std() / lifetime.std()
print(f"warnings {len(caught)}, one-year simulations {len(cdr):,}")
print(f"share flagged (Pareto k above 0.7) above 95%: {quick.flagged_share > 0.95}")
print(f"CDR sd / lifetime reserve sd between 0.6 and 0.95: {0.6 < ratio < 0.95}")
```

```text
warnings 0, one-year simulations 1,000
share flagged (Pareto k above 0.7) above 95%: True
CDR sd / lifetime reserve sd between 0.6 and 0.95: True
```

On Windows at seed 0 this gives 99% flagged and a CDR sd of 1,970 against a
lifetime sd of 2,538, a ratio of 0.78 (1,000 one-year simulations keep the
block quick; the default is 2,000). The finding measures the ratio at 0.77
with a seed-to-seed sd of 0.08.

The full view, with its moment matching and refits, recorded from a real run
on a release build (Windows, 16 threads, 62 s). **This block is not re-run by
CI**:

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

About three quarters of the lifetime uncertainty emerges in the first year,
close to the ODP bootstrap's share on the same triangle (about 0.80). The
one warning says some of the refits failed their own MCMC diagnostics; read
`full.refit_warnings` before relying on the tail of the CDR.

## 9. What this guide does not reproduce

* **The 200-triangle validation.** The p-p plots, KS statistics and
  model-preference counts need Meyers' panel with its outcomes. prospicio's
  validation data hold group 353 and the six CLRD lines summed over
  companies (upper triangles only), not the panel; the Reserving lane lists
  the calibration study as its next step. prospicio's diagonal backtest,
  which holds out the latest diagonals and scores forecasts on them, is in
  Rust only for now (finding
  [`meyers-segments-backtest.md`](../../knowledge/findings/meyers-segments-backtest.md)).
* **SCC and IPI.** The stochastic Cape Cod (chapter 6) is not implemented;
  Meyers drops it himself. The integrated paid and incurred model (chapter 9:
  shared accident year levels, CSR on paid and CAY on incurred) is planned in
  the Reserving lane, not built.
* **Dependence between lines (chapter 10).** Meyers estimates a correlation
  parameter between two lines' residuals, given each line's posterior.
  prospicio's `Meyers(..., dependence="rank_correlation", spearman=...)`
  with `fit_segments` instead takes the rank correlation between the lines'
  total reserves as an input (Iman-Conover on the simulations), so it can use
  his estimate but does not produce it.
* **Rounding in his prediction.** His R code rounds each simulated
  120-month value up to an integer (`ceiling`), about +4.5 on the total;
  prospicio does not, which is far inside the Monte Carlo error.
