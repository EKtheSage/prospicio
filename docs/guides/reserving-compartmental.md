# Hierarchical compartmental reserving: Gesmann and Morris' models 1 and 2

Markus Gesmann and Jake Morris, *Hierarchical Compartmental Reserving
Models* (CAS Research Paper, 2020), model a reserve the way a
pharmacologist models a drug in the body: as quantities flowing between
compartments at rates. Claims arise from premium, sit in outstanding while
they are known but unpaid, and move to paid as they settle. One small set of
parameters, each with a business meaning, then describes the paid and the
outstanding triangle at once. This guide explains the idea in new words, fits
the paper's two models on its illustrative company with prospicio
(`Compartmental`), and sets the paper's printed posterior summaries next to
prospicio's.

The paper's text is the authors'; read it for the derivations, the market
cycle model (their model 3) and the plots, which this guide does not
reproduce (section 10). The paper fits its models with the R package brms,
which writes the Stan program behind the scenes; prospicio samples the same
posterior with its own NUTS sampler. Money is in thousands of US dollars, as
in the data. For the choice between reserving methods in general, read
[`reserving-methods.md`](reserving-methods.md); for Meyers' Bayesian models on
a single triangle, [`reserving-meyers.md`](reserving-meyers.md).

**How the numbers here are checked.** Every Python block below that is
followed by a text block runs in CI (`python/tests/test_guides.py`) and must
print exactly that text. Model 1 runs in CI with prospicio's default sampler
(4 chains of 1,000 warm-up and 2,000 kept draws), which takes about 5 to 9
seconds in a release build. Model 2 and the one-year views do not: a model 2
fit takes about a minute, and a one-year view several minutes. Two things need
saying plainly:

* A Bayesian result is a Monte Carlo estimate: another seed gives a slightly
  different number, and so did the paper's own run. Each comparison states
  how far a number moves from seed to seed, from the 200-seed studies recorded
  in `knowledge/findings/compartmental-gm1-parity.md` and
  `compartmental-gm2-parity.md` (not re-run by CI).
* The exact digits of a NUTS run depend on the platform: the last bit of
  `exp` and `ln` differs between Windows and Linux, the sampler's paths then
  diverge, and the draws differ. So the checked blocks print the paper's
  value (or, where the paper prints none, that of its code rerun in Stan), a
  tolerance a few Monte Carlo errors wide, and whether prospicio's value
  falls inside it. prospicio's own values come from blocks marked
  `python no-run`, recorded from a real run of this guide (Windows x86_64,
  release build, seed 0). **The `no-run` blocks, which include every model 2
  fit and both one-year views, are not re-run by CI.**

## 1. The compartmental idea

Picture three tanks. The first, exposure (EX), starts each accident year full
with that year's expected losses. Claims leave it as they are reported, at a
rate `k_er` a year, and flow into the second tank, outstanding (OS). From
outstanding they leave as they are settled, at a rate `k_p` a year, into the
third tank, paid (PD). Linear differential equations describe the flow,
and with constant rates they have a closed form. With `t` the development
time in years (12 months is `t = 1`) and

    g(t) = (exp(-k_p t) - exp(-k_er t)) / (k_er - k_p)

the expected outstanding and cumulative paid of origin `i` with premium `P_i`
are

    OS_i(t) = P_i RLR_i k_er g(t)
    PD_i(t) = P_i RLR_i RRF_i [(1 - exp(-k_p t)) - k_p g(t)]

Outstanding rises from zero, peaks and runs off; paid rises toward its
ultimate. The four parameters read as statements about the business:

* `RLR_i`, the reported loss ratio: the share of premium that ever passes
  through outstanding, valued at the case reserves' level.
* `RRF_i`, the reserve robustness factor: how much of what was reported is
  eventually paid. Below 1, case reserves are redundant; above 1, they fall
  short.
* `RLR_i RRF_i`, the ultimate loss ratio. It plays the part of the expected
  loss ratio (ELR) in a Bornhuetter-Ferguson or Cape Cod reserve, but it is
  estimated jointly from paid and outstanding rather than chosen beside them.
* `k_er` and `k_p`, the speeds of reporting and payment: shared by every
  accident year in model 1, varying by accident and development year in
  model 2.

**The hierarchy.** `RLR` and `RRF` vary by accident year, but not freely:
each year's value is the population's times a random effect, the two effects
correlated, with their spread estimated from the data. A young year with
little data borrows from the others, much as a credibility-weighted ELR
would; a mature year mostly speaks for itself. An accident year with premium
and no data at all (1997 below) takes a new effect from the population in
each simulation.

**Identification.** The two rates can swap places with `RLR` and `RRF`
rescaled and every mean stays exactly the same
(`knowledge/findings/compartmental-rate-swap.md`). Only the priors tell the
two mirror solutions apart; prospicio warns if any posterior draw has
`k_er <= k_p`, and none does here.

## 2. Company 337

The paper's company is NAIC company 337, workers' compensation, accident
years 1988 to 1997, from the CAS loss reserve database. Its outstanding is
incurred (with bulk and IBNR reserves) less paid, and its premium is direct
earned premium. The model is trained on the calendar years up to 1996: 45
paid and 45 outstanding cells for 1988 to 1996, and 1997 with its premium
only. The full 10 by 10 square is in the data, so the 1997 calendar diagonal
and the 120-month outcome are known and can test the forecast.

```python
import csv
import math
import textwrap
import warnings
from pathlib import Path

import numpy as np

from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle

DATA = Path("../../validation/data/clrd_wkcomp_337.csv")
with open(DATA, newline="") as f:
    square = list(csv.DictReader(line for line in f if not line.startswith("#")))
for r in square:
    r["origin"], r["age"] = int(r["origin"]), int(r["development"])
    r["paid"], r["incurred"], r["premium"] = (float(r[k]) for k in ("paid", "incurred", "premium"))
    r["os"] = r["incurred"] - r["paid"]
cell = {(r["origin"], r["age"]): r for r in square}
premium = {r["origin"]: r["premium"] for r in square if r["age"] == 12}

# Training cells: calendar years up to 1996. 1997 enters with its premium only.
train = [r for r in square if r["origin"] + r["age"] // 12 - 1 <= 1996]
tri = Triangle.from_long(
    origin=[r["origin"] for r in train] + [1997],
    development=[r["age"] for r in train] + [12],
    values={
        "paid": [r["paid"] for r in train] + [math.nan],
        "incurred": [r["incurred"] for r in train] + [math.nan],
        "premium": [r["premium"] for r in train] + [premium[1997]],
    },
)
latest_paid = sum(r["paid"] for r in train if r["origin"] + r["age"] // 12 - 1 == 1996)
ultimate_paid = sum(cell[o, 120]["paid"] for o in premium)
print(f"training cells {len(train)} of each type, premium {sum(premium.values()):,.0f}")
c = cell[1988, 12]
print(f"1988 at 12 months: premium {c['premium']:,.0f}, paid {c['paid']:,.0f}, outstanding {c['os']:,.0f}")
print(f"latest paid {latest_paid:,.0f}; paid at 120 months {ultimate_paid:,.0f}; "
      f"outcome reserve {ultimate_paid - latest_paid:,.0f}")
```

```text
training cells 45 of each type, premium 903,152
1988 at 12 months: premium 104,437, paid 9,558, outstanding 53,121
latest paid 402,015; paid at 120 months 589,435; outcome reserve 187,420
```

Morris (2016), the paper's predecessor, printed the same 1988 cell in millions:
premium 104, outstanding 53, paid 10. The outcome reserve is what was paid to
120 months after the end of 1996, by every accident year including 1997: the
number the models below are trying to predict.

## 3. Priors, and the prior predictive check

The priors are a required argument of `Compartmental`: the rates and loss
ratios are specific to a line and a book, so prospicio does not choose them
silently. `CompartmentalPriors.gesmann_morris()` is the paper's model 1 set.
Each positive parameter is a median times `exp(sd o)` with `o` standard
normal: `k_er` median 3 (sd 0.1 on the log scale), `k_p` 1 (0.1), `RLR` 0.7
(0.2), `RRF` 0.8 (0.1). The sds of the accident-year effects get half
Student-t priors (scales 0.2 and 0.1 on that `o` scale), their correlation a
uniform LKJ(1) prior, and each `log sigma` a Student-t prior so wide it is
effectively flat.

The paper urges a prior predictive check before fitting: draw the parameters
from the priors alone, run them through the curves and see whether the
implied development looks like a plausible book. `prior_predictive` returns
those paths (the means, without the likelihood's noise). Here are the paid
and outstanding loss ratios of 1988 they imply, next to what 1988 actually
did:

```python
GM1 = CompartmentalPriors.gesmann_morris()
model1 = Compartmental(GM1, n_sims=2000)


def prior_check(model, label):
    paths = model.prior_predictive(tri, "premium", n_sims=2000)
    print(f"{label}: 1988 loss ratios under the priors alone (5%, 50%, 95%) and actual")
    for kind, key in (("paid", "paid"), ("outstanding", "os")):
        for age in (12, 36, 120):
            m = paths[kind].marginal(("1988", age))
            q = [m.quantile(p) / premium[1988] for p in (0.05, 0.5, 0.95)]
            actual = cell[1988, age][key] / premium[1988]
            print(f"  {kind:11s} {age:3d} months  {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}   actual {actual:.2f}")


prior_check(model1, "model 1")
```

```text
model 1: 1988 loss ratios under the priors alone (5%, 50%, 95%) and actual
  paid         12 months  0.18 0.26 0.39   actual 0.09
  paid         36 months  0.36 0.52 0.76   actual 0.32
  paid        120 months  0.39 0.56 0.82   actual 0.50
  outstanding  12 months  0.24 0.33 0.47   actual 0.51
  outstanding  36 months  0.03 0.05 0.09   actual 0.31
  outstanding 120 months  0.00 0.00 0.00   actual 0.01
```

The check finds a conflict before any fitting. Under the preset, payment is
fast (`k_p` near 1 a year), so the priors put about a quarter of premium in
paid after 12 months and empty outstanding within a few years. Company 337 pays
far more slowly: 9% paid at 12 months, and 31% of premium still outstanding
at 36 months where the priors expect about 5%. The data will move the rates many prior sds away, and they do (section
4: `k_er` about 1.76 and `k_p` about 0.43 against medians 3 and 1). The paper
keeps the preset; a user fitting their own book should widen or recentre the
rate priors after a check like this one, with `GM1.replace(k_p=(0.5, 0.3))`
for instance.

## 4. Model 1: one likelihood for paid and outstanding amounts

Model 1 puts a normal likelihood on each observed cumulative paid and
outstanding amount about its curve, with one `sigma` for each type. `RLR`
and `RRF` vary by accident year; the rates do not. The paper prints its
posterior summary (`m1fit`, section 7.2.2) on brms' scale: the intercepts
`b_*` on the standardised `o` scale of each parameter, the sds `tau` of the
accident-year effects on that scale, their correlation `rho`, and
`log sigma` of each type. prospicio reports the same names.

The tolerances: prospicio's means move by at most 0.032 posterior sds from
seed to seed and its sds by at most 2.3% (200 seeds); the paper's run had
about 1,300 to 2,200 effective draws, so its means carry about 0.03 sds of
error too, and it rounds to two decimals. A tolerance of 0.15 printed sds
(plus 0.005 for the rounding) on a mean and 10% on an sd is about three
combined errors wide.

```python
def check(label, target, ours, tol, fmt=".3f", source="paper"):
    agrees = abs(ours - target) <= tol
    print(f"{label:24s} {source:5s} {target:>9{fmt}}  tolerance {tol:>7{fmt}}  agrees: {agrees}")
    return agrees


def fit_quietly(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = model.fit(tri, "paid", "premium", incurred="incurred")
    return fit, len(caught)


def check_posterior(fit, printed):
    s = fit.summary().set_index("parameter")
    ok = 0
    for name, (mean, sd) in printed.items():
        ok += check(f"{name} mean", mean, s.loc[name, "mean"], 0.15 * sd + 0.005)
        ok += check(f"{name} sd", sd, s.loc[name, "sd"], 0.10 * sd + 0.005)
    return ok


# The paper's printed m1fit, mean and sd.
M1FIT = {
    "b_ker": (-5.40, 0.81), "b_kp": (-8.49, 0.36), "b_RLR": (1.44, 0.26), "b_RRF": (-1.36, 0.42),
    "tau_RLR": (0.64, 0.15), "tau_RRF": (0.74, 0.22), "rho": (0.54, 0.26),
    "log_sigma_os": (8.25, 0.16), "log_sigma_pd": (6.66, 0.20),
}
fit1, n_warn1 = fit_quietly(model1)
print("warnings", n_warn1, "| rate swaps", fit1.rate_swaps)
print("rows agreeing", check_posterior(fit1, M1FIT), "of", 2 * len(M1FIT))
```

```text
warnings 0 | rate swaps 0
b_ker mean               paper    -5.400  tolerance   0.127  agrees: True
b_ker sd                 paper     0.810  tolerance   0.086  agrees: True
b_kp mean                paper    -8.490  tolerance   0.059  agrees: True
b_kp sd                  paper     0.360  tolerance   0.041  agrees: True
b_RLR mean               paper     1.440  tolerance   0.044  agrees: True
b_RLR sd                 paper     0.260  tolerance   0.031  agrees: True
b_RRF mean               paper    -1.360  tolerance   0.068  agrees: True
b_RRF sd                 paper     0.420  tolerance   0.047  agrees: True
tau_RLR mean             paper     0.640  tolerance   0.028  agrees: True
tau_RLR sd               paper     0.150  tolerance   0.020  agrees: True
tau_RRF mean             paper     0.740  tolerance   0.038  agrees: True
tau_RRF sd               paper     0.220  tolerance   0.027  agrees: True
rho mean                 paper     0.540  tolerance   0.044  agrees: True
rho sd                   paper     0.260  tolerance   0.031  agrees: True
log_sigma_os mean        paper     8.250  tolerance   0.029  agrees: True
log_sigma_os sd          paper     0.160  tolerance   0.021  agrees: True
log_sigma_pd mean        paper     6.660  tolerance   0.035  agrees: True
log_sigma_pd sd          paper     0.200  tolerance   0.025  agrees: True
rows agreeing 18 of 18
```

Every row agrees. On Windows at seed 0 prospicio gives `b_ker` -5.395
(0.833), `b_kp` -8.485 (0.356), `b_RLR` 1.446 (0.260), `b_RRF` -1.357
(0.414), `tau_RLR` 0.643 (0.150), `tau_RRF` 0.739 (0.220), `rho` 0.533
(0.259), `log_sigma_os` 8.249 (0.160) and `log_sigma_pd` 6.671 (0.204). The
largest gap to the paper, 0.011 on `log_sigma_pd`, is 0.055 printed sds. The
same model written as the Stan program brms generates and run under CmdStan
for 100,000 draws gives the same values to within 0.04 sds of the paper
(`knowledge/references/gesmann-morris-2020.md`).

On the natural scale this is `k_er` about 1.76 and `k_p` about 0.43 a year:
claims are reported about seven months after the start on average
(`1 / k_er` years) and paid about 2.3 years after that (`1 / k_p`). The population `RLR` is about 0.94 and
`RRF` about 0.70, so the case reserves (with bulk) are redundant by about 30%
and the population ultimate loss ratio is about 0.65:

```python no-run
s1 = fit1.summary().set_index("parameter")
for name in ("k_er", "k_p", "RLR", "RRF"):
    print(f"{name:4s} posterior mean {s1.loc[name, 'mean']:.2f}, 90% interval "
          f"{s1.loc[name, 'q05']:.2f} to {s1.loc[name, 'q95']:.2f}")
```

```text
k_er posterior mean 1.76, 90% interval 1.54 to 2.02
k_p  posterior mean 0.43, 90% interval 0.40 to 0.45
RLR  posterior mean 0.94, 90% interval 0.86 to 1.02
RRF  posterior mean 0.70, 90% interval 0.65 to 0.75
```

## 5. The fitted paid and outstanding development

The fitted curves come straight from the posterior draws: each draw's rates
and its accident year's `RLR` and `RRF` give a curve, and the posterior mean
of the curves is the fit. Here are two accident years, a mature one and a
young one, actual beside fitted at each observed age:

```python no-run
names = fit1.parameter_names
draws = np.array(fit1.draws)
par = {n: draws[:, i] for i, n in enumerate(names)}


def fitted(origin, age):
    ker, kp = par["k_er"], par["k_p"]
    rlr, rrf, t = par[f"RLR[{origin}]"], par[f"RRF[{origin}]"], age / 12
    g = (np.exp(-kp * t) - np.exp(-ker * t)) / (ker - kp)
    os_ = premium[origin] * rlr * ker * g
    pd_ = premium[origin] * rlr * rrf * ((1 - np.exp(-kp * t)) - kp * g)
    return os_.mean(), pd_.mean()


for origin in (1988, 1994):
    print(f"{origin}  age   paid actual  fitted   outstanding actual  fitted")
    for age in range(12, 12 * (1997 - origin) + 1, 12):
        f_os, f_pd = fitted(origin, age)
        c = cell[origin, age]
        print(f"      {age:3d}  {c['paid']:>11,.0f} {f_pd:>7,.0f}  {c['os']:>18,.0f} {f_os:>7,.0f}")
```

```text
1988  age   paid actual  fitted   outstanding actual  fitted
       12        9,558  10,266              53,121  53,150
       24       22,778  23,740              41,222  43,904
       36       33,298  33,712              32,309  30,258
       48       40,348  40,423              24,944  20,017
       60       45,146  44,835              17,104  13,101
       72       48,048  47,718              13,137   8,551
       84       49,782  49,599               9,605   5,578
       96       50,623  50,826               6,515   3,638
      108       51,812  51,626               1,661   2,373
1994  age   paid actual  fitted   outstanding actual  fitted
       12       11,194  11,693              62,434  58,627
       24       26,893  27,040              46,661  48,432
       36       38,488  38,401              32,248  33,380
```

Paid fits closely at every age. Outstanding does not: 1988's actual
outstanding runs off more slowly than the two-rate curve allows (24,944 at 48
months against 20,017 fitted, 13,137 at 72 against 8,551), and the curve
makes up for it at the latest age. With one `k_p` for every age, the only
way to hold outstanding up longer would be slower payment, which the paid
cells refuse. A normal likelihood with one `sigma` per type also weighs a
cell's error in absolute terms, so the large early cells dominate and the
small late ones fit loosely. Model 2 (section 7) answers both points.

The test diagonal, calendar year 1997, was not used in fitting. The fit's
`paid` and `outstanding` are the posterior predictive distributions of every
future cell, with the likelihood's noise:

```python no-run
print("origin age   paid actual  predicted (5% to 95%)      outstanding actual  predicted (5% to 95%)")
inside = 0
for origin in range(1988, 1998):
    age = 12 * (1997 - origin + 1)
    row = f"{origin}  {age:3d}"
    for kind, key in (("paid", "paid"), ("outstanding", "os")):
        m = getattr(fit1, kind).marginal((str(origin), age))
        lo, hi, actual = m.quantile(0.05), m.quantile(0.95), cell[origin, age][key]
        inside += lo <= actual <= hi
        row += f"  {actual:>11,.0f} {m.mean():>9,.0f} ({lo:>7,.0f} to {hi:>7,.0f})"
    print(row)
print(f"actual inside the 90% interval: {inside} of 20 cells")
```

```text
origin age   paid actual  predicted (5% to 95%)      outstanding actual  predicted (5% to 95%)
1988  120       51,939    52,134 ( 50,555 to  53,704)        1,322     1,442 ( -4,767 to   7,654)
1989  108       46,229    46,471 ( 44,974 to  47,984)        2,071     2,225 ( -4,221 to   8,463)
1990   96       53,956    53,918 ( 52,229 to  55,396)        3,015     3,400 ( -2,962 to   9,698)
1991   84       66,566    68,301 ( 66,629 to  69,847)        3,966     6,093 (   -214 to  12,605)
1992   72       59,437    59,640 ( 57,874 to  61,288)        8,408     9,719 (  3,262 to  16,244)
1993   60       50,742    51,569 ( 49,766 to  53,265)       16,955    15,255 (  8,524 to  21,299)
1994   48       45,580    46,036 ( 44,052 to  47,800)       24,140    21,936 ( 15,299 to  28,437)
1995   36       44,045    44,009 ( 41,575 to  46,110)       35,336    32,662 ( 26,005 to  39,666)
1996   24       31,474    29,551 ( 26,296 to  32,548)       41,707    43,566 ( 35,806 to  51,858)
1997   12        9,372     6,225 (  4,061 to   8,740)       40,799    29,044 ( 19,668 to  38,538)
actual inside the 90% interval: 17 of 20 cells
```

Seventeen of the twenty cells fall inside their 90% intervals. Two of the
three misses are the youngest accident year: 1997's first paid (9,372) and outstanding
(40,799) are well above anything the model expected from its premium, a
first sign of what section 7 finds in the reserve. The outstanding intervals
of the old years reach below zero: the normal likelihood with a constant
`sigma` (about 3,900) knows nothing of a cell's size, another reason for
model 2's log-normal.

## 6. The reserve distribution

The reserve of an accident year is its paid at 120 months less its latest
paid; for 1997 that is all of it. `fit1.reserves` is the joint posterior
predictive distribution by accident year, with parameter and process error.
The paper prints no reserve, so the comparison is with the Stan
reconstruction of its code (100,000 draws; prospicio's mean moves by about
0.016 sds from seed to seed and its sd by about 1.4%, at 10,000 simulations;
these use 2,000, so the tolerances are 0.1 sds and 7%):

```python
r1 = np.array(fit1.reserves.draw_matrix())
total1 = r1.sum(axis=1)
check("total reserve mean", 173_341, total1.mean(), 1_000, ",.0f", "Stan")
check("total reserve sd", 10_076, total1.std(), 700, ",.0f", "Stan")
i97 = fit1.origins.index("1997")
check("1997 reserve mean", 31_321, r1[:, i97].mean(), 600, ",.0f", "Stan")
check("1997 reserve sd", 6_125, r1[:, i97].std(), 430, ",.0f", "Stan")
outcome = ultimate_paid - latest_paid
lo, hi = np.quantile(total1, [0.05, 0.95])
print(f"outcome {outcome:,.0f} inside the 90% interval: {bool(lo <= outcome <= hi)}")
```

```text
total reserve mean       Stan    173,341  tolerance   1,000  agrees: True
total reserve sd         Stan     10,076  tolerance     700  agrees: True
1997 reserve mean        Stan     31,321  tolerance     600  agrees: True
1997 reserve sd          Stan      6,125  tolerance     430  agrees: True
outcome 187,420 inside the 90% interval: True
```

```python no-run
print("origin  premium  latest paid  reserve mean      sd  ultimate loss ratio")
for row in fit1.to_frame().itertuples():
    print(f"{row.origin:6s} {row.premium:>8,.0f} {row.latest_paid:>12,.0f} {row.mean:>13,.0f} "
          f"{row.std_dev:>7,.0f} {row.ultimate_loss_ratio:>20.3f}")
print(f"total: mean {total1.mean():,.0f}, sd {total1.std():,.0f}, 5% {np.quantile(total1, 0.05):,.0f}, "
      f"95% {np.quantile(total1, 0.95):,.0f}; outcome {outcome:,.0f} at percentile "
      f"{100 * np.mean(total1 <= outcome):.1f}")
```

```text
origin  premium  latest paid  reserve mean      sd  ultimate loss ratio
1988    104,437       51,812           322     948                0.509
1989     88,883       45,871         1,056     953                0.538
1990     85,956       52,879         2,471     984                0.656
1991     99,339       65,039         6,776   1,113                0.736
1992    104,897       56,406         8,756   1,172                0.633
1993    119,427       46,722        13,272   1,208                0.512
1994    110,784       38,488        20,927   1,420                0.546
1995     77,731       31,604        36,486   2,228                0.893
1996     63,646       13,194        51,755   4,388                1.041
1997     48,052            0        31,636   6,109                0.665
total: mean 173,457, sd 10,260, 5% 156,936, 95% 189,868; outcome 187,420 at percentile 91.9
```

The total reserve is about 173,500 with an sd of about 10,300 (Stan's
reconstruction: 173,341 and 10,076). The two youngest years carry half of
it, and 1997, with no data, carries the widest uncertainty: its loss ratio
is the population's, drawn afresh in every simulation. The ultimate loss
ratios are each year's `RLR RRF`, the model's ELR by accident year, shrunk
toward the population's. The outcome, 187,420, falls at about the 92nd
percentile: inside the 90% interval, but high.

## 7. Model 2: log-normal loss ratios and effects on the rates

Model 2 changes three things.

* **The response.** It models loss ratios to premium, not amounts, and paid
  as the incremental paid loss ratio of each year rather than the cumulative.
  An incremental cell is the difference of the paid curve between `t - 1` and
  `t`. Cumulative paid cells are strongly correlated with each other, and an
  independent likelihood on them overstates how much information they hold;
  incrementals are closer to independent.
* **The likelihood.** Each response is log-normal about its curve, so the
  noise is proportional to the cell's size, and a small late cell is not
  swamped by a large early one, as it is under model 1's constant `sigma`.
  The `log sigma` priors are now informative, normal with median 0.2 for
  `sigma` (the code's value; the paper's text gives 0.1, see the reference
  note).
* **The hierarchy.** All four parameters get accident-year effects (`RLR`
  and `RRF` correlated, the rates independent), and all four get
  development-year effects too, so each cell has its own `k_er`, `k_p`,
  `RLR` and `RRF`. A development effect lets the curve bend where the
  two-rate shape is too stiff; an age with no observed cell (120 months
  here) draws a new effect from the population.

`Compartmental.model_2(CompartmentalPriors.gesmann_morris_2())` builds it,
with a longer sampler than model 1's (4 chains of 2,000 warm-up and 8,000
kept draws at target acceptance 0.999): with 87 parameters, several effect
sds near zero and a slowly mixing `k_p`, model 1's sampler left divergent
transitions on most seeds. Its prior predictive check is quick, so it runs:

```python
GM2 = CompartmentalPriors.gesmann_morris_2()
model2 = Compartmental.model_2(GM2, n_sims=2000)
print(model2.likelihood, model2.scale, model2.paid_response, model2.tune, model2.draws, model2.target_accept)
prior_check(model2, "model 2")
```

```text
lognormal loss_ratio incremental 2000 8000 0.999
model 2: 1988 loss ratios under the priors alone (5%, 50%, 95%) and actual
  paid         12 months  0.15 0.26 0.45   actual 0.09
  paid         36 months  0.32 0.51 0.85   actual 0.32
  paid        120 months  0.35 0.56 0.92   actual 0.50
  outstanding  12 months  0.20 0.33 0.57   actual 0.51
  outstanding  36 months  0.03 0.05 0.10   actual 0.31
  outstanding 120 months  0.00 0.00 0.00   actual 0.01
```

The medians are model 1's, since the rate and ratio priors are the same;
the intervals are wider, because each development year and accident year can
now shift the rates and ratios too. The conflict with company 337's slow
payment is the same, and the posterior again moves far from the preset,
though less than model 1's (`k_er` about 2.6 and `k_p` about 0.60 at the
population, against model 1's 1.76 and 0.43), because the development
effects can now bend the curve instead.

The fit takes about a minute, so it is not run in CI. The comparison with the
paper's printed `m2fit` (section 7.2.3), with prospicio's value and the gap in
printed sds:

```python no-run
M2FIT = {
    "b_ker": (-1.31, 1.08), "b_kp": (-5.07, 1.75), "b_RLR": (1.54, 0.44), "b_RRF": (-1.45, 0.69),
    "tau_RLR": (0.69, 0.31), "tau_RRF": (0.73, 0.47), "tau_ker": (0.26, 0.22), "tau_kp": (2.23, 1.35),
    "rho": (0.37, 0.47), "tau_dev_RLR": (0.35, 0.26), "tau_dev_RRF": (1.11, 0.49),
    "tau_dev_ker": (0.27, 0.23), "tau_dev_kp": (0.45, 0.43),
    "log_sigma_os": (-1.78, 0.16), "log_sigma_pd": (-1.89, 0.16),
}
fit2, n_warn2 = fit_quietly(model2)
print("warnings", n_warn2, "| draws with k_er <= k_p in an observed cell", fit2.cell_rate_swaps)
s2 = fit2.summary().set_index("parameter")
print(f"{'':14s} {'paper mean':>10s} {'ours':>7s} {'gap/sd':>7s}   {'paper sd':>8s} {'ours':>6s}")
for name, (mean, sd) in M2FIT.items():
    m, d = s2.loc[name, "mean"], s2.loc[name, "sd"]
    print(f"{name:14s} {mean:>10.2f} {m:>7.3f} {(m - mean) / sd:>7.3f}   {sd:>8.2f} {d:>6.3f}")
```

```text
warnings 0 | draws with k_er <= k_p in an observed cell 0.0
               paper mean    ours  gap/sd   paper sd   ours
b_ker               -1.31  -1.294   0.015       1.08  1.064
b_kp                -5.07  -5.112  -0.024       1.75  1.692
b_RLR                1.54   1.539  -0.002       0.44  0.447
b_RRF               -1.45  -1.456  -0.009       0.69  0.697
tau_RLR              0.69   0.699   0.030       0.31  0.306
tau_RRF              0.73   0.731   0.003       0.47  0.470
tau_ker              0.26   0.263   0.015       0.22  0.214
tau_kp               2.23   2.175  -0.041       1.35  1.315
rho                  0.37   0.358  -0.025       0.47  0.474
tau_dev_RLR          0.35   0.352   0.009       0.26  0.264
tau_dev_RRF          1.11   1.124   0.029       0.49  0.498
tau_dev_ker          0.27   0.274   0.016       0.23  0.228
tau_dev_kp           0.45   0.483   0.076       0.43  0.511
log_sigma_os        -1.78  -1.787  -0.045       0.16  0.165
log_sigma_pd        -1.89  -1.887   0.017       0.16  0.162
```

Every mean is within 0.08 printed sds of the paper's, and every sd within 4%
but one: `tau_dev_kp`, the sd of the development effects on `k_p`, 0.511
against the printed 0.43. That sd is heavy-tailed and mixes slowly; the
paper's own run had only about 500 effective draws on `k_p`'s parameters, and
the Stan reconstruction of its code with 100,000 draws gives 0.518, so the
gap is the paper's Monte Carlo error rather than a difference of model.
prospicio's means move by at most 0.036 posterior sds from seed to seed
(200 seeds), and every draw keeps `k_er` above `k_p` in every observed cell.

The two models' reserves and ultimate loss ratios, with what the accident
years had paid at 120 months:

```python no-run
r2 = np.array(fit2.reserves.draw_matrix())
total2 = r2.sum(axis=1)
for label, t in (("model 1", total1), ("model 2", total2)):
    print(f"{label}: total reserve mean {t.mean():,.0f}, sd {t.std():,.0f}, 5% {np.quantile(t, 0.05):,.0f}, "
          f"95% {np.quantile(t, 0.95):,.0f}; outcome at percentile {100 * np.mean(t <= outcome):.1f}")
print("ultimate loss ratio by accident year, model 1 and model 2")
for o, a, b in zip(fit1.origins, fit1.ultimate_loss_ratio, fit2.ultimate_loss_ratio):
    actual = cell[int(o), 120]["paid"] / premium[int(o)]
    print(f"  {o}  {a:.3f}  {b:.3f}   paid at 120 months {actual:.3f}")
```

```text
model 1: total reserve mean 173,457, sd 10,260, 5% 156,936, 95% 189,868; outcome at percentile 91.9
model 2: total reserve mean 151,861, sd 13,729, 5% 130,835, 95% 175,476; outcome at percentile 99.2
ultimate loss ratio by accident year, model 1 and model 2
  1988  0.509  0.543   paid at 120 months 0.497
  1989  0.538  0.582   paid at 120 months 0.523
  1990  0.656  0.686   paid at 120 months 0.638
  1991  0.736  0.727   paid at 120 months 0.694
  1992  0.633  0.657   paid at 120 months 0.604
  1993  0.512  0.565   paid at 120 months 0.485
  1994  0.546  0.586   paid at 120 months 0.499
  1995  0.893  0.849   paid at 120 months 0.862
  1996  1.041  0.885   paid at 120 months 1.072
  1997  0.665  0.674   paid at 120 months 1.152
```

Model 2's total, about 151,900 (Stan's reconstruction: 152,242, sd 13,938),
is about 21,600 below model 1's, with a wider spread. Neither model saw what
happened next: the outcome, 187,420, is at model 1's 92nd percentile and
model 2's 99th. The miss is in the youngest years. 1996 and 1997 paid 1.07
and 1.15 of their premium by 120 months, against ultimate loss ratios of
0.89 and 0.67 under model 2 and 1.04 and 0.67 under model 1: premium fell
from 119,427 in 1993 to 48,052 in 1997 while losses did not, the pattern of
a softening market. Both models shrink a young year toward the population's
loss ratio, and model 2, with more effects to share information through,
shrinks harder. The paper's model 3 adds market cycle multipliers for this;
prospicio does not have it (section 10). One triangle is one outcome, so
this says nothing about the models' calibration in general, only that the
shrinkage of young years deserves a judgement when the market is turning.

## 8. Comparing the two models: elpd_loo does not

Both fits report PSIS-LOO (`fit.loo()`), the expected log predictive density
of each observed cell left out in turn. It is tempting to compare model 1's
and model 2's, and it would be wrong: model 1's responses are amounts in
thousands, cumulative paid; model 2's are loss ratios, incremental paid. A
log density depends on the unit of its response, so the two numbers are on
different scales and say nothing about which model predicts better. Compare
elpd_loo only between fits of the same responses (two prior sets for model 1,
say). Model 1's runs in CI, checked against the Stan reconstruction (which
moves by about 0.35 from seed to seed):

```python
loo1 = fit1.loo()
check("model 1 elpd_loo", -811.0, loo1.elpd, 2.0, ".1f", "Stan")
k1 = np.array(loo1.pareto_k)
print(f"model 1 cells with Pareto k above 0.7: at least one {bool((k1 > 0.7).any())}")
```

```text
model 1 elpd_loo         Stan     -811.0  tolerance     2.0  agrees: True
model 1 cells with Pareto k above 0.7: at least one True
```

```python no-run
for label, fit in (("model 1", fit1), ("model 2", fit2)):
    loo = fit.loo()
    k = np.array(loo.pareto_k)
    by_type = fit.loo_by_type()
    print(f"{label}: elpd_loo {loo.elpd:.1f} (se {loo.se:.1f}), outstanding {by_type['elpd_outstanding']:.1f}, "
          f"paid {by_type['elpd_paid']:.1f}; Pareto k above 0.7 in {(k > 0.7).sum()} of {len(k)} cells, "
          f"largest {k.max():.2f}")
```

```text
model 1: elpd_loo -810.7 (se 10.2), outstanding -440.8, paid -370.0; Pareto k above 0.7 in 3 of 90 cells, largest 0.96
model 2: elpd_loo 203.8 (se 11.5), outstanding 71.5, paid 132.2; Pareto k above 0.7 in 9 of 90 cells, largest 1.30
```

A few cells have Pareto `k` above 0.7, where the importance-sampling
approximation behind PSIS-LOO is unreliable: the left-out cell moves the
posterior too much for the full posterior's draws to stand in for the fit
without it. The elpd_loo is then an estimate with an unknown error, not a
measure of predictive accuracy; refitting without those cells (exact LOO)
would settle it, and prospicio does not automate that. For model 2, its
elpd_loo matched arviz's computation on the same draws (finding
`compartmental-gm2-parity.md`), so the row checks the arithmetic, not the
fit. To choose between the two models, compare their forecasts on the test
diagonal and the outcome (sections 5 to 7), on the same scale.

## 9. The one-year view

A solvency view asks not how far the reserve may move over its whole life
but how far the best estimate may move in the coming year: the claims
development result (CDR), the opening reserve less the year's paid and the
reserve re-estimated a year later. `Compartmental.one_year` simulates the
next calendar diagonal (paid and outstanding of every accident year) from the
posterior, and for each simulated year re-estimates the posterior by
reweighting the existing draws with the likelihood of the new cells (Pareto
smoothed importance sampling), instead of refitting.

The new cells move this posterior a lot, so most simulations' weights fail
the Pareto `k` test (`k` above 0.7). For model 1, prospicio then moment
matches the draws (small affine shifts of the draws that make the weights
reliable again) and refits with NUTS only the few that matching cannot
rescue; on company 337 matching rescues about 95% within about 0.05 sds of a
refit (`knowledge/findings/compartmental-one-year.md`). The view runs at most
2,000 simulations by default and takes several minutes, so it is not run in
CI:

```python no-run
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    view1 = model1.one_year(tri, "paid", "premium", incurred="incurred")
cdr1 = np.array(view1.cdr.draw_matrix()).sum(axis=1)
print(f"model 1: flagged {view1.flagged_share:.0%}, moment-matched {len(view1.moment_matched):,}, "
      f"refitted {len(view1.refitted):,}, refits with a diagnostic warning {view1.refit_warnings}")
print("warnings", len(caught))
print(f"total CDR mean {cdr1.mean():,.0f}, sd {cdr1.std():,.0f}; lifetime reserve sd {total1.std():,.0f}; "
      f"ratio {cdr1.std() / total1.std():.2f}")
```

```text
model 1: flagged 81%, moment-matched 1,549, refitted 78, refits with a diagnostic warning 8
warnings 1
total CDR mean 55, sd 7,982; lifetime reserve sd 10,260; ratio 0.78
```

About four in five simulations are flagged; matching rescues about 95% of
them and the rest are refitted with NUTS (most of the run time). The warning
is about the refits: the count above missed an MCMC diagnostic threshold, so
read `view1.refit_warnings` before relying on the far tail of the CDR. About 78% of the lifetime reserve's sd emerges in the
first year, in line with the 200-seed study (0.80, sd 0.03). The CDR's mean
is about zero, as it should be.

For model 2 matching rescues none of the flagged simulations (its posterior
has 95 sampled values, several effect sds near zero, far from the
near-normal shape the shifts can follow), and a refit takes as long as a fit,
so a full refit view would take days. Its default view therefore reweights
only and warns that it is approximate: on company 337 such a view sits about
0.4 sds of the total CDR from a refit per simulation. `refit` and
`max_refits` buy accuracy with time.

```python no-run
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    view2 = model2.one_year(tri, "paid", "premium", incurred="incurred")
cdr2 = np.array(view2.cdr.draw_matrix()).sum(axis=1)
print(f"model 2: flagged {view2.flagged_share:.0%}, moment-matched {len(view2.moment_matched):,}, "
      f"refitted {len(view2.refitted):,}")
print("warnings", len(caught))
print(textwrap.fill(f"approximate: {view2.approximate}", 88))
print(f"total CDR mean {cdr2.mean():,.0f}, sd {cdr2.std():,.0f}; lifetime reserve sd {total2.std():,.0f}; "
      f"ratio {cdr2.std() / total2.std():.2f}")
```

```text
model 2: flagged 100%, moment-matched 0, refitted 0
warnings 1
approximate: the one-year view reweights its flagged simulations only, the default with
development effects (model 2): moment matching rescues none of them and each refit takes
about as long as a fit. The view is approximate, about 0.4 sd of the total claims
development result off a refit per simulation on company 337; set refit (with max_refits
as a budget) to refit them
total CDR mean 103, sd 11,379; lifetime reserve sd 13,729; ratio 0.83
```

Every simulation is flagged and reweighted only, and the view says so in
`approximate` and in a warning. Its CDR sd is 0.83 of the lifetime sd (0.82
in the study, 0.78 to 0.84 over three seeds); read it as an approximation
good to about 0.4 sds, not as a refit-quality number.

## 10. What this guide does not reproduce

* **The paper's plots.** The fitted curves against the data, the
  posterior predictive bands and the prior predictive fans are figures in the
  paper; this guide prints the numbers behind such plots (sections 3 and 5)
  and draws none.
* **Model 3 and the extensions.** The market-cycle model, the multistage
  exposure and time-varying rates are outside prospicio's `Compartmental`
  for now (the design note lists them as later work).
* **Several triangles at once and the backtest.** `Compartmental` fits one
  triangle. Fitting several segments jointly (`fit_segments`) and the
  diagonal backtest, which other Bayesian models in prospicio have, are not
  yet built for it (the Reserving lane's next step).
* **Exact digits.** The checked blocks hold on every platform; the recorded
  `no-run` values are Windows', and another platform prints others within
  the stated Monte Carlo errors.
