# Iman-Conover, uncertainty and capital by layer: CAS Monograph 15, chapter 8

<!-- site: execute -->

CAS Monograph 15 is *Introduction to Capital Modeling and Portfolio
Management* (John Major and Stephen Mildenhall, 2026). Chapter 8 covers
parameter uncertainty, the Iman-Conover method, capital allocation by layer
and Bodoff's layer allocation. This guide runs each topic in prospicio and
shows the monograph's numbers beside prospicio's.

Exact numbers (Example 2's regional means and percentiles, the InsCo tables of
sections 8.3 and 8.4) agree with prospicio to the printed digits. Numbers from
one sample of 1,000 simulations (the achieved correlations of Examples 1 and 2)
agree within sampling error, whose size the guide shows. The explanations are
in new words; the table numbers are the monograph's.

Companion guides: [chapter 9](capital-modeling-monograph-15.md),
[chapters 3 and 4](capital-modeling-monograph-15-theory.md),
[chapters 5 and 6](capital-modeling-monograph-15-applications.md) and
[chapters 1, 2 and 7](capital-modeling-monograph-15-practice.md).

## 1. Parameter uncertainty

Section 8.1 separates process risk (the randomness of a loss with a known
distribution) from parameter uncertainty (an unknown distribution). It gives
four responses: ignore the uncertainty, report a range, absorb it into a
predictive distribution, or isolate a credible worst case. Take a lognormal
severity with `sigma` 1.2 and a mean of about 100, whose log-scale location is
known only to within 0.3 in each direction:

```python
import math

import pandas as pd
from great_tables import GT

from prospicio.distributions import Lognormal

SIGMA = 1.2
best = math.log(100.0) - SIGMA**2 / 2
severities = [Lognormal(best + shift, SIGMA) for shift in (-0.3, 0.0, 0.3)]

def mixture_quantile(dists, p, lo=0.0, hi=1e6):
    """The p quantile of an equal mixture, by bisection on its cdf."""
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(d.cdf(mid) for d in dists) / len(dists) < p else (lo, mid)
    return (lo + hi) / 2

responses = pd.DataFrame(
    [{"response": f"report, {name}", "mean": d.mean(), "q": d.quantile(0.995)}
     for name, d in zip(("low", "best", "high"), severities)]
    + [{"response": "ignore it", "mean": None, "q": severities[1].quantile(0.995)},
       {"response": "absorb it", "mean": None, "q": mixture_quantile(severities, 0.995)},
       {"response": "isolate it", "mean": None, "q": severities[2].quantile(0.995)}]
)
(GT(responses, rowname_col="response")
 .tab_header("Responses to parameter uncertainty", "Lognormal severity with sigma 1.2; the location moves by 0.3")
 .cols_label(q="99.5% quantile")
 .fmt_number(["mean", "q"], decimals=1, use_seps=False)
 .sub_missing(missing_text=""))
```

```text
               mean  99.5% quantile
report, low    74.1           793.3
report, best  100.0          1070.8
report, high  135.0          1445.4
ignore it                    1070.8
absorb it                    1139.9
isolate it                   1445.4
```

Absorbing the uncertainty, with equal weights on the three locations, gives a
quantile about 6% above the best estimate. This is the downward bias that the
monograph finds in a quantile that ignores parameter uncertainty.

`PredictiveDistribution.blend` does the same mixing of whole simulations: it
takes each simulation from one of several models with given weights. The
Bayesian models (for example `BayesOdp` in reserving) return predictive
distributions that already include parameter uncertainty.

## 2. Example 1: three units

Section 8.2.2 correlates three units: Auto, GL and Property. Each unit has a
lognormal severity limited per occurrence and a gamma-mixed Poisson count:

| Unit | Expected loss | Limit | Severity mean | Severity CV | Mixing CV |
|---|---|---|---|---|---|
| Auto | 5,000 | 2,000 | 30 | 2 | 0.2 |
| GL | 2,000 | 5,000 | 50 | 5 | 0.3 |
| Property | 7,000 | 10,000 | 100 | 5 | 0.1 |

prospicio compounds each unit by FFT. Then it draws 1,000 independent years
from the three distributions, as the monograph does:

```python
import numpy as np

from prospicio.aggregate import fft
from prospicio.distributions import Count, Grid, PredictiveDistribution
from prospicio.risk import GaussianCopula, iman_conover, simulate

def unit(loss, limit, mean, cv, mixing_cv, step=1.0, points=2**16):
    """A unit's annual loss: limited lognormal claims, gamma-mixed Poisson count."""
    severity, _ = Grid.local_moment(Lognormal.from_mean_cv(mean, cv), step, points)
    claim, _ = severity.map(lambda x: min(x, limit))
    annual, _ = fft(Count.mixed_poisson(loss / claim.mean(), mixing_cv), claim, points)
    return annual

UNITS = {"Auto": (5000, 2000, 30, 2, 0.2), "GL": (2000, 5000, 50, 5, 0.3),
         "Property": (7000, 10000, 100, 5, 0.1)}
grids = [unit(*args) for args in UNITS.values()]
KEYS = [(name,) for name in UNITS]
IDENTITY = np.eye(3).tolist()
sample = simulate(GaussianCopula(IDENTITY), grids, 1000, 815, keys=KEYS, dims=["unit"])
X = np.array(sample.draw_matrix())

BOOK_81 = {"Auto": (5011.7, 0.257), "GL": (2092.4, 0.660), "Property": (7117.8, 0.467)}
table_81 = pd.DataFrame(
    [{"unit": name, "fft_mean": g.mean(), "fft_cv": g.variance() ** 0.5 / g.mean(),
      "mean": X[:, j].mean(), "cv": X[:, j].std(ddof=1) / X[:, j].mean(),
      "book_mean": BOOK_81[name][0], "book_cv": BOOK_81[name][1]}
     for j, (name, g) in enumerate(zip(UNITS, grids))]
)
(GT(table_81, rowname_col="unit")
 .tab_header("Table 8.1: three units, model and sample", "Annual loss; the sample has 1,000 years")
 .cols_label(fft_mean="FFT mean", fft_cv="FFT cv", mean="sample mean", cv="sample cv",
             book_mean="book mean", book_cv="book cv")
 .fmt_number(["fft_mean", "mean", "book_mean"], decimals=1, use_seps=False)
 .fmt_number(["fft_cv", "cv", "book_cv"], decimals=3))
```

```text
          FFT mean  FFT cv  sample mean  sample cv  book mean  book cv
Auto        5000.0   0.262       4998.8      0.262     5011.7    0.257
GL          2000.0   0.672       2042.0      0.644     2092.4    0.660
Property    7000.0   0.465       7017.0      0.454     7117.8    0.467
```

The FFT columns are the model. Both samples differ from it by a few percent,
which is sampling error, not a model difference.

## 3. The target and the achieved correlation

The target (Table 8.2) has a correlation of -0.3 between Auto and GL, 0.8
between GL and Property, and 0 between Auto and Property. A target must be a
correlation matrix: symmetric and positive definite. prospicio refuses one
that is not:

```python
TARGET = [[1.0, -0.3, 0.0], [-0.3, 1.0, 0.8], [0.0, 0.8, 1.0]]
print("eigenvalues of the target:", np.linalg.eigvalsh(TARGET).round(3))
try:
    iman_conover(sample, [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]], 1)
except ValueError:
    print("not positive definite: ValueError")
```

```text
eigenvalues of the target: [0.146 1.    1.854]
not positive definite: ValueError
```

`iman_conover` reorders each column of the sample. Each unit keeps exactly its
own 1,000 values, and only the grouping of the values into years changes:

```python
PAIRS = [(0, 1), (0, 2), (1, 2)]
NAMES = ["Auto-GL", "Auto-Property", "GL-Property"]

def ranks(a):
    """Ranks from 0, ties in order of appearance (as R's rank(ties.method = "first"))."""
    return np.argsort(np.argsort(a, axis=0, kind="stable"), axis=0, kind="stable")

def pairs(m):
    """Pearson and Spearman correlation of each pair of columns."""
    pearson, spearman = np.corrcoef(m.T), np.corrcoef(ranks(m).T)
    return [(pearson[i, j], spearman[i, j]) for i, j in PAIRS]

joint = iman_conover(sample, TARGET, 1)
Y = np.array(joint.draw_matrix())
print("marginals unchanged:", all(np.array_equal(np.sort(X[:, j]), np.sort(Y[:, j])) for j in range(3)))
```

```text
marginals unchanged: True
```

The achieved correlations, against the monograph's Table 8.3:

```python
BOOK_83 = [(-0.271, -0.287), (0.003, 0.003), (0.772, 0.779)]  # (linear, rank)
table_83 = pd.DataFrame(
    [{"pair": name, "target": TARGET[i][j], "pearson": pe, "book_pearson": bp,
      "spearman": sp, "book_spearman": bs}
     for name, (i, j), (pe, sp), (bp, bs) in zip(NAMES, PAIRS, pairs(Y), BOOK_83)]
)
(GT(table_83, rowname_col="pair")
 .tab_header("Table 8.3: achieved correlation", "Of the reordered sample, against the target")
 .cols_label(pearson="Pearson", book_pearson="Pearson, book", spearman="Spearman",
             book_spearman="Spearman, book")
 .fmt_number(["target"], decimals=1)
 .fmt_number(["pearson", "book_pearson", "spearman", "book_spearman"], decimals=3))
```

```text
               target  Pearson  Pearson, book  Spearman  Spearman, book
Auto-GL          -0.3   -0.278         -0.271    -0.270          -0.287
Auto-Property     0.0   -0.002          0.003     0.003           0.003
GL-Property       0.8    0.773          0.772     0.784           0.779
```

As in the monograph, Auto-GL and GL-Property land a little inside the target.
To see how far a sample of 1,000 moves the digits, repeat with 100 samples and
shuffles:

```python
runs = []
for s in range(100):
    draw = simulate(GaussianCopula(IDENTITY), grids, 1000, 1000 + s, keys=KEYS, dims=["unit"])
    runs.append(pairs(np.array(iman_conover(draw, TARGET, s).draw_matrix())))
runs = np.array(runs)  # run, pair, (Pearson, Spearman)

def flat(x):
    """Round to 3 places and turn -0.0 into 0.0."""
    return round(float(x), 3) + 0.0

spread_rows = []
for k, (name, (i, j)) in enumerate(zip(NAMES, PAIRS)):
    pe, sp = runs[:, k, 0], runs[:, k, 1]
    spread_rows.append({
        "pair": name, "p5": flat(np.quantile(pe, 0.05)), "pm": flat(pe.mean()), "p95": flat(np.quantile(pe, 0.95)),
        "s5": flat(np.quantile(sp, 0.05)), "sm": flat(sp.mean()), "s95": flat(np.quantile(sp, 0.95)),
        "gauss": flat(6 / np.pi * np.arcsin(TARGET[i][j] / 2)),
    })
(GT(pd.DataFrame(spread_rows), rowname_col="pair")
 .tab_header("Achieved correlation over 100 samples", "5%, mean and 95% quantiles; Gaussian Spearman = (6/pi) asin(r/2)")
 .cols_label(p5="Pearson 5%", pm="Pearson mean", p95="Pearson 95%", s5="Spearman 5%", sm="Spearman mean",
             s95="Spearman 95%", gauss="Gaussian Spearman")
 .fmt_number(["p5", "pm", "p95", "s5", "sm", "s95", "gauss"], decimals=3))
```

```text
               Pearson 5%  Pearson mean  Pearson 95%  Spearman 5%  Spearman mean  Spearman 95%  Gaussian Spearman
Auto-GL            -0.286        -0.269       -0.252       -0.298         -0.286        -0.271             -0.288
Auto-Property      -0.020         0.000        0.021       -0.012          0.000         0.016              0.000
GL-Property         0.752         0.773        0.791        0.778          0.787         0.795              0.786
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for k, name in enumerate(NAMES):
    i, j = PAIRS[k]
    ax.plot([TARGET[i][j]] * 2, [k - 0.3, k + 0.3], color="0.75", lw=3, label="target (Pearson)" if k == 0 else None)
    ax.plot([6 / np.pi * np.arcsin(TARGET[i][j] / 2)] * 2, [k - 0.3, k + 0.3], color="0.4", lw=1, ls="--",
            label="Gaussian Spearman of the target" if k == 0 else None)
    for off, kind, label in ((-0.1, 0, "Pearson"), (0.1, 1, "Spearman")):
        v = runs[:, k, kind]
        ax.hlines(k + off, np.quantile(v, 0.05), np.quantile(v, 0.95), color=f"C{kind}", lw=2,
                  label=label if k == 0 else None)
        ax.plot(v.mean(), k + off, "o", color=f"C{kind}")
ax.set_yticks(range(3), NAMES)
ax.invert_yaxis()
ax.set_xlabel("achieved correlation (mean and 90% range of 100 samples)")
ax.set_title("Achieved against target correlation")
ax.legend()
```

Every value of Table 8.3 is inside these 90% ranges. Two results are not
sampling error:

- **The rank correlation is the Gaussian rank correlation of the target.**
  Iman-Conover copies the ranks of reference scores that have exactly the
  target correlation, so Spearman's rho is near `(6/pi) asin(r/2)`, the rank
  correlation of a bivariate normal with correlation `r`.
- **The linear correlation is less than the target.** Pearson's correlation
  depends on the marginals: skewed marginals cannot reach a high linear
  correlation even when their ranks agree. The maximum is the correlation of
  the two samples sorted together:

```python
def bounds(a, b):
    """The largest and smallest Pearson correlation any pairing of a and b can have."""
    return np.corrcoef(np.sort(a), np.sort(b))[0, 1], np.corrcoef(np.sort(a), np.sort(b)[::-1])[0, 1]

hi, lo = bounds(X[:, 1], X[:, 2])
print(f"GL-Property: at most {hi:.3f}, at least {lo:.3f}")
hi, lo = bounds(X[:, 0], X[:, 1])
print(f"Auto-GL:     at most {hi:.3f}, at least {lo:.3f}")
```

```text
GL-Property: at most 0.997, at least -0.779
Auto-GL:     at most 0.963, at least -0.890
```

So a request for a very high Pearson correlation can disappoint. Set the input
a little above the output you need, or use the rank correlation, which
Iman-Conover controls directly.

## 4. The output's row order

Item 6 of section 8.2.5 warns that some implementations return an
Iman-Conover sample in a systematic row order. prospicio does: the first column
of its reference holds the scores in increasing order, so the first unit comes
out sorted:

```python
print("Auto sorted ascending:", bool(np.all(np.diff(Y[:, 0]) >= 0)))
print(f"mean of the first 100 years: Auto {Y[:100, 0].mean():7.1f}  (all 1,000: {Y[:, 0].mean():7.1f})")
u = np.array(GaussianCopula(IDENTITY).sample(1000, 99))
shuffled = Y[np.argsort(u[:, 0])]
print(f"after a random row order:     Auto {shuffled[:100, 0].mean():7.1f}")
```

```text
Auto sorted ascending: True
mean of the first 100 years: Auto  2965.4  (all 1,000:  4998.8)
after a random row order:     Auto  4856.5
```

So the first 100 rows are not a random sample. Statistics over all rows
(means, quantiles, a `Portfolio` built from them) do not depend on the row
order. Shuffle the rows, as above, before you take a subset or pair them with
another simulation.

## 5. The algorithm by hand

Section 8.2.8 derives the method in four steps:

1. Take `n` scores with mean 0 and variance 1. The monograph prefers the
   evenly spaced normal quantiles `Phi^-1(i / (n + 1))`.
2. Put an independently permuted copy of the scores in each of `m` columns to
   make `M`. Then `M'M / n` is close to the identity.
3. Let `S = C'C` be the Cholesky factorization of the target, and `M'M / n =
   F'F` that of the actual correlation of `M`. Then `T = M F^-1 C` has
   correlation exactly `S`.
4. Give each column of the input the rank order of the matching column of `T`.

```python
from statistics import NormalDist

def iman_conover_by_hand(x, target, scores, seed):
    n, m = x.shape
    scores = (scores - scores.mean()) / np.sqrt(np.mean((scores - scores.mean()) ** 2))
    u = np.array(GaussianCopula(np.eye(m).tolist()).sample(n, seed))
    M = np.column_stack([scores[np.argsort(u[:, j])] for j in range(m)])
    F = np.linalg.cholesky(M.T @ M / n).T  # upper triangular, M'M/n = F'F
    C = np.linalg.cholesky(np.array(target)).T
    T = M @ np.linalg.inv(F) @ C
    out = np.column_stack([np.sort(x[:, j])[ranks(T[:, j])] for j in range(m)])
    return out, M, T

n = 1000
normal_scores = np.array([NormalDist().inv_cdf(i / (n + 1)) for i in range(1, n + 1)])
Z, M, T = iman_conover_by_hand(X, TARGET, normal_scores, 3)
print(f"largest error of M'M/n against the identity: {np.abs(M.T @ M / n - np.eye(3)).max():.3f}")
print("T'T/n equals the target to 1e-12:", bool(np.abs(T.T @ T / n - np.array(TARGET)).max() < 1e-12))
```

```text
largest error of M'M/n against the identity: 0.035
T'T/n equals the target to 1e-12: True
```

The Spearman correlations of the by-hand result and of prospicio's:

```python
by_hand = pd.DataFrame(
    {"pair": NAMES, "hand": [s for _, s in pairs(Z)], "lib": [s for _, s in pairs(Y)]}
)
(GT(by_hand, rowname_col="pair")
 .tab_header("Spearman correlation of the reordered sample", "By hand and by prospicio")
 .cols_label(hand="by hand", lib="prospicio")
 .fmt_number(["hand", "lib"], decimals=3))
```

```text
               by hand  prospicio
Auto-GL         -0.281     -0.270
Auto-Property   -0.002      0.003
GL-Property      0.788      0.784
```

The `F^-1` step removes the few hundredths of error that the permutations
leave in the correlation of `M`. The results differ because the permutations
differ.

## 6. Other reference distributions and copulas

Section 8.2.3 replaces the normal scores with Student t scores, which put more
weight on joint extremes. `iman_conover_by_hand` takes any scores. The
monograph's extreme example is t with 2 degrees of freedom, with quantile
`(2u - 1) / sqrt(2 u (1 - u))`. Count the years in which GL and Property are
both in their worst 1%, out of 10,000, under each reference:

```python
n = 10_000
draws = np.array(simulate(GaussianCopula(IDENTITY), grids, n, 4, keys=KEYS, dims=["unit"]).draw_matrix())
uu = np.arange(1, n + 1) / (n + 1)
SCORES = {
    "normal": np.array([NormalDist().inv_cdf(v) for v in uu]),
    "t, 2 dof": (2 * uu - 1) / np.sqrt(2 * uu * (1 - uu)),
}
for name, scores in SCORES.items():
    Z, _, _ = iman_conover_by_hand(draws, TARGET, scores, 5)
    worst = (ranks(Z[:, 1]) >= 0.99 * n) & (ranks(Z[:, 2]) >= 0.99 * n)
    print(f"{name:9s} GL-Property Spearman {pairs(Z)[2][1]:.3f}, years in both worst 1%: {worst.sum()}")
print("if independent, about", n // 10_000)
```

```text
normal    GL-Property Spearman 0.787, years in both worst 1%: 34
t, 2 dof  GL-Property Spearman 0.714, years in both worst 1%: 68
if independent, about 1
```

With t scores the joint extremes are about twice as frequent although the rank
correlation is lower (the pinched scatter of Figure 8.3). The monograph warns
that this flexibility is not a reason to use it.

Section 8.2.4 asks if a copula can be the reference. A multivariate
Archimedean copula has a single parameter and cannot match a correlation
matrix. Instead, simulate directly with a prospicio copula (`simulate` with a
`GaussianCopula`, `StudentTCopula`, `ArchimedeanCopula`,
`NestedArchimedeanCopula` or `VineCopula`). Nested Archimedean copulas and
vines avoid the single parameter: a vine uses one bivariate copula per edge, so
each pair has its own strength and tail.

## 7. Iman-Conover against the normal copula

The normal copula method (section 8.2.5) draws correlated normals and maps
each to its unit's distribution through the quantile function. With normal
scores, Iman-Conover does the same to an existing sample and corrects the
reference to the exact correlation. Both methods, 100 times on the same units:

```python
normal_copula = GaussianCopula(TARGET)
spread = {"Iman-Conover": [], "normal copula": []}
for s in range(100):
    draw = simulate(GaussianCopula(IDENTITY), grids, 1000, 3000 + s, keys=KEYS, dims=["unit"])
    spread["Iman-Conover"].append(pairs(np.array(iman_conover(draw, TARGET, s).draw_matrix()))[2][1])
    copula_draw = simulate(normal_copula, grids, 1000, 5000 + s, keys=KEYS, dims=["unit"])
    spread["normal copula"].append(pairs(np.array(copula_draw.draw_matrix()))[2][1])
for name, values in spread.items():
    print(f"{name:14s} GL-Property Spearman: mean {np.mean(values):.3f}, sd {np.std(values):.4f}")
```

```text
Iman-Conover   GL-Property Spearman: mean 0.787, sd 0.0052
normal copula  GL-Property Spearman: mean 0.787, sd 0.0142
```

```python
fig, ax = plt.subplots()
bins = np.linspace(0.74, 0.83, 28)
for name, values in spread.items():
    ax.hist(values, bins=bins, alpha=0.6, label=name)
ax.axvline(6 / np.pi * np.arcsin(0.4), color="0.75", lw=2)
ax.set_xlabel("GL-Property Spearman correlation of one sample (grey line: 0.786)")
ax.set_ylabel("samples out of 100")
ax.set_title("Iman-Conover varies less than the normal copula")
ax.legend()
```

Both methods centre on 0.786, the Gaussian rank correlation. Iman-Conover
varies much less, because only the small gap between the rank and linear
correlations of the normal reference remains. But the normal copula makes each
year independently, so you can split, subset or extend its output freely. An
Iman-Conover sample is made in one piece and needs the care of section 4.

## 8. Example 2: block Iman-Conover

A large correlation matrix estimated from data is mostly noise. Block
Iman-Conover (section 8.2.6) uses the correlations that data can estimate:
between lines within each region, and between the region totals.

Example 2 (section 8.2.7) has four regions, each with Auto (AL), GL and
Property (Pr). Each line has the model of Example 1, except that the GL
severity CV is 4. The first step computes the region totals with independent
lines, by FFT (Table 8.4):

```python
from prospicio.pricing import Portfolio

LINES = {"AL": (2000, 30, 2, 0.2), "GL": (5000, 50, 4, 0.3), "Pr": (10000, 100, 5, 0.1)}
REGIONS = {"North": (4000, 2000, 2000), "East": (3000, 1000, 3000),
           "South": (5000, 2000, 7000), "West": (5000, 2000, 2000)}
BOOK_84 = {"North": (8000, 0.298, 16237), "East": (7000, 0.347, 15920),
           "South": (14000, 0.266, 25737), "West": (9000, 0.276, 17331)}
region_grids = {r: [unit(loss, *LINES[line]) for loss, line in zip(losses, LINES)]
                for r, losses in REGIONS.items()}
rows = []
for r, gs in region_grids.items():
    total = Portfolio.from_independent(list(LINES), gs)
    xs, ps = np.array(total.totals), np.array(total.probs)
    mean = (xs * ps).sum()
    cv = np.sqrt(((xs - mean) ** 2 * ps).sum()) / mean
    rows.append({"region": r, "mean": mean, "cv": cv, "p99": total.assets(0.99), "book_mean": BOOK_84[r][0],
                 "book_cv": BOOK_84[r][1], "book_p99": BOOK_84[r][2]})
(GT(pd.DataFrame(rows), rowname_col="region")
 .tab_header("Table 8.4: region totals with independent lines", "Annual loss")
 .cols_label(p99="99th pct", book_mean="book mean", book_cv="book cv", book_p99="book 99th pct")
 .fmt_number(["mean"], decimals=1, use_seps=False)
 .fmt_number(["cv", "book_cv"], decimals=3)
 .fmt_number(["p99", "book_p99", "book_mean"], decimals=0, use_seps=False))
```

```text
          mean     cv  99th pct  book mean  book cv  book 99th pct
North   8000.0  0.298     16236       8000    0.298          16237
East    7000.0  0.347     15920       7000    0.347          15920
South  14000.0  0.266     25737      14000    0.266          25737
West    9000.0  0.276     17330       9000    0.276          17331
```

The means and CVs agree to the printed digits. The 99th percentiles differ by
at most 1.5 in 16,000 to 26,000, from the discretization step: a step of 0.5
gives 16,236.5, 15,920.5, 25,737 and 17,329.5.

The block method has three steps:

1. Correlate the lines within each region by ordinary Iman-Conover. Here every
   region has the same 3 x 3 matrix (Table 8.6 (e)), but each region can have
   its own matrix.
2. Stack the regions into one `PredictiveDistribution` with a `region`
   dimension.
3. Correlate the region totals: move the years of each region as whole rows,
   which keeps the structure within the region.

`reorder_groups` does the last step:

```python
INTRA = [[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]]
INTER = [[1.0, 0.7, 0.2, 0.0], [0.7, 1.0, 0.4, 0.2], [0.2, 0.4, 1.0, 0.8], [0.0, 0.2, 0.8, 1.0]]
parts = []
for k, (r, gs) in enumerate(region_grids.items()):
    draw = simulate(GaussianCopula(IDENTITY), gs, 1000, 100 + k,
                    keys=[(line,) for line in LINES], dims=["line"])
    parts.append((r, iman_conover(draw, INTRA, k)))
stacked = PredictiveDistribution.join(parts, "region")
bic = stacked.reorder_groups("region", INTER, 7)
print(bic.dims, bic.components()[:4], "...")
```

```text
['region', 'line'] [('North', 'AL'), ('North', 'GL'), ('North', 'Pr'), ('East', 'AL')] ...
```

The correlations between the region totals (Table 8.5):

```python
B = np.array(bic.draw_matrix())
totals = B.reshape(1000, 4, 3).sum(axis=2)
achieved = np.corrcoef(totals.T)
BOOK_85 = {(0, 1): 0.683, (0, 2): 0.167, (0, 3): -0.002, (1, 2): 0.369, (1, 3): 0.197, (2, 3): 0.799}
names = list(REGIONS)
table_85 = pd.DataFrame(
    [{"pair": f"{names[i]}-{names[j]}", "target": INTER[i][j], "achieved": achieved[i, j], "book": book}
     for (i, j), book in BOOK_85.items()]
)
(GT(table_85, rowname_col="pair")
 .tab_header("Table 8.5: correlation of the region totals", "Target, achieved and the monograph's sample")
 .fmt_number(["target"], decimals=1)
 .fmt_number(["achieved", "book"], decimals=3))
```

```text
             target  achieved    book
North-East      0.7     0.684   0.683
North-South     0.2     0.199   0.167
North-West      0.0     0.004  -0.002
East-South      0.4     0.376   0.369
East-West       0.2     0.187   0.197
South-West      0.8     0.786   0.799
```

These are close to the target and to the monograph's values. Within a region
the lines keep the first step's correlation, because the reordering moves the
three lines together (Table 8.6):

```python
C12 = np.corrcoef(B.T)
BOOK_86 = {"North": (0.579, 0.180, 0.332), "East": (0.544, 0.191, 0.318),
           "South": (0.571, 0.202, 0.354), "West": (0.580, 0.176, 0.330)}
rows = []
for k, r in enumerate(names):
    block = C12[3 * k:3 * k + 3, 3 * k:3 * k + 3]
    rows.append({"region": r, "al_gl": block[0, 1], "al_pr": block[0, 2], "gl_pr": block[1, 2],
                 "b_al_gl": BOOK_86[r][0], "b_al_pr": BOOK_86[r][1], "b_gl_pr": BOOK_86[r][2]})
(GT(pd.DataFrame(rows), rowname_col="region")
 .tab_header("Table 8.6: correlation between lines within a region", "The target is 0.6, 0.2 and 0.4")
 .cols_label(al_gl="AL-GL", al_pr="AL-Pr", gl_pr="GL-Pr", b_al_gl="AL-GL, book", b_al_pr="AL-Pr, book",
             b_gl_pr="GL-Pr, book")
 .fmt_number(["al_gl", "al_pr", "gl_pr", "b_al_gl", "b_al_pr", "b_gl_pr"], decimals=3))
```

```text
       AL-GL  AL-Pr  GL-Pr  AL-GL, book  AL-Pr, book  GL-Pr, book
North  0.567  0.165  0.299        0.579        0.180        0.332
East   0.512  0.192  0.331        0.544        0.191        0.318
South  0.576  0.213  0.385        0.571        0.202        0.354
West   0.557  0.173  0.317        0.580        0.176        0.330
```

The full 12 x 12 matrix (Table 8.7) has the block structure of Figure 8.4.
Here is the North AL row against the lines of the other regions:

```python
BOOK_87 = [0.32, 0.29, 0.42, 0.07, 0.04, 0.12, -0.01, -0.07, 0.01]
table_87 = pd.DataFrame(
    {"line": [f"{r} {line}" for r in names[1:] for line in LINES], "mine": C12[0, 3:], "book": BOOK_87}
)
(GT(table_87, rowname_col="line")
 .tab_header("Table 8.7: North AL with the lines of the other regions", "Correlation of annual loss")
 .cols_label(mine="prospicio")
 .fmt_number(["mine", "book"], decimals=2))
```

```text
          prospicio   book
East AL        0.40   0.32
East GL        0.33   0.29
East Pr        0.40   0.42
South AL       0.11   0.07
South GL       0.14   0.04
South Pr       0.12   0.12
West AL       -0.04  -0.01
West GL        0.01  -0.07
West Pr        0.05   0.01
```

Lines in correlated regions are correlated across regions (North and East at
0.7); lines in uncorrelated regions (North and West) are not. Nine numbers set
all 66 correlations.

## 9. Capital by layer, if you must

Section 8.3 returns to InsCo (ten equally likely years, three units) and its
Wang distortion calibrated to a 15% return with assets of 100. The assets
divide into layers between consecutive distinct totals. In a layer, `alpha` is
each unit's expected share of the total in the years that reach the layer, and
`beta` shares the premium the same way under the distorted probabilities. The
layer's margin is its premium less its loss, and its capital is the part of
its width that premium does not fund. The cost of capital in a layer,
`iota = (g(S) - S) / (1 - g(S))`, is the same for every unit, so a unit's
capital there is its margin divided by `iota` (equation 8.5).

```python
from prospicio.risk import Distortion

ROWS = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
        [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]
insco = Portfolio(["A", "B", "C"], ROWS)
wang = insco.calibrate("wang", assets=100, return_on_capital=0.15)

def layers(port, g):
    """Per layer: bottom, width, S, g(S), iota, and per unit alpha S, beta g(S),
    margin and capital, all per unit of width."""
    x, p = [0.0] + port.totals, port.probs
    share = {u: [k / t for k, t in zip(port.kappa(u), port.totals)] for u in port.units}
    S = [1.0] + [max(0.0, 1.0 - sum(p[:k])) for k in range(1, len(p) + 1)]
    G = [g.g(s) for s in S]
    q = [G[j] - G[j + 1] for j in range(len(p))]
    out = []
    for k in range(len(p)):
        aS = [sum(p[j] * share[u][j] for j in range(k, len(p))) for u in port.units]
        bG = [sum(q[j] * share[u][j] for j in range(k, len(p))) for u in port.units]
        margin = [b - a for a, b in zip(aS, bG)]
        iota = (G[k] - S[k]) / (1 - G[k]) if G[k] < 1 else None
        capital = [m / iota if iota else 0.0 for m in margin]
        out.append((x[k], x[k + 1] - x[k], S[k], G[k], iota, aS, bG, margin, capital))
    return out

wang_layers = pd.DataFrame(
    [
        {"layer": f"{x:.0f} to {x + dx:.0f}", "S": s, "gS": g, "iota": iota,
         **{f"margin {u}": m for u, m in zip("ABC", margin)},
         **{f"capital {u}": q for u, q in zip("ABC", capital)}}
        for x, dx, s, g, iota, _, _, margin, capital in layers(insco, wang)
    ]
)
(GT(wang_layers, rowname_col="layer")
 .tab_header("Table 8.10: InsCo's layers under the Wang distortion",
             "Margin and capital per unit of layer width; the bottom layer has no iota")
 .tab_spanner("margin", ["margin A", "margin B", "margin C"])
 .tab_spanner("capital", ["capital A", "capital B", "capital C"])
 .cols_label(gS="g(S)", **{f"{k} {u}": u for k in ("margin", "capital") for u in "ABC"})
 .fmt_number("S", decimals=1)
 .fmt_number(["gS", "iota", "margin A", "margin B", "margin C", "capital A", "capital B", "capital C"],
             decimals=3)
 .sub_missing(missing_text="-"))
```

```text
             S   g(S)   iota       A       B      C      A       B      C
0 to 22    1.0  1.000      -  -0.030  -0.032  0.061  0.000   0.000  0.000
22 to 28   0.9  0.948  0.917   0.003  -0.017  0.061  0.003  -0.018  0.067
28 to 36   0.8  0.882  0.693   0.021  -0.001  0.061  0.031  -0.001  0.089
36 to 40   0.7  0.807  0.555   0.025   0.013  0.069  0.045   0.024  0.125
40 to 55   0.3  0.428  0.224   0.030   0.026  0.072  0.134   0.115  0.323
55 to 65   0.2  0.309  0.158   0.021   0.019  0.069  0.133   0.121  0.437
65 to 100  0.1  0.174  0.089   0.012   0.015  0.047  0.132   0.165  0.529
```

The first three columns are the margin of A, B and C and the last three their
capital. These values agree with Table 8.10 (and `g(S)` with Table 8.9) to the printed
digits. The code keeps `alpha S` and `beta g(S)`, the columns of Tables 8.8 and
8.9, for section 10. Every year pays for the bottom layer, so it needs no
capital and `iota` is undefined; its unit margins sum to zero (cross-subsidies).
Summing each column times the layer width gives each unit's margin and
capital:

```python
def by_unit(port, g):
    rows = layers(port, g)
    margin = [sum(r[1] * r[7][i] for r in rows) for i in range(3)]
    capital = [sum(r[1] * r[8][i] for r in rows) for i in range(3)]
    return margin, capital

margin, capital = by_unit(insco, wang)
lifted = [u.capital for u in insco.price(wang, assets=100, allocation="lifted").allocated]
BOOK_MARGIN, BOOK_CAPITAL, BOOK_RETURN = [0.709, 0.337, 5.919], [8.411, 8.691, 29.333], [0.084, 0.039, 0.202]
by_unit_table = pd.DataFrame(
    [{"unit": u, "margin": m, "book_margin": bm, "capital": q, "book_capital": bq, "lifted": lq,
      "ret": m / q, "book_ret": br}
     for u, m, bm, q, bq, lq, br in zip("ABC", margin, BOOK_MARGIN, capital, BOOK_CAPITAL, lifted, BOOK_RETURN)]
    + [{"unit": "total", "margin": sum(margin), "book_margin": None, "capital": sum(capital),
        "book_capital": None, "lifted": sum(lifted), "ret": sum(margin) / sum(capital), "book_ret": 0.150}]
)
(GT(by_unit_table, rowname_col="unit")
 .tab_header("Margin and capital by unit", "Wang distortion, assets of 100; lifted = Portfolio.price(..., allocation=\"lifted\")")
 .cols_label(book_margin="book margin", book_capital="book capital", lifted="lifted capital",
             ret="return", book_ret="book return")
 .fmt_number(["margin", "book_margin", "capital", "book_capital", "lifted"], decimals=3, use_seps=False)
 .fmt_percent(["ret", "book_ret"], decimals=1)
 .sub_missing(missing_text=""))
```

```text
       margin  book margin  capital  book capital  lifted capital  return  book return
A       0.709        0.709    8.411         8.411           8.411    8.4%         8.4%
B       0.337        0.337    8.691         8.691           8.691    3.9%         3.9%
C       5.919        5.919   29.333        29.333          29.333   20.2%        20.2%
total   6.965                46.435                        46.435   15.0%        15.0%
```

The returns differ by unit although every layer pays every unit the same
return, because the units use different layers. Low layers are expensive but
mostly funded by premium. High layers are cheap but mostly capital.

For the CCoC and the proportional hazard, the numbers (Table 8.11) depend on
the treatment of the bottom layer. Equation 8.5 as written gives it no capital.
prospicio, like `aggregate` 1.0.1, converts its margins at the limit of `iota`
as `S` rises to 1: `r = 0.15` for the CCoC and `(1 - rho) / rho` for the
proportional hazard. For the Wang, dual power and TVaR distortions the limit is
infinite and the conventions agree. The monograph's printed capital comes from
an earlier `aggregate` and matches neither convention:

```python
BOOK_Q = {"ccoc": (4.874, 5.288, 36.273), "ph": (7.074, 6.565, 32.796)}
rows = []
for family in ("ccoc", "ph"):
    d = insco.calibrate(family, assets=100, return_on_capital=0.15)
    _, eq85 = by_unit(insco, d)
    lifted = [u.capital for u in insco.price(d, assets=100, allocation="lifted").allocated]
    for k, u in enumerate("ABC"):
        rows.append({"case": f"{family}, {u}", "eq85": eq85[k], "lifted": lifted[k], "book": BOOK_Q[family][k]})
(GT(pd.DataFrame(rows), rowname_col="case")
 .tab_header("Table 8.11: capital by unit under two conventions", "Assets of 100, 15% return")
 .cols_label(eq85="equation 8.5", lifted="prospicio")
 .fmt_number(["eq85", "lifted", "book"], decimals=3, use_seps=False))
```

```text
         equation 8.5  prospicio    book
ccoc, A         5.419      2.261   4.874
ccoc, B         6.082      1.478   5.288
ccoc, C        34.934     42.696  36.273
ph, A           7.765      6.481   7.074
ph, B           7.694      5.597   6.565
ph, C          30.975     34.356  32.796
```

All three conventions sum to the same total capital of 46.435 and differ only
in how they spread the bottom layer's cross-subsidies. Under the prospicio
convention every unit earns exactly 15% under the CCoC, which is what a
constant cost of capital means (see section 5 of the
[chapter 9 guide](capital-modeling-monograph-15.md)). The monograph advises
against any allocation of capital, and this result is one more reason.

## 10. Bodoff's layer allocation

Section 8.4 compares the natural allocation with the allocation of Bodoff
(2007). Bodoff shares every layer of the assets by `alpha`, not only its
expected loss (Table 8.12):

```python
bodoff = insco.bodoff(assets=100)
first = layers(insco, wang)[0]
layer_share = [first[1] * a / first[2] for a in first[5]]
BOOK_BODOFF, BOOK_FIRST = [24.159, 29.756, 46.086], [7.152, 9.694, 5.154]
bodoff_table = pd.DataFrame(
    [{"unit": u, "assets": a, "book": ba, "first": f, "book_first": bf}
     for u, a, ba, f, bf in zip("ABC", bodoff, BOOK_BODOFF, layer_share, BOOK_FIRST)]
    + [{"unit": "total", "assets": sum(bodoff), "book": 100.0, "first": None, "book_first": None}]
)
(GT(bodoff_table, rowname_col="unit")
 .tab_header("Table 8.12: Bodoff's allocation of the assets", "Assets of 100; the first layer is 0 to 22")
 .cols_label(assets="Bodoff assets", book="book", first="layer 0 to 22", book_first="layer 0 to 22, book")
 .fmt_number(["assets", "book", "first", "book_first"], decimals=3, use_seps=False)
 .sub_missing(missing_text=""))
```

```text
       Bodoff assets     book  layer 0 to 22  layer 0 to 22, book
A             24.159   24.159          7.152                7.152
B             29.756   29.756          9.694                9.694
C             46.086   46.086          5.154                5.154
total        100.000  100.000                                    
```

Bodoff's allocation splits the assets but not the premium. The monograph adds
a CCoC: in each layer the premium is the expected loss plus a margin of
`d = r / (1 + r)` on the unfunded part, `alpha (S + d (1 - S))`, the CCoC
`g(S)` shared by the loss shares:

```python
ccoc = Distortion.ccoc(0.15)
bodoff_premium = [sum(dx * g * aS[i] / s for _, dx, s, g, _, aS, _, _, _ in layers(insco, ccoc) if s > 0)
                  for i in range(3)]
natural = [u.premium for u in insco.price(ccoc, assets=100).allocated]
premiums = pd.DataFrame(
    {"unit": ["A", "B", "C", "total"], "bodoff": bodoff_premium + [sum(bodoff_premium)],
     "natural": natural + [sum(natural)]}
)
(GT(premiums, rowname_col="unit")
 .tab_header("Premium by unit under the CCoC", "Assets of 100")
 .cols_label(bodoff="Bodoff with CCoC", natural="natural CCoC")
 .fmt_number(["bodoff", "natural"], decimals=3, use_seps=False))
```

```text
       Bodoff with CCoC  natural CCoC
A                14.803        13.739
B                19.794        18.522
C                18.968        21.304
total            53.565        53.565
```

```python
fig, ax = plt.subplots()
width = 0.38
at = np.arange(3)
ax.bar(at - width / 2, bodoff_premium, width, label="Bodoff with CCoC")
ax.bar(at + width / 2, natural, width, label="natural CCoC")
ax.set_xticks(at, ["A", "B", "C"])
ax.set_xlabel("unit")
ax.set_ylabel("premium (assets of 100)")
ax.set_title("Premium by unit: the same total of 53.565")
ax.legend()
```

The totals are the same, but the splits differ. Bodoff's shares are the
expected loss shares in every layer. The natural allocation uses the distorted
shares, which move premium toward unit C, which dominates the years the
distortion weights more. Another distortion gives other splits. So Bodoff's
method needs one more assumption before it can price anything.
