# Iman-Conover, uncertainty and capital by layer: CAS Monograph 15, chapter 8

Chapter 8 of CAS Monograph 15 (John Major and Stephen Mildenhall, *Introduction
to Capital Modeling and Portfolio Management*, 2026) collects four advanced
topics: parameter uncertainty, the Iman-Conover method for correlating
simulated units, capital allocation layer by layer, and Bodoff's layer
allocation. This guide runs each of them in prospicio, with the monograph's
printed numbers beside prospicio's.

Two kinds of number appear in the chapter. Some are exact, such as the
regional means and percentiles of Example 2 (computed by FFT) and every table
of InsCo in sections 8.3 and 8.4; those should agree with prospicio to the
digits printed. Others come from one random sample of 1,000 simulations,
such as the achieved correlations of Examples 1 and 2; prospicio draws its own
sample, so those can only agree within sampling error, and the guide shows how
large that error is. The explanations are in new words; read the monograph for
the argument. Table numbers are the monograph's.

Companion guides: [chapter 9](capital-modeling-monograph-15.md),
[chapters 3 and 4](capital-modeling-monograph-15-theory.md),
[chapters 5 and 6](capital-modeling-monograph-15-applications.md) and
[chapters 1, 2 and 7](capital-modeling-monograph-15-practice.md).

## 1. Parameter uncertainty

Section 8.1 separates process risk (the randomness of a loss whose
distribution is known) from parameter uncertainty (not knowing the
distribution), and lists four responses: ignore it, report a range, absorb it
into a predictive distribution, or isolate a credible worst case. Take a
lognormal severity with a shape `sigma` of 1.2 and an expected value of about
100, whose log-scale location is known only to within 0.3 either way, and
see what each response does to its 99.5% quantile:

```python
import math

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


for d in severities:
    print(f"report it:   mean {d.mean():5.1f}, 99.5% quantile {d.quantile(0.995):6.1f}")
print(f"ignore it:                     99.5% quantile {severities[1].quantile(0.995):6.1f}")
print(f"absorb it:                     99.5% quantile {mixture_quantile(severities, 0.995):6.1f}")
print(f"isolate it:                    99.5% quantile {severities[2].quantile(0.995):6.1f}")
```

```text
report it:   mean  74.1, 99.5% quantile  793.3
report it:   mean 100.0, 99.5% quantile 1070.8
report it:   mean 135.0, 99.5% quantile 1445.4
ignore it:                     99.5% quantile 1070.8
absorb it:                     99.5% quantile 1139.9
isolate it:                    99.5% quantile 1445.4
```

Reporting gives a range, here from about 790 to 1,450. Absorbing the
uncertainty, by weighting the three locations equally, gives a single answer
about 6% above the best estimate's: the quantile of a mixture is pulled up by
its wider spread, which is the downward bias of a quantile that ignores
parameter uncertainty that the monograph mentions. Isolating a worst case
gives the highest answer. In prospicio the same mixing of whole simulations
is `PredictiveDistribution.blend`, which takes each simulation from one of
several models with given weights, and the Bayesian models (for example
`BayesOdp` in reserving) return predictive distributions with the parameter
uncertainty already in them.

## 2. Example 1: three units

Section 8.2.2 correlates three units, Auto, GL and Property. Each has a
lognormal severity limited per occurrence and a gamma-mixed Poisson count:

| Unit | Expected loss | Limit | Severity mean | Severity CV | Mixing CV |
|---|---|---|---|---|---|
| Auto | 5,000 | 2,000 | 30 | 2 | 0.2 |
| GL | 2,000 | 5,000 | 50 | 5 | 0.3 |
| Property | 7,000 | 10,000 | 100 | 5 | 0.1 |

prospicio compounds each unit by FFT, then draws 1,000 independent years
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
print(f"{'':9s} {'FFT mean':>9s} {'FFT cv':>7s} {'sample mean':>12s} {'cv':>6s}   book sample")
for j, (name, g) in enumerate(zip(UNITS, grids)):
    sd = X[:, j].std(ddof=1)
    print(f"{name:9s} {g.mean():9.1f} {g.variance() ** 0.5 / g.mean():7.3f} "
          f"{X[:, j].mean():12.1f} {sd / X[:, j].mean():6.3f}   {BOOK_81[name]}")
```

```text
           FFT mean  FFT cv  sample mean     cv   book sample
Auto         5000.0   0.262       4998.8  0.262   (5011.7, 0.257)
GL           2000.0   0.672       2042.0  0.644   (2092.4, 0.66)
Property     7000.0   0.465       7017.0  0.454   (7117.8, 0.467)
```

The FFT means are the requested 5,000, 2,000 and 7,000, and the CVs are those
of the model. The monograph's Table 8.1 describes its own sample (means of
5,011.7, 2,092.4 and 7,117.8, CVs of 0.257, 0.660 and 0.467), and prospicio's
sample differs from it by about as much as either differs from the model:
sampling differences of a few percent, not model differences.

## 3. The target and the achieved correlation

The target (Table 8.2) has Auto and GL at -0.3, GL and Property at 0.8, and
Auto and Property uncorrelated. A target must be a correlation matrix, that
is, symmetric and positive definite; prospicio refuses one that is not:

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

`iman_conover` reorders each column of the sample. Every unit keeps exactly
its own 1,000 values; only which values share a year changes. The
correlations achieved are those of Table 8.3:

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
BOOK_83 = [(-0.271, -0.287), (0.003, 0.003), (0.772, 0.779)]  # (linear, rank)
print(f"{'':14s} {'target':>7s} {'Pearson':>8s} {'book':>7s} {'Spearman':>9s} {'book':>7s}")
for name, (i, j), (pe, sp), (bp, bs) in zip(NAMES, PAIRS, pairs(Y), BOOK_83):
    print(f"{name:14s} {TARGET[i][j]:7.1f} {pe:8.3f} {bp:7.3f} {sp:9.3f} {bs:7.3f}")
```

```text
marginals unchanged: True
                target  Pearson    book  Spearman    book
Auto-GL           -0.3   -0.278  -0.271    -0.270  -0.287
Auto-Property      0.0   -0.002   0.003     0.003   0.003
GL-Property        0.8    0.773   0.772     0.784   0.779
```

The pattern is the monograph's: Auto-GL and GL-Property land a little inside
the target and Auto-Property stays near zero. The digits differ because the
sample and the shuffles differ. To see how far a sample of 1,000 moves them,
repeat with 100 different samples and shuffles:

```python
runs = []
for s in range(100):
    draw = simulate(GaussianCopula(IDENTITY), grids, 1000, 1000 + s, keys=KEYS, dims=["unit"])
    runs.append(pairs(np.array(iman_conover(draw, TARGET, s).draw_matrix())))
runs = np.array(runs)  # run, pair, (Pearson, Spearman)
print(f"{'':14s} {'Pearson 5%':>11s} {'mean':>7s} {'95%':>7s} {'Spearman 5%':>12s} {'mean':>7s} {'95%':>7s}")
for k, name in enumerate(NAMES):
    pe, sp = runs[:, k, 0], runs[:, k, 1]
    cells = [np.quantile(pe, 0.05), pe.mean(), np.quantile(pe, 0.95),
             np.quantile(sp, 0.05), sp.mean(), np.quantile(sp, 0.95)]
    cells = [f"{c:.3f}".replace("-0.000", "0.000") for c in cells]
    print(f"{name:14s} {cells[0]:>11s} {cells[1]:>7s} {cells[2]:>7s} {cells[3]:>12s} {cells[4]:>7s} {cells[5]:>7s}")
print("Gaussian Spearman (6/pi) asin(r/2):",
      [round(float(6 / np.pi * np.arcsin(TARGET[i][j] / 2)), 3) for i, j in PAIRS])
```

```text
                Pearson 5%    mean     95%  Spearman 5%    mean     95%
Auto-GL             -0.286  -0.269  -0.252       -0.298  -0.286  -0.271
Auto-Property       -0.020   0.000   0.021       -0.012   0.000   0.016
GL-Property          0.752   0.773   0.791        0.778   0.787   0.795
Gaussian Spearman (6/pi) asin(r/2): [-0.288, 0.0, 0.786]
```

Every value of Table 8.3 lies inside these 90% ranges, so the monograph's
sample and prospicio's agree as far as a sample of 1,000 can tell. Two things
are not sampling error:

- **The rank correlation is the target's Gaussian rank correlation.** prospicio
  gives the reference scores exactly the target correlation, and Iman-Conover
  copies their ranks, so Spearman's rho sits near `(6/pi) asin(r/2)`, the rank
  correlation of a bivariate normal with correlation `r`: -0.288 and 0.786 for
  -0.3 and 0.8. The monograph's -0.287 and 0.779 are within a sample's
  distance of those values.
- **The linear correlation falls short of the target.** Pearson's correlation
  depends on the marginals, and skewed marginals cannot reach a high linear
  correlation even when ranked perfectly together. The ceiling for GL and
  Property is the correlation of the two samples sorted together:

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

The monograph's warning that a request for very high correlation may
disappoint is this ceiling. When Pearson's correlation is what matters, aim
the input a little above the output you need, or work with the rank
correlation, which Iman-Conover controls directly.

## 4. The output's row order

Item 6 of the monograph's comparison in section 8.2.5 warns that an
Iman-Conover sample must be used whole, because some implementations return
the rows in a systematic order. prospicio's does: the first column of its
reference holds the scores in increasing order, so the first unit comes back
sorted:

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

The first 100 rows of the output are therefore not a random sample of 100
years. Statistics over all rows (means, quantiles, a `Portfolio` built from
them) do not depend on the row order and are unaffected. Shuffle the rows, as
above, before taking a subset or pairing the output row by row with another
simulation.

## 5. The algorithm by hand

Section 8.2.8 derives the method. Take `n` scores with mean 0 and variance 1
(the monograph prefers the evenly spaced normal quantiles
`Phi^-1(i / (n + 1))`), put an independently permuted copy in each of `m`
columns to make `M`, and note that `M'M / n` is close to the identity. If
`S = C'C` is the target's Cholesky factorization and `M'M / n = F'F` that of
the actual correlation of `M`, then `T = M F^-1 C` has correlation exactly
`S`. Finally give each column of the input the rank order of the matching
column of `T`. In a dozen lines:

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
print("by hand,   Spearman:", [round(float(s), 3) for _, s in pairs(Z)])
print("prospicio, Spearman:", [round(float(s), 3) for _, s in pairs(Y)])
```

```text
largest error of M'M/n against the identity: 0.035
T'T/n equals the target to 1e-12: True
by hand,   Spearman: [-0.281, -0.002, 0.788]
prospicio, Spearman: [-0.27, 0.003, 0.784]
```

The random permutations alone leave errors of a few hundredths in the
correlation of `M`; the `F^-1` step removes them. The two results are not
equal row by row, since the permutations differ, but both have the
correlation structure of a normal reference with the target correlation.

## 6. Other reference distributions and copulas

Section 8.2.3 replaces the normal scores by Student t scores, which put more
weight on joint extremes. The monograph's `aggregate` takes a `dof` argument
for this; prospicio's `iman_conover` uses normal scores only. The function
above takes any scores, and the t distribution with 2 degrees of freedom,
the monograph's extreme example, has the closed-form quantile
`(2u - 1) / sqrt(2 u (1 - u))`. Compare how often GL and Property are both in
their worst 1% of 10,000 years under each reference:

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

With t scores the joint extremes are about twice as frequent, even though the
overall rank correlation is lower: the pinched scatter of the monograph's Figure 8.3. The
monograph also warns that this flexibility is no reason to use it.

Section 8.2.4 asks whether a copula could serve as the reference. A
multivariate Archimedean copula has a single parameter and cannot match a
correlation matrix, as the monograph says. prospicio's copulas can instead be
used to simulate directly (`simulate` with a `GaussianCopula`,
`StudentTCopula`, `ArchimedeanCopula`, `NestedArchimedeanCopula` or
`VineCopula`). The nested Archimedean copulas and the C- and D-vines are the
ways around the single parameter: a vine builds the joint law from one
bivariate copula per edge, so it can carry a different strength and tail for
each pair.

## 7. Iman-Conover against the normal copula

The normal copula method (section 8.2.5) simulates each year by drawing
correlated normals and transforming each to its unit's distribution through
the quantile function. With normal scores, Iman-Conover does the same to a
sample that already exists, with one difference: it corrects the reference to
the exact correlation. Repeating both 100 times on the same units shows what
that buys:

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

Both centre on the Gaussian rank correlation of about 0.786. The normal copula's
correlation varies from sample to sample as any sample statistic does;
Iman-Conover's varies much less, because only the small gap between the
normal reference's rank and linear correlation is left. In exchange the
normal copula produces one independent year at a time, so its output can be
split, subset or extended freely, while an Iman-Conover sample is made in one
piece from inputs that already exist (and needs the care of section 4).

## 8. Example 2: block Iman-Conover

A large correlation matrix estimated from data is mostly noise. Section 8.2.6's
block Iman-Conover uses what can be estimated: a correlation between lines
within each region and a correlation between the regions' totals. Example 2
(section 8.2.7) has four regions, each with Auto (AL), GL and Property (Pr),
each line the same model as in Example 1 except GL's severity CV of 4. First
the regions' totals with independent lines, by FFT, as in Table 8.4:

```python
from prospicio.pricing import Portfolio

LINES = {"AL": (2000, 30, 2, 0.2), "GL": (5000, 50, 4, 0.3), "Pr": (10000, 100, 5, 0.1)}
REGIONS = {"North": (4000, 2000, 2000), "East": (3000, 1000, 3000),
           "South": (5000, 2000, 7000), "West": (5000, 2000, 2000)}
BOOK_84 = {"North": (8000, 0.298, 16237), "East": (7000, 0.347, 15920),
           "South": (14000, 0.266, 25737), "West": (9000, 0.276, 17331)}
region_grids = {r: [unit(loss, *LINES[line]) for loss, line in zip(losses, LINES)]
                for r, losses in REGIONS.items()}
print(f"{'':6s} {'mean':>8s} {'cv':>6s} {'99th pct':>9s}   book")
for r, gs in region_grids.items():
    total = Portfolio.from_independent(list(LINES), gs)
    xs, ps = np.array(total.totals), np.array(total.probs)
    mean = (xs * ps).sum()
    cv = np.sqrt(((xs - mean) ** 2 * ps).sum()) / mean
    print(f"{r:6s} {mean:8.1f} {cv:6.3f} {total.assets(0.99):9.0f}   {BOOK_84[r]}")
```

```text
           mean     cv  99th pct   book
North    8000.0  0.298     16236   (8000, 0.298, 16237)
East     7000.0  0.347     15920   (7000, 0.347, 15920)
South   14000.0  0.266     25737   (14000, 0.266, 25737)
West     9000.0  0.276     17330   (9000, 0.276, 17331)
```

Means and CVs agree to the digits printed. The 99th percentiles agree within
1.5 in 16,000 to 26,000; they move by that much with the discretization step
(a step of 0.5 gives 16,236.5, 15,920.5, 25,737 and 17,329.5), so what is
left is the grid, not the model.

The block method then has three steps. Correlate the lines within each
region by ordinary Iman-Conover (the same 3 x 3 matrix in every region here,
Table 8.6 (e)), stack the regions into one `PredictiveDistribution` with a
`region` dimension, and correlate the regions' totals by moving each region's
years as whole rows, which keeps the structure within the region. prospicio's
`reorder_groups` is that last step:

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

B = np.array(bic.draw_matrix())
totals = B.reshape(1000, 4, 3).sum(axis=2)
achieved = np.corrcoef(totals.T)
BOOK_85 = {(0, 1): 0.683, (0, 2): 0.167, (0, 3): -0.002, (1, 2): 0.369, (1, 3): 0.197, (2, 3): 0.799}
names = list(REGIONS)
print(f"{'':12s} {'target':>7s} {'achieved':>9s} {'book':>7s}")
for (i, j), book in BOOK_85.items():
    print(f"{names[i] + '-' + names[j]:12s} {INTER[i][j]:7.1f} {achieved[i, j]:9.3f} {book:7.3f}")
```

```text
['region', 'line'] [('North', 'AL'), ('North', 'GL'), ('North', 'Pr'), ('East', 'AL')] ...
              target  achieved    book
North-East       0.7     0.684   0.683
North-South      0.2     0.199   0.167
North-West       0.0     0.004  -0.002
East-South       0.4     0.376   0.369
East-West        0.2     0.187   0.197
South-West       0.8     0.786   0.799
```

The correlations between regions land close to the target, as in Table 8.5,
and within sampling error of the monograph's. Within each region the lines
keep the correlation set in the first step, because the reordering moves a
region's three lines together (Table 8.6):

```python
C12 = np.corrcoef(B.T)
BOOK_86 = {"North": (0.579, 0.180, 0.332), "East": (0.544, 0.191, 0.318),
           "South": (0.571, 0.202, 0.354), "West": (0.580, 0.176, 0.330)}
print(f"{'':6s} {'AL-GL':>6s} {'AL-Pr':>6s} {'GL-Pr':>6s}   book (target 0.6, 0.2, 0.4)")
for k, r in enumerate(names):
    block = C12[3 * k:3 * k + 3, 3 * k:3 * k + 3]
    print(f"{r:6s} {block[0, 1]:6.3f} {block[0, 2]:6.3f} {block[1, 2]:6.3f}   {BOOK_86[r]}")
print("North AL with the other regions' lines:", C12[0, 3:].round(2).tolist())
print("book (Table 8.7):                       ",
      [0.32, 0.29, 0.42, 0.07, 0.04, 0.12, -0.01, -0.07, 0.01])
```

```text
        AL-GL  AL-Pr  GL-Pr   book (target 0.6, 0.2, 0.4)
North   0.567  0.165  0.299   (0.579, 0.18, 0.332)
East    0.512  0.192  0.331   (0.544, 0.191, 0.318)
South   0.576  0.213  0.385   (0.571, 0.202, 0.354)
West    0.557  0.173  0.317   (0.58, 0.176, 0.33)
North AL with the other regions' lines: [0.4, 0.33, 0.4, 0.11, 0.14, 0.12, -0.04, 0.01, 0.05]
book (Table 8.7):                        [0.32, 0.29, 0.42, 0.07, 0.04, 0.12, -0.01, -0.07, 0.01]
```

The full 12 x 12 matrix (Table 8.7) has the block structure of the
monograph's Figure 8.4: lines in regions that are correlated with each other
are correlated across regions too (North and East at 0.7), and lines in
regions that are not (North and West) are not. Nine numbers set all 66
correlations, which is the point of the method.

## 9. Capital by layer, if you must

Section 8.3 returns to InsCo (ten equally likely years, three units) and its
Wang distortion calibrated to a 15% return with assets of 100. It divides the
assets into layers between consecutive distinct totals. In each layer the
expected loss is shared between units by `alpha`, each unit's expected share
of the total in the years that reach the layer, and the premium by `beta`, the
same share under the distorted probabilities. A layer's margin is its premium
less its loss, its capital is the part of its width not funded by premium,
and its cost of capital `iota = (g(S) - S) / (1 - g(S))` is the same for every
unit. So a unit's capital in a layer is its margin there divided by `iota`
(equation 8.5).

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


print(f"{'x':>4s} {'dx':>3s} {'S':>4s} {'g(S)':>6s} {'iota':>6s} {'margin A, B, C':>20s}   {'capital A, B, C':>20s}")
for x, dx, s, g, iota, aS, bG, margin, capital in layers(insco, wang):
    it = f"{iota:6.3f}" if iota is not None else "     -"
    print(f"{x:4.0f} {dx:3.0f} {s:4.1f} {g:6.3f} {it} " + " ".join(f"{m:6.3f}" for m in margin)
          + "   " + " ".join(f"{q:6.3f}" for q in capital))
```

```text
   x  dx    S   g(S)   iota       margin A, B, C        capital A, B, C
   0  22  1.0  1.000      - -0.030 -0.032  0.061    0.000  0.000  0.000
  22   6  0.9  0.948  0.917  0.003 -0.017  0.061    0.003 -0.018  0.067
  28   8  0.8  0.882  0.693  0.021 -0.001  0.061    0.031 -0.001  0.089
  36   4  0.7  0.807  0.555  0.025  0.013  0.069    0.045  0.024  0.125
  40  15  0.3  0.428  0.224  0.030  0.026  0.072    0.134  0.115  0.323
  55  10  0.2  0.309  0.158  0.021  0.019  0.069    0.133  0.121  0.437
  65  35  0.1  0.174  0.089  0.012  0.015  0.047    0.132  0.165  0.529
```

These are Table 8.10 (with `g(S)` from Table 8.9), per unit of layer width,
and agree with it to the digits printed; `alpha S` and `beta g(S)`, the
columns of Tables 8.8 and 8.9, are kept for section 10. The bottom layer,
from 0 to 22, is paid for by losses in every year, so it needs no capital and
`iota` is undefined; its unit margins sum to zero and are cross-subsidies
between the units. Summing each column times the layer width gives each
unit's margin and capital:

```python
def by_unit(port, g):
    rows = layers(port, g)
    margin = [sum(r[1] * r[7][i] for r in rows) for i in range(3)]
    capital = [sum(r[1] * r[8][i] for r in rows) for i in range(3)]
    return margin, capital


margin, capital = by_unit(insco, wang)
lifted = insco.price(wang, assets=100, allocation="lifted")
print("margin   ", [round(m, 3) for m in margin], " book [0.709, 0.337, 5.919]")
print("capital  ", [round(q, 3) for q in capital], " book [8.411, 8.691, 29.333]")
print("Portfolio", [round(u.capital, 3) for u in lifted.allocated])
print("return   ", [f"{m / q:.1%}" for m, q in zip(margin, capital)],
      f"total {sum(margin) / sum(capital):.1%}", " book 8.4%, 3.9%, 20.2%, 15.0%")
```

```text
margin    [0.709, 0.337, 5.919]  book [0.709, 0.337, 5.919]
capital   [8.411, 8.691, 29.333]  book [8.411, 8.691, 29.333]
Portfolio [8.411, 8.691, 29.333]
return    ['8.4%', '3.9%', '20.2%'] total 15.0%  book 8.4%, 3.9%, 20.2%, 15.0%
```

The returns differ by unit (A earns 8.4%, C 20.2%) although every layer pays
the same return to every unit, because the units draw on different layers:
the low layers are expensive but mostly funded by premium, the high ones
cheap but mostly capital. `Portfolio.price(..., allocation="lifted")` gives
the same capital.

Table 8.11 sets this against the CCoC distortion, and there the numbers
depend on what the bottom layer gets. By equation 8.5 as written, the bottom
layer carries no capital. prospicio (like `aggregate` 1.0.1) instead converts
the bottom layer's margins at the limit of `iota` as `S` rises to 1, which is
`r = 0.15` for the CCoC and `(1 - rho) / rho` for the proportional hazard;
for the Wang, dual power and TVaR distortions that limit is infinite and the
two conventions agree. The monograph's printed capital for these two
families, from an earlier `aggregate`, matches neither:

```python
BOOK_Q = {"ccoc": (4.874, 5.288, 36.273), "ph": (7.074, 6.565, 32.796)}
for family in ("ccoc", "ph"):
    d = insco.calibrate(family, assets=100, return_on_capital=0.15)
    _, eq85 = by_unit(insco, d)
    lifted = [u.capital for u in insco.price(d, assets=100, allocation="lifted").allocated]
    print(f"{family:4s} equation 8.5 {[round(q, 3) for q in eq85]}  "
          f"prospicio {[round(q, 3) for q in lifted]}  book {list(BOOK_Q[family])}")
```

```text
ccoc equation 8.5 [5.419, 6.082, 34.934]  prospicio [2.261, 1.478, 42.696]  book [4.874, 5.288, 36.273]
ph   equation 8.5 [7.765, 7.694, 30.975]  prospicio [6.481, 5.597, 34.356]  book [7.074, 6.565, 32.796]
```

All three add up to the same total capital of 46.435; they differ only in how
the bottom layer's cross-subsidies, which sum to zero, are spread. Under
prospicio's convention every unit earns exactly 15% under the CCoC, which is
what a constant cost of capital means (see section 5 of the
[chapter 9 guide](capital-modeling-monograph-15.md)). The monograph's advice
is not to allocate capital at all; this is one more reason.

## 10. Bodoff's layer allocation

Section 8.4 compares the natural allocation with Bodoff's (2007): share every
layer of the assets, not only its expected loss, by `alpha`. Table 8.12:

```python
bodoff = insco.bodoff(assets=100)
print("Bodoff assets", [round(a, 3) for a in bodoff], " total", round(sum(bodoff), 3))
print("book          [24.159, 29.756, 46.086]  total 100.0")
first = layers(insco, wang)[0]
print("layer 0 to 22:", [round(first[1] * a / first[2], 3) for a in first[5]], " book [7.152, 9.694, 5.154]")
```

```text
Bodoff assets [24.159, 29.756, 46.086]  total 100.0
book          [24.159, 29.756, 46.086]  total 100.0
layer 0 to 22: [7.152, 9.694, 5.154]  book [7.152, 9.694, 5.154]
```

Bodoff's allocation splits the assets but not the premium. The monograph
fills the gap with a CCoC: in each layer, expected loss plus a margin of
`d = r / (1 + r)` on the unfunded part, `alpha (S + d (1 - S))`, which is the
CCoC's `g(S)` shared by the loss shares:

```python
ccoc = Distortion.ccoc(0.15)
bodoff_premium = [sum(dx * g * aS[i] / s for _, dx, s, g, _, aS, _, _, _ in layers(insco, ccoc) if s > 0)
                  for i in range(3)]
natural = [u.premium for u in insco.price(ccoc, assets=100).allocated]
print("Bodoff with CCoC", [round(v, 3) for v in bodoff_premium], round(sum(bodoff_premium), 3))
print("natural CCoC    ", [round(v, 3) for v in natural], round(sum(natural), 3))
```

```text
Bodoff with CCoC [14.803, 19.794, 18.968] 53.565
natural CCoC     [13.739, 18.522, 21.304] 53.565
```

The totals are the same 53.565, but the split is not: Bodoff's shares are the
expected loss shares in every layer, the natural allocation's the distorted
ones, which move premium toward the unit that dominates the years the
distortion weights up (C). Another distortion in place of the CCoC gives
other splits, which is the monograph's point that Bodoff's method needs a
further assumption before it prices anything.

## 11. What prospicio does not do here

- **Other reference scores.** `iman_conover` uses normal scores only; the
  monograph's `aggregate` also takes t scores (`dof`). Section 6 does it by
  hand. Filed as #378, with the row order below.
- **A block Iman-Conover function.** There is none by that name, but
  `iman_conover` within groups, `PredictiveDistribution.join` and
  `reorder_groups` compose it (section 8), with a different correlation
  matrix per region if wanted.
- **Row order.** The output's first component comes back sorted (section 4);
  shuffle before taking a subset.
- **Capital by unit under the CCoC and PH** depends on the bottom-layer
  convention (section 9): prospicio follows `aggregate` 1.0.1, not the
  monograph's table or equation 8.5 as written. Filed as #379, to document
  the convention or offer equation 8.5's.
