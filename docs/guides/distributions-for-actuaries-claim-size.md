# Claim size, layers, limits and deductibles: Bahnemann's *Distributions for Actuaries* reproduced

<!-- site: execute -->

CAS Monograph 2, *Distributions for Actuaries* by David Bahnemann (2015), is the
CAS textbook on the probability behind pricing. This guide follows chapters 2, 5
and 6 (claim size, excess claims, limits and deductibles) and shows the
monograph's printed value (`book`) beside prospicio's. Example and problem
numbers ("Ex 2.7", "P6.13") are the monograph's. Where the two disagree, the
guide gives the likely reason. Read the monograph for the derivations. The
companion guide,
[claim counts and aggregate claims](distributions-for-actuaries-counts-aggregates.md),
covers chapters 3 and 4 and the aggregate limits of section 6.4.

## 1. Translating the parameters

Most apparent disagreements between a textbook and a library come from names
and scales:

| Monograph | Its formula | prospicio |
|---|---|---|
| gamma (α, β), β a scale | mean αβ | `Gamma(shape=α, scale=β)` |
| exponential β | mean β | `Gamma(1.0, β)` |
| lognormal (μ, σ) | log X normal (μ, σ) | `Lognormal(μ, σ)` |
| Weibull (β, δ) | F = 1 − exp(−(x/β)^δ) | `Weibull(shape=δ, scale=β)` |
| Pareto (α, β), the "shifted" one used in all examples | F = 1 − (β/(x+β))^α, from 0 | `GeneralizedPareto(xi=1/α, beta=β/α)`, or `Burr(α, 1.0, β)` |
| classical Pareto (γ, α) | F = 1 − (γ/x)^α, from γ | `Pareto(t=γ, alpha=α)` |
| Burr (α, β, δ) | F = 1 − (β/(x^δ+β))^α | `Burr(α, δ, β^(1/δ))` |

The monograph's Pareto starts at zero (a Lomax); prospicio's `Pareto` starts at
a threshold. A helper builds the Lomax as a generalized Pareto with shape 1/α and
scale β/α. This block also defines the two table helpers, and the print checks
the Lomax against the closed form of Ex 2.15's limited mean:

```python
import math

import pandas as pd
from great_tables import GT
from scipy.optimize import minimize

from prospicio.distributions import (
    Beta, Burr, Gamma, GeneralizedPareto, Lognormal, Pareto, Truncated, fit_severity,
)
from prospicio.pricing import ilf, layer_cv, loss_elimination_ratio

NAN = float("nan")


def shifted_pareto(alpha, beta):
    """The monograph's Pareto, F(x) = 1 - (beta / (x + beta))**alpha."""
    return GeneralizedPareto(1.0 / alpha, beta / alpha)


def survival(sev, x):
    return 1.0 - sev.cdf(x)


def limited_second_moment(sev, limit):
    """E[X^2; limit]. Truncated() with no bounds adds layer moments to any severity."""
    return Truncated(sev).layer_second_moment(limit, 0.0)


def book_table(rows, title, subtitle=None):
    """One row per item, (label, book, prospicio, decimals): the book's value beside prospicio's."""
    frame = pd.DataFrame({"item": [r[0] for r in rows], "book": [r[1] for r in rows],
                          "prospicio": [r[2] for r in rows]})
    table = GT(frame, rowname_col="item")
    for places in sorted({r[3] for r in rows}):
        table = table.fmt_number(["book", "prospicio"], decimals=places,
                                 rows=[i for i, r in enumerate(rows) if r[3] == places])
    return table.sub_missing(missing_text="").tab_header(title, subtitle)


def grid_table(stub, index, measures, title, subtitle=None):
    """One measure: book and prospicio columns. Several: rows alternate prospicio and `, book`.

    Each measure is (name, book values, prospicio values, decimals).
    """
    at = [f"{x:,.0f}" if isinstance(x, (int, float)) else x for x in index]
    if len(measures) == 1:
        name, book, ours, places = measures[0]
        table = (GT(pd.DataFrame({"at": at, "book": book, "prospicio": ours}), rowname_col="at")
                 .tab_spanner(name, ["book", "prospicio"]).fmt_number(["book", "prospicio"], decimals=places))
    else:
        frame = pd.DataFrame({"at": [label for a in at for label in (a, f"{a}, book")]})
        for name, book, ours, _ in measures:
            frame[name] = [v for pair in zip(ours, book) for v in pair]
        table = GT(frame, rowname_col="at")
        for name, _, _, places in measures:
            table = table.fmt_number(name, decimals=places)
    return table.tab_stubhead(stub).sub_missing(missing_text="").tab_header(title, subtitle)


p = shifted_pareto(2.0, 3000.0)
closed_form = 3000.0 / (2 - 1) * (1 - 3000.0 / (8000.0 + 3000.0))
print(f"E[X; 8,000] of Pareto (2; 3,000): book 2,182, closed form {closed_form:,.2f}, "
      f"prospicio {p.lev(8000.0):,.2f}, Burr {Burr(2.0, 1.0, 3000.0).lev(8000.0):,.2f}")
```

```text
E[X; 8,000] of Pareto (2; 3,000): book 2,182, closed form 2,181.82, prospicio 2,181.82, Burr 2,181.82
```

## 2. Limited expected values

The limited expected value E[X; l] is the average claim under a cap l. Increased
limit factors, layers, deductibles and trend are differences or ratios of it.
Every severity has it as `lev`, and `Truncated(sev).layer_second_moment` gives
E[X²; l]. Ex 2.4 caps an exponential (mean 250) at 400. P2.1, P2.17 and P2.19
give the usual values for a classical Pareto, a lognormal and a shifted Pareto:

```python
x = Gamma(1.0, 250.0)
rows = [("Ex 2.4 E[X; 400]", 199.53, x.lev(400.0), 2),
        ("Ex 2.4 E[X^2; 400]", 59_384, limited_second_moment(x, 400.0), 0)]

x = Pareto(1000.0, 3.0)  # classical: F = 1 - (1,000 / x)^3 from 1,000
rows += [("P2.1 mean", 1_500, x.mean(), 0),
         ("P2.1 variance", 750_000, x.variance(), 0),
         ("P2.1 Pr{X > 2,000}", 0.1250, x.survival(2000.0), 4),
         ("P2.1 E[X; 2,000]", 1_375, x.lev(2000.0), 0)]

for name, x, book in [
    ("P2.17", Lognormal(6.3210, 1.6000), (556, 0.1461, 0.2108, 1_023, 9_689)),
    ("P2.19", shifted_pareto(3.0, 4000.0), (1_040, 0.1866, 0.3254, 1_347, 6_500)),
]:
    s3 = survival(x, 3000.0)
    rows += [(f"{name} median", book[0], x.quantile(0.5), 0),
             (f"{name} Pr{{X > 3,000}}", book[1], s3, 4),
             (f"{name} Pr{{1,000 < X <= 3,000}}", book[2], x.cdf(3000.0) - x.cdf(1000.0), 4),
             (f"{name} E[X; 3,000]", book[3], x.lev(3000.0), 0),
             (f"{name} E[X | X > 3,000]", book[4], 3000.0 + x.stop_loss(3000.0) / s3, 0)]
book_table(rows, "Limited expected values and tail probabilities", "Chapter 2, examples and problems")
```

```text
                                 book  prospicio
Ex 2.4 E[X; 400]               199.53     199.53
Ex 2.4 E[X^2; 400]             59,384     59,384
P2.1 mean                       1,500      1,500
P2.1 variance                 750,000    750,000
P2.1 Pr{X > 2,000}             0.1250     0.1250
P2.1 E[X; 2,000]                1,375      1,375
P2.17 median                      556        556
P2.17 Pr{X > 3,000}            0.1461     0.1461
P2.17 Pr{1,000 < X <= 3,000}   0.2108     0.2108
P2.17 E[X; 3,000]               1,023      1,023
P2.17 E[X | X > 3,000]          9,689      9,689
P2.19 median                    1,040      1,040
P2.19 Pr{X > 3,000}            0.1866     0.1866
P2.19 Pr{1,000 < X <= 3,000}   0.3254     0.3254
P2.19 E[X; 3,000]               1,347      1,347
P2.19 E[X | X > 3,000]          6,500      6,500
```

Two facts for later:

- The mean of the claims above x is x plus the **mean excess**
  e(x) = E[(X − x)+] / Pr{X > x}. `stop_loss(x)` is its numerator.
- The average claim above 3,000 is nearly ten times the lognormal mean (about
  2,000). The tail sets excess prices.

P2.25 gives only the mean (1,000) and the standard deviation (2,000). How many
of 5,000 claims exceed 1,000 under each family?

```python
book = {"gamma": 1_282, "lognormal": 1_315, "Pareto": 1_428}
# A shifted Pareto with cv 2 has alpha / (alpha - 2) = 4, so alpha = 8/3.
fits = {
    "gamma": Gamma.from_mean_cv(1000.0, 2.0),
    "lognormal": Lognormal.from_mean_cv(1000.0, 2.0),
    "Pareto": shifted_pareto(8 / 3, 1000.0 * (8 / 3 - 1)),
}
book_table([(name, book[name], 5000 * survival(x, 1000.0), 0) for name, x in fits.items()],
           "P2.25: claims above 1,000 out of 5,000", "Mean 1,000 and standard deviation 2,000, by family")
```

```text
            book  prospicio
gamma      1,282      1,282
lognormal  1,315      1,315
Pareto     1,428      1,428
```

## 3. Fitting grouped data

The monograph's data are counts of claims per size band. It fits them on a
spreadsheet solver by moments, by minimum chi-square or by the distance between
fitted and empirical limited means. SciPy's Nelder–Mead replaces the solver
here. Ex 2.6 has 200 claims in nine bands, with sample moments M1 = 1,555 and
M2 = 3,036,875. Ex 2.7 matches a gamma to them:

```python
M1, M2 = 1555.0, 3_036_875.0
alpha = M1**2 / (M2 - M1**2)
beta = (M2 - M1**2) / M1
print(f"alpha {alpha:.6f} (book 3.907288), beta {beta:.3f} (book 397.931)")
```

```text
alpha 3.907288 (book 3.907288), beta 397.974 (book 397.931)
```

The printed scale is a slip: 618,850 / 1,555 = 397.974, not 397.931. The
monograph's tables use 397.931, so the next blocks do too. A chi-square test on
six cells measures the fit:

```python
CUTS = [0, 1000, 1500, 2000, 2500, 3000, math.inf]
OBSERVED = [42, 61, 47, 26, 14, 10]


def cell_probs(sev, cuts):
    cdf = [0.0 if c == 0 else 1.0 if c == math.inf else sev.cdf(c) for c in cuts]
    return [b - a for a, b in zip(cdf, cdf[1:])]


def chi_square(sev, cuts=CUTS, observed=OBSERVED):
    n = sum(observed)
    return sum((o - n * p) ** 2 / (n * p) for o, p in zip(observed, cell_probs(sev, cuts)))


g = Gamma(3.907288, 397.931)
xs = [1000, 1500, 2000, 2500, 3000]
grid_table("x", xs, [
    ("Pr{X > x}", [0.7382, 0.4604, 0.2465, 0.1186, 0.0528], [survival(g, x) for x in xs], 4),
    ("E[X; x]", [924, 1223, 1396, 1484, 1525], [g.lev(x) for x in xs], 0),
], "Ex 2.7: the gamma fitted by moments")
```

```text
             Pr{X > x}  E[X; x]
1,000           0.7382      924
1,000, book     0.7382      924
1,500           0.4604    1,223
1,500, book     0.4604    1,223
2,000           0.2465    1,396
2,000, book     0.2465    1,396
2,500           0.1186    1,484
2,500, book     0.1186    1,484
3,000           0.0528    1,525
3,000, book     0.0528    1,525
```

```python
print(f"chi-square: book 3.096, prospicio {chi_square(g):.3f}")
```

```text
chi-square: book 3.096, prospicio 3.095
```

The monograph rounds the expected counts to two decimals before summing, hence
the third-decimal difference.

Ex 2.8 fits a lognormal to the same cells by **minimum chi-square**, the
parameters with the smallest statistic. The monograph reports
(7.274670, 0.442525) and a minimum of 1.828:

```python
ln = Lognormal(7.274670, 0.442525)
grid_table("x", xs, [
    ("Pr{X > x}", [0.7965, 0.4653, 0.2305, 0.1072, 0.0491], [survival(ln, x) for x in xs], 4),
    ("E[X; x]", [958, 1273, 1441, 1522, 1559], [ln.lev(x) for x in xs], 0),
], "Ex 2.8: the lognormal at the monograph's parameters")
```

```text
             Pr{X > x}  E[X; x]
1,000           0.7965      958
1,000, book     0.7965      958
1,500           0.4653    1,273
1,500, book     0.4653    1,273
2,000           0.2305    1,441
2,000, book     0.2305    1,441
2,500           0.1072    1,522
2,500, book     0.1072    1,522
3,000           0.0491    1,559
3,000, book     0.0491    1,559
```

```python
def min_chi_square(make, start, **kw):
    fit = minimize(lambda p: chi_square(make(*p), **kw), start, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    return fit.x, fit.fun


(mu, sigma), chi2 = min_chi_square(lambda m, s: Lognormal(m, abs(s)), [7.2, 0.5])
(a, b), _ = min_chi_square(lambda a, b: Gamma(abs(a), abs(b)), [4.0, 400.0])
book_table([("Ex 2.8 chi-square at the book's parameters", 1.828, chi_square(ln), 3),
            ("Ex 2.8 smallest chi-square", NAN, chi2, 3),
            ("Ex 2.8 mu at the smallest", NAN, mu, 4),
            ("Ex 2.8 sigma at the smallest", NAN, abs(sigma), 4),
            ("P2.16 gamma a", 4.7432, abs(a), 4),
            ("P2.16 gamma b", 337.31, abs(b), 2)],
           "Minimum chi-square fits", "Ex 2.8's lognormal and P2.16's gamma")
```

```text
                                              book  prospicio
Ex 2.8 chi-square at the book's parameters   1.828      1.020
Ex 2.8 smallest chi-square                              0.614
Ex 2.8 mu at the smallest                              7.2857
Ex 2.8 sigma at the smallest                           0.4570
P2.16 gamma a                               4.7432     4.7432
P2.16 gamma b                               337.31     337.31
```

The tail probabilities and limited means match Table 2.3, so these are the
monograph's parameters. Their statistic is 1.020, not 1.828, and they are not
the minimum: the solver stopped early. The conclusion stands, since the fit is
far below the 5% limit of 7.815. For P2.16's gamma, the same search gives the
printed answer.

Each band's count beside the two fits shows where they differ:

```python
import matplotlib.pyplot as plt

bands = ["0-1k", "1-1.5k", "1.5-2k", "2-2.5k", "2.5-3k", "3k+"]
fits = {"gamma, Ex 2.7": g, "lognormal, least chi-square": Lognormal(mu, abs(sigma))}
fig, ax = plt.subplots()
ax.bar(bands, OBSERVED, color="0.75", label="observed claims")
for name, sev in fits.items():
    ax.plot(bands, [200 * q for q in cell_probs(sev, CUTS)], marker="o", label=name)
ax.set_ylabel("claims in the band")
ax.set_title("Ex 2.6: 200 claims by size band, and two fitted distributions")
ax.legend()
```

Ex 2.9 fits the shifted Pareto to a heavier sample by the limited-mean
distance, with α fixed at 6. At the printed parameters, prospicio gives
Table 2.4:

```python
p = shifted_pareto(6.0, 34_355.3719)
xs = [2000, 4000, 6000, 8000, 10000]
grid_table("x", xs, [
    ("Pr{X > x}", [0.7121, 0.5164, 0.3807, 0.2848, 0.2159], [p.survival(x) for x in xs], 4),
    ("E[X; x]", [1693, 2910, 3799, 4459, 4956], [p.lev(x) for x in xs], 0),
], "Ex 2.9: the shifted Pareto (6; 34,355.37)")
```

```text
              Pr{X > x}  E[X; x]
2,000            0.7121    1,693
2,000, book      0.7121    1,693
4,000            0.5164    2,910
4,000, book      0.5164    2,910
6,000            0.3807    3,799
6,000, book      0.3807    3,799
8,000            0.2848    4,459
8,000, book      0.2848    4,459
10,000           0.2159    4,956
10,000, book     0.2159    4,956
```

## 4. Censored and truncated data

A **policy limit censors**: a claim at the limit is only known to be at least
that large. A **deductible truncates**: claims below it do not appear. Fitting
the recorded claims as the population understates the tail, so fit the
ground-up distribution through the modification.

Ex 2.10 has 1,500 claims under a 300,000 limit, 23 of them at the limit. The
data are counts and totals below eight sizes, so the empirical limited mean at
each size is exact. The monograph minimizes its squared distance to the
lognormal's `lev`. P2.39 does the same for another censored sample:

```python
def min_distance(sizes, empirical, start):
    def distance(p):
        ln = Lognormal(p[0], abs(p[1]))
        return sum((ln.lev(x) - e) ** 2 for x, e in zip(sizes, empirical))

    fit = minimize(distance, start, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    return fit.x[0], abs(fit.x[1])


sizes = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 300000]
below = [729, 1096, 1208, 1326, 1391, 1440, 1468, 1500]
total_below = [225_138, 1_102_272, 1_918_947, 3_752_091,
               6_007_543, 9_234_739, 13_100_561, 22_343_455]
# A claim above x contributes x to E[X; x]; those below contribute themselves.
emp = [(s + x * (1500 - n)) / 1500 for x, n, s in zip(sizes, below, total_below)]
mu_2_10, sigma_2_10 = min_distance(sizes, emp, [7.0, 2.5])
fit_2_10 = Lognormal(mu_2_10, sigma_2_10)
grid_table("x", sizes, [
    ("E[X; x]", [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850], [fit_2_10.lev(x) for x in sizes], 0),
], "Ex 2.10: limited means of the lognormal fitted by minimum distance")
```

```text
           book  prospicio
1,000       648        648
5,000     2,091      2,091
10,000    3,239      3,239
25,000    5,410      5,410
50,000    7,580      7,580
100,000  10,168     10,168
200,000  13,069     13,069
300,000  14,850     14,850
```

```python
sizes = [10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000]
counts = [78, 27, 21, 15, 12, 10, 8, 7]          # 22 more at the 100,000 limit
mids = [5000, 15000, 25000, 35000, 45000, 55000, 70000, 90000]
emp = [(sum(n * m for n, m in zip(counts[:j + 1], mids[:j + 1]))
        + x * (200 - sum(counts[:j + 1]))) / 200 for j, x in enumerate(sizes)]
mu, sigma = min_distance(sizes, emp, [9.7, 1.4])
book_table([("Ex 2.10 mu", 6.9852, mu_2_10, 4), ("Ex 2.10 sigma", 2.5850, sigma_2_10, 4),
            ("P2.39 mu", 9.778102, mu, 4), ("P2.39 sigma", 1.444776, sigma, 4)],
           "Lognormals fitted to censored samples", "Minimum limited-mean distance")
```

```text
                 book  prospicio
Ex 2.10 mu     6.9852     6.9852
Ex 2.10 sigma  2.5850     2.5850
P2.39 mu       9.7781     9.7781
P2.39 sigma    1.4448     1.4448
```

Under a deductible, each cell's probability is conditional on a claim above it.
Ex 2.11 has 770 claims above a 1,000 deductible, censored at 200,000. P2.43
keeps only claims up to 50,000. Both are fitted by minimum chi-square, with
prospicio's `Truncated` giving the conditional distribution:

```python
cuts = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000,
        200000, math.inf]
observed = [367, 112, 118, 65, 36, 13, 10, 8, 6, 4, 31]
(mu, sigma), chi2 = min_chi_square(
    lambda m, s: Truncated(Lognormal(m, abs(s)), lower=1000.0), [6.7, 2.7],
    cuts=cuts, observed=observed)
fitted = Truncated(Lognormal(mu, abs(sigma)), lower=1000.0)
rows = [("Ex 2.11 mu", 6.6916, mu, 4), ("Ex 2.11 sigma", 2.6965, abs(sigma), 4),
        ("Ex 2.11 chi-square", 4.691, chi2, 3)]

cuts_43 = [5000 * k for k in range(11)]
observed_43 = [104, 120, 81, 54, 45, 32, 26, 18, 12, 8]
(mu, sigma), _ = min_chi_square(
    lambda m, s: Truncated(Lognormal(m, abs(s)), upper=50000.0), [9.5, 1.1],
    cuts=cuts_43, observed=observed_43)
rows += [("P2.43 mu", 9.495111, mu, 4), ("P2.43 sigma", 1.084180, abs(sigma), 4)]
book_table(rows, "Lognormals fitted by minimum chi-square to truncated samples",
           "Ex 2.11 (deductible 1,000, limit 200,000) and P2.43 (up to 50,000)")
```

```text
                      book  prospicio
Ex 2.11 mu          6.6916     6.6916
Ex 2.11 sigma       2.6965     2.6965
Ex 2.11 chi-square   4.691      4.691
P2.43 mu            9.4951     9.4951
P2.43 sigma         1.0842     1.0842
```

Ex 2.11 band by band, the observed claims beside the expected counts:

```python
ex_2_11 = pd.DataFrame({
    "band": ["1k to 5k", "5k to 10k", "10k to 25k", "25k to 50k", "50k to 75k", "75k to 100k",
             "100k to 125k", "125k to 150k", "150k to 175k", "175k to 200k", "at the 200k limit"],
    "observed": observed,
    "book": [360, 122, 121, 63, 27, 16, 10, 7, 5, 4, 34],
    "prospicio": [770 * q for q in cell_probs(fitted, cuts)],
})
(GT(ex_2_11, rowname_col="band")
 .tab_header("Ex 2.11: 770 claims above a 1,000 deductible, censored at 200,000",
             "Expected counts of the lognormal fitted by minimum chi-square")
 .tab_spanner("expected", ["book", "prospicio"])
 .fmt_integer(["observed", "book", "prospicio"]))
```

```text
                   observed  book  prospicio
1k to 5k                367   360        360
5k to 10k               112   122        122
10k to 25k              118   121        121
25k to 50k               65    63         63
50k to 75k               36    27         27
75k to 100k              13    16         16
100k to 125k             10    10         10
125k to 150k              8     7          7
150k to 175k              6     5          5
175k to 200k              4     4          4
at the 200k limit        31    34         34
```

With individual claims, `fit_severity` does the same by maximum likelihood.
`reporting_thresholds` conditions each claim on exceeding its own threshold (the
truncation), and `censored` marks a claim at a limit as "at least this value".
The monograph prints no claim-level data, so the next block draws 20,000
ground-up claims from Ex 2.11's lognormal and applies its deductible and
limit:

```python
truth = Lognormal(6.6916, 2.6965)
ground_up = truth.sample(20_000, seed=2)
kept = [x for x in ground_up if x > 1000.0]          # the deductible hides the rest
recorded = [min(x, 200_000.0) for x in kept]
at_limit = [x >= 200_000.0 for x in kept]
print(f"{len(kept):,} of 20,000 claims recorded, {sum(at_limit)} at the limit")
```

```text
9,551 of 20,000 claims recorded, 415 at the limit
```

Three fits: no correction, the limit only, both:

```python
rows = []
for label, kw in [
    ("ignoring both", {}),
    ("censoring only", {"censored": at_limit}),
    ("truncation and censoring", {"censored": at_limit,
                                  "reporting_thresholds": [1000.0] * len(kept)}),
]:
    fit = fit_severity("lognormal", recorded, **kw)
    rows.append((label, fit.parameters["meanlog"], fit.standard_errors["meanlog"],
                 fit.parameters["sdlog"]))
(GT(pd.DataFrame(rows, columns=["fit", "mu", "se of mu", "sigma"]), rowname_col="fit")
 .tab_header("Ex 2.11 refitted from the 20,000 simulated claims", "The population has mu 6.69 and sigma 2.70")
 .fmt_number(["mu", "se of mu", "sigma"], decimals=2))
```

```text
                            mu  se of mu  sigma
ignoring both             8.94      0.01   1.46
censoring only            8.96      0.02   1.52
truncation and censoring  6.76      0.15   2.67
```

Only the last fit recovers the population within its standard errors. The
other two describe the recorded claims, with a thinner tail. The deductible,
which hides more than half of the population, causes the error. Few claims
reach the limit.

On a log scale: the 20,000 ground-up claims (never seen), the recorded ones, and
each fit's expected count per bin:

```python
edges = [10 ** (k / 10) for k in range(0, 71)]          # 1 to 10,000,000, ten bins a decade
fits = {}
for label, kw in [
    ("fit ignoring both", {}),
    ("fit with both", {"censored": at_limit, "reporting_thresholds": [1000.0] * len(kept)}),
]:
    fit = fit_severity("lognormal", recorded, **kw)
    fits[label] = Lognormal(fit.parameters["meanlog"], fit.parameters["sdlog"])
fig, ax = plt.subplots()
ax.hist(ground_up, bins=edges, color="0.82", label="ground-up")
ax.hist(recorded, bins=edges, color="0.55", label="recorded")
for label, sev in fits.items():
    expected = [20_000 * (sev.cdf(b) - sev.cdf(a)) for a, b in zip(edges, edges[1:])]
    ax.stairs(expected, edges, linewidth=2, label=label)
ax.axvline(1000.0, color="0.4", linestyle=":")
ax.axvline(200_000.0, color="0.4", linestyle=":")
ax.set_xscale("log")
ax.set_xlabel("claim size (dotted: the deductible and the limit)")
ax.set_ylabel("claims in the bin")
ax.set_title("Ex 2.11: only the corrected fit follows the population")
ax.legend(loc="upper left")
```

## 5. Inflation and claim size

A uniform inflation of t − 1 multiplies every claim by t. A family keeps its
form with its scale multiplied by t (a lognormal's μ moves by log t). The
limited mean grows more slowly, because the cap does not inflate:
E[tX; l] = t E[X; l/t]. Ex 2.15 applies +10% and −5% to a Pareto (2; 3,000)
capped at 8,000:

```python
p = shifted_pareto(2.0, 3000.0)
base = p.lev(8000.0)
up = shifted_pareto(2.0, 3000.0 * 1.10).lev(8000.0)   # rescale the family
down = 0.95 * p.lev(8000.0 / 0.95)                    # or E[tX; l] = t E[X; l/t]
book_table([("E[X; 8,000]", 2_182, base, 0),
            ("after +10%", 2_336, up, 0),
            ("effective rate, %", 7.1, 100 * (up / base - 1), 1),
            ("after -5%", 2_101, down, 0),
            ("effective rate, %", -3.7, 100 * (down / base - 1), 1)],
           "Ex 2.15: inflation on a capped Pareto (2; 3,000)")
```

```text
                    book  prospicio
E[X; 8,000]        2,182      2,182
after +10%         2,336      2,336
effective rate, %    7.1        7.1
after -5%          2,101      2,101
effective rate, %   -3.7       -3.7
```

A cap absorbs part of any inflation, so basic-limits rates trend more slowly
than ground-up severity. In Ex 2.16 inflation grows with claim size,
t(x) = 0.96 x^0.0183. A power of a lognormal is lognormal, so the inflated
claims are lognormal (1.0183 μ + log 0.96, 1.0183 σ):

```python
before = Lognormal(7.2, 0.476)
after = Lognormal(1.0183 * 7.2 + math.log(0.96), 1.0183 * 0.476)
print(f"mean before {before.mean():,.0f} (book 1,500), after {after.mean():,.0f} (book 1,650)")
print(f"inflation of a claim of 1,000: {0.96 * 1000**0.0183 - 1:.2%}, "
      f"of 5,000: {0.96 * 5000**0.0183 - 1:.2%}")
```

```text
mean before 1,500 (book 1,500), after 1,650 (book 1,650)
inflation of a claim of 1,000: 8.94%, of 5,000: 12.19%
```

## 6. Excess claims and layers

The claims above an attachment a, measured from a, are the **excess** claims
X_a, with mean e(a) = E[(X − a)+] / Pr{X > a}. A layer of width l above a has,
per claim that reaches it, the mean (E[X; a + l] − E[X; a]) / Pr{X > a}. The
numerators are `stop_loss(a)` and `layer(l, a)`. The shape of e(x) identifies
the family:

- constant for the exponential,
- decreasing for a gamma with shape above 1,
- a rising straight line for the shifted Pareto, e(x) = (x + β)/(α − 1),
- rising and curved for the lognormal.

P5.2 asks for e(3,000) of five distributions with mean 2,000. The uniform on
[0, 4,000] is `Beta(1, 1, scale)`. Ex 5.4 fits a Pareto by regressing the
empirical e(x) on x. A slope of 0.257335 and an intercept of 1,208.50 give
α − 1 = 1/slope and β = intercept · (α − 1):

```python
cases = [
    ("uniform [0; 4,000]", 500, Beta(1.0, 1.0, 4000.0)),
    ("gamma (2; 1,000)", 1_250, Gamma(2.0, 1000.0)),
    ("exponential 2,000", 2_000, Gamma(1.0, 2000.0)),
    ("Pareto (3; 4,000)", 3_500, shifted_pareto(3.0, 4000.0)),
    ("lognormal (5.9809; 1.8)", 8_518, Lognormal(5.9809, 1.8)),
]
rows = [(f"P5.2 e(3,000), {name}", book, x.stop_loss(3000.0) / survival(x, 3000.0), 0)
        for name, book, x in cases]
alpha = 1 + 1 / 0.257335
beta = 1208.50 * (alpha - 1)
rows += [("Ex 5.4 alpha", 4.88599, alpha, 5), ("Ex 5.4 beta", 4_696.22, beta, 2)]
book_table(rows, "Mean excess at 3,000 (P5.2) and the Pareto fitted by regression (Ex 5.4)")
```

```text
                                            book  prospicio
P5.2 e(3,000), uniform [0; 4,000]            500        500
P5.2 e(3,000), gamma (2; 1,000)            1,250      1,250
P5.2 e(3,000), exponential 2,000           2,000      2,000
P5.2 e(3,000), Pareto (3; 4,000)           3,500      3,500
P5.2 e(3,000), lognormal (5.9809; 1.8)     8,518      8,518
Ex 5.4 alpha                             4.88599    4.88599
Ex 5.4 beta                             4,696.22   4,696.21
```

```python
p = shifted_pareto(alpha, beta)
xs = [500, 1000, 2000, 5000, 10000]
grid_table("x", xs, [
    ("F(x)", [0.3900, 0.6106, 0.8233, 0.9711, 0.9962], [p.cdf(x) for x in xs], 4),
    ("E[X; x]", [393, 638, 904, 1136, 1194], [p.lev(x) for x in xs], 0),
], "Ex 5.4: the fitted Pareto")
```

```text
                F(x)  E[X; x]
500           0.3900      393
500, book     0.3900      393
1,000         0.6106      638
1,000, book   0.6106      638
2,000         0.8233      904
2,000, book   0.8233      904
5,000         0.9711    1,136
5,000, book   0.9711    1,136
10,000        0.9962    1,194
10,000, book  0.9962    1,194
```

Ex 5.7 prices a layer of 4,000 above 5,000 under a Pareto (2; 3,000). The excess
of a shifted Pareto over d is a shifted Pareto (α; β + d), so the layer's mean
per excess claim is the limited mean of a Pareto (2; 8,000) at 4,000. `layer`
agrees:

```python
p = shifted_pareto(2.0, 3000.0)
book_table([("E[X; 5,000]", 1_875, p.lev(5000.0), 0),
            ("E[X; 9,000]", 2_250, p.lev(9000.0), 0),
            ("layer mean per excess claim", 2_667, p.layer(4000.0, 5000.0) / p.survival(5000.0), 0),
            ("as Pareto (2; 8,000) at 4,000", 2_667, shifted_pareto(2.0, 8000.0).lev(4000.0), 0)],
           "Ex 5.7: the layer 4,000 xs 5,000 of a Pareto (2; 3,000)")
```

```text
                                book  prospicio
E[X; 5,000]                    1,875      1,875
E[X; 9,000]                    2,250      2,250
layer mean per excess claim    2,667      2,667
as Pareto (2; 8,000) at 4,000  2,667      2,667
```

Coefficients of variation (CV) grow large in layers. Ex 5.8 and P5.10 measure
them for the lognormal (5.9809; 1.8), whose mean is 2,000:

```python
ln = Lognormal(5.9809, 1.8)
s3 = survival(ln, 3000.0)
excess_mean = ln.stop_loss(3000.0) / s3
layer_mean = ln.layer(5000.0, 3000.0) / s3
layer_m2 = Truncated(ln).layer_second_moment(5000.0, 3000.0) / s3
# The monograph's own rounded inputs reproduce its printed layer CV:
m = (1276 - 891) / 0.130239
m2 = (5_774_970 - 1_853_050 - 2 * 3000 * (1276 - 891)) / 0.130239


def cv_limited(limit):
    lev = ln.lev(limit)
    return math.sqrt(limited_second_moment(ln, limit) - lev**2) / lev


book_table([
    ("Ex 5.8 F(3,000)", 0.869761, ln.cdf(3000.0), 6),
    ("E[X; 3,000]", 891, ln.lev(3000.0), 0),
    ("E[X^2; 3,000]", 1_853_050, limited_second_moment(ln, 3000.0), 0),
    ("E[X^2; 8,000]", 5_774_970, limited_second_moment(ln, 8000.0), 0),
    ("CV[X]", 4.9531, ln.std() / ln.mean(), 4),
    ("E[X_3K], the excess mean", 8_515, excess_mean, 0),
    ("CV of the layer 5K xs 3K", 0.6452, math.sqrt(layer_m2 - layer_mean**2) / layer_mean, 4),
    ("same, from the book's rounded inputs", 0.6452, math.sqrt(m2 - m * m) / m, 4),
    ("P5.10 CV[X; 3,000]", 1.1551, cv_limited(3000.0), 4),
    ("P5.10 CV[X; 8,000]", 1.5959, cv_limited(8000.0), 4),
], "Moments and CVs of the lognormal (5.9809; 1.8)")
```

```text
                                           book  prospicio
Ex 5.8 F(3,000)                        0.869761   0.869761
E[X; 3,000]                                 891        891
E[X^2; 3,000]                         1,853,050  1,853,050
E[X^2; 8,000]                         5,774,970  5,774,970
CV[X]                                    4.9531     4.9532
E[X_3K], the excess mean                  8,515      8,518
CV of the layer 5K xs 3K                 0.6452     0.6385
same, from the book's rounded inputs     0.6452     0.6452
P5.10 CV[X; 3,000]                       1.1551     1.1561
P5.10 CV[X; 8,000]                       1.5959     1.5955
```

The derived numbers differ in the third or fourth figure because the monograph
subtracts rounded limited means (891 for 890.56, 1,276 for 1,276.24). The
difference cancels most of their size, so half a unit becomes one percent of the
layer's CV, as the rounded-inputs row shows. The excess mean of 8,515 has the
same cause: P5.2 prints 8,518, prospicio's value.

prospicio's `layer_cv` is a different quantity (2.90 here): the CV of the layer
loss per ground-up claim, zeros included. Use it to price the layer from
ground-up counts.

P5.12 splits a policy with a 100 deductible and a 3,000 limit into three layers,
for a Pareto (3; 5,000) and 5 expected ground-up claims. The layers' expected
losses add up to the unlimited total:

```python
p = shifted_pareto(3.0, 5000.0)
shares, severities, losses = [], [], []
for a, width in [(0.0, 100.0), (100.0, 2900.0), (3000.0, math.inf)]:
    reach = p.survival(a)
    loss = p.stop_loss(a) if width == math.inf else p.layer(width, a)
    shares.append(reach)
    severities.append(loss / reach)
    losses.append(5 * loss)
grid_table("layer", ["[0; 100]", "(100; 3,000]", "(3,000; inf)", "total, 5 x mean"], [
    ("share reaching", [1.0000, 0.9423, 0.2441, NAN], shares + [NAN], 4),
    ("severity", [97.08, 1_513.66, 4_000.00, NAN], severities + [NAN], 2),
    ("expected loss", [485, 7_132, 4_883, 12_500], losses + [sum(losses)], 0),
], "P5.12: three layers of a policy with a 100 deductible and a 3,000 limit")
```

```text
                       share reaching  severity  expected loss
[0; 100]                       1.0000     97.08            485
[0; 100], book                 1.0000     97.08            485
(100; 3,000]                   0.9423  1,513.66          7,132
(100; 3,000], book             0.9423  1,513.66          7,132
(3,000; inf)                   0.2441  4,000.00          4,883
(3,000; inf), book             0.2441  4,000.00          4,883
total, 5 x mean                                         12,500
total, 5 x mean, book                                   12,500
```

Deductibles truncate counts too. Ex 5.3 fits a lognormal by minimum chi-square
to 300 claims, net of a 1,500 deductible and censored at 50,000, then estimates
the claims the deductible hid. P5.23 does the same with a 500 deductible. From
the printed parameters, prospicio gives the printed counts:

```python
for name, book, (mu, sigma), n, d in [
    ("Ex 5.3", 43, (8.67593, 1.18109), 300, 1500.0),
    ("P5.23", 125, (7.960294, 1.428801), 1000, 500.0),
]:
    f = Lognormal(mu, sigma).cdf(d)
    print(f"{name} claims hidden by the deductible: book {book}, prospicio {n * f / (1 - f):.0f}")
```

```text
Ex 5.3 claims hidden by the deductible: book 43, prospicio 43
P5.23 claims hidden by the deductible: book 125, prospicio 125
```

The fits do not reproduce. At Ex 5.3's printed parameters, the chi-square as the
monograph defines it is 1.806, not the printed 1.6610; the minimum is 1.790 at
(8.6869, 1.1684). For P5.23, the printed parameters are further from the minimum
under every reading of the cells that we tried. Both are spreadsheet solver
results, and a footnote in the monograph warns that they can vary. The hidden
claims change very little.

## 7. Inflation above an attachment: the leveraging effect

Inflation hits an excess layer harder than the ground-up loss. At a fixed
attachment, claims below it move into the layer (more excess claims) and claims
above it grow (a larger excess severity). The layer's loss grows as their
product. Ex 5.10 to 5.12 use the Pareto (2; 3,000), an attachment of 5,000 and
10% inflation. Ex 5.12 adds 5% growth in the ground-up count:

```python
p, t = shifted_pareto(2.0, 3000.0), 1.10
e0 = p.stop_loss(5000.0) / p.survival(5000.0)
e1 = t * p.stop_loss(5000.0 / t) / p.survival(5000.0 / t)      # e of tX at 5,000
layer0 = p.layer(4000.0, 5000.0) / p.survival(5000.0)
layer1 = t * p.layer(4000.0 / t, 5000.0 / t) / p.survival(5000.0 / t)
count = p.survival(5000.0 / t) / p.survival(5000.0)
book_table([
    ("Ex 5.10 e(5,000)", 8_000, e0, 0),
    ("e(5,000) after inflation", 8_300, e1, 0),
    ("excess severity trend, %", 3.75, 100 * (e1 / e0 - 1), 2),
    ("layer 4K xs 5K severity, trended", 2_701, layer1, 0),
    ("layer severity trend, %", 1.27, 100 * (layer1 / layer0 - 1), 2),
    ("Ex 5.11 excess count factor", 1.1241, count, 4),
    ("Ex 5.12 excess count trend, %", 18.03, 100 * (1.05 * count - 1), 2),
    ("excess loss trend, %", 22.46, 100 * (1.05 * count * e1 / e0 - 1), 2),
    ("of which from claim size, %", 16.6, 100 * (count * e1 / e0 - 1), 1),
], "Ex 5.10 to 5.12: 10% inflation on the layer above 5,000")
```

```text
                                    book  prospicio
Ex 5.10 e(5,000)                   8,000      8,000
e(5,000) after inflation           8,300      8,300
excess severity trend, %            3.75       3.75
layer 4K xs 5K severity, trended   2,701      2,699
layer severity trend, %             1.27       1.22
Ex 5.11 excess count factor       1.1241     1.1241
Ex 5.12 excess count trend, %      18.03      18.03
excess loss trend, %               22.46      22.46
of which from claim size, %         16.6       16.6
```

The monograph's 2,701 comes from rounded limited means. prospicio's 2,699 is
exact, so its trend is 1.22%, not 1.27%. The same split for an exponential with
mean 1,000 and a layer of 2,000 above 1,000 (P5.1), and for the lognormal of
Ex 5.8 (P5.21, P5.22):

```python
x = Gamma(1.0, 1000.0)
s = x.survival(1000.0)
mean = x.layer(2000.0, 1000.0) / s
sev = t * x.layer(2000.0 / t, 1000.0 / t) / x.survival(1000.0 / t) / mean - 1
cnt = x.survival(1000.0 / t) / s - 1
rows = [("P5.1 layer mean per excess claim", 865, mean, 0),
        ("its variance", 440_343, x.layer_second_moment(2000.0, 1000.0) / s - mean**2, 0),
        ("excess claims, of 20", 7.4, 20 * s, 1),
        ("severity trend, %", 6.6, 100 * sev, 1),
        ("count trend, %", 9.5, 100 * cnt, 1),
        ("loss trend, %", 16.7, 100 * ((1 + sev) * (1 + cnt) - 1), 1)]
ln = Lognormal(5.9809, 1.8)
for a, book in [(3000.0, 8.9), (8000.0, 11.5)]:
    rows.append((f"P5.21 count trend above {a:,.0f}, %", book,
                 100 * (survival(ln, a / t) / survival(ln, a) - 1), 1))
for width, book in [(2000.0, 9.5), (5000.0, 10.2)]:
    rows.append((f"P5.22 loss trend, layer to {3000 + width:,.0f}, %", book,
                 100 * (t * ln.layer(width / t, 3000.0 / t) / ln.layer(width, 3000.0) - 1), 1))
book_table(rows, "P5.1, P5.21 and P5.22: severity, count and loss trends", "10% inflation")
```

```text
                                        book  prospicio
P5.1 layer mean per excess claim         865        865
its variance                         440,343    440,343
excess claims, of 20                     7.4        7.4
severity trend, %                        6.6        6.6
count trend, %                           9.5        9.5
loss trend, %                           16.7       16.7
P5.21 count trend above 3,000, %         8.9        8.9
P5.21 count trend above 8,000, %        11.5       11.5
P5.22 loss trend, layer to 5,000, %      9.5        9.5
P5.22 loss trend, layer to 8,000, %     10.2       10.2
```

A layer with an upper limit has less leverage than an unlimited excess cover,
and a narrower layer has less again, because the limit caps the growth of
claims already in the layer. Compare P5.22's 9.5% with the 8.9% count growth of
P5.21 alone.

With a Poisson count, the aggregate layer loss's moments follow from the
layer's limited moments. Ex 5.13 takes 15 expected ground-up claims and the
layer of 5,000 above 3,000 of the same lognormal:

```python
lam, a, w = 15.0, 3000.0, 5000.0
book_table([("E[S]", 5_775, lam * ln.layer(w, a), 0),
            ("Var[S]", 24_178_800, lam * Truncated(ln).layer_second_moment(w, a), 0),
            ("expected layer claims", 1.9536, lam * survival(ln, a), 4)],
           "Ex 5.13: the aggregate loss of the layer 5,000 xs 3,000")
```

```text
                             book   prospicio
E[S]                        5,775       5,785
Var[S]                 24,178,800  24,117,273
expected layer claims      1.9536      1.9536
```

The monograph again uses the rounded limited values of Ex 5.8:
15 × (1,276 − 891) = 5,775. The expected count needs no difference and agrees.

## 8. Increased limit factors

An increased limit factor I(l) is the premium for a limit l over the premium
for the basic limit b. Without expenses and risk load, it is E[X; l]/E[X; b],
which `prospicio.pricing.ilf` returns. The monograph adds allocated loss
adjustment expense (ALAE) in two ways: a fixed amount e per claim, giving
(E[X; l] + e)/(E[X; b] + e), or a proportion u of the indemnity, which cancels
out of the ratio. Ex 6.3 uses a lognormal (7.000; 2.400), a basic limit of
100,000, e = 2,200 or u = 20%, a frequency of 0.2 and a 35% variable expense
ratio:

```python
ln = Lognormal(7.0, 2.4)
b = 100_000.0
limits = [100e3, 500e3, 1e6, 2e6, 3e6, 4e6, 5e6]
grid_table("limit", limits, [
    ("E[X; l]", [8896, 13626, 15345, 16738, 17390, 17782, 18048], [ln.lev(l) for l in limits], 0),
    ("I(l), fixed e", [1.0, 1.4263, 1.5812, 1.7067, 1.7655, 1.8008, 1.8248],
     [(ln.lev(l) + 2200) / (ln.lev(b) + 2200) for l in limits], 4),
    ("I(l), proportional u", [1.0, 1.5317, 1.7249, 1.8815, 1.9548, 1.9989, 2.0288],
     [ilf(ln, l, b) for l in limits], 4),
], "Ex 6.3: limited means and increased limit factors")
```

```text
                 E[X; l]  I(l), fixed e  I(l), proportional u
100,000            8,896         1.0000                1.0000
100,000, book      8,896         1.0000                1.0000
500,000           13,626         1.4262                1.5316
500,000, book     13,626         1.4263                1.5317
1,000,000         15,345         1.5812                1.7249
1,000,000, book   15,345         1.5812                1.7249
2,000,000         16,738         1.7067                1.8815
2,000,000, book   16,738         1.7067                1.8815
3,000,000         17,390         1.7655                1.9548
3,000,000, book   17,390         1.7655                1.9548
4,000,000         17,782         1.8008                1.9989
4,000,000, book   17,782         1.8008                1.9989
5,000,000         18,048         1.8248                2.0288
5,000,000, book   18,048         1.8248                2.0288
```

Ex 6.4 prices the layers between limits:

```python
basic_e = 0.2 * (ln.lev(b) + 2200) / (1 - 0.35)
basic_u = 0.2 * ln.lev(b) * 1.2 / (1 - 0.35)
rows = [("Ex 6.3 basic premium, e", 3_414, basic_e, 0),
        ("1M premium, e", 5_398, basic_e * (ln.lev(1e6) + 2200) / (ln.lev(b) + 2200), 0),
        ("basic premium, u", 3_285, basic_u, 0),
        ("1M premium, u", 5_666, basic_u * ilf(ln, 1e6, b), 0)]
for (lo, hi), book in zip([(1e6, 2e6), (2e6, 3e6), (3e6, 4e6), (4e6, 5e6)], [428, 201, 121, 82]):
    rows.append((f"Ex 6.4 layer {lo / 1e6:.0f}M to {hi / 1e6:.0f}M premium, e", book,
                 basic_e * (ln.layer(hi - lo, lo)) / (ln.lev(b) + 2200), 0))
book_table(rows, "Ex 6.3 and 6.4: premiums by limit and by layer")
```

```text
                                   book  prospicio
Ex 6.3 basic premium, e           3,414      3,414
1M premium, e                     5,398      5,399
basic premium, u                  3,285      3,285
1M premium, u                     5,666      5,666
Ex 6.4 layer 1M to 2M premium, e    428        429
Ex 6.4 layer 2M to 3M premium, e    201        201
Ex 6.4 layer 3M to 4M premium, e    121        121
Ex 6.4 layer 4M to 5M premium, e     82         82
```

The monograph forms its factors from limited means rounded to whole units,
hence the fourth-decimal differences. The 2M layer's 428 against 429 comes from
the factor 0.1255, which it rounds before multiplying. P6.5 repeats Ex 6.3 for a
lognormal (6.8; 2.6) with e = 2,500 or u = 25%:

```python
ln2 = Lognormal(6.8, 2.6)
limits2 = [100e3, 250e3, 500e3, 1e6, 2e6, 5e6]
grid_table("limit", limits2, [
    ("E[X; l]", [9178, 12548, 15180, 17702, 19968, 22404], [ln2.lev(l) for l in limits2], 0),
    ("I(l), fixed e", [1.0, 1.2885, 1.5139, 1.7299, 1.9240, 2.1325],
     [(ln2.lev(l) + 2500) / (ln2.lev(b) + 2500) for l in limits2], 4),
    ("I(l), proportional u", [1.0, 1.3671, 1.6539, 1.9288, 2.1756, 2.4410],
     [ilf(ln2, l, b) for l in limits2], 4),
], "P6.5: limited means and increased limit factors")
```

```text
                 E[X; l]  I(l), fixed e  I(l), proportional u
100,000            9,178         1.0000                1.0000
100,000, book      9,178         1.0000                1.0000
250,000           12,548         1.2885                1.3671
250,000, book     12,548         1.2885                1.3671
500,000           15,180         1.5139                1.6539
500,000, book     15,180         1.5139                1.6539
1,000,000         17,702         1.7299                1.9288
1,000,000, book   17,702         1.7299                1.9288
2,000,000         19,968         1.9240                2.1756
2,000,000, book   19,968         1.9240                2.1756
5,000,000         22,404         2.1325                2.4410
5,000,000, book   22,404         2.1325                2.4410
```

P6.7 uses a hybrid ALAE: 20% of each claim up to 750,000 and a flat 150,000
above it, so E_l[Y] = E[X; l] + 0.2 E[X; min(l, 750,000)]:

```python
def hybrid(l):
    return ln.lev(l) + 0.2 * ln.lev(min(l, 750e3))


limits3 = [100e3, 500e3, 750e3, 1e6, 2e6, 3e6, 4e6, 5e6]
grid_table("limit", limits3, [
    ("I(l)", [1.0, 1.5317, 1.6488, 1.7122, 1.8427, 1.9038, 1.9405, 1.9655],
     [hybrid(l) / hybrid(b) for l in limits3], 4),
], "P6.7: increased limit factors with a hybrid ALAE")
```

```text
             book  prospicio
100,000    1.0000     1.0000
500,000    1.5317     1.5316
750,000    1.6488     1.6488
1,000,000  1.7122     1.7123
2,000,000  1.8427     1.8427
3,000,000  1.9038     1.9038
4,000,000  1.9405     1.9405
5,000,000  1.9655     1.9654
```

## 9. Risk loads

Higher limits carry more uncertainty per unit of expected loss, so the monograph
adds a risk load ρ(l). It uses the aggregate loss variance per expected claim,
E[X²; l] + δ E[X; l]², where δ = Var[N]/E[N] − 1 is zero for a Poisson count.
The load is k times its square root (the standard deviation method) or k times
itself (the variance method). Ex 6.5 uses the standard deviation method with
k = 0.0277 and δ = 0 on Ex 6.3's lognormal with u = 20%:

```python
def sd_load(sev, l, k, delta):
    return k * math.sqrt(limited_second_moment(sev, l) + delta * sev.lev(l) ** 2)


def loaded_ilf(sev, l, base, k, delta, load=sd_load, u=0.2):
    def premium(x):
        return (1 + u) * sev.lev(x) + load(sev, x, k, delta)
    return premium(l) / premium(base)


loaded = [loaded_ilf(ln, l, b, 0.0277, 0.0) for l in limits]
grid_table("limit", limits, [
    ("load rho(l)", [627, 1473, 2013, 2663, 3090, 3410, 3668],
     [sd_load(ln, l, 0.0277, 0.0) for l in limits], 0),
    ("loaded I(l)", [1.0, 1.5770, 1.8074, 2.0128, 2.1197, 2.1897, 2.2407], loaded, 4),
], "Ex 6.5: the standard deviation load")
```

```text
                 load rho(l)  loaded I(l)
100,000                  627       1.0000
100,000, book            627       1.0000
500,000                1,473       1.5770
500,000, book          1,473       1.5770
1,000,000              2,013       1.8074
1,000,000, book        2,013       1.8074
2,000,000              2,663       2.0127
2,000,000, book        2,663       2.0128
3,000,000              3,090       2.1197
3,000,000, book        3,090       2.1197
4,000,000              3,410       2.1897
4,000,000, book        3,410       2.1897
5,000,000              3,668       2.2407
5,000,000, book        3,668       2.2407
```

Averaged over a mix of policies by limit:

```python
mix = [0.15, 0.10, 0.30, 0.20, 0.10, 0.10, 0.05]
plain = sum(w * ilf(ln, l, b) for w, l in zip(mix, limits))
risky = sum(w * x for w, x in zip(mix, loaded))
book_table([("average I, no load", 1.6938, plain, 4),
            ("average I, loaded", 1.7955, risky, 4),
            ("the load's effect, %", 6.0, 100 * (risky / plain - 1), 1)],
           "Ex 6.5: the average increased limit factor")
```

```text
                        book  prospicio
average I, no load    1.6938     1.6938
average I, loaded     1.7955     1.7954
the load's effect, %     6.0        6.0
```

P6.8 repeats Ex 6.5 for a Pareto (3; 6,000) with k = 0.5 and δ = 0.1:

```python
p = shifted_pareto(3.0, 6000.0)
limits4 = [1000, 2000, 3000, 4000, 5000, 7500, 10000]
grid_table("limit", limits4, [
    ("I(l), no load", [1.0, 1.6490, 2.0940, 2.4123, 2.6478, 3.0247, 3.2392],
     [ilf(p, l, 1000) for l in limits4], 4),
    ("I(l), loaded", [1.0, 1.6787, 2.1645, 2.5267, 2.8055, 3.2805, 3.5759],
     [loaded_ilf(p, l, 1000, 0.5, 0.1) for l in limits4], 4),
], "P6.8: the load on a Pareto (3; 6,000)")
```

```text
              I(l), no load  I(l), loaded
1,000                1.0000        1.0000
1,000, book          1.0000        1.0000
2,000                1.6490        1.6787
2,000, book          1.6490        1.6787
3,000                2.0940        2.1645
3,000, book          2.0940        2.1645
4,000                2.4123        2.5267
4,000, book          2.4123        2.5267
5,000                2.6478        2.8055
5,000, book          2.6478        2.8055
7,500                3.0247        3.2805
7,500, book          3.0247        3.2805
10,000               3.2392        3.5759
10,000, book         3.2392        3.5759
```

P6.10 uses the variance method (k = 0.0000005, δ = 0) on a Pareto with
α = 0.78, so heavy that its mean is infinite. The basic limit is 1,000,000. The
problem asks for the factor of each additional million of limit:

```python
def var_load(sev, l, k, delta):
    return k * (limited_second_moment(sev, l) + delta * sev.lev(l) ** 2)


p = shifted_pareto(0.78, 100.0)
millions = [m * 1e6 for m in range(1, 11)]
f = [loaded_ilf(p, l, 1e6, 5e-7, 0.0, load=var_load, u=0.0) for l in millions]
grid_table("limit", millions, [
    ("I(l)", [1.0, 1.3858, 1.7458, 2.1014, 2.4585, 2.8194, 3.1849, 3.5553, 3.9309, 3.4114], f, 4),
    ("layer factor", [NAN] * 3 + [0.3556, 0.3571, 0.3609, 0.3655, 0.3705, 0.3755, 0.3806],
     [NAN] * 3 + [hi - lo for lo, hi in zip(f[2:], f[3:])], 4),
], "P6.10: the variance load on a Pareto with infinite mean",
   "The layer factor is I(l) less I at the limit one million below")
```

```text
                    I(l)  layer factor
1,000,000         1.0000
1,000,000, book   1.0000
2,000,000         1.3858
2,000,000, book   1.3858
3,000,000         1.7458
3,000,000, book   1.7458
4,000,000         2.1014        0.3556
4,000,000, book   2.1014        0.3556
5,000,000         2.4585        0.3571
5,000,000, book   2.4585        0.3571
6,000,000         2.8194        0.3609
6,000,000, book   2.8194        0.3609
7,000,000         3.1849        0.3655
7,000,000, book   3.1849        0.3655
8,000,000         3.5553        0.3705
8,000,000, book   3.5553        0.3705
9,000,000         3.9309        0.3755
9,000,000, book   3.9309        0.3755
10,000,000        4.3114        0.3806
10,000,000, book  3.4114        0.3806
```

P6.10's last printed factor, 3.4114, is a misprint for 4.3114 (its own layer
factor of 0.3806 on top of 3.9309 gives 4.3115). The layer factors show the
point: the variance load grows so fast that each additional million costs more
than the one below it, although fewer claims reach it. This is a known weakness
of the variance method for heavy tails.

## 10. Deductibles

A straight deductible d removes min(X, d) from every claim. The monograph's
**credit** C(d) is the share of the basic-limit premium it eliminates:
(E[X; d] + e F(d)) / (E[X; b] + e), with e = 0 when ALAE is a proportion. This
loss elimination ratio is against the basic limit. prospicio's
`loss_elimination_ratio(sev, d)` is against the unlimited mean, E[X; d]/E[X]. A
**franchise** deductible pays the whole claim once it exceeds d, so it
eliminates only the claims below d. A **diminishing** deductible shrinks from d
at a claim of d to nothing at a claim of D, and eliminates
E[X; d] − d (E[X; D] − E[X; d])/(D − d) per claim. Ex 6.7 to 6.9 use Ex 6.3's
lognormal, u = 20%, a frequency of 0.0005 per exposure and D = d + 1,000:

```python
base = ln.lev(b)
deds = [1000, 2000, 3000, 4000, 5000, 10000]
straight = [ln.lev(d) / base for d in deds]
franchise = [(ln.lev(d) - d * survival(ln, d)) / base for d in deds]
diminishing = [(ln.lev(d) - d * (ln.lev(d + 1000) - ln.lev(d)) / 1000) / base for d in deds]
grid_table("deductible", deds, [
    ("C(d), straight", [0.0741, 0.1249, 0.1661, 0.2016, 0.2328, 0.3534], straight, 4),
    ("frequency x 10^6", [258, 201, 169, 147, 132, 89], [1e6 * 0.0005 * survival(ln, d) for d in deds], 0),
    ("pure premium", [4.942, 4.671, 4.451, 4.262, 4.095, 3.451],
     [0.0005 * 1.2 * (base - ln.lev(d)) for d in deds], 3),
], "Ex 6.7: the straight deductible")
```

```text
              C(d), straight  frequency x 10^6  pure premium
1,000                 0.0741               258         4.942
1,000, book           0.0741               258         4.942
2,000                 0.1249               201         4.671
2,000, book           0.1249               201         4.671
3,000                 0.1661               169         4.451
3,000, book           0.1661               169         4.451
4,000                 0.2015               147         4.262
4,000, book           0.2016               147         4.262
5,000                 0.2328               132         4.095
5,000, book           0.2328               132         4.095
10,000                0.3535                89         3.451
10,000, book          0.3534                89         3.451
```

```python
grid_table("deductible", deds, [
    ("Ex 6.8 C(d), franchise", [0.0162, 0.0347, 0.0523, 0.0690, 0.0846, 0.1528], franchise, 4),
    ("Ex 6.9 C(d), diminishing", [0.0233, 0.0424, 0.0599, 0.0766, 0.0917, NAN],
     diminishing[:5] + [NAN], 4),
], "Franchise and diminishing deductibles")
```

```text
              Ex 6.8 C(d), franchise  Ex 6.9 C(d), diminishing
1,000                         0.0162                    0.0234
1,000, book                   0.0162                    0.0233
2,000                         0.0347                    0.0424
2,000, book                   0.0347                    0.0424
3,000                         0.0523                    0.0599
3,000, book                   0.0523                    0.0599
4,000                         0.0689                    0.0763
4,000, book                   0.0690                    0.0766
5,000                         0.0846                    0.0917
5,000, book                   0.0846                    0.0917
10,000                        0.1528
10,000, book                  0.1528
```

```python
print(f"Ex 6.7, 1M limit with a 2,000 deductible: book 5,527, prospicio "
      f"{basic_u * (loaded_ilf(ln, 1e6, b, 0.0277, 0.0) - straight[1]):,.0f}")
print(f"prospicio's loss_elimination_ratio(2,000) = E[X; 2,000] / E[X] = "
      f"{loss_elimination_ratio(ln, 2000.0):.4f}")
```

```text
Ex 6.7, 1M limit with a 2,000 deductible: book 5,527, prospicio 5,527
prospicio's loss_elimination_ratio(2,000) = E[X; 2,000] / E[X] = 0.0569
```

The two ratios differ by the factor E[X]/E[X; b], more than two for a
distribution this heavy. Table 6.7 prints the franchise credit at 5,000 as
0.0864, but Table 6.6 has 0.0846, which prospicio gives. The diminishing
credits that differ in the fourth decimal are rounding again: the formula
multiplies a difference of limited means by d, so the whole-unit rounding grows
up to fourfold at d = 4,000.

P6.13 and P6.14 use twelve actual losses, a 20,000 or 12,000 limit and a 2,000
deductible. The loss elimination ratio (LER) here is the share of the unlimited
losses that the insurer does not pay, so with the lower limit it also counts
what the limit eliminates:

```python
losses = [1000, 1550, 1700, 2200, 2500, 3000, 5200, 9000, 11000, 12500, 15000, 19800]
total = sum(losses)
rows = []
for name, limit, book in [("P6.13", 20000, (62_200, 0.2635, 80_200, 0.0503)),
                          ("P6.14", 12000, (50_900, 0.3973, 68_900, 0.1841))]:
    paid_straight = sum(max(min(x, limit) - 2000, 0) for x in losses)
    paid_franchise = sum(min(x, limit) for x in losses if x > 2000)
    rows += [(f"{name} straight: insurer pays", book[0], paid_straight, 0),
             (f"{name} straight: LER", book[1], 1 - paid_straight / total, 4),
             (f"{name} franchise: insurer pays", book[2], paid_franchise, 0),
             (f"{name} franchise: LER", book[3], 1 - paid_franchise / total, 4)]
book_table(rows, "P6.13 and P6.14: deductibles on twelve actual losses")
```

```text
                                 book  prospicio
P6.13 straight: insurer pays   62,200     62,200
P6.13 straight: LER            0.2635     0.2635
P6.13 franchise: insurer pays  80,200     80,200
P6.13 franchise: LER           0.0503     0.0503
P6.14 straight: insurer pays   50,900     50,900
P6.14 straight: LER            0.3973     0.3973
P6.14 franchise: insurer pays  68,900     68,900
P6.14 franchise: LER           0.1841     0.1841
```

A fixed ALAE per claim raises the credit: a deductible removes the claims below
it and their expense with them. P6.16 reruns Ex 6.7 and 6.8 with e = 500 and
u = 0:

```python
e = 500.0
cred = [(ln.lev(d) + e * ln.cdf(d)) / (base + e) for d in deds]
pp = [0.0005 * (base - ln.lev(d) + e * survival(ln, d)) for d in deds]
fr = [(ln.lev(d) - d * survival(ln, d) + e * ln.cdf(d)) / (base + e) for d in deds]
grid_table("deductible", deds, [
    ("C(d), straight", [0.0959, 0.1501, 0.1926, 0.2283, 0.2596, 0.3783], cred, 4),
    ("pure premium", [4.247, 3.993, 3.973, 3.625, 3.478, 2.921], pp, 3),
    ("C(d), franchise", [0.0411, 0.0647, 0.0848, 0.1028, 0.1193, 0.1884], fr, 4),
], "P6.16: a fixed ALAE of 500 per claim")
```

```text
              C(d), straight  pure premium  C(d), franchise
1,000                 0.0960         4.247           0.0411
1,000, book           0.0959         4.247           0.0411
2,000                 0.1501         3.993           0.0647
2,000, book           0.1501         3.993           0.0647
3,000                 0.1925         3.794           0.0848
3,000, book           0.1926         3.973           0.0848
4,000                 0.2283         3.625           0.1028
4,000, book           0.2283         3.625           0.1028
5,000                 0.2596         3.478           0.1193
5,000, book           0.2596         3.478           0.1193
10,000                0.3784         2.920           0.1884
10,000, book          0.3783         2.921           0.1884
```

P6.17 uses a Pareto (3; 10,000), a basic limit of 5,000, a frequency of 0.005
and e = 250:

```python
p = shifted_pareto(3.0, 10000.0)
deds2 = [0, 250, 500, 750, 1000]
base2 = p.lev(5000.0) + 250.0
grid_table("deductible", deds2, [
    ("C(d), straight", [0.0, 0.0855, 0.1648, 0.2385, 0.3071],
     [(p.lev(d) + 250 * p.cdf(d)) / base2 for d in deds2], 4),
    ("pure premium", [15.139, 13.845, 12.644, 11.528, 10.489],
     [0.005 * (base2 - p.lev(d) - 250 * p.cdf(d)) for d in deds2], 3),
    ("C(d), franchise", [NAN, 0.0088, 0.0221, 0.0391, 0.0590],
     [NAN] + [(p.lev(d) - d * p.survival(d) + 250 * p.cdf(d)) / base2 for d in deds2[1:]], 4),
], "P6.17: a Pareto (3; 10,000) with a fixed ALAE of 250")
```

```text
             C(d), straight  pure premium  C(d), franchise
0                    0.0000        15.139
0, book              0.0000        15.139
250                  0.0855        13.845           0.0088
250, book            0.0855        13.845           0.0088
500                  0.1648        12.644           0.0221
500, book            0.1648        12.644           0.0221
750                  0.2385        11.528           0.0391
750, book            0.2385        11.528           0.0391
1,000                0.3071        10.489           0.0590
1,000, book          0.3071        10.489           0.0590
```

P6.21 counts the claims, out of 500, that Ex 6.7's deductibles eliminate, now
and after 10% inflation:

```python
grid_table("deductible", deds[:5], [
    ("claims removed", [242, 299, 331, 353, 368], [500 * ln.cdf(d) for d in deds[:5]], 0),
    ("after inflation", [234, 292, 324, 346, 362], [500 * ln.cdf(d / 1.1) for d in deds[:5]], 0),
    ("share removed, %", [7.1, 12.0, 16.1, 19.5, 22.6],
     [100 * ln.lev(d / 1.1) / ln.lev(b / 1.1) for d in deds[:5]], 1),
], "P6.21: claims removed by a fixed deductible")
```

```text
             claims removed  after inflation  share removed, %
1,000                   242              234               7.1
1,000, book             242              234               7.1
2,000                   299              292              12.0
2,000, book             299              292              12.0
3,000                   331              324              16.1
3,000, book             331              324              16.1
4,000                   353              346              19.5
4,000, book             353              346              19.5
5,000                   368              362              22.6
5,000, book             368              362              22.6
```

P6.16's printed pure premium at 3,000, 3.973, is out of sequence: it makes the
3,000 deductible almost no cheaper than the 2,000 one. prospicio's 3.794 fits
the pattern, and the printed figure has two middle digits transposed.

Inflation erodes a fixed deductible: each year it removes fewer claims and a
smaller share of the loss. This is section 7's leveraging effect seen from
below. Ex 6.10 measures it for a Pareto (2; 8,000), a 500 deductible, a
frequency of 0.25, e = 50 and 5% inflation (of claims and of e), without a
limit and with a 5,000 limit:

```python
def pure_premium(sev, top, d, e, phi=0.25):
    capped = sev.mean() if top is None else sev.lev(top)
    return phi * (capped - sev.lev(d) + survival(sev, d) * e)


now, later = shifted_pareto(2.0, 8000.0), shifted_pareto(2.0, 8400.0)
rows = []
for top, book, dp in [(None, (1_893, 1_994, 5.3), 0), (5000.0, (662.7, 677.3, 2.2), 1)]:
    p0, p1 = pure_premium(now, top, 500.0, 50.0), pure_premium(later, top, 500.0, 52.5)
    label = "no limit" if top is None else "5,000 limit"
    rows += [(f"Ex 6.10 {label}: pure premium", book[0], p0, dp),
             ("after 5% inflation", book[1], p1, dp),
             ("effective trend, %", book[2], 100 * (p1 / p0 - 1), 1)]
book_table(rows, "Ex 6.10: inflation on a deductible of 500")
```

```text
                                    book  prospicio
Ex 6.10 no limit: pure premium     1,893      1,893
after 5% inflation                 1,994      1,994
effective trend, %                   5.3        5.3
Ex 6.10 5,000 limit: pure premium  662.7      662.7
after 5% inflation                 677.3      677.3
effective trend, %                   2.2        2.2
```

A limit on top cuts the trend because it absorbs more inflation than the
deductible adds.

## 11. What agreed and what did not

Every number agrees with prospicio to its printed digits, or differs for a
reason found in the monograph itself:

- **Rounding carried into differences** (Ex 5.8, 5.10, 5.13, some ILFs, Ex 6.4):
  the monograph rounds limited means to whole units before it subtracts. Its
  rounded inputs reproduce its figures.
- **Slips**: Ex 2.7's scale, P6.10's last factor, Table 6.7's franchise credit,
  P6.16's 3.973.
- **Solver results**: Ex 2.8's and Ex 5.3's chi-squares and P5.23's parameters.
  Ex 2.10, 2.11, P2.16, P2.39 and P2.43 reproduce to four decimals by the same
  methods.

Where to go next:

- [Severity and frequency](severity-and-frequency.md): fitting a severity in
  practice.
- [Aggregate distributions](aggregate-distributions.md): compound distributions.
- [Experience and exposure rating](experience-and-exposure-rating.md): increased
  limit factors, exposure curves and layer rating.
- [Claim counts and aggregate claims](distributions-for-actuaries-counts-aggregates.md):
  chapters 3 and 4 of the monograph.
