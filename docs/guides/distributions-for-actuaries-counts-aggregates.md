# Claim counts and aggregate claims: Bahnemann's *Distributions for Actuaries* reproduced

<!-- site: execute -->

This guide recomputes the worked examples and problem answers of chapters 3 and
4 and section 6.4 of CAS Monograph 2, *Distributions for Actuaries* (David
Bahnemann, 2015): counts, aggregate claims and aggregate limits. The
[first guide](distributions-for-actuaries-claim-size.md) covers chapters 2, 5
and 6 and explains the conventions. Each table shows the monograph's printed
value beside prospicio's. Read the monograph for the derivations.

## 1. Translating the parameters

The monograph writes the negative binomial with mean ν and gamma shape α
(Var[N] = ν + ν²/α), later with the **contagion** γ = 1/α (Var[N] = ν + γν²).
prospicio's `NegativeBinomial(r, beta)` has mean rβ and variance rβ(1 + β).
`claim_count(mean, dispersion)` takes the variance-to-mean ratio:

| Monograph | prospicio |
|---|---|
| Poisson λ | `Poisson(λ)` |
| binomial (m, p) | `Binomial(m, p)` |
| negative binomial (α, ν) | `NegativeBinomial(r=α, beta=ν/α)` |
| negative binomial, mean ν and contagion γ | `claim_count(ν, 1 + γν)`, or `NegativeBinomial(1/γ, γν)` |
| Poisson mixed by a gamma with mean ν and shape α | `Count.mixed_poisson(ν, cv=1/√α, mixing="gamma")` |
| fitted by the method of moments | `NegativeBinomial.from_mean_variance(m, v)` |
| geometric with mean (1 − p)/p | `Count.geometric(mean)` |
| any count on 0, 1, 2, ... | `Count.empirical(probs)` |

```python
import math

import pandas as pd
from great_tables import GT

from prospicio.aggregate import compound_moments, fft, panjer, simulate_events
from prospicio.distributions import (
    Binomial, Count, Gamma, Grid, Lognormal, NegativeBinomial, Poisson, claim_count,
)


def compare(rows, title, subtitle=None):
    """Book against prospicio: rows of (label, book, prospicio) or (label, book, prospicio, decimals)."""
    frame = pd.DataFrame([(r[0], r[1], r[2], r[3] if len(r) > 3 else 4) for r in rows],
                         columns=["row", "book", "prospicio", "decimals"])
    table = GT(frame.drop(columns="decimals"), rowname_col="row").tab_header(title, subtitle)
    for places in sorted(set(frame["decimals"])):
        table = table.fmt_number(["book", "prospicio"],
                                 rows=list(frame.index[frame["decimals"] == places]), decimals=places)
    return table


def chi_square(observed, probs):
    n = sum(observed)
    return sum((o - n * p) ** 2 / (n * p) for o, p in zip(observed, probs))
```

## 2. The Poisson and the binomial

A Poisson count is the limit of a binomial with many trials and a small chance
in each. Ex 3.1 counts days with a claim (0.3% a day). Ex 3.2 and P3.4 use
annual rates of 0.04 and 0.10:

```python
rows = [("Ex 3.1, 30 days, Pr{N = 0}", 0.9138, Binomial(30, 0.003).pmf(0)),
        ("Ex 3.1, 30 days, Pr{N = 1}", 0.0825, Binomial(30, 0.003).pmf(1)),
        ("Ex 3.1, 365 days, Pr{N = 2}", 0.2009, Binomial(365, 0.003).pmf(2))]
for t, book in [(1.0, (0.9608, 0.0384, 0.0008)), (1.5, (0.9418, 0.0565, 0.0017))]:
    for n in range(3):
        rows.append((f"Ex 3.2, {t} years, Pr{{N = {n}}}", book[n], Poisson(0.04 * t).pmf(n)))
for months, book in [(6, 0.0012), (15, 0.0069), (24, 0.0164)]:
    rows.append((f"P3.4, {months} months, Pr{{N = 2}}", book, Poisson(0.10 * months / 12).pmf(2)))
compare(rows, "Ex 3.1, Ex 3.2 and P3.4: binomial and Poisson probabilities")
```

```text
                                book  prospicio
Ex 3.1, 30 days, Pr{N = 0}    0.9138     0.9138
Ex 3.1, 30 days, Pr{N = 1}    0.0825     0.0825
Ex 3.1, 365 days, Pr{N = 2}   0.2009     0.2009
Ex 3.2, 1.0 years, Pr{N = 0}  0.9608     0.9608
Ex 3.2, 1.0 years, Pr{N = 1}  0.0384     0.0384
Ex 3.2, 1.0 years, Pr{N = 2}  0.0008     0.0008
Ex 3.2, 1.5 years, Pr{N = 0}  0.9418     0.9418
Ex 3.2, 1.5 years, Pr{N = 1}  0.0565     0.0565
Ex 3.2, 1.5 years, Pr{N = 2}  0.0017     0.0017
P3.4, 6 months, Pr{N = 2}     0.0012     0.0012
P3.4, 15 months, Pr{N = 2}    0.0069     0.0069
P3.4, 24 months, Pr{N = 2}    0.0164     0.0164
```

Ex 3.3 fits a Poisson at the sample mean to 1,000 policies (868 with no claim,
118 with one, 11 with two, 2 with three, 1 with four). A chi-square test on the
cells 0, 1 and 2 or more accepts it. Ex 3.8, P3.3, P3.30 and P3.31 add
independent Poissons:

```python
policies = [868, 118, 11, 2, 1]
lam = sum(k * n for k, n in enumerate(policies)) / sum(policies)
p = Poisson(lam)
rows = [("Ex 3.3, rate", 0.15, lam, 2)]
for k, book in enumerate([0.8607, 0.1291, 0.0097, 0.0005]):
    rows.append((f"Ex 3.3, Pr{{N = {k}}}", book, p.pmf(k)))
rows += [("Ex 3.3, chi-square", 2.43, chi_square([868, 118, 14], [p.pmf(0), p.pmf(1), 1 - p.cdf(1)]), 2),
         ("Ex 3.8, Pr{N >= 5}, 100 x 0.035", 0.2746, 1 - Poisson(3.5).cdf(4))]
for rate, book in [(2.5, 0.7576), (6.25, 0.1303), (2.0, 0.8571), (4.0, 0.4335)]:
    rows.append((f"P3.3, Pr{{N <= 3}}, rate {rate}", book, Poisson(rate).cdf(3)))
rows += [("P3.30, Pr{N >= 6}, 20 cars x 0.30", 0.5543, 1 - Poisson(6.0).cdf(5)),
         ("P3.31, policies with a claim, of 8,000", 624, 8000 * (1 - Poisson(650 / 8000).pmf(0)), 0)]
compare(rows, "Ex 3.3, Ex 3.8, P3.3, P3.30 and P3.31: fitting and using a Poisson")
```

```text
                                          book  prospicio
Ex 3.3, rate                              0.15       0.15
Ex 3.3, Pr{N = 0}                       0.8607     0.8607
Ex 3.3, Pr{N = 1}                       0.1291     0.1291
Ex 3.3, Pr{N = 2}                       0.0097     0.0097
Ex 3.3, Pr{N = 3}                       0.0005     0.0005
Ex 3.3, chi-square                        2.43       2.45
Ex 3.8, Pr{N >= 5}, 100 x 0.035         0.2746     0.2746
P3.3, Pr{N <= 3}, rate 2.5              0.7576     0.7576
P3.3, Pr{N <= 3}, rate 6.25             0.1303     0.1303
P3.3, Pr{N <= 3}, rate 2.0              0.8571     0.8571
P3.3, Pr{N <= 3}, rate 4.0              0.4335     0.4335
P3.30, Pr{N >= 6}, 20 cars x 0.30       0.5543     0.5543
P3.31, policies with a claim, of 8,000     624        624
```

The monograph rounds the expected counts to one decimal before it sums them,
which gives its chi-square of 2.43 against 2.45.

## 3. Parameter uncertainty: mixed Poissons

When each policy has its own rate, the portfolio count is a **mixed Poisson**.
Its variance is the mean rate plus the variance of the rates, so it has more
zeros and a heavier tail than a Poisson of the same mean. Ex 3.4 mixes four
rates. Ex 3.5 draws the rate from an exponential with mean 0.25: a geometric
count:

```python
rates, weights = [0.10, 0.35, 0.70, 1.40], [0.20, 0.40, 0.30, 0.10]
mixed = [sum(w * Poisson(r).pmf(n) for r, w in zip(rates, weights)) for n in range(6)]
mean = sum(w * r for r, w in zip(rates, weights))
var = mean + sum(w * r * r for r, w in zip(rates, weights)) - mean**2
geometric = Count.mixed_poisson(0.25, 1.0, mixing="gamma")   # exponential: cv 1

pmf = pd.DataFrame({
    "n": [f"N = {n}" for n in range(6)],
    "mixed_book": [0.6365, 0.2556, 0.0788, 0.0218, 0.0056, 0.0013], "mixed": mixed,
    "poisson_book": [0.6005, 0.3063, 0.0781, 0.0133, 0.0017, 0.0002],
    "poisson": [Poisson(mean).pmf(n) for n in range(6)],
    "geo_book": [0.8, 0.16, 0.032, 0.0064, 0.0013, 0.0003],
    "geo_mixed": [geometric.pmf(n) for n in range(6)],
    "geo": [Count.geometric(0.25).pmf(n) for n in range(6)],
})
(GT(pmf, rowname_col="n")
 .tab_header("Ex 3.4 and Ex 3.5: probability of N claims")
 .tab_spanner("Ex 3.4, mixed Poisson", ["mixed_book", "mixed"])
 .tab_spanner("Ex 3.4, Poisson with mean 0.51", ["poisson_book", "poisson"])
 .tab_spanner("Ex 3.5, exponential mixing", ["geo_book", "geo_mixed", "geo"])
 .fmt_number(["mixed_book", "mixed", "poisson_book", "poisson", "geo_book", "geo_mixed", "geo"], decimals=4)
 .cols_label(mixed_book="book", mixed="prospicio", poisson_book="book", poisson="prospicio",
             geo_book="book", geo_mixed="prospicio", geo="geometric"))
```

```text
         book  prospicio    book  prospicio    book  prospicio  geometric
N = 0  0.6365     0.6365  0.6005     0.6005  0.8000     0.8000     0.8000
N = 1  0.2556     0.2556  0.3063     0.3063  0.1600     0.1600     0.1600
N = 2  0.0788     0.0788  0.0781     0.0781  0.0320     0.0320     0.0320
N = 3  0.0218     0.0218  0.0133     0.0133  0.0064     0.0064     0.0064
N = 4  0.0056     0.0056  0.0017     0.0017  0.0013     0.0013     0.0013
N = 5  0.0013     0.0013  0.0002     0.0002  0.0003     0.0003     0.0003
```

Ex 3.4 against the Poisson with the same mean:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(range(6), [Poisson(mean).pmf(n) for n in range(6)], color="0.75",
       label="Poisson, mean 0.51")
ax.plot(range(6), mixed, marker="o", label="mixed Poisson, Ex 3.4")
ax.set_xlabel("number of claims N")
ax.set_ylabel("probability")
ax.set_title("Ex 3.4: the mixture against the Poisson with the same mean")
ax.legend()
```

Gamma rates give a negative binomial, the standard overdispersed count. P3.11
uses a gamma with shape 2 and scale 1, P3.12 one with mean 0.100 and variance
0.005. P3.13 compares a Poisson, a two-point mixture and an exponential mixture
of one mean:

```python
nb = NegativeBinomial(2.0, 1.0)
rows = [("Ex 3.4, Var[N]", 0.6439, var),
        ("Ex 3.5, Var[N]", 0.3125, geometric.variance()),
        ("P3.11, mean", 2, nb.mean(), 0),
        ("P3.11, variance", 4, nb.variance(), 0),
        ("P3.11, Pr{N <= 3}", 0.8125, nb.cdf(3))]
nb = NegativeBinomial.from_mean_variance(0.100, 0.100 + 0.005)  # Var[N] = E + Var[rate]
for n, book in enumerate([0.9070, 0.0864, 0.0062, 0.0004]):
    rows.append((f"P3.12, Pr{{N = {n}}}", book, nb.pmf(n)))
two_point_var = 0.75 + (0.75 * 0.6**2 + 0.25 * 1.2**2 - 0.75**2)
rows += [("P3.13, Var, Poisson 0.75", 0.7500, Poisson(0.75).variance()),
         ("P3.13, Var, two-point mixture", 0.8175, two_point_var),
         ("P3.13, Var, exponential mixture", 1.3125, Count.mixed_poisson(0.75, 1.0).variance())]
compare(rows, "Ex 3.4, Ex 3.5 and P3.11 to P3.13: variances and gamma mixtures")
```

```text
                                   book  prospicio
Ex 3.4, Var[N]                   0.6439     0.6439
Ex 3.5, Var[N]                   0.3125     0.3125
P3.11, mean                           2          2
P3.11, variance                       4          4
P3.11, Pr{N <= 3}                0.8125     0.8125
P3.12, Pr{N = 0}                 0.9070     0.9070
P3.12, Pr{N = 1}                 0.0864     0.0864
P3.12, Pr{N = 2}                 0.0062     0.0062
P3.12, Pr{N = 3}                 0.0004     0.0004
P3.13, Var, Poisson 0.75         0.7500     0.7500
P3.13, Var, two-point mixture    0.8175     0.8175
P3.13, Var, exponential mixture  1.3125     1.3125
```

## 4. The negative binomial and contagion

Ex 3.6 and P3.15 fit a negative binomial by moments to 5,000 auto liability and
10,000 policies. The sample variance exceeds the mean, so a Poisson is too thin.
A chi-square test on the cells 0, 1, 2 and 3 or more tests each fit:

```python
def nb_moments(counts):
    n = sum(counts)
    m1 = sum(k * c for k, c in enumerate(counts)) / n
    m2 = sum(k * k * c for k, c in enumerate(counts)) / n
    return m1, m2 - m1**2


rows = []
for name, counts, book in [
    ("Ex 3.6", [4429, 528, 39, 3, 1], (0.1238, 2.44285, 0.7753)),
    ("P3.15", [8956, 907, 120, 15, 2], (0.1200, 0.685714, 0.3288)),
]:
    mean, var = nb_moments(counts)
    nb = NegativeBinomial.from_mean_variance(mean, var)
    probs = [nb.pmf(0), nb.pmf(1), nb.pmf(2), 1 - nb.cdf(2)]
    observed = counts[:3] + [sum(counts[3:])]
    rows += [(f"{name}, mean", book[0], mean),
             (f"{name}, alpha = r", book[1], nb.r, 5),
             (f"{name}, chi-square", book[2], chi_square(observed, probs))]
    if name == "Ex 3.6":
        for k, book_pmf in enumerate([0.8862, 0.1044, 0.0087, 0.0006]):
            rows.append((f"Ex 3.6, Pr{{N = {k}}}", book_pmf, nb.pmf(k)))
compare(rows, "Ex 3.6 and P3.15: negative binomials fitted by moments")
```

```text
                       book  prospicio
Ex 3.6, mean         0.1238     0.1238
Ex 3.6, alpha = r   2.44285    2.44302
Ex 3.6, chi-square   0.7753     0.6464
Ex 3.6, Pr{N = 0}    0.8862     0.8862
Ex 3.6, Pr{N = 1}    0.1044     0.1044
Ex 3.6, Pr{N = 2}    0.0087     0.0087
Ex 3.6, Pr{N = 3}    0.0006     0.0006
P3.15, mean          0.1200     0.1200
P3.15, alpha = r    0.68571    0.68571
P3.15, chi-square    0.3288     0.3288
```

In Ex 3.6, α differs in the fourth decimal because the monograph rounds the
sample variance to 0.130074 (it is 0.13007356) before it divides by the small
excess over the mean. The fitted probabilities agree. Its chi-square of 0.7753
does not reproduce: with the printed probabilities, or with α = 2.44285, the
statistic is near 0.6 to 0.65, so the printed value probably comes from other
cells or expected counts.

**Contagion** is the monograph's other route to the negative binomial. If each
claim makes the next more likely, Var[N] = ν + γν², and a Pólya urn gives
r = 1/γ. For any count, γ = (Var[N] − E[N])/E[N]²: 0 for a Poisson, negative for
a binomial. P3.28 asks for mean 1 and γ = 0.2. Ex 3.9 and P3.29 add independent
policies. Means and variances add, so the contagion shrinks as the portfolio
grows:

```python
nb = claim_count(1.0, 1 + 0.2 * 1.0)       # dispersion Var/E = 1 + gamma * mean


def contagion(counts):
    mean = sum(c.mean() for c in counts)
    var = sum(c.variance() for c in counts)
    return (var - mean) / mean**2


rows = [(f"P3.28, Pr{{N = {n}}}", book, nb.pmf(n))
        for n, book in enumerate([0.4019, 0.3349, 0.1674, 0.0651, 0.0217, 0.0065])]
rows += [("Ex 3.9, contagion, 20 policies, nu 0.15, gamma 0.4", 0.020,
          contagion([claim_count(0.15, 1 + 0.4 * 0.15)] * 20), 3),
         ("P3.29, contagion, three lines", 0.1000,
          contagion([claim_count(10, 1 + 0.35 * 10), claim_count(25, 1 + 0.20 * 25), Poisson(5.0)]))]
compare(rows, "P3.28, Ex 3.9 and P3.29: contagion")
```

```text
                                                      book  prospicio
P3.28, Pr{N = 0}                                    0.4019     0.4019
P3.28, Pr{N = 1}                                    0.3349     0.3349
P3.28, Pr{N = 2}                                    0.1674     0.1674
P3.28, Pr{N = 3}                                    0.0651     0.0651
P3.28, Pr{N = 4}                                    0.0217     0.0217
P3.28, Pr{N = 5}                                    0.0065     0.0065
Ex 3.9, contagion, 20 policies, nu 0.15, gamma 0.4   0.020      0.020
P3.29, contagion, three lines                       0.1000     0.1000
```

For claims above an attachment, each claim stays with probability p. A Poisson λ
becomes Poisson pλ. A negative binomial keeps its γ and its mean becomes pν:
build the thinned count from them (section 7 of the first guide, Ex 5.9 and
5.11).

## 5. Aggregate claims and their moments

The year's total S sums N claims: E[S] = E[N]E[X] and
Var[S] = E[N]Var[X] + Var[N]E[X]². `compound_moments` returns the first three
moments, `fft` the whole distribution. Ex 4.1 and P4.1 are small discrete
examples:

```python
count = Count.empirical([0.6, 0.3, 0.1])
size = Grid(100.0, [0.0, 0.4, 0.5, 0.1])             # 100, 200, 300
agg, _ = fft(count, size, 16)
rows = [(f"Ex 4.1, F_S({100 * k})", book, agg.cdf(100.0 * k))
        for k, book in enumerate([0.6, 0.72, 0.886, 0.956, 0.989, 0.999, 1.0])]
rows += [("Ex 4.1, E[S]", 85, agg.mean(), 0), ("Ex 4.1, Var[S]", 15_055, agg.variance(), 0)]
compare(rows, "Ex 4.1: a small discrete aggregate")
```

```text
                    book  prospicio
Ex 4.1, F_S(0)    0.6000     0.6000
Ex 4.1, F_S(100)  0.7200     0.7200
Ex 4.1, F_S(200)  0.8860     0.8860
Ex 4.1, F_S(300)  0.9560     0.9560
Ex 4.1, F_S(400)  0.9890     0.9890
Ex 4.1, F_S(500)  0.9990     0.9990
Ex 4.1, F_S(600)  1.0000     1.0000
Ex 4.1, E[S]          85         85
Ex 4.1, Var[S]    15,055     15,055
```

```python
count = Count.empirical([0.20, 0.40, 0.25, 0.15])
size = Grid(500.0, [0.0, 0.1, 0.4, 0.3, 0.2])        # 500 to 2,000
agg, _ = fft(count, size, 16)
book = [0.2, 0.24, 0.4025, 0.5427, 0.6795, 0.758, 0.8418, 0.9047, 0.9507]
compare([(f"P4.1, F_S({500 * k:,})", b, agg.cdf(500.0 * k)) for k, b in enumerate(book)],
        "P4.1: the aggregate distribution")
```

```text
                    book  prospicio
P4.1, F_S(0)      0.2000     0.2000
P4.1, F_S(500)    0.2400     0.2400
P4.1, F_S(1,000)  0.4025     0.4025
P4.1, F_S(1,500)  0.5427     0.5427
P4.1, F_S(2,000)  0.6795     0.6795
P4.1, F_S(2,500)  0.7580     0.7580
P4.1, F_S(3,000)  0.8418     0.8418
P4.1, F_S(3,500)  0.9047     0.9047
P4.1, F_S(4,000)  0.9507     0.9507
```

Ex 4.2 has a Poisson 2.5 count and gamma claims (shape 3, scale 400). A sum of
n claims is a gamma with shape 3n, so the distribution is an exact series over
n. Table 4.1 is that series:

```python
lam, size = 2.5, Gamma(3.0, 400.0)
moments = compound_moments(Poisson(lam), size)


def poisson_gamma_cdf(s, lam, shape, scale, terms=400):
    n = Poisson(lam)
    if s <= 0:
        return n.pmf(0)
    return n.pmf(0) + sum(n.pmf(k) * Gamma(shape * k, scale).cdf(s) for k in range(1, terms))


S = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
book = [0.0821, 0.1096, 0.1867, 0.3755, 0.5613, 0.7152, 0.9013, 0.9723, 0.9934]
rows = [("Ex 4.2, E[S]", 3_000, moments.mean, 0),
        ("Ex 4.2, Var[S]", 4_800_000, moments.variance, 0),
        ("Ex 4.2, skewness", 0.9129, moments.skew)]
rows += [(f"Ex 4.2, F({s:,})", b, poisson_gamma_cdf(s, lam, 3.0, 400.0)) for s, b in zip(S, book)]
compare(rows, "Ex 4.2 and Table 4.1: Poisson 2.5 with gamma (3; 400) claims")
```

```text
                        book  prospicio
Ex 4.2, E[S]           3,000      3,000
Ex 4.2, Var[S]     4,800,000  4,800,000
Ex 4.2, skewness      0.9129     0.9129
Ex 4.2, F(0)          0.0821     0.0821
Ex 4.2, F(500)        0.1096     0.1096
Ex 4.2, F(1,000)      0.1867     0.1867
Ex 4.2, F(2,000)      0.3755     0.3755
Ex 4.2, F(3,000)      0.5613     0.5613
Ex 4.2, F(4,000)      0.7152     0.7152
Ex 4.2, F(6,000)      0.9013     0.9013
Ex 4.2, F(8,000)      0.9723     0.9723
Ex 4.2, F(10,000)     0.9934     0.9934
```

## 6. Approximations by matching moments

Actuaries once matched a distribution to the aggregate's mean, variance and,
where the family allows, skewness. The normal ignores skewness. The shifted gamma
matches three moments. `CompoundMoments.approximate` offers these and the gamma,
lognormal, shifted lognormal, normal power and Wilson–Hilferty. Table 4.3 is
Ex 4.2. Table 4.4 is far more skewed (Poisson 10, gamma with shape 0.05 and scale
6,000). P4.13 has a Poisson 8 with gamma shape 0.2 and scale 3,750:

```python
cases = [
    ("Table 4.3", 4, 2.5, 3.0, 400.0, [0, 2000, 4000, 6000, 8000],
     [0.0821, 0.3755, 0.7152, 0.9013, 0.9723], [0.0855, 0.3240, 0.6760, 0.9145, 0.9888],
     [0.0459, 0.3680, 0.7185, 0.9038, 0.9724]),
    ("Table 4.4", 5, 10.0, 0.05, 6000.0, [0, 2000, 6000, 10000, 18000],
     [0.00005, 0.5922, 0.8401, 0.9294, 0.9846], [0.2451, 0.4090, 0.7549, 0.9463, 0.9997],
     [0.1228, 0.5886, 0.8402, 0.9298, 0.9847]),
    ("P4.13", 4, 8.0, 0.2, 3750.0, [0, 3000, 9000, 15000, 27000],
     [0.0003, 0.3420, 0.7774, 0.9349, 0.9954], [0.1241, 0.2819, 0.7181, 0.9584, 1.0000],
     [0.0263, 0.3362, 0.7782, 0.9356, 0.9953]),
]
moment_rows = []
for name, _, lam, shape, scale, *_rest in cases:
    m = compound_moments(Poisson(lam), Gamma(shape, scale))
    moment_rows.append((name, m.mean, m.variance ** 0.5, m.skew))
(GT(pd.DataFrame(moment_rows, columns=["case", "mean", "sd", "skewness"]), rowname_col="case")
 .tab_header("Tables 4.3, 4.4 and P4.13: the aggregate's moments")
 .fmt_number(["mean", "sd"], decimals=0)
 .fmt_number("skewness", decimals=4))
```

```text
            mean     sd  skewness
Table 4.3  3,000  2,191    0.9129
Table 4.4  3,000  4,347    2.8293
P4.13      6,000  5,196    1.5877
```

```python
rows = []
for name, _, lam, shape, scale, S, exact, normal, sgamma in cases:
    m = compound_moments(Poisson(lam), Gamma(shape, scale))
    for i, s in enumerate(S):
        rows.append((f"{name}, s = {s:,}",
                     exact[i], poisson_gamma_cdf(s, lam, shape, scale),
                     normal[i], m.approximate("normal").cdf(s),
                     sgamma[i], m.approximate("shifted_gamma").cdf(s),
                     m.approximate("shifted_lognormal").cdf(s)))
approx = pd.DataFrame(rows, columns=["case", "exact_book", "exact", "normal_book", "normal",
                                     "sgamma_book", "sgamma", "slognormal"])
(GT(approx, rowname_col="case")
 .tab_header("F(s) by the exact series and three approximations")
 .tab_spanner("exact series", ["exact_book", "exact"])
 .tab_spanner("normal approximation", ["normal_book", "normal"])
 .tab_spanner("shifted gamma", ["sgamma_book", "sgamma"])
 .fmt_number(["exact_book", "exact", "normal_book", "normal", "sgamma_book", "sgamma", "slognormal"],
             decimals=4)
 .fmt_number(["exact_book", "exact"], rows=[5], decimals=5)   # Table 4.4 at s = 0
 .cols_label(exact_book="book", exact="prospicio", normal_book="book", normal="prospicio",
             sgamma_book="book", sgamma="prospicio", slognormal="shifted lognormal"))
```

```text
                          book  prospicio    book  prospicio    book  prospicio  shifted lognormal
Table 4.3, s = 0        0.0821     0.0821  0.0855     0.0855  0.0459     0.0459             0.0497
Table 4.3, s = 2,000    0.3755     0.3755  0.3240     0.3240  0.3680     0.3680             0.3609
Table 4.3, s = 4,000    0.7152     0.7152  0.6760     0.6760  0.7185     0.7185             0.7197
Table 4.3, s = 6,000    0.9013     0.9013  0.9145     0.9145  0.9038     0.9038             0.9064
Table 4.3, s = 8,000    0.9723     0.9723  0.9888     0.9888  0.9724     0.9724             0.9729
Table 4.4, s = 0       0.00005    0.00005  0.2451     0.2451  0.1228     0.1228             0.2182
Table 4.4, s = 2,000    0.5922     0.5922  0.4090     0.4090  0.5886     0.5886             0.5234
Table 4.4, s = 6,000    0.8401     0.8401  0.7549     0.7549  0.8402     0.8402             0.8341
Table 4.4, s = 10,000   0.9294     0.9294  0.9463     0.9463  0.9298     0.9298             0.9364
Table 4.4, s = 18,000   0.9846     0.9846  0.9997     0.9997  0.9847     0.9847             0.9874
P4.13, s = 0            0.0003     0.0003  0.1241     0.1241  0.0263     0.0263             0.0586
P4.13, s = 3,000        0.3420     0.3420  0.2819     0.2819  0.3362     0.3362             0.3139
P4.13, s = 9,000        0.7774     0.7774  0.7181     0.7181  0.7782     0.7782             0.7790
P4.13, s = 15,000       0.9349     0.9349  0.9584     0.9584  0.9356     0.9356             0.9402
P4.13, s = 27,000       0.9954     0.9954  1.0000     1.0000  0.9953     0.9953             0.9951
```

The normal leaves the exact curve at both ends. The shifted gamma stays on it
except near zero:

```python
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
for ax, (name, _, lam, shape, scale, *_rest) in zip(axes, cases[:2]):
    m = compound_moments(Poisson(lam), Gamma(shape, scale))
    top = m.mean + 4.5 * m.variance ** 0.5
    xs = [top * i / 40 for i in range(41)]
    ax.plot(xs, [poisson_gamma_cdf(x, lam, shape, scale, terms=60) for x in xs],
            color="0.75", linewidth=4, label="exact")
    ax.plot(xs, [m.approximate("normal").cdf(x) for x in xs], label="normal")
    ax.plot(xs, [m.approximate("shifted_gamma").cdf(x) for x in xs], label="shifted gamma")
    ax.set_title(f"{name}: skewness {m.skew:.1f}")
    ax.set_xlabel("total claims s")
axes[0].set_ylabel("F(s)")
axes[0].legend()
```

Above the mean the shifted gamma is within a few units of the fourth decimal,
even at a skewness of 2.8. It fails near zero, where it has no mass for the
chance of no claim. The normal misses the right tail by several percent. For
capital or stop-loss questions use the exact distribution (next section), which
is cheap.

## 7. Recursion and Fourier inversion

**Panjer's recursion** is exact for claim sizes on a grid and a Poisson,
negative binomial or binomial count. Ex 4.4 has a Poisson 1.75 and sizes of
1,000 to 5,000:

```python
size = Grid(1000.0, [0.0, 0.20, 0.40, 0.20, 0.15, 0.05])
agg, _ = panjer(Poisson(1.75), size, 64)
book = [0.1738, 0.0608, 0.1323, 0.1046, 0.117, 0.0932, 0.0786, 0.0641]
rows = [(f"Ex 4.4, f_S({1000 * k:,})", b, agg.probs[k]) for k, b in enumerate(book)]
rows += [(f"Ex 4.4, F_S({s:,})", b, agg.cdf(float(s)))
         for s, b in [(9000, 0.9121), (12000, 0.9729), (16000, 0.9955)]]
compare(rows, "Ex 4.4: Panjer's recursion, a Poisson 1.75")
```

```text
                       book  prospicio
Ex 4.4, f_S(0)       0.1738     0.1738
Ex 4.4, f_S(1,000)   0.0608     0.0608
Ex 4.4, f_S(2,000)   0.1323     0.1323
Ex 4.4, f_S(3,000)   0.1046     0.1046
Ex 4.4, f_S(4,000)   0.1170     0.1170
Ex 4.4, f_S(5,000)   0.0932     0.0932
Ex 4.4, f_S(6,000)   0.0786     0.0786
Ex 4.4, f_S(7,000)   0.0641     0.0641
Ex 4.4, F_S(9,000)   0.9121     0.9121
Ex 4.4, F_S(12,000)  0.9729     0.9729
Ex 4.4, F_S(16,000)  0.9955     0.9955
```

A continuous severity goes on a grid first. The **midpoint** method gives each
point the probability of its interval (`Grid.rounding`). Ex 4.5 does this to the
gamma of Ex 4.2 with steps of 100 and 20. The finer step brings the recursion
within about 1% of the exact values at small totals, and closer above:

```python
S = [500, 1000, 2000, 4000, 6000, 10000]
ex_45 = {}
for h, book in [(100.0, [0.1158, 0.1956, 0.3852, 0.7218, 0.9042, 0.9937]),
                (20.0, [0.1108, 0.1885, 0.3775, 0.7165, 0.9019, 0.9935])]:
    grid, _ = Grid.rounding(Gamma(3.0, 400.0), h, int(6000 / h) + 1)
    agg, _ = panjer(Poisson(2.5), grid, int(30000 / h))
    ex_45[h] = (book, [agg.cdf(s) for s in S])
    print(f"Ex 4.5  step {h:3.0f}  grid mean {grid.mean():,.2f}  variance {grid.variance():,.0f}")
```

```text
Ex 4.5  step 100  grid mean 1,199.98  variance 480,642
Ex 4.5  step  20  grid mean 1,199.98  variance 479,846
```

```python
(GT(pd.DataFrame({"s": [f"F_S({s:,})" for s in S],
                  "book_100": ex_45[100.0][0], "ours_100": ex_45[100.0][1],
                  "book_20": ex_45[20.0][0], "ours_20": ex_45[20.0][1]}), rowname_col="s")
 .tab_header("Ex 4.5: Panjer's recursion on the midpoint grid")
 .tab_spanner("step 100", ["book_100", "ours_100"])
 .tab_spanner("step 20", ["book_20", "ours_20"])
 .fmt_number(["book_100", "ours_100", "book_20", "ours_20"], decimals=4)
 .cols_label(book_100="book", ours_100="prospicio", book_20="book", ours_20="prospicio"))
```

```text
               book  prospicio    book  prospicio
F_S(500)     0.1158     0.1158  0.1108     0.1108
F_S(1,000)   0.1956     0.1956  0.1885     0.1885
F_S(2,000)   0.3852     0.3852  0.3775     0.3775
F_S(4,000)   0.7218     0.7218  0.7165     0.7165
F_S(6,000)   0.9042     0.9042  0.9019     0.9019
F_S(10,000)  0.9937     0.9937  0.9935     0.9935
```

Every figure agrees, and so do the printed grid moments (1,199.98 and 480,642
for the step of 100, 479,846 for 20), except the mean for the step of 20: the
monograph prints 1,199.88, a slipped digit. P4.16 recurses a Poisson 1.35 on the
sizes of P4.1:

```python
agg, _ = panjer(Poisson(1.35), Grid(500.0, [0.0, 0.1, 0.4, 0.3, 0.2]), 64)
book = [0.2592, 0.2942, 0.4366, 0.5606, 0.6838, 0.8306, 0.9223, 0.967, 0.9872]
at = (0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000)
compare([(f"P4.16, F_S({s:,})", b, agg.cdf(float(s))) for s, b in zip(at, book)],
        "P4.16: Panjer's recursion, a Poisson 1.35")
```

```text
                     book  prospicio
P4.16, F_S(0)      0.2592     0.2592
P4.16, F_S(500)    0.2942     0.2942
P4.16, F_S(1,000)  0.4366     0.4366
P4.16, F_S(1,500)  0.5606     0.5606
P4.16, F_S(2,000)  0.6838     0.6838
P4.16, F_S(3,000)  0.8306     0.8306
P4.16, F_S(4,000)  0.9223     0.9223
P4.16, F_S(5,000)  0.9670     0.9670
P4.16, F_S(6,000)  0.9872     0.9872
```

The monograph's **Fourier** method (Ex 4.7) is Heckman and Meyers' inversion of
the characteristic function. prospicio inverts with the FFT on a grid (`fft`, or
`fft_auto`, which picks the grid). It takes any count prospicio supports, and
counts much larger than the recursion allows, since e^(−λ) underflows for large
λ. On Ex 4.2 with a step of 1, a mean-preserving grid (`Grid.local_moment`) and
2^16 points, it reproduces the exact series:

```python
grid, _ = Grid.local_moment(Gamma(3.0, 400.0), 1.0, 20001)
agg, report = fft(Poisson(2.5), grid, 2**16)
S = [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
print(f"Ex 4.7  aliasing error below 1e-12: {report.aliasing_error < 1e-12}")
```

```text
Ex 4.7  aliasing error below 1e-12: True
```

```python
book = [0.1094, 0.1869, 0.3757, 0.5614, 0.7151, 0.9011, 0.9722, 0.9934]
ex_47 = pd.DataFrame({"s": [f"F_S({s:,})" for s in S], "book": book,
                      "exact": [poisson_gamma_cdf(s, 2.5, 3.0, 400.0) for s in S],
                      "fft": [agg.cdf(s) for s in S]})
(GT(ex_47, rowname_col="s")
 .tab_header("Ex 4.7: Heckman and Meyers against the exact series and the FFT")
 .fmt_number(["book", "exact", "fft"], decimals=4)
 .cols_label(exact="exact series", fft="FFT, prospicio"))
```

```text
               book  exact series  FFT, prospicio
F_S(500)     0.1094        0.1096          0.1096
F_S(1,000)   0.1869        0.1867          0.1868
F_S(2,000)   0.3757        0.3755          0.3756
F_S(3,000)   0.5614        0.5613          0.5614
F_S(4,000)   0.7151        0.7152          0.7152
F_S(6,000)   0.9011        0.9013          0.9014
F_S(8,000)   0.9722        0.9723          0.9723
F_S(10,000)  0.9934        0.9934          0.9934
```

The FFT is within a unit in the fourth decimal (the grid's half-step), the same
accuracy as Heckman and Meyers' method.

A per-claim limit puts a mass in the severity, so the aggregate jumps at
multiples of the limit. Ex 4.8 caps gamma (2.5; 500) claims at 2,000 under a
Poisson 1.308. The jump at 2,000 is Pr{N = 1} Pr{X > 2,000}:

```python
size = Gamma(2.5, 500.0)
compare([("Ex 4.8, E[X; 2,000]", 1_147, size.lev(2000.0), 0),
         ("Ex 4.8, mass at the limit", 0.1562, size.survival(2000.0)),
         ("Ex 4.8, jump in F_S at 2,000", 0.0552, Poisson(1.308).pmf(1) * size.survival(2000.0))],
        "Ex 4.8: a per-claim limit of 2,000")
```

```text
                                book  prospicio
Ex 4.8, E[X; 2,000]            1,147      1,147
Ex 4.8, mass at the limit     0.1562     0.1562
Ex 4.8, jump in F_S at 2,000  0.0552     0.0552
```

## 8. Simulation

`simulate_events` draws a count, then that many claims, from a seed. Ex 4.14
draws 10,000 years of Ex 4.2. Another generator draws other years, so the check
is accuracy, not digits:

```python
years = simulate_events(Poisson(2.5), Gamma(3.0, 400.0), 10_000, 2015).totals().total()
sim = years.cdf(3000.0)
exact = poisson_gamma_cdf(3000.0, 2.5, 3.0, 400.0)
print(f"Ex 4.14  F(3,000): book's sample 0.5599, exact {exact:.4f}, "
      f"within two standard errors here: {abs(sim - exact) < 2 * math.sqrt(exact * (1 - exact) / 10_000)}")
print(f"Ex 4.13  lognormal (4.956; 2.3), 1.5 claims a year: E[S] = "
      f"{1.5 * Lognormal(4.956, 2.3).mean():,.0f} (book 3,000)")
```

```text
Ex 4.14  F(3,000): book's sample 0.5599, exact 0.5613, within two standard errors here: True
Ex 4.13  lognormal (4.956; 2.3), 1.5 claims a year: E[S] = 3,000 (book 3,000)
```

With 10,000 years the standard error of a probability near one half is 0.005,
so a sampled table is good to about two decimals (Table 4.11).

## 9. Aggregate limits

A policy can cap the year's total as well as each claim. The expected loss under
a per-claim limit l and an aggregate limit L is E[S_l; L]; the increased limit
factor is that value over the expected loss at the basic limit. Ex 6.6 has 1.2
expected claims, contagion 0.1 (a negative binomial with r = 10) and lognormal
claims with σ = 2.4.

The monograph states μ = 7.000, but its basic-limit loss 1.2 × 21,743 and its 2M
loss 1.2 × 28,338 come from μ = 7.6 (μ = 7.0 gives 13,626 and 16,738, the numbers
of Ex 6.3). Every figure of Ex 6.6 and Table 6.4 reproduces with 7.6, so 7.000 is
a misprint:

```python
size = Lognormal(7.6, 2.4)
count = claim_count(1.2, 1 + 0.1 * 1.2)
basic = count.mean() * size.lev(500e3)
h = 250.0
capped_2m, _ = Grid.local_moment(size, h, int(2e6 / h) + 1)   # mass at 2M is the cap
m = compound_moments(count, capped_2m)
compare([("Ex 6.6, E[S], 500K per claim", 26_092, basic, 0),
         ("Ex 6.6, E[S], 2M per claim", 34_006, count.mean() * size.lev(2e6), 0),
         ("Ex 6.6, sd of S, 2M per claim", 151_311, m.variance ** 0.5, 0),
         ("Ex 6.6, skewness, 2M per claim", 9.4728, m.skew)],
        "Ex 6.6: lognormal (7.6; 2.4) claims, 1.2 expected claims, contagion 0.1")
```

```text
                                   book  prospicio
Ex 6.6, E[S], 500K per claim     26,092     26,091
Ex 6.6, E[S], 2M per claim       34,006     34,006
Ex 6.6, sd of S, 2M per claim   151,311    151,312
Ex 6.6, skewness, 2M per claim   9.4728     9.4728
```

The aggregate limit removes E[(S_l − L)+]. The monograph takes it from the
shifted gamma approximation, whose stop-loss is a gamma's (`Gamma.stop_loss` of
the unshifted part). The FFT gives it exactly on the same grid:

```python
def shifted_gamma_excess(moments, L):
    a = dict(moments.approximate("shifted_gamma").parameters)
    return Gamma(a["shape"], a["scale"]).stop_loss(L - a["shift"])


sg = shifted_gamma_excess(m, 3e6)
agg, _ = fft(count, capped_2m, 2**16)
exact = agg.stop_loss(3e6)
removed = pd.DataFrame({
    "row": ["removed by a 3M aggregate limit", "I(2M, 3M)"],
    "book": [91, 1.2998],
    "shifted": [sg, (m.mean - sg) / basic],
    "exact": [exact, (m.mean - exact) / basic],
})
(GT(removed, rowname_col="row")
 .tab_header("Ex 6.6: a 3M aggregate limit on 2M claims")
 .fmt_number(["book", "shifted", "exact"], rows=[0], decimals=0)
 .fmt_number(["book", "shifted", "exact"], rows=[1], decimals=4)
 .cols_label(shifted="shifted gamma", exact="exact, FFT"))
```

```text
                                   book  shifted gamma  exact, FFT
removed by a 3M aggregate limit      91             91           8
I(2M, 3M)                        1.2998         1.2999      1.3031
```

Matched to a skewness of 9.5, the shifted gamma puts far too much probability in
the far tail, so it overstates what the limit removes: 91 against 8. The gap
grows as the aggregate limit comes down towards the per-claim limit (Table 6.4):

```python
book = {  # Table 6.4: E[S] by per-claim limit (rows) and aggregate limit
    1e6: {1e6: 29_702, 2e6: 30_306, 3e6: 30_333, 5e6: 30_335},
    2e6: {2e6: 33_524, 3e6: 33_915, 5e6: 34_002},
    3e6: {3e6: 35_421, 5e6: 35_781},
    5e6: {5e6: 37_428},
}
rows, exact_at = [], {}
for l, row in book.items():
    grid, _ = Grid.local_moment(size, h, int(l / h) + 1)
    moments = compound_moments(count, grid)
    agg, _ = fft(count, grid, 2**16)
    for L, value in row.items():
        exact_at[l, L] = agg.lev(L)
        rows.append((f"{l / 1e6:.0f}M per claim, {L / 1e6:.0f}M aggregate",
                     value, moments.mean - shifted_gamma_excess(moments, L), agg.lev(L)))
    exact_at[l, math.inf] = agg.mean()
(GT(pd.DataFrame(rows, columns=["limits", "book", "shifted", "exact"]), rowname_col="limits")
 .tab_header("Table 6.4: E[S] by per-claim and aggregate limit")
 .fmt_number(["book", "shifted", "exact"], decimals=0)
 .cols_label(shifted="shifted gamma", exact="exact, FFT"))
```

```text
                              book  shifted gamma  exact, FFT
1M per claim, 1M aggregate  29,702         29,702      30,133
1M per claim, 2M aggregate  30,306         30,306      30,334
1M per claim, 3M aggregate  30,333         30,333      30,335
1M per claim, 5M aggregate  30,335         30,335      30,335
2M per claim, 2M aggregate  33,524         33,523      33,910
2M per claim, 3M aggregate  33,915         33,915      33,999
2M per claim, 5M aggregate  34,002         34,002      34,006
3M per claim, 3M aggregate  35,421         35,420      35,762
3M per claim, 5M aggregate  35,781         35,780      35,819
5M per claim, 5M aggregate  37,428         37,427      37,703
```

P6.11 asks for the share an aggregate limit equal to the per-claim limit
removes:

```python
compare([(f"P6.11, {l / 1e6:.0f}M per claim and aggregate", b,
          1 - exact_at[l, l] / exact_at[l, math.inf])
         for l, b in [(1e6, 0.0209), (2e6, 0.0142), (3e6, 0.0112), (5e6, 0.0081)]],
        "P6.11: share of the loss an aggregate limit removes")
```

```text
                                     book  prospicio
P6.11, 1M per claim and aggregate  0.0209     0.0066
P6.11, 2M per claim and aggregate  0.0142     0.0028
P6.11, 3M per claim and aggregate  0.0112     0.0016
P6.11, 5M per claim and aggregate  0.0081     0.0008
```

P6.12 prices a 1M per-claim limit with v = 25%:

```python
print(f"P6.12  premium, 1M per claim, v = 25%: book 40,447, prospicio "
      f"{exact_at[1e6, math.inf] / 0.75:,.0f}")
print(f"P6.12  credit for a 1M aggregate limit: book 844, "
      f"exact {(exact_at[1e6, math.inf] - exact_at[1e6, 1e6]) / 0.75:,.0f}")
```

```text
P6.12  premium, 1M per claim, v = 25%: book 40,447, prospicio 40,446
P6.12  credit for a 1M aggregate limit: book 844, exact 269
```

The shifted gamma column matches the monograph to a unit or so (it integrated
numerically), which confirms the method and μ = 7.6. Where the aggregate limit
equals the per-claim limit, the approximation removes three to ten times the loss
that is there, and P6.12's credit for a 1M aggregate limit falls from 844 to
about a third of that. P6.12's unlimited premium needs no approximation. It
agrees up to the monograph's rounding of 30,335 before it divides, the same
rounding that gives its basic loss of 26,092 and standard deviation of 151,311.

## 10. What agreed and what did not

These values agree to the printed digits:

- Ex 3.1 to 3.5, 3.8, 3.9, P3.3, P3.4, P3.11 to P3.13, P3.15 and P3.28 to P3.31.
- Ex 4.1, 4.2 (moments and Table 4.1), 4.3, 4.4, 4.5, 4.8, P4.1 and P4.16, and the
  exact, normal and shifted gamma columns of Tables 4.3 and 4.4 and P4.13.
- Ex 6.6's moments and its 91, Table 6.4's shifted gamma values and P6.12's
  premium, up to the monograph's rounding of intermediate values (a unit or so).

These values did not agree:

- **Ex 3.3's chi-square**, 2.45 against 2.43: rounded expected counts.
- **Ex 3.6's α**, 2.44302 against 2.44285: a rounded sample variance. Its
  chi-square of 0.7753 is not reproduced (near 0.65 here).
- **Ex 6.6's μ**: printed 7.000, used 7.6.
- **Table 6.4, P6.11 and P6.12's credit** are right for the monograph's shifted
  gamma, several times too large for the actual aggregate.

Where to go next:

- [Severity and frequency](severity-and-frequency.md): count families by
  dispersion.
- [Aggregate distributions](aggregate-distributions.md): when FFT, Panjer,
  simulation or moments are exact, fast or unsafe.
- [Experience and exposure rating](experience-and-exposure-rating.md): excess
  counts and layers.
- [Claim size, layers, limits and deductibles](distributions-for-actuaries-claim-size.md):
  chapters 2, 5 and 6.
