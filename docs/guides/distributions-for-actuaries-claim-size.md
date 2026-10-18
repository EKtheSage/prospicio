# Claim size, layers, limits and deductibles: Bahnemann's *Distributions for Actuaries* reproduced

CAS Monograph 2, *Distributions for Actuaries* by David Bahnemann (2015), is the
CAS's textbook on the probability behind pricing: the size of a claim, the number
of claims, the total of a year, and what policy limits, deductibles and
inflation do to each of them. Almost every section ends in a worked example
with numbers, and the appendix prints answers to many of the problems. This
guide follows the monograph's chapters 2, 5 and 6 (claim size, excess claims,
limits and deductibles) and recomputes those numbers with prospicio, printing
the monograph's value next to prospicio's. Its companion,
[claim counts and aggregate claims](distributions-for-actuaries-counts-aggregates.md),
does the same for chapters 3, 4 and the aggregate limits of section 6.4.

The explanations here are new words, not the monograph's; read the monograph for
the derivations and the full problem statements. Example and problem numbers are
the monograph's ("Ex 2.7", "P6.13"), and so are its symbols where a formula is
quoted. Where a printed number and prospicio disagree, the guide says so and
gives the most likely reason; section 10 collects them.

## 1. Translating the parameters

The monograph and prospicio name and scale their distributions differently, and
most apparent disagreements between a textbook and a library come from here, so
it is worth fixing the dictionary first:

| Monograph | Its formula | prospicio |
|---|---|---|
| gamma (α, β), β a scale | mean αβ | `Gamma(shape=α, scale=β)` |
| exponential β | mean β | `Gamma(1.0, β)` |
| lognormal (μ, σ) | log X normal (μ, σ) | `Lognormal(μ, σ)` |
| Weibull (β, δ) | F = 1 − exp(−(x/β)^δ) | `Weibull(shape=δ, scale=β)` |
| Pareto (α, β), the "shifted" one used in all examples | F = 1 − (β/(x+β))^α, from 0 | `GeneralizedPareto(xi=1/α, beta=β/α)`, or `Burr(α, 1.0, β)` |
| classical Pareto (γ, α) | F = 1 − (γ/x)^α, from γ | `Pareto(t=γ, alpha=α)` |
| Burr (α, β, δ) | F = 1 − (β/(x^δ+β))^α | `Burr(α, δ, β^(1/δ))` |

The trap is the Pareto. The monograph's Pareto starts at zero (a Lomax), while
prospicio's `Pareto` is the single-parameter one that starts at a threshold,
the form reinsurance pricing uses above an attachment. The Lomax is a
generalized Pareto with shape 1/α and scale β/α. Every block below uses one
helper for it, and checks it against the monograph's closed form for the
limited mean of Ex 2.15:

```python
import math

from scipy.optimize import minimize

from prospicio.distributions import (
    Beta, Burr, Gamma, GeneralizedPareto, Lognormal, Pareto, Truncated, fit_severity,
)
from prospicio.pricing import ilf, layer_cv, loss_elimination_ratio


def shifted_pareto(alpha, beta):
    """The monograph's Pareto, F(x) = 1 - (beta / (x + beta))**alpha."""
    return GeneralizedPareto(1.0 / alpha, beta / alpha)


def survival(sev, x):
    return 1.0 - sev.cdf(x)


def limited_second_moment(sev, limit):
    """E[X^2; limit]. Truncated() with no bounds adds layer moments to any severity."""
    return Truncated(sev).layer_second_moment(limit, 0.0)


def show(label, book, ours, digits=0):
    print(f"{label:<40s} book {book:>12,.{digits}f}   prospicio {ours:>12,.{digits}f}")


p = shifted_pareto(2.0, 3000.0)
closed_form = 3000.0 / (2 - 1) * (1 - 3000.0 / (8000.0 + 3000.0))
show("E[X; 8,000], Pareto (2; 3,000)", 2182, p.lev(8000.0))
print(f"closed form {closed_form:,.2f}, prospicio {p.lev(8000.0):,.2f}, "
      f"Burr {Burr(2.0, 1.0, 3000.0).lev(8000.0):,.2f}")
```

```text
E[X; 8,000], Pareto (2; 3,000)           book        2,182   prospicio        2,182
closed form 2,181.82, prospicio 2,181.82, Burr 2,181.82
```

`show` prints the monograph's printed number (a literal in the code) and then
prospicio's, rounded the same way.

## 2. Limited expected values

The limited expected value E[X; l], the average claim when every claim is capped
at l, is the monograph's main tool: increased limit factors, layers, deductibles
and trend effects are all differences or ratios of it. Every prospicio severity
has it as `lev`, and `Truncated(sev).layer_second_moment` gives the limited
second moment E[X²; l]. Ex 2.4 caps an exponential with mean 250 at 400;
P2.1, P2.17 and P2.19 ask for the usual list of a classical Pareto, a lognormal
and a shifted Pareto:

```python
x = Gamma(1.0, 250.0)
show("Ex 2.4  E[X; 400]", 199.53, x.lev(400.0), 2)
show("Ex 2.4  E[X^2; 400]", 59_384, limited_second_moment(x, 400.0))

x = Pareto(1000.0, 3.0)  # classical: F = 1 - (1,000 / x)^3 from 1,000
show("P2.1   mean", 1_500, x.mean())
show("P2.1   variance", 750_000, x.variance())
show("P2.1   Pr{X > 2,000}", 0.1250, x.survival(2000.0), 4)
show("P2.1   E[X; 2,000]", 1_375, x.lev(2000.0))

for name, x, book in [
    ("P2.17", Lognormal(6.3210, 1.6000), (556, 0.1461, 0.2108, 1_023, 9_689)),
    ("P2.19", shifted_pareto(3.0, 4000.0), (1_040, 0.1866, 0.3254, 1_347, 6_500)),
]:
    s3 = survival(x, 3000.0)
    show(f"{name}  median", book[0], x.quantile(0.5))
    show(f"{name}  Pr{{X > 3,000}}", book[1], s3, 4)
    show(f"{name}  Pr{{1,000 < X <= 3,000}}", book[2], x.cdf(3000.0) - x.cdf(1000.0), 4)
    show(f"{name}  E[X; 3,000]", book[3], x.lev(3000.0))
    show(f"{name}  E[X | X > 3,000]", book[4], 3000.0 + x.stop_loss(3000.0) / s3)
```

```text
Ex 2.4  E[X; 400]                        book       199.53   prospicio       199.53
Ex 2.4  E[X^2; 400]                      book       59,384   prospicio       59,384
P2.1   mean                              book        1,500   prospicio        1,500
P2.1   variance                          book      750,000   prospicio      750,000
P2.1   Pr{X > 2,000}                     book       0.1250   prospicio       0.1250
P2.1   E[X; 2,000]                       book        1,375   prospicio        1,375
P2.17  median                            book          556   prospicio          556
P2.17  Pr{X > 3,000}                     book       0.1461   prospicio       0.1461
P2.17  Pr{1,000 < X <= 3,000}            book       0.2108   prospicio       0.2108
P2.17  E[X; 3,000]                       book        1,023   prospicio        1,023
P2.17  E[X | X > 3,000]                  book        9,689   prospicio        9,689
P2.19  median                            book        1,040   prospicio        1,040
P2.19  Pr{X > 3,000}                     book       0.1866   prospicio       0.1866
P2.19  Pr{1,000 < X <= 3,000}            book       0.3254   prospicio       0.3254
P2.19  E[X; 3,000]                       book        1,347   prospicio        1,347
P2.19  E[X | X > 3,000]                  book        6,500   prospicio        6,500
```

All agree. Two things carry over to the rest of the guide. First, the mean of
the claims above a point is the point plus the **mean excess**
e(x) = E[(X − x)+] / Pr{X > x}, and `stop_loss(x)` is its numerator. Second, the
average of claims above 3,000 is nearly ten times the overall lognormal mean
(about 2,000 here): the tail, not the middle, decides excess prices.

P2.25 shows how much the family matters when only the mean and standard
deviation are known: with mean 1,000 and standard deviation 2,000, how many of
5,000 claims exceed 1,000?

```python
book = {"gamma": 1_282, "lognormal": 1_315, "Pareto": 1_428}
# A shifted Pareto with cv 2 has alpha / (alpha - 2) = 4, so alpha = 8/3.
fits = {
    "gamma": Gamma.from_mean_cv(1000.0, 2.0),
    "lognormal": Lognormal.from_mean_cv(1000.0, 2.0),
    "Pareto": shifted_pareto(8 / 3, 1000.0 * (8 / 3 - 1)),
}
for name, x in fits.items():
    show(f"P2.25  claims > 1,000, {name}", book[name], 5000 * survival(x, 1000.0))
```

```text
P2.25  claims > 1,000, gamma             book        1,282   prospicio        1,282
P2.25  claims > 1,000, lognormal         book        1,315   prospicio        1,315
P2.25  claims > 1,000, Pareto            book        1,428   prospicio        1,428
```

## 3. Fitting grouped data

The monograph's data come grouped: a count of claims in each size band, no
individual amounts. It fits them three ways, the method of moments, minimum
chi-square, and a minimum distance between the fitted and the empirical
limited expected values, all on a spreadsheet's solver. prospicio fits by
maximum likelihood on individual claims (section 4), so here its distributions
supply the cdf and `lev`, and SciPy's Nelder–Mead does the minimizing.

The running sample of Ex 2.6 has 200 claims in nine bands. Its first two sample
moments are M1 = 1,555 and M2 = 3,036,875, and Ex 2.7 matches a gamma to them:

```python
M1, M2 = 1555.0, 3_036_875.0
alpha = M1**2 / (M2 - M1**2)
beta = (M2 - M1**2) / M1
show("Ex 2.7  alpha", 3.907288, alpha, 6)
show("Ex 2.7  beta", 397.931, beta, 3)
```

```text
Ex 2.7  alpha                            book     3.907288   prospicio     3.907288
Ex 2.7  beta                             book      397.931   prospicio      397.974
```

The shape agrees to every printed digit and the scale does not:
(3,036,875 − 1,555²) / 1,555 is 618,850 / 1,555 = 397.974, not 397.931. The
monograph's own tables were computed with its 397.931 (its fitted tail
probabilities match prospicio's with that scale, not with 397.974), so the
slip is in the printed division, and the comparison below uses the printed
scale. The fit is judged by a chi-square test on six cells:

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
print("Ex 2.7  Pr{X > x}  book ", [0.7382, 0.4604, 0.2465, 0.1186, 0.0528])
print("                   ours ", [round(survival(g, x), 4) for x in xs])
print("        E[X; x]    book ", [924, 1223, 1396, 1484, 1525])
print("                   ours ", [round(g.lev(x)) for x in xs])
show("Ex 2.7  chi-square", 3.096, chi_square(g), 3)
```

```text
Ex 2.7  Pr{X > x}  book  [0.7382, 0.4604, 0.2465, 0.1186, 0.0528]
                   ours  [0.7382, 0.4604, 0.2465, 0.1186, 0.0528]
        E[X; x]    book  [924, 1223, 1396, 1484, 1525]
                   ours  [924, 1223, 1396, 1484, 1525]
Ex 2.7  chi-square                       book        3.096   prospicio        3.095
```

The chi-square differs in the third decimal because the monograph rounds the
expected counts to two decimals before summing.

Ex 2.8 fits a lognormal to the same cells by **minimum chi-square**: choose the
parameters that make the chi-square statistic smallest. The monograph reports
(7.274670, 0.442525) with a minimum of 1.828:

```python
def min_chi_square(make, start, **kw):
    fit = minimize(lambda p: chi_square(make(*p), **kw), start, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    return fit.x, fit.fun


ln = Lognormal(7.274670, 0.442525)
print("Ex 2.8  Pr{X > x}  book ", [0.7965, 0.4653, 0.2305, 0.1072, 0.0491])
print("                   ours ", [round(survival(ln, x), 4) for x in xs])
print("        E[X; x]    book ", [958, 1273, 1441, 1522, 1559])
print("                   ours ", [round(ln.lev(x)) for x in xs])
show("Ex 2.8  chi-square at its fit", 1.828, chi_square(ln), 3)
(mu, sigma), chi2 = min_chi_square(lambda m, s: Lognormal(m, abs(s)), [7.2, 0.5])
print(f"        smallest chi-square {chi2:.3f} at ({mu:.4f}, {abs(sigma):.4f})")
(a, b), chi2 = min_chi_square(lambda a, b: Gamma(abs(a), abs(b)), [4.0, 400.0])
show("P2.16  gamma by min chi-square: a", 4.7432, abs(a), 4)
show("P2.16  b", 337.31, abs(b), 2)
```

```text
Ex 2.8  Pr{X > x}  book  [0.7965, 0.4653, 0.2305, 0.1072, 0.0491]
                   ours  [0.7965, 0.4653, 0.2305, 0.1072, 0.0491]
        E[X; x]    book  [958, 1273, 1441, 1522, 1559]
                   ours  [958, 1273, 1441, 1522, 1559]
Ex 2.8  chi-square at its fit            book        1.828   prospicio        1.020
        smallest chi-square 0.614 at (7.2857, 0.4570)
P2.16  gamma by min chi-square: a        book       4.7432   prospicio       4.7432
P2.16  b                                 book       337.31   prospicio       337.31
```

The fitted lognormal's tail probabilities and limited means match the
monograph's Table 2.3 exactly, so the parameters are the ones it used; but at
those parameters the statistic is 1.020, not 1.828, and they are not the
minimum: the solver stopped early, and the true minimum, 0.614, lies at
(7.2857, 0.4570). The conclusion (a good fit, far under the 5% limit of 7.815)
stands. For the gamma of P2.16 the same search lands on the printed answer.

Ex 2.9 fits the shifted Pareto to a heavier sample by the limited-mean distance,
with α held at 6. With the printed parameters prospicio reproduces Table 2.4:

```python
p = shifted_pareto(6.0, 34_355.3719)
xs = [2000, 4000, 6000, 8000, 10000]
print("Ex 2.9  Pr{X > x}  book ", [0.7121, 0.5164, 0.3807, 0.2848, 0.2159])
print("                   ours ", [round(p.survival(x), 4) for x in xs])
print("        E[X; x]    book ", [1693, 2910, 3799, 4459, 4956])
print("                   ours ", [round(p.lev(x)) for x in xs])
```

```text
Ex 2.9  Pr{X > x}  book  [0.7121, 0.5164, 0.3807, 0.2848, 0.2159]
                   ours  [0.7121, 0.5164, 0.3807, 0.2848, 0.2159]
        E[X; x]    book  [1693, 2910, 3799, 4459, 4956]
                   ours  [1693, 2910, 3799, 4459, 4956]
```

## 4. Censored and truncated data

Claims reach the actuary modified. A **policy limit censors** them: a claim
recorded at the limit is known only to be at least that large. A **deductible
truncates** them: claims below it never appear, and those above are reported
net of it (or, added back, from the deductible up). Fitting the recorded claims
as if they were the population understates the tail; the fix is to fit the
ground-up distribution through the modification.

Ex 2.10 has 1,500 claims under a 300,000 limit, 23 of them at the limit,
summarized as counts and totals below eight sizes. From those, the empirical
limited mean at each size is exact even though the claims are grouped, and the
monograph fits a lognormal by minimizing the squared distance between it and the
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
mu, sigma = min_distance(sizes, emp, [7.0, 2.5])
show("Ex 2.10 mu", 6.9852, mu, 4)
show("Ex 2.10 sigma", 2.5850, sigma, 4)
print("Ex 2.10 E[X; x] book ", [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850])
print("            prospicio", [round(Lognormal(mu, sigma).lev(x)) for x in sizes])

sizes = [10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000]
counts = [78, 27, 21, 15, 12, 10, 8, 7]          # 22 more at the 100,000 limit
mids = [5000, 15000, 25000, 35000, 45000, 55000, 70000, 90000]
emp = [(sum(n * m for n, m in zip(counts[:j + 1], mids[:j + 1]))
        + x * (200 - sum(counts[:j + 1]))) / 200 for j, x in enumerate(sizes)]
mu, sigma = min_distance(sizes, emp, [9.7, 1.4])
show("P2.39 mu", 9.778102, mu, 4)
show("P2.39 sigma", 1.444776, sigma, 4)
```

```text
Ex 2.10 mu                               book       6.9852   prospicio       6.9852
Ex 2.10 sigma                            book       2.5850   prospicio       2.5850
Ex 2.10 E[X; x] book  [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850]
            prospicio [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850]
P2.39 mu                                 book       9.7781   prospicio       9.7781
P2.39 sigma                              book       1.4448   prospicio       1.4448
```

When a deductible truncates the data, each cell's probability is conditioned on
exceeding it. Ex 2.11 has 770 claims above a 1,000 deductible (added back),
censored at 200,000, fitted by minimum chi-square. P2.43's claims are truncated
from above instead (only claims up to 50,000 were kept), and prospicio's
`Truncated` gives the conditional distribution in both cases:

```python
cuts = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000,
        200000, math.inf]
observed = [367, 112, 118, 65, 36, 13, 10, 8, 6, 4, 31]
(mu, sigma), chi2 = min_chi_square(
    lambda m, s: Truncated(Lognormal(m, abs(s)), lower=1000.0), [6.7, 2.7],
    cuts=cuts, observed=observed)
show("Ex 2.11 mu", 6.6916, mu, 4)
show("Ex 2.11 sigma", 2.6965, abs(sigma), 4)
show("Ex 2.11 chi-square", 4.691, chi2, 3)
fitted = Truncated(Lognormal(mu, abs(sigma)), lower=1000.0)
print("Ex 2.11 expected book ", [360, 122, 121, 63, 27, 16, 10, 7, 5, 4, 34])
print("            prospicio ", [round(770 * q) for q in cell_probs(fitted, cuts)])

cuts = [5000 * k for k in range(11)]
observed = [104, 120, 81, 54, 45, 32, 26, 18, 12, 8]
(mu, sigma), chi2 = min_chi_square(
    lambda m, s: Truncated(Lognormal(m, abs(s)), upper=50000.0), [9.5, 1.1],
    cuts=cuts, observed=observed)
show("P2.43 mu", 9.495111, mu, 4)
show("P2.43 sigma", 1.084180, abs(sigma), 4)
```

```text
Ex 2.11 mu                               book       6.6916   prospicio       6.6916
Ex 2.11 sigma                            book       2.6965   prospicio       2.6965
Ex 2.11 chi-square                       book        4.691   prospicio        4.691
Ex 2.11 expected book  [360, 122, 121, 63, 27, 16, 10, 7, 5, 4, 34]
            prospicio  [360, 122, 121, 63, 27, 16, 10, 7, 5, 4, 34]
P2.43 mu                                 book       9.4951   prospicio       9.4951
P2.43 sigma                              book       1.0842   prospicio       1.0842
```

With individual claims, prospicio's `fit_severity` does the same job by maximum
likelihood: each claim is conditioned on exceeding its own reporting threshold
(`reporting_thresholds`, the truncation) and a claim at a limit enters as "at
least this" (`censored`). The monograph prints no claim-level data, so the next
block draws 20,000 ground-up claims from Ex 2.11's fitted lognormal, applies its
deductible and limit, and fits three times: ignoring both modifications,
handling the limit only, and handling both:

```python
truth = Lognormal(6.6916, 2.6965)
ground_up = truth.sample(20_000, seed=2)
kept = [x for x in ground_up if x > 1000.0]          # the deductible hides the rest
recorded = [min(x, 200_000.0) for x in kept]
at_limit = [x >= 200_000.0 for x in kept]
print(f"{len(kept):,} of 20,000 claims recorded, {sum(at_limit)} at the limit")
for label, kw in [
    ("ignoring both", {}),
    ("censoring only", {"censored": at_limit}),
    ("truncation and censoring", {"censored": at_limit,
                                  "reporting_thresholds": [1000.0] * len(kept)}),
]:
    fit = fit_severity("lognormal", recorded, **kw)
    m, s = fit.parameters["meanlog"], fit.parameters["sdlog"]
    se = fit.standard_errors["meanlog"]
    print(f"{label:<25s} mu {m:5.2f} (se {se:.2f})  sigma {s:4.2f}")
```

```text
9,551 of 20,000 claims recorded, 415 at the limit
ignoring both             mu  8.94 (se 0.01)  sigma 1.46
censoring only            mu  8.96 (se 0.02)  sigma 1.52
truncation and censoring  mu  6.76 (se 0.15)  sigma 2.67
```

Only the last fit recovers the population, (6.69, 2.70), within its standard
errors; the other two describe the claims an insurer with this deductible and
limit sees, with a far thinner tail. The first two are close because only a
few percent of claims reach the limit; the deductible, which hides more than
half of the population, does the damage.

## 5. Inflation and claim size

A uniform inflation of t − 1 multiplies every claim by t, and each family stays
in its family with its scale multiplied (a lognormal's μ moves by log t). The
limited mean does not grow at the full rate, because the cap does not inflate:
E[tX; l] = t E[X; l/t]. Ex 2.15 puts 10% and −5% on a Pareto (2; 3,000) capped
at 8,000:

```python
p = shifted_pareto(2.0, 3000.0)
base = p.lev(8000.0)
up = shifted_pareto(2.0, 3000.0 * 1.10).lev(8000.0)   # rescale the family
down = 0.95 * p.lev(8000.0 / 0.95)                    # or E[tX; l] = t E[X; l/t]
show("Ex 2.15 E[X; 8,000]", 2_182, base)
show("        after +10%", 2_336, up)
show("        effective rate, %", 7.1, 100 * (up / base - 1), 1)
show("        after -5%", 2_101, down)
show("        effective rate, %", -3.7, 100 * (down / base - 1), 1)
```

```text
Ex 2.15 E[X; 8,000]                      book        2,182   prospicio        2,182
        after +10%                       book        2,336   prospicio        2,336
        effective rate, %                book          7.1   prospicio          7.1
        after -5%                        book        2,101   prospicio        2,101
        effective rate, %                book         -3.7   prospicio         -3.7
```

A cap absorbs part of any inflation: a basic-limits rate trends more slowly
than the ground-up severity. Ex 2.16 has inflation that rises with the size of
the claim, t(x) = 0.96 x^0.0183. A power of a lognormal is lognormal, so the
inflated claims are lognormal (1.0183 μ + log 0.96, 1.0183 σ):

```python
before = Lognormal(7.2, 0.476)
after = Lognormal(1.0183 * 7.2 + math.log(0.96), 1.0183 * 0.476)
show("Ex 2.16 mean before", 1_500, before.mean())
show("        mean after", 1_650, after.mean())
print(f"        inflation of a claim of 1,000: {0.96 * 1000**0.0183 - 1:.2%}, "
      f"of 5,000: {0.96 * 5000**0.0183 - 1:.2%}")
```

```text
Ex 2.16 mean before                      book        1,500   prospicio        1,500
        mean after                       book        1,650   prospicio        1,650
        inflation of a claim of 1,000: 8.94%, of 5,000: 12.19%
```

## 6. Excess claims and layers

Above an attachment a, the claims that pierce it, measured from a, are the
**excess** claims X_a. Their mean is the mean excess
e(a) = E[(X − a)+] / Pr{X > a}, and a layer of width l above a, per claim that
reaches the layer, has mean (E[X; a + l] − E[X; a]) / Pr{X > a}. In prospicio
the numerators are `stop_loss(a)` and `layer(l, a)`.

How e(x) moves with x tells families apart: constant for the exponential,
falling for a gamma with shape above 1, rising in a straight line for the
shifted Pareto (e(x) = (x + β)/(α − 1)), rising and bending for the lognormal.
P2.39's sister question P5.2 takes five distributions with mean 2,000 and asks
for e(3,000); the uniform on [0, 4,000] is prospicio's `Beta(1, 1, scale)`.
Ex 5.4 fits a Pareto by regressing the empirical e(x) on x: a slope of 0.257335
and an intercept of 1,208.50 give α − 1 = 1/slope and β = intercept · (α − 1):

```python
cases = [
    ("uniform [0; 4,000]", 500, Beta(1.0, 1.0, 4000.0)),
    ("gamma (2; 1,000)", 1_250, Gamma(2.0, 1000.0)),
    ("exponential 2,000", 2_000, Gamma(1.0, 2000.0)),
    ("Pareto (3; 4,000)", 3_500, shifted_pareto(3.0, 4000.0)),
    ("lognormal (5.9809; 1.8)", 8_518, Lognormal(5.9809, 1.8)),
]
for name, book, x in cases:
    show(f"P5.2  e(3,000), {name}", book, x.stop_loss(3000.0) / survival(x, 3000.0))

alpha = 1 + 1 / 0.257335
beta = 1208.50 * (alpha - 1)
show("Ex 5.4  alpha", 4.88599, alpha, 5)
show("Ex 5.4  beta", 4_696.22, beta, 2)
p = shifted_pareto(alpha, beta)
xs = [500, 1000, 2000, 5000, 10000]
print("Ex 5.4  F(x)     book ", [0.3900, 0.6106, 0.8233, 0.9711, 0.9962])
print("                 ours ", [round(p.cdf(x), 4) for x in xs])
print("        E[X; x]  book ", [393, 638, 904, 1136, 1194])
print("                 ours ", [round(p.lev(x)) for x in xs])
```

```text
P5.2  e(3,000), uniform [0; 4,000]       book          500   prospicio          500
P5.2  e(3,000), gamma (2; 1,000)         book        1,250   prospicio        1,250
P5.2  e(3,000), exponential 2,000        book        2,000   prospicio        2,000
P5.2  e(3,000), Pareto (3; 4,000)        book        3,500   prospicio        3,500
P5.2  e(3,000), lognormal (5.9809; 1.8)  book        8,518   prospicio        8,518
Ex 5.4  alpha                            book      4.88599   prospicio      4.88599
Ex 5.4  beta                             book     4,696.22   prospicio     4,696.21
Ex 5.4  F(x)     book  [0.39, 0.6106, 0.8233, 0.9711, 0.9962]
                 ours  [0.39, 0.6106, 0.8233, 0.9711, 0.9962]
        E[X; x]  book  [393, 638, 904, 1136, 1194]
                 ours  [393, 638, 904, 1136, 1194]
```

Ex 5.7 prices a layer of 4,000 above 5,000 under a Pareto (2; 3,000). The
excess of a shifted Pareto over d is again a shifted Pareto, (α; β + d), so
the layer's mean per excess claim is the limited mean of a Pareto (2; 8,000)
at 4,000, and prospicio's `layer` gives the same:

```python
p = shifted_pareto(2.0, 3000.0)
show("Ex 5.7  E[X; 5,000]", 1_875, p.lev(5000.0))
show("        E[X; 9,000]", 2_250, p.lev(9000.0))
show("        layer mean per excess claim", 2_667, p.layer(4000.0, 5000.0) / p.survival(5000.0))
show("        as Pareto (2; 8,000) at 4,000", 2_667, shifted_pareto(2.0, 8000.0).lev(4000.0))
```

```text
Ex 5.7  E[X; 5,000]                      book        1,875   prospicio        1,875
        E[X; 9,000]                      book        2,250   prospicio        2,250
        layer mean per excess claim      book        2,667   prospicio        2,667
        as Pareto (2; 8,000) at 4,000    book        2,667   prospicio        2,667
```

Layers are where coefficients of variation explode. Ex 5.8 and P5.10 measure
them for the lognormal (5.9809; 1.8), whose mean is 2,000:

```python
ln = Lognormal(5.9809, 1.8)
s3 = survival(ln, 3000.0)
excess_mean = ln.stop_loss(3000.0) / s3
layer_mean = ln.layer(5000.0, 3000.0) / s3
layer_m2 = Truncated(ln).layer_second_moment(5000.0, 3000.0) / s3
show("Ex 5.8  F(3,000)", 0.869761, ln.cdf(3000.0), 6)
show("        E[X; 3,000]", 891, ln.lev(3000.0))
show("        E[X^2; 3,000]", 1_853_050, limited_second_moment(ln, 3000.0))
show("        E[X^2; 8,000]", 5_774_970, limited_second_moment(ln, 8000.0))
show("        CV[X]", 4.9531, ln.std() / ln.mean(), 4)
show("        E[X_3K], the excess mean", 8_515, excess_mean)
show("        CV of the layer 5K xs 3K", 0.6452, math.sqrt(layer_m2 - layer_mean**2) / layer_mean, 4)
show("P5.10   CV[X; 3,000]",  1.1551,
     math.sqrt(limited_second_moment(ln, 3000.0) - ln.lev(3000.0) ** 2) / ln.lev(3000.0), 4)
show("P5.10   CV[X; 8,000]", 1.5959,
     math.sqrt(limited_second_moment(ln, 8000.0) - ln.lev(8000.0) ** 2) / ln.lev(8000.0), 4)
# The monograph's own rounded inputs reproduce its printed layer CV:
m = (1276 - 891) / 0.130239
m2 = (5_774_970 - 1_853_050 - 2 * 3000 * (1276 - 891)) / 0.130239
print(f"        layer CV from the printed, rounded inputs: {math.sqrt(m2 - m * m) / m:.4f}")
```

```text
Ex 5.8  F(3,000)                         book     0.869761   prospicio     0.869761
        E[X; 3,000]                      book          891   prospicio          891
        E[X^2; 3,000]                    book    1,853,050   prospicio    1,853,050
        E[X^2; 8,000]                    book    5,774,970   prospicio    5,774,970
        CV[X]                            book       4.9531   prospicio       4.9532
        E[X_3K], the excess mean         book        8,515   prospicio        8,518
        CV of the layer 5K xs 3K         book       0.6452   prospicio       0.6385
P5.10   CV[X; 3,000]                     book       1.1551   prospicio       1.1561
P5.10   CV[X; 8,000]                     book       1.5959   prospicio       1.5955
        layer CV from the printed, rounded inputs: 0.6452
```

The inputs agree to the printed digits. The derived numbers differ in the third
or fourth figure, and the last line shows why: the monograph carries the
rounded limited means (891 for 890.56, 1,276 for 1,276.24) into differences that
cancel most of their size, which turns a rounding of half a unit into a
percent of the layer's CV. Its excess mean of 8,515 is the same rounding: P5.2
prints the same quantity as 8,518, prospicio's value. prospicio's `layer_cv` is
the CV of the layer loss per ground-up claim, zeros included, a different
quantity (2.90 here), suited to pricing the layer from ground-up counts.

P5.12 splits a policy with a 100 deductible and a 3,000 limit into three
layers, for a Pareto (3; 5,000) with 5 ground-up claims expected: the share of
claims that reach each layer, the expected count and severity in it, and its
expected loss, which add up to the unlimited total:

```python
p = shifted_pareto(3.0, 5000.0)
book = {"[0; 100]": (1.0000, 97.08, 485), "(100; 3,000]": (0.9423, 1_513.66, 7_132),
        "(3,000; inf)": (0.2441, 4_000.00, 4_883)}
total = 0.0
for (name, (b_share, b_sev, b_loss)), (a, width) in zip(
        book.items(), [(0.0, 100.0), (100.0, 2900.0), (3000.0, math.inf)]):
    reach = p.survival(a)
    loss = p.stop_loss(a) if width == math.inf else p.layer(width, a)
    total += 5 * loss
    show(f"P5.12 {name} share reaching", b_share, reach, 4)
    show(f"P5.12 {name} severity", b_sev, loss / reach, 2)
    show(f"P5.12 {name} expected loss", b_loss, 5 * loss)
show("P5.12 total = 5 x mean", 12_500, total)
```

```text
P5.12 [0; 100] share reaching            book       1.0000   prospicio       1.0000
P5.12 [0; 100] severity                  book        97.08   prospicio        97.08
P5.12 [0; 100] expected loss             book          485   prospicio          485
P5.12 (100; 3,000] share reaching        book       0.9423   prospicio       0.9423
P5.12 (100; 3,000] severity              book     1,513.66   prospicio     1,513.66
P5.12 (100; 3,000] expected loss         book        7,132   prospicio        7,132
P5.12 (3,000; inf) share reaching        book       0.2441   prospicio       0.2441
P5.12 (3,000; inf) severity              book     4,000.00   prospicio     4,000.00
P5.12 (3,000; inf) expected loss         book        4,883   prospicio        4,883
P5.12 total = 5 x mean                   book       12,500   prospicio       12,500
```

Deductibles truncate counts as well as amounts. Ex 5.3 fits a lognormal by
minimum chi-square to 300 claims net of a 1,500 deductible and censored by a
50,000 limit, then estimates how many claims the deductible hid; P5.23 does the
same with a 500 deductible. From the printed parameters prospicio gives the
printed counts:

```python
for name, book, (mu, sigma), n, d in [
    ("Ex 5.3", 43, (8.67593, 1.18109), 300, 1500.0),
    ("P5.23", 125, (7.960294, 1.428801), 1000, 500.0),
]:
    f = Lognormal(mu, sigma).cdf(d)
    show(f"{name} claims hidden by the deductible", book, n * f / (1 - f))
```

```text
Ex 5.3 claims hidden by the deductible   book           43   prospicio           43
P5.23 claims hidden by the deductible    book          125   prospicio          125
```

The fits themselves do not reproduce. At Ex 5.3's printed parameters the
chi-square statistic, as the monograph defines it, is 1.806 where it prints
1.6610, and its minimum is 1.790 at (8.6869, 1.1684); for P5.23 the printed
parameters are further from the minimum under every reading of the cells
tried. Both are spreadsheet solver results (a footnote in the monograph warns
they can vary); the estimates of hidden claims hardly move.

## 7. Inflation above an attachment: the leveraging effect

Inflation does more to an excess layer than to the ground-up loss. With a fixed
attachment, inflation pushes claims that were below it into the layer (more
excess claims) and makes those already above it larger (a larger excess
severity), so the layer's loss grows faster than the inflation rate. The
monograph splits the growth into a severity factor, a count factor and their
product. Ex 5.10 to 5.12 use the Pareto (2; 3,000), an attachment of 5,000,
10% inflation and, in Ex 5.12, 5% growth in the ground-up count:

```python
p, t = shifted_pareto(2.0, 3000.0), 1.10
e0 = p.stop_loss(5000.0) / p.survival(5000.0)
e1 = t * p.stop_loss(5000.0 / t) / p.survival(5000.0 / t)      # e of tX at 5,000
show("Ex 5.10 e(5,000)", 8_000, e0)
show("        e(5,000) after inflation", 8_300, e1)
show("        excess severity trend, %", 3.75, 100 * (e1 / e0 - 1), 2)
layer0 = p.layer(4000.0, 5000.0) / p.survival(5000.0)
layer1 = t * p.layer(4000.0 / t, 5000.0 / t) / p.survival(5000.0 / t)
show("        layer 4K xs 5K severity, trended", 2_701, layer1)
show("        layer severity trend, %", 1.27, 100 * (layer1 / layer0 - 1), 2)
count = p.survival(5000.0 / t) / p.survival(5000.0)
show("Ex 5.11 excess count factor", 1.1241, count, 4)
show("Ex 5.12 excess count trend, %", 18.03, 100 * (1.05 * count - 1), 2)
show("        excess loss trend, %", 22.46, 100 * (1.05 * count * e1 / e0 - 1), 2)
show("        of which from claim size, %", 16.6, 100 * (count * e1 / e0 - 1), 1)
```

```text
Ex 5.10 e(5,000)                         book        8,000   prospicio        8,000
        e(5,000) after inflation         book        8,300   prospicio        8,300
        excess severity trend, %         book         3.75   prospicio         3.75
        layer 4K xs 5K severity, trended book        2,701   prospicio        2,699
        layer severity trend, %          book         1.27   prospicio         1.22
Ex 5.11 excess count factor              book       1.1241   prospicio       1.1241
Ex 5.12 excess count trend, %            book        18.03   prospicio        18.03
        excess loss trend, %             book        22.46   prospicio        22.46
        of which from claim size, %      book         16.6   prospicio         16.6
```

A 10% inflation becomes 16.6% on the excess loss (22.5% with the count growth).
The layer severity printed as 2,701 is computed from rounded limited means;
prospicio's 2,699 is exact, and so its trend is 1.22% rather than 1.27%. The
same decomposition for an exponential with mean 1,000 and a layer of 2,000
above 1,000 (P5.1), and for the lognormal of Ex 5.8 (P5.21 and P5.22):

```python
x = Gamma(1.0, 1000.0)
s = x.survival(1000.0)
mean = x.layer(2000.0, 1000.0) / s
show("P5.1  layer mean per excess claim", 865, mean)
show("      its variance", 440_343, x.layer_second_moment(2000.0, 1000.0) / s - mean**2)
show("      excess claims, of 20", 7.4, 20 * s, 1)
sev = t * x.layer(2000.0 / t, 1000.0 / t) / x.survival(1000.0 / t) / mean - 1
cnt = x.survival(1000.0 / t) / s - 1
show("      severity trend, %", 6.6, 100 * sev, 1)
show("      count trend, %", 9.5, 100 * cnt, 1)
show("      loss trend, %", 16.7, 100 * ((1 + sev) * (1 + cnt) - 1), 1)
ln = Lognormal(5.9809, 1.8)
for a, book in [(3000.0, 8.9), (8000.0, 11.5)]:
    show(f"P5.21 count trend above {a:,.0f}, %", book,
         100 * (survival(ln, a / t) / survival(ln, a) - 1), 1)
for width, book in [(2000.0, 9.5), (5000.0, 10.2)]:
    show(f"P5.22 loss trend, layer to {3000 + width:,.0f}, %", book,
         100 * (t * ln.layer(width / t, 3000.0 / t) / ln.layer(width, 3000.0) - 1), 1)
```

```text
P5.1  layer mean per excess claim        book          865   prospicio          865
      its variance                       book      440,343   prospicio      440,343
      excess claims, of 20               book          7.4   prospicio          7.4
      severity trend, %                  book          6.6   prospicio          6.6
      count trend, %                     book          9.5   prospicio          9.5
      loss trend, %                      book         16.7   prospicio         16.7
P5.21 count trend above 3,000, %         book          8.9   prospicio          8.9
P5.21 count trend above 8,000, %         book         11.5   prospicio         11.5
P5.22 loss trend, layer to 5,000, %      book          9.5   prospicio          9.5
P5.22 loss trend, layer to 8,000, %      book         10.2   prospicio         10.2
```

A layer with an upper limit leverages less than an unlimited excess cover, and
less the narrower it is, because the limit caps the growth of the claims that
were already in it (P5.22's 9.5% against P5.21's 8.9% count growth alone).

With a Poisson count, the moments of the aggregate layer loss follow from the
layer's limited moments. Ex 5.13 takes 15 expected ground-up claims and the
layer of 5,000 above 3,000 of the same lognormal:

```python
lam, a, w = 15.0, 3000.0, 5000.0
mean_s = lam * ln.layer(w, a)
var_s = lam * Truncated(ln).layer_second_moment(w, a)
show("Ex 5.13 E[S]", 5_775, mean_s)
show("        Var[S]", 24_178_800, var_s)
show("        expected layer claims", 1.9536, lam * survival(ln, a), 4)
```

```text
Ex 5.13 E[S]                             book        5,775   prospicio        5,785
        Var[S]                           book   24,178,800   prospicio   24,117,273
        expected layer claims            book       1.9536   prospicio       1.9536
```

Again the printed mean and variance are computed from the rounded limited
values of Ex 5.8 (15 × (1,276 − 891) = 5,775); the expected count, which
needs no difference, agrees.

## 8. Increased limit factors

An increased limit factor I(l) is the premium for a limit l over the premium for
the basic limit b. Without expenses and risk load it is E[X; l]/E[X; b], which
`prospicio.pricing.ilf` returns. The monograph adds allocated loss adjustment
expense (ALAE) two ways: a fixed amount e per claim, giving
(E[X; l] + e)/(E[X; b] + e), or a proportion u of the indemnity, which cancels
out of the ratio. Ex 6.3 uses a lognormal (7.000; 2.400), a basic limit of
100,000, e = 2,200 or u = 20%, a frequency giving 0.2 expected claims and a 35%
variable expense ratio; Ex 6.4 prices the layers between limits:

```python
ln = Lognormal(7.0, 2.4)
b = 100_000.0
limits = [100e3, 500e3, 1e6, 2e6, 3e6, 4e6, 5e6]
print("Ex 6.3  E[X; l]      book ", [8896, 13626, 15345, 16738, 17390, 17782, 18048])
print("                     ours ", [round(ln.lev(l)) for l in limits])
print("        I(l), e      book ", [1.0, 1.4263, 1.5812, 1.7067, 1.7655, 1.8008, 1.8248])
print("                     ours ", [round((ln.lev(l) + 2200) / (ln.lev(b) + 2200), 4) for l in limits])
print("        I(l), u      book ", [1.0, 1.5317, 1.7249, 1.8815, 1.9548, 1.9989, 2.0288])
print("                     ours ", [round(ilf(ln, l, b), 4) for l in limits])
basic_e = 0.2 * (ln.lev(b) + 2200) / (1 - 0.35)
basic_u = 0.2 * ln.lev(b) * 1.2 / (1 - 0.35)
show("Ex 6.3  basic premium, e", 3_414, basic_e)
show("        1M premium, e", 5_398, basic_e * (ln.lev(1e6) + 2200) / (ln.lev(b) + 2200))
show("        basic premium, u", 3_285, basic_u)
show("        1M premium, u", 5_666, basic_u * ilf(ln, 1e6, b))
for (lo, hi), book in zip([(1e6, 2e6), (2e6, 3e6), (3e6, 4e6), (4e6, 5e6)], [428, 201, 121, 82]):
    show(f"Ex 6.4  layer {lo / 1e6:.0f}M to {hi / 1e6:.0f}M premium, e", book,
         basic_e * (ln.layer(hi - lo, lo)) / (ln.lev(b) + 2200))
```

```text
Ex 6.3  E[X; l]      book  [8896, 13626, 15345, 16738, 17390, 17782, 18048]
                     ours  [8896, 13626, 15345, 16738, 17390, 17782, 18048]
        I(l), e      book  [1.0, 1.4263, 1.5812, 1.7067, 1.7655, 1.8008, 1.8248]
                     ours  [1.0, 1.4262, 1.5812, 1.7067, 1.7655, 1.8008, 1.8248]
        I(l), u      book  [1.0, 1.5317, 1.7249, 1.8815, 1.9548, 1.9989, 2.0288]
                     ours  [1.0, 1.5316, 1.7249, 1.8815, 1.9548, 1.9989, 2.0288]
Ex 6.3  basic premium, e                 book        3,414   prospicio        3,414
        1M premium, e                    book        5,398   prospicio        5,399
        basic premium, u                 book        3,285   prospicio        3,285
        1M premium, u                    book        5,666   prospicio        5,666
Ex 6.4  layer 1M to 2M premium, e        book          428   prospicio          429
Ex 6.4  layer 2M to 3M premium, e        book          201   prospicio          201
Ex 6.4  layer 3M to 4M premium, e        book          121   prospicio          121
Ex 6.4  layer 4M to 5M premium, e        book           82   prospicio           82
```

The fourth decimals that differ come from the monograph forming its factors
from limited means rounded to whole units; the 2M layer's premium of 428
against 429 is the factor 0.1255 rounded before multiplying. P6.5 is the same
exercise for a lognormal (6.8; 2.6) with e = 2,500 or u = 25%, and P6.7 a hybrid
ALAE, 20% of each claim up to 750,000 and a flat 150,000 above it, so that
E_l[Y] = E[X; l] + 0.2 E[X; min(l, 750,000)]:

```python
ln2 = Lognormal(6.8, 2.6)
limits2 = [100e3, 250e3, 500e3, 1e6, 2e6, 5e6]
print("P6.5  E[X; l]   book ", [9178, 12548, 15180, 17702, 19968, 22404])
print("                ours ", [round(ln2.lev(l)) for l in limits2])
print("      I(l), e   book ", [1.0, 1.2885, 1.5139, 1.7299, 1.9240, 2.1325])
print("                ours ", [round((ln2.lev(l) + 2500) / (ln2.lev(b) + 2500), 4) for l in limits2])
print("      I(l), u   book ", [1.0, 1.3671, 1.6539, 1.9288, 2.1756, 2.4410])
print("                ours ", [round(ilf(ln2, l, b), 4) for l in limits2])


def hybrid(l):
    return ln.lev(l) + 0.2 * ln.lev(min(l, 750e3))


limits3 = [100e3, 500e3, 750e3, 1e6, 2e6, 3e6, 4e6, 5e6]
print("P6.7  I(l)      book ", [1.0, 1.5317, 1.6488, 1.7122, 1.8427, 1.9038, 1.9405, 1.9655])
print("                ours ", [round(hybrid(l) / hybrid(b), 4) for l in limits3])
```

```text
P6.5  E[X; l]   book  [9178, 12548, 15180, 17702, 19968, 22404]
                ours  [9178, 12548, 15180, 17702, 19968, 22404]
      I(l), e   book  [1.0, 1.2885, 1.5139, 1.7299, 1.924, 2.1325]
                ours  [1.0, 1.2885, 1.5139, 1.7299, 1.924, 2.1325]
      I(l), u   book  [1.0, 1.3671, 1.6539, 1.9288, 2.1756, 2.441]
                ours  [1.0, 1.3671, 1.6539, 1.9288, 2.1756, 2.441]
P6.7  I(l)      book  [1.0, 1.5317, 1.6488, 1.7122, 1.8427, 1.9038, 1.9405, 1.9655]
                ours  [1.0, 1.5316, 1.6488, 1.7123, 1.8427, 1.9038, 1.9405, 1.9654]
```

## 9. Risk loads

Higher limits carry more uncertainty per unit of expected loss, and the
monograph loads each limit's premium with a risk load ρ(l) built from the
variance of the aggregate loss per expected claim, E[X²; l] + δ E[X; l]², where
δ = Var[N]/E[N] − 1 is zero for a Poisson count. Two forms: k times its square
root (the standard deviation method) and k times itself (the variance method).
Ex 6.5 takes the standard deviation method with k = 0.0277 and δ = 0 on the
lognormal of Ex 6.3 with u = 20%, and averages the factors over a mix of
policies by limit:

```python
def sd_load(sev, l, k, delta):
    return k * math.sqrt(limited_second_moment(sev, l) + delta * sev.lev(l) ** 2)


def loaded_ilf(sev, l, base, k, delta, load=sd_load, u=0.2):
    def premium(x):
        return (1 + u) * sev.lev(x) + load(sev, x, k, delta)
    return premium(l) / premium(base)


print("Ex 6.5  rho(l)     book ", [627, 1473, 2013, 2663, 3090, 3410, 3668])
print("                   ours ", [round(sd_load(ln, l, 0.0277, 0.0)) for l in limits])
loaded = [loaded_ilf(ln, l, b, 0.0277, 0.0) for l in limits]
print("        loaded I   book ", [1.0, 1.5770, 1.8074, 2.0128, 2.1197, 2.1897, 2.2407])
print("                   ours ", [round(x, 4) for x in loaded])
mix = [0.15, 0.10, 0.30, 0.20, 0.10, 0.10, 0.05]
plain = sum(w * ilf(ln, l, b) for w, l in zip(mix, limits))
risky = sum(w * x for w, x in zip(mix, loaded))
show("        average I, no load", 1.6938, plain, 4)
show("        average I, loaded", 1.7955, risky, 4)
show("        the load's effect, %", 6.0, 100 * (risky / plain - 1), 1)
```

```text
Ex 6.5  rho(l)     book  [627, 1473, 2013, 2663, 3090, 3410, 3668]
                   ours  [627, 1473, 2013, 2663, 3090, 3410, 3668]
        loaded I   book  [1.0, 1.577, 1.8074, 2.0128, 2.1197, 2.1897, 2.2407]
                   ours  [1.0, 1.577, 1.8074, 2.0127, 2.1197, 2.1897, 2.2407]
        average I, no load               book       1.6938   prospicio       1.6938
        average I, loaded                book       1.7955   prospicio       1.7954
        the load's effect, %             book          6.0   prospicio          6.0
```

P6.8 repeats it for a Pareto (3; 6,000) with k = 0.5 and δ = 0.1. P6.10 takes the
variance method (k = 0.0000005, δ = 0) on a Pareto with α = 0.78, so heavy that
its mean is infinite, from a basic limit of 1,000,000, and asks for the factor
of each additional million of limit:

```python
p = shifted_pareto(3.0, 6000.0)
limits4 = [1000, 2000, 3000, 4000, 5000, 7500, 10000]
print("P6.8   I, no load  book ", [1.0, 1.6490, 2.0940, 2.4123, 2.6478, 3.0247, 3.2392])
print("                   ours ", [round(ilf(p, l, 1000), 4) for l in limits4])
print("       I, loaded   book ", [1.0, 1.6787, 2.1645, 2.5267, 2.8055, 3.2805, 3.5759])
print("                   ours ", [round(loaded_ilf(p, l, 1000, 0.5, 0.1), 4) for l in limits4])


def var_load(sev, l, k, delta):
    return k * (limited_second_moment(sev, l) + delta * sev.lev(l) ** 2)


p = shifted_pareto(0.78, 100.0)
millions = [m * 1e6 for m in range(1, 11)]
f = [loaded_ilf(p, l, 1e6, 5e-7, 0.0, load=var_load, u=0.0) for l in millions]
print("P6.10  I(l)        book ", [1.0, 1.3858, 1.7458, 2.1014, 2.4585, 2.8194, 3.1849,
                                    3.5553, 3.9309, 3.4114])
print("                   ours ", [round(x, 4) for x in f])
print("       layer factor book ", [0.3556, 0.3571, 0.3609, 0.3655, 0.3705, 0.3755, 0.3806])
print("                   ours ", [round(hi - lo, 4) for lo, hi in zip(f[2:], f[3:])])
```

```text
P6.8   I, no load  book  [1.0, 1.649, 2.094, 2.4123, 2.6478, 3.0247, 3.2392]
                   ours  [1.0, 1.649, 2.094, 2.4123, 2.6478, 3.0247, 3.2392]
       I, loaded   book  [1.0, 1.6787, 2.1645, 2.5267, 2.8055, 3.2805, 3.5759]
                   ours  [1.0, 1.6787, 2.1645, 2.5267, 2.8055, 3.2805, 3.5759]
P6.10  I(l)        book  [1.0, 1.3858, 1.7458, 2.1014, 2.4585, 2.8194, 3.1849, 3.5553, 3.9309, 3.4114]
                   ours  [1.0, 1.3858, 1.7458, 2.1014, 2.4585, 2.8194, 3.1849, 3.5553, 3.9309, 4.3114]
       layer factor book  [0.3556, 0.3571, 0.3609, 0.3655, 0.3705, 0.3755, 0.3806]
                   ours  [0.3556, 0.3571, 0.3609, 0.3655, 0.3705, 0.3755, 0.3806]
```

P6.10's last printed factor, 3.4114, is a misprint for 4.3114 (its own layer
factor of 0.3806 on top of 3.9309 gives 4.3115). The point of the problem is in
the last row: the variance load grows so fast with the limit that each
additional million costs more than the one below it. A layer higher up should
never cost more than a layer of the same width below it (fewer claims reach
it), so the factors are inconsistent, a known weakness of the variance method
for heavy tails.

## 10. Deductibles

A straight deductible d removes min(X, d) from every claim. The monograph's
**credit** C(d) is the share of the basic-limit premium it eliminates:
(E[X; d] + e F(d)) / (E[X; b] + e), with e = 0 when ALAE is a proportion. It is
a loss elimination ratio measured against the basic limit, where prospicio's
`loss_elimination_ratio(sev, d)` measures against the unlimited mean
E[X; d]/E[X]. A **franchise** deductible pays the whole claim once it exceeds
d, so it eliminates only the claims below d; a **diminishing** deductible
shrinks from d at a claim of d to nothing at a claim of D, eliminating
E[X; d] − d (E[X; D] − E[X; d])/(D − d) per claim. Ex 6.7 to 6.9 use the
lognormal of Ex 6.3, u = 20%, a frequency of 0.0005 per exposure and D = d + 1,000:

```python
base = ln.lev(b)
deds = [1000, 2000, 3000, 4000, 5000, 10000]
straight = [ln.lev(d) / base for d in deds]
franchise = [(ln.lev(d) - d * survival(ln, d)) / base for d in deds]
diminishing = [(ln.lev(d) - d * (ln.lev(d + 1000) - ln.lev(d)) / 1000) / base for d in deds]
print("Ex 6.7  C(d) straight     book ", [0.0741, 0.1249, 0.1661, 0.2016, 0.2328, 0.3534])
print("                          ours ", [round(c, 4) for c in straight])
print("        frequency x 10^6  book ", [258, 201, 169, 147, 132, 89])
print("                          ours ", [round(1e6 * 0.0005 * survival(ln, d)) for d in deds])
print("        pure premium      book ", [4.942, 4.671, 4.451, 4.262, 4.095, 3.451])
print("                          ours ", [round(0.0005 * 1.2 * (base - ln.lev(d)), 3) for d in deds])
print("Ex 6.8  C(d) franchise    book ", [0.0162, 0.0347, 0.0523, 0.0690, 0.0846, 0.1528])
print("                          ours ", [round(c, 4) for c in franchise])
print("Ex 6.9  C(d) diminishing  book ", [0.0233, 0.0424, 0.0599, 0.0766, 0.0917])
print("                          ours ", [round(c, 4) for c in diminishing[:5]])
show("Ex 6.7  1M limit, 2,000 deductible", 5_527,
     basic_u * (loaded_ilf(ln, 1e6, b, 0.0277, 0.0) - straight[1]))
print(f"prospicio's loss_elimination_ratio(2,000) = E[X; 2,000] / E[X] = "
      f"{loss_elimination_ratio(ln, 2000.0):.4f}")
```

```text
Ex 6.7  C(d) straight     book  [0.0741, 0.1249, 0.1661, 0.2016, 0.2328, 0.3534]
                          ours  [0.0741, 0.1249, 0.1661, 0.2015, 0.2328, 0.3535]
        frequency x 10^6  book  [258, 201, 169, 147, 132, 89]
                          ours  [258, 201, 169, 147, 132, 89]
        pure premium      book  [4.942, 4.671, 4.451, 4.262, 4.095, 3.451]
                          ours  [4.942, 4.671, 4.451, 4.262, 4.095, 3.451]
Ex 6.8  C(d) franchise    book  [0.0162, 0.0347, 0.0523, 0.069, 0.0846, 0.1528]
                          ours  [0.0162, 0.0347, 0.0523, 0.0689, 0.0846, 0.1528]
Ex 6.9  C(d) diminishing  book  [0.0233, 0.0424, 0.0599, 0.0766, 0.0917]
                          ours  [0.0234, 0.0424, 0.0599, 0.0763, 0.0917]
Ex 6.7  1M limit, 2,000 deductible       book        5,527   prospicio        5,527
prospicio's loss_elimination_ratio(2,000) = E[X; 2,000] / E[X] = 0.0569
```

The last line shows that the two ratios differ by the factor E[X]/E[X; b],
here more than two, for a distribution this heavy. Table 6.7
prints the franchise credit at 5,000 as 0.0864 where Table 6.6 has 0.0846;
prospicio's 0.0846 settles it. The diminishing credits that differ in the
fourth decimal are rounding again: the formula multiplies a difference of
limited means by d, so the monograph's whole-unit rounding of E[X; d] is
magnified up to fourfold at d = 4,000.

P6.13 and P6.14 do the same with twelve actual losses, a 20,000 or 12,000 limit
and a 2,000 deductible. The monograph's loss elimination ratio here is the
share of the unlimited losses that the insurer does not pay, so with the lower
limit it counts the limit's elimination too:

```python
losses = [1000, 1550, 1700, 2200, 2500, 3000, 5200, 9000, 11000, 12500, 15000, 19800]
total = sum(losses)
for name, limit, book in [("P6.13", 20000, (62_200, 0.2635, 80_200, 0.0503)),
                          ("P6.14", 12000, (50_900, 0.3973, 68_900, 0.1841))]:
    paid_straight = sum(max(min(x, limit) - 2000, 0) for x in losses)
    paid_franchise = sum(min(x, limit) for x in losses if x > 2000)
    show(f"{name} straight: insurer pays", book[0], paid_straight)
    show(f"{name} straight: LER", book[1], 1 - paid_straight / total, 4)
    show(f"{name} franchise: insurer pays", book[2], paid_franchise)
    show(f"{name} franchise: LER", book[3], 1 - paid_franchise / total, 4)
```

```text
P6.13 straight: insurer pays             book       62,200   prospicio       62,200
P6.13 straight: LER                      book       0.2635   prospicio       0.2635
P6.13 franchise: insurer pays            book       80,200   prospicio       80,200
P6.13 franchise: LER                     book       0.0503   prospicio       0.0503
P6.14 straight: insurer pays             book       50,900   prospicio       50,900
P6.14 straight: LER                      book       0.3973   prospicio       0.3973
P6.14 franchise: insurer pays            book       68,900   prospicio       68,900
P6.14 franchise: LER                     book       0.1841   prospicio       0.1841
```

With a fixed ALAE per claim the credit grows: a deductible removes the claims
below it, and their expense with them. P6.16 reruns Ex 6.7 and 6.8 with e = 500
and u = 0; P6.17 uses a Pareto (3; 10,000), a basic limit of 5,000, a frequency
of 0.005 and e = 250; P6.21 counts the claims of 500 that the deductibles of
Ex 6.7 eliminate, now and after 10% inflation:

```python
e = 500.0
cred = [(ln.lev(d) + e * ln.cdf(d)) / (base + e) for d in deds]
pp = [0.0005 * (base - ln.lev(d) + e * survival(ln, d)) for d in deds]
fr = [(ln.lev(d) - d * survival(ln, d) + e * ln.cdf(d)) / (base + e) for d in deds]
print("P6.16  straight C(d)    book ", [0.0959, 0.1501, 0.1926, 0.2283, 0.2596, 0.3783])
print("                        ours ", [round(c, 4) for c in cred])
print("       pure premium     book ", [4.247, 3.993, 3.973, 3.625, 3.478, 2.921])
print("                        ours ", [round(x, 3) for x in pp])
print("       franchise C(d)   book ", [0.0411, 0.0647, 0.0848, 0.1028, 0.1193, 0.1884])
print("                        ours ", [round(c, 4) for c in fr])

p = shifted_pareto(3.0, 10000.0)
deds2 = [0, 250, 500, 750, 1000]
base2 = p.lev(5000.0) + 250.0
print("P6.17  straight C(d)    book ", [0.0, 0.0855, 0.1648, 0.2385, 0.3071])
print("                        ours ", [round((p.lev(d) + 250 * p.cdf(d)) / base2, 4) for d in deds2])
print("       pure premium     book ", [15.139, 13.845, 12.644, 11.528, 10.489])
print("                        ours ",
      [round(0.005 * (base2 - p.lev(d) - 250 * p.cdf(d)), 3) for d in deds2])
print("       franchise C(d)   book ", [0.0088, 0.0221, 0.0391, 0.0590])
print("                        ours ",
      [round((p.lev(d) - d * p.survival(d) + 250 * p.cdf(d)) / base2, 4) for d in deds2[1:]])

print("P6.21  claims removed   book ", [242, 299, 331, 353, 368])
print("                        ours ", [round(500 * ln.cdf(d)) for d in deds[:5]])
print("       after inflation  book ", [234, 292, 324, 346, 362])
print("                        ours ", [round(500 * ln.cdf(d / 1.1)) for d in deds[:5]])
print("       share removed, % book ", [7.1, 12.0, 16.1, 19.5, 22.6])
print("                        ours ", [round(100 * ln.lev(d / 1.1) / ln.lev(b / 1.1), 1) for d in deds[:5]])
```

```text
P6.16  straight C(d)    book  [0.0959, 0.1501, 0.1926, 0.2283, 0.2596, 0.3783]
                        ours  [0.096, 0.1501, 0.1925, 0.2283, 0.2596, 0.3784]
       pure premium     book  [4.247, 3.993, 3.973, 3.625, 3.478, 2.921]
                        ours  [4.247, 3.993, 3.794, 3.625, 3.478, 2.92]
       franchise C(d)   book  [0.0411, 0.0647, 0.0848, 0.1028, 0.1193, 0.1884]
                        ours  [0.0411, 0.0647, 0.0848, 0.1028, 0.1193, 0.1884]
P6.17  straight C(d)    book  [0.0, 0.0855, 0.1648, 0.2385, 0.3071]
                        ours  [0.0, 0.0855, 0.1648, 0.2385, 0.3071]
       pure premium     book  [15.139, 13.845, 12.644, 11.528, 10.489]
                        ours  [15.139, 13.845, 12.644, 11.528, 10.489]
       franchise C(d)   book  [0.0088, 0.0221, 0.0391, 0.059]
                        ours  [0.0088, 0.0221, 0.0391, 0.059]
P6.21  claims removed   book  [242, 299, 331, 353, 368]
                        ours  [242, 299, 331, 353, 368]
       after inflation  book  [234, 292, 324, 346, 362]
                        ours  [234, 292, 324, 346, 362]
       share removed, % book  [7.1, 12.0, 16.1, 19.5, 22.6]
                        ours  [7.1, 12.0, 16.1, 19.5, 22.6]
```

P6.16's printed pure premium at 3,000, 3.973, is out of sequence (it would make
the 3,000 deductible cheaper than the 2,000 by almost nothing and the 4,000 far
cheaper again); prospicio's 3.794 fits the pattern, and the printed figure is
its middle digits transposed. Inflation erodes a fixed deductible: it
removes fewer claims and a smaller share of the loss each year, which is the
leveraging effect of section 7 seen from below. Ex 6.10 measures it for a
Pareto (2; 8,000), a 500 deductible, a frequency of 0.25, e = 50 and 5%
inflation (of the claims and of e), without a limit and with a 5,000 limit:

```python
def pure_premium(sev, top, d, e, phi=0.25):
    capped = sev.mean() if top is None else sev.lev(top)
    return phi * (capped - sev.lev(d) + survival(sev, d) * e)


now, later = shifted_pareto(2.0, 8000.0), shifted_pareto(2.0, 8400.0)
for top, book, dp in [(None, (1_893, 1_994, 5.3), 0), (5000.0, (662.7, 677.3, 2.2), 1)]:
    p0, p1 = pure_premium(now, top, 500.0, 50.0), pure_premium(later, top, 500.0, 52.5)
    label = "no limit" if top is None else "5,000 limit"
    show(f"Ex 6.10 {label}: pure premium", book[0], p0, dp)
    show(f"        after 5% inflation", book[1], p1, dp)
    show(f"        effective trend, %", book[2], 100 * (p1 / p0 - 1), 1)
```

```text
Ex 6.10 no limit: pure premium           book        1,893   prospicio        1,893
        after 5% inflation               book        1,994   prospicio        1,994
        effective trend, %               book          5.3   prospicio          5.3
Ex 6.10 5,000 limit: pure premium        book        662.7   prospicio        662.7
        after 5% inflation               book        677.3   prospicio        677.3
        effective trend, %               book          2.2   prospicio          2.2
```

The deductible raises the 5% trend to 5.3%; a limit on top cuts it to 2.2%,
because the limit absorbs more inflation than the deductible adds.

## 11. What agreed, what did not, and what prospicio does not do

Every printed number this guide recomputes agrees with prospicio to its printed
digits, or differs for a reason found in the monograph itself:

- **Rounding carried into differences** (Ex 5.8's excess mean and layer CV,
  Ex 5.10's 2,701, Ex 5.13, the fourth decimal of some ILFs, Ex 6.4's 428):
  the monograph forms differences of limited means rounded to whole units.
  Feeding its rounded inputs to the same formula reproduces its figures.
- **Arithmetic or typesetting slips**: Ex 2.7's scale 397.931 (618,850 / 1,555
  is 397.974; the tables use the printed value), P6.10's 3.4114 (4.3114),
  Table 6.7's franchise 0.0864 (0.0846), P6.16's 3.973 (3.794).
- **Solver results**: Ex 2.8's chi-square of 1.828 (1.020 at its own
  parameters, and the true minimum is 0.614 elsewhere); Ex 5.3's 1.6610 (1.806
  at its parameters, minimum 1.790) and P5.23's parameters. Ex 2.10, 2.11,
  P2.16, P2.39 and P2.43 reproduce to four decimals by the same methods.

What these chapters use that prospicio does not provide:

- **Grouped-data estimators.** Minimum chi-square, minimum limited-mean
  distance and the mean-excess regression are not in the library; the blocks
  above build them from prospicio's `cdf` and `lev` and SciPy's optimizer.
  prospicio's `fit_severity` (lognormal, gamma, Weibull, Burr) and
  `Pareto.fit` fit individual claims by maximum likelihood, with left
  truncation and censoring but not right truncation (P2.43's kind of data)
  and not interval-grouped counts.
- **The shifted (Lomax) Pareto by name**, and fitting it: it is a
  `GeneralizedPareto` or a `Burr` with δ = 1, but `fit_severity` has no
  Pareto family of this form (`GeneralizedPareto.fit_riegel` fits Riegel's
  form above a threshold).
- **Limited moments above the second.** `layer_second_moment` exists for most
  families (for the lognormal through `Truncated`), but not E[X³; l], which
  Ex 2.4 and the skewness of Ex 5.13 use.
- **Premium formulas.** Increased limit factors with ALAE, risk loads,
  deductible credits and the franchise and diminishing deductibles are a line
  each on top of `lev`; `ilf` and `loss_elimination_ratio` cover the plain
  ratios only, and `loss_elimination_ratio` measures against the unlimited
  mean, not a basic limit.
- **Empirical limited means**: `Sampled` has no `lev`, so the twelve losses of
  P6.13 are summed directly.
- **Chapter 1**, the probability background, has nothing to compute that
  prospicio adds to.

Where to go next: [severity and frequency](severity-and-frequency.md) for
choosing and fitting a severity in practice (maximum likelihood with
thresholds and limits, the Pareto family, splicing),
[aggregate distributions](aggregate-distributions.md) for the compound
distributions of the companion guide, and
[experience and exposure rating](experience-and-exposure-rating.md) for
increased limit factors, exposure curves and layer rating in reinsurance. The
companion guide, [claim counts and aggregate claims](distributions-for-actuaries-counts-aggregates.md),
covers the monograph's chapters 3 and 4 and its aggregate limits.
