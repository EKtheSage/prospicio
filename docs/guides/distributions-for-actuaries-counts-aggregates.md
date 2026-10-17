# Claim counts and aggregate claims: Bahnemann's *Distributions for Actuaries* reproduced

This is the second of two guides that recompute the worked examples and problem
answers of CAS Monograph 2, *Distributions for Actuaries* by David Bahnemann
(2015), with prospicio. The first,
[claim size, layers, limits and deductibles](distributions-for-actuaries-claim-size.md),
covers the monograph's chapters 2, 5 and 6 and explains the conventions: the
monograph's printed value is a literal in the code, printed next to
prospicio's, and every disagreement is explained. This one covers chapter 3
(claim counts), chapter 4 (aggregate claims) and the aggregate limits of
section 6.4. The words are new; read the monograph for the derivations.

## 1. Translating the parameters

The monograph writes the negative binomial in terms of its mean ν and a gamma
mixing shape α, so that Var[N] = ν + ν²/α, and later in terms of the
**contagion** parameter γ = 1/α, so that Var[N] = ν + γν². prospicio's
`NegativeBinomial(r, beta)` has mean rβ and variance rβ(1 + β), and its
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

The helper `show` prints the monograph's value and prospicio's side by side:

```python
import math

from prospicio.aggregate import compound_moments, fft, panjer, simulate_events
from prospicio.distributions import (
    Binomial, Count, Gamma, Grid, Lognormal, NegativeBinomial, Poisson, claim_count,
)


def show(label, book, ours, digits=4):
    print(f"{label:<40s} book {book:>12,.{digits}f}   prospicio {ours:>12,.{digits}f}")


def chi_square(observed, probs):
    n = sum(observed)
    return sum((o - n * p) ** 2 / (n * p) for o, p in zip(observed, probs))
```

## 2. The Poisson and the binomial

A claim count is Poisson when claims arrive independently at a steady rate; it
is the limit of a binomial with many trials and a small chance in each. Ex 3.1
counts days with a claim when each day has a 0.3% chance; Ex 3.2 has a rate of
0.04 a year over one year and a year and a half; P3.4 a rate of 0.10 over 6, 15
and 24 months:

```python
show("Ex 3.1  30 days, Pr{N = 0}", 0.9138, Binomial(30, 0.003).pmf(0))
show("        30 days, Pr{N = 1}", 0.0825, Binomial(30, 0.003).pmf(1))
show("        365 days, Pr{N = 2}", 0.2009, Binomial(365, 0.003).pmf(2))
for t, book in [(1.0, (0.9608, 0.0384, 0.0008)), (1.5, (0.9418, 0.0565, 0.0017))]:
    for n in range(3):
        show(f"Ex 3.2  {t} years, Pr{{N = {n}}}", book[n], Poisson(0.04 * t).pmf(n))
print("P3.4   Pr{N = 2}   book  [0.0012, 0.0069, 0.0164]")
print("                   ours ", [round(Poisson(0.10 * m / 12).pmf(2), 4) for m in (6, 15, 24)])
```

```text
Ex 3.1  30 days, Pr{N = 0}               book       0.9138   prospicio       0.9138
        30 days, Pr{N = 1}               book       0.0825   prospicio       0.0825
        365 days, Pr{N = 2}              book       0.2009   prospicio       0.2009
Ex 3.2  1.0 years, Pr{N = 0}             book       0.9608   prospicio       0.9608
Ex 3.2  1.0 years, Pr{N = 1}             book       0.0384   prospicio       0.0384
Ex 3.2  1.0 years, Pr{N = 2}             book       0.0008   prospicio       0.0008
Ex 3.2  1.5 years, Pr{N = 0}             book       0.9418   prospicio       0.9418
Ex 3.2  1.5 years, Pr{N = 1}             book       0.0565   prospicio       0.0565
Ex 3.2  1.5 years, Pr{N = 2}             book       0.0017   prospicio       0.0017
P3.4   Pr{N = 2}   book  [0.0012, 0.0069, 0.0164]
                   ours  [0.0012, 0.0069, 0.0164]
```

Ex 3.3 fits a Poisson to 1,000 policies, of which 868 had no claim, 118 one,
11 two, 2 three and 1 four: the maximum likelihood rate is the sample mean,
0.15. A chi-square test on the cells 0, 1 and 2 or more accepts it. A Poisson
sum of independent Poissons is Poisson, which Ex 3.8 uses for 100 policies at
0.035 each; P3.3, P3.30 and P3.31 are more of the same:

```python
policies = [868, 118, 11, 2, 1]
lam = sum(k * n for k, n in enumerate(policies)) / sum(policies)
p = Poisson(lam)
show("Ex 3.3  rate", 0.15, lam, 2)
print("        pmf   book  [0.8607, 0.1291, 0.0097, 0.0005]")
print("              ours ", [round(p.pmf(k), 4) for k in range(4)])
show("        chi-square", 2.43, chi_square([868, 118, 14], [p.pmf(0), p.pmf(1), 1 - p.cdf(1)]), 2)
show("Ex 3.8  Pr{N >= 5}, 100 x 0.035", 0.2746, 1 - Poisson(3.5).cdf(4))
for lam, book in [(2.5, 0.7576), (6.25, 0.1303), (2.0, 0.8571), (4.0, 0.4335)]:
    show(f"P3.3   Pr{{N <= 3}}, rate {lam}", book, Poisson(lam).cdf(3))
show("P3.30  Pr{N >= 6}, 20 cars x 0.30", 0.5543, 1 - Poisson(6.0).cdf(5))
show("P3.31  policies with a claim, of 8,000", 624, 8000 * (1 - Poisson(650 / 8000).pmf(0)), 0)
```

```text
Ex 3.3  rate                             book         0.15   prospicio         0.15
        pmf   book  [0.8607, 0.1291, 0.0097, 0.0005]
              ours  [0.8607, 0.1291, 0.0097, 0.0005]
        chi-square                       book         2.43   prospicio         2.45
Ex 3.8  Pr{N >= 5}, 100 x 0.035          book       0.2746   prospicio       0.2746
P3.3   Pr{N <= 3}, rate 2.5              book       0.7576   prospicio       0.7576
P3.3   Pr{N <= 3}, rate 6.25             book       0.1303   prospicio       0.1303
P3.3   Pr{N <= 3}, rate 2.0              book       0.8571   prospicio       0.8571
P3.3   Pr{N <= 3}, rate 4.0              book       0.4335   prospicio       0.4335
P3.30  Pr{N >= 6}, 20 cars x 0.30        book       0.5543   prospicio       0.5543
P3.31  policies with a claim, of 8,000   book          624   prospicio          624
```

The chi-square of 2.45 against the printed 2.43 is the monograph rounding the
expected counts to one decimal before summing.

## 3. Parameter uncertainty: mixed Poissons

A rate that is the same for every policy is rarely true. When each policy has
its own rate, drawn from a distribution, the portfolio's count is a **mixed
Poisson**: its mean is the mean rate and its variance the mean rate plus the
variance of the rates, so it is overdispersed and has more zeros and more large
counts than a Poisson with the same mean. Ex 3.4 mixes four rates over 100
policies; Ex 3.5 draws the rate from an exponential with mean 0.25, which
gives a geometric count:

```python
rates, weights = [0.10, 0.35, 0.70, 1.40], [0.20, 0.40, 0.30, 0.10]
mixed = [sum(w * Poisson(r).pmf(n) for r, w in zip(rates, weights)) for n in range(6)]
mean = sum(w * r for r, w in zip(rates, weights))
var = mean + sum(w * r * r for r, w in zip(rates, weights)) - mean**2
print("Ex 3.4  mixed pmf     book  [0.6365, 0.2556, 0.0788, 0.0218, 0.0056, 0.0013]")
print("                      ours ", [round(x, 4) for x in mixed])
print("        Poisson 0.51  book  [0.6005, 0.3063, 0.0781, 0.0133, 0.0017, 0.0002]")
print("                      ours ", [round(Poisson(mean).pmf(n), 4) for n in range(6)])
show("        Var[N]", 0.6439, var)

geometric = Count.mixed_poisson(0.25, 1.0, mixing="gamma")   # exponential: cv 1
print("Ex 3.5  pmf           book  [0.8, 0.16, 0.032, 0.0064, 0.0013, 0.0003]")
print("                      ours ", [round(geometric.pmf(n), 4) for n in range(6)])
print("        geometric          ", [round(Count.geometric(0.25).pmf(n), 4) for n in range(6)])
show("        Var[N]", 0.3125, geometric.variance())
```

```text
Ex 3.4  mixed pmf     book  [0.6365, 0.2556, 0.0788, 0.0218, 0.0056, 0.0013]
                      ours  [0.6365, 0.2556, 0.0788, 0.0218, 0.0056, 0.0013]
        Poisson 0.51  book  [0.6005, 0.3063, 0.0781, 0.0133, 0.0017, 0.0002]
                      ours  [0.6005, 0.3063, 0.0781, 0.0133, 0.0017, 0.0002]
        Var[N]                           book       0.6439   prospicio       0.6439
Ex 3.5  pmf           book  [0.8, 0.16, 0.032, 0.0064, 0.0013, 0.0003]
                      ours  [0.8, 0.16, 0.032, 0.0064, 0.0013, 0.0003]
        geometric           [0.8, 0.16, 0.032, 0.0064, 0.0013, 0.0003]
        Var[N]                           book       0.3125   prospicio       0.3125
```

When the rates follow a gamma distribution the mixed count is a negative
binomial, which is why that family is the standard overdispersed count. P3.11
mixes by a gamma with shape 2 and scale 1; P3.12 by one with mean 0.100 and
variance 0.005; P3.13 compares the variances of a Poisson, a two-point mixture
and an exponential mixture with the same mean:

```python
nb = NegativeBinomial(2.0, 1.0)
show("P3.11  mean", 2, nb.mean(), 0)
show("       variance", 4, nb.variance(), 0)
show("       Pr{N <= 3}", 0.8125, nb.cdf(3))
nb = NegativeBinomial.from_mean_variance(0.100, 0.100 + 0.005)  # Var[N] = E + Var[rate]
print("P3.12  pmf   book  [0.9070, 0.0864, 0.0062, 0.0004]")
print("             ours ", [round(nb.pmf(n), 4) for n in range(4)])
two_point_var = 0.75 + (0.75 * 0.6**2 + 0.25 * 1.2**2 - 0.75**2)
show("P3.13  Var, Poisson 0.75", 0.7500, Poisson(0.75).variance())
show("       Var, two-point mixture", 0.8175, two_point_var)
show("       Var, exponential mixture", 1.3125, Count.mixed_poisson(0.75, 1.0).variance())
```

```text
P3.11  mean                              book            2   prospicio            2
       variance                          book            4   prospicio            4
       Pr{N <= 3}                        book       0.8125   prospicio       0.8125
P3.12  pmf   book  [0.9070, 0.0864, 0.0062, 0.0004]
             ours  [0.907, 0.0864, 0.0062, 0.0004]
P3.13  Var, Poisson 0.75                 book       0.7500   prospicio       0.7500
       Var, two-point mixture            book       0.8175   prospicio       0.8175
       Var, exponential mixture          book       1.3125   prospicio       1.3125
```

## 4. The negative binomial and contagion

Ex 3.6 fits a negative binomial to 5,000 auto liability policies by the method
of moments (the sample variance exceeds the mean, so a Poisson is too thin),
and P3.15 to 10,000 policies; each is tested by chi-square on the cells 0, 1, 2
and 3 or more:

```python
def nb_moments(counts):
    n = sum(counts)
    m1 = sum(k * c for k, c in enumerate(counts)) / n
    m2 = sum(k * k * c for k, c in enumerate(counts)) / n
    return m1, m2 - m1**2


for name, counts, book in [
    ("Ex 3.6", [4429, 528, 39, 3, 1], (0.1238, 2.44285, 0.7753)),
    ("P3.15", [8956, 907, 120, 15, 2], (0.1200, 0.685714, 0.3288)),
]:
    mean, var = nb_moments(counts)
    nb = NegativeBinomial.from_mean_variance(mean, var)
    probs = [nb.pmf(0), nb.pmf(1), nb.pmf(2), 1 - nb.cdf(2)]
    observed = counts[:3] + [sum(counts[3:])]
    show(f"{name} mean", book[0], mean)
    show(f"{name} alpha = r", book[1], nb.r, 5)
    show(f"{name} chi-square", book[2], chi_square(observed, probs))
print("Ex 3.6 pmf   book  [0.8862, 0.1044, 0.0087, 0.0006]")
mean, var = nb_moments([4429, 528, 39, 3, 1])
print("             ours ", [round(NegativeBinomial.from_mean_variance(mean, var).pmf(k), 4)
                            for k in range(4)])
```

```text
Ex 3.6 mean                              book       0.1238   prospicio       0.1238
Ex 3.6 alpha = r                         book      2.44285   prospicio      2.44302
Ex 3.6 chi-square                        book       0.7753   prospicio       0.6464
P3.15 mean                               book       0.1200   prospicio       0.1200
P3.15 alpha = r                          book      0.68571   prospicio      0.68571
P3.15 chi-square                         book       0.3288   prospicio       0.3288
Ex 3.6 pmf   book  [0.8862, 0.1044, 0.0087, 0.0006]
             ours  [0.8862, 0.1044, 0.0087, 0.0006]
```

P3.15 agrees to every printed digit. Ex 3.6 does not quite: the monograph
rounds the sample variance to 0.130074 (it is 0.13007356) before dividing by its
small excess over the mean, which moves α in the fourth decimal; the fitted
probabilities agree. Its chi-square of 0.7753 we could not reproduce: with the
printed probabilities, or with α = 2.44285, the statistic on these four cells
is near 0.6 to 0.65, so the printed value probably comes from different cells
or expected counts.

**Contagion** is the monograph's other route to the negative binomial: if each
claim makes the next more likely (or a common cause raises all of them), the
count's variance is ν + γν², with γ the contagion parameter, and a Pólya urn
gives exactly the negative binomial with r = 1/γ. For any count,
γ = (Var[N] − E[N])/E[N]², which is 0 for a Poisson and negative for a
binomial. P3.28 asks for the negative binomial with mean 1 and γ = 0.2; Ex 3.9
and P3.29 combine independent policies, whose contagion shrinks as the
portfolio grows, since variances and means add:

```python
nb = claim_count(1.0, 1 + 0.2 * 1.0)       # dispersion Var/E = 1 + gamma * mean
print("P3.28  pmf   book  [0.4019, 0.3349, 0.1674, 0.0651, 0.0217, 0.0065]")
print("             ours ", [round(nb.pmf(n), 4) for n in range(6)])


def contagion(counts):
    mean = sum(c.mean() for c in counts)
    var = sum(c.variance() for c in counts)
    return (var - mean) / mean**2


show("Ex 3.9  20 policies, nu 0.15, gamma 0.4", 0.020,
     contagion([claim_count(0.15, 1 + 0.4 * 0.15)] * 20), 3)
show("P3.29  three lines", 0.1000,
     contagion([claim_count(10, 1 + 0.35 * 10), claim_count(25, 1 + 0.20 * 25), Poisson(5.0)]))
```

```text
P3.28  pmf   book  [0.4019, 0.3349, 0.1674, 0.0651, 0.0217, 0.0065]
             ours  [0.4019, 0.3349, 0.1674, 0.0651, 0.0217, 0.0065]
Ex 3.9  20 policies, nu 0.15, gamma 0.4  book        0.020   prospicio        0.020
P3.29  three lines                       book       0.1000   prospicio       0.1000
```

Two more facts from chapter 5 belong here. When only claims above an
attachment count, each claim is kept with the probability p that it pierces
the attachment: a Poisson λ becomes a Poisson pλ, and a negative binomial keeps
its contagion γ with its mean scaled to pν (prospicio has no thinning method;
build the thinned count from its mean and γ). That is how section 7 of the
first guide computes the excess counts of Ex 5.9 and 5.11.

## 5. Aggregate claims and their moments

The year's total S is the sum of N claims. Its mean is E[N]E[X], its variance
E[N]Var[X] + Var[N]E[X]², and its third moment follows the same way;
`compound_moments` returns all three for a prospicio count and severity. Ex 4.1
and P4.1 build the whole distribution of S for small discrete examples, which
`fft` does from a count and a severity on a grid (`Count.empirical` takes any
count probabilities):

```python
count = Count.empirical([0.6, 0.3, 0.1])
size = Grid(100.0, [0.0, 0.4, 0.5, 0.1])             # 100, 200, 300
agg, _ = fft(count, size, 16)
print("Ex 4.1  F_S   book  [0.6, 0.72, 0.886, 0.956, 0.989, 0.999, 1.0]")
print("              ours ", [round(agg.cdf(100.0 * k), 4) for k in range(7)])
show("        E[S]", 85, agg.mean(), 0)
show("        Var[S]", 15_055, agg.variance(), 0)

count = Count.empirical([0.20, 0.40, 0.25, 0.15])
size = Grid(500.0, [0.0, 0.1, 0.4, 0.3, 0.2])        # 500 to 2,000
agg, _ = fft(count, size, 16)
print("P4.1   F_S   book  [0.2, 0.24, 0.4025, 0.5427, 0.6795, 0.758, 0.8418, 0.9047, 0.9507]")
print("             ours ", [round(agg.cdf(500.0 * k), 4) for k in range(9)])
```

```text
Ex 4.1  F_S   book  [0.6, 0.72, 0.886, 0.956, 0.989, 0.999, 1.0]
              ours  [0.6, 0.72, 0.886, 0.956, 0.989, 0.999, 1.0]
        E[S]                             book           85   prospicio           85
        Var[S]                           book       15,055   prospicio       15,055
P4.1   F_S   book  [0.2, 0.24, 0.4025, 0.5427, 0.6795, 0.758, 0.8418, 0.9047, 0.9507]
             ours  [0.2, 0.24, 0.4025, 0.5427, 0.6795, 0.758, 0.8418, 0.9047, 0.9507]
```

The monograph's main continuous example, Ex 4.2, is a Poisson count with mean
2.5 and gamma claims with shape 3 and scale 400. Because a sum of n such claims
is a gamma with shape 3n, its distribution is an exact series over n, which
prospicio's own `Poisson` and `Gamma` evaluate; Table 4.1 is that series:

```python
lam, size = 2.5, Gamma(3.0, 400.0)
moments = compound_moments(Poisson(lam), size)
show("Ex 4.2  E[S]", 3_000, moments.mean, 0)
show("        Var[S]", 4_800_000, moments.variance, 0)
show("        skewness", 0.9129, moments.skew)


def poisson_gamma_cdf(s, lam, shape, scale, terms=400):
    n = Poisson(lam)
    if s <= 0:
        return n.pmf(0)
    return n.pmf(0) + sum(n.pmf(k) * Gamma(shape * k, scale).cdf(s) for k in range(1, terms))


S = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
print("        F(s)  book  [0.0821, 0.1096, 0.1867, 0.3755, 0.5613, 0.7152, 0.9013, 0.9723, 0.9934]")
print("              ours ", [round(poisson_gamma_cdf(s, lam, 3.0, 400.0), 4) for s in S])
```

```text
Ex 4.2  E[S]                             book        3,000   prospicio        3,000
        Var[S]                           book    4,800,000   prospicio    4,800,000
        skewness                         book       0.9129   prospicio       0.9129
        F(s)  book  [0.0821, 0.1096, 0.1867, 0.3755, 0.5613, 0.7152, 0.9013, 0.9723, 0.9934]
              ours  [0.0821, 0.1096, 0.1867, 0.3755, 0.5613, 0.7152, 0.9013, 0.9723, 0.9934]
```

## 6. Approximations by matching moments

Before computers made the exact distribution cheap, the aggregate was
approximated by a distribution with the same mean, variance and, where the
family allows, skewness. The normal matches two moments and ignores the
skewness, so it is too light in the tail; the shifted gamma matches three. The
monograph also uses the normal power and Wilson–Hilferty approximations, which
prospicio does not have; it offers the normal, gamma, lognormal, shifted gamma
and shifted lognormal through `CompoundMoments.approximate`. Table 4.3 compares
them for Ex 4.2, Table 4.4 for a far more skewed case (Poisson 10, gamma claims
with shape 0.05 and scale 6,000) and P4.13 for a Poisson 8 with gamma (0.2;
3,750):

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
for name, at_zero, lam, shape, scale, S, exact, normal, sgamma in cases:
    m = compound_moments(Poisson(lam), Gamma(shape, scale))
    print(f"{name}: mean {m.mean:,.0f}, sd {m.variance ** 0.5:,.0f}, skewness {m.skew:.4f}")
    print("  exact          book ", exact)
    print("                 ours ", [round(poisson_gamma_cdf(s, lam, shape, scale), at_zero if s == 0 else 4)
                                 for s in S])
    print("  normal         book ", normal)
    print("                 ours ", [round(m.approximate("normal").cdf(s), 4) for s in S])
    print("  shifted gamma  book ", sgamma)
    print("                 ours ", [round(m.approximate("shifted_gamma").cdf(s), 4) for s in S])
    print("  shifted lognormal   ", [round(m.approximate("shifted_lognormal").cdf(s), 4) for s in S])
```

```text
Table 4.3: mean 3,000, sd 2,191, skewness 0.9129
  exact          book  [0.0821, 0.3755, 0.7152, 0.9013, 0.9723]
                 ours  [0.0821, 0.3755, 0.7152, 0.9013, 0.9723]
  normal         book  [0.0855, 0.324, 0.676, 0.9145, 0.9888]
                 ours  [0.0855, 0.324, 0.676, 0.9145, 0.9888]
  shifted gamma  book  [0.0459, 0.368, 0.7185, 0.9038, 0.9724]
                 ours  [0.0459, 0.368, 0.7185, 0.9038, 0.9724]
  shifted lognormal    [0.0497, 0.3609, 0.7197, 0.9064, 0.9729]
Table 4.4: mean 3,000, sd 4,347, skewness 2.8293
  exact          book  [5e-05, 0.5922, 0.8401, 0.9294, 0.9846]
                 ours  [5e-05, 0.5922, 0.8401, 0.9294, 0.9846]
  normal         book  [0.2451, 0.409, 0.7549, 0.9463, 0.9997]
                 ours  [0.2451, 0.409, 0.7549, 0.9463, 0.9997]
  shifted gamma  book  [0.1228, 0.5886, 0.8402, 0.9298, 0.9847]
                 ours  [0.1228, 0.5886, 0.8402, 0.9298, 0.9847]
  shifted lognormal    [0.2182, 0.5234, 0.8341, 0.9364, 0.9874]
P4.13: mean 6,000, sd 5,196, skewness 1.5877
  exact          book  [0.0003, 0.342, 0.7774, 0.9349, 0.9954]
                 ours  [0.0003, 0.342, 0.7774, 0.9349, 0.9954]
  normal         book  [0.1241, 0.2819, 0.7181, 0.9584, 1.0]
                 ours  [0.1241, 0.2819, 0.7181, 0.9584, 1.0]
  shifted gamma  book  [0.0263, 0.3362, 0.7782, 0.9356, 0.9953]
                 ours  [0.0263, 0.3362, 0.7782, 0.9356, 0.9953]
  shifted lognormal    [0.0586, 0.3139, 0.779, 0.9402, 0.9951]
```

The shifted gamma is good to two or three units in the fourth decimal above
the mean, even with a skewness of 2.8, and poor only near zero, where the
approximation has no probability mass at zero to match the chance of no claim.
The normal misses the right tail by several percent in every case. For a
capital or stop-loss question the tail is what matters, so a moment
approximation should be at least the shifted gamma, and the exact distribution
(next section) is cheap.

## 7. Recursion and Fourier inversion

**Panjer's recursion** computes the aggregate distribution exactly when the
claim sizes sit on a grid and the count is Poisson, negative binomial or
binomial. Ex 4.4 has a Poisson 1.75 and claim sizes of 1,000 to 5,000:

```python
size = Grid(1000.0, [0.0, 0.20, 0.40, 0.20, 0.15, 0.05])
agg, _ = panjer(Poisson(1.75), size, 64)
print("Ex 4.4  f_S   book  [0.1738, 0.0608, 0.1323, 0.1046, 0.117, 0.0932, 0.0786, 0.0641]")
print("              ours ", [round(x, 4) for x in agg.probs[:8]])
print("        F_S at 9,000, 12,000, 16,000  book  [0.9121, 0.9729, 0.9955]")
print("                                      ours ", [round(agg.cdf(s), 4) for s in (9000.0, 12000.0, 16000.0)])
```

```text
Ex 4.4  f_S   book  [0.1738, 0.0608, 0.1323, 0.1046, 0.117, 0.0932, 0.0786, 0.0641]
              ours  [0.1738, 0.0608, 0.1323, 0.1046, 0.117, 0.0932, 0.0786, 0.0641]
        F_S at 9,000, 12,000, 16,000  book  [0.9121, 0.9729, 0.9955]
                                      ours  [0.9121, 0.9729, 0.9955]
```

A continuous severity is first put on a grid. The monograph's **midpoint**
method gives each grid point the probability of the interval around it, which
is prospicio's `Grid.rounding`. Ex 4.5 discretizes Ex 4.2's gamma with steps
of 100 and 20; the finer step brings the recursion within about 1% of the exact
values at small totals and closer above. P4.16 recurses a Poisson 1.35 on P4.1's
discrete sizes:

```python
S = [500, 1000, 2000, 4000, 6000, 10000]
for h, book in [(100.0, [0.1158, 0.1956, 0.3852, 0.7218, 0.9042, 0.9937]),
                (20.0, [0.1108, 0.1885, 0.3775, 0.7165, 0.9019, 0.9935])]:
    grid, _ = Grid.rounding(Gamma(3.0, 400.0), h, int(6000 / h) + 1)
    agg, _ = panjer(Poisson(2.5), grid, int(30000 / h))
    print(f"Ex 4.5  step {h:3.0f}  grid mean {grid.mean():,.2f}  variance {grid.variance():,.0f}")
    print("        F_S  book ", book)
    print("             ours ", [round(agg.cdf(s), 4) for s in S])
agg, _ = panjer(Poisson(1.35), Grid(500.0, [0.0, 0.1, 0.4, 0.3, 0.2]), 64)
print("P4.16   F_S  book  [0.2592, 0.2942, 0.4366, 0.5606, 0.6838, 0.8306, 0.9223, 0.967, 0.9872]")
print("             ours ", [round(agg.cdf(s), 4) for s in (0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000)])
```

```text
Ex 4.5  step 100  grid mean 1,199.98  variance 480,642
        F_S  book  [0.1158, 0.1956, 0.3852, 0.7218, 0.9042, 0.9937]
             ours  [0.1158, 0.1956, 0.3852, 0.7218, 0.9042, 0.9937]
Ex 4.5  step  20  grid mean 1,199.98  variance 479,846
        F_S  book  [0.1108, 0.1885, 0.3775, 0.7165, 0.9019, 0.9935]
             ours  [0.1108, 0.1885, 0.3775, 0.7165, 0.9019, 0.9935]
P4.16   F_S  book  [0.2592, 0.2942, 0.4366, 0.5606, 0.6838, 0.8306, 0.9223, 0.967, 0.9872]
             ours  [0.2592, 0.2942, 0.4366, 0.5606, 0.6838, 0.8306, 0.9223, 0.967, 0.9872]
```

Every figure agrees, and so do the grid moments the monograph prints (1,199.98
and 480,642 for the step of 100; 479,846 for 20), except the mean for the step
of 20, printed 1,199.88 for 1,199.98, a slipped digit.

The monograph's **Fourier** method is Heckman and Meyers' inversion of the
characteristic function with a piecewise linear severity (Ex 4.7). prospicio
inverts with the fast Fourier transform on a grid instead (`fft`, or `fft_auto`
which picks the grid), which handles any count it supports and much larger
claim counts than the recursion, whose starting probability e^(−λ) underflows
for large λ. On Ex 4.2 with a step of 1, a mean-preserving grid
(`Grid.local_moment`) and 2^16 points, it reproduces the exact series:

```python
grid, _ = Grid.local_moment(Gamma(3.0, 400.0), 1.0, 20001)
agg, report = fft(Poisson(2.5), grid, 2**16)
S = [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
print("Ex 4.7  Heckman-Meyers  book  [0.1094, 0.1869, 0.3757, 0.5614, 0.7151, 0.9011, 0.9722, 0.9934]")
print("        exact series         ", [round(poisson_gamma_cdf(s, 2.5, 3.0, 400.0), 4) for s in S])
print("        FFT                  ", [round(agg.cdf(s), 4) for s in S])
print(f"        aliasing error below 1e-12: {report.aliasing_error < 1e-12}")
```

```text
Ex 4.7  Heckman-Meyers  book  [0.1094, 0.1869, 0.3757, 0.5614, 0.7151, 0.9011, 0.9722, 0.9934]
        exact series          [0.1096, 0.1867, 0.3755, 0.5613, 0.7152, 0.9013, 0.9723, 0.9934]
        FFT                   [0.1096, 0.1868, 0.3756, 0.5614, 0.7152, 0.9014, 0.9723, 0.9934]
        aliasing error below 1e-12: True
```

The FFT is within a unit in the fourth decimal (the grid's own half-step),
the same accuracy as Heckman and Meyers' method.

A per-claim limit puts a probability mass in the severity, and the aggregate
inherits jumps at multiples of the limit. Ex 4.8 caps gamma (2.5; 500) claims
at 2,000 under a Poisson 1.308; the jump at a total of 2,000 from a single
claim at the limit is Pr{N = 1} Pr{X > 2,000}:

```python
size = Gamma(2.5, 500.0)
show("Ex 4.8  E[X; 2,000]", 1_147, size.lev(2000.0), 0)
show("        mass at the limit", 0.1562, size.survival(2000.0))
show("        jump in F_S at 2,000", 0.0552, Poisson(1.308).pmf(1) * size.survival(2000.0))
```

```text
Ex 4.8  E[X; 2,000]                      book        1,147   prospicio        1,147
        mass at the limit                book       0.1562   prospicio       0.1562
        jump in F_S at 2,000             book       0.0552   prospicio       0.0552
```

## 8. Simulation

Section 4.4 simulates the aggregate by drawing a count and then that many
claims, which `simulate_events` does with a seed. Ex 4.14 draws 10,000 years of
Ex 4.2 and reads F(3,000) off the sample, 0.5599 against the exact 0.5613. A
different generator draws different years, so the comparison is of accuracy,
not of digits:

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
so a sampled table is good to about two decimals, as the monograph's Table 4.11
shows; the exact methods above are better by orders of magnitude and faster.

## 9. Aggregate limits

A policy can cap the total it pays in a year as well as each claim. The
expected loss under a per-claim limit l and an aggregate limit L is E[S_l; L],
the limited expected value of the aggregate of capped claims, and the
increased limit factor for the pair is that over the basic limit's expected
loss. Ex 6.6 prices such pairs for liability policies with 1.2 expected claims,
contagion 0.1 and lognormal claims with σ = 2.4.

The monograph states μ = 7.000 for this example, but its basic-limit expected
loss of 1.2 × 21,743 and its 2M expected loss of 1.2 × 28,338 both come out of a
lognormal with μ = 7.6, not 7.0 (which gives 13,626 and 16,738, the numbers of
Ex 6.3); every figure of the example and of Table 6.4 reproduces with 7.6, so
7.000 is a misprint and the code below uses 7.6. The count is a negative
binomial with r = 1/γ = 10:

```python
size = Lognormal(7.6, 2.4)
count = claim_count(1.2, 1 + 0.1 * 1.2)
basic = count.mean() * size.lev(500e3)
show("Ex 6.6  E[S], 500K per claim", 26_092, basic, 0)
show("        E[S], 2M per claim", 34_006, count.mean() * size.lev(2e6), 0)
h = 250.0
capped_2m, _ = Grid.local_moment(size, h, int(2e6 / h) + 1)   # mass at 2M is the cap
m = compound_moments(count, capped_2m)
show("        sd of S, 2M per claim", 151_311, m.variance ** 0.5, 0)
show("        skewness", 9.4728, m.skew)
```

```text
Ex 6.6  E[S], 500K per claim             book       26,092   prospicio       26,091
        E[S], 2M per claim               book       34,006   prospicio       34,006
        sd of S, 2M per claim            book      151,311   prospicio      151,312
        skewness                         book       9.4728   prospicio       9.4728
```

The aggregate limit removes E[(S_l − L)+]. The monograph evaluates it under the
shifted gamma approximation, whose stop-loss is a gamma's (`Gamma.stop_loss`
of the unshifted part); prospicio can also compute it from the exact aggregate
by FFT, on the same grid of capped claims:

```python
def shifted_gamma_excess(moments, L):
    a = dict(moments.approximate("shifted_gamma").parameters)
    return Gamma(a["shape"], a["scale"]).stop_loss(L - a["shift"])


sg = shifted_gamma_excess(m, 3e6)
agg, _ = fft(count, capped_2m, 2**16)
exact = agg.stop_loss(3e6)
show("Ex 6.6  removed by a 3M aggregate limit", 91, sg, 0)
print(f"        the same by FFT, exactly: {exact:,.0f}")
show("        I(2M, 3M), shifted gamma", 1.2998, (m.mean - sg) / basic)
print(f"        I(2M, 3M), exact: {(m.mean - exact) / basic:.4f}")
```

```text
Ex 6.6  removed by a 3M aggregate limit  book           91   prospicio           91
        the same by FFT, exactly: 8
        I(2M, 3M), shifted gamma         book       1.2998   prospicio       1.2999
        I(2M, 3M), exact: 1.3031
```

The shifted gamma, matched to a skewness of 9.5, puts far more probability in
the far tail than the real aggregate has, and so overstates what the aggregate
limit removes: 91 against 8. Table 6.4 and P6.11 and P6.12 are built the same
way, and the gap grows as the aggregate limit comes down towards the per-claim
limit:

```python
book = {  # Table 6.4: E[S] by per-claim limit (rows) and aggregate limit
    1e6: {1e6: 29_702, 2e6: 30_306, 3e6: 30_333, 5e6: 30_335},
    2e6: {2e6: 33_524, 3e6: 33_915, 5e6: 34_002},
    3e6: {3e6: 35_421, 5e6: 35_781},
    5e6: {5e6: 37_428},
}
print(f"{'per claim':>9s} {'aggregate':>9s} {'book':>8s} {'sh. gamma':>9s} {'exact':>8s}")
exact_at = {}
for l, row in book.items():
    grid, _ = Grid.local_moment(size, h, int(l / h) + 1)
    moments = compound_moments(count, grid)
    agg, _ = fft(count, grid, 2**16)
    for L, value in row.items():
        sg = moments.mean - shifted_gamma_excess(moments, L)
        exact_at[l, L] = agg.lev(L)
        print(f"{l / 1e6:8.0f}M {L / 1e6:8.0f}M {value:8,d} {sg:9,.0f} {agg.lev(L):8,.0f}")
    exact_at[l, math.inf] = agg.mean()

for l, book_ler in [(1e6, 0.0209), (2e6, 0.0142), (3e6, 0.0112), (5e6, 0.0081)]:
    ler = 1 - exact_at[l, l] / exact_at[l, math.inf]
    print(f"P6.11  aggregate limit = per-claim limit {l / 1e6:.0f}M: share removed, "
          f"book {book_ler:.4f}, exact {ler:.4f}")
show("P6.12  premium, 1M per claim, v = 25%", 40_447, exact_at[1e6, math.inf] / 0.75, 0)
print(f"P6.12  credit for a 1M aggregate limit: book 844, "
      f"exact {(exact_at[1e6, math.inf] - exact_at[1e6, 1e6]) / 0.75:,.0f}")
```

```text
per claim aggregate     book sh. gamma    exact
       1M        1M   29,702    29,702   30,133
       1M        2M   30,306    30,306   30,334
       1M        3M   30,333    30,333   30,335
       1M        5M   30,335    30,335   30,335
       2M        2M   33,524    33,523   33,910
       2M        3M   33,915    33,915   33,999
       2M        5M   34,002    34,002   34,006
       3M        3M   35,421    35,420   35,762
       3M        5M   35,781    35,780   35,819
       5M        5M   37,428    37,427   37,703
P6.11  aggregate limit = per-claim limit 1M: share removed, book 0.0209, exact 0.0066
P6.11  aggregate limit = per-claim limit 2M: share removed, book 0.0142, exact 0.0028
P6.11  aggregate limit = per-claim limit 3M: share removed, book 0.0112, exact 0.0016
P6.11  aggregate limit = per-claim limit 5M: share removed, book 0.0081, exact 0.0008
P6.12  premium, 1M per claim, v = 25%    book       40,447   prospicio       40,446
P6.12  credit for a 1M aggregate limit: book 844, exact 269
```

The shifted gamma column reproduces the monograph's table to a unit or so
(the monograph integrated numerically), which confirms the reading of its
method and of μ = 7.6. The exact column shows the approximation's cost: where
the aggregate limit equals the per-claim limit it removes three to ten times
the loss that is really there, and the credit of P6.12 for a 1M aggregate limit
falls from 844 to about a third of that. The unlimited premium of P6.12 needs
no approximation and agrees up to the monograph's rounding of 30,335 before
dividing (the same rounding puts its basic loss at 26,092 and its standard
deviation at 151,311).

## 10. What agreed, what did not, and what prospicio does not do

Agreed to the printed digits: Ex 3.1 to 3.5, 3.8, 3.9, P3.3, P3.4, P3.11 to
P3.13, P3.15, P3.28 to P3.31; Ex 4.1, 4.2 (moments and Table 4.1), 4.3 (the
exact, normal and shifted gamma columns of Tables 4.3 and 4.4, and P4.13), 4.4,
4.5, 4.8, P4.1, P4.16; Ex 6.6's moments and its 91, Table 6.4's shifted gamma
values within a unit or so, and P6.12's premium (the last three up to the
monograph's rounding of intermediate values).

Did not agree, and why:

- **Ex 3.3's chi-square**, 2.45 against 2.43: expected counts rounded before
  summing.
- **Ex 3.6's α**, 2.44302 against 2.44285: the sample variance rounded before
  a near-cancelling subtraction. Its chi-square of 0.7753 is not reproduced
  (near 0.65 here) and its source is unclear.
- **Ex 6.6's μ**: printed 7.000, used 7.6 (above).
- **Table 6.4, P6.11 and P6.12's credit** are right for the shifted gamma
  approximation the monograph uses and several times too large for the actual
  aggregate.

What these chapters use that prospicio does not provide:

- **The normal power and Wilson–Hilferty approximations** (prospicio has the
  normal, gamma, lognormal, shifted gamma and shifted lognormal).
- **Heckman and Meyers' inversion**: prospicio uses the FFT, which serves the
  same purpose.
- **Thinning a count** to the claims above an attachment, and a contagion
  parameter on the count objects: both are a line from the mean and variance.
- **The Pólya urn** and the binomial-to-Poisson limit, which are derivations.
- **A stop-loss or limited mean on an `Approximation`**: the shifted gamma's
  comes from `Gamma.stop_loss` above.
- **Fitting counts**: there is no count fitting by maximum likelihood with a
  goodness of fit test; the method of moments is `from_mean_variance`, and
  the chi-square tests above are written out.

Where to go next: [severity and frequency](severity-and-frequency.md) for
choosing a count family by dispersion and fitting it,
[aggregate distributions](aggregate-distributions.md) for when FFT, Panjer,
simulation or moments are exact, fast or unsafe and how to choose the grid, and
[experience and exposure rating](experience-and-exposure-rating.md) for
excess counts and layers in reinsurance. The first guide,
[claim size, layers, limits and deductibles](distributions-for-actuaries-claim-size.md),
covers the monograph's chapters 2, 5 and 6.
