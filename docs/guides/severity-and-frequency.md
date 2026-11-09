# Choosing and fitting severity and frequency

<!-- site: execute -->

## Use this guide when

You have claim counts and claim sizes and need a loss model that prospicio can
price: a **frequency** (claims a year) and a **severity** (the size of each
claim). The guide says which count and size model to use in each situation, and
the calls to make. It assumes you know GLMs from primary pricing. In
reinsurance, the **tail** (the few very large claims) sets the price of the
upper layers.

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| Yearly counts, variance near the mean | Poisson | `Poisson(lam)` | One parameter is sufficient |
| Yearly counts, variance above the mean | Count chosen by dispersion | `claim_count(mean, variance / mean)` | Matches mean and variance, and selects binomial, Poisson or negative binomial |
| Counts driven by a shared year effect with a heavy right tail | Mixed Poisson | `Count.mixed_poisson(mean, cv, mixing="inverse_gaussian")` | You choose the shape of the year effect |
| Counts seen only when at least one claim occurred, or too many or too few zeros | Zero-truncated or zero-modified | `Count.zero_truncated(base)`, `Count.zero_modified(base, p0)` | The data has no zeros, or a different share of zeros |
| Counts by rating cell and exposure | Quasi-Poisson GLM with an exposure offset | `Glm("poisson", "log", dispersion="pearson")` | Rates by cell, and standard errors that include over-dispersion |
| Mean claim size by rating cell | Gamma GLM, log link | `Glm("gamma", "log")` | Multiplicative relativities. The dispersion gives the gamma shape |
| Small claims, working layers, primary | A body family | `Lognormal`, `Gamma`, `Weibull`, `Burr`, `Loglogistic` | Smooth fit to most of the claims |
| Large claims, high layers | Pareto family above a threshold | `Pareto.fit`, `PiecewisePareto.fit`, `GeneralizedPareto.fit_riegel` | Gives the tail in the units the market quotes (the alpha) |
| Both matter | A body spliced onto a tail | `Mixture.splice` | One severity. Body and tail each fitted where they have data |
| Losses capped by policy limits, or reported above different thresholds | Pareto fit with flags | `Pareto.fit(..., censored=, reporting_thresholds=)` | If you ignore either, the alpha has a bias |
| Which threshold? How sure is the alpha? | Alpha by threshold, mean excess, Hill, bootstrap | `mean_excess`, `hill`, `Pareto.fit` on resamples | The choice can change the price of a high layer by a multiple |
| A tail quantile or TVaR of the claim size | Peaks over threshold | `PotTail.fit`, `Gpd.fit` | The extreme value fit, with VaR and TVaR |
| Few observations in some rating cells | Bayesian GLM with a prior on the relativities | `BayesGlm(..., prior_sd=)` | The prior keeps thin cells near the base. The draws carry the uncertainty |
| Few large losses | An alpha from the market or a larger book, priced at several alphas | `Pareto(t, alpha)` at each alpha (section 7.3) | The data alone cannot set the alpha |

## 1. Before you fit

A fit cannot correct claims that are not on the basis of the contract you price.

1. **Define the claim** that the contract pays on: per claim or per occurrence
   (an event that aggregates several claims). Decide if expenses (ALAE) add to
   the loss or sit outside the limit. Counts and sizes must use the same
   definition.
2. **Develop** each large loss to its ultimate value: an open claim at its
   current estimate understates the tail. Develop the **counts** too (IBNR
   counts).
3. **Trend** each loss to the contract year, and move the large-loss threshold
   with the trend. A threshold of 1 today was about 1 / (1 + trend)^years in an
   old year, so a fixed threshold removes old losses that are large now.

   `LargeLossExperience(...).prepare(rate=, to=, development=)` does steps 2
   and 3: it converts to the pricing currency (`fx=`), develops to ultimate
   with a `DevelopmentPattern` (or a list of factors), and trends from the
   accident date to the contract period. `PreparedLosses.complete_above` is the
   moved threshold (the reporting threshold times the largest trend factor).
   The burning cost leaves out losses whose trended reported amount is below it.

   For the counts, a claim-count chain ladder's `DevelopmentPattern` gives each
   year's IBNR factor at the mid-point of the year:
   `pattern.at(valuation - (year + 0.5))`. See
   [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md),
   section 1.
4. **On-level** the counts to the current exposure before you calculate their
   mean and dispersion (section 3.1).
5. For working layers, keep the **empirical** distribution (the claims): its
   burning cost checks the fitted body (section 5).

## 2. The running example

Money is in millions of a synthetic currency. Layers are priced **per claim**
unless the text says otherwise. `truth` is a splice (section 8): a lognormal
body below 1 (98 percent of the claims) and a Pareto tail above 1 with alpha
1.8 (2 percent). `losses` is what a cedant's claims file holds, 8,000 claims.
Each fit can be compared with the truth (a real analysis has none).

```python
import math
import random

from prospicio.distributions import (
    Count, Gamma, GeneralizedPareto, Lognormal, Mixture, NegativeBinomial,
    Pareto, PiecewisePareto, Poisson, Sampled, Truncated, claim_count,
    fit_severity, goodness_of_fit,
)
from prospicio.models import BayesGlm, Glm, Terms
from prospicio.risk import Gpd, PotTail, hill, mean_excess

INF = float("inf")
truth = Mixture.splice([(0.98, Lognormal(-2.0, 1.2)), (0.02, Pareto(1.0, 1.8))], [0.0, 1.0, INF])
losses = truth.sample(8000, seed=7)
large = [x for x in losses if x > 1.0]
print(f"{len(losses)} claims, {len(large)} above 1, mean {sum(losses) / len(losses):.3f}, largest {max(losses):.1f}")
```

```text
8000 claims, 156 above 1, mean 0.234, largest 12.6
```

## 3. Claim counts

### 3.1 Poisson, negative binomial, binomial: choose by dispersion

The dispersion (variance over mean of the yearly counts) sets the family, and
`claim_count` applies the rule:

| Dispersion (variance / mean) | Family | When |
|---|---|---|
| Below 1 | Binomial | A fixed number of risks that each claim at most once. Rare |
| About 1 | Poisson | Independent claims, no shared shocks |
| Above 1 | Negative binomial | The usual case: years differ for reasons that the whole book shares |

A binomial needs a whole number of trials. Below 1, `claim_count` keeps the
mean and raises the dispersion to the nearest value that a whole number of
trials permits: read the returned object.

```python
yearly = NegativeBinomial.from_mean_variance(6.4, 12.0).sample(25, seed=3)
mean = sum(yearly) / len(yearly)
var = sum((k - mean) ** 2 for k in yearly) / (len(yearly) - 1)
count = claim_count(mean, var / mean)
print(f"{len(yearly)} years: mean {mean:.2f}, variance {var:.2f}, dispersion {var / mean:.2f}"
      f" (the model that made them has {12.0 / 6.4:.2f})")
print(f"{type(count).__name__}: r {count.r:.2f}, beta {count.beta:.3f}")
print(f"P(15 or more claims): Poisson {1 - Poisson(mean).cdf(14):.5f}, negative binomial {1 - count.cdf(14):.5f}")
```

```text
25 years: mean 5.20, variance 7.42, dispersion 1.43 (the model that made them has 1.88)
NegativeBinomial: r 12.20, beta 0.426
P(15 or more claims): Poisson 0.00034, negative binomial 0.00323
```

**Why it matters.** The two counts have the same mean, so a per-claim layer has
the same expected loss. But a year with 15 or more claims is about ten times
more likely under the negative binomial. Every term that is not linear in the
yearly claim count changes with the count model: aggregate limits and
deductibles, reinstatements, profit commissions, capital and risk load. A
Poisson fitted to over-dispersed counts understates the bad years and
misprices those terms.

**Pitfalls.**

- With 25 years the dispersion has a wide error (1.43 against 1.88 above). With
  few years, take it from a larger, similar book (a market benchmark, or a GLM
  across many cells, section 3.4).
- Count the **same claims** as the severity fit. Counts of large losses and of
  all claims have different means and dispersions.
- Put the history **on level** first (exposure growth and frequency trend).
  Otherwise the trend adds to the dispersion.

### 3.2 Mixed Poisson: when the year effect has a shape

A mixed Poisson count is a Poisson whose mean is multiplied each year by a
random `theta` with mean 1 and coefficient of variation `cv`. Gamma mixing
gives the negative binomial. `Count.mixed_poisson` sets the shape of the year
effect, and `shift` fixes part of it (the share of the mean that does not
change).

```python
cv = math.sqrt((var - mean) / mean**2)
for name, c in (
    ("gamma", Count.mixed_poisson(mean, cv)),
    ("inverse Gaussian", Count.mixed_poisson(mean, cv, mixing="inverse_gaussian")),
):
    print(f"{name:>16}: mean {c.mean():.2f}, variance {c.variance():.2f}, P(15 or more) {1 - c.cdf(14):.5f}")
```

```text
           gamma: mean 5.20, variance 7.42, P(15 or more) 0.00323
inverse Gaussian: mean 5.20, variance 7.42, P(15 or more) 0.00368
```

The gamma row equals the negative binomial of section 3.1. The inverse
Gaussian has a heavier right tail: choose it when a few bad years are worse
than a negative binomial permits. With only a mean and a variance, keep the
negative binomial (25 years cannot show the difference).

### 3.3 Zero-truncated and zero-modified counts

A claims file that lists only the policies with a claim has no zeros. Model the
count **given at least one** with `Count.zero_truncated`. If the share of zeros
differs from the base count's (more policies without a claim than a Poisson
permits), `Count.zero_modified` sets the probability of zero.

```python
zt = Count.zero_truncated(Poisson(0.4))
zm = Count.zero_modified(Poisson(0.4), 0.9)
print(f"zero-truncated Poisson(0.4): P(0) {zt.pmf(0):.3f}, P(1) {zt.pmf(1):.3f}, mean {zt.mean():.3f}")
print(f"zero-modified, P(0) set to 0.9: mean {zm.mean():.3f}")
```

```text
zero-truncated Poisson(0.4): P(0) 0.000, P(1) 0.813, mean 1.213
zero-modified, P(0) set to 0.9: mean 0.121
```

Choose the base Poisson mean so that the truncated mean agrees with the data.

### 3.4 Counts by rating cell: a GLM with an exposure offset

For counts by segment, year and exposure, fit a Poisson GLM with the log of
exposure as an **offset**, as in primary pricing. Counts are over-dispersed, so
set `dispersion="pearson"` (quasi-Poisson): the coefficients are the Poisson
coefficients, and the estimated dispersion scales the standard errors. The
negative binomial GLM is an alternative (here with a fixed `theta` of 10).

```python
rng = random.Random(4)
seg, expo, claims = [], [], []
for s, rate in (("A", 0.05), ("B", 0.08), ("C", 0.14)):
    for year in range(10):
        e = 100.0 + 10 * year
        mu = rate * e
        n = NegativeBinomial.from_mean_variance(mu, 2.0 * mu).sample(1, seed=rng.randrange(10**6))[0]
        seg.append(s), expo.append(e), claims.append(float(n))
data = {"seg": seg, "expo": expo}
coding = Terms().intercept().factor("seg").fit(data)
design = coding.design(data, offset=[math.log(e) for e in expo])

poisson = Glm("poisson", "log").fit(design, claims)
quasi = Glm("poisson", "log", dispersion="pearson").fit(design, claims)
negbin = Glm("negative_binomial", "log", theta=10.0).fit(design, claims)
print(f"dispersion: Poisson {poisson.dispersion:.2f}, quasi-Poisson {quasi.dispersion:.2f}")
print(f"quasi-Poisson standard errors over Poisson: {quasi.std_errors[0] / poisson.std_errors[0]:.2f}")
```

```text
dispersion: Poisson 1.00, quasi-Poisson 2.53
quasi-Poisson standard errors over Poisson: 1.59
```

```python
import pandas as pd
from great_tables import GT

glm_se = pd.DataFrame({
    "term": ["base rate", "B", "C"],
    "estimate": [math.exp(c) for c in quasi.coefficients],
    "Poisson": poisson.std_errors,
    "quasi-Poisson": quasi.std_errors,
    "neg. binomial": negbin.std_errors,
})
(GT(glm_se, rowname_col="term")
 .tab_header("Base rate and relativities", "Standard errors are on the log scale")
 .tab_spanner("standard error", ["Poisson", "quasi-Poisson", "neg. binomial"])
 .fmt_number(["estimate", "Poisson", "quasi-Poisson", "neg. binomial"], decimals=3))
```

```text
           estimate  Poisson  quasi-Poisson  neg. binomial
base rate     0.070    0.099          0.157          0.141
B             1.343    0.131          0.208          0.193
C             1.853    0.123          0.195          0.188
```

The Poisson standard errors are too small by the square root of the dispersion
(the ratio above), so a Poisson fit finds differences that the data does not
support. Pass a segment's predicted mean and the fitted dispersion to
`claim_count` for next year's count.

```python
new = coding.design({"seg": ["C"], "expo": [120.0]}, offset=[math.log(120.0)])
mu = quasi.predict(new)[0]
c = claim_count(mu, quasi.dispersion)
print(f"segment C on 120 exposure: mean {mu:.2f}, {type(c).__name__} with r {c.r:.2f}")
```

```text
segment C on 120 exposure: mean 15.64, NegativeBinomial with r 10.24
```

For a frequency trend, add a numeric term (`Terms().numeric("year")`): the GLM
then gives the annual trend and its standard error.

## 4. Claim sizes: which family

| Family | Tail | Use it for | Notes |
|---|---|---|---|
| `Lognormal(meanlog, sdlog)` | Medium: its local alpha continues to increase | The body of most lines | Closed-form fit from the mean and standard deviation of the logs |
| `Gamma(shape, scale)` | Light (exponential) | Average size by segment. Homogeneous or capped claims | The GLM family for severity. `Gamma.from_mean_cv` |
| `Weibull` | Light for shape above 1. Heavier than exponential (but lighter than a power tail) for shape below 1 | Sizes with a natural scale | |
| `Burr`, `Loglogistic` | Heavy (a power tail) with a flexible body | One family for body and moderate tail. Liability | `fit_severity("burr", ...)` |
| `InverseGamma` | Heavy (a power tail) | Skewed bodies with a heavy tail | |
| `InverseGaussian` | Light (exponential) | Skewed bodies | Lighter in the tail than the lognormal |
| `Pareto(t, alpha)` and its relatives | Heavy (a power tail) | Large losses above a threshold | Fitted with truncation and censoring, section 7 |
| `Truncated(severity, lower, upper)` | As the base | A claim given that it is in a window | Section 6 |
| `Mixture`, `Mixture.splice` | What you give it | Body and tail together | Section 8 |

Each family has `cdf`, `quantile`, `mean`, `sample`, `lev` (the limited
expected value `E[min(X, limit)]`) and `layer`. The Pareto family and the `Gpd`
of section 7 have their own fits. `fit_severity` fits the lognormal, gamma,
Weibull and Burr, with reporting thresholds and censoring (section 6). Give the
other families their parameters directly.

A lognormal has a closed-form fit (the mean and standard deviation of the
logs). Here it is fitted to all claims:

```python
logs = [math.log(x) for x in losses]
meanlog = sum(logs) / len(logs)
sdlog = math.sqrt(sum((v - meanlog) ** 2 for v in logs) / len(logs))
one = Lognormal(meanlog, sdlog)


def burning_cost(attachment, limit):
    """Average payment per claim to the layer, from the claims themselves."""
    return sum(min(max(x - attachment, 0.0), limit) for x in losses) / len(losses)


print(f"lognormal fitted to all claims: meanlog {meanlog:.2f}, sdlog {sdlog:.2f}")
```

```text
lognormal fitted to all claims: meanlog -2.08, sdlog 1.14
```

```python
cases = ((1, 1), (4, 1), (5, 5), (10, 10))
layer_tab = pd.DataFrame({
    "layer": [f"{limit} xs {att}" for limit, att in cases],
    "observed": [burning_cost(att, limit) for limit, att in cases],
    "lognormal": [one.layer(limit, att) for limit, att in cases],
    "truth": [truth.layer(limit, att) for limit, att in cases],
})
(GT(layer_tab, rowname_col="layer")
 .tab_header("Expected payment per claim to each layer", "Millions")
 .fmt_number(["observed", "lognormal", "truth"], decimals=5))
```

```text
          observed  lognormal    truth
1 xs 1     0.01121    0.01629  0.01064
4 xs 1     0.01778    0.02332  0.01810
5 xs 5     0.00255    0.00104  0.00294
10 xs 10   0.00075    0.00018  0.00169
```

The lognormal prices the layers badly: too high near 1 and too low in the high
layers, because one pair of parameters cannot fit body and tail together. (The
observed 10 xs 10 average comes from very few claims.) This is the main point
of the guide: **a family fitted for the body is not reliable in the tail**.

### Severity by segment: a gamma GLM

For the average size by rating cell, fit a gamma GLM with a log link to losses
**capped at the large-loss threshold**, and use the section 7 tail model for
the part above. Uncapped, a few large losses move the relativities a lot. The
fitted dispersion is `1 / shape`, so a segment's mean and the dispersion give a
`Gamma` for its claim size.

```python
rng = random.Random(8)
sseg = ["A", "B", "C"] * 100
mean_size = {"A": 5.0, "B": 8.0, "C": 12.0}
sizes = [Gamma(2.0, mean_size[s] / 2).sample(1, seed=rng.randrange(10**6))[0] for s in sseg]
sdata = {"seg": sseg}
scoding = Terms().intercept().factor("seg").fit(sdata)
gamma_fit = Glm("gamma", "log").fit(scoding.design(sdata), sizes)
phi = gamma_fit.dispersion
b = gamma_fit.coefficients
means = [math.exp(b[0]), math.exp(b[0] + b[1]), math.exp(b[0] + b[2])]
print(f"dispersion {phi:.2f}, so the shape is {1 / phi:.2f} (true 2)")
segment_c = Gamma(1 / phi, means[2] * phi)
print(f"segment C: mean {segment_c.mean():.2f}, 10 xs 10 per claim {segment_c.layer(10, 10):.4f}")
```

```text
dispersion 0.52, so the shape is 1.93 (true 2)
segment C: mean 12.50, 10 xs 10 per claim 3.2147
```

```python
gamma_tab = pd.DataFrame({
    "segment": ["A", "B", "C"],
    "fitted": means,
    "true": [5.0, 8.0, 12.0],
    "se": gamma_fit.std_errors,
})
(GT(gamma_tab, rowname_col="segment")
 .tab_header("Mean claim size by segment", "Gamma GLM; the standard errors are of the log coefficients")
 .fmt_number(["fitted", "true"], decimals=2)
 .fmt_number("se", decimals=3)
 .cols_label(se="log-coefficient se"))
```

```text
   fitted   true  log-coefficient se
A    5.23   5.00               0.072
B    7.98   8.00               0.102
C   12.50  12.00               0.102
```

With 100 claims per segment, a fitted mean several percent from the truth is
noise. A gamma has a light tail: use it for the **mean** by segment, not to
price layers far above the average claim.

## 5. The body and the tail matter for different layers

| Layer | Depends mostly on | Fit it with |
|---|---|---|
| Primary and working layers (at or below the typical claim) | The body and the frequency | A body family, or the empirical distribution |
| Layers that attach in the transition (a few times the mean) | Both | A splice (section 8) |
| High layers, above the largest claims seen | The alpha of the tail and its uncertainty | The Pareto family (section 7) |

- A working layer's burning cost (the average of what the layer paid on past
  claims) is a good price, and a body model checks it.
- A high layer has almost no observed payments. Its price **extrapolates** from
  the claims above the threshold, so the tail family and threshold set it.

## 6. Truncated and censored data

Large-loss data is rarely complete:

- **Reporting thresholds**: a source reports only losses above some amount, and
  sources differ. Ignoring this makes the tail look heavier (the alpha is too
  low): the source with the higher threshold shows no losses between the two
  thresholds, so large losses look more frequent.
- **Censoring**: a policy limit capped the loss, so the file shows the limit.
  Ignoring this makes the tail look thinner (the alpha is too high).

`Pareto.fit`, `PiecewisePareto.fit` and `GeneralizedPareto.fit_riegel` take
both, `censored` (a flag per loss) and `reporting_thresholds` (a threshold per
loss), and fit by maximum likelihood. `Pareto.fit` also takes `weights` (for
example to give old years less weight) and `truncation`, an upper end above
which no loss can occur, such as a sum insured.

In the example, each large loss has a policy limit of 1.5, 2 or 3:

```python
rng = random.Random(1)
policy_limit = [rng.choice((1.5, 2.0, 3.0)) for _ in large]
capped = [min(x, lim) for x, lim in zip(large, policy_limit)]
flags = [x >= lim for x, lim in zip(large, policy_limit)]
print(f"{sum(flags)} of {len(large)} large losses reach their policy limit")
print(f"alpha, all losses known:   {Pareto.fit(large, 1.0).alpha:.3f}")
print(f"alpha, capped, ignored:    {Pareto.fit(capped, 1.0).alpha:.3f}")
print(f"alpha, capped, censored:   {Pareto.fit(capped, 1.0, censored=flags).alpha:.3f}")

rng = random.Random(8)
source_threshold = [1.0 if rng.random() < 0.5 else 2.0 for _ in large]
seen = [(x, u) for x, u in zip(large, source_threshold) if x >= u]
kept, thresholds = [x for x, _ in seen], [u for _, u in seen]
print(f"{len(kept)} of {len(large)} large losses are reported (half the sources start at 2)")
print(f"alpha, thresholds ignored: {Pareto.fit(kept, 1.0).alpha:.3f}")
print(f"alpha, thresholds used:    {Pareto.fit(kept, 1.0, reporting_thresholds=thresholds).alpha:.3f}")
```

```text
43 of 156 large losses reach their policy limit
alpha, all losses known:   1.765
alpha, capped, ignored:    2.415
alpha, capped, censored:   1.749
100 of 156 large losses are reported (half the sources start at 2)
alpha, thresholds ignored: 1.418
alpha, thresholds used:    1.590
```

Both flags move the alpha as theory says. Ignoring the caps makes 43 large
losses look smaller, so the alpha is far too high: a thin tail and cheap high
layers. The `censored` flags bring it back. The thresholds case is noisier
because it has fewer losses (section 7.3).

### Fitting the body: `fit_severity`

`Truncated(severity, lower, upper)` models a claim **given** that it is in a
window: use it for a body fitted to truncated data.

`fit_severity(family, losses, reporting_thresholds=, censored=)` fits a
lognormal, gamma, Weibull or Burr by maximum likelihood. It conditions each
loss on exceeding its own reporting threshold, counts a capped loss as at least
its limit, and returns the distribution, its parameters with standard errors,
and the AIC and BIC.

`goodness_of_fit(distribution, losses, reporting_thresholds=)` compares a fit
with the product-limit estimate of the same losses: the largest gap between the
two survival functions, and the Q-Q pairs.

A sample cut at 1, because the claims above 1 go to the tail fit, is not claims
capped at 1. Do not treat the cut as censoring: that puts the body's own mass
above 1, where the tail's mass is. `fit_lognormal_between` below fits, by
maximum likelihood, a lognormal seen only between `lower` and `upper`. Its
likelihood subtracts the log of the window's probability:

```python
def normal_between(a, b):
    """P(a < Z <= b) for a standard normal Z, accurate in either tail."""
    if a > 0:
        return 0.5 * (math.erfc(a / math.sqrt(2.0)) - math.erfc(b / math.sqrt(2.0)))
    return 0.5 * (math.erfc(-b / math.sqrt(2.0)) - math.erfc(-a / math.sqrt(2.0)))


def fit_lognormal_between(xs, lower=0.0, upper=INF):
    """Maximum likelihood meanlog, sdlog of a lognormal seen only in (lower, upper] (grid search)."""
    n = len(xs)
    s1 = sum(math.log(x) for x in xs)
    s2 = sum(math.log(x) ** 2 for x in xs)

    def nll(mu, sd):
        a = (math.log(lower) - mu) / sd if lower > 0 else -INF
        b = (math.log(upper) - mu) / sd if upper < INF else INF
        p = max(normal_between(a, b), 1e-300)
        return n * math.log(sd) + (s2 - 2 * mu * s1 + n * mu**2) / (2 * sd**2) + n * math.log(p)

    mu, sd, half_mu, half_sd = s1 / n, math.sqrt(s2 / n - (s1 / n) ** 2), 2.0, 0.8
    for _ in range(6):
        grid = [(mu + half_mu * i / 10, sd + half_sd * j / 10) for i in range(-10, 11) for j in range(-10, 11)]
        grid = [(m, s) for m, s in grid if s > 0.05]
        mu, sd = min(grid, key=lambda p: nll(*p))
        half_mu, half_sd = half_mu / 3, half_sd / 3
    return mu, sd


reported = [x for x in losses if x > 0.1]
thresholds = [0.1] * len(reported)
naive_mu = sum(math.log(x) for x in reported) / len(reported)
fit = fit_severity("lognormal", reported, reporting_thresholds=thresholds)
mu_hat, sd_hat = fit.parameters["meanlog"], fit.parameters["sdlog"]
se = fit.standard_errors
window = [x for x in reported if x <= 1.0]
mu_win, sd_win = fit_lognormal_between(window, lower=0.1, upper=1.0)
print(f"{len(reported)} of {len(losses)} claims are above the reporting threshold of 0.1")
print(f"ignoring the threshold: meanlog {naive_mu:.2f}")
print(f"standard errors of the fit above 0.1: meanlog {se['meanlog']:.2f}, sdlog {se['sdlog']:.2f}")
gof = goodness_of_fit(fit.distribution, reported, reporting_thresholds=thresholds)
print(f"largest gap to the product-limit survival: {gof['max_distance']:.3f}")
print(f"largest three losses: {', '.join(f'{x:.1f}' for x in gof['x'][-3:])}; "
      f"the fit's quantiles there: {', '.join(f'{q:.1f}' for q in gof['model_quantile'][-3:])}")
body_fits = pd.DataFrame({
    "fit": ["above 0.1, truncated at 0.1", "0.1 to 1, truncated at both", "truth"],
    "meanlog": [mu_hat, mu_win, -2.0],
    "sdlog": [sd_hat, sd_win, 1.2],
})
(GT(body_fits, rowname_col="fit")
 .tab_header("Lognormal fits of the body")
 .fmt_number(["meanlog", "sdlog"], decimals=2))
```

```text
4681 of 8000 claims are above the reporting threshold of 0.1
ignoring the threshold: meanlog -1.32
standard errors of the fit above 0.1: meanlog 0.05, sdlog 0.03
largest gap to the product-limit survival: 0.017
largest three losses: 11.0, 11.7, 12.6; the fit's quantiles there: 5.0, 5.8, 7.8
                             meanlog  sdlog
above 0.1, truncated at 0.1    -1.93   1.04
0.1 to 1, truncated at both    -2.21   1.33
truth                          -2.00   1.20
```

With the threshold in the fit, the meanlog moves from -1.32 to -1.93, near the
true -2.00. The sdlog (1.04) stays below 1.20, because the claims above 1
follow the Pareto, not the lognormal (section 4).

The survival functions differ by under 2 percent, because most claims are in the
body. But the largest losses are far above the fit's quantiles: a small
distance does not show that the tail is right.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0.1, 20], [0.1, 20], color="0.75", label="perfect fit")
ax.plot(gof["x"], gof["model_quantile"], label="lognormal fit")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("observed loss (millions)")
ax.set_ylabel("quantile of the fit (millions)")
ax.set_title("Q-Q plot of the body fit: the largest observed losses exceed the fit")
ax.legend()
```

The window fit cuts the sample at 1 and puts both cuts in the likelihood, which
removes the mix of body and tail. Its estimate is noisier, because a window
from 0.1 to 1 shows only part of the body's shape. Section 8 uses the same
function for the body below the tail threshold.

## 7. The tail

### 7.1 When to use the single-parameter Pareto

Above a threshold `t`, a Pareto has survival `(t / x)^alpha`. One parameter,
the alpha, gives the whole tail, in closed form. It is the default tail for
excess of loss because it scales (an alpha fitted on old losses stays correct
after inflation) and layers price from each other (see
`docs/design/pareto.md`).

Use it when the alpha is stable as the threshold changes (section 7.2) and
enough losses lie above the threshold. The alpha's standard error is about
`alpha / sqrt(n)` for `n` losses. `Pareto.fit_with_standard_error` returns it
from the observed information, which also covers truncation and censoring.

```python
tail, a_se = Pareto.fit_with_standard_error(large, 1.0)
a_hat = tail.alpha
print(f"alpha {a_hat:.3f} from {len(large)} losses above 1 (the truth is 1.8)")
print(f"standard error {a_se:.3f}; alpha / sqrt(n) = {a_hat / math.sqrt(len(large)):.3f}")
print(f"expected loss per large loss, 5 xs 5: {tail.layer(5, 5):.4f}; 10 xs 10: {tail.layer(10, 10):.4f}")
```

```text
alpha 1.765 from 156 losses above 1 (the truth is 1.8)
standard error 0.141; alpha / sqrt(n) = 0.141
expected loss per large loss, 5 xs 5: 0.1572; 10 xs 10: 0.0925
```

```python
n_losses = [10, 30, 100, 156]
range_tab = pd.DataFrame({
    "losses": [str(n) for n in n_losses],
    "range": [1.645 / math.sqrt(n) for n in n_losses],
})
(GT(range_tab, rowname_col="losses")
 .tab_header("90% range of the alpha", "Plus or minus this share of the alpha, by number of losses")
 .fmt_percent("range", decimals=0))
```

```text
     range
10     52%
30     30%
100    16%
156    13%
```

Below about 30 losses above the threshold the range is wide: do not take the
alpha from the cedant's data alone. Price at several alphas and anchor on an
alpha from the market or a larger book (sections 7.3 and 9). Do not use the
single Pareto when the local alpha clearly changes with size (section 7.4).

### 7.2 Choosing the threshold

A low threshold adds **bias** (body claims bend the fit). A high one adds
**variance** (few losses remain). Three tools help:

- **Alpha by threshold**: where the Pareto applies, the alpha is flat (the
  Pareto is threshold invariant), apart from noise.
- **Mean excess** `e(u) = E[X - u | X > u]`: for a Pareto it rises linearly in
  `u` with slope `1 / (alpha - 1)`. Find where it becomes a straight line.
- **Hill estimates** of `xi = 1 / alpha` from the `k` largest losses: find a
  range of `k` where the estimate is flat.

```python
us = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
counts = [sum(1 for x in losses if x >= u) for u in us]
alphas = [Pareto.fit([x for x in losses if x >= u], u).alpha for u in us]
errs = [a / math.sqrt(n) for a, n in zip(alphas, counts)]
by_threshold = pd.DataFrame({"threshold": us, "losses": counts, "alpha": alphas, "std err": errs})
(GT(by_threshold, rowname_col="threshold")
 .tab_header("Pareto alpha by threshold", "Synthetic losses, true alpha 1.8 above 1")
 .fmt_number(["alpha", "std err"], decimals=2)
 .fmt_integer("losses"))
```

```text
      losses  alpha  std err
0.25   2,217   1.50     0.03
0.5      919   2.16     0.07
1.0      156   1.76     0.14
2.0       46   2.09     0.31
3.0       21   2.24     0.49
5.0        5   1.47     0.66
```

```python
mex = mean_excess(losses, [0.5, 1, 2, 3, 5])
mex_tab = pd.DataFrame({
    "threshold": [f"{u:g}" for u, _, _ in mex],
    "mean excess": [e for _, e, _ in mex],
    "losses": [n for _, _, n in mex],
})
(GT(mex_tab, rowname_col="threshold")
 .tab_header("Mean excess over a threshold", "Millions, with the number of losses above the threshold")
 .fmt_number("mean excess", decimals=2)
 .fmt_integer("losses"))
```

```text
     mean excess  losses
0.5         0.43     919
1           1.08     156
2           1.72      46
3           2.31      21
5           5.28       5
```

```python
ks = [20, 40, 80, 156, 300]
hill_tab = pd.DataFrame({"k": [str(k) for k in ks], "alpha": [1 / h for h in hill(losses, ks)]})
(GT(hill_tab, rowname_col="k")
 .tab_header("Hill alpha", "From the k largest losses")
 .fmt_number("alpha", decimals=2))
```

```text
     alpha
20    2.30
40    2.05
80    2.02
156   1.76
300   2.36
```

```python
cuts = [1.5, 2.0, 3.0, 5.0, 8.0]
n_above = [sum(1 for v in large if v > x) for x in cuts]
share_tab = pd.DataFrame({
    "size": [f"{x:g}" for x in cuts],
    "losses above": n_above,
    "observed share": [n / len(large) for n in n_above],
    "Pareto": [tail.survival(x) for x in cuts],
})
(GT(share_tab, rowname_col="size")
 .tab_header("Share of the large losses above each size", "Observed against the fitted Pareto")
 .fmt_integer("losses above")
 .fmt_number(["observed share", "Pareto"], decimals=3))
```

```text
     losses above  observed share  Pareto
1.5            84           0.538   0.489
2              46           0.295   0.294
3              21           0.135   0.144
5               5           0.032   0.058
8               4           0.026   0.025
```

```python
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
axes[0].errorbar(us, alphas, yerr=errs, marker="o", capsize=3, label="fitted alpha")
axes[0].hlines(1.8, 1.0, 5.0, color="0.75", label="true alpha above 1")
axes[0].set_xlabel("threshold (millions)")
axes[0].set_ylabel("alpha")
axes[0].set_title("Alpha by threshold")
axes[0].legend()
axes[1].plot([u for u, e, n in mex], [e for u, e, n in mex], marker="o")
axes[1].set_xlabel("threshold (millions)")
axes[1].set_ylabel("mean excess (millions)")
axes[1].set_title("Mean excess")
axes[2].axhline(1.8, color="0.75", label="true alpha above 1")
axes[2].plot(ks, [1 / h for h in hill(losses, ks)], marker="o", label="Hill alpha")
axes[2].set_xlabel("number of largest losses k")
axes[2].set_ylabel("Hill alpha")
axes[2].set_title("Hill estimate")
axes[2].legend()
fig.tight_layout()
```

Here the right threshold is 1, because the truth joins there. With real data
it is a judgment from several diagnostics together:

- **Alpha by threshold.** At 0.5 the body still bends the alpha, and at 0.25
  the small standard error is false precision. From 1 up, the alphas differ by
  about one standard error or less.
- **Mean excess.** It rises steeply up to 1, where the tail replaces the body.
  The last points come from 21 and 5 losses.
- **Hill.** The alpha is flat up to 156 losses and jumps at 300, where body
  claims enter. The high values at small `k` are noise (above 1 this example is
  exactly Pareto). With real data the estimate is also biased where the tail is
  not exactly Pareto: look for a flat range, not for its value.
- **Fit check.** The size table compares the observed share above each size with
  the fitted Pareto's. They agree from 2 up; the 5 losses above 5 are too few
  to read.

The rule: **take the lowest threshold above which the alpha is stable, because
it keeps the most data.** The cedant's reporting threshold and the layers you
price also matter. Record the alpha at several thresholds.

### 7.3 Parameter uncertainty: how sure is the alpha?

The standard error `alpha / sqrt(n)` is a start. For a layer price, 200 times:
resample the losses with replacement, fit again, and price the layer. The
spread of the price is the parameter uncertainty. It is much larger than the
alpha's spread suggests, because a layer price is a steep function of the
alpha.

```python
rng = random.Random(5)
layers = ((5, 5), (10, 10), (20, 20))
boot = []
for _ in range(200):
    refit = Pareto.fit([rng.choice(large) for _ in large], 1.0)
    boot.append((refit.alpha, [refit.layer(limit, att) for limit, att in layers]))
alphas = sorted(a for a, _ in boot)
print(f"alpha {a_hat:.2f}, 90% bootstrap range {alphas[9]:.2f} to {alphas[190]:.2f} (10th smallest and largest of 200)")
```

```text
alpha 1.76, 90% bootstrap range 1.57 to 1.96 (10th smallest and largest of 200)
```

```python
true_tail = Pareto(1.0, 1.8)
rows = []
for j, (limit, att) in enumerate(layers):
    prices = sorted(p[j] for _, p in boot)
    rows.append((f"{limit} xs {att}", prices[9], tail.layer(limit, att), prices[190],
                 true_tail.layer(limit, att), prices[190] / prices[9]))
boot_tab = pd.DataFrame(rows, columns=["layer", "5%", "fitted", "95%", "truth", "95% / 5%"])
(GT(boot_tab, rowname_col="layer")
 .tab_header("Bootstrap range of the layer price", "Expected loss per large loss, 200 resamples")
 .fmt_number(["5%", "fitted", "95%", "truth"], decimals=4)
 .fmt_number("95% / 5%", decimals=1))
```

```text
              5%  fitted     95%   truth  95% / 5%
5 xs 5    0.1072  0.1572  0.2292  0.1468       2.1
10 xs 10  0.0550  0.0925  0.1545  0.0843       2.8
20 xs 20  0.0282  0.0545  0.1041  0.0484       3.7
```

The layers are per large loss. The alpha's 90 percent range is 1.57 to 1.96,
yet the price of 10 xs 10 varies by a factor of 2.8 and that of 20 xs 20 by
3.7: the higher the layer, the wider the range. A reviewer needs this range
first. Report a layer price from a fitted tail **with its range**, or load it
for the uncertainty (see the guide on risk loads).

A layer price falls as the alpha rises, so the 5 percent price is the price at
the 95 percent alpha. The code prices each resample, which also works for
quantities that are not monotone in the alpha.

The bootstrap has two limits:

- It resamples only the sizes, so the uncertainty in the number of large losses
  (section 3) is additional.
- With few losses it is optimistic: it can only reproduce the losses you have.
  Anchor on an alpha from the market or a larger book (section 9).

### 7.4 When one alpha is not enough: piecewise Pareto and the generalized Pareto

If the alpha clearly differs between ranges, use:

| Tool | Local alpha | Use it when | Cost |
|---|---|---|---|
| `PiecewisePareto.fit(losses, t=[...])` | A step function, one alpha for each piece | The market or the data gives distinct ranges (a working range and a catastrophe range) | A parameter for each piece, each fitted on fewer losses |
| `GeneralizedPareto.fit_riegel(losses, t)` | Moves smoothly from an initial alpha at `t` to a tail alpha far out | The tail looks lighter, or heavier, with size | Two parameters. The far tail is an extrapolation |
| `LogAffinePareto(t, alpha0, gamma)` | Increases linearly in the log of the size, like a lognormal | You want a lognormal-type tail in alpha units | Give the two numbers from a benchmark |

```python
piecewise = PiecewisePareto.fit(large, [1.0, 3.0])
smooth = GeneralizedPareto.fit_riegel(large, 1.0)
print(f"piecewise alphas: {[round(a, 2) for a in piecewise.alpha]} on pieces starting at {piecewise.t}")
print(f"generalized Pareto: initial alpha {1.0 / smooth.beta:.2f}, tail alpha {1 / smooth.xi:.2f}")
```

```text
piecewise alphas: [1.71, 2.24] on pieces starting at [1.0, 3.0]
generalized Pareto: initial alpha 1.45, tail alpha 2.77
```

```python
tail_cases = ((5, 5), (10, 10), (20, 20))
tail_tab = pd.DataFrame({
    "layer": [f"{limit} xs {att}" for limit, att in tail_cases],
    "Pareto": [tail.layer(limit, att) for limit, att in tail_cases],
    "piecewise": [piecewise.layer(limit, att) for limit, att in tail_cases],
    "general": [smooth.layer(limit, att) for limit, att in tail_cases],
    "truth": [true_tail.layer(limit, att) for limit, att in tail_cases],
})
(GT(tail_tab, rowname_col="layer")
 .tab_header("Expected loss per large loss under three tails", "Millions")
 .fmt_number(["Pareto", "piecewise", "general", "truth"], decimals=4))
```

```text
          Pareto  piecewise  general   truth
5 xs 5    0.1572     0.1133   0.0965  0.1468
10 xs 10  0.0925     0.0479   0.0337  0.0843
20 xs 20  0.0545     0.0203   0.0108  0.0484
```

The three tails already differ at 5 xs 5. At 20 xs 20 the single Pareto charges
about five times the generalized Pareto, and the 156 losses cannot show which
is right. The generalized Pareto bends the alpha up to its tail value, so it
prices the highest layers far below the single Pareto: right only if the
thinning is real.

The single Pareto is a special case of the other two: equal alphas in both
pieces, or, for the generalized Pareto, `xi` equal to `beta` over the
threshold. A likelihood ratio test therefore applies, and AIC compares all
three. `log_likelihood` sums the log densities of a fitted tail, each the
numerical slope of the `cdf`:

```python
def log_likelihood(dist, xs, h=1e-6):
    """Sum of the log densities, each the slope of the cdf over a small relative step."""
    return sum(math.log((dist.cdf(x * (1 + h)) - dist.cdf(x * (1 - h))) / (2 * h * x)) for x in xs)


lls = {}
for name, d in (("Pareto", tail), ("piecewise Pareto", piecewise), ("generalized Pareto", smooth)):
    lls[name] = log_likelihood(d, large)
```

```python
n_par = {"Pareto": 1, "piecewise Pareto": 2, "generalized Pareto": 2}
aic_tab = pd.DataFrame({
    "tail": list(n_par),
    "parameters": list(n_par.values()),
    "log-lik": [lls[n] for n in n_par],
    "AIC": [2 * k - 2 * lls[n] for n, k in n_par.items()],
})
(GT(aic_tab, rowname_col="tail")
 .tab_header("Three tails fitted to the large losses")
 .fmt_integer("parameters")
 .fmt_number(["log-lik", "AIC"], decimals=2))
```

```text
                    parameters  log-lik     AIC
Pareto                       1  -155.82  313.63
piecewise Pareto             2  -155.19  314.38
generalized Pareto           2  -154.14  312.29
```

```python
for name in ("piecewise Pareto", "generalized Pareto"):
    lr = 2 * (lls[name] - lls["Pareto"])
    print(f"{name} against the single Pareto: likelihood ratio {lr:.2f}, p-value {math.erfc(math.sqrt(lr / 2)):.3f}")
```

```text
piecewise Pareto against the single Pareto: likelihood ratio 1.26, p-value 0.262
generalized Pareto against the single Pareto: likelihood ratio 3.35, p-value 0.067
```

The generalized Pareto has the best AIC, by a small amount, but the likelihood
ratio test does not reject the single Pareto (p-value 0.067, and 0.262 for the
piecewise Pareto). The 156 losses cannot separate these tails, yet the tails
price 20 xs 20 very differently. (The truth here is a single Pareto; in
practice you do not know that.)

More flexibility is not more information. **Prefer the simpler tail unless a
likelihood comparison and market evidence both support the extra parameter.
Show the price under the alternatives.** For a high layer, the lower alpha of a
range is a conservative choice, not an unbiased one: it loads the price for the
uncertainty and does not estimate it.

### 7.5 Extreme value theory: Gpd and PotTail

The **generalized Pareto distribution** (`Gpd(xi, beta)`) gives the amount by
which a loss exceeds a high threshold: extreme value theory shows that the
exceedances of any heavy-tailed loss approach it. `Gpd.fit(exceedances)` fits
it to the amounts **above** the threshold. The tail index `xi` is `1 / alpha`.

`PotTail.fit(draws, level)` does the same on a `Sampled` set of losses, with
the threshold at a quantile level, and gives the `var` and `tvar` of the claim
size.

Use EVT for a **quantile or TVaR of the claim size** (for example a 1-in-1000
claim) and the Pareto family for **layers** and the market's alphas. They are
the same distribution: the shifted `Gpd` equals Riegel's generalized Pareto,
with the same likelihood.

```python
exceedances = [x - 1.0 for x in large]
g = Gpd.fit(exceedances)
print(f"Gpd on exceedances of 1: xi {g.xi:.4f} (alpha {1 / g.xi:.2f}), beta {g.beta:.4f}")
print(f"fit_riegel on the same losses: xi {smooth.xi:.4f}, beta {smooth.beta:.4f}")
```

```text
Gpd on exceedances of 1: xi 0.3611 (alpha 2.77), beta 0.6887
fit_riegel on the same losses: xi 0.3611, beta 0.6887
```

```python
q = truth.quantile(0.999)
pots = {level: PotTail.fit(Sampled(losses), level) for level in (0.98, 0.99)}
rows = [("truth", 1 / 1.8, q, q + truth.stop_loss(q) / 0.001)]
rows += [(f"PotTail at {lv}, threshold {p.threshold:.2f}", p.gpd.xi, p.var(0.999), p.tvar(0.999))
         for lv, p in pots.items()]
pot_tab = pd.DataFrame(rows, columns=["fit", "xi", "VaR 99.9%", "TVaR 99.9%"])
(GT(pot_tab, rowname_col="fit")
 .tab_header("Tail of the claim size at the 99.9% level", "Millions")
 .fmt_number(["xi", "VaR 99.9%", "TVaR 99.9%"], decimals=2))
```

```text
                                   xi  VaR 99.9%  TVaR 99.9%
truth                            0.56       5.28       11.88
PotTail at 0.98, threshold 0.99  0.38       4.68        8.02
PotTail at 0.99, threshold 1.52  0.41       4.66        8.24
```

Use `PotTail` on complete data. When losses are censored or reported above
thresholds, use the Pareto fits.

Both VaRs are below the truth and both TVaRs far below it, because the fitted
`xi` is below the true 0.56. A TVaR at 99.9 percent extrapolates the tail
index, so a moderate error in `xi` gives a large error in the TVaR. Show it at
more than one threshold, beside the Pareto fits of this section.

## 8. Splicing a body onto a tail

`Mixture.splice(components, breaks)` makes one severity from pieces, each
conditioned on its own range of sizes, with the probabilities of the ranges as
weights. `breaks` has one more entry than the components: the lower end (0),
the joins and the upper end (infinity).

Fit a lognormal below the threshold and a Pareto above it, and join them. The
body sample is cut at the join, so fit it with that cut in the likelihood
(`fit_lognormal_between` of section 6, with `upper=1.0`). The table compares a
plain fit, which treats the cut sample as complete, with the truncated fit:

```python
body = [x for x in losses if x <= 1.0]
body_logs = [math.log(x) for x in body]
plain_mu = sum(body_logs) / len(body_logs)
plain_sd = math.sqrt(sum((v - plain_mu) ** 2 for v in body_logs) / len(body_logs))
body_mu, body_sd = fit_lognormal_between(body, upper=1.0)
share_large = len(large) / len(losses)


def splice(mu, sd):
    return Mixture.splice([(1 - share_large, Lognormal(mu, sd)), (share_large, tail)], [0.0, 1.0, INF])


plain, fitted = splice(plain_mu, plain_sd), splice(body_mu, body_sd)
print(f"body, plain fit: meanlog {plain_mu:.2f}, sdlog {plain_sd:.2f}; "
      f"truncated at 1: meanlog {body_mu:.2f}, sdlog {body_sd:.2f} (true -2.00, 1.20)")
print(f"share above 1: {share_large:.4f}; alpha {tail.alpha:.2f}")
print(f"variance of one claim under the splice: {fitted.variance()}")
```

```text
body, plain fit: meanlog -2.14, sdlog 1.08; truncated at 1: meanlog -2.02, sdlog 1.20 (true -2.00, 1.20)
share above 1: 0.0195; alpha 1.76
variance of one claim under the splice: inf
```

```python
splice_cases = ((0.4, 0.1), (1, 1), (4, 1), (5, 5), (10, 10))
splice_tab = pd.DataFrame({
    "layer": [f"{limit:g} xs {att:g}" for limit, att in splice_cases],
    "observed": [burning_cost(att, limit) for limit, att in splice_cases],
    "lognormal": [one.layer(limit, att) for limit, att in splice_cases],
    "plain": [plain.layer(limit, att) for limit, att in splice_cases],
    "truncated": [fitted.layer(limit, att) for limit, att in splice_cases],
    "truth": [truth.layer(limit, att) for limit, att in splice_cases],
})
(GT(splice_tab, rowname_col="layer")
 .tab_header("Expected payment per claim to each layer", "Millions; the splice with a plain and a truncated body fit")
 .tab_spanner("splice", ["plain", "truncated"])
 .fmt_number(["observed", "lognormal", "plain", "truncated", "truth"], decimals=5))
```

```text
            observed  lognormal    plain  truncated    truth
0.4 xs 0.1   0.10556    0.10283  0.09213    0.10568  0.10715
1 xs 1       0.01121    0.01629  0.01049    0.01049  0.01064
4 xs 1       0.01778    0.02332  0.01805    0.01805  0.01810
5 xs 5       0.00255    0.00104  0.00307    0.00307  0.00294
10 xs 10     0.00075    0.00018  0.00180    0.00180  0.00169
```

The truncated fit recovers the body (-2.02 and 1.20 against -2.00 and 1.20) and
prices the working layer 0.4 xs 0.1 close to the truth. The plain fit's sdlog
(1.08) is too small, because the cut removed the upper end of the body, and it
prices that layer too low.

From 1 xs 1 up the two splices agree, because those layers depend on the tail.
For 10 xs 10 the observed average comes from very few claims, and the splice is
the better guide, because it uses all the large losses. In the working layer
the observed burning cost is as good as the splice (section 5) and checks it.

Notes:

- To model only the large losses, use the tail alone, `Pareto.fit(large, 1.0)`,
  with a count of large losses (section 3).
- Any tail of section 7 can replace the Pareto: a `PiecewisePareto`, a
  `GeneralizedPareto`, or a `LogAffinePareto` with parameters from a benchmark.
- The spliced `Mixture` has the same `layer`, `lev`, `quantile` and `sample` as
  the other severities, so it goes into the aggregate and tower functions of the
  related guides.
- Its variance is infinite (the printed `inf`), because the tail alpha is below
  2. Use layer moments (`layer_variance`) instead.

## 9. When there is very little data

With few observations, segment relativities and the tail alpha become
unreliable. `BayesGlm(family, link)` fits the same GLM with a normal prior of
mean 0 on each coefficient. Its posterior draws (`coefficient_draws`) carry the
uncertainty into every value you calculate from the fit, including a segment's
mean, and `predict_distribution` gives predictive draws of new claims.

The prior on the relativities has standard deviation `prior_sd` on the log
scale, 2.5 by default (almost no effect). A smaller `prior_sd` states that
relativities are modest and pulls thin cells towards the base. Before you read
the posterior, check that the chains agree (`rhat` near 1) and that there are
no divergences.

Here are 10 claims per segment, with true means 5, 8 and 12 again:

```python
rng = random.Random(9)
tseg = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
tsizes = [Gamma(2.0, mean_size[s] / 2).sample(1, seed=rng.randrange(10**6))[0] for s in tseg]
tdata = {"seg": tseg}
tcoding = Terms().intercept().factor("seg").fit(tdata)
tdesign = tcoding.design(tdata)
mle = Glm("gamma", "log").fit(tdesign, tsizes)
cov = mle.covariance
se_c = math.sqrt(cov[0][0] + cov[2][2] + 2 * cov[0][2])
mle_c = mle.coefficients[0] + mle.coefficients[2]
print("observed mean size by segment:",
      [round(sum(y for y, s in zip(tsizes, tseg) if s == g) / 10, 2) for g in "ABC"])
print(f"{'fit':>13} {'mean A':>7} {'relativity C':>13} {'mean C, 90% range':>18}")
b = mle.coefficients
print(f"{'GLM':>13} {math.exp(b[0]):7.2f} {math.exp(b[2]):13.2f} "
      f"{math.exp(mle_c - 1.645 * se_c):8.2f} to {math.exp(mle_c + 1.645 * se_c):5.2f}")
print(f"{'truth':>13} {5.0:7.2f} {2.4:13.2f} {12.0:8.2f}")
glm_width = math.exp(mle_c + 1.645 * se_c) - math.exp(mle_c - 1.645 * se_c)
# A sampler's draws differ in the last digits from one platform to another, so
# the Bayesian fits are read as checks rather than printed to two decimals.
for prior_sd in (2.5, 0.5):
    bayes = BayesGlm("gamma", "log", prior_sd=prior_sd, chains=2, tune=300, draws=300, seed=1).fit(tdesign, tsizes)
    ok = all(row["rhat"] < 1.05 for row in bayes.summary()) and bayes.divergences == 0
    c_means = sorted(math.exp(d[0] + d[2]) for d in bayes.coefficient_draws)
    k = len(c_means) // 20
    pm = bayes.posterior_mean
    mean_a, rel_c = math.exp(pm[0]), math.exp(pm[2])
    narrower = c_means[-k - 1] - c_means[k] < glm_width
    print(f"prior sd {prior_sd}: chains fine {ok}")
    if prior_sd == 2.5:
        print(f"  relativity C within 10% of the GLM's {math.exp(b[2]):.2f}: {abs(rel_c / math.exp(b[2]) - 1) < 0.10}")
    else:
        print(f"  mean A between 4.5 and 6 (GLM {math.exp(b[0]):.2f}, truth 5.00): {4.5 < mean_a < 6.0}")
        print(f"  relativity C between 1.6 and 2.2 (GLM {math.exp(b[2]):.2f}, truth 2.40): {1.6 < rel_c < 2.2}")
        print(f"  90% range of mean C narrower than the GLM's: {narrower}")
```

```text
observed mean size by segment: [3.27, 8.7, 11.82]
          fit  mean A  relativity C  mean C, 90% range
          GLM    3.27          3.61     7.54 to 18.53
        truth    5.00          2.40    12.00
prior sd 2.5: chains fine True
  relativity C within 10% of the GLM's 3.61: True
prior sd 0.5: chains fine True
  mean A between 4.5 and 6 (GLM 3.27, truth 5.00): True
  relativity C between 1.6 and 2.2 (GLM 3.61, truth 2.40): True
  90% range of mean C narrower than the GLM's: True
```

Segment A's 10 claims average 3.27 against a true 5, so the GLM puts the
relativity of C at 3.61 against a true 2.40. The default prior barely moves it.
A `prior_sd` of 0.5 pulls the mean of segment A to between 4.5 and 6, the
relativity of C to between 1.6 and 2.2, and narrows the 90 percent range of
segment C's mean below the GLM's (7.54 to 18.53).

A prior is an assumption. It helps here because the true relativities are
modest. The code prints checks, not posterior values, because a sampler's draws
differ in their last digits between platforms. Take `prior_sd` from a larger
book or a market study, never from the answer you want.

With very few losses above the threshold, use the same method for the tail: a
prior alpha from the market or a larger similar book, updated by the fit. Price
the layer at several alphas (section 7.3), weighted by your belief.

## 10. Checklist and pitfalls

- Fit **frequency and severity on the same definition of a claim**: the same
  threshold, trend, development and currency basis.
- Fit the alpha at **several thresholds**, find the flat range, and show the
  layer price at the ends of the alpha's range.
- Compare a flexible tail with the single Pareto at the layers you price.
- For the tail fit, keep the losses that the burning cost keeps: those whose
  `amount` times `factor` (in `to_dict()`) is at least
  `PreparedLosses.complete_above`. Then fit their `trended` values.
  Development and conversion change a loss's size, not whether its record is
  complete. With one FX rate, the threshold in the pricing currency is
  `complete_above` times the rate.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): which pricing method to use.
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): loading a
  price for uncertainty and capital.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md):
  pricing a layer from losses and from exposure.
- [`aggregate-distributions.md`](aggregate-distributions.md): from frequency and
  severity to the annual loss.
- [`reserving-methods.md`](reserving-methods.md): developing the losses.
- [`reinsurance-price.md`](reinsurance-price.md): what a reinsurance price is.

## Appendix: terms

- **Frequency** and **severity**: the number of claims in a period, and the
  size of one claim. Together they make the **collective model**: the annual
  loss is the sum of a random number of random sizes.
- **Attritional** and **large** losses: the many small, stable claims, and the
  few large claims that cause the volatility. The **threshold** separates them.
- **Dispersion**: for a claim count, its variance divided by its mean. A
  Poisson count has dispersion 1. **Over-dispersion** is a value above 1. Counts
  of all claims in a book often have it (weather, or a good or bad year for the
  whole book). Counts of large losses above a high threshold are often near
  Poisson. Test it, do not assume it.
- In a GLM, the dispersion `phi` scales the variance function, `Var = phi *
  V(mu)`. For a Poisson, `phi` is the variance over the mean. For a gamma, it
  is `1 / shape`, the squared coefficient of variation.
- **Local Pareto alpha**: how fast the probability of a loss above a size falls
  as the size rises. With alpha 2, a size two times larger has a quarter of the
  frequency. A low alpha is a heavy tail. The market quotes tails as alphas.
- **Censored**: a loss known only to be at least some amount, because a policy
  limit capped it. **Reporting threshold**: a loss is in the data only if it is
  more than this amount (**left truncation**).
- **Layer** `limit xs attachment`: the part of a loss between the attachment
  and attachment plus limit. `5 xs 5` pays the part between 5 and 10.
