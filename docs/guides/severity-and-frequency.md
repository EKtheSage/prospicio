# Choosing and fitting severity and frequency

## Use this guide when

You have claim counts and claim sizes, from a cedant or from your own book. You
must make a loss model that prospicio can price: a **frequency** (how many
claims a year) and a **severity** (how big each claim is). This guide tells you
which count model and which size model to use in each situation, why, and the
calls to make. It is for a reader who knows GLMs and loss ratios from primary
pricing and is new to reinsurance. In reinsurance, the **tail** (the few very
large claims) sets the price of the upper layers.

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

## 1. Words you need first

- **Frequency** and **severity**: the number of claims in a period, and the
  size of one claim. Together they make the **collective model**: the annual
  loss is the sum of a random number of random sizes.
- **Attritional** and **large** losses: the many small claims that are stable
  from year to year, and the few large claims that cause the volatility. The
  **threshold** separates them.
- **Dispersion**: for a claim count, its variance divided by its mean. A
  Poisson count has dispersion 1. **Over-dispersion** is a value above 1. Counts
  of all claims in a book often have it (weather, or a good or bad year for the
  whole book). Counts of large losses above a high threshold are often near
  Poisson. Test the dispersion. Do not assume it.
- In a GLM, the dispersion `phi` scales the variance function, `Var = phi *
  V(mu)`. For a Poisson, `phi` is the variance over the mean. For a gamma, it
  is `1 / shape`, the squared coefficient of variation.
- **Local Pareto alpha**: how fast the probability of a loss above a size
  decreases as the size increases. With alpha 2, a size two times larger has a
  quarter of the frequency. A low alpha is a heavy tail. The market quotes
  tails as alphas.
- **Censored**: a loss known only to be at least some amount, because a policy
  limit capped it. **Reporting threshold**: a loss is in the data only if it is
  more than this amount (**left truncation**).
- **Layer** `limit xs attachment`: the part of a loss between the attachment
  and attachment plus limit. `5 xs 5` pays the part of a loss between 5 and 10.

Money is in millions of a synthetic currency. Layers are priced **per claim**
unless the text says per large loss.

### Before you fit

Each fit below assumes that the claims are already on the basis of the
contract that you price. This preparation is most of the work, and a fit
cannot correct it. Do these steps:

1. **Define the claim** that the contract pays on: per claim or per occurrence
   (an event that aggregates several claims). Decide if expenses (ALAE) are
   added to the loss or paid in addition to the limit. Counts and sizes must
   use the same definition.
2. **Develop** each large loss to its ultimate value. An open claim at its
   current estimate understates the tail. Develop the **counts** too: the
   recent years still have claims that are not yet reported (IBNR counts).
3. **Trend** each loss to the contract year, and move the large-loss threshold
   with the trend. A threshold of 1 today was about 1 / (1 + trend)^years in an
   old year. A fixed threshold thus removes old losses that are large now.

   `LargeLossExperience(...).prepare(rate=, to=, development=)` does steps 2
   and 3 for each loss. It converts the loss to the pricing currency (with
   `fx=` on `LargeLossExperience`). It develops the loss to ultimate with a
   `DevelopmentPattern` (or a list of factors). It trends the loss from its
   accident date to the contract period.

   `PreparedLosses.complete_above` is the moved threshold: the reporting
   threshold times the largest trend factor. The burning cost compares it with
   the reported amount of each loss, trended, before development and
   conversion. It leaves out losses below it.

   For the counts, a `DevelopmentPattern` from a claim-count chain ladder gives
   the IBNR factor of each year, read at the mid-point of the year:
   `pattern.at(valuation - (year + 0.5))`. See
   [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md),
   section 2.
4. **On-level** the counts to the current exposure before you calculate their
   mean and dispersion (section 3.1).
5. For working layers, keep the **empirical** distribution (the claims) with
   each fitted body. Its burning cost is the check on the parametric one
   (section 5).

## 2. The running example

The data is synthetic, so you can compare each fit with the truth. A real
analysis does not have the truth. Claims have a lognormal body below 1 (98
percent of the claims) and a Pareto tail above 1 with alpha 1.8 (2 percent).
`truth` is that severity. `losses` is what a cedant's claims file holds, 8,000
claims. Section 8 explains the splice that makes `truth`.

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

Start from the history of yearly counts. Calculate the mean and the variance,
and divide. The dispersion sets the family, and `claim_count` applies the rule:

| Dispersion (variance / mean) | Family | When |
|---|---|---|
| Below 1 | Binomial | A fixed number of risks that each claim at most once. Rare |
| About 1 | Poisson | Independent claims, no shared shocks |
| Above 1 | Negative binomial | The usual case: years differ for reasons that the whole book shares |

A binomial needs a whole number of trials. Thus below 1, `claim_count` keeps
the mean and moves the dispersion up to the nearest value that a whole number
of trials permits. Read the returned object, not the dispersion that you asked
for.

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

**Why it matters.** The two counts have the same mean, so the expected loss of
a per-claim layer does not change. The probability of a year with 15 or more
claims is about ten times higher under the negative binomial. The count model
changes each term that is not linear in the number of claims in a year. These
terms are aggregate limits and deductibles, reinstatements, profit commissions, and the
capital and risk load. A Poisson fitted to over-dispersed counts understates
the bad years, and so gives the wrong price for those terms.

**Pitfalls.**

- With 25 years, the dispersion has a wide error, as the two printed values
  show. With few years of history, take the dispersion from a larger, similar
  book (a market benchmark, or a GLM across many cells, section 3.4), not from
  the cedant's own years.
- Count the **same claims** that you use in the severity fit. Counts of large
  losses (above a threshold) and counts of all claims have different means and
  different dispersions.
- Put the history **on level** first. Adjust for exposure growth and for the
  frequency trend before you calculate the mean and variance. If you do not,
  the trend increases the dispersion.

### 3.2 Mixed Poisson: when the year effect has a shape

A mixed Poisson count is a Poisson whose mean is multiplied each year by a
random `theta` with mean 1 and coefficient of variation `cv`. Gamma mixing
gives the negative binomial. Use `Count.mixed_poisson` to choose the shape of
the year effect, or when part of it is fixed (`shift`, the share of the mean
that does not change).

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
Gaussian has the same mean and variance and a heavier right tail. Choose it
when a few bad years in the history are worse than a negative binomial
permits. When you know only the mean and the variance, keep the negative
binomial: 25 years cannot show the difference between the two shapes.

### 3.3 Zero-truncated and zero-modified counts

Some data never has a zero: for example, a claims file that lists only the
policies with a claim. Model the count **given at least one** with
`Count.zero_truncated`. Sometimes the share of zeros is different from what
the base count gives, for example more policies without a claim than a Poisson
permits. Then use `Count.zero_modified` to set the probability of zero.

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

Choose the base Poisson mean so that the truncated mean agrees with the data,
then wrap it.

### 3.4 Counts by rating cell: a GLM with an exposure offset

When you have counts by segment, year and exposure, fit a Poisson GLM with the
log of exposure as an **offset**, as in primary pricing. Claim counts are
over-dispersed, so set `dispersion="pearson"` (a quasi-Poisson fit). The
coefficients are the Poisson coefficients, and the estimated dispersion scales
the standard errors. The negative binomial GLM is an alternative. The code
gives it a fixed `theta` of 10.

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
print("base rate and relativities:", [round(math.exp(b), 3) for b in quasi.coefficients])
print(f"dispersion: Poisson {poisson.dispersion:.2f}, quasi-Poisson {quasi.dispersion:.2f}")
print("standard errors, Poisson:      ", [round(s, 3) for s in poisson.std_errors])
print("standard errors, quasi-Poisson:", [round(s, 3) for s in quasi.std_errors])
print("standard errors, neg. binomial:", [round(s, 3) for s in negbin.std_errors])
print(f"quasi-Poisson over Poisson: {quasi.std_errors[0] / poisson.std_errors[0]:.2f}")
```

```text
base rate and relativities: [0.07, 1.343, 1.853]
dispersion: Poisson 1.00, quasi-Poisson 2.53
standard errors, Poisson:       [0.099, 0.131, 0.123]
standard errors, quasi-Poisson: [0.157, 0.208, 0.195]
standard errors, neg. binomial: [0.141, 0.193, 0.188]
quasi-Poisson over Poisson: 1.59
```

The Poisson standard errors are too small by the square root of the
dispersion, the last printed number. Thus a Poisson fit finds significant
differences between segments that the data does not support. The fitted
dispersion also gives the count for next year: give the predicted mean of a
segment and the dispersion to `claim_count`.

```python
new = coding.design({"seg": ["C"], "expo": [120.0]}, offset=[math.log(120.0)])
mu = quasi.predict(new)[0]
c = claim_count(mu, quasi.dispersion)
print(f"segment C on 120 exposure: mean {mu:.2f}, {type(c).__name__} with r {c.r:.2f}")
```

```text
segment C on 120 exposure: mean 15.64, NegativeBinomial with r 10.24
```

To fit a frequency trend, add a numeric term (`Terms().numeric("year")`). The
same GLM then gives the annual trend and its standard error.

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

Each parametric family has `cdf`, `quantile`, `mean`, `sample`, `lev` (the
limited expected value `E[min(X, limit)]`) and `layer`. The Pareto family and
the `Gpd` of section 7 have their own fits. `fit_severity` fits the lognormal,
gamma, Weibull and Burr, with reporting thresholds and censoring (section 6).
Give the other families their parameters directly.

The lognormal fit has a closed form (the mean and standard deviation of the
logs). The code fits one to all the claims and shows how it prices layers:

```python
logs = [math.log(x) for x in losses]
meanlog = sum(logs) / len(logs)
sdlog = math.sqrt(sum((v - meanlog) ** 2 for v in logs) / len(logs))
one = Lognormal(meanlog, sdlog)


def burning_cost(attachment, limit):
    """Average payment per claim to the layer, from the claims themselves."""
    return sum(min(max(x - attachment, 0.0), limit) for x in losses) / len(losses)


print(f"lognormal fitted to all claims: meanlog {meanlog:.2f}, sdlog {sdlog:.2f}")
print(f"{'layer':>9} {'observed':>9} {'lognormal':>10} {'truth':>8}")
for limit, att in ((1, 1), (4, 1), (5, 5), (10, 10)):
    print(f"{limit:>3} xs {att:<3} {burning_cost(att, limit):9.5f} {one.layer(limit, att):10.5f} {truth.layer(limit, att):8.5f}")
```

```text
lognormal fitted to all claims: meanlog -2.08, sdlog 1.14
    layer  observed  lognormal    truth
  1 xs 1     0.01121    0.01629  0.01064
  4 xs 1     0.01778    0.02332  0.01810
  5 xs 5     0.00255    0.00104  0.00294
 10 xs 10    0.00075    0.00018  0.00169
```

The lognormal fits most of the claims, but it prices the layers badly. It
overstates the layers near 1 and understates the high layers, because one pair
of parameters cannot fit the body and the tail together. The observed average
in the highest layer is also poor, because it comes from very few claims. This
is the main point of the guide: **a family chosen and fitted for the body is
not reliable in the tail**. Sections 7 and 8 show what to do.

### Severity by segment: a gamma GLM

For the average size by rating cell, fit a gamma GLM with a log link to claim
sizes (the severity model of primary pricing). Fit it to losses **capped at the
large-loss threshold**, and use the tail model of section 7 for the part above.
If you do not cap them, a few large losses move the relativities a lot. The
fitted dispersion is `1 / shape`. Thus the mean of a segment and the dispersion
give a `Gamma` for the claim size of that segment.

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
print("fitted mean sizes:", [round(m, 2) for m in means], "(true 5, 8, 12)")
print("standard errors of the log coefficients:", [round(s, 3) for s in gamma_fit.std_errors])
print(f"dispersion {phi:.2f}, so the shape is {1 / phi:.2f} (true 2)")
segment_c = Gamma(1 / phi, means[2] * phi)
print(f"segment C: mean {segment_c.mean():.2f}, 10 xs 10 per claim {segment_c.layer(10, 10):.4f}")
```

```text
fitted mean sizes: [5.23, 7.98, 12.5] (true 5, 8, 12)
standard errors of the log coefficients: [0.072, 0.102, 0.102]
dispersion 0.52, so the shape is 1.93 (true 2)
segment C: mean 12.50, 10 xs 10 per claim 3.2147
```

With 100 claims per segment and a shape near 2, the standard error of the log
base mean is 0.072, and of each log relativity 0.102. Thus a fitted mean
several percent from the truth is noise, not a finding. A gamma has a light
tail. Use this model for the **mean** by segment, because primary pricing needs
its relativities. Do not use it to price layers far above the average claim.

## 5. The body and the tail matter for different layers

| Layer | Depends mostly on | Fit it with |
|---|---|---|
| Primary and working layers (at or below the typical claim) | The body and the frequency | A body family, or the empirical distribution |
| Layers that attach in the transition (a few times the mean) | Both | A splice (section 8) |
| High layers, above the largest claims seen | The alpha of the tail and its uncertainty | The Pareto family (section 7) |

This has two results:

- The burning cost of a working layer (the average of what the layer pays on
  past claims) gives a good price, and a body model checks it.
- A high layer has almost no observed payments. Its price is an
  **extrapolation** from the claims above the threshold, and the choice of tail
  family and threshold sets the price.

## 6. Truncated and censored data

Large-loss data is rarely complete, in two ways:

- **Reporting thresholds**: a source reports only losses above some amount,
  and different sources use different amounts. If you ignore this, the tail
  looks heavier than it is (the alpha is too low). The source with the higher
  threshold shows no losses between the two thresholds, so large losses look
  relatively more frequent.
- **Censoring**: a policy limit capped the loss, so the file shows the limit
  and not the loss. If you ignore this, the tail looks thinner (the alpha is
  too high).

`Pareto.fit`, `PiecewisePareto.fit` and `GeneralizedPareto.fit_riegel` take
both: `censored` (a flag for each loss) and `reporting_thresholds` (the
threshold for each loss). Their estimates are maximum likelihood with the
correct likelihood for each loss. `Pareto.fit` also takes `weights` (one per
loss, for example to give old years less weight). It also takes `truncation`,
an upper end above which no loss can occur, such as a sum insured that caps
each risk.

In the example, each large loss has a policy limit of 1.5, 2 or 3, so the
limits cap some losses at different amounts:

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

Both flags move the alpha in the direction that the theory gives. If you
ignore the caps, 43 large losses look smaller than they were. The alpha is then
much too high: 2.415 against 1.765 with all losses known, which gives a thin
tail and cheap high layers. The `censored` flags bring it back to 1.749. The
thresholds case has more noise because it has fewer losses. Section 7.3 shows
how much a small sample can move the alpha.

### Fitting the body: `fit_severity`

`Truncated(severity, lower, upper)` models a claim **given** that it is in a
window. Use it after you fit a body to truncated data.

`fit_severity(family, losses, reporting_thresholds=, censored=)` fits a
lognormal, gamma, Weibull or Burr by maximum likelihood. It conditions each
loss on being more than its own reporting threshold. It counts a loss capped by
a policy limit as at least that limit. It returns the distribution, its
parameters with standard errors, and the AIC and BIC.

`goodness_of_fit(distribution, losses, reporting_thresholds=)` compares a fit
with the product-limit estimate of the same losses. It gives the largest gap
between the two survival functions, and the Q-Q pairs.

A sample cut at 1, because the claims above 1 go to the tail fit, is not the
same as claims capped at 1. Do not treat the cut as censoring: that puts the
body's own mass above 1, where the tail's mass is. `fit_lognormal_between`
below fits a lognormal seen only between `lower` and `upper` by maximum
likelihood. Its likelihood uses the data through the count, the sum of the log
losses and the sum of their squares. The normalizing term subtracts the log of
the probability of the window:

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
print(f"ignoring the threshold:        meanlog {naive_mu:.2f}")
print(f"above 0.1, truncated at 0.1:   meanlog {mu_hat:.2f} (se {se['meanlog']:.2f}), "
      f"sdlog {sd_hat:.2f} (se {se['sdlog']:.2f})")
print(f"0.1 to 1, truncated at both:   meanlog {mu_win:.2f}, sdlog {sd_win:.2f}")
print("truth for the body:            meanlog -2.00, sdlog 1.20")
gof = goodness_of_fit(fit.distribution, reported, reporting_thresholds=thresholds)
print(f"largest gap to the product-limit survival: {gof['max_distance']:.3f}")
print(f"largest three losses: {', '.join(f'{x:.1f}' for x in gof['x'][-3:])}; "
      f"the fit's quantiles there: {', '.join(f'{q:.1f}' for q in gof['model_quantile'][-3:])}")
```

```text
4681 of 8000 claims are above the reporting threshold of 0.1
ignoring the threshold:        meanlog -1.32
above 0.1, truncated at 0.1:   meanlog -1.93 (se 0.05), sdlog 1.04 (se 0.03)
0.1 to 1, truncated at both:   meanlog -2.21, sdlog 1.33
truth for the body:            meanlog -2.00, sdlog 1.20
largest gap to the product-limit survival: 0.017
largest three losses: 11.0, 11.7, 12.6; the fit's quantiles there: 5.0, 5.8, 7.8
```

With the threshold in the fit, the meanlog moves from -1.32 to -1.93, near the
true -2.00. The sdlog, 1.04, stays below 1.20, because the claims above 1
follow the Pareto and not the lognormal (section 4).

The goodness of fit shows where the fit fails. The survival functions never
differ by more than 2 percent, because most claims are in the body. But the
largest losses are much larger than the quantiles of the fit. A small distance
alone does not show that the tail is correct.

The window fit also cuts the sample at 1 and puts both cuts in the likelihood.
This removes the mix of body and tail. The estimate (-2.21 and 1.33) has more
noise, because the window from 0.1 to 1 shows only part of the shape of the
body. Section 8 uses the same function to fit the body below the tail
threshold.

## 7. The tail

### 7.1 When to use the single-parameter Pareto

Above a threshold `t`, a Pareto has survival `(t / x)^alpha`. One parameter,
the alpha, gives the whole tail, and the estimate has a closed form. It is the
default tail for excess of loss for two reasons. It scales: an alpha fitted on
old losses stays correct after inflation. And layers price from each other
(see `docs/design/pareto.md`).

Use it when the alpha is stable as you change the threshold (section 7.2), and
when there are sufficient losses above the threshold to set it. The standard
error of the alpha is about `alpha / sqrt(n)` for `n` losses.
`Pareto.fit_with_standard_error` returns it from the observed information,
which also includes truncation and censoring. The last line below shows the
width of a 90 percent range of the alpha for a given number of losses:

```python
tail, a_se = Pareto.fit_with_standard_error(large, 1.0)
a_hat = tail.alpha
print(f"alpha {a_hat:.3f} from {len(large)} losses above 1 (the truth is 1.8)")
print(f"standard error {a_se:.3f}; alpha / sqrt(n) = {a_hat / math.sqrt(len(large)):.3f}")
print(f"expected loss per large loss, 5 xs 5: {tail.layer(5, 5):.4f}; 10 xs 10: {tail.layer(10, 10):.4f}")
print("90% range of the alpha, as a share of it, for n losses:",
      ", ".join(f"n={n}: +-{1.645 / math.sqrt(n):.0%}" for n in (10, 30, 100, 156)))
```

```text
alpha 1.765 from 156 losses above 1 (the truth is 1.8)
standard error 0.141; alpha / sqrt(n) = 0.141
expected loss per large loss, 5 xs 5: 0.1572; 10 xs 10: 0.0925
90% range of the alpha, as a share of it, for n losses: n=10: +-52%, n=30: +-30%, n=100: +-16%, n=156: +-13%
```

With 30 losses, the 90 percent range of the alpha is about plus or minus 30
percent of it. With 10 losses, it is plus or minus 52 percent. As a rule, below
about 30 losses above the threshold, do not take the alpha from the cedant's
data alone. Price at several alphas and anchor on an alpha from the market or a
larger book (sections 7.3 and 9). Do not use the single Pareto when the local
alpha clearly changes with size (section 7.4).

### 7.2 Choosing the threshold

The threshold balances **bias** against **variance**. If it is too low, body
claims that do not follow the Pareto bend the fit. If it is too high, few
losses remain. Three tools help, and none gives the answer alone:

- **Alpha by threshold**: fit at several thresholds. Where the Pareto applies,
  the alpha is flat (the Pareto is threshold invariant), apart from noise.
- **Mean excess** `e(u) = E[X - u | X > u]`: for a Pareto it increases
  linearly in `u` with slope `1 / (alpha - 1)`. Find the point above which it
  is a straight line.
- **Hill estimates** of `xi = 1 / alpha` from the `k` largest losses: find a
  range of `k` where the estimate is flat.

```python
print(f"{'threshold':>9} {'losses':>7} {'alpha':>6} {'std err':>8}")
for u in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
    above = [x for x in losses if x >= u]
    a = Pareto.fit(above, u).alpha
    print(f"{u:9.2f} {len(above):7d} {a:6.2f} {a / math.sqrt(len(above)):8.2f}")
print("mean excess:", [(u, round(e, 2), n) for u, e, n in mean_excess(losses, [0.5, 1, 2, 3, 5])])
print("Hill alpha for the k largest:", [(k, round(1 / h, 2)) for k, h in zip((20, 40, 80, 156, 300), hill(losses, [20, 40, 80, 156, 300]))])
print(f"{'size':>5} {'above':>6} {'observed share':>15} {'Pareto':>7}")
for x in (1.5, 2.0, 3.0, 5.0, 8.0):
    n_above = sum(1 for v in large if v > x)
    print(f"{x:5g} {n_above:6d} {n_above / len(large):15.3f} {tail.survival(x):7.3f}")
```

```text
threshold  losses  alpha  std err
     0.25    2217   1.50     0.03
     0.50     919   2.16     0.07
     1.00     156   1.76     0.14
     2.00      46   2.09     0.31
     3.00      21   2.24     0.49
     5.00       5   1.47     0.66
mean excess: [(0.5, 0.43, 919), (1.0, 1.08, 156), (2.0, 1.72, 46), (3.0, 2.31, 21), (5.0, 5.28, 5)]
Hill alpha for the k largest: [(20, 2.3), (40, 2.05), (80, 2.02), (156, 1.76), (300, 2.36)]
 size  above  observed share  Pareto
  1.5     84           0.538   0.489
    2     46           0.295   0.294
    3     21           0.135   0.144
    5      5           0.032   0.058
    8      4           0.026   0.025
```

In this synthetic example, the correct threshold is 1, because the truth has
its join there. With real data, the choice is a judgment from several
diagnostics together. Read them as follows:

- **Alpha by threshold.** At 0.5, the alpha is 2.16 with a standard error of
  0.07, far from the 1.76 at 1, so the body still bends it. At 0.25 the fit is
  1.50 with a standard error of 0.03, which is false precision. From 1 up, the
  alphas differ by about one standard error or less, and the standard error
  increases quickly.
- **Mean excess.** It increases from 0.43 at 0.5 to 1.08 at 1, where the tail
  replaces the body. It then continues to increase, but the last points come
  from 21 and 5 losses.
- **Hill.** The alphas are between 1.76 and 2.3 for up to 156 losses, and
  reach 2.36 at 300, where body claims are included. Above 1 this example is
  exactly Pareto, so the values above 1.8 at small `k` are noise. With real
  data, the Hill estimate also has a bias where the tail is not exactly Pareto.
  Thus look for a flat range, not for its value.
- **Fit check.** The last table gives the same information as a QQ plot or a
  log-log survival plot. It shows the share of the large losses above each
  size, with the survival of the fitted Pareto. They agree from 2 up (0.295 against 0.294). At 1.5 the
  data has a few more losses than the Pareto (0.538 against 0.489). The 5
  losses above 5 are too few to read.

The rule: **take the lowest threshold above which the alpha is stable, because
it keeps the most data.** In practice, the cedant's reporting threshold and the
units of the layers that you price also affect the choice. Record the alpha at
several thresholds in the file.

### 7.3 Parameter uncertainty: how sure is the alpha?

The standard error `alpha / sqrt(n)` is a start. For the price of a layer, do
these steps 200 times: resample the losses with replacement, fit again, and
price the layer. The spread of the price is the parameter uncertainty of the
model. It is usually much larger than the spread of the alpha suggests,
because the layer price is a steep function of the alpha.

```python
rng = random.Random(5)
layers = ((5, 5), (10, 10), (20, 20))
boot = []
for _ in range(200):
    refit = Pareto.fit([rng.choice(large) for _ in large], 1.0)
    boot.append((refit.alpha, [refit.layer(limit, att) for limit, att in layers]))
alphas = sorted(a for a, _ in boot)
print(f"alpha {a_hat:.2f}, 90% bootstrap range {alphas[9]:.2f} to {alphas[190]:.2f} (10th smallest and largest of 200)")
true_tail = Pareto(1.0, 1.8)
print(f"{'layer':>9} {'5%':>8} {'fitted':>8} {'95%':>8} {'truth':>8} {'95% / 5%':>9}")
for j, (limit, att) in enumerate(layers):
    prices = sorted(p[j] for _, p in boot)
    print(f"{limit:>3} xs {att:<3} {prices[9]:8.4f} {tail.layer(limit, att):8.4f} {prices[190]:8.4f} "
          f"{true_tail.layer(limit, att):8.4f} {prices[190] / prices[9]:9.1f}")
```

```text
alpha 1.76, 90% bootstrap range 1.57 to 1.96 (10th smallest and largest of 200)
    layer       5%   fitted      95%    truth  95% / 5%
  5 xs 5     0.1072   0.1572   0.2292   0.1468       2.1
 10 xs 10    0.0550   0.0925   0.1545   0.0843       2.8
 20 xs 20    0.0282   0.0545   0.1041   0.0484       3.7
```

The layers are per large loss. The 90 percent range of the alpha is 1.57 to
1.96. For 10 xs 10, this gives a price range from 0.0550 to 0.1545, a factor
of 2.8. For 20 xs 20 the factor is 3.7: the higher the layer, the wider the
range. A reviewer needs this range first.

The price of a layer decreases as the alpha increases. Thus the 5 percent price
is the price at the 95 percent alpha. The code prices each resample, which also
works for quantities that are not monotone in the alpha. Report a layer price
from a fitted tail **with its range**, or load it for the uncertainty (see the
guide on risk loads).

The bootstrap has two limits:

- It resamples only the sizes, so the uncertainty in the number of large losses
  (section 3) is additional.
- With few losses, it is optimistic, because it can only reproduce the losses
  that you have. Then an alpha from the market or a larger book is the better
  anchor (section 9).

### 7.4 When one alpha is not enough: piecewise Pareto and the generalized Pareto

If the alpha clearly differs between ranges, these tools permit it:

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
print(f"{'layer':>9} {'Pareto':>8} {'piecewise':>10} {'general':>8} {'truth':>8}")
for limit, att in ((5, 5), (10, 10), (20, 20)):
    row = [d.layer(limit, att) for d in (tail, piecewise, smooth)]
    print(f"{limit:>3} xs {att:<3} {row[0]:8.4f} {row[1]:10.4f} {row[2]:8.4f} {true_tail.layer(limit, att):8.4f}")
```

```text
piecewise alphas: [1.71, 2.24] on pieces starting at [1.0, 3.0]
generalized Pareto: initial alpha 1.45, tail alpha 2.77
    layer   Pareto  piecewise  general    truth
  5 xs 5     0.1572     0.1133   0.0965   0.1468
 10 xs 10    0.0925     0.0479   0.0337   0.0843
 20 xs 20    0.0545     0.0203   0.0108   0.0484
```

The three tails differ already at 5 xs 5. At 20 xs 20, the single Pareto
charges 0.0545 against 0.0108 for the generalized Pareto, and the 156 losses
cannot show which is correct. Here the generalized Pareto bends the alpha up to
its tail value. It prices the highest layers far below the single Pareto,
which is correct only if the thinning is real.

Which tail does the data support? The single Pareto is a special case of the
other two: equal alphas in both pieces, or for the generalized Pareto, `xi`
equal to `beta` over the threshold. Thus a likelihood ratio test applies, and
AIC compares all three.

`log_likelihood` below sums the log densities of a fitted tail. It takes each
density as the slope of the `cdf`, calculated numerically:

```python
def log_likelihood(dist, xs, h=1e-6):
    """Sum of the log densities, each the slope of the cdf over a small relative step."""
    return sum(math.log((dist.cdf(x * (1 + h)) - dist.cdf(x * (1 - h))) / (2 * h * x)) for x in xs)


print(f"{'tail':>18} {'parameters':>10} {'log-lik':>8} {'AIC':>7}")
lls = {}
for name, d, k in (("Pareto", tail, 1), ("piecewise Pareto", piecewise, 2), ("generalized Pareto", smooth, 2)):
    lls[name] = log_likelihood(d, large)
    print(f"{name:>18} {k:10d} {lls[name]:8.2f} {2 * k - 2 * lls[name]:7.2f}")
for name in ("piecewise Pareto", "generalized Pareto"):
    lr = 2 * (lls[name] - lls["Pareto"])
    print(f"{name} against the single Pareto: likelihood ratio {lr:.2f}, p-value {math.erfc(math.sqrt(lr / 2)):.3f}")
```

```text
              tail parameters  log-lik     AIC
            Pareto          1  -155.82  313.63
  piecewise Pareto          2  -155.19  314.38
generalized Pareto          2  -154.14  312.29
piecewise Pareto against the single Pareto: likelihood ratio 1.26, p-value 0.262
generalized Pareto against the single Pareto: likelihood ratio 3.35, p-value 0.067
```

The generalized Pareto has the best AIC, by a small amount (312.29 against
313.63). But the likelihood ratio test does not reject the single Pareto
(p-value 0.067, and 0.262 for the piecewise Pareto). The 156 losses cannot
separate these tails, but the tails price 20 xs 20 very differently. In this
example the truth is a single Pareto. In practice, you do not know that.

More flexibility does not give more information. **Prefer the simpler tail
unless a likelihood comparison and market evidence both support the extra
parameter. Show the price under the alternatives.** For a high layer, the
lower alpha of a range is a conservative choice, not an unbiased one. It loads
the price for the uncertainty and does not estimate it.

### 7.5 Extreme value theory: Gpd and PotTail

The **generalized Pareto distribution** (`Gpd(xi, beta)`) gives the amount by
which a loss exceeds a high threshold. Extreme value theory shows that the
exceedances of any heavy-tailed loss approach it. `Gpd.fit(exceedances)` fits
it to the amounts **above** the threshold (the loss minus the threshold). The
tail index `xi` is `1 / alpha`.

`PotTail.fit(draws, level)` does the same on a `Sampled` set of losses, with
the threshold at a quantile level. It gives the `var` and `tvar` (the value at
risk and the tail value at risk) of the claim size directly.

Use EVT for a **quantile or TVaR of the claim size** (for example, a 1-in-1000
claim). Use the Pareto family for **layers** and the market's alphas. They are
the same distribution: the shifted `Gpd` equals Riegel's generalized Pareto,
with the same likelihood.

```python
exceedances = [x - 1.0 for x in large]
g = Gpd.fit(exceedances)
print(f"Gpd on exceedances of 1: xi {g.xi:.4f} (alpha {1 / g.xi:.2f}), beta {g.beta:.4f}")
print(f"fit_riegel on the same losses: xi {smooth.xi:.4f}, beta {smooth.beta:.4f}")
q = truth.quantile(0.999)
print(f"truth: VaR 99.9% {q:.2f}, TVaR 99.9% {q + truth.stop_loss(q) / 0.001:.2f}, xi {1 / 1.8:.2f}")
for level in (0.98, 0.99):
    pot = PotTail.fit(Sampled(losses), level)
    print(f"PotTail at {level}: threshold {pot.threshold:.2f}, xi {pot.gpd.xi:.2f}, "
          f"VaR 99.9% {pot.var(0.999):.2f}, TVaR 99.9% {pot.tvar(0.999):.2f}")
```

```text
Gpd on exceedances of 1: xi 0.3611 (alpha 2.77), beta 0.6887
fit_riegel on the same losses: xi 0.3611, beta 0.6887
truth: VaR 99.9% 5.28, TVaR 99.9% 11.88, xi 0.56
PotTail at 0.98: threshold 0.99, xi 0.38, VaR 99.9% 4.68, TVaR 99.9% 8.02
PotTail at 0.99: threshold 1.52, xi 0.41, VaR 99.9% 4.66, TVaR 99.9% 8.24
```

Use `PotTail` on complete data. When losses are censored or reported above
thresholds, use the Pareto fits.

Both VaRs (4.68 and 4.66) are less than the true 5.28. Both TVaRs (8.02 and
8.24) are far below the true 11.88, because the fitted `xi` (0.38 and 0.41) is
below the true 0.56. A TVaR at 99.9 percent is an extrapolation of the tail
index, and a moderate error in `xi` gives a large error in the TVaR. Show it at
more than one threshold, and with the Pareto fits of this section.

## 8. Splicing a body onto a tail

`Mixture.splice(components, breaks)` makes one severity from pieces. Each piece
is conditioned on its own range of sizes. The weights are the probabilities of
the ranges. `breaks` has one more entry than the components: the lower end
(0), the joins and the upper end (infinity).

Fit a lognormal to the claims below the threshold and a Pareto to the claims
above, and join them. Each model then applies where it has data. The example's
`truth` has this form, and this fit uses the tools of this guide.

The body sample is cut at the join: it has no claim above 1. Fit the body with
that cut in the likelihood (`fit_lognormal_between` of section 6, with
`upper=1.0`). The table shows two fits. The plain fit treats the cut sample as
complete. The truncated fit includes the cut:

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
print(f"{'layer':>10} {'observed':>9} {'lognormal':>10} {'plain':>8} {'splice':>8} {'truth':>8}")
for limit, att in ((0.4, 0.1), (1, 1), (4, 1), (5, 5), (10, 10)):
    label = f"{limit:g} xs {att:g}"
    print(f"{label:>10} {burning_cost(att, limit):9.5f} {one.layer(limit, att):10.5f} "
          f"{plain.layer(limit, att):8.5f} {fitted.layer(limit, att):8.5f} {truth.layer(limit, att):8.5f}")
print(f"variance of one claim under the splice: {fitted.variance()}")
```

```text
body, plain fit: meanlog -2.14, sdlog 1.08; truncated at 1: meanlog -2.02, sdlog 1.20 (true -2.00, 1.20)
share above 1: 0.0195; alpha 1.76
     layer  observed  lognormal    plain   splice    truth
0.4 xs 0.1   0.10556    0.10283  0.09213  0.10568  0.10715
    1 xs 1   0.01121    0.01629  0.01049  0.01049  0.01064
    4 xs 1   0.01778    0.02332  0.01805  0.01805  0.01810
    5 xs 5   0.00255    0.00104  0.00307  0.00307  0.00294
  10 xs 10   0.00075    0.00018  0.00180  0.00180  0.00169
variance of one claim under the splice: inf
```

The truncated fit recovers the body (-2.02 and 1.20 against -2.00 and 1.20).
With it, the splice prices the working layer 0.4 xs 0.1 at 0.10568 against a
true 0.10715. The sdlog of the plain fit, 1.08, is too small, because the cut
removed the upper end of the body. The plain fit prices that layer at 0.09213,
which is too low.

From 1 xs 1 up, the two splices agree, because those layers depend on the
tail. Both are near the truth. In the bottom row, 10 xs 10, the observed
average (0.00075) comes from very few claims. The splice (0.00180 against a
true 0.00169) is the better guide, because it carries the information of all
the large losses up the layers. In the working layer, the observed burning
cost (0.10556) is as good as the splice, as section 5 says, and it is the check
on the splice.

Notes:

- The weights set the share of claims above the join. To model only the large
  losses, use the tail alone, `Pareto.fit(large, 1.0)`, with a count of large
  losses (section 3) in place of a count of claims.
- Any tail of section 7 can replace the Pareto: a `PiecewisePareto`, a
  `GeneralizedPareto`, or a `LogAffinePareto` with parameters from a
  benchmark.
- The spliced `Mixture` has the same `layer`, `lev`, `quantile` and `sample`
  as the other severities. Thus it goes directly into the aggregate and tower
  functions of the related guides.
- Its variance is infinite, as the last line shows, because the tail alpha is
  below 2. The standard deviation of one claim thus has no meaning. Use layer
  moments (`layer_variance`) instead.

## 9. When there is very little data

With few observations, the segment relativities and the tail alpha both become
unreliable. `BayesGlm(family, link)` fits the same GLM with a normal prior of
mean 0 on each coefficient. It returns posterior draws (`coefficient_draws`),
so the uncertainty goes into each value that you calculate from the fit,
including the mean of a segment. `predict_distribution` gives the predictive
draws of new claims.

The prior on the relativities has standard deviation `prior_sd` on the log
scale, 2.5 by default. This prior is so wide that it has almost no effect on
the answer. A smaller `prior_sd` states that relativities are modest, and it
pulls thin cells towards the base. Before you read the posterior, make sure
that the chains agree (`rhat` near 1) and that there are no divergences.

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

The 10 claims of segment A have an average of 3.27 against a true 5. Thus the
GLM puts the relativity of C at 3.61 against a true 2.40. The default prior
(standard deviation 2.5) has almost no effect: the relativity of C stays within
10% of the GLM's value.

A prior standard deviation of 0.5 states that relativities are modest. It has
three effects:

- it pulls the mean of segment A from 3.27 to between 4.5 and 6, near the true
  5.
- it pulls the relativity of C to between 1.6 and 2.2.
- it makes the 90 percent range of the mean of segment C narrower than the
  GLM's range (7.54 to 18.53).

A prior is an assumption. It helps here because the true relativities are
modest, and it already pulls the relativity of C below the true 2.40. The code
prints checks, not posterior values, because a sampler's draws differ in their
last digits between platforms. Take `prior_sd` from a larger book or a market
study, never from the answer that you want.

With very few losses above the threshold, the same method applies to the tail.
Use a prior alpha from the market or a larger similar book, and treat the fit
as an update to it. Price the layer at several alphas (section 7.3), and give
each alpha a weight from your belief.

## 10. Checklist and pitfalls

- Fit **frequency and severity on the same definition of a claim**: the same
  threshold, the same trend and development adjustments, and the same currency
  basis.
- Check the **dispersion** of the counts before you choose Poisson. Take it
  from a larger book when you have few years.
- Fit the body and the tail **separately**, and splice them. A single family
  fitted to all claims gives the wrong price at both ends. Put the cut at the
  join (and any reporting threshold) in the likelihood of the body.
- With fewer than about 30 losses above the threshold, anchor the alpha on the
  market or a larger book, and price at several alphas.
- Fit the alpha at **several thresholds** and find the flat range. Report the
  alpha with its range, and show the layer price at the ends of the range.
- Give **`censored`** and **`reporting_thresholds`** when the data has them.
- A tail more flexible than the data supports is not more accurate. Compare it
  with the single Pareto at the layers that you price.
- Develop and trend the large losses before the tail fit. An open claim at its
  current estimate understates the tail. `LargeLossExperience.prepare` does
  both.
- For the tail fit, keep the losses that the burning cost keeps: those whose
  `amount` times `factor` (in `to_dict()`) is at least
  `PreparedLosses.complete_above`. Then fit their `trended` values.
  Development and conversion change the size of a loss, but not whether the
  record is complete for it. With one FX rate, the threshold in the pricing
  currency is `complete_above` times the rate.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): which pricing method to use
  in which situation.
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): loading a
  price for parameter uncertainty and for capital.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md):
  pricing a layer from the cedant's losses and from its exposure.
- [`aggregate-distributions.md`](aggregate-distributions.md): from a frequency
  and a severity to the annual loss distribution.
- [`reserving-methods.md`](reserving-methods.md): developing the losses that
  these models start from.
- [`reinsurance-price.md`](reinsurance-price.md): what a reinsurance price is,
  end to end.
