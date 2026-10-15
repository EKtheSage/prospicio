# Choosing and fitting severity and frequency

## Use this guide when

You have claim counts and claim sizes, from a cedant or from your own book, and
must turn them into a loss model that prospicio can price from: a **frequency**
(how many claims a year) and a **severity** (how big each one is). The guide
tells you which count model and which size model to use in which situation, why,
and the exact calls. It assumes you know GLMs and loss ratios from primary
pricing and are new to the reinsurance side, where the **tail** (the few very
large claims) decides the price of the upper layers.

| Situation | Method | prospicio call | Why |
|---|---|---|---|
| Yearly counts, variance near the mean | Poisson | `Poisson(lam)` | One parameter; nothing to gain from more |
| Yearly counts, variance above the mean | Count chosen by dispersion | `claim_count(mean, variance / mean)` | Matches mean and variance; picks binomial, Poisson or negative binomial |
| Counts driven by a shared year effect with a heavy right tail | Mixed Poisson | `Count.mixed_poisson(mean, cv, mixing="inverse_gaussian")` | You choose the shape of the year effect |
| Counts seen only when at least one claim happened, or too many or too few zeros | Zero-truncated or zero-modified | `Count.zero_truncated(base)`, `Count.zero_modified(base, p0)` | The data has no zeros, or a different share |
| Counts by rating cell and exposure | Quasi-Poisson GLM with an exposure offset | `Glm("poisson", "log", dispersion="pearson")` | Rates by cell and standard errors that allow for over-dispersion |
| Mean claim size by rating cell | Gamma GLM, log link | `Glm("gamma", "log")` | Multiplicative relativities; the dispersion gives the gamma shape |
| Small claims, working layers, primary | A body family | `Lognormal`, `Gamma`, `Weibull`, `Burr`, `Loglogistic` | Fits the bulk of the claims, smooth |
| Large claims, high layers | Pareto family above a threshold | `Pareto.fit`, `PiecewisePareto.fit`, `GeneralizedPareto.fit_riegel` | Describes the tail in the units the market quotes (the alpha) |
| Both matter | A body spliced onto a tail | `Mixture.splice` | One severity; body and tail each fitted where they have data |
| Losses capped by policy limits, or reported above different thresholds | Pareto fit with flags | `Pareto.fit(..., censored=, reporting_thresholds=)` | Ignoring either biases the alpha |
| Which threshold? How sure is the alpha? | Alpha by threshold, mean excess, Hill, bootstrap | `mean_excess`, `hill`, `Pareto.fit` on resamples | The choice moves a high layer by a multiple |
| A tail quantile or TVaR of the claim size | Peaks over threshold | `PotTail.fit`, `Gpd.fit` | The extreme value fit, with VaR and TVaR built in |
| Few observations in some rating cells | Bayesian GLM with a prior on the relativities | `BayesGlm(..., prior_sd=)` | The prior holds thin cells near the base; the draws carry the uncertainty |
| Few large losses | An alpha from the market or a larger book, priced at several alphas | `Pareto(t, alpha)` at each alpha (section 7.3) | prospicio has no Bayesian tail fit; the data cannot pin the alpha down alone |

## 1. Words you need first

- **Frequency** and **severity**: the number of claims in a period, and the size
  of one claim. Together they are the **collective model**: the annual loss is
  the sum of a random number of random sizes.
- **Attritional** and **large** losses: the many small claims that are stable
  from year to year, and the few big ones that drive the volatility. The line
  between them is the **threshold**.
- **Dispersion**: for a claim count, its variance divided by its mean. A Poisson
  count has dispersion 1; **over-dispersion** is a value above 1. Counts of all
  claims across a book often have it (weather, a good or bad year for the whole
  book); counts of large losses above a high threshold are often close to
  Poisson. Test it; do not assume it. In a GLM the dispersion `phi` scales the
  variance function, `Var = phi * V(mu)`: for a Poisson that is the variance over
  the mean above, and for a gamma it is `1 / shape`, the squared coefficient of
  variation.
- **Local Pareto alpha**: how fast the chance of a loss above a size falls as the
  size rises. With alpha 2, doubling the size cuts the frequency to a quarter.
  A low alpha is a heavy tail. The market quotes tails as alphas.
- **Censored**: a loss known only to be at least some amount, because a policy
  limit capped it. **Reporting threshold**: a loss is in the data only if it
  exceeds this amount (**left truncation**).
- **Layer** `limit xs attachment`: the part of a loss between the attachment and
  attachment plus limit. `5 xs 5` pays the part of a loss between 5 and 10.

Money is in millions of a synthetic currency. Layers are priced **per claim**
unless the text says per large loss.

### Before you fit

Every fit below assumes the claims are already on the basis of the contract you
price. Getting them there is most of the work, and a fit cannot repair it.

- **Define the claim** the contract pays on: per claim or per occurrence (an
  event that aggregates several claims), and whether expenses (ALAE) are added
  to the loss or paid in addition to the limit. Counts and sizes must use the
  same definition.
- **Develop** each large loss to its ultimate value (an open claim at its
  current estimate understates the tail), and develop the **counts** too: the
  recent years still have claims not yet reported (IBNR counts).
- **Trend** each loss to the contract year, and move the large-loss threshold
  with the trend: a threshold of 1 today was about 1 / (1 + trend)^years in an
  old year, so a fixed threshold drops old losses that would be large now.

  prospicio does these two steps for you:
  `LargeLossExperience(...).prepare(rate=, to=, development=)` converts each
  loss to the pricing currency (with `fx=` on `LargeLossExperience`), develops
  it to ultimate with a `DevelopmentPattern` (or a list of factors) and trends
  it from its accident date to the contract period.
  `PreparedLosses.complete_above` is the moved threshold (the reporting
  threshold times the largest trend factor). It is compared with each loss's
  reported amount trended, before development and conversion; losses below it
  are left out of the burning cost. For the counts, a `DevelopmentPattern`
  built from a claim-count chain ladder gives each year's IBNR factor, read at
  the year's mid-point: `pattern.at(valuation - (year + 0.5))`. See
  [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md),
  section 2.
- **On-level** the counts to the current exposure before reading their mean and
  dispersion (section 3.1).
- For working layers, keep the **empirical** distribution (the claims
  themselves) next to any fitted body: its burning cost is the check on the
  parametric one (section 5).

## 2. The running example

The data is synthetic, so every fit can be compared with the truth, which a real
analysis never has. Claims have a lognormal body below 1 (98 percent of the
claims) and a Pareto tail above 1 with alpha 1.8 (2 percent). `truth` is that
severity; `losses` is what a cedant's claims file would hold, 8,000 claims.
Section 8 explains the splice that builds `truth`.

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

Start from the history of yearly counts. Compute the mean and the variance and
divide. The dispersion decides the family, and `claim_count` applies the rule:

| Dispersion (variance / mean) | Family | When |
|---|---|---|
| Below 1 | Binomial | A fixed number of risks that each claim at most once; rare |
| About 1 | Poisson | Independent claims, no shared shocks |
| Above 1 | Negative binomial | The usual case: years differ for reasons the whole book shares |

A binomial needs a whole number of trials, so below 1 the mean is kept and the
dispersion moves up to the nearest value that number of trials allows. Read the
returned object, not the dispersion you asked for.

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

**Why it matters.** Both counts have the same mean, so the expected loss of a
per-claim layer does not change. The chance of a year with 15 or more claims is
about ten times higher under the negative binomial. What changes is every term
that is not linear in the number of claims in a year: aggregate limits and
aggregate deductibles, reinstatements, profit commissions, and the capital and
risk load. A Poisson fitted to over-dispersed counts understates the bad years
and so misprices those terms.

**Pitfalls.**

- With 25 years the dispersion has a wide error, as the two values printed above
  show. With a few years of history, take the dispersion from a larger, similar
  book (a market benchmark, or a GLM across many cells, section 3.4) instead of
  from the cedant's own years.
- Count the **same claims** you use in the severity fit. Counts of large losses
  (above a threshold) and counts of all claims have different means and
  different dispersions.
- Bring the history **on level** first: adjust for exposure growth and for the
  frequency trend before taking the mean and variance, or the trend inflates the
  dispersion.

### 3.2 Mixed Poisson: when the year effect has a shape

A mixed Poisson count is a Poisson whose mean is drawn each year as a multiplier
`theta` of mean 1 and coefficient of variation `cv`. Gamma mixing gives the
negative binomial. Use `Count.mixed_poisson` when you want to choose the shape of
the year effect, or when part of it is fixed (`shift`, the share of the mean that
does not vary).

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

The gamma row equals the negative binomial of section 3.1. The inverse Gaussian
has the same mean and variance and a heavier right tail. Choose it when a few
bad years in the history look worse than a negative binomial allows. When you
know only the mean and the variance, keep the negative binomial: 25 years cannot
tell the two shapes apart.

### 3.3 Zero-truncated and zero-modified counts

Some data never contains a zero: a claims file lists only policies that had a
claim. Model the count **given at least one** with `Count.zero_truncated`. When
the share of zeros differs from what the base count implies (more policies
without a claim than a Poisson allows), `Count.zero_modified` lets you set the
probability of zero.

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

Neither estimates the base parameter for you. Choose the Poisson mean so that
the truncated mean matches the data, then wrap it.

### 3.4 Counts by rating cell: a GLM with an exposure offset

When you have counts by segment, year and exposure, fit a Poisson GLM with the
log of exposure as an **offset**, exactly as in primary pricing. Claim counts are
over-dispersed, so set `dispersion="pearson"` (a quasi-Poisson fit): the
coefficients are the Poisson ones and the standard errors are scaled by the
estimated dispersion. A negative binomial GLM is the alternative, but prospicio
takes its parameter `theta` as given and does not estimate it.

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

The Poisson standard errors are too small by the square root of the dispersion,
the last number printed, so a Poisson fit would call differences between
segments significant that the data does not support. The fitted dispersion also
gives the count for next year: take the predicted mean for a segment and give it,
with the dispersion, to `claim_count`.

```python
new = coding.design({"seg": ["C"], "expo": [120.0]}, offset=[math.log(120.0)])
mu = quasi.predict(new)[0]
c = claim_count(mu, quasi.dispersion)
print(f"segment C on 120 exposure: mean {mu:.2f}, {type(c).__name__} with r {c.r:.2f}")
```

```text
segment C on 120 exposure: mean 15.64, NegativeBinomial with r 10.24
```

A frequency trend goes in as a numeric term (`Terms().numeric("year")`); the
same GLM then gives the annual trend and its standard error.

## 4. Claim sizes: which family

| Family | Tail | Use it for | Notes |
|---|---|---|---|
| `Lognormal(meanlog, sdlog)` | Medium: its local alpha keeps rising | The body of most lines | Closed-form fit from the mean and standard deviation of the logs |
| `Gamma(shape, scale)` | Light (exponential) | Average size by segment; homogeneous or capped claims | The GLM family for severity; `Gamma.from_mean_cv` |
| `Weibull` | Light for shape above 1; heavier than exponential (but lighter than a power tail) for shape below 1 | Sizes with a natural scale | |
| `Burr`, `Loglogistic` | Heavy (a power tail) with a flexible body | One family for body and moderate tail; liability | `fit_severity("burr", ...)`; the loglogistic is fitted outside prospicio |
| `InverseGamma` | Heavy (a power tail) | Skewed bodies with a heavy tail | |
| `InverseGaussian` | Light (exponential) | Skewed bodies | Lighter in the tail than the lognormal |
| `Pareto(t, alpha)` and its relatives | Heavy (a power tail) | Large losses above a threshold | Fitted with truncation and censoring, section 7 |
| `Truncated(severity, lower, upper)` | As the base | A claim given that it falls in a window | Section 6 |
| `Mixture`, `Mixture.splice` | What you give it | Body and tail together | Section 8 |

Every parametric family has `cdf`, `quantile`, `mean`, `sample`, `lev` (the
limited expected value `E[min(X, limit)]`) and `layer`. Besides the Pareto
family and the `Gpd` of section 7, `fit_severity` fits the lognormal, gamma,
Weibull and Burr, with reporting thresholds and censoring (section 6); for the
others, fit with `scipy` or a formula and pass the parameters in. The lognormal is closed
form (the mean and standard deviation of the logs), so we can fit one to all the
claims and see how it prices layers:

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

The lognormal fits the bulk of the claims, and still prices the layers badly: it
overstates the layers near 1 and understates the high ones, because one pair of
parameters cannot bend to the body and the tail at once. The observed average in
the highest layer is also poor, since it rests on a handful of claims. This is
the central point of the guide: **a family chosen and fitted for the body cannot
be trusted in the tail**, and sections 7 and 8 show what to do instead.

### Severity by segment: a gamma GLM

For the average size by rating cell, fit a gamma GLM with a log link to claim
sizes (primary pricing's severity model). Fit it to losses **capped at the
large-loss threshold**, and leave the part above to the tail model of section
7: a few large losses would otherwise swing the relativities. The fitted
dispersion is `1 / shape`, so a segment's mean and the dispersion give a `Gamma`
for the claim size of that segment.

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

With 100 claims a segment and a shape near 2, the standard error of the log
base mean is 0.072, and of each log relativity 0.102, so a fitted mean several
percent from the truth is noise, not a finding. A gamma has a
light tail, so use this model for the **mean** by segment, as the relativities
it gives are what primary pricing needs. Do not use it to price layers far above
the average claim.

## 5. The body and the tail matter for different layers

| Layer | Depends mostly on | Fit it with |
|---|---|---|
| Primary and working layers (at or below the typical claim) | The body and the frequency | A body family, or the empirical distribution |
| Layers attaching in the transition (a few times the mean) | Both | A splice (section 8) |
| High layers, above the largest claims seen | The tail's alpha and its uncertainty | The Pareto family (section 7) |

Two consequences follow. First, a working layer is priced well from its own
burning cost (the average of what the layer would have paid on past claims)
and a body model checks it. Second, a high layer has almost no observed
payments, so its price is an **extrapolation** from the claims above the
threshold, and the choice of tail family and threshold decides the price.

## 6. Truncated and censored data

Large-loss data is rarely complete, in two ways:

- **Reporting thresholds**: a source reports only losses above some amount, and
  different sources use different amounts. Ignoring this makes the tail look
  heavier than it is (the alpha is too low): the source with the higher
  threshold shows no losses between the two thresholds, as if there were none,
  so large losses look relatively more common.
- **Censoring**: a policy limit capped the loss, so the file shows the limit and
  not the loss. Ignoring this makes the tail look thinner (the alpha is too high).

`Pareto.fit`, `PiecewisePareto.fit` and `GeneralizedPareto.fit_riegel` take both:
`censored` (a flag for each loss) and `reporting_thresholds` (the threshold for
each loss). Their estimates are maximum likelihood with the correct likelihood
for each loss. `Pareto.fit` also takes `weights` (one per loss, for example to
down-weight old years) and `truncation` (an upper end above which no loss can
occur, such as a sum insured that caps every risk).

In the example each large loss sits under a policy limit of 1.5, 2 or 3, so
some are capped at different amounts:

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

Both flags move the alpha in the direction the theory says. Ignoring the caps
treats 43 large losses as smaller than they were, and the alpha comes out far
too high (2.415 against 1.765 with every loss known: a thin tail, cheap high
layers); the `censored` flags bring it back to 1.749. The thresholds case is
the noisier because it has fewer losses; section 7.3 puts a number on how much a
small sample can move the alpha.

### Fitting the body: `fit_severity`

`Truncated(severity, lower, upper)` models a claim **given** that it lies in a
window, and is the right object once you have fitted a body to truncated data.
To **fit** one, `fit_severity(family, losses, reporting_thresholds=,
censored=)` takes a lognormal, gamma, Weibull or Burr by maximum likelihood,
each loss conditioned on exceeding its own reporting threshold and a loss capped
by a policy limit counted as at least that limit. It returns the distribution,
its parameters with standard errors, and the AIC and BIC.
`goodness_of_fit(distribution, losses, reporting_thresholds=)` sets a fit beside
the product-limit estimate of the same losses: the largest gap between the two
survival functions, and the Q-Q pairs.

`fit_severity` has no upper truncation: a sample cut at 1 because the claims
above 1 went to the tail fit is not the same as claims capped at 1, and
censoring them at 1 would put the body's own mass above 1 where the tail's is.
For a window, write the likelihood. For a lognormal seen only between `lower`
and `upper`, it depends on the data through the count, the sum of the log losses
and the sum of their squares, and the normalizing term subtracts the log of the
probability of the window:

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

Accounting for the threshold moves the meanlog from -1.32 to -1.93, close to the
true -2.00. The sdlog, 1.04, stays short of 1.20, because the claims above 1
follow the Pareto and not the lognormal (section 4). The goodness of fit shows
where: the survival functions never differ by more than 2 percent, since most
claims are in the body, but the largest losses run well past the fit's
quantiles. A small distance alone does not clear a tail. Cutting the sample at 1
as well, and putting both cuts in the likelihood, removes that mix-up; the
estimate (-2.21 and 1.33) is noisier, because the window from 0.1 to 1 shows
only part of the body's shape. Section 8 uses the same function to fit the body
below the tail threshold.

## 7. The tail

### 7.1 When to use the single-parameter Pareto

Above a threshold `t`, a Pareto has survival `(t / x)^alpha`. One parameter,
the alpha, describes the whole tail, and the estimate has a closed form. It is
the default tail for excess of loss because it scales (an alpha fitted on old
losses still holds after inflation) and because layers price from each other
(see `docs/design/pareto.md`). Use it when the alpha looks stable as you vary
the threshold (section 7.2) and there are enough losses above the threshold to
pin it down. The standard error of the alpha is about `alpha / sqrt(n)` for `n`
losses (`Pareto.fit_with_standard_error` returns it, from the observed
information, which also covers truncation and censoring), so the last line below
shows how wide a 90 percent range of the alpha is for a given number of losses:

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

With 30 losses the 90 percent range of the alpha is about plus or minus 30
percent of it, and with 10 losses plus or minus 52 percent. As a rule, below
about 30 losses above the threshold do not take the alpha from the cedant's data
alone: price at several alphas and anchor on one from the market or a larger
book (sections 7.3 and 9). Do not use the single Pareto when the local alpha
clearly changes with size (section 7.4).

### 7.2 Choosing the threshold

The threshold trades **bias** against **variance**: too low, and body claims
that do not follow the Pareto bend the fit; too high, and few losses remain.
Three tools help, and none gives the answer alone.

- **Alpha by threshold**: fit at several thresholds. Where the Pareto holds, the
  alpha is flat (the Pareto is threshold invariant), up to noise.
- **Mean excess** `e(u) = E[X - u | X > u]`: for a Pareto it rises linearly in `u`
  with slope `1 / (alpha - 1)`. Look for the point above which it is a straight
  line.
- **Hill estimates** of `xi = 1 / alpha` from the `k` largest losses: look for a
  range of `k` where it is flat.

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

In this synthetic example the right threshold is known to be 1, because the
truth was built with its join there. With real data the choice is a judgment,
made from several diagnostics together; here is how they read.

- **Alpha by threshold.** At 0.5 the alpha is 2.16 with a standard error of
  0.07, far from the 1.76 at 1, so the body still bends it; at 0.25 the fit is
  1.50 with a standard error of 0.03, which is false precision. From 1 up the
  alphas differ by about one standard error or less, and the standard error
  grows quickly.
- **Mean excess.** It jumps from 0.43 at 0.5 to 1.08 at 1, as the body gives way
  to the tail, and then keeps rising, but the last points rest on 21 and 5
  losses.
- **Hill.** The alphas lie between 1.76 and 2.3 for up to 156 losses, and reach
  2.36 at 300, where body claims have entered. Above 1 this example is exactly
  Pareto, so the values above 1.8 at small `k` are noise; with real data the
  Hill estimate is also biased where the tail is not exactly Pareto, so read it
  for a flat range, not for its value.
- **Fit check.** The last table is what a QQ plot or a log-log survival plot
  shows: the share of the large losses above each size next to the fitted
  Pareto's survival. They agree from 2 up (0.295 against 0.294); at 1.5 the data
  has a few more losses than the Pareto (0.538 against 0.489), and the 5 losses
  above 5 are too few to read.

The rule is: **take the lowest threshold above which the alpha is stable,
because it keeps the most data.** In practice the cedant's reporting threshold
and the units of the layers you must price also set the choice; keep a record
of the alpha at several thresholds in the file.

### 7.3 Parameter uncertainty: how sure is the alpha?

The standard error `alpha / sqrt(n)` is a start. For the price of a layer,
resample the losses with replacement, refit and price the layer each time. The
spread of the price is the parameter uncertainty of the model, and it is
usually much larger than the spread of the alpha suggests, because the layer is
a steep function of the alpha.

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

The layers are per large loss. A 90 percent range of the alpha from 1.57 to
1.96 becomes a range for the price of 10 xs 10 from 0.0550 to 0.1545, a factor
of 2.8, and a factor of 3.7 for 20 xs 20: the higher the layer, the wider. That
range is the first number a reviewer needs to see. Since a layer's price falls
as the alpha rises, the 5 percent price is the price at the 95 percent alpha;
pricing every resample, as here, also works for quantities that are not
monotone in the alpha. Report a layer price from a fitted tail **with its
range**, or load it for the uncertainty (see the sibling guide on risk loads).

The bootstrap has two limits. It resamples the sizes only, so the uncertainty in
the number of large losses (section 3) comes on top. And with few losses it is
optimistic, because it can only reproduce the losses in hand; then an alpha from
the market or a larger book is the better anchor (section 9).

### 7.4 When one alpha is not enough: piecewise Pareto and the generalized Pareto

If the alpha clearly differs between ranges, two tools allow it.

| Tool | Local alpha | Use it when | Cost |
|---|---|---|---|
| `PiecewisePareto.fit(losses, t=[...])` | A step function, one alpha for each piece | The market or the data gives distinct ranges (a working range and a catastrophe range) | A parameter for each piece, each fitted on fewer losses |
| `GeneralizedPareto.fit_riegel(losses, t)` | Moves smoothly from an initial alpha at `t` to a tail alpha far out | The tail seems lighter, or heavier, with size | Two parameters; the far tail is an extrapolation |
| `LogAffinePareto(t, alpha0, gamma)` | Rises linearly in the log of the size, lognormal-like | You want a lognormal-type tail in alpha units | No fitting routine; give the two numbers from a benchmark |

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

The three differ already at 5 xs 5, and at 20 xs 20 the single Pareto charges
0.0545 against 0.0108 for the generalized Pareto, where nothing in 156 losses
can settle the matter. The generalized Pareto here bends the alpha up to its
tail value and prices the highest layers far below the single Pareto, which is
right only if the thinning is real.

Which tail does the data support? The single Pareto is a special case of the
other two (equal alphas in both pieces; for the generalized Pareto, `xi` equal
to `beta` over the threshold), so a likelihood ratio test applies, and AIC
compares all three.

The fitted tails do not report their likelihood, so compute it. The density is
the slope of the `cdf`, taken numerically here:

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

The generalized Pareto has the best AIC, by a little (312.29 against 313.63),
but the likelihood ratio test does not reject the single Pareto (p-value 0.067,
and 0.262 for the piecewise one). The 156 losses cannot tell these tails apart,
although they price 20 xs 20 very differently. In this example the truth is a
single Pareto; in practice you do not know that. Added flexibility is not added
information: **prefer the simpler tail unless a likelihood comparison and
market evidence both support the extra parameter, and show the price under the
alternatives.** Choosing the lower alpha of a range for a high layer is a
conservative choice, not an unbiased one: it loads the price for the
uncertainty instead of estimating it.

### 7.5 Extreme value theory: Gpd and PotTail

The **generalized Pareto distribution** (`Gpd(xi, beta)`) describes how much a
loss exceeds a high threshold, and is what extreme value theory says the
exceedances of any heavy-tailed loss approach. `Gpd.fit(exceedances)` fits it to
the amounts **above** the threshold (the loss minus the threshold). The tail
index `xi` is `1 / alpha`. `PotTail.fit(draws, level)` does the same on a
`Sampled` set of losses with the threshold at a quantile level, and gives
`var` and `tvar` (the value at risk and the tail value at risk) of the claim
size directly.

Use EVT when you want a **quantile or TVaR of the claim size** (a 1-in-1000
claim, say), and use the Pareto family when you want **layers** and the market's
alphas. They are the same distribution: the shifted `Gpd` equals Riegel's
generalized Pareto, and its likelihood is the same.

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

`PotTail` does not accept censoring or reporting thresholds: use it on complete
data, and the Pareto fits when the data has those defects. Both VaRs (4.68 and
4.66) fall short of the true 5.28, and both TVaRs (8.02 and 8.24) are far below
the true 11.88, because the fitted `xi` (0.38 and 0.41) is below the true 0.56.
A TVaR at 99.9 percent is an extrapolation of the tail index, and a moderate
error in `xi` becomes a large one in the TVaR. Show it at more than one
threshold, and next to the Pareto fits of this section.

## 8. Splicing a body onto a tail

`Mixture.splice(components, breaks)` builds one severity from pieces, each
conditioned on its own range of sizes. The weights are the probabilities of the
ranges, and `breaks` has one more entry than the components: the lower end (0),
the joins and the upper end (infinity). A lognormal fitted to the claims below
the threshold, joined to a Pareto fitted to the claims above, uses each model
where it has data. This is how the example's `truth` was built, and this is the
fit that uses the tools of this guide.

The body sample is cut at the join: no claim above 1 is in it. Fit the body with
that cut in the likelihood (`fit_lognormal_between` of section 6, with
`upper=1.0`). The table shows both the plain fit, which treats the cut sample as
if it were whole, and the truncated one:

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

The truncated fit recovers the body (-2.02 and 1.20 against -2.00 and 1.20),
and with it the splice prices the working layer 0.4 xs 0.1 at 0.10568 against a
true 0.10715. The plain fit's sdlog of 1.08 is too small, because the cut
removed the body's upper end, and it prices that layer at 0.09213, too low. From
1 xs 1 up the two splices agree, since those layers depend on the tail, and both
are close to the truth. In the bottom row, 10 xs 10, the observed average
(0.00075) rests on a handful of claims, and the splice (0.00180 against a true
0.00169) is the better guide, since it carries the information of all the large
losses up the layers. In the working layer the observed burning cost (0.10556)
is as good as the splice, as section 5 says, and is the check on it. Notes:

- The weights set the share of claims above the join. To model only the large
  losses, use the tail alone, `Pareto.fit(large, 1.0)`, with a count of large
  losses (section 3) in place of a count of claims.
- Any tail of section 7 can take the Pareto's place: a `PiecewisePareto`, a
  `GeneralizedPareto`, or a `LogAffinePareto` with parameters from a benchmark.
- The spliced `Mixture` has the same `layer`, `lev`, `quantile` and `sample`
  as the other severities, so it goes straight into the aggregate and tower
  functions of the sibling guides. Its variance is infinite, as the last line
  shows, because the tail alpha is below 2, so the standard deviation of a
  single claim carries no meaning; use layer moments (`layer_variance`) instead.

## 9. When there is very little data

Both the segment relativities and the tail alpha get unreliable with few
observations. `BayesGlm(family, link)` fits the same GLM with a normal prior of
mean 0 on each coefficient and returns posterior draws (`coefficient_draws`), so
the uncertainty carries to anything you compute from the fit, a segment's mean
included (`predict_distribution` gives the predictive draws of new claims). The
prior on the relativities has standard deviation `prior_sd` on the log scale,
2.5 by default, which is so wide that it hardly moves the answer. A tighter
`prior_sd` is a statement that relativities are modest, and it pulls thin cells
towards the base. Check that the chains agree (`rhat` near 1) and that there
are no divergences before reading the posterior.

Here are 10 claims a segment, true means 5, 8 and 12 again:

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
relativity of C at 3.61 against a true 2.40. The default prior (standard
deviation 2.5) changes almost nothing: the relativity of C stays within 10% of
the GLM's. A prior standard deviation of 0.5, a belief that relativities are
modest, pulls segment A's mean from 3.27 to between 4.5 and 6, near the true 5,
pulls the relativity of C to between 1.6 and 2.2, and narrows the 90 percent
range of segment C's mean (7.54 to 18.53 for the GLM). A prior is an
assumption: it helps here because the true relativities are modest, and it
already pulls C's relativity below the true 2.40. The checks are printed rather
than the posterior values because a sampler's draws differ in their last digits
from one platform to another. Take `prior_sd` from a larger book or a market
study, never from the answer you want.

With a handful of losses above the threshold, the same idea applies to the tail:
use a prior alpha from the market or a larger similar book, and treat the fit as
an update to it. prospicio has no Bayesian Pareto fit yet; the workable route
today is to price the layer at several alphas (section 7.3) and weigh them by
your belief.

## 10. Checklist and pitfalls

- Fit **frequency and severity on the same definition of a claim**: same
  threshold, same trend and development adjustments, same currency basis.
- Check the **dispersion** of the counts before choosing Poisson, and take it
  from a larger book when you have few years.
- Fit the body and the tail **separately** and splice. A single family fitted to
  all claims mis-prices both ends. Put the cut at the join (and any reporting
  threshold) in the body's likelihood.
- With fewer than about 30 losses above the threshold, anchor the alpha on the
  market or a larger book and price at several alphas.
- Fit the alpha at **several thresholds** and look for the flat range. Report
  the alpha with its range and show the layer price at the ends.
- Pass **`censored`** and **`reporting_thresholds`** whenever the data has them.
- A tail more flexible than the data supports is not more accurate. Compare it
  with the single Pareto at the layers you price.
- Develop and trend the large losses before the tail fit. An open claim at its
  current estimate understates the tail. `LargeLossExperience.prepare` does
  both. For the tail fit, keep the losses the burning cost keeps: those whose
  `amount` times `factor` (in `to_dict()`) is at least
  `PreparedLosses.complete_above`. Then fit their `trended` values.
  Development and conversion change a loss's size but not whether the record
  is complete for it. With one FX rate, the threshold in the pricing currency
  is `complete_above` times the rate.

## What prospicio does not do yet

- `fit_severity` covers the lognormal, gamma, Weibull and Burr with reporting
  thresholds and censoring, but not upper truncation (a sample cut at a
  window's top), and not the loglogistic or the inverse families.
- `Lognormal` has no `survival` and no `layer_variance`, which the other
  families have.
- `Glm("negative_binomial")` needs `theta` and does not estimate it.
- `PotTail.fit` takes a `Sampled`, while `Gpd.fit` takes a list of exceedances.
- The fitted tails (`Pareto`, `PiecewisePareto`, `GeneralizedPareto`) do not
  report their log-likelihood or AIC, so comparing them takes the hand-written
  likelihood of section 7.4.
- There is no Bayesian Pareto or tail fit, and `LogAffinePareto` has no fit.

## See also

- [`choosing-a-method.md`](choosing-a-method.md): which pricing method to use
  for which situation.
- [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md): loading a
  price for parameter uncertainty and for capital.
- [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md):
  pricing a layer from the cedant's losses and from its exposure.
- [`aggregate-distributions.md`](aggregate-distributions.md): turning a
  frequency and a severity into the annual loss distribution.
- [`reserving-methods.md`](reserving-methods.md): developing the losses these
  models start from.
- [`reinsurance-price.md`](reinsurance-price.md): what a reinsurance price is,
  end to end.
