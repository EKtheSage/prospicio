# GLMs for insurance rating: Goldburd, Khare, Tevet and Guller with prospicio

Mark Goldburd, Anand Khare, Dan Tevet and Dmitriy Guller's *Generalized Linear
Models for Insurance Rating* (CAS Monograph 5, second edition, 2025 revision)
is the CAS text on building a classification rating plan with GLMs: the log
link as a multiplicative plan, offsets and weights, frequency, severity and
Tweedie pure premium models, continuous variables and interactions, the
measures that refine a model (deviance, F-tests, AIC, residuals) and those
that choose between finished ones (lift, double lift and loss ratio charts,
the Gini index), and extensions such as GAMs and elastic nets.

This guide follows that workflow in prospicio's Python API (`prospicio.models`,
with `prospicio.distributions` for the Tweedie and the F-test). The
monograph's examples come from insurers' data the PDF does not contain: its
coefficient tables (Tables 4 to 12, Figure 9, Table 14) and its Gini of 56.1%
cannot be rerun, and this guide does not invent data to match them. Where the
monograph prints arithmetic that can be redone from what it prints (rating
factors, offsets, the F-statistic, an interaction's reparameterization, the
Tweedie parameters, confusion matrix rates) the code recomputes it and prints
the monograph's value beside prospicio's. The rest of the workflow runs on a
portfolio the repository ships.

This first part covers the GLM as a rating plan, offsets and weights, the
choice of distribution and the tests of significance (the monograph's chapters
2 to 6.2); the second,
[`glm-ratemaking-goldburd-validation.md`](glm-ratemaking-goldburd-validation.md),
covers continuous variables, interactions, residuals, validation on held-out
data and regularization (chapters 5.4 to 7 and 10). The monograph's text and
exhibits are the authors'; what follows explains them in new words. Section
numbers in parentheses are the monograph's. Each part runs in about a second.

## 1. The data

`validation/data/glm_policies.csv` is a synthetic portfolio of 600 policies
written by `validation/scripts/statsmodels_glm.py` for the GLM parity tests:
a driver age (18 to 80), a region (A to D, with A the most populous and D the
thinnest), an exposure in years and responses for every family, all drawn
under a fixed seed from a known model. That is its value here: the true
relativities are known, so we can see what a GLM recovers and what it misses.
The claim count `claims` is Poisson with mean
`exposure * exp(-1 - 0.01 (age - 40)) * r` and region factors
`r = 1.0, 1.2, 0.8, 1.5` for A to D; `wavy_claims` is Poisson with a
deliberately non-linear age effect, `exposure * exp(0.5 + 0.8 sin(age / 8))`;
`avg_sev` is a gamma average of `n_sev` claims; `pure` is a compound
Poisson-gamma pure premium. The portfolio drew 155 claims where the true model expects 134, about 1.8
standard deviations more, which shows up below as a base rate above the true
one. The French motor portfolio freMTPL2 that the
library's large parity test uses is downloaded on demand and is not in the
repository, so a guide that CI runs offline cannot use it.

```python
import csv
import math
from collections import Counter
from pathlib import Path

from prospicio.models import Design, Glm, Terms

rows = list(csv.DictReader(open(Path("../../validation/data/glm_policies.csv"))))
NUMERIC = ["age", "exposure", "claims", "n_sev", "avg_sev", "ig", "nb", "trials", "pure", "wavy_claims"]
data = {k: [float(r[k]) for r in rows] for k in NUMERIC}
data["region"] = [r["region"] for r in rows]
n = len(rows)
log_e = [math.log(e) for e in data["exposure"]]

print(n, "policies,", round(sum(data["exposure"]), 1), "years of exposure,",
      int(sum(data["claims"])), "claims")
print("policies by region:", dict(sorted(Counter(data["region"]).items())))
true_r = {"A": 1.0, "B": 1.2, "C": 0.8, "D": 1.5}
true_mu = [e * math.exp(-1 - 0.01 * (a - 40)) * true_r[g] for e, a, g in zip(data["exposure"], data["age"], data["region"])]
print(f"claims expected under the true model: {sum(true_mu):.1f}")
```

```text
600 policies, 358.0 years of exposure, 155 claims
policies by region: {'A': 243, 'B': 163, 'C': 120, 'D': 74}
claims expected under the true model: 134.0
```

## 2. The GLM as a multiplicative rating plan (sections 2.1 to 2.4)

A GLM says the response is drawn from an exponential-family distribution with
mean `mu` and dispersion `phi`, and that a link function of the mean is linear
in the predictors. With the log link the linear predictor turns into a product
of factors: `mu = exp(b0) * exp(b1 x1) * ...`, a base rate times one
relativity per rating variable. That is why the monograph treats the log link
as the default for rating: premiums stay positive and a surcharge is a
percentage, not a dollar amount.

Section 2.1.3 works this through for a gamma severity model with intercept
5.8, a driver-age slope of 0.1 and a married coefficient of -0.15, dispersion
0.3. The variance of a GLM response is `phi * V(mu)`, with `V(mu) = mu^2` for
the gamma, so a gamma with dispersion `phi` has coefficient of variation
`sqrt(phi)` whatever its mean; prospicio's `Gamma.from_mean_cv` shows the
variances of section 2.2:

```python
from prospicio.distributions import Gamma

b0, b_age, b_married, phi = 5.8, 0.1, -0.15, 0.3
eta = b0 + b_age * 25 + b_married * 1
print(f"25-year-old married: linear predictor {eta:.2f}, severity {math.exp(eta):,.2f}"
      "   (monograph 8.15 and 3,463.38)")
factors = [round(math.exp(b0), 2), round(math.exp(b_age * 25), 3), round(math.exp(b_married), 3)]
print("base and factors:", factors, "   (monograph 330.30, 12.182, 0.861)")
print(f"product of the rounded factors {math.prod(factors):,.2f}   (monograph 3,464.42)")
print(f"35-year-old unmarried: {math.exp(b0 + b_age * 35):,.0f}   (monograph 10,938)")
for mean, book in [(3464.0, "3.6"), (10938.0, "35.9")]:
    var = Gamma.from_mean_cv(mean, math.sqrt(phi)).variance()
    print(f"variance at mean {mean:,.0f}: {var / 1e6:.1f} million   (monograph {book} million)")
```

```text
25-year-old married: linear predictor 8.15, severity 3,463.38   (monograph 8.15 and 3,463.38)
base and factors: [330.3, 12.182, 0.861]    (monograph 330.30, 12.182, 0.861)
product of the rounded factors 3,464.42   (monograph 3,464.42)
35-year-old unmarried: 10,938   (monograph 10,938)
variance at mean 3,464: 3.6 million   (monograph 3.6 million)
variance at mean 10,938: 35.9 million   (monograph 35.9 million)
```

Both monograph figures for the 25-year-old are right: 3,463.38 is the exact
mean and 3,464.42 is the product of the factors rounded to the precision a
rating manual prints, which is the monograph's point about rounding.

A categorical variable enters as one 0/1 column per level except a base level,
and each coefficient is a log relativity to the base. Section 2.4.2's vehicle
type coefficients (SUV 1.23, truck 0.57, van -0.30 against sedan) become
factors by exponentiating; moving the base to van (Table 5) subtracts the van
coefficient from every level. A logged continuous predictor gives a power
curve (section 2.4.1: a coefficient of 0.62 on log amount of insurance makes
$200,000 cost `2^0.62` times $100,000), and a logistic model's coefficients
are log odds ratios (section 2.8: 0.24 means 27% higher odds):

```python
vtype = {"SUV": 1.23, "truck": 0.57, "van": -0.30}
print("factors:", {k: round(math.exp(b), 3) for k, b in vtype.items()},
      "  (monograph SUV 3.421, van 0.741)")
print(f"SUV {math.exp(1.23) - 1:+.0%}, van {math.exp(-0.30) - 1:+.1%}   (monograph +242%, -25.9%)")
rebased = {"sedan": 0.0 - vtype["van"], "SUV": vtype["SUV"] - vtype["van"], "truck": vtype["truck"] - vtype["van"]}
print("van as base:", {k: round(b, 2) for k, b in rebased.items()},
      "  (monograph Table 5: 0.30, 1.53, 0.88)")
print(f"AOI 200k against 100k: {(200 / 100) ** 0.62:.2f}   (monograph 1.54)")
print(f"odds factor for a coefficient of 0.24: {math.exp(0.24) - 1:.0%}   (monograph 27%)")
```

```text
factors: {'SUV': 3.421, 'truck': 1.768, 'van': 0.741}   (monograph SUV 3.421, van 0.741)
SUV +242%, van -25.9%   (monograph +242%, -25.9%)
van as base: {'sedan': 0.3, 'SUV': 1.53, 'truck': 0.87}   (monograph Table 5: 0.30, 1.53, 0.88)
AOI 200k against 100k: 1.54   (monograph 1.54)
odds factor for a coefficient of 0.24: 27%   (monograph 27%)
```

The truck coefficient rebased is 0.87 from the two-decimal coefficients Table 4
prints and 0.88 in Table 5: the unrounded estimates behind Table 4 must differ
by 0.875 or more (0.574 and -0.304 would do). It is rounding, not an error.

What changes with the base level is the standard errors (section 2.4.3). A
coefficient measures a level against the base, so a thin base level makes every
comparison uncertain. On our portfolio, region D has 74 policies and A has 243.
prospicio's `Terms` builds the design in treatment coding and takes the
reference level as an argument; the claim count model has the log of exposure
as its offset (section 2.6, next):

```python
def coding(*terms, reference=None):
    """Intercept, numeric columns and the region factor with the given base level."""
    t = Terms().intercept()
    for name in terms:
        t = t.numeric(name)
    return t.factor("region", reference=reference).fit(data)


poisson = Glm("poisson", "log")
for base in ["A", "D"]:
    fit = poisson.fit(coding("age", reference=base).design(data, offset=log_e), data["claims"])
    cells = [f"{name} {b:+.3f} ({se:.3f})" for name, b, se in
             zip(fit.names[2:], fit.coefficients[2:], fit.std_errors[2:])]
    print(f"base {base}: " + ", ".join(cells))
```

```text
base A: region[B] +0.078 (0.192), region[C] -0.258 (0.249), region[D] +0.204 (0.245)
base D: region[A] -0.204 (0.245), region[B] -0.126 (0.253), region[C] -0.462 (0.299)
```

The two fits are the same model: D against A is +0.204 in the first and A
against D is -0.204 in the second, and the predictions agree. But with D as
the base every standard error is larger (0.25 to 0.30 against 0.19 to 0.25),
because every coefficient now leans on the thinnest level. The true factors
are 1.2, 0.8 and 1.5 for B, C and D against A, that is log relativities of
+0.18, -0.22 and +0.41; with 155 claims none of the three is estimated sharply,
which is the monograph's remark that a GLM gives the data full credibility and
reports the uncertainty only in the standard errors (section 2.10).

## 3. Offsets and weights (sections 2.5 and 2.6)

An offset is a predictor whose coefficient is fixed at 1: a part of the rating
plan the GLM must respect but not re-estimate, such as deductible factors
derived from loss elimination ratios or territorial base rates. On the log
link it enters as the log of the factor, and several offsets add. A weight
divides the variance: a row that averages `w` claims has `1/w` of one claim's
variance.

```python
print("deductible offsets:", [round(math.log(f), 4) for f in (1.00, 0.95, 0.90)],
      "  (monograph 0, -0.0513, -0.1054)")
print(f"base loss cost 265 with deductible factor 0.90: {math.log(265) + math.log(0.90):.4f}"
      "   (monograph 5.5797 - 0.1054 = 5.4744)")
```

```text
deductible offsets: [0.0, -0.0513, -0.1054]   (monograph 0, -0.0513, -0.1054)
base loss cost 265 with deductible factor 0.90: 5.4744   (monograph 5.5797 - 0.1054 = 5.4744)
```

Section 2.6 states an equivalence worth checking: a Poisson model of the claim
count with log exposure as the offset and a Poisson model of the frequency
(claims per year) with exposure as the weight give the same coefficients and
standard errors. A Poisson response need not be an integer in a GLM, so
prospicio fits the frequency as given:

```python
c = coding("age")
counts = poisson.fit(c.design(data, offset=log_e), data["claims"])
freq = [k / e for k, e in zip(data["claims"], data["exposure"])]
rates = poisson.fit(c.design(data, weights=data["exposure"]), freq)
print("coefficients agree:", max(abs(a - b) for a, b in zip(counts.coefficients, rates.coefficients)) < 1e-9)
print("standard errors agree:", max(abs(a - b) for a, b in zip(counts.std_errors, rates.std_errors)) < 1e-9)
print("count model:", [f"{b:.4f}" for b in counts.coefficients])
```

```text
coefficients agree: True
standard errors agree: True
count model: ['-0.3636', '-0.0105', '0.0778', '-0.2581', '0.2036']
```

The age coefficient, -0.0105, is close to the true -0.01. Footnote 5 adds that
the equivalence fails for the negative binomial; section 4 below shows why.

An offset carries any factor fixed from outside the same way: had the region
factors been filed and not up for review, `log(exposure) + log(factor)` would
be the offset and the GLM would fit age around them.

## 4. Choosing the distribution (sections 2.7 and 5.2)

**Frequency.** The Poisson fixes `phi = 1`; the overdispersed Poisson
estimates it (prospicio: `dispersion="pearson"`), which leaves the
coefficients alone and scales the standard errors by `sqrt(phi)`. The negative
binomial instead adds a parameter `k` with variance `mu (1 + k mu)`;
prospicio's `"negative_binomial"` takes `theta = 1/k` as an input and does not
estimate it, so the code profiles the log-likelihood over a grid by hand. The
`nb` column was drawn with `theta = 2`:

```python
c = coding("age")
odp = Glm("poisson", "log", dispersion="pearson").fit(c.design(data, offset=log_e), data["claims"])
print(f"overdispersed Poisson: phi {odp.dispersion:.3f}, same coefficients:",
      max(abs(a - b) for a, b in zip(odp.coefficients, counts.coefficients)) < 1e-12,
      f", age se {counts.std_errors[1]:.5f} -> {odp.std_errors[1]:.5f}")
profile = {}
for theta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    nb = Glm("negative_binomial", "log", theta=theta).fit(c.design(data, offset=log_e), data["nb"])
    profile[theta] = nb.log_likelihood
best = max(profile, key=profile.get)
print("negative binomial log-likelihood by theta:", {t: round(v, 1) for t, v in profile.items()})
print("best theta on the grid:", best, "(drawn with 2)")
```

```text
overdispersed Poisson: phi 0.944, same coefficients: True , age se 0.00452 -> 0.00439
negative binomial log-likelihood by theta: {0.5: -932.4, 1.0: -902.6, 1.5: -899.9, 2.0: -902.3, 2.5: -905.9, 3.0: -909.4, 4.0: -915.7}
best theta on the grid: 1.5 (drawn with 2)
```

The Poisson data here are not overdispersed (`phi` is just under 1). Because
`theta` is part of the negative binomial's variance, a count model with an
offset and a frequency model with a weight estimate it differently, which is
footnote 5's caveat.

**Severity.** The gamma (`V = mu^2`) is the usual choice and the inverse
Gaussian (`V = mu^3`) suits more skewed data. Section 6.3.1 decides between
them with deviance residuals: if the family is right they look roughly normal,
and a right-skewed histogram says the data are more skewed than the family
assumes. prospicio's `GlmFit` does not return residuals, but `deviance` with
one row gives each row's unit deviance, so they are a line of code. The `ig`
column was drawn from an inverse Gaussian:

```python
from prospicio.models import deviance


def deviance_residuals(family, y, mu, weights=None, **kw):
    """sign(y - mu) * sqrt(unit deviance), one row at a time."""
    w = weights or [1.0] * len(y)
    return [math.copysign(math.sqrt(max(deviance(family, [a], [m], weights=[wi], **kw), 0.0)), a - m)
            for a, m, wi in zip(y, mu, w)]


def skewness(xs):
    m = sum(xs) / len(xs)
    s2 = sum((x - m) ** 2 for x in xs) / len(xs)
    return sum((x - m) ** 3 for x in xs) / len(xs) / s2 ** 1.5


c = coding("age")
for family in ["gamma", "inverse_gaussian"]:
    fit = Glm(family, "log").fit(c.design(data), data["ig"])
    r = deviance_residuals(family, data["ig"], fit.fitted)
    print(f"{family:16s} AIC {fit.aic:8.1f}, skewness of deviance residuals {skewness(r):+.2f}")
```

```text
gamma            AIC   8240.2, skewness of deviance residuals +0.52
inverse_gaussian AIC   8196.7, skewness of deviance residuals +0.05
```

The gamma leaves right-skewed residuals and the inverse Gaussian brings them
near symmetric, with the lower AIC: the pattern of the monograph's Figures 16
and 17. Severity rows that average several claims carry the claim count as the
weight (section 2.5); `avg_sev` averages `n_sev` claims with true region factors
1.2, 0.8 and 1.5, which a weighted gamma fit recovers:

```python
sev = Glm("gamma", "log").fit(coding("age").design(data, weights=data["n_sev"]), data["avg_sev"])
print("severity factors B, C, D:", [round(math.exp(b), 3) for b in sev.coefficients[2:]],
      f" dispersion {sev.dispersion:.3f}")
```

```text
severity factors B, C, D: [1.268, 0.853, 1.474]  dispersion 0.533
```

**Pure premium.** The Tweedie with power `1 < p < 2` is a Poisson number of
gamma losses, with a point mass at zero. Section 2.7.3 relates its parameters
to the Poisson mean `lambda` and the gamma shape `alpha` and scale `theta`:
mean `lambda alpha theta` (equation 7), power `(alpha + 2) / (alpha + 1)`
(equation 8) and dispersion `lambda^(1-p) (alpha theta)^(2-p) / (2 - p)`
(equation 9). prospicio's `Tweedie.from_poisson_gamma` does the conversion;
the `pure` column was drawn with gamma shape 2 and scale 500, so equation 8
gives `p = 4/3`:

```python
from prospicio.distributions import Tweedie

lam, alpha, scale = 0.4, 2.0, 500.0
tw = Tweedie.from_poisson_gamma(lam, alpha, scale)
p = (alpha + 2) / (alpha + 1)
print(f"mean {tw.mean():.1f} (eq. 7: {lam * alpha * scale:.1f}), power {tw.power:.4f} (eq. 8: {p:.4f}),"
      f" dispersion {tw.dispersion:.3f} (eq. 9: {lam ** (1 - p) * (alpha * scale) ** (2 - p) / (2 - p):.3f})")
```

```text
mean 400.0 (eq. 7: 400.0), power 1.3333 (eq. 8: 1.3333), dispersion 203.581 (eq. 9: 203.581)
```

To pick `p` the monograph suggests estimating it, comparing candidates by
log-likelihood or cross-validation, or taking a customary 1.6 to 1.7. The Rust
crate has a profile-likelihood search (`prospicio_glm::tweedie`) that the
bindings do not expose, so the profile is done by hand: for each `p`, maximize
the GLM's exact Tweedie log-likelihood over `phi`. With the dispersion fixed
(`dispersion=phi`) the coefficients do not change and `log_likelihood` is the
Dunn and Smyth series density summed over the rows, each with dispersion
`phi / w` for exposure weight `w`:

```python
def tweedie_loglik(design, y, power, phi):
    """Exact Tweedie log-likelihood of the GLM refitted with its dispersion fixed at phi."""
    return Glm("tweedie", "log", power=power, dispersion=phi).fit(design, y).log_likelihood


def best_phi(design, y, power, lo=math.log(1.0), hi=math.log(1e4)):
    """Golden-section search for the maximum-likelihood dispersion, on the log scale."""
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if tweedie_loglik(design, y, power, math.exp(a)) > tweedie_loglik(design, y, power, math.exp(b)):
            hi = b
        else:
            lo = a
    phi = math.exp((lo + hi) / 2)
    return phi, tweedie_loglik(design, y, power, phi)


pp = coding("age").design(data, weights=data["exposure"])
for power in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
    phi, ll = best_phi(pp, data["pure"], power)
    print(f"p {power}: log-likelihood {ll:9.1f} at phi {phi:7.1f}")
```

```text
p 1.2: log-likelihood   -1430.1 at phi   329.0
p 1.3: log-likelihood   -1417.0 at phi   212.8
p 1.4: log-likelihood   -1421.0 at phi   137.0
p 1.5: log-likelihood   -1433.4 at phi    89.9
p 1.6: log-likelihood   -1452.3 at phi    61.1
p 1.7: log-likelihood   -1479.2 at phi    44.2
```

The profile peaks near the true 4/3. The monograph's advice (section 5.1.1)
still applies: where the data allow it, separate frequency and severity models
show which effects drive which component, and the Tweedie's constant `phi`
assumes frequency and severity move together. In this portfolio region moves
frequency only, which a pure premium model cannot tell apart from a severity
effect.

## 5. Variable significance and comparing models (sections 2.3 and 6.1 to 6.2)

Each coefficient comes with a standard error and a p-value (prospicio:
`std_errors`, `p_values`, normal for a fixed dispersion and Student's t for an
estimated one). To ask whether a group of parameters earns its place, such as
all levels of a factor, the monograph uses the F-test on nested models
(section 6.2.1): the drop in unscaled deviance divided by the number of added
parameters times the larger model's dispersion, referred to an F distribution
with that many numerator and `n - p` denominator degrees of freedom. Its
example adds a five-level territory (4 parameters) to a 6-parameter model on
972 rows. prospicio has no F distribution, but the F distribution function is
a regularized incomplete beta, `P(F <= f) = I(d1 f / (d1 f + d2); d1/2, d2/2)`,
which prospicio's `Beta` provides:

```python
from prospicio.distributions import Beta


def f_cdf(f, d1, d2):
    return Beta(d1 / 2, d2 / 2).cdf(d1 * f / (d1 * f + d2))


F = (365.8 - 352.1) / (4 * 1.42)
print(f"F = {F:.3f}  (monograph 2.412); percentile on F(4, 962): {f_cdf(F, 4, 972 - 10):.1%}  (monograph 95.2%)")
```

```text
F = 2.412  (monograph 2.412); percentile on F(4, 962): 95.2%  (monograph 95.2%)
```

The same test on our portfolio asks whether region belongs in the claim
frequency model, using the overdispersed Poisson so that `phi` is estimated.
Section 6.2.2 adds the penalized measures AIC (`-2 loglik + 2p`, on `GlmFit`)
and BIC (`-2 loglik + p ln n`, by hand):

```python
quasi = Glm("poisson", "log", dispersion="pearson")
age_only = Terms().intercept().numeric("age").fit(data)
small = quasi.fit(age_only.design(data, offset=log_e), data["claims"])
big = quasi.fit(coding("age").design(data, offset=log_e), data["claims"])
added = len(big.coefficients) - len(small.coefficients)
F = (small.deviance - big.deviance) / (added * big.dispersion)
print(f"region: F = {F:.3f} on ({added}, {big.df_resid:.0f}), percentile {f_cdf(F, added, big.df_resid):.1%}")
for name, model in [("age", age_only), ("age + region", coding("age"))]:
    fit = poisson.fit(model.design(data, offset=log_e), data["claims"])
    k = len(fit.coefficients)
    print(f"{name:13s} deviance {fit.deviance:6.1f}, AIC {fit.aic:6.1f}, BIC {-2 * fit.log_likelihood + k * math.log(n):6.1f}")
```

```text
region: F = 0.970 on (3, 595), percentile 59.4%
age           deviance  445.3, AIC  730.0, BIC  738.8
age + region  deviance  442.6, AIC  733.3, BIC  755.2
```

Region does have an effect in the true model, but with 155 claims the test
cannot see it (the 59th percentile is far short of 95%) and both criteria
prefer the smaller model, BIC more strongly. This is the monograph's caution
that a high p-value is absence of evidence, not evidence of absence; the
choice to keep a rating variable rests on business knowledge too.

## 6. What prospicio does not do (yet)

The monograph's workflow runs end to end in this guide and its second part,
but several steps were done by hand. These are the gaps, with the issues filed for the Models lane:

| Monograph step | In prospicio | Issue |
|---|---|---|
| Interactions, polynomials, hinge functions, bins and natural cubic splines as model terms (5.4, 5.6) | `Terms` has numeric columns and factors only; the columns are built by hand (the R binding takes a formula, so `poly`, `splines::ns` and `a:b` work there) | #350 |
| Residuals (deviance, Pearson, working, randomized quantile), hat values, Cook's distance (6.3, 6.4) | not on `GlmFit`; per-row `deviance` and the working residual by hand | #334 |
| F-test of nested models, BIC, VIF (2.9, 6.2) | by hand; no F distribution (the beta function stands in) | #351 |
| Negative binomial `k` estimated with the model (2.7.2) | `theta` is an input; profiled by hand | #352 |
| Tweedie power by profile likelihood (2.7.3) | in Rust (`prospicio_glm::tweedie`), not in Python or R | #352 |
| Double lift chart, loss ratio chart, ROC curve and AUROC (7.2, 7.3) | `lift` and `gini` only | #351 |
| GLMMs, double GLMs (dispersion modelling), MARS (10.1, 10.2, 10.4) | none; a one-way random effect has a Bayesian counterpart in `prospicio.credibility`, and the double GLM's iterative scheme of section 10.2 can be run with `Glm` weights | #353 |

Beyond the monograph's list, prospicio has GAMs with P-spline, cyclic,
monotone and tensor smooths, elastic nets for every family, sandwich standard
errors (`GlmFit.robust_std_errors`), `predict_distribution` for a fitted GLM's
full predictive distribution, and JSON artifacts for a fitted rating model.
