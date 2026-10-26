# GLMs for insurance rating: Goldburd, Khare, Tevet and Guller with prospicio

Mark Goldburd, Anand Khare, Dan Tevet and Dmitriy Guller wrote *Generalized
Linear Models for Insurance Rating* (CAS Monograph 5, second edition, 2025
revision). It is the CAS text on how to build a classification rating plan
with GLMs. It covers:

- the log link as a multiplicative plan,
- offsets and weights,
- frequency, severity and Tweedie pure premium models,
- continuous variables and interactions,
- the measures that refine a model (deviance, F-tests, AIC, residuals),
- the measures that choose between finished models (lift, double lift and
  loss ratio charts, the Gini index),
- extensions such as GAMs and elastic nets.

This guide follows that workflow in the prospicio Python API
(`prospicio.models`, with `prospicio.distributions` for the Tweedie and the
F-test). The PDF does not contain the insurers' data behind the examples of the
monograph. Thus the guide does not rerun its coefficient tables (Tables 4 to 12,
Figure 9, Table 14) or its Gini of 56.1%. Where the monograph prints arithmetic
that a reader can redo from the printed values, the code recomputes it and
prints the monograph's value beside prospicio's. This covers rating factors,
offsets, the F-statistic, the reparameterization of an interaction, the Tweedie
parameters and the confusion matrix rates. The rest of the workflow runs on a
portfolio that the repository ships.

This is the first of two parts. It covers the GLM as a rating plan, offsets and
weights, the choice of distribution and the tests of significance (chapters 2
to 6.2). The second part,
[`glm-ratemaking-goldburd-validation.md`](glm-ratemaking-goldburd-validation.md),
covers continuous variables, interactions, residuals, validation on held-out
data and regularization (chapters 5.4 to 7 and 10). The text and exhibits are
the authors'. This guide explains them in new words. Section numbers in
parentheses are the monograph's. Each part runs in about a second.

## 1. The data

`validation/data/glm_policies.csv` is a synthetic portfolio of 600 policies.
`validation/scripts/statsmodels_glm.py` writes it for the GLM parity tests. It
has a driver age (18 to 80), a region (A to D), an exposure in years and a
response for every family. Region A is the most populous and region D is the
thinnest. A fixed seed draws all of them from a known model. The drawing model
sets the true relativities, so the guide can show what a GLM recovers and what
it misses.

- `claims` is Poisson with mean `exposure * exp(-1 - 0.01 (age - 40)) * r`. The
  region factors are `r = 1.0, 1.2, 0.8, 1.5` for A to D.
- `wavy_claims` is Poisson with a non-linear age effect,
  `exposure * exp(0.5 + 0.8 sin(age / 8))`.
- `avg_sev` is a gamma average of `n_sev` claims.
- `pure` is a compound Poisson-gamma pure premium.

The portfolio drew 155 claims where the true model expects 134. This is about
1.8 standard deviations more. It shows below as a base rate above the true one.

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

A GLM says that the response comes from an exponential-family distribution with
mean `mu` and dispersion `phi`. A link function of the mean is linear in the
predictors. With the log link, the linear predictor becomes a product of
factors: `mu = exp(b0) * exp(b1 x1) * ...`. This is a base rate times one
relativity for each rating variable. The monograph uses the log link as the
default for rating because premiums stay positive and a surcharge is a
percentage, not a dollar amount.

Section 2.1.3 works through a gamma severity model. It has intercept 5.8, a
driver-age slope of 0.1, a married coefficient of -0.15 and dispersion 0.3. The
variance of a GLM response is `phi * V(mu)`. For the gamma, `V(mu) = mu^2`, so a
gamma with dispersion `phi` has coefficient of variation `sqrt(phi)` for any
mean. The code uses `Gamma.from_mean_cv` to show the variances of section 2.2:

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

Both monograph figures for the 25-year-old are right. The value 3,463.38 is the
exact mean. The value 3,464.42 is the product of the factors rounded to the
precision of a rating manual. This is the monograph's point about rounding.

A categorical variable enters as one 0/1 column for each level except a base
level. Each coefficient is a log relativity to the base. Exponentiating a
coefficient gives a factor. The vehicle type coefficients of section 2.4.2 (SUV
1.23, truck 0.57, van -0.30 against sedan) show this. To move the base to van
(Table 5), subtract the van coefficient from every level. A logged continuous
predictor gives a power curve (section 2.4.1). A coefficient of 0.62 on log
amount of insurance makes $200,000 cost `2^0.62` times $100,000. The
coefficients of a logistic model are log odds ratios (section 2.8). A
coefficient of 0.24 means 27% higher odds:

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

The rebased truck coefficient is 0.87 from the two-decimal coefficients of
Table 4, and Table 5 prints 0.88. The unrounded estimates behind Table 4 must
differ by 0.875 or more (0.574 and -0.304 would do). This is rounding, not an
error.

The base level changes the standard errors (section 2.4.3). A coefficient
measures a level against the base, so a thin base level makes every comparison
uncertain. In this portfolio, region D has 74 policies and region A has 243.
`Terms` builds the design in treatment coding and takes the reference level as
an argument. The claim count model has the log of exposure as its offset
(section 2.6, next):

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

The two fits are the same model. D against A is +0.204 in the first fit, and A
against D is -0.204 in the second. The predictions agree. With D as the base,
every standard error is larger (0.25 to 0.30 against 0.19 to 0.25). Every
coefficient now depends on the thinnest level. The true factors for B, C
and D against A are 1.2, 0.8 and 1.5. These are log relativities of +0.18, -0.22
and +0.41. With 155 claims, the fit estimates none of the three sharply. The
monograph makes a related point: a GLM gives the data full credibility and
reports the uncertainty only in the standard errors (section 2.10).

## 3. Offsets and weights (sections 2.5 and 2.6)

An offset is a predictor with a coefficient fixed at 1. It is a part of the
rating plan that the GLM must respect but not re-estimate. Examples are
deductible factors from loss elimination ratios and territorial base rates. On
the log link, the offset is the log of the factor. Several offsets add. A weight
divides the variance: a row that averages `w` claims has `1/w` of the variance
of one claim.

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

Section 2.6 states an equivalence. Fit two Poisson models. One models the claim
count with log exposure as the offset. The other models the frequency (claims
per year) with exposure as the weight. They give the same coefficients and
standard errors. A Poisson response does not have to be an integer in a GLM, so
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

The age coefficient, -0.0105, is close to the true value of -0.01. Footnote 5
adds that the equivalence fails for the negative binomial. Section 4 below shows
why.

An offset carries any externally fixed factor in the same way. Suppose the
filed region factors are not under review. The offset would then be
`log(exposure) + log(factor)`, and the GLM would fit age around the factors.

## 4. Choosing the distribution (sections 2.7 and 5.2)

**Frequency.** The Poisson fixes `phi = 1`. The overdispersed Poisson estimates
`phi` (prospicio: `dispersion="pearson"`). This leaves the coefficients
unchanged and scales the standard errors by `sqrt(phi)`. The negative binomial
adds a parameter `k` with variance `mu (1 + k mu)`. The `"negative_binomial"`
family takes `theta = 1/k` as an input. `NegativeBinomialGlm` estimates `theta`
with the coefficients by maximum likelihood (as R's `MASS::glm.nb` does). The
`nb` column was drawn with `theta = 2`:

```python
from prospicio.models import NegativeBinomialGlm

c = coding("age")
odp = Glm("poisson", "log", dispersion="pearson").fit(c.design(data, offset=log_e), data["claims"])
print(f"overdispersed Poisson: phi {odp.dispersion:.3f}, same coefficients:",
      max(abs(a - b) for a, b in zip(odp.coefficients, counts.coefficients)) < 1e-12,
      f", age se {counts.std_errors[1]:.5f} -> {odp.std_errors[1]:.5f}")
nb = NegativeBinomialGlm().fit(c.design(data, offset=log_e), data["nb"])
print(f"negative binomial: theta {nb.theta:.2f} (se {nb.se_theta:.2f}, drawn with 2),"
      f" log-likelihood {nb.log_likelihood:.1f}")
```

```text
overdispersed Poisson: phi 0.944, same coefficients: True , age se 0.00452 -> 0.00439
negative binomial: theta 1.40 (se 0.21, drawn with 2), log-likelihood -899.8
```

The Poisson data here are not overdispersed (`phi` is just under 1). The
estimated `theta`, 1.40, is below the value 2 that drew the column. This is the
sampling variation of one portfolio of 600 policies. The variance of the
negative binomial contains `theta`. Thus a count model with an offset and a
frequency model with a weight estimate `theta` differently. This is the caveat
of footnote 5.

**Severity.** The gamma (`V = mu^2`) is the usual choice. The inverse Gaussian
(`V = mu^3`) suits more skewed data. Section 6.3.1 decides between them with
deviance residuals. If the family is right, the residuals look roughly normal. A
right-skewed histogram shows that the data are more skewed than the family
assumes. `GlmFit.diagnostics(design, y)` returns the deviance residuals, the
Pearson and working residuals, the leverages and Cook's distances. The `ig`
column was drawn from an inverse Gaussian:

```python
def skewness(xs):
    m = sum(xs) / len(xs)
    s2 = sum((x - m) ** 2 for x in xs) / len(xs)
    return sum((x - m) ** 3 for x in xs) / len(xs) / s2 ** 1.5


sev_design = coding("age").design(data)
for family in ["gamma", "inverse_gaussian"]:
    fit = Glm(family, "log").fit(sev_design, data["ig"])
    r = fit.diagnostics(sev_design, data["ig"])["deviance"]
    print(f"{family:16s} AIC {fit.aic:8.1f}, skewness of deviance residuals {skewness(r):+.2f}")
```

```text
gamma            AIC   8240.2, skewness of deviance residuals +0.52
inverse_gaussian AIC   8196.7, skewness of deviance residuals +0.05
```

The gamma leaves right-skewed residuals. The inverse Gaussian gives nearly
symmetric residuals and the lower AIC. This is the pattern of Figures 16 and 17
in the monograph. A severity row that averages several claims carries the claim
count as its weight (section 2.5). The `avg_sev` column averages `n_sev` claims.
Its true region factors are 1.2, 0.8 and 1.5. A weighted gamma fit recovers
them:

```python
sev = Glm("gamma", "log").fit(coding("age").design(data, weights=data["n_sev"]), data["avg_sev"])
print("severity factors B, C, D:", [round(math.exp(b), 3) for b in sev.coefficients[2:]],
      f" dispersion {sev.dispersion:.3f}")
```

```text
severity factors B, C, D: [1.268, 0.853, 1.474]  dispersion 0.533
```

**Pure premium.** The Tweedie with power `1 < p < 2` is a Poisson number of
gamma losses, with a point mass at zero. Section 2.7.3 relates its parameters to
the Poisson mean `lambda`, the gamma shape `alpha` and the gamma scale `theta`:

- mean: `lambda alpha theta` (equation 7),
- power: `(alpha + 2) / (alpha + 1)` (equation 8),
- dispersion: `lambda^(1-p) (alpha theta)^(2-p) / (2 - p)` (equation 9).

`Tweedie.from_poisson_gamma` does the conversion. The `pure` column was drawn
with gamma shape 2 and scale 500, so equation 8 gives `p = 4/3`:

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

The monograph gives three ways to pick `p`: estimate it, compare candidates by
log-likelihood or cross-validation, or take a customary 1.6 to 1.7.
`tweedie_profile` does the first. For each `p`, it maximizes the exact Tweedie
log-likelihood of the GLM over `phi`. This is the Dunn and Smyth series density,
summed over the rows, with dispersion `phi / w` for exposure weight `w`.
`TweedieGlm` estimates `p` directly, with a 95% profile-likelihood interval:

```python
from prospicio.models import TweedieGlm, tweedie_profile

pp = coding("age").design(data, weights=data["exposure"])
profile = tweedie_profile(pp, data["pure"], [1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
for power, ll, phi in zip(profile["powers"], profile["log_likelihood"], profile["dispersion"]):
    print(f"p {power}: log-likelihood {ll:9.1f} at phi {phi:7.1f}")
tw_fit = TweedieGlm().fit(pp, data["pure"])
lo, hi = tw_fit.interval
print(f"estimated p {tw_fit.power:.3f} (95% interval {lo:.3f} to {hi:.3f}), phi {tw_fit.dispersion:.1f}")
```

```text
p 1.2: log-likelihood   -1430.1 at phi   329.0
p 1.3: log-likelihood   -1417.0 at phi   212.8
p 1.4: log-likelihood   -1421.0 at phi   137.0
p 1.5: log-likelihood   -1433.4 at phi    89.9
p 1.6: log-likelihood   -1452.3 at phi    61.1
p 1.7: log-likelihood   -1479.2 at phi    44.2
estimated p 1.314 (95% interval 1.264 to 1.371), phi 200.1
```

The profile peaks near the true value of 4/3, which lies inside the interval of
the estimate. The advice of the monograph (section 5.1.1) still applies. Where
the data allow it, separate frequency and severity models show which effects
drive which component. The constant `phi` of the Tweedie assumes that frequency
and severity move together. In this portfolio, region moves frequency only. A
pure premium model cannot tell this apart from a severity effect.

## 5. Variable significance and comparing models (sections 2.3 and 6.1 to 6.2)

Each coefficient has a standard error and a p-value (prospicio: `std_errors`,
`p_values`). The p-value is normal for a fixed dispersion and Student's t for an
estimated one. To test a group of parameters, such as all levels of a factor,
the monograph uses the F-test on nested models (section 6.2.1). The statistic is the drop in unscaled deviance, divided by the number of
added parameters times the dispersion of the larger model. The reference is an F
distribution with that number of numerator degrees of freedom and `n - p`
denominator degrees of freedom.

The example of the monograph adds a five-level territory (4 parameters) to a
6-parameter model on 972 rows. For two fitted models, `f_test` does the whole
test (as R's `anova(small, big, test = "F")` does). The monograph prints only
the deviances, so the code refers the statistic to the F distribution directly.
The F distribution function is a regularized incomplete beta,
`P(F <= f) = I(d1 f / (d1 f + d2); d1/2, d2/2)`. The prospicio `Beta` provides
it:

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

The same test on this portfolio asks whether region belongs in the claim
frequency model. It uses the overdispersed Poisson, so that `f_test` estimates
`phi`. Section 6.2.2 adds the penalized measures AIC (`-2 loglik + 2p`) and BIC
(`-2 loglik + p ln n`). Both are on `GlmFit`:

```python
from prospicio.models import f_test

quasi = Glm("poisson", "log", dispersion="pearson")
age_only = Terms().intercept().numeric("age").fit(data)
small = quasi.fit(age_only.design(data, offset=log_e), data["claims"])
big = quasi.fit(coding("age").design(data, offset=log_e), data["claims"])
test = f_test(small, big)
print(f"region: F = {test['statistic']:.3f} on ({test['df']:.0f}, {test['df_resid']:.0f}),"
      f" percentile {1 - test['p_value']:.1%}")
for name, model in [("age", age_only), ("age + region", coding("age"))]:
    fit = poisson.fit(model.design(data, offset=log_e), data["claims"])
    print(f"{name:13s} deviance {fit.deviance:6.1f}, AIC {fit.aic:6.1f}, BIC {fit.bic:6.1f}")
```

```text
region: F = 0.970 on (3, 595), percentile 59.4%
age           deviance  445.3, AIC  730.0, BIC  738.8
age + region  deviance  442.6, AIC  733.3, BIC  755.2
```

Region has an effect in the true model, but with 155 claims the test cannot see
it (the 59th percentile is far short of 95%). Both criteria prefer the smaller
model, and BIC prefers it more strongly. This agrees with the caution of the
monograph: a high p-value is absence of evidence, not evidence of absence. The
choice to keep a rating variable also rests on business knowledge.

## 6. Where each step lives in prospicio

This guide and its second part run the workflow of the monograph from start to
end. The table maps the steps to prospicio, including steps that the guides only
mention:

| Monograph step | In prospicio |
|---|---|
| Interactions, polynomials, hinge functions, bins and natural cubic splines as model terms (5.4, 5.6) | `Terms.interaction`, `poly`, `hinges`, `bins` and `natural_spline` (the R binding takes a formula, so `poly`, `splines::ns` and `a:b` work there) |
| Residuals (deviance, Pearson, working, randomized quantile), hat values, Cook's distance (6.3, 6.4) | `GlmFit.diagnostics` and `GlmFit.quantile_residuals` |
| F-test of nested models, BIC, VIF (2.9, 6.2) | `f_test`, `likelihood_ratio_test`, `GlmFit.bic` and `vif` |
| Negative binomial `k` estimated with the model (2.7.2) | `NegativeBinomialGlm` |
| Tweedie power by profile likelihood (2.7.3) | `TweedieGlm` and `tweedie_profile` |
| Double lift chart, loss ratio chart, ROC curve and AUROC (7.2, 7.3) | `double_lift`, `loss_ratio_chart`, `roc`, `confusion_matrix` and `normalized_gini` |
| GLMMs, double GLMs (dispersion modelling), MARS (10.1, 10.2, 10.4) | `Glmm` (and `BayesGlmm`), `DoubleGlm` and `mars` |

prospicio also has GAMs with P-spline, cyclic, monotone and tensor smooths, and
elastic nets for every family. It has sandwich standard errors
(`GlmFit.robust_std_errors`), `predict_distribution` for the full predictive
distribution of a fitted GLM, and JSON artifacts for a fitted rating model.
