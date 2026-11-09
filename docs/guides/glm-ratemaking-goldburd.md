# GLMs for insurance rating: Goldburd, Khare, Tevet and Guller with prospicio

<!-- site: execute -->

Mark Goldburd, Anand Khare, Dan Tevet and Dmitriy Guller wrote *Generalized
Linear Models for Insurance Rating* (CAS Monograph 5, second edition, 2025
revision), the CAS text on building a classification rating plan with GLMs:
the log link as a multiplicative plan, frequency, severity and Tweedie models,
the measures that refine and choose between models, and extensions such as GAMs.

This guide follows that workflow in the prospicio Python API
(`prospicio.models`, with `prospicio.distributions` for the Tweedie and the
F-test). The PDF does not contain the insurers' data, so the guide does not
rerun the monograph's coefficient tables (Tables 4 to 12, Figure 9, Table 14) or
its Gini of 56.1%. Where the monograph prints arithmetic that a reader can redo,
the tables show the monograph's value beside prospicio's. The rest runs on a
portfolio that the repository ships.

This is part 1 (chapters 2 to 6.2: the rating plan, offsets and weights, the
choice of distribution, tests of significance). Part 2,
[`glm-ratemaking-goldburd-validation.md`](glm-ratemaking-goldburd-validation.md),
covers continuous variables, interactions, residuals, validation and
regularization (chapters 5.4 to 7 and 10). Section numbers in parentheses are
the monograph's.

## 1. The data

`validation/data/glm_policies.csv` is a synthetic portfolio of 600 policies,
written by `validation/scripts/statsmodels_glm.py`. It has a driver age (18 to
80), a region (A to D, A the most populous and D the thinnest), an exposure in
years and a response for every family. A fixed seed draws them from a known
model, so the guide shows what a GLM recovers and what it misses.

- `claims` is Poisson with mean `exposure * exp(-1 - 0.01 (age - 40)) * r`. The
  region factors are `r = 1.0, 1.2, 0.8, 1.5` for A to D.
- `wavy_claims` is Poisson with a non-linear age effect,
  `exposure * exp(0.5 + 0.8 sin(age / 8))`.
- `avg_sev` is a gamma average of `n_sev` claims.
- `pure` is a compound Poisson-gamma pure premium.

The portfolio drew 155 claims where the true model expects 134 (1.8 standard
deviations more), so fitted base rates run above the true one.

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

A GLM draws the response from an exponential-family distribution with mean `mu`
and dispersion `phi`, with a link function of the mean linear in the
predictors. With the log link, `mu = exp(b0) * exp(b1 x1) * ...`: a base rate
times one relativity for each rating variable. Premiums stay positive and a
surcharge is a percentage.

Section 2.1.3 uses a gamma severity model with intercept 5.8, a driver-age
slope of 0.1, a married coefficient of -0.15 and dispersion 0.3. The variance
is `phi * V(mu)`, and for the gamma `V(mu) = mu^2`, so the coefficient of
variation is `sqrt(phi)` at any mean. `Gamma.from_mean_cv` gives the
variances of section 2.2:

```python
import pandas as pd
from great_tables import GT

from prospicio.distributions import Gamma


def book_table(rows, title, subtitle=None, book="monograph"):
    """Rows of (label, prospicio, book, decimals[, signed]) as a table; a missing book value stays blank."""
    frame = pd.DataFrame([r[:3] for r in rows], columns=["item", "prospicio", book])
    table = GT(frame, rowname_col="item").tab_header(title, subtitle)
    for i, row in enumerate(rows):
        table = table.fmt_number(["prospicio", book], rows=[i], decimals=row[3],
                                 force_sign=len(row) > 4 and row[4])
    return table.sub_missing(missing_text="")


b0, b_age, b_married, phi = 5.8, 0.1, -0.15, 0.3
eta = b0 + b_age * 25 + b_married * 1
factors = [round(math.exp(b0), 2), round(math.exp(b_age * 25), 3), round(math.exp(b_married), 3)]
variance = lambda mean: Gamma.from_mean_cv(mean, math.sqrt(phi)).variance() / 1e6
book_table([
    ("linear predictor, 25, married", eta, 8.15, 2),
    ("severity, 25, married", math.exp(eta), 3463.38, 2),
    ("base rate", factors[0], 330.30, 2),
    ("age factor, 25 years", factors[1], 12.182, 3),
    ("married factor", factors[2], 0.861, 3),
    ("product of the rounded factors", math.prod(factors), 3464.42, 2),
    ("severity, 35, unmarried", math.exp(b0 + b_age * 35), 10938, 0),
    ("variance at mean 3,464, millions", variance(3464.0), 3.6, 1),
    ("variance at mean 10,938, millions", variance(10938.0), 35.9, 1),
], "Gamma severity model of section 2.1.3")
```

```text
                                   prospicio  monograph
linear predictor, 25, married           8.15       8.15
severity, 25, married               3,463.38   3,463.38
base rate                             330.30     330.30
age factor, 25 years                  12.182     12.182
married factor                         0.861      0.861
product of the rounded factors      3,464.42   3,464.42
severity, 35, unmarried               10,938     10,938
variance at mean 3,464, millions         3.6        3.6
variance at mean 10,938, millions       35.9       35.9
```

Both monograph figures are right. The value 3,463.38 is the exact mean. The
value 3,464.42 is the product of factors rounded to the precision of a rating
manual.

A categorical variable enters as one 0/1 column for each level except a base
level. Each coefficient is a log relativity to the base, and its exponential
is a factor (section 2.4.2: SUV 1.23, truck 0.57, van -0.30 against sedan). To
move the base to van (Table 5), subtract the van coefficient from every level.
A logged continuous predictor gives a power curve (section 2.4.1): a
coefficient of 0.62 on log amount of insurance makes $200,000 cost `2^0.62`
times $100,000. A logistic coefficient is a log odds ratio (section 2.8):
0.24 means 27% higher odds.

```python
vtype = {"SUV": 1.23, "truck": 0.57, "van": -0.30}
factor = {k: math.exp(b) for k, b in vtype.items()}
rebased = {"sedan": 0.0 - vtype["van"], "SUV": vtype["SUV"] - vtype["van"], "truck": vtype["truck"] - vtype["van"]}
book_table([
    ("factor, SUV", factor["SUV"], 3.421, 3),
    ("factor, truck", factor["truck"], None, 3),
    ("factor, van", factor["van"], 0.741, 3),
    ("change in %, SUV", 100 * (factor["SUV"] - 1), 242, 0, True),
    ("change in %, van", 100 * (factor["van"] - 1), -25.9, 1, True),
    ("van as base, sedan", rebased["sedan"], 0.30, 2),
    ("van as base, SUV", rebased["SUV"], 1.53, 2),
    ("van as base, truck", rebased["truck"], 0.88, 2),
    ("AOI 200k against 100k", (200 / 100) ** 0.62, 1.54, 2),
    ("odds change in %, coefficient 0.24", 100 * (math.exp(0.24) - 1), 27, 0),
], "Vehicle type, amount of insurance and odds")
```

```text
                                    prospicio  monograph
factor, SUV                             3.421      3.421
factor, truck                           1.768           
factor, van                             0.741      0.741
change in %, SUV                         +242       +242
change in %, van                        -25.9      -25.9
van as base, sedan                       0.30       0.30
van as base, SUV                         1.53       1.53
van as base, truck                       0.87       0.88
AOI 200k against 100k                    1.54       1.54
odds change in %, coefficient 0.24         27         27
```

The rebased truck coefficient is 0.87 from the two-decimal coefficients of
Table 4, and Table 5 prints 0.88. Unrounded estimates of 0.574 and -0.304
would give 0.878. This is rounding, not an error.

The base level changes the standard errors (section 2.4.3). A coefficient
measures a level against the base, so a thin base level makes every comparison
uncertain. Region D has 74 policies and region A has 243. `Terms` takes the
reference level as an argument. The claim count model has the log of exposure
as its offset (section 2.6, next):

```python
def coding(*terms, reference=None):
    """Intercept, numeric columns and the region factor with the given base level."""
    t = Terms().intercept()
    for name in terms:
        t = t.numeric(name)
    return t.factor("region", reference=reference).fit(data)


poisson = Glm("poisson", "log")
estimates = {}
for base in ["A", "D"]:
    fit = poisson.fit(coding("age", reference=base).design(data, offset=log_e), data["claims"])
    estimates[base] = {name[-2]: (b, se) for name, b, se in
                       zip(fit.names[2:], fit.coefficients[2:], fit.std_errors[2:])}
nan = float("nan")
by_base = pd.DataFrame({"region": ["A", "B", "C", "D"]})
for base in ["A", "D"]:
    for k, kind in enumerate(["estimate", "std. error"]):
        by_base[f"{kind} {base}"] = [estimates[base].get(g, (nan, nan))[k] for g in by_base["region"]]
(GT(by_base, rowname_col="region")
 .tab_header("Region coefficients by base level", "Poisson claim counts on age and region, log exposure as offset")
 .tab_spanner("Base A", ["estimate A", "std. error A"])
 .tab_spanner("Base D", ["estimate D", "std. error D"])
 .fmt_number(["estimate A", "estimate D"], decimals=3, force_sign=True)
 .fmt_number(["std. error A", "std. error D"], decimals=3)
 .sub_missing(missing_text=""))
```

```text
   estimate A  std. error A  estimate D  std. error D
A                                -0.204         0.245
B      +0.078         0.192      -0.126         0.253
C      -0.258         0.249      -0.462         0.299
D      +0.204         0.245                          
```

The two fits are the same model. With D as the base, every standard error is
larger, because every coefficient depends on the thinnest level. The true
factors for B, C and D against A are 1.2, 0.8 and 1.5 (log relativities +0.18,
-0.22 and +0.41), and with 155 claims the fit pins down none of them. A GLM
gives the data full credibility and reports uncertainty only in the standard
errors (section 2.10).

The figure shows each log relativity against base A with its 95% interval.
The diamond is the true value:

```python
import matplotlib.pyplot as plt

fit_a = poisson.fit(coding("age").design(data, offset=log_e), data["claims"])
levels = ["B", "C", "D"]
xs = range(len(levels))
fig, ax = plt.subplots()
ax.axhline(0.0, color="0.75")
ax.errorbar([x - 0.06 for x in xs], fit_a.coefficients[2:], yerr=[1.96 * s for s in fit_a.std_errors[2:]],
            fmt="o", capsize=4, label="estimate with 95% interval")
ax.plot([x + 0.06 for x in xs], [math.log(true_r[g]) for g in levels], "D", label="true value")
ax.set_xticks(list(xs), [f"region {g}" for g in levels])
ax.set_ylabel("log relativity against region A")
ax.set_title("Region relativities: every interval covers the truth and zero")
ax.legend()
```

## 3. Offsets and weights (sections 2.5 and 2.6)

An offset is a predictor with its coefficient fixed at 1: a part of the rating
plan that the GLM must respect but not re-estimate, such as deductible factors
or territorial base rates. On the log link the offset is the log of the factor,
and several offsets add. A weight divides the variance: a row that averages
`w` claims has `1/w` of the variance of one claim.

```python
offsets = [(f"offset, deductible factor {f:.2f}", math.log(f), book, 4)
           for f, book in zip((1.00, 0.95, 0.90), (0.0, -0.0513, -0.1054))]
book_table(offsets + [("log base loss cost 265", math.log(265), 5.5797, 4),
                      ("log loss cost, base 265, factor 0.90", math.log(265) + math.log(0.90), 5.4744, 4)],
           "Offsets of section 2.5")
```

```text
                                      prospicio  monograph
offset, deductible factor 1.00           0.0000     0.0000
offset, deductible factor 0.95          -0.0513    -0.0513
offset, deductible factor 0.90          -0.1054    -0.1054
log base loss cost 265                   5.5797     5.5797
log loss cost, base 265, factor 0.90     5.4744     5.4744
```

Section 2.6: a Poisson model of the claim count with log exposure as the
offset and a Poisson model of the frequency (claims per year) with exposure as
the weight give the same coefficients and standard errors. prospicio fits the
non-integer frequency as given:

```python
c = coding("age")
counts = poisson.fit(c.design(data, offset=log_e), data["claims"])
freq = [k / e for k, e in zip(data["claims"], data["exposure"])]
rates = poisson.fit(c.design(data, weights=data["exposure"]), freq)
print("coefficients agree:", max(abs(a - b) for a, b in zip(counts.coefficients, rates.coefficients)) < 1e-9)
print("standard errors agree:", max(abs(a - b) for a, b in zip(counts.std_errors, rates.std_errors)) < 1e-9)
```

```text
coefficients agree: True
standard errors agree: True
```

```python
(GT(pd.DataFrame({"term": counts.names, "coef, offset": counts.coefficients,
                  "coef, weight": rates.coefficients, "se, offset": counts.std_errors,
                  "se, weight": rates.std_errors}), rowname_col="term")
 .tab_header("Claim counts with an offset, frequency with a weight")
 .fmt_number(decimals=4))
```

```text
             coef, offset  coef, weight  se, offset  se, weight
(Intercept)       -0.3636       -0.3636      0.2394      0.2394
age               -0.0105       -0.0105      0.0045      0.0045
region[B]          0.0778        0.0778      0.1921      0.1921
region[C]         -0.2581       -0.2581      0.2490      0.2490
region[D]          0.2036        0.2036      0.2448      0.2448
```

The age coefficient is close to the true value of -0.01. Footnote 5: the
equivalence fails for the negative binomial (section 4).

A fixed external factor works the same way. If the filed region factors are not
under review, the offset is `log(exposure) + log(factor)`.

## 4. Choosing the distribution (sections 2.7 and 5.2)

**Frequency.** The Poisson fixes `phi = 1`. The overdispersed Poisson estimates
`phi` (prospicio: `dispersion="pearson"`), which keeps the coefficients and
scales the standard errors by `sqrt(phi)`. The negative binomial adds a
parameter `k` with variance `mu (1 + k mu)`. The `"negative_binomial"` family
takes `theta = 1/k` as an input. `NegativeBinomialGlm` estimates `theta` with
the coefficients by maximum likelihood (as R's `MASS::glm.nb` does). The `nb`
column was drawn with `theta = 2`:

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

The Poisson data are not overdispersed (`phi` is just under 1). The estimated
`theta` is below the 2 that drew the column: sampling variation in 600
policies. Because the variance contains `theta`, a count model with an offset
and a frequency model with a weight estimate it differently (footnote 5).

**Severity.** The gamma (`V = mu^2`) is the usual choice. The inverse Gaussian
(`V = mu^3`) suits more skewed data. Section 6.3.1 decides between them with
deviance residuals, which look roughly normal if the family is right. A
right-skewed histogram means the data are more skewed than the family assumes. `GlmFit.diagnostics(design, y)` returns the residuals, leverages and
Cook's distances. The `ig` column was drawn from an inverse Gaussian:

```python
def skewness(xs):
    m = sum(xs) / len(xs)
    s2 = sum((x - m) ** 2 for x in xs) / len(xs)
    return sum((x - m) ** 3 for x in xs) / len(xs) / s2 ** 1.5


sev_design = coding("age").design(data)
fits = []
for family in ["gamma", "inverse_gaussian"]:
    fit = Glm(family, "log").fit(sev_design, data["ig"])
    r = fit.diagnostics(sev_design, data["ig"])["deviance"]
    fits.append((family, fit.aic, skewness(r)))
(GT(pd.DataFrame(fits, columns=["family", "AIC", "residual skewness"]), rowname_col="family")
 .tab_header("Severity families on the inverse Gaussian column", "Log link, age and region")
 .fmt_number("AIC", decimals=1)
 .fmt_number("residual skewness", decimals=2, force_sign=True))
```

```text
                      AIC  residual skewness
gamma             8,240.2              +0.52
inverse_gaussian  8,196.7              +0.05
```

The gamma leaves right-skewed residuals. The inverse Gaussian gives nearly
symmetric ones and the lower AIC, the pattern of Figures 16 and 17:

```python
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
for ax, family in zip(axes, ["gamma", "inverse_gaussian"]):
    fit = Glm(family, "log").fit(sev_design, data["ig"])
    r = fit.diagnostics(sev_design, data["ig"])["deviance"]
    sd = (sum(x * x for x in r) / len(r)) ** 0.5
    ax.hist([x / sd for x in r], bins=30)
    ax.set_title(f"{family}: skewness {skewness(r):+.2f}")
    ax.set_xlabel("deviance residual (sd units)")
axes[0].set_ylabel("policies")
```

A severity row that averages several claims carries the claim count as its
weight (section 2.5). A gamma fit weighted by `n_sev` recovers the true region
factors:

```python
sev = Glm("gamma", "log").fit(coding("age").design(data, weights=data["n_sev"]), data["avg_sev"])
print(f"weighted gamma dispersion {sev.dispersion:.3f}")
```

```text
weighted gamma dispersion 0.533
```

```python
(GT(pd.DataFrame({"region": ["B", "C", "D"], "true": [true_r[g] for g in "BCD"],
                  "fitted": [math.exp(b) for b in sev.coefficients[2:]]}), rowname_col="region")
 .tab_header("Severity relativities against region A", "Gamma, claim count as weight")
 .fmt_number(["true", "fitted"], decimals=3))
```

```text
    true  fitted
B  1.200   1.268
C  0.800   0.853
D  1.500   1.474
```

**Pure premium.** The Tweedie with power `1 < p < 2` is a Poisson number of
gamma losses, with a point mass at zero. Section 2.7.3 gives its parameters from
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
book_table([
    ("mean", tw.mean(), lam * alpha * scale, 1),
    ("power", tw.power, p, 4),
    ("dispersion", tw.dispersion, lam ** (1 - p) * (alpha * scale) ** (2 - p) / (2 - p), 3),
], "Tweedie parameters from Poisson-gamma parameters", "Lambda 0.4, alpha 2, scale 500", book="equations 7 to 9")
```

```text
            prospicio  equations 7 to 9
mean            400.0             400.0
power          1.3333            1.3333
dispersion    203.581           203.581
```

The monograph picks `p` by estimating it, by comparing candidates on
log-likelihood or cross-validation, or by a customary 1.6 to 1.7.
`tweedie_profile` does the first: for each `p` it maximizes the exact Tweedie
log-likelihood over `phi` (the Dunn and Smyth series density, with dispersion
`phi / w` for exposure weight `w`). `TweedieGlm` estimates `p` directly, with a
95% profile-likelihood interval:

```python
from prospicio.models import TweedieGlm, tweedie_profile

pp = coding("age").design(data, weights=data["exposure"])
profile = tweedie_profile(pp, data["pure"], [1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
tw_fit = TweedieGlm().fit(pp, data["pure"])
lo, hi = tw_fit.interval
print(f"estimated p {tw_fit.power:.3f} (95% interval {lo:.3f} to {hi:.3f}), phi {tw_fit.dispersion:.1f}")
```

```text
estimated p 1.314 (95% interval 1.264 to 1.371), phi 200.1
```

```python
profile_table = pd.DataFrame({"p": profile["powers"], "log-likelihood": profile["log_likelihood"],
                              "dispersion phi": profile["dispersion"]})
(GT(profile_table, rowname_col="p")
 .tab_header("Tweedie profile likelihood by power", "Pure premium on age and region, exposure as weight")
 .fmt_number(["log-likelihood", "dispersion phi"], decimals=1))
```

```text
     log-likelihood  dispersion phi
1.2        -1,430.1           329.0
1.3        -1,417.0           212.8
1.4        -1,421.0           137.0
1.5        -1,433.4            89.9
1.6        -1,452.3            61.1
1.7        -1,479.2            44.2
```

The profile peaks near the true value of 4/3, inside the interval. The
monograph advises separate frequency and severity models where the data allow
(section 5.1.1), because the Tweedie's constant `phi` assumes that frequency and
severity move together. Here region moves frequency only, and a pure premium
model cannot tell this from a severity effect.

## 5. Variable significance and comparing models (sections 2.3 and 6.1 to 6.2)

Each coefficient has a standard error and a p-value (prospicio: `std_errors`,
`p_values`), normal for a fixed dispersion and Student's t for an estimated one.
To test a group of parameters, such as all levels of a factor, the monograph
uses the F-test on nested models (section 6.2.1). The statistic is the drop in
unscaled deviance divided by the number of added parameters times the
dispersion of the larger model. It refers to an F distribution with that many
numerator degrees of freedom and `n - p` denominator degrees of freedom.

The monograph's example adds a five-level territory (4 parameters) to a
6-parameter model on 972 rows. For two fitted models, `f_test` does the whole
test (as R's `anova(small, big, test = "F")` does). The monograph prints only
the deviances, so the code uses the F distribution function, a regularized
incomplete beta, `P(F <= f) = I(d1 f / (d1 f + d2); d1/2, d2/2)`, from `Beta`:

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

The same test asks whether region belongs in the claim frequency model. It
uses the overdispersed Poisson, so that `f_test` estimates `phi`. Section 6.2.2
adds AIC (`-2 loglik + 2p`) and BIC (`-2 loglik + p ln n`), both on `GlmFit`:

```python
from prospicio.models import f_test

quasi = Glm("poisson", "log", dispersion="pearson")
age_only = Terms().intercept().numeric("age").fit(data)
small = quasi.fit(age_only.design(data, offset=log_e), data["claims"])
big = quasi.fit(coding("age").design(data, offset=log_e), data["claims"])
test = f_test(small, big)
print(f"region: F = {test['statistic']:.3f} on ({test['df']:.0f}, {test['df_resid']:.0f}),"
      f" percentile {1 - test['p_value']:.1%}")
```

```text
region: F = 0.970 on (3, 595), percentile 59.4%
```

```python
scores = []
for name, model in [("age", age_only), ("age + region", coding("age"))]:
    fit = poisson.fit(model.design(data, offset=log_e), data["claims"])
    scores.append((name, fit.deviance, fit.aic, fit.bic))
(GT(pd.DataFrame(scores, columns=["model", "deviance", "AIC", "BIC"]), rowname_col="model")
 .tab_header("Claim frequency models", "Poisson, log exposure as offset")
 .fmt_number(["deviance", "AIC", "BIC"], decimals=1))
```

```text
              deviance    AIC    BIC
age              445.3  730.0  738.8
age + region     442.6  733.3  755.2
```

Region has an effect in the true model, but with 155 claims the test cannot see
it (the 59th percentile is far short of 95%). Both criteria prefer the smaller
model, BIC more strongly. As the monograph cautions, a high p-value is absence
of evidence, not evidence of absence, and business knowledge also decides
whether to keep a variable.

## 6. Where each step lives in prospicio

The monograph's steps in prospicio, including steps that the guides only
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
