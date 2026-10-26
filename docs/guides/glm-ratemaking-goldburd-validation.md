# GLMs for insurance rating, part 2: refinement and validation (Goldburd, Khare, Tevet and Guller)

This is the second part of a guide to *Generalized Linear Models for Insurance
Rating* (CAS Monograph 5, second edition, 2025 revision). The authors are
Goldburd, Khare, Tevet and Guller. The guide uses the prospicio Python API. The
first part, [`glm-ratemaking-goldburd.md`](glm-ratemaking-goldburd.md), explains the GLM as
a multiplicative rating plan, offsets and weights. It also explains the choice
of distribution and the F-test, and it maps the steps of the monograph to
prospicio.

This part follows the model refinement and validation of the monograph:

- continuous variables (section 5.4),
- interactions (5.6),
- deviance and residuals (6.1 and 6.3),
- validation on held-out data with lift, double lift and loss ratio charts, the
  Gini index and the ROC curve (4.3 and 7),
- stability and the elastic net (6.4 and 10.5).

The PDF does not contain the data behind the examples of the monograph. Thus
the guide recomputes the printed arithmetic where it can. The workflow runs on
the synthetic portfolio of the repository, `validation/data/glm_policies.csv`.
The text and exhibits are the authors'. This guide explains them in new words.
Section numbers in parentheses are the monograph's. The whole part runs in about
a second.

The setup repeats the first part. It has the 600 policies and the log of
exposure as the claim count offset. It also has a helper for the region factor,
and the Poisson and overdispersed Poisson families. This part uses `claims`. It is Poisson with a
log-linear age effect and region factors 1.0, 1.2, 0.8 and 1.5 for A to D. This
part also uses `wavy_claims`, a Poisson count with the non-linear true age effect
`0.8 sin(age / 8)`:

```python
import csv
import math
from pathlib import Path

from prospicio.models import Design, Glm, Terms

rows = list(csv.DictReader(open(Path("../../validation/data/glm_policies.csv"))))
data = {k: [float(r[k]) for r in rows] for k in ["age", "exposure", "claims", "trials", "n_sev", "wavy_claims"]}
data["region"] = [r["region"] for r in rows]
n = len(rows)
log_e = [math.log(e) for e in data["exposure"]]


def coding(*terms, reference=None):
    """Intercept, numeric columns and the region factor with the given base level."""
    t = Terms().intercept()
    for name in terms:
        t = t.numeric(name)
    return t.factor("region", reference=reference).fit(data)


poisson = Glm("poisson", "log")
quasi = Glm("poisson", "log", dispersion="pearson")
age_only = Terms().intercept().numeric("age").fit(data)
big = quasi.fit(coding("age").design(data, offset=log_e), data["claims"])
print(n, "policies,", int(sum(data["claims"])), "claims,", int(sum(data["wavy_claims"])), "wavy_claims")
```

```text
600 policies, 155 claims, 693 wavy_claims
```

## 1. Continuous variables: partial residuals, bins, polynomials, hinges and splines (section 5.4)

A continuous predictor enters linearly on the link scale. To see whether this is
enough, section 5.4.1 plots partial residuals against `x`. A partial residual is
the working residual `(y - mu) / mu` (log link) plus the own term `b x` of the
predictor. A plot of thousands of points hides the pattern, so the code bins the
points and averages each bin with the working weights. For a Poisson on the log
link, the working weight is `mu` times the prior weight (section 6.3.2).
`GlmFit.diagnostics` returns both. The code does this for the `wavy_claims`
count, whose true age effect is `0.8 sin(age / 8)`, fitted with a straight line
in age:

```python
order = sorted(range(n), key=lambda i: data["age"][i])
y = data["wavy_claims"]
linear = poisson.fit(age_only.design(data, offset=log_e), y)
b_age = linear.coefficients[1]
diag = linear.diagnostics(age_only.design(data, offset=log_e), y)
partial = [r + b_age * a for r, a in zip(diag["working"], data["age"])]
print(f"linear age slope {b_age:+.4f}")
print("age band   mean age  binned partial residual  fitted b*age  true 0.8 sin(age/8), centred")
true_c = sum(0.8 * math.sin(a / 8) for a in data["age"]) / n
for k in range(8):
    idx = order[k * n // 8:(k + 1) * n // 8]
    ww = [diag["working_weights"][i] for i in idx]
    pr = sum(partial[i] * w for i, w in zip(idx, ww)) / sum(ww)
    age = sum(data["age"][i] * w for i, w in zip(idx, ww)) / sum(ww)
    lo, hi = data["age"][idx[0]], data["age"][idx[-1]]
    print(f"{lo:4.1f}-{hi:4.1f}  {age:8.1f}  {pr:23.3f}  {b_age * age:12.3f}  {0.8 * math.sin(age / 8) - true_c:+.3f}")
```

```text
linear age slope +0.0121
age band   mean age  binned partial residual  fitted b*age  true 0.8 sin(age/8), centred
18.1-24.7      21.6                    0.840         0.263  +0.315
24.8-30.6      28.2                    0.307         0.341  -0.318
30.7-38.3      34.3                   -0.106         0.416  -0.750
38.5-46.8      42.9                    0.005         0.520  -0.660
47.0-56.7      52.0                    0.747         0.631  +0.152
56.8-62.9      59.7                    1.216         0.724  +0.716
63.0-71.4      67.6                    1.117         0.820  +0.640
71.5-79.9      76.1                    0.603         0.922  -0.088
```

The fitted line rises gently. The binned partial residuals start high, fall to a
trough in the thirties and forties, peak around 60 and fall again. This is the
true sine, shifted by a constant of about 0.6 (partial residuals carry the share
of the intercept in the fit). The monograph offers four fixes, and prospicio
fits all of them. `Terms` builds the bins, polynomials and hinges as model
terms:

- **bins** (5.4.2): a factor of age bands of about equal counts. Each band
  floats freely. The cost is many parameters and possible reversals.
- **polynomials** (5.4.3): powers of a centred and scaled age. They bend, but
  they misbehave at the edges.
- **hinge functions** (5.4.4): `max(0, age - knot)` terms. The slope changes at
  chosen knots. The partial residual plot shows the knots (here the trough near
  40 and the peak near 60: knots at 38 and 62).
- **splines** (5.4.5 and 10.3): the `Gam` of prospicio fits a penalized P-spline
  with the smoothness chosen by UBRE. This is the GAM of section 10.3. It
  replaces the natural cubic spline (`Terms.natural_spline`).

Each fit has three scores. The first is the training deviance. The second is
the AIC (with the effective degrees of freedom of the GAM). The third is the
mean Poisson deviance on held-out folds from `cross_validate` with five folds. `k_fold` gives the same folds in Python
and R:

```python
from prospicio.models import Gam, cross_validate, deviance, deviance_score, k_fold

ages = data["age"]
cuts = [sorted(ages)[k * n // 8] for k in range(1, 8)]
band = [str(sum(a >= c for c in cuts)) for a in ages]
cols = {"age": ages, "band": band, "z": [(a - 50) / 20 for a in ages]}
terms = {
    "linear": Terms().intercept().numeric("age"),
    "8 bins": Terms().intercept().factor("band"),
    "cubic": Terms().intercept().poly("z", 3),
    "hinges 38, 62": Terms().intercept().numeric("age").hinges("age", [38, 62]),
}
designs = {name: t.fit(cols).design(cols, offset=log_e) for name, t in terms.items()}
folds = k_fold(n, 5, 2025)
score = deviance_score("poisson")
print("model           parameters  deviance     AIC  CV deviance")
for name, d in designs.items():
    fit = poisson.fit(d, y)
    cv = sum(cross_validate(poisson, d, y, folds, score)) / len(folds)
    print(f"{name:14s} {len(fit.coefficients):11d} {fit.deviance:9.1f} {fit.aic:7.1f} {cv:12.4f}")
gam = Gam(poisson, [("age", 10)])
gd = designs["linear"]
gfit = gam.fit(gd, y)
ll = sum(yi * math.log(m) - m - math.lgamma(yi + 1) for yi, m in zip(y, gfit.fitted))
cv = sum(cross_validate(gam, gd, y, folds, score)) / len(folds)
print(f"{'P-spline GAM':14s} {gfit.edf:11.1f} {gfit.deviance:9.1f} {-2 * ll + 2 * gfit.edf:7.1f} {cv:12.4f}")
```

```text
model           parameters  deviance     AIC  CV deviance
linear                   2     769.9  1673.9       1.2923
8 bins                   8     658.6  1574.6       1.1338
cubic                    4     635.5  1543.5       1.0730
hinges 38, 62            4     624.0  1532.0       1.0534
P-spline GAM           5.8     621.0  1532.7       1.0612
```

All four non-linear forms beat the straight line by a wide margin. The hinges,
with knots read off the partial residual table, do best out of sample, just
ahead of the GAM. The GAM found its shape without being told where to bend. The
cubic is close behind. The bins pay for their extra parameters with a step
function that ignores the variation inside each band.

The building-age example of the monograph reads the same way from its printed
coefficients. Table 6 has a slope of 1.225 below the knot at log age 2.75 and
1.225 - 2.269 above it. Table 7 adds a knot at 3.6 with coefficient 1.170. The
quadratic of Figure 10 (4.749 x - 0.866 x^2) peaks where its derivative is zero,
at `x = 4.749 / (2 x 0.866)`. This is close to the knot that the authors chose by
eye. It is also close to the peak of the GAM at about `e^2.8`, 16 years (section
10.3):

```python
print(f"Table 6 slopes: {1.225:.3f} then {1.225 - 2.269:.3f}   (monograph 1.225 and -1.044)")
print(f"Table 7 slopes: {1.289:.3f}, {1.289 - 2.472:.3f}, {1.289 - 2.472 + 1.170:.3f}")
peak = 4.749 / (2 * 0.866)
print(f"quadratic peak at log age {peak:.2f}, about {math.exp(peak):.0f} years; GAM peak e^2.8 = {math.exp(2.8):.0f} years")
```

```text
Table 6 slopes: 1.225 then -1.044   (monograph 1.225 and -1.044)
Table 7 slopes: 1.289, -1.183, -0.013
quadratic peak at log age 2.74, about 16 years; GAM peak e^2.8 = 16 years
```

With the second knot, the slope past log age 3.6 is almost flat. This is the
"leveling off" that the monograph describes.

## 2. Interactions (section 5.6)

An interaction lets the relativity of one variable depend on the level of
another. In a multiplicative plan, it is one more factor, which applies only to
the combination. The two-by-two example of Figure 13 has main factors 2.0 and
1.5 and an interaction factor of 1.1 on the BY cell. Table 9 gives occupancy
class 2 a sprinkler factor from the sum of the main and interaction
coefficients:

```python
base, f_b, f_y, f_by = 10.0, 2.0, 1.5, 1.1
print("cells X/A, X/B, Y/A, Y/B:", [base, base * f_b, base * f_y, round(base * f_b * f_y * f_by, 1)],
      "  (monograph 10, 20, 15, 33)")
print(f"sprinklered discount, main effects only: {math.exp(-0.3046) - 1:.1%}   (monograph -26.3%)")
print(f"class 1 with the interaction: {math.exp(-0.2895) - 1:.1%}   (monograph -25.1%)")
print(f"class 2: factor {math.exp(-0.2895 - 0.2847):.3f}   (monograph 0.563, a 43.7% discount)")
```

```text
cells X/A, X/B, Y/A, Y/B: [10.0, 20.0, 15.0, 33.0]   (monograph 10, 20, 15, 33)
sprinklered discount, main effects only: -26.3%   (monograph -26.3%)
class 1 with the interaction: -25.1%   (monograph -25.1%)
class 2: factor 0.563   (monograph 0.563, a 43.7% discount)
```

An interaction between a factor and a continuous variable shifts the slope. The
main effect then refers to the point where the continuous variable is zero. In
Table 11, that point is log AOI = 0, an amount of $1. This is why the sprinkler
main effect turns positive. Table 12 refits with `log(AOI / 200,000)`, so the
main effect is the sprinkler relativity at the base amount. The two tables are
the same model. The coefficients of Table 12 follow from Table 11: each
intercept-like term picks up its slope times `ln 200,000`:

```python
shift = math.log(200_000)
print(f"intercept: {-8.9456 + 0.4239 * shift:.4f}   (monograph Table 12: -3.7710)")
print(f"sprinklered: {0.7447 - 0.1032 * shift:.4f}   (monograph Table 12: -0.5153)")
```

```text
intercept: -3.7714   (monograph Table 12: -3.7710)
sprinklered: -0.5150   (monograph Table 12: -0.5153)
```

The gaps of 0.0004 and 0.0003 come from the four-decimal slopes, multiplied by
`ln 200,000 = 12.2`. A rounding of 0.00005 in a slope moves the product by
0.0006, so both tables are consistent.

On this portfolio, `Terms.interaction` adds region times centred age, and
`f_test` compares the models with and without it. The true model has no
interaction, and the F-test agrees. Centring age at 40 also makes the region main
effects the relativities of a 40-year-old, as in Table 12:

```python
from prospicio.models import f_test

data["ac"] = [a - 40 for a in data["age"]]
main_terms = Terms().intercept().numeric("ac").factor("region")
mains = quasi.fit(main_terms.fit(data).design(data, offset=log_e), data["claims"])
inter_terms = main_terms.interaction("region", "ac").fit(data)
inter = quasi.fit(inter_terms.design(data, offset=log_e), data["claims"])
test = f_test(mains, inter)
print(f"region x age: F = {test['statistic']:.3f}, percentile {1 - test['p_value']:.1%}")
print("same fit as uncentred age:", abs(mains.deviance - big.deviance) < 1e-8,
      f"; base rate at age 40 {math.exp(mains.coefficients[0]):.4f} (true {math.exp(-1):.4f})")
```

```text
region x age: F = 1.024, percentile 61.8%
same fit as uncentred age: True ; base rate at age 40 0.4574 (true 0.3679)
```

## 3. Deviance and residuals (sections 6.1 and 6.3)

The scaled deviance is twice the gap in log-likelihood between the saturated
model (one parameter per row) and the fitted model. Multiplied by `phi`, it is
the unscaled deviance, which `GlmFit.deviance` and `deviance` report. For the
Poisson, each row contributes `2 [y ln(y / mu) - (y - mu)]`. For the gamma, each
row contributes `2 [-ln(y / mu) + (y - mu) / mu]`. On a tiny example:

```python
y3, mu3 = [0.0, 1.0, 3.0], [0.5, 1.2, 2.0]
pois = sum(2 * ((a * math.log(a / m) if a > 0 else 0.0) - (a - m)) for a, m in zip(y3, mu3))
print(f"Poisson deviance by hand {pois:.6f}, prospicio {deviance('poisson', y3, mu3):.6f}")
y3, mu3 = [100.0, 250.0, 900.0], [150.0, 200.0, 600.0]
gam_d = sum(2 * (-math.log(a / m) + (a - m) / m) for a, m in zip(y3, mu3))
print(f"gamma deviance by hand   {gam_d:.6f}, prospicio {deviance('gamma', y3, mu3):.6f}")
```

```text
Poisson deviance by hand 1.468148, prospicio 1.468148
gamma deviance by hand   0.387046, prospicio 0.387046
```

Section 6.3.2 defines the working residuals `(y - mu) g'(mu)`. The code bins them
with equal sums of working weights `w / (V(mu) g'(mu)^2)`. For a well-specified
model, the binned residuals have no bias and constant variance. The appendix shows
that the residual of each bin has mean 0 and variance `phi m / SWW`, for `m` bins
and total working weight `SWW`. It suggests a choice of `m` that makes this
variance about 0.01 or less. For the Poisson on the log link, the working weight
is `mu` (with the exposure in the offset).

Binned over a predictor (Figure 19 of the monograph), the residuals show a missed
non-linear effect. The code bins the `wavy_claims` residuals over age for the
straight-line model and for the GAM. If the model is right, the sum of the
squared bin residuals over their variance is about chi-square. This
distribution has 10 degrees of freedom (mean 10, 95th percentile 18.3):

```python
def binned_working_residuals(key, y, mu, m):
    """Working residuals (y - mu) / mu averaged with working weights mu in m bins of equal weight, sorted by key."""
    idx = sorted(range(len(y)), key=lambda i: key[i])
    total, out, cum, cur = sum(mu), [], 0.0, []
    for i in idx:
        cur.append(i)
        cum += mu[i]
        if cum >= total * (len(out) + 1) / m - 1e-9 or i == idx[-1]:
            out.append(sum(y[j] - mu[j] for j in cur) / sum(mu[j] for j in cur))
            cur = []
    return out


m = 10
sd = math.sqrt(m / sum(linear.fitted))
print(f"{m} bins by age, total working weight {sum(linear.fitted):.0f}, sd of a bin under the model {sd:.3f}")
for name, fit_mu in [("straight line", linear.fitted), ("GAM", gfit.fitted)]:
    bins = binned_working_residuals(ages, y, fit_mu, m)
    print(f"{name:13s}", " ".join(f"{r:+.2f}" for r in bins), f"  sum of squares / sd^2 {sum(r * r for r in bins) / sd ** 2:5.1f}")
```

```text
10 bins by age, total working weight 693, sd of a bin under the model 0.120
straight line +0.55 -0.21 -0.53 -0.52 -0.06 +0.41 +0.76 +0.10 -0.04 -0.48   sum of squares / sd^2 130.8
GAM           -0.01 +0.05 -0.09 +0.03 +0.05 -0.09 +0.08 +0.02 -0.01 -0.04   sum of squares / sd^2   2.1
```

The binned residuals of the straight line follow the sine, far beyond their
standard deviation of 0.12. The residuals of the GAM show no pattern. Their sum
of squares is even below the value that chance gives. This is the normal result
in the training data for a model that adapted its shape to them.

Section 6.3.1 also notes that the deviance residuals of counts cluster and do not
look normal. It suggests randomized quantile residuals (Dunn and Smyth). Each
count's residual is drawn at random within the normal scores of its probability
step. Under the right model, these residuals are exactly standard normal.
prospicio has them as `GlmFit.quantile_residuals`, and as
`prospicio.models.quantile_residuals` for any family. The latter uses the
randomized probability integral transform of `prospicio.models.pit`. The claim
count model has the true form:

```python
def moments(r):
    """Mean, standard deviation, skewness and share beyond 1.96 in absolute value."""
    m = sum(r) / len(r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    skew = sum((x - m) ** 3 for x in r) / len(r) / (sum((x - m) ** 2 for x in r) / len(r)) ** 1.5
    return f"mean {m:+.2f}, sd {sd:.2f}, skewness {skew:+.2f}, beyond 1.96: {sum(abs(x) > 1.96 for x in r) / len(r):.1%}"


claims_design = coding("age").design(data, offset=log_e)
claims_fit = poisson.fit(claims_design, data["claims"])
claims_diag = claims_fit.diagnostics(claims_design, data["claims"])
print("deviance residuals:", moments(claims_diag["deviance"]))
print("quantile residuals:", moments(claims_fit.quantile_residuals(claims_design, data["claims"], seed=1)))
```

```text
deviance residuals: mean -0.26, sd 0.82, skewness +1.45, beyond 1.96: 1.3%
quantile residuals: mean -0.05, sd 1.03, skewness +0.05, beyond 1.96: 4.5%
```

With about a quarter of a claim expected for each policy, the deviance residuals
fall into two clusters (no claim and one claim). They show skew and look too
narrow for a normal. The quantile residuals have the mean, spread, symmetry and tails of
a standard normal. A departure from these would point at the model.

## 4. Validation on held-out data (sections 4.3 and 7)

The monograph requires a test on data that the fit did not use. The analyst does
all the building on a training set and touches the test set once at the end.
The code holds out one fold of five from `k_fold`. It compares a "current plan"
(the straight line in age) with a "proposed model" (the GAM) on the claims of the
`wavy_claims` count. `lift` sorts the test rows by predicted claims per year and
cuts them into bands of equal exposure (the quantile plot of section 7.2.1). A
good model has actual claims that track its predictions and spread widely from
the first band to the last:

```python
from prospicio.models import gini, lift

train, test = folds[0]
yt = [y[i] for i in test]
et = [data["exposure"][i] for i in test]
current = poisson.fit(designs["linear"].select(train), [y[i] for i in train]).predict(designs["linear"].select(test))
proposed = gam.fit(gd.select(train), [y[i] for i in train]).predict(gd.select(test))
for name, pred in [("current", current), ("proposed", proposed)]:
    bands = lift(yt, pred, et, bands=5)
    avg = sum(yt) / sum(et)
    print(f"{name:9s} predicted", " ".join(f"{b['expected'] / b['exposure'] / avg:.2f}" for b in bands),
          "| actual", " ".join(f"{b['actual'] / b['exposure'] / avg:.2f}" for b in bands))
```

```text
current   predicted 0.85 0.97 1.16 1.35 1.56 | actual 0.91 0.39 1.13 1.43 1.12
proposed  predicted 0.49 0.77 1.16 1.70 2.15 | actual 0.46 0.50 1.03 1.46 1.55
```

Both lines are relative to the average frequency of the test set. The predictions
of the current plan spread from 0.85 to 1.56 of average. Its actual
frequencies do not rise with them (the second band is the lowest). The actuals of
the proposed model rise in every band, from 0.46 to 1.55.

The double lift chart (section 7.2.2) sorts by the ratio of the predictions of
the two models. The end bands hold the risks on which the models disagree most.
The chart shows which model's predictions the actuals follow. The loss ratio
chart (section 7.2.3) sorts by the proposed model over the current premium. If
the proposed model improves on the plan, the loss ratios rise across the bands.
`double_lift` and `loss_ratio_chart` draw both, in bands of about equal exposure,
as `lift` does:

```python
from prospicio.models import double_lift, loss_ratio_chart

dl = double_lift(yt, proposed, current, et, bands=5)
print("double lift, band average over overall average:")
for name, key in [("current", "b_relativity"), ("proposed", "a_relativity"), ("actual", "actual_relativity")]:
    print(f"  {name:9s}", " ".join(f"{band[key]:.2f}" for band in dl))
lr = loss_ratio_chart(yt, proposed, current, et, bands=5)
print("loss ratio (actual / current premium):", " ".join(f"{band['loss_ratio']:.2f}" for band in lr))
rate = lambda pred: [round(p / e, 10) for p, e in zip(pred, et)]  # equal ages tie exactly
print(f"Gini on the test set: current {gini(yt, rate(current), et):.3f}, proposed {gini(yt, rate(proposed), et):.3f}")
```

```text
double lift, band average over overall average:
  current   0.96 1.03 1.01 1.01 0.98
  proposed  0.42 0.66 0.97 1.36 1.59
  actual    0.48 0.65 1.32 0.75 1.82
loss ratio (actual / current premium): 0.42 0.54 1.10 0.62 1.57
Gini on the test set: current 0.121, proposed 0.256
```

In the double lift chart, the actuals run with the proposed model in both end
bands. The actuals are 0.48 and 1.82. The proposed model predicts 0.42 and 1.59,
and the current plan predicts about 1 in every band. Where the two models disagree, the proposed model was
right. The loss ratio chart tells the same story in a form that any stakeholder
reads. Under the current premium, the first band runs at a loss ratio of 0.42 and
the last at 1.57. The 120 test policies show one reversal.

The Gini index (section 7.2.4) sorts by predicted rate. It plots the cumulative
share of exposure against the cumulative share of losses (the Lorenz curve). The
index is twice the area between the curve and the diagonal. `gini` computes it
with exposure and takes ties together. The output above shows the values for the test
set. The predicted rate depends only on age, so policies of the same age tie.
They tie only up to the last bit of floating point, which differs between
platforms and between the Python and R fits. Rounding the rates to 10 decimals
makes those ties exact everywhere.

For a yes or no outcome, section 7.3 adds the ROC curve. The confusion matrices
of Table 13 give the rates that the monograph quotes. Footnote 17 states that the
area under the ROC curve is `0.5 x normalized Gini + 0.5`. The normalized Gini is
the Gini of the model over the Gini of a perfect model (each row predicted by its
own outcome). The code checks the identity with a logistic model of whether a
policy had a claim. It takes the area under the curve from `roc` (which counts
ties as half, as the Mann-Whitney statistic does) and the normalized Gini from
`normalized_gini`:

```python
from prospicio.models import normalized_gini, roc

print(f"threshold 50%: sensitivity {39 / 109:.3f}, specificity {673 / 704:.3f}, false positive rate {1 - 673 / 704:.3f}"
      "   (monograph 0.358, 0.956, 0.044)")
print(f"threshold 25%: sensitivity {75 / 109:.1%}, false positive rate {103 / 704:.1%}   (monograph 68.8%, 14.6%)")

had_claim = [1.0 if k > 0 else 0.0 for k in data["wavy_claims"]]
logit_design = Design([[1.0] * n, ages, [max(0.0, a - 38) for a in ages], [max(0.0, a - 62) for a in ages], log_e],
                      ["(Intercept)", "age", "h38", "h62", "log exposure"])
prob = Glm("binomial", "logit").fit(logit_design, had_claim).fitted
auc = roc(had_claim, prob)["auc"]
print(f"AUROC {auc:.6f}; 0.5 x normalized Gini + 0.5 = {0.5 * normalized_gini(had_claim, prob) + 0.5:.6f}")
```

```text
threshold 50%: sensitivity 0.358, specificity 0.956, false positive rate 0.044   (monograph 0.358, 0.956, 0.044)
threshold 25%: sensitivity 68.8%, false positive rate 14.6%   (monograph 68.8%, 14.6%)
AUROC 0.743339; 0.5 x normalized Gini + 0.5 = 0.743339
```

The two agree to the last digit printed. For a binary outcome, the Lorenz curve
and the ROC curve carry the same information. This is why the monograph warns
against reporting both as separate evidence.

## 5. Stability and regularization (sections 6.4 and 10.5)

A stable model gives similar answers on similar data. Section 6.4 suggests a
refit on the cross-validation training sets (or bootstrap samples) to watch the
coefficients:

```python
c = coding("age")
full = c.design(data, offset=log_e)
print("fold  age      B       C       D")
for k, (tr, _) in enumerate(folds):
    fit = poisson.fit(full.select(tr), [data["claims"][i] for i in tr])
    print(f"{k:4d}", " ".join(f"{b:+.3f}" if j else f"{b:+.4f}" for j, b in enumerate(fit.coefficients[1:])))
```

```text
fold  age      B       C       D
   0 -0.0091 +0.166 -0.314 +0.274
   1 -0.0097 -0.099 -0.284 +0.122
   2 -0.0112 +0.207 -0.318 +0.061
   3 -0.0093 +0.052 -0.122 +0.345
   4 -0.0132 +0.051 -0.270 +0.187
```

The age slope stays between -0.009 and -0.013. The region coefficients move by up
to 0.3, about their standard errors. The 155 claims leave them unsettled.

Section 6.4 also looks for single records with an outsized pull on the
coefficients. Cook's distance measures how far all the coefficients move, scaled
by their covariance, when the fit leaves out one row. It is large for a row with both a
large residual and a high leverage (hat value). `GlmFit.diagnostics` returns both
for the claim count model of section 3:

```python
cook = claims_diag["cooks_distance"]
top = sorted(range(n), key=lambda i: -cook[i])[:3]
print("claims  expected  region   age  leverage  Cook's D")
for i in top:
    print(f"{data['claims'][i]:6.0f} {claims_fit.fitted[i]:9.2f}  {data['region'][i]:>6s} {data['age'][i]:5.1f}"
          f" {claims_diag['leverage'][i]:9.4f} {cook[i]:9.4f}")
rest = [i for i in range(n) if i != top[0]]
refit = poisson.fit(claims_design.select(rest), [data["claims"][i] for i in rest])
print(f"region B coefficient {claims_fit.coefficients[2]:+.3f}, without that policy {refit.coefficients[2]:+.3f}")
```

```text
claims  expected  region   age  leverage  Cook's D
     3      0.32       B  69.9    0.0116    0.0522
     2      0.34       C  22.2    0.0201    0.0334
     2      0.31       C  33.7    0.0151    0.0292
region B coefficient +0.078, without that policy +0.018
```

The most influential policy had three claims in 0.9 years where 0.32 were
expected. Leaving it out alone moves the region B coefficient from +0.078 to
+0.018. No leverage is high (the average is 5/600, 0.008), so the pull comes from
the residual. With 155 claims in all, one policy can shift the relativity of a
thin level by several points. This is the reason that the monograph gives to
look.

The elastic net (section 10.5) minimizes the deviance plus a penalty on the size
of the coefficients. The penalty is `lambda` times a mix of their absolute values
(the lasso) and squares (ridge), on standardized columns with the intercept
unpenalized. As `lambda` falls from the value that zeroes every coefficient,
variables enter one by one. Cross-validation picks `lambda`. `ElasticNet`
minimizes the objective of glmnet, for every family including the gamma and the
Tweedie. The code gives the claim count model age, region and two columns
unrelated to claims (`trials` and `n_sev`, drawn independently). The lasso then
sorts them:

```python
from prospicio.models import ElasticNet

dummies = {g: [1.0 if r == g else 0.0 for r in data["region"]] for g in "BCD"}
cols = ["age", "trials", "n_sev"]
names = ["(Intercept)"] + cols + ["B", "C", "D"]
X = Design([[1.0] * n] + [data[k] for k in cols] + [dummies[g] for g in "BCD"], names, offset=log_e)
net = ElasticNet("poisson", "log", alpha=1.0)
lams = net.lambda_path(X, data["claims"], n=40, min_ratio=1e-3)
cv = net.cross_validate(X, data["claims"], lams, folds)
print(f"lambda_max {lams[0]:.5f}, CV minimum at {cv.lam_min:.5f}, one-SE choice {cv.lam_1se:.5f}")
for lam_name, lam in [("lambda_min", cv.lam_min), ("smaller", lams[-1])]:
    fit = net.with_lam(lam).fit(X, data["claims"])
    print(f"{lam_name:10s}", ", ".join(f"{nm} {b:+.4f}" for nm, b in zip(names[1:], fit.coefficients[1:])))
glm_fit = poisson.fit(X, data["claims"])
print(f"{'GLM':10s}", ", ".join(f"{nm} {b:+.4f}" for nm, b in zip(names[1:], glm_fit.coefficients[1:])))
```

```text
lambda_max 0.05114, CV minimum at 0.02518, one-SE choice 0.05114
lambda_min age -0.0055, trials +0.0000, n_sev +0.0000, B +0.0000, C -0.0486, D +0.0000
smaller    age -0.0106, trials -0.0254, n_sev +0.0629, B +0.0696, C -0.2758, D +0.1984
GLM        age -0.0106, trials -0.0255, n_sev +0.0631, B +0.0700, C -0.2762, D +0.1990
```

At the cross-validated `lambda`, the lasso keeps age (shrunk by half) and a small
region C discount. It drops the rest, including both noise columns. With a
cross-validation curve this flat, the one-standard-error rule drops everything.
At a small `lambda`, the lasso is close to the unpenalized GLM, which gives the
two noise columns small but non-zero coefficients. The shrinkage of the thin
levels towards zero is the credibility-like behaviour that the monograph points
to.

Section 6 of the first part maps each step of the monograph workflow to
prospicio.
