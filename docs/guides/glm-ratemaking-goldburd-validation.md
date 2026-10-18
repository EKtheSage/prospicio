# GLMs for insurance rating, part 2: refinement and validation (Goldburd, Khare, Tevet and Guller)

This is the second part of a guide to Goldburd, Khare, Tevet and Guller's
*Generalized Linear Models for Insurance Rating* (CAS Monograph 5, second
edition, 2025 revision) with prospicio's Python API. The first part,
[`glm-ratemaking-goldburd.md`](glm-ratemaking-goldburd.md), explains the GLM
as a multiplicative rating plan, offsets and weights, the choice of
distribution and the F-test, and lists what prospicio does not support. This
part follows the monograph's model refinement and validation: continuous
variables (section 5.4), interactions (5.6), deviance and residuals (6.1 and
6.3), validation on held-out data with lift, double lift and loss ratio
charts, the Gini index and the ROC curve (4.3 and 7), and stability and the
elastic net (6.4 and 10.5). As in the first part, the monograph's own examples
use data the PDF does not contain, so its printed arithmetic is recomputed
where it can be and the workflow runs on the repository's synthetic portfolio
`validation/data/glm_policies.csv`. The monograph's text and exhibits are the
authors'; what follows explains them in new words, and section numbers in
parentheses are the monograph's. The whole part runs in about a second.

The setup repeats the first part's: the 600 policies, the log of exposure as
the claim count offset, a helper for the region factor, the Poisson and
overdispersed Poisson families, and the F distribution function from
prospicio's `Beta`. Besides `claims` (Poisson with a log-linear age effect and
region factors 1.0, 1.2, 0.8 and 1.5 for A to D), this part uses
`wavy_claims`, a Poisson count whose true age effect is the non-linear
`0.8 sin(age / 8)`:

```python
import csv
import math
from pathlib import Path

from prospicio.distributions import Beta
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


def f_cdf(f, d1, d2):
    """F distribution function as a regularized incomplete beta."""
    return Beta(d1 / 2, d2 / 2).cdf(d1 * f / (d1 * f + d2))


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

A continuous predictor enters linearly on the link scale. To see whether that
is enough, section 5.4.1 plots partial residuals: the working residual
`(y - mu) / mu` (log link) plus the predictor's own term `b x`, against `x`.
Plotting thousands of points hides the pattern, so they are binned, averaged
with the working weights (`mu` times the prior weight for a Poisson on the log
link, section 6.3.2). We do this for the `wavy_claims` count, whose true age
effect is `0.8 sin(age / 8)`, fitted with a straight line in age:

```python
order = sorted(range(n), key=lambda i: data["age"][i])
linear = poisson.fit(age_only.design(data, offset=log_e), data["wavy_claims"])
b_age = linear.coefficients[1]
mu = linear.fitted
y = data["wavy_claims"]
partial = [(y[i] - mu[i]) / mu[i] + b_age * data["age"][i] for i in range(n)]
print(f"linear age slope {b_age:+.4f}")
print("age band   mean age  binned partial residual  fitted b*age  true 0.8 sin(age/8), centred")
true_c = sum(0.8 * math.sin(a / 8) for a in data["age"]) / n
for k in range(8):
    idx = order[k * n // 8:(k + 1) * n // 8]
    ww = [mu[i] for i in idx]
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

The fitted line rises gently while the binned partial residuals start high,
fall to a trough in the thirties and forties, peak around 60 and fall again:
the true sine, shifted by a constant of about 0.6 (partial residuals carry the
intercept's share of the fit). The monograph offers four fixes, and prospicio fits all of them,
though the bins, polynomials and hinges are design columns built by hand since
`Terms` has only numeric columns and factors:

- **bins** (5.4.2): a factor of age bands of about equal counts; each band
  floats freely, at the cost of many parameters and possible reversals;
- **polynomials** (5.4.3): powers of a centred and scaled age, which bend but
  misbehave at the edges;
- **hinge functions** (5.4.4): `max(0, age - knot)` terms, so the slope changes
  at chosen knots, read off the partial residual plot (here the trough near
40 and the peak near 60: knots at 38 and 62);
- **splines** (5.4.5 and 10.3): prospicio's `Gam` fits a penalized P-spline
  with the smoothness chosen by UBRE, the GAM of section 10.3, in place of the
  natural cubic spline, which the Python bindings do not build.

Each is scored three ways: training deviance, AIC (with the GAM's effective
degrees of freedom), and the mean Poisson deviance on held-out folds from
`cross_validate` with five folds. The folds come from `k_fold`, the same in
Python and R:

```python
from prospicio.models import Gam, cross_validate, deviance, deviance_score, k_fold

ages = data["age"]
cuts = [sorted(ages)[k * n // 8] for k in range(1, 8)]
band = [str(sum(a >= c for c in cuts)) for a in ages]
z = [(a - 50) / 20 for a in ages]
designs = {
    "linear": Design([[1.0] * n, ages], ["(Intercept)", "age"], offset=log_e),
    "8 bins": Terms().intercept().factor("band").fit({"band": band}).design({"band": band}, offset=log_e),
    "cubic": Design([[1.0] * n, z, [v ** 2 for v in z], [v ** 3 for v in z]], ["(Intercept)", "z", "z2", "z3"], offset=log_e),
    "hinges 38, 62": Design([[1.0] * n, ages, [max(0.0, a - 38) for a in ages], [max(0.0, a - 62) for a in ages]],
                            ["(Intercept)", "age", "h38", "h62"], offset=log_e),
}
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
ahead of the GAM, which found its shape without being told where to bend; the
cubic is close behind. The bins pay for their extra parameters with a step
function that ignores the variation inside each band.

The monograph's building-age example reads the same way from its printed
coefficients. Table 6 has a slope of 1.225 below the knot at log age 2.75 and
1.225 - 2.269 above it; Table 7 adds a knot at 3.6 with coefficient 1.170; the
quadratic of Figure 10 (4.749 x - 0.866 x^2) peaks where its derivative is
zero, at `x = 4.749 / (2 x 0.866)`, close to the knot the authors chose by eye
and to the GAM's peak at about `e^2.8`, 16 years (section 10.3):

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

With the second knot the slope past log age 3.6 is almost flat, the "leveling
off" the monograph describes.

## 2. Interactions (section 5.6)

An interaction lets one variable's relativity depend on another's level. In a
multiplicative plan it is one more factor, applied only to the combination.
Figure 13's two-by-two example has main factors 2.0 and 1.5 and an interaction
factor of 1.1 on the BY cell; Table 9 gives occupancy class 2 a sprinkler
factor from the sum of the main and interaction coefficients:

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

An interaction between a factor and a continuous variable shifts the slope,
and the main effect then refers to the point where the continuous variable is
zero. In Table 11 that is log AOI = 0, an amount of $1, which is why the
sprinkler main effect turns positive. Table 12 refits with
`log(AOI / 200,000)`, so the main effect is the sprinkler relativity at the
base amount. The two tables are the same model, and Table 12's coefficients
follow from Table 11's: each intercept-like term picks up its slope times
`ln 200,000`:

```python
shift = math.log(200_000)
print(f"intercept: {-8.9456 + 0.4239 * shift:.4f}   (monograph Table 12: -3.7710)")
print(f"sprinklered: {0.7447 - 0.1032 * shift:.4f}   (monograph Table 12: -0.5153)")
```

```text
intercept: -3.7714   (monograph Table 12: -3.7710)
sprinklered: -0.5150   (monograph Table 12: -0.5153)
```

The 0.0004 and 0.0003 gaps come from the four-decimal slopes multiplied by
`ln 200,000 = 12.2`: a rounding of 0.00005 in a slope moves the product by
0.0006, so both tables are consistent.

`Terms` has no interaction term, so on our portfolio the interaction columns
are products built by hand: region times centred age. The true model has no
interaction, and the F-test agrees; centring age at 40 also makes the region
main effects the relativities of a 40-year-old, as in Table 12:

```python
ac = [a - 40 for a in data["age"]]
dummies = {g: [1.0 if r == g else 0.0 for r in data["region"]] for g in "BCD"}
main_cols = [[1.0] * n, ac] + [dummies[g] for g in "BCD"]
main_names = ["(Intercept)", "age-40", "B", "C", "D"]
inter_cols = [[d * a for d, a in zip(dummies[g], ac)] for g in "BCD"]
mains = quasi.fit(Design(main_cols, main_names, offset=log_e), data["claims"])
inter = quasi.fit(Design(main_cols + inter_cols, main_names + ["B:age", "C:age", "D:age"], offset=log_e), data["claims"])
F = (mains.deviance - inter.deviance) / (3 * inter.dispersion)
print(f"region x age: F = {F:.3f}, percentile {f_cdf(F, 3, inter.df_resid):.1%}")
print("same fit as uncentred age:", abs(mains.deviance - big.deviance) < 1e-8,
      f"; base rate at age 40 {math.exp(mains.coefficients[0]):.4f} (true {math.exp(-1):.4f})")
```

```text
region x age: F = 1.024, percentile 61.8%
same fit as uncentred age: True ; base rate at age 40 0.4574 (true 0.3679)
```

## 3. Deviance and residuals (sections 6.1 and 6.3)

The scaled deviance is twice the gap in log-likelihood between the saturated
model (one parameter per row) and the fitted one; multiplied by `phi` it is the
unscaled deviance, which `GlmFit.deviance` and `deviance` report. For the
Poisson each row contributes `2 [y ln(y / mu) - (y - mu)]`, and for the gamma
`2 [-ln(y / mu) + (y - mu) / mu]`. On a tiny example:

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

Section 6.3.2's working residuals `(y - mu) g'(mu)`, binned with equal sums of
working weights `w / (V(mu) g'(mu)^2)`, are unbiased and homoscedastic for a
well-specified model: the appendix shows each bin's residual has mean 0 and
variance `phi m / SWW` for `m` bins and total working weight `SWW`, and
suggests choosing `m` so that this is about 0.01 or less. For the Poisson on
the log link the working weight is `mu` (with the exposure in the offset).
Binned over a predictor (the monograph's Figure 19), they show a missed
non-linear effect. We bin the `wavy_claims` residuals over age for the
straight-line model and the GAM; if the model is right, the sum of the squared
bin residuals over their variance is about chi-square with 10 degrees of
freedom (mean 10, 95th percentile 18.3):

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

The straight line's binned residuals follow the sine, far beyond their
standard deviation of 0.12. The GAM's show no pattern; their sum of squares is
even below what chance gives, as expected in the training data for a model
that adapted its shape to them. Section 6.3.1 also notes
that deviance residuals of counts cluster and do not look normal, and suggests
randomized quantile residuals (Dunn and Smyth) instead; prospicio's
`prospicio.models.pit` gives the randomized probability integral transform
those residuals are built from. Section 6.4's Cook's distance needs the hat
values, which `GlmFit` does not expose either (issue #334).

## 4. Validation on held-out data (sections 4.3 and 7)

The monograph insists that a model be judged on data it was not fitted on:
a training set for all the building, a test set touched once at the end. We
hold out one fold of five from `k_fold` and compare a "current plan" (the
straight line in age) with a "proposed model" (the GAM) on the claims of the
`wavy_claims` count. `lift` sorts the test rows by predicted claims per year
and cuts them into bands of equal exposure (the quantile plot of section
7.2.1); a good model's actual claims track its predictions and spread widely
from the first band to the last:

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

Both lines are relative to the test set's average frequency. The current
plan's predictions spread from 0.85 to 1.56 of average but its actual
frequencies do not rise with them (the second band is the lowest); the
proposed model's actuals rise in every band, from 0.46 to 1.55.

The double lift chart (section 7.2.2) sorts by the ratio of the two models'
predictions, so the end bands hold the risks on which they disagree most, and
asks which model's predictions the actuals follow. The loss ratio chart
(section 7.2.3) sorts by the proposed model over the current premium, and a
plan the proposed model improves on shows loss ratios that rise across the
bands. prospicio has `lift` but neither of these charts, so both are a few
lines over `lift`'s banding:

```python
def bands_by(key, cols, m=5):
    """Sum each column in m bands of about equal exposure after sorting by key."""
    idx = sorted(range(len(key)), key=lambda i: key[i])
    total, cum, out = sum(et), 0.0, [[0.0] * len(cols) for _ in range(m)]
    for i in idx:
        b = min(int((cum + 0.5 * et[i]) / total * m), m - 1)
        cum += et[i]
        for j, col in enumerate(cols):
            out[b][j] += col[i]
    return out


ratio = [p / c for p, c in zip(proposed, current)]
dl = bands_by(ratio, [et, current, proposed, yt])
tot = [sum(col) for col in (et, current, proposed, yt)]
print("double lift, band average over overall average:")
for name, j in [("current", 1), ("proposed", 2), ("actual", 3)]:
    print(f"  {name:9s}", " ".join(f"{(band[j] / band[0]) / (tot[j] / tot[0]):.2f}" for band in dl))
lr = bands_by(ratio, [yt, current])
print("loss ratio (actual / current premium):", " ".join(f"{a / c:.2f}" for a, c in lr))
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

In the double lift chart the actuals run with the proposed model in both end
bands (0.48 and 1.82 against its 0.42 and 1.59, while the current plan
predicts about 1 in every band), so where the two disagree the proposed model
was right. The loss ratio chart tells the same story in a form any
stakeholder reads: under the current premium the first band runs at a loss
ratio of 0.42 and the last at 1.57, with one reversal on 120 test policies.

The Gini index (section 7.2.4) sorts by predicted rate, plots cumulative share
of exposure against cumulative share of losses (the Lorenz curve), and is
twice the area between the curve and the diagonal; `gini` computes it with
exposure, ties taken together; the test set's values are printed above. The
predicted rate depends only on age, so policies of the same age tie, but only
up to the last bit of floating point, which differs between platforms and
between the Python and R fits; rounding the rates to 10 decimals makes those
ties exact everywhere.

For a yes or no outcome, section 7.3 adds the ROC curve. Table 13's confusion
matrices give the rates the monograph quotes, and footnote 17 states that the
area under the ROC curve is `0.5 x normalized Gini + 0.5`, with the normalized
Gini the model's Gini over that of a perfect model (each row predicted by its
own outcome). We check the identity with a logistic model of whether a policy
had a claim, computing the area under the curve by ranks
(Mann-Whitney) and the Gini with `gini`:

```python
print(f"threshold 50%: sensitivity {39 / 109:.3f}, specificity {673 / 704:.3f}, false positive rate {1 - 673 / 704:.3f}"
      "   (monograph 0.358, 0.956, 0.044)")
print(f"threshold 25%: sensitivity {75 / 109:.1%}, false positive rate {103 / 704:.1%}   (monograph 68.8%, 14.6%)")

had_claim = [1.0 if k > 0 else 0.0 for k in data["wavy_claims"]]
logit_design = Design([[1.0] * n, ages, [max(0.0, a - 38) for a in ages], [max(0.0, a - 62) for a in ages], log_e],
                      ["(Intercept)", "age", "h38", "h62", "log exposure"])
prob = Glm("binomial", "logit").fit(logit_design, had_claim).fitted
pos = [p for p, h in zip(prob, had_claim) if h]
neg = [p for p, h in zip(prob, had_claim) if not h]
auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
normalized = gini(had_claim, prob) / gini(had_claim, had_claim)
print(f"AUROC by ranks {auc:.6f}; 0.5 x normalized Gini + 0.5 = {0.5 * normalized + 0.5:.6f}")
```

```text
threshold 50%: sensitivity 0.358, specificity 0.956, false positive rate 0.044   (monograph 0.358, 0.956, 0.044)
threshold 25%: sensitivity 68.8%, false positive rate 14.6%   (monograph 68.8%, 14.6%)
AUROC by ranks 0.743339; 0.5 x normalized Gini + 0.5 = 0.743339
```

The two agree to the last digit printed: for a binary outcome the Lorenz curve
and the ROC curve carry the same information, which is why the monograph
warns against reporting both as separate evidence.

## 5. Stability and regularization (sections 6.4 and 10.5)

A stable model gives similar answers on similar data. Section 6.4 suggests
refitting on the cross-validation training sets (or bootstrap samples) and
watching the coefficients:

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

The age slope stays between -0.009 and -0.013 while the region coefficients
move by up to 0.3, about their standard errors: 155 claims leave them
unsettled.

The elastic net (section 10.5) minimizes the deviance plus a penalty on the
size of the coefficients, `lambda` times a mix of their absolute values (the
lasso) and squares (ridge), on standardized columns with the intercept
unpenalized. As `lambda` falls from the value that zeroes every coefficient,
variables enter one by one; cross-validation picks `lambda`. prospicio's
`ElasticNet` minimizes glmnet's objective, for every family including the
gamma and the Tweedie (footnote 30 notes glmnet itself lacks them). We give
the claim count model age, region and two columns unrelated to claims
(`trials` and `n_sev`, drawn independently) and let the lasso sort them:

```python
from prospicio.models import ElasticNet

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

At the cross-validated `lambda` the lasso keeps age (shrunk by half) and a
small region C discount and drops the rest, including both noise columns; the
one-standard-error rule, with a cross-validation curve this flat, drops
everything. At a small `lambda` the lasso is close to the unpenalized GLM,
which gives the two noise columns small but non-zero coefficients. The shrinkage of the thin
levels towards zero is the credibility-like behaviour the monograph points to.

What prospicio does not do (yet) for this workflow is listed in the first
part's last section, with the issues filed for the Models lane.
