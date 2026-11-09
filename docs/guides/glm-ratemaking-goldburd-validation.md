# GLMs for insurance rating, part 2: refinement and validation (Goldburd, Khare, Tevet and Guller)

<!-- site: execute -->

This is part 2 of a guide to *Generalized Linear Models for Insurance Rating*
(CAS Monograph 5, second edition, 2025 revision; Goldburd, Khare, Tevet and
Guller), with the prospicio Python API. [Part 1](glm-ratemaking-goldburd.md)
covers the multiplicative rating plan, offsets, weights, the choice of
distribution and the F-test, and maps the monograph's workflow to prospicio.

Section numbers are the monograph's. The PDF does not hold its data, so the
guide recomputes the printed arithmetic where it can and runs the workflow on
the synthetic portfolio `validation/data/glm_policies.csv`.

The setup is that of part 1: 600 policies, with the log of exposure as the
claim count offset. `claims` is Poisson with a log-linear age effect and region
factors 1.0, 1.2, 0.8 and 1.5 for A to D. `wavy_claims` is Poisson with the
non-linear age effect `0.8 sin(age / 8)`:

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

A continuous predictor enters linearly on the link scale. Section 5.4.1 tests
this with partial residuals against `x`: the working residual `(y - mu) / mu`
(log link) plus the predictor's own term `b x`. Each bin is averaged with the
working weights (`mu` times the prior weight for a Poisson on the log link,
section 6.3.2). `GlmFit.diagnostics` returns both. The fit is a straight line in
age on `wavy_claims`:

```python
order = sorted(range(n), key=lambda i: data["age"][i])
y = data["wavy_claims"]
linear = poisson.fit(age_only.design(data, offset=log_e), y)
b_age = linear.coefficients[1]
diag = linear.diagnostics(age_only.design(data, offset=log_e), y)
partial = [r + b_age * a for r, a in zip(diag["working"], data["age"])]
print(f"linear age slope {b_age:+.4f}")
```

```text
linear age slope +0.0121
```

```python
import pandas as pd
from great_tables import GT

true_c = sum(0.8 * math.sin(a / 8) for a in data["age"]) / n
bands = []
for k in range(8):
    idx = order[k * n // 8:(k + 1) * n // 8]
    ww = [diag["working_weights"][i] for i in idx]
    pr = sum(partial[i] * w for i, w in zip(idx, ww)) / sum(ww)
    age = sum(data["age"][i] * w for i, w in zip(idx, ww)) / sum(ww)
    lo, hi = data["age"][idx[0]], data["age"][idx[-1]]
    bands.append({"band": f"{lo:.1f}-{hi:.1f}", "mean_age": age, "partial": pr, "fitted": b_age * age,
                 "true": 0.8 * math.sin(age / 8) - true_c})
(GT(pd.DataFrame(bands), rowname_col="band")
 .tab_header("Partial residuals of wavy_claims against age",
             "Eight bands of equal count, averaged with the working weights")
 .cols_label(mean_age="mean age", partial="binned partial residual", fitted="fitted b x age",
             true="true 0.8 sin(age/8), centred")
 .fmt_number(["mean_age"], decimals=1)
 .fmt_number(["partial", "fitted"], decimals=3)
 .fmt_number(["true"], decimals=3, force_sign=True))
```

```text
           mean age  binned partial residual  fitted b x age  true 0.8 sin(age/8), centred
18.1-24.7      21.6                    0.840           0.263                        +0.315
24.8-30.6      28.2                    0.307           0.341                        -0.318
30.7-38.3      34.3                   -0.106           0.416                        -0.750
38.5-46.8      42.9                    0.005           0.520                        -0.660
47.0-56.7      52.0                    0.747           0.631                        +0.152
56.8-62.9      59.7                    1.216           0.724                        +0.716
63.0-71.4      67.6                    1.117           0.820                        +0.640
71.5-79.9      76.1                    0.603           0.922                        -0.088
```

```python
import matplotlib.pyplot as plt

bins = [order[k * n // 8:(k + 1) * n // 8] for k in range(8)]
wsum = [sum(diag["working_weights"][i] for i in idx) for idx in bins]
bin_age = [sum(data["age"][i] * diag["working_weights"][i] for i in idx) / w for idx, w in zip(bins, wsum)]
bin_pr = [sum(partial[i] * diag["working_weights"][i] for i in idx) / w for idx, w in zip(bins, wsum)]
grid = [18 + 0.5 * j for j in range(125)]
fig, ax = plt.subplots()
ax.plot(grid, [b_age * a for a in grid], label="fitted straight line, b x age")
ax.plot(grid, [0.8 * math.sin(a / 8) - true_c for a in grid], label="true effect, 0.8 sin(age / 8), centred")
ax.scatter(bin_age, bin_pr, color="0.4", label="binned partial residual")
ax.set_xlabel("age (years)")
ax.set_ylabel("partial residual (log scale)")
ax.set_title("Partial residuals of wavy_claims against age: a line misses the wave")
ax.legend(loc="upper left")
```

The binned residuals follow the true sine, shifted by about 0.6 (the intercept's
share), with a trough near 40 and a peak near 60. The monograph offers four
fixes, and prospicio fits all of them. `Terms` builds the first three:

- **bins** (5.4.2): a factor of age bands of about equal counts. Each band
  floats freely, at the cost of many parameters and possible reversals.
- **polynomials** (5.4.3): powers of a centred and scaled age. They bend, but
  misbehave at the edges.
- **hinge functions** (5.4.4): `max(0, age - knot)` terms. The slope changes at
  chosen knots, here 38 and 62, read off the trough near 40 and the peak near 60.
- **splines** (5.4.5 and 10.3): `Gam` fits a penalized P-spline with the
  smoothness chosen by UBRE (the GAM of section 10.3). It replaces the natural
  cubic spline (`Terms.natural_spline`).

Each fit gets its training deviance, its AIC (with the GAM's effective degrees
of freedom) and its mean Poisson deviance on five held-out folds from
`cross_validate`. `k_fold` gives the same folds in Python and R:

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
scores = []
for name, d in designs.items():
    fit = poisson.fit(d, y)
    cv = sum(cross_validate(poisson, d, y, folds, score)) / len(folds)
    scores.append((name, str(len(fit.coefficients)), fit.deviance, fit.aic, cv))
gam = Gam(poisson, [("age", 10)])
gd = designs["linear"]
gfit = gam.fit(gd, y)
ll = sum(yi * math.log(m) - m - math.lgamma(yi + 1) for yi, m in zip(y, gfit.fitted))
cv = sum(cross_validate(gam, gd, y, folds, score)) / len(folds)
scores.append(("P-spline GAM", f"{gfit.edf:.1f}", gfit.deviance, -2 * ll + 2 * gfit.edf, cv))
(GT(pd.DataFrame(scores, columns=["model", "parameters", "deviance", "aic", "cv"]), rowname_col="model")
 .tab_header("Five age effects on wavy_claims", "Training deviance and AIC; mean deviance on five held-out folds")
 .cols_label(aic="AIC", cv="CV deviance")
 .fmt_number(["deviance", "aic"], decimals=1, use_seps=False)
 .fmt_number(["cv"], decimals=4))
```

```text
               parameters  deviance     AIC  CV deviance
linear                  2     769.9  1673.9       1.2923
8 bins                  8     658.6  1574.6       1.1338
cubic                   4     635.5  1543.5       1.0730
hinges 38, 62           4     624.0  1532.0       1.0534
P-spline GAM          5.8     621.0  1532.7       1.0612
```

All four non-linear forms beat the straight line. The hinges, with knots read
off the partial residuals, do best out of sample, just ahead of the GAM, which
found its shape unaided.

The monograph's building-age example, from its printed coefficients: Table 6 has
a slope of 1.225 below the knot at log age 2.75 and 1.225 - 2.269 above it, and
Table 7 adds a knot at 3.6 with coefficient 1.170. The quadratic of Figure 10
(4.749 x - 0.866 x^2) peaks at `x = 4.749 / (2 x 0.866)`, near the authors' knot
and the GAM's peak at about `e^2.8` (section 10.3):

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

Past the second knot at log age 3.6 the slope is almost flat, the "leveling
off" of the monograph.

## 2. Interactions (section 5.6)

An interaction makes one variable's relativity depend on the level of another.
In a multiplicative plan it is one more factor, applied only to the combination.
Figure 13 has main factors 2.0 and 1.5 and an interaction
factor of 1.1 on the BY cell. Table 9 gives occupancy class 2 a sprinkler factor
from the sum of the main and interaction coefficients:

```python
base, f_b, f_y, f_by = 10.0, 2.0, 1.5, 1.1
checks = pd.DataFrame({
    "quantity": ["cell X/A", "cell X/B", "cell Y/A", "cell Y/B", "sprinkler discount, main effects only",
                 "sprinkler discount, class 1 with the interaction", "sprinkler factor, class 2 (a 43.7% discount)"],
    "prospicio": [base, base * f_b, base * f_y, round(base * f_b * f_y * f_by, 1), math.exp(-0.3046) - 1,
                  math.exp(-0.2895) - 1, math.exp(-0.2895 - 0.2847)],
    "book": [10, 20, 15, 33, -0.263, -0.251, 0.563],
})
(GT(checks, rowname_col="quantity")
 .tab_header("Interaction arithmetic against the monograph", "Figure 13 and Table 9")
 .fmt_number(["prospicio", "book"], rows=[0, 1, 2, 3], decimals=0)
 .fmt_percent(["prospicio", "book"], rows=[4, 5], decimals=1)
 .fmt_number(["prospicio", "book"], rows=[6], decimals=3))
```

```text
                                                  prospicio    book
cell X/A                                                 10      10
cell X/B                                                 20      20
cell Y/A                                                 15      15
cell Y/B                                                 33      33
sprinkler discount, main effects only                -26.3%  -26.3%
sprinkler discount, class 1 with the interaction     -25.1%  -25.1%
sprinkler factor, class 2 (a 43.7% discount)          0.563   0.563
```

A factor-by-continuous interaction shifts the slope, and the main effect then
refers to the point where the continuous variable is zero. In Table 11 that is
log AOI = 0, an amount of $1, which is why the sprinkler main effect turns
positive. Table 12 uses `log(AOI / 200,000)`, so the main effect is the
relativity at the base amount. The tables are the same model: each
intercept-like term of Table 12 picks up its slope times `ln 200,000`:

```python
shift = math.log(200_000)
print(f"intercept: {-8.9456 + 0.4239 * shift:.4f}   (monograph Table 12: -3.7710)")
print(f"sprinklered: {0.7447 - 0.1032 * shift:.4f}   (monograph Table 12: -0.5153)")
```

```text
intercept: -3.7714   (monograph Table 12: -3.7710)
sprinklered: -0.5150   (monograph Table 12: -0.5153)
```

The gaps of 0.0004 and 0.0003 are rounding: a slope rounded by 0.00005 moves its
product with `ln 200,000 = 12.2` by 0.0006.

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

Section 6.3.2 defines the working residuals `(y - mu) g'(mu)` and bins them with
equal sums of working weights `w / (V(mu) g'(mu)^2)` (`mu` for the Poisson on
the log link). For a well-specified model, each of `m` binned residuals has mean 0 and variance
`phi m / SWW`, for total working weight `SWW`; the appendix suggests an `m` that
makes this about 0.01 or less. Binned over a predictor (Figure 19), the
residuals show a missed non-linear effect. Here the `wavy_claims` residuals are
binned over age for the straight line and the GAM. If the model is right, the
sum of the squared bin residuals over their variance is about chi-square with 10
degrees of freedom (mean 10, 95th percentile 18.3):

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
```

```text
10 bins by age, total working weight 693, sd of a bin under the model 0.120
```

```python
resid = []
for name, fit_mu in [("straight line", linear.fitted), ("GAM", gfit.fitted)]:
    bins = binned_working_residuals(ages, y, fit_mu, m)
    resid.append({"model": name, **{str(j + 1): r for j, r in enumerate(bins)},
                  "ss": sum(r * r for r in bins) / sd ** 2})
bin_cols = [str(j + 1) for j in range(m)]
(GT(pd.DataFrame(resid), rowname_col="model")
 .tab_header("Binned working residuals over age", "Ten bins of equal working weight")
 .tab_spanner("bin", bin_cols)
 .cols_label(ss="sum of squares / sd^2")
 .fmt_number(bin_cols, decimals=2, force_sign=True)
 .fmt_number(["ss"], decimals=1))
```

```text
                   1      2      3      4      5      6      7      8      9     10  sum of squares / sd^2
straight line  +0.55  -0.21  -0.53  -0.52  -0.06  +0.41  +0.76  +0.10  -0.04  -0.48                  130.8
GAM            -0.01  +0.05  -0.09  +0.03  +0.05  -0.09  +0.08  +0.02  -0.01  -0.04                    2.1
```

The straight line's residuals follow the sine, far beyond their standard
deviation of 0.12. The GAM's show no pattern; their sum of squares is below
chance, as expected for a model fitted to the same data.

Section 6.3.1 notes that deviance residuals of counts cluster and look
non-normal, and suggests randomized quantile residuals (Dunn and Smyth): each
count's residual is drawn at random within the normal scores of its probability
step, so under the right model it is exactly standard normal. prospicio has them
as `GlmFit.quantile_residuals`, and as `prospicio.models.quantile_residuals` for
any family (the randomized probability integral transform of
`prospicio.models.pit`). The claim count model has the true form:

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

With about a quarter of a claim expected per policy, the deviance residuals form
two clusters (no claim, one claim), skewed and too narrow. The quantile
residuals match a standard normal in mean, spread, symmetry and tails; a
departure would point at the model.

## 4. Validation on held-out data (sections 4.3 and 7)

The monograph requires a test on data the fit did not use: build on a training
set and touch the test set once at the end. Here one fold of five from `k_fold`
is held out, and a "current plan" (the straight line in age) is compared with a
"proposed model" (the GAM) on `wavy_claims`. `lift` sorts the test rows by
predicted claims per year and cuts them into bands of equal exposure (the
quantile plot of section 7.2.1). A good model's actuals track its predictions
and spread widely from the first band to the last. Values are relative to the
test set's average frequency:

```python
from prospicio.models import gini, lift

train, test = folds[0]
yt = [y[i] for i in test]
et = [data["exposure"][i] for i in test]
current = poisson.fit(designs["linear"].select(train), [y[i] for i in train]).predict(designs["linear"].select(test))
proposed = gam.fit(gd.select(train), [y[i] for i in train]).predict(gd.select(test))
avg = sum(yt) / sum(et)
by_band = {}
for name, pred in [("current", current), ("proposed", proposed)]:
    bands = lift(yt, pred, et, bands=5)
    by_band[f"{name}_pred"] = [b["expected"] / b["exposure"] / avg for b in bands]
    by_band[f"{name}_act"] = [b["actual"] / b["exposure"] / avg for b in bands]
(GT(pd.DataFrame({"band": range(1, 6), **by_band}), rowname_col="band")
 .tab_header("Lift on the test set", "Claim frequency relative to the test average, 5 bands of equal exposure")
 .cols_label(current_pred="current predicted", current_act="current actual",
             proposed_pred="proposed predicted", proposed_act="proposed actual")
 .fmt_number(list(by_band), decimals=2))
```

```text
   current predicted  current actual  proposed predicted  proposed actual
1               0.85            0.91                0.49             0.46
2               0.97            0.39                0.77             0.50
3               1.16            1.13                1.16             1.03
4               1.35            1.43                1.70             1.46
5               1.56            1.12                2.15             1.55
```

The current plan's actuals do not rise with its predictions (band 2 is the
lowest). The proposed model's actuals rise in every band.

The double lift chart (section 7.2.2) sorts by the ratio of the predictions of
the two models. The end bands hold the risks on which the models disagree most.
The loss ratio chart (section 7.2.3) sorts by the proposed model over the
current premium. If the proposed model improves on the plan, the loss ratios
rise across the bands. `double_lift` and `loss_ratio_chart` draw both, in bands
of about equal exposure. The Gini index (section 7.2.4) sorts by predicted rate and plots the cumulative
share of exposure against the cumulative share of losses (the Lorenz curve); it
is twice the area between the curve and the diagonal. `gini` computes it with
exposure and takes ties together. Here the rate depends only on age, so policies
of the same age tie, but only up to floating-point bits that differ between
platforms and between the Python and R fits. Rounding the rates to 10 decimals
makes the ties exact:

```python
from prospicio.models import double_lift, loss_ratio_chart

dl = double_lift(yt, proposed, current, et, bands=5)
lr = loss_ratio_chart(yt, proposed, current, et, bands=5)
rate = lambda pred: [round(p / e, 10) for p, e in zip(pred, et)]  # equal ages tie exactly
print(f"Gini on the test set: current {gini(yt, rate(current), et):.3f}, proposed {gini(yt, rate(proposed), et):.3f}")
```

```text
Gini on the test set: current 0.121, proposed 0.256
```

```python
lift_bands = pd.DataFrame({
    "band": list(range(1, 6)),
    "current": [b["b_relativity"] for b in dl],
    "proposed": [b["a_relativity"] for b in dl],
    "actual": [b["actual_relativity"] for b in dl],
    "loss_ratio": [b["loss_ratio"] for b in lr],
})
(GT(lift_bands, rowname_col="band")
 .tab_header("Double lift and loss ratio, 120 test policies in 5 bands",
             "Relativities to the overall average; loss ratio = actual / current premium")
 .tab_spanner("relativity", ["current", "proposed", "actual"])
 .cols_label(loss_ratio="loss ratio")
 .fmt_number(["current", "proposed", "actual", "loss_ratio"], decimals=2))
```

```text
   current  proposed  actual  loss ratio
1     0.96      0.42    0.48        0.42
2     1.03      0.66    0.65        0.54
3     1.01      0.97    1.32        1.10
4     1.01      1.36    0.75        0.62
5     0.98      1.59    1.82        1.57
```

```python
fig, ax = plt.subplots()
ax.plot(lift_bands["band"], lift_bands["actual"], color="0.4", marker="o", label="actual")
ax.plot(lift_bands["band"], lift_bands["current"], marker="o", label="current plan")
ax.plot(lift_bands["band"], lift_bands["proposed"], marker="o", label="proposed model")
ax.axhline(1.0, color="0.75", linestyle=":")
ax.set_xlabel("band (sorted by proposed / current prediction)")
ax.set_ylabel("relative claim frequency (overall = 1)")
ax.set_xticks(lift_bands["band"])
ax.set_title("Double lift chart: the actuals follow the proposed model")
ax.legend(loc="upper left")
```

In both end bands the actuals run with the proposed model, and the current plan
predicts about 1 in every band: where the two models disagree, the proposed
model was right. The loss ratio chart says the same, with one reversal in the
120 test policies.

For a yes or no outcome, section 7.3 adds the ROC curve. Table 13's confusion
matrices give the rates the monograph quotes:

```python
roc_rates = pd.DataFrame({
    "rate": ["sensitivity at 50%", "specificity at 50%", "false positive rate at 50%",
             "sensitivity at 25%", "false positive rate at 25%"],
    "prospicio": [39 / 109, 673 / 704, 1 - 673 / 704, 75 / 109, 103 / 704],
    "book": [0.358, 0.956, 0.044, 0.688, 0.146],
})
(GT(roc_rates, rowname_col="rate")
 .tab_header("Rates from the confusion matrices of Table 13", "Threshold on the predicted claim probability")
 .fmt_number(["prospicio", "book"], rows=[0, 1, 2], decimals=3)
 .fmt_percent(["prospicio", "book"], rows=[3, 4], decimals=1))
```

```text
                            prospicio   book
sensitivity at 50%              0.358  0.358
specificity at 50%              0.956  0.956
false positive rate at 50%      0.044  0.044
sensitivity at 25%              68.8%  68.8%
false positive rate at 25%      14.6%  14.6%
```

Footnote 17 states that the area under the ROC curve is `0.5 x normalized Gini +
0.5`, where the normalized Gini is the model's Gini over that of a perfect model
(each row predicted by its own outcome). A logistic model of whether a policy
had a claim checks the identity: the area comes from `roc` (ties count as half,
as in the Mann-Whitney statistic) and the normalized Gini from
`normalized_gini`:

```python
from prospicio.models import normalized_gini, roc

had_claim = [1.0 if k > 0 else 0.0 for k in data["wavy_claims"]]
logit_design = Design([[1.0] * n, ages, [max(0.0, a - 38) for a in ages], [max(0.0, a - 62) for a in ages], log_e],
                      ["(Intercept)", "age", "h38", "h62", "log exposure"])
prob = Glm("binomial", "logit").fit(logit_design, had_claim).fitted
auc = roc(had_claim, prob)["auc"]
print(f"AUROC {auc:.6f}; 0.5 x normalized Gini + 0.5 = {0.5 * normalized_gini(had_claim, prob) + 0.5:.6f}")
```

```text
AUROC 0.743339; 0.5 x normalized Gini + 0.5 = 0.743339
```

For a binary outcome, the Lorenz curve and the ROC curve carry the same
information, so the monograph warns against reporting both as separate evidence.

## 5. Stability and regularization (sections 6.4 and 10.5)

Section 6.4 checks stability by refitting on the cross-validation training sets
(or bootstrap samples) and watching the coefficients:

```python
c = coding("age")
full = c.design(data, offset=log_e)
stab = []
for k, (tr, _) in enumerate(folds):
    fit = poisson.fit(full.select(tr), [data["claims"][i] for i in tr])
    stab.append([k, *fit.coefficients[1:]])
(GT(pd.DataFrame(stab, columns=["fold", "age", "B", "C", "D"]), rowname_col="fold")
 .tab_header("Coefficients refitted without one fold", "Claim count model, log scale")
 .fmt_number(["age"], decimals=4, force_sign=True)
 .fmt_number(["B", "C", "D"], decimals=3, force_sign=True))
```

```text
       age       B       C       D
0  -0.0091  +0.166  -0.314  +0.274
1  -0.0097  -0.099  -0.284  +0.122
2  -0.0112  +0.207  -0.318  +0.061
3  -0.0093  +0.052  -0.122  +0.345
4  -0.0132  +0.051  -0.270  +0.187
```

The age slope stays between -0.009 and -0.013. The region coefficients move by up
to 0.3, about their standard errors. The 155 claims leave them unsettled.

Section 6.4 also looks for single records with an outsized pull. Cook's distance
is how far all the coefficients move, scaled by their covariance, when the fit
leaves out one row; it is large when a row has both a large residual and high
leverage (hat value). `GlmFit.diagnostics` returns both for the claim count
model of section 3:

```python
cook = claims_diag["cooks_distance"]
top = sorted(range(n), key=lambda i: -cook[i])[:3]
influential = pd.DataFrame({
    "rank": [1, 2, 3],
    "claims": [int(data["claims"][i]) for i in top],
    "expected": [claims_fit.fitted[i] for i in top],
    "region": [data["region"][i] for i in top],
    "age": [data["age"][i] for i in top],
    "leverage": [claims_diag["leverage"][i] for i in top],
    "cook": [cook[i] for i in top],
})
(GT(influential, rowname_col="rank")
 .tab_header("The three most influential policies", "Claim count model")
 .cols_label(cook="Cook's D")
 .fmt_number(["expected"], decimals=2)
 .fmt_number(["age"], decimals=1)
 .fmt_number(["leverage", "cook"], decimals=4))
```

```text
   claims  expected  region   age  leverage  Cook's D
1       3      0.32       B  69.9    0.0116    0.0522
2       2      0.34       C  22.2    0.0201    0.0334
3       2      0.31       C  33.7    0.0151    0.0292
```

```python
rest = [i for i in range(n) if i != top[0]]
refit = poisson.fit(claims_design.select(rest), [data["claims"][i] for i in rest])
print(f"region B coefficient {claims_fit.coefficients[2]:+.3f}, without that policy {refit.coefficients[2]:+.3f}")
```

```text
region B coefficient +0.078, without that policy +0.018
```

The most influential policy had three claims in 0.9 years where 0.32 were
expected. No leverage is high (the average is 5/600, 0.008), so the pull comes
from the residual. With 155 claims in all, one policy can shift a thin level's
relativity by several points, which is why the monograph says to look.

The elastic net (section 10.5) minimizes the deviance plus `lambda` times a mix
of the coefficients' absolute values (lasso) and squares (ridge), on
standardized columns with the intercept unpenalized. As `lambda` falls from the
value that zeroes every coefficient, variables enter one by one, and
cross-validation picks `lambda`. `ElasticNet` minimizes glmnet's objective, for
every family including the gamma and the Tweedie. The claim count model gets
age, region and two columns unrelated to claims (`trials` and `n_sev`, drawn
independently), and the lasso sorts them:

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
```

```text
lambda_max 0.05114, CV minimum at 0.02518, one-SE choice 0.05114
```

```python
paths = []
for lam_name, lam in [("lambda_min", cv.lam_min), ("smaller", lams[-1])]:
    paths.append([lam_name, *net.with_lam(lam).fit(X, data["claims"]).coefficients[1:]])
paths.append(["GLM", *poisson.fit(X, data["claims"]).coefficients[1:]])
(GT(pd.DataFrame(paths, columns=["fit"] + names[1:]), rowname_col="fit")
 .tab_header("Lasso coefficients at two penalties, and the plain GLM", "Claim count model, log scale")
 .fmt_number(names[1:], decimals=4, force_sign=True))
```

```text
                age   trials    n_sev        B        C        D
lambda_min  -0.0055   0.0000   0.0000   0.0000  -0.0486   0.0000
smaller     -0.0106  -0.0254  +0.0629  +0.0696  -0.2758  +0.1984
GLM         -0.0106  -0.0255  +0.0631  +0.0700  -0.2762  +0.1990
```

At the cross-validated `lambda` the lasso keeps age (shrunk by half) and a small
region C discount, and drops the rest, both noise columns included. With a
cross-validation curve this flat, the one-standard-error rule drops everything.
At a small `lambda` it is close to the unpenalized GLM, which gives the noise
columns small non-zero coefficients. Shrinking thin levels towards zero is the
credibility-like behaviour the monograph points to.
