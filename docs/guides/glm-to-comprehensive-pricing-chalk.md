# Beyond the GLM: CAS Monograph 16 with prospicio

<!-- site: execute -->

CAS Monograph 16 is *From GLMs to Comprehensive Insurance Pricing: Techniques
and Challenges* (Alan Chalk, David Deacon, Montserrat Guillen and Max
Martinelli, 2025). It starts where the GLM monograph of Goldburd and others
stops (the [Goldburd guide](glm-ratemaking-goldburd.md) covers the GLM) and asks
what a pricing team does next. This guide covers chapters 3 to 6: measuring a
model on unseen data, penalized regression, step and hinge functions, and
factors with thousands of levels. The second part,
[`glm-to-comprehensive-pricing-chalk-2.md`](glm-to-comprehensive-pricing-chalk-2.md),
covers chapters 7 to 10. Table and section numbers are the monograph's.

**What this guide reproduces.** The monograph's case studies use data that is
not available offline (the FAA aircraft registry joined to NTSB accident
records, UK road accident counts, and a simulated auto portfolio whose rows are
not published), so this guide does not recompute their coefficients or plots. It
reproduces every worked example whose inputs the monograph prints, each with the
monograph's value beside prospicio's. To show the methods on data, it simulates
a motor portfolio with a known ground truth, as section 2.3 of the monograph
does (section 1.2 builds it with a fixed seed). Models are compared by ranking
and by pseudo-R2 rounded to three decimals, because gradient boosting libraries
change the later digits between versions.

## 1. Measuring performance (chapter 3)

### 1.1 Pseudo-R2

The monograph scores every model by the share of the possible improvement in
log-likelihood that it achieves, from the null model (the mean for everyone) to
the saturated model (each record at its own outcome). Deviance is twice the gap
to the saturated model, so this share is `1 - D_model / D_null`, the deviance
explained. Table 3.1 has four records, two claims and predictions 0.2 to 0.8:

```python
import math

import pandas as pd
from great_tables import GT
from prospicio.models import deviance

y = [0.0, 0.0, 1.0, 1.0]
mu = [0.2, 0.4, 0.6, 0.8]
null = [sum(y) / len(y)] * len(y)


def loglik(y, mu):
    """Poisson log-likelihood without the log(y!) constant, as in section 3.1."""
    return sum(yi * math.log(mi) - mi for yi, mi in zip(y, mu))


ll_model, ll_null = loglik(y, mu), loglik(y, null)
ll_sat = sum(yi * math.log(yi) - yi for yi in y if yi > 0)
d_model, d_null = deviance("poisson", y, mu), deviance("poisson", y, null)
check = pd.DataFrame({
    "measure": ["log-likelihood, model", "log-likelihood, null", "log-likelihood, saturated",
                "pseudo-R2 from log-likelihoods", "pseudo-R2 from deviances"],
    "prospicio": [ll_model, ll_null, ll_sat, (ll_model - ll_null) / (ll_sat - ll_null),
                  1 - d_model / d_null],
    "book": ["-2.734", "-3.386", "-2", "47%", "47%"],
})
(GT(check, rowname_col="measure")
 .tab_header("Table 3.1: pseudo-R2 of four records", "Poisson log-likelihood without the log(y!) constant")
 .fmt_number("prospicio", decimals=3))
```

```text
                                prospicio    book
log-likelihood, model              -2.734  -2.734
log-likelihood, null               -3.386  -3.386
log-likelihood, saturated          -2.000      -2
pseudo-R2 from log-likelihoods      0.471     47%
pseudo-R2 from deviances            0.471     47%
```

`deviance` also takes prior weights, for the exposure-weighted version of
section 3.7.2.

Two more checks are arithmetic on printed values (Table 3.6 and the rebasing
factor of section 3.7.4):

```python
(GT(pd.DataFrame({
    "quantity": ["Table 3.6, linear predictor", "Table 3.6, frequency", "Section 3.7.4, rebasing factor"],
    "prospicio": [-2.447 + 0.410 + 5 * -0.086, math.exp(-2.467), 0.004593 / 0.004579],
    "book": ["-2.467", "0.085", "1.0031"],
}), rowname_col="quantity")
 .tab_header("Arithmetic on printed values")
 .fmt_number("prospicio", rows=[0, 1], decimals=3)
 .fmt_number("prospicio", rows=[2], decimals=4))
```

```text
                                prospicio    book
Table 3.6, linear predictor        -2.467  -2.467
Table 3.6, frequency                0.085   0.085
Section 3.7.4, rebasing factor     1.0031  1.0031
```

### 1.2 A portfolio with a known truth, and folds from the policy number

The simulated portfolio takes the place of the monograph's case studies: 20,000
policies with driver age, vehicle age, region, power, a 300-level make, a noise
feature and a young-driver, high-power interaction. The true log frequency has a
steep young-driver effect and a rise after age 60, a vehicle-age effect that
falls until age 6 and is flat after, region and power effects, a make effect for
each of 300 makes (make sizes fall like a Zipf law) and an extra load for young
drivers of high-power cars.

`noise` has no effect. `exper` (years since the driving test) is highly
correlated with age and has no effect of its own.

Section 3.7.3 recommends folds from the policy number, so that anyone can
rebuild them and all records of one policy fall in one fold. Here the fold is
the last digit of the row number. Folds 7 to 9 (30%) are the test data; folds 0
to 6 are the seven cross-validation folds.

```python
import numpy as np

rng = np.random.default_rng(2025)
n = 20_000
age = rng.integers(18, 81, n)
veh_age = rng.integers(0, 21, n)
power = rng.choice(["low", "mid", "high"], n, p=[0.5, 0.35, 0.15])
region = rng.choice(list("ABCDEF"), n, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
zipf = 1 / np.arange(1, 301) ** 1.1
make = rng.choice(300, n, p=zipf / zipf.sum())
make_effect = rng.normal(0, 0.4, 300)
noise = rng.normal(size=n)
exper = np.maximum(age - 17 - rng.poisson(2, n), 0)
exposure = rng.uniform(0.5, 1.0, n).round(3)
true_lp = (
    np.log(0.25)
    + 0.9 * np.exp(-(age - 18) / 6)
    + 0.0015 * np.maximum(age - 60, 0) ** 1.5
    - 0.08 * np.minimum(veh_age, 6)
    + np.select([region == "B", region == "C", region == "F"], [0.15, -0.2, 0.3], 0.0)
    + make_effect[make]
    + np.where(power == "high", 0.25, 0.0)
    + np.where((power == "high") & (age < 25), 0.6, 0.0)
)
claims = rng.poisson(np.exp(true_lp) * exposure).astype(float)
fold = np.arange(n) % 10
train = fold < 7
print(f"{n} policies, {exposure.sum():.0f} car-years, {claims.sum():.0f} claims")
print(f"frequency {claims.sum() / exposure.sum():.4f}; makes with under 20 policies: "
      f"{(np.bincount(make, minlength=300) < 20).sum()}")
print(f"correlation of age and exper {np.corrcoef(age, exper)[0, 1]:.3f}")
```

```text
20000 policies, 15000 car-years, 3659 claims
frequency 0.2439; makes with under 20 policies: 166
correlation of age and exper 0.997
```

The null model is the mean frequency times each record's exposure. `cv_score`
trains on six of the seven folds, scores the seventh and averages, for any
function that fits and predicts (a GLM, a GAM or a boosting model).

```python
from prospicio.models import Design


def pseudo_r2(y, mu, expo):
    y, mu, expo = (np.asarray(v, float) for v in (y, mu, expo))
    null = expo * y.sum() / expo.sum()
    return 1 - deviance("poisson", y.tolist(), mu.tolist()) / deviance("poisson", y.tolist(), null.tolist())


def rows(mask):
    return np.flatnonzero(mask).tolist()


def cv_score(fit_predict, X, folds=range(7)):
    scores = []
    for k in folds:
        fit_rows, val = rows(train & (fold != k)), rows(fold == k)
        mu = fit_predict(X.select(fit_rows), claims[fit_rows].tolist(), X.select(val))
        scores.append(pseudo_r2(claims[val], mu, exposure[val]))
    return float(np.mean(scores)), scores


def columns(*cols):
    return [np.asarray(c, float).tolist() for c in cols]


ones = np.ones(n)
log_expo = np.log(exposure).tolist()
region_d = [(region == r).astype(float) for r in "BCDEF"]
power_d = [(power == p).astype(float) for p in ["mid", "high"]]
factor_names = [f"region[{r}]" for r in "BCDEF"] + ["power[mid]", "power[high]"]
truth = np.exp(true_lp) * exposure
print(f"pseudo-R2 of the true frequencies on the test data {pseudo_r2(claims[~train], truth[~train], exposure[~train]):.3f}")
```

```text
pseudo-R2 of the true frequencies on the test data 0.089
```

A perfect model explains only a few percent of the deviance of individual claim
counts, because most of each count is chance (section 3.7.1; the monograph's
perfect model scored 0.051), so compare models on the same data.

### 1.3 Null, baseline and benchmark models

The null model predicts the training mean. Its validation pseudo-R2 is slightly
negative, not zero, because the training mean is not the validation fold's mean
(Table 3.3). Rebasing to the fold's mean (section 3.7.4) makes it exactly zero.

The baseline is a Poisson GLM with age and vehicle age as straight lines and
region and power as factors (section 3.5). The benchmark is LightGBM through
`prospicio.boosting.Booster` (section 3.6), with the make as a plain number so
the trees can split on it.

```python
from prospicio.boosting import Booster
from prospicio.models import Glm

X_null = Design(columns(ones), ["(Intercept)"], offset=log_expo)
X_base = Design(columns(ones, age, veh_age, *region_d, *power_d),
                ["(Intercept)", "age", "veh_age", *factor_names], offset=log_expo)
X_tree = Design(columns(age, veh_age, np.unique(region, return_inverse=True)[1],
                        np.unique(power, return_inverse=True)[1], make, noise),
                ["age", "veh_age", "region", "power", "make", "noise"], offset=log_expo)


def glm(d, y, new):
    return Glm("poisson").fit(d, y).predict(new)


def gbm(d, y, new, rounds=150):
    params = {"num_leaves": 7, "min_data_in_leaf": 200, "num_threads": 1}
    return Booster("poisson", n_rounds=rounds, learning_rate=0.05, params=params, seed=1).fit(d, y).predict(new)


null_cv, null_folds = cv_score(glm, X_null)
print("null model by fold     ", [f"{s:.4f}" for s in null_folds])
fit_rows, val = rows(train & (fold != 0)), rows(fold == 0)
mu0 = np.asarray(glm(X_null.select(fit_rows), claims[fit_rows].tolist(), X_null.select(val)))
rebased = mu0 * claims[val].sum() / mu0.sum()
print(f"fold 0 rebased          {abs(pseudo_r2(claims[val], rebased, exposure[val])):.4f}")
print(f"baseline GLM  cv        {cv_score(glm, X_base)[0]:.3f}")
print(f"benchmark GBM cv        {cv_score(gbm, X_tree)[0]:.3f}")
```

```text
null model by fold      ['-0.0000', '-0.0000', '-0.0031', '-0.0000', '-0.0009', '-0.0012', '-0.0028']
fold 0 rebased          0.0000
baseline GLM  cv        0.021
benchmark GBM cv        0.045
```

The order is the monograph's: null, GLM, boosting model.

## 2. Penalized regression (chapter 4)

### 2.1 Why fewer parameters can predict better

In the rare-car example of section 4.1, ten insurers each see one of ten cars
for a year, and one car has a claim. An unpenalized GLM at each insurer predicts
100% or 0%: right on average (10%), far off each time. The spread of the ten
predictions is the variance of the model.

```python
preds = np.array([1.0] + [0.0] * 9)
print(f"mean {preds.mean():.2f}, variance {preds.var():.2f}, sd {preds.std():.2f} (printed 0.09, 0.30)")
```

```text
mean 0.10, variance 0.09, sd 0.30 (printed 0.09, 0.30)
```

A model that ignores this car and predicts the 5% average for every make has
bias but no variance, and every insurer is closer to the truth with it.
Penalized regression trades bias for variance continuously, by maximizing the
log-likelihood less a penalty on the coefficients:

- ridge: the sum of their squares,
- lasso: the sum of their absolute values,
- elastic net: a mix of the two, with weight `alpha` on the lasso part.

prospicio's `ElasticNet` minimizes the glmnet objective: half the mean deviance
plus `lam` times the penalty. The intercept is not penalized and the columns are
standardized first. Its `lam` and `alpha` are the monograph's lambda and alpha
(section 4.6.3).

### 2.2 Ridge in closed form, and why features are standardized

For one feature, a normal response and no intercept, ridge divides the mean of
`x * y` by the mean of `x**2` plus `lam` (section 4.7.1), which pulls the least
squares estimate towards zero. In section 4.7.2 the means are 10 and 5 with
length in metres, and `lam` = 5. In centimetres they are 100 and 10,000 times
larger, so the same `lam` shrinks almost nothing:

```python
units = []
for unit, xy, xx, scale in [("metres", 10.0, 5.0, 1.0), ("centimetres", 1_000.0, 50_000.0, 100.0)]:
    ols, ridge_b = xy / xx, xy / (xx + 5.0)
    units.append({"unit": unit, "least squares": ols, "ridge": ridge_b, "frequency at 1 m": ridge_b * scale})
(GT(pd.DataFrame(units), rowname_col="unit")
 .tab_header("Section 4.7.2: ridge on one length in two units", "lam = 5")
 .fmt_number("least squares", decimals=4)
 .fmt_number("ridge", decimals=6)
 .fmt_number("frequency at 1 m", decimals=4))
```

```text
             least squares     ridge  frequency at 1 m
metres              2.0000  1.000000            1.0000
centimetres         0.0200  0.019998            1.9998
```

The monograph prints 0.019 and 1.9 for centimetres: it truncated the exact
values. The units, not the data, set the effect of the penalty.

Standardizing each feature to unit standard deviation before the penalty removes
this. `ElasticNet` does it by default and returns coefficients on the original
scale, so ridge on metres and on centimetres differ by exactly the factor of
100:

```python
from prospicio.models import ElasticNet

x_m = rng.uniform(1.0, 5.0, 200)
y_len = 2.0 * x_m + rng.normal(0, 1, 200)
ridge = ElasticNet("gaussian", alpha=0.0, lam=0.5)
b_m = ridge.fit(Design(columns(np.ones(200), x_m), ["(Intercept)", "x"]), y_len.tolist()).coefficients[1]
b_cm = ridge.fit(Design(columns(np.ones(200), 100 * x_m), ["(Intercept)", "x"]), y_len.tolist()).coefficients[1]
print(f"metres {b_m:.6f}, centimetres x 100 {100 * b_cm:.6f}")
```

```text
metres 1.343762, centimetres x 100 1.343762
```

### 2.3 The lasso sets coefficients to exactly zero

Section 4.7.3: for one standardized feature, the lasso estimate is the least
squares estimate moved towards zero by `lam`, and exactly zero beyond it
(soft-thresholding). With this objective the least squares estimate is the mean
of `x * y`:

```python
x_std = rng.normal(size=500)
x_std = (x_std - x_std.mean()) / x_std.std()
y_soft = 0.3 * x_std + rng.normal(0, 1, 500)
y_soft -= y_soft.mean()
ols = float(np.mean(x_std * y_soft))
X_soft = Design(columns(np.ones(500), x_std), ["(Intercept)", "x"])
soft = []
for lam in [0.0, 0.1, 0.2, 0.4]:
    b = ElasticNet("gaussian", alpha=1.0, lam=lam).fit(X_soft, y_soft.tolist()).coefficients[1]
    soft.append({"lam": f"{lam:.1f}", "lasso": b, "soft threshold": np.sign(ols) * max(abs(ols) - lam, 0.0)})
(GT(pd.DataFrame(soft), rowname_col="lam")
 .tab_header("Section 4.7.3: the lasso is the soft threshold", "One standardized feature")
 .fmt_number(["lasso", "soft threshold"], decimals=4))
```

```text
      lasso  soft threshold
0.0  0.3382          0.3382
0.1  0.2382          0.2382
0.2  0.1382          0.1382
0.4  0.0000          0.0000
```

### 2.4 Paths and choosing the penalty by cross-validation

Tables 4.1 and 4.2 show the coefficient of log aircraft age shrinking as the
penalty grows: smoothly for ridge, to zero for the lasso. The block repeats this
with log driver age. For a Poisson model, records with equal features can be
summed (section 8.3), so the age-only models use 63 ages times 10 folds:

```python
key = age * 10 + fold
cells, cell = np.unique(key, return_inverse=True)
cell_claims = np.bincount(cell, claims)
cell_expo = np.bincount(cell, exposure)
cell_age, cell_fold = cells // 10, cells % 10
cell_off = np.log(cell_expo).tolist()
tr_cells = rows(cell_fold < 7)
X_logage = Design(columns(np.ones(len(cells)), np.log(cell_age)), ["(Intercept)", "log_age"], offset=cell_off)
Xa, ya = X_logage.select(tr_cells), cell_claims[tr_cells].tolist()
paths = {}
for name, a in [("ridge", 0.0), ("lasso", 1.0)]:
    net = ElasticNet("poisson", alpha=a)
    top = ElasticNet("poisson", alpha=1.0).lambda_max(Xa, ya)
    scale = 100.0 if a == 0.0 else 1.0
    path = net.path(Xa, ya, [scale * top * f for f in [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]])
    paths[name] = [fit.coefficients[1] for fit in path]
(GT(pd.DataFrame({"penalty": ["1", "0.3", "0.1", "0.03", "0.01", "0 (GLM)"], **paths}), rowname_col="penalty")
 .tab_header("Tables 4.1 and 4.2: the log-age coefficient along the path",
             "Penalty as a share of the largest one; ridge starts at 100 times the lasso's")
 .fmt_number(["ridge", "lasso"], decimals=3))
```

```text
          ridge   lasso
1        -0.023   0.000
0.3      -0.067  -0.292
0.1      -0.153  -0.373
0.03     -0.275  -0.401
0.01     -0.354  -0.409
0 (GLM)  -0.413  -0.413
```

Section 4.3 chooses `lam` by fitting the path on six folds and scoring the
seventh. `ElasticNet.cross_validate` does this on the policy-number folds. It
returns the `lam` with the best mean validation deviance and, for the
one-standard-error rule of section 4.6.5, the largest `lam` within one standard
error of it:

```python
def cell_splits(cell_rows, cell_folds):
    pos = {r: i for i, r in enumerate(cell_rows)}
    return [([pos[r] for r in cell_rows if cell_folds[r] != k],
             [pos[r] for r in cell_rows if cell_folds[r] == k]) for k in range(7)]


splits = cell_splits(tr_cells, cell_fold)
lasso = ElasticNet("poisson", alpha=0.99)
lams = lasso.lambda_path(Xa, ya, n=30, min_ratio=1e-3)
path_cv = lasso.cross_validate(Xa, ya, lams, splits)
print(f"lam_min at position {lams.index(path_cv.lam_min)} of 30, lam_1se at position {lams.index(path_cv.lam_1se)}")
```

```text
lam_min at position 29 of 30, lam_1se at position 3
```

### 2.5 The relaxed lasso

The relaxed lasso (section 4.5) separates selection from shrinkage: it refits
the features that the lasso selected with less penalty or none. In prospicio
that is a `Glm` on the selected columns (section 3.2).

## 3. Nonlinear effects (chapter 5)

### 3.1 Step and hinge functions, from the printed coefficients

A step function is 1 at or above a threshold and 0 below it. A hinge is `max(0,
x - knot)`. A single step at age 30 gives `log frequency = -5.2815 - 1.4689 x
step_30` (section 5.2). With steps at 5, 15 and 30 the monograph prints four
frequencies:

```python
steps = pd.DataFrame({
    "quantity": ["one step, frequency", "one step, relativity", "three steps, band 1", "three steps, band 2",
                 "three steps, band 3", "three steps, band 4"],
    "prospicio": [math.exp(-5.2815), math.exp(-1.4689),
                  *[math.exp(sum([-4.8292, -0.6881, -0.5711, -0.6237][: i + 1])) for i in range(4)]],
    "book": [0.00509, 0.2302, 0.00783, 0.00394, 0.00222, 0.00119],
})
steps["book / prospicio"] = steps["book"] / steps["prospicio"]
(GT(steps, rowname_col="quantity")
 .tab_header("Section 5.2: frequencies from the printed coefficients", "Bands are cut at ages 5, 15 and 30")
 .fmt_number(["prospicio", "book"], rows=[0, 2, 3, 4, 5], decimals=5)
 .fmt_number(["prospicio", "book"], rows=[1], decimals=4)
 .fmt_number("book / prospicio", decimals=3))
```

```text
                      prospicio     book  book / prospicio
one step, frequency     0.00508  0.00509             1.001
one step, relativity     0.2302   0.2302             1.000
three steps, band 1     0.00799  0.00783             0.980
three steps, band 2     0.00402  0.00394             0.981
three steps, band 3     0.00227  0.00222             0.978
three steps, band 4     0.00122  0.00119             0.979
```

The single step's base frequency differs in the last digit: exp(-5.2815) is
0.005082 (an intercept of -5.2809 or more gives the printed 0.00509).

The three-step frequencies are all about 2% below the printed ones, by almost
the same factor in every band, which points at the intercept: -4.85, not
-4.8292, gives the printed values and keeps the ratios between bands. Without
the data we cannot say which printed number is wrong.

Table 5.7 codes an ordinal feature (pilot quality, five levels) as five
cumulative steps. A level's relativity is the exponential of the running sum of
the coefficients:

```python
coef = [0, 0.515, 0, 0.211, 0.206]
(GT(pd.DataFrame({
    "level": ["1", "2", "3", "4", "5"],
    "coefficient": coef,
    "prospicio": [math.exp(sum(coef[: i + 1])) for i in range(5)],
    "book": ["1", "1.673", "1.673", "2.06", "2.539"],
}), rowname_col="level")
 .tab_header("Table 5.7: cumulative relativities of pilot quality")
 .fmt_number(["coefficient", "prospicio"], decimals=3))
```

```text
   coefficient  prospicio   book
1        0.000      1.000      1
2        0.515      1.674  1.673
3        0.000      1.674  1.673
4        0.211      2.067   2.06
5        0.206      2.540  2.539
```

The exact fourth value is 2.067: Table 5.7 truncates it and Table 5.8 rounds it
to 2.07.

### 3.2 Many steps or hinges, and the lasso picks

Section 5.2 gives the lasso one step at each age and lets cross-validation
decide how many stay (section 5.6 uses 25 equally spaced steps). Here the steps
are every three years from 20 to 77, on the aggregated ages:

```python
knots = list(range(20, 80, 3))
step_names = ["(Intercept)"] + [f"age_stp_{a}" for a in knots]
X_steps = Design(columns(np.ones(len(cells)), *[cell_age >= a for a in knots]), step_names, offset=cell_off)
Xt, yt = X_steps.select(tr_cells), cell_claims[tr_cells].tolist()
net = ElasticNet("poisson", alpha=0.99)
lams = net.lambda_path(Xt, yt, n=25, min_ratio=1e-3)
cvp = net.cross_validate(Xt, yt, lams, splits)
fits = net.path(Xt, yt, lams)
best, simple = fits[lams.index(cvp.lam_min)], fits[lams.index(cvp.lam_1se)]
kept = [a for a, b in zip(knots, best.coefficients[1:]) if b != 0.0]
print(f"{len(knots)} steps offered, {best.df} kept at lam_min, {simple.df} at lam_1se")
print("kept:", kept)
```

```text
20 steps offered, 8 kept at lam_min, 5 at lam_1se
kept: [20, 23, 26, 29, 32, 62, 65, 77]
```

The cross-validation curve, with the two chosen penalties marked:

```python
import matplotlib.pyplot as plt

x = np.log10(cvp.lams)
fig, ax = plt.subplots()
ax.errorbar(x, cvp.mean, yerr=cvp.se, color="0.75", capsize=2, label="mean validation deviance")
ax.axvline(np.log10(cvp.lam_min), linestyle="--", label=f"lam_min ({best.df} steps)")
ax.axvline(np.log10(cvp.lam_1se), linestyle=":", label=f"lam_1se ({simple.df} steps)")
ax.set_xlabel("log10 of the penalty lam")
ax.set_ylabel("deviance on the validation fold")
ax.set_title("Cross-validation of the lasso on 20 age steps")
ax.legend()
```

The next block compares with the truth by age (the exposure-weighted average of
the policies' true frequencies) the lasso at `lam_min`, the relaxed lasso (the
kept steps refitted by an unpenalized `Glm`, section 4.5) and a P-spline GAM
(`Gam`, which the monograph uses for its spatial second stage in chapter 8):

```python
from prospicio.models import Gam

grid = np.arange(18, 81)
true_by_age = np.bincount(age - 18, np.exp(true_lp) * exposure) / np.bincount(age - 18, exposure)
grid_steps = Design(columns(np.ones(63), *[grid >= a for a in knots]), step_names)
lasso_curve = np.array(best.predict(grid_steps))
kept_names = ["(Intercept)"] + [f"age_stp_{a}" for a in kept]
X_kept = Design(columns(np.ones(len(cells)), *[cell_age >= a for a in kept]), kept_names, offset=cell_off)
relaxed = Glm("poisson").fit(X_kept.select(tr_cells), cell_claims[tr_cells].tolist())
relaxed_curve = np.array(relaxed.predict(Design(columns(np.ones(63), *[grid >= a for a in kept]), kept_names)))
X_age = Design(columns(np.ones(len(cells)), cell_age), ["(Intercept)", "age"], offset=cell_off)
gam_fit = Gam(Glm("poisson"), [("age", 12)]).fit(X_age.select(tr_cells), cell_claims[tr_cells].tolist())
gam_curve = np.array(gam_fit.predict(Design(columns(np.ones(63), grid), ["(Intercept)", "age"])))
ages = [18, 20, 23, 26, 35, 50, 65, 80]
(GT(pd.DataFrame({
    "age": [str(a) for a in ages],
    "truth": [true_by_age[a - 18] for a in ages],
    "lasso": [lasso_curve[a - 18] for a in ages],
    "relaxed": [relaxed_curve[a - 18] for a in ages],
    "GAM": [gam_curve[a - 18] for a in ages],
}), rowname_col="age")
 .tab_header("Claim frequency by age: truth and three fits", "Claims per policy year")
 .fmt_number(["truth", "lasso", "relaxed", "GAM"], decimals=3))
```

```text
    truth  lasso  relaxed    GAM
18  0.565  0.531    0.553  0.530
20  0.481  0.408    0.415  0.449
23  0.363  0.304    0.310  0.355
26  0.264  0.256    0.260  0.290
35  0.216  0.207    0.200  0.206
50  0.207  0.207    0.200  0.195
65  0.215  0.236    0.239  0.228
80  0.256  0.239    0.253  0.249
```

```python
print(f"GAM effective degrees of freedom {gam_fit.edf:.1f}")
```

```text
GAM effective degrees of freedom 5.7
```

```python
fig, ax = plt.subplots()
ax.plot(grid, true_by_age, color="0.75", linewidth=3, label="truth")
ax.plot(grid, lasso_curve, label="lasso at lam_min")
ax.plot(grid, relaxed_curve, label="relaxed lasso")
ax.plot(grid, gam_curve, label="GAM")
ax.set_xlabel("driver age (years)")
ax.set_ylabel("claims per policy year")
ax.set_title("Claim frequency by age: truth and three fits")
ax.legend()
```

All three fits find the young-driver slope and the rise after 60 without being
told where they are. Lasso shrinkage pulls the extremes towards the middle (as
section 5.2 sees above aircraft age 20), and the relaxed fit removes part of
that. The monograph prefers steps and hinges because the result is a short
rating table. A GAM gives a smooth curve, which you tabulate afterwards.

Section 5.8.1 links steps to the fused lasso: a step at each level is a dummy
at each level with each step the difference between neighbouring dummies, so a
lasso penalty on steps penalizes the jumps between neighbours.

### 3.3 Monotone effects

Section 5.7.2 makes an ordinal effect monotone with non-negative step
coefficients. A GAM can set the shape directly: `"mpd"` is a monotone decreasing
P-spline (from scam). The block fits it to vehicle age, whose true effect falls
until age 6 and is flat after:

```python
veh_cells, veh_cell = np.unique(veh_age * 10 + fold, return_inverse=True)
veh_train = rows(veh_cells % 10 < 7)
X_veh = Design(columns(np.ones(len(veh_cells)), veh_cells // 10), ["(Intercept)", "veh_age"],
               offset=np.log(np.bincount(veh_cell, exposure)).tolist())
mono = Gam(Glm("poisson"), [{"column": "veh_age", "n_basis": 8, "bs": "mpd"}]).fit(
    X_veh.select(veh_train), np.bincount(veh_cell, claims)[veh_train].tolist())
curve = np.array(mono.predict(Design(columns(np.ones(21), np.arange(21)), ["(Intercept)", "veh_age"])))
show = [0, 2, 4, 6, 10, 15, 20]
(GT(pd.DataFrame({
    "vehicle age": [str(v) for v in show],
    "fitted": (curve / curve[0])[show],
    "true": [math.exp(-0.08 * min(v, 6)) for v in show],
}), rowname_col="vehicle age")
 .tab_header("Relativity of claim frequency to vehicle age 0", "Monotone P-spline against the true effect")
 .fmt_number(["fitted", "true"], decimals=2))
```

```text
    fitted  true
0     1.00  1.00
2     0.89  0.85
4     0.80  0.73
6     0.75  0.62
10    0.70  0.62
15    0.64  0.62
20    0.61  0.62
```

The monotone fit bends more gradually than the true hinge and keeps falling a
little after age 6, but it cannot turn back up. A constraint hides a reversal if
the data has one, so use it where the shape is known (claim history,
deductibles), not to make a factor look tidy.

## 4. High cardinality categorical variables (chapter 6)

### 4.1 Target encoding, and keeping it out of its own fold

A factor with thousands of levels (aircraft make, city, zip code) is a problem
as dummies in a GLM: it uses much memory and gives unstable estimates for thin
levels. Target encoding, the main tool of chapter 6, replaces each level by one
number, its actual-to-expected ratio shrunk towards 1 by the credibility factor
`z = n / (n + K)`, where `n` is the level's exposure. The log of that number is
the feature.

Encoding a level from its own records leaks the target into the feature (section
6.3), so a record in fold `k` is encoded only from the other folds. Table 6.1
has one make with six aircraft in three folds:

```python
te_fold = np.array([1, 1, 2, 2, 3, 3])
te_acc = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
encodings = []
for k, printed in zip([1, 2, 3], [0.75, 0.50, 0.25]):
    other = te_fold != k
    encodings.append({"fold": str(k), "accidents elsewhere": te_acc[other].sum(), "aircraft elsewhere": other.sum(),
                      "encoding": te_acc[other].mean(), "book": printed})
(GT(pd.DataFrame(encodings), rowname_col="fold")
 .tab_header("Table 6.1: each fold encoded from the other folds")
 .fmt_integer(["accidents elsewhere", "aircraft elsewhere"])
 .fmt_number(["encoding", "book"], decimals=2))
```

```text
   accidents elsewhere  aircraft elsewhere  encoding  book
1                    3                   4      0.75  0.75
2                    2                   4      0.50  0.50
3                    1                   4      0.25  0.25
```

### 4.2 The credibility constant, and the printed encodings

Section 6.6 estimates `K` as the ratio of the within-make variance (an
aircraft-year's accident count, Bernoulli at the mean rate 0.004584) to the
between-make variance, which the monograph estimates from the mean residuals of
large makes as 7.038492e-6:

```python
p = 0.004584
within = p * (1 - p)
K_est = within / 7.038492e-6
(GT(pd.DataFrame({
    "quantity": ["within variance", "K", "credibility at 1,000 aircraft-years"],
    "prospicio": [within, K_est, 1000 / (1000 + 648)],
    "book": ["0.004562987", "648", "0.61"],
}), rowname_col="quantity")
 .tab_header("Section 6.6: the credibility constant")
 .fmt_number("prospicio", rows=[0], decimals=9)
 .fmt_number("prospicio", rows=[1], decimals=0)
 .fmt_number("prospicio", rows=[2], decimals=2))
```

```text
                                       prospicio         book
within variance                      0.004562987  0.004562987
K                                            648          648
credibility at 1,000 aircraft-years         0.61         0.61
```

Table 6.5 lists encoded makes at `K` = 1,024, the monograph's final value. The
next block recomputes them from the printed inputs:

```python
table_6_5 = [("DJI", 11900, 0.0, 0.9208, 0.079), ("TEXTRON AVIATION INC", 5483, 0.24356, 0.8426, 0.363),
             ("MARLIN JOHN E", 13, 186.59373, 0.0122, 3.266), ("STUMP GREAT LAKES", 12, 223.74936, 0.0116, 3.579)]
encoded = []
for make_name, expo, ave, z_printed, te_printed in table_6_5:
    z = expo / (expo + 1024)
    encoded.append({"make": make_name, "z": z, "book z": z_printed, "encoding": 1 - z + z * ave,
                    "from book z": 1 - z_printed + z_printed * ave, "book encoding": te_printed})
(GT(pd.DataFrame(encoded), rowname_col="make")
 .tab_header("Table 6.5: encoded makes at K = 1,024")
 .fmt_number(["z", "book z"], decimals=4)
 .fmt_number(["encoding", "from book z", "book encoding"], decimals=3))
```

```text
                           z  book z  encoding  from book z  book encoding
DJI                   0.9208  0.9208     0.079        0.079          0.079
TEXTRON AVIATION INC  0.8426  0.8426     0.363        0.363          0.363
MARLIN JOHN E         0.0125  0.0122     3.327        3.264          3.266
STUMP GREAT LAKES     0.0116  0.0116     3.580        3.584          3.579
```

DJI and Textron match. Marlin's printed credibility (0.0122) implies about
12.65 aircraft-years, not 13, so the exposure column is rounded (the encoding
from the printed `z` is 3.264). Stump Great Lakes differs for the same reason.

Section 6.7.1's worked sentence multiplies Stump's `z` by 222.22, but Table 6.5
has 223.75, which gives the printed 3.58: the text is wrong. The table's caption
also describes AvE upside down (its values are actual over expected).

Two smaller checks come from the same chapter: the Bell helicopter example of
section 6.7.2 (16 per thousand against 4 for the overall mean, against 20 for
the GLM prediction for helicopters), and the logistic blending of section 6.7.6
at a level with no data, which should get no credibility but gets some:

```python
print(f"Bell: {16 / 4:.2f} against the mean, {16 / 20:.2f} against the GLM (printed 4 and 0.80)")
for smoothing in [100, 500]:
    print(f"logistic credibility at n = 0, inflection 500, smoothing {smoothing}: "
          f"{1 / (1 + math.exp((500 - 0) / smoothing)):.3f}")
```

```text
Bell: 4.00 against the mean, 0.80 against the GLM (printed 4 and 0.80)
logistic credibility at n = 0, inflection 500, smoothing 100: 0.007
logistic credibility at n = 0, inflection 500, smoothing 500: 0.269
```

The second value is section 6.7.6's "about 25%" for a level with no data.

### 4.3 Ridge regression on dummies is credibility

Section 6.8.1: ridge on the dummies of one factor, with a centred normal response and no intercept, gives each level the
coefficient `n_k (mean_k - mean) / (n_k + lambda)`, a credibility-weighted
deviation from the overall mean. prospicio minimizes the mean squared error over
two plus `lam` over two times the sum of squares, so the monograph's `lambda` is
`lam` times the number of records. A design with no all-ones column has no
intercept, so `ElasticNet` fits exactly that model:

```python
sizes = [1, 5, 50, 400]
levels = np.repeat(np.arange(4), sizes)
y_lvl = rng.normal(np.array([2.0, -1.0, 0.5, 0.0])[levels], 1.0)
yc = y_lvl - y_lvl.mean()
dummies = [(levels == j).astype(float) for j in range(4)]
lam = 0.1
fit = ElasticNet("gaussian", alpha=0.0, lam=lam, standardize=False).fit(
    Design(columns(*dummies), [f"level{j}" for j in range(4)]), yc.tolist())
K = lam * len(yc)
ridge_rows = []
for j, size in enumerate(sizes):
    z = size / (size + K)
    ridge_rows.append({"level": f"level {j}", "records": size, "ridge": fit.coefficients[j],
                       "z x deviation": z * yc[levels == j].mean(), "z": z})
(GT(pd.DataFrame(ridge_rows), rowname_col="level")
 .tab_header("Section 6.8.1: ridge on dummies is credibility", f"lam = {lam}, so lambda = {K:.1f}")
 .fmt_integer("records")
 .fmt_number(["ridge", "z x deviation"], decimals=4, force_sign=True)
 .fmt_number("z", decimals=3))
```

```text
         records    ridge  z x deviation      z
level 0        1  +0.0456        +0.0456  0.021
level 1        5  -0.0470        -0.0470  0.099
level 2       50  +0.2407        +0.2407  0.523
level 3      400  -0.0511        -0.0511  0.898
```

The penalty does the work of the Buhlmann `K`. Credibility theory estimates `K`
from variance components; penalized regression chooses it by cross-validation.

### 4.4 Target encoding the portfolio's 300 makes

On the portfolio, the encoding follows section 6.7.2. A cross-fitted first GLM
without make gives each policy's expected count, so no model predicts a record
that it saw in training. A make's encoding is its shrunk actual-to-expected
ratio, and its log goes into a second-stage GLM with the first-GLM prediction as
offset (section 6.9). For each `K`, seven-fold cross-validation repeats the
whole procedure inside each fold:

```python
expected = np.zeros(n)
for j in range(7):
    fit_j = Glm("poisson").fit(X_base.select(rows(train & (fold != j))), claims[train & (fold != j)].tolist())
    expected[fold == j] = fit_j.predict(X_base.select(rows(fold == j)))
expected[~train] = Glm("poisson").fit(X_base.select(rows(train)), claims[train].tolist()).predict(
    X_base.select(rows(~train)))


def encode(source, K):
    """Each make's shrunk actual-to-expected ratio from the rows in `source`."""
    actual = np.bincount(make[source], claims[source], minlength=300)
    exp_ = np.bincount(make[source], expected[source], minlength=300)
    expo = np.bincount(make[source], exposure[source], minlength=300)
    z = expo / (expo + K)
    return 1 - z + z * np.divide(actual, exp_, out=np.ones(300), where=exp_ > 0)


def make_feature(K, held_out=None):
    """Per policy: a training row in fold j is encoded from the training folds other than j
    (and other than the held-out fold); every other row from all training folds but the held-out one."""
    te = encode(train & (fold != held_out), K)[make]
    for j in range(7):
        if j != held_out:
            te[fold == j] = encode(train & (fold != j) & (fold != held_out), K)[make[fold == j]]
    return np.log(te)


stage1 = {}
for k in range(7):
    fit_k = Glm("poisson").fit(X_base.select(rows(train & (fold != k))), claims[train & (fold != k)].tolist())
    stage1[k] = np.asarray(fit_k.predict(X_base))
results = []
for K in [16, 64, 256, 1024, 4096]:
    scores, betas = [], []
    for k in range(7):
        feature = make_feature(K, k)
        X2 = Design(columns(ones, feature), ["(Intercept)", "log_te"], offset=np.log(stage1[k]).tolist())
        fit_rows, val = rows(train & (fold != k)), rows(fold == k)
        fit2 = Glm("poisson").fit(X2.select(fit_rows), claims[fit_rows].tolist())
        scores.append(pseudo_r2(claims[val], fit2.predict(X2.select(val)), exposure[val]))
        betas.append(fit2.coefficients[1])
    results.append({"K": str(K), "cv pseudo-R2": np.mean(scores), "coefficient": np.mean(betas)})
(GT(pd.DataFrame(results), rowname_col="K")
 .tab_header("Target encoding of the 300 makes", "Second-stage GLM by credibility constant K")
 .fmt_number("cv pseudo-R2", decimals=3)
 .fmt_number("coefficient", decimals=2))
```

```text
      cv pseudo-R2  coefficient
16           0.040         0.82
64           0.041         1.08
256          0.039         1.41
1024         0.037         2.24
4096         0.035         4.98
```

The baseline GLM scored 0.021 without make; the encoded make roughly doubles
that. As in Figure 6.1, the score is flat over a wide range of `K`, best near
64. As in Figure 6.2, the coefficient rises with `K`: a heavily shrunk encoding
needs a coefficient above 1 to restore the spread. The best score and a
coefficient near 1 occur at the same `K`, as section 6.6 expects when the
encoding is correct. The rest of the guide uses `K` = 64.

A generalized linear mixed model (section 6.5) treats make as a random effect
and shrinks it the same way. Section 6.8.2 gives the Bayesian view of ridge,
which is `BayesGlm` with normal priors.

## 5. The printed numbers of chapters 3 to 6 and prospicio's

| Where | Printed | Recomputed | Note |
|---|---|---|---|
| Table 3.1 pseudo-R2 | 47% | 0.471 | matches |
| Section 3.5 linear predictor | -2.467, 0.085 | -2.467, 0.085 | matches |
| Section 3.7.4 rebasing | 1.0031 | 1.0031 | matches |
| Section 4.1 variance | 0.09, 0.30 | 0.09, 0.30 | matches |
| Section 4.7.2 ridge in cm | 0.019, 1.9 | 0.019998, 1.9998 | printed digits truncated |
| Section 5.2 one step | 0.00509, 0.2302 | 0.00508, 0.2302 | relativity matches, the frequency is 0.005082 |
| Section 5.2 three steps | 0.00783 to 0.00119 | 0.00799 to 0.00122 | all about 2% higher, which points at the printed intercept |
| Table 5.7 cumulative | 2.06 | 2.067 | Table 5.8 prints 2.07 |
| Section 6.6 K | 648, 0.61 | 648, 0.61 | matches |
| Table 6.5 encodings | 0.079, 0.363, 3.266, 3.579 | 0.079, 0.363, 3.264, 3.580 | Marlin from its printed `z`, exposures printed rounded, 222.22 in the text is 223.75 |

Next: [part 2](glm-to-comprehensive-pricing-chalk-2.md), chapters 7 to 10.
