# Beyond the GLM: CAS Monograph 16 with prospicio

CAS Monograph 16, *From GLMs to Comprehensive Insurance Pricing: Techniques and
Challenges* (Alan Chalk, David Deacon, Montserrat Guillen and Max Martinelli,
2025), starts where the GLM monograph of Goldburd and others stops. It takes a
GLM as known and asks what a pricing team does next: how to measure a model on
data it has not seen, how penalized regression picks features and shrinks them,
how step and hinge functions let a GLM bend, what to do with a factor that has
thousands of levels, how to handle highly correlated features, how to fit a
model in two stages with offsets, and how to learn from a gradient boosting
model that beats the GLM. The GLM itself (families, links, offsets, the
rating plan from coefficients) is the subject of the Goldburd guide; this one
covers the step beyond it.

This guide explains each chapter in new words and runs its methods in prospicio.
The text is the authors'; what follows is an explanation with the numbers
recomputed. Read the monograph for the reasoning and this guide for the
mechanics. Table and section numbers are the monograph's.

**What can be reproduced.** The monograph's case studies run on data a reader of
this repository does not have offline: the FAA aircraft registry joined to NTSB
accident records (published as an R package on GitHub, 2 million rows, of which
the authors used a 25% sample), UK road accident counts summarized with census
fields by the authors, and a simulated auto portfolio whose rows are not
published. The fitted coefficients, pseudo-R2 values and plots of those case
studies therefore cannot be recomputed, and this guide does not invent data to
match them. What it does reproduce, in full, is every worked example whose
inputs the monograph prints: the pseudo-R2 of Table 3.1, the closed forms of
section 4.7, the encodings of Tables 6.1 and 6.5, the offset example of chapter
8 (whose cell counts and true relativities Tables 8.2 and 8.3 print), the
permutation example of Tables 9.1 to 9.4 and the arithmetic built on printed
coefficients. Each prints the monograph's value next to prospicio's.

To show the methods on data, the guide simulates a motor portfolio with a known
ground truth, as the monograph's section 2.3 does: 20,000 policies with driver
age, vehicle age, region, power, a 300-level make, a noise feature and an
interaction between young drivers and powerful cars. It is built with a fixed
seed in the first block of section 1.2, so every number below is reproducible.
Models are compared by ranking and by pseudo-R2 rounded to three decimals,
since gradient boosting libraries move the later digits between versions.

## 1. Measuring performance (chapter 3)

### 1.1 Pseudo-R2

The monograph measures every model by one number that does not grow with the
size of the data: the share of the possible improvement in log-likelihood that
the model achieves, from the null model (the mean for everyone) to the saturated
model (each record predicted at its own outcome). Since deviance is twice the
gap to the saturated model, this is `1 - D_model / D_null`, the deviance
explained. Table 3.1 has four records, two claims and predictions 0.2 to 0.8:

```python
import math

from prospicio.models import deviance

y = [0.0, 0.0, 1.0, 1.0]
mu = [0.2, 0.4, 0.6, 0.8]
null = [sum(y) / len(y)] * len(y)


def loglik(y, mu):
    """Poisson log-likelihood without the log(y!) constant, as in section 3.1."""
    return sum(yi * math.log(mi) - mi for yi, mi in zip(y, mu))


ll_model, ll_null = loglik(y, mu), loglik(y, null)
ll_sat = sum(yi * math.log(yi) - yi for yi in y if yi > 0)
print(f"log-likelihood model {ll_model:.3f} (printed -2.734)")
print(f"log-likelihood null  {ll_null:.3f} (printed -3.386)")
print(f"log-likelihood sat.  {ll_sat:.3f} (printed -2)")
print(f"pseudo-R2 from log-likelihoods {(ll_model - ll_null) / (ll_sat - ll_null):.3f}")
d_model, d_null = deviance("poisson", y, mu), deviance("poisson", y, null)
print(f"pseudo-R2 from deviances       {1 - d_model / d_null:.3f} (printed 47%)")
```

```text
log-likelihood model -2.734 (printed -2.734)
log-likelihood null  -3.386 (printed -3.386)
log-likelihood sat.  -2.000 (printed -2)
pseudo-R2 from log-likelihoods 0.471
pseudo-R2 from deviances       0.471 (printed 47%)
```

Both routes give the printed 47%. `deviance` takes prior weights too, which is
section 3.7.2's exposure-weighted version: each record's deviance is multiplied
by its exposure in both the model and the null deviance.

Two more numbers in the chapter are arithmetic on printed values. Table 3.6's
corporation with a five-year-old aircraft has linear predictor -2.447 + 0.410 +
5 x (-0.086) = -2.467 and frequency exp(-2.467) = 0.085; and in section 3.7.4 the
null model trained on folds 2 to 7 is rebased to fold 1's mean by
0.004593 / 0.004579:

```python
print(f"linear predictor {-2.447 + 0.410 + 5 * -0.086:.3f}, frequency {math.exp(-2.467):.3f} (printed -2.467, 0.085)")
print(f"rebasing factor {0.004593 / 0.004579:.4f} (printed 1.0031)")
```

```text
linear predictor -2.467, frequency 0.085 (printed -2.467, 0.085)
rebasing factor 1.0031 (printed 1.0031)
```

### 1.2 A portfolio with a known truth, and folds from the policy number

The simulated portfolio below plays the part of the monograph's case studies.
The true log frequency has a steep young-driver effect, a rise after 60, a
vehicle-age effect that falls until age 6 and is flat after, region and power
effects, a make effect drawn for each of 300 makes whose sizes fall off like a
Zipf law, and an extra load for young drivers of high-power cars. `noise` has no
effect, and `exper` (years since the driving test) is driver age less 17 less a
small random amount, so it is highly correlated with age and has no effect of
its own.

Section 3.7.3 recommends deriving folds from the policy number rather than from a
random draw, so they can be rebuilt by anyone at any time and every record of
one policy lands in one fold. Here the policy number is the row number, the fold
is its last digit, folds 7 to 9 (30%) are the test data, and folds 0 to 6 are
the seven cross-validation folds, as in the monograph.

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

The helper below is the monograph's measure for a frequency model with exposure:
`y` are claim counts, `mu` expected counts, and the null model is the data's own
mean frequency times each record's exposure. `cv_score` trains on six of the
seven folds, scores the seventh, and averages; it takes any function that fits
on one design and predicts on another, so a GLM, a GAM and a gradient boosting
model are scored the same way.

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

That is section 3.7.1's lesson: a perfect model explains only a few percent of
the deviance of individual claim counts, because most of each count is chance.
The monograph's perfect model scored 0.051 on its own simulation. The level says
little; comparisons between models on the same data say everything.

### 1.3 Null, baseline and benchmark models

The null model predicts the training mean. Its validation pseudo-R2 should be
zero but comes out slightly negative, because the training mean is not the
validation fold's mean (Table 3.3); rebasing the prediction to the fold's own
mean (section 3.7.4) brings it to exactly zero. The baseline is a plain Poisson
GLM with age and vehicle age as straight lines and region and power as factors
(section 3.5), and the benchmark is a gradient boosting model through
`prospicio.boosting.Booster` (section 3.6), here LightGBM with the make given
as a plain number, so the trees can split on it.

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

The order is the monograph's: null at zero, a GLM above it, the boosting model
well above the GLM. The rest of the monograph is about closing that gap with
models a pricing team can explain.

## 2. Penalized regression (chapter 4)

### 2.1 Why fewer parameters can predict better

Section 4.1's rare car: ten insurers each see one of ten cars for a year, and one
car has a claim. An unpenalized GLM at each insurer predicts 100% or 0%. On
average across the ten insurers that is right (10%), but each insurer's
prediction is far from it: the spread of the ten predictions is the model's
variance.

```python
preds = np.array([1.0] + [0.0] * 9)
print(f"mean {preds.mean():.2f}, variance {preds.var():.2f}, sd {preds.std():.2f} (printed 0.09, 0.30)")
```

```text
mean 0.10, variance 0.09, sd 0.30 (printed 0.09, 0.30)
```

A model that ignores this car's own experience and predicts the 5% average for
every make is biased but has no variance, and here every insurer is closer to
the truth with it. Penalized regression trades bias for variance continuously:
it maximizes the log-likelihood less a penalty on the coefficients, the sum of
their squares for ridge, of their absolute values for the lasso, and a mix of
the two (weight `alpha` on the lasso part) for the elastic net. prospicio's
`ElasticNet` minimizes glmnet's objective: the mean deviance over two plus
`lam` times the penalty, with an unpenalized intercept and the columns
standardized before the penalty is applied. Its `lam` and `alpha` are the
monograph's lambda and alpha (section 4.6.3).

### 2.2 Ridge in closed form, and why features are standardized

For one feature, a normal response and no intercept, section 4.7.1 shows that
ridge divides the mean of `x * y` by the mean of `x**2` plus `lam`, so it pulls
the least squares estimate towards zero. Section 4.7.2's example has mean
`x * y` of 10 and mean `x**2` of 5, length in metres, and `lam` = 5. In
centimetres the first mean is 100 times bigger and the second 10,000 times, so
the same `lam` barely shrinks at all:

```python
for unit, xy, xx, scale in [("metres", 10.0, 5.0, 1.0), ("centimetres", 1_000.0, 50_000.0, 100.0)]:
    ols, ridge_b = xy / xx, xy / (xx + 5.0)
    print(f"{unit:12s} least squares {ols:.4f}, ridge {ridge_b:.6f}, frequency at 1 m {ridge_b * scale:.4f}")
```

```text
metres       least squares 2.0000, ridge 1.000000, frequency at 1 m 1.0000
centimetres  least squares 0.0200, ridge 0.019998, frequency at 1 m 1.9998
```

The monograph prints the centimetre estimate as 0.019 and the frequency as 1.9.
The exact values are 0.019998 and 1.9998, which round to 0.020 and 2.0; the
printed digits are truncated, and the point stands either way: the units, not
the data, decided how much the penalty bit. Standardizing every feature to unit
standard deviation before applying the penalty removes the dependence, and
`ElasticNet` does so by default and reports coefficients on the original scale.
Fitting the same ridge to a length in metres and in centimetres gives
coefficients that differ by exactly the factor of 100:

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

Section 4.7.3 shows that for one standardized feature the lasso estimate is the
least squares estimate moved towards zero by `lam`, and exactly zero once `lam`
exceeds it (soft-thresholding). With prospicio's objective the same holds, the
least squares estimate being the mean of `x * y`:

```python
x_std = rng.normal(size=500)
x_std = (x_std - x_std.mean()) / x_std.std()
y_soft = 0.3 * x_std + rng.normal(0, 1, 500)
y_soft -= y_soft.mean()
ols = float(np.mean(x_std * y_soft))
X_soft = Design(columns(np.ones(500), x_std), ["(Intercept)", "x"])
for lam in [0.0, 0.1, 0.2, 0.4]:
    b = ElasticNet("gaussian", alpha=1.0, lam=lam).fit(X_soft, y_soft.tolist()).coefficients[1]
    print(f"lam {lam:.1f}: lasso {b:.4f}, soft threshold {np.sign(ols) * max(abs(ols) - lam, 0.0):.4f}")
```

```text
lam 0.0: lasso 0.3382, soft threshold 0.3382
lam 0.1: lasso 0.2382, soft threshold 0.2382
lam 0.2: lasso 0.1382, soft threshold 0.1382
lam 0.4: lasso 0.0000, soft threshold 0.0000
```

### 2.4 Paths and choosing the penalty by cross-validation

Tables 4.1 and 4.2 fit accident frequency on the log of aircraft age alone and
show the coefficient shrink as the penalty grows: smoothly towards zero for
ridge, and to exactly zero for the lasso. The same on the portfolio, with log
driver age. For a Poisson model the records can be added up within each value
of the features without changing the fit (section 8.3 notes the same), so the
age-only models below run on 63 ages times 10 folds rather than 20,000 policies:

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
for name, a in [("ridge", 0.0), ("lasso", 1.0)]:
    net = ElasticNet("poisson", alpha=a)
    top = ElasticNet("poisson", alpha=1.0).lambda_max(Xa, ya)
    scale = 100.0 if a == 0.0 else 1.0
    path = net.path(Xa, ya, [scale * top * f for f in [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]])
    print(f"{name:5s} log-age coefficient:", [f"{f.coefficients[1]:.3f}" for f in path])
```

```text
ridge log-age coefficient: ['-0.023', '-0.067', '-0.153', '-0.275', '-0.354', '-0.413']
lasso log-age coefficient: ['0.000', '-0.292', '-0.373', '-0.401', '-0.409', '-0.413']
```

The last value of each row is the unpenalized GLM. To choose `lam`, section 4.3
fits the whole path on six folds, scores it on the seventh, and averages over
the seven choices of validation fold. `ElasticNet.cross_validate` does exactly
this from `(train, test)` splits; here the splits are the policy-number folds.
It reports the `lam` with the best mean validation deviance and, for the
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

The lasso both selects features and shrinks the survivors. Section 4.5's relaxed
lasso separates the two: keep the features the lasso selected, then refit them
with less or no penalty (hard thresholding). In prospicio the second stage is a
`Glm` on the selected columns, used in section 3.2 below.

## 3. Nonlinear effects (chapter 5)

### 3.1 Step and hinge functions, from the printed coefficients

A step function is 1 at or above a threshold and 0 below; a hinge is zero below
its knot and rises one per unit above it, `max(0, x - knot)`. Given a step at
age 30, section 5.2's GLM is `log frequency = -5.2815 - 1.4689 x step_30`; given
steps at 5, 15 and 30 it prints four frequencies. Recomputing them from the
printed coefficients:

```python
print(f"one step: {math.exp(-5.2815):.5f}, relativity {math.exp(-1.4689):.4f} (printed 0.00509, 0.2302)")
lp = [-4.8292, -0.6881, -0.5711, -0.6237]
three = [math.exp(sum(lp[: i + 1])) for i in range(4)]
printed = [0.00783, 0.00394, 0.00222, 0.00119]
print("three steps:", [f"{f:.5f}" for f in three], "printed", printed)
print("printed / recomputed:", [f"{p / f:.3f}" for p, f in zip(printed, three)])
```

```text
one step: 0.00508, relativity 0.2302 (printed 0.00509, 0.2302)
three steps: ['0.00799', '0.00402', '0.00227', '0.00122'] printed [0.00783, 0.00394, 0.00222, 0.00119]
printed / recomputed: ['0.980', '0.981', '0.978', '0.979']
```

The single step matches. The three-step frequencies do not: every printed value
is about 2% below the one the printed coefficients give, by nearly the same
factor in all four bands. A common factor points at the intercept rather than the
slopes: an intercept near -4.85 instead of -4.8292 would give the printed
frequencies, and the ratios between bands agree with the printed slopes either
way. Without the data we cannot say which printed number is off.

Table 5.7 codes an ordinal feature (pilot quality, five levels) as five
cumulative steps and multiplies the exponentiated coefficients:

```python
coef = [0, 0.515, 0, 0.211, 0.206]
print("cumulative relativities:", [f"{math.exp(sum(coef[: i + 1])):.3f}" for i in range(5)])
```

```text
cumulative relativities: ['1.000', '1.674', '1.674', '2.067', '2.540']
```

The printed cumulative row is 1, 1.673, 1.673, 2.06, 2.539, and Table 5.8 gives
the fourth as 2.07; the exact product is 2.067, so Table 5.7 truncates where
Table 5.8 rounds.

### 3.2 Many steps or hinges, and the lasso picks

Rather than choosing knots by eye, section 5.2 gives the lasso one step function
at every age and lets cross-validation decide how many survive; section 5.6 uses
25 equally spaced steps or hinges per feature. On the portfolio's ages
(aggregated as above), with a step every three years from 20 to 77:

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

To see how close the result gets to the truth, compare frequencies by age with
the true ones (the exposure-weighted average of the true frequencies of the
policies at each age) for three fits: the lasso at `lam_min`, the relaxed lasso
(the kept steps refitted by an unpenalized `Glm`, section 4.5), and a P-spline
GAM (`Gam`, the smoothing model the monograph itself uses for its spatial
second stage in chapter 8):

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
for a in [18, 20, 23, 26, 35, 50, 65, 80]:
    i = a - 18
    print(f"age {a}: truth {true_by_age[i]:.3f}  lasso {lasso_curve[i]:.3f}  "
          f"relaxed {relaxed_curve[i]:.3f}  GAM {gam_curve[i]:.3f}")
print(f"GAM effective degrees of freedom {gam_fit.edf:.1f}")
```

```text
age 18: truth 0.565  lasso 0.531  relaxed 0.553  GAM 0.530
age 20: truth 0.481  lasso 0.408  relaxed 0.415  GAM 0.449
age 23: truth 0.363  lasso 0.304  relaxed 0.310  GAM 0.355
age 26: truth 0.264  lasso 0.256  relaxed 0.260  GAM 0.290
age 35: truth 0.216  lasso 0.207  relaxed 0.200  GAM 0.206
age 50: truth 0.207  lasso 0.207  relaxed 0.200  GAM 0.195
age 65: truth 0.215  lasso 0.236  relaxed 0.239  GAM 0.228
age 80: truth 0.256  lasso 0.239  relaxed 0.253  GAM 0.249
GAM effective degrees of freedom 5.7
```

All three find the young-driver slope and the rise after 60 without being told
where they are. The lasso's shrinkage pulls the extremes towards the middle, the
effect section 5.2 sees above aircraft age 20, and the relaxed refit partly undoes it.
The monograph prefers steps and hinges because the result is a short rating
table; a GAM gives a smooth curve that is tabulated afterwards. Section 5.8.1's
link to the fused lasso is a change of coordinates: a step at each level is the
same model as a dummy at each level, each step's coefficient being the
difference between neighbouring dummies, so a lasso penalty on steps is a
penalty on the jumps between neighbours.

Hinges work the same way in principle, but `ElasticNet` is slow on them: the
columns of neighbouring hinges are almost identical, and coordinate descent
creeps. Twenty hinges on these 441 cells took over 40 seconds where twenty steps
took a quarter of a second, so this guide offers the lasso steps only (see the
list of gaps at the end). MARS (section 5.5), which places hinges greedily, is
not in prospicio.

### 3.3 Monotone effects

Section 5.7.2 forces an ordinal effect to be monotone by constraining step
coefficients to be non-negative. `ElasticNet` has no bounds on coefficients,
but a GAM can impose the shape directly: `"mpd"` is a monotone decreasing
P-spline (scam's). On vehicle age, whose true effect falls until 6 and is flat
after:

```python
veh_cells, veh_cell = np.unique(veh_age * 10 + fold, return_inverse=True)
veh_train = rows(veh_cells % 10 < 7)
X_veh = Design(columns(np.ones(len(veh_cells)), veh_cells // 10), ["(Intercept)", "veh_age"],
               offset=np.log(np.bincount(veh_cell, exposure)).tolist())
mono = Gam(Glm("poisson"), [{"column": "veh_age", "n_basis": 8, "bs": "mpd"}]).fit(
    X_veh.select(veh_train), np.bincount(veh_cell, claims)[veh_train].tolist())
curve = np.array(mono.predict(Design(columns(np.ones(21), np.arange(21)), ["(Intercept)", "veh_age"])))
show = [0, 2, 4, 6, 10, 15, 20]
print("fitted relativity to age 0:", [f"{v:.2f}" for v in (curve / curve[0])[show]])
print("true relativity:           ", [f"{math.exp(-0.08 * min(v, 6)):.2f}" for v in show])
```

```text
fitted relativity to age 0: ['1.00', '0.89', '0.80', '0.75', '0.70', '0.64', '0.61']
true relativity:            ['1.00', '0.85', '0.73', '0.62', '0.62', '0.62', '0.62']
```

The monotone fit bends more gently than the true hinge and keeps falling a
little after age 6 where the truth is flat, but it can never turn back up. As
the monograph warns, a constraint hides a reversal if the data has one: use it
where the shape is known (claim history, deductibles), not to tidy a factor.

## 4. High cardinality categorical variables (chapter 6)

### 4.1 Target encoding, and keeping it out of its own fold

A factor with thousands of levels (aircraft make, city, zip code) cannot go into
a GLM as dummies without memory trouble and wild estimates for the thin levels.
Chapter 6's main tool is target encoding: replace each level by one number, its
actual-to-expected ratio shrunk towards 1 by a credibility factor
`z = n / (n + K)`, where `n` is the level's exposure, then use the log of that
number as a feature. Encoding a level from its own records leaks the target into
the feature (section 6.3), so a record in fold `k` is encoded only from the
other folds. Table 6.1 has one make with six aircraft in three folds:

```python
te_fold = np.array([1, 1, 2, 2, 3, 3])
te_acc = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
for k in [1, 2, 3]:
    other = te_fold != k
    print(f"fold {k}: accidents elsewhere {te_acc[other].sum():.0f} on {other.sum()} aircraft, "
          f"encoding {te_acc[other].mean():.2f}")
print("printed: 0.75, 0.50, 0.25")
```

```text
fold 1: accidents elsewhere 3 on 4 aircraft, encoding 0.75
fold 2: accidents elsewhere 2 on 4 aircraft, encoding 0.50
fold 3: accidents elsewhere 1 on 4 aircraft, encoding 0.25
printed: 0.75, 0.50, 0.25
```

### 4.2 The credibility constant, and the printed encodings

Section 6.6 estimates `K` as the ratio of the within-make variance (one
aircraft-year's accident count, Bernoulli at the mean rate 0.004584) to the
between-make variance (estimated from mean residuals of large makes as
7.038492e-6), and Table 6.5 lists encoded makes at the `K` = 1,024 finally
used. Recomputing from the printed inputs:

```python
p = 0.004584
within = p * (1 - p)
K_est = within / 7.038492e-6
print(f"within variance {within:.9f} (printed 0.004562987), K {K_est:.0f} (printed 648), "
      f"credibility at 1,000 aircraft-years {1000 / (1000 + 648):.2f} (printed 0.61)")
table_6_5 = [("DJI", 11900, 0.0, 0.9208, 0.079), ("TEXTRON AVIATION INC", 5483, 0.24356, 0.8426, 0.363),
             ("MARLIN JOHN E", 13, 186.59373, 0.0122, 3.266), ("STUMP GREAT LAKES", 12, 223.74936, 0.0116, 3.579)]
for make_name, expo, ave, z_printed, te_printed in table_6_5:
    z = expo / (expo + 1024)
    print(f"{make_name:21s} z {z:.4f} (printed {z_printed})  encoding {1 - z + z * ave:.3f}, "
          f"from the printed z {1 - z_printed + z_printed * ave:.3f} (printed {te_printed})")
```

```text
within variance 0.004562987 (printed 0.004562987), K 648 (printed 648), credibility at 1,000 aircraft-years 0.61 (printed 0.61)
DJI                   z 0.9208 (printed 0.9208)  encoding 0.079, from the printed z 0.079 (printed 0.079)
TEXTRON AVIATION INC  z 0.8426 (printed 0.8426)  encoding 0.363, from the printed z 0.363 (printed 0.363)
MARLIN JOHN E         z 0.0125 (printed 0.0122)  encoding 3.327, from the printed z 3.264 (printed 3.266)
STUMP GREAT LAKES     z 0.0116 (printed 0.0116)  encoding 3.580, from the printed z 3.584 (printed 3.579)
```

DJI and Textron match. For the two one-claim makes the exposure column shows a
whole number of aircraft-years, and the printed credibility of Marlin (0.0122)
implies about 12.65 aircraft-years, not 13: the column is rounded, and from the
printed `z` the encoding is 3.264. Stump Great Lakes recomputes to 3.580
against 3.579 (3.584 from its rounded printed `z`), the gap a rounding of its
exposure too. The worked sentence for
Stump in section 6.7.1 multiplies `z` by 222.22 where Table 6.5 has an AvE
ratio of 223.75; 223.75 is the value that reproduces the printed 3.58, so the
222.22 is a slip in the text. (The table's caption also describes the AvE ratio
upside down, as predicted over actual; its values are actual over expected.)

Two smaller checks from the same chapter: the Bell helicopter example of section
6.7.2, encoded against the overall mean (16 per thousand against 4) and against
the GLM's own prediction for helicopters (16 against 20), and the logistic
blending of the software in section 6.7.6 at a level with no data, which should
give no credibility but does not:

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

The second value is the "about 25%" of section 6.7.6, credibility for a level
with no data at all.

### 4.3 Ridge regression on dummies is credibility

Section 6.8.1 shows why target encoding is natural: ridge regression on the
dummies of a single factor, normal response, response centred and no intercept,
gives each level the coefficient `n_k (mean_k - mean) / (n_k + lambda)`, a
credibility-weighted deviation from the overall mean. In prospicio's
parameterization (mean squared error over two, plus `lam` over two times the sum
of squares) the monograph's `lambda` is `lam` times the number of records. A
design with no all-ones column has no intercept, so `ElasticNet` fits exactly
that model:

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
for j, size in enumerate(sizes):
    z = size / (size + K)
    print(f"level {j} ({size:3d} records): ridge {fit.coefficients[j]:+.4f}, "
          f"z x deviation {z * yc[levels == j].mean():+.4f}  (z = {z:.3f})")
```

```text
level 0 (  1 records): ridge +0.0456, z x deviation +0.0456  (z = 0.021)
level 1 (  5 records): ridge -0.0470, z x deviation -0.0470  (z = 0.099)
level 2 ( 50 records): ridge +0.2407, z x deviation +0.2407  (z = 0.523)
level 3 (400 records): ridge -0.0511, z x deviation -0.0511  (z = 0.898)
```

The penalty plays Buhlmann's `K`, with one difference the monograph stresses:
credibility theory estimates `K` from variance components, while the penalized
regression chooses it by cross-validation.

### 4.4 Target encoding the portfolio's 300 makes

On the portfolio, makes follow section 6.7.2: the expected count of each policy
is a first GLM's prediction without make (cross-fitted, so no record is
predicted by a model that saw it), and each make's encoding is its shrunk
actual-to-expected ratio. The log encoding then enters a second-stage GLM with
the first GLM's prediction as offset, as the monograph does in section 6.9. For
each `K` in a grid, the seven-fold cross-validation repeats the whole recipe
inside each fold:

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
print(f"{'K':>6s} {'cv pseudo-R2':>13s} {'coefficient':>12s}")
for K in [16, 64, 256, 1024, 4096]:
    scores, betas = [], []
    for k in range(7):
        feature = make_feature(K, k)
        X2 = Design(columns(ones, feature), ["(Intercept)", "log_te"], offset=np.log(stage1[k]).tolist())
        fit_rows, val = rows(train & (fold != k)), rows(fold == k)
        fit2 = Glm("poisson").fit(X2.select(fit_rows), claims[fit_rows].tolist())
        scores.append(pseudo_r2(claims[val], fit2.predict(X2.select(val)), exposure[val]))
        betas.append(fit2.coefficients[1])
    print(f"{K:6d} {np.mean(scores):13.3f} {np.mean(betas):12.2f}")
```

```text
     K  cv pseudo-R2  coefficient
    16         0.040         0.82
    64         0.041         1.08
   256         0.039         1.41
  1024         0.037         2.24
  4096         0.035         4.98
```

The baseline GLM scored 0.021 without make; the encoded make roughly doubles
that, and as in the monograph's Figure 6.1 the score is flat over a wide range
of `K`, here best near 64. The coefficient rises with `K`, as in Figure 6.2: a
heavily shrunk encoding needs a coefficient above 1 to restore the spread, and a
coefficient near 1 means the credibility was about right. Here the best score
and a coefficient near 1 fall at the same `K`, which section 6.6 says to expect
when the encoding is done correctly; the rest of the guide uses `K` = 64.

A generalized linear mixed model (section 6.5) would treat make as a random
effect and shrink it the same way. prospicio's GLMs have no random effects; its
nearest tools are Bayesian hierarchical credibility for a single ratio per group
(`prospicio.credibility`) and `BayesGlm` with normal priors (the Bayesian view
of ridge in section 6.8.2), neither a Poisson GLMM.

## 5. The printed numbers of chapters 3 to 6 and prospicio's

| Where | Printed | Recomputed | Note |
|---|---|---|---|
| Table 3.1 pseudo-R2 | 47% | 0.471 | matches |
| Section 3.5 linear predictor | -2.467, 0.085 | -2.467, 0.085 | matches |
| Section 3.7.4 rebasing | 1.0031 | 1.0031 | matches |
| Section 4.1 variance | 0.09, 0.30 | 0.09, 0.30 | matches |
| Section 4.7.2 ridge in cm | 0.019, 1.9 | 0.019998, 1.9998 | printed digits truncated |
| Section 5.2 one step | 0.00509, 0.2302 | 0.00509, 0.2302 | matches |
| Section 5.2 three steps | 0.00783 to 0.00119 | 0.00799 to 0.00122 | all about 2% higher; points at the printed intercept |
| Table 5.7 cumulative | 2.06 | 2.067 | Table 5.8 prints 2.07 |
| Section 6.6 K | 648, 0.61 | 648, 0.61 | matches |
| Table 6.5 encodings | 0.079, 0.363, 3.266, 3.579 | 0.079, 0.363, 3.264, 3.580 | Marlin from its printed `z`; exposures printed rounded; 222.22 in the text is 223.75 |

Chapters 7 to 10 (highly correlated variables, two-stage models, learning from
the boosting model, and the challenges of chapter 10), with the list of what
prospicio does not cover, are in the second part of this guide,
[`glm-to-comprehensive-pricing-chalk-2.md`](glm-to-comprehensive-pricing-chalk-2.md).
