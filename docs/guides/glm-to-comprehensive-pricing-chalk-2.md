# Beyond the GLM, part 2: correlated features, two stages and black boxes

<!-- site: execute -->

Part 2 of the guide to CAS Monograph 16, *From GLMs to Comprehensive Insurance
Pricing: Techniques and Challenges* (Chalk, Deacon, Guillen and Martinelli,
2025). [Part 1](glm-to-comprehensive-pricing-chalk.md) covers chapters 3 to 6
and tells which examples the guide reproduces. This part covers chapters 7 to
10. Table and section numbers are the monograph's.

The block below rebuilds what part 1 built: the simulated portfolio (20,000
policies, folds from the policy number, folds 7 to 9 for testing), the
pseudo-R2 helper and cross-validation loop, the baseline GLM and boosting
designs, the age steps that the lasso keeps, and the out-of-fold target
encoding of make. Part 1 explains each piece.

```python
import math

from prospicio.models import ElasticNet, deviance
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

from prospicio.boosting import Booster
from prospicio.models import Glm

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



key = age * 10 + fold
cells, cell = np.unique(key, return_inverse=True)
cell_claims = np.bincount(cell, claims)
cell_expo = np.bincount(cell, exposure)
cell_age, cell_fold = cells // 10, cells % 10
cell_off = np.log(cell_expo).tolist()
tr_cells = rows(cell_fold < 7)

def cell_splits(cell_rows, cell_folds):
    pos = {r: i for i, r in enumerate(cell_rows)}
    return [([pos[r] for r in cell_rows if cell_folds[r] != k],
             [pos[r] for r in cell_rows if cell_folds[r] == k]) for k in range(7)]


splits = cell_splits(tr_cells, cell_fold)

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

expected = np.zeros(n)
for j in range(7):
    fit_j = Glm("poisson").fit(X_base.select(rows(train & (fold != j))), claims[train & (fold != j)].tolist())
    expected[fold == j] = fit_j.predict(X_base.select(rows(fold == j)))
expected[~train] = Glm("poisson").fit(X_base.select(rows(train)), claims[train].tolist()).predict(
    X_base.select(rows(~train)))


from prospicio.models import target_encode

make_level = make.astype(str).tolist()


def make_feature(K, held_out=None):
    """Per policy, the log of the make's shrunk actual-to-expected ratio: a training row in fold j is
    encoded from the training folds other than j (and other than the held-out fold), every other row
    from all training folds but the held-out one."""
    src = np.flatnonzero(train & (fold != held_out))
    splits = [(np.flatnonzero(fold[src] != j).tolist(), np.flatnonzero(fold[src] == j).tolist())
              for j in range(7) if j != held_out]
    oof, encoding = target_encode([make_level[i] for i in src], claims[src].tolist(),
                                  expected[src].tolist(), splits, K, exposure[src].tolist())
    te = np.array(encoding.encode(make_level))
    te[src] = oof
    return np.log(te)


print(f"{n} policies, {claims.sum():.0f} claims; age steps kept by the lasso at {kept}")
```

```text
20000 policies, 3659 claims; age steps kept by the lasso at [20, 23, 26, 29, 32, 62, 65, 77]
```

## 1. Highly correlated variables (chapter 7)

A GLM can split the joint effect of two near-identical features in many ways
(section 7.2), so small changes in the data move the coefficients. The
portfolio's `exper` is driver age less 17 less a small random amount:
correlation 0.997 with age and no effect of its own. The block fits both on each
of the seven cross-validation training sets, with and without a penalty (section
7.4):

```python
import pandas as pd
from great_tables import GT

X_corr = Design(columns(ones, age, exper), ["(Intercept)", "age", "exper"], offset=log_expo)
fits_by = {"GLM": lambda d, y: Glm("poisson").fit(d, y),
           "ridge": lambda d, y: ElasticNet("poisson", alpha=0.0, lam=0.01).fit(d, y),
           "lasso": lambda d, y: ElasticNet("poisson", alpha=1.0, lam=0.001).fit(d, y)}
summary = {}
for name, fitter in fits_by.items():
    coefs = np.array([fitter(X_corr.select(rows(train & (fold != k))), claims[train & (fold != k)].tolist()).coefficients[1:]
                      for k in range(7)])
    summary[name] = {"age min": coefs[:, 0].min(), "age max": coefs[:, 0].max(),
                     "exper min": coefs[:, 1].min(), "exper max": coefs[:, 1].max(),
                     "sum": coefs.sum(1).mean()}
(GT(pd.DataFrame(summary).T.rename_axis("fit").reset_index(), rowname_col="fit")
 .tab_header("Age and experience fitted together", "Coefficient over the seven training sets")
 .fmt_number(decimals=4, force_sign=True))
```

```text
       age min  age max  exper min  exper max      sum
GLM    -0.0540  -0.0409    +0.0340    +0.0467  -0.0072
ridge  -0.0061  -0.0052    -0.0016    -0.0010  -0.0070
lasso  -0.0074  -0.0066     0.0000     0.0000  -0.0070
```

The unpenalized coefficients change much between training sets, but their sum
(one year of age and experience together) hardly changes. Ridge shares the
effect and keeps both stable. The lasso keeps one feature, which is stable here
but can flip when the features are even closer (section 7.5.3).

For principal components (section 7.4.5), take `numpy.linalg.svd` of the
standardized columns and put the components into a `Design` like any other
column.

## 2. Modelling in two stages (chapter 8)

### 2.1 The marketing discount as an offset

Chapter 8 has the only fully specified data set in the monograph: one million
simulated auto policies. Age group (A to E) and no-claims bonus (NCB 0 to 4) are
strongly related. Table 8.2 gives the policies per cell, Table 8.3 the true
relativities, section 8.1 a base frequency of 10% for age A and NCB 0. A Poisson
GLM on the 25 cells, with expected claims as response, gives the fit that the
monograph's simulation estimates with noise:

```python
from prospicio.models import Terms

table_8_2 = {"A": [132381, 54519, 12194, 1237, 26], "B": [55624, 82213, 49354, 12283, 952],
             "C": [11053, 49903, 77029, 50248, 11291], "D": [920, 12187, 49027, 81946, 55690],
             "E": [22, 1178, 12396, 54286, 132041]}
age_rel = dict(zip("ABCDE", [1.0, 0.7, 0.6, 0.5, 0.4]))
ncb_rel = [1.0, 0.95, 0.9, 0.85, 0.8]
marketing = [1.0, 0.65, 0.50, 0.40, 0.20]
cells_8 = [(g, b, c) for g, counts in table_8_2.items() for b, c in enumerate(counts)]
data_8 = {"age": [g for g, _, _ in cells_8], "ncb": [str(b) for _, b, _ in cells_8]}
policies = np.array([c for _, _, c in cells_8], float)
expected_claims = [c * 0.10 * age_rel[g] * ncb_rel[b] for g, b, c in cells_8]
print(f"policies {policies.sum():,.0f}")
```

```text
policies 1,000,000
```

```python
both = Terms().intercept().factor("age").factor("ncb").fit(data_8)
full = Glm("poisson").fit(both.design(data_8, offset=np.log(policies).tolist()), expected_claims)
true_rel = [0.10 * age_rel["A"] * ncb_rel[0], *[age_rel[g] for g in "BCDE"], *ncb_rel[1:]]
(GT(pd.DataFrame({"coefficient": ["intercept", *[f"age {g}" for g in "BCDE"], *[f"NCB {b}" for b in range(1, 5)]],
                  "true": true_rel, "fitted": [math.exp(b) for b in full.coefficients]}),
    rowname_col="coefficient")
 .tab_header("Relativities with age and NCB in the model", "Poisson GLM on the 25 cells")
 .fmt_number(decimals=3))
```

```text
            true  fitted
intercept  0.100   0.100
age B      0.700   0.700
age C      0.600   0.600
age D      0.500   0.500
age E      0.400   0.400
NCB 1      0.950   0.950
NCB 2      0.900   0.900
NCB 3      0.850   0.850
NCB 4      0.800   0.800
```

The fit gives the truth; the monograph's own fit on its noisy simulation (Table
8.4) is within about 0.025 of it. Next, the marketing department's NCB scale
goes in as an offset and age is fitted again. Age relativities now increase with
age, because age carries part of the NCB effect that the offset removed. The
book's differences come from the noise in its simulation:

```python
age_only = Terms().intercept().factor("age").fit(data_8)
mkt_offset = np.log(policies * np.array([marketing[b] for _, b, _ in cells_8])).tolist()
offset_fit = Glm("poisson").fit(age_only.design(data_8, offset=mkt_offset), expected_claims)
rel = [math.exp(b) for b in offset_fit.coefficients]
(GT(pd.DataFrame({"coefficient": ["intercept", *[f"age {g}" for g in "BCDE"]],
                  "book": [0.112, 0.849, 0.925, 0.980, 1.057], "prospicio": rel}),
    rowname_col="coefficient")
 .tab_header("Table 8.6: age relativities with the NCB scale as offset")
 .fmt_number("book", decimals=3)
 .fmt_number("prospicio", decimals=4))
```

```text
            book  prospicio
intercept  0.112     0.1125
age B      0.849     0.8485
age C      0.925     0.9175
age D      0.980     0.9825
age E      1.057     1.0586
```

Section 8.1 then gives risk premiums at a claim size of 10,000:

```python
premiums = pd.DataFrame({
    "premium": ["true A/0", "true E/4", "GLM E/4, printed coefficients",
                "GLM age with marketing NCB", "offset model E/4, printed coefficients",
                "offset model E/4, recomputed coefficients"],
    "book": [1000, 320, 322, 78, 237, 237],
    "prospicio": [0.10 * 1.0 * 1.0 * 10000, 0.10 * 0.4 * 0.8 * 10000, 0.10 * 0.39 * 0.826 * 10000,
                  0.10 * 0.39 * 0.20 * 10000, 0.112 * 1.057 * 0.20 * 10000, rel[0] * rel[4] * 0.20 * 10000]})
(GT(premiums, rowname_col="premium")
 .tab_header("Section 8.1: risk premiums at a claim size of 10,000")
 .fmt_integer(["book", "prospicio"]))
```

```text
                                            book  prospicio
true A/0                                   1,000      1,000
true E/4                                     320        320
GLM E/4, printed coefficients                322        322
GLM age with marketing NCB                    78         78
offset model E/4, printed coefficients       237        237
offset model E/4, recomputed coefficients    237        238
```

### 2.2 Offsets and residuals give the same second stage

A second stage can continue from a first stage in two ways (sections 8.3 to
8.5): with the log of the first-stage predicted counts as an offset, or with the
residual (actual over predicted counts) as the response and the predicted counts
as weights. Yan and others, and Shi, show that the two are identical for a
Poisson log-link model. The block fits a first stage without region, then a
second stage with region only:

```python
X_s1 = Design(columns(ones, age, veh_age, *power_d), ["(Intercept)", "age", "veh_age", "power[mid]", "power[high]"],
              offset=log_expo)
tr = rows(train)
mu1 = np.asarray(Glm("poisson").fit(X_s1.select(tr), claims[tr].tolist()).predict(X_s1.select(tr)))
reg_cols = columns(np.ones(len(tr)), *[d[tr] for d in region_d])
reg_names = ["(Intercept)", *factor_names[:5]]
by_offset = Glm("poisson").fit(Design(reg_cols, reg_names, offset=np.log(mu1).tolist()), claims[tr].tolist())
by_residual = Glm("poisson").fit(Design(reg_cols, reg_names, weights=mu1.tolist()), (claims[tr] / mu1).tolist())
(GT(pd.DataFrame({"coefficient": ["intercept", *[f"region {r}" for r in "BCDEF"]],
                  "offset": [math.exp(b) for b in by_offset.coefficients],
                  "residual": [math.exp(b) for b in by_residual.coefficients]}),
    rowname_col="coefficient")
 .tab_header("Second stage by offset and by residual", "Region relativities")
 .fmt_number(decimals=4))
```

```text
           offset  residual
intercept  0.9301    0.9301
region B   1.2148    1.2148
region C   0.8598    0.8598
region D   1.0619    1.0619
region E   1.0436    1.0436
region F   1.4061    1.4061
```

In the zip code summary of section 8.4, the first rows have residuals 0 / 0.014
= 0 and 1 / 0.739 = 1.353.

Section 8.9.1 adds a caution: cross-fit the first-stage predictions, as part 1
did for `expected` in section 4.4, so that a model predicts only records it did
not see. An overfitted first stage leaves nothing for the second stage to find
(Table 8.12); cross-fitted predictions leave residuals such as 3 / 5 = 0.6 and 7
/ 5 = 1.4 (Table 8.13).

For a spatial second stage like Figure 8.2, use a `Gam` with the tensor
product smooth `tensors=[["lon", "lat"]]` and the first stage as offset.

### 2.3 Combining tables and area groups

Section 8.7 multiplies the seven aircraft tables into one relativity for the
Cessna 172M. Section 8.9.2 replaces 30,000 zip code relativities by 100 area
groups, in equal multiplicative steps from the smallest (0.379) to the largest
(1.851) relativity:

```python
table_8_11 = [1.30, 1.15, 1.00, 1.36, 1.69, 1.00, 0.41, 0.97]
step = (1.851 / 0.379) ** (1 / 99)
checks = pd.DataFrame({
    "item": ["Cessna 172M relativity", "range, largest over smallest", "step", "range from 99 printed steps"],
    "book": [1.37, 4.884, 1.0162, 4.908],
    "prospicio": [math.prod(table_8_11), 1.851 / 0.379, step, 1.0162 ** 99]})
(GT(checks, rowname_col="item")
 .tab_header("Sections 8.7 and 8.9.2")
 .fmt_number(["book", "prospicio"], rows=[0], decimals=2)
 .fmt_number(["book", "prospicio"], rows=[1, 3], decimals=3)
 .fmt_number("book", rows=[2], decimals=4)
 .fmt_number("prospicio", rows=[2], decimals=5))
```

```text
                                book  prospicio
Cessna 172M relativity          1.37       1.37
range, largest over smallest   4.884      4.884
step                          1.0162    1.01615
range from 99 printed steps    4.908      4.908
```

The printed step 1.0162 is the exact 1.01615 rounded up, so 99 steps of it
overshoot the range (4.908 against 4.884). Table 8.15 uses the rounded step:

```python
groups = [1, 2, 3, 22, 56]
(GT(pd.DataFrame({"group": [str(g) for g in groups], "book": [0.379, 0.385, 0.391, 0.531, 0.917],
                  "prospicio": [0.379 * 1.0162 ** (g - 1) for g in groups]}), rowname_col="group")
 .tab_header("Table 8.15: relativities of selected area groups")
 .fmt_number(decimals=3))
```

```text
     book  prospicio
1   0.379      0.379
2   0.385      0.385
3   0.391      0.391
22  0.531      0.531
56  0.917      0.917
```

## 3. Learning from black-box models (chapter 9)

### 3.1 The boosting model's variance

The monograph calls the gradient boosting benchmark a very complicated GLM. Its
training score keeps rising with more trees, but its validation score stops
rising (Figure 9.1). The block shows this on fold 0 of the portfolio:

```python
fit_rows, val = rows(train & (fold != 0)), rows(fold == 0)
trees = []
for rounds in [50, 150, 600]:
    booster = Booster("poisson", n_rounds=rounds, learning_rate=0.05,
                      params={"num_leaves": 7, "min_data_in_leaf": 200, "num_threads": 1}, seed=1)
    b_fit = booster.fit(X_tree.select(fit_rows), claims[fit_rows].tolist())
    gap_train = pseudo_r2(claims[fit_rows], b_fit.predict(X_tree.select(fit_rows)), exposure[fit_rows])
    gap_val = pseudo_r2(claims[val], b_fit.predict(X_tree.select(val)), exposure[val])
    trees.append({"trees": str(rounds), "train": gap_train, "validation": gap_val})
(GT(pd.DataFrame(trees), rowname_col="trees")
 .tab_header("Boosting on fold 0", "Pseudo-R2 by number of trees")
 .fmt_number(decimals=3))
```

```text
     train  validation
50   0.044       0.042
150  0.067       0.052
600  0.109       0.051
```

### 3.2 Permutation feature importance

Section 9.2 measures a feature's importance by how much worse the model predicts
held-out data after you shuffle that feature's column. The shuffle breaks the
link to the outcome and needs only predictions, so it works for any model.
Tables 9.1 to 9.4 use three insureds, frequency age / 100, a model that has
learned exactly that, and the mean absolute error:

```python
toy_age = np.array([20.0, 30.0, 40.0])
toy_height = np.array([1.4, 1.5, 1.6])
toy_freq = toy_age / 100


def toy_model(a, h):
    return a / 100


base_mae = np.mean(np.abs(toy_model(toy_age, toy_height) - toy_freq))
mae_age = np.mean(np.abs(toy_model(np.array([40.0, 20.0, 30.0]), toy_height) - toy_freq))
mae_height = np.mean(np.abs(toy_model(toy_age, np.array([1.6, 1.4, 1.5])) - toy_freq))
gains = {"age": mae_age - base_mae, "height": mae_height - base_mae}
for name, g in gains.items():
    print(f"{name:6s} increase in MAE {g:.3f}, importance {100 * g / max(gains.values()):.0f}")
print("printed: age 0.133 and 100, height 0 and 0")
```

```text
age    increase in MAE 0.133, importance 100
height increase in MAE 0.000, importance 0
printed: age 0.133 and 100, height 0 and 0
```

On the portfolio, "our GLM" is the monograph's end point: the age steps that the
lasso kept, steps for vehicle age, region, power and the make encoding (at `K` =
64), without noise or experience features (part 1, sections 3.2 and 4.4). The
boosting model gets the raw features and the same make encoding, but not
`exper`: this near-copy of age would blur every comparison (the caveat of
section 9.2 about correlated features).

As in section 9.5.1, each importance is the average over the seven
cross-validation models of the loss of pseudo-R2 on the held-out fold when the
feature is shuffled. `permutation_importance` takes a score to minimize, so the
pseudo-R2 goes in with a negative sign.

The make encoding changes with each held-out fold, so the `splits` argument
cannot fit the models here. The loop fits both models on each fold, explains
them on the held-out rows (the fold number seeds the shuffle) and combines the
seven results. It also runs `partial_dependence` on age and the pairwise
`interaction_screen` for the next two sections:

```python
veh_knots = [2, 4, 6, 8, 10]
features = ["age", "veh_age", "region", "power", "log_te", "noise"]
ages = [18, 21, 25, 40, 60, 75]
young_high = lambda f: (f["age"] < 25) & (f["power"] == "high")
gbm_params = {"num_leaves": 7, "min_data_in_leaf": 200, "num_threads": 1}


def glm_design(f, interaction=False):
    cols = [np.ones(len(f["age"]))] + [f["age"] >= a for a in kept] + [f["veh_age"] >= v for v in veh_knots]
    cols += [f["region"] == r for r in "BCDEF"] + [f["power"] == p for p in ["mid", "high"]] + [f["log_te"]]
    names = (["(Intercept)"] + [f"age_stp_{a}" for a in kept] + [f"veh_stp_{v}" for v in veh_knots]
             + factor_names + ["log_te"])
    if interaction:
        cols, names = cols + [young_high(f)], names + ["young_high"]
    return Design(columns(*cols), names, offset=np.log(f["exposure"]).tolist())


def tree_design(f):
    cols = [f["age"], f["veh_age"], np.unique(f["region"], return_inverse=True)[1],
            np.unique(f["power"], return_inverse=True)[1], f["log_te"], f["noise"]]
    return Design(columns(*cols), features, offset=np.log(f["exposure"]).tolist())


def frame(mask, log_te):
    return {"age": age[mask], "veh_age": veh_age[mask], "region": region[mask], "power": power[mask],
            "log_te": log_te[mask], "noise": noise[mask], "exposure": exposure[mask]}


from prospicio.models import (InteractionScreen, PartialDependence, PermutationImportance,
                              interaction_screen, partial_dependence, permutation_importance)

designs = {"GLM": glm_design, "GBM": tree_design}
neg_r2 = lambda y, mu, d: -pseudo_r2(y, mu, np.exp(d.offset))
imp_by, pd_by, screen_by = {m: [] for m in designs}, {m: [] for m in designs}, []
glm_with_interaction = []
for k in range(7):
    log_te_k = make_feature(64, held_out=k)
    fit_mask, val_mask = train & (fold != k), fold == k
    fit_f, val_f, y_val = frame(fit_mask, log_te_k), frame(val_mask, log_te_k), claims[val_mask]
    models = {"GLM": Glm("poisson").fit(glm_design(fit_f), claims[fit_mask].tolist()),
              "GBM": Booster("poisson", n_rounds=150, learning_rate=0.05, params=gbm_params, seed=1)
              .fit(tree_design(fit_f), claims[fit_mask].tolist())}
    for m, model in models.items():
        imp_by[m].append(permutation_importance(model, val_f, y_val, designs[m], neg_r2, features, seed=k))
        pd_by[m].append(partial_dependence(model, val_f, designs[m], "age", ages))
    screen_by.append(interaction_screen(models["GBM"], val_f, y_val, tree_design, neg_r2, features, seed=k))
    with_i = Glm("poisson").fit(glm_design(fit_f, interaction=True), claims[fit_mask].tolist())
    glm_with_interaction.append(pseudo_r2(y_val, with_i.predict(glm_design(val_f, interaction=True)), val_f["exposure"]))

importance = {m: PermutationImportance.combine(v) for m, v in imp_by.items()}
pd_age = {m: PartialDependence.combine(v) for m, v in pd_by.items()}
screen = InteractionScreen.combine(screen_by)
score_full = {m: -np.mean(imp.base) for m, imp in importance.items()}
mean_loss = {m: dict(zip(imp.features, imp.importance)) for m, imp in importance.items()}
print(f"cv pseudo-R2: GLM {score_full['GLM']:.3f}, GBM {score_full['GBM']:.3f}")
```

```text
cv pseudo-R2: GLM 0.055, GBM 0.048
```

```python
(GT(pd.DataFrame([{"feature": x, "GLM": mean_loss["GLM"][x], "GBM": mean_loss["GBM"][x]}
                  for x in sorted(features, key=lambda x: -mean_loss["GLM"][x])]), rowname_col="feature")
 .tab_header("Permutation importance", "Loss of pseudo-R2 when the feature is shuffled")
 .fmt_number(decimals=3))
```

```text
           GLM    GBM
log_te   0.039  0.033
age      0.037  0.034
power    0.019  0.019
veh_age  0.011  0.009
region   0.008  0.003
noise    0.000  0.000
```

```python
import matplotlib.pyplot as plt

order = sorted(features, key=lambda x: mean_loss["GLM"][x])
ypos = np.arange(len(order))
fig, ax = plt.subplots()
ax.barh(ypos + 0.2, [mean_loss["GLM"][x] for x in order], height=0.4, label="GLM")
ax.barh(ypos - 0.2, [mean_loss["GBM"][x] for x in order], height=0.4, label="GBM")
ax.set_yticks(ypos, order)
ax.set_xlabel("loss of pseudo-R2 when the feature is shuffled")
ax.set_title("Permutation importance, mean over the seven held-out folds")
ax.legend()
```

The two models agree on which features matter and roughly on their order. The
GBM gives the noise feature almost no importance, although it can use it. With
the monograph's treatment of nonlinear effects and of make, our GLM now scores
above this untuned boosting model (in section 1.3 of part 1 the raw-feature
baseline GLM was far below it). The GLM loss for noise is exactly zero because
the GLM does not use noise; the monograph's GLM importances are never zero,
because its lasso kept each feature in at least one fold.

### 3.3 Partial dependence

Section 9.3 isolates the effect of a feature in a black box: set the feature to
one value for every record, average the predictions, and repeat for another
value. For a GLM without interactions, the ratio of the two averages is exactly
the ratio of the feature's relativities. The printed `nu_registered` steps give
1.107 between 1 and 10; Tables 9.7 to 9.9 give the same ratio from predictions,
and the boosting model of Table 9.10 gives 1.104:

```python
nu = lambda v: -0.0001407 * v + 0.02936 * (v >= 2) + 0.07342 * (v >= 7) - 0.15822 * (v >= 16) - 0.17117 * (v >= 35)
(GT(pd.DataFrame({"ratio": ["GLM steps, 10 vs 1", "Table 9.9, ratio of totals", "Table 9.10, GBM"],
                  "book": [1.107, 1.107, 1.104],
                  "prospicio": [math.exp(nu(10) - nu(1)), 0.01225 / 0.01107, 0.005892 / 0.005336]}),
    rowname_col="ratio")
 .tab_header("Partial dependence ratios of nu_registered")
 .fmt_number(decimals=3))
```

```text
                             book  prospicio
GLM steps, 10 vs 1          1.107      1.107
Table 9.9, ratio of totals  1.107      1.107
Table 9.10, GBM             1.104      1.104
```

On the portfolio, the table gives each model's partial dependence on driver age,
relative to age 40 and averaged over the seven held-out folds, beside the truth
for a low-power car. The true young-driver effect is larger for high power,
which is the interaction:

```python
true_age = np.exp(0.9 * np.exp(-(np.array(ages) - 18) / 6) + 0.0015 * np.maximum(np.array(ages) - 60, 0) ** 1.5)
pd_rel = {m: pd.relative(40) for m, pd in pd_age.items()}
(GT(pd.DataFrame({"age": [str(a) for a in ages], "GLM": pd_rel["GLM"], "GBM": pd_rel["GBM"],
                  "truth, low power": true_age / true_age[3]}), rowname_col="age")
 .tab_header("Young-driver effect", "Partial dependence relative to age 40")
 .fmt_number(decimals=2))
```

```text
     GLM   GBM  truth, low power
18  2.81  2.39              2.40
21  2.06  1.85              1.69
25  1.54  1.18              1.29
40  1.00  1.00              1.00
60  1.00  1.02              0.98
75  1.18  1.09              1.07
```

```python
fig, ax = plt.subplots()
ax.plot(ages, true_age / true_age[3], color="0.75", linewidth=3, label="truth, low power")
ax.plot(ages, pd_rel["GLM"], marker="o", label="GLM")
ax.plot(ages, pd_rel["GBM"], marker="o", label="GBM")
ax.set_xlabel("driver age (years)")
ax.set_ylabel("partial dependence relative to age 40")
ax.set_title("Young-driver effect: the GLM overstates it without the interaction")
ax.legend()
```

A partial dependence is an average over the portfolio (section 9.3), so both
models' young-driver figures mix high-power cars with the other cars. The GLM
figure is higher: without an interaction term it charges every young driver part
of the extra load of high-power cars. The next section finds that interaction.

### 3.4 Interactions from permutations

The interaction screen of section 9.4 shuffles two features separately and
together. With losses `L_A`, `L_B` and `L_AB`, the sum `L_A + L_B - L_AB` is
zero if neither feature interacts with anything, and positive if the two
interact with each other or with a common third feature. Table 9.11 prints the
three losses for three pairs with aircraft age (from the values, its columns are
`L_AB`, `L_A` and `L_B`):

```python
table_9_11 = [("region", 0.291, 0.253, 0.135, 0.096), ("nu_registered", 0.115, 0.016, 0.135, 0.037),
              ("faa_eng_hp_char", 0.123, 0.002, 0.135, 0.014)]
(GT(pd.DataFrame([{"pair": f"{name} x aircraft age", "L_AB": l_ab, "L_A": l_a, "L_B": l_b,
                   "prospicio": l_a + l_b - l_ab, "book": printed}
                  for name, l_ab, l_a, l_b, printed in table_9_11]), rowname_col="pair")
 .tab_header("Table 9.11: interaction with aircraft age")
 .fmt_number(decimals=3))
```

```text
                                 L_AB    L_A    L_B  prospicio   book
region x aircraft age           0.291  0.253  0.135      0.097  0.096
nu_registered x aircraft age    0.115  0.016  0.135      0.036  0.037
faa_eng_hp_char x aircraft age  0.123  0.002  0.135      0.014  0.014
```

Our sums differ from the printed values by 0.001, because the three inputs are
rounded. The same screen on the boosting model, over each pair of portfolio
features and averaged over the seven folds:

```python
top = pd.DataFrame([{"pair": f"{r['a']} x {r['b']}", "strength": r["strength"]} for r in screen.table()[:6]])
(GT(top, rowname_col="pair")
 .tab_header("Interaction screen of the boosting model",
             "Strength of each pair of features, mean over seven folds (pseudo-R2 units)")
 .fmt_number("strength", decimals=4))
```

```text
                  strength
age x power         0.0051
age x log_te        0.0012
veh_age x noise     0.0006
veh_age x region    0.0005
region x power      0.0003
region x noise      0.0000
```

Age and power come first, which is the planted interaction. The next pairs have
a quarter of its strength or less, and their order changes with the shuffle (the
third pair includes the noise feature, which interacts with nothing). One more
column for the first pair raises the cross-validated GLM score:

```python
print(f"GLM cv pseudo-R2 {score_full['GLM']:.3f}, with a young-driver high-power term "
      f"{np.mean(glm_with_interaction):.3f}")
```

```text
GLM cv pseudo-R2 0.055, with a young-driver high-power term 0.058
```

In the monograph, the leaves of a decision tree fitted to the ratio of boosting
to GLM predictions did this work (cross-validated pseudo-R2 from 0.151 to
0.156). A `Booster` with one tree and a few leaves is a decision tree, and the
hand-made column above uses the same idea.

### 3.5 The test data, once

All decisions above used only the training folds. By the rule of section 3.2,
the test data is looked at once, at the end. The block fits each model again on
all seven training folds, with the make encoding from all of them, and scores
each on folds 7 to 9:

```python
log_te_all = make_feature(64)
fit_all, test_all = frame(train, log_te_all), frame(~train, log_te_all)
y_fit, y_test, e_test = claims[train].tolist(), claims[~train], exposure[~train]
final = {
    "baseline GLM": Glm("poisson").fit(X_base.select(rows(train)), y_fit).predict(X_base.select(rows(~train))),
    "our GLM": Glm("poisson").fit(glm_design(fit_all), y_fit).predict(glm_design(test_all)),
    "our GLM + interaction": Glm("poisson").fit(glm_design(fit_all, interaction=True), y_fit)
    .predict(glm_design(test_all, interaction=True)),
    "GBM": Booster("poisson", n_rounds=150, learning_rate=0.05, params=gbm_params, seed=1)
    .fit(tree_design(fit_all), y_fit).predict(tree_design(test_all)),
    "truth": truth[~train],
}
for name, mu_test in final.items():
    print(f"{name:22s} {pseudo_r2(y_test, mu_test, e_test):.3f}")
```

```text
baseline GLM           0.025
our GLM                0.068
our GLM + interaction  0.071
GBM                    0.064
truth                  0.089
```

Cross-validation predicted this ranking. The last digit of the boosting model
can change with the LightGBM version; the last digits of the GLMs cannot.

## 4. Challenges in ratemaking models (chapter 10)

The last chapter lists ways that a good cross-validation score can still
mislead:

- data with a misunderstood meaning (the monograph's aircraft age was the years
  since the registration certificate, not since manufacture),
- technical errors, such as factor levels coded differently in training and
  production (a `Terms` coding that you fit once and reuse prevents this),
- data that breaks the assumptions of the model,
- coefficients and business mix that drift over time,
- regulation,
- outliers and data quality.

Cross-validation fails when the future differs from the folds, so monitor the
model in production. `actual_vs_expected` in `prospicio.models` compares actual
and expected claims by period, with a z-score for each period and a test for a
trend in the ratio (section 10.4).
