# Beyond the GLM, part 2: correlated features, two stages and black boxes

The second part of the guide to CAS Monograph 16, *From GLMs to Comprehensive
Insurance Pricing: Techniques and Challenges* (Chalk, Deacon, Guillen and
Martinelli, 2025). The first part,
[`glm-to-comprehensive-pricing-chalk.md`](glm-to-comprehensive-pricing-chalk.md),
covers chapters 3 to 6 (measuring performance, penalized regression, nonlinear
effects, high cardinality factors) and explains what can and cannot be
reproduced. This part covers chapters 7 to 10. The text is the authors'; what
follows is an explanation in new words with the numbers recomputed, and table
and section numbers are the monograph's.

The block below rebuilds what part 1 built and this part uses, in one go: the
simulated portfolio with its known truth (20,000 policies, folds from the
policy number, folds 7 to 9 for testing), the pseudo-R2 helper and
cross-validation loop, the baseline GLM and boosting designs, the age steps the
lasso keeps, and the make's out-of-fold target encoding. Part 1 explains each
piece.

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


print(f"{n} policies, {claims.sum():.0f} claims; age steps kept by the lasso at {kept}")
```

```text
20000 policies, 3659 claims; age steps kept by the lasso at [20, 23, 26, 29, 32, 62, 65, 77]
```

## 1. Highly correlated variables (chapter 7)

When two features carry nearly the same information, a GLM can split their joint
effect between them in many ways that fit almost equally well, so small changes
in the data swing the coefficients (section 7.2). The portfolio's `exper` is
driver age less 17 less a small random amount, correlated 0.997 with age, and has
no effect of its own. Fitting age and experience together on each of the seven
training sets of the cross-validation shows the swing, and what ridge and the
lasso do with it (section 7.4):

```python
X_corr = Design(columns(ones, age, exper), ["(Intercept)", "age", "exper"], offset=log_expo)
fits_by = {"GLM": lambda d, y: Glm("poisson").fit(d, y),
           "ridge": lambda d, y: ElasticNet("poisson", alpha=0.0, lam=0.01).fit(d, y),
           "lasso": lambda d, y: ElasticNet("poisson", alpha=1.0, lam=0.001).fit(d, y)}
for name, fitter in fits_by.items():
    coefs = np.array([fitter(X_corr.select(rows(train & (fold != k))), claims[train & (fold != k)].tolist()).coefficients[1:]
                      for k in range(7)])
    print(f"{name:5s} age {coefs[:, 0].min():+.4f} to {coefs[:, 0].max():+.4f}   "
          f"exper {coefs[:, 1].min():+.4f} to {coefs[:, 1].max():+.4f}   sum {coefs.sum(1).mean():+.4f}")
```

```text
GLM   age -0.0540 to -0.0409   exper +0.0340 to +0.0467   sum -0.0072
ridge age -0.0061 to -0.0052   exper -0.0016 to -0.0010   sum -0.0070
lasso age -0.0074 to -0.0066   exper +0.0000 to +0.0000   sum -0.0070
```

The unpenalized coefficients move a lot from one training set to the next while
their sum, the effect of a year of age-and-experience together, barely moves.
Ridge shares the effect between the two and holds both steady; the lasso keeps
mostly one and drops the other, which is stable here but can flip between
training sets when the two are closer still (section 7.5.3). Principal
components (section 7.4.5) are not in prospicio; `numpy.linalg.svd` on the
standardized columns gives them, and the components go into a `Design` like any
other column.

## 2. Modelling in two stages (chapter 8)

### 2.1 The marketing discount as an offset

Chapter 8's example is the one fully specified data set in the monograph: a
million simulated private auto policies, one year each, with age group (A to E)
and no-claims bonus (NCB 0 to 4) strongly related. Table 8.2 prints the number
of policies in each cell, Table 8.3 the true relativities, and section 8.1 a base
frequency of 10% for age A and NCB 0. A Poisson GLM can be fitted to the 25
cells with expected claims as the response, which gives the fit that the
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

both = Terms().intercept().factor("age").factor("ncb").fit(data_8)
full = Glm("poisson").fit(both.design(data_8, offset=np.log(policies).tolist()), expected_claims)
print("age and NCB fitted:", [f"{math.exp(b):.3f}" for b in full.coefficients])
age_only = Terms().intercept().factor("age").fit(data_8)
mkt_offset = np.log(policies * np.array([marketing[b] for _, b, _ in cells_8])).tolist()
offset_fit = Glm("poisson").fit(age_only.design(data_8, offset=mkt_offset), expected_claims)
rel = [math.exp(b) for b in offset_fit.coefficients]
print(f"NCB offset at the marketing scale: intercept {rel[0]:.4f} (printed 0.112)")
print("  age B to E:", [f"{r:.4f}" for r in rel[1:]], "printed [0.849, 0.925, 0.980, 1.057]")
```

```text
policies 1,000,000
age and NCB fitted: ['0.100', '0.700', '0.600', '0.500', '0.400', '0.950', '0.900', '0.850', '0.800']
NCB offset at the marketing scale: intercept 0.1125 (printed 0.112)
  age B to E: ['0.8485', '0.9175', '0.9825', '1.0586'] printed [0.849, 0.925, 0.980, 1.057]
```

With age and NCB both in the model the GLM returns the truth (intercept 0.10,
then the age and NCB relativities of Table 8.3); the monograph's fit on its noisy
simulation, Table 8.4, is within about 0.025 of it. Offsetting the marketing
department's NCB scale and refitting age gives the intercept 0.112 that section
8.1 prints (0.1125, printed as 0.112) and age relativities of 0.849, 0.918,
0.983 and 1.059 against the printed 0.849, 0.925, 0.980 and 1.057. The differences are the monograph's
simulation noise; the reversal (age relativities rising with age, because age
now carries part of the NCB effect the marketing scale removed) is the same.

The risk premiums of the section follow, at a claim size of 10,000:

```python
print(f"true A/0 {0.10 * 1.0 * 1.0 * 10000:.0f}, true E/4 {0.10 * 0.4 * 0.8 * 10000:.0f} (printed 1,000 and 320)")
print(f"GLM E/4 at printed coefficients {0.10 * 0.39 * 0.826 * 10000:.0f} (printed 322)")
print(f"GLM age with marketing NCB {0.10 * 0.39 * 0.20 * 10000:.0f} (printed 78)")
print(f"offset model E/4: printed coefficients {0.112 * 1.057 * 0.20 * 10000:.0f} (printed 237), "
      f"recomputed {rel[0] * rel[4] * 0.20 * 10000:.0f}")
```

```text
true A/0 1000, true E/4 320 (printed 1,000 and 320)
GLM E/4 at printed coefficients 322 (printed 322)
GLM age with marketing NCB 78 (printed 78)
offset model E/4: printed coefficients 237 (printed 237), recomputed 238
```

### 2.2 Offsets and residuals give the same second stage

A second-stage model can start where a first stage stopped in two ways (sections
8.3 to 8.5): an offset of the log of the first stage's predicted counts, or the
residual (actual over predicted counts) as the response with the predicted
counts as weights. Yan and others, and Shi, show the two are identical for a
Poisson log-link model. On the portfolio, a first-stage GLM without region and a
second stage that fits region alone:

```python
X_s1 = Design(columns(ones, age, veh_age, *power_d), ["(Intercept)", "age", "veh_age", "power[mid]", "power[high]"],
              offset=log_expo)
tr = rows(train)
mu1 = np.asarray(Glm("poisson").fit(X_s1.select(tr), claims[tr].tolist()).predict(X_s1.select(tr)))
reg_cols = columns(np.ones(len(tr)), *[d[tr] for d in region_d])
reg_names = ["(Intercept)", *factor_names[:5]]
by_offset = Glm("poisson").fit(Design(reg_cols, reg_names, offset=np.log(mu1).tolist()), claims[tr].tolist())
by_residual = Glm("poisson").fit(Design(reg_cols, reg_names, weights=mu1.tolist()), (claims[tr] / mu1).tolist())
print("offset:  ", [f"{math.exp(b):.4f}" for b in by_offset.coefficients])
print("residual:", [f"{math.exp(b):.4f}" for b in by_residual.coefficients])
```

```text
offset:   ['0.9301', '1.2148', '0.8598', '1.0619', '1.0436', '1.4061']
residual: ['0.9301', '1.2148', '0.8598', '1.0619', '1.0436', '1.4061']
```

The fitted second stage is the same to the printed digits. In section 8.4's
summary by zip code the first rows have residuals 0 / 0.014 = 0 and 1 / 0.739 =
1.353. Section 8.9.1 adds the caution that the first stage's predictions should
be cross-fitted (each record predicted by a model that did not see it), as
`expected` was in part 1's section 4.4: an overfitted first stage leaves the
second nothing to find (Table 8.12), while cross-fitted predictions leave
residuals such as 3 / 5 = 0.6 and 7 / 5 = 1.4 (Table 8.13). prospicio's `Gam`
also takes tensor product smooths, so a spatial second stage of the kind in
Figure 8.2 is a `Gam` with `tensors=[["lon", "lat"]]` and the first stage as
offset.

### 2.3 Combining tables and area groups

Section 8.7 multiplies the seven aircraft tables into one relativity for the
Cessna 172M, and section 8.9.2 replaces 30,000 zip code relativities by 100
area groups in equal multiplicative steps between the smallest (0.379) and
largest (1.851) relativity:

```python
table_8_11 = [1.30, 1.15, 1.00, 1.36, 1.69, 1.00, 0.41, 0.97]
print(f"Cessna 172M relativity {math.prod(table_8_11):.2f} (printed 1.37)")
step = (1.851 / 0.379) ** (1 / 99)
print(f"range {1.851 / 0.379:.3f} (printed 4.884), exact step {step:.5f}, printed step 1.0162, "
      f"1.0162^99 = {1.0162 ** 99:.3f} (printed 4.908)")
print("groups 1, 2, 3, 22, 56:", [f"{0.379 * 1.0162 ** (g - 1):.3f}" for g in [1, 2, 3, 22, 56]],
      "printed [0.379, 0.385, 0.391, 0.531, 0.917]")
```

```text
Cessna 172M relativity 1.37 (printed 1.37)
range 4.884 (printed 4.884), exact step 1.01615, printed step 1.0162, 1.0162^99 = 4.908 (printed 4.908)
groups 1, 2, 3, 22, 56: ['0.379', '0.385', '0.391', '0.531', '0.917'] printed [0.379, 0.385, 0.391, 0.531, 0.917]
```

The printed step 1.0162 is the exact 1.01615 rounded up, which is why 99 steps
of it overshoot the range (4.908 against 4.884); the group relativities of
Table 8.15 are built from the rounded step.

## 3. Learning from black-box models (chapter 9)

### 3.1 The boosting model's variance

The gradient boosting benchmark is, in the monograph's words, a very complicated
GLM: thousands of steps and interactions. Its training performance keeps rising
as trees are added while its validation performance stalls (Figure 9.1). On fold
0 of the portfolio:

```python
fit_rows, val = rows(train & (fold != 0)), rows(fold == 0)
for rounds in [50, 150, 600]:
    booster = Booster("poisson", n_rounds=rounds, learning_rate=0.05,
                      params={"num_leaves": 7, "min_data_in_leaf": 200, "num_threads": 1}, seed=1)
    b_fit = booster.fit(X_tree.select(fit_rows), claims[fit_rows].tolist())
    gap_train = pseudo_r2(claims[fit_rows], b_fit.predict(X_tree.select(fit_rows)), exposure[fit_rows])
    gap_val = pseudo_r2(claims[val], b_fit.predict(X_tree.select(val)), exposure[val])
    print(f"{rounds:4d} trees: train {gap_train:.3f}, validation {gap_val:.3f}")
```

```text
  50 trees: train 0.044, validation 0.042
 150 trees: train 0.067, validation 0.052
 600 trees: train 0.109, validation 0.051
```

### 3.2 Permutation feature importance

Section 9.2 measures a feature's importance by how much worse the model predicts
held-out data when that feature's column is shuffled, which breaks its link to
the outcome and leaves everything else in place. It needs only predictions, so
it works for any model. Tables 9.1 to 9.4: three insureds, frequency age / 100,
a model that has learned exactly that, and the mean absolute error:

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

On the portfolio, "our GLM" for this chapter is the monograph's end point: the
age steps the lasso kept, steps for vehicle age, region, power and the make
encoding (at `K` = 64), with the noise and experience features left out (part 1, sections 3.2 and 4.4). The
boosting model gets the raw features (less `exper`, whose near-copy of age would
blur every comparison below, the caveat of section 9.2 about correlated
features) and the same make encoding, as the monograph does: it notes that the
two models would share one encoding but for its boosting library's built-in
one. As in section 9.5.1, each importance is computed on the held-out fold of
each of the seven cross-validation models and averaged, as the loss of
pseudo-R2 when the feature is shuffled. The loop also keeps what the next two
sections need: the losses from shuffling pairs and the partial dependence on age.

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


def shuffled(f, names, perm):
    return {k: (v[perm] if k in names else v) for k, v in f.items()}


designs = {"GLM": glm_design, "GBM": tree_design}
score_full = {m: [] for m in designs}
loss = {m: {x: [] for x in features} for m in designs}
pair_loss = {}
pd_by_age = {m: [] for m in designs}
glm_with_interaction = []
for k in range(7):
    log_te_k = make_feature(64, held_out=k)
    fit_mask, val_mask = train & (fold != k), fold == k
    fit_f, val_f, y_val = frame(fit_mask, log_te_k), frame(val_mask, log_te_k), claims[val_mask]
    models = {"GLM": Glm("poisson").fit(glm_design(fit_f), claims[fit_mask].tolist()),
              "GBM": Booster("poisson", n_rounds=150, learning_rate=0.05, params=gbm_params, seed=1)
              .fit(tree_design(fit_f), claims[fit_mask].tolist())}
    perm = np.random.default_rng(k).permutation(val_mask.sum())
    for m, model in models.items():
        score = lambda f: pseudo_r2(y_val, model.predict(designs[m](f)), f["exposure"])
        full = score(val_f)
        score_full[m].append(full)
        for x in features:
            loss[m][x].append(full - score(shuffled(val_f, [x], perm)))
        if m == "GBM":
            for i, a in enumerate(features):
                for b in features[i + 1:]:
                    pair_loss.setdefault((a, b), []).append(full - score(shuffled(val_f, [a, b], perm)))
        pd_by_age[m].append([np.mean(model.predict(designs[m](dict(val_f, age=np.full(len(y_val), v)))))
                             for v in ages])
    with_i = Glm("poisson").fit(glm_design(fit_f, interaction=True), claims[fit_mask].tolist())
    glm_with_interaction.append(pseudo_r2(y_val, with_i.predict(glm_design(val_f, interaction=True)), val_f["exposure"]))

mean_loss = {m: {x: float(np.mean(v)) for x, v in loss[m].items()} for m in designs}
print(f"cv pseudo-R2: GLM {np.mean(score_full['GLM']):.3f}, GBM {np.mean(score_full['GBM']):.3f}")
print(f"{'feature':8s} {'GLM loss':>9s} {'GBM loss':>9s}")
for x in sorted(features, key=lambda x: -mean_loss["GLM"][x]):
    print(f"{x:8s} {mean_loss['GLM'][x]:9.3f} {mean_loss['GBM'][x]:9.3f}")
```

```text
cv pseudo-R2: GLM 0.055, GBM 0.048
feature   GLM loss  GBM loss
log_te       0.037     0.032
age          0.036     0.035
power        0.017     0.017
veh_age      0.010     0.008
region       0.010     0.004
noise        0.000    -0.001
```

The two models agree on which features matter and roughly in which order, and
the GBM gives the noise feature next to nothing although it was allowed to use
it. With the monograph's treatment of nonlinear effects and of the make, our
GLM now scores above this (untuned) boosting model, which the raw-feature
baseline GLM was far below in part 1's section 1.3.

The GLM's loss for noise is exactly zero because the GLM does not use it;
the monograph's GLM importances are never exactly zero because its lasso kept
each feature in at least one fold.

### 3.3 Partial dependence

Section 9.3 isolates a feature's effect in a black box by setting it to one value
for every record, averaging the predictions, and repeating for another value.
For a GLM without interactions the ratio of the two averages is exactly the
ratio of the feature's relativities. The printed `nu_registered` steps give 1.107
between 1 and 10, Tables 9.7 to 9.9 the same ratio from predictions, and the
boosting model of Table 9.10 1.104:

```python
nu = lambda v: -0.0001407 * v + 0.02936 * (v >= 2) + 0.07342 * (v >= 7) - 0.15822 * (v >= 16) - 0.17117 * (v >= 35)
print(f"GLM relativity 10 vs 1 from the steps {math.exp(nu(10) - nu(1)):.3f} (printed 1.107)")
print(f"Table 9.9 ratio of totals {0.01225 / 0.01107:.3f}, Table 9.10 GBM {0.005892 / 0.005336:.3f} (printed 1.107, 1.104)")
```

```text
GLM relativity 10 vs 1 from the steps 1.107 (printed 1.107)
Table 9.9 ratio of totals 1.107, Table 9.10 GBM 1.104 (printed 1.107, 1.104)
```

On the portfolio, the partial dependence of each model on driver age, relative
to age 40 and averaged over the seven held-out folds, next to the truth for a
low-power car (the true young-driver effect is larger for high power, which is
the interaction):

```python
true_age = np.exp(0.9 * np.exp(-(np.array(ages) - 18) / 6) + 0.0015 * np.maximum(np.array(ages) - 60, 0) ** 1.5)
pd_mean = {m: np.mean(np.array(v), axis=0) for m, v in pd_by_age.items()}
for i, a in enumerate(ages):
    print(f"age {a}: GLM {pd_mean['GLM'][i] / pd_mean['GLM'][3]:.2f}  GBM {pd_mean['GBM'][i] / pd_mean['GBM'][3]:.2f}  "
          f"true (low power) {true_age[i] / true_age[3]:.2f}")
```

```text
age 18: GLM 2.81  GBM 2.39  true (low power) 2.40
age 21: GLM 2.06  GBM 1.85  true (low power) 1.69
age 25: GLM 1.54  GBM 1.18  true (low power) 1.29
age 40: GLM 1.00  GBM 1.00  true (low power) 1.00
age 60: GLM 1.00  GBM 1.02  true (low power) 0.98
age 75: GLM 1.18  GBM 1.09  true (low power) 1.07
```

As section 9.3 cautions, a partial dependence is an average over the portfolio.
Both models' young-driver figures blend high-power cars, where the effect is
stronger, with the rest; the GLM's is the higher because, without an
interaction term, it charges every young driver part of the high-power cars'
extra load. The next section finds that interaction.

### 3.4 Interactions from permutations

Section 9.4's screen for interactions shuffles two features separately and
together: with `L_A`, `L_B` and `L_AB` the losses, `L_A + L_B - L_AB` is zero if
neither feature interacts with anything and positive if they interact with each
other or with a common third feature. Table 9.11 prints the three losses for
three pairs with aircraft age; its columns are, from the values, `L_AB`, `L_A`
and `L_B`:

```python
table_9_11 = [("region", 0.291, 0.253, 0.135, 0.096), ("nu_registered", 0.115, 0.016, 0.135, 0.037),
              ("faa_eng_hp_char", 0.123, 0.002, 0.135, 0.014)]
for name, l_ab, l_a, l_b, printed in table_9_11:
    print(f"{name:15s} x aircraft age: {l_a + l_b - l_ab:.3f} (printed {printed})")
```

```text
region          x aircraft age: 0.097 (printed 0.096)
nu_registered   x aircraft age: 0.036 (printed 0.037)
faa_eng_hp_char x aircraft age: 0.014 (printed 0.014)
```

The recomputed values differ from the printed ones by 0.001, the rounding of the
three inputs. The same screen on the boosting model, over every pair of the
portfolio's features and averaged over the seven folds, and then the GLM's
cross-validated score with and without one more column for the pair it puts
first:

```python
strength = {pair: mean_loss["GBM"][pair[0]] + mean_loss["GBM"][pair[1]] - np.mean(v) for pair, v in pair_loss.items()}
for (a, b), s in sorted(strength.items(), key=lambda kv: -kv[1])[:3]:
    print(f"{a:8s} x {b:8s} {s:.4f}")
print(f"GLM cv pseudo-R2 {np.mean(score_full['GLM']):.3f}, with a young-driver high-power term "
      f"{np.mean(glm_with_interaction):.3f}")
```

```text
age      x power    0.0044
age      x log_te   0.0025
region   x log_te   0.0010
GLM cv pseudo-R2 0.055, with a young-driver high-power term 0.058
```

Age and power come first, which is the interaction that was planted, and the
GLM gains when given it. In the monograph, the leaves of a decision tree fitted
to the ratio of boosting to GLM predictions played this role and lifted the
cross-validated pseudo-R2 from 0.151 to 0.156; prospicio has no decision tree,
but a `Booster` with one tree and a few leaves is one, and the hand-made column
above is the same idea.

### 3.5 The test data, once

Everything above was decided on the training folds. Section 3.2's rule is that
the test data is looked at once, at the end. Refitting each model on all seven
training folds (the make encoding from all of them) and scoring on folds 7 to 9:

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

The ranking on the test data is the one cross-validation promised: the truth
on top, then our GLM with the interaction, our GLM, the boosting model and the
baseline. The boosting model's last digit can move with the LightGBM version;
the GLMs' cannot.

## 4. Challenges in ratemaking models (chapter 10)

The last chapter is a list of ways a good cross-validation score can still mislead:
data whose meaning was misunderstood (the monograph's own aircraft age was the
years since the registration certificate, not since manufacture), technical
errors such as factor levels coded differently in training and production (a
`Terms` coding fitted once and reused avoids this), data that breaks the model's
assumptions, coefficients and business mix that drift over time, regulation,
outliers and data quality. Cross-validation fails when the future differs from
the folds, so a model in production is monitored: `actual_vs_expected` in
`prospicio.models` compares actual and expected claims by period with a z-score
for each period and a test for a trend in the ratio, the check that section 10.4
asks for.

## 5. What prospicio does not cover

Gaps found while writing this guide, filed for the Models lane:

- Model-agnostic explanations: permutation importance, partial dependence and
  the pairwise interaction screen are written out by hand above; a helper that
  takes any fitted model and a design would save the loops.
- Bounds on elastic net coefficients (glmnet's `lower.limits` and
  `upper.limits`) for monotone step functions (section 5.7.2), a built-in
  relaxed lasso (section 4.5), and speed on hinge bases: twenty hinges took over
  40 seconds against a quarter of a second for twenty steps.
- Step, hinge and MARS bases as `Terms` (section 5.5), and out-of-fold target
  encoding with a credibility constant (chapter 6).
- Generalized linear mixed models with a random intercept per level (section
  6.5).

Not in prospicio and not filed, since general tools do them: principal
components (section 7.4.5), word embeddings (section 6.4.4) and decision trees
for interaction leaves (section 9.4; a one-tree `Booster` is one). The monograph
mentions neural networks only through the universal approximation theorem
(section 8.2); prospicio's `Cann`, a GLM offset plus a network correction, is
the two-stage idea of chapter 8 with a network in the second stage, but the
monograph has nothing to reproduce with it. Price optimization and demand
modelling are outside the monograph.

## 6. The printed numbers of chapters 8 and 9 and prospicio's

| Where | Printed | Recomputed | Note |
|---|---|---|---|
| Table 8.6 offset fit | 0.112; 0.849, 0.925, 0.980, 1.057 | 0.1125; 0.849, 0.918, 0.983, 1.059 | the monograph's simulation noise |
| Section 8.1 premiums | 1,000, 320, 322, 78, 237 | 1,000, 320, 322, 78, 237 | matches (238 from unrounded coefficients) |
| Section 8.9.2 area groups | 1.0162, 4.908 | 1.01615, 4.908 | step rounded |
| Tables 9.1 to 9.4 | 0.133, 100 | 0.133, 100 | matches |
| Section 9.3 partial dependence | 1.107, 1.104 | 1.107, 1.104 | matches |
| Table 9.11 interactions | 0.096, 0.037, 0.014 | 0.097, 0.036, 0.014 | inputs rounded |
