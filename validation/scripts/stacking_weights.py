"""Regenerate validation/data/stacking_lpd.csv and
validation/reference/stacking_weights.csv.

    pip install numpy scipy statsmodels
    python validation/scripts/stacking_weights.py

Pointwise held-out log predictive densities of four claim-count models on
validation/data/glm_policies.csv (10 folds, row i in fold i % 10, log
exposure offset): Poisson with intercept only, Poisson with age, Poisson
with age and region, and negative binomial (alpha 0.5) with age and
region, fitted by statsmodels.

Reference weights, from the lpd matrix:

- stacking: maximize sum_i log sum_k w_k exp(lpd_ik) over the simplex
  (Yao et al. 2018), by SLSQP with an analytic gradient as BayesBlend's
  MleStacking does (MIT, Ledger Investing), from several starts, then
  polished by Newton's method on the models kept;
- pseudo-BMA: softmax of the models' total lpd (no bootstrap).

R's loo::pseudobma_weights(BB = FALSE) agrees to 1e-12. loo's
stacking_weights stops early (constrOptim): its weights here are about 1e-3
off and its log score 3e-5 lower; the optimum above satisfies the KKT
conditions (gradient n on the models kept, below n on those dropped).
"""

import csv
import sys

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

DATA = "validation/data/glm_policies.csv"
LPD = "validation/data/stacking_lpd.csv"
OUT = "validation/reference/stacking_weights.csv"
MODELS = ["poisson_intercept", "poisson_age", "poisson_age_region", "nb_age_region"]


def load():
    rows = list(csv.DictReader(open(DATA)))
    age = np.array([float(r["age"]) for r in rows])
    region = [r["region"] for r in rows]
    dummies = [np.array([float(g == lv) for g in region]) for lv in "BCD"]
    y = np.array([float(r["claims"]) for r in rows])
    off = np.log(np.array([float(r["exposure"]) for r in rows]))
    n = len(rows)
    designs = {
        "poisson_intercept": np.ones((n, 1)),
        "poisson_age": np.column_stack([np.ones(n), age]),
        "poisson_age_region": np.column_stack([np.ones(n), age] + dummies),
    }
    designs["nb_age_region"] = designs["poisson_age_region"]
    return designs, y, off


def lpd_matrix(designs, y, off):
    n = len(y)
    fold = np.arange(n) % 10
    out = np.zeros((n, len(MODELS)))
    for k, name in enumerate(MODELS):
        X = designs[name]
        for f in range(10):
            tr, te = fold != f, fold == f
            if name.startswith("nb"):
                fam = sm.families.NegativeBinomial(alpha=0.5)
            else:
                fam = sm.families.Poisson()
            res = sm.GLM(y[tr], X[tr], family=fam, offset=off[tr]).fit(tol=1e-14)
            mu = np.exp(X[te] @ res.params + off[te])
            if name.startswith("nb"):
                size = 1 / 0.5
                out[te, k] = nbinom.logpmf(y[te], size, size / (size + mu))
            else:
                out[te, k] = poisson.logpmf(y[te], mu)
    return out


def stacking(lpd):
    # Scale each row by its maximum for stability; the weights do not change.
    e = np.exp(lpd - lpd.max(axis=1, keepdims=True))
    k = lpd.shape[1]
    fun = lambda w: -np.log(e @ w).sum()
    grad = lambda w: -(e / (e @ w)[:, None]).sum(axis=0)
    best = None
    for start in [np.full(k, 1 / k)] + [np.eye(k)[j] * 0.7 + 0.3 / k for j in range(k)]:
        res = minimize(fun, start, jac=grad, method="SLSQP",
                       bounds=[(0, 1)] * k,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"ftol": 1e-15, "maxiter": 10_000})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    w[w < 1e-10] = 0.0
    w /= w.sum()
    # SLSQP stops at about 1e-7 in the gradient: polish with Newton's method
    # on the models it keeps (w = (v, 1 - sum v)).
    s = np.flatnonzero(w)
    for _ in range(50):
        es = e[:, s]
        p = es / (es @ w[s])[:, None]
        g = p.sum(axis=0)
        h = -(p.T @ p)
        r = g[:-1] - g[-1]
        hr = h[:-1, :-1] - h[:-1, -1:] - h[-1:, :-1] + h[-1, -1]
        step = np.linalg.solve(hr, -r)
        w[s[:-1]] += step
        w[s[-1]] = 1.0 - w[s[:-1]].sum()
    return w


def pseudo_bma(lpd):
    elpd = lpd.sum(axis=0)
    z = np.exp(elpd - elpd.max())
    return z / z.sum()


def main():
    designs, y, off = load()
    lpd = lpd_matrix(designs, y, off)
    with open(LPD, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(MODELS)
        for row in lpd:
            w.writerow([repr(float(v)) for v in row])
    rows = []
    for method, weights, tol in [("stacking", stacking(lpd), 1e-10),
                                 ("pseudo_bma", pseudo_bma(lpd), 1e-12)]:
        for name, v in zip(MODELS, weights):
            rows.append([method, name, repr(float(v)), tol, 0.0, "scipy SLSQP / NumPy"])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", "model", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {LPD} and {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
