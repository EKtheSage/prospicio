"""Regenerate validation/reference/stacking_grid.csv.

    pip install numpy
    python validation/scripts/stacking_grid.py

Exact posterior moments of Bayesian stacking and hierarchical stacking by
grid integration (no sampler), for the NUTS fits of act_bayes::stacking.
Two models from validation/data/stacking_lpd.csv: poisson_intercept and
poisson_age_region (the second is the reference, logit 0); the covariate
is age from validation/data/glm_policies.csv, centred and divided by twice
its standard deviation, as BayesBlend scales continuous covariates.

- bayes: w ~ Dirichlet(1, 1), likelihood sum_i log(w e_i1 + (1 - w) e_i2);
  posterior mean and sd of w (the first model's weight).
- hierarchical: w_i1 = logistic(a + b x_i), a ~ N(0, 1), b ~ N(0, 1)
  (BayesBlend's no-pooling defaults); posterior mean and sd of a and b,
  and the posterior mean weight of the first model at x = -1, 0, 1.
- synthetic: the same hierarchical model on validation/data/stacking_synthetic.csv
  (written here, numpy seed 20261005): x ~ U(-1, 1), y ~ N(m(x), 1) with
  m = 0 for x < 0 and 1 above; model A predicts N(0, 1), model B N(1, 1).
  The data decide the weights, so the posterior is far from the prior.
"""

import csv
import sys

import numpy as np

LPD = "validation/data/stacking_lpd.csv"
DATA = "validation/data/glm_policies.csv"
OUT = "validation/reference/stacking_grid.csv"
SYN = "validation/data/stacking_synthetic.csv"


def load():
    rows = list(csv.reader(open(LPD)))
    names = rows[0]
    lpd = np.array([[float(v) for v in r] for r in rows[1:]])
    a = lpd[:, names.index("poisson_intercept")]
    b = lpd[:, names.index("poisson_age_region")]
    age = np.array([float(r["age"]) for r in csv.DictReader(open(DATA))])
    x = (age - age.mean()) / (2 * age.std())
    m = np.maximum(a, b)
    return np.exp(a - m), np.exp(b - m), x


def synthetic():
    rng = np.random.default_rng(20261005)
    x = rng.uniform(-1, 1, 300).round(4)
    y = (rng.normal(0, 1, 300) + (x > 0)).round(4)
    la = -0.5 * y ** 2 - 0.5 * np.log(2 * np.pi)
    lb = -0.5 * (y - 1) ** 2 - 0.5 * np.log(2 * np.pi)
    with open(SYN, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["x", "lpd_a", "lpd_b"])
        for row in zip(x, la, lb):
            out.writerow([repr(float(v)) for v in row])
    m = np.maximum(la, lb)
    return np.exp(la - m), np.exp(lb - m), x


def hierarchical(case, e1, e2, x):
    a = np.linspace(-8, 8, 801)
    A, B = np.meshgrid(a, a, indexing="ij")
    logp = -0.5 * (A ** 2 + B ** 2)
    for i in range(len(x)):
        w1 = 1 / (1 + np.exp(-(A + B * x[i])))
        logp += np.log(w1 * e1[i] + (1 - w1) * e2[i])
    p = normalize(logp)
    rows = []
    for name, v in [("alpha", A), ("beta", B)]:
        m = (p * v).sum()
        s = np.sqrt((p * (v - m) ** 2).sum())
        rows += [[case, name, "mean", m, 0.1 * s, 0.0], [case, name, "sd", s, 0.0, 0.08]]
    for xv in (-1.0, 0.0, 1.0):
        w1 = 1 / (1 + np.exp(-(A + B * xv)))
        m = (p * w1).sum()
        s = np.sqrt((p * (w1 - m) ** 2).sum())
        rows.append([case, f"weight_at_{xv:g}", "mean", m, max(0.1 * s, 1e-3), 0.0])
    return rows


def normalize(logp):
    w = np.exp(logp - logp.max())
    return w / w.sum()


def main():
    e1, e2, x = load()
    rows = []
    # Bayesian stacking: w on (0, 1), Dirichlet(1, 1) is uniform.
    w = np.linspace(0, 1, 200_001)[1:-1]
    logp = np.array([np.log(wi * e1 + (1 - wi) * e2).sum() for wi in w])
    p = normalize(logp)
    m = (p * w).sum()
    s = np.sqrt((p * (w - m) ** 2).sum())
    rows += [["bayes", "w", "mean", m, 0.1 * s, 0.0], ["bayes", "w", "sd", s, 0.0, 0.08]]

    rows += hierarchical("hierarchical", e1, e2, x)
    rows += hierarchical("synthetic", *synthetic())
    with open(OUT, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["case", "param", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for r in rows:
            out.writerow([r[0], r[1], r[2], repr(float(r[3])), float(r[4]), r[5],
                          "NumPy grid integration"])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
