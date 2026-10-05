"""Regenerate validation/reference/bayes_glm_grid.csv.

    pip install numpy scipy
    python validation/scripts/bayes_glm_grid.py

Exact posterior means and standard deviations of two Bayesian GLMs, by
brute-force integration on a grid (no sampler involved), for the NUTS
fits of act_bayes::glm to be checked against. Data: the first 200 rows
of validation/data/glm_policies.csv, with x = (age - 50) / 10.

- poisson: claims ~ Poisson(exposure * exp(b0 + b1 x)), priors
  b0 ~ N(0, 10^2), b1 ~ N(0, 2.5^2); grid over (b0, b1).
- gaussian: (gauss - 200) / 10 ~ N(b0 + b1 x, phi), priors as above and
  phi ~ half-normal(10) (on the variance); grid over (b0, b1, log phi)
  with the Jacobian phi.

Each grid spans +-8 posterior standard deviations (found from a coarse
pass) with 401 points a side for two dimensions and 161 for three; the
moments are converged to well below the Monte Carlo error they are
compared at.
"""

import csv
import sys

import numpy as np
from scipy.special import gammaln

DATA = "validation/data/glm_policies.csv"
OUT = "validation/reference/bayes_glm_grid.csv"


def load():
    rows = list(csv.DictReader(open(DATA)))[:200]
    x = (np.array([float(r["age"]) for r in rows]) - 50.0) / 10.0
    claims = np.array([float(r["claims"]) for r in rows])
    expo = np.array([float(r["exposure"]) for r in rows])
    gauss = (np.array([float(r["gauss"]) for r in rows]) - 200.0) / 10.0
    return x, claims, expo, gauss


def moments(logp, axes, exp_last=False):
    w = np.exp(logp - logp.max())
    w /= w.sum()
    out = []
    for k, a in enumerate(axes):
        if exp_last and k == len(axes) - 1:
            a = np.exp(a)
        m = (w * a).sum()
        out.append((m, np.sqrt((w * (a - m) ** 2).sum())))
    return out


def poisson_logp(b0, b1, x, y, e):
    eta = b0[..., None] + b1[..., None] * x
    ll = (y * (np.log(e) + eta) - e * np.exp(eta) - gammaln(y + 1)).sum(-1)
    return ll - 0.5 * (b0 / 10) ** 2 - 0.5 * (b1 / 2.5) ** 2


def gaussian_logp(b0, b1, lphi, x, y):
    phi = np.exp(lphi)
    out = np.empty(b0.shape)
    for i in range(b0.shape[0]):  # chunk over the first axis for memory
        r = y - (b0[i][..., None] + b1[i][..., None] * x)
        ss = (r * r).sum(-1)
        n = len(y)
        out[i] = -0.5 * ss / phi[i] - 0.5 * n * np.log(2 * np.pi * phi[i])
    return (out - 0.5 * (b0 / 10) ** 2 - 0.5 * (b1 / 2.5) ** 2
            - 0.5 * (phi / 10) ** 2 + lphi)


def grid(centres, sds, n):
    axes = [np.linspace(c - 8 * s, c + 8 * s, n) for c, s in zip(centres, sds)]
    return np.meshgrid(*axes, indexing="ij")


def refine(logp_fn, centres, sds, n):
    for size in (41, n):
        g = grid(centres, sds, size)
        mom = moments(logp_fn(*g), g)
        centres = [m for m, _ in mom]
        sds = [s for _, s in mom]
    return mom


def main():
    x, claims, expo, gauss = load()
    pois = refine(lambda a, b: poisson_logp(a, b, x, claims, expo),
                  [-2.0, 0.0], [0.5, 0.5], 401)
    ols = np.polyfit(x, gauss, 1)
    resid = gauss - np.polyval(ols, x)
    gaus = refine(lambda a, b, l: gaussian_logp(a, b, l, x, gauss),
                  [ols[1], ols[0], np.log(resid.var())], [0.2, 0.2, 0.2], 161)
    # The dispersion's moments on its own scale, from the final grid.
    g = grid([m for m, _ in gaus], [s for _, s in gaus], 161)
    gaus = gaus[:2] + moments(gaussian_logp(*g, x, gauss), g, exp_last=True)[2:]
    rows = []
    for case, names, mom in [("poisson", ["b0", "b1"], pois),
                             ("gaussian", ["b0", "b1", "dispersion"], gaus)]:
        for name, (m, s) in zip(names, mom):
            # Monte Carlo tolerances for 4 chains of 1000 draws (ESS in the
            # thousands): the mean to 0.1 sd (about 4 MCSE), the sd to 8%.
            rows.append([case, name, "mean", repr(float(m)), 0.1 * float(s), 0.0])
            rows.append([case, name, "sd", repr(float(s)), 0.0, 0.08])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "param", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for r in rows:
            w.writerow(r + ["NumPy grid integration"])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
