"""Regenerate validation/reference/bonus_malus_scipy.csv.

    python3 validation/scripts/bonus_malus_reference.py   # numpy and scipy

An independent implementation of `prospicio_pricing::bonus_malus` (issue
#563) for the validation test `bonus_malus_matches_scipy`. No printed table
from Lemaire (1995) or Denuit, Marechal, Pitrebois and Walhin (2007) was
available when this was written, so this is a second implementation of the
same formulas with a different integrator and solver, not parity with a
published source:

- the scale's table is built here from its rule, and the transition matrix
  from Poisson probabilities with the last column taking `P(N >= K)`;
- the stationary distribution solves `pi (P - I) = 0`, `sum pi = 1` by
  least squares (prospicio: Gaussian elimination with one row replaced);
- the portfolio integrals over `Theta` (mean 1, gamma or inverse Gaussian)
  are QUADPACK's adaptive `quad` on `ln theta` between the quantiles 1e-16
  and 1 - 1e-16 (prospicio: Gauss-Legendre panels doubled to convergence);
- Loimaranta's efficiency is a central difference of `ln r(lambda)` in
  `ln lambda` with a step of 1e-5 (prospicio: the exact derivative of `pi`);
- the years to stationarity count steps of the chain from the start level
  until the total variation distance is below the tolerance.

Quantities: `stationary` and `relativity` at level `arg` over the
portfolio; `stationary_at` at level `arg` for frequency `arg2`;
`efficiency` at frequency `arg` with the case's optimal relativities;
`years` to tolerance `arg` at frequency `arg2`.
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy import integrate, stats

SOURCE = "scipy reimplementation"
OUT = Path(__file__).resolve().parent.parent / "reference" / "bonus_malus_scipy.csv"

CASES = [
    # -1/+top on six levels, a negative binomial portfolio.
    dict(name="top6", levels=6, start=5, down=1, up=5, mixing="gamma", cv=1.0,
         weights=[1.0], lambdas=[0.1]),
    # -1/+2 on ten levels, a Poisson-inverse Gaussian portfolio.
    dict(name="ten", levels=10, start=6, down=1, up=2, mixing="inverse_gaussian", cv=0.8,
         weights=[1.0], lambdas=[0.15]),
    # -1/+3 on twelve levels with three a priori classes.
    dict(name="classes", levels=12, start=7, down=1, up=3, mixing="gamma", cv=1.3,
         weights=[0.5, 0.3, 0.2], lambdas=[0.06, 0.1, 0.18]),
]


def table(levels, down, up):
    top = levels - 1
    k_max = -(-top // up)
    return [[max(l - down, 0) if k == 0 else min(l + k * up, top) for k in range(k_max + 1)]
            for l in range(levels)]


def matrix(t, lam):
    s, k_max = len(t), len(t[0]) - 1
    p = [stats.poisson.pmf(k, lam) for k in range(k_max)]
    p.append(stats.poisson.sf(k_max - 1, lam))
    m = np.zeros((s, s))
    for l, row in enumerate(t):
        for k, j in enumerate(row):
            m[l, j] += p[k]
    return m


def stationary(t, lam):
    m = matrix(t, lam)
    s = len(t)
    a = np.vstack([(m - np.eye(s)).T, np.ones(s)])
    b = np.zeros(s + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(a, b, rcond=None)[0]


def mixing(case):
    cv = case["cv"]
    if case["mixing"] == "gamma":
        a = 1.0 / cv**2
        return stats.gamma(a, scale=1.0 / a)
    phi = 1.0 / cv**2
    # scipy's invgauss(mu, scale) has mean mu * scale and shape scale.
    return stats.invgauss(1.0 / phi, scale=phi)


def portfolio(case):
    t = table(case["levels"], case["down"], case["up"])
    d = mixing(case)
    lo, hi = math.log(d.ppf(1e-16)), math.log(d.isf(1e-16))
    w = np.array(case["weights"]) / sum(case["weights"])
    s = case["levels"]
    pi, theta_pi = np.zeros(s), np.zeros(s)
    for l in range(s):
        def f(x, moment):
            th = math.exp(x)
            v = sum(wk * stationary(t, lk * th)[l] for wk, lk in zip(w, case["lambdas"]))
            return v * th**moment * d.pdf(th) * th

        pi[l] = integrate.quad(f, lo, hi, args=(0,), epsabs=1e-15, epsrel=1e-13, limit=400)[0]
        theta_pi[l] = integrate.quad(f, lo, hi, args=(1,), epsabs=1e-15, epsrel=1e-13,
                                     limit=400)[0]
    return pi, theta_pi / pi


def efficiency(t, lam, r):
    h = 1e-5
    up = math.log(stationary(t, lam * (1 + h)) @ r)
    down = math.log(stationary(t, lam * (1 - h)) @ r)
    return (up - down) / (math.log1p(h) - math.log1p(-h))


def years(t, start, lam, tol):
    m, pi = matrix(t, lam), stationary(t, lam)
    p = np.zeros(len(t))
    p[start] = 1.0
    n = 0
    while 0.5 * np.abs(p - pi).sum() >= tol:
        p = p @ m
        n += 1
    return n


def main():
    rows = []
    for case in CASES:
        params = ";".join([
            f"levels={case['levels']}", f"start={case['start']}", f"down={case['down']}",
            f"up={case['up']}", f"mixing={case['mixing']}", f"cv={case['cv']}",
            "weights=" + "|".join(str(w) for w in case["weights"]),
            "lambdas=" + "|".join(str(x) for x in case["lambdas"]),
        ])

        def add(quantity, arg, arg2, value, abs_tol, rel_tol):
            rows.append([case["name"], params, quantity, arg, arg2, repr(float(value)),
                         abs_tol, rel_tol, SOURCE])

        t = table(case["levels"], case["down"], case["up"])
        pi, r = portfolio(case)
        assert abs(pi.sum() - 1) < 1e-11 and abs(pi @ r - 1) < 1e-11, case["name"]
        for l in range(case["levels"]):
            add("stationary", l, "", pi[l], 1e-11, 0.0)
            add("relativity", l, "", r[l], 1e-10, 1e-10)
        lam = case["lambdas"][0]
        for l, v in enumerate(stationary(t, lam)):
            add("stationary_at", l, lam, v, 1e-13, 0.0)
        for x in (0.05, 0.1, 0.2):
            add("efficiency", x, "", efficiency(t, x, r), 1e-8, 0.0)
        for tol in (1e-2, 1e-3):
            add("years", tol, lam, years(t, case["start"], lam, tol), 0.0, 0.0)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"{len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
