# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-12T00:00:00Z"
# ///
"""Regenerate the severity-fit parity data and references from SciPy.

Run from the repository root:

    python validation/scripts/scipy_severity_fits.py

Writes `validation/data/severity_fits.csv`, synthetic large losses
reported above 150,000 (a stand-in for issue #800's 89 losses, which live in
another repository), and `validation/reference/severity_fits_scipy.csv`: the
maximum likelihood fit of every `prospicio_prob::severity_fit::Family` on two
data sets, by Nelder-Mead and then BFGS on SciPy's own `logpdf` and `logsf`,
an implementation independent of the Rust one.

- `plain`: every loss reported above 150,000, none censored.
- `mixed`: a second sample with reporting thresholds 0, 50,000 and 150,000
  in turn (a loss kept only above its own) and losses above 400,000
  capped there (censored). The Pareto families, which need every loss above
  `t`, are fitted on `plain` only.

A fit that runs to a boundary of its family (the gamma's shape to 0 on
`plain`) is checked on its log-likelihood alone, with a looser tolerance.

Each loss contributes `logpdf(y) - logsf(a)` (`logsf(y)` when censored),
`a` its reporting threshold; the Pareto families condition on `max(r, t)`
with `t = 150,000`, which here is `r`. Rows give each parameter, the
log-likelihood and the AIC, with the tolerances the Rust fit must meet.
"""

import csv
import math
import warnings

import numpy as np
import scipy
from scipy import optimize, stats

DATA = "validation/data/severity_fits.csv"
OUT = "validation/reference/severity_fits_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"
T = 150_000.0
CAP = 400_000.0


def datasets():
    rng = np.random.default_rng(800)
    x = np.exp(rng.normal(11.6, 1.03, 240))
    y = np.sort(x[x >= T])
    plain = (y, np.full(len(y), T), np.zeros(len(y), bool))
    x = np.exp(rng.normal(11.0, 1.0, 300))
    r = np.array([0.0, 50_000.0, T])[np.arange(len(x)) % 3]
    keep = x > r
    x, r = x[keep], r[keep]
    mixed = (np.minimum(x, CAP), r, x > CAP)
    return {"plain": plain, "mixed": mixed}


# family -> (parameter names, scipy distribution from the natural parameters,
# indices of the parameters optimized on the log scale)
FAMILIES = {
    "lognormal": (["meanlog", "sdlog"], lambda p: stats.lognorm(s=p[1], scale=math.exp(p[0]))),
    "gamma": (["shape", "scale"], lambda p: stats.gamma(p[0], scale=p[1])),
    "weibull": (["shape", "scale"], lambda p: stats.weibull_min(p[0], scale=p[1])),
    "loglogistic": (["shape", "scale"], lambda p: stats.fisk(p[0], scale=p[1])),
    "exponential": (["mean"], lambda p: stats.expon(scale=p[0])),
    "lomax": (["alpha", "scale"], lambda p: stats.lomax(p[0], scale=p[1])),
    "pareto": (["alpha"], lambda p: stats.pareto(p[0], scale=T)),
    # Riegel: xi = 1/alpha_tail, beta = t/alpha_ini, location t.
    "gen_pareto": (
        ["alpha_ini", "alpha_tail"],
        lambda p: stats.genpareto(1.0 / p[1], loc=T, scale=T / p[0]),
    ),
}


def loglik(dist, y, r, c):
    own = np.where(c, dist.logsf(y), dist.logpdf(y))
    return float(np.sum(own - dist.logsf(r)))


def starts(family, y):
    ly = np.log(y)
    m, s = ly.mean(), ly.std()
    if family == "lognormal":
        return [[m, s]]
    if family == "gamma":
        return [[0.5, y.mean() / 0.5], [0.1, y.mean() / 0.1], [2.0, y.mean() / 2.0]]
    if family == "weibull":
        return [[0.5, math.exp(m)], [1.0, math.exp(m)], [0.3, math.exp(m) / 10]]
    if family == "loglogistic":
        k = math.pi / (s * math.sqrt(3))
        return [[k, math.exp(m)], [2 * k, math.exp(m)], [0.5 * k, math.exp(m)]]
    if family == "exponential":
        return [[y.mean()]]
    if family == "lomax":
        return [[a, math.exp(m) / (2 ** (1 / a) - 1)] for a in (1.2, 2.5, 8.0)]
    if family == "pareto":
        return [[1.0], [2.0]]
    return [[1.0, 2.0], [0.5, 3.0], [2.0, 1.5]]


def fit(family, y, r, c):
    names, make = FAMILIES[family]
    lognormal = family == "lognormal"

    def natural(u):
        return [u[0], math.exp(u[1])] if lognormal else [math.exp(v) for v in u]

    def objective(u):
        if np.any(np.abs(u) > 700):
            return np.inf
        v = -loglik(make(natural(u)), y, r, c)
        return v if np.isfinite(v) else np.inf

    best = None
    for s in starts(family, y):
        u0 = [s[0], math.log(s[1])] if lognormal else [math.log(v) for v in s]
        res = optimize.minimize(
            objective, u0, method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 40_000, "maxfev": 40_000},
        )
        res = optimize.minimize(objective, res.x, method="BFGS", options={"gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    p = natural(best.x)
    return names, p, -best.fun


def rows(sets):
    for name, (y, r, c) in sets.items():
        for family in FAMILIES:
            if name != "plain" and family in ("pareto", "gen_pareto"):
                continue
            names, p, ll = fit(family, y, r, c)
            boundary = min(p[1:] if family == "lognormal" else p) < 1e-6
            if not boundary:
                for n, v in zip(names, p):
                    yield (family, name, n, v, 0.0, 1e-5)
            tol = 1e-4 if boundary else 1e-6
            yield (family, name, "log_likelihood", ll, tol, 0.0)
            yield (family, name, "aic", 2 * len(names) - 2 * ll, 2 * tol, 0.0)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    sets = datasets()
    with open(DATA, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["data", "loss", "threshold", "censored"])
        for name, (y, r, c) in sets.items():
            for v, a, k in zip(y, r, c):
                w.writerow([name, repr(float(v)), repr(float(a)), int(k)])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "data", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for family, data, q, v, a, rel in rows(sets):
            w.writerow([family, data, q, repr(float(v)), a, rel, SOURCE])
    sizes = {k: len(v[0]) for k, v in sets.items()}
    print(f"{sizes} losses; wrote {DATA} and {OUT}")


if __name__ == "__main__":
    main()
