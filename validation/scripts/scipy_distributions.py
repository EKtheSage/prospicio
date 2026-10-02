"""Regenerate validation/reference/distributions_scipy.csv from SciPy.

Run from the repository root:

    python validation/scripts/scipy_distributions.py

Each row is one parity case. `abs_tol` and `rel_tol` are the tolerances the
Rust result must meet (either one suffices).
"""

import csv
import math
import sys

import scipy
from scipy import stats

OUT = "validation/reference/distributions_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"

# (meanlog, sdlog) pairs covering light to very heavy tails.
LOGNORMALS = [(0.0, 1.0), (7.0, 0.5), (10.0, 2.0), (-1.0, 0.1)]
PROBS = [1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 0.995, 0.999999]


def rows():
    for meanlog, sdlog in LOGNORMALS:
        d = stats.lognorm(s=sdlog, scale=math.exp(meanlog))
        params = f"meanlog={meanlog};sdlog={sdlog}"
        yield ("lognormal", params, "mean", "", d.mean(), 0.0, 1e-13)
        yield ("lognormal", params, "variance", "", d.var(), 0.0, 1e-12)
        for p in PROBS:
            x = d.ppf(p)
            yield ("lognormal", params, "quantile", p, x, 0.0, 1e-12)
            yield ("lognormal", params, "cdf", x, d.cdf(x), 1e-15, 1e-12)


# Claim counts: Poisson(lambda), negative binomial (r, beta), with SciPy's
# nbinom(n=r, p=1/(1+beta)), and binomial (n, p).
POISSONS = [0.5, 3.0, 40.0]
NEGBINS = [(0.7, 10.0), (2.5, 1.5), (50.0, 0.2)]
BINOMIALS = [(12, 0.35), (300, 0.02), (40, 0.9)]
COUNT_PROBS = [0.01, 0.25, 0.5, 0.9, 0.999]


def count_rows(name, params, d):
    yield (name, params, "mean", "", d.mean(), 0.0, 1e-13)
    yield (name, params, "variance", "", d.var(), 0.0, 1e-12)
    lo, hi = int(d.ppf(0.001)), int(d.ppf(0.999))
    for k in sorted({0, 1, lo, int(d.mean()), hi}):
        yield (name, params, "pmf", k, d.pmf(k), 1e-300, 1e-11)
        yield (name, params, "cdf", k, d.cdf(k), 0.0, 1e-11)
    for p in COUNT_PROBS:
        yield (name, params, "quantile", p, d.ppf(p), 0.0, 0.0)


def counts():
    for lam in POISSONS:
        yield from count_rows("poisson", f"lambda={lam}", stats.poisson(lam))
    for r, beta in NEGBINS:
        yield from count_rows("negative_binomial", f"r={r};beta={beta}", stats.nbinom(r, 1 / (1 + beta)))
    for n, p in BINOMIALS:
        yield from count_rows("binomial", f"n={n};p={p}", stats.binom(n, p))


def main():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for r in list(rows()) + list(counts()):
            dist, params, qty, arg, expected, abs_tol, rel_tol = r
            w.writerow([dist, params, qty, repr(float(arg)) if arg != "" else "", repr(float(expected)), abs_tol, rel_tol, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
