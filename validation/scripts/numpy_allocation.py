"""Regenerate validation/reference/allocation_numpy.csv.

    pip install numpy
    python validation/scripts/numpy_allocation.py

Capital allocation of a distortion risk measure for a fixed joint sample of
four components over 400 equally likely simulations. The draws are integers
from the formulas in `draws()` (repeated in validation/tests/distributions.rs),
so both sides see identical inputs, and many totals tie, which exercises tie
handling.

Everything is written from definitions, independently of the Rust code:

- rho(X) for a discrete X = sum over distinct values x of
  x * (g(P(X >= x)) - g(P(X > x))).
- Euler: each distinct total s carries the weight g(P(S >= s)) - g(P(S > s)),
  shared equally by the simulations at s; component j gets the weighted sum
  of its draws.
- Covariance: rho(S) Cov(X_j, S) / Var(S), population moments.
- Proportional: rho(S) rho(X_j) / sum_k rho(X_k).
- Marginal: rho(S) - rho(S - X_j).
- Shapley: the average over all 24 orders of joining of each component's
  increase in rho (not the subset formula the Rust code uses).
"""

import csv
import itertools
import math

import numpy as np

OUT = "validation/reference/allocation_numpy.csv"
N, M = 400, 4


def draws():
    x = np.zeros((N, M))
    for i in range(N):
        a = (i * 37) % 101
        x[i, 0] = a
        x[i, 1] = (i * 53) % 97 + a // 2
        x[i, 2] = (i * i) % 89
        x[i, 3] = 3 * ((i * 29) % 41) + (a // 10) ** 2
    return x


def distortions():
    return {
        ("tvar", 0.9): lambda s: min(s / 0.1, 1.0),
        ("tvar", 0.75): lambda s: min(s / 0.25, 1.0),
        ("proportional_hazard", 0.5): lambda s: s**0.5,
        ("dual_power", 3.0): lambda s: 1.0 - (1.0 - s) ** 3,
    }


def rho(g, values):
    values = np.asarray(values)
    total = 0.0
    for v in np.unique(values):
        at_least = np.mean(values >= v)
        above = np.mean(values > v)
        total += v * (g(at_least) - g(above))
    return total


def euler(g, x):
    s = x.sum(axis=1)
    out = np.zeros(M)
    for v in np.unique(s):
        w = g(np.mean(s >= v)) - g(np.mean(s > v))
        at = s == v
        out += w * x[at].mean(axis=0)
    return out


def shapley(g, x):
    out = np.zeros(M)
    for order in itertools.permutations(range(M)):
        before = 0.0
        acc = np.zeros(N)
        for j in order:
            acc = acc + x[:, j]
            after = rho(g, acc)
            out[j] += after - before
            before = after
    return out / math.factorial(M)


def main():
    x = draws()
    s = x.sum(axis=1)
    rows = []
    for (name, arg), g in distortions().items():
        total = rho(g, s)
        standalone = np.array([rho(g, x[:, j]) for j in range(M)])
        cov = np.array([np.mean((x[:, j] - x[:, j].mean()) * (s - s.mean())) for j in range(M)])
        methods = {
            "standalone": standalone,
            "euler": euler(g, x),
            "covariance": total * cov / np.var(s),
            "proportional": total * standalone / standalone.sum(),
            "marginal": np.array([total - rho(g, s - x[:, j]) for j in range(M)]),
            "shapley": shapley(g, x),
        }
        rows.append([name, arg, "total", "", repr(float(total))])
        for method, values in methods.items():
            for j, v in enumerate(values):
                rows.append([name, arg, method, j, repr(float(v))])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distortion", "arg", "method", "component", "expected", "abs_tol", "rel_tol", "source"])
        for r in rows:
            w.writerow(r + ["1e-9", "1e-12", f"numpy {np.__version__}, from definitions"])


if __name__ == "__main__":
    main()
