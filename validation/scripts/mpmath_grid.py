"""Regenerate validation/reference/grid_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_grid.py

Grid masses for the three discretization methods, computed at 30 digits
from the lognormal cdf and limited expected value, written from the
textbook definitions (Klugman, Panjer & Willmot, "Loss Models", local
moment matching and rounding), independently of the Rust code.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/grid_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} at 30 digits"

MU, SIGMA = mp.mpf(7), mp.mpf("0.5")
CASES = [(100, 60), (250, 12)]


def cdf(x):
    return mp.mpf(0) if x <= 0 else mp.erfc(-(mp.log(x) - MU) / (SIGMA * mp.sqrt(2))) / 2


def lev(d):
    if d <= 0:
        return mp.mpf(d)
    phi = lambda z: mp.erfc(-z / mp.sqrt(2)) / 2
    z = (mp.log(d) - MU) / SIGMA
    return mp.exp(MU + SIGMA**2 / 2) * phi(z - SIGMA) + d * phi(-z)


def local_moment(h, n):
    L = [lev(j * h) for j in range(n)]
    f = [1 - L[1] / h]
    f += [(2 * L[j] - L[j - 1] - L[j + 1]) / h for j in range(1, n - 1)]
    f.append((L[n - 1] - L[n - 2]) / h)
    return f


def from_edges(edges):
    f, below = [], mp.mpf(0)
    for e in edges:
        f.append(cdf(e) - below)
        below = cdf(e)
    f.append(1 - below)
    return f


def rounding(h, n):
    return from_edges([(j + mp.mpf("0.5")) * h for j in range(n - 1)])


def lower(h, n):
    return from_edges([j * h for j in range(1, n)])


def main():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "method", "step", "points", "index", "expected", "abs_tol", "rel_tol", "source"])
        for h, n in CASES:
            for name, method in [("local_moment", local_moment), ("rounding", rounding), ("lower", lower)]:
                for j, p in enumerate(method(mp.mpf(h), n)):
                    w.writerow(["lognormal", f"meanlog={float(MU)};sdlog={float(SIGMA)}", name, h, n, j, repr(float(p)), 1e-12, 0.0, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
