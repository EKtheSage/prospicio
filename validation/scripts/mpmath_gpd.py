"""Regenerate validation/reference/gpd_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_gpd.py

Maximum likelihood fits of the generalized Pareto distribution at 30
digits. Each data set is the GPD(xi0, beta0) quantiles at (i - 0.5) / n,
i = 1..n, which the Rust test recomputes in double precision; the
estimates are not (xi0, beta0) but the exact MLE for those values.

The MLE solves the profile likelihood in theta = xi / beta: for fixed theta,
xi(theta) = mean(log(1 + theta y)) and beta = xi / theta, so the estimate is
the root of d/dtheta [-log(xi(theta) / theta) - xi(theta)] (differentiated
analytically), found by mpmath.findroot (Illinois method) from a bracket
around the generating parameters.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/gpd_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} profile likelihood at 30 digits"

CASES = [(0.4, 1.0, 200), (-0.2, 2.0, 150), (0.05, 3.0, 400), (1.1, 0.5, 100)]


def data(xi, beta, n):
    xi, beta = mp.mpf(xi), mp.mpf(beta)
    # Double-precision values, as Rust computes them: beta * expm1(xi * -log1p(-p)) / xi.
    out = []
    for i in range(1, n + 1):
        p = (i - 0.5) / n
        e = -mp.log1p(-mp.mpf(p))
        out.append(mp.mpf(float(beta * mp.expm1(xi * e) / xi)))
    return out


def fit(y):
    n = len(y)

    def xi_of(theta):
        return mp.fsum(mp.log1p(theta * v) for v in y) / n

    def score(theta):
        # d/dtheta of -log(xi / theta) - xi, with xi' = mean(y / (1 + theta y)).
        xi = xi_of(theta)
        dxi = mp.fsum(v / (1 + theta * v) for v in y) / n
        return -dxi / xi + 1 / theta - dxi

    return xi_of, score


def bracketed_root(f, start, ymax):
    """Root of f by the Illinois method, from a bracket found by stepping
    out from start (staying inside theta > -1 / max(y))."""
    lo = -1 / ymax
    a, b = start * mp.mpf("0.9"), start * mp.mpf("1.1")
    a, b = min(a, b), max(a, b)
    while f(a) * f(b) > 0:
        a = max(a - (b - a), lo + (a - lo) / 2)
        b = b + (b - a)
    return mp.findroot(f, (a, b), solver="illinois", tol=mp.mpf(10) ** -28)


def rows():
    for xi0, beta0, n in CASES:
        y = data(xi0, beta0, n)
        xi_of, score = fit(y)
        theta = bracketed_root(score, mp.mpf(xi0) / beta0, max(y))
        xi = xi_of(theta)
        beta = xi / theta
        params = f"xi0={xi0};beta0={beta0};n={n}"
        yield params, "xi", xi
        yield params, "beta", beta


def main():
    w = csv.writer(sys.stdout if "--stdout" in sys.argv else open(OUT, "w", newline=""), lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
    for params, qty, value in rows():
        w.writerow(["gpd_fit", params, qty, "", mp.nstr(value, 17, strip_zeros=False), 1e-9, 1e-8, SOURCE])


if __name__ == "__main__":
    main()
