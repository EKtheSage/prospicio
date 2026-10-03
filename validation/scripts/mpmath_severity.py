"""Regenerate validation/reference/severity_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_severity.py

Limited expected values, stop-loss and layer means for the lognormal and
the gamma, by numerically integrating the survival function at 30
significant digits:

    LEV(d)          = integral of S(x) from 0 to d
    stop_loss(d)    = integral of S(x) from d to infinity
    layer(l xs a)   = integral of S(x) from a to a + l

This is independent of the closed forms the Rust code uses. (actuar's
levlnorm is the plan's other reference, but CRAN is not reachable from the
environment that generated this file.)
"""

import csv
import sys

import mpmath as mp
from scipy import stats

mp.mp.dps = 30
OUT = "validation/reference/severity_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} quad of survival at 30 digits"

LOGNORMALS = [(0.0, 1.0), (7.0, 0.5), (10.0, 2.0), (-1.0, 0.1)]
# Limits and retentions at these quantiles; 1 - 1e-9 tests tail accuracy.
PROBS = ["0.01", "0.5", "0.9", "0.99", "0.9999", "0.999999999"]


def survival(mu, sigma):
    return lambda x: mp.erfc((mp.log(x) - mu) / (sigma * mp.sqrt(2))) / 2


def quantile(mu, sigma, p):
    return mp.exp(mu + sigma * mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1))


def integral(f, a, b, median):
    points = sorted({a, b} | ({median} if a < median < b else set()))
    if b == mp.inf:
        points = [a, 2 * a, 10 * a, mp.inf] if a > 0 else [a, median, mp.inf]
    return mp.quad(f, points)


GAMMAS = [(0.3, 2.0), (2.5, 400.0), (250.0, 0.01)]


def gamma_rows():
    for shape, scale in GAMMAS:
        a, t = mp.mpf(shape), mp.mpf(scale)
        S = lambda x, a=a, t=t: mp.gammainc(a, x / t, mp.inf, regularized=True)
        median = a * t
        params = f"shape={shape};scale={scale}"
        for p in PROBS:
            # Evaluation points at quantiles; SciPy only picks the points.
            d = mp.mpf(float(stats.gamma(shape, scale=scale).isf(1 - float(p))))
            yield "gamma", params, "lev", d, "", integral(S, mp.mpf(0), d, median), 1e-12
            yield "gamma", params, "stop_loss", d, "", integral(S, d, mp.inf, median), 1e-11
            yield "gamma", params, "layer", d, d, integral(S, d, 2 * d, median), 1e-11


def rows():
    for mu, sigma in LOGNORMALS:
        S = survival(mp.mpf(mu), mp.mpf(sigma))
        median = mp.exp(mu)
        params = f"meanlog={mu};sdlog={sigma}"
        for p in PROBS:
            d = quantile(mu, sigma, p)
            # Round the limit to a double so Rust evaluates at the same point.
            d = mp.mpf(float(d))
            yield "lognormal", params, "lev", d, "", integral(S, mp.mpf(0), d, median), 1e-12
            yield "lognormal", params, "stop_loss", d, "", integral(S, d, mp.inf, median), 1e-11
            # Layer of width d attaching at d.
            yield "lognormal", params, "layer", d, d, integral(S, d, 2 * d, median), 1e-11


def main():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol", "rel_tol", "source"])
        for dist, params, qty, arg, arg2, expected, rel in list(rows()) + list(gamma_rows()):
            w.writerow([
                dist, params, qty, repr(float(arg)),
                repr(float(arg2)) if arg2 != "" else "",
                repr(float(expected)),
                0.0, rel, SOURCE,
            ])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
