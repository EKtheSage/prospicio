"""Regenerate validation/reference/distortion_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_distortion.py

Distortion risk measures rho(X) = integral of g(S(x)) dx, at 30 significant
digits:

- `discrete`: a fixed discrete distribution (DISCRETE below, repeated in
  validation/tests/distributions.rs) with a 1e-10 atom in the tail, so
  the sum is exact and the tolerance tests rounding alone.
- `lognormal`: the continuous integral for a lognormal. Rust evaluates it
  on a local-moment grid with the step and point count in `params`, which
  differs from the integral in two ways: the grid's discretization, held
  under a 2e-4 relative error, and the tail above the last grid point,
  whose integral this script computes and adds to the tolerance. So
  `rel_tol = 2e-4 + (integral of g(S) above the grid) / rho`.

The distortions are written from their definitions, independently of the
Rust code: TVaR min(s / (1 - p), 1), Wang Phi(Phi^-1(s) + lambda),
proportional hazard s^rho, dual power 1 - (1 - s)^beta.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/distortion_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} at 30 digits"

DISCRETE = ([0, 1, 2, 5, 10, 100], ["0.5", "0.2", "0.15", "0.1", "0.0499999999", "1e-10"])
DISTORTIONS = [
    ("tvar", "0.9"),
    ("tvar", "0.99"),
    ("wang", "0.25"),
    ("wang", "1.0"),
    ("proportional_hazard", "0.5"),
    ("proportional_hazard", "0.9"),
    ("dual_power", "2.0"),
    ("dual_power", "5.0"),
]
# (meanlog, sdlog, step, points).
LOGNORMALS = [(0.0, 0.5, 0.0005, 40_000), (0.0, 1.0, 0.002, 100_000)]


def phi(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def phi_inv(p):
    return mp.sqrt(2) * mp.erfinv(2 * p - 1)


def g(kind, a):
    a = mp.mpf(a)
    if kind == "tvar":
        return lambda s: min(s / (1 - a), mp.mpf(1))
    if kind == "wang":
        return lambda s: mp.mpf(0) if s <= 0 else mp.mpf(1) if s >= 1 else phi(phi_inv(s) + a)
    if kind == "proportional_hazard":
        return lambda s: s**a
    if kind == "dual_power":
        return lambda s: 1 - (1 - s) ** a
    raise ValueError(kind)


def discrete(kind, a):
    gg = g(kind, a)
    values, probs = DISCRETE
    probs = [mp.mpf(p) for p in probs]
    total = mp.mpf(0)
    for k, x in enumerate(values):
        at_or_above = sum(probs[k:])
        above = sum(probs[k + 1 :])
        total += x * (gg(at_or_above) - gg(above))
    return total


def lognormal(kind, a, mu, sigma, top):
    """The integral of g(S(x)) over (0, inf), and over (top, inf)."""
    gg = g(kind, a)
    survival = lambda x: mp.erfc((mp.log(x) - mu) / (sigma * mp.sqrt(2))) / 2
    f = lambda x: gg(survival(x))
    median = mp.exp(mu)
    points = {mp.mpf(0), median / 4, median, 4 * median, 50 * median, top}
    if kind == "tvar":
        # The kink at the VaR.
        points.add(mp.exp(mu + sigma * phi_inv(mp.mpf(a))))
    points = sorted(x for x in points if x <= top)
    body = mp.quad(f, points)
    tail = mp.quad(f, [top, 10 * top, 1000 * top, mp.inf])
    return body + tail, tail


def rows():
    for kind, a in DISTORTIONS:
        yield "discrete", "", kind, a, discrete(kind, a), 1e-14
    for mu, sigma, step, n in LOGNORMALS:
        params = f"meanlog={mu};sdlog={sigma};step={step};points={n}"
        top = mp.mpf(step) * (n - 1)
        for kind, a in DISTORTIONS:
            value, tail = lognormal(kind, a, mu, sigma, top)
            yield "lognormal", params, kind, a, value, float(mp.mpf("2e-4") + tail / value)


def main():
    w = csv.writer(sys.stdout if "--stdout" in sys.argv else open(OUT, "w", newline=""))
    w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol", "rel_tol", "source"])
    for dist, params, kind, a, value, rel in rows():
        rel = float(mp.nstr(rel, 3))
        w.writerow([dist, params, kind, a, "", mp.nstr(value, 17, strip_zeros=False), 0.0, rel, SOURCE])


if __name__ == "__main__":
    main()
