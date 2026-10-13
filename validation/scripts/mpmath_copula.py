"""Regenerate validation/reference/copula_mpmath.csv.

    uv run --with mpmath python validation/scripts/mpmath_copula.py

Copula cdfs at 30 significant digits:

- `bvn`: the standard bivariate normal cdf Phi_2(x, y; rho), by Plackett's
  identity Phi_2 = Phi(x) Phi(y) + int_0^rho phi_2(x, y; r) dr, the
  integral by mpmath's quadrature. Rust uses Genz's BVND (Gauss-Legendre
  on 6, 12 or 20 points by |rho|), accurate to about 1e-15 absolute.
- `gaussian`: the Gaussian copula C(u, v) = Phi_2(Phi^-1 u, Phi^-1 v; rho).
- `clayton`, `gumbel`, `frank`, `joe`: the Archimedean copulas'
  C(u, v) = psi(psi^-1(u) + psi^-1(v)) with prospicio's parameter theta
  (the generators of `ArchimedeanCopula`), and one three-dimensional point.
"""

import csv
from pathlib import Path

import mpmath as mp

mp.mp.dps = 30
SOURCE = f"mpmath {mp.__version__} at 30 digits"


def phi2(x, y, r):
    d = 1 - r * r
    return mp.exp(-(x * x - 2 * r * x * y + y * y) / (2 * d)) / (2 * mp.pi * mp.sqrt(d))


def bvn(x, y, rho):
    x, y, rho = mp.mpf(x), mp.mpf(y), mp.mpf(rho)
    return mp.ncdf(x) * mp.ncdf(y) + mp.quad(lambda r: phi2(x, y, r), [0, rho])


def inv(family, theta, u):
    u, th = mp.mpf(u), mp.mpf(theta)
    if family == "clayton":
        return u ** (-th) - 1
    if family == "gumbel":
        return (-mp.log(u)) ** th
    if family == "frank":
        return -mp.log(-mp.expm1(-th * u) / -mp.expm1(-th))
    if family == "joe":
        return -mp.log(1 - (1 - u) ** th)
    raise ValueError(family)


def psi(family, theta, t):
    th = mp.mpf(theta)
    if family == "clayton":
        return (1 + t) ** (-1 / th)
    if family == "gumbel":
        return mp.exp(-(t ** (1 / th)))
    if family == "frank":
        return -mp.log(1 + mp.exp(-t) * mp.expm1(-th)) / th
    if family == "joe":
        return 1 - (1 - mp.exp(-t)) ** (1 / th)
    raise ValueError(family)


rows = []
for rho in ["-0.95", "-0.5", "0", "0.3", "0.7", "0.9", "0.99"]:
    for x, y in [("-3", "-1"), ("-1", "0.5"), ("0", "0"), ("0.5", "2"), ("2", "2"), ("-2", "1.5")]:
        rows.append(("bvn", f"rho={rho}", "cdf", x, y, bvn(x, y, rho), "1e-14", "0"))

points = [("0.1", "0.2"), ("0.5", "0.5"), ("0.9", "0.95"), ("0.01", "0.99"), ("0.3", "0.8")]
for rho in ["0.5", "-0.3", "0.9"]:
    for u, v in points:
        x, y = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(u) - 1), mp.sqrt(2) * mp.erfinv(2 * mp.mpf(v) - 1)
        rows.append(("gaussian", f"rho={rho}", "cdf", u, v, bvn(x, y, rho), "1e-14", "0"))

for family, theta in [("clayton", "2"), ("gumbel", "1.5"), ("frank", "4"), ("joe", "2")]:
    for u, v in points:
        c = psi(family, theta, inv(family, theta, u) + inv(family, theta, v))
        rows.append((family, f"theta={theta}", "cdf", u, v, c, "0", "1e-13"))
    # A three-dimensional point, the third argument in `params`.
    c = psi(family, theta, inv(family, theta, "0.4") + inv(family, theta, "0.6") + inv(family, theta, "0.7"))
    rows.append((family, f"theta={theta};w=0.7", "cdf", "0.4", "0.6", c, "0", "1e-13"))

out = Path(__file__).resolve().parents[1] / "reference" / "copula_mpmath.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol", "rel_tol", "source"])
    for d, p, q, a, b, e, at, rt in rows:
        w.writerow([d, p, q, a, b, mp.nstr(e, 17, min_fixed=-30, max_fixed=30), at, rt, SOURCE])
print(f"wrote {len(rows)} rows to {out}")
