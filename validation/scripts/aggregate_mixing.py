"""Regenerate validation/reference/mixing_aggregate.csv.

    python3 -m venv agg && agg/bin/pip install aggregate==1.2.0
    agg/bin/python validation/scripts/aggregate_mixing.py

The beta and Sichel mixed Poisson counts against Mildenhall's `aggregate`
1.2.0, as `validation/scripts/aggregate_counts.py` does for the gamma and
inverse Gaussian: each count is an aggregate with a unit severity
(`dsev [1]`), so its distribution is the count's, and the script records
P(N = k) for k = 0..59 on a grid of 1024 points with bucket 1. `params`
gives the count in prospicio's terms (`Mixing::Beta { cv, max }`,
`Mixing::Sichel { cv, shape }`, `Mixing::sichel_gamma(cv, f)`,
`Mixing::sichel_ig(cv, f)`). For the Sichel kinds the script also records
the generalized inverse Gaussian aggregate solved, `shape` (its `lam`),
`omega` (`mu / beta`) and `eta` (`mu`); aggregate's Broyden solve stops
at `f_rtol = 1e-11`, so those and the probabilities agree to about 1e-9.

aggregate's `sichel.gamma 0.4 0.3` is wrong: its Broyden solve stops,
with no warning, at a mixing with mean 1.476 and cv 0.071, so the count's
mean is 14.76, not 10. For that case the reference is SciPy's instead: the
shape and `omega` that give the mean 1, cv 0.4 and the Delaporte's
skewness `2 cv / (1 - f)`, solved by nested `brentq`, and the probabilities
`(lambda eta)^k / k! s^-(k + shape) K_{k + shape}(omega s) / K_shape(omega)`,
`s = sqrt(1 + 2 lambda eta / omega)`, by `scipy.special.kve`.

aggregate's beta pgf, `hyp1f1(a, a + b, r (z - 1))` at complex points, is
off by up to about 1e-8 on the first case, so its probabilities are kept
as `pmf_aggregate` at `2e-8`, and `pmf` is the probability by quadrature,
the integral of the Poisson(`max x`) probability against the Beta(a, b)
density (`scipy.integrate.quad`), at `1e-14`.
"""

import csv
import sys
import warnings

import math

import aggregate
import numpy as np
import scipy
from aggregate import build
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, kve
from scipy.stats import beta as beta_dist
from scipy.stats import poisson

OUT = "validation/reference/mixing_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
SCIPY = f"scipy {scipy.__version__}"


def gig_moments(shape, omega):
    eta = kve(shape, omega) / kve(shape + 1, omega)
    m2, m3 = (eta ** r * kve(shape + r, omega) / kve(shape, omega) for r in (2, 3))
    return eta, m2, m3


def gig_omega(shape, cv):
    return math.exp(brentq(lambda lw: gig_moments(shape, math.exp(lw))[1] - 1 - cv * cv,
                           math.log(1e-6), math.log(1e6), xtol=1e-15, rtol=1e-15))


def gig_skewness(shape, cv):
    _, m2, m3 = gig_moments(shape, gig_omega(shape, cv))
    return (m3 - 3 * m2 + 2) / (m2 - 1) ** 1.5


def sichel_pmf(lam, shape, omega, eta, kmax):
    s = math.sqrt(1 + 2 * lam * eta / omega)
    k = np.arange(kmax)
    return np.exp(k * np.log(lam * eta) - gammaln(k + 1) - (k + shape) * np.log(s)) * \
        kve(k + shape, omega * s) / kve(shape, omega) * np.exp(-omega * (s - 1))


def main():
    warnings.filterwarnings("ignore")
    cases = [
        ("beta", "lambda=10.0;cv=0.3;max=30.0", "mixed beta 0.3 30", "2e-8"),
        ("beta", "lambda=10.0;cv=0.5;max=20.0", "mixed beta 0.5 20", "2e-8"),
        ("sichel", "lambda=10.0;cv=0.5;shape=-0.5", "mixed sichel 0.5 -0.5", "1e-9"),
        ("sichel", "lambda=10.0;cv=0.4;shape=1.5", "mixed sichel 0.4 1.5", "1e-9"),
        ("sichel", "lambda=10.0;cv=0.6;shape=-2.0", "mixed sichel 0.6 -2.0", "1e-9"),
        ("sichel_gamma", "lambda=10.0;cv=0.4;f=0.3", "mixed sichel.gamma 0.4 0.3", "1e-9"),
        ("sichel_ig", "lambda=10.0;cv=0.5;f=0.3", "mixed sichel.ig 0.5 0.3", "1e-9"),
    ]
    rows = []
    for name, params, mixing, tol in cases:
        if name == "sichel_gamma":
            cv, f = 0.4, 0.3
            target = 2 * cv / (1 - f)
            shape = brentq(lambda l: gig_skewness(l, cv) - target, -5.0, 6.0,
                           xtol=1e-14, rtol=1e-14)
            omega = gig_omega(shape, cv)
            eta = gig_moments(shape, omega)[0]
            p = sichel_pmf(10.0, shape, omega, eta, 60)
            for k in range(60):
                rows.append([name, params, "pmf", str(k), "", repr(float(p[k])), "1e-12", "0.0",
                             SCIPY])
            for q, v in [("shape", shape), ("omega", omega), ("eta", eta)]:
                rows.append([name, params, q, "", "", repr(float(v)), "0.0", "1e-10", SCIPY])
            continue
        a = build(f"agg A 10 claims dsev [1] {mixing}", log2=10, bs=1)
        p = a.density_df.p_total
        quantity = "pmf_aggregate" if name == "beta" else "pmf"
        for k in range(60):
            rows.append([name, params, quantity, str(k), "", repr(float(p.iloc[k])), tol, "0.0",
                         SOURCE])
        if name == "beta":
            cv, r = (float(x.split("=")[1]) for x in params.split(";")[1:])
            c, n = cv * cv, 10.0
            al = (r - n * (1 + c)) / (c * r)
            be = al * (r - n) / n
            for k in range(60):
                q, _ = quad(lambda x: poisson.pmf(k, r * x) * beta_dist.pdf(x, al, be), 0, 1,
                            epsabs=1e-17, epsrel=1e-14, limit=500)
                rows.append([name, params, "pmf", str(k), "", repr(float(q)), "1e-14", "0.0",
                             SCIPY])
        if name.startswith("sichel"):
            f = a.frequency
            mu, beta, lam = f._mu, f._beta, f._lam
            for q, v in [("shape", lam), ("omega", mu / beta), ("eta", mu)]:
                rows.append([name, params, q, "", "", repr(float(v)), "0.0", "1e-8", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
