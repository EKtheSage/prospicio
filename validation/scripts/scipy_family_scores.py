"""Regenerate validation/reference/family_scores_scipy.csv.

    python validation/scripts/scipy_family_scores.py

Each family's predictive distribution, as `Family::draw` samples it, from
SciPy: the log density (log probability for counts), behind the log score,
and P(Y < y) and P(Y <= y), behind the randomized PIT. Mean mu, dispersion
phi and weight w:

- gaussian: norm(mu, sqrt(phi / w));
- poisson: (phi / w) N with N ~ poisson(mu w / phi), at N = y w / phi;
  with phi != 1 the log density is that of the continuous extension in y,
  (w / phi) lam^N e^-lam / (Gamma(N + 1) C(lam)), lam = mu w / phi,
  C(lam) = integral over x >= 0 of lam^x e^-lam / Gamma(x + 1) (by
  scipy.integrate.quad), at any y >= 0, on the lattice or off it;
- binomial: N / w with N ~ binom(w, mu), at N = y w;
- negative_binomial: nbinom(theta, theta / (theta + mu)) (w, phi unused);
- gamma: gamma(w / phi, scale = mu phi / w);
- inverse_gaussian: invgauss(mu / lam, scale = lam) with lam = w / phi.
"""

import csv
import sys

import math

import scipy
from scipy import integrate, special, stats

OUT = "validation/reference/family_scores_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"

# (family, mu, phi, w, theta, y)
CASES = [
    ("gaussian", 2.0, 1.5, 2.0, None, 3.1),
    ("gaussian", -1.0, 0.2, 1.0, None, -1.9),
    ("poisson", 2.0, 1.0, 1.0, None, 1.0),
    ("poisson", 0.3, 1.0, 1.0, None, 0.0),
    ("poisson", 4.0, 2.0, 1.0, None, 6.0),
    ("poisson", 1.5, 1.0, 3.0, None, 4.0 / 3.0),
    # Over-dispersed Poisson off its lattice (log density only): small,
    # moderate and large lam.
    ("poisson", 0.8, 2.0, 1.0, None, 1.1),
    ("poisson", 9.0, 1.5, 2.0, None, 8.3),
    ("poisson", 120.0, 2.5, 1.0, None, 117.3),
    ("binomial", 0.3, 1.0, 10.0, None, 0.4),
    ("binomial", 0.8, 1.0, 1.0, None, 1.0),
    ("negative_binomial", 3.0, 1.0, 1.0, 1.5, 5.0),
    ("negative_binomial", 0.5, 1.0, 1.0, 0.7, 0.0),
    ("gamma", 2.0, 0.4, 1.0, None, 1.3),
    ("gamma", 500.0, 1.2, 3.0, None, 900.0),
    ("inverse_gaussian", 2.0, 0.3, 1.0, None, 1.1),
    ("inverse_gaussian", 10.0, 0.05, 2.0, None, 14.0),
]


def dist(family, mu, phi, w, theta):
    if family == "gaussian":
        return stats.norm(mu, (phi / w) ** 0.5), 1.0
    if family == "poisson":
        return stats.poisson(mu * w / phi), w / phi
    if family == "binomial":
        return stats.binom(int(round(w)), mu), round(w)
    if family == "negative_binomial":
        return stats.nbinom(theta, theta / (theta + mu)), 1.0
    if family == "gamma":
        return stats.gamma(w / phi, scale=mu * phi / w), 1.0
    lam = w / phi
    return stats.invgauss(mu / lam, scale=lam), 1.0


def odp_log_density(y, mu, phi, w):
    lam = mu * w / phi
    n = y * w / phi
    c, _ = integrate.quad(
        lambda x: math.exp(x * math.log(lam) - lam - special.gammaln(x + 1.0)),
        0.0,
        lam + 40.0 * math.sqrt(lam) + 40.0,
        epsabs=0.0,
        epsrel=1e-13,
        limit=500,
        points=[lam],
    )
    return n * math.log(lam) - lam - special.gammaln(n + 1.0) - math.log(c) + math.log(w / phi)


def main():
    with open(OUT, "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["family", "params", "y", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for family, mu, phi, w, theta, y in CASES:
            d, scale = dist(family, mu, phi, w, theta)
            params = f"mu={mu};phi={phi};w={w}" + (f";theta={theta}" if theta else "")
            discrete = family in ("poisson", "binomial", "negative_binomial")
            on_lattice = not discrete or abs(y * scale - round(y * scale)) < 1e-9
            if discrete:
                k = round(y * scale)
                logp, lo, hi = d.logpmf(k), d.cdf(k - 1), d.cdf(k)
            else:
                logp, lo, hi = d.logpdf(y), d.cdf(y), d.cdf(y)
            if family == "poisson" and phi != 1.0:
                logp = odp_log_density(y, mu, phi, w)
            # The quadrature in C agrees to about 1e-12.
            tol = (1e-11, 1e-11) if family == "poisson" and phi != 1.0 else (1e-13, 1e-12)
            out.writerow([family, params, repr(y), "log_density", repr(float(logp)), *tol, SOURCE])
            if on_lattice:
                out.writerow([family, params, repr(y), "cdf_lower", repr(float(lo)), 1e-15, 1e-11, SOURCE])
                out.writerow([family, params, repr(y), "cdf_upper", repr(float(hi)), 1e-15, 1e-11, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
