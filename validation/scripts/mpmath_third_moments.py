"""Regenerate validation/reference/third_moments_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_third_moments.py

Third raw moments `E[X^3]` of the severity families and claim counts, at
30 significant digits and independent of the closed forms in Rust:

- severities: `3 ∫ x² S(x) dx` over `(0, ∞)` by `mpmath.quad` on the
  survival function;
- counts: `Σ k³ P(N = k)` summed exactly over `0..K`, with `K` far enough
  past the mean that the rest is below the working precision;
- the Tweedie: the compound Poisson–gamma's third moment from its count
  and severity, `E[N] m₃ + 3 E[N(N−1)] m₁ m₂ + E[N(N−1)(N−2)] m₁³`.

Divergent moments are checked in the Rust unit tests, not here.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/third_moments_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} at 30 digits"


def third(survival, lo=0, breaks=()):
    pts = [mp.mpf(lo), *[mp.mpf(b) for b in breaks], mp.inf]
    return 3 * mp.quad(lambda x: x * x * survival(x), pts)


def lognormal(mu, s):
    return lambda x: mp.erfc((mp.log(x) - mu) / (s * mp.sqrt(2))) / 2


def gamma(a, t):
    return lambda x: mp.gammainc(a, x / t, mp.inf, regularized=True)


def weibull(k, t):
    return lambda x: mp.exp(-(x / t) ** k)


def loglogistic(b, t):
    return lambda x: 1 / (1 + (x / t) ** b)


def inverse_gamma(a, t):
    return lambda x: mp.gammainc(a, 0, t / x, regularized=True)


def burr(a, g, t):
    return lambda x: (1 + (x / t) ** g) ** (-a)


def beta(a, b, scale):
    return lambda x: mp.betainc(a, b, x / scale, 1, regularized=True) if x < scale else mp.mpf(0)


def inverse_gaussian(m, lam):
    def s(x):
        r = mp.sqrt(lam / x)
        return mp.ncdf(-r * (x / m - 1)) - mp.exp(2 * lam / m) * mp.ncdf(-r * (x / m + 1))
    return s


def pareto(t, a, trunc=None):
    if trunc is None:
        return lambda x: mp.mpf(1) if x < t else (t / x) ** a
    mass = 1 - (t / trunc) ** a

    def s(x):
        if x < t:
            return mp.mpf(1)
        if x >= trunc:
            return mp.mpf(0)
        return ((t / x) ** a - (t / trunc) ** a) / mass
    return s


def gpd(xi, b):
    return lambda x: (1 + xi * x / b) ** (-1 / xi)


SEVERITIES = [  # (name, params, survival, breaks)
    ("lognormal", "meanlog=4;sdlog=1", lognormal(4, 1), (100,)),
    ("lognormal", "meanlog=3.8;sdlog=1.2686362411795196", lognormal(mp.mpf("3.8"), mp.mpf("1.2686362411795196")), (100,)),
    ("gamma", "shape=2.0408163265306123;scale=24.5", gamma(mp.mpf("2.0408163265306123"), mp.mpf("24.5")), (50,)),
    ("gamma", "shape=0.5;scale=10", gamma(mp.mpf("0.5"), 10), (10,)),
    ("weibull", "shape=0.7;scale=1000", weibull(mp.mpf("0.7"), 1000), (1000,)),
    ("weibull", "shape=2;scale=5", weibull(2, 5), (5,)),
    ("loglogistic", "shape=5;scale=100", loglogistic(5, 100), (100,)),
    ("inverse_gamma", "shape=5;scale=400", inverse_gamma(5, 400), (100,)),
    ("burr", "alpha=3;gamma=2;scale=100", burr(3, 2, 100), (100,)),
    ("beta", "a=2;b=5;scale=1000", beta(2, 5, 1000), (1000,)),
    ("inverse_gaussian", "mean=100;shape=50", inverse_gaussian(100, 50), (100,)),
    ("pareto", "t=100;alpha=4.5", pareto(100, mp.mpf("4.5")), (100,)),
    ("pareto", "t=100;alpha=2;truncation=10000", pareto(100, 2, 10000), (100, 10000)),
    ("pareto", "t=100;alpha=3;truncation=5000", pareto(100, 3, 5000), (100, 5000)),
    ("gpd", "xi=0.2;beta=50", gpd(mp.mpf("0.2"), 50), (50,)),
    ("gpd", "xi=-0.3;beta=50", lambda x: gpd(mp.mpf("-0.3"), 50)(x) if x < mp.mpf(50) / mp.mpf("0.3") else mp.mpf(0), (100, mp.mpf(50) / mp.mpf("0.3"))),
]


def poisson(l):
    return lambda k: mp.exp(-l) * l ** k / mp.factorial(k)


def negbin(r, b):
    return lambda k: mp.binomial(k + r - 1, k) * (1 / (1 + b)) ** r * (b / (1 + b)) ** k


def binomial(n, p):
    return lambda k: mp.binomial(n, k) * p ** k * (1 - p) ** (n - k) if k <= n else mp.mpf(0)


COUNTS = [  # (name, params, pmf, K)
    ("poisson", "lambda=2", poisson(2), 200),
    ("poisson", "lambda=1000", poisson(1000), 3000),
    ("negative_binomial", "r=1.5;beta=2", negbin(mp.mpf("1.5"), 2), 600),
    ("negative_binomial", "r=11.111111111111111;beta=0.9", negbin(mp.mpf("11.111111111111111"), mp.mpf("0.9")), 600),
    ("binomial", "n=20;p=0.3", binomial(20, mp.mpf("0.3")), 20),
]


def tweedie(mu, phi, p):
    lam = mu ** (2 - p) / (phi * (2 - p))
    a = (2 - p) / (p - 1)
    t = phi * (p - 1) * mu ** (p - 1)
    m1, m2, m3 = a * t, a * (a + 1) * t ** 2, a * (a + 1) * (a + 2) * t ** 3
    return lam * m3 + 3 * lam ** 2 * m1 * m2 + lam ** 3 * m1 ** 3


def main():
    rows = []
    for name, params, s, breaks in SEVERITIES:
        rows.append((name, params, third(s, 0, breaks)))
    for name, params, pmf, top in COUNTS:
        rows.append((name, params, mp.fsum(mp.mpf(k) ** 3 * pmf(k) for k in range(top + 1))))
    for mu, phi, p in [("100", "2", "1.5"), ("10", "0.5", "1.8")]:
        rows.append(("tweedie", f"mean={mu};dispersion={phi};power={p}",
                     tweedie(mp.mpf(mu), mp.mpf(phi), mp.mpf(p))))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for name, params, val in rows:
            w.writerow([name, params, "third_moment", "", mp.nstr(val, 20), 0.0, 1e-10, SOURCE])
    print(f"wrote {OUT}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
