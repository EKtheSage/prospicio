"""Regenerate validation/reference/tweedie_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_tweedie.py

Tweedie distributions with power 1 < p < 2 at 40 significant digits,
written from the compound Poisson-gamma definition, independently of the
Rust code:

    N ~ Poisson(lam),  lam = mu^(2-p) / (phi (2-p))
    X ~ Gamma(alpha, theta),  alpha = (2-p)/(p-1),  theta = phi (p-1) mu^(p-1)

- `ln_pdf`: log of sum_n Pois(n) * gamma density of shape n*alpha, with
  terms summed until they no longer change the sum at 40 digits;
- `cdf`, `survival`: e^-lam + sum_n Pois(n) P(n alpha, y/theta), and
  sum_n Pois(n) Q(n alpha, y/theta);
- `lev`: sum_n Pois(n) LEV of Gamma(n alpha, theta) at d, each from its
  regularized incomplete gammas.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 40
OUT = "validation/reference/tweedie_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} series at 40 digits"

# (mu, phi, p): mid-range, small lambda, power near 1 and near 2, and a
# large lambda (about 500 claims).
CASES = [
    (500.0, 40.0, 1.6),
    (10.0, 2.0, 1.4),
    (1.0, 0.5, 1.1),
    (1000.0, 5.0, 1.9),
    (100.0, 0.1, 1.2),
]
# Points as multiples of the mean, from deep in the body to the far tail.
MULTIPLES = ["0.001", "0.1", "0.5", "1", "2", "5", "20"]


def gammainc(a, lo, hi):
    """Regularized incomplete gamma over [lo, hi], lo = 0 or hi = inf. Where
    mpmath cannot converge on one part (it happens for huge shapes far from
    x = a, where that part is within 2^-136 of 0 or 1), take the complement
    of the other."""
    try:
        return mp.gammainc(a, lo, hi, regularized=True)
    except ValueError:
        if lo == 0:
            return 1 - mp.gammainc(a, hi, mp.inf, regularized=True)
        return 1 - mp.gammainc(a, 0, lo, regularized=True)


def params(mu, phi, p):
    mu, phi, p = mp.mpf(mu), mp.mpf(phi), mp.mpf(p)
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    theta = phi * (p - 1) * mu ** (p - 1)
    return lam, alpha, theta


def series(lam, f):
    """sum_{n>=1} Pois(n; lam) f(n), past the Poisson mode until terms vanish."""
    total = mp.mpf(0)
    n = 1
    small = 0
    while True:
        w = mp.exp(n * mp.log(lam) - lam - mp.loggamma(n + 1))
        term = w * f(n)
        total += term
        if n > lam and (term == 0 or abs(term) < total * mp.mpf(10) ** -45):
            small += 1
            if small > 5:
                return total
        else:
            small = 0
        n += 1


def rows():
    for mu, phi, p in CASES:
        lam, alpha, theta = params(mu, phi, p)
        label = f"mu={mu};phi={phi};p={p}"
        surv = lambda y: series(lam, lambda n: gammainc(n * alpha, y / theta, mp.inf))
        for m in MULTIPLES:
            y = mp.mpf(float(mp.mpf(m) * mu))  # a double, so Rust evaluates the same point
            dens = series(
                lam,
                lambda n: mp.exp((n * alpha - 1) * mp.log(y) - y / theta - mp.loggamma(n * alpha) - n * alpha * mp.log(theta)),
            )
            yield label, "ln_pdf", y, mp.log(dens), 1e-12
            s = surv(y)
            yield label, "survival", y, s, 1e-11
            # Summed directly: 1 - s would cancel where the cdf is tiny.
            c = mp.exp(-lam) + series(lam, lambda n: gammainc(n * alpha, 0, y / theta))
            yield label, "cdf", y, c, 1e-12
            lev = series(
                lam,
                lambda n: n * alpha * theta * gammainc(n * alpha + 1, 0, y / theta)
                + y * gammainc(n * alpha, y / theta, mp.inf),
            )
            yield label, "lev", y, lev, 1e-11
        yield label, "cdf", mp.mpf(0), mp.exp(-lam), 1e-13


def main():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for label, qty, arg, value, rel in rows():
            w.writerow(["tweedie", label, qty, repr(float(arg)), mp.nstr(value, 17), "1e-300", rel, SOURCE])
            f.flush()
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
