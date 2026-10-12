"""Regenerate validation/reference/moments_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_moments.py

Compound moments to third order and the moment approximations against
Mildenhall's `aggregate` 1.0.1:

- the compound mean, cv and skewness, exactly at 30 digits by mpmath from
  the count's cumulants and the severity's raw moments, at `1e-12`. They
  are checked here against `aggregate`'s analytic statistics,
  `stats_df['mixed']`, to `1e-8`: `aggregate` computes the Weibull's
  moments numerically (6.5e-10 off the mean, 2e-9 off the skewness) and
  converts raw moments, which cancels at a Poisson mean of 1,000 (2.5e-10
  off the skewness);
- the five method-of-moments fits of `aggregate.approximate` (normal,
  gamma, lognormal, shifted gamma, shifted lognormal), computed by its own
  `_fits.approximate_from_mcvsk` on the exact moments, at `1e-10`.
  `Aggregate.approximate` itself fits the moments estimated from its grid
  (`stats_df['empirical']`), which differ from the analytic ones by the
  discretization, so the formulas are checked on the exact inputs.

`params` gives each book in prospicio's terms; scipy's parameters are
recorded as `shape`, `loc` and `scale` (lognormal: `shape` is `σ`, `scale`
is `e^μ`; the shifts are `loc`).
"""

import csv
import sys
import warnings

import aggregate
import mpmath as mp
from aggregate import build
from aggregate._fits import approximate_from_mcvsk

OUT = "validation/reference/moments_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__} fits on mpmath {mp.__version__} moments"

mp.mp.dps = 30


def severity_moments(params):
    """E[X], E[X^2], E[X^3] in closed form."""
    kv = dict(x.split("=") for x in params.split(";"))
    sev = kv["severity"]
    if sev == "weibull":
        k, t = mp.mpf(kv["shape"]), mp.mpf(kv["scale"])
        return [t ** j * mp.gamma(1 + j / k) for j in (1, 2, 3)]
    m, cv = mp.mpf(kv["mean"]), mp.mpf(kv["cv"])
    if sev == "gamma":
        a = 1 / cv ** 2
        t = m / a
        return [a * t, a * (a + 1) * t ** 2, a * (a + 1) * (a + 2) * t ** 3]
    s2 = mp.log(1 + cv ** 2)
    mu = mp.log(m) - s2 / 2
    return [mp.exp(j * mu + j * j * s2 / 2) for j in (1, 2, 3)]


def count_cumulants(params):
    kv = dict(x.split("=") for x in params.split(";"))
    lam = mp.mpf(kv["lambda"])
    if "mixing_cv" not in kv:
        return lam, lam, lam
    c2 = mp.mpf(kv["mixing_cv"]) ** 2
    r, b = 1 / c2, lam * c2
    return r * b, r * b * (1 + b), r * b * (1 + b) * (1 + 2 * b)


def exact(params):
    x1, x2, x3 = severity_moments(params)
    n1, n2, n3 = count_cumulants(params)
    vx = x2 - x1 ** 2
    k3x = x3 - 3 * x1 * vx - x1 ** 3
    mean = n1 * x1
    var = n1 * vx + n2 * x1 ** 2
    k3 = n1 * k3x + 3 * n2 * x1 * vx + n3 * x1 ** 3
    return mean, mp.sqrt(var) / mean, k3 / var ** mp.mpf(1.5)


BOOKS = [  # (name, DecL, params)
    ("B", "agg B 10 claims sev gamma 50 cv 0.7 poisson",
     "lambda=10;severity=gamma;mean=50;cv=0.7"),
    ("O", "agg O 2 claims sev lognorm 100 cv 2 poisson",
     "lambda=2;severity=lognormal;mean=100;cv=2"),
    ("W", "agg W 0.5 claims sev 1000 * weibull_min 0.7 poisson",
     "lambda=0.5;severity=weibull;shape=0.7;scale=1000"),
    ("D", "agg D 50 claims sev lognorm 1000 cv 3 mixed gamma 0.3",
     "lambda=50;mixing_cv=0.3;severity=lognormal;mean=1000;cv=3"),
    ("L", "agg L 1000 claims sev lognorm 10 cv 0.5 poisson",
     "lambda=1000;severity=lognormal;mean=10;cv=0.5"),
]

FITS = ["norm", "gamma", "lognorm", "sgamma", "slognorm"]


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for name, decl, params in BOOKS:
        a = build(decl, bs=1, log2=10)
        s = a.stats_df["mixed"]
        theirs = [float(s[("agg", k)]) for k in ("mean", "cv", "skew")]
        m, cv, skew = (float(v) for v in exact(params))
        for t, e in zip(theirs, (m, cv, skew)):
            assert abs(t / e - 1) < 1e-8, (name, t, e)
        rows += [(name, params, "mean", "", m, 1e-12), (name, params, "cv", "", cv, 1e-12),
                 (name, params, "skew", "", skew, 1e-12)]
        for fit in FITS:
            rv = approximate_from_mcvsk(m, cv, skew, "x", "agg x 1 claim ", "", fit, "scipy")
            shape = rv.args[0] if rv.args else None
            for key, val in [("shape", shape), ("loc", rv.kwds.get("loc", 0.0)),
                             ("scale", rv.kwds.get("scale", 1.0))]:
                if val is not None:
                    rows.append((name, params, f"{fit}_{key}", "", float(val), 1e-10))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for name, params, qty, arg, val, tol in rows:
            # A zero location is exact; give it an absolute tolerance.
            abs_tol = 1e-9 if val == 0.0 else 0.0
            w.writerow([name, params, qty, arg, repr(val), abs_tol, tol, SOURCE])
    print(f"wrote {OUT}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
