# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-12T00:00:00Z"
# ///
"""Regenerate the KS p-value and tail-moment parity reference from SciPy (#805).

Run from the repository root, after `scipy_severity_fits.py`:

    python validation/scripts/scipy_fit_tails.py

For each family's maximum likelihood fit (`scipy_severity_fits.py`) to the
`plain` losses (89 above t = 150,000), writes to `validation/reference/
fit_tails_scipy.csv`: the Kolmogorov-Smirnov distance and p-value of
`scipy.stats.kstest` against the fit's distribution conditional on t
(`method="exact"`, `kstwo`), and the mean and CV of `X | X > t`, integrated on the survival
scale (checked against the lognormal's closed form), "inf" where the
moment diverges.
"""

import csv
import importlib.util
import math
import pathlib
import warnings

import numpy as np
from scipy import integrate, stats

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("fits", HERE / "scipy_severity_fits.py")
fits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fits)

OUT = "validation/reference/fit_tails_scipy.csv"
T = fits.T


def divergent(family, p):
    """The order of the first moment that diverges, or None."""
    if family == "pareto":
        a = p[0]
    elif family == "gen_pareto":
        a = p[1]
    elif family == "loglogistic":
        a = p[0]
    elif family == "lomax":
        a = p[0]
    else:
        return None
    return a


def tail_expectation(d, g):
    """E[g(X) | X > T] over the survival scale, q = S(x) = e^(-v):
    (1/S(T)) * integral from -ln S(T) to infinity of g(isf(e^-v)) e^-v dv.
    SciPy's own `expect(..., lb=T, conditional=True)` does not converge on
    these heavy tails (it warns and returns 20% too little for the
    lognormal), so it is not used."""
    v0 = -d.logsf(T)

    def f(v):
        return g(d.isf(math.exp(-v))) * math.exp(-v)

    # Beyond v = 700, e^-v underflows; what is left of a finite moment
    # there is below double precision.
    edges = [v0 + k for k in range(0, 40)] + [v0 + 40 + 20 * k for k in range(0, 33)]
    total = sum(integrate.quad(f, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                for lo, hi in zip(edges, edges[1:]))
    return total / math.exp(-v0)


def lognormal_check():
    """The closed form E[X | X > t] of the lognormal against the
    integral above."""
    mu, s = 11.0, 1.2
    d = stats.lognorm(s=s, scale=math.exp(mu))
    z = (math.log(T) - mu) / s
    exact = math.exp(mu + s * s / 2) * stats.norm.sf(z - s) / stats.norm.sf(z)
    got = tail_expectation(d, lambda x: x)
    assert abs(got / exact - 1) < 1e-9, (got, exact)


def main():
    lognormal_check()
    warnings.simplefilter("ignore")
    y, r, c = fits.datasets()["plain"]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "quantity", "expected", "abs_tol", "rel_tol", "source"])
        for family, (names, make) in fits.FAMILIES.items():
            _, p, _ = fits.fit(family, y, r, c)
            d = make(p)
            st = d.sf(T)
            boundary = min(p[1:] if family == "lognormal" else p) < 1e-6
            ks = stats.kstest(y, lambda v: 1.0 - d.sf(np.maximum(v, T)) / st, method="exact")
            w.writerow([family, "ks_distance", repr(float(ks.statistic)), 1e-4 if boundary else 1e-7, 0.0, fits.SOURCE])
            w.writerow([family, "ks_p_value", repr(float(ks.pvalue)), 1e-3 if boundary else 1e-6, 0.0, fits.SOURCE])
            a = divergent(family, p)
            if a is not None and a <= 1:
                mean, cv = math.inf, math.inf
            else:
                m1 = tail_expectation(d, lambda x: x)
                if a is not None and a <= 2:
                    cv = math.inf
                else:
                    m2 = tail_expectation(d, lambda x: (x - m1) ** 2)
                    cv = math.sqrt(m2) / m1
                mean = m1
            rel = 1e-4 if boundary else 1e-6
            w.writerow([family, "tail_mean", repr(float(mean)), 0.0, rel, fits.SOURCE])
            w.writerow([family, "tail_cv", repr(float(cv)), 0.0, rel, fits.SOURCE])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
