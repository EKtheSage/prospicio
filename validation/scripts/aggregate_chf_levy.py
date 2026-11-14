# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aggregate==2.1.0",
#     "numpy",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-10T00:00:00Z"
# ///
"""Regenerate validation/reference/chf_levy_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_chf_levy.py

Laws of Levy processes computed by inverting their characteristic function
against Mildenhall's `aggregate` 2.1.0 (`AggregateCHF`, the `chfagg`
families `norminvgauss`, `variancegamma` and `temperedstable`; #780, first
PR): the normal inverse Gaussian, the variance gamma (the Laplace law
among them, and a variance rate whose slow decay walks the bucket down)
and the CGMY tempered stable at four indices, each on the grid rule A
chooses. The quantities are those of `aggregate_chf.py`. `params` gives
the law in prospicio's terms.
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_levy_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (law, aggregate family, aggregate keywords, chf_scale, chf_loc, log2, bs,
# prospicio's parameters). An affine law is `scale * base + shift`, the
# base's parameters prefixed `base_`.
CASES = [
    ("normal_inverse_gaussian", "norminvgauss", dict(alpha=2, beta=0.5, delta=1.5, mu=0.3), 1, 0,
     12, 0, dict(alpha=2, beta=0.5, delta=1.5, mu=0.3)),
    ("normal_inverse_gaussian", "norminvgauss", dict(alpha=1, beta=-0.8, delta=10, mu=100), 1, 0,
     14, 0, dict(alpha=1, beta=-0.8, delta=10, mu=100)),
    ("variance_gamma", "variancegamma", dict(sigma=0.2, nu=0.5, theta=-0.1, mu=0.05), 1, 0,
     12, 0, dict(sigma=0.2, nu=0.5, theta=-0.1, mu=0.05)),
    ("variance_gamma", "variancegamma", dict(sigma=1, nu=1, theta=0, mu=0), 1, 0,
     12, 0, dict(sigma=1, nu=1, theta=0, mu=0)),
    ("variance_gamma", "variancegamma", dict(sigma=0.3, nu=4, theta=0.2, mu=0), 1, 0,
     12, 0, dict(sigma=0.3, nu=4, theta=0.2, mu=0)),
    ("tempered_stable", "temperedstable", dict(C=1, G=5, M=10, Y=0.5, mu=0), 1, 0,
     12, 0, dict(C=1, G=5, M=10, Y=0.5, mu=0)),
    ("tempered_stable", "temperedstable", dict(C=0.5, G=2, M=8, Y=1.5, mu=1), 1, 0,
     12, 0, dict(C=0.5, G=2, M=8, Y=1.5, mu=1)),
    ("tempered_stable", "temperedstable", dict(C=2, G=3, M=4, Y=-0.5, mu=0), 1, 0,
     12, 0, dict(C=2, G=3, M=4, Y=-0.5, mu=0)),
    ("tempered_stable", "temperedstable", dict(C=0.1, G=1, M=1, Y=1.8, mu=0), 1, 0,
     14, 0, dict(C=0.1, G=1, M=1, Y=1.8, mu=0)),
]


LEVELS = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]

# (abs_tol, rel_tol) by quantity. The grid is exact; the masses differ by
# FFT round-off; the Chernoff bounds by where Brent stops on a flat minimum.
TOLERANCE = {
    "step": (0.0, 0.0),
    "x_min": (1e-12, 0.0),
    "truncation_error": (1e-20, 1e-9),
    "outside_below": (1e-20, 1e-6),
    "outside_above": (1e-20, 1e-6),
    "sum": (1e-13, 0.0),
    "mean": (1e-12, 1e-12),
    "sd": (0.0, 1e-10),
    "quantile": (1e-9, 0.0),
    "mass": (1e-13, 0.0),
}


def main():
    rows = []
    for law, family, kw, c, b, log2, bs, params in CASES:
        a = AggregateCHF(law, family, chf_scale=c, chf_loc=b, **kw)
        a.update(log2=log2, bs=bs)
        p = np.asarray(a.agg_density, dtype=float)
        n = len(p)
        x = a.x_min + a.bs * np.arange(n)
        mean = float((x * p).sum() / p.sum())
        sd = float(np.sqrt(((x - mean) ** 2 * p).sum() / p.sum()))
        text = ";".join(f"{k}={v}" for k, v in {**params, "log2": log2, "bs": bs}.items())

        def put(quantity, arg, value):
            abs_tol, rel_tol = TOLERANCE[quantity]
            rows.append(dict(distribution=law, params=text, quantity=quantity, arg=arg,
                             arg2="", expected=repr(float(value)), abs_tol=abs_tol,
                             rel_tol=rel_tol, source=SOURCE))

        put("step", "", a.bs)
        put("x_min", "", a.x_min)
        put("truncation_error", "", a.truncation_error)
        put("outside_below", "", a.outside_below)
        put("outside_above", "", a.outside_above)
        put("sum", "", p.sum())
        put("mean", "", mean)
        put("sd", "", sd)
        cum = np.cumsum(p)
        for level in LEVELS:
            k = int(np.searchsorted(cum, level))
            put("quantile", level, x[min(k, n - 1)])
        idx = sorted(set(np.linspace(0, n - 1, 257).astype(int).tolist()
                         + [int(np.argmax(p))]))
        for k in idx:
            put("mass", k, p[k])

    fields = ["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol",
              "rel_tol", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
