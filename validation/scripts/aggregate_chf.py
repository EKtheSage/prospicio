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
"""Regenerate validation/reference/chf_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_chf.py

Distributions computed by inverting their characteristic function against
Mildenhall's `aggregate` 2.1.0 (`aggregate.characteristic_function.AggregateCHF`,
the `chfagg` statement), untilted (`#779`, first PR). Each case is a law
with a closed-form characteristic function on the grid `aggregate`'s rule A
chooses (`bs = 0`: the window from the Chernoff edges at mass
`1 - bucket_sizing_p` beyond it, `bucket_sizing_p = 0.99999`) or rule C
(`bs` given). The script records the grid (`bs`, `x_min`), the frequency
truncation `|phi(pi / bs)|` of the continuous part, the Chernoff bounds on
the mass outside the window (`outside_below`, `outside_above`), the sum,
mean and standard deviation of the masses, the lower quantiles of the
masses at a few levels and the masses themselves at 257 evenly spaced
indices and at the largest one. `params` gives the law in prospicio's
terms.
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (law, aggregate family, aggregate keywords, chf_scale, chf_loc, log2, bs,
# prospicio's parameters). An affine law is `scale * base + shift`, the
# base's parameters prefixed `base_`.
CASES = [
    ("gamma", "gamma", dict(shape=4, scale=25), 1, 0, 12, 0, dict(shape=4, scale=25)),
    ("normal", "norm", dict(mean=100, sd=10), 1, 0, 10, 0, dict(mu=100, sigma=10)),
    ("normal", "norm", dict(mean=0, sd=1), 1, 0, 12, 0, dict(mu=0, sigma=1)),
    ("inverse_gaussian", "invgauss", dict(mean=10, shape=20), 1, 0, 12, 0,
     dict(mean=10, shape=20)),
    ("inverse_gaussian", "invgauss", dict(mean=1000, shape=4000), 1, 0, 14, 0,
     dict(mean=1000, shape=4000)),
    ("tweedie", "tweedie", dict(p=1.5, mean=100, dispersion=2), 1, 0, 12, 0,
     dict(mean=100, dispersion=2, power=1.5)),
    ("tweedie", "tweedie", dict(p=1.8, mean=50, dispersion=0.5), 1, 0, 12, 0,
     dict(mean=50, dispersion=0.5, power=1.8)),
    ("gamma", "gamma", dict(shape=0.5, scale=10), 1, 0, 12, 0, dict(shape=0.5, scale=10)),
    ("affine_gamma", "gamma", dict(shape=3), 2, 5, 10, 0,
     dict(scale=2, shift=5, base_shape=3, base_scale=1)),
    ("affine_tweedie", "tweedie", dict(p=1.5, mean=100, dispersion=2), 0.5, 10, 12, 0,
     dict(scale=0.5, shift=10, base_mean=100, base_dispersion=2, base_power=1.5)),
    ("gamma", "gamma", dict(shape=4, scale=25), 1, 0, 11, 0.5, dict(shape=4, scale=25)),
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
