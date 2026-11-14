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
"""Regenerate validation/reference/chf_tweedie_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_chf_tweedie.py

The Tweedie law at any power computed by inverting its characteristic
function against Mildenhall's `aggregate` 2.1.0 (`AggregateCHF`, `chfagg
tweedie p mean dispersion`; #780, second PR): the normal (`p = 0`), the
extreme stable laws (`p < 0`), the compound Poisson-gamma with its atom
(`1 < p < 2`), the gamma (`p = 2`), the inverse Gaussian (`p = 3`) and the
positive extreme stable laws (`p > 2`), each on the grid rule A chooses.
The quantities are those of `aggregate_chf.py`. `params` gives the law in
prospicio's terms (`TweediePower(power, mean, dispersion)`).
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_tweedie_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (law, aggregate family, aggregate keywords, chf_scale, chf_loc, log2, bs,
# prospicio's parameters). An affine law is `scale * base + shift`, the
# base's parameters prefixed `base_`.
CASES = [
    ("tweedie_power", "tweedie", dict(p=0, mean=5, dispersion=4), 1, 0, 12, 0,
     dict(power=0, mean=5, dispersion=4)),
    ("tweedie_power", "tweedie", dict(p=-1, mean=3, dispersion=2), 1, 0, 12, 0,
     dict(power=-1, mean=3, dispersion=2)),
    ("tweedie_power", "tweedie", dict(p=-3, mean=1, dispersion=0.5), 1, 0, 12, 0,
     dict(power=-3, mean=1, dispersion=0.5)),
    ("tweedie_power", "tweedie", dict(p=1.2, mean=10, dispersion=1), 1, 0, 12, 0,
     dict(power=1.2, mean=10, dispersion=1)),
    ("tweedie_power", "tweedie", dict(p=1.5, mean=100, dispersion=2), 1, 0, 12, 0,
     dict(power=1.5, mean=100, dispersion=2)),
    ("tweedie_power", "tweedie", dict(p=2, mean=10, dispersion=0.25), 1, 0, 12, 0,
     dict(power=2, mean=10, dispersion=0.25)),
    ("tweedie_power", "tweedie", dict(p=2.5, mean=3, dispersion=0.4), 1, 0, 12, 0,
     dict(power=2.5, mean=3, dispersion=0.4)),
    ("tweedie_power", "tweedie", dict(p=3, mean=10, dispersion=0.5), 1, 0, 12, 0,
     dict(power=3, mean=10, dispersion=0.5)),
    ("tweedie_power", "tweedie", dict(p=5, mean=2, dispersion=0.1), 1, 0, 12, 0,
     dict(power=5, mean=2, dispersion=0.1)),
    ("tweedie_power", "tweedie", dict(p=2.5, mean=10, dispersion=0.28460498941515416), 1, 0, 14, 0,
     dict(power=2.5, mean=10, dispersion=0.28460498941515416)),
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
