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
"""Regenerate validation/reference/chf_split_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_chf_split.py

Stable laws with two power tails (`|beta| < 1`) under the tilt against
Mildenhall's `aggregate` 2.1.0 (`AggregateCHF`, `chfagg ... stable`; #780):
each law split into its two totally skewed halves, each half inverted on
twice the grid with its own tilt (a normal moved onto the narrower half
near `|beta| = 1`), convolved and cropped to equal mass outside on each
side. Cases at `alpha` 0.8, 1, 1.1, 1.5 and 1.8, `beta` from -0.9 to 0.99,
with a scale and location, a fixed tilt and a given bucket. A symmetric
law (`beta = 0`) is left out: its crop is a tie between two neighbouring
windows (gaps 1.011078e-7 and 1.011076e-7 at `alpha = 1.5`) that
round-off decides, so the two engines may place it a bucket apart; every
case here has its best crop at least 15% below the next. `params` and
the quantities are those of `aggregate_chf_tilt.py`.
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_split_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (alpha, beta, scale, loc, S0?, log2, tilt, bs)
CASES = [
    (1.5, 0.5, 1, 0, False, 12, True, 0),
    (1.5, 0.1, 1, 0, False, 12, True, 0),
    (1.5, 0.99, 1, 0, False, 12, True, 0),
    (1.1, -0.3, 2, 5, False, 12, True, 0),
    (0.8, 0.2, 1, 0, False, 12, True, 0),
    (1.0, 0.5, 1, 0, False, 12, True, 0),
    (1.8, -0.9, 3, 10, False, 12, True, 0),
    (1.5, 0.5, 1, 0, False, 12, 6.0, 0),
    (1.5, 0.5, 1, 0, False, 12, True, 0.05),
    (1.5, 0.5, 2, 1, True, 13, True, 0),
]

LEVELS = [0.01, 0.1, 0.5, 0.9, 0.99]

# (abs_tol, rel_tol) by quantity: the grid exactly; the masses to FFT
# round-off times the tilt's amplification e^tau at the heavy end (e^14.6
# = 2.3e6 at alpha 0.7), so the sum of 4,096 of them to 1e-9 (8e-11 apart
# at alpha 0.7); the tilt and edge error to the Brent stop of the 1e-40
# Chernoff edge.
# The split adds two tail-series values and a convolution of 2n x 2n
# tilted masses, so the first bucket (which holds the mass below the
# window) and the edge error (the larger of the one-sided model and the
# outside mass) carry a few more ulps of their round-off than the
# one-sided grids: masses to 1e-10, the edge error to 1e-7 relative.
TOLERANCE = {
    "step": (0.0, 0.0),
    "x_min": (1e-12, 0.0),
    "tilt_left": (1e-9, 1e-9),
    "tilt_right": (1e-9, 1e-9),
    "edge_error": (0.0, 1e-7),
    "truncation_error": (1e-300, 1e-9),
    "outside_below": (1e-11, 1e-9),
    "outside_above": (1e-11, 1e-9),
    "sum": (1e-9, 0.0),
    "quantile": (1e-9, 0.0),
    "mass": (1e-10, 0.0),
}


def tilt_code(tilt):
    if tilt is True:
        return -1
    if tilt is False:
        return -2
    return tilt


def main():
    rows = []
    for alpha, beta, scale, loc, s0, log2, tilt, bs in CASES:
        a = AggregateCHF("S", "stable", alpha=alpha, beta=beta, scale=scale, loc=loc,
                         parameterization="S0" if s0 else "S1")
        a.update(log2=log2, bs=bs, tilt=tilt)
        p = np.asarray(a.agg_density, dtype=float)
        n = len(p)
        x = a.x_min + a.bs * np.arange(n)
        params = dict(alpha=alpha, beta=beta, scale=scale, loc=loc,
                      parameterization=int(s0), log2=log2, bs=bs, tilt=tilt_code(tilt))
        text = ";".join(f"{k}={v}" for k, v in params.items())

        def put(quantity, arg, value):
            abs_tol, rel_tol = TOLERANCE[quantity]
            rows.append(dict(distribution="stable", params=text, quantity=quantity, arg=arg,
                             arg2="", expected=repr(float(value)), abs_tol=abs_tol,
                             rel_tol=rel_tol, source=SOURCE))

        put("step", "", a.bs)
        put("x_min", "", a.x_min)
        put("tilt_left", "", a.tilt[0])
        put("tilt_right", "", a.tilt[1])
        if np.isfinite(a.edge_error):
            put("edge_error", "", a.edge_error)
        put("truncation_error", "", a.truncation_error)
        put("outside_below", "", a.outside_below)
        put("outside_above", "", a.outside_above)
        put("sum", "", p.sum())
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
