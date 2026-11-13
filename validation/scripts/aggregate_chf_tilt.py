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
"""Regenerate validation/reference/chf_tilt_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_chf_tilt.py

Stable laws inverted from their characteristic function against
Mildenhall's `aggregate` 2.1.0 (`AggregateCHF`, `chfagg ... stable`), the
second PR of #779: totally skewed laws (`beta = 1`, and `beta = -1` by
reflection) under the automatic tilt, a fixed tilt and no tilt, at
`alpha` 0.7, 1, 1.1, 1.5 and 1.8, with a scale, a location in S1 or S0
and a given bucket; and a law with `|beta| < 1` untilted. `params` gives
the law in prospicio's terms (`parameterization` 1 for S0, 0 for S1) and
the grid (`tilt` -1 automatic, -2 off, else the e-folds). The script
records the grid, the tilt on each side, the modelled edge error, the
outside masses, the truncation, the sum and lower quantiles of the masses
and the masses at 257 evenly spaced indices and at the largest one.
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_tilt_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (alpha, beta, scale, loc, S0?, log2, tilt, bs)
CASES = [
    (1.5, 1, 1, 0, False, 12, True, 0),
    (0.7, 1, 1, 0, False, 12, True, 0),
    (1.1, 1, 1, 0, False, 14, True, 0),
    (1.0, 1, 1, 0, False, 12, True, 0),
    (1.5, -1, 2, 3, False, 12, True, 0),
    (1.5, 1, 1, 0, False, 12, False, 0),
    (1.5, 1, 1, 0, False, 12, 8.0, 0),
    (1.8, 1, 10, 100, True, 12, True, 0),
    (1.5, 1, 1, 0, False, 12, True, 0.05),
    (1.0, -1, 3, -2, False, 12, True, 0),
    (1.5, 0.5, 1, 0, False, 12, False, 0),
]

LEVELS = [0.01, 0.1, 0.5, 0.9, 0.99]

# (abs_tol, rel_tol) by quantity: the grid exactly; the masses to FFT
# round-off times the tilt's amplification e^tau at the heavy end (e^14.6
# = 2.3e6 at alpha 0.7), so the sum of 4,096 of them to 1e-9 (8e-11 apart
# at alpha 0.7); the tilt and edge error to the Brent stop of the 1e-40
# Chernoff edge.
TOLERANCE = {
    "step": (0.0, 0.0),
    "x_min": (1e-12, 0.0),
    "tilt_left": (1e-9, 1e-9),
    "tilt_right": (1e-9, 1e-9),
    "edge_error": (0.0, 1e-8),
    "truncation_error": (1e-300, 1e-9),
    "outside_below": (1e-20, 1e-9),
    "outside_above": (1e-20, 1e-9),
    "sum": (1e-9, 0.0),
    "quantile": (1e-9, 0.0),
    "mass": (1e-11, 0.0),
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
