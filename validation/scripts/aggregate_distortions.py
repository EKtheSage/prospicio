"""Regenerate validation/reference/distortion_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_distortions.py

Mildenhall's `aggregate` package (1.0.1, Python 3.12 or later) is the
reference implementation of the distortions in Mildenhall and Major,
*Pricing Insurance Risk* (2022), and of CAS Monograph 15 (Major and
Mildenhall, 2026). Three kinds of row:

- `g`: `g(s)` of each distortion kind at a grid of `s`, from
  `aggregate.Distortion`. The kinds are named as in prospicio
  (`ccoc`, `bitvar`, ...); `params` holds their parameters.
- `insco_price`: the price of the monograph's InsCo example (ten equally
  likely totals; assets 100, the largest) under the distortions that
  `Portfolio.calibrate_distortions(0.15, p=1)` returns, at aggregate's
  calibrated parameter. Rust prices the same totals at the same parameter.
- `insco_calibrate`: the calibrated parameter itself. aggregate stops its
  Newton iteration at a premium error (`error` in its `distortion_df`), so
  the tolerance is that error over the price's slope in the parameter,
  times ten, plus `1e-9`.
"""

import csv
import sys

import aggregate
import numpy as np
import pandas as pd
from aggregate import Distortion, Portfolio

OUT = "validation/reference/distortion_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
S = [1e-6, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999999]

# (prospicio kind, params as written in the csv, aggregate distortion)
KINDS = [
    ("tvar", {"p": 0.3}, Distortion("tvar", p=0.3)),
    ("wang", {"lambda": 0.4}, Distortion("wang", lam=0.4)),
    ("ph", {"rho": 0.6}, Distortion("ph", a=0.6)),
    ("dual", {"beta": 2.5}, Distortion("dual", b=2.5)),
    ("ccoc", {"r": 0.15}, Distortion("ccoc", r=0.15)),
    ("bitvar", {"p0": 0.2, "p1": 0.9, "w": 0.3}, Distortion("bitvar", p0=0.2, p1=0.9, w1=0.3)),
    ("bitvar", {"p0": 0.0, "p1": 1.0, "w": 0.2}, Distortion("bitvar", p0=0.0, p1=1.0, w1=0.2)),
    ("clin", {"r0": 0.05, "slope": 1.4}, Distortion("clin", r0=0.05, slope=1.4)),
    ("cll", {"r0": 0.1, "b": 0.8}, Distortion("cll", r0=0.1, b=0.8)),
    ("lep", {"r0": 0.02, "r": 0.2}, Distortion("lep", r0=0.02, r=0.2)),
    ("ly", {"r0": 0.03, "r": 0.5}, Distortion("ly", r0=0.03, r=0.5)),
    ("beta", {"a": 0.6, "b": 1.5}, Distortion("beta", a=0.6, b=1.5)),
]
WTD = ([0.1, 0.5, 0.9], [0.2, 0.5, 0.3])

INSCO = pd.DataFrame({
    "X1": [15, 15, 5, 7, 13, 5, 15, 26, 17, 16],
    "X2": [7, 13, 20, 33, 20, 27, 16, 19, 8, 20],
    "X3": [0, 0, 11, 0, 7, 8, 9, 10, 40, 64],
    "p_total": 0.1,
})


def fmt(params):
    return ";".join(f"{k}={v!r}" for k, v in params.items())


def row(distribution, params, quantity, arg, expected, abs_tol, rel_tol):
    return [distribution, params, quantity, repr(arg), "", repr(float(expected)),
            repr(abs_tol), repr(rel_tol), SOURCE]


def insco_price(x, p, d):
    """aggregate's price of the discrete distribution under d."""
    return float(d.price(pd.Series(p, index=x), a=100, kind="ask").ask)


def main():
    rows = []
    for kind, params, d in KINDS:
        for s in S:
            rows.append(row("g", fmt(params), kind, s, d.g(np.array([s]))[0], 1e-15, 1e-12))
    wtd = Distortion("wtdtvar", ps=WTD[0], wts=WTD[1])
    for s in S:
        rows.append(row("g", "ps=0.1|0.5|0.9;wts=0.2|0.5|0.3", "wtdtvar", s,
                        wtd.g(np.array([s]))[0], 1e-15, 1e-12))

    port = Portfolio.create_from_sample("InsCo", INSCO, bs=1, log2=8)
    res = port.calibrate_distortions(0.15, p=1)
    total = INSCO[["X1", "X2", "X3"]].sum(axis=1)
    dist = total.value_counts(normalize=True).sort_index()
    x, p = dist.index.to_numpy(float), dist.to_numpy(float)
    for name, d in res.distortions.items():
        param = float(res.distortion_df.loc[name, "param"])
        err = float(res.distortion_df.loc[name, "error"])
        price = insco_price(x, p, d)
        rows.append(row("insco_price", f"param={param!r}", name, param, price, 0.0, 1e-12))
        if name == "ccoc":
            continue
        h = 1e-6 * max(abs(param), 1e-3)
        key = {"ph": "a", "wang": "lam", "dual": "b", "tvar": "p"}[name]
        up = insco_price(x, p, Distortion(name, **{key: param + h}))
        dn = insco_price(x, p, Distortion(name, **{key: param - h}))
        slope = abs(up - dn) / (2 * h)
        tol = 10 * abs(err) / slope + 1e-9
        rows.append(row("insco_calibrate", "premium=53.565217391304344", name,
                        53.565217391304344, param, tol, 0.0))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
