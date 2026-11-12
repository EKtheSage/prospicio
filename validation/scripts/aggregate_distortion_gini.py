# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aggregate==2.1.0",
#     "numpy",
#     "pandas",
# ]
# [tool.uv]
# exclude-newer = "2026-11-11T00:00:00Z"
# ///
"""Regenerate validation/reference/distortion_gini_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_distortion_gini.py

The distortions `aggregate` 2.1.0 added (#769, #770, #771) against
Mildenhall's package. Four kinds of row:

- `g`: `g(s)` of the Tsallis distortion of order `alpha` (0.5, 1, 2, 3, 10)
  at a grid of `s`, from `aggregate.Distortion("tsallis", alpha=)`.
- `gini`: `Distortion.gini`, `2 int g - 1`, of each family at one member,
  named and parameterised as in prospicio (`params`). aggregate has a
  closed form for the one-parameter families and integrates the others
  with `quad` to 1e-11.
- `from_gini`: the parameter of `Distortion.from_gini(kind, G)` for the
  six one-parameter families at G = 0.1, 0.25, 0.5 and 0.8, in prospicio's
  parameterisation (`ccoc`'s return `r`; aggregate sets `d = r / (1 + r)`
  to G).
- `insco_extended`: Monograph 15's InsCo (ten equally likely totals, assets
  100, a 15% cost of capital) calibrated by
  `Portfolio.calibrate_distortions(0.15, p=1, extended=True)`, aggregate's
  extended set with Tsallis between Wang and dual: each family's parameter,
  with the tolerance from aggregate's premium error as in
  `aggregate_distortions.py`, and the price at that parameter.
"""

import csv
import sys

import aggregate
import numpy as np
import pandas as pd
from aggregate import Distortion, Portfolio

OUT = "validation/reference/distortion_gini_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
S = [1e-12, 1e-6, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999999]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 10.0]

# (prospicio kind, params as written in the csv, aggregate distortion)
KINDS = [
    ("tvar", {"p": 0.3}, Distortion("tvar", p=0.3)),
    ("wang", {"lambda": 0.4}, Distortion("wang", lam=0.4)),
    ("ph", {"rho": 0.6}, Distortion("ph", a=0.6)),
    ("dual", {"beta": 2.5}, Distortion("dual", b=2.5)),
    ("ccoc", {"r": 0.15}, Distortion("ccoc", r=0.15)),
    ("tsallis", {"alpha": 3.0}, Distortion("tsallis", alpha=3.0)),
    ("tsallis", {"alpha": 0.5}, Distortion("tsallis", alpha=0.5)),
    ("bitvar", {"p0": 0.2, "p1": 0.9, "w": 0.3}, Distortion("bitvar", p0=0.2, p1=0.9, w1=0.3)),
    ("clin", {"r0": 0.05, "slope": 1.4}, Distortion("clin", r0=0.05, slope=1.4)),
    ("cll", {"r0": 0.1, "b": 0.8}, Distortion("cll", r0=0.1, b=0.8)),
    ("lep", {"r0": 0.02, "r": 0.2}, Distortion("lep", r0=0.02, r=0.2)),
    ("ly", {"r0": 0.03, "r": 0.5}, Distortion("ly", r0=0.03, r=0.5)),
    ("beta", {"a": 0.6, "b": 1.5}, Distortion("beta", a=0.6, b=1.5)),
]
# The attribute of each one-parameter family that is prospicio's parameter
# (ccoc's return r, which aggregate's from_gini sets through d = G).
ONE = {"ccoc": "r", "ph": "a", "wang": "lam", "tsallis": "alpha", "dual": "b", "tvar": "p"}

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
    for alpha in ALPHAS:
        d = Distortion("tsallis", alpha=alpha)
        for s in S:
            rows.append(row("g", f"alpha={alpha!r}", "tsallis", s, d.g(np.array([s]))[0],
                            1e-15, 1e-12))
    for kind, params, d in KINDS:
        # Closed forms to 1e-14; quad (clin, cll, lep, ly, beta, bitvar) to 1e-11.
        rows.append(row("gini", fmt(params), kind, 0.0, d.gini, 1e-11, 0.0))
    for kind, key in ONE.items():
        for g in [0.1, 0.25, 0.5, 0.8]:
            d = Distortion.from_gini(kind, g)
            rows.append(row("from_gini", f"gini={g!r}", kind, g, getattr(d, key), 0.0, 1e-12))

    port = Portfolio.create_from_sample("InsCo", INSCO, bs=1, log2=8)
    res = port.calibrate_distortions(0.15, p=1, extended=True)
    total = INSCO[["X1", "X2", "X3"]].sum(axis=1)
    dist = total.value_counts(normalize=True).sort_index()
    x, p = dist.index.to_numpy(float), dist.to_numpy(float)
    for name, d in res.distortions.items():
        param = float(res.distortion_df.loc[name, "param"])
        err = float(res.distortion_df.loc[name, "error"])
        rows.append(row("insco_extended_price", f"param={param!r}", name, param,
                        insco_price(x, p, d), 0.0, 1e-12))
        if name == "ccoc":
            continue
        key = {"ph": "a", "wang": "lam", "tsallis": "alpha", "dual": "b", "tvar": "p"}[name]
        h = 1e-6 * max(abs(param), 1e-3)
        up = insco_price(x, p, Distortion(name, **{key: param + h}))
        dn = insco_price(x, p, Distortion(name, **{key: param - h}))
        slope = abs(up - dn) / (2 * h)
        tol = 10 * abs(err) / slope + 1e-9
        rows.append(row("insco_extended", "premium=53.565217391304344", name,
                        53.565217391304344, param, tol, 0.0))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
