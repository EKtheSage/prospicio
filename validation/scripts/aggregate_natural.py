"""Regenerate validation/reference/natural_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_natural.py

Monograph 15's InsCo example (Major and Mildenhall, 2026): three units
over ten equally likely events. For each distortion (CCoC at 15%, and the
PH, Wang, dual and TVaR distortions aggregate calibrates to the same
premium), each allocation (linear and lifted) and each asset level (100,
the largest total, so no default; 65, the 0.85 quantile; and 40), the
script records aggregate's `Portfolio.price` loss, margin, premium,
capital and assets by unit and in total. Then Bodoff's percentile layer
of capital at 100 and 65 (`Portfolio.bodoff(p=0, a=)`).

The same for the Discrete case of *Pricing Insurance Risk* (Mildenhall and
Major, 2022): two independent units, X1 0, 8 or 10 and X2 0, 1 or 90,
each with probabilities 1/2, 1/4, 1/4, built by `aggregate.build` (which
convolves the units by FFT), priced at a 10% cost of capital at assets
100, 90 and 11. Its rows carry `case=discrete` in `params`.

Rows: `distribution` is `price` or `bodoff`; `params` names the
distortion, allocation and assets; `quantity` is the unit and the column
(`X1.P`); `expected` is aggregate's value. aggregate computes on a grid
with bucket 1, which holds every total, so the only differences are the
order of floating-point sums: tolerance `1e-10` absolute.
"""

import csv
import sys
import warnings

import aggregate
import pandas as pd
from aggregate import Distortion, Portfolio, build

OUT = "validation/reference/natural_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
INSCO = pd.DataFrame({
    "X1": [15, 15, 5, 7, 13, 5, 15, 26, 17, 16],
    "X2": [7, 13, 20, 33, 20, 27, 16, 19, 8, 20],
    "X3": [0, 0, 11, 0, 7, 8, 9, 10, 40, 64],
    "p_total": 0.1,
})
KEYS = {"ccoc": "r", "ph": "a", "wang": "lam", "dual": "b", "tvar": "p"}


def priced(case, port, coc, levels, units, rows):
    res = port.calibrate_distortions(coc, p=1)
    for name in ["ccoc", "ph", "wang", "dual", "tvar"]:
        param = float(res.distortion_df.loc[name, "param"])
        d = Distortion(name, **{KEYS[name]: param})
        for alloc in ["linear", "lifted"]:
            for a in levels:
                frame = port.price(a, d, allocation=alloc).df.loc[str(d)]
                params = f"case={case};family={name};param={param!r};assets={a}"
                for u in units + ["total"]:
                    for col in ["L", "M", "P", "Q", "a"]:
                        rows.append(["price", params + f";allocation={alloc}", f"{u}.{col}",
                                     "", "", repr(float(frame.loc[u, col])), "1e-10", "0.0", SOURCE])
    for a in levels[:2]:
        b = port.bodoff(p=0, a=a)
        for u in units:
            rows.append(["bodoff", f"case={case};assets={a}", u, "", "",
                         repr(float(b[u].iloc[0])), "1e-10", "0.0", SOURCE])


def main():
    warnings.filterwarnings("ignore")
    rows = []
    insco = Portfolio.create_from_sample("InsCo", INSCO, bs=1, log2=8)
    priced("insco", insco, 0.15, [100, 65, 40], ["X1", "X2", "X3"], rows)
    discrete = build("port Discrete agg X1 1 claim dsev [0 8 10] [1/2 1/4 1/4] fixed "
                     "agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed", bs=1, log2=8)
    priced("discrete", discrete, 0.1, [100, 90, 11], ["X1", "X2"], rows)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
