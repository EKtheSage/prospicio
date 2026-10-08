"""Regenerate validation/reference/counts_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_counts.py

Claim counts beyond the (a, b, 0) class, against Mildenhall's `aggregate`
1.0.1. Each count is built as an aggregate with a unit severity
(`dsev [1]`), so its distribution is the count's; the script records
P(N = k) for k = 0..39. `params` gives the count in prospicio's terms:

- `zm`: Poisson with base mean `lambda`, `P(N = 0) = p0`
  (`poisson zm p0 !`, the trailing `!` fixing the base mean);
- `zt`: zero-truncated Poisson whose mean is 2 (`poisson zt`); the base
  mean `lambda` solves `lambda / (1 - exp(-lambda)) = 2`;
- `logarithmic`: mean 3; `p` solves the logarithmic mean;
- `mixed`: Poisson mean `lambda` mixed by gamma or inverse Gaussian with
  coefficient of variation `cv` (of the whole mixing variable) and fixed
  part `shift` (`mixed gamma`, `delaporte`, `ig`, `sig`);
- `neyman`: Poisson(`lambda`) clusters of Poisson(`theta`) claims
  (`neymana theta`, mean `lambda theta`).

aggregate computes these by FFT on a grid of 1024 points with bucket 1,
which holds all the mass: tolerance `1e-12` absolute.
"""

import csv
import math
import sys
import warnings

import aggregate
from aggregate import build
from scipy.optimize import brentq

OUT = "validation/reference/counts_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"


def main():
    warnings.filterwarnings("ignore")
    zt_lambda = brentq(lambda l: l / (1 - math.exp(-l)) - 2, 1e-6, 10)
    log_p = brentq(lambda p: -p / ((1 - p) * math.log1p(-p)) - 3, 1e-9, 1 - 1e-12)
    cases = [
        ("zm", "lambda=2.0;p0=0.3", "agg A 2 claims dsev [1] poisson zm 0.3 !"),
        ("zt", f"lambda={zt_lambda!r}", "agg A 2 claims dsev [1] poisson zt"),
        ("logarithmic", f"p={log_p!r}", "agg A 3 claims dsev [1] logarithmic"),
        ("mixed", "lambda=10.0;mixing=gamma;cv=0.4;shift=0.0", "agg A 10 claims dsev [1] mixed gamma 0.4"),
        ("mixed", "lambda=10.0;mixing=gamma;cv=0.4;shift=0.3",
         "agg A 10 claims dsev [1] mixed delaporte 0.4 0.3"),
        ("mixed", "lambda=10.0;mixing=ig;cv=0.5;shift=0.0", "agg A 10 claims dsev [1] mixed ig 0.5"),
        ("mixed", "lambda=10.0;mixing=ig;cv=0.5;shift=0.3", "agg A 10 claims dsev [1] mixed sig 0.5 0.3"),
        ("neyman", "lambda=2.0;theta=3.0", "agg A 6 claims dsev [1] neymana 3"),
    ]
    rows = []
    for name, params, program in cases:
        p = build(program, log2=10, bs=1).density_df.p_total
        for k in range(40):
            rows.append([name, params, "pmf", str(k), "", repr(float(p.iloc[k])), "1e-12", "0.0", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
