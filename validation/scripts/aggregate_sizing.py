"""Regenerate validation/reference/sizing_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_sizing.py

FFT grid sizing against Mildenhall's `aggregate` 1.0.1:

- `round_bucket`: aggregate's own docstring cases and their reciprocals,
  exact.
- Five books built with no `bs` (aggregate sizes the grid at `log2 = 16`):
  the bucket aggregate chose (`bs`, recorded, not asserted: prospicio sizes
  with two moments and a single big jump, aggregate with three-moment
  fits), the exact mean, and aggregate's quantiles (built with
  `normalize=False`, see below) at 0.5, 0.99, 0.999 and 0.9999. A quantile
  matches to within two buckets of the coarser of the two grids, the mean
  to `1e-3` relative.

`params` gives each book in prospicio's terms (Poisson `lambda`, or a
gamma-mixed Poisson with mixing `cv`; the severity family and its
parameters).
"""

import csv
import math
import sys
import warnings

import aggregate
from aggregate import build
from aggregate.utilities import round_bucket

OUT = "validation/reference/sizing_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

ROUNDS = [1, 1.1, 2, 2.5, 4, 5, 5.5, 8.7, 9.9, 10, 13, 15, 20, 50, 100, 99, 101, 200, 250, 400,
          457, 500, 750, 1000, 2412, 12323, 57000, 119000, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21]

BOOKS = [  # (name, DecL, params, exact mean)
    ("A", "agg A 10 claims sev gamma 50 cv 0.7 poisson",
     "lambda=10;severity=gamma;mean=50;cv=0.7", 500.0),
    ("B", "agg B 5 claims sev lognorm 100 cv 1 poisson",
     "lambda=5;severity=lognormal;mean=100;cv=1", 500.0),
    ("D", "agg D 50 claims sev lognorm 1000 cv 3 mixed gamma 0.3",
     "lambda=50;mixing_cv=0.3;severity=lognormal;mean=1000;cv=3", 50000.0),
    ("E", "agg E 2 claims sev 1000 * weibull_min 0.7 poisson",
     "lambda=2;severity=weibull;shape=0.7;scale=1000", 2 * 1000 * math.gamma(1 + 1 / 0.7)),
    ("F", "agg F 20 claims sev 100 * pareto 2.5 - 100 poisson",
     "lambda=20;severity=lomax;alpha=2.5;scale=100", 20 * 100 / 1.5),
]


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for x in ROUNDS:
        rows.append(("round_bucket", "", "round_bucket", x, round_bucket(x), 0.0, 0.0))
        rows.append(("round_bucket", "", "round_bucket", 1 / x, round_bucket(1 / x), 0.0, 0.0))
    for name, decl, params, mean in BOOKS:
        a = build(decl)
        bs = float(a.bs)
        # Quantiles without aggregate's default renormalization of the
        # severity cut at the grid top, which thins the tail (on F it moves
        # the 0.9999 quantile 0.65% down); prospicio lumps that mass on the
        # last point instead, and so does aggregate with normalize=False.
        a = build(decl, normalize=False)
        assert float(a.bs) == bs
        # aggregate's bucket and its grid mean (est_m), for the record.
        p = f"{params};agg_bs={bs!r};agg_grid_mean={float(a.est_m)!r}"
        rows.append((name, p, "mean", "", mean, 0.0, 1e-3))
        for q in [0.5, 0.99, 0.999, 0.9999]:
            rows.append((name, p, "quantile", q, float(a.q(q)), 2 * bs, 0.0))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for r in rows:
            name, params, qty, arg, val, abs_tol, rel_tol = r
            arg = "" if arg == "" else repr(float(arg))
            w.writerow([name, params, qty, arg, repr(float(val)), abs_tol, rel_tol, SOURCE])
    print(f"wrote {OUT}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
