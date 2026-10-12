"""Regenerate validation/reference/oep_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_oep.py

The occurrence PML (OEP) against Mildenhall's `aggregate` 1.0.1,
`aggregate.utilities.oep`, on Poisson books: at each annual probability
the loss, the severity's exceedance at it and the occurrence return
period. `oep` is closed form from the severity (it never reads the
aggregate's grid), so the tolerance is `1e-10` relative.

`params` gives each book in prospicio's terms (Poisson `lambda`; the
severity family and its parameters).
"""

import csv
import sys
import warnings

import aggregate
from aggregate import build
from aggregate.utilities import oep

OUT = "validation/reference/oep_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

BOOKS = [  # (name, DecL, params)
    ("O", "agg O 2 claims sev lognorm 100 cv 2 poisson",
     "lambda=2;severity=lognormal;mean=100;cv=2"),
    ("G", "agg G 10 claims sev gamma 50 cv 0.7 poisson",
     "lambda=10;severity=gamma;mean=50;cv=0.7"),
    ("W", "agg W 0.5 claims sev 1000 * weibull_min 0.7 poisson",
     "lambda=0.5;severity=weibull;shape=0.7;scale=1000"),
]

PROBABILITIES = [0.3, 0.1, 0.04, 0.01, 0.004, 0.001, 1e-4, 1e-5]


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for name, decl, params in BOOKS:
        a = build(decl, bs=1, log2=10)
        lam = float(a.n)
        ps = [p for p in PROBABILITIES if p < 1 - pow(2.718281828459045, -lam)]
        df = oep(a, ps)
        for p in ps:
            r = df.loc[p]
            rows.append((name, params, "loss", p, float(r["loss"]), 0.0, 1e-10))
            rows.append((name, params, "severity_survival", p, float(r["S_sev"]), 0.0, 1e-9))
            rows.append((name, params, "occurrence_return_period", p,
                         float(r["occurrence_return_period"]), 0.0, 1e-9))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for name, params, qty, arg, val, abs_tol, rel_tol in rows:
            w.writerow([name, params, qty, repr(float(arg)), repr(val), abs_tol, rel_tol, SOURCE])
    print(f"wrote {OUT}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
