# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aggregate==2.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-11-10T00:00:00Z"
# ///
"""Regenerate validation/reference/validation_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_validation.py

The verdict on a built aggregate against `aggregate` 2.1.0's `valid`
(`Validation` flags and `Validation.passes`, #778): each book built at an
explicit `bs` and `log2`, the severity discretized by rounding (aggregate's
default) and not normalized, so mass beyond the grid stays a deficit.
`quantity` is a flag (`SEV_MEAN`, `SEV_CV`, `SEV_SKEW`, `AGG_MEAN`,
`AGG_CV`, `AGG_SKEW`, `ALIASING`, `DEFECTIVE`) or `passes`, `expected` 1 when
set and 0 when not. The books are chosen with verdicts away from the
tolerances (each moment error under a third or over three times its bound),
so the two discretizations, aggregate's and prospicio's, agree on them.
"""

import csv
import sys
import warnings

import aggregate
from aggregate import build
from aggregate.constants import Validation

warnings.filterwarnings("ignore")
OUT = "validation/reference/validation_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
FLAGS = ["SEV_MEAN", "SEV_CV", "SEV_SKEW", "AGG_MEAN", "AGG_CV", "AGG_SKEW",
         "ALIASING", "DEFECTIVE"]
# (name, program, bs, log2): prospicio's params give the same book
BOOKS = [
    ("gamma_fine", "agg G 50 claims sev gamma 10 cv 0.5 poisson", 0.5, 11,
     "count=poisson;n=50;severity=gamma;mean=10;cv=0.5"),
    ("gamma_coarse", "agg G 50 claims sev gamma 10 cv 0.5 poisson", 4.0, 9,
     "count=poisson;n=50;severity=gamma;mean=10;cv=0.5"),
    ("gamma_short", "agg G 50 claims sev gamma 10 cv 0.5 poisson", 0.25, 10,
     "count=poisson;n=50;severity=gamma;mean=10;cv=0.5"),
    ("lognormal_capped_fine", "agg L 100 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson", 4.0, 13,
     "count=poisson;n=100;severity=lognormal;mean=100;cv=1.5;cap=1000"),
    ("lognormal_capped_short", "agg L 100 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson", 1.0, 13,
     "count=poisson;n=100;severity=lognormal;mean=100;cv=1.5;cap=1000"),
    ("lognormal_capped_coarse", "agg L 100 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson", 64.0, 10,
     "count=poisson;n=100;severity=lognormal;mean=100;cv=1.5;cap=1000"),
]


def main():
    rows = []
    for name, prog, bs, log2, params in BOOKS:
        a = build(prog, bs=bs, log2=log2, normalize=False, update=True)
        v = a.valid
        for flag in FLAGS:
            on = bool(v & getattr(Validation, flag))
            rows.append([name, f"{params};bs={bs!r};log2={log2}", flag, "", "",
                         "1.0" if on else "0.0", "0.0", "0.0", SOURCE])
        rows.append([name, f"{params};bs={bs!r};log2={log2}", "passes", "", "",
                     "1.0" if v.passes else "0.0", "0.0", "0.0", SOURCE])
        print(name, v, file=sys.stderr)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
