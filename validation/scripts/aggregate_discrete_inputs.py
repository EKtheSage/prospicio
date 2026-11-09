"""Regenerate validation/reference/discrete_inputs_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_discrete_inputs.py

Which discrete inputs `aggregate` 2.1.0 builds and which it refuses (#774).
`dsev` probabilities must sum to 1 within a relative 1e-5: each row is a
two-point severity `dsev [1 2] [0.5 q]` with a fixed count of 2, `expected`
1 when aggregate builds it and 0 when it raises. prospicio's
`Discrete::new` must accept and refuse the same probabilities. (A
fractional fixed count, `2.5 claims ... fixed`, is refused too; prospicio's
Rust count takes a whole number by its type, and the Python and R tests
check the bindings' refusal. `dfreq` probabilities are not checked by
aggregate: `dfreq [1 2] [.5 .49]` builds with a density summing to 0.99,
while prospicio's `EmpiricalCount` refuses a sum more than 1e-9 from 1.)
"""

import csv
import sys
import warnings

import aggregate
from aggregate import build

warnings.filterwarnings("ignore")
OUT = "validation/reference/discrete_inputs_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
QS = [0.5, 0.49999, 0.499991, 0.5000099, 0.500011, 0.49998, 0.48, 0.52, 1.5]


def main():
    rows = []
    for q in QS:
        try:
            build(f"agg T 2 claims dsev [1 2] [0.5 {q!r}] fixed")
            ok = 1.0
        except ValueError:
            ok = 0.0
        rows.append(["dsev", f"p1=0.5;p2={q!r}", "builds", "", "", repr(ok), "0.0", "0.0", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
