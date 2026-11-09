"""Regenerate validation/reference/elt_units_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 --with eltq==2.15.0 \
        python validation/scripts/aggregate_elt_units.py

Event loss tables of several units that share events against `aggregate`
2.1.0 (#777): `agg NAME elt unit A [[...]] elt unit B [[...]]`, each row
`event rate mean max sdi sdc`, an event on two units one occurrence rolled
up as aggregate does by default (the event's rate; means, maxima and
correlated sds summed, independent sds in quadrature). Rows: the
aggregate's analytic `('freq', 'ex1')` (the rate of events, `quantity`
`rate`) and `('sev', 'ex1'..'ex3')` (`ex1` to `ex3`), for the whole table
(`unit=all`) and for each unit alone (`unit=A`, ...). The tables are in
`TABLES` below, which the Rust test repeats. Moments by eltq, at `1e-10`.
"""

import csv
import sys
import warnings

import aggregate
from aggregate import build

warnings.filterwarnings("ignore")
OUT = "validation/reference/elt_units_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
# name: {unit: [[event, rate, mean, max, sdi, sdc], ...]}
TABLES = {
    "two": {
        "A": [[1, 0.01, 800, 2000, 200, 100], [2, 0.05, 200, 500, 0, 0], [4, 0.002, 1500, 5000, 300, 300]],
        "B": [[1, 0.01, 300, 1000, 50, 40], [3, 0.02, 100, 400, 30, 0], [4, 0.002, 600, 2000, 100, 50]],
    },
    "three": {
        "X": [[10, 0.1, 50, 200, 10, 5], [11, 0.03, 400, 1200, 80, 60]],
        "Y": [[10, 0.1, 20, 100, 5, 5], [12, 0.004, 900, 3000, 200, 150]],
        "Z": [[10, 0.1, 5, 50, 2, 0], [11, 0.03, 100, 600, 40, 10], [12, 0.004, 300, 1500, 90, 40]],
    },
}


def body(units):
    def row(r):
        return "[" + ", ".join([str(int(r[0]))] + [repr(float(v)) for v in r[1:]]) + "]"
    return " ".join(f"elt unit {u} [{', '.join(row(r) for r in rows)}]" for u, rows in units.items())


def main():
    rows = []
    for name, units in TABLES.items():
        views = {"all": units, **{u: {u: r} for u, r in units.items()}}
        for view, sub in views.items():
            a = build(f"agg {name.upper()} {body(sub)}", update=False)
            s = a.stats_df["mixed"]
            params = f"table={name};unit={view}"
            rows.append(["elt", params, "rate", "", "", repr(float(s[("freq", "ex1")])), "0.0", "1e-10", SOURCE])
            for k in (1, 2, 3):
                rows.append(["elt", params, f"ex{k}", "", "", repr(float(s[("sev", f"ex{k}")])),
                             "0.0", "1e-10", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
