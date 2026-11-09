"""Regenerate validation/reference/disappearing_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_disappearing.py

Deductibles on a ground-up loss against `aggregate` 2.1.0 (#772): the
disappearing deductible `y xs d1 to d2 disappearing` and, beside it, the
franchise `y xs d franchise` and the ordinary `y xs d`. Each row is a raw
moment (`quantity` `ex1`, `ex2`, `ex3`) of the payment of one claim that
exceeds `d1`, from
aggregate's analytic severity statistics (`stats_df`, `('sev', 'exk')`,
column `mixed`), for a lognormal (mean 100, cv 1) or gamma (mean 80, cv
0.6) loss: `params` gives the deductible, `d1`, `d2` (none for the ordinary
and franchise forms), the limit `y` (`inf` for none) and the severity.
aggregate's layered severity is conditional on the loss exceeding the
attachment (its count is of claims that reach the layer), so prospicio's
unconditional moments are divided by `P(X > d1)`. aggregate integrates the
moments numerically, prospicio from the
severity's layer moments in closed form, so they agree to `1e-9` relative.
"""

import csv
import math
import sys
import warnings

import aggregate
from aggregate import build

warnings.filterwarnings("ignore")
OUT = "validation/reference/disappearing_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

SEVS = {"lognormal": "lognorm 100 cv 1", "gamma": "gamma 80 cv 0.6"}
CASES = [
    # (form, d1, d2, y, severity)
    ("disappearing", 50.0, 100.0, 300.0, "lognormal"),
    ("disappearing", 50.0, 100.0, 30.0, "lognormal"),
    ("disappearing", 50.0, 100.0, math.inf, "lognormal"),
    ("disappearing", 20.0, 250.0, 150.0, "lognormal"),
    ("disappearing", 50.0, 100.0, 300.0, "gamma"),
    ("franchise", 50.0, None, 300.0, "lognormal"),
    ("franchise", 50.0, None, 30.0, "lognormal"),
    ("ordinary", 50.0, None, 300.0, "lognormal"),
]


def main():
    rows = []
    for form, d1, d2, y, sev in CASES:
        lim = "inf" if math.isinf(y) else f"{y:g}"
        if form == "disappearing":
            layer = f"{lim} xs {d1:g} to {d2:g} disappearing"
        else:
            layer = f"{lim} xs {d1:g}" + (" franchise" if form == "franchise" else "")
        a = build(f"agg T 2 claims {layer} sev {SEVS[sev]} poisson", bs=1 / 4, log2=13)
        params = f"form={form};d1={d1!r};d2={d2!r};y={y!r};severity={sev}"
        for k in (1, 2, 3):
            v = float(a.stats_df.loc[("sev", f"ex{k}"), "mixed"])
            rows.append(["deductible", params, f"ex{k}", "", "", repr(v), "0.0", "1e-9", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
