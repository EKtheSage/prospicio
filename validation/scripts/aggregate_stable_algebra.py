# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aggregate==2.1.0",
#     "numpy",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-10T00:00:00Z"
# ///
"""Regenerate validation/reference/stable_algebra_aggregate.csv.

    python3.13 -m venv agg && agg/bin/pip install aggregate==2.1.0
    agg/bin/python validation/scripts/aggregate_stable_algebra.py

The alpha-stable law's algebra against Mildenhall's `aggregate` 2.1.0
(`aggregate.stable.Stable`; #780): sums of laws with a common alpha,
shifts, scaling by positive and negative numbers (the log term at
`alpha = 1`), negation, `t @ X` (`t` independent copies), the S0 location,
the tail constants, `Pr(X > loc)` and the generalized central limit
theorem's `Stable.from_compound_pareto`. Each row is one parameter of the
result in S1. `params` gives the operands as `x_alpha;x_beta;...`.
"""

import csv

import aggregate
from aggregate.stable import Stable

OUT = "validation/reference/stable_algebra_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

LAWS = {
    "a": (1.5, 0.4, 2.0, 1.0),
    "b": (1.5, -0.8, 1.0, -3.0),
    "c": (1.0, 0.6, 1.7, 0.5),
    "d": (1.0, -0.2, 0.3, 2.0),
    "e": (0.7, 1.0, 1.2, 0.0),
    "f": (0.7, 0.3, 0.5, -1.0),
}


def law(key):
    return Stable(*LAWS[key])


def text(**kw):
    return ";".join(f"{k}={v}" for k, v in kw.items())


def main():
    rows = []

    def put(op, operands, st):
        p = text(**operands)
        for quantity, value in (("alpha", st.alpha), ("beta", st.beta), ("scale", st.scale),
                                ("loc", st.loc), ("loc0", st.loc0),
                                ("tail_constant", st.tail_constant),
                                ("left_tail_constant", st.left_tail_constant),
                                ("prob_above_loc", st.prob_above_loc)):
            if value != value:
                # Pr(X > loc) is nan at alpha = 1 unless beta = 0; the
                # Rust unit tests check that case
                continue
            rows.append(dict(distribution="stable", params=p, quantity=quantity, arg=op,
                             arg2="", expected=repr(float(value)), abs_tol=1e-13,
                             rel_tol=1e-13, source=SOURCE))

    def operands(key, prefix="x"):
        a, b, g, d = LAWS[key]
        return {f"{prefix}_alpha": a, f"{prefix}_beta": b, f"{prefix}_scale": g, f"{prefix}_loc": d}

    for x, y in (("a", "b"), ("c", "d"), ("e", "f")):
        put("add", {**operands(x), **operands(y, "y")}, law(x) + law(y))
    for x in LAWS:
        put("identity", operands(x), law(x))
        put("shift", {**operands(x), "c": 2.5}, law(x) + 2.5)
        put("neg", operands(x), -law(x))
        for c in (3.0, -0.4):
            put("scale", {**operands(x), "c": c}, c * law(x))
        put("copies", {**operands(x), "c": 2.7}, 2.7 @ law(x))
    put("s0", {"x_alpha": 1.5, "x_beta": 0.4, "x_scale": 2.0, "x_loc": 1.0, "c": 0},
        Stable(1.5, 0.4, 2.0, 1.0, parameterization="S0"))
    put("s0", {"x_alpha": 1.0, "x_beta": 0.6, "x_scale": 1.7, "x_loc": 0.5, "c": 0},
        Stable(1.0, 0.6, 1.7, 0.5, parameterization="S0"))
    for lam, a, th, m in ((50.0, 1.6, 3.0, 12.0), (1000.0, 1.2, 0.5, 4.0)):
        put("compound_pareto", {"frequency": lam, "alpha": a, "theta": th, "sev_mean": m},
            Stable.from_compound_pareto(lam, a, th, m))

    fields = ["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol",
              "rel_tol", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
