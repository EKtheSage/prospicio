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
"""Regenerate validation/reference/chf_tvar_aggregate.csv.

    uv run validation/scripts/aggregate_chf_tvar.py

The tail value at risk of a law inverted from its characteristic function,
`AggregateCHF.tvar` in Mildenhall's `aggregate` 2.1.0 (#779, #780): its
`GridDistribution`'s kernel on the buckets of positive mass, their masses
cumulated as inverted (not renormalized), the largest point from the
next-to-last cumulative on. One or two laws of each family the engine
inverts: light laws on rule A's grid, a Tweedie with its atom, the Lévy
laws, a tilted stable law with a right and one with a left power tail,
and a two-sided one split into its halves. `params` are those of the
other `aggregate_chf_*.py` scripts (`bs` 0 sizes the bucket, `tilt` -1 is
automatic, `parameterization` 1 is S0).
"""

import csv

import aggregate
import numpy as np
from aggregate.characteristic_function import AggregateCHF

OUT = "validation/reference/chf_tvar_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

# (law, aggregate family, aggregate keywords, log2, update keywords,
# prospicio's parameters)
CASES = [
    ("gamma", "gamma", dict(shape=4, scale=25), 12, {}, dict(shape=4, scale=25)),
    ("normal", "norm", dict(mean=0, sd=1), 12, {}, dict(mu=0, sigma=1)),
    ("inverse_gaussian", "invgauss", dict(mean=10, shape=20), 12, {}, dict(mean=10, shape=20)),
    ("tweedie_power", "tweedie", dict(p=1.5, mean=100, dispersion=2), 12, {},
     dict(power=1.5, mean=100, dispersion=2)),
    ("tweedie_power", "tweedie", dict(p=3, mean=10, dispersion=0.5), 12, {},
     dict(power=3, mean=10, dispersion=0.5)),
    ("normal_inverse_gaussian", "norminvgauss", dict(alpha=2, beta=0.5, delta=1.5, mu=0.3), 12, {},
     dict(alpha=2, beta=0.5, delta=1.5, mu=0.3)),
    ("variance_gamma", "variancegamma", dict(sigma=0.2, nu=0.5, theta=-0.1, mu=0.05), 12, {},
     dict(sigma=0.2, nu=0.5, theta=-0.1, mu=0.05)),
    ("tempered_stable", "temperedstable", dict(C=0.5, G=2, M=8, Y=1.5, mu=1), 12, {},
     dict(C=0.5, G=2, M=8, Y=1.5, mu=1)),
    ("stable", "stable", dict(alpha=1.5, beta=1, scale=1, loc=0, parameterization="S1"), 12,
     dict(tilt=True), dict(alpha=1.5, beta=1, scale=1, loc=0, parameterization=0, tilt=-1)),
    ("stable", "stable", dict(alpha=1.5, beta=-1, scale=2, loc=3, parameterization="S1"), 12,
     dict(tilt=True), dict(alpha=1.5, beta=-1, scale=2, loc=3, parameterization=0, tilt=-1)),
    ("stable", "stable", dict(alpha=1.8, beta=1, scale=10, loc=100, parameterization="S0"), 12,
     dict(tilt=True), dict(alpha=1.8, beta=1, scale=10, loc=100, parameterization=1, tilt=-1)),
    ("stable", "stable", dict(alpha=1.5, beta=0.5, scale=1, loc=0, parameterization="S1"), 12,
     dict(tilt=True), dict(alpha=1.5, beta=0.5, scale=1, loc=0, parameterization=0, tilt=-1)),
]

LEVELS = [0.0, 0.01, 0.5, 0.9, 0.99, 0.995, 0.999]

# (abs_tol, rel_tol). A TVaR sums the masses times the grid points above
# the VaR and divides by 1 - p. Untilted, the masses agree to FFT
# round-off (1e-13), so the TVaR to 1e-9 relative (1e-9 absolute for one
# near 0). Under the tilt the round-off at the heavy end is amplified by
# up to e^tau (the masses agree to 1e-10 there), and the TVaR at 0.99 to
# 0.999 of a tilted stable law differs from aggregate's by up to 4e-9
# relative: 1e-8 for those.
TOLERANCE = (1e-9, 1e-9)
TILTED_TOLERANCE = (1e-9, 1e-8)


def main():
    rows = []
    for law, family, kw, log2, update, params in CASES:
        a = AggregateCHF("L", family, **kw)
        a.update(log2=log2, bs=0, **update)
        text = ";".join(f"{k}={v}" for k, v in {**params, "log2": log2, "bs": 0}.items())
        tilted = "tilt" in update and a.tilt != (0, 0)
        for level in LEVELS:
            abs_tol, rel_tol = TILTED_TOLERANCE if tilted else TOLERANCE
            rows.append(dict(distribution=law, params=text, quantity="tvar", arg=level, arg2="",
                             expected=repr(float(a.tvar(level))), abs_tol=abs_tol,
                             rel_tol=rel_tol, source=SOURCE))

    fields = ["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol",
              "rel_tol", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
