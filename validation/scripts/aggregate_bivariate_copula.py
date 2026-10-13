"""Regenerate validation/reference/bivariate_copula_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_bivariate_copula.py

Two lines hit by one count of events, with the losses of an event that
hits both coupled by a copula, against Mildenhall's `aggregate` 1.0.1
(`bivariate NAME 12 claims agg A dfreq [0 1] [1-pA pA] sev ... agg B ...
copula KIND P <freq>`). Each event hits line A with probability 1/4 and
loses a gamma (mean 10, cv 1.5), and B with probability 1/3 and loses a
lognormal (mean 5, cv 0.5); the per-event severity is the copula
rectangle pmf of the two triggered marginals,
`S[i, j] = C(G_A[i], G_B[j]) − C(G_A[i−1], G_B[j]) − C(G_A[i], G_B[j−1])
+ C(G_A[i−1], G_B[j−1])`. Copulas: `gumbel` with Kendall's tau 0.4
(theta 5/3), `clayton` with tau 0.4 (theta 4/3) and `normal` with rho
0.5. Counts: Poisson with mean 12, and gamma-mixed (cv 0.5). Grid: steps
1 and 0.5 on 2^11 and 2^10 points, padding 1.

`aggregate`'s normal copula evaluates scipy's bivariate normal cdf (exact
in two dimensions); replacing it with a 1e-13 one moves the density by
2e-17. As for clash, `aggregate`'s 1e-15 fuzz clip is turned off and the
deficit is asserted below 1e-13. Probabilities at 1e-12 absolute,
moments at 1e-9 relative.
"""

import csv
import sys
import warnings

import numpy as np

import aggregate
import aggregate.bivariate
from aggregate import build

OUT = "validation/reference/bivariate_copula_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
UNITS = (
    "agg A dfreq [0 1] [0.75 0.25] sev gamma 10 cv 1.5 "
    "agg B dfreq [0 1] [0.6666666666666666 0.3333333333333333] sev lognorm 5 cv 0.5"
)


def main():
    warnings.filterwarnings("ignore")
    aggregate.bivariate._clip_density_fuzz = lambda density, where: None
    rows = []
    for copula in ["gumbel 0.4", "clayton 0.4", "normal 0.5"]:
        for count, freq in [("poisson", "poisson"), ("mixed", "mixed gamma 0.5")]:
            b = build(f"bivariate T 12 claims {UNITS} copula {copula} {freq}", update=False)
            b.update(bs=(1.0, 0.5), log2=(11, 10))
            assert b.padding == 1
            assert abs(b.deficit) < 1e-13, b.deficit
            kind, param = copula.split()
            params = f"copula={kind};param={param};count={count}"

            def add(quantity, arg, arg2, value, abs_tol, rel_tol):
                rows.append([kind, params, quantity, repr(float(arg)),
                             "" if arg2 is None else repr(float(arg2)),
                             repr(float(value)), abs_tol, rel_tol, SOURCE])

            d = b.density
            for i, j in [(0, 0), (0, 10), (10, 0), (5, 7), (30, 40), (60, 20), (100, 50),
                         (200, 100)]:
                add("density", i, j, d[i, j], "1e-12", "0.0")
            ma, mb = b.marginals
            for k in [0, 1, 10, 30, 60, 100, 200, 400]:
                add("marginal_a", k, None, ma[k], "1e-12", "0.0")
            for k in [0, 1, 10, 30, 60, 100, 200]:
                add("marginal_b", k, None, mb[k], "1e-12", "0.0")
            tp = np.asarray(b.total.p, dtype=float)
            for k in [0, 1, 10, 50, 100, 200, 400]:
                add("total", k, None, tp[k], "1e-12", "0.0")
            m = b.moments(2).to_numpy()
            add("mean_a", 0, None, m[1, 0], "0.0", "1e-9")
            add("mean_b", 0, None, m[0, 1], "0.0", "1e-9")
            add("var_a", 0, None, m[2, 0] - m[1, 0] ** 2, "0.0", "1e-9")
            add("var_b", 0, None, m[0, 2] - m[0, 1] ** 2, "0.0", "1e-9")
            add("moment_ab", 0, None, m[1, 1], "0.0", "1e-9")
            add("correlation", 0, None, b.corr, "0.0", "1e-9")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
