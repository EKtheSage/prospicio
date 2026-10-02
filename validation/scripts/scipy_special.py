"""Regenerate validation/reference/special_scipy.csv from SciPy.

Run from the repository root:

    python validation/scripts/scipy_special.py

Special functions in act-math: the regularized incomplete beta
(scipy.special.betainc) and Student's t distribution function
(scipy.stats.t.cdf). Lower-tail arguments down to t = -40 check that small
probabilities keep their relative precision.
"""

import csv
import sys

import scipy
from scipy import special, stats

OUT = "validation/reference/special_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"

BETA = [(0.5, 0.5), (1.5, 4.0), (10.0, 0.7), (50.0, 40.0), (200.0, 3.0)]
BETA_X = [1e-8, 0.01, 0.3, 0.5, 0.9, 0.999]
T_NU = [1.0, 2.5, 4.0, 30.0, 300.0]
T_X = [-40.0, -3.0, -0.5, 0.0, 0.25, 2.0, 12.0]


def rows():
    for a, b in BETA:
        for x in BETA_X:
            yield "beta_inc", f"a={a};b={b}", x, special.betainc(a, b, x), 1e-300, 1e-12
    for nu in T_NU:
        for t in T_X:
            yield "student_t", f"nu={nu}", t, stats.t.cdf(t, nu), 1e-300, 1e-12


def main():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for fn, params, x, value, abs_tol, rel_tol in rows():
            w.writerow([fn, params, "cdf", repr(float(x)), repr(float(value)), abs_tol, rel_tol, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
