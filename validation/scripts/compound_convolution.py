"""Regenerate validation/reference/compound_convolution.csv.

    pip install numpy scipy
    python validation/scripts/compound_convolution.py

Compound distributions S = X_1 + ... + X_N computed by brute force,
P(S = k) = sum_n P(N = n) f^{*n}(k), with repeated numpy convolutions and
SciPy claim-count pmfs. This is independent of Panjer's recursion (and of
FFT), which prospicio-aggregate checks against it.
"""

import csv
import sys

import numpy as np
import scipy
from scipy import stats

OUT = "validation/reference/compound_convolution.csv"
SOURCE = f"numpy {np.__version__} convolution; scipy {scipy.__version__} pmf"

SEVERITY = [0.1, 0.3, 0.25, 0.2, 0.1, 0.05]  # P(X = 0..5), step 1
POINTS = 40
CASES = [
    ("poisson", "lambda=3.0", stats.poisson(3.0)),
    ("negative_binomial", "r=2.5;beta=1.5", stats.nbinom(2.5, 1 / 2.5)),
]


def compound(count):
    f = np.array(SEVERITY)
    s = np.zeros(POINTS)
    conv = np.zeros(POINTS)
    conv[0] = 1.0  # f^{*0}
    n = 0
    while count.sf(n) > 1e-18 or n < 5:
        s += count.pmf(n) * conv
        conv = np.convolve(conv, f)[:POINTS]
        n += 1
    return s


def main():
    sev = ";".join(repr(p) for p in SEVERITY)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frequency", "params", "severity", "points", "index", "expected", "abs_tol", "rel_tol", "source"])
        for name, params, count in CASES:
            for k, p in enumerate(compound(count)):
                w.writerow([name, params, sev, POINTS, k, repr(float(p)), 1e-14, 0.0, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
