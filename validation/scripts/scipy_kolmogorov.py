"""Regenerate the Kolmogorov-Smirnov p-value parity reference from SciPy (#805).

Run from the repository root:

    python validation/scripts/scipy_kolmogorov.py

Writes `validation/reference/kolmogorov_scipy.csv`: `scipy.stats.kstwo.sf(d, n)`,
the two-sided one-sample p-value, on a grid of `n` and `d` that crosses
every one of its methods (Ruben-Gambino's ends, twice Smirnov's tail,
Durbin's matrix, Pomeranz's recursion, Pelz and Good's series), and
`kstwobign.sf(z)`, Kolmogorov's limit, for `n` = "inf".
"""

import csv

import numpy as np
import scipy
from scipy import stats

OUT = "validation/reference/kolmogorov_scipy.csv"
SOURCE = f"scipy {scipy.__version__}"


def main():
    rows = []
    for n in [1, 2, 3, 5, 10, 20, 50, 89, 140, 141, 500, 2000, 20000, 200000]:
        for z in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]:
            d = z / np.sqrt(n)
            if d >= 1:
                continue
            p = float(stats.kstwo.sf(d, n))
            # SciPy's own asymptotic series beyond the exact range; ours is
            # Kolmogorov's limit there, close but not equal.
            exact = n <= 140 or n * d**1.5 <= 1.4 or n * d * d >= 2.2
            tol = 1e-10 if exact else 2e-3
            rows.append((n, repr(float(d)), repr(p), tol))
        for d in [0.6, 0.75, 0.95]:
            if n * d < n - 1 or n <= 3:
                rows.append((n, repr(d), repr(float(stats.kstwo.sf(d, n))), 1e-12))
    for z in [0.2, 0.5, 0.8, 1.0, 1.36, 2.0, 4.0]:
        rows.append(("inf", repr(z), repr(float(stats.kstwobign.sf(z))), 1e-14))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "d", "expected", "abs_tol", "rel_tol", "source"])
        for n, d, p, tol in rows:
            w.writerow([n, d, p, tol, 0.0, SOURCE])
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
