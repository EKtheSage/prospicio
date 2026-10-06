"""Regenerate validation/reference/mbbefd_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_mbbefd.py

MBBEFD exposure curves (Bernegger 1997) by integrating the survival
function of the destruction rate at 30 digits, independently of the
closed-form G in act_pricing::exposure:

    G(x) = integral of (1 - F) over (0, x) / integral over (0, 1)
    mean = integral of (1 - F) over (0, 1)

with F(x) = 1 - (1 - b) / ((g - 1) b^(1 - x) + 1 - g b) for x < 1 (and the
b = 1 and bg = 1 limits), for the Swiss Re curves c = 1.5, 2, 3, 4, 5
(b = exp(3.1 - 0.15 (1 + c) c), g = exp((0.78 + 0.12 c) c)) and three
direct (b, g) pairs covering the cases.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/mbbefd_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} quad of survival at 30 digits"
XS = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.9"]


def survival(b, g):
    if b == 1:
        return lambda x: 1 / (1 + (g - 1) * x)
    if b * g == 1:
        return lambda x: b**x
    return lambda x: (1 - b) / ((g - 1) * b ** (1 - x) + 1 - g * b)


def main():
    curves = []
    for c in ["1.5", "2", "3", "4", "5"]:
        cm = mp.mpf(c)
        b = mp.e ** (mp.mpf("3.1") - mp.mpf("0.15") * (1 + cm) * cm)
        g = mp.e ** ((mp.mpf("0.78") + mp.mpf("0.12") * cm) * cm)
        curves.append((f"c={c}", b, g))
    for b, g in [("0.5", "4"), ("1", "7"), ("0.25", "4")]:
        curves.append((f"b={b};g={g}", mp.mpf(b), mp.mpf(g)))
    rows = []
    for name, b, g in curves:
        s = survival(b, g)
        mean = mp.quad(s, [0, 1])
        rows.append([name, "mean", "", mean])
        for x in XS:
            rows.append([name, "G", x, mp.quad(s, [0, mp.mpf(x)]) / mean])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["curve", "quantity", "x", "expected", "abs_tol", "rel_tol", "source"])
        for name, q, x, v in rows:
            w.writerow([name, q, x, mp.nstr(v, 17), 1e-14, 1e-12, SOURCE])
    print(f"wrote {OUT}: {len(rows)} values", file=sys.stderr)


if __name__ == "__main__":
    main()
