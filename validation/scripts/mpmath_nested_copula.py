# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mpmath==1.4.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-13T00:00:00Z"
# ///
"""Regenerate validation/reference/nested_copula_mpmath.csv.

    uv run --with mpmath python validation/scripts/mpmath_nested_copula.py

The distribution function of nested Archimedean copulas (McNeil 2008),
evaluated with mpmath at 30 digits: a node is
psi(sum over its leaves of psi^-1(u_j) + sum over its children of
psi^-1(C_child(u))), with the node's own theta.

Each case names a tree (`TREES` below, also written out in the test as a
node table): a node is (theta, leaves, children), leaves 0-based.
"""

import csv
from pathlib import Path

import mpmath as mp

mp.mp.dps = 30


def psi(family, theta, t):
    if family == "clayton":
        return (1 + t) ** (-1 / theta)
    return mp.exp(-(t ** (1 / theta)))


def psi_inv(family, theta, u):
    if family == "clayton":
        return u ** (-theta) - 1
    return (-mp.log(u)) ** theta


def cdf(family, node, u):
    theta, leaves, children = node
    t = sum((psi_inv(family, theta, u[j]) for j in leaves), mp.mpf(0))
    for child in children:
        t += psi_inv(family, theta, cdf(family, child, u))
    return psi(family, theta, t)


TREES = {
    # Python's docstring example: 0 alone at 1.5, {1, 2} at 3, {3, 4} at 2.
    "gumbel5": ("gumbel", (mp.mpf("1.5"), [0], [(mp.mpf(3), [1, 2], []), (mp.mpf(2), [3, 4], [])])),
    # Three levels: 0 at 1, 1 at 2.5 below it, {2, 3} at 4 below that.
    "clayton4": ("clayton", (mp.mpf(1), [0], [(mp.mpf("2.5"), [1], [(mp.mpf(4), [2, 3], [])])])),
    # Two groups and no leaf at the root.
    "clayton4b": ("clayton", (mp.mpf("0.5"), [], [(mp.mpf(2), [0, 1], []), (mp.mpf(3), [2, 3], [])])),
}

POINTS = {
    5: [
        ["0.5", "0.5", "0.5", "0.5", "0.5"],
        ["0.1", "0.9", "0.3", "0.7", "0.5"],
        ["0.95", "0.99", "0.97", "0.9", "0.92"],
        ["0.02", "0.05", "0.1", "0.03", "0.2"],
    ],
    4: [
        ["0.5", "0.5", "0.5", "0.5"],
        ["0.1", "0.9", "0.3", "0.7"],
        ["0.95", "0.99", "0.97", "0.9"],
        ["0.02", "0.05", "0.1", "0.03"],
    ],
}


def dim(node):
    _, leaves, children = node
    return len(leaves) + sum(dim(c) for c in children)


rows = []
for name, (family, tree) in TREES.items():
    for p in POINTS[dim(tree)]:
        u = [mp.mpf(x) for x in p]
        rows.append([name, family, ";".join(p), mp.nstr(cdf(family, tree, u), 17), "1e-14", "0", "mpmath %s at 30 digits" % mp.__version__])

out = Path(__file__).resolve().parents[1] / "reference" / "nested_copula_mpmath.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["tree", "family", "point", "expected", "abs_tol", "rel_tol", "source"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
