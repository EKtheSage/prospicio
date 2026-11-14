# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aggregate==2.1.0",
#     "eltq==2.15.0",
#     "numpy",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-10T00:00:00Z"
# ///
"""Regenerate validation/reference/elt_grid_aggregate.csv.

    uv run validation/scripts/aggregate_elt_grid.py

An event loss table as the severity of a compound Poisson on a grid
against `aggregate` 2.1.0 (#781): `agg NAME elt [...]` updated with
`bs`, `log2` and `padding=2` (an FFT on four times the grid, the buffer
`fft` returns), the annual mass above the grid left out, as aggregate
leaves it. Rows, by `table` in `params`:

- `means`: an ELT3 table (event, rate, loss), which aggregate lowers to
  `dhistogram` atoms placed by the mean-preserving linear scatter
  (`EltBucketing::MeansLinear`): the severity `sev` and the annual total
  `agg` at every node where either is above 1e-300, to 1e-13.
- `means_ceded`: the same table with `occurrence ceded to 1000 xs 500`,
  the layer's annual ceded loss `agg` at every node, to 1e-13.
- `secondary`: an ELT6 table (event, rate, mean, max, sdi, sdc), which
  aggregate quantizes with eltq 2.15.0's survival method by the nearest
  node (`EltBucketing::Nearest`); eltq interpolates each row's survival
  between log-spaced nodes, holding it within about 1e-4 relative, so
  `sev` and `agg` are compared to 1e-7 absolute.
- `secondary_exact`: the same rule computed exactly from SciPy's beta
  survival at the bucket edges `(k + 1/2) bs`, the mass above the top edge
  on the top node, to 1e-12 (`sev` only; source SciPy).

The tables are in `TABLES` below, which the Rust test repeats.
"""

import csv
import sys
import warnings

import aggregate
import numpy as np
import scipy
from aggregate import build
from scipy import stats

warnings.filterwarnings("ignore")
OUT = "validation/reference/elt_grid_aggregate.csv"
AGG = f"aggregate {aggregate.__version__}"
SCIPY = f"scipy {scipy.__version__}"
BS, LOG2 = 10.0, 10
N = 1 << LOG2
# rows: event, rate, mean, max, sdi, sdc
MEANS = [[1, 0.5, 130.0], [2, 0.25, 270.5], [3, 0.05, 1234.5], [4, 0.01, 2999.9], [5, 0.002, 7777.7]]
SECONDARY = [
    [1, 0.01, 800.0, 2000.0, 200.0, 100.0],
    [2, 0.05, 250.0, 500.0, 0.0, 0.0],
    [3, 0.002, 1500.0, 5000.0, 300.0, 300.0],
    [4, 0.2, 40.0, 300.0, 30.0, 10.0],
    [5, 0.0005, 6000.0, 20000.0, 3000.0, 1000.0],
]
LAYER = "occurrence ceded to 1000 xs 500"


def body(rows):
    def row(r):
        return "[" + ", ".join([str(int(r[0]))] + [repr(float(v)) for v in r[1:]]) + "]"
    return "elt [" + ", ".join(row(r) for r in rows) + "]"


def run(spec):
    a = build(spec, update=False)
    a.update(log2=LOG2, bs=BS, padding=2)
    return np.asarray(a.sev_density, float), a.density_df.p_total.values.astype(float)


def exact_nearest(rows):
    p = np.zeros(N)
    total = sum(r[1] for r in rows)
    edges = (np.arange(N) + 0.5) * BS
    for _, rate, m, mx, si, sc in rows:
        sd = si + sc
        if sd == 0:
            p[min(int(np.floor(m / BS + 0.5)), N - 1)] += rate / total
            continue
        mu, v = m / mx, (sd / mx) ** 2
        k = mu * (1 - mu) / v - 1
        sf = stats.beta(mu * k, (1 - mu) * k, scale=mx).sf(edges)
        q = np.empty(N)
        q[0] = 1 - sf[0]
        q[1:] = sf[:-1] - sf[1:]
        q[-1] += sf[-1]
        p += rate / total * q
    return p


def emit(rows, table, quantity, values, tol, source):
    for k, v in enumerate(values):
        if v > 1e-300:
            rows.append(["elt_grid", f"table={table}", quantity, str(k), "", repr(float(v)),
                         repr(tol), "0.0", source])


def main():
    out = []
    sev, agg = run(f"agg M {body(MEANS)}")
    emit(out, "means", "sev", sev, 1e-13, AGG)
    emit(out, "means", "agg", agg, 1e-13, AGG)
    _, ceded = run(f"agg MC {body(MEANS)} {LAYER}")
    emit(out, "means_ceded", "agg", ceded, 1e-13, AGG)
    sev, agg = run(f"agg S {body(SECONDARY)}")
    emit(out, "secondary", "sev", sev, 1e-7, AGG)
    emit(out, "secondary", "agg", agg, 1e-7, AGG)
    emit(out, "secondary_exact", "sev", exact_nearest(SECONDARY), 1e-12, SCIPY)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(out)
    print(f"wrote {len(out)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
