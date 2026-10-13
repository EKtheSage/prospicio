"""Regenerate validation/reference/picks_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_picks.py

Layer loss picks against Mildenhall's `aggregate` 1.0.1
(`Aggregate.picks(attachments, layer_loss_picks)`, which returns the
adjusted severity's probabilities): a gamma severity with mean 100 and cv
1.5 on a step of 1 with 2^14 points, 10 expected claims, layers 100, 400 xs
100 and 500 xs 500.

- `scaled`: picks 480, 420 and 60, near the severity's own (495.1, 447.6,
  52.5), so every layer is a rescaled survival function;
- `straight`: picks 700, 250 and 50, where the bottom layer's rescaled
  survival would pass 1 and takes a straight line instead. `aggregate`
  rounds that line's length to the grid and reaches 704.90 in the bottom
  layer (it warns); prospicio solves the length on the grid and reaches
  700. So this case compares only the points above the bottom layer.

Quantities: `p` at loss `arg`, the adjusted probability (every fifth point
to 1,010, from 101 in `straight`, and the points around each attachment); `layer`
with `arg` the layer's top, the adjusted loss to the layer for 10 claims.
The adjusted survival is a few sums and products of the severity's, so
`1e-12` absolute on `p` and `1e-9` relative on the layers.
"""

import csv
import sys
import warnings

import aggregate
import numpy as np
from aggregate import build

OUT = "validation/reference/picks_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
ATTACHMENTS = [100, 500, 1000]


def main():
    warnings.filterwarnings("ignore")
    rows = []
    agg = build("agg G 10 claims sev gamma 100 cv 1.5 poisson", bs=1, log2=14)
    xs = agg.xs
    for case, picks, first in [("scaled", [480, 420, 60], 0), ("straight", [700, 250, 50], 101)]:
        params = "picks=" + ":".join(str(p) for p in picks)
        p = agg.picks(ATTACHMENTS, picks)
        near = {a + d for a in ATTACHMENTS for d in (-1, 0, 1, 2)}
        for j in sorted(set(range(first, 1011, 5)) | {j for j in near if j >= first}):
            rows.append([case, params, "p", str(j), "", repr(float(p[j])), "1e-12", "0.0", SOURCE])
        lev = [0.0] + [float(np.sum(np.minimum(xs, a) * p)) for a in ATTACHMENTS]
        for i, a in enumerate(ATTACHMENTS):
            if case == "straight" and i == 0:
                continue
            rows.append([case, params, "layer", str(a), "",
                         repr(float(agg.n * (lev[i + 1] - lev[i]))), "0.0", "1e-9", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
