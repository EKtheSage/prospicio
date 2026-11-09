"""Regenerate validation/reference/lattice_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_lattice.py

The dyadic lattice of a discrete severity against `aggregate` 2.1.0
(#776): `aggregate.utilities._dyadic_lattice_step` (`quantity` `step` and
`residual`, absent when there is no lattice: `none` is 1) and
`_lattice_candidates` (`candidate` with `arg` its rank), for atom sets of
whole, binary-fraction, scaled and decimal values. Exact.
"""

import csv
import sys

import aggregate
from aggregate.utilities import _dyadic_lattice_step, _lattice_candidates

OUT = "validation/reference/lattice_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
ATOMS = [
    [1, 2, 4], [3, 6, 9], [0.75, 1.5], [2e5, 4e5, 1e6], [0.1, 0.2], [1, 2.5],
    [1000, 3000, 7000], [12, 15], [0, 0], [-6, 9, 15], [0.375, 1.125], [300, 450],
    [7, 1e-7], [1 / 3, 2 / 3], [2.5e-5, 5e-5], [123456, 7],
]


def main():
    rows = []
    for atoms in ATOMS:
        label = "|".join(repr(float(a)) for a in atoms)
        res = _dyadic_lattice_step(atoms)
        if res is None:
            rows.append(["lattice", f"atoms={label}", "none", "0", "", "1.0", "0.0", "0.0", SOURCE])
            continue
        h, residual = res
        rows.append(["lattice", f"atoms={label}", "step", "0", "", repr(float(h)), "0.0", "0.0", SOURCE])
        rows.append(["lattice", f"atoms={label}", "residual", "0", "", repr(float(residual)),
                     "1e-15", "0.0", SOURCE])
        for i, c in enumerate(_lattice_candidates(h)):
            rows.append(["lattice", f"atoms={label}", "candidate", str(i), "", repr(float(c)),
                         "0.0", "0.0", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
