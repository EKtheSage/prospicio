"""Regenerate validation/reference/program_oep_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_program_oep.py

The occurrence PML of a program, net or ceded, against `aggregate` 2.1.0's
`oep` (`aggregate.utilities.oep`, #773) on an aggregate with an
`occurrence net of` or `occurrence ceded to` clause: `loss`, the PML of the
view at annual probability `arg`; `S_sev`, the gross survival at the right
end of the view's flat stretch through that loss; `oep`, the achieved
annual probability. Poisson counts with a lognormal (mean 100, cv 2) or
gamma (mean 50, cv 1.5) severity; programs: one layer `400 xs 100`, two
layers with a gap and a share (`50% po 200 xs 100 and 500 xs 400`, a 50% share of 200 xs 100). The
PML is the gross PML through the program's function of an occurrence, a
closed form, so `loss` and `S_sev` agree to `1e-9` relative; the probabilities
past the program's top or inside a flat stretch check the atoms.
"""

import csv
import sys
import warnings

import aggregate
import numpy as np
from aggregate import build
from aggregate.utilities import oep

warnings.filterwarnings("ignore")
OUT = "validation/reference/program_oep_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
SEVS = {"lognormal": "sev lognorm 100 cv 2", "gamma": "sev gamma 50 cv 1.5"}
PROGRAMS = {"one": "400 xs 100", "two": "50% po 200 xs 100 and 500 xs 400"}
PS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.004, 0.001, 1e-4]


def main():
    rows = []
    for sev_name, sev in SEVS.items():
        for prog_name, prog in PROGRAMS.items():
            for view in ("net", "ceded"):
                clause = "net of" if view == "net" else "ceded to"
                a = build(f"agg P 3 claims {sev} occurrence {clause} {prog} poisson", update=False)
                df = oep(a, PS)
                params = f"severity={sev_name};program={prog_name};view={view};lambda=3"
                for p, r in df.iterrows():
                    for col in ("loss", "S_sev", "oep"):
                        rows.append(["program_oep", params, col, repr(float(p)), "",
                                     repr(float(r[col])), "1e-12", "1e-9", SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
