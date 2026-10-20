"""Print aggregate 1.2.0's `sharpen` probes behind `grid_audit_matches_aggregate_sharpen`.

    python3.13 -m venv agg && agg/bin/pip install aggregate==1.2.0
    agg/bin/python validation/scripts/aggregate_sharpen.py

Two books, each probed from an explicit center: a capped lognormal from
a step of 1 on 2^13 points (the grid ends short of the book) and a gamma
from 0.5 on 2^10 (nothing sound meets the target, so log2 grows). The
test in `validation/tests/aggregate.rs` copies the scores of the sound
cells, the six terms of the selected cell and the selection.
"""

import warnings

import pandas as pd
from aggregate import build

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

COLUMNS = ["bs", "log2", "score", "u_sev_mean", "u_sev_cv", "u_sev_skew",
           "u_agg_mean", "u_agg_cv", "u_agg_skew", "deficit", "defective",
           "selected"]

for program, bs, log2 in [
    ("agg L 100 claims 1000 xs 0 sev lognorm 100 cv 1.5 poisson", 1, 13),
    ("agg G 50 claims sev gamma 10 cv 0.5 poisson", 0.5, 10),
]:
    a = build(program, update=False)
    a.update(bs=bs, log2=log2)
    a.sharpen(bs=bs, log2=log2)
    print(program)
    print(a.sharpen_df[COLUMNS].to_string())
    print()
