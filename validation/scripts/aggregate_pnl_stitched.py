"""Reference values for the stitched peel of several occurrence layers,
from `aggregate` 1.0.1 (docs/design/pnl.md, follow-up 7c).

A book of 2 claims (Poisson) of 1, 2, 4 or 6, premium 20, 10% premium
expense; occurrence covers 1 xs 1 (deposit 1) and 2 xs 2 (deposit 1, a
20% ceding commission):

- `td`: peeled top down;
- `bu`: peeled bottom up, with an aggregate cover 3 xs 5 (deposit 0.5)
  behind them.

Every row of a stitched peel is its own marginal, so the sheet's mean,
standard deviation, CV and skewness and the card are recorded. The
scenario (`k`) columns are not: `aggregate` prints a Palm ladder here that
conditions each row through the claims, or through the retained aggregate
once an aggregate cover exists (the gross loss row reads `k01 = -16` on
`td` and `-11.55` on `bu`), and prospicio does not build it.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_stitched.py

writes validation/reference/pnl_stitched_aggregate.csv.
"""

from pathlib import Path

import numpy as np
from aggregate import build

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

B = 'agg ExSt 2 claims dsev [1 2 4 6] [.4 .3 .2 .1]'
OCC = 'occurrence net of 1 xs 1 deposit 1 and 2 xs 2 deposit 1 cede 20% poisson'
EXP = 'less 10% premium expense'
CASES = {
    'td': f'xpnl S 20 premium less {B} {OCC} {EXP} peel top-down',
    'bu': f'xpnl S 20 premium less {B} {OCC} aggregate net of 3 xs 5 deposit 0.5 '
          f'{EXP} peel bottom-up',
}
STATS = ('EX', 'SD', 'CV', 'Skew')

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


for case, prog in CASES.items():
    pnl = build(prog, log2=6, bs=1)
    for idx, r in pnl.economic_df.iterrows():
        label = '|'.join(idx)
        for col in STATS:
            add(case, 'sheet', label, col, r[col], 1e-10 if col == 'Skew' else 1e-12)
    for idx, r in pnl.summary_df.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v, 1e-10 if col == 'Skew' else 1e-12)
    for _, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX)

(ref / 'pnl_stitched_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
