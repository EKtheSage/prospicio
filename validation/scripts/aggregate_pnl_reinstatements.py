"""Reference values for reinstatement premiums in the P&L ledger, from
`aggregate` 1.0.1 (docs/design/pnl.md, PR 6b).

An occurrence cover 2 xs 2 with a deposit of 2 on example 4's book, over
the joint of the gross loss and the cover's unlimited annual cession `R`:

- `r1`, `r1c`: one reinstatement at 100% (an annual limit of 4), walked
  and consolidated: the recovery `A(R)` and the premium `D + h(R)`;
- `r0`, `r0c`: no reinstatements, an annual limit of 2 and no
  reinstatement premium;
- `r2`, `r2c`: one free and one at 50%, a 25% ceding commission, and an
  aggregate cover 3 xs 5 (deposit 0.5) on the net of `A(R)`, walked and
  consolidated (aggregate refuses to peel a reinstated program).

Every case runs on the same joint as example 4
(validation/reference/pnl_occurrence_sources.csv, `ex4`); the script
checks that.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_reinstatements.py

writes validation/reference/pnl_reinstatements_aggregate.csv.
"""

import csv
from pathlib import Path

import numpy as np
from aggregate import build

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

B = 'agg ExRe 2 claims dsev [1 2 4] [.5 .3 .2]'
EXP = 'less 10% premium expense'
OCC = 'occurrence net of 2 xs 2 deposit 2'
CASES = {
    'r1': f'xpnl R 20 premium less {B} {OCC} reinstatements 1 at 100% poisson {EXP}',
    'r1c': f'pnl R 20 premium less {B} {OCC} reinstatements 1 at 100% poisson {EXP}',
    'r0': f'xpnl R 20 premium less {B} {OCC} no reinstatements poisson {EXP}',
    'r0c': f'pnl R 20 premium less {B} {OCC} no reinstatements poisson {EXP}',
}
TWO = (f'{B} {OCC} cede 25% reinstatements 1 free and 1 at 50% poisson '
       'aggregate net of 3 xs 5 deposit 0.5')
CASES['r2'] = f'xpnl R 20 premium less {TWO} {EXP}'
CASES['r2c'] = f'pnl R 20 premium less {TWO} {EXP}'

with open(ref / 'pnl_occurrence_sources.csv') as f:
    ex4 = sorted((float(r['gross']), float(r['ceded']), float(r['prob']))
                 for r in csv.DictReader(f) if r['source'] == 'ex4')

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


for case, prog in CASES.items():
    pnl = build(prog, log2=6, bs=1)
    src = pnl.source
    a0, a1 = np.asarray(src.axis0, dtype=float), np.asarray(src.axis1, dtype=float)
    x, y = np.repeat(a0, len(a1)), np.tile(a1, len(a0))
    p = np.asarray(pnl._probs, dtype=float)
    keep = p > 0
    atoms = sorted(zip(x[keep].tolist(), y[keep].tolist(), p[keep].tolist()))
    assert atoms == ex4, f'{case}: not example 4 joint'
    # SD and CV relative to 1e-10: E[X^2] - E[X]^2 on the net premium (a
    # mean of 17.4, an SD of 0.24) loses about 1e-12 to cancellation, and
    # the two sides sum in different orders.
    def tols(col):
        if col == 'Skew':
            return 1e-10, 0.0
        return (1e-12, 1e-10) if col in ('SD', 'CV') else (1e-12, 0.0)
    for idx, r in pnl.economic_df.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            col = col.replace('κ', 'k')
            add(case, 'sheet', label, col, v, *tols(col))
    for idx, r in pnl.summary_df.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v, *tols(col))
    for _, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX)

(ref / 'pnl_reinstatements_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
