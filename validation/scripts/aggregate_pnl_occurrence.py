"""Reference values for the P&L ledger of occurrence covers on the joint
(gross, ceded) source, from `aggregate` 1.0.1 (docs/design/pnl.md, PR 5).

- `ex4`, `ex4c`: example 4, a book with an occurrence cover 2 xs 2
  (deposit 2, a 25% ceding commission) on a Poisson count, walked and
  consolidated;
- `ex7`, `ex7c`, `ex7td`: the same with an aggregate cover 3 xs 5 on the
  net of the occurrence cover (deposit 0.5), walked, consolidated and
  peeled top down;
- `joint`: the joint (gross, ceded) density of example 4, cell by cell.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_occurrence.py

writes validation/reference/pnl_occurrence_aggregate.csv and
validation/reference/pnl_occurrence_sources.csv (each walk's joint source
with the probabilities PnL uses; the consolidated faces are built from it).
"""

from pathlib import Path

import numpy as np
from aggregate import build

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

A = 'agg ExOcc 2 claims dsev [1 2 4] [.5 .3 .2]'
OCC = 'occurrence net of 2 xs 2 deposit 2 cede 25%'
AGG = 'aggregate net of 3 xs 5 deposit 0.5'
CASES = {
    'ex4': f'xpnl Ex4 20 premium less {A} {OCC} poisson less 10% premium expense',
    'ex4c': f'pnl Ex4 20 premium less {A} {OCC} poisson less 10% premium expense',
    'ex7': f'xpnl Ex7 20 premium less {A} {OCC} poisson {AGG} less 10% premium expense',
    'ex7c': f'pnl Ex7 20 premium less {A} {OCC} poisson {AGG} less 10% premium expense',
    'ex7td': f'xpnl Ex7 20 premium less {A} {OCC} poisson {AGG} less 10% premium expense '
             'peel top-down',
}

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


def record(case, pnl):
    # A consolidated face's source is aggregate's net marginal from its 1-D
    # engine; prospicio builds it from the joint, and the two agree to about
    # 2e-12 in the standard deviations.
    tol = 1e-11 if case.endswith('c') else 1e-12
    for idx, r in pnl.economic_df.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            col = col.replace('κ', 'k')
            add(case, 'sheet', label, col, v, 1e-10 if col == 'Skew' else tol)
    for idx, r in pnl.summary_df.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v, 1e-10 if col == 'Skew' else tol)
    for _, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX)


sources = []
for case, prog in CASES.items():
    pnl = build(prog, log2=6, bs=1)
    record(case, pnl)
    p = np.asarray(pnl._probs, dtype=float)
    src = pnl.source
    if not hasattr(src, 'axis1'):
        continue  # a consolidated face's net marginal; prospicio builds it from the joint
    a0, a1 = np.asarray(src.axis0, dtype=float), np.asarray(src.axis1, dtype=float)
    x = np.repeat(a0, len(a1))
    y = np.tile(a1, len(a0))
    keep = p > 0
    sources += [f'{case},{u!r},{v!r},{q!r}'
                for u, v, q in zip(x[keep].tolist(), y[keep].tolist(), p[keep].tolist())]
    if case == 'ex4':
        d = np.asarray(src.density, dtype=float)
        add('joint', 'shape', '', 'rows', d.shape[0])
        add('joint', 'shape', '', 'cols', d.shape[1])
        for i, j in zip(*np.nonzero(d)):
            add('joint', 'density', f'{i}|{j}', 'p', d[i, j], 1e-15)

(ref / 'pnl_occurrence_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
(ref / 'pnl_occurrence_sources.csv').write_text(
    'source,gross,ceded,prob\n' + '\n'.join(sources) + '\n')
