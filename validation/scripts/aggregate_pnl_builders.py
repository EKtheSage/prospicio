"""Reference values for the P&L builders over annual covers, from `aggregate`
1.0.1 (docs/design/pnl.md, PR 4).

- `ex3a`, `ex3b`: the walk (`xpnl`) and the consolidated ledger (`pnl`)
  of a book with an aggregate cover 3 xs 5 (deposit 1, a 10% ceding
  commission) and a 10% premium expense;
- `ex6w`, `ex6td`, `ex6bu`: two aggregate layers (2 xs 3 for 0.5, 3 xs 5
  for 1 with a 10% commission) as one walk step and peeled top down and
  bottom up;
- `ex1`, `ex1w`: the plain book of example 1 consolidated and as a
  one-step walk; `exp1` to `exp3`: expense groups and their names;
  `lbl`: premium and loss labels on the walk and the consolidated face,
  with a loss-basis expense;
- `ex5`: a continuous book (lognormal severity, FFT), its source written
  to validation/reference/pnl_builder_sources.csv;
- `derive`: the gross premium that leaves a technical premium after its
  expenses.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_builders.py

writes validation/reference/pnl_builders_aggregate.csv and
validation/reference/pnl_builder_sources.csv.
"""

from pathlib import Path

import numpy as np
from aggregate import build
from aggregate._pnl_builders import (
    build_consolidated_pnl,
    build_xpnl_walk,
    derive_consideration,
)

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

A = 'agg ExGross 2 claims dsev [1 2 4] [.5 .3 .2] fixed'
COVER = 'aggregate net of 3 xs 5 deposit 1 cede 10%'
TWO = 'aggregate net of 2 xs 3 deposit 0.5 as Low and 3 xs 5 deposit 1 cede 10% as High'
CASES = {
    'ex3a': f'xpnl Ex3 10 premium less {A} {COVER} less 10% premium expense',
    'ex3b': f'pnl Ex3 10 premium less {A} {COVER} less 10% premium expense',
    'ex6w': f'xpnl Ex6 10 premium less {A} {TWO} less 10% premium expense',
    'ex6td': f'xpnl Ex6 10 premium less {A} {TWO} less 10% premium expense peel top-down',
    'ex6bu': f'xpnl Ex6 10 premium less {A} {TWO} less 10% premium expense peel bottom-up',
    'ex1': f'pnl Ex1 10 premium less {A} less 10% premium expense',
    'ex1w': f'xpnl Ex1 10 premium less {A} less 10% premium expense',
    'exp1': f'pnl E1 10 premium less {A} less 10% premium expense 5% loss expense '
            '1 fixed expense',
    'exp2': f'pnl E2 10 premium less {A} less 10% premium expense and 5% loss expense',
    'exp3': f'pnl E3 10 premium less {A} less 10% premium expense as Acq '
            '5% loss expense as LAE',
    'ex5': 'pnl Ex5 1000 premium less agg Ex5e 10 claims sev lognorm 50 cv 1.5 poisson '
           'less 5% loss expense as LAE 25% premium expense as Acq',
}
GRID = {'ex5': dict(log2=12, bs=1)}

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


def record(case, pnl, abs_tol=1e-12, rel_tol=0.0):
    for idx, r in pnl.economic_df.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            col = col.replace('κ', 'k')
            add(case, 'sheet', label, col, v,
                max(abs_tol, 1e-10) if col == 'Skew' else abs_tol, rel_tol)
    for idx, r in pnl.summary_df.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v,
                max(abs_tol, 1e-10) if col == 'Skew' else abs_tol, rel_tol)
    for _, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX, abs_tol, rel_tol)


pnls = {}
for case, prog in CASES.items():
    pnl = build(prog, **GRID.get(case, dict(log2=6, bs=1)))
    pnls[case] = pnl
    # The FFT book's source has 4,096 atoms; its values agree to rounding.
    record(case, pnl, *((1e-9, 1e-12) if case == 'ex5' else ()))

# Premium and loss labels, with a loss-basis expense, on both faces.
engine = pnls['ex3a'].engine
econ = pnls['ex3a'].economics
spec = [('premium', 0.1), ('loss', 0.05)]
record('lblw', build_xpnl_walk(engine, gross=10.0, ceded=1.0, gcn_economics=econ,
                               expense_spec=spec, consideration_label='GWP',
                               loss_label='Claims', name='Lbl', label='Book'))
record('lblc', build_consolidated_pnl(engine, gross=10.0, ceded=1.0, gcn_economics=econ,
                                      expense_spec=spec, consideration_label='GWP',
                                      loss_label='Claims', name='Lbl'))

DERIVE = {
    'none': ([], 5.0),
    'premium': ([('premium', 0.1)], 5.0),
    'mixed': ([('premium', 0.1), ('fixed', 1.0)], 5.0),
    'groups': ([(None, [('premium', 0.1)]), ('Acq', [('premium', 0.15), ('fixed', 2.0)])], 5.0),
}
for key, (spec, t) in DERIVE.items():
    add('derive', 'premium', key, 'P', derive_consideration(spec, t, 'x'))

(ref / 'pnl_builders_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')

# The gross sources: the engine's density, every atom with mass.
lines = []
for case in ('ex3a', 'ex5'):
    # The probabilities PnL uses: the engine's grid normalized over all its
    # points (a sum over the atoms with mass alone can differ in the last
    # bit, and move a cumulative that lands on a ladder level).
    x = np.asarray(pnls[case].source.x, dtype=float)
    p = np.asarray(pnls[case]._probs, dtype=float)
    keep = p > 0
    lines += [f'{case},{v!r},{q!r}' for v, q in zip(x[keep].tolist(), p[keep].tolist())]
(ref / 'pnl_builder_sources.csv').write_text('source,value,prob\n' + '\n'.join(lines) + '\n')
