"""Reference values for variable terms in the P&L ledger and for
`pnl_program`'s premium ladder, from `aggregate` 1.0.1
(docs/design/pnl.md, PR 6a).

- `swing`, `slide`, `pc`, `corr` (and `...c`, the consolidated face): an
  aggregate cover 3 xs 5 on example 1's book, swing rated, with a sliding
  scale, a profit commission or a loss-ratio corridor;
- `half`, `halfc`: the profit commission on half of the layer placed;
- `occsw`, `occswc`: the swing-rated cover behind example 4's occurrence
  cover, over the joint source (validation/reference/pnl_occurrence_sources.csv);
- `retro`: a retrospectively rated book (consolidated only, as aggregate);
- `program`: `pnl_program`'s ladder, the inputs (expected net and ceded
  losses) read off aggregate's engines and the premium and deposits it
  writes.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_variable.py

writes validation/reference/pnl_variable_aggregate.csv.
"""

from pathlib import Path

import numpy as np
from aggregate import build
from aggregate._program import _pnl_technical, _require_program

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

A = 'agg ExVar 2 claims dsev [1 2 4] [.5 .3 .2] fixed'
B = 'agg ExOcc 2 claims dsev [1 2 4] [.5 .3 .2]'
EXP = 'less 10% premium expense'
SWING = 'swing basic 0.5 lcm 1.1 min 0.6 max 2.5'
COVERS = {
    'swing': f'aggregate net of 3 xs 5 {SWING}',
    'slide': 'aggregate net of 3 xs 5 deposit 1 slide 0.3 at 0.2 and 0.1 at 0.6',
    'pc': 'aggregate net of 3 xs 5 deposit 1 pc 0.25 after 0.1',
    'corr': 'aggregate net of 3 xs 5 deposit 1 corridor 0.5 po 0.3 xs 0.2',
    'half': 'aggregate net of 1.5 po 3 xs 5 deposit 1 pc 0.25 after 0.1',
}
CASES = {}
for key, cover in COVERS.items():
    CASES[key] = f'xpnl V 10 premium less {A} {cover} {EXP}'
    CASES[f'{key}c'] = f'pnl V 10 premium less {A} {cover} {EXP}'
OCC = f'{B} occurrence net of 2 xs 2 deposit 2 cede 25% poisson aggregate net of 3 xs 5 {SWING}'
CASES['occsw'] = f'xpnl V 20 premium less {OCC} {EXP}'
CASES['occswc'] = f'pnl V 20 premium less {OCC} {EXP}'
CASES['retro'] = f'pnl V retro basic 2 lcm 1.1 min 3 max 9 premium less {A} {EXP}'

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


def record(case, pnl, tol):
    for idx, r in pnl.economic_df.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            col = col.replace('κ', 'k')
            add(case, 'sheet', label, col, v, max(tol, 1e-10) if col == 'Skew' else tol)
    for idx, r in pnl.summary_df.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v, max(tol, 1e-10) if col == 'Skew' else tol)
    for _, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX, tol)


for case, prog in CASES.items():
    pnl = build(prog, log2=6, bs=1)
    # The consolidated face behind an occurrence cover reads aggregate's
    # 1-D net-of-occurrence engine; prospicio builds it from the joint.
    record(case, pnl, 1e-11 if case == 'occswc' else 1e-12)

# pnl_program's ladder: T_net = E_net / c_net, Q_j = E_j / c_j, deposits
# d_j = round(Q_j / s_j), P = round((T_net + sum s_j d_j) / (1 - ER)).
PROGRAMS = {
    'note': ('agg Prog 10 claims sev lognorm 50 cv 1.5 '
             'occurrence net of 200 xs 100 poisson aggregate net of 500 xs 1000',
             13, dict(expense_ratio=0.25, net_combined_ratio=0.9,
                      occ_combined_ratio=0.6, agg_combined_ratio=0.5)),
    'half': ('agg ProgH 10 claims sev lognorm 50 cv 1.5 '
             'occurrence net of 100 po 200 xs 100 poisson',
             13, dict(expense_ratio=0.3, net_combined_ratio=0.8,
                      occ_combined_ratio=0.55, agg_combined_ratio=None)),
}
for key, (prog, log2, kw) in PROGRAMS.items():
    ob = build(prog, log2=log2, bs=1 / 8)
    _kind, _name, spec = _require_program(ob, 'x')
    stats = ob.reins_stats_df
    add('program', 'input', key, 'net_loss', float(ob.est_m))
    add('program', 'input', key, 'expense_ratio', kw['expense_ratio'])
    add('program', 'input', key, 'net_combined_ratio', kw['net_combined_ratio'])
    booked, occ, agg = _pnl_technical(ob, spec, kw['expense_ratio'],
                                      kw['net_combined_ratio'],
                                      kw['occ_combined_ratio'],
                                      kw['agg_combined_ratio'], 'x')
    j = 0
    for tier, k, entries in (('occurrence', 'occ', occ), ('aggregate', 'agg', agg)):
        ratio = kw[f'{k}_combined_ratio'] or kw['net_combined_ratio']
        for i, (share, _limit, _attach) in enumerate(spec.get(f'{k}_reins') or []):
            ceded = float(stats.loc[('agg', 'mean'), (k, f'layer.{i + 1}')])
            add('program', 'input', f'{key}|{j}', 'ceded_loss', ceded)
            add('program', 'input', f'{key}|{j}', 'share', share)
            add('program', 'input', f'{key}|{j}', 'combined_ratio', ratio)
            add('program', 'deposit', f'{key}|{j}', 'deposit', entries[i][1], 0.0)
            j += 1
    add('program', 'premium', key, 'premium', booked, 0.0)

(ref / 'pnl_variable_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
