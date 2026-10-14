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

# pnl_program's ladder: T_net = E_net / c_net (or the engine's stated
# premium), Q_j = E_j / c_j, deposits d_j = round(Q_j / s_j),
# P = round((T_net + sum s_j d_j) / (1 - ER)); premium_style='rate' respells
# each priced deposit as a rate of P (the aggregate tier's of P less the
# occurrence premiums), and a layer's own deposit clause is kept.
PROGRAMS = {
    'note': ('agg Prog 10 claims sev lognorm 50 cv 1.5 '
             'occurrence net of 200 xs 100 poisson aggregate net of 500 xs 1000',
             dict(expense_ratio=0.25, net_combined_ratio=0.9,
                  occ_combined_ratio=0.6, agg_combined_ratio=0.5)),
    'half': ('agg ProgH 10 claims sev lognorm 50 cv 1.5 '
             'occurrence net of 100 po 200 xs 100 poisson',
             dict(expense_ratio=0.3, net_combined_ratio=0.8,
                  occ_combined_ratio=0.55, agg_combined_ratio=None)),
    'stated': ('agg ProgS 1000 premium at 0.65 lr sev lognorm 50 cv 1.5 '
               'occurrence net of 200 xs 100 poisson aggregate net of 500 xs 1000',
               dict(expense_ratio=0.25, net_combined_ratio=0.9,
                    occ_combined_ratio=0.6, agg_combined_ratio=0.5)),
    'statedn': ('agg ProgN 1000 premium at 0.65 lr sev lognorm 50 cv 1.5 poisson',
                dict(expense_ratio=0.25, net_combined_ratio=0.9,
                     occ_combined_ratio=None, agg_combined_ratio=None)),
    'kept': ('agg ProgK 10 claims sev lognorm 50 cv 1.5 occurrence net of '
             '100 po 200 xs 100 deposit 40 and 300 xs 300 poisson',
             dict(expense_ratio=0.25, net_combined_ratio=0.9,
                  occ_combined_ratio=0.6, agg_combined_ratio=None)),
}
for key, (prog, kw) in PROGRAMS.items():
    ob = build(prog, log2=13, bs=1 / 8)
    _kind, _name, spec = _require_program(ob, 'x')
    stats = ob.reins_stats_df
    stated = float(np.sum(np.asarray(ob.exp_premium if ob.exp_premium is not None else 0.0)))
    if stated:
        add('program', 'input', key, 'stated', stated)
    else:
        add('program', 'input', key, 'net_loss', float(ob.est_m))
    add('program', 'input', key, 'expense_ratio', kw['expense_ratio'])
    add('program', 'input', key, 'net_combined_ratio', kw['net_combined_ratio'])
    out = {}
    for style in ('deposit', 'rate'):
        _, _, spec = _require_program(ob, 'x')
        out[style] = _pnl_technical(ob, spec, kw['expense_ratio'],
                                    kw['net_combined_ratio'],
                                    kw['occ_combined_ratio'],
                                    kw['agg_combined_ratio'], 'x',
                                    premium_style=style)
    booked, occ, agg = out['deposit']
    _, occ_r, agg_r = out['rate']
    j = 0
    for k, entries, rated in (('occ', occ, occ_r), ('agg', agg, agg_r)):
        ratio = kw[f'{k}_combined_ratio'] or kw['net_combined_ratio']
        existing = spec.get(f'{k}_reins_premium')
        for i, (share, _limit, _attach) in enumerate(spec.get(f'{k}_reins') or []):
            row = f'{key}|{j}'
            add('program', 'input', row, 'tier', 1.0 if k == 'agg' else 0.0)
            add('program', 'input', row, 'share', share)
            if existing is not None and existing[i] is not None:
                # a clause the author wrote: kept, at the placed share
                add('program', 'input', row, 'stated', share * existing[i][1])
            else:
                ceded = float(stats.loc[('agg', 'mean'), (k, f'layer.{i + 1}')])
                add('program', 'input', row, 'ceded_loss', ceded)
                add('program', 'input', row, 'combined_ratio', ratio)
                add('program', 'rate', row, 'rate', rated[i][1], 0.0)
            add('program', 'deposit', row, 'deposit', entries[i][1], 0.0)
            j += 1
    if isinstance(booked, float):
        add('program', 'premium', key, 'premium', booked, 0.0)
    else:
        # derive premium: the stated premium grossed up at build, unrounded
        add('program', 'premium', key, 'premium', stated / (1 - kw['expense_ratio']))

(ref / 'pnl_variable_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
