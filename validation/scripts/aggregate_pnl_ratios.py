"""Reference values for rebucketed legs, the ratio exhibit and the stack of
P&L ledgers, from `aggregate` 1.0.1 (docs/design/pnl.md, PR 3).

- `ex2b`: a seller with consideration 20 and the loss `1.03 x` rebucketed
  on a step of 4: the sheet, each row's own quantiles (the rebucketed one
  read off its signed grid), the card and `validation_df`;
- `economic_ratios_df` on the PR 2 ledgers with kinds (examples 1, 2 and
  3a) and on the three-group ledger without kinds (the unclassified
  fallback);
- `stack_marginal_pnls` of examples 2b and 2c with their difference.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_ratios.py

writes validation/reference/pnl_ratios_aggregate.csv; the sources are in
validation/reference/pnl_ledger_sources.csv (aggregate_pnl_ledger.py).
"""

from pathlib import Path

import numpy as np
from aggregate import Group, Leg, PnL, build, stack_marginal_pnls

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

a = build('agg PnLSource 2 claims dsev [1 2 4] [.5 .3 .2] fixed', log2=6, bs=1, update=True)
dd = a.density_df
mask = dd.p_total.to_numpy() > 0
loss = (dd.loss.to_numpy()[mask], dd.p_total.to_numpy()[mask])
hand = (np.array([0.0, 10.0, 20.0, 50.0]), np.array([0.4, 0.3, 0.2, 0.1]))


def seller(prem=10.0):
    return Group('Gross', 'sell', [Leg('Premium', prem, kind='premium')],
                 [Leg('Loss', lambda x: x, kind='loss'),
                  Leg('expense', 1.0, kind='expense')])


LEDGERS = {
    'ex1': PnL(name='Ex1', source=loss, role='sell',
               consideration=[Leg('Premium', 10.0, kind='premium')],
               obligation=[Leg('Loss', lambda x: x, kind='loss'),
                           Leg('expense', 1.0, kind='expense')],
               result_name='margin'),
    'ex2': PnL(name='Ex2', source=hand, groups=[
        Group('Gross', 'sell',
              [Leg('Premium', lambda x: np.clip(5 + 1.1 * x, 12, 40), kind='premium')],
              [Leg('Loss', lambda x: x, kind='loss'),
               Leg('LAE', lambda x: 0.05 * x, kind='expense')]),
        Group('XL', 'buy', [Leg('XL premium', 4.0, kind='premium')],
              [Leg('XL recovery', lambda x: np.clip(x - 15, 0, 25), kind='recovery'),
               Leg('XL commission', 0.4, kind='commission')])],
        result_name='margin'),
    'ex3': PnL(name='Ex3', source=loss, groups=[
        seller(),
        Group('agg 3 xs 5', 'buy', [Leg('agg premium', 1.0, kind='premium')],
              [Leg('agg recovery', lambda x: 0.1 * np.clip(x - 5, 0, 3), kind='recovery')])],
        result_name='margin'),
    'tiers': PnL(name='Tiers', source=hand, groups=[
        Group('Gross', 'sell', [Leg('Premium', 30.0)], [Leg('Loss', lambda x: x)],
              margin_label='Direct'),
        Group('XL1', 'buy', [Leg('XL1 premium', 3.0)],
              [Leg('XL1 recovery', lambda x: np.clip(x - 10, 0, 10))]),
        Group('XL2', 'buy', [Leg('XL2 premium', 2.0)],
              [Leg('XL2 recovery', lambda x: np.clip(x - 20, 0, 30))])],
        tier_spans=(('All XL', 1, 3), ('All gross and XL1', 0, 2))),
    'ex2b': PnL(name='Ex2b', role='sell', source=hand, consideration=20.0,
                obligation=[Leg('Loss', lambda x: x * 1.03, bs=4)]),
    'ex2c': PnL(name='Ex2c', role='sell', source=hand,
                consideration=[Leg('Prem', lambda x: np.where(x == 20, 30.0, 10.0))],
                obligation=[Leg('Loss', lambda x: x)]),
}

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},0.0,{SOURCE}')


for case in ('ex1', 'ex2', 'ex3', 'tiers'):
    for step, r in LEDGERS[case].economic_ratios_df.iterrows():
        for col, v in r.items():
            add(case, 'ratios', step, col, v)

pnl = LEDGERS['ex2b']
for quantity, frame in (('sheet', pnl.economic_df), ('marginal', pnl.economic_marginal_df)):
    for idx, r in frame.iterrows():
        for col, v in r.items():
            if quantity == 'marginal' and not col.startswith('P'):
                continue
            add('ex2b', quantity, '|'.join(idx), col.replace('κ', 'k'), v,
                1e-10 if col == 'Skew' else 1e-12)
for idx, r in pnl.summary_df.iterrows():
    for col, v in r.items():
        add('ex2b', 'card', idx, col, v, 1e-10 if col == 'Skew' else 1e-12)
for leg, r in pnl.validation_df.iterrows():
    for col, v in r.items():
        add('ex2b', 'validation', leg, col, v, 1e-14)
gd = pnl._rows['Loss'].gd
for x, p in zip(gd.x, gd.p):
    add('ex2b', 'grid', 'Loss', repr(float(x)), p, 1e-15)

stack = stack_marginal_pnls({'ex2b': LEDGERS['ex2b'], 'ex2c': LEDGERS['ex2c']},
                            impacts=[('c less b', 'ex2c', 'ex2b')])
for label, r in stack.iterrows():
    for col, v in r.items():
        add('stack', 'stack', label, col, v, 1e-10 if col == 'Skew' else 1e-12)

(ref / 'pnl_ratios_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
