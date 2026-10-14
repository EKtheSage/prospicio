"""Reference values for the P&L ledger kernel, from `aggregate` 1.0.1.

`aggregate`'s `PnL` built by hand from `Leg` and `Group` over a
one-dimensional source (docs/design/pnl.md, PR 2): every row's moments and
scenario (kappa) columns (`economic_df`), each row's own quantiles
(`economic_marginal_df`), the card (`summary_df`) and `prob_eq_0`. Cases:

- `ex1`: the design note's example 1, a seller over the loss of two claims
  of 1, 2 or 4 (aggregate's own grid of it);
- `ex2`: example 2, a seller and a buyer on four outcomes;
- `ex2c`: example 2c, a premium that is not monotone in the loss;
- `ex3`: example 3a by hand, the seller and an aggregate cover 10% of
  3 xs 5 bought for 1, on example 1's source;
- `tiers`: three groups with a tier subtotal and a net-of block.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_pnl_ledger.py

writes validation/reference/pnl_ledger_aggregate.csv and the sources to
validation/reference/pnl_ledger_sources.csv.
"""

from pathlib import Path

import numpy as np
from aggregate import Group, Leg, PnL, build

SOURCE = 'aggregate 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

a = build('agg PnLSource 2 claims dsev [1 2 4] [.5 .3 .2] fixed', log2=6, bs=1, update=True)
dd = a.density_df
mask = dd.p_total.to_numpy() > 0
loss_x = dd.loss.to_numpy()[mask]
loss_p = dd.p_total.to_numpy()[mask]
hand_x = np.array([0.0, 10.0, 20.0, 50.0])
hand_p = np.array([0.4, 0.3, 0.2, 0.1])
SOURCES = {'loss': (loss_x, loss_p), 'hand': (hand_x, hand_p)}


def seller(prem=10.0, expense=1.0):
    return Group('Gross', 'sell', [Leg('Premium', prem, kind='premium')],
                 [Leg('Loss', lambda x: x, kind='loss'),
                  Leg('expense', expense, kind='expense')])


CASES = {
    'ex1': ('loss', dict(name='Ex1', role='sell',
                         consideration=[Leg('Premium', 10.0, kind='premium')],
                         obligation=[Leg('Loss', lambda x: x, kind='loss'),
                                     Leg('expense', 1.0, kind='expense')],
                         result_name='margin')),
    'ex2': ('hand', dict(name='Ex2', groups=[
        Group('Gross', 'sell',
              [Leg('Premium', lambda x: np.clip(5 + 1.1 * x, 12, 40), kind='premium')],
              [Leg('Loss', lambda x: x, kind='loss'),
               Leg('LAE', lambda x: 0.05 * x, kind='expense')]),
        Group('XL', 'buy', [Leg('XL premium', 4.0, kind='premium')],
              [Leg('XL recovery', lambda x: np.clip(x - 15, 0, 25), kind='recovery'),
               Leg('XL commission', 0.4, kind='commission')])],
        result_name='margin')),
    'ex2c': ('hand', dict(name='Ex2c', role='sell',
                          consideration=[Leg('Prem', lambda x: np.where(x == 20, 30.0, 10.0))],
                          obligation=[Leg('Loss', lambda x: x)])),
    'ex3': ('loss', dict(name='Ex3', groups=[
        seller(),
        Group('agg 3 xs 5', 'buy', [Leg('agg premium', 1.0, kind='premium')],
              [Leg('agg recovery', lambda x: 0.1 * np.clip(x - 5, 0, 3), kind='recovery')])],
        result_name='margin')),
    'tiers': ('hand', dict(name='Tiers', groups=[
        Group('Gross', 'sell', [Leg('Premium', 30.0)], [Leg('Loss', lambda x: x)],
              margin_label='Direct'),
        Group('XL1', 'buy', [Leg('XL1 premium', 3.0)],
              [Leg('XL1 recovery', lambda x: np.clip(x - 10, 0, 10))]),
        Group('XL2', 'buy', [Leg('XL2 premium', 2.0)],
              [Leg('XL2 recovery', lambda x: np.clip(x - 20, 0, 30))])],
        tier_spans=(('All XL', 1, 3), ('All gross and XL1', 0, 2)))),
}

rows = []


def add(case, quantity, row, col, value, abs_tol=1e-12, rel_tol=0.0):
    v = float(value)
    if np.isnan(v):
        return
    rows.append(f'{case},{quantity},{row},{col},{v!r},{abs_tol},{rel_tol},{SOURCE}')


for case, (src, kw) in CASES.items():
    x, p = SOURCES[src]
    pnl = PnL(source=(x, p), **kw)
    sheet = pnl.economic_df
    for idx, r in sheet.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            col = col.replace('κ', 'k')
            add(case, 'sheet', label, col, v, 1e-10 if col == 'Skew' else 1e-12)
    for idx, r in pnl.economic_marginal_df.iterrows():
        label = '|'.join(idx)
        for col, v in r.items():
            if col.startswith('P'):
                add(case, 'marginal', label, col, v)
    card = pnl.summary_df
    for idx, r in card.iterrows():
        label = '|'.join(idx) if isinstance(idx, tuple) else idx
        for col, v in r.items():
            add(case, 'card', label, col, v, 1e-10 if col == 'Skew' else 1e-12)
    add(case, 'prob_eq_0', '', '', pnl.prob_eq_0)
    for i, r in pnl.legs_df.iterrows():
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'EX', r.EX)
        add(case, 'legs', f'{r.Step}|{r.Side}|{r.Label}', 'SD', r.SD)

(ref / 'pnl_ledger_aggregate.csv').write_text(
    'case,quantity,row,col,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
# The probabilities PnL uses: normalized once, as PnL does.
(ref / 'pnl_ledger_sources.csv').write_text(
    'source,value,prob\n' + '\n'.join(
        f'{name},{v!r},{q!r}' for name, (xs, ps) in SOURCES.items()
        for v, q in zip(xs.tolist(), (ps / ps.sum()).tolist())) + '\n')
