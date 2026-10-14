"""Reference values for a signed discrete distribution, from `aggregate` 1.0.1.

`GridDistribution` on irregular, possibly negative supports, as the P&L
ledger builds its rows (docs/design/pnl.md, PR 1): the lower and upper
quantiles, TVaR (the upper tail), the cdf, the moments as the ledger takes
them (raw second moment less the squared mean) and the return-period
table. Cases: the results of the design note's examples 1 and 2, a loss
grid with equal masses (aggregate's k / n branch), a signed sample, and a
lone atom.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_discrete.py

writes validation/reference/discrete_aggregate.csv and the random case's
atoms to validation/reference/discrete_aggregate_atoms.csv.
"""

from pathlib import Path

import numpy as np
from aggregate._aggregate import return_period_frame
from aggregate._grid_distribution import GridDistribution

rng = np.random.default_rng(20261014)
signed = np.round(rng.normal(0.0, 100.0, 40), 1)
CASES = {
    # The design note's example 1: the margin 9 - loss of two claims of
    # 1, 2 or 4.
    'ex1': ([1.0, 3.0, 4.0, 5.0, 6.0, 7.0], [0.04, 0.12, 0.2, 0.09, 0.3, 0.25]),
    # Example 2: the net result's atoms.
    'ex2': ([1.9, 7.4, 8.4, 8.9], [0.3, 0.2, 0.4, 0.1]),
    # Equal masses: aggregate reads the cumulative as k / n.
    'equal': ([-30.0, -10.0, 0.0, 5.0, 25.0, 60.0, 61.0], [1 / 7] * 7),
    # A signed sample with unequal masses.
    'signed': (np.unique(signed).tolist(),
               (lambda w: (w / w.sum()).tolist())(rng.uniform(0.5, 1.5, len(np.unique(signed))))),
    'single': ([-4.5], [1.0]),
}
LEVELS = [0.0, 0.001, 0.01, 0.04, 0.05, 0.1, 0.16, 0.25, 0.36, 0.5, 0.55, 0.75,
          0.9, 0.95, 0.99, 0.999, 1.0]
SOURCE = 'aggregate 1.0.1'

ref = Path(__file__).resolve().parents[1] / 'reference'
rows = []


def row(case, quantity, arg, value, abs_tol=1e-12, rel_tol=0.0):
    rows.append(f'{case},{quantity},{float(arg)!r},{float(value)!r},{abs_tol},{rel_tol},{SOURCE}')


for case, (x, p) in CASES.items():
    x, p = np.asarray(x, float), np.asarray(p, float)
    g = GridDistribution(x, p, is_loss_value=False)
    m = float((x * p).sum())
    var = max(float((x * x * p).sum()) - m * m, 0.0)
    sd = var ** 0.5
    row(case, 'mean', 0.0, m)
    row(case, 'sd', 0.0, sd)
    if sd > 0:
        row(case, 'skew', 0.0, float((((x - m) ** 3) * p).sum()) / sd ** 3, 1e-10)
    for v in sorted(set(x.tolist()) | {x.min() - 1, x.max() + 1, float(np.median(x)) + 0.25}):
        row(case, 'cdf', v, g.cdf(v))
    for lv in LEVELS:
        row(case, 'q_lower', lv, g.q(lv, 'lower'))
        qu = float(g.q(lv, 'upper'))
        if np.isfinite(qu):
            row(case, 'q_upper', lv, qu)
        # aggregate's tvar indexes F[-2], which a lone atom lacks.
        if lv < 1 and len(x) > 1:
            row(case, 'tvar', lv, g.tvar(lv), 1e-10)
    if len(x) == 1:
        continue
    tail = return_period_frame(g.q, g.tvar, m)
    for pp, r in tail.iterrows():
        row(case, 'tail_var', pp, r['VaR'])
        row(case, 'tail_tvar', pp, r['TVaR'], 1e-10)
        row(case, 'tail_period', pp, r['T'])

# The random case's atoms, for the test to rebuild it.
atoms = [f'signed,{v!r},{q!r}' for v, q in zip(*CASES['signed'])]
(ref / 'discrete_aggregate_atoms.csv').write_text('case,value,prob\n' + '\n'.join(atoms) + '\n')
(ref / 'discrete_aggregate.csv').write_text(
    'case,quantity,arg,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
