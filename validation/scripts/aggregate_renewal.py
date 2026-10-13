"""Reference values for renewal claim counts, from `aggregate` 1.0.1.

The count N(T) of a Sparre-Andersen (renewal) process by horizon T, for the
renewal cases in `aggregate`'s own test suite (`agg/_test_suite.agg`, the
`Y.Renewal.*` lines; the unconditional splice is left out). For each case
the wait grid `aggregate` chose is written next to the count, so a test can
check both the sizing and the count.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_renewal.py

writes validation/reference/renewal_count_aggregate.csv.
"""

from pathlib import Path

import scipy.stats as ss
from aggregate import build

CASES = {
    'expon': (10.0, 'sev lognorm 100 cv 1 wait expon'),
    'gamma': (5.0, 'sev lognorm 10 cv .3 wait gamma 2'),
    'determin': (3.0, 'dsev [1] dwait [1]'),
    'cluster': (2.0, 'dsev [1] dwait [0 1] [.5 .5]'),
    'defective': (2.0, 'dsev [1] dwait [1 2] [.4 .5] !'),
    'mixture': (5.0, 'sev gamma 10 cv .3 wait [.6 .4] * expon wts [.5 .5]'),
}

ref = Path(__file__).resolve().parents[1] / 'reference'
SOURCE = 'aggregate 1.0.1'
rows = []
for case, (years, spec) in CASES.items():
    a = build(f'agg Renewal.{case} {years} years {spec}', update=False)
    f = a.frequency
    grid = [('step', f.wait_bs, 0.0, 1e-14), ('points', 2 ** f.wait_log2, 0.0, 0.0),
            ('closed', int(bool(f.wait_lattice)), 0.0, 0.0),
            ('p0', f.wait_p0, 1e-15, 0.0), ('defect', float(f.wait_defect), 1e-15, 0.0),
            ('kmax', f.kmax, 0.0, 0.0)]
    rows += [f'{case},{years!r},{q},,{float(v)!r},{a_},{r_},{SOURCE}'
             for q, v, a_, r_ in grid]
    rows += [f'{case},{years!r},p,{k},{float(v)!r},1e-8,0.0,{SOURCE}'
             for k, v in enumerate(f.freq_df.p.to_numpy())]

# Exact values: exponential waits give a Poisson count, and N(T) = 0 is
# the first wait beyond T. aggregate is within 6e-9 of them at k = 0 and 1
# (the round-off of its tilted sums), prospicio within 5e-11.
exact = [('expon', 10.0, k, ss.poisson(10.0).pmf(k)) for k in range(40)]
exact += [('gamma', 5.0, 0, ss.gamma(2.0).sf(5.0)),
          ('mixture', 5.0, 0, 0.5 * ss.expon(scale=0.6).sf(5.0)
           + 0.5 * ss.expon(scale=0.4).sf(5.0))]
rows += [f'{case},{years!r},p,{k},{float(v)!r},1e-9,0.0,exact'
         for case, years, k, v in exact]

(ref / 'renewal_count_aggregate.csv').write_text(
    'case,horizon,quantity,arg,expected,abs_tol,rel_tol,source\n'
    + '\n'.join(rows) + '\n')
