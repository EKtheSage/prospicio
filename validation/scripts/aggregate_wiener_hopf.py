"""Reference values for eventual ruin with renewal claims, from `aggregate` 1.0.1.

`Aggregate.wiener_hopf(rho)`: the chance of eventual ruin when the waits
between claims follow any law (Sparre Andersen), by the cepstral
Wiener-Hopf factorization of the claim walk `X - cW`, with the premium
rate `c = (1 + rho) E[X] / E[W]`. Severities on a step of 1/32 over 2^16
points; psi on the same grid.

    uv run --python 3.12 --with aggregate==1.0.1 \
        python validation/scripts/aggregate_wiener_hopf.py

writes validation/reference/wiener_hopf_aggregate.csv.
"""

from pathlib import Path

from aggregate import build

CASES = {
    'expon': 'sev gamma 10 cv .3 wait expon',
    'gamma': 'sev gamma 10 cv .3 wait gamma 2',
    'mixture': 'sev gamma 10 cv .3 wait [.6 .4] * expon wts [.5 .5]',
    'heavy': 'sev lognorm 10 cv 1 wait gamma 0.5',
}
BS, LOG2 = 1 / 32, 16
SOURCE = 'aggregate 1.0.1'

ref = Path(__file__).resolve().parents[1] / 'reference'
rows = []
for case, spec in CASES.items():
    a = build(f'agg WH.{case} 5 years {spec}', update=False)
    a.update(log2=LOG2, bs=BS)
    for rho in (0.1, 0.3):
        r = a.wiener_hopf(rho)
        rows.append(f'{case},{rho},mean,,{r.mean!r},0.0,1e-12,{SOURCE}')
        for u in (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
            rows.append(f'{case},{rho},psi,{u},{float(r.ruin.loc[u])!r},1e-10,0.0,{SOURCE}')
        for p in (0.1, 0.01, 0.001):
            rows.append(f'{case},{rho},u_index,{p},{float(r.find_u(p))!r},0.0,0.0,{SOURCE}')
        rows.append(f'{case},{rho},max_pmf,0,{float(r.density[0])!r},1e-10,0.0,{SOURCE}')

(ref / 'wiener_hopf_aggregate.csv').write_text(
    'case,rho,quantity,arg,expected,abs_tol,rel_tol,source\n' + '\n'.join(rows) + '\n')
