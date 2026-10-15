"""Reference values for pair copulas and C- and D-vines, from pyvinecopulib
1.0.1 (docs/design/risk.md, vine copulas).

- `pair`: each family and rotation's cdf, density, `h2(u | v) = dC/dv`
  (pyvinecopulib's `hfunc2`), `h1(v | u) = dC/du` (`hfunc1`) and their
  inverses (`hinv2`, `hinv1`) at a few points;
- `dvine`, `cvine`: a three-dimensional vine's log-density and inverse
  Rosenblatt transform with asymmetric (rotated) pairs.

    uv run --no-project --python 3.12 --with pyvinecopulib==1.0.1 \\
        python validation/scripts/pyvinecopulib_vines.py

writes validation/reference/vine_pyvinecopulib.csv.
"""

from pathlib import Path

import numpy as np
import pyvinecopulib as pv

F = pv.families
SOURCE = 'pyvinecopulib 1.0.1'
ref = Path(__file__).resolve().parents[1] / 'reference'

PAIRS = [
    ('gaussian', F.gaussian, 0, 0.6), ('gaussian', F.gaussian, 0, -0.4),
    ('frank', F.frank, 0, 4.0), ('frank', F.frank, 0, -3.0),
]
for name, fam in [('clayton', F.clayton), ('gumbel', F.gumbel), ('joe', F.joe)]:
    for rot in (0, 90, 180, 270):
        PAIRS.append((name, fam, rot, 2.5))
POINTS = [(0.3, 0.6), (0.8, 0.2), (0.55, 0.45), (0.1, 0.9), (0.02, 0.97)]

rows = []


def add(case, family, rotation, parameter, quantity, u, expected, rel_tol=1e-9, abs_tol=1e-12):
    rows.append(f'{case},{family},{rotation},{parameter!r},{quantity},{u},'
                f'{float(expected)!r},{abs_tol},{rel_tol},{SOURCE}')


for name, fam, rot, par in PAIRS:
    c = pv.Bicop(family=fam, rotation=rot, parameters=np.array([[par]]))
    for u, v in POINTS:
        x = np.array([[u, v]])
        pt = f'{u!r};{v!r}'
        add('pair', name, rot, par, 'cdf', pt, c.cdf(x)[0])
        add('pair', name, rot, par, 'log_pdf', pt, np.log(c.pdf(x)[0]), 1e-9, 1e-10)
        add('pair', name, rot, par, 'h2', pt, c.hfunc2(x)[0])
        add('pair', name, rot, par, 'h1', pt, c.hfunc1(x)[0])
        # Inverses at p = u given v (hinv2) and p = v given u (hinv1).
        add('pair', name, rot, par, 'h2_inverse', pt, c.hinv2(x)[0], 1e-7, 1e-9)
        add('pair', name, rot, par, 'h1_inverse', pt, c.hinv1(x)[0], 1e-7, 1e-9)

# Three-dimensional vines in prospicio's order (1, 2, 0), 0-based, each
# edge's first argument its earlier variable: tree 1 joins (1, 2) and (2, 0)
# for the D-vine, (1, 2) and (1, 0) for the C-vine. pyvinecopulib's D-vine
# order (2, 3, 1), 1-based, has the same edges and arguments. Its C-vine
# puts the root last in the order and as each edge's second argument, so
# the same C-vine is its order (1, 3, 2) with the pairs transposed (90 and
# 270 swapped) and listed by the other variable: (1, 2), (3, 2), (1, 3 | 2).
# Its inverse Rosenblatt transform runs through the order backwards.


def bicop(fam, rot, par):
    return pv.Bicop(family=fam, rotation=rot, parameters=np.array([[par]]))


D = [[bicop(F.gumbel, 0, 2.0), bicop(F.clayton, 90, 1.5)], [bicop(F.joe, 270, 1.8)]]
C = [[bicop(F.clayton, 270, 1.5), bicop(F.gumbel, 0, 2.0)], [bicop(F.joe, 90, 1.8)]]
X = np.array([[0.2, 0.7, 0.4], [0.5, 0.5, 0.5], [0.9, 0.15, 0.6], [0.05, 0.3, 0.8]])
for case, structure, pairs in [('dvine', pv.DVineStructure(order=[2, 3, 1]), D),
                               ('cvine', pv.CVineStructure(order=[1, 3, 2]), C)]:
    v = pv.Vinecop.from_structure(structure=structure, pair_copulas=pairs)
    for x in X:
        pt = ';'.join(repr(float(a)) for a in x)
        add(case, '', '', '', 'log_pdf', pt, np.log(v.pdf(x[None, :])[0]), 1e-9, 1e-10)
        r = v.inverse_rosenblatt(x[None, :])[0]
        for j in range(3):
            add(case, '', '', '', f'inverse_rosenblatt_{j}', pt, r[j], 1e-7, 1e-9)

(ref / 'vine_pyvinecopulib.csv').write_text(
    'case,family,rotation,parameter,quantity,u,expected,abs_tol,rel_tol,source\n'
    + '\n'.join(rows) + '\n')
