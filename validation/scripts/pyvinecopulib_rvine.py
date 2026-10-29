"""Reference values for R-vine copulas and vine fitting, from pyvinecopulib
1.0.1 (docs/design/risk.md, vine copulas; #568, second part).

- `density` and `inverse_rosenblatt`: a five-dimensional R-vine that is
  neither a C- nor a D-vine, in full and truncated after two trees, with
  rotated, asymmetric and Student t pairs, at a few points;
- `fit_structure`: the vine's own matrix fitted tree by tree to data
  sampled from it (`Vinecop.from_data(u, controls, matrix=...)`), each
  edge's family, rotation and parameters and the log-likelihood;
- `select`: the structure chosen by Dissmann's algorithm on the same data
  (`Vinecop.from_data(u, controls)`), each edge as its conditioned pair
  and conditioning set with its pair copula, and the log-likelihood.

Fits use independence and the six families, AIC and no family
preselection, so every family is tried on every edge. The data sets are in
`validation/reference/vine_rvine_data.csv`, one row an observation.

    uv run --no-project --python 3.12 --with pyvinecopulib==1.0.1 \\
        python validation/scripts/pyvinecopulib_rvine.py

writes validation/reference/vine_rvine_pyvinecopulib.csv and
validation/reference/vine_rvine_data.csv.
"""

import csv
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

OUT = Path(__file__).resolve().parents[1] / "reference"
F = pv.BicopFamily
NAMES = {
    F.indep: "independence",
    F.gaussian: "gaussian",
    F.student: "student",
    F.clayton: "clayton",
    F.gumbel: "gumbel",
    F.frank: "frank",
    F.joe: "joe",
}

# A structure that is neither a C- nor a D-vine: tree 1 is 1-2, 2-3, 2-4,
# 4-5. Each edge (a, b, conditioning) with its pair copula, first argument a.
TREES = [
    [((1, 2, []), (F.gumbel, 0, [2.5])), ((2, 3, []), (F.clayton, 0, [3.0])),
     ((2, 4, []), (F.student, 0, [0.7, 4.0])), ((4, 5, []), (F.frank, 0, [8.0]))],
    [((1, 3, [2]), (F.gaussian, 0, [0.4])), ((3, 4, [2]), (F.joe, 90, [1.6])),
     ((2, 5, [4]), (F.clayton, 180, [1.2]))],
    [((1, 4, [2, 3]), (F.gaussian, 0, [-0.3])), ((3, 5, [2, 4]), (F.frank, 0, [2.0]))],
    [((1, 5, [2, 3, 4]), (F.gaussian, 0, [0.2]))],
]
MATRIX = pv.RVineStructure.from_trees(5, [[e for e, _ in tree] for tree in TREES]).matrix


def pairs():
    """The pair copulas in the matrix's places, transposed where the
    matrix puts an edge's second variable on the diagonal."""
    d = 5
    out = []
    for t in range(4):
        tree = []
        for j in range(d - 1 - t):
            diag, other = int(MATRIX[d - 1 - j, j]), int(MATRIX[t, j])
            given = sorted(int(MATRIX[i, j]) for i in range(t))
            for (a, b, c), (fam, rot, par) in TREES[t]:
                if sorted(c) == given and {a, b} == {diag, other}:
                    if a != diag:
                        rot = {90: 270, 270: 90}.get(rot, rot)
                    tree.append(pv.Bicop(family=fam, rotation=rot, parameters=np.array(par).reshape(-1, 1)))
        assert len(tree) == d - 1 - t
        out.append(tree)
    return out


def matrix_text(m):
    return ";".join(" ".join(str(int(x)) for x in row) for row in m)


def pair_text(c):
    params = [float(x) for x in np.asarray(c.parameters).ravel()]
    return ":".join([NAMES[c.family], str(int(c.rotation))] + [repr(p) for p in params])


def pairs_text(pcs):
    return "/".join(";".join(pair_text(c) for c in tree) for tree in pcs)


def nums(x):
    return ";".join(repr(float(v)) for v in np.asarray(x).ravel())


def main():
    rows = []
    points = np.array(
        [
            [0.2, 0.7, 0.4, 0.9, 0.35],
            [0.55, 0.1, 0.8, 0.3, 0.6],
            [0.95, 0.9, 0.85, 0.99, 0.97],
            [0.05, 0.12, 0.02, 0.08, 0.1],
        ]
    )
    for trunc in (4, 2):
        pcs = pairs()[:trunc]
        m = MATRIX.copy()
        for t in range(trunc, 4):
            for j in range(4 - t):
                m[t, j] = 0
        v = pv.Vinecop.from_structure(matrix=m, pair_copulas=pcs)
        for p in points:
            u = p.reshape(1, -1)
            rows.append(dict(case="density", matrix=matrix_text(m), pairs=pairs_text(pcs),
                             data=nums(p), expected=nums(v.loglik(u))))
            rows.append(dict(case="inverse_rosenblatt", matrix=matrix_text(m), pairs=pairs_text(pcs),
                             data=nums(p), expected=nums(v.inverse_rosenblatt(u))))
    controls = pv.FitControlsVinecop(
        family_set=[F.indep, F.gaussian, F.student, F.clayton, F.gumbel, F.frank, F.joe],
        selection_criterion="aic",
        preselect_families=False,
    )
    data_rows = []
    for k, (n, seed) in enumerate([(800, 5), (800, 17)]):
        truth = pv.Vinecop.from_structure(matrix=MATRIX, pair_copulas=pairs())
        u = truth.sample(n, seeds=[seed, seed + 1, seed + 2, seed + 3, seed + 4])
        for i, r in enumerate(u):
            data_rows.append(dict(set=k, obs=i, **{f"u{j + 1}": repr(float(x)) for j, x in enumerate(r)}))
        own = pv.Vinecop.from_data(u, controls, matrix=MATRIX)
        rows.append(dict(case="fit_structure", matrix=matrix_text(own.matrix), pairs=pairs_text(own.pair_copulas),
                         data=str(k), expected=repr(float(own.loglik(u)))))
        sel = pv.Vinecop.from_data(u, controls)
        edges = []
        for tree in sel.get_trees():
            for e in tree:
                a, b = e["conditioned"]
                given = " ".join(str(x) for x in sorted(e["conditioning"]))
                edges.append(f"{a} {b}|{given}|{pair_text(e['pair_copula'])}")
        rows.append(dict(case="select", matrix=matrix_text(sel.matrix), pairs="/".join(edges),
                         data=str(k), expected=repr(float(sel.loglik(u)))))
    with open(OUT / "vine_rvine_pyvinecopulib.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "matrix", "pairs", "data", "expected"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "vine_rvine_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "obs"] + [f"u{j}" for j in range(1, 6)])
        w.writeheader()
        w.writerows(data_rows)


if __name__ == "__main__":
    main()
