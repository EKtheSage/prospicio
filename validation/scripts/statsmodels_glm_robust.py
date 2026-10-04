"""Regenerate validation/reference/glm_robust_statsmodels.csv.

    pip install numpy statsmodels
    python validation/scripts/statsmodels_glm_robust.py

Sandwich standard errors for every case of statsmodels_glm.py (same data,
design, offsets and weights): statsmodels' cov_type="HC0", and
cov_type="cluster" with clusters of five-year age bands (floor(age / 5))
and statsmodels' default small-sample correction. statsmodels builds the
sandwich from the observed information, which differs from the expected
for non-canonical links.
"""

import csv
import sys
import warnings

import numpy as np
import statsmodels
import statsmodels.api as sm

from statsmodels_glm import cases, data, design

OUT = "validation/reference/glm_robust_statsmodels.csv"
SOURCE = f"statsmodels {statsmodels.__version__}"
NAMES = ["(Intercept)", "age", "region[B]", "region[C]", "region[D]"]


def main():
    warnings.simplefilter("ignore")
    d = data()
    X = design(d)
    groups = np.floor(d["age"] / 5).astype(int)
    rows = []
    for name, y, fam, offset, vw, _ in cases(d):
        model = sm.GLM(y, X, family=fam, offset=offset, var_weights=vw)
        for kind, kw in [("hc0", {"cov_type": "HC0"}),
                         ("cluster", {"cov_type": "cluster", "cov_kwds": {"groups": groups}})]:
            res = model.fit(tol=1e-14, **kw)
            for j, nm in enumerate(NAMES):
                rows.append([name, kind, nm, res.bse[j]])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "kind", "term", "expected", "abs_tol", "rel_tol", "source"])
        for case, kind, term, value in rows:
            w.writerow([case, kind, term, repr(float(value)), 1e-12, 1e-7, SOURCE])
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
