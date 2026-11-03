"""Exact Kernel SHAP (#547): references for python/tests/test_models.py and
R/prospicio/tests/test-models.R from shap 0.54.1's KernelExplainer.

On validation/data/explain_frame.csv (written by r_ale.R) with the factor g
coded 0 to 3, the same prediction function as r_ale.R:
    exp(0.3 x1) + x1 x2 + (0, 0.5, -0.3, 1)[g] + 0.5 x3^2 + 0.4 x3 (g == 3)
the background is the first 40 rows and the rows explained are rows 41 to
48. With four features and nsamples = 2^4, KernelExplainer enumerates every
coalition, so its values are the exact Shapley values of the interventional
game; they are written with its expected value to
validation/reference/shap_kernel.csv, and also on log(f + 10), each
prediction transformed before the mean over the background.

Run from the repository root, in an environment with shap and numpy:
    python validation/scripts/shap_kernel.py
"""

import csv
import pathlib

import numpy as np
import shap

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVELS = {"a": 0, "b": 1, "c": 2, "d": 3}
EFFECT = np.array([0.0, 0.5, -0.3, 1.0])


def f(x):
    x1, x2, x3, g = x[:, 0], x[:, 1], x[:, 2], x[:, 3].astype(int)
    return np.exp(0.3 * x1) + x1 * x2 + EFFECT[g] + 0.5 * x3**2 + 0.4 * x3 * (g == 3)


def main():
    with open(ROOT / "data" / "explain_frame.csv") as fh:
        rows = list(csv.DictReader(fh))
    x = np.array([[float(r["x1"]), float(r["x2"]), float(r["x3"]), LEVELS[r["g"]]] for r in rows])
    background, explain = x[:40], x[40:48]
    out = []
    for name, fun in [("identity", f), ("log10", lambda v: np.log(f(v) + 10.0))]:
        explainer = shap.KernelExplainer(fun, background)
        values = explainer.shap_values(explain, nsamples=2**4, l1_reg=False, silent=True)
        out.append((name, "baseline", "", "", float(explainer.expected_value)))
        for i in range(len(explain)):
            for j, feature in enumerate(["x1", "x2", "x3", "g"]):
                out.append((name, "shap", str(41 + i), feature, float(values[i][j])))
    path = ROOT / "reference" / "shap_kernel.csv"
    with open(path, "w", newline="") as fh:
        fh.write(f"# shap {shap.__version__}, numpy {np.__version__}\n")
        w = csv.writer(fh)
        w.writerow(["output", "quantity", "row", "feature", "expected", "source"])
        for name, q, row, feature, v in out:
            w.writerow([name, q, row, feature, repr(v), f"shap {shap.__version__} KernelExplainer exact"])
    print(len(out), "rows")


if __name__ == "__main__":
    main()
