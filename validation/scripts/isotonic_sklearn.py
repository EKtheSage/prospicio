"""Reference values for isotonic regression and the calibration curve, from
scikit-learn 1.5.2 (docs/design/models.md, auto-calibration; #545).

- `isotonic`: `IsotonicRegression(increasing=..., out_of_bounds="clip")`
  fitted on seeded data with ties in x and weights (or none), and its
  `predict` at query points inside and outside the data's range;
- `balance`: the multiplier sum(w y) / sum(w pred).

    uv run --no-project --python 3.12 --with scikit-learn==1.5.2 \\
        python validation/scripts/isotonic_sklearn.py

writes validation/reference/isotonic_sklearn.csv.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

OUT = Path(__file__).resolve().parents[1] / "reference" / "isotonic_sklearn.csv"


def nums(a):
    return ";".join(repr(float(v)) for v in np.asarray(a).ravel())


def main():
    rows = []
    for k, (n, ties, weighted, increasing) in enumerate(
        [(50, False, False, True), (80, True, True, True), (60, True, False, False), (120, True, True, False), (7, False, True, True)]
    ):
        rng = np.random.default_rng(100 + k)
        x = rng.integers(0, n // 3, n).astype(float) if ties else rng.normal(size=n)
        sign = 1.0 if increasing else -1.0
        y = sign * x + rng.normal(scale=2.0, size=n)
        w = rng.uniform(0.1, 3.0, size=n) if weighted else None
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, y, sample_weight=w)
        q = np.concatenate([np.linspace(x.min() - 1, x.max() + 1, 15), x[:5]])
        rows.append(dict(case="isotonic", increasing=int(increasing), x=nums(x), y=nums(y),
                         w="" if w is None else nums(w), query=nums(q), expected=nums(iso.predict(q))))
        pred = np.abs(x) + 1.0
        tw = np.ones(n) if w is None else w
        rows.append(dict(case="balance", increasing="", x="", y=nums(np.abs(y)), w="" if w is None else nums(w),
                         query=nums(pred), expected=repr(float((tw * np.abs(y)).sum() / (tw * pred).sum()))))
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["case", "increasing", "x", "y", "w", "query", "expected"])
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
