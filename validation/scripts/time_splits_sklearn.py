"""Reference index sets for time-ordered splits with a window, a horizon and
a gap, from scikit-learn 1.5.2's TimeSeriesSplit (docs/design/models.md,
resampling; #549).

On rows 0 .. n-1, one period each, TimeSeriesSplit(n_splits, test_size,
max_train_size, gap) is prospicio's TimeSplits with n_test = n_splits,
horizon = step = test_size, a rolling window of max_train_size (expanding
when None) and the same gap.

    uv run --no-project --python 3.12 --with scikit-learn==1.5.2 \\
        python validation/scripts/time_splits_sklearn.py

writes validation/reference/time_splits_sklearn.csv.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

OUT = Path(__file__).resolve().parents[1] / "reference" / "time_splits_sklearn.csv"


def main():
    rows = []
    cases = [(20, 3, 2, None, 0), (20, 4, 3, 5, 1), (15, 2, 1, 3, 2), (30, 5, 4, None, 3), (12, 2, 3, 4, 0)]
    for k, (n, n_splits, test_size, max_train, gap) in enumerate(cases):
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train, gap=gap)
        for f, (train, test) in enumerate(tss.split(np.zeros(n))):
            rows.append(dict(case=k, n=n, n_test=n_splits, horizon=test_size,
                             window="" if max_train is None else max_train, gap=gap, split=f,
                             train=" ".join(map(str, train)), test=" ".join(map(str, test))))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "n", "n_test", "horizon", "window", "gap", "split", "train", "test"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
