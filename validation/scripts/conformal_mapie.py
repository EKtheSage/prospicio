"""Reference intervals for conformal prediction, from MAPIE 0.9.1 with
scikit-learn 1.5.2 (docs/design/models.md, conformal prediction; #546).

Each case fixes the data, the model and the folds, so the predictions
prospicio's functions take are exactly the ones MAPIE uses:

- `split`: `MapieRegressor(LinearRegression fitted on a training set,
  cv="prefit", method="base")` calibrated on a held-out set;
- `cqr`: `MapieQuantileRegressor` on three prefit linear quantile
  regressions (alpha/2, 1 - alpha/2 and the median), cv="prefit", at a
  calibration size where its "higher" quantile is the conformal order
  statistic;
- `cv_plus`: `MapieRegressor(LinearRegression, cv=KFold(5),
  method="plus")`, with each fold's model's out-of-fold and test
  predictions;
- `jackknife_plus`: the same with `cv=LeaveOneOut()`.

    uv run --no-project --python 3.12 --with scikit-learn==1.5.2 \\
        --with mapie==0.9.1 python validation/scripts/conformal_mapie.py

writes validation/reference/conformal_mapie.csv.
"""

import csv
from pathlib import Path

import numpy as np
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.regression import MapieRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut

OUT = Path(__file__).resolve().parents[1] / "reference" / "conformal_mapie.csv"


def nums(a):
    return ";".join(repr(float(v)) for v in np.asarray(a).ravel())


def data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=(n, 2))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=1.0 + 0.3 * x[:, 0], size=n)
    return x, y


def main():
    rows = []
    alpha = 0.1
    # Split conformal.
    x, y = data(400, 1)
    xt, yt, xc, yc, xs = x[:150], y[:150], x[150:350], y[150:350], x[350:360]
    lr = LinearRegression().fit(xt, yt)
    m = MapieRegressor(lr, cv="prefit", method="base").fit(xc, yc)
    _, b = m.predict(xs, alpha=alpha)
    rows.append(dict(case="split", alpha=alpha, y=nums(yc), pred=nums(lr.predict(xc)), folds="",
                     test=nums(lr.predict(xs)), lo=nums(b[:, 0, 0]), hi=nums(b[:, 1, 0])))
    # CQR.
    qs = [QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(xt, yt) for q in (alpha / 2, 1 - alpha / 2, 0.5)]
    mq = MapieQuantileRegressor(qs, alpha=alpha, cv="prefit").fit(xc, yc)
    _, b = mq.predict(xs)
    rows.append(dict(case="cqr", alpha=alpha, y=nums(yc), pred=nums(qs[0].predict(xc)) + "|" + nums(qs[1].predict(xc)),
                     folds="", test=nums(qs[0].predict(xs)) + "|" + nums(qs[1].predict(xs)),
                     lo=nums(b[:, 0, 0]), hi=nums(b[:, 1, 0])))
    # CV+ and jackknife+.
    for case, n, cv in [("cv_plus", 120, KFold(5)), ("jackknife_plus", 40, LeaveOneOut())]:
        x, y = data(n + 10, 7)
        xtr, ytr, xs = x[:n], y[:n], x[n:]
        m = MapieRegressor(LinearRegression(), cv=cv, method="plus").fit(xtr, ytr)
        _, b = m.predict(xs, alpha=alpha)
        folds = np.zeros(n, dtype=int)
        oof = np.zeros(n)
        test = []
        for k, (tr, te) in enumerate(cv.split(xtr)):
            f = LinearRegression().fit(xtr[tr], ytr[tr])
            folds[te] = k
            oof[te] = f.predict(xtr[te])
            test.append(nums(f.predict(xs)))
        rows.append(dict(case=case, alpha=alpha, y=nums(ytr), pred=nums(oof), folds=nums(folds),
                         test="|".join(test), lo=nums(b[:, 0, 0]), hi=nums(b[:, 1, 0])))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "alpha", "y", "pred", "folds", "test", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
