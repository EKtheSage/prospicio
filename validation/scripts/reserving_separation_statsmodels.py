# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.5", "pandas==2.3.3", "statsmodels==0.14.5"]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"
# ///
"""Regenerate validation/reference/reserving_separation_statsmodels.csv.

Taylor's (1977) arithmetic separation method as the Poisson GLM it is
(Verbeek 1972; `reserving-inflation.md` decision 8): the increments of
each origin divided by its normaliser, `s_ij`, regressed on development
and calendar factors with a log link, no origin factor and no offset.
From the fit, with `b_j` the development and `c_h` the calendar effects
(first levels zero) and `a` the intercept:

    r_j = exp(b_j) / sum_m exp(b_m),   lambda_h = exp(a + c_h) sum_m exp(b_m)

so that `sum r_j = 1`, and the fitted amounts are `n_i r_j lambda_{i+j}`.
Future indices grow from the last one at 5% a year; the future cell of
origin `i` at age `j` is `n_i r_j lambda_{i+j}`. Run from the repository
root:

    uv run validation/scripts/reserving_separation_statsmodels.py

Rows: `proportion` (arg: age in months), `index` (arg: calendar year),
`fitted` and `future` (arg: `origin=<year>;age=<months>`), `reserve` (arg:
origin year) and `total_reserve`. Methods: `arithmetic` (no normaliser) on
GenIns and ABC, `arithmetic_premium` on GenIns with the premium of
`genins_premium.csv` as the normaliser.

The geometric method (PR 4, decisions 12 and 13) is ordinary least squares
of `ln s_ij` on the same factors, mapped to `r` and `lambda` the same way;
`geometric` takes the fitted values back as `exp(eta)`, `geometric_lognormal`
as `exp(eta + sigma^2 / 2)` with `sigma^2` the residual sum of squares over
the residual degrees of freedom (every index times `exp(sigma^2 / 2)`), and
both report `log_variance`. `geometric_premium` is the geometric method with
the premium normaliser.
"""

import csv

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

OUT = "validation/reference/reserving_separation_statsmodels.csv"
REL = 1e-9
RATE = 0.05


def read(name, value="value"):
    d = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    d = d.sort_values(["origin", "development"])
    d["inc"] = d.groupby("origin")[value].diff().fillna(d[value])
    return d


def separation(d, normaliser, kind="arithmetic"):
    origins = sorted(d.origin.unique())
    ages = sorted(d.development.unique())
    n = len(origins)
    d = d.copy()
    d["i"] = d.origin.map({o: k for k, o in enumerate(origins)})
    d["j"] = d.development.map({a: k for k, a in enumerate(ages)})
    d["h"] = d.i + d.j
    d["n"] = d.origin.map(normaliser)
    d["s"] = d.inc / d.n
    x = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="const"),
            pd.get_dummies(d.j, prefix="dev", drop_first=True, dtype=float),
            pd.get_dummies(d.h, prefix="cal", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    if kind == "arithmetic":
        fit = sm.GLM(d.s, x, family=sm.families.Poisson()).fit(tol=1e-15, maxiter=500)
        sigma2 = None
    else:
        fit = sm.OLS(np.log(d.s), x).fit()
        sigma2 = fit.ssr / fit.df_resid
    p = fit.params
    b = np.array([0.0] + [p[f"dev_{j}"] for j in range(1, n)])
    c = np.array([0.0] + [p[f"cal_{h}"] for h in range(1, n)])
    scale = np.exp(b).sum()
    r = np.exp(b) / scale
    lam = np.exp(p["const"] + c) * scale
    if kind == "geometric_lognormal":
        lam = lam * np.exp(sigma2 / 2)
    per = (1.0 + RATE) ** np.arange(1, n)
    lam_all = np.concatenate([lam, lam[-1] * per])
    if kind == "arithmetic":
        fitted = fit.fittedvalues * d.n
    else:
        fitted = d.n * r[d.j.values] * lam[d.h.values]
    return origins, ages, r, lam, lam_all, d, fitted, sigma2


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value):
        rows.append([dataset, method, quantity, arg, f"{value:.17g}", 0, REL, f"statsmodels {statsmodels.__version__}"])

    genins_premium = read("genins_premium", "paid")
    premium = genins_premium.groupby("origin").premium.first().to_dict()
    cases = [
        ("genins", "arithmetic", read("genins"), None),
        ("abc", "arithmetic", read("abc"), None),
        ("genins", "arithmetic_premium", genins_premium, premium),
        ("genins", "geometric", read("genins"), None),
        ("abc", "geometric", read("abc"), None),
        ("genins", "geometric_lognormal", read("genins"), None),
        ("abc", "geometric_lognormal", read("abc"), None),
        ("genins", "geometric_premium", genins_premium, premium),
    ]
    for dataset, method, d, normaliser in cases:
        norm = normaliser or {o: 1.0 for o in d.origin.unique()}
        kind = {"arithmetic_premium": "arithmetic", "geometric_premium": "geometric"}.get(method, method)
        origins, ages, r, lam, lam_all, cells, mu, sigma2 = separation(d, norm, kind)
        if sigma2 is not None:
            emit(dataset, method, "log_variance", "", sigma2)
        n = len(origins)
        for j, a in enumerate(ages):
            emit(dataset, method, "proportion", a, r[j])
        for h in range(n):
            emit(dataset, method, "index", origins[0] + h, lam[h])
        for (_, cell), m in zip(cells.iterrows(), mu):
            emit(dataset, method, "fitted", f"origin={cell.origin};age={cell.development}", m)
        total = 0.0
        for i, o in enumerate(origins):
            reserve = 0.0
            for j in range(n - i, n):
                v = norm[o] * r[j] * lam_all[i + j]
                emit(dataset, method, "future", f"origin={o};age={ages[j]}", v)
                reserve += v
            emit(dataset, method, "reserve", o, reserve)
            total += reserve
        emit(dataset, method, "total_reserve", "", total)

    with open(OUT, "w", newline="") as f:
        f.write(
            f"# Generated by validation/scripts/reserving_separation_statsmodels.py with statsmodels "
            f"{statsmodels.__version__}, numpy {np.__version__}, pandas {pd.__version__}.\n"
            "# Taylor's arithmetic separation method as a Poisson GLM on development and calendar factors,\n"
            "# the geometric method as least squares on the logs;\n"
            f"# future indices at {RATE:.0%} a year from the last (reserving-inflation.md, PR 3, #979).\n"
        )
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
