# /// script
# requires-python = "==3.11.*"
# dependencies = ["apc==1.0.2", "setuptools<81", "numpy==1.26.4", "pandas==1.5.3", "statsmodels==0.14.2", "scipy<1.14"]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"
# ///
"""Regenerate validation/reference/reserving_apc_python.csv.

The age-period-cohort reserving GLM (`reserving-inflation.md`, PR 6, #979)
on Taylor-Ashe (GenIns), from Python `apc` 1.0.2 (J. Harnau; Kuang, Nielsen
and Nielsen 2008's canonical parameterisation): the Poisson APC and its AC,
AP and PC sub-models, their deviances and fitted values, the canonical
level, slopes and second differences, and the reserves under the three
extrapolation rules of the period effect. Run from the repository root:

    uv run validation/scripts/reserving_apc_python.py

Environment (see `knowledge/references/apc-packages.md`):

* `apc` 1.0.2 imports `pkg_resources` (hence `setuptools<81`) and was written
  for old pandas: with pandas 2 its design builder indexes string columns by
  position, and even with pandas 1.5.3 its loop over the age and cohort
  second differences ends with one empty selection that pandas rejects. The
  script runs the package's own `_get_design_components` with that one
  empty iteration skipped (a no-op in the pandas the package was written
  for); nothing else is changed. numpy, pandas, scipy and statsmodels are
  pinned to the versions that work with it, so this script's environment
  differs from the other generators'.
* Every forecast row is computed by this script, not by `apc`: `apc`'s
  `forecast()` is documented for models without parameter extrapolation
  and fails on this triangle, and it has no `I1` or `I0`. The script
  takes `apc`'s fitted canonical coefficients and its own forecast design
  (`_get_fc_design`, the hinge functions continued past the latest
  diagonal) and applies the rules of R `apc`'s `apc.forecast.apc` (read
  from its source on the CRAN mirror, 2026-11-24) to the fitted period
  second differences `DDbeta` (`n = k - 2` of them):
  - `I2`: the future `DDbeta` are 0;
  - `I1`: `Dbeta = cumsum(DDbeta)`; the first future `DDbeta` is
    `mean(Dbeta) - Dbeta[n]` (the mean over all `n`), then 0;
  - `I0`: `beta = cumsum(cumsum(DDbeta))`; a least-squares line through
    all `n` points of `beta` is extrapolated to `n + 1` and `n + 2`; the
    first two future `DDbeta` are the second differences of `beta, f1, f2`
    at its end, then 0.
  The `I0` total on Taylor-Ashe is checked against R `apc`'s documented
  19,061,428 (documentation of `apc.forecast.apc`); the
  `I1` and `I2` totals have no independent number to meet.

Rows: `deviance` (no arg), `level`, `slope_age` and `slope_cohort` (no
arg), `dd_age`, `dd_period` and `dd_cohort` (arg: the 0-based index of the
second difference, 2 to 9), `fitted` and `future` (arg:
`origin=<year>;age=<months>`), `reserve` (arg: origin year) and
`total_reserve`. Methods: `poisson_apc`, `poisson_ac`, `poisson_ap`,
`poisson_pc` (fits), `poisson_apc_i2`, `poisson_apc_i1`, `poisson_apc_i0`,
`poisson_ap_i2`, `poisson_ap_i1`, `poisson_ap_i0`, `poisson_pc_i2`,
`poisson_pc_i1`, `poisson_pc_i0` (forecasts) and `poisson_ac` (whose
forecast needs no extrapolation).
"""

import csv
import inspect
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore")

import apc
import numpy as np
import pandas as pd
import statsmodels.api as sm

M = sys.modules["apc.Model"]

OUT = "validation/reference/reserving_apc_python.csv"
REL = 1e-6
ABS = 1e-6
R_DOCUMENTED_I0_TOTAL = 19061428.0
FC = "apc 1.0.2 coefficients and forecast design; rule of R apc applied by the generator"

# apc's design builder, with the empty last iteration of its two
# second-difference loops skipped.
_src = textwrap.dedent(inspect.getsource(M.Model._get_design_components))
for _slope, _v, _big in (("slope_age", "i", "I"), ("slope_coh", "k", "K")):
    _old = f"    for {_v} in range(1, {_big} - anchor_index):\n"
    assert _old in _src, _old
    _src = _src.replace(_old, _old + f"        if not ({_slope} == {_v}+1).any():\n            continue\n")
exec(_src, M.__dict__)
M.Model._get_design_components = M.__dict__["_get_design_components"]


def read_genins():
    d = pd.read_csv("validation/data/genins.csv", comment="#")
    d = d.sort_values(["origin", "development"])
    d["inc"] = d.groupby("origin")["value"].diff().fillna(d["value"])
    return d


def extrapolate(dd, rule):
    """The first two future period second differences."""
    if rule == "I2":
        return [0.0, 0.0]
    db = np.cumsum(dd)
    if rule == "I1":
        return [db.mean() - db[-1], 0.0]
    beta = np.cumsum(db)
    n = len(beta)
    slope, intercept = np.polyfit(np.arange(n), beta, 1)
    f1, f2 = intercept + slope * n, intercept + slope * (n + 1)
    return [f1 - 2 * beta[-1] + beta[-2], f2 - 2 * f1 + beta[-1]]


def prediction_errors(emit, method, X, y, F, offset, origins, source):
    """Delta-method prediction errors of the future cells' sums, as the Rust
    `OdpGlmFit::prediction_error`: a Poisson GLM with Pearson's dispersion
    on the observed design `X` (rows `y`); the future design `F` (and
    `offset`); process variance `phi sum mu`, parameter variance
    `g' Cov g` with `g = sum mu x`, by origin and in total."""
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale="X2", tol=1e-14, maxiter=500)
    beta, cov, phi = np.asarray(res.params), np.asarray(res.cov_params()), res.scale
    mu = np.exp(F @ beta + offset)
    origins = np.array(origins)
    groups = [(int(o), origins == o) for o in sorted(set(origins))]
    groups.append(("", np.ones(len(origins), dtype=bool)))
    for arg, sel in groups:
        g = (mu[sel, None] * F[sel]).sum(axis=0)
        emit(method, "process_variance", arg, phi * mu[sel].sum(), source=source)
        emit(method, "parameter_variance", arg, g @ cov @ g, source=source)
    return mu


SM = "statsmodels GLM (Poisson, Pearson scale)"


def separation_cases(emit, genins, rate=0.05, trend=4):
    """Taylor's arithmetic separation as the Poisson GLM on development and
    calendar factors, with a rate future (an offset on the last calendar
    column) and a trend future (the log slope of the last `trend` calendar
    effects)."""
    d = genins.copy()
    origins = sorted(d.origin.unique())
    ages = sorted(d.development.unique())
    k = len(origins)
    d["i"] = d.origin.map({o: i for i, o in enumerate(origins)})
    d["j"] = d.development.map({a: j for j, a in enumerate(ages)})
    d["h"] = d.i + d.j

    def row(j, h, tw=None):
        r = np.zeros(1 + 2 * (k - 1))
        r[0] = 1.0
        if j > 0:
            r[j] = 1.0
        if h < k:
            if h > 0:
                r[k - 1 + h] = 1.0
        else:
            r[k - 1 + (k - 1)] += 1.0
            if tw is not None:
                for hh in range(1, k):
                    r[k - 1 + hh] += (h - k + 1) * tw[hh]
        return r

    X = np.array([row(j, h) for j, h in zip(d.j, d.h)])
    y = d.inc.values.astype(float)
    w = np.zeros(k)
    for hh in range(k - trend, k):
        e = np.zeros(trend)
        e[hh - (k - trend)] = 1.0
        w[hh] = np.polyfit(np.arange(trend), e, 1)[0]
    fut = [(i, j) for i in range(k) for j in range(k) if i + j >= k]
    fo = [origins[i] for i, _ in fut]
    for method, tw, off in (
        ("separation_rate", None, np.array([np.log1p(rate) * (i + j - k + 1) for i, j in fut])),
        ("separation_trend", w, np.zeros(len(fut))),
    ):
        F = np.array([row(j, i + j, tw) for i, j in fut])
        mu = prediction_errors(emit, method, X, y, F, off, fo, SM)
        emit(method, "total_reserve", "", float(mu.sum()), source=SM)


def main():
    ta = apc.loss_TA()
    genins = read_genins()
    # apc's loss_TA is the incremental Taylor-Ashe triangle: the same data.
    for _, r in genins.iterrows():
        o, a = int(r.origin) - 2000, int(r.development) // 12
        assert ta.iloc[o - 1, a - 1] == r.inc, (o, a, ta.iloc[o - 1, a - 1], r.inc)
    assert ta.notna().sum().sum() == len(genins)

    rows = []

    def emit(method, quantity, arg, value, abs_tol=ABS, rel_tol=REL, source=None):
        rows.append([
            "genins", method, quantity, arg, f"{value:.17g}", abs_tol, rel_tol,
            source or "Python apc 1.0.2",
        ])

    k = ta.shape[0]
    for name, predictor in (("apc", "APC"), ("ac", "AC"), ("ap", "AP"), ("pc", "PC")):
        m = apc.Model()
        m.data_from_df(ta, data_format="CL")
        m.fit("poisson_response", predictor)
        method = f"poisson_{name}"
        coef = m.parameters["coef"]
        emit(method, "deviance", "", m.deviance)
        emit(method, "level", "", coef["level"])
        emit(method, "slope_age", "", coef["slope_age"])
        emit(method, "slope_cohort", "", coef["slope_coh"])
        for label, key in (("dd_age", "dd_age_"), ("dd_period", "dd_per_"), ("dd_cohort", "dd_coh_")):
            for c in coef.index:
                if c.startswith(key):
                    emit(method, label, int(c[len(key):]) - 1, coef[c])
        for (age, coh, _per), v in m.fitted_values.items():
            emit(method, "fitted", f"origin={2000 + coh};age={12 * age}", v)

        fc = m._get_fc_design(m.predictor)[coef.index]
        eta0 = fc.dot(coef).values
        hs = np.array([p - 1 for p in fc.index.get_level_values("Period")])
        cells = [(2000 + c, 12 * a) for (c, a, _p) in fc.index]
        dd_per = coef[[c for c in coef.index if c.startswith("dd_per_")]].values
        rules = ["I2", "I1", "I0"] if len(dd_per) else ["I2"]
        for rule in rules:
            fut = extrapolate(dd_per, rule) if len(dd_per) else [0.0, 0.0]
            extra = sum(fut[t] * np.maximum(hs - (k + t) + 1, 0) for t in range(2))
            mu = np.exp(eta0 + extra)
            fm = f"{method}_{rule.lower()}" if name != "ac" else method
            by_origin = {2000 + o: 0.0 for o in range(1, k + 1)}
            for (o, a), v in zip(cells, mu):
                emit(fm, "future", f"origin={o};age={a}", v, source=FC)
                by_origin[o] += v
            for o, v in by_origin.items():
                emit(fm, "reserve", o, v, source=FC)
            F = fc.values.astype(float).copy()
            if len(dd_per):
                n = len(dd_per)
                cols = [i for i, c in enumerate(coef.index) if c.startswith("dd_per_")]
                for t in range(2):
                    for mm, ci in enumerate(cols):
                        wt = extrapolate(np.eye(n)[mm], rule)[t]
                        F[:, ci] += wt * np.maximum(hs - (k + t) + 1, 0)
            prediction_errors(
                emit, fm, m.design.values.astype(float), m.data_vector["response"].values.astype(float),
                F, 0.0, [o for o, _ in cells], SM + " on apc's design",
            )
            total = float(mu.sum())
            emit(fm, "total_reserve", "", total, source=FC)
            if name == "apc" and rule == "I0":
                assert abs(total - R_DOCUMENTED_I0_TOTAL) < 1.0, total
                emit(fm, "documented_total", "", R_DOCUMENTED_I0_TOTAL, abs_tol=1.0, rel_tol=0.0,
                     source="R apc documentation, apc.forecast.apc (Taylor-Ashe)")

    separation_cases(emit, genins)

    with open(OUT, "w", newline="") as f:
        f.write(
            f"# Generated by validation/scripts/reserving_apc_python.py with apc 1.0.2, "
            f"numpy {np.__version__}, pandas {pd.__version__}.\n"
            "# The age-period-cohort reserving GLM on Taylor-Ashe (GenIns): Poisson APC, AC, AP and PC fits,\n"
            "# canonical second differences, and reserves under the I2, I1 and I0 rules for the period effect\n"
            "# (reserving-inflation.md, PR 6, #979).\n"
        )
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
