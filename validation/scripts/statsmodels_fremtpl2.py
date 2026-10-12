"""Regenerate validation/reference/fremtpl2_statsmodels.csv.

    pip install numpy statsmodels
    python validation/scripts/fetch_fremtpl2.py
    python validation/scripts/statsmodels_fremtpl2.py

The claim frequency GLM of Noll, Salzmann and Wüthrich (2018) on all
678,013 policies of freMTPL2freq (prepared by fetch_fremtpl2.py): ClaimNb
on Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density
and Region, log link, offset log(Exposure). The design is built as
prospicio_models::Terms builds it: an intercept, then each term in order,
numeric terms as they are and factors in treatment coding with one column
per level other than the reference, levels in sorted order, named
`name[level]`.

Cases:
- poisson_log: Poisson maximum likelihood, coefficients, standard errors,
  deviance, null deviance, log-likelihood and AIC;
- quasi_poisson_log: the same fit with Pearson's dispersion, its scale and
  standard errors;
- poisson_log_hc0: the Poisson fit's HC0 sandwich standard errors;
- binomial_logit: whether a policy has a claim (ClaimNb > 0) by logistic
  regression on the same design, without an offset, with coefficients,
  standard errors, deviances, log-likelihood and AIC, and
  binomial_logit_hc0 its HC0 standard errors;
- gamma_log_weighted: claim severity on freMTPL2sev joined to the
  portfolio (validation/data/external/freMTPL2sev_glm.csv, 24,944
  policies with claim amounts): the average amount AvgClaim on the same
  design, without an offset, gamma family and log link, with the number
  of amounts NSev as var_weights (prospicio's prior weights), Pearson's
  dispersion; the same quantities plus the scale, and
  gamma_log_weighted_hc0 its HC0 standard errors.
"""

import csv
import sys

import numpy as np
import statsmodels
import statsmodels.api as sm

DATA = "validation/data/external/freMTPL2freq_glm.csv"
SEV_DATA = "validation/data/external/freMTPL2sev_glm.csv"
OUT = "validation/reference/fremtpl2_statsmodels.csv"
SOURCE = f"statsmodels {statsmodels.__version__}"

# (column, kind, reference level)
TERMS = [
    ("Area", "numeric", None),
    ("VehPower", "factor", None),
    ("VehAge", "factor", "6-12"),
    ("DrivAge", "factor", "41-50"),
    ("BonusMalus", "numeric", None),
    ("VehBrand", "factor", None),
    ("VehGas", "factor", None),
    ("Density", "numeric", None),
    ("Region", "factor", "R24"),
]


def read(path=DATA):
    with open(path) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    return {name: [r[j] for r in body] for j, name in enumerate(header)}


def design(d):
    n = len(d["ClaimNb"])
    cols, names = [np.ones(n)], ["(Intercept)"]
    for name, kind, ref in TERMS:
        values = d[name]
        if kind == "numeric":
            cols.append(np.array(values, dtype=float))
            names.append(name)
            continue
        levels = sorted(set(values))
        ref = ref if ref is not None else levels[0]
        v = np.array(values)
        for level in levels:
            if level != ref:
                cols.append((v == level).astype(float))
                names.append(f"{name}[{level}]")
    return np.column_stack(cols), names


def full(case, model, names, scale):
    """Coefficients, standard errors, deviances, log-likelihood, AIC (and
    the scale), then HC0 standard errors as case_hc0."""
    fit = model.fit(tol=1e-14)
    rows = []
    for j, nm in enumerate(names):
        rows.append([case, "coef", nm, fit.params[j], 1e-8])
        rows.append([case, "std_error", nm, fit.bse[j], 1e-7])
    rows.append([case, "deviance", "", fit.deviance, 1e-10])
    rows.append([case, "null_deviance", "", fit.null_deviance, 1e-10])
    rows.append([case, "log_likelihood", "", fit.llf, 1e-10])
    rows.append([case, "aic", "", fit.aic, 1e-10])
    if scale:
        rows.append([case, "scale", "", fit.scale, 1e-8])
    hc0 = model.fit(tol=1e-14, cov_type="HC0")
    for j, nm in enumerate(names):
        rows.append([f"{case}_hc0", "std_error", nm, hc0.bse[j], 1e-7])
    return rows


def main():
    d = read()
    X, names = design(d)
    y = np.array(d["ClaimNb"], dtype=float)
    offset = np.log(np.array(d["Exposure"], dtype=float))
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    rows = []
    fit = model.fit(tol=1e-14)
    for j, nm in enumerate(names):
        rows.append(["poisson_log", "coef", nm, fit.params[j], 1e-8])
        rows.append(["poisson_log", "std_error", nm, fit.bse[j], 1e-7])
    rows.append(["poisson_log", "deviance", "", fit.deviance, 1e-10])
    rows.append(["poisson_log", "null_deviance", "", fit.null_deviance, 1e-10])
    rows.append(["poisson_log", "log_likelihood", "", fit.llf, 1e-10])
    rows.append(["poisson_log", "aic", "", fit.aic, 1e-10])
    quasi = model.fit(scale="X2", tol=1e-14)
    rows.append(["quasi_poisson_log", "scale", "", quasi.scale, 1e-8])
    for j, nm in enumerate(names):
        rows.append(["quasi_poisson_log", "std_error", nm, quasi.bse[j], 1e-7])
    hc0 = model.fit(tol=1e-14, cov_type="HC0")
    for j, nm in enumerate(names):
        rows.append(["poisson_log_hc0", "std_error", nm, hc0.bse[j], 1e-7])

    has_claim = (y > 0).astype(float)
    binomial = sm.GLM(has_claim, X, family=sm.families.Binomial())
    rows += full("binomial_logit", binomial, names, scale=False)

    s = read(SEV_DATA)
    Xs, sev_names = design(s)
    gamma = sm.GLM(
        np.array(s["AvgClaim"], dtype=float), Xs,
        family=sm.families.Gamma(sm.families.links.Log()),
        var_weights=np.array(s["NSev"], dtype=float),
    )
    rows += full("gamma_log_weighted", gamma, sev_names, scale=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "quantity", "term", "expected", "abs_tol", "rel_tol", "source"])
        for case, qty, term, value, rel in rows:
            w.writerow([case, qty, term, repr(float(value)), 1e-12, rel, SOURCE])
    print(f"wrote {OUT}: {len(names)} coefficients", file=sys.stderr)


if __name__ == "__main__":
    main()
