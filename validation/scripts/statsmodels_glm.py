"""Regenerate validation/data/glm_policies.csv and
validation/reference/glm_statsmodels.csv.

    pip install numpy statsmodels
    python validation/scripts/statsmodels_glm.py

A synthetic portfolio of 600 policies (numeric `age`, categorical `region`
with four levels, `exposure`), with responses for each family drawn from
numpy's generator under a fixed seed and written to the data file, so the
Rust test reads exactly the inputs statsmodels saw. Each case fits
statsmodels' GLM on the design intercept, age, region[B], region[C],
region[D] (treatment coding, reference "A"), with the offset and
var_weights given, and records coefficients, standard errors, deviance,
null deviance, scale and, where statsmodels computes it exactly, the
log-likelihood and AIC.
"""

import csv
import sys

import numpy as np
import statsmodels
import statsmodels.api as sm

DATA = "validation/data/glm_policies.csv"
OUT = "validation/reference/glm_statsmodels.csv"
SOURCE = f"statsmodels {statsmodels.__version__}"
N = 600


def data():
    rng = np.random.default_rng(20261003)
    age = rng.uniform(18, 80, N).round(1)
    region = rng.choice(["A", "B", "C", "D"], N, p=[0.4, 0.3, 0.2, 0.1])
    exposure = rng.uniform(0.2, 1.0, N).round(3)
    rel = {"A": 1.0, "B": 1.2, "C": 0.8, "D": 1.5}
    r = np.array([rel[g] for g in region])
    base = np.exp(-1.0 - 0.01 * (age - 40)) * r
    claims = rng.poisson(base * exposure)
    severity = rng.gamma(2.0, 1000.0 * r / 2.0, N).round(2)
    # Average severity over a positive number of claims, for gamma with
    # weights: n claims of mean severity.
    n_sev = rng.integers(1, 5, N)
    avg_sev = np.array([rng.gamma(2.0 * k, 1000.0 * ri / (2.0 * k)) for k, ri in zip(n_sev, r)]).round(2)
    ig = np.array([rng.wald(m, 2000.0) for m in 500.0 * r]).round(3)
    trials = rng.integers(1, 10, N)
    p = 1 / (1 + np.exp(-(-0.5 + 0.02 * (age - 40) + np.log(r))))
    successes = rng.binomial(trials, p)
    nb = rng.negative_binomial(2.0, 2.0 / (2.0 + 3.0 * base))
    gauss = (100.0 + 2.0 * age + 30.0 * np.log(r) + rng.normal(0, 15, N)).round(3)
    # Tweedie pure premium: claims x gamma severities per unit exposure.
    pure = np.array([rng.gamma(2.0, 500.0, k).sum() for k in claims]) / exposure
    # Non-linear in age, for GAMs (validation/scripts/r_gam.R); drawn last
    # so the columns above do not change.
    wavy = (50.0 + 20.0 * np.sin(age / 8.0) + rng.normal(0, 5, N)).round(3)
    wavy_claims = rng.poisson(exposure * np.exp(0.5 + 0.8 * np.sin(age / 8.0)))
    return {
        "age": age, "region": region, "exposure": exposure, "claims": claims,
        "severity": severity, "n_sev": n_sev, "avg_sev": avg_sev, "ig": ig,
        "trials": trials, "successes": successes, "nb": nb, "gauss": gauss,
        "pure": pure.round(4), "wavy": wavy, "wavy_claims": wavy_claims,
    }


def design(d):
    reg = d["region"]
    return np.column_stack([
        np.ones(N), d["age"],
        (reg == "B").astype(float), (reg == "C").astype(float), (reg == "D").astype(float),
    ])


def cases(d):
    X = design(d)
    log_e = np.log(d["exposure"])
    # name, endog, family, offset, var_weights, exact_llf
    yield "poisson_log", d["claims"], sm.families.Poisson(), log_e, None, True
    yield "quasi_poisson_log", d["claims"], sm.families.Poisson(), log_e, None, False
    yield "gamma_log", d["severity"], sm.families.Gamma(sm.families.links.Log()), None, None, True
    yield "gamma_inverse", d["severity"], sm.families.Gamma(), None, None, True
    yield "gamma_log_weighted", d["avg_sev"], sm.families.Gamma(sm.families.links.Log()), None, d["n_sev"].astype(float), True
    yield "inverse_gaussian_log", d["ig"], sm.families.InverseGaussian(sm.families.links.Log()), None, None, True
    yield "binomial_logit", d["successes"] / d["trials"], sm.families.Binomial(), None, d["trials"].astype(float), False
    yield "negative_binomial_log", d["nb"], sm.families.NegativeBinomial(alpha=0.5), None, None, True
    yield "gaussian_identity", d["gauss"], sm.families.Gaussian(), None, None, True
    yield "tweedie_log", d["pure"], sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()), None, d["exposure"], False


def main():
    d = data()
    with open(DATA, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        cols = ["age", "region", "exposure", "claims", "severity", "n_sev", "avg_sev", "ig",
                "trials", "successes", "nb", "gauss", "pure", "wavy", "wavy_claims"]
        w.writerow(cols)
        for i in range(N):
            w.writerow([d[c][i] for c in cols])
    X = design(d)
    names = ["(Intercept)", "age", "region[B]", "region[C]", "region[D]"]
    rows = []
    for name, y, fam, offset, vw, exact in cases(d):
        model = sm.GLM(y, X, family=fam, offset=offset, var_weights=vw)
        if name == "quasi_poisson_log":
            res = model.fit(scale="X2", tol=1e-14)
        else:
            res = model.fit(tol=1e-14)
        for j, nm in enumerate(names):
            rows.append([name, "coef", nm, res.params[j], 1e-8])
            rows.append([name, "std_error", nm, res.bse[j], 1e-7])
        rows.append([name, "deviance", "", res.deviance, 1e-9])
        rows.append([name, "null_deviance", "", res.null_deviance, 1e-9])
        rows.append([name, "scale", "", res.scale, 1e-8])
        if exact:
            rows.append([name, "log_likelihood", "", res.llf, 1e-9])
            rows.append([name, "aic", "", res.aic, 1e-9])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "quantity", "term", "expected", "abs_tol", "rel_tol", "source"])
        for case, qty, term, value, rel in rows:
            w.writerow([case, qty, term, repr(float(value)), 1e-12, rel, SOURCE])
    print(f"wrote {DATA} and {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
