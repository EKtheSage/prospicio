"""Regenerate validation/data/fremtpl2_sample.csv,
validation/data/fremtpl2_sev_sample.csv and
validation/reference/glm_fremtpl2_statsmodels.csv.

    pip install numpy pandas statsmodels
    python validation/scripts/statsmodels_fremtpl2.py

Downloads freMTPL2freq (OpenML data id 41214, 678,013 French motor policies),
applies the usual cleaning of Noll, Salzmann and Wuthrich (ClaimNb capped at
4, Exposure capped at 1), and keeps a fixed-seed sample of 50,000 policies so
the committed file stays small. Each case fits statsmodels' GLM on the
intercept, DrivAge, BonusMalus, log(Density) and the factors Area, VehPower,
VehBrand, VehGas and Region (treatment coding, reference the first level in
sorted order), and records coefficients, standard errors, deviance, null
deviance and scale.

The gamma case uses freMTPL2sev (OpenML data id 41215, one row per claim):
for the sampled policies with at least one claim amount, the mean amount per
policy is fitted with a log link and the number of amounts as var_weights.
"""

import csv
import io
import sys
import urllib.request

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

URL = "https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff"
SEV_URL = "https://www.openml.org/data/v1/download/20649149/freMTPL2sev.arff"
DATA = "validation/data/fremtpl2_sample.csv"
SEV_DATA = "validation/data/fremtpl2_sev_sample.csv"
OUT = "validation/reference/glm_fremtpl2_statsmodels.csv"
SOURCE = f"statsmodels {statsmodels.__version__}"
SAMPLE = 50_000
SEED = 20261005
FACTORS = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]
NUMERIC = ["DrivAge", "BonusMalus"]


def load():
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    # The ARFF has a string attribute scipy cannot read; the data section is
    # plain CSV with single-quoted strings.
    head, body = raw.split("@data\n", 1)
    names = [l.split()[1] for l in head.splitlines() if l.lower().startswith("@attribute")]
    df = pd.read_csv(io.StringIO(body), names=names, quotechar="'")
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(int)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df["VehPower"] = df["VehPower"].astype(int).astype(str)
    df["IDpol"] = df["IDpol"].astype(int)
    df["Density"] = df["Density"].astype(float)
    df = df.sample(n=SAMPLE, random_state=SEED).sort_values("IDpol").reset_index(drop=True)
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    return df[["IDpol", "ClaimNb", "HasClaim", "Exposure", "Area", "VehPower", "VehAge",
               "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Region", "Density"]]


def load_severity(df):
    raw = urllib.request.urlopen(SEV_URL).read().decode("utf-8")
    body = raw.split("@data\n", 1)[1]
    sev = pd.read_csv(io.StringIO(body), names=["IDpol", "ClaimAmount"])
    sev["IDpol"] = sev["IDpol"].astype(int)
    agg = sev.groupby("IDpol")["ClaimAmount"].agg(NSev="count", AvgClaim="mean").reset_index()
    out = df.merge(agg, on="IDpol", how="inner").sort_values("IDpol").reset_index(drop=True)
    out["AvgClaim"] = out["AvgClaim"].round(6)
    return out


def design(df):
    cols = [np.ones(len(df))]
    names = ["(Intercept)"]
    for c in NUMERIC:
        cols.append(df[c].to_numpy(float))
        names.append(c)
    cols.append(np.log(df["Density"].to_numpy(float)))
    names.append("log_Density")
    for f in FACTORS:
        levels = sorted(df[f].unique())
        for lv in levels[1:]:
            cols.append((df[f] == lv).to_numpy(float))
            names.append(f"{f}[{lv}]")
    return np.column_stack(cols), names


def main():
    df = load()
    df.to_csv(DATA, index=False, lineterminator="\n")
    sev = load_severity(df)
    sev.to_csv(SEV_DATA, index=False, lineterminator="\n")
    X, names = design(df)
    Xs, _ = design(sev)
    log_e = np.log(df["Exposure"].to_numpy())
    cases = [
        ("poisson_log", df["ClaimNb"].to_numpy(), sm.families.Poisson(), log_e, None),
        ("quasi_poisson_log", df["ClaimNb"].to_numpy(), sm.families.Poisson(), log_e, "X2"),
        ("binomial_logit", df["HasClaim"].to_numpy(), sm.families.Binomial(), None, None),
    ]
    cases.append((
        "gamma_log_weighted", sev["AvgClaim"].to_numpy(),
        sm.families.Gamma(sm.families.links.Log()), None, None,
    ))
    rows = []
    for name, y, fam, offset, scale in cases:
        x, w = (Xs, sev["NSev"].to_numpy(float)) if name.startswith("gamma") else (X, None)
        model = sm.GLM(y, x, family=fam, offset=offset, var_weights=w)
        res = model.fit(scale=scale, tol=1e-12) if scale else model.fit(tol=1e-12)
        for j, nm in enumerate(names):
            rows.append([name, "coef", nm, res.params[j], 1e-7])
            rows.append([name, "std_error", nm, res.bse[j], 1e-6])
        rows.append([name, "deviance", "", res.deviance, 1e-8])
        rows.append([name, "null_deviance", "", res.null_deviance, 1e-8])
        rows.append([name, "scale", "", res.scale, 1e-7])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "quantity", "term", "expected", "abs_tol", "rel_tol", "source"])
        for case, qty, term, value, rel in rows:
            w.writerow([case, qty, term, repr(float(value)), 1e-12, rel, SOURCE])
    print(f"wrote {DATA}, {SEV_DATA} and {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
