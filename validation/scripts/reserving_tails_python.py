"""Regenerate validation/reference/reserving_tails_python.csv.

Tail factors of chainladder-python (TailConstant, TailCurve, TailBondy)
applied to the volume-weighted chain ladder, and MackChainladder with each
tail. Run from the repository root:

    uv run --no-project --with chainladder==0.10.1 python validation/scripts/reserving_tails_python.py

Triangles are read from validation/data/*.csv as in
reserving_chainladder_python.py. `arg` is the 0-based development index k
for per-age quantities (`ldf` and `cdf` run past the oldest age, k = n - 1,
into the factors beyond the triangle), the origin year for per-origin
quantities, and empty for totals and tail quantities.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_tails_python.csv"
DATASETS = ["raa", "genins", "abc"]
VER = cl.__version__
REL = 1e-9
ZERO_ABS = 1e-9
# TailBondy fits its exponent b with scipy's least_squares at the default
# tolerances, which stops once the cost changes by less than 1e-8 of itself:
# on GenIns (earliest_age=36) b is 3.6e-6 (relative) short of the least-squares
# optimum. The tail raises a factor to b / (1 - b) and Mack extrapolates from
# it, so the tail's share of each result carries up to about 3e-5 of relative
# error. A generalized Bondy fit (earliest_age set) is checked to 1e-4; the
# classic one keeps b at its starting value 1/2 and is exact.
BONDY_REL = 1e-4

# method -> (constructor source, tail estimator).
METHODS = {
    "tail_constant": ("TailConstant(tail=1.05)", lambda: cl.TailConstant(tail=1.05)),
    "tail_constant_decay": (
        "TailConstant(tail=1.1, decay=0.75)",
        lambda: cl.TailConstant(tail=1.1, decay=0.75),
    ),
    "tail_constant_attach": (
        "TailConstant(tail=1.05, attachment_age=72)",
        lambda: cl.TailConstant(tail=1.05, attachment_age=72),
    ),
    # A tail below 1 scales the ultimates and carries the risk read where a
    # tail of 1.001 would be (R's MackChainLadder ignores such a tail).
    "tail_constant_below_one": ("TailConstant(tail=0.98)", lambda: cl.TailConstant(tail=0.98)),
    "tail_curve_exponential": ("TailCurve()", lambda: cl.TailCurve()),
    "tail_curve_inverse_power": (
        "TailCurve(curve='inverse_power')",
        lambda: cl.TailCurve(curve="inverse_power"),
    ),
    "tail_curve_fit_period": (
        "TailCurve(fit_period=(36, 108), extrap_periods=50)",
        lambda: cl.TailCurve(fit_period=(36, 108), extrap_periods=50),
    ),
    # Ages off the 12-month grid: positions int(age / 12 - 1), so (24, 96).
    "tail_curve_off_grid": (
        "TailCurve(fit_period=(30, 102))",
        lambda: cl.TailCurve(fit_period=(30, 102)),
    ),
    "tail_curve_attach": (
        "TailCurve(attachment_age=60)",
        lambda: cl.TailCurve(attachment_age=60),
    ),
    "tail_bondy": ("TailBondy()", lambda: cl.TailBondy()),
    "tail_bondy_generalized": (
        "TailBondy(earliest_age=36)",
        lambda: cl.TailBondy(earliest_age=36),
    ),
    # Off the grid: ddims[int(30 / 12) - 1], age 24.
    "tail_bondy_off_grid": (
        "TailBondy(earliest_age=30)",
        lambda: cl.TailBondy(earliest_age=30),
    ),
    "tail_bondy_attach": (
        "TailBondy(earliest_age=36, attachment_age=72)",
        lambda: cl.TailBondy(earliest_age=36, attachment_age=72),
    ),
}


def read_triangle(name):
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    df["origin"] = df["origin"].astype(str)
    df["valuation"] = (df["origin"].astype(int) + df["development"] // 12 - 1).astype(str)
    return cl.Triangle(
        df, origin="origin", development="valuation", columns="value", cumulative=True
    )


def fmt(x):
    return repr(float(x))


def main():
    rows = []
    notes = []

    def emit(dataset, method, quantity, arg, value, source, rel=REL):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{value} for {dataset} {method} {quantity} {arg}")
        abs_tol = ZERO_ABS if value == 0 else 0
        rows.append([dataset, method, quantity, arg, fmt(value), abs_tol, rel, source])

    for name in DATASETS:
        tri = read_triangle(name)
        n = tri.shape[3]
        origins = [str(d.year) for d in tri.origin.to_timestamp()]
        latest = tri.latest_diagonal.values[0, 0, :, 0]

        for method, (ctor, make) in METHODS.items():
            rel = BONDY_REL if method.startswith("tail_bondy") and method != "tail_bondy" else REL
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dev = cl.Development().fit_transform(tri)
                tail = make().fit(dev)
                tailed = tail.transform(dev)
                model = cl.MackChainladder().fit(tailed)
            for w in caught:
                if "deprecated" in str(w.message):
                    continue  # numpy/pandas deprecation noise, not about the model
                notes.append(f"# {name} {method}: Python warning: {w.message}".replace("\n", " "))
            src = f"chainladder-python {VER}: {ctor}.fit(Development().fit_transform(tri))"
            msrc = f"chainladder-python {VER}: MackChainladder().fit({ctor}.fit_transform(Development().fit_transform(tri)))"

            ldf = tail.ldf_.values[0, 0, 0, :]
            for k, f in enumerate(ldf):
                emit(name, method, "ldf", k, f, src + ".ldf_", rel)
            cdf = tail.cdf_.values[0, 0, 0, :]
            for k, f in enumerate(cdf):
                emit(name, method, "cdf", k, f, src + ".cdf_", rel)
            emit(name, method, "tail_factor", "", tail.tail_.values[0, 0], src + ".tail_", rel)
            emit(name, method, "tail_sigma", "", tail.sigma_.values[0, 0, 0, -1], src + ".sigma_[..., -1]", rel)
            emit(name, method, "tail_std_err", "", tail.std_err_.values[0, 0, 0, -1],
                 src + ".std_err_[..., -1]", rel)

            ult = model.ultimate_.values[0, 0, :, 0]
            se = model.mack_std_err_.values[0, 0, :, -1]
            proc = model.process_risk_.values[0, 0, :, -1]
            par = model.parameter_risk_.values[0, 0, :, -1]
            for i, o in enumerate(origins):
                emit(name, method, "ultimate", o, ult[i], msrc + ".ultimate_", rel)
                emit(name, method, "reserve", o, ult[i] - latest[i], msrc + ".ultimate_ - latest_diagonal", rel)
                emit(name, method, "se", o, se[i], msrc + ".mack_std_err_[..., -1]", rel)
                emit(name, method, "process_risk", o, proc[i], msrc + ".process_risk_[..., -1]", rel)
                emit(name, method, "parameter_risk", o, par[i], msrc + ".parameter_risk_[..., -1]", rel)
            emit(name, method, "total_ultimate", "", ult.sum(), msrc + ".ultimate_.sum()", rel)
            emit(name, method, "total_reserve", "", (ult - latest).sum(), msrc + ".ibnr_.sum()", rel)
            emit(name, method, "total_standard_error", "", model.total_mack_std_err_.values[0, 0],
                 msrc + ".total_mack_std_err_", rel)
            emit(name, method, "total_process_risk", "", model.total_process_risk_.values[0, 0, 0, -1],
                 msrc + ".total_process_risk_[..., -1]", rel)
            emit(name, method, "total_parameter_risk", "", model.total_parameter_risk_.values[0, 0, 0, -1],
                 msrc + ".total_parameter_risk_[..., -1]", rel)

    with open(OUT, "w", newline="") as f:
        f.write(f"# Generated by validation/scripts/reserving_tails_python.py with chainladder-python {VER}.\n")
        f.write("# Triangles from validation/data/{raa,genins,abc}.csv; volume-weighted factors, log-linear\n")
        f.write("# sigma. ldf and cdf run past the oldest age into the tail's factors (projection_period 12).\n")
        f.write("# Mack risks are standard errors at ultimate, including the tail.\n")
        for note in dict.fromkeys(notes):
            f.write(note + "\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
