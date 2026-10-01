"""Regenerate validation/reference/reserving_chainladder_python.csv.

Uses chainladder-python. Run from the repository root:

    uv run --no-project --with chainladder==0.10.1 python validation/scripts/reserving_chainladder_python.py

Triangles are read from validation/data/*.csv (not load_sample) so the R and
Python generators see identical input. `arg` is the 0-based development index
k (k links age k to k+1) for per-age quantities, the origin year for
per-origin quantities, and empty for totals.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_chainladder_python.csv"
DATASETS = ["raa", "genins", "abc"]
VER = cl.__version__
REL = 1e-9
ZERO_ABS = 1e-9

# method -> Development keyword arguments. alpha=1 volume, 0 simple, 2 regression.
MACK_METHODS = {
    "mack": dict(average="volume", sigma_interpolation="log-linear"),
    "mack_sigma_mack": dict(average="volume", sigma_interpolation="mack"),
    "mack_alpha0": dict(average="simple", sigma_interpolation="log-linear"),
    "mack_alpha2": dict(average="regression", sigma_interpolation="log-linear"),
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

    def emit(dataset, method, quantity, arg, value, source):
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"NaN for {dataset} {method} {quantity} {arg}")
        abs_tol = ZERO_ABS if value == 0 else 0
        rows.append([dataset, method, quantity, arg, fmt(value), abs_tol, REL, source])

    for name in DATASETS:
        tri = read_triangle(name)
        n = tri.shape[3]
        origins = [str(d.year) for d in tri.origin.to_timestamp()]
        latest = tri.latest_diagonal.values[0, 0, :, 0]

        def dev_source(kw):
            args = ", ".join(f"{k}='{v}'" for k, v in kw.items())
            return f"chainladder-python {VER}: Development({args})"

        # Volume-weighted chain ladder, no tail.
        dev = cl.Development().fit(tri)
        src = dev_source({})
        ldf = dev.ldf_.values[0, 0, 0, : n - 1]
        for k in range(n - 1):
            emit(name, "chain_ladder", "ata_factor", k, ldf[k], src + ".ldf_")
        cdf = dev.cdf_.values[0, 0, 0, : n - 1]
        for k in range(n - 1):
            emit(name, "chain_ladder", "cdf", k, cdf[k], src + ".cdf_")
        emit(name, "chain_ladder", "cdf", n - 1, 1.0, src + " (no tail: ultimate age cdf = 1)")
        model = cl.Chainladder().fit(dev.transform(tri))
        ult = model.ultimate_.values[0, 0, :, 0]
        csrc = f"chainladder-python {VER}: Chainladder().fit(Development().fit_transform(tri))"
        for i, o in enumerate(origins):
            emit(name, "chain_ladder", "ultimate", o, ult[i], csrc + ".ultimate_")
            emit(name, "chain_ladder", "reserve", o, ult[i] - latest[i], csrc + ".ultimate_ - latest_diagonal")
        emit(name, "chain_ladder", "total_ultimate", "", ult.sum(), csrc + ".ultimate_.sum()")
        emit(name, "chain_ladder", "total_reserve", "", (ult - latest).sum(), csrc + ".ibnr_.sum()")

        # Simple average of link ratios.
        dev = cl.Development(average="simple").fit(tri)
        src = dev_source({"average": "simple"})
        ldf = dev.ldf_.values[0, 0, 0, : n - 1]
        for k in range(n - 1):
            emit(name, "chain_ladder_simple", "ata_factor", k, ldf[k], src + ".ldf_")
        ult = cl.Chainladder().fit(dev.transform(tri)).ultimate_.values[0, 0, :, 0]
        emit(name, "chain_ladder_simple", "total_reserve", "", (ult - latest).sum(),
             src + " then Chainladder().ibnr_.sum()")

        # Mack standard errors.
        for method, kw in MACK_METHODS.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dev = cl.Development(**kw).fit(tri)
                m = cl.MackChainladder().fit(dev.transform(tri))
            for w in caught:
                if "deprecated" in str(w.message):
                    continue  # numpy/pandas deprecation noise, not about the model
                notes.append(f"# {name} {method}: Python warning: {w.message}".replace("\n", " "))
            src = dev_source(kw) + " + MackChainladder()"
            sigma = dev.sigma_.values[0, 0, 0, : n - 1]
            f_se = dev.std_err_.values[0, 0, 0, : n - 1]
            for k in range(n - 1):
                emit(name, method, "sigma", k, sigma[k], src + " .sigma_")
                emit(name, method, "f_se", k, f_se[k], src + " .std_err_")
            se = m.mack_std_err_.values[0, 0, :, -1]
            proc = m.process_risk_.values[0, 0, :, -1]
            par = m.parameter_risk_.values[0, 0, :, -1]
            for i, o in enumerate(origins):
                # mack_std_err_ is NaN for the fully developed origin (its IBNR is
                # 0); both component risks are 0 there, so the SE is 0.
                s = se[i]
                if math.isnan(s) and proc[i] == 0 and par[i] == 0:
                    s = 0.0
                emit(name, method, "se", o, s, src + " .mack_std_err_[..., -1] (NaN -> 0 when fully developed)")
                emit(name, method, "process_risk", o, proc[i], src + " .process_risk_[..., -1]")
                emit(name, method, "parameter_risk", o, par[i], src + " .parameter_risk_[..., -1]")
            emit(name, method, "total_standard_error", "",
                 m.total_mack_std_err_.values[0, 0], src + " .total_mack_std_err_")
            emit(name, method, "total_process_risk", "",
                 m.total_process_risk_.values[0, 0, 0, -1], src + " .total_process_risk_[..., -1]")
            emit(name, method, "total_parameter_risk", "",
                 m.total_parameter_risk_.values[0, 0, 0, -1], src + " .total_parameter_risk_[..., -1]")

    with open(OUT, "w", newline="") as f:
        f.write(f"# Generated by validation/scripts/reserving_chainladder_python.py with chainladder-python {VER}.\n")
        f.write("# Triangles from validation/data/{raa,genins,abc}.csv. No tail factor. Mack risks are\n")
        f.write("# standard errors (not variances) at ultimate. sigma/f_se include the extrapolated last age.\n")
        for note in dict.fromkeys(notes):
            f.write(note + "\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
