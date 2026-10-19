"""Regenerate validation/reference/reserving_averages_python.csv.

Averages over the latest periods and excluding high and low link ratios
(decision 13 of docs/design/reserving-v02.md) against chainladder-python's
Development(n_periods, drop_high, drop_low). Run from the repository root:

    uv run --no-project --with chainladder==0.10.1 python validation/scripts/reserving_averages_python.py

raa and genins (column value) are fitted with each of the volume, simple
and regression averages under each setting: n_periods=3, drop_high=1 with
drop_low=1, both together, n_periods=1 (factors only: sigma is not
estimable), and drop_high=2 with n_periods=5. Rows give the factor (ldf),
sigma, the standard error of the factor (std_err), the number of link ratios
behind each factor (n_links) and the chain ladder's ultimate per origin and
in total. clrd_wkcomp (paid, exposure the latest diagonal of premium) is
fitted with Bornhuetter-Ferguson and Cape Cod under two settings, for the
total ultimate.

chainladder-python's weights w_ count the starting cell of the latest
diagonal, which has no link ratio, under n_periods; n_links here counts the
link ratios actually used (weight above 0 and a finite ratio), which is what
the weighted regression sees. `arg` is the age in months for per-age
quantities, the origin year for ultimate, empty for totals.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_averages_python.csv"
VER = cl.__version__
REL = 1e-9

AVERAGES = ["volume", "simple", "regression"]
SETTINGS = [
    {"n_periods": 3},
    {"drop_high": 1, "drop_low": 1},
    {"n_periods": 3, "drop_high": 1, "drop_low": 1},
    {"n_periods": 1},
    {"n_periods": 5, "drop_high": 2},
]
EXPOSURE_METHODS = [
    ("bornhuetter_ferguson;apriori=0.75", lambda: cl.BornhuetterFerguson(apriori=0.75)),
    ("cape_cod;trend=0.05;decay=0.75", lambda: cl.CapeCod(trend=0.05, decay=0.75)),
]
EXPOSURE_SETTINGS = [
    {"n_periods": 3},
    {"n_periods": 4, "drop_high": 1, "drop_low": 1},
]


def read_triangle(name, columns):
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    df["origin"] = df["origin"].astype(str)
    df["valuation"] = (df["origin"].astype(int) + df["development"] // 12 - 1).astype(str)
    return cl.Triangle(
        df, origin="origin", development="valuation", columns=columns, cumulative=True
    )


def fmt(x):
    return repr(float(x))


def label(setting):
    return ";".join(f"{k}={v}" for k, v in setting.items())


def call(setting, average):
    args = ", ".join(f"{k}={v}" for k, v in {"average": repr(average), **setting}.items())
    return f"Development({args})"


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value, source, allow_nan=False):
        value = float(value)
        if math.isnan(value):
            if allow_nan:
                return
            raise ValueError(f"NaN for {dataset} {method} {quantity} {arg}")
        rows.append([dataset, method, quantity, arg, fmt(value), 0, REL, source])

    for name in ["raa", "genins"]:
        tri = read_triangle(name, ["value"])
        ages = [int(a) for a in tri.development.values]
        origins = [str(d.year) for d in tri.origin.to_timestamp()]
        x = tri.values[0, 0, :, :-1]
        y = tri.values[0, 0, :, 1:]
        for average in AVERAGES:
            for setting in SETTINGS:
                method = f"average={average};{label(setting)}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dev = cl.Development(average=average, **setting).fit(tri)
                src = f"chainladder-python {VER}: {call(setting, average)}"
                ldf = dev.ldf_.values[0, 0, 0, :]
                sigma = dev.sigma_.values[0, 0, 0, :]
                std_err = dev.std_err_.values[0, 0, 0, :]
                w = np.nan_to_num(np.asarray(dev.w_)[0, 0])
                links = np.isfinite(y / x) & (w > 0)
                n_links = links.sum(axis=0)
                for k, age in enumerate(ages[:-1]):
                    emit(name, method, "ldf", age, ldf[k], src + ".fit(tri).ldf_")
                    emit(name, method, "n_links", age, n_links[k], src + ".fit(tri).w_ links")
                    # Too few link ratios (n_periods=1, or the drops) leave a sigma
                    # and every one after it unestimable: no row, the factor is
                    # checked. Where a sigma is filled in, chainladder-python's
                    # standard error divides it by the square root of the first
                    # origin's earlier value, not of the volume used, which is
                    # the same only when the first origin is the one link left:
                    # std_err is checked only where the sigma is estimated.
                    emit(name, method, "sigma", age, sigma[k], src + ".fit(tri).sigma_", True)
                    if n_links[k] > 1:
                        emit(name, method, "std_err", age, std_err[k], src + ".fit(tri).std_err_", True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ult = cl.Chainladder().fit(dev.transform(tri)).ultimate_.values[0, 0, :, 0]
                full = f"chainladder-python {VER}: Chainladder().fit({call(setting, average)}.fit_transform(tri))"
                for i, o in enumerate(origins):
                    emit(name, method, "ultimate", o, ult[i], full + ".ultimate_")
                emit(name, method, "total_ultimate", "", ult.sum(), full + ".ultimate_.sum()")

    name = "clrd_wkcomp"
    tri = read_triangle(name, ["paid", "premium"])
    paid = tri["paid"]
    exposure = tri["premium"].latest_diagonal
    for model_label, make in EXPOSURE_METHODS:
        for setting in EXPOSURE_SETTINGS:
            method = f"{model_label};average=volume;{label(setting)}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dev = cl.Development(average="volume", **setting).fit_transform(paid)
                model = make()
                model.fit(dev, sample_weight=exposure)
            ult = model.ultimate_.values[0, 0, :, 0]
            full = (
                f"chainladder-python {VER}: {type(model).__name__}(...).fit("
                f"{call(setting, 'volume')}.fit_transform(paid), "
                "sample_weight=premium.latest_diagonal)"
            )
            emit(name, method, "total_ultimate", "", ult.sum(), full + ".ultimate_.sum()")

    with open(OUT, "w", newline="") as f:
        f.write(
            "# Generated by validation/scripts/reserving_averages_python.py with "
            f"chainladder-python {VER}.\n"
        )
        f.write("# Development(n_periods, drop_high, drop_low) on raa and genins (value), and\n")
        f.write("# Bornhuetter-Ferguson and Cape Cod on clrd_wkcomp. See the script's docstring.\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
