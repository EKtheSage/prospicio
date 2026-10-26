"""Regenerate validation/reference/reserving_latest_diagonal_python.csv.

A latest-diagonal-only triangle (one observed cell per origin, each at its own
age) with a fully selected pattern, against chainladder-python's
DevelopmentConstant. Run from the repository root:

    uv run --no-project --with chainladder==0.10.1 --with pandas python validation/scripts/reserving_latest_diagonal_python.py

The pattern is the full triangle's own volume-weighted pattern (no scaling):
e_k the age-to-age factors, c_k the factors to ultimate with 1 at the oldest age.
It is given to DevelopmentConstant, which is fitted on the latest diagonal
(`Triangle.latest_diagonal`) alone:

* style=ldf: the factors e_k plus 1.0 at the oldest age (without that entry the
  oldest origin's ultimate is NaN);
* style=cdf: the factors to ultimate c_k.

genins (value) is fitted with the chain ladder; clrd_wkcomp (paid, exposure the
latest diagonal of premium) with the expected-loss methods, as
reserving_patterns_python.py does, for the cdf pattern. Rows with quantity
`pattern` are the inputs. `arg` is the age in months for pattern, the origin
year for per-origin quantities, empty for totals.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_latest_diagonal_python.csv"
VER = cl.__version__
REL = 1e-9

EXPOSURE_METHODS = [
    ("bornhuetter_ferguson;apriori=0.6", lambda: cl.BornhuetterFerguson(apriori=0.6)),
    ("benktander;apriori=0.6;n_iters=3", lambda: cl.Benktander(apriori=0.6, n_iters=3)),
    ("cape_cod;trend=0.02;decay=0.75", lambda: cl.CapeCod(trend=0.02, decay=0.75)),
]


def read_triangle(name, columns):
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    df["origin"] = df["origin"].astype(str)
    df["valuation"] = (df["origin"].astype(int) + df["development"] // 12 - 1).astype(str)
    return cl.Triangle(
        df, origin="origin", development="valuation", columns=columns, cumulative=True
    )


def own_pattern(tri):
    ages = [int(a) for a in tri.development.values]
    e = cl.Development(average="volume").fit(tri).ldf_.values[0, 0, 0, : len(ages) - 1]
    cdf = list(np.cumprod(e[::-1])[::-1]) + [1.0]
    return ages, e, cdf


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value, source):
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"NaN for {dataset} {method} {quantity} {arg}")
        rows.append([dataset, method, quantity, arg, repr(value), 0, REL, source])

    # genins: chain ladder on the latest diagonal.
    tri = read_triangle("genins", ["value"])
    ages, e, cdf = own_pattern(tri)
    diag = tri.latest_diagonal
    origins = [str(d.year) for d in tri.origin.to_timestamp()]
    for style in ["ldf", "cdf"]:
        method = f"chain_ladder;style={style}"
        if style == "ldf":
            given = dict(zip(ages[:-1], [float(x) for x in e]))
            given[ages[-1]] = 1.0
        else:
            given = dict(zip(ages, [float(x) for x in cdf]))
        for age, v in given.items():
            emit("genins", method, "pattern", age, v, "input: the selected pattern")
        dev = cl.DevelopmentConstant(patterns=given, style=style).fit_transform(diag)
        ult = cl.Chainladder().fit(dev).ultimate_.values[0, 0, :, 0]
        src = (
            f"chainladder-python {VER}: Chainladder().fit(DevelopmentConstant("
            f"patterns=<{style} pattern>, style='{style}').fit_transform(latest_diagonal))"
        )
        for i, o in enumerate(origins):
            emit("genins", method, "ultimate", o, ult[i], src + ".ultimate_")
        emit("genins", method, "total_ultimate", "", ult.sum(), src + ".ultimate_.sum()")

    # clrd_wkcomp: expected-loss methods on the paid latest diagonal.
    name = "clrd_wkcomp"
    tri = read_triangle(name, ["paid", "premium"])
    paid = tri["paid"]
    ages, e, cdf = own_pattern(paid)
    given = dict(zip(ages, [float(x) for x in cdf]))
    diag = paid.latest_diagonal
    exposure = tri["premium"].latest_diagonal
    origins = [str(d.year) for d in paid.origin.to_timestamp()]
    dev = cl.DevelopmentConstant(patterns=given, style="cdf").fit_transform(diag)
    for label, make in EXPOSURE_METHODS:
        method = f"{label};style=cdf"
        for age, v in given.items():
            emit(name, method, "pattern", age, v, "input: the selected pattern")
        model = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(dev, sample_weight=exposure)
        ult = model.ultimate_.values[0, 0, :, 0]
        src = (
            f"chainladder-python {VER}: {type(model).__name__}(...).fit(DevelopmentConstant("
            "patterns=<cdf pattern>, style='cdf').fit_transform(paid.latest_diagonal), "
            "sample_weight=premium.latest_diagonal)"
        )
        for i, o in enumerate(origins):
            emit(name, method, "ultimate", o, ult[i], src + ".ultimate_")
        emit(name, method, "total_ultimate", "", ult.sum(), src + ".ultimate_.sum()")

    with open(OUT, "w", newline="") as f:
        f.write(
            "# Generated by validation/scripts/reserving_latest_diagonal_python.py with "
            f"chainladder-python {VER}.\n"
        )
        f.write("# A latest-diagonal triangle with the full triangle's own volume-weighted pattern\n")
        f.write("# through DevelopmentConstant: genins (value) with the chain ladder, clrd_wkcomp\n")
        f.write("# (paid, exposure the latest premium) with the expected-loss methods.\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
