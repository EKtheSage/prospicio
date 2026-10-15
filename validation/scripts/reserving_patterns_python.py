"""Regenerate validation/reference/reserving_patterns_python.csv.

Selected development patterns (decision 12 of docs/design/reserving-v02.md)
against chainladder-python's DevelopmentConstant. Run from the repository
root:

    uv run --no-project --with chainladder==0.10.1 python validation/scripts/reserving_patterns_python.py

Each case selects a full benchmark pattern, made from the triangle's own
volume-weighted factors as s_k = 1 + 1.1 (e_k - 1) for every age but the
oldest, and gives it to DevelopmentConstant in one of five ways:

* style=ldf: the age-to-age factors s_k;
* style=cdf: the factors to ultimate, 1 at the oldest age (where the two
  libraries' conventions for the level agree);
* style=cdf;level=1.03: the factors to ultimate times 1.03 and no tail: both
  libraries keep the level 1.03 at the oldest age as the tail (prospicio's
  implied tail, decided 2026-10-15);
* style=cdf;tail=1.05: the factors to ultimate times 1.03, so the level at
  the oldest age is 1.03, followed by TailConstant(tail=1.05), which replaces
  that level with the tail (chainladder-python and prospicio agree here too);
* style=ldf;weight=0.6: chainladder-python has no blend, so the script blends
  itself, 0.6 s_k + 0.4 e_k, and passes the result as style=ldf. This checks
  how the blended factors reach each method, not the blend's arithmetic.

Rows with quantity `pattern` are the inputs, the values the Rust test passes
as the selected pattern (s_k, or the factors to ultimate as given): an
age-to-age pattern is read back from the fit's selected factors, a
cumulative one as the fit's factor to ultimate times its pattern_rescale.
`arg` is the age in months for pattern, ldf and cdf, the origin year for
per-origin quantities, empty for totals.

raa and genins (column value) are fitted with the chain ladder; clrd_wkcomp
(paid, exposure the latest diagonal of premium) with the expected-loss
methods, as reserving_expected_loss_python.py does.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_patterns_python.csv"
VER = cl.__version__
REL = 1e-9
SCALE = 1.1
LEVEL = 1.03
TAIL = 1.05
WEIGHT = 0.6

STYLES = [
    "style=ldf",
    "style=cdf",
    "style=cdf;level=1.03",
    "style=cdf;tail=1.05",
    "style=ldf;weight=0.6",
]
CHAIN_LADDER = [("raa", "value"), ("genins", "value")]
EXPOSURE_METHODS = [
    ("expected_loss;apriori=0.75", lambda: cl.ExpectedLoss(apriori=0.75)),
    ("bornhuetter_ferguson;apriori=0.75", lambda: cl.BornhuetterFerguson(apriori=0.75)),
    ("benktander;apriori=0.75;n_iters=2", lambda: cl.Benktander(apriori=0.75, n_iters=2)),
    ("cape_cod;trend=0.05;decay=0.75", lambda: cl.CapeCod(trend=0.05, decay=0.75)),
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


def patterns(tri, style):
    """The pattern the Rust test selects (`pattern` rows), the one given to
    DevelopmentConstant, its style and the tail, for `style`."""
    ages = [int(a) for a in tri.development.values]
    e = cl.Development(average="volume").fit(tri).ldf_.values[0, 0, 0, : len(ages) - 1]
    s = 1 + SCALE * (e - 1)
    cdf = list(np.cumprod(s[::-1])[::-1]) + [1.0]
    if style == "style=ldf":
        given = dict(zip(ages[:-1], s))
        return given, given, "ldf", None
    if style == "style=cdf":
        given = dict(zip(ages, cdf))
        return given, given, "cdf", None
    if style == "style=cdf;level=1.03":
        given = {a: LEVEL * c for a, c in zip(ages, cdf)}
        return given, given, "cdf", None
    if style == "style=cdf;tail=1.05":
        given = {a: LEVEL * c for a, c in zip(ages, cdf)}
        return given, given, "cdf", TAIL
    if style == "style=ldf;weight=0.6":
        blended = WEIGHT * s + (1 - WEIGHT) * e
        return dict(zip(ages[:-1], s)), dict(zip(ages[:-1], blended)), "ldf", None
    raise ValueError(style)


def developed(tri, style):
    selected, passed, kind, tail = patterns(tri, style)
    out = cl.DevelopmentConstant(
        patterns={k: float(v) for k, v in passed.items()}, style=kind
    ).fit_transform(tri)
    src = f"DevelopmentConstant(patterns=<{kind} pattern>, style='{kind}').fit_transform(tri)"
    if style.endswith("weight=0.6"):
        src = (
            "DevelopmentConstant(patterns=<0.6 s + 0.4 Development().ldf_, blended by the "
            "script>, style='ldf').fit_transform(tri)"
        )
    if tail is not None:
        out = cl.TailConstant(tail=tail).fit_transform(out)
        src = f"TailConstant(tail={tail}).fit_transform({src})"
    return selected, out, src


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value, source):
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"NaN for {dataset} {method} {quantity} {arg}")
        rows.append([dataset, method, quantity, arg, fmt(value), 0, REL, source])

    def common(name, method, tri, selected, dev, src):
        ages = [int(a) for a in tri.development.values]
        for age, v in selected.items():
            emit(name, method, "pattern", age, v, "input: the selected pattern")
        ldf = dev.ldf_.values[0, 0, 0, :]
        cdf = dev.cdf_.values[0, 0, 0, :]
        for k, age in enumerate(ages[:-1]):
            emit(name, method, "ldf", age, ldf[k], f"chainladder-python {VER}: {src}.ldf_")
        for k, age in enumerate(ages[: len(cdf)]):
            emit(name, method, "cdf", age, cdf[k], f"chainladder-python {VER}: {src}.cdf_")

    for name, column in CHAIN_LADDER:
        tri = read_triangle(name, [column])
        origins = [str(d.year) for d in tri.origin.to_timestamp()]
        for style in STYLES:
            method = f"chain_ladder;{style}"
            selected, dev, src = developed(tri, style)
            common(name, method, tri, selected, dev, src)
            ult = cl.Chainladder().fit(dev).ultimate_.values[0, 0, :, 0]
            full = f"chainladder-python {VER}: Chainladder().fit({src})"
            for i, o in enumerate(origins):
                emit(name, method, "ultimate", o, ult[i], full + ".ultimate_")
            emit(name, method, "total_ultimate", "", ult.sum(), full + ".ultimate_.sum()")

    name = "clrd_wkcomp"
    tri = read_triangle(name, ["paid", "premium"])
    paid = tri["paid"]
    exposure = tri["premium"].latest_diagonal
    origins = [str(d.year) for d in paid.origin.to_timestamp()]
    for label, make in EXPOSURE_METHODS:
        for style in STYLES:
            method = f"{label};{style}"
            selected, dev, src = developed(paid, style)
            common(name, method, paid, selected, dev, src)
            model = make()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(dev, sample_weight=exposure)
            ult = model.ultimate_.values[0, 0, :, 0]
            full = (
                f"chainladder-python {VER}: {type(model).__name__}(...).fit({src}, "
                "sample_weight=premium.latest_diagonal)"
            )
            for i, o in enumerate(origins):
                emit(name, method, "ultimate", o, ult[i], full + ".ultimate_")
            emit(name, method, "total_ultimate", "", ult.sum(), full + ".ultimate_.sum()")

    with open(OUT, "w", newline="") as f:
        f.write(
            "# Generated by validation/scripts/reserving_patterns_python.py with "
            f"chainladder-python {VER}.\n"
        )
        f.write("# Selected patterns through DevelopmentConstant: raa and genins (value) with the\n")
        f.write("# chain ladder, clrd_wkcomp (paid, exposure the latest premium) with the expected-loss\n")
        f.write("# methods. quantity pattern holds the inputs; see the script's docstring.\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
