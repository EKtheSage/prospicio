"""Regenerate validation/reference/reserving_fit_apply_python.csv.

A development fitted on one book and applied to another, or to a latest
diagonal: chainladder-python's `fit(source).predict(target)`. Run from the
repository root:

    uv run --no-project --with chainladder==0.10.1 --with pandas python validation/scripts/reserving_fit_apply_python.py

The triangles are read as reserving_latest_diagonal_python.py reads them. The
source is always the full triangle; `method` names the source's settings and,
after `target=`, the book the fit is applied to:

* genins (value), the chain ladder with no tail, `TailConstant(1.05)`,
  `TailCurve()` and `TailCurve(curve='inverse_power')`, applied to
  `diagonal` (its own latest diagonal), `perturbed` (every value times
  1 + 0.01 (origin - 2000)), `double` (every value times 2), `from_2006` and
  `to_2004` (the origins from 2006 and to 2004), `shifted` (the origins
  moved 10 years later) and `early` (the cells valued before 2008);
* clrd_wkcomp (paid, exposure the latest premium), Bornhuetter-Ferguson,
  Benktander and Cape Cod, applied to `diagonal`, `perturbed` (paid and
  premium times 1 + 0.01 (origin - 1987), and the latest premium times 1.1
  more), `shifted` and `from_1993`.

Three of predict's results are not what is wanted, so they are taken
another way:

* Cape Cod on a latest diagonal: `predict(diagonal)` aligns every origin to
  the first factor to ultimate, so the diagonal is given with
  `.val_to_dev()`;
* `early`: predict values the target at the source's valuation and gives NaN,
  so the expected values are `DevelopmentConstant(<source cdf_>,
  style='cdf')` on the target, with the chain ladder;
* a target with a missing latest cell is not here: predict gives that origin
  NaN, and the Rust tests check it by hand.

Rows with quantity `pattern` are the source's `cdf_` at its ages (the last one
the tail factor), and `tail` its tail factor. `arg` is the age in months for
pattern, the origin year for per-origin quantities, empty for totals. A NaN
anywhere is an error.
"""

import csv
import math
import warnings

import chainladder as cl
import numpy as np
import pandas as pd

OUT = "validation/reference/reserving_fit_apply_python.csv"
VER = cl.__version__
REL = 1e-9

TAILS = [
    ("tail=none", None, "no tail"),
    ("tail=constant;factor=1.05", lambda: cl.TailConstant(1.05), "TailConstant(1.05)"),
    ("tail=curve;curve=exponential", lambda: cl.TailCurve(), "TailCurve()"),
    (
        "tail=curve;curve=inverse_power",
        lambda: cl.TailCurve(curve="inverse_power"),
        "TailCurve(curve='inverse_power')",
    ),
]

EXPOSURE_METHODS = [
    ("bornhuetter_ferguson;apriori=0.6", lambda: cl.BornhuetterFerguson(apriori=0.6)),
    ("benktander;apriori=0.6;n_iters=3", lambda: cl.Benktander(apriori=0.6, n_iters=3)),
    ("cape_cod;trend=0;decay=1", lambda: cl.CapeCod()),
    ("cape_cod;trend=0.02;decay=0.75", lambda: cl.CapeCod(trend=0.02, decay=0.75)),
]


def read_triangle(name, columns, scale=None, shift=0):
    """The triangle of `name`, with every value of `columns` times
    `1 + scale (origin - first origin + 1)` and the origins `shift` years
    later when asked."""
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    if scale is not None:
        k = df["origin"] - df["origin"].min() + 1
        for c in columns:
            df[c] = df[c] * (1 + scale * k)
    df["origin"] = df["origin"] + shift
    df["origin"] = df["origin"].astype(str)
    df["valuation"] = (df["origin"].astype(int) + df["development"] // 12 - 1).astype(str)
    return cl.Triangle(
        df, origin="origin", development="valuation", columns=columns, cumulative=True
    )


def years(tri):
    return [str(d.year) for d in tri.origin.to_timestamp()]


def main():
    rows = []

    def emit(dataset, method, quantity, arg, value, source):
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"NaN for {dataset} {method} {quantity} {arg}")
        rows.append([dataset, method, quantity, arg, repr(value), 0, REL, source])

    def ultimates(dataset, method, fitted, src):
        ult = fitted.ultimate_.values[0, 0, :, 0]
        for i, o in enumerate(years(fitted.ultimate_)):
            emit(dataset, method, "ultimate", o, ult[i], src + ".ultimate_")
        emit(dataset, method, "total_ultimate", "", ult.sum(), src + ".ultimate_.sum()")

    # genins: the chain ladder, with each tail.
    full = read_triangle("genins", ["value"])
    ages = [int(a) for a in full.development.values]
    targets = {
        "diagonal": (full.latest_diagonal, "latest_diagonal"),
        "perturbed": (read_triangle("genins", ["value"], scale=0.01), "<perturbed>"),
        "double": (full * 2, "2 * triangle"),
        "from_2006": (full[full.origin >= "2006"], "triangle[origin >= 2006]"),
        "to_2004": (full[full.origin <= "2004"], "triangle[origin <= 2004]"),
        "shifted": (read_triangle("genins", ["value"], shift=10), "<origins + 10>"),
    }
    early = full[full.valuation < "2008"]
    for label, make, name in TAILS:
        steps = [("dev", cl.Development())]
        if make is not None:
            steps.append(("tail", make()))
        steps.append(("cl", cl.Chainladder()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = cl.Pipeline(steps).fit(full)
        model = pipe.named_steps.cl
        method = f"chain_ladder;{label}"
        cdf = model.cdf_.values[0, 0, 0, : len(ages)]
        for age, c in zip(ages, cdf):
            emit("genins", method, "pattern", age, c, f"chainladder-python {VER}: {name}, cdf_")
        emit("genins", method, "tail", "", cdf[-1], f"chainladder-python {VER}: {name}, cdf_ at 120")
        src = f"chainladder-python {VER}: Pipeline(Development(), {name}, Chainladder()).fit(genins)"
        chosen = targets if make is None else {k: targets[k] for k in ["diagonal", "perturbed"]}
        for target, (tri, what) in chosen.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = pipe.predict(tri)
            ultimates("genins", f"{method};target={target}", fitted, f"{src}.predict({what})")
        if make is None:
            # predict values `early` at the source's valuation (NaN), so the
            # source's factors to ultimate are given to DevelopmentConstant.
            given = dict(zip(ages, [float(c) for c in cdf]))
            dc = cl.DevelopmentConstant(patterns=given, style="cdf").fit_transform(early)
            fitted = cl.Chainladder().fit(dc)
            ultimates(
                "genins",
                f"{method};target=early",
                fitted,
                f"chainladder-python {VER}: Chainladder().fit(DevelopmentConstant(patterns="
                "<source cdf_>, style='cdf').fit_transform(triangle[valuation < 2008]))",
            )

    # clrd_wkcomp: the expected-loss methods on paid, exposure the latest
    # premium.
    name = "clrd_wkcomp"
    tri = read_triangle(name, ["paid", "premium"])
    paid, premium = tri["paid"], tri["premium"].latest_diagonal
    perturbed = read_triangle(name, ["paid", "premium"], scale=0.01)
    shifted = read_triangle(name, ["paid", "premium"], shift=10)
    targets = {
        "diagonal": (paid.latest_diagonal, premium, "paid.latest_diagonal"),
        "perturbed": (
            perturbed["paid"],
            perturbed["premium"].latest_diagonal * 1.1,
            "<perturbed paid>, sample_weight=<perturbed premium> * 1.1",
        ),
        "shifted": (
            shifted["paid"],
            shifted["premium"].latest_diagonal,
            "<origins + 10>",
        ),
        "from_1993": (
            paid[paid.origin >= "1993"],
            premium[premium.origin >= "1993"],
            "paid[origin >= 1993]",
        ),
    }
    for label, make in EXPOSURE_METHODS:
        model = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(paid, sample_weight=premium)
        ages = [int(a) for a in paid.development.values]
        cdf = model.cdf_.values[0, 0, 0, : len(ages)]
        src = f"chainladder-python {VER}: {type(model).__name__}(...).fit(paid, sample_weight=premium.latest_diagonal)"
        for age, c in zip(ages, cdf):
            emit(name, label, "pattern", age, c, src + ".cdf_")
        cape_cod = label.startswith("cape_cod")
        for target, (x, w, what) in targets.items():
            if target == "from_1993" and not cape_cod:
                continue
            if cape_cod and target == "diagonal":
                # Otherwise every origin is aligned to the first cdf.
                x, what = x.val_to_dev(), what + ".val_to_dev()"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.predict(x, sample_weight=w)
            method = f"{label};target={target}"
            ultimates(name, method, fitted, f"{src}.predict({what})")
            if cape_cod:
                apriori = fitted.apriori_.values[0, 0, :, 0]
                for i, o in enumerate(years(fitted.apriori_)):
                    emit(name, method, "apriori", o, apriori[i], f"{src}.predict({what}).apriori_")

    with open(OUT, "w", newline="") as f:
        f.write(
            "# Generated by validation/scripts/reserving_fit_apply_python.py with "
            f"chainladder-python {VER}.\n"
        )
        f.write("# A development fitted on the full triangle and applied to another book or a latest\n")
        f.write("# diagonal (fit(source).predict(target)): genins (value) with the chain ladder and\n")
        f.write("# tails, clrd_wkcomp (paid, exposure the latest premium) with the expected-loss methods.\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
