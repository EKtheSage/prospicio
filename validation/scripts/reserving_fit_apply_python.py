"""Regenerate validation/reference/reserving_fit_apply_python.csv.

A development fitted on one book and applied to another, or to a latest
diagonal: chainladder-python's `fit(source).predict(target)`. Run from the
repository root:

    uv run --no-project --with chainladder==0.10.1 --with pandas python validation/scripts/reserving_fit_apply_python.py

The triangles are read as reserving_latest_diagonal_python.py reads them. The
source is always the full triangle; `method` names the source's settings and,
after `target=`, the book the fit is applied to:

* genins (value), the chain ladder with no tail, `TailConstant(1.05)`,
  `TailCurve()` and `TailCurve(curve='inverse_power')`, each applied to
  `diagonal` (its own latest diagonal) and `perturbed` (every value times
  1 + 0.01 (origin - 2000)); the source with no tail also to `double`
  (every value times 2), `from_2006` and `to_2004` (the origins from 2006
  and to 2004), `shifted` (the origins moved 10 years later) and `early`
  (the cells valued before 2008);
* clrd_wkcomp (paid, exposure the latest premium), Bornhuetter-Ferguson,
  Benktander and Cape Cod, applied to `diagonal`, `perturbed` (paid and
  premium times 1 + 0.01 (origin - 1987), and the latest premium times 1.1
  more) and `shifted`; Cape Cod also to `from_1993` and `diagonal_raw`.

Three of predict's results are not what is wanted. The script asserts each
(it raises if chainladder-python stops behaving so) and takes the wanted
result another way:

* Cape Cod on a latest diagonal: `predict(diagonal)` aligns every origin to
  the first factor to ultimate, so `diagonal` is given with `.val_to_dev()`;
  the rows of `diagonal_raw` record the apriori of predict on the diagonal
  as it is, which the Rust tests reproduce with every origin at the first
  age;
* `early`: predict values the target at the source's valuation and gives NaN
  for every origin, so the expected values are
  `DevelopmentConstant(<source cdf_>, style='cdf')` on the target, with the
  chain ladder;
* a target with GenIns' 2005 cell at 72 months removed: predict gives that
  origin NaN and the others their ultimates, so it has no rows, and the
  Rust tests check it by hand.

The curve tails are not refitted on the target: quantity
`refitted_tail_perturbed` is the tail factor of the same pipeline fitted on
`perturbed` itself, which differs from the source's.

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


def read_triangle(name, columns, scale=None, shift=0, drop=None):
    """The triangle of `name`, with every value of `columns` times
    `1 + scale (origin - first origin + 1)`, the origins `shift` years
    later and the cell `drop` (origin, age) removed when asked."""
    df = pd.read_csv(f"validation/data/{name}.csv", comment="#")
    if drop is not None:
        df = df[~((df["origin"] == drop[0]) & (df["development"] == drop[1]))].copy()
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
        if label.startswith("tail=curve"):
            # The same pipeline fitted on the perturbed book has a tail of its
            # own, which predict does not use.
            steps = [("dev", cl.Development()), ("tail", make()), ("cl", cl.Chainladder())]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refit = cl.Pipeline(steps).fit(targets["perturbed"][0]).named_steps.cl
            emit(
                "genins",
                method,
                "refitted_tail_perturbed",
                "",
                refit.cdf_.values[0, 0, 0, len(ages) - 1],
                f"chainladder-python {VER}: Pipeline(Development(), {name}, Chainladder()).fit(<perturbed>), cdf_ at 120",
            )
        src = f"chainladder-python {VER}: Pipeline(Development(), {name}, Chainladder()).fit(genins)"
        chosen = targets if make is None else {k: targets[k] for k in ["diagonal", "perturbed"]}
        for target, (tri, what) in chosen.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = pipe.predict(tri)
            ultimates("genins", f"{method};target={target}", fitted, f"{src}.predict({what})")
        if make is None:
            # The traps of predict: a missing latest cell and an earlier
            # valuation give NaN.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                missing = pipe.predict(read_triangle("genins", ["value"], drop=(2005, 72)))
                early_predict = pipe.predict(early)
            ult = missing.ultimate_.values[0, 0, :, 0]
            nan = [o for o, u in zip(years(missing.ultimate_), ult) if math.isnan(u)]
            if nan != ["2005"]:
                raise AssertionError(f"predict with 2005 at 72 months removed: NaN for {nan}, not 2005 alone")
            if not np.isnan(early_predict.ultimate_.values[0, 0, :, 0]).all():
                raise AssertionError("predict on the book valued before 2008 is no longer NaN")
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
        "diagonal_raw": (paid.latest_diagonal, premium, "paid.latest_diagonal"),
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
        aprioris = {}
        for target, (x, w, what) in targets.items():
            if target in ["from_1993", "diagonal_raw"] and not cape_cod:
                continue
            if cape_cod and target == "diagonal":
                # Otherwise every origin is aligned to the first cdf
                # (`diagonal_raw`).
                x, what = x.val_to_dev(), what + ".val_to_dev()"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.predict(x, sample_weight=w)
            method = f"{label};target={target}"
            if target != "diagonal_raw":
                # Only the apriori of the raw diagonal is recorded: the Rust
                # tests reproduce it, not its ultimates.
                ultimates(name, method, fitted, f"{src}.predict({what})")
            if cape_cod:
                apriori = fitted.apriori_.values[0, 0, :, 0]
                for i, o in enumerate(years(fitted.apriori_)):
                    emit(name, method, "apriori", o, apriori[i], f"{src}.predict({what}).apriori_")
                aprioris[target] = apriori
        if cape_cod and np.allclose(aprioris["diagonal"], aprioris["diagonal_raw"]):
            raise AssertionError(f"{label}: predict on the raw latest diagonal now reads each origin's age")

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
