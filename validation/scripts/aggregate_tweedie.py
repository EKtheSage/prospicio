"""Regenerate validation/reference/tweedie_aggregate.csv.

    uv run --python 3.13 --with aggregate==2.1.0 \
        python validation/scripts/aggregate_tweedie.py

The Tweedie parameter conversions of `aggregate` 2.1.0 (#775): one law
spelled three ways by `TweedieConversion`, from its reproductive
parameters (`from_dispersion(p, mean, dispersion)`), from its mean and
coefficient of variation (`from_cv(p, mean, cv)`) and from a Poisson count
with a gamma severity (`from_compound(frequency, sev_m, sev_cv)`). Each row
is one output (`quantity`: `p`, `mean`, `dispersion`, `cv`, `frequency`,
`sev_m`, `sev_cv`, `p0`) of one input (`distribution` the constructor,
`params` its arguments), at `1e-13` relative: the conversions are closed
forms (`p0 = exp(-frequency)` at `1e-13` times the frequency).
"""

import csv
import sys

import aggregate
from aggregate.tweedie import TweedieConversion

OUT = "validation/reference/tweedie_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
FIELDS = ["p", "mean", "dispersion", "cv", "frequency", "sev_m", "sev_cv", "p0"]

CASES = [
    ("from_dispersion", {"p": 1.5, "mean": 100.0, "dispersion": 2.0}),
    ("from_dispersion", {"p": 1.6, "mean": 500.0, "dispersion": 40.0}),
    ("from_dispersion", {"p": 1.05, "mean": 0.3, "dispersion": 1.2}),
    ("from_dispersion", {"p": 1.95, "mean": 2500.0, "dispersion": 0.8}),
    ("from_cv", {"p": 1.5, "mean": 100.0, "cv": 0.5}),
    ("from_cv", {"p": 1.3, "mean": 0.07, "cv": 4.0}),
    ("from_cv", {"p": 1.8, "mean": 1e6, "cv": 0.2}),
    ("from_compound", {"frequency": 10.0, "sev_m": 10.0, "sev_cv": 1.0}),
    ("from_compound", {"frequency": 0.05, "sev_m": 8000.0, "sev_cv": 1.7}),
    ("from_compound", {"frequency": 250.0, "sev_m": 40.0, "sev_cv": 0.3}),
]


def main():
    rows = []
    for ctor, params in CASES:
        t = getattr(TweedieConversion, ctor)(*params.values())
        label = ";".join(f"{k}={v!r}" for k, v in params.items())
        for field in FIELDS:
            # p0 = exp(-frequency) carries the rounding of the frequency
            # magnified by the frequency itself.
            rel = 1e-13 * max(1.0, t.frequency) if field == "p0" else 1e-13
            rows.append([ctor, label, field, "", "", repr(float(getattr(t, field))),
                         "0.0", repr(rel), SOURCE])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
