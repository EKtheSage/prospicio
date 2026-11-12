"""Helpers shared by the Python test files (#845).

Import what a file needs: ``from helpers import dataset, spearman``. pytest
puts this directory on ``sys.path`` (the tests are not a package), so the
import works under ``pytest tests`` and under pytest-xdist alike. Helpers
whose tolerance rule differs from file to file, such as each file's
``close()``, stay in their files.
"""

import csv
import math
from pathlib import Path

from prospicio.reserving import Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def read_csv(path):
    """The rows of a reference CSV as dicts, ``#`` comment lines skipped."""
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def data_rows(name):
    """The rows of ``validation/data/<name>.csv``."""
    return read_csv(VALIDATION / "data" / f"{name}.csv")


def dataset(name, key="values"):
    """The long-format triangle ``validation/data/<name>.csv`` (columns
    origin, development, value), its values under ``key``."""
    rs = data_rows(name)
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rs],
        development=[int(r["development"]) for r in rs],
        values={key: [float(r["value"]) for r in rs]},
    )


def ranks(xs):
    """Ordinal ranks from 0, ties in order of appearance."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def pearson(x, y):
    """Pearson's correlation."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    """Spearman's rank correlation: Pearson's on the ordinal ranks."""
    return pearson(ranks(x), ranks(y))
