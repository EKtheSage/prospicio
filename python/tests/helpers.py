"""Helpers shared by the Python tests (#845): the reference CSV reader and
the triangles of validation/data. Copies that only look alike but differ
(the `close` checks, each file's hand-built `tri`) stay in their files."""

import csv
from pathlib import Path

from prospicio.reserving import Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def read_csv(path):
    """The rows of a validation CSV as dicts, comment lines (`#`) skipped."""
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def dataset(name):
    """validation/data/<name>.csv as a triangle with one column, `values`."""
    rows = read_csv(VALIDATION / "data" / f"{name}.csv")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )
