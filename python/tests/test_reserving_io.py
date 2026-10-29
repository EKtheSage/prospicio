"""Triangles read from files (docs/design/triangle.md, decision 6).

The Rust readers are checked in crates/prospicio-reserving/src/io.rs and
validation/tests/reserving_io.rs; these tests check the binding:
``Triangle.read_csv`` in both layouts, ``Triangle.read_excel`` on the
committed workbook, the errors, and ``schedule_p`` against the committed
square of group 353 when the cas-schedule-p package is installed.
"""

import csv
import math
import sys
from pathlib import Path

import pytest

from prospicio.reserving import Triangle, schedule_p

DATA = Path(__file__).resolve().parents[2] / "validation" / "data"


def raa():
    return Triangle.read_csv(DATA / "raa.csv", "origin", "development", comment="#")


def cells(tri):
    """Shape, origins, ages and every value, with nan as None."""
    values = [
        [[[None if math.isnan(v) else v for v in row] for row in column] for column in index]
        for index in tri.values
    ]
    return tri.shape, tri.origins, tri.development, tri.valuation, values


def test_read_csv_long_equals_from_frame():
    rows = list(csv.DictReader(line for line in open(DATA / "raa.csv") if not line.startswith("#")))
    frame = {
        "origin": [int(r["origin"]) for r in rows],
        "development": [int(r["development"]) for r in rows],
        "value": [float(r["value"]) for r in rows],
    }
    expected = Triangle.from_frame(frame, "origin", "development", "value")
    tri = raa()
    assert tri.columns == ["value"]
    assert cells(tri) == cells(expected)


def test_read_csv_wide_and_ages(tmp_path):
    tri = raa()
    path = tmp_path / "wide.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["origin"] + tri.development)
        for o, row in zip(tri.origins, tri.values[0][0]):
            w.writerow([o] + ["" if math.isnan(v) else v for v in row])
    wide = Triangle.read_csv(path, "origin", columns="paid")
    assert wide.columns == ["paid"]
    assert cells(wide) == cells(tri)

    path.write_text("year,first,second\n2020,100,150\n2021,110,\n")
    given = Triangle.read_csv(str(path), "year", ages=[12, 24])
    assert given.columns == ["values"]
    assert given.development == [12, 24]


def test_read_csv_keys_and_valuations(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text(
        "company,ay,dy,paid,incurred\n353,1988,1988,10,20\n353,1988,1989,15,22\n"
        "353,1989,1989,12,\n388,1988,1988,7,NA\n"
    )
    tri = Triangle.read_csv(path, "ay", "dy", keys="company", development_is_valuation=True)
    assert tri.keys == ["company"]
    assert tri.index == ["353", "388"]
    assert tri.columns == ["paid", "incurred"]
    assert tri.development == [12, 24]


def test_read_excel_both_sheets():
    expected = cells(raa())
    wide = Triangle.read_excel(DATA / "raa.xlsx", "origin", columns="paid")
    assert cells(wide) == expected
    long = Triangle.read_excel(DATA / "raa.xlsx", "origin", "development", "paid", sheet="long")
    assert cells(long) == expected


def test_errors_name_the_cell(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("year,age,paid\n2020,12,100\n2020,24,1O0\n")
    with pytest.raises(ValueError, match='line 3, column "paid": "1O0" is not a number'):
        Triangle.read_csv(path, "year", "age")
    with pytest.raises(ValueError, match='no column named "dev"'):
        Triangle.read_csv(path, "year", "dev")
    with pytest.raises(ValueError, match="wide layout"):
        Triangle.read_csv(path, "year", "age", ages=[12])
    with pytest.raises(ValueError, match="one measure"):
        Triangle.read_csv(path, "year", columns=["a", "b"])
    with pytest.raises(ValueError, match="one ASCII character"):
        Triangle.read_csv(path, "year", "age", comment="##")
    with pytest.raises(ValueError, match=r'cell C3 \(column "paid"\) of sheet "bad"'):
        Triangle.read_excel(DATA / "raa.xlsx", "origin", "development", "paid", sheet="bad")
    with pytest.raises(ValueError, match='has no sheet "none"'):
        Triangle.read_excel(DATA / "raa.xlsx", "origin", sheet="none")


def test_schedule_p_without_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "cas_schedule_p", None)
    with pytest.raises(ImportError, match="pip install cas-schedule-p pyarrow"):
        schedule_p("ca", company=353)


def test_schedule_p_group_353_matches_the_committed_square():
    pytest.importorskip("cas_schedule_p")
    pytest.importorskip("pyarrow")
    # calibration_panel.py's own check: group 353 commercial auto equals
    # validation/data/clrd_comauto_353.csv (paid, case incurred, net
    # premium) cell by cell.
    square = schedule_p("ca", company=353, columns=["paid", "case_incurred", "premium"], lower=True)
    assert square.keys == ["company"]
    assert square.index == ["353"]
    assert square.shape == (1, 3, 10, 10)
    expected = Triangle.read_csv(
        DATA / "clrd_comauto_353.csv",
        "origin",
        "development",
        ["paid", "incurred", "premium"],
        comment="#",
    )
    assert cells(square) == cells(expected)

    upper = schedule_p("comauto", company="353", columns="paid")
    assert sum(not math.isnan(v) for row in upper.values[0][0] for v in row) == 55
    assert upper.valuation.year == 1997

    line = schedule_p("ca", columns="paid")
    assert len(line.index) > 50
    with pytest.raises(ValueError, match="unknown Schedule P line"):
        schedule_p("auto")
    with pytest.raises(ValueError, match="unknown Schedule P column"):
        schedule_p("ca", columns="reported")
    with pytest.raises(ValueError, match="no rows for company 1 "):
        schedule_p("ca", company=1)
