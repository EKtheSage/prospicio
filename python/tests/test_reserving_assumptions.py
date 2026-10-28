"""Mack's calendar-year effect and development-factor correlation tests
(Mack 1994, appendices H and G; ``docs/design/reserving-v02.md``, decision
26). The Rust core is checked against R ChainLadder and chainladder-python in
``validation/tests/reserving_assumptions.rs``; these tests check the
bindings' arguments, results, repr and errors against the same R
reference through Python."""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import (
    CalendarYearTest,
    DevelopmentCorrelationTest,
    Mack,
    Triangle,
    calendar_year_effect,
    development_correlation,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def load(name):
    rows = read_csv(VALIDATION / "data" / f"{name}.csv")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


R = {}
for row in read_csv(VALIDATION / "reference" / "reserving_assumptions_r.csv"):
    R[(row["dataset"], row["test"], row["quantity"], row["arg"])] = float(row["expected"])


@pytest.mark.parametrize("name", ["raa", "genins", "abc"])
def test_calendar_year_effect_matches_r(name):
    tri = load(name)
    for ci in (0.95, 0.9):
        if ci == 0.95:
            cy = calendar_year_effect(tri, "values")
        else:
            cy = calendar_year_effect(tri, "values", ci)
        assert isinstance(cy, CalendarYearTest)
        test = f"calendar_year_effect;ci={ci}"
        assert cy.z == R[(name, test, "total_z", "")]
        for got, quantity in [
            (cy.expected, "total_expected"),
            (cy.variance, "total_variance"),
            (cy.lower, "lower"),
            (cy.upper, "upper"),
        ]:
            assert got == pytest.approx(R[(name, test, quantity, "")], rel=1e-12)
        assert cy.passes == bool(R[(name, test, "passes", "")])
        assert cy.ci == ci
        for d in cy.diagonals:
            year = d["valuation"][:4]
            assert d["valuation"].endswith("-12")
            for q in ["small", "large", "z", "n", "m"]:
                assert d[q] == R[(name, test, q, year)]
            assert d["expected"] == pytest.approx(R[(name, test, "expected", year)], rel=1e-12)
            assert d["variance"] == pytest.approx(R[(name, test, "variance", year)], rel=1e-12)
        assert len(cy.diagonals) == R[(name, test, "n_diagonals", "")]


@pytest.mark.parametrize("name", ["raa", "genins", "abc"])
def test_development_correlation_matches_r(name):
    tri = load(name)
    for ci in (0.5, 0.75):
        dc = development_correlation(tri, "values", ci=ci)
        assert isinstance(dc, DevelopmentCorrelationTest)
        test = f"development_correlation;ci={ci}"
        for got, quantity in [
            (dc.t, "t"),
            (dc.variance, "variance"),
            (dc.lower, "lower"),
            (dc.upper, "upper"),
        ]:
            assert got == pytest.approx(R[(name, test, quantity, "")], rel=1e-12, abs=1e-15)
        assert dc.passes == bool(R[(name, test, "passes", "")])
        assert len(dc.pairs) == R[(name, test, "n_pairs", "")]
        for i, p in enumerate(dc.pairs):
            want = R[(name, test, "t_k", str(i))]
            assert p["t"] == pytest.approx(want, rel=1e-12, abs=1e-15)
            assert p["weight"] == R[(name, test, "weight", str(i))]
            assert p["n"] == p["weight"] + 1
            assert p["age"] == 12 * (i + 1)


def test_mack_prints_raa():
    # Mack (1994), appendices G and H, on RAA: T = .070, Z = 14,
    # E(Z) = 12.875, Var(Z) = 3.9785.
    fit = Mack().fit(load("raa"), "values")
    dc = fit.development_correlation()
    assert dc.t == pytest.approx(0.070, abs=5e-4)
    assert dc.variance == pytest.approx(1 / 28)
    assert dc.ci == 0.5
    cy = fit.calendar_year_effect()
    assert (cy.z, cy.expected, cy.ci) == (14.0, 12.875, 0.95)
    assert cy.variance == pytest.approx(3.9785, abs=5e-5)
    assert cy.passes and dc.passes


def test_a_fit_tests_its_kept_link_ratios():
    tri = load("raa")
    whole = calendar_year_effect(tri, "values")
    assert Mack().fit(tri, "values").calendar_year_effect().z == whole.z
    kept = Mack(n_periods=4).fit(tri, "values").calendar_year_effect()
    assert [d["valuation"] for d in kept.diagonals] == [
        "1987-12",
        "1988-12",
        "1989-12",
        "1990-12",
    ]
    dropped = Mack(drop_high=1, drop_low=1).fit(tri, "values")
    n_kept = sum(d["n"] for d in dropped.calendar_year_effect().diagonals)
    assert n_kept < sum(d["n"] for d in whole.diagonals)
    assert all(p["n"] >= 2 for p in dropped.development_correlation().pairs)


def test_repr_and_frame():
    tri = load("raa")
    cy = calendar_year_effect(tri, "values")
    assert repr(cy).startswith("CalendarYearTest(z=14.0, expected=12.875,")
    assert repr(cy).endswith("passes=True)")
    dc = development_correlation(tri, "values")
    assert repr(dc).startswith("DevelopmentCorrelationTest(t=")
    pd = pytest.importorskip("pandas")
    frame = cy.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "valuation",
        "small",
        "large",
        "z",
        "n",
        "m",
        "expected",
        "variance",
    ]
    assert frame["z"].sum() == 14
    pairs = dc.to_frame()
    assert list(pairs.columns) == ["age", "n", "t", "weight"]
    assert pairs["t"].iloc[0] == pytest.approx(4 / 21)


def test_errors():
    tri = load("raa")
    for ci in (0.0, 1.0, math.nan):
        with pytest.raises(ValueError, match="ci"):
            calendar_year_effect(tri, "values", ci)
        with pytest.raises(ValueError, match="ci"):
            development_correlation(tri, "values", ci)
    with pytest.raises(ValueError):
        calendar_year_effect(tri, "missing")
    small = Triangle.from_long(
        origin=[2020, 2020, 2021], development=[12, 24, 12], values={"values": [1.0, 2.0, 1.0]}
    )
    with pytest.raises(ValueError):
        calendar_year_effect(small, "values")
    rows = read_csv(VALIDATION / "data" / "raa.csv")
    keyed = Triangle.from_long(
        origin=[int(r["origin"]) for r in rows] * 2,
        development=[int(r["development"]) for r in rows] * 2,
        values={"values": [float(r["value"]) for r in rows] * 2},
        keys={"lob": ["a"] * len(rows) + ["b"] * len(rows)},
    )
    fit = Mack().fit(keyed, "values")
    with pytest.raises(ValueError, match="segment"):
        fit.calendar_year_effect()
    with pytest.raises(ValueError, match="segment"):
        fit.development_correlation()
