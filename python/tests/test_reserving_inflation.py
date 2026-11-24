"""Indexation by a calendar index and re-inflation (issue #979,
docs/design/reserving-inflation.md, PR 2): the Python wrappers of
validation/tests/reserving_inflation.rs."""

import datetime
import pickle

import pytest
from helpers import VALIDATION, dataset, read_csv
from prospicio.reserving import (
    CalendarIndex,
    ChainLadder,
    FutureCells,
    Indexation,
    OdpGlm,
    Reinflated,
    Triangle,
)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def cells_of(tri, column=0):
    """The first segment's values of one column as a list of rows."""
    return tri.values[0][column]


def same_cells(got, want, tol=1e-12):
    for row, expected in zip(cells_of(got), cells_of(want)):
        for a, b in zip(row, expected):
            assert (a != a and b != b) or close(a, b, tol)


def test_the_index_reads_its_periods_and_extends_at_the_rate():
    cpi = CalendarIndex("2020", [100.0, 104.0, 110.0], future_rate=0.05)
    assert cpi.value("2021-06") == 104.0
    assert cpi.value(datetime.date(2022, 12, 31)) == 110.0
    assert close(cpi.value("2024-01"), 110.0 * 1.05**2)
    assert cpi.periods == ["2020", "2021", "2022"]
    assert cpi.values == [100.0, 104.0, 110.0]
    assert cpi.future_rate == 0.05
    q = CalendarIndex("2020Q2", [100.0, 101.0, 103.0])
    assert q.periods == ["2020Q2", "2020Q3", "2020Q4"]
    assert q.value("2020-06") == 100.0
    assert q.future_rate is None
    assert CalendarIndex(2020, [1.0]) == CalendarIndex("2020", [1.0])
    assert pickle.loads(pickle.dumps(cpi)) == cpi
    assert "CalendarIndex(" in repr(cpi)


def test_a_unit_index_restates_to_the_same_triangle_and_chain_ladder():
    tri = dataset("genins")
    unit = Indexation(CalendarIndex("2001", [1.0], future_rate=0.0), "2010-12")
    restated = unit.restate(tri, "values")
    assert isinstance(restated, Triangle)
    same_cells(restated, tri)
    want = ChainLadder().fit(tri, "values").reserve
    got = ChainLadder().fit(restated, "values").reserve
    assert all(close(a, b) for a, b in zip(got, want))


def test_inflate_undoes_restate_and_the_factor_is_the_index_ratio():
    tri = dataset("genins")
    index = CalendarIndex(
        "2001", [100.0, 103.0, 101.5, 108.0, 112.0, 111.0, 118.0, 125.0, 131.0, 140.0]
    )
    at = Indexation(index, "2006-07")
    assert at.base == "2006-07"
    assert at.index == index
    assert close(at.factor("2008-03"), 125.0 / 111.0)
    back = at.inflate(at.restate(tri, "values"), "values")
    for row, want in zip(cells_of(back), cells_of(tri)):
        for a, b in zip(row, want):
            assert (a != a and b != b) or close(a, b, 1e-12)
    # The first cell is paid in 2001: restated by 111 / 100.
    first = cells_of(at.restate(tri, "values"))[0][0]
    assert close(first, cells_of(tri)[0][0] * 111.0 / 100.0)
    assert pickle.loads(pickle.dumps(at)) == at


def test_chain_ladder_cells_sum_to_its_reserves_and_reinflate_at_a_constant_rate():
    tri = dataset("genins")
    fit = ChainLadder().fit(tri, "values")
    cells = FutureCells.from_chain_ladder(fit)
    assert len(cells) == 45
    assert all(close(a, b) for a, b in zip(cells.reserves(), fit.reserve))
    unit = Indexation(CalendarIndex("2001", [1.0], future_rate=0.0), "2010-12")
    out = unit.reinflate(cells)
    assert all(close(a, b) for a, b in zip(out.reserves, fit.reserve))
    # A restated triangle's chain ladder, re-inflated at 4% a year.
    cpi = Indexation(CalendarIndex("2001", [1.0] * 10, future_rate=0.04), "2010-12")
    again = cpi.reinflate(FutureCells.from_chain_ladder(ChainLadder().fit(cpi.restate(tri, "values"), "values")))
    assert again.total_reserve > again.total_base_reserve
    for base, got, month in zip(
        FutureCells.from_chain_ladder(ChainLadder().fit(cpi.restate(tri, "values"), "values")).cells["value"],
        again.cells.cells["value"],
        again.cells.cells["valuation"],
    ):
        years = (int(month[:4]) - 2010) + (int(month[5:]) - 12) / 12
        assert close(got, base * 1.04**years, 1e-12)


def test_chain_ladder_cells_of_several_segments_need_a_choice():
    rows = read_csv(VALIDATION / "data" / "genins.csv")
    stacked = Triangle.from_long(
        origin=[int(r["origin"]) for r in rows] * 2,
        development=[int(r["development"]) for r in rows] * 2,
        values={"values": [float(r["value"]) for r in rows] * 2},
        keys={"lob": ["auto"] * len(rows) + ["home"] * len(rows)},
    )
    fit = ChainLadder().fit(stacked, "values")
    with pytest.raises(ValueError):
        FutureCells.from_chain_ladder(fit)
    one = FutureCells.from_chain_ladder(fit, segment={"lob": "home"})
    assert len(one) == 45


def test_odp_glm_cells_reinflate_at_a_constant_rate():
    tri = dataset("genins")
    fit = OdpGlm().fit(tri, "values")
    cells = FutureCells.from_odp_glm(fit)
    assert all(close(a, b) for a, b in zip(cells.reserves(), fit.reserves))
    assert len(cells) == len(cells.cells["value"]) == 45
    assert cells.origins[0] == "2001"
    # A zero rate re-inflates nothing.
    flat = Indexation(CalendarIndex("2001", [1.0], future_rate=0.0), "2010-12")
    out = flat.reinflate(cells)
    assert isinstance(out, Reinflated)
    assert close(out.total_reserve, fit.total_reserve)
    assert close(out.total_base_reserve, fit.total_reserve)
    # A 4% rate from 2010 grows each cell by its years past 2010-12.
    at = Indexation(CalendarIndex("2001", [1.0] * 10, future_rate=0.04), "2010-12")
    grown = at.reinflate(cells)
    assert grown.total_reserve > grown.total_base_reserve
    assert len(grown.reserves) == len(grown.base_reserves) == 10
    for base, cell in zip(cells.cells["value"], grown.cells.cells["value"]):
        assert cell >= base - 1e-12


def test_taylor_1977_example_1_reinflation():
    r = [0.5835, 0.2878, 0.0866, 0.0421]
    lam = [86.4, 98.9, 102.0, 113.9]
    origins = ["2001", "2002", "2003", "2004"]
    origin, age, valuation, value = [], [], [], []
    for i, label in enumerate(origins):
        for j in range(4 - i, 4):
            origin.append(label)
            age.append(12 * (j + 1))
            valuation.append(f"{2001 + i + j}-12")
            value.append(r[j] * lam[3])
    cells = FutureCells(origins, origin, age, valuation, value)
    at_last = Indexation(CalendarIndex("2001", lam, future_rate=0.10), "2004-01")
    out = at_last.reinflate(cells)
    printed = [5.3, 10.8, 5.8, 36.1, 11.9, 6.4]
    got = out.cells.cells["value"]
    assert len(got) == len(printed)
    for g, w in zip(got, printed):
        assert abs(g - w) <= 0.06
    assert out.cells.cells["origin"] == ["2002", "2003", "2003", "2004", "2004", "2004"]
    assert out.cells.cells["age"][0] == 48
    assert out.cells.cells["valuation"][0] == "2005-12"
    assert close(sum(out.reserves), out.total_reserve)
    assert close(out.total_base_reserve, sum(value))
    assert out.reserves[0] == 0.0


def test_future_cells_from_columns():
    cells = FutureCells(
        ["2020", "2021"],
        ["2020", "2021", "2020"],
        [24, 24, 36],
        ["2021-12", datetime.date(2022, 12, 31), "2022-12"],
        [50.0, 60.0, 25.0],
    )
    assert cells.reserves() == [75.0, 60.0]
    assert cells.cells["valuation"] == ["2021-12", "2022-12", "2022-12"]
    assert cells.origins == ["2020", "2021"]
    assert len(cells) == 3


def test_errors():
    tri = dataset("genins")
    with pytest.raises(ValueError, match="positive"):
        CalendarIndex("2020", [1.0, 0.0])
    with pytest.raises(ValueError, match="at least one"):
        CalendarIndex("2020", [])
    with pytest.raises(ValueError, match="period"):
        CalendarIndex("twenty", [1.0])
    with pytest.raises(ValueError, match="future_rate"):
        CalendarIndex("2020", [1.0], future_rate=-1.0)
    index = CalendarIndex("2002", [1.0] * 9)
    with pytest.raises(ValueError, match="starts in 2002"):
        index.value("2001-06")
    with pytest.raises(ValueError, match="no future rate"):
        index.value("2020-01")
    with pytest.raises(TypeError, match="month"):
        index.value(2005)
    with pytest.raises(ValueError, match="starts in 2002"):
        Indexation(index, "2001-01")
    late = Indexation(index, "2005-01")
    with pytest.raises(ValueError, match="starts in 2002"):
        late.restate(tri, "values")
    with pytest.raises(ValueError, match="nope"):
        Indexation(CalendarIndex("2001", [1.0] * 10), "2005-01").restate(tri, "nope")
    short = Indexation(CalendarIndex("2001", [1.0] * 10), "2005-01")
    cells = FutureCells(["2009"], ["2009"], [24], ["2011-12"], [1.0])
    with pytest.raises(ValueError, match="no future rate"):
        short.reinflate(cells)
    with pytest.raises(ValueError, match="not one of the origins"):
        FutureCells(["2009"], ["2010"], [24], ["2011-12"], [1.0])
    with pytest.raises(ValueError, match="one entry per cell"):
        FutureCells(["2009"], ["2009"], [24, 36], ["2011-12"], [1.0])
