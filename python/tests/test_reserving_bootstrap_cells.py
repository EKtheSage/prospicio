"""The bootstraps' future cells (issue #979, docs/design/reserving-inflation.md,
"The bootstraps' cells", decisions A to I): ``fit(cells=True)`` of
``OdpBootstrap`` and ``MackBootstrap`` and the fits' ``cells``, the Python
wrappers of validation/tests/reserving_bootstrap_cells.rs."""

from collections import defaultdict

import pytest
from helpers import VALIDATION, dataset, read_csv
from prospicio.reserving import (
    CalendarIndex,
    IndexPaths,
    Indexation,
    MackBootstrap,
    OdpBootstrap,
    Triangle,
)

BOOTSTRAPS = {"odp": OdpBootstrap, "mack": MackBootstrap}


def close(a, b, rel):
    return abs(a - b) <= rel * max(1.0, abs(b))


def stacked(parts):
    """Segments under ``lob``: each part a name, the rows of
    validation/data/<file>.csv (or rows given as tuples) and a multiplier."""
    keys, origin, development, values = [], [], [], []
    for lob, rows, times in parts:
        for o, d, v in rows:
            keys.append(lob)
            origin.append(o)
            development.append(d)
            values.append(times * v)
    return Triangle.from_long(origin, development, {"values": values}, keys={"lob": keys})


def rows_of(name):
    return [
        (int(r["origin"]), int(r["development"]), float(r["value"]))
        for r in read_csv(VALIDATION / "data" / f"{name}.csv")
    ]


def twins(name):
    """validation/data/<name>.csv under ``lob`` A, and doubled under B."""
    rows = rows_of(name)
    return stacked([("A", rows, 1.0), ("B", rows, 2.0)])


def square():
    """A 3 by 3 annual triangle with every cell observed."""
    return Triangle.from_long(
        [2020] * 3 + [2021] * 3 + [2022] * 3,
        [12, 24, 36] * 3,
        [100.0, 150.0, 165.0, 110.0, 170.0, 182.0, 120.0, 175.0, 196.0],
    )


def check_sums(reserves, cells, rel=1e-9):
    """Each origin's cells sum to its reserve in every simulation, by hand
    and through ``aggregate``; an origin without a cell has a reserve of 0."""
    assert cells.dims == reserves.dims + ["development"]
    owner = defaultdict(list)
    for j, key in enumerate(cells.components()):
        owner[tuple(key[:-1])].append(j)
    keys = [tuple(k) for k in reserves.components()]
    assert set(owner) <= set(keys)
    for crow, rrow in zip(cells.draw_matrix(), reserves.draw_matrix()):
        for o, key in enumerate(keys):
            if key in owner:
                assert close(sum(crow[j] for j in owner[key]), rrow[o], rel)
            else:
                assert rrow[o] == 0.0
    summed = cells.aggregate(reserves.dims)
    column = {tuple(k): j for j, k in enumerate(summed.components())}
    assert set(column) == set(owner)
    for srow, rrow in zip(summed.draw_matrix(), reserves.draw_matrix()):
        for o, key in enumerate(keys):
            if key in column:
                assert close(srow[column[key]], rrow[o], rel)


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_cells_leave_the_reserves_unchanged_and_sum_to_them(name):
    raa = dataset("raa")
    boot = BOOTSTRAPS[name](n_sims=200, seed=979)
    without = boot.fit(raa, "values")
    with_cells = boot.fit(raa, "values", cells=True)
    assert without.cells is None
    assert with_cells.reserves.draw_matrix() == without.reserves.draw_matrix()
    assert with_cells.reserves.components() == without.reserves.components()
    cells = with_cells.cells
    assert cells.dims == ["origin", "development"]
    assert len(cells.draw_matrix()) == 200
    # RAA is 10 by 10: the origin k years after 1981 has k future cells,
    # after its latest age, in age order.
    components = cells.components()
    assert len(components) == 45
    assert components[0] == ("1982", 120)
    assert [k[1] for k in components if k[0] == "1990"] == list(range(24, 121, 12))
    check_sums(with_cells.reserves, cells)


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_a_tail_adds_one_cell_per_origin_past_the_oldest_age(name):
    raa = dataset("raa")
    fit = BOOTSTRAPS[name](n_sims=200, seed=979, tail=1.05).fit(raa, "values", cells=True)
    cells = fit.cells
    components = cells.components()
    assert len(components) == 55
    assert [k[0] for k in components if k[1] == 132] == [str(y) for y in range(1981, 1991)]
    # The oldest origin's only cell is its tail step, its whole reserve.
    assert components[0] == ("1981", 132)
    for crow, rrow in zip(cells.draw_matrix(), fit.reserves.draw_matrix()):
        assert close(crow[0], rrow[0], 1e-12)
    check_sums(fit.reserves, cells)


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_a_fully_developed_triangle_has_no_cells_without_a_tail(name):
    boot = BOOTSTRAPS[name](n_sims=50, seed=979)
    assert boot.fit(square(), "values").cells is None
    with pytest.raises(ValueError, match="no future cells"):
        boot.fit(square(), "values", cells=True)
    cells = (
        BOOTSTRAPS[name](n_sims=50, seed=979, tail=1.05)
        .fit(square(), "values", cells=True)
        .cells
    )
    assert [k[1] for k in cells.components()] == [48, 48, 48]


@pytest.mark.parametrize("name", ["odp", "mack"])
@pytest.mark.parametrize("dependence", ["independent", "synchronized", "rank_correlation"])
def test_segments_keep_their_own_cells(name, dependence):
    tri = twins("raa")
    spearman = [[1.0, 0.6], [0.6, 1.0]] if dependence == "rank_correlation" else None
    boot = BOOTSTRAPS[name](n_sims=200, seed=979, dependence=dependence, spearman=spearman)
    without = boot.fit(tri, "values")
    fit = boot.fit(tri, "values", cells=True)
    assert fit.reserves.draw_matrix() == without.reserves.draw_matrix()
    cells = fit.cells
    assert cells.dims == ["lob", "origin", "development"]
    assert len(cells.components()) == 90
    check_sums(fit.reserves, cells)
    # A segment picked alone keeps its part of the cells.
    a = fit.segment(lob="A")
    assert a.cells.dims == ["lob", "origin", "development"]
    assert a.cells.components() == [k for k in cells.components() if k[0] == "A"]
    check_sums(a.reserves, a.cells)
    assert without.segment(lob="A").cells is None


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_a_segment_without_future_cells_has_none(name):
    sq = [(2018, 12, 100.0), (2018, 24, 150.0), (2018, 36, 165.0)]
    sq += [(2019, 12, 110.0), (2019, 24, 170.0), (2019, 36, 182.0)]
    sq += [(2020, 12, 120.0), (2020, 24, 175.0), (2020, 36, 196.0)]
    tri = stacked([("raa", rows_of("raa"), 1.0), ("square", sq, 1.0)])
    fit = BOOTSTRAPS[name](n_sims=50, seed=979).fit(tri, "values", cells=True)
    assert {k[0] for k in fit.cells.components()} == {"raa"}
    check_sums(fit.reserves, fit.cells)
    assert fit.segment(lob="square").cells is None
    assert fit.segment(lob="raa").cells is not None


def genins_index(rate):
    """GenIns' index, 2001 to 2010 at 4% a year, base March 2006."""
    given = [100.0 * 1.04**k for k in range(10)]
    return Indexation(CalendarIndex("2001", given, rate), "2006-03"), given


@pytest.mark.parametrize("name", ["odp", "mack"])
@pytest.mark.parametrize("tail", [1.0, 1.03])
def test_reinflate_draws_takes_the_cells(name, tail):
    fit = BOOTSTRAPS[name](n_sims=300, seed=979, tail=tail).fit(
        dataset("genins"), "values", cells=True
    )
    cells = fit.cells
    n_sims = len(cells.draw_matrix())
    rate = 0.05
    at, given = genins_index(rate)
    # Paths at the index's own future rate, 2011 to 2021 (the tail is paid
    # one period past the oldest age), give the deterministic re-inflation.
    path = [given[9] * (1.0 + rate) ** k for k in range(1, 12)]
    with_paths = at.reinflate_draws(cells, IndexPaths("2011", [path] * n_sims))
    without = at.reinflate_draws(cells)
    assert with_paths.components() == cells.components()
    for a_row, b_row in zip(with_paths.draw_matrix(), without.draw_matrix()):
        for a, b in zip(a_row, b_row):
            assert close(a, b, 1e-12)
    # A unit index gives back the reserves by origin.
    unit = Indexation(CalendarIndex("2001", [1.0] * 10, 0.0), "2006-03")
    by_origin = unit.reinflate_draws(cells).aggregate(["origin"])
    column = {tuple(k): j for j, k in enumerate(fit.reserves.components())}
    for brow, rrow in zip(by_origin.draw_matrix(), fit.reserves.draw_matrix()):
        for j, key in enumerate(by_origin.components()):
            assert close(brow[j], rrow[column[tuple(key)]], 1e-9)


@pytest.mark.parametrize("name", ["odp", "mack"])
@pytest.mark.parametrize("tail", [1.0, 1.03])
def test_each_cell_is_reinflated_at_its_own_payment_year(name, tail):
    # A flat index to 2010 then 4% a year, based at 2010-12: the cell of
    # origin o at age d months is paid in year o + d / 12 - 1, so it grows
    # by 1.04 ** (o + d / 12 - 1 - 2010), the tail cell one year later than
    # the oldest age's.
    fit = BOOTSTRAPS[name](n_sims=50, seed=979, tail=tail).fit(
        dataset("genins"), "values", cells=True
    )
    at = Indexation(CalendarIndex("2001", [1.0] * 10, 0.04), "2010-12")
    out = at.reinflate_draws(fit.cells)
    keys = fit.cells.components()
    assert out.components() == keys
    years = [int(o) + d // 12 - 1 - 2010 for o, d in keys]
    assert min(years) == 1
    assert max(years) == (10 if tail > 1.0 else 9)
    for crow, orow in zip(fit.cells.draw_matrix(), out.draw_matrix()):
        for c, o, y in zip(crow, orow, years):
            assert abs(o - c * 1.04**y) <= 1e-12 * max(1.0, abs(c * 1.04**y))


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_cells_is_a_keyword_argument(name):
    with pytest.raises(TypeError):
        BOOTSTRAPS[name](n_sims=10, seed=979).fit(dataset("raa"), "values", True)


@pytest.mark.parametrize("name", ["odp", "mack"])
def test_reinflate_draws_takes_the_segments_cells(name):
    fit = BOOTSTRAPS[name](n_sims=200, seed=979).fit(twins("genins"), "values", cells=True)
    unit = Indexation(CalendarIndex("2001", [1.0] * 10, 0.0), "2006-03")
    out = unit.reinflate_draws(fit.cells)
    assert out.dims == ["lob", "origin", "development"]
    assert out.components() == fit.cells.components()
    check_sums(fit.reserves, out)
