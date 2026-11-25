"""Stochastic re-inflation with one index path per simulation (issue #979,
docs/design/reserving-inflation.md, PR 8, decision 11): the Python wrappers
of validation/tests/reserving_reinflate_draws.rs."""

import pickle
from collections import defaultdict

import numpy as np
import pytest
from helpers import VALIDATION, dataset, read_csv
from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import (
    CalendarIndex,
    FutureCells,
    IndexPaths,
    Indexation,
    OdpGlm,
)


def close(a, b, rel=1e-9, abs_tol=0.0):
    return abs(a - b) <= max(abs_tol, rel * max(1.0, abs(b)))


def args(row):
    """`name=value;name=value` as a dict."""
    return dict(pair.split("=", 1) for pair in row["arg"].split(";") if pair)


def build(rows):
    """One method's index, paths and draws from its rows, and the result."""
    of = defaultdict(list)
    for r in rows:
        of[r["quantity"]].append(r)
    index_rows = of["index"]
    rate = [float(r["expected"]) for r in of["future_rate"]]
    index = CalendarIndex(
        args(index_rows[0])["period"],
        [float(r["expected"]) for r in index_rows],
        future_rate=rate[0] if rate else None,
    )
    at = Indexation(index, args(of["base"][0])["month"])
    draw_rows = of["draw"]
    keys = []
    for r in draw_rows:
        a = args(r)
        if a["sim"] == "0":
            keys.append((a["origin"], int(a["age"])))
    n_sims = len(draw_rows) // len(keys)
    matrix = [
        [float(r["expected"]) for r in draw_rows[i * len(keys) : (i + 1) * len(keys)]]
        for i in range(n_sims)
    ]
    for k, r in enumerate(draw_rows):
        a = args(r)
        assert int(a["sim"]) == k // len(keys)
        assert (a["origin"], int(a["age"])) == keys[k % len(keys)]
    draws = PredictiveDistribution(["origin", "development"], keys, matrix)
    paths = None
    if of["path"]:
        per_sim = len(of["path"]) // n_sims
        levels = [float(r["expected"]) for r in of["path"]]
        paths = IndexPaths(
            args(of["path"][0])["period"],
            [levels[i * per_sim : (i + 1) * per_sim] for i in range(n_sims)],
        )
    return at, paths, draws, keys, at.reinflate_draws(draws, paths)


def test_reinflated_draws_match_the_closed_form():
    rows = read_csv(VALIDATION / "reference" / "reserving_reinflate_draws.csv")
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    assert sorted(by_method) == [
        "annual_paths",
        "annual_rate",
        "monthly_paths",
        "quarterly_paths",
    ]
    for method, cases in by_method.items():
        _, paths, draws, keys, out = build(cases)
        assert (paths is None) == (method == "annual_rate")
        assert out.dims == draws.dims
        assert out.components() == draws.components()
        assert out.n_sims == draws.n_sims
        assert "reinflated" in dict(out.provenance()["parameters"])
        matrix = out.draw_matrix()
        by_origin = out.aggregate(["origin"])
        origins = [k[0] for k in by_origin.components()]
        reserves = by_origin.draw_matrix()
        checked = defaultdict(int)
        for r in cases:
            a, want = args(r), float(r["expected"])
            tol = {"rel": float(r["rel_tol"]), "abs_tol": float(r["abs_tol"])}
            q = r["quantity"]
            if q == "reinflated":
                got = matrix[int(a["sim"])][keys.index((a["origin"], int(a["age"])))]
            elif q == "reserve":
                got = reserves[int(a["sim"])][origins.index(a["origin"])]
            elif q == "total":
                got = sum(matrix[int(a["sim"])])
            else:
                continue
            checked[q] += 1
            assert close(got, want, **tol), (method, q, r["arg"], got, want)
        assert checked["reinflated"] == draws.n_sims * len(keys), method
        assert checked["reserve"] > 0 and checked["total"] > 0, method


def test_without_paths_each_simulation_reinflates_as_reinflate():
    fit = OdpGlm().fit(dataset("genins"), "values")
    draws = fit.predict_distribution(5, 11)
    cells = FutureCells.from_odp_glm(fit).cells
    at = Indexation(CalendarIndex("2001", [1.0 + 0.03 * k for k in range(10)], 0.04), "2005-06")
    out = at.reinflate_draws(draws)
    assert out.components() == draws.components()
    assert out.dims == ["origin", "development"]
    column = {key: j for j, key in enumerate(draws.components())}
    order = [column[(o, a)] for o, a in zip(cells["origin"], cells["age"])]
    for base, got in zip(draws.draw_matrix(), out.draw_matrix()):
        one = FutureCells(
            sorted(set(cells["origin"])),
            cells["origin"],
            cells["age"],
            cells["valuation"],
            [base[j] for j in order],
        )
        want = at.reinflate(one)
        for j, value in zip(order, want.cells.cells["value"]):
            assert close(got[j], value, 1e-12)
    # Origin 2001 is fully developed, so nine origins have reserves.
    assert len(out.aggregate(["origin"]).components()) == 9


def test_paths_read_lists_and_numpy_and_integer_keys_pass_through():
    levels = [[1.21, 1.331], [1.155, 1.21]]
    paths = IndexPaths(2023, np.array(levels))
    assert paths == IndexPaths("2023", levels)
    assert paths.first == "2023" and paths.n_sims == 2 and paths.n_periods == 2
    assert paths.periods == ["2023", "2024"] and paths.values == levels
    assert pickle.loads(pickle.dumps(paths)) == paths
    assert repr(paths) == 'IndexPaths(first="2023", n_sims=2, n_periods=2)'
    assert IndexPaths("2023Q3", [[1.0, 1.1, 1.2]]).periods == ["2023Q3", "2023Q4", "2024Q1"]
    # Integer origins and an extra dimension: the keys come back as given.
    draws = PredictiveDistribution(
        ["lob", "origin", "age"],
        [("auto", 2022, 24), ("auto", 2022, 36)],
        [[100.0, 50.0], [120.0, 40.0]],
    )
    at = Indexation(CalendarIndex("2021", [1.0, 1.1]), "2022-01")
    out = at.reinflate_draws(draws, paths)
    assert out.components() == [("auto", 2022, 24), ("auto", 2022, 36)]
    want = [[100 * 1.21 / 1.1, 50 * 1.331 / 1.1], [120 * 1.155 / 1.1, 40 * 1.21 / 1.1]]
    for got_row, want_row in zip(out.draw_matrix(), want):
        assert all(close(g, w, 1e-12) for g, w in zip(got_row, want_row))


def test_errors():
    with pytest.raises(ValueError, match="at least one simulation"):
        IndexPaths("2023", [])
    with pytest.raises(ValueError, match="row 1"):
        IndexPaths("2023", [[1.0, 1.1], [1.0]])
    with pytest.raises(ValueError, match="positive multiple"):
        IndexPaths("2023", [[], []])
    with pytest.raises(ValueError, match="finite and positive"):
        IndexPaths("2023", [[1.0, float("nan")]])
    with pytest.raises(ValueError, match="period"):
        IndexPaths("twenty", [[1.0]])
    at = Indexation(CalendarIndex("2021", [1.0, 1.1]), "2022-01")
    cells = [("2022", 24), ("2022", 36)]
    draws = PredictiveDistribution(["origin", "development"], cells, [[1.0, 1.0]] * 2)
    by_origin = PredictiveDistribution(["origin"], [("2022",)], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="no development or age"):
        at.reinflate_draws(by_origin)
    with pytest.raises(ValueError, match="no origin"):
        at.reinflate_draws(PredictiveDistribution(["line"], [("a",)], [[1.0]]))
    with pytest.raises(ValueError, match="not a period"):
        at.reinflate_draws(
            PredictiveDistribution(["origin", "age"], [("twenty", 24)], [[1.0]])
        )
    with pytest.raises(ValueError, match="both a development and an age"):
        at.reinflate_draws(
            PredictiveDistribution(
                ["origin", "development", "age"], [("2022", 24, 24)], [[1.0]]
            )
        )
    with pytest.raises(ValueError, match="positive number of months"):
        at.reinflate_draws(PredictiveDistribution(["origin", "age"], [("2022", 0)], [[1.0]]))
    same = PredictiveDistribution(
        ["origin", "age"], [("2022", 24), (2022, 24)], [[1.0, 1.0]]
    )
    with pytest.raises(ValueError, match=r'\("2022", 24\) and \(2022, 24\) both read'):
        at.reinflate_draws(same)
    with pytest.raises(ValueError, match="not an age"):
        at.reinflate_draws(
            PredictiveDistribution(["origin", "age"], [("2022", "24")], [[1.0]])
        )
    with pytest.raises(ValueError, match="no future rate"):
        at.reinflate_draws(draws)
    with pytest.raises(ValueError, match="not in 2023"):
        at.reinflate_draws(draws, IndexPaths("2024", [[1.2], [1.2]]))
    with pytest.raises(ValueError, match="another grain"):
        at.reinflate_draws(draws, IndexPaths("2023Q1", [[1.2], [1.2]]))
    with pytest.raises(ValueError, match="2 simulations"):
        at.reinflate_draws(draws, IndexPaths("2023", [[1.2, 1.3]] * 3))
    with pytest.raises(ValueError, match="the paths end in"):
        at.reinflate_draws(draws, IndexPaths("2023", [[1.2], [1.2]]))
    outside = Indexation(CalendarIndex("2021", [1.0, 1.1], future_rate=0.1), "2023-01")
    with pytest.raises(ValueError, match="outside the given index"):
        outside.reinflate_draws(draws, IndexPaths("2023", [[1.2, 1.3]] * 2))
    with pytest.raises(TypeError):
        at.reinflate_draws([[1.0]])
