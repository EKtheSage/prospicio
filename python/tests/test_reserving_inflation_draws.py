"""Stochastic re-inflation of simulated future cells (issue #979,
docs/design/reserving-inflation.md, PR 8): the Python wrapper of
validation/tests/reserving_inflation_draws.rs."""

import pytest
from helpers import dataset
from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import CalendarIndex, Indexation, OdpGlm


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def setup():
    fit = OdpGlm().fit(dataset("genins"), "values")
    draws = fit.predict_distribution(200, 11)
    history = [1.04**k for k in range(10)]
    at_2010 = Indexation(CalendarIndex("2001", history, future_rate=0.04), "2010-01")
    return draws, at_2010


def test_without_paths_each_simulation_takes_the_indexation():
    draws, at_2010 = setup()
    out = at_2010.reinflate_draws(draws)
    assert out.dims == draws.dims and out.n_sims == draws.n_sims
    # Each cell is scaled by its payment period's factor, the same in every
    # simulation: development d of origin o is paid in o + d / 12 - 1.
    factors = [
        at_2010.factor(f"{int(key[0]) + key[1] // 12 - 1}-12") for key in draws.components()
    ]
    for got, row in zip(out.draw_matrix()[:5], draws.draw_matrix()[:5]):
        assert all(close(g, x * f) for g, x, f in zip(got, row, factors))


def test_each_simulation_takes_its_own_path():
    draws, at_2010 = setup()
    index = at_2010.index
    paths = [
        CalendarIndex("2001", index.values, future_rate=0.1 * i / draws.n_sims)
        for i in range(draws.n_sims)
    ]
    out = at_2010.reinflate_draws(draws, paths)
    # The first path has no future inflation.
    assert out.draw_matrix()[0] == draws.draw_matrix()[0]
    # The last one grows every cell.
    assert all(g > x for g, x in zip(out.draw_matrix()[-1], draws.draw_matrix()[-1]) if x > 0)


def test_the_paths_are_one_per_simulation():
    draws, at_2010 = setup()
    with pytest.raises(ValueError, match="one per simulation"):
        at_2010.reinflate_draws(draws, [at_2010.index, at_2010.index])
    lines = PredictiveDistribution(["line"], [("A",)], [[1.0]])
    with pytest.raises(ValueError, match="origin"):
        at_2010.reinflate_draws(lines)
