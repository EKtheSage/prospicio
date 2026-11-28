"""The scenario generator and the projection from Python (esg-dfa.md, PR 5):
the matrices' shapes and their first column, a seed's paths replayed, the
Rust doctests' numbers, ruin and breach, the losses' dependence, and the
errors for names outside the vocabularies."""

import math

import numpy as np
import pytest

from prospicio.esg import (
    Cir,
    Cycle,
    Esg,
    HullWhite,
    Inflation,
    Projection,
    ProjectionPaths,
    Superimposed,
    Svensson,
    Vasicek,
)


def economy(step="quarterly"):
    return Esg(
        Vasicek(0.3, 0.04, 0.01, 0.02),
        inflation=Inflation(0.5, 0.025, 0.02, 0.03),
        superimposed=Superimposed(0.01, 0.5, 0.005),
        correlation=[[1.0, 0.4, 0.0], [0.4, 1.0, 0.3], [0.0, 0.3, 1.0]],
        step=step,
        maturities=[1.0, 5.0],
    )


def test_the_scenarios_come_back_as_matrices_by_path_and_step():
    sc = economy().simulate(3, 50, seed=4)
    assert (sc.n_sims, sc.n_steps, sc.step) == (50, 12, "quarterly")
    assert sc.times()[:2] == [0.0, 0.25]
    assert sc.maturities() == [1.0, 5.0]
    for m in (sc.short_rates(), sc.deflators(), sc.inflation_rates(), sc.index_levels("price"),
              sc.index_levels(), sc.bond_prices(1)):
        assert m.shape == (50, 13)
    assert np.all(sc.short_rates()[:, 0] == 0.02)
    assert np.all(sc.deflators()[:, 0] == 1.0)
    assert np.all(sc.index_levels()[:, 0] == 1.0)
    assert sc.superimposed().shape == (50, 3)
    curve = sc.curve(0, 0)
    assert curve["maturities"] == [1.0, 5.0] and len(curve["rates"]) == 2
    again = economy().simulate(3, 50, seed=4)
    assert np.array_equal(sc.short_rates(), again.short_rates())


def test_the_models_report_their_moments():
    v = Vasicek(0.3, 0.04, 0.01, 0.02)
    assert v.mean(2.0) == pytest.approx(0.04 + (0.02 - 0.04) * math.exp(-0.6))
    assert v.variance(2.0) == pytest.approx(0.01**2 / 0.6 * (1 - math.exp(-1.2)))
    assert 0.0 < v.bond_price(0.02, 5.0) < 1.0
    c = Cir(0.4, 0.03, 0.08, 0.02)
    assert c.feller() and c.mean(0.0) == pytest.approx(0.02)
    s = Svensson([0.04, -0.02, 0.01, 0.0], [2.0, 8.0])
    assert s.discount(10.0) == pytest.approx(math.exp(-10.0 * s.spot(10.0)))
    fitted = Svensson.fit([1.0, 2.0, 5.0, 10.0, 20.0, 30.0], [s.spot(m) for m in (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)])
    assert fitted.spot(7.0) == pytest.approx(s.spot(7.0), abs=1e-5)
    hw = HullWhite(0.1, 0.01, svensson=s)
    assert hw.discount(4.0) == pytest.approx(s.discount(4.0))
    i = Inflation(0.5, 0.025, 0.01, 0.03)
    assert i.variance(50.0) == pytest.approx(i.stationary_variance())
    assert Superimposed(0.0, 0.5, 0.1).variance(1) == pytest.approx(0.01)
    assert Cycle(0.0, 1.2, -0.6, 0.05, 0.0).period() == pytest.approx(9.176, abs=1e-3)
    assert Cycle(0.0, 0.5, 0.0, 0.05, 0.0).period() is None


def test_the_projection_matches_the_rust_doctest():
    p = Projection([[60.0, 70.0, 80.0], [50.0, 120.0, 40.0]], 200.0,
                   premiums=[100.0] * 3, expense_ratio=0.3, payout=[0.5, 0.5])
    paths = p.run()
    assert (paths.n_sims, paths.years, paths.opening_surplus) == (2, 3, 200.0)
    assert paths.values("result")[1, 1] == pytest.approx(-50.0)
    assert paths.values("surplus")[1, 2] == pytest.approx(200.0)
    assert paths.ruin(3) == {"probability": 0.0, "standard_error": 0.0}
    assert "surplus_before_capital" in ProjectionPaths.lines()
    for capital in (150.0, [150.0, 150.0, 150.0], ("premium", 1.5)):
        b = paths.breach(3, capital)
        assert set(b) == {"probability", "standard_error", "shortfall"}
    assert paths.breach(3, 0.0)["shortfall"] is None


def test_the_rules_and_the_economy_run_together():
    years, n = 4, 2_000
    sc = economy().simulate(years, n, seed=7)
    rng = np.random.default_rng(3)
    losses = rng.gamma(4.0, 20.0, size=(n, years))
    p = Projection(losses, 300.0, premiums=[100.0] * years, expense_ratio=0.25,
                   opening_reserves=[40.0, 20.0], payout=[0.6, 0.4], bonds=[(0.5, 5.0)],
                   cycle=Cycle(0.0, 1.2, -0.6, 0.05, 0.0), cycle_seed=2, tax_rate=0.2,
                   dividend_above=2.0, inject_below=0.5,
                   dependence="gumbel", dependence_parameter=2.0, dependence_seed=5)
    paths = p.run(sc)
    surplus = paths.values("surplus")
    moves = (paths.values("result") - paths.values("tax") - paths.values("dividends")
             + paths.values("injections"))
    before = np.column_stack([np.full(n, paths.opening_surplus), surplus[:, :-1]])
    assert np.allclose(surplus - before, moves)
    # The dependence keeps each year's losses as a set.
    levels = sc.index_levels()[:, 4::4]
    base = paths.values("incurred_losses") / levels
    assert np.allclose(np.sort(base, axis=0), np.sort(losses, axis=0))


def test_names_outside_the_vocabularies_are_refused():
    with pytest.raises(ValueError, match='step must be "monthly", "quarterly" or "annual"'):
        Esg(Vasicek(0.3, 0.04, 0.01, 0.02), step="weekly")
    p = Projection([[1.0]], 10.0)
    with pytest.raises(ValueError, match="line must be"):
        p.run().values("profit")
    with pytest.raises(ValueError, match="dependence must be"):
        Projection([[1.0]], 10.0, dependence="vine")
    with pytest.raises(ValueError):
        Projection([[1.0]], 10.0, dependence="gaussian", dependence_parameter=0.5).run()
    with pytest.raises(ValueError):
        HullWhite(0.1, 0.01)
