import math

import pytest

import actuarialrs as ar
from actuarialrs.aggregate import fft, panjer, simulate_events
from actuarialrs.reinsurance import Layer, Tower
from actuarialrs.distributions import Grid, Lognormal, NegativeBinomial, Poisson

SEV = Grid(1.0, [0.1, 0.3, 0.25, 0.2, 0.1, 0.05])


def test_panjer_and_fft_agree_and_match_compound_moments():
    for n in (Poisson(3.0), NegativeBinomial(2.5, 1.5)):
        a, ra = panjer(n, SEV, 300)
        b, rb = fft(n, SEV, 300)
        assert ra.method == "panjer" and rb.method == "fft"
        assert max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-13
        assert rb.aliasing_error < 1e-12
        # E[S] = E[N] E[X]; Var[S] = E[N] Var[X] + Var[N] E[X]^2.
        assert a.mean() == pytest.approx(n.mean() * SEV.mean(), rel=1e-12)
        var = n.mean() * SEV.variance() + n.variance() * SEV.mean() ** 2
        assert a.variance() == pytest.approx(var, rel=1e-10)


def test_panjer_underflow_points_to_fft():
    unit = Grid(1.0, [0.0, 1.0])
    with pytest.raises(ValueError, match="FFT"):
        panjer(Poisson(800.0), unit, 10)
    with pytest.raises(TypeError):
        panjer(Lognormal(0.0, 1.0), unit, 10)


def test_events_are_reproducible_and_feed_a_tower():
    sev = Lognormal.from_mean_cv(3e6, 1.5)
    events = simulate_events(Poisson(2.0), sev, 20_000, 11)
    again = simulate_events(Poisson(2.0), sev, 20_000, 11)
    assert events.events(123) == again.events(123)
    assert events.n_sims == 20_000 and events.seed == 11
    assert sum(events.counts()) > 0
    with pytest.raises(IndexError):
        events.events(20_000)

    tower = Tower([Layer("5x5", 5e6, 5e6, reinstatements=1), Layer("15x10", 15e6, 10e6, share=0.6)])
    result = tower.apply(events)
    assert result.dims == ["kind", "layer"]
    for row in result.draw_matrix()[:500]:
        gross, ceded_a, ceded_b, net = row
        assert gross == pytest.approx(ceded_a + ceded_b + net, abs=1e-6)
    by_kind = result.aggregate(["kind"])
    assert [k[0] for k in by_kind.components()] == ["gross", "ceded", "net"]
    assert result.provenance()["seed"] == 11


def test_mean_ceded_matches_the_exact_layer_value():
    sev = Lognormal.from_mean_cv(3e6, 1.5)
    events = simulate_events(Poisson(2.0), sev, 100_000, 3)
    ceded = Tower([Layer("5x5", 5e6, 5e6)]).apply(events).marginal(("ceded", "5x5"))
    exact = 2.0 * sev.layer(5e6, 5e6)
    se = ceded.variance() ** 0.5 / len(ceded) ** 0.5
    assert abs(ceded.mean() - exact) < 4 * se


def test_layer_terms():
    layer = Layer("L", 10.0, 5.0, share=0.5, aggregate_deductible=4.0, aggregate_limit=15.0)
    assert layer.ceded([8.0, 20.0, 12.0]) == 7.5
    with pytest.raises(ValueError, match="at most one"):
        Layer("L", 1.0, 0.0, aggregate_limit=2.0, reinstatements=1)
    with pytest.raises(ValueError):
        Layer("L", 1.0, 0.0, share=1.5)
    with pytest.raises(ValueError):
        Tower([Layer("L", 1.0, 0.0), Layer("L", 2.0, 1.0)])


def test_quota_share_and_stop_loss():
    assert Layer.quota_share("QS", 0.25).ceded([8.0, 4.0]) == 3.0
    sl = Layer.stop_loss("SL", 50.0, 100.0)
    assert sl.ceded([60.0, 70.0]) == 30.0
    assert sl.ceded([200.0]) == 50.0
    with pytest.raises(ValueError):
        Layer.quota_share("QS", 0.0)


def test_inuring_tower():
    tower = Tower.inuring(
        [
            [Layer("A", 10.0, 5.0, aggregate_limit=15.0), Layer("B", 100.0, 18.0)],
            [Layer.stop_loss("SL", 20.0, 20.0)],
        ]
    )
    assert tower.stages == [0, 0, 1]
    assert tower.ceded([20.0, 20.0]) == [15.0, 4.0, 1.0]
    with pytest.raises(ValueError):
        Tower.inuring([[], [Layer("A", 1.0, 0.0)]])


def test_ceded_by_event_and_reinstatement_premiums():
    layer = Layer("L", 10.0, 5.0, share=0.5, aggregate_deductible=4.0, aggregate_limit=15.0)
    assert layer.ceded_by_event([8.0, 20.0, 12.0]) == [0.0, 4.5, 3.0]
    paid = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
    assert paid.aggregate_limit == 30.0
    assert paid.reinstatement_rates == [1.0, 0.5]
    assert paid.reinstatement_premium([15.0, 25.0]) == 2.5
    with pytest.raises(ValueError):
        Layer("L", 1.0, 0.0, reinstatements=1, reinstatement_rates=[1.0])

    events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 2_000, 3)
    result = Tower([Layer("5x5", 5e6, 5e6, premium=1e6, reinstatement_rates=[1.0])]).apply(events)
    assert [k[0] for k in result.components()] == ["gross", "ceded", "net", "reinstatement_premium"]
    ceded = result.marginal(("ceded", "5x5")).draws
    premium = result.marginal(("reinstatement_premium", "5x5")).draws
    for c, p in zip(ceded, premium):
        assert abs(p - 1e6 * min(c, 5e6) / 5e6) <= 1e-6


def test_tower_on_grid():
    sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
    tower = Tower([Layer("2x2", 2.0, 2.0), Layer("QS", math.inf, 0.0, share=0.5)])
    r = tower.on_grid(Poisson(3.0), sev, 200)
    # Net keeps half of each loss, which falls between unit points.
    assert not r.on_points
    assert abs(r.ceded[0].mean() - 1.2) < 1e-12
    assert r.ceded[1].step == 0.5
    assert abs(r.gross.mean() - sum(g.mean() for g in r.ceded) - r.net.mean()) < 1e-10
    assert r.gross_report.tail_mass < 1e-12
    assert len(r.ceded_reports) == 2
    assert r.expected_reinstatement_premium == [0.0, 0.0]

    with_terms = Tower([Layer("2x2", 2.0, 2.0, reinstatements=1)]).on_grid(Poisson(3.0), sev, 200)
    assert with_terms.on_points
    assert with_terms.net is None
    with pytest.raises(ValueError):
        Tower.inuring(
            [[Layer("A", 2.0, 2.0, reinstatements=1)], [Layer("B", 4.0, 4.0)]]
        ).on_grid(Poisson(3.0), sev, 100)


def test_portfolio_join_reorder_and_aggregate_cover():
    from actuarialrs.distributions import PredictiveDistribution

    n = 3000
    reserve = PredictiveDistribution(
        ["origin"], [(2023,), (2024,)],
        [[float((i * 7919) % n), float((i * 31) % n) + 1.0] for i in range(n)])
    premium = PredictiveDistribution(["lob"], [("motor",)], [[float((i * 104729) % n)] for i in range(n)])
    adc = Tower([Layer.stop_loss("adc", 500.0, 3000.0)]).apply_aggregate(reserve)
    kinds = adc.aggregate(["kind"])
    assert all(abs(g - c - nt) < 1e-9 for g, c, nt in kinds.draw_matrix())
    assert max(r[1] for r in kinds.draw_matrix()) == 500.0
    p = PredictiveDistribution.join([("reserve", reserve), ("premium", premium)], "risk")
    assert p.dims == ["risk", "origin", "lob"] and p.n_components == 3
    r = p.reorder_groups("risk", [[1.0, 0.7], [0.7, 1.0]], 5)
    assert sorted(r.total().draws) != sorted(p.total().draws)
    assert sorted(r.marginal(("premium", "", "motor")).draws) == sorted(premium.marginal(("motor",)).draws)
    with pytest.raises(ValueError):
        PredictiveDistribution.join([("a", reserve), ("a", premium)], "risk")
