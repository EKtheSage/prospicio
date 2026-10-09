import math

import pytest

import prospicio as ar
from prospicio.aggregate import fft, panjer, simulate_events
from prospicio.reinsurance import Layer, Tower
from prospicio.distributions import Grid, Lognormal, NegativeBinomial, Poisson

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
    prov = result.provenance()
    assert prov["seed"] == 11
    assert prov["stream_scheme"] == "chacha20/sim-index/v1"
    assert prov["samplers"] == [("gamma", "marsaglia-tsang/2026-10")]


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
    from prospicio.distributions import PredictiveDistribution

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


def test_surplus_treaty_on_events_with_sums_insured():
    from prospicio.aggregate import EventSet
    from prospicio.reinsurance import Layer, Tower

    s = Layer.surplus("surplus", 1e6, 4.0)
    assert s.needs_sums_insured
    # Cessions 0, 1/2, 4/5 and 2/5.
    assert s.ceded_with_sums_insured([0.5e6, 2e6, 1e6, 10e6], [0.5e6, 2e6, 5e6, 10e6]) == \
        pytest.approx(0.0 + 1e6 + 0.8e6 + 4e6)
    events = EventSet.from_years([[0.5e6, 2e6], [1e6, 10e6], []],
                                 [[0.5e6, 2e6], [5e6, 10e6], []])
    assert events.has_sums_insured and events.sums_insured(1) == [5e6, 10e6]
    tower = Tower.inuring([[s], [Layer("1x1", 1e6, 1e6)]])
    pd = tower.apply(events)
    draws = {k: pd.marginal(k).mean() for k in [("ceded", "surplus"), ("ceded", "1x1")]}
    assert draws[("ceded", "surplus")] == pytest.approx((1e6 + 4.8e6) / 3)
    assert draws[("ceded", "1x1")] == pytest.approx(1e6 / 3)
    # Losses from elsewhere: the seed is recorded, the samplers are not.
    assert pd.provenance()["seed"] == 0 and pd.provenance()["samplers"] is None
    with pytest.raises(ValueError, match="sum insured"):
        tower.apply(EventSet.from_years([[1.0]]))
    with pytest.raises(ValueError):
        EventSet.from_years([[5.0]], [[4.0]])
    with pytest.raises(ValueError):
        EventSet.from_years([[5.0]], [[6.0, 7.0]])


def test_reinstatements_pro_rata_as_to_time():
    from prospicio.aggregate import EventSet

    amount = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
    timed = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5],
                  pro_rata_time=True)
    assert timed.pro_rata_time and not amount.pro_rata_time
    assert math.isnan(timed.reinstatement_premium([22.0]))
    assert amount.reinstatement_premium([22.0, 12.0], [0.25, 0.5]) == pytest.approx(2.2)
    # 10 at t = 0.25 (rate 1), then 2 of the second limit at t = 0.5 (rate 0.5).
    want = 2.0 * (10.0 * 0.75 + 0.5 * 2.0 * 0.5) / 10.0
    assert timed.reinstatement_premium([22.0, 12.0], [0.25, 0.5]) == pytest.approx(want)
    with pytest.raises(ValueError):
        timed.reinstatement_premium([22.0], [0.1, 0.2])
    with pytest.raises(ValueError):
        Layer("free", 10.0, 10.0, reinstatements=1, pro_rata_time=True)

    # One loss a year exhausting the limit at a uniform time: 2 E[1 - t] = 1.
    n = 20_000
    events = EventSet.from_years([[20.0]] * n, seed=9)
    one = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0], pro_rata_time=True)
    with pytest.raises(ValueError):
        Tower([one]).apply(events)
    dated = events.with_uniform_times()
    assert dated.has_times and not events.has_times and events.times(0) is None
    rp = Tower([one]).apply(dated).marginal(("reinstatement_premium", "10x10"))
    assert abs(rp.mean() - 1.0) < 4 * (rp.variance() / n) ** 0.5

    e = EventSet.from_years([[5.0, 2.0], [9.0]], times=[[0.1, 0.6], [0.3]])
    assert e.times(1) == [0.3]
    with pytest.raises(ValueError):
        EventSet.from_years([[5.0, 2.0]], times=[[0.6, 0.1]])
    with pytest.raises(ValueError):
        EventSet.from_years([[5.0, 2.0]], times=[[0.1]])
    # Uniform times leave the losses alone and replay per year.
    a = simulate_events(Poisson(3.0), SEV, 20, 4).with_uniform_times()
    b = simulate_events(Poisson(3.0), SEV, 200, 4).with_uniform_times()
    assert [a.times(i) for i in range(20)] == [b.times(i) for i in range(20)]
    assert a.events(3) == simulate_events(Poisson(3.0), SEV, 20, 4).events(3)

    # Losses only in the second half of the year: 2 E[1 - t] = 0.5.
    late = events.with_seasonal_times([0.0, 1.0])
    assert all(0.5 <= late.times(i)[0] <= 1.0 for i in range(100))
    rp = Tower([one]).apply(late).marginal(("reinstatement_premium", "10x10"))
    assert abs(rp.mean() - 0.5) < 4 * (rp.variance() / n) ** 0.5
    flat = simulate_events(Poisson(3.0), SEV, 20, 4).with_seasonal_times([1.0] * 12)
    for i in range(20):
        assert flat.times(i) == pytest.approx(a.times(i), abs=1e-15)
    for bad in ([], [0.0, 0.0], [1.0, -1.0]):
        with pytest.raises(ValueError):
            events.with_seasonal_times(bad)


def test_loss_corridors():
    qs = Layer.quota_share("QS", 0.3).with_loss_corridor(70.0, 90.0)
    assert qs.loss_corridor == (70.0, 90.0, 1.0)
    assert Layer.quota_share("QS", 0.3).loss_corridor is None
    assert qs.ceded([50.0, 30.0]) == pytest.approx(21.0)
    assert qs.ceded([120.0]) == pytest.approx(30.0)
    # Deductible, corridor, then the annual limit.
    layer = Layer("L", 10.0, 5.0, aggregate_deductible=4.0, aggregate_limit=12.0)
    layer = layer.with_loss_corridor(5.0, 9.0, 0.5)
    assert layer.ceded([8.0, 20.0, 12.0]) == 12.0
    assert layer.ceded_by_event([8.0, 20.0, 12.0]) == [0.0, 7.0, 5.0]
    for bad in ((2.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 2.0, 0.0), (1.0, math.inf, 1.0)):
        with pytest.raises(ValueError):
            Layer("L", 1.0, 1.0).with_loss_corridor(*bad)
    tower = Tower.inuring([[qs], [Layer("4x4", 4.0, 4.0)]])
    back = Tower.from_json(tower.to_json())
    assert back.to_json() == tower.to_json()
    events = simulate_events(Poisson(3.0), SEV, 2_000, 5)
    a, b = tower.apply(events), back.apply(events)
    assert a.draw_matrix() == b.draw_matrix()


def test_towers_save_and_load_as_json():
    import pickle

    from prospicio.aggregate import EventSet

    tower = Tower.inuring([
        [Layer.quota_share("QS", 0.3), Layer.surplus("S", 1e6, 4.0)],
        [Layer("xl", 2e6, 1e6, share=0.6, aggregate_deductible=5e5, premium=3e5,
               reinstatement_rates=[1.0, 0.5], pro_rata_time=True),
         Layer("top", math.inf, 3e6)],
    ])
    text = tower.to_json()
    back = Tower.from_json(text)
    assert back.to_json() == text
    assert pickle.loads(pickle.dumps(tower)).to_json() == text
    ev = EventSet.from_years([[3e6, 0.5e6], [8e6]], sums_insured=[[5e6, 1e6], [2e7]],
                             times=[[0.2, 0.7], [0.5]])
    a, b = tower.apply(ev), back.apply(ev)
    assert a.draw_matrix() == b.draw_matrix()
    with pytest.raises(ValueError):
        Tower.from_json(text.replace('"share":0.6', '"share":1.6'))
    with pytest.raises(ValueError):
        Tower.from_json("{}")


def test_grid_sizing():
    from prospicio.aggregate import fft_auto, recommend_grid, round_bucket
    from prospicio.distributions import Gamma, Pareto, Poisson

    # aggregate 1.0.1's round_bucket rungs.
    assert [round_bucket(x) for x in (1.1, 2.5, 5.5, 2412.0, 0.3, 1e-21)] == [
        2.0, 4.0, 8.0, 4000.0, 0.5, 2.0**-69,
    ]
    with pytest.raises(ValueError):
        round_bucket(0.0)
    # aggregate sizes this book at bs = 1/16 too.
    g = recommend_grid(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    assert (g.step, g.points, g.method) == (0.0625, 65536, "moments")
    # Infinite variance: the single big jump sizes the grid.
    heavy = recommend_grid(Poisson(20.0), Pareto(100.0, 1.5), log2=14)
    assert heavy.method == "single_big_jump" and heavy.moment_extent is None
    agg, report, size = fft_auto(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    assert report.aliasing_error < 1e-12 and size.points == len(agg.probs)
    assert agg.mean() == pytest.approx(500.0, rel=1e-6)
    with pytest.raises(ValueError):
        recommend_grid(Poisson(1.0), Pareto(1.0, 0.8))
