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


def test_reorder_groups_to_a_spearman_target_from_a_long_table():
    from prospicio.distributions import PredictiveDistribution

    n = 20000
    mult = [1, 7919, 104729]
    pd = PredictiveDistribution(
        ["class"], [("A",), ("B",), ("C",)],
        [[float((i * m) % n) for m in mult] for i in range(n)])
    assert pd.groups("class") == ["A", "B", "C"]
    long = [("B", "A", 0.5), ("A", "C", 0.3), ("C", "B", 0.7)]
    rho = pd.correlation_from_pairs("class", long)
    assert rho == [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]]
    s = pd.reorder_groups("class", rho, 9, kind="spearman")
    groups, got = s.group_spearman("class")
    assert groups == ["A", "B", "C"]
    assert all(abs(got[i][j] - rho[i][j]) < 0.01 for i in range(3) for j in range(3))
    # Read as normal scores, the same matrix falls short by (6/pi) asin(r/2).
    _, short = pd.reorder_groups("class", rho, 9).group_spearman("class")
    assert short[1][2] < 0.695
    with pytest.raises(ValueError):
        pd.correlation_from_pairs("class", long[:2])
    with pytest.raises(ValueError):
        pd.reorder_groups("class", rho, 9, kind="kendall")


def test_reorder_groups_by_a_copula_keeps_rows_and_takes_its_ranks():
    from prospicio.distributions import PredictiveDistribution
    from prospicio.risk import ArchimedeanCopula, GaussianCopula, StudentTCopula

    n = 3000
    pd = PredictiveDistribution(
        ["seg", "item"], [("a", 0), ("a", 1), ("b", 0)],
        [[float((i * 7919) % n), 3.0 * ((i * 7919) % n), float((i * 104729) % n)]
         for i in range(n)])
    gumbel = ArchimedeanCopula("gumbel", 2.0, 2)
    r = pd.reorder_groups_by_copula("seg", gumbel, 9)
    assert all(row[1] == 3.0 * row[0] for row in r.draw_matrix())
    assert sorted(r.marginal(("b", 0)).draws) == sorted(pd.marginal(("b", 0)).draws)
    # The totals' ranks are the ranks of the copula's draws on the same seed.
    u = gumbel.sample(n, seed=9)
    rank = lambda v: sorted(range(n), key=lambda i: v[i])
    totals = r.aggregate(["seg"]).draw_matrix()
    assert rank([t[1] for t in totals]) == rank([x[1] for x in u])
    assert abs(r.group_spearman("seg")[1][0][1] - 0.68) < 0.03
    # Few degrees of freedom: more joint extremes than normal scores.
    rank_of = lambda d, g: _ranks([t[g] for t in d.aggregate(["seg"]).draw_matrix()])
    top = lambda d: sum(
        1 for x, y in zip(rank_of(d, 0), rank_of(d, 1)) if x > 0.99 * n and y > 0.99 * n)
    t = pd.reorder_groups_by_copula("seg", StudentTCopula([[1, 0.5], [0.5, 1]], 2.0), 3)
    normal = pd.reorder_groups_by_copula("seg", GaussianCopula([[1, 0.5], [0.5, 1]]), 3)
    assert top(t) > top(normal)
    with pytest.raises(ValueError):
        pd.reorder_groups_by_copula("seg", ArchimedeanCopula("clayton", 1.0, 3), 1)


def _ranks(v):
    out = [0] * len(v)
    for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i])):
        out[i] = r + 1
    return out


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


def test_annual_limit_pro_rata_across_events():
    # 10x5 with one reinstatement on [15, 20, 10]: recoveries 10, 10, 5 and
    # an annual limit of 20.
    order = Layer("10x5", 10.0, 5.0, reinstatements=1)
    pro = Layer("10x5", 10.0, 5.0, reinstatements=1, allocation="pro_rata")
    assert order.allocation == "event_order" and pro.allocation == "pro_rata"
    assert order.ceded_by_event([15.0, 20.0, 10.0]) == pytest.approx([10.0, 10.0, 0.0])
    assert pro.ceded_by_event([15.0, 20.0, 10.0]) == pytest.approx([8.0, 8.0, 4.0])
    assert pro.ceded([15.0, 20.0, 10.0]) == pytest.approx(order.ceded([15.0, 20.0, 10.0]))
    # What an inuring 5x5 sees: 5, 10, 10 in event order, 7, 12, 6 pro rata.
    second = Layer("5x5", 5.0, 5.0)
    assert Tower.inuring([[order], [second]]).ceded([15.0, 20.0, 10.0]) == pytest.approx([20.0, 10.0])
    inuring = Tower.inuring([[pro], [second]])
    assert inuring.ceded([15.0, 20.0, 10.0]) == pytest.approx([20.0, 8.0])
    assert Tower.from_json(inuring.to_json()).ceded([15.0, 20.0, 10.0]) == pytest.approx([20.0, 8.0])
    with pytest.raises(ValueError):
        Layer("10x5", 10.0, 5.0, allocation="by_size")
    with pytest.raises(ValueError):
        Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0], pro_rata_time=True,
              allocation="pro_rata")


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


def test_contract_terms():
    from prospicio.reinsurance import LossSensitivePremium

    # Premiums from quotes, at the placed share.
    assert Layer("10x10", 10.0, 10.0, share=0.4).with_rate_on_line(0.2).premium == pytest.approx(0.8)
    assert Layer("10x10", 10.0, 10.0, share=0.4).with_deposit_premium(3.0).premium == pytest.approx(1.2)
    # A premium in the constructor is kept without paid reinstatements.
    assert Layer("L", 10.0, 10.0, premium=2.5).premium == 2.5
    qs = Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
    assert qs.premium == 100.0
    with pytest.raises(ValueError):
        Layer.quota_share("QS", 0.5).with_rate_on_line(0.1)

    # aggregate's sliding scale and profit commission examples.
    slide = qs.with_sliding_scale([(0.25, 0.70), (0.45, 0.60), (0.19, 0.80)])
    assert slide.sliding_scale == [(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)]
    assert slide.ceding_commission is None
    for lr, c in [(0.55, 0.45), (0.65, 0.35), (0.75, 0.22), (0.90, 0.19)]:
        assert slide.ceding_commission_for(100.0 * lr) == pytest.approx(100.0 * c)
    pc = qs.with_profit_commission(0.25, 0.10)
    assert pc.profit_commission == (0.25, 0.10)
    assert pc.profit_commission_for(60.0) == pytest.approx(7.5)
    assert pc.profit_commission_for(95.0) == 0.0
    assert qs.with_ceding_commission(0.3).ceding_commission_for(500.0) == pytest.approx(30.0)
    with pytest.raises(ValueError, match="premium"):
        Layer.quota_share("QS", 0.5).with_sliding_scale([(0.3, 0.6)])
    with pytest.raises(ValueError):
        qs.with_sliding_scale([(0.2, 0.6), (0.3, 0.7)])

    # Deficits and credits carried over a term.
    base = Layer.quota_share("QS", 1.0).with_premium_rate(1.0, 100.0)
    pcc = base.with_profit_commission(0.5, 0.2).with_profit_commission_carry_forward()
    assert pcc.profit_commission_carry == {"years": None}
    assert pc.profit_commission_carry is None
    ceded = [120.0, 70.0, 30.0, 85.0, 50.0]
    assert pcc.profit_commissions_over(ceded) == pytest.approx([0.0, 0.0, 10.0, 0.0, 12.5])
    pc1 = base.with_profit_commission(0.5, 0.2).with_profit_commission_carry_forward(years=1)
    assert pc1.profit_commissions_over(ceded) == pytest.approx([0.0, 0.0, 25.0, 0.0, 12.5])
    ss = base.with_sliding_scale([(0.4, 0.5), (0.2, 0.7)])
    ssc = ss.with_sliding_scale_carry_forward(deficit=False, credit=True)
    assert ssc.sliding_scale_carry == {"deficit": False, "credit": True, "years": None}
    assert ssc.ceding_commissions_over([90.0, 40.0, 30.0, 60.0]) == pytest.approx(
        [20.0, 40.0, 40.0, 40.0]
    )
    assert ssc.with_ceding_commission(0.3).sliding_scale_carry is None
    with pytest.raises(ValueError, match="profit commission first"):
        base.with_profit_commission_carry_forward()
    with pytest.raises(ValueError):
        ss.with_sliding_scale_carry_forward(deficit=False, credit=False)
    with pytest.raises(ValueError):
        ss.with_sliding_scale_carry_forward(years=0)

    # Swing rating at a 40% share: amounts scale, the factor does not.
    swing = Layer("L", 1000.0, 0.0, share=0.4).with_swing_rating(20.0, 1.25, 50.0, 300.0)
    assert swing.swing_rating == (20.0, 1.25, 50.0, 300.0)
    assert swing.premium_for(40.0) == pytest.approx(58.0)
    assert swing.premium_for(1000.0) == pytest.approx(120.0)
    with pytest.raises(ValueError, match="swing"):
        qs.with_sliding_scale([(0.3, 0.6)]).with_swing_rating(0.0, 1.0)
    with pytest.raises(ValueError):
        Layer("10x10", 10.0, 10.0, premium=1.0, reinstatement_rates=[1.0]).with_swing_rating(0.0, 1.0)

    retro = LossSensitivePremium(1000.0, 1.1, maximum=2500.0)
    assert retro.minimum == 1000.0 and retro.maximum == 2500.0
    assert retro.premium([0.0, 500.0, 1500.0]) == pytest.approx([1000.0, 1550.0, 2500.0])

    # In a tower: components per term, joint with the ceded losses.
    xl = Layer("3x3", 3.0, 3.0, share=0.5).with_swing_rating(1.0, 1.2, 2.0, 6.0).with_ceding_commission(0.1)
    book = Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 30.0)
    book = book.with_sliding_scale([(0.4, 0.5), (0.2, 0.9)]).with_profit_commission(0.3, 0.25)
    tower = Tower([book, xl])
    result = tower.apply(simulate_events(Poisson(3.0), SEV, 20_000, 9))
    assert [k[0] for k in result.components()] == [
        "gross", "ceded", "ceded", "net", "swing_premium", "ceding_commission", "ceding_commission",
        "profit_commission",
    ]
    rows = result.draw_matrix()
    for row in rows[:500]:
        assert row[4] == xl.premium_for(row[2])
        assert row[5] == book.ceding_commission_for(row[1])
        assert row[7] == book.profit_commission_for(row[1])
    # The grid's expectations agree with the simulation.
    grids = tower.on_grid(Poisson(3.0), SEV, 400)
    assert grids.expected_premium[0] == pytest.approx(15.0)
    sim = result.marginal(("ceding_commission", "QS")).draws
    mean = sum(sim) / len(sim)
    se = math.sqrt(sum((x - mean) ** 2 for x in sim) / len(sim) / len(sim))
    assert abs(grids.expected_ceding_commission[0] - mean) < 4 * se
    assert grids.expected_profit_commission[1] == 0.0
    # Saved and loaded with every term.
    assert Tower.from_json(tower.to_json()).to_json() == tower.to_json()


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


def test_grid_audit():
    from prospicio.aggregate import (
        audit_grid,
        audit_portfolio_grid,
        recommend_grid,
        recommend_portfolio_grid,
    )
    from prospicio.distributions import Gamma, Lognormal, PolicyLimits, Poisson

    # aggregate 1.2.0's sharpen on "100 claims 1000 xs 0 sev lognorm 100 cv 1.5
    # poisson" from bs 1, log2 13: it moves to bs 4.
    capped = PolicyLimits(Lognormal.from_mean_cv(100.0, 1.5), [1000.0], [1.0])
    a = audit_grid(Poisson(100.0), capped, step=1.0, log2=13)
    assert (a.step, a.log2, a.grew, a.probed) == (4.0, 13, False, True)
    assert a.best["score"] == pytest.approx(0.035326, rel=1e-4)
    assert a.best["aggregate_cv"] == pytest.approx(0.020144, rel=1e-4)
    assert a.cells[0]["note"] == "center" and a.cells[0]["defective"]
    assert len(a) == len(a.cells) and a.cells[a.selected]["selected"]
    # From aggregate's 0.5, 2^10 on the gamma book log2 grows to 11.
    g = audit_grid(Poisson(50.0), Gamma.from_mean_cv(10.0, 0.5), step=0.5, log2=10)
    assert (g.step, g.log2, g.grew) == (1.0, 11, True)
    assert g.best["score"] == pytest.approx(0.722254, rel=1e-5)
    # Without a center, recommend_grid's grid is audited.
    auto = audit_grid(Poisson(50.0), Gamma.from_mean_cv(10.0, 0.5), log2=12)
    first = recommend_grid(Poisson(50.0), Gamma.from_mean_cv(10.0, 0.5), log2=12)
    assert (auto.cells[0]["step"], 2 ** auto.cells[0]["log2"]) == (first.step, first.points)
    # A one-unit portfolio audits as the unit.
    units = [(Poisson(50.0), Gamma.from_mean_cv(10.0, 0.5))]
    book = audit_portfolio_grid(units, step=0.5, log2=10)
    assert [c["score"] for c in book.cells] == pytest.approx([c["score"] for c in g.cells])
    two = [(Poisson(10.0), Gamma(2.0, 50.0)), (Poisson(2.0), Gamma(0.5, 1000.0))]
    size = recommend_portfolio_grid(two, log2=12)
    assert size.points <= 2**12
    assert audit_portfolio_grid(two, log2=12).best["score"] < 0.5
    with pytest.raises(ValueError):
        audit_grid(Poisson(5.0), capped, step=-1.0, log2=10)
    with pytest.raises(ValueError):
        recommend_portfolio_grid([])


def test_occurrence_pml():
    from prospicio.aggregate import occurrence_exceedance, occurrence_pml, simulate_events
    from prospicio.distributions import Lognormal, NegativeBinomial, Poisson

    sev = Lognormal.from_mean_cv(100.0, 2.0)
    # aggregate 1.0.1's oep on "agg O 2 claims sev lognorm 100 cv 2 poisson".
    for p, loss, rp in [(0.1, 348.934754485, 9.491222), (0.001, 2906.643201, 999.499917)]:
        pml = occurrence_pml(Poisson(2.0), sev, p)
        assert pml.loss == pytest.approx(loss, rel=1e-9)
        assert pml.occurrence_return_period == pytest.approx(rp, rel=1e-6)
        assert pml.annual_return_period == pytest.approx(1 / p)
    nb = NegativeBinomial(1.5, 2.0)
    pml = occurrence_pml(nb, sev, 0.05)
    assert occurrence_exceedance(nb, sev, pml.loss) == pytest.approx(0.05, rel=1e-9)
    # The simulated OEP at the 10% PML, within four standard errors.
    maxima = [row[0] for row in simulate_events(Poisson(2.0), sev, 20_000, 7).maxima().draw_matrix()]
    share = sum(m > 348.934754485 for m in maxima) / len(maxima)
    assert abs(share - 0.1) < 4 * math.sqrt(0.1 * 0.9 / 20_000)
    with pytest.raises(ValueError):
        occurrence_pml(Poisson(0.1), sev, 0.2)


def test_compound_moments_and_approximations():
    from prospicio.aggregate import compound_moments
    from prospicio.distributions import Gamma, Pareto, Poisson

    m = compound_moments(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    # aggregate 1.0.1 on "agg B 10 claims sev gamma 50 cv 0.7 poisson".
    assert m.mean == pytest.approx(500.0, rel=1e-12)
    assert m.cv == pytest.approx(0.38600518131237566, rel=1e-12)
    assert m.skew == pytest.approx(0.5129464825493272, rel=1e-12)
    expected = {
        "gamma": {"shape": 6.7114093959731544, "scale": 74.5},
        "lognormal": {"sdlog": 0.3726821686995752},
        "shifted_gamma": {"shape": 15.202530354045754, "shift": -252.52525252525868},
        "shifted_lognormal": {"sdlog": 0.16816686089041927, "shift": -639.5805099091635},
    }
    for kind, params in expected.items():
        fit = m.approximate(kind)
        got = dict(fit.parameters)
        for k, v in params.items():
            assert got[k] == pytest.approx(v, rel=1e-10), (kind, k)
        assert fit.mean() == pytest.approx(500.0, rel=1e-12)
        assert fit.cdf(fit.quantile(0.99)) == pytest.approx(0.99, abs=1e-10)
    with pytest.raises(ValueError):
        m.approximate("weibull")
    with pytest.raises(ValueError):
        compound_moments(Poisson(2.0), Pareto(100.0, 2.5))


def test_layer_statistics_and_prices_by_view():
    from prospicio.aggregate import compound_moments
    from prospicio.reinsurance import occurrence_layer_statistics
    from prospicio.risk import Distortion

    sev, _ = Grid.rounding(Lognormal.from_mean_cv(100.0, 2.0), 1.0, 1001)
    n = Poisson(2.0)
    s = occurrence_layer_statistics(n, sev, 1.0, 500.0, 100.0)
    # aggregate 1.0.1, reins_stats_df (validation/scripts/aggregate_reinsurance.py)
    assert s.pr_attach == pytest.approx(0.261655626646, rel=1e-10)
    assert s.count.mean == pytest.approx(2.0 * s.pr_attach, rel=1e-14)
    assert s.loss_on_line == pytest.approx(0.151169062665, rel=1e-9)
    assert s.aggregate.mean == pytest.approx(s.count.mean * s.severity.mean, rel=1e-12)
    half = occurrence_layer_statistics(n, sev, 0.5, 500.0, 100.0)
    assert half.aggregate.skew == pytest.approx(s.aggregate.skew, rel=1e-12)
    with pytest.raises(ValueError):
        occurrence_layer_statistics(n, sev, 0.0, 500.0, 100.0)

    grids = Tower([Layer("occ", 500.0, 100.0)]).on_grid(n, sev, 1 << 14)
    p = grids.price(Distortion.proportional_hazard(0.7))
    assert p.gross.premium == pytest.approx(297.443189953534, rel=2e-8)
    assert p.ceded[0].loss == pytest.approx(s.aggregate.mean, rel=1e-9)
    assert p.net.margin == pytest.approx(p.net.premium - p.net.loss)
    # Distortion prices are subadditive.
    assert p.ceded[0].premium + p.net.premium >= p.gross.premium
    assert compound_moments(n, sev).mean == pytest.approx(p.gross.loss, rel=1e-9)


def test_eventual_ruin():
    from prospicio.aggregate import eventual_ruin
    from prospicio.distributions import Grid

    # Claims of 1, 2 or 3; aggregate 1.0.1's pollaczeck_khinchine at rho 0.1.
    sev = Grid(1.0, [0.0, 0.5, 0.3, 0.2] + [0.0] * 1020)
    ruin = eventual_ruin(sev, rho=0.1)
    assert ruin.u[:3] == [0.0, 1.0, 2.0]
    assert ruin.psi_at(10.0) == pytest.approx(0.15362022867039649, abs=1e-12)
    assert ruin.severity_mean == pytest.approx(1.7)
    assert ruin.capital(0.01) is not None
    assert ruin.lundberg_exponent > 0
    assert ruin.report.aliasing_error < 1e-12
    same = eventual_ruin(sev, loss_ratio=1 / 1.1)
    assert same.psi == pytest.approx(ruin.psi, abs=1e-14)
    capped = eventual_ruin(sev, rho=0.1, cap=2.0)
    assert capped.severity_mean == pytest.approx((0.5 + 0.6) / 0.8)
    with pytest.raises(ValueError):
        eventual_ruin(sev, rho=0.1, loss_ratio=0.9)
    with pytest.raises(ValueError):
        eventual_ruin(sev, rho=0.1, cap=2.0, excess=1.0)
    with pytest.raises(ValueError):
        eventual_ruin(sev, rho=0.0)



def test_wiener_hopf():
    from prospicio.aggregate import wiener_hopf
    from prospicio.distributions import Gamma, Grid

    # Claims of 1, 2 or 3 after gamma waits (shape 2); aggregate 1.0.1's
    # wiener_hopf at rho 0.1 on the same grid.
    sev = Grid(1.0, [0.0, 0.5, 0.3, 0.2] + [0.0] * 1020)
    ruin = wiener_hopf(sev, Gamma(2.0, 1.0), rho=0.1)
    assert ruin.psi_at(10.0) == pytest.approx(0.1973674584712206, abs=1e-10)
    assert ruin.capital(0.01) == 31.0
    assert ruin.premium_rate == pytest.approx(1.1 * 1.7 / 2.0)
    assert ruin.mean_step < 0.0
    assert len(ruin.maximum) == len(ruin.psi) == 1024
    same = wiener_hopf(sev, Gamma(2.0, 1.0), loss_ratio=1 / 1.1)
    assert same.psi == pytest.approx(ruin.psi, abs=1e-14)
    with pytest.raises(ValueError):
        wiener_hopf(sev, Gamma(2.0, 1.0), rho=0.0)
    with pytest.raises(ValueError):
        wiener_hopf(sev, Gamma(2.0, 1.0), rho=0.1, points=8)


def test_clash():
    from prospicio.aggregate import Clash, JointGrid, fft_bivariate
    from prospicio.distributions import Grid, NegativeBinomial, Poisson

    # aggregate 1.0.1: clash C 2 3 1 claims dsev [1 2] [.5 .5] dsev [1] poisson,
    # bs 1 and log2 (7, 7) on each axis.
    clash = Clash(2.0, 3.0, 1.0)
    assert (clash.neither, clash.events) == (6.0, 12.0)
    assert clash.trigger_b == pytest.approx(1 / 3)
    a = Grid(1.0, [0.0, 0.5, 0.5])
    b = Grid(1.0, [0.0, 1.0])
    joint, report = clash.aggregate(Poisson(clash.events), a, b, (128, 128))
    assert joint.shape == (128, 128) and joint.steps == (1.0, 1.0)
    assert joint.prob(3, 2) == pytest.approx(0.02323830165624712, abs=1e-12)
    assert joint.probs[0][0] == pytest.approx(0.0024787521766663594, abs=1e-12)
    assert joint.correlation() == pytest.approx(0.2738612787524785, rel=1e-9)
    assert joint.total().probs[5] == pytest.approx(0.08374051103504515, abs=1e-12)
    assert max(report.aliasing_mass) < 1e-14
    assert joint.marginal(0).mean() == pytest.approx(report.expected_means[0])
    # The same through the general entry point.
    same, _ = fft_bivariate(Poisson(12.0), clash.severity(a, b), (128, 128))
    assert same.covariance() == pytest.approx(joint.covariance(), rel=1e-12)
    with pytest.raises(ValueError):
        clash.aggregate(NegativeBinomial(1.0, 1.0), a, b, (8, 8))
    with pytest.raises(ValueError):
        Clash(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        JointGrid((1.0, 1.0), [[0.5, 0.5], [0.5]])
    with pytest.raises(IndexError):
        joint.prob(128, 0)


def test_layer_loss_picks():
    from prospicio.aggregate import layer_loss_picks
    from prospicio.distributions import Gamma, Grid

    sev, _ = Grid.rounding(Gamma.from_mean_cv(100.0, 1.5), 1.0, 1 << 14)
    picked = layer_loss_picks(sev, [100, 500, 1000], [480, 420, 60], claims=10)
    assert not picked.straight_bottom
    # aggregate 1.0.1's Aggregate.picks at loss 200 (picks_aggregate.csv).
    assert picked.severity.probs[200] == pytest.approx(0.0008386073686922624, abs=1e-12)
    layers = picked.to_dict()
    assert layers["top"] == [100.0, 500.0, 1000.0]
    assert layers["target"] == pytest.approx([48.0, 42.0, 6.0])
    assert layer_loss_picks(sev, [100, 500, 1000], [700, 250, 50], claims=10).straight_bottom
    with pytest.raises(ValueError):
        layer_loss_picks(sev, [100.5], [10.0])
    with pytest.raises(ValueError):
        layer_loss_picks(sev, [100, 500], [480, 1], claims=10)


def test_stage_cession_and_aggregate_cover_statistics():
    from prospicio.distributions import Gamma, Grid, Poisson
    from prospicio.reinsurance import (
        Layer,
        Tower,
        aggregate_layer_statistics,
        occurrence_cession_statistics,
    )
    from prospicio.risk import Distortion

    # Book T of validation/scripts/aggregate_reinsurance.py.
    sev, _ = Grid.rounding(Gamma.from_mean_cv(50.0, 1.5), 1.0, 1001)
    n = Poisson(5.0)
    layers = [(1.0, 100.0, 100.0), (1.0, 200.0, 200.0)]
    c = occurrence_cession_statistics(n, sev, layers)
    occ = Tower([Layer("a", 100.0, 100.0), Layer("b", 200.0, 200.0)]).on_grid(n, sev, 1 << 14)
    total = occ.stage_ceded[0]
    # aggregate 1.0.1's reins_stats_df and reins_price_df.
    assert c.aggregate.mean == pytest.approx(67.33279562141576, rel=1e-9)
    price = occ.price(Distortion.proportional_hazard(0.7))
    assert price.stages[0].premium == pytest.approx(111.0007062267948, rel=2e-8)
    assert price.stages[0].loss == pytest.approx(sum(v.loss for v in price.ceded), rel=1e-10)
    assert len(occ.stage_ceded) == 1
    s = aggregate_layer_statistics(occ.net, 1.0, 300.0, 200.0)
    assert s.pr_attach == pytest.approx(0.3912338371214888, abs=1e-12)
    assert s.aggregate.mean == pytest.approx(38.7761195844111, rel=1e-9)
    assert s.loss_on_line == pytest.approx(s.aggregate.mean / 300.0)


def test_quota_share_on_an_annual_grid():
    from prospicio.reinsurance import Layer, Tower

    # Loss ratios 50% to 80% on a premium of 100, equally likely (#206).
    probs = [0.0] * 81
    for k in (50, 60, 70, 80):
        probs[k] = 0.25
    qs = (
        Layer.quota_share("QS", 0.5)
        .with_premium_rate(1.0, 200.0)
        .with_sliding_scale([(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)])
    )
    r = Tower([qs]).on_annual_grid(Grid(1.0, probs))
    expected = sum(0.25 * qs.ceding_commission_for(0.5 * k) for k in (50, 60, 70, 80))
    assert r.expected_ceding_commission[0] == pytest.approx(expected, abs=1e-12)
    assert r.ceded[0].mean() + r.net.mean() == pytest.approx(65.0, abs=1e-12)


def test_event_loss_table_against_its_oep():
    from prospicio.aggregate import EventLossTable, EventSet

    elt = EventLossTable(
        [0.01, 0.04, 0.2], [10_000.0, 3000.0, 800.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [10_000.0, 3000.0, 800.0],
    )
    assert elt.oep(5000.0) == pytest.approx(1 - math.exp(-0.01), rel=1e-14)
    assert elt.exceedance_frequency(500.0) == pytest.approx(0.25, rel=1e-14)
    n = 50_000
    events, ids = elt.simulate(n, seed=3)
    hit = sum(1 for y in range(n) if max(events.events(y), default=0.0) > 2000.0) / n
    p = elt.oep(2000.0)
    assert abs(hit - p) < 4 * math.sqrt(p * (1 - p) / n)
    assert [len(i) for i in ids] == events.counts()
    beta = EventLossTable([0.05], [500.0], [200.0], [50.0], [2000.0])
    assert beta.oep(beta.occurrence_pml(0.01)) == pytest.approx(0.01, abs=1e-9)
    with pytest.raises(ValueError, match="row 0"):
        EventLossTable([0.1], [50.0], [60.0], [0.0], [100.0])
    long = EventSet.from_long([1, 1], [3.0, 4.0], 3)
    assert long.counts() == [0, 2, 0]


def test_merge_event_sets_and_hours_clause():
    from prospicio.aggregate import EventSet

    h = 1 / 8760
    cat = EventSet.from_years([[100.0], []], times=[[0.5], []])
    other = EventSet.from_years([[3.0, 4.0], [5.0]], times=[[0.2, 0.7], [0.1]])
    m = cat.merge(other)
    assert m.events(0) == [3.0, 100.0, 4.0] and m.events(1) == [5.0]
    with pytest.raises(ValueError):
        cat.merge(EventSet.from_years([[1.0], []]))
    e = EventSet.from_years([[1.0, 2.0, 4.0]], times=[[0.0, 72 * h, 72.5 * h]])
    assert e.grouped(72.0).events(0) == [3.0, 4.0]
    with pytest.raises(ValueError):
        EventSet.from_years([[1.0]]).grouped(72.0)


def test_copula_coupled_lines():
    from prospicio.aggregate import JointGrid, fft_bivariate
    from prospicio.risk import ArchimedeanCopula, GaussianCopula, StudentTCopula

    a = Grid(1.0, [0.1, 0.3, 0.4, 0.2])
    b = Grid(2.0, [0.2, 0.5, 0.3])
    s = JointGrid.from_copula(a, b, ArchimedeanCopula("clayton", 3.0, 2), trigger_a=0.5)
    # The marginals come back: A is triggered half the time.
    ma = s.marginal(0).probs
    assert ma[0] == pytest.approx(0.5 + 0.5 * 0.1, abs=1e-15)
    assert s.marginal(1).probs == pytest.approx(b.probs, abs=1e-15)
    assert s.correlation() > 0
    zero = JointGrid.from_copula(a, b, GaussianCopula([[1.0, 0.0], [0.0, 1.0]]))
    assert zero.probs[2][1] == pytest.approx(0.4 * 0.5, abs=1e-15)
    joint, _ = fft_bivariate(Poisson(3.0), s, (64, 64))
    assert joint.correlation() > 0
    with pytest.raises(ValueError):
        JointGrid.from_copula(a, b, StudentTCopula([[1.0, 0.5], [0.5, 1.0]], 4.0))


def test_ledger():
    import math

    from prospicio.aggregate import Group, Ledger, Leg
    from prospicio.distributions import Grid

    # The P&L design note's example 2 against aggregate 1.0.1's PnL.
    x, p = [0.0, 10.0, 20.0, 50.0], [0.4, 0.3, 0.2, 0.1]
    gross = Group(
        "Gross", "sell",
        [Leg("Premium", lambda v: [min(max(5 + 1.1 * a, 12), 40) for a in v], kind="premium")],
        [Leg("Loss", lambda v: v, kind="loss"), Leg("LAE", lambda v: [0.05 * a for a in v])],
    )
    xl = Group(
        "XL", "buy", [Leg("XL premium", 4.0)],
        [Leg("XL recovery", lambda v: [min(max(a - 15, 0), 25) for a in v]),
         Leg("XL commission", 0.4)],
    )
    pnl = Ledger((x, p), [gross, xl], name="Ex2", result_name="margin")
    s = pnl.sheet()
    i = s["row"].index("margin")
    assert s["EX"][i] == pytest.approx(6.3, abs=1e-12)
    assert s["SD"][i] == pytest.approx(2.913760457, abs=1e-9)
    assert [s[k][i] for k in ("k01", "k50", "k99")] == pytest.approx([1.9, 7.4, 8.9], abs=1e-12)
    j = s["row"].index("Premium")
    assert s["k50"][j] == pytest.approx(27.0, abs=1e-12) and s["k75"][j] == pytest.approx(12.0)
    assert s["step"][-1] == "All" and s["label"][-1] == "Impact"
    card = pnl.card()
    assert card["side"][-1] == "Impact" and card["EX"][-1] == pytest.approx(-0.1, abs=1e-12)
    assert pnl.legs()["kind"][:2] == ["premium", "loss"]
    assert pnl.marginal()["P99"][i] == pytest.approx(8.9)
    assert pnl.result.tvar(0.5) == pytest.approx(8.5, abs=1e-12)
    assert pnl.row("Loss").mean() == pytest.approx(-12.0)
    assert pnl.prob_eq_0 == 0.0
    # A grid as the source; one group without a buyer reads Total.
    g = Grid(1.0, [0.5, 0.0, 0.5])
    one = Ledger(g, [Group("Book", "sell", [Leg("P", 3.0)], [Leg("L", lambda v: v)])])
    assert one.sheet()["label"] == ["P", "L", "Total"] and one.result.mean() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        Leg("L", 1.0, kind="bogus")
    with pytest.raises(ValueError):
        Group("A", "lend")
    with pytest.raises(ValueError):
        Ledger(g, [Group("A", "sell", [Leg("P", 1.0)]), Group("B", "buy", [Leg("P", 1.0)])])
    assert not math.isnan(one.sheet()["EX"][0])


def test_ledger_ratios_and_stack():
    from prospicio.aggregate import Group, Ledger, Leg, stack

    # The P&L design note's examples 2, 2b and 2c against aggregate 1.0.1.
    x, p = [0.0, 10.0, 20.0, 50.0], [0.4, 0.3, 0.2, 0.1]
    gross = Group(
        "Gross", "sell",
        [Leg("Premium", lambda v: [min(max(5 + 1.1 * a, 12), 40) for a in v], kind="premium")],
        [Leg("Loss", lambda v: v, kind="loss"),
         Leg("LAE", lambda v: [0.05 * a for a in v], kind="expense")],
    )
    xl = Group(
        "XL", "buy", [Leg("XL premium", 4.0, kind="premium")],
        [Leg("XL recovery", lambda v: [min(max(a - 15, 0), 25) for a in v], kind="recovery"),
         Leg("XL commission", 0.4, kind="commission")],
    )
    r = Ledger((x, p), [gross, xl], name="Ex2", result_name="margin").ratios()
    assert r["step"] == ["Gross", "XL", "All"]
    assert [r[k][0] for k in ("P", "L", "E", "M")] == pytest.approx([19.0, 12.0, 0.6, 6.4])
    assert r["E_LR"][0] == pytest.approx(0.46064814814814825, abs=1e-12)
    assert r["E_ER"][2] == pytest.approx(-0.006449275362318844, abs=1e-12)
    assert r["M_share"][2] == pytest.approx(0.984375, abs=1e-12)

    b = Ledger((x, p), [Group("Ex2b", "sell", [Leg("consideration", 20.0)],
                              [Leg("Loss", lambda v: v * 1.03, step=4.0)])])
    v = b.validation()
    assert v["label"] == ["Loss"] and v["EX"][0] == pytest.approx(-12.36, abs=1e-12)
    assert v["abs_err"][0] < 1e-12
    m = b.marginal()
    i = m["row"].index("Loss")
    assert [m[k][i] for k in ("P01", "P10", "P50")] == [-52.0, -48.0, -8.0]
    # The sheet's scenario columns stay on the exact values.
    assert b.sheet()["k01"][i] == pytest.approx(-51.5)
    c = Ledger((x, p), [Group("Ex2c", "sell",
                              [Leg("Prem", lambda v: [30.0 if a == 20 else 10.0 for a in v])],
                              [Leg("Loss", lambda v: v)])])
    s = stack({"ex2b": b, "ex2c": c}, [("c less b", "ex2c", "ex2b")])
    assert s["label"] == ["ex2b", "ex2c", "c less b"]
    assert s["EX"][2] == pytest.approx(-5.64, abs=1e-12)
    assert s["P01"][2] == pytest.approx(-8.5, abs=1e-12)
    with pytest.raises(ValueError):
        stack({"ex2b": b}, [("x", "ex2b", "nope")])


def test_book_builders():
    from prospicio.aggregate import Book, Expense, derive_premium
    from prospicio.distributions import Grid
    from prospicio.reinsurance import Layer, Tower

    # The P&L design note's examples 3a, 3b and 6 against aggregate 1.0.1.
    gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    high = Layer.stop_loss("High", 3.0, 5.0).with_deposit_premium(1.0).with_ceding_commission(0.1)
    low = Layer.stop_loss("Low", 2.0, 3.0).with_deposit_premium(0.5)
    book = Book("Ex6", 10.0, [Expense("premium", 0.1)])
    walk = book.ledger(gross, Tower([high]), "walk")
    s = walk.sheet()
    i = s["row"].index("High recovery")
    assert s["EX"][i] == pytest.approx(0.24) and s["k01"][i] == pytest.approx(2 / 3)
    net = book.ledger(gross, Tower([high]))
    assert net.result.mean() == pytest.approx(4.54) and "commission" in net.rows
    assert net.row("net premium").mean() == pytest.approx(9.0)
    peel = book.ledger(gross, Tower([low, high]), "peel-top-down")
    steps = peel.sheet()["step"]
    assert steps.index("High") < steps.index("Low") and "All aggregate" in steps
    assert peel.result.mean() == pytest.approx(4.85)
    # The plain book; a Grid works as the source.
    plain = Book("P", 10.0, [Expense("premium", 0.1, label="Acq"), Expense("loss", 0.05)])
    g = Grid(1.0, [0.0, 0.0, 0.5, 0.5])
    assert plain.ledger(g).result.mean() == pytest.approx(10 - 2.5 - 1 - 0.125)
    mixed = Expense([("premium", 0.1), ("loss", 0.05)])
    assert mixed.terms == [("premium", 0.1), ("loss", 0.05)]
    assert derive_premium(5.0, [Expense("premium", 0.1), Expense("fixed", 1.0)]) == pytest.approx(6 / 0.9)
    with pytest.raises(ValueError):
        derive_premium(5.0, [Expense("loss", 0.1)])
    with pytest.raises(ValueError):
        book.ledger(gross, None, "peel-top-down")
    with pytest.raises(ValueError):
        book.ledger(gross, None, "sideways")
    with pytest.raises(ValueError):
        Expense("bogus", 1.0)


def test_book_occurrence_covers():
    from prospicio.aggregate import Book, Expense, occurrence_source
    from prospicio.distributions import Grid, Poisson
    from prospicio.reinsurance import Layer, Tower

    # The P&L design note's example 4 against aggregate 1.0.1: gross margin
    # 14.2, the occurrence cover's recovery 0.8 and margin -0.7, net 13.5.
    sev = Grid(1.0, [0.0, 0.5, 0.3, 0.0, 0.2])
    occ = Tower([Layer("occ 2 xs 2", 2.0, 2.0).with_deposit_premium(2.0).with_ceding_commission(0.25)])
    joint, report = occurrence_source(Poisson(2.0), sev, occ, (64, 32))
    assert len(joint) == 3 and len(joint[0]) == 280
    book = Book("Ex4", 20.0, [Expense("premium", 0.1)])
    walk = book.occurrence_ledger(joint, occ, face="walk")
    s = walk.sheet()
    net = s["row"].index("net through occ 2 xs 2")
    assert s["EX"][net] == pytest.approx(13.5, abs=1e-10)
    assert s["SD"][net] == pytest.approx(2.236067977, abs=1e-8)
    assert s["k01"][net] == pytest.approx(7.5, abs=1e-10)
    rec = s["row"].index("occ 2 xs 2 recovery")
    assert s["EX"][rec] == pytest.approx(0.8, abs=1e-10)
    agg = Tower([Layer.stop_loss("agg 3 xs 5", 3.0, 5.0).with_deposit_premium(0.5)])
    c = book.occurrence_ledger(joint, occ, agg)
    w = book.occurrence_ledger(joint, occ, agg, "walk")
    assert c.result.mean() == pytest.approx(w.result.mean(), abs=1e-12)
    assert "Net of occurrence" in w.sheet()["step"]
    with pytest.raises(ValueError):
        book.occurrence_ledger(([1.0], [1.0]), occ)
    two = Tower([Layer("a", 1.0, 1.0), Layer("b", 2.0, 2.0)])
    with pytest.raises(ValueError):
        book.occurrence_ledger(joint, two, face="peel-top-down")


def test_book_variable_terms_and_program():
    from prospicio.aggregate import Book, Expense, pnl_program, round_premium, size_premium
    from prospicio.reinsurance import Layer, LossSensitivePremium, Tower

    # aggregate 1.0.1: a 3 xs 5 cover with a sliding scale, or a profit
    # commission, on example 1's book.
    gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    book = Book("V", 10.0, [Expense("premium", 0.1)])
    base = Layer.stop_loss("ceded agg", 3.0, 5.0).with_deposit_premium(1.0)
    slide = Tower([base.with_sliding_scale([(0.3, 0.2), (0.1, 0.6)])])
    s = book.ledger(gross, slide, "walk").sheet()
    assert s["EX"][s["row"].index("sliding commission")] == pytest.approx(0.268, abs=1e-12)
    pc = Tower([base.with_profit_commission(0.25, 0.1)])
    c = book.ledger(gross, pc)
    assert c.result.mean() == pytest.approx(4.629, abs=1e-12)
    assert c.sheet()["SD"][c.sheet()["row"].index("commission")] > 0
    # Swing rating: the ceded premium varies with the ceded loss.
    swing = Tower([Layer.stop_loss("ceded agg", 3.0, 5.0).with_swing_rating(0.5, 1.1, 0.6, 2.5)])
    w = book.ledger(gross, swing, "walk").sheet()
    assert w["EX"][w["row"].index("ceded agg premium")] == pytest.approx(-0.796, abs=1e-12)
    # Retro: the premium is the collar at the gross loss.
    retro = Book("V", 2.0, [Expense("premium", 0.1)],
                 retro=LossSensitivePremium(2.0, 1.1, minimum=3.0, maximum=9.0))
    assert retro.ledger(gross).row("Premium").mean() == pytest.approx(6.108, abs=1e-12)
    with pytest.raises(ValueError):
        retro.ledger(gross, pc)
    # pnl_program's ladder, against aggregate's on the design note's book.
    p = pnl_program(405.97794085441933, 0.9, 0.25, [
        dict(ceded_loss=86.78504710067914, share=1.0, combined_ratio=0.6),
        dict(ceded_loss=8.144345556686755, share=1.0, combined_ratio=0.5)])
    assert p["premium"] == 817.0 and p["deposits"] == [145.0, 16.29]
    kept = pnl_program(406.0, 0.9, 0.25, [dict(share=0.5, stated=50.0)])
    assert kept["deposits"] == [100.0] and kept["ceded_premium"] == 50.0
    assert round_premium(146.5) == 146.0 and size_premium(499.885, 0.7) == 714.0
    with pytest.raises(ValueError):
        size_premium(-1.0, 0.7)


def test_book_reinstatements():
    from prospicio.aggregate import Book, Expense, occurrence_source
    from prospicio.distributions import Grid, Poisson
    from prospicio.reinsurance import Layer, Tower

    # aggregate 1.0.1: an occurrence cover 2 xs 2 with one reinstatement at
    # 100% on example 4's book; the premium D + h(R) varies with the year.
    sev = Grid(1.0, [0.0, 0.5, 0.3, 0.0, 0.2])
    occ = Tower([Layer("occ 2 xs 2", 2.0, 2.0, premium=2.0, reinstatement_rates=[1.0])])
    joint, _ = occurrence_source(Poisson(2.0), sev, occ, (64, 32))
    book = Book("R", 20.0, [Expense("premium", 0.1)])
    s = book.occurrence_ledger(joint, occ, face="walk").sheet()
    prem = s["row"].index("occ 2 xs 2 premium")
    assert s["EX"][prem] == pytest.approx(-2.65936, abs=1e-5)
    assert s["SD"][prem] == pytest.approx(0.940194, abs=1e-6)
    net = book.occurrence_ledger(joint, occ)
    assert net.result.mean() == pytest.approx(12.323104, abs=1e-6)
    with pytest.raises(ValueError):
        book.ledger(([1.0, 5.0], [0.5, 0.5]), occ, "walk")


def test_pnl_program_rates_and_stated_premium():
    from prospicio.aggregate import pnl_program, pnl_program_stated

    # aggregate 1.0.1's premium_style='rate' on the design note's engine.
    p = pnl_program(405.97794085441933, 0.9, 0.25, [
        dict(ceded_loss=86.78504710067914, share=1.0, combined_ratio=0.6),
        dict(ceded_loss=8.144345556686755, share=1.0, combined_ratio=0.5, aggregate=True)])
    assert p["rates"] == [0.17747858, 0.024241071]
    # An engine stating 1000 of premium, with and without cover.
    s = pnl_program_stated(1000.0, 0.25, [
        dict(ceded_loss=112.82015639464626, share=1.0, combined_ratio=0.6),
        dict(ceded_loss=25.56634375990714, share=1.0, combined_ratio=0.5, aggregate=True)])
    assert s["premium"] == 1652.0 and s["rates"] == [0.11380145, 0.034924863]
    assert pnl_program_stated(1000.0, 0.25)["premium"] == pytest.approx(1000 / 0.75, abs=1e-9)
    kept = pnl_program(406.0, 0.9, 0.25, [dict(share=0.5, stated=20.0)])
    assert kept["rates"] == [None]


def test_consolidated_loss_expense_each_year():
    from prospicio.aggregate import Book, Expense
    from prospicio.reinsurance import Layer, Tower

    gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    cover = Tower([Layer.stop_loss("SL", 3.0, 5.0).with_deposit_premium(1.0)])
    book = Book("B", 10.0, [Expense("loss", 0.05)])
    c, w = book.ledger(gross, cover), book.ledger(gross, cover, "walk")
    # The consolidated margin is the walk's net margin, year by year.
    assert c.result.std_dev() == pytest.approx(w.result.std_dev(), abs=1e-12)
    m = Book("B", 10.0, [Expense("loss", 0.05)], expense_at_expected_loss=True).ledger(gross, cover)
    assert m.result.mean() == pytest.approx(c.result.mean(), abs=1e-12)
    assert m.row("expense").std_dev() == 0.0 and c.row("expense").std_dev() > 0.0


def test_stitched_occurrence_peel():
    from prospicio.aggregate import Book, Expense
    from prospicio.distributions import Grid, Poisson
    from prospicio.reinsurance import Layer, Tower

    # aggregate 1.0.1's stitched peel: two occurrence layers, top down.
    sev = Grid(1.0, [0.0, 0.4, 0.3, 0.0, 0.2, 0.0, 0.1])
    occ = Tower([Layer("occ 1 xs 1", 1.0, 1.0).with_deposit_premium(1.0),
                 Layer("occ 2 xs 2", 2.0, 2.0).with_deposit_premium(1.0).with_ceding_commission(0.2)])
    book = Book("S", 20.0, [Expense("premium", 0.1)])
    peel = book.occurrence_peel(Poisson(2.0), sev, occ, points=64)
    assert peel.is_stitched
    s = peel.sheet()
    assert "k01" not in s
    net = s["row"].index("net through occ 1 xs 1")
    assert s["SD"][net] == pytest.approx(1.897366596, abs=1e-8)
    assert s["EX"][net] == pytest.approx(13.8, abs=1e-9)
    with pytest.raises(ValueError):
        peel.row("total impact")
    with pytest.raises(ValueError):
        book.occurrence_peel(Poisson(2.0), sev, occ, face="walk", points=64)


def test_select_ceded_before_joining_a_book():
    # Issue #193: join only the ceded part of a tower's result to a book.
    from prospicio.aggregate import simulate_events
    from prospicio.distributions import Lognormal, Poisson, PredictiveDistribution
    from prospicio.reinsurance import Layer, Tower

    ev = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 2000, seed=7)
    res = Tower([Layer("5x5", 5e6, 5e6), Layer("10x10", 10e6, 10e6)]).apply(ev)
    ceded = res.select("kind", ["ceded"])
    assert ceded.dims == ["kind", "layer"] and len(ceded.components()) == 2
    by_kind = res.aggregate(["kind"])
    assert ceded.mean() == pytest.approx(by_kind.marginal(["ceded"]).mean(), rel=1e-12)
    with pytest.raises(ValueError):
        res.select("kind", ["nothing"])
    with pytest.raises(ValueError):
        res.select("lob", ["ceded"])
    two = res.select("kind", ["gross", "net"])
    assert len(two.components()) == 2


def test_tower_over_a_term_carries_deficits():
    events = simulate_events(Poisson(3.0), Lognormal.from_mean_cv(10.0, 1.5), 3_000, seed=11)
    years = events.term_years(3)
    assert [y.n_sims for y in years] == [1_000] * 3
    assert years[2].events(7) == events.events(2_007)
    layer = (
        Layer.quota_share("QS", 1.0)
        .with_premium_rate(1.0, 30.0)
        .with_profit_commission(0.3, 0.25)
        .with_profit_commission_carry_forward()
    )
    tower = Tower([layer])
    term = tower.apply_term(years)
    assert term.dims == ["kind", "layer", "year"]
    import numpy as np

    m = np.array(term.draw_matrix())
    assert m.shape == (1_000, 12)
    for j in (0, 16, 499):
        assert list(m[j, [3, 7, 11]]) == pytest.approx(
            layer.profit_commissions_over(list(m[j, [1, 5, 9]]))
        )
    alone = np.array(tower.apply(years[0]).draw_matrix())
    assert (alone == m[:, :4]).all()
    agg = tower.apply_aggregate_term([y.totals() for y in years[:2]])
    assert np.array(agg.draw_matrix()).shape == (1_000, 8)
    assert Tower.from_json(Tower([layer]).to_json()).to_json() == Tower([layer]).to_json()
    with pytest.raises(ValueError):
        events.term_years(7)


def test_normal_power_and_wilson_hilferty():
    import statistics

    from prospicio.aggregate import compound_moments
    from prospicio.distributions import Gamma, Poisson

    m = compound_moments(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    z = statistics.NormalDist().inv_cdf(0.99)
    sd = m.mean * m.cv
    np_ = m.approximate("normal_power")
    assert np_.kind == "normal_power"
    assert dict(np_.parameters)["skew"] == pytest.approx(m.skew)
    assert np_.quantile(0.99) == pytest.approx(m.mean + sd * (z + m.skew * (z * z - 1) / 6), rel=1e-12)
    wh = m.approximate("wilson_hilferty")
    c = 1 + m.skew * z / 6 - m.skew**2 / 36
    assert wh.quantile(0.99) == pytest.approx(m.mean + sd * 2 / m.skew * (c**3 - 1), rel=1e-12)
    assert wh.cdf(wh.quantile(0.9)) == pytest.approx(0.9, abs=1e-12)


def test_layer_third_moment():
    from prospicio.distributions import Gamma, Pareto

    x = Gamma(1.0, 2.0)
    t = 1.5
    want = 48.0 * (1 - math.exp(-t) * (1 + t + t * t / 2))
    assert x.layer_third_moment(3.0, 0.0) == pytest.approx(want, rel=1e-10)
    assert x.layer_third_moment(math.inf, 0.0) == pytest.approx(48.0, rel=1e-10)
    assert Pareto(100.0, 2.5).layer_third_moment(math.inf, 0.0) == math.inf


def test_tower_results_by_event_label():
    from prospicio.aggregate import EventSet

    events = EventSet.from_long(
        [0, 0, 1, 1], [15.0, 20.0, 10.0, 4.0], 2,
        labels={"peril": ["wind", "quake", "quake", "wind"]},
    )
    assert events.has_labels and events.label_dims == ["peril"]
    assert events.labels(1) == {"peril": ["quake", "wind"]}
    tower = Tower([Layer("10x5", 10.0, 5.0, aggregate_limit=15.0)])
    result = tower.apply(events)
    assert result.dims == ["kind", "layer", "peril"]
    assert result.marginal(("ceded", "10x5", "wind")).draws == pytest.approx([10.0, 0.0])
    assert result.marginal(("ceded", "10x5", "quake")).draws == pytest.approx([5.0, 5.0])
    assert result.marginal(("net", "retained", "quake")).draws == pytest.approx([15.0, 5.0])
    plain = tower.apply(EventSet.from_years([[15.0, 20.0], [10.0, 4.0]]))
    back = result.aggregate(["kind", "layer"])
    for key in [("gross", "ground_up"), ("ceded", "10x5"), ("net", "retained")]:
        assert back.marginal(key).draws == pytest.approx(plain.marginal(key).draws)
    # Merged with an unlabelled set, whose losses get "".
    other = EventSet.from_years([[1.0], [2.0]])
    merged = events.merge(other)
    assert merged.label_combinations == [[""], ["quake"], ["wind"]]
    with pytest.raises(ValueError):
        events.with_labels({"kind": ["a"] * 4})


def test_scale_to_totals_carries_a_capped_total_back():
    from prospicio.distributions import PredictiveDistribution

    pd = PredictiveDistribution(
        ["class"], [("motor",), ("home",)], [[6.0, 9.0], [3.0, 1.0], [0.0, 0.0]]
    )
    capped = [min(sum(r), 10.0) for r in pd.draw_matrix()]
    scaled = pd.scale_to_totals(capped)
    assert scaled.draw_matrix() == [[4.0, 6.0], [3.0, 1.0], [0.0, 0.0]]
    assert scaled.components() == pd.components()
    with pytest.raises(ValueError):
        pd.scale_to_totals([1.0, 1.0, 1.0])


def test_apply_annual_on_plain_losses():
    from prospicio.distributions import PredictiveDistribution

    # A loss-ratio cap of 120% on a premium of 100, with a commission
    # sliding from 30% at a 50% loss ratio to 10% at 90% (#477).
    cover = (Layer.stop_loss("cover", 120.0, 0.0)
             .with_premium_rate(1.0, 100.0)
             .with_sliding_scale([(0.3, 0.5), (0.1, 0.9)]))
    losses = [40.0, 70.0, 150.0]
    res = Tower([cover]).apply_annual(losses)
    ceded = res.marginal(["ceded", "cover"]).draws
    assert ceded == [40.0, 70.0, 120.0]
    cc = res.marginal(["ceding_commission", "cover"]).draws
    assert cc == pytest.approx([cover.ceding_commission_for(c) for c in ceded], rel=1e-12)
    assert cc == pytest.approx([30.0, 20.0, 10.0], rel=1e-12)
    # The same as apply_aggregate on the losses as draws.
    agg = Tower([cover]).apply_aggregate(PredictiveDistribution(["line"], [("a",)], [[x] for x in losses]))
    assert agg.marginal(["ceding_commission", "cover"]).draws == pytest.approx(cc, rel=1e-12)
    with pytest.raises(ValueError):
        Tower([cover]).apply_annual([1.0, -1.0])
    with pytest.raises(ValueError):
        Tower([cover]).apply_annual([])


YLT_CSV = (
    "year,peril,slot,class,value,item\n"
    "1,wind,5,home,10,a\n"
    "1,wind,5,auto,4,b\n"
    "2,quake,1,home,30,c\n"
    "1,wind,2,home,2,d\n"
)


def test_read_ylt_orders_checks_and_tiles(tmp_path):
    from prospicio.aggregate import EventSet, read_ylt, ylt_hash

    path = tmp_path / "ylt.csv"
    path.write_text(YLT_CSV)
    e = read_ylt(path, labels=["peril", "class"], n_years=3, n_sims=6)
    assert e.n_sims == 6
    # Year 1 by slot, the file's order breaking the tie in slot 5.
    assert e.events(0) == [2.0, 10.0, 4.0]
    assert e.events(1) == [30.0] and e.events(2) == []
    assert e.events(3) == e.events(0)
    assert e.labels(0)["class"] == ["home", "home", "auto"]
    assert e.input_hash == ylt_hash(path.read_bytes())
    assert e.totals().provenance()["input_hash"] == e.input_hash

    # One slot hits two classes: unique by peril and class, not by peril.
    with pytest.raises(ValueError, match="rows 1 and 2"):
        read_ylt(path, labels=["peril", "class"], unique_by=["peril"])
    with pytest.raises(ValueError, match="not one of"):
        read_ylt(path, labels=["peril", "class"], expected={"peril": ["wind"]})
    with pytest.raises(ValueError, match="whole multiple"):
        read_ylt(path, labels=["peril", "class"], n_sims=5)
    with pytest.raises(ValueError, match="no column 'region'"):
        read_ylt(path, labels=["region"])

    r = read_ylt(path, labels=["peril", "class"], n_sims=7, expand="resample", seed=3)
    again = EventSet.from_ylt([1, 1, 2, 1], [10.0, 4.0, 30.0, 2.0], slots=[5, 5, 1, 2], labels={"class": ["home", "auto", "home", "home"]}, n_sims=7, expand="resample", seed=3)
    assert [r.events(i) for i in range(7)] == [again.events(i) for i in range(7)]
    assert r.seed == 3


def test_read_ylt_from_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from prospicio.aggregate import read_ylt

    table = pa.table({"year": [1, 1, 2], "slot": [2, 1, 1], "peril": ["wind", "quake", "wind"], "value": [5.0, 1.0, 7.0]})
    path = tmp_path / "ylt.parquet"
    pq.write_table(table, path)
    e = read_ylt(path, labels=["peril"])
    assert e.events(0) == [1.0, 5.0] and e.events(1) == [7.0]
    assert e.labels(0) == {"peril": ["quake", "wind"]}
