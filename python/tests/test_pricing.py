import math

import pytest

import prospicio as ar

D = ar.distributions
P = ar.pricing

ATTACHMENTS = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 5000.0, 10_000.0]
LOSSES = [100.0, 90.0, 50.0, 40.0, 100.0, 50.0, 50.0]


def test_collective_model_layer_moments():
    m = P.CollectiveModel(D.claim_count(2.0, 1.5), D.Pareto(1e6, 2.0))
    assert m.layer_mean(4e6, 1e6) == pytest.approx(1.6e6, rel=1e-12)
    assert m.excess_frequency(2e6) == pytest.approx(0.5, rel=1e-12)
    # E[N] E[Y^2] + (Var N - E N) E[Y]^2.
    sev = D.Pareto(1e6, 2.0)
    want = 2.0 * sev.layer_second_moment(4e6, 1e6) + (3.0 - 2.0) * sev.layer(4e6, 1e6) ** 2
    assert m.layer_variance(4e6, 1e6) == pytest.approx(want, rel=1e-12)
    assert math.isinf(m.variance())
    events = m.simulate(100, 7)
    assert events.n_sims == 100


def test_rating_helpers():
    p = D.Pareto(100.0, 2.0)
    assert P.ilf(p, 1000.0, 200.0) == pytest.approx(p.lev(1000.0) / p.lev(200.0))
    assert P.loss_elimination_ratio(p, 150.0) == pytest.approx(p.lev(150.0) / p.mean())
    r = P.pareto_extrapolation((1000.0, 1000.0), (3000.0, 2000.0), 1.7)
    assert P.alpha_between_layers((1000.0, 1000.0, 1.0), (3000.0, 2000.0, r)) == pytest.approx(1.7)
    assert P.alpha_between_frequency_and_layer(1e6, 2.0, 1e6, 1e6, 1e6) == pytest.approx(2.0)
    assert P.alpha_between_frequencies(1e6, 4.0, 2e6, 1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        P.alpha_between_layers((1000.0, 1000.0, 1.0), (1000.0, 2000.0, 1.0))


def test_tower_matching_reproduces_the_tower():
    m = P.match_tower(ATTACHMENTS, LOSSES, [0.25] + [None] * 6)
    for i, e in enumerate(LOSSES):
        limit = ATTACHMENTS[i + 1] - ATTACHMENTS[i] if i < 6 else math.inf
        assert m.layer_loss(limit, ATTACHMENTS[i]) == pytest.approx(e, rel=1e-10)
    # Riegel (2018), Table 4.
    assert m.severity.alpha[:3] == pytest.approx([2.374, 0.199, 0.175], abs=5e-4)
    assert m.frequency == 0.25
    mid = P.match_tower(ATTACHMENTS, LOSSES, rule="midpoint")
    assert mid.layer_loss(500.0, 1500.0) == pytest.approx(90.0, rel=1e-10)
    with pytest.raises(ValueError, match="rule"):
        P.match_tower(ATTACHMENTS, LOSSES, rule="nope")


def test_pml_and_reference_fits():
    m = P.fit_pml_curve([5.0, 20.0, 50.0], [2e6, 5e6, 9e6], tail_alpha=1.8)
    assert m.excess_frequency(5e6) == pytest.approx(1 / 20.0, rel=1e-12)
    r = P.fit_references([(1000.0, 1000.0, 120.0), (1500.0, 1500.0, 110.0)], [(2500.0, 0.05)])
    assert r.layer_loss(1500.0, 1500.0) == pytest.approx(110.0, rel=1e-11)
    assert r.excess_frequency(2500.0) == pytest.approx(0.05, rel=1e-11)
    # The fitted severity is a PiecewisePareto usable anywhere.
    assert isinstance(r.severity, D.PiecewisePareto)


def test_risk_loaded_prices():
    R = ar.risk
    pd = D.PredictiveDistribution(
        ["cover"], [("a",), ("b",)], [[0.0, 2.0], [1.0, 1.0], [4.0, 0.0], [8.0, 0.0]]
    )
    assets = R.Distortion.tvar(0.5)
    p = P.price_portfolio(pd, assets, cost_of_capital=0.1)
    assert p.components() == [("a",), ("b",)]
    assert sum(c.premium for c in p.allocated) == pytest.approx(p.total.premium)
    for c in p.allocated:
        assert c.return_on_capital == pytest.approx(0.1)
    assert p.diversification() > 0.0
    total = P.price(pd, assets, cost_of_capital=0.1)
    assert total.premium == pytest.approx(p.total.premium)
    wang = P.price(pd, assets, distortion=R.Distortion.wang(0.2))
    assert wang.expected_loss < wang.premium < wang.assets
    with pytest.raises(ValueError):
        P.price(pd, assets)
    with pytest.raises(ValueError):
        P.price(pd, assets, distortion=R.Distortion.tvar(0.9))


def test_tabulated_curve_and_destruction_rates():
    from prospicio.pricing import Mbbefd, TabulatedCurve

    t = TabulatedCurve([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0])
    assert t.curve([0.3])[0] == pytest.approx(0.6, abs=1e-15)
    assert t.mean_rate() == pytest.approx(0.25)
    # Atoms at 0.1, 0.5 and 1 with probabilities 0.75, 0.15, 0.1.
    assert t.rate_quantile([0.75, 0.76, 0.91]) == [0.1, 0.5, 1.0]
    assert t.layer_share(5e6, 5e6, 10e6) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        TabulatedCurve([0.0, 0.5, 1.0], [0.0, 0.3, 1.0])  # convex
    # MBBEFD rate quantiles reproduce the curve's mean.
    c3 = Mbbefd.swiss_re(3.0)
    n = 100_000
    rates = c3.rate_quantile([(i + 0.5) / n for i in range(n)])
    assert sum(rates) / n == pytest.approx(c3.mean(), rel=1e-3)
    import pickle
    assert pickle.loads(pickle.dumps(t)).x == t.x


def test_risk_profile_simulation_matches_exposure_rating():
    from prospicio.pricing import Mbbefd, RiskProfile, TabulatedCurve
    from prospicio.reinsurance import Layer, Tower

    curves = [Mbbefd.swiss_re(2.0), Mbbefd.swiss_re(3.0),
              TabulatedCurve([0.0, 0.02, 0.2, 1.0], [0.0, 0.3, 0.8, 1.0])]
    p = RiskProfile([0.5e6, 3e6, 20e6], [2000, 300, 20], curves,
                    expected_losses=[0.6e6, 0.5e6, 0.4e6])
    assert p.expected_loss() == pytest.approx(1.5e6)
    events = p.simulate(50_000, 11)
    tower = Tower.inuring([[Layer.surplus("surplus", 1e6, 5.0)], [Layer("xl", 1e6, 0.5e6)]])
    pd = tower.apply(events)
    n = 50_000
    for key, want in [(("ceded", "surplus"), p.expected_surplus_loss(1e6, 5.0)),
                      (("ceded", "xl"), p.expected_layer_loss(1e6, 0.5e6, 1e6, 5.0))]:
        m = pd.marginal(key)
        assert abs(m.mean() - want) < 4 * m.variance() ** 0.5 / n ** 0.5, key
    # Premium times loss ratio; one curve for all bands.
    q = RiskProfile([1e6, 10e6], [800, 50], Mbbefd.swiss_re(3.0), premiums=[2e6, 1e6],
                    loss_ratio=[0.6, 0.5])
    assert q.expected_loss() == pytest.approx(1.7e6)
    assert q.expected_layer_loss(math.inf, 0.0) == pytest.approx(1.7e6)
    with pytest.raises(ValueError):
        RiskProfile([1e6], [1], Mbbefd.swiss_re(3.0), premiums=[1.0])
    with pytest.raises(ValueError):
        RiskProfile([1e6], [1], Mbbefd.swiss_re(3.0), expected_losses=[1.0], premiums=[1.0])
    with pytest.raises(TypeError):
        RiskProfile([1e6], [1], "curve", expected_losses=[1.0])


def test_risk_profile_spreads_sums_insured_between_bounds():
    from prospicio.pricing import Mbbefd, RiskProfile
    from prospicio.reinsurance import Layer, Tower

    c = Mbbefd.swiss_re(3.0)
    # The second band's risks run from 1m to 5m; the first has no bounds.
    p = RiskProfile([0.5e6, 3e6], [2000, 400], c, expected_losses=[0.6e6, 1.2e6],
                    lower=[None, 1e6], upper=[None, 5e6])
    point = RiskProfile([0.5e6, 3e6], [2000, 400], c, expected_losses=[0.6e6, 1.2e6])
    assert p.expected_claims() == pytest.approx(point.expected_claims(), rel=1e-12)
    # A 2m retention: the 3m risk cedes 1/3; the spread band, weighted by
    # sum insured, cedes 3/8.
    assert point.expected_surplus_loss(2e6, 4.0) == pytest.approx(1.2e6 / 3, rel=1e-12)
    assert p.expected_surplus_loss(2e6, 4.0) == pytest.approx(0.375 * 1.2e6, rel=1e-9)
    n = 50_000
    pd = Tower.inuring([[Layer.surplus("surplus", 2e6, 4.0)]]).apply(p.simulate(n, 3))
    m = pd.marginal(("ceded", "surplus"))
    assert abs(m.mean() - 0.45e6) < 4 * m.variance() ** 0.5 / n ** 0.5
    with pytest.raises(ValueError):
        RiskProfile([3e6], [1], c, expected_losses=[1.0], lower=[1e6])
    with pytest.raises(ValueError):
        RiskProfile([3e6], [1], c, expected_losses=[1.0], lower=[1e6], upper=[None])
    with pytest.raises(ValueError):
        RiskProfile([3e6], [1], c, expected_losses=[1.0], lower=[5e6], upper=[1e6])


def test_risk_profile_tilts_a_spread_to_its_mean_sum_insured():
    from prospicio.pricing import Mbbefd, RiskProfile
    from prospicio.reinsurance import Layer, Tower

    c = Mbbefd.swiss_re(3.0)
    # 400 risks from 1m to 5m totalling 800m: mean 2m, not the midpoint.
    tilted = RiskProfile([2e6], [400], c, expected_losses=[1.2e6], lower=[1e6], upper=[5e6],
                         spread="tilted")
    point = RiskProfile([2e6], [400], c, expected_losses=[1.2e6])
    assert tilted.expected_claims() == pytest.approx(point.expected_claims(), rel=1e-12)
    # Leaning to small risks, it cedes less than the uniform spread's 3/8.
    ceded = tilted.expected_surplus_loss(2e6, 4.0)
    assert ceded < 0.375 * 1.2e6
    n = 50_000
    pd = Tower([Layer.surplus("surplus", 2e6, 4.0)]).apply(tilted.simulate(n, 3))
    m = pd.marginal(("ceded", "surplus"))
    assert abs(m.mean() - ceded) < 4 * m.variance() ** 0.5 / n ** 0.5
    # At the midpoint the tilted spread is the uniform one.
    kw = dict(expected_losses=[1.2e6], lower=[1e6], upper=[5e6])
    mid = RiskProfile([3e6], [400], c, spread="tilted", **kw)
    assert mid.expected_surplus_loss(2e6, 4.0) == pytest.approx(0.375 * 1.2e6, rel=1e-9)
    with pytest.raises(ValueError):
        RiskProfile([5e6], [400], c, spread="tilted", **kw)
    with pytest.raises(ValueError):
        RiskProfile([3e6], [400], c, spread="steep", **kw)


INSCO_ROWS = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
              [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]


def test_natural_allocation_on_insco():
    from prospicio.pricing import Pentagon, Portfolio
    from prospicio.risk import Distortion

    port = Portfolio(["A", "B", "C"], INSCO_ROWS)
    assert port.totals == [22, 28, 36, 40, 55, 65, 100]
    assert port.assets(0.85) == 65
    assert port.kappa("C")[-1] == 64
    # aggregate 1.0.1: CCoC at 15% and assets 100.
    p = port.price(Distortion.ccoc(0.15), assets=100)
    assert p.total.premium == pytest.approx(53.565217391304344, abs=1e-12)
    assert [u.assets for u in p.allocated] == pytest.approx([16, 20, 64], abs=1e-12)
    # PH at aggregate's calibrated parameter, lifted, at assets 65.
    ph = port.calibrate("ph", assets=100, return_on_capital=0.15)
    lifted = port.price(ph, p=0.85, allocation="lifted")
    assert [u.capital for u in lifted.allocated] == pytest.approx(
        [1.947189207380, -0.070344428313, 16.219694433834], abs=1e-6)
    d = lifted.to_dict()
    assert d["unit"] == ["A", "B", "C", "total"]
    assert sum(d["premium"][:3]) == pytest.approx(d["premium"][3])
    total, units = port.epd(65)
    assert total == pytest.approx(3.5 / 46.6)
    assert port.assets_for_epd(3.5 / 46.6) == pytest.approx(65)
    assert sum(port.bodoff(assets=100)) == pytest.approx(100)
    pent = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
    assert pent.premium == pytest.approx(53.565217391304344)
    with pytest.raises(ValueError):
        Pentagon.solve(loss=1.0, premium=2.0)
    with pytest.raises(ValueError):
        port.price(ph, allocation="other")


def test_natural_from_independent_and_predictive():
    from prospicio.distributions import Grid, PredictiveDistribution
    from prospicio.pricing import Portfolio

    a = Grid(1.0, [0.5, 0.5])
    b = Grid(1.0, [0.5, 0.0, 0.5])
    port = Portfolio.from_independent(["a", "b"], [a, b])
    assert port.totals == [0.0, 1.0, 2.0, 3.0]
    assert port.kappa("a")[1] == pytest.approx(1.0)
    pd = PredictiveDistribution(["unit"], [("A",), ("B",), ("C",)], [[float(x) for x in row] for row in INSCO_ROWS])
    from_pd = Portfolio.from_predictive(pd)
    assert from_pd.units == ["A", "B", "C"]
    assert from_pd.totals == [22, 28, 36, 40, 55, 65, 100]


def test_premium_bounds_and_classical():
    from prospicio.distributions import Sampled
    from prospicio.pricing import Portfolio, calibrate_classical, classical_premium

    port = Portfolio(["A", "B", "C"], INSCO_ROWS)
    b = port.premium_bounds(53.565217391304344, assets=100)
    assert [round(u["lower"], 9) for u in b] == [13.097826087, 17.465726051, 19.254738016]
    assert b[2]["upper"] == pytest.approx(22.098038028339595, abs=1e-9)
    x = Sampled([22, 28, 36, 40, 40, 40, 40, 55, 65, 100])
    assert calibrate_classical("esscher", x, 53.565217391304344) == pytest.approx(0.012851355964986997, rel=1e-8)
    assert classical_premium("standard_deviation", Sampled([0.0, 10.0]), 0.2) == 6.0
    with pytest.raises(ValueError):
        classical_premium("nope", x, 0.1)
