import math

import pytest

import actuarialrs as ar

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
    from actuarialrs.pricing import Mbbefd, TabulatedCurve

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
