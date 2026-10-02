import math
import pickle

import pytest

import actuarialrs as ar

D = ar.distributions


def test_pareto_layers_and_truncation():
    p = D.Pareto(500.0, 2.0)
    assert p.layer(4000.0, 1000.0) == pytest.approx(200.0, rel=1e-12)
    assert p.mean() == pytest.approx(1000.0)
    assert p.survival(1000.0) == pytest.approx(0.25)
    assert math.isinf(D.Pareto(1.0, 0.9).mean())
    t = D.Pareto(500.0, 2.0, truncation=8000.0)
    assert t.truncation == 8000.0
    assert t.quantile(1.0) == pytest.approx(8000.0)
    assert t.lev(1e5) == pytest.approx(t.mean())
    assert t.layer_variance(1e4, 0.0) == pytest.approx(t.variance(), rel=1e-12)
    with pytest.raises(ValueError, match="alpha"):
        D.Pareto(1.0, 0.0)


def test_pareto_fit_closed_form():
    fit = D.Pareto.fit([1500.0, 2500.0, 4000.0, 10_000.0], 1000.0,
                       censored=[False, False, False, True])
    assert fit.alpha == pytest.approx(3.0 / math.log(150.0), rel=1e-12)
    assert isinstance(fit, D.Pareto)


def test_piecewise_pareto():
    pp = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0])
    assert pp.survival(4000.0) == pytest.approx(0.125, rel=1e-12)
    assert pp.stop_loss(2000.0) == pytest.approx(1000.0, rel=1e-12)
    assert pp.t == [1000.0, 2000.0] and pp.alpha == [1.0, 2.0]
    lp = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0)
    wd = D.PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0, truncation_type="wd")
    assert lp.truncation_type == "lp" and wd.truncation_type == "wd"
    assert lp.survival(1500.0) != wd.survival(1500.0)
    with pytest.raises(ValueError, match="truncation_type"):
        D.PiecewisePareto([1.0], [1.0], truncation=2.0, truncation_type="xx")
    fit = D.PiecewisePareto.fit([1200.0, 1500.0, 2500.0, 6000.0], [1000.0, 2000.0])
    want = 2.0 / (math.log(1.2) + math.log(1.5) + 2.0 * math.log(2.0))
    assert fit.alpha[0] == pytest.approx(want, rel=1e-12)


def test_log_affine_and_generalized_pareto():
    d = D.LogAffinePareto.from_delta(1e6, 1.5, 0.5)
    assert d.local_alpha(2e6) == pytest.approx(2.0)
    assert d.delta == pytest.approx(0.5)
    # gamma = 0 is the Pareto.
    flat = D.LogAffinePareto(1000.0, 2.5, 0.0)
    assert flat.layer(4000.0, 1000.0) == pytest.approx(D.Pareto(1000.0, 2.5).layer(4000.0, 1000.0))
    g = D.GeneralizedPareto.riegel(1000.0, 2.0, 1.5)
    assert g.survival(2000.0) == pytest.approx((7.0 / 3.0) ** -1.5, rel=1e-12)
    assert g.location == 1000.0
    assert g.lev(5000.0) + g.stop_loss(5000.0) == pytest.approx(g.mean(), rel=1e-12)


def test_counts_by_dispersion():
    assert isinstance(D.claim_count(4.0, 1.0), D.Poisson)
    assert isinstance(D.claim_count(4.0, 2.5), D.NegativeBinomial)
    b = D.claim_count(10.0, 0.3)
    assert isinstance(b, D.Binomial)
    assert (b.n, b.mean()) == (15, pytest.approx(10.0))
    assert D.Binomial(10, 0.3).pmf(11) == 0.0
    assert sum(D.Binomial(10, 0.3).pmf(k) for k in range(11)) == pytest.approx(1.0)


def test_pareto_family_feeds_aggregation():
    # Truncated inside the grid, so no severity or aggregate mass is lumped
    # into the last point (up to 20 claims of at most 50,000).
    sev = D.PiecewisePareto([1000.0, 3000.0], [1.2, 2.0], truncation=50_000.0)
    grid, _ = D.Grid.local_moment(sev, 100.0, 501)
    assert grid.mean() == pytest.approx(sev.mean(), rel=1e-12)
    agg, _ = ar.aggregate.panjer(D.Binomial(20, 0.1), grid, 10_001)
    assert agg.mean() == pytest.approx(2.0 * grid.mean(), rel=1e-9)


def test_pickle_round_trip():
    for d in (D.Pareto(500.0, 2.0, 8000.0),
              D.PiecewisePareto([1.0, 2.0], [1.0, 2.0], 5.0, "wd"),
              D.LogAffinePareto(1.0, 1.5, 0.3),
              D.GeneralizedPareto(0.5, 2.0, 1.0),
              D.Binomial(5, 0.2)):
        back = pickle.loads(pickle.dumps(d))
        assert repr(back) == repr(d)
