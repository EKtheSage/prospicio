import math
import pickle

import pytest
from scipy import stats

import actuarialrs as ar


def test_matches_scipy():
    d = ar.distributions.Lognormal(7.0, 0.5)
    ref = stats.lognorm(s=0.5, scale=math.exp(7.0))
    assert d.mean() == pytest.approx(ref.mean(), rel=1e-13)
    assert d.variance() == pytest.approx(ref.var(), rel=1e-12)
    for p in (0.01, 0.5, 0.995):
        assert d.quantile(p) == pytest.approx(ref.ppf(p), rel=1e-12)
    assert d.cdf(1500.0) == pytest.approx(ref.cdf(1500.0), rel=1e-12)


def test_from_mean_cv():
    d = ar.distributions.Lognormal.from_mean_cv(1000.0, 0.5)
    assert d.mean() == pytest.approx(1000.0)
    assert d.std() == pytest.approx(500.0)


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError, match="sdlog"):
        ar.distributions.Lognormal(0.0, -1.0)
    with pytest.raises(ValueError, match="probability"):
        ar.distributions.Lognormal(0.0, 1.0).quantile(2.0)


def test_sample_reproducible_and_matches_rust_stream():
    d = ar.distributions.Lognormal(0.0, 1.0)
    a = d.sample(5, seed=42, stream=3)
    assert a == d.sample(5, seed=42, stream=3)
    assert a != d.sample(5, seed=42, stream=4)
    # Same values are pinned in the Rust and R tests.
    assert a[:3] == [1.0007760893701914, 1.6293872534754683, 1.0763869265482304]


def test_pickle_round_trip():
    d = ar.distributions.Lognormal(1.5, 0.25)
    back = pickle.loads(pickle.dumps(d))
    assert (back.meanlog, back.sdlog) == (1.5, 0.25)
    assert repr(back) == "Lognormal(meanlog=1.5, sdlog=0.25)"


def test_lognormal_severity_methods():
    d = ar.distributions.Lognormal(7.0, 0.5)
    for limit in (100.0, 1500.0, 1e5):
        assert d.lev(limit) + d.stop_loss(limit) == pytest.approx(d.mean(), rel=1e-12)
    assert d.layer(1000.0, 500.0) == pytest.approx(d.lev(1500.0) - d.lev(500.0), rel=1e-12)


def test_claim_counts_match_scipy():
    n = ar.distributions.Poisson(3.0)
    ref = stats.poisson(3.0)
    for k in range(10):
        assert n.pmf(k) == pytest.approx(ref.pmf(k), rel=1e-12)
        assert n.cdf(k) == pytest.approx(ref.cdf(k), rel=1e-12)
    assert n.quantile(0.9) == ref.ppf(0.9)

    nb = ar.distributions.NegativeBinomial(2.5, 1.5)
    ref = stats.nbinom(2.5, 1 / 2.5)
    for k in range(10):
        assert nb.pmf(k) == pytest.approx(ref.pmf(k), rel=1e-12)
    assert nb.variance() == pytest.approx(ref.var(), rel=1e-12)
    assert nb.sample(5, seed=1) == nb.sample(5, seed=1)
    with pytest.raises(ValueError, match="variance"):
        ar.distributions.NegativeBinomial.from_mean_variance(10.0, 5.0)


def test_grid_discretization():
    d = ar.distributions.Lognormal(7.0, 0.5)
    grid, report = ar.distributions.Grid.local_moment(d, 100.0, 200)
    assert len(grid) == 200 and report.method == "local_moment"
    assert sum(grid.probs) == pytest.approx(1.0, abs=1e-12)
    # Local moment matching keeps the limited mean up to the last point.
    assert grid.mean() == pytest.approx(d.lev(199 * 100.0), rel=1e-9)
    assert report.mean_error() == pytest.approx(-d.stop_loss(199 * 100.0), abs=1e-9)
    lower, _ = ar.distributions.Grid.lower(d, 100.0, 200)
    assert lower.mean() <= d.mean()
    # A grid can be rediscretized, and is itself a severity.
    coarse, _ = ar.distributions.Grid.rounding(grid, 200.0, 100)
    assert coarse.layer(1000.0, 1000.0) >= 0.0
    with pytest.raises(TypeError):
        ar.distributions.Grid.rounding(3.0, 100.0, 10)
    with pytest.raises(ValueError):
        ar.distributions.Grid(1.0, [0.5, 0.4])


def test_sampled_risk_measures():
    s = ar.distributions.Sampled([1.0, 2.0, 3.0, 4.0])
    assert s.var(0.5) == 2.0
    assert s.tvar(0.375) == pytest.approx(3.2)
    assert pickle.loads(pickle.dumps(s)).draws == [1.0, 2.0, 3.0, 4.0]


def test_predictive_distribution_is_joint():
    pd = ar.distributions.PredictiveDistribution(
        ["lob", "origin"],
        [("Auto", 2023), ("Auto", 2024), ("Home", 2023), ("Home", 2024)],
        [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0], [5.0, 6.0, 50.0, 60.0]],
    )
    assert (pd.n_sims, pd.n_components) == (3, 4)
    assert pd.components()[3] == ("Home", 2024)
    by_lob = pd.aggregate(["lob"])
    assert by_lob.draw_matrix() == [[3.0, 30.0], [7.0, 70.0], [11.0, 110.0]]
    assert pd.total().draws == [33.0, 77.0, 121.0]
    assert pd.marginal(("Home", 2023)).draws == [10.0, 30.0, 50.0]
    assert pd.marginal(("Home", 2025)) is None
    assert pd.provenance()["model"] == "python"
    with pytest.raises(ValueError, match="row 1"):
        ar.distributions.PredictiveDistribution(["x"], [(1,)], [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError):
        pd.aggregate(["state"])
