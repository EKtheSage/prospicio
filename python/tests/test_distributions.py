import math
import pickle

import pytest
from scipy import stats

import prospicio as ar


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
    with pytest.raises(TypeError, match="got float"):
        ar.distributions.Grid.rounding(3.0, 100.0, 10)
    # Draws have no exact layer moments, so they are no severity.
    draws = ar.distributions.Sampled([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Sampled has no exact layer moments"):
        ar.distributions.Grid.rounding(draws, 100.0, 10)
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
    assert pd.provenance()["samplers"] is None
    with pytest.raises(ValueError, match="row 1"):
        ar.distributions.PredictiveDistribution(["x"], [(1,)], [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError):
        pd.aggregate(["state"])


def test_gamma_and_tweedie():
    import math

    from prospicio.distributions import Gamma, Grid, Poisson, Tweedie

    g = Gamma.from_mean_cv(1000.0, 0.5)
    assert g.shape == 4.0
    assert abs(g.lev(1500.0) + g.stop_loss(1500.0) - 1000.0) < 1e-9
    assert abs(Gamma(1.0, 3.0).survival(6.0) - math.exp(-2.0)) < 1e-16
    assert abs(Gamma.from_mean_dispersion(200.0, 0.25).shape - 4.0) < 1e-12

    y = Tweedie(500.0, 40.0, 1.6)
    assert abs(y.cdf(0.0) - math.exp(-y.lambda_)) < 1e-15
    assert abs(y.variance() - 40.0 * 500.0**1.6) < 1e-6
    assert y.ln_pdf(0.0) == -y.lambda_
    z = Tweedie.from_poisson_gamma(y.lambda_, y.severity.shape, y.severity.scale)
    assert abs(z.mean() / 500.0 - 1.0) < 1e-12
    with pytest.raises(ValueError):
        Tweedie(1.0, 1.0, 2.0)

    # Severities discretize and compound like any other.
    grid, _ = Grid.local_moment(g, 50.0, 400)
    assert abs(grid.mean() - g.lev(399 * 50.0)) < 1e-9
    assert Poisson(2.0).mean() == 2.0


def test_weibull_mixture_and_tail_diagnostics():
    import math

    from prospicio.distributions import Grid, Lognormal, Mixture, Pareto, Weibull
    from prospicio.risk import hill, mean_excess

    w = Weibull(1.0, 2.0)
    assert abs(w.stop_loss(3.0) - 2.0 * math.exp(-1.5)) < 1e-14
    assert abs(w.quantile(w.cdf(1.7)) - 1.7) < 1e-12
    m = Mixture([(0.9, Lognormal.from_mean_cv(1e4, 1.0)), (0.1, Pareto(1e5, 2.0))])
    assert abs(m.mean() - 29000.0) < 1e-6
    assert m.weights == [0.9, 0.1]
    with pytest.raises(ValueError):
        Mixture([(0.5, w)])
    grid, _ = Grid.local_moment(m, 1000.0, 2000)
    assert abs(grid.mean() - m.lev(1999 * 1000.0)) < 1e-6
    assert mean_excess([1.0, 2.0, 3.0, 4.0], [2.0]) == [(2.0, 1.5, 2)]
    n = 2000
    x = [(1 - (i - 0.5) / n) ** -0.5 for i in range(1, n + 1)]
    assert abs(hill(x, [200])[0] - 0.5) < 0.02


def test_loglogistic_growth_curve_and_heavy_tail():
    import math

    from prospicio.distributions import Grid, Loglogistic

    d = Loglogistic(1.0, 2.0)
    # Shape 1: F(x) = x / (x + theta), LEV = theta ln(1 + u / theta).
    assert abs(d.cdf(3.0) - 0.6) < 1e-15
    assert abs(d.lev(3.0) - 2.0 * math.log(2.5)) < 1e-13
    assert math.isinf(d.mean())
    # Clark's growth curve: the median age reports half of ultimate.
    g = Loglogistic(1.5, 24.0)
    assert abs(g.cdf(24.0) - 0.5) < 1e-15
    assert abs(g.lev(1e4) + g.stop_loss(1e4) - g.mean()) < 1e-9 * g.mean()
    grid, _ = Grid.local_moment(g, 1.0, 500)
    assert abs(grid.mean() - g.lev(499.0)) < 1e-9
    with pytest.raises(ValueError):
        Loglogistic(0.0, 1.0)


def test_marginal_selects_an_origin_period_by_its_label():
    from prospicio.reserving import OdpBootstrap, Triangle

    tri = Triangle.from_long(
        [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
        [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    )
    reserves = OdpBootstrap(n_sims=500, seed=1).fit(tri, "values").reserves
    by_text = reserves.marginal(("2023",))
    by_int = reserves.marginal((2023,))
    assert by_text is not None and by_int is not None
    assert by_text.draws == by_int.draws
    assert reserves.marginal(("2024",)) is None


def test_distributions_save_and_load_as_json():
    import json
    import math

    from prospicio.distributions import (
        Custom, Gamma, GeneralizedPareto, Grid, LogAffinePareto, Loglogistic, Lognormal,
        Mixture, Pareto, PiecewisePareto, Sampled, Tweedie, Weibull, from_json, to_json,
    )

    dists = [
        Lognormal(7.0, 0.5), Pareto(1e5, 1.5), Pareto(1e5, 1.5, 1e7),
        PiecewisePareto([1.0, 10.0, 100.0], [1.2, 1.8, 2.5]), LogAffinePareto(100.0, 1.5, 0.3),
        GeneralizedPareto(0.25, 3.0), Gamma(2.0, 500.0), Tweedie(1000.0, 2.0, 1.5),
        Weibull(1.5, 1000.0), Loglogistic(4.0, 900.0),
        Mixture([(0.7, Lognormal(7.0, 0.5)), (0.3, Pareto(1e5, 2.0))]),
        Grid(0.5, [0.1, 0.4, 0.3, 0.2]), Sampled([3.0, 1.0, 2.0]),
    ]
    for d in dists:
        text = to_json(d)
        assert json.loads(text)["format"] == "risk_rs.distribution"
        back = from_json(text)
        assert type(back) is type(d)
        assert to_json(back) == text
        assert back.mean() == d.mean() or (math.isinf(d.mean()) and math.isinf(back.mean()))
    with pytest.raises(ValueError, match="cannot be saved"):
        to_json(Custom(lambda x: 1 - math.exp(-x)))
    with pytest.raises(ValueError):
        from_json('{"format": "risk_rs.distribution", "format_version": 1, "family": "gamma", "shape": -1, "scale": 1}')


def test_more_claim_counts():
    from prospicio.aggregate import fft, panjer
    from prospicio.distributions import Count, Grid, NegativeBinomial, Poisson

    zt = Count.zero_truncated(Poisson(2.0))
    assert zt.pmf(0) == 0.0
    assert zt.mean() == pytest.approx(2.0 / (1.0 - math.exp(-2.0)))
    assert zt.panjer_ab() == (0.0, 2.0)
    zm = Count.zero_modified(NegativeBinomial(2.0, 1.5), 0.3)
    assert zm.pmf(0) == pytest.approx(0.3)
    # aggregate 1.0.1's Poisson-inverse Gaussian, mean 10, cv 0.5.
    pig = Count.mixed_poisson(10.0, 0.5, mixing="inverse_gaussian")
    assert pig.pmf(0) == pytest.approx(0.0030337404, abs=1e-10)
    assert pig.panjer_ab() is None
    ney = Count.compound_poisson(2.0, Poisson(3.0))
    assert ney.mean() == pytest.approx(6.0)
    assert Count.logarithmic(0.5).pmf(1) == pytest.approx(0.5 / math.log(2.0))
    emp = Count.empirical([0.5, 0.25, 0.25])
    assert emp.mean() == 0.75 and len(emp.sample(10, seed=1)) == 10
    sev = Grid(1.0, [0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
    a, _ = panjer(zm, sev, 100)
    b, _ = fft(zm, sev, 100)
    assert max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-12
    c, _ = fft(pig, sev, 400)
    assert c.mean() == pytest.approx(10.0 * sev.mean(), rel=1e-8)
    with pytest.raises(ValueError):
        panjer(pig, sev, 100)
    with pytest.raises(ValueError):
        Count.mixed_poisson(1.0, 0.5, mixing="beta")


def test_added_severities_and_splicing():
    import json

    from prospicio.distributions import (
        Beta, Burr, Gamma, InverseGamma, InverseGaussian, Loglogistic, Lognormal, Mixture,
        Pareto, Truncated, from_json, to_json,
    )

    # SciPy 1.18: invgamma(3, scale=2000).cdf(1000), burr12(2, 1.5, scale=900).sf(1000),
    # invgauss(0.5, scale=2000).cdf(1000), beta(2, 3, scale=5000).cdf(1000).
    assert InverseGamma(3.0, 2000.0).cdf(1000.0) == pytest.approx(0.6766764161830634, rel=1e-12)
    assert Burr(1.5, 2.0, 900.0).survival(1000.0) == pytest.approx(
        (1 + (1000 / 900) ** 2) ** -1.5, rel=1e-13
    )
    assert InverseGaussian(1000.0, 2000.0).cdf(1000.0) == pytest.approx(0.6276978381552528, rel=1e-12)
    assert Beta(2.0, 3.0, 5000.0).cdf(1000.0) == pytest.approx(0.1808, rel=1e-13)
    # Burr with alpha = 1 is the loglogistic.
    assert Burr(1.0, 4.0, 300.0).lev(500.0) == pytest.approx(Loglogistic(4.0, 300.0).lev(500.0))

    t = Truncated(Gamma(2.0, 500.0), 100.0, 4000.0)
    assert t.cdf(100.0) == 0.0 and t.cdf(4000.0) == 1.0
    assert t.lev(1e9) == pytest.approx(t.mean())
    assert type(t.severity) is Gamma
    with pytest.raises(ValueError):
        Truncated(Pareto(100.0, 2.0), 0.0, 50.0)

    body, tail = Lognormal.from_mean_cv(50.0, 1.0), Pareto(100.0, 2.5)
    s = Mixture.splice([(0.9, body), (0.1, tail)], [0.0, 100.0, float("inf")])
    assert s.cdf(100.0) == pytest.approx(0.9)
    assert s.survival(400.0) == pytest.approx(0.1 * 0.25**2.5)

    for d in [InverseGamma(3.5, 2500.0), InverseGaussian(1000.0, 2000.0), Burr(2.0, 1.5, 900.0),
              Beta(2.0, 3.0, 5000.0), t, s]:
        text = to_json(d)
        back = from_json(text)
        assert type(back) is type(d)
        assert to_json(back) == text
        assert back.mean() == d.mean()
    assert json.loads(to_json(t))["inner"]["family"] == "gamma"
